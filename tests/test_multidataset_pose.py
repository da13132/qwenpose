from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest import mock
import warnings

from PIL import Image
import torch

from qwenpose.data import (
    ALL_POSE_PROMPT,
    EpochRandomRefHumanDataset,
    InterleavedPoseDataset,
    parse_dataset_mix_weights,
    PoseAugmentConfig,
    PoseRecord,
    PoseRecordDataset,
    aic_to_union,
    load_aic_records,
    load_coco_records,
    load_mpii_records,
    letterbox_pose_image,
    mpii_boxes_from_center_scale,
    pose_collate,
    set_pose_dataset_epoch,
    transform_pose_boxes,
    transform_pose_keypoints,
)
from qwenpose.eval_pose import predictions_to_coco_results, tensor_to_prediction_rows
from qwenpose.eagle_lora import (
    EagleFeatureExtractor,
    _load_eagle_vision_projector_weights,
    build_eagle_inputs,
    find_eagle_lora_targets,
)
from qwenpose.losses import (
    LossWeights,
    compute_keypoint_denoising_loss,
    compute_person_confidence_rescue_loss,
    compute_pose_losses,
    compute_refhuman_match_loss,
    per_joint_oks_loss,
)
from qwenpose.metrics import (
    _mpii_bbox_from_center_scale,
    prediction_rows_to_instances,
    targets_to_gt_instances,
)
from qwenpose.model import (
    QwenPoseConfig,
    QwenPoseModel,
    SinePositionEncoding,
    apply_refhuman_box_refinement_safety,
    apply_keypoint_decode_mode,
    build_schema_joint_priors,
    nonsemantic_joint_reference_points,
)
from qwenpose.qwen_lora import QwenFeatureRefiner
from qwenpose.spatial_features import MultiScaleSpatialFeatureBatch, SpatialFeatureBatch
from qwenpose.train_pose import (
    REFHUMAN_FALLBACK_MARKER,
    SCHEMA_POSE_EDGE_INDICES,
    HomogeneousDatasetBatchSampler,
    align_targets_to_person_queries,
    backbone_adapter_state_dict,
    build_locate_generation_prompts,
    build_optimizer_param_groups,
    build_progress_loss_postfix,
    configure_backbone_train_scope,
    estimate_locate_vision_tokens,
    _locate_coord_token,
    locate_generation_token_budget,
    nms_box_indices_xyxy,
    pair_keypoint_denoising_with_box_denoising,
    parse_locate_bbox_response,
    prepare_box_denoising,
    prepare_keypoint_denoising,
    prepare_joint_pbd_conditioning,
    prepare_locate_generated_box_conditioning_from_responses,
    prepare_person_query_conditioning,
    save_pose_visualization,
    select_informative_visualization_sample,
    validate_pose_batch_contract,
)
from qwenpose.schemas import (
    SCHEMA_INDICES,
    SCHEMA_TO_ID,
    UNION_KEYPOINTS,
    UNION_SIGMAS,
    UNION_TO_ID,
    coco_to_union,
    crowdpose_to_union,
    mpii_to_union,
)


class _TinyDataset(torch.utils.data.Dataset):
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> int:
        return index


class _DummyBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision_model = torch.nn.Module()
        self.vision_model.block = torch.nn.Module()
        self.vision_model.block.lora_A = torch.nn.Linear(4, 2, bias=False)
        self.language_model = torch.nn.Module()
        self.language_model.block = torch.nn.Module()
        self.language_model.block.lora_A = torch.nn.Linear(4, 2, bias=False)
        self.mlp1 = torch.nn.Linear(4, 4)
        self.base_weight = torch.nn.Parameter(torch.ones(1))


class _LoRAProjection(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_A = torch.nn.Linear(4, 2, bias=False)
        self.lora_B = torch.nn.Linear(2, 4, bias=False)


class _DummySelectiveBackbone(torch.nn.Module):
    def __init__(self, vision_blocks: int = 3, llm_layers: int = 4) -> None:
        super().__init__()
        self.vision_model = torch.nn.Module()
        self.vision_model.encoder = torch.nn.Module()
        self.vision_model.encoder.blocks = torch.nn.ModuleList()
        for _ in range(vision_blocks):
            block = torch.nn.Module()
            block.wqkv = _LoRAProjection()
            block.wo = _LoRAProjection()
            block.mlp = torch.nn.Module()
            block.mlp.fc0 = _LoRAProjection()
            block.mlp.fc1 = _LoRAProjection()
            self.vision_model.encoder.blocks.append(block)

        self.language_model = torch.nn.Module()
        self.language_model.model = torch.nn.Module()
        self.language_model.model.layers = torch.nn.ModuleList()
        for _ in range(llm_layers):
            layer = torch.nn.Module()
            layer.self_attn = torch.nn.Module()
            layer.self_attn.q_proj = _LoRAProjection()
            layer.self_attn.k_proj = _LoRAProjection()
            layer.self_attn.v_proj = _LoRAProjection()
            layer.self_attn.o_proj = _LoRAProjection()
            layer.mlp = torch.nn.Module()
            layer.mlp.gate_proj = _LoRAProjection()
            layer.mlp.up_proj = _LoRAProjection()
            layer.mlp.down_proj = _LoRAProjection()
            self.language_model.model.layers.append(layer)

        self.mlp1 = torch.nn.Linear(4, 4)
        self.base_weight = torch.nn.Parameter(torch.ones(1))


class _RawEagleTargets(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision_model = torch.nn.Module()
        self.vision_model.encoder = torch.nn.Module()
        block = torch.nn.Module()
        block.wqkv = torch.nn.Linear(4, 12)
        block.wo = torch.nn.Linear(4, 4)
        block.mlp = torch.nn.Module()
        block.mlp.fc0 = torch.nn.Linear(4, 8)
        block.mlp.fc1 = torch.nn.Linear(8, 4)
        self.vision_model.encoder.blocks = torch.nn.ModuleList([block])
        self.language_model = torch.nn.Module()
        self.language_model.model = torch.nn.Module()
        layer = torch.nn.Module()
        layer.self_attn = torch.nn.Module()
        layer.self_attn.q_proj = torch.nn.Linear(4, 4)
        layer.self_attn.v_proj = torch.nn.Linear(4, 4)
        self.language_model.model.layers = torch.nn.ModuleList([layer])


class _TinyEagle(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=8))
        self.vision_model = torch.nn.Module()
        self.vision_model.merge_kernel_size = (2, 2)
        self.vision_model.proj = torch.nn.Linear(8, 8, bias=False)
        self.mlp1 = torch.nn.Identity()

    def extract_feature(
        self,
        pixel_values: torch.Tensor,
        image_grid_hws: torch.Tensor,
    ) -> list[torch.Tensor]:
        del image_grid_hws
        return [self.vision_model.proj(sample) for sample in pixel_values]


class _VisionOnlyImageProcessor:
    _qwenpose_vision_only = True
    patch_size = 14

    def __init__(self) -> None:
        self.in_token_limit = 25600
        self.calls = 0
        self.seen_limits: list[int] = []

    def __call__(self, *, images, return_tensors: str):
        self.calls += 1
        self.seen_limits.append(int(self.in_token_limit))
        if return_tensors != "pt":
            raise AssertionError(return_tensors)
        batch = len(images)
        return {
            "pixel_values": torch.ones(batch, 3, 14, 14),
            "image_grid_hws": torch.full((batch, 2), 2, dtype=torch.long),
        }


class _CostDataset(torch.utils.data.Dataset):
    def __init__(self, sizes: list[tuple[int, int]]) -> None:
        self.records = [
            SimpleNamespace(
                width=width,
                height=height,
                boxes_xyxy=torch.zeros((1, 4)),
            )
            for width, height in sizes
        ]
        self.max_instances = 80

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        return {
            "record_index": index,
            "estimated_cost": estimate_locate_vision_tokens(
                record.width,
                record.height,
                4096,
            ) + 32,
        }


class MultiDatasetPoseTests(unittest.TestCase):
    def test_joint_pbd_conditioning_covers_all_pose_people_and_ref_target(self) -> None:
        all_boxes = torch.tensor(
            [
                [0.60, 0.60, 0.90, 0.95],
                [0.10, 0.05, 0.30, 0.50],
            ]
        )
        ref_boxes = torch.tensor(
            [
                [0.05, 0.10, 0.25, 0.90],
                [0.65, 0.10, 0.90, 0.95],
            ]
        )
        batch = {
            "task_ids": torch.tensor([0, 1]),
            "targets": [
                {
                    "boxes": all_boxes,
                    "ref_target": torch.tensor(-1),
                    "dataset": "coco",
                },
                {
                    "boxes": ref_boxes,
                    "ref_target": torch.tensor(1),
                    "dataset": "refhuman",
                },
            ],
        }

        boxes, mask, targets, metadata = prepare_joint_pbd_conditioning(
            batch,
            torch.device("cpu"),
            max_instances=30,
        )

        self.assertEqual(metadata["box_counts"].tolist(), [2, 1])
        self.assertEqual(tuple(metadata["gt_boxes"].shape), (3, 4))
        self.assertTrue(torch.equal(targets[0]["boxes"][0], all_boxes[1]))
        self.assertTrue(torch.equal(targets[0]["boxes"][1], all_boxes[0]))
        self.assertTrue(torch.equal(targets[1]["boxes"][0], ref_boxes[1]))
        self.assertEqual(int(targets[1]["ref_target"]), 0)
        self.assertEqual(mask.sum(dim=1).tolist(), [2, 1])
        self.assertEqual(tuple(boxes.shape), (2, 2, 4))

    def test_person_query_conditioning_keeps_all_refhuman_candidates(self) -> None:
        union = len(UNION_KEYPOINTS)
        boxes = torch.tensor(
            [[0.05, 0.10, 0.25, 0.90], [0.65, 0.10, 0.90, 0.95]]
        )
        target = {
            "boxes": boxes.clone(),
            "loss_boxes": boxes.clone(),
            "loss_areas": torch.tensor([0.16, 0.2375]),
            "keypoints": torch.zeros(2, union, 3),
            "keypoint_valid": torch.zeros(2, union, dtype=torch.bool),
            "visibility_valid": torch.zeros(2, union, dtype=torch.bool),
            "box_context_scale": torch.ones(2),
            "box_jitter_scale": torch.zeros(2),
            "box_jitter_shift": torch.zeros(2),
            "ref_target": torch.tensor(1),
        }
        placeholder, mask, selected = prepare_person_query_conditioning(
            [target], torch.tensor([1]), torch.device("cpu"), max_instances=80
        )
        self.assertEqual(tuple(placeholder.shape), (1, 1, 4))
        self.assertFalse(bool(mask.any()))
        self.assertEqual(int(selected[0]["boxes"].shape[0]), 2)
        self.assertEqual(int(selected[0]["ref_target"]), 1)

        outputs = {
            "pred_boxes": torch.tensor(
                [[[0.65, 0.10, 0.90, 0.95], [0.40, 0.40, 0.50, 0.50], [0.05, 0.10, 0.25, 0.90]]]
            ),
            "box_mask": torch.ones(1, 3, dtype=torch.bool),
        }
        aligned = align_targets_to_person_queries(outputs, selected, torch.tensor([1]))
        self.assertEqual(int(aligned[0]["ref_target"]), 0)
        self.assertEqual(aligned[0]["matched_gt_indices"].tolist(), [1, -1, 0])

    def test_global_person_queries_run_without_coordinate_prompts(self) -> None:
        model = QwenPoseModel(
            QwenPoseConfig(
                hidden_dim=32,
                external_dim=16,
                pose_decoder_layers=1,
                refinement_steps=1,
                human_decoder_layers=1,
                decoder_heads=4,
                pose_roi_size=4,
                use_refinement=False,
                pose_feature_channels=8,
                use_global_person_queries=True,
                num_person_queries=6,
            )
        ).eval()
        common = {
            "schema_ids": torch.tensor([SCHEMA_TO_ID["COCO17"]]),
            "task_ids": torch.tensor([1]),
            "images": None,
            "external_feature_map": torch.randn(1, 16, 8, 8),
            "target_boxes": None,
            "target_box_mask": None,
        }
        with torch.no_grad():
            first = model(external_text_embed=torch.zeros(1, 16), **common)
            second = model(external_text_embed=torch.randn(1, 16), **common)
        self.assertEqual(tuple(first["pred_boxes"].shape), (1, 6, 4))
        self.assertEqual(tuple(first["pred_keypoints"].shape[:2]), (1, 6))
        self.assertTrue(bool(first["box_mask"].all()))
        self.assertTrue(torch.allclose(first["pred_boxes"], second["pred_boxes"], atol=1e-6))
        self.assertTrue(torch.allclose(first["pred_keypoints"], second["pred_keypoints"], atol=1e-6))

    def test_progress_pose_includes_weighted_quality_head_contribution(self) -> None:
        postfix = build_progress_loss_postfix(
            {
                "loss_total": 1.0,
                "loss_person_confidence": 0.8,
                "person_quality_target_mean": 0.4,
                "person_confidence_std": 0.02,
            },
            LossWeights(person_confidence=0.5),
        )
        self.assertEqual(postfix["pose"], "0.400")
        self.assertNotIn("pconf", postfix)
        self.assertEqual(postfix["q"], "0.400")
        self.assertEqual(postfix["sstd"], "0.020")

    def test_progress_lm_reports_raw_and_weighted_box_loss(self) -> None:
        postfix = build_progress_loss_postfix(
            {
                "loss_total": 2.0,
                "loss_lm": 4.0,
                "loss_lm_weight": 0.02,
            },
            LossWeights(lm=0.02),
        )
        self.assertEqual(postfix["lm"], "0.080")
        self.assertEqual(postfix["lmraw"], "4.000")

    def test_progress_pose_dn_is_outer_weighted_contribution(self) -> None:
        postfix = build_progress_loss_postfix(
            {"loss_total": 2.0, "loss_keypoint_dn": 1.2},
            LossWeights(keypoint_dn=0.5),
        )
        self.assertEqual(postfix["posedn"], "0.600")

    def test_progress_splits_main_box_and_box_dn(self) -> None:
        postfix = build_progress_loss_postfix(
            {
                "loss_total": 2.5,
                "loss_box_l1": 0.2,
                "loss_box_dn": 1.5,
            },
            LossWeights(box_l1=5.0, box_dn=1.0),
        )
        self.assertEqual(postfix["box"], "1.000")
        self.assertEqual(postfix["boxdn"], "1.500")

    def test_progress_pose_reports_single_coarse_coord_objective(self) -> None:
        postfix = build_progress_loss_postfix(
            {
                "loss_total": 3.0,
                "loss_coord_coarse": 2.0,
            },
            LossWeights(
                oks=0.5,
                coord=3.0,
                image_coord=5.0,
                coarse_coord=0.5,
            ),
        )
        self.assertEqual(postfix["pose"], "1.000")

    def test_locate_generation_budget_is_task_aware(self) -> None:
        self.assertEqual(
            locate_generation_token_budget(8192, 80, task_id=0),
            656,
        )
        self.assertEqual(
            locate_generation_token_budget(8192, 80, task_id=1),
            24,
        )
        self.assertEqual(
            locate_generation_token_budget(512, 80, task_id=0),
            512,
        )

    def test_visualization_selector_skips_tiny_or_unannotated_people(self) -> None:
        union = len(UNION_KEYPOINTS)
        valid = torch.zeros(2, union, dtype=torch.bool)
        valid[1, :3] = True
        batch = {
            "targets": [
                {
                    "boxes": torch.tensor(
                        [[0.1, 0.1, 0.8, 0.8], [0.1, 0.1, 0.11, 0.11]]
                    ),
                    "keypoint_valid": valid,
                }
            ]
        }
        outputs = {"box_mask": torch.tensor([[True, True]])}
        self.assertIsNone(
            select_informative_visualization_sample(
                outputs,
                batch,
                min_gt_area_ratio=0.005,
            )
        )
        batch["targets"][0]["boxes"][1] = torch.tensor([0.1, 0.1, 0.4, 0.5])
        self.assertEqual(
            select_informative_visualization_sample(
                outputs,
                batch,
                min_gt_area_ratio=0.005,
            ),
            0,
        )

    def test_horizontal_flip_maps_boxes_and_swaps_left_right_joints(self) -> None:
        width, height = 100, 80
        matrix = torch.tensor(
            [[-1.0, 0.0, float(width)], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
        )
        boxes = torch.tensor([[0.10, 0.20, 0.40, 0.60]])
        flipped_boxes = transform_pose_boxes(boxes, matrix, width, height, clamp=True)
        self.assertTrue(
            torch.allclose(flipped_boxes, torch.tensor([[0.60, 0.20, 0.90, 0.60]]), atol=1e-6)
        )

        keypoints = torch.zeros(1, len(UNION_KEYPOINTS), 3)
        valid = torch.zeros(1, len(UNION_KEYPOINTS), dtype=torch.bool)
        visibility_valid = torch.zeros_like(valid)
        left = UNION_TO_ID["left_wrist"]
        right = UNION_TO_ID["right_wrist"]
        keypoints[0, left] = torch.tensor([0.20, 0.30, 1.0])
        keypoints[0, right] = torch.tensor([0.75, 0.65, 1.0])
        valid[0, [left, right]] = True
        visibility_valid[0, [left, right]] = True
        mapped, mapped_valid, mapped_visibility = transform_pose_keypoints(
            keypoints,
            valid,
            visibility_valid,
            matrix,
            width,
            height,
            horizontal_flip=True,
        )
        self.assertTrue(torch.allclose(mapped[0, left, :2], torch.tensor([0.25, 0.65]), atol=1e-6))
        self.assertTrue(torch.allclose(mapped[0, right, :2], torch.tensor([0.80, 0.30]), atol=1e-6))
        self.assertTrue(bool(mapped_valid[0, left] and mapped_valid[0, right]))
        self.assertTrue(bool(mapped_visibility[0, left] and mapped_visibility[0, right]))

    def test_affine_maps_keypoints_boxes_and_area_scale(self) -> None:
        width, height = 100, 200
        matrix = torch.tensor(
            [[1.5, 0.0, 10.0], [0.0, 1.5, 20.0], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
        )
        boxes = torch.tensor([[0.10, 0.10, 0.30, 0.30]])
        mapped_boxes = transform_pose_boxes(boxes, matrix, width, height, clamp=False)
        expected_boxes = torch.tensor([[0.25, 0.25, 0.55, 0.55]])
        self.assertTrue(torch.allclose(mapped_boxes, expected_boxes, atol=1e-6))

        keypoints = torch.zeros(1, len(UNION_KEYPOINTS), 3)
        valid = torch.zeros(1, len(UNION_KEYPOINTS), dtype=torch.bool)
        visibility_valid = torch.zeros_like(valid)
        wrist = UNION_TO_ID["left_wrist"]
        keypoints[0, wrist] = torch.tensor([0.20, 0.20, 1.0])
        valid[0, wrist] = True
        visibility_valid[0, wrist] = True
        mapped, _, _ = transform_pose_keypoints(
            keypoints,
            valid,
            visibility_valid,
            matrix,
            width,
            height,
            horizontal_flip=False,
        )
        self.assertTrue(torch.allclose(mapped[0, wrist, :2], torch.tensor([0.40, 0.40]), atol=1e-6))
        self.assertAlmostEqual(abs(float(torch.det(matrix[:2, :2]))), 2.25, places=6)

    def test_letterbox_scales_long_side_and_pads_annotations_to_square(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "wide.png"
            Image.new("RGB", (40, 20), (10, 20, 30)).save(image_path)
            union = len(UNION_KEYPOINTS)
            keypoints = torch.zeros(1, union, 3)
            valid = torch.zeros(1, union, dtype=torch.bool)
            visibility_valid = torch.zeros_like(valid)
            nose = UNION_TO_ID["nose"]
            keypoints[0, nose] = torch.tensor([0.5, 0.5, 1.0])
            valid[0, nose] = True
            visibility_valid[0, nose] = True
            record = PoseRecord(
                image_path=image_path,
                width=40,
                height=20,
                boxes_xyxy=torch.tensor([[0.25, 0.0, 0.75, 1.0]]),
                loss_boxes_xyxy=torch.tensor([[0.25, 0.0, 0.75, 1.0]]),
                loss_areas=torch.tensor([0.5]),
                keypoints=keypoints,
                keypoint_valid=valid,
                visibility_valid=visibility_valid,
                box_context_scale=torch.ones(1),
                box_jitter_scale=torch.zeros(1),
                box_jitter_shift=torch.zeros(1),
                schema="COCO17",
                task="ALL_POSE",
                prompt=ALL_POSE_PROMPT,
                dataset_name="coco",
                image_id="wide",
            )
            dataset = PoseRecordDataset(
                [record],
                load_image_tensors=False,
                load_vision_images=True,
                letterbox_size=800,
                letterbox_fill=127,
            )
            item = dataset[0]
            target = item["target"]
            self.assertEqual(tuple(item["vision_image"].shape), (3, 800, 800))
            self.assertEqual((target["width"], target["height"]), (800, 800))
            self.assertEqual((target["original_width"], target["original_height"]), (40, 20))
            self.assertEqual(target["letterbox_left"], 0)
            self.assertEqual(target["letterbox_top"], 200)
            self.assertEqual(target["letterbox_resized_width"], 800)
            self.assertEqual(target["letterbox_resized_height"], 400)
            self.assertTrue(
                torch.allclose(
                    target["boxes"],
                    torch.tensor([[0.25, 0.25, 0.75, 0.75]]),
                    atol=1e-6,
                )
            )
            self.assertTrue(
                torch.allclose(target["keypoints"][0, nose, :2], torch.tensor([0.5, 0.5]))
            )
            self.assertTrue(torch.allclose(target["loss_areas"], torch.tensor([0.25])))
            self.assertEqual(int(item["vision_image"][0, 0, 0]), 127)
            self.assertEqual(int(item["vision_image"][0, 400, 400]), 10)

    def test_dataset_reads_once_shares_augmented_image_and_drops_stage1_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            Image.new("RGB", (40, 30), (10, 20, 30)).save(image_path)
            union = len(UNION_KEYPOINTS)
            keypoints = torch.zeros(1, union, 3)
            valid = torch.zeros(1, union, dtype=torch.bool)
            visibility_valid = torch.zeros_like(valid)
            keypoints[0, UNION_TO_ID["nose"]] = torch.tensor([0.5, 0.5, 1.0])
            valid[0, UNION_TO_ID["nose"]] = True
            visibility_valid[0, UNION_TO_ID["nose"]] = True
            record = PoseRecord(
                image_path=image_path,
                width=40,
                height=30,
                boxes_xyxy=torch.tensor([[0.2, 0.2, 0.8, 0.8]]),
                loss_boxes_xyxy=torch.tensor([[0.2, 0.2, 0.8, 0.8]]),
                loss_areas=torch.tensor([0.36]),
                keypoints=keypoints,
                keypoint_valid=valid,
                visibility_valid=visibility_valid,
                box_context_scale=torch.ones(1),
                box_jitter_scale=torch.zeros(1),
                box_jitter_shift=torch.zeros(1),
                schema="COCO17",
                task="ALL_POSE",
                prompt=ALL_POSE_PROMPT,
                dataset_name="coco",
                image_id="sample",
            )
            dataset = PoseRecordDataset(
                [record],
                image_size=16,
                load_image_tensors=True,
                load_vision_images=True,
                augment_config=PoseAugmentConfig(enabled=False),
                use_prompts=False,
            )
            real_open = Image.open
            with mock.patch("qwenpose.data.Image.open", wraps=real_open) as open_mock:
                item = dataset[0]
            self.assertEqual(open_mock.call_count, 1)
            self.assertEqual(item["prompt"], "")
            self.assertEqual(tuple(item["image"].shape), (3, 16, 16))
            self.assertEqual(tuple(item["vision_image"].shape), (3, 30, 40))

            processor = _VisionOnlyImageProcessor()
            with mock.patch("qwenpose.eagle_lora.Image.open", side_effect=AssertionError("second read")):
                inputs = build_eagle_inputs(
                    processor,
                    [str(image_path)],
                    None,
                    torch.device("cpu"),
                    image_tensors=[item["vision_image"]],
                )
            self.assertIn("pixel_values", inputs)
            self.assertNotIn("input_ids", inputs)

            path_backed_dataset = PoseRecordDataset(
                [record],
                image_size=16,
                load_image_tensors=True,
                load_vision_images=False,
                augment_config=PoseAugmentConfig(enabled=False),
                use_prompts=False,
            )
            path_backed_item = path_backed_dataset[0]
            self.assertIsNone(path_backed_item["vision_image"])
            path_backed_batch = pose_collate([path_backed_item])
            self.assertIsNone(path_backed_batch["vision_images"])
            with mock.patch("qwenpose.eagle_lora.Image.open", wraps=real_open) as open_mock:
                inputs = build_eagle_inputs(
                    processor,
                    [str(image_path)],
                    None,
                    torch.device("cpu"),
                    image_tensors=path_backed_batch["vision_images"],
                )
            self.assertEqual(open_mock.call_count, 1)
            self.assertIn("pixel_values", inputs)

    def test_selective_loader_reads_only_vision_and_projector_tensors(self) -> None:
        try:
            from safetensors.torch import save_file
        except ImportError:
            self.skipTest("safetensors is not installed in this test environment")

        vision = torch.nn.Linear(3, 2)
        projector = torch.nn.Sequential(torch.nn.Linear(2, 2))
        expected = {
            "vision_model.weight": torch.randn_like(vision.weight),
            "vision_model.bias": torch.randn_like(vision.bias),
            "mlp1.0.weight": torch.randn_like(projector[0].weight),
            "mlp1.0.bias": torch.randn_like(projector[0].bias),
            "language_model.unused": torch.randn(5),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shard = "model-00001-of-00001.safetensors"
            save_file(expected, root / shard)
            weight_map = {key: shard for key in expected}
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"metadata": {}, "weight_map": weight_map}),
                encoding="utf-8",
            )
            _load_eagle_vision_projector_weights(root, vision, projector)

        self.assertTrue(torch.equal(vision.weight, expected["vision_model.weight"]))
        self.assertTrue(torch.equal(vision.bias, expected["vision_model.bias"]))
        self.assertTrue(torch.equal(projector[0].weight, expected["mlp1.0.weight"]))
        self.assertTrue(torch.equal(projector[0].bias, expected["mlp1.0.bias"]))

    def test_batch_token_limit_caps_per_image_processor_budget(self) -> None:
        processor = _VisionOnlyImageProcessor()
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            for index in range(2):
                path = Path(tmpdir) / f"budget-{index}.png"
                Image.new("RGB", (32, 32), (0, index * 20, 0)).save(path)
                paths.append(str(path))
            build_eagle_inputs(
                processor,
                paths,
                ["ignored", "ignored"],
                torch.device("cpu"),
                image_token_limit=4096,
                batch_token_limit=800,
            )
        self.assertEqual(processor.seen_limits, [350])
        self.assertEqual(processor.in_token_limit, 25600)

    def test_cross_rank_sampler_balances_vision_token_cost(self) -> None:
        sizes = [
            (224, 224), (280, 280), (336, 336), (392, 392),
            (448, 448), (504, 504), (560, 560), (616, 616),
            (672, 672), (728, 728), (784, 784), (840, 840),
            (896, 896), (952, 952), (1008, 1008), (1064, 1064),
        ]
        mixed = InterleavedPoseDataset([("pose", _CostDataset(sizes))], weights=None, seed=7)
        def collect(balance: bool) -> list[list[list[int]]]:
            result = []
            for rank in range(4):
                sampler = HomogeneousDatasetBatchSampler(
                    mixed,
                    batch_size=2,
                    seed=19,
                    rank=rank,
                    world_size=4,
                    shuffle=False,
                    fill_last=True,
                    balance_vision_tokens=balance,
                    vision_token_limit=4096,
                )
                result.append(list(sampler))
            return result

        balanced = collect(True)
        unbalanced = collect(False)
        self.assertTrue(all(len(batches) == len(balanced[0]) for batches in balanced))
        balanced_spreads = []
        unbalanced_spreads = []
        for step in range(len(balanced[0])):
            balanced_costs = [
                sum(int(mixed[index]["estimated_cost"]) for index in balanced[rank][step])
                for rank in range(4)
            ]
            unbalanced_costs = [
                sum(int(mixed[index]["estimated_cost"]) for index in unbalanced[rank][step])
                for rank in range(4)
            ]
            balanced_spreads.append(max(balanced_costs) - min(balanced_costs))
            unbalanced_spreads.append(max(unbalanced_costs) - min(unbalanced_costs))
            self.assertLessEqual(
                (max(balanced_costs) - min(balanced_costs)) / max(balanced_costs),
                0.20,
            )
        self.assertLess(max(balanced_spreads), max(unbalanced_spreads))

    def test_homogeneous_sampler_resume_offset_skips_indices_without_loading_samples(self) -> None:
        mixed = InterleavedPoseDataset(
            [("pose", _CostDataset([(224, 224)] * 12))],
            weights=None,
            seed=7,
        )
        sampler = HomogeneousDatasetBatchSampler(
            mixed,
            batch_size=2,
            seed=19,
            rank=0,
            world_size=1,
            shuffle=False,
            fill_last=True,
        )
        all_batches = list(sampler)
        sampler.set_start_batch(3)

        self.assertEqual(list(sampler), all_batches[3:])
        self.assertEqual(len(sampler), len(all_batches))
        sampler.set_epoch(1)
        self.assertEqual(sampler.start_batch, 0)

    def test_vision_only_processor_builds_no_language_inputs(self) -> None:
        processor = _VisionOnlyImageProcessor()
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            for index in range(2):
                path = Path(tmpdir) / f"image-{index}.png"
                Image.new("RGB", (32, 32), (index * 20, 0, 0)).save(path)
                paths.append(str(path))
            inputs = build_eagle_inputs(
                processor,
                paths,
                ["prompt must be ignored", "another prompt"],
                torch.device("cpu"),
                image_token_limit=128,
            )
        self.assertEqual(processor.calls, 1)
        self.assertEqual(processor.in_token_limit, 25600)
        self.assertNotIn("input_ids", inputs)
        self.assertNotIn("attention_mask", inputs)
        self.assertEqual(tuple(inputs["pixel_values"].shape), (2, 3, 14, 14))
        self.assertEqual(tuple(inputs["image_grid_hws"].shape), (2, 2))

    def test_vision_only_extractor_skips_language_model_and_backpropagates(self) -> None:
        backbone = _TinyEagle()
        extractor = EagleFeatureExtractor(
            backbone,
            refiner_layers=1,
            refiner_bottleneck_dim=4,
            refiner_init_scale=0.05,
            feature_source="vision_only",
        )

        def fail_language(*args, **kwargs):
            del args, kwargs
            raise AssertionError("vision_only unexpectedly called the language model")

        extractor.run_language_hidden = fail_language  # type: ignore[method-assign]
        inputs = {
            "pixel_values": torch.randn(2, 4, 8),
            "image_grid_hws": torch.tensor([[4, 4], [4, 4]]),
        }
        feature_map, text_embed = extractor(inputs, freeze_eagle=False)
        self.assertIsInstance(feature_map, MultiScaleSpatialFeatureBatch)
        p2, p3 = feature_map.levels
        self.assertEqual(tuple(p2.shape), (2, 2, 4, 4))
        self.assertEqual(p2.spatial_shapes.tolist(), [[4, 4], [4, 4]])
        self.assertEqual(tuple(p3.shape), (2, 8, 2, 2))
        self.assertEqual(p3.spatial_shapes.tolist(), [[2, 2], [2, 2]])
        self.assertEqual(tuple(text_embed.shape), (2, 8))
        self.assertEqual(float(text_embed.abs().sum()), 0.0)
        (p2.tensor.square().mean() + p3.tensor.square().mean()).backward()
        self.assertIsNotNone(backbone.vision_model.proj.weight.grad)

    def test_fast_stage_backbone_scope_only_opens_vision_lora(self) -> None:
        model = _DummyBackbone()
        counts = configure_backbone_train_scope(
            model, "vision_lora", train_projector=False
        )
        trainable = {name for name, param in model.named_parameters() if param.requires_grad}
        self.assertEqual(trainable, {"vision_model.block.lora_A.weight"})
        self.assertGreater(counts["vision_lora"], 0)
        self.assertEqual(counts["language_lora"], 0)
        self.assertEqual(counts["projector"], 0)

        counts = configure_backbone_train_scope(model, "all_lora", train_projector=True)
        trainable = {name for name, param in model.named_parameters() if param.requires_grad}
        self.assertEqual(
            trainable,
            {
                "vision_model.block.lora_A.weight",
                "language_model.block.lora_A.weight",
                "mlp1.weight",
                "mlp1.bias",
            },
        )
        self.assertGreater(counts["language_lora"], 0)
        self.assertGreater(counts["projector"], 0)

    def test_default_selective_vision_scope_opens_all_blocks_and_projector(self) -> None:
        model = _DummySelectiveBackbone(vision_blocks=27)
        counts = configure_backbone_train_scope(model, "selective_vision_lora")
        trainable = {name for name, param in model.named_parameters() if param.requires_grad}
        self.assertIn("mlp1.weight", trainable)
        self.assertIn("mlp1.bias", trainable)
        vision_lora = {name for name in trainable if "vision_model" in name}
        self.assertEqual(len(vision_lora), 27 * 4 * 2)
        self.assertTrue(any("blocks.0." in name for name in vision_lora))
        self.assertTrue(any("blocks.26." in name for name in vision_lora))
        self.assertGreater(counts["vision_lora"], 0)
        self.assertGreater(counts["projector"], 0)
        self.assertEqual(counts["language_lora"], 0)

        frozen_counts = configure_backbone_train_scope(model, "frozen")
        self.assertFalse(any(param.requires_grad for param in model.parameters()))
        self.assertEqual(frozen_counts, {"vision_lora": 0, "language_lora": 0, "projector": 0})

    def test_selective_lora_opens_only_requested_layers_and_modules(self) -> None:
        model = _DummySelectiveBackbone()
        counts = configure_backbone_train_scope(
            model,
            "selective_lora",
            train_projector=False,
            llm_layers="2-3",
            vision_layers="1-2",
            llm_modules="q_proj,v_proj",
            vision_modules="wqkv,wo",
        )
        trainable = {name for name, param in model.named_parameters() if param.requires_grad}
        self.assertTrue(trainable)
        for name in trainable:
            if "vision_model" in name:
                self.assertRegex(name, r"blocks\.[12]\.(wqkv|wo)\.lora_[AB]\.weight$")
            elif "language_model" in name:
                self.assertRegex(
                    name,
                    r"layers\.[23]\.self_attn\.(q_proj|v_proj)\.lora_[AB]\.weight$",
                )
            else:
                self.fail(f"Unexpected selective LoRA parameter: {name}")
        self.assertGreater(counts["vision_lora"], 0)
        self.assertGreater(counts["language_lora"], 0)
        self.assertEqual(counts["projector"], 0)

    def test_locate_coordinate_tokens_use_native_unpadded_format(self) -> None:
        self.assertEqual(_locate_coord_token(0.029, 1.0), "<29>")
        self.assertEqual(_locate_coord_token(0.0, 1.0), "<0>")
        self.assertEqual(_locate_coord_token(1.0, 1.0), "<1000>")

    def test_decoupled_selective_scopes_open_only_one_backbone_side(self) -> None:
        vision_model = _DummySelectiveBackbone()
        vision_counts = configure_backbone_train_scope(
            vision_model,
            "selective_vision_lora",
            train_projector=False,
            vision_layers="1-2",
            vision_modules="wqkv,wo",
        )
        vision_trainable = {
            name for name, param in vision_model.named_parameters() if param.requires_grad
        }
        self.assertTrue(vision_trainable)
        self.assertTrue(all("vision_model" in name for name in vision_trainable))
        self.assertGreater(vision_counts["vision_lora"], 0)
        self.assertEqual(vision_counts["language_lora"], 0)

        llm_model = _DummySelectiveBackbone()
        llm_counts = configure_backbone_train_scope(
            llm_model,
            "selective_llm_lora",
            train_projector=False,
            llm_layers="2-3",
            llm_modules="q_proj,v_proj",
        )
        llm_trainable = {
            name for name, param in llm_model.named_parameters() if param.requires_grad
        }
        self.assertTrue(llm_trainable)
        self.assertTrue(all("language_model" in name for name in llm_trainable))
        self.assertEqual(llm_counts["vision_lora"], 0)
        self.assertGreater(llm_counts["language_lora"], 0)

    def test_backbone_adapter_checkpoint_keeps_frozen_vision_lora(self) -> None:
        model = _DummySelectiveBackbone()
        configure_backbone_train_scope(
            model,
            "selective_llm_lora",
            train_projector=False,
            llm_layers="2-3",
            llm_modules="q_proj,v_proj",
        )
        state = backbone_adapter_state_dict(model)
        self.assertTrue(any("vision_model" in name and "lora_" in name for name in state))
        self.assertTrue(any("language_model" in name and "lora_" in name for name in state))
        self.assertIn("mlp1.weight", state)
        self.assertIn("mlp1.bias", state)
        self.assertNotIn("base_weight", state)

    def test_selective_lora_optimizer_separates_projector_group(self) -> None:
        model = torch.nn.Module()
        model.pose_weight = torch.nn.Parameter(torch.ones(1))
        model.backbone_model = _DummySelectiveBackbone()
        configure_backbone_train_scope(
            model.backbone_model,
            "selective_lora",
            train_projector=True,
            llm_layers="2-3",
            vision_layers="1-2",
            llm_modules="q_proj,v_proj",
            vision_modules="wqkv,wo",
        )
        args = SimpleNamespace(
            backbone="eagle",
            lr=2e-4,
            locate_llm_scale=0.01,
            locate_vision_scale=0.05,
            qwen_lora_lr_scale=1.0,
            qwen_vision_lr_scale=0.01,
            weight_decay=1e-4,
        )
        groups, printable = build_optimizer_param_groups(model, args)
        self.assertAlmostEqual(printable["pose"][2], 2e-4)
        self.assertAlmostEqual(printable["backbone_vision_lora"][2], 1e-5)
        self.assertAlmostEqual(printable["backbone_projector"][2], 1e-5)
        self.assertAlmostEqual(printable["backbone_lora"][2], 2e-6)
        decay_by_lr = [
            (float(group["lr"]), float(group["weight_decay"])) for group in groups
        ]
        pose_decay = next(decay for lr, decay in decay_by_lr if abs(lr - 2e-4) < 1e-12)
        vision_decays = [
            decay for lr, decay in decay_by_lr if abs(lr - 1e-5) < 1e-12
        ]
        llm_decay = next(decay for lr, decay in decay_by_lr if abs(lr - 2e-6) < 1e-12)
        self.assertAlmostEqual(pose_decay, 1e-4)
        self.assertIn(0.0, vision_decays)
        self.assertIn(1e-4, vision_decays)
        self.assertEqual(llm_decay, 0.0)

    def test_eagle_lora_targets_match_local_moonvit_names(self) -> None:
        targets, rank_pattern, alpha_pattern = find_eagle_lora_targets(_RawEagleTargets())
        self.assertIn("vision_model.encoder.blocks.0.wqkv", targets)
        self.assertIn("vision_model.encoder.blocks.0.wo", targets)
        self.assertIn("vision_model.encoder.blocks.0.mlp.fc0", targets)
        self.assertIn("vision_model.encoder.blocks.0.mlp.fc1", targets)
        self.assertIn("language_model.model.layers.0.self_attn.q_proj", targets)
        self.assertIn("language_model.model.layers.0.self_attn.v_proj", targets)
        self.assertEqual(rank_pattern["vision_model.encoder.blocks.0.wqkv"], -1)
        self.assertEqual(alpha_pattern["vision_model.encoder.blocks.0.wqkv"], -1)

    def test_feature_refiner_starts_as_exact_identity(self) -> None:
        refiner = QwenFeatureRefiner(8, num_layers=2, bottleneck_dim=4, init_scale=0.05)
        inputs = torch.randn(2, 8, 5, 5)
        outputs = refiner(inputs)
        self.assertTrue(torch.equal(outputs, inputs))

    def test_mpii_geometry_matches_mmpose(self) -> None:
        geometry = mpii_boxes_from_center_scale(
            center=[501.0, 401.0],
            scale=1.0,
            width=1000.0,
            height=1000.0,
        )
        self.assertIsNotNone(geometry)
        condition_box, loss_box, loss_area = geometry  # type: ignore[misc]
        self.assertEqual(loss_box, [400.0, 315.0, 600.0, 515.0])
        self.assertEqual(condition_box, [375.0, 290.0, 625.0, 540.0])
        self.assertAlmostEqual(loss_area, 200.0 * 200.0 * 0.53)

        boundary_geometry = mpii_boxes_from_center_scale(
            center=[1.0, 1.0],
            scale=1.0,
            width=100.0,
            height=100.0,
        )
        self.assertIsNotNone(boundary_geometry)
        boundary_condition, boundary_loss, _ = boundary_geometry  # type: ignore[misc]
        self.assertEqual(boundary_condition, [0.0, 0.0, 100.0, 100.0])
        self.assertEqual(boundary_loss, [-100.0, -85.0, 100.0, 115.0])

    def test_metric_geometry_uses_mmpose_box_and_loss_area(self) -> None:
        self.assertEqual(
            _mpii_bbox_from_center_scale([501.0, 401.0], 1.0),
            [375.0, 290.0, 625.0, 540.0],
        )
        union = len(UNION_KEYPOINTS)
        target = {
            "dataset": "coco",
            "image_id": "1",
            "schema": "COCO17",
            "width": 100.0,
            "height": 200.0,
            "boxes": torch.tensor([[0.1, 0.1, 0.9, 0.9]]),
            "loss_areas": torch.tensor([0.2]),
            "keypoints": torch.zeros(1, union, 3),
            "keypoint_valid": torch.zeros(1, union, dtype=torch.bool),
        }
        target["keypoint_valid"][:, :17] = True
        instances = targets_to_gt_instances([target])
        self.assertEqual(len(instances), 1)
        self.assertAlmostEqual(instances[0].area, 4000.0, delta=1e-3)

    def test_training_visualization_does_not_hide_mpii_coordinates(self) -> None:
        union = len(UNION_KEYPOINTS)
        mpii_indices = SCHEMA_INDICES["MPII16"]
        keypoints = torch.zeros(1, 1, union, 3)
        for offset, joint_idx in enumerate(mpii_indices.tolist()):
            keypoints[0, 0, joint_idx, 0] = 0.4 + 0.02 * (offset % 5)
            keypoints[0, 0, joint_idx, 1] = 0.5 + 0.03 * (offset // 5)
            keypoints[0, 0, joint_idx, 2] = 0.0
        schema_valid = torch.zeros(1, union, dtype=torch.bool)
        schema_valid[:, mpii_indices] = True
        outputs = {
            "boxes": torch.tensor([[[0.1, 0.1, 0.9, 0.9]]]),
            "pose_boxes": torch.tensor([[[0.1, 0.1, 0.9, 0.9]]]),
            "box_mask": torch.tensor([[True]]),
            "person_logits": torch.tensor([[10.0]]),
            "keypoints": keypoints,
            "keypoint_valid_mask": schema_valid,
        }
        target = {
            "boxes": torch.tensor([[0.1, 0.1, 0.9, 0.9]]),
            "keypoints": keypoints[0].clone(),
            "keypoint_valid": schema_valid[0].unsqueeze(0),
            "dataset": "mpii",
            "schema": "MPII16",
            "task": "ALL_POSE",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "source.png"
            output_path = Path(tmpdir) / "vis.png"
            Image.new("RGB", (128, 128), (255, 255, 255)).save(image_path)
            save_pose_visualization(
                outputs,
                {
                    "image_paths": [str(image_path)],
                    "targets": [target],
                    "source_datasets": ["mpii"],
                },
                output_path,
                draw_all_schema_keypoints=True,
            )
            pixels = list(Image.open(output_path).convert("RGB").getdata())
        self.assertIn((240, 70, 70), pixels)
        self.assertNotIn((255, 220, 0), pixels)
        self.assertNotIn((255, 140, 0), pixels)

    def test_training_visualization_keeps_matched_gt_when_person_score_is_low(self) -> None:
        union = len(UNION_KEYPOINTS)
        coco_indices = SCHEMA_INDICES["COCO17"]
        schema_valid = torch.zeros(1, union, dtype=torch.bool)
        schema_valid[:, coco_indices] = True
        keypoints = torch.zeros(1, 2, union, 3)
        for offset, joint_idx in enumerate(coco_indices.tolist()):
            keypoints[0, 0, joint_idx, 0] = 0.25 + 0.01 * (offset % 4)
            keypoints[0, 0, joint_idx, 1] = 0.30 + 0.02 * (offset // 4)
            keypoints[0, 1, joint_idx, :2] = 0.75
        outputs = {
            "boxes": torch.tensor(
                [[[0.15, 0.15, 0.55, 0.75], [0.60, 0.20, 0.90, 0.80]]]
            ),
            "box_mask": torch.tensor([[True, True]]),
            "person_logits": torch.tensor([[-10.0, -10.0]]),
            "keypoints": keypoints,
            "keypoint_valid_mask": schema_valid,
        }
        target_keypoints = keypoints[0].clone()
        target_valid = torch.zeros(2, union, dtype=torch.bool)
        target_valid[0, coco_indices] = True
        target = {
            "boxes": torch.tensor(
                [[0.10, 0.10, 0.50, 0.80], [0.60, 0.20, 0.90, 0.80]]
            ),
            "keypoints": target_keypoints,
            "keypoint_valid": target_valid,
            "matched_gt_indices": torch.tensor([0, -1]),
            "dataset": "coco",
            "schema": "COCO17",
            "task": "ALL_POSE",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "source.png"
            output_path = Path(tmpdir) / "vis.png"
            Image.new("RGB", (128, 128), (255, 255, 255)).save(image_path)
            save_pose_visualization(
                outputs,
                {
                    "image_paths": [str(image_path)],
                    "targets": [target],
                    "source_datasets": ["coco"],
                },
                output_path,
                draw_all_schema_keypoints=True,
            )
            pixels = list(Image.open(output_path).convert("RGB").getdata())
        self.assertIn((50, 200, 80), pixels)
        self.assertIn((80, 180, 255), pixels)
        self.assertIn((240, 70, 70), pixels)

    def test_crowdpose_visualization_uses_separate_head_joint(self) -> None:
        crowdpose_head = UNION_TO_ID["crowdpose_head"]
        head_top = UNION_TO_ID["head_top"]
        neck = UNION_TO_ID["neck"]
        edges = set(SCHEMA_POSE_EDGE_INDICES["CrowdPose14"])
        self.assertIn((crowdpose_head, neck), edges)
        self.assertNotIn((head_top, neck), edges)

    def test_all_annotated_joints_train_as_visible_and_crowdpose_head_is_separate(self) -> None:
        mpii_kpts, mpii_valid, mpii_vis_valid = mpii_to_union(
            [[10.0, 20.0]] * 16,
            [1] * 16,
            100.0,
            100.0,
        )
        self.assertTrue(torch.equal(mpii_vis_valid, mpii_valid))
        self.assertEqual(float(mpii_kpts[..., 2].sum()), 16.0)

        coco_flat = [10.0, 20.0, 1.0] + [0.0, 0.0, 0.0] * 16
        coco_kpts, coco_valid, coco_vis_valid = coco_to_union(coco_flat, 100.0, 100.0)
        self.assertTrue(bool(coco_valid[UNION_TO_ID["nose"]]))
        self.assertTrue(bool(coco_vis_valid[UNION_TO_ID["nose"]]))
        self.assertEqual(float(coco_kpts[UNION_TO_ID["nose"], 2]), 1.0)

        flat = []
        for idx in range(14):
            flat.extend([float(idx + 1), float(idx + 2), 1.0])
        crowd_kpts, crowd_valid, crowd_vis_valid = crowdpose_to_union(flat, 100.0, 100.0)
        head_idx = UNION_TO_ID["crowdpose_head"]
        old_head_top_idx = UNION_TO_ID["head_top"]
        self.assertTrue(bool(crowd_valid[head_idx]))
        self.assertTrue(bool(crowd_vis_valid[head_idx]))
        self.assertFalse(bool(crowd_valid[old_head_top_idx]))
        self.assertEqual(float(crowd_kpts[head_idx, 2]), 1.0)

        aic_flat = [10.0, 20.0, 2.0] + [0.0, 0.0, 3.0] * 13
        aic_kpts, aic_valid, aic_vis_valid = aic_to_union(aic_flat, 100.0, 100.0)
        aic_joint = int(SCHEMA_INDICES["AIC14"][0])
        self.assertTrue(bool(aic_valid[aic_joint]))
        self.assertTrue(bool(aic_vis_valid[aic_joint]))
        self.assertEqual(float(aic_kpts[aic_joint, 2]), 1.0)

    def test_schema_priors_have_correct_left_right_order(self) -> None:
        priors = build_schema_joint_priors("configs/schema_joint_priors.json")
        for schema_name, schema_idx in SCHEMA_TO_ID.items():
            for suffix in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle"):
                left = f"left_{suffix}"
                right = f"right_{suffix}"
                schema_joints = {
                    UNION_KEYPOINTS[int(index)]
                    for index in SCHEMA_INDICES[schema_name].tolist()
                }
                if left not in schema_joints or right not in schema_joints:
                    continue
                self.assertGreater(
                    float(priors[schema_idx, UNION_TO_ID[left], 0]),
                    float(priors[schema_idx, UNION_TO_ID[right], 0]),
                    msg=f"bad prior ordering for {schema_name}/{suffix}",
                )

    def test_auto_interleave_keeps_dataset_size_ratio(self) -> None:
        mixed = InterleavedPoseDataset(
            [("large", _TinyDataset(6)), ("small", _TinyDataset(2))],
            weights=None,
            seed=1,
        )
        self.assertEqual(mixed.weights, [6, 2])
        self.assertEqual(len(mixed), 8)

    def test_refhuman_uses_one_caption_per_instance_and_rotates_each_epoch(self) -> None:
        union = len(UNION_KEYPOINTS)

        def make_record(caption: str) -> PoseRecord:
            return PoseRecord(
                image_path=Path("/tmp/refhuman-caption-rotation.jpg"),
                width=100,
                height=100,
                boxes_xyxy=torch.tensor([[0.1, 0.1, 0.8, 0.9]]),
                loss_boxes_xyxy=torch.tensor([[0.1, 0.1, 0.8, 0.9]]),
                loss_areas=torch.tensor([0.56]),
                keypoints=torch.zeros(1, union, 3),
                keypoint_valid=torch.zeros(1, union, dtype=torch.bool),
                visibility_valid=torch.zeros(1, union, dtype=torch.bool),
                box_context_scale=torch.ones(1),
                box_jitter_scale=torch.zeros(1),
                box_jitter_shift=torch.zeros(1),
                schema="COCO17",
                task="REF_POSE",
                prompt=f'Locate a single person that matches: "{caption}".',
                ref_text=caption,
                ref_target=0,
                dataset_name="refhuman",
                image_id="same-person",
            )

        records = [make_record(text) for text in ("red shirt", "left person", "wearing a hat")]
        dataset = EpochRandomRefHumanDataset(
            records,
            captions_per_instance=1,
            seed=17,
            load_image_tensors=False,
            load_vision_images=False,
        )
        self.assertEqual(len(dataset), 1)

        captions = []
        for epoch in range(3):
            set_pose_dataset_epoch(dataset, epoch)
            first = dataset[0]
            second = dataset[0]
            self.assertEqual(first["ref_text"], second["ref_text"])
            self.assertIn(first["ref_text"], first["prompt"])
            captions.append(first["ref_text"])
        self.assertEqual(set(captions), {"red shirt", "left person", "wearing a hat"})

        set_pose_dataset_epoch(dataset, 3)
        self.assertEqual(dataset[0]["ref_text"], captions[0])

        replica = EpochRandomRefHumanDataset(
            records,
            captions_per_instance=1,
            seed=17,
            load_image_tensors=False,
            load_vision_images=False,
        )
        set_pose_dataset_epoch(replica, 1)
        self.assertEqual(replica[0]["ref_text"], captions[1])

        mixed = InterleavedPoseDataset([("refhuman", replica)], weights=None, seed=5)
        sampler = HomogeneousDatasetBatchSampler(mixed, batch_size=1, seed=5)
        sampler.set_epoch(2)
        self.assertEqual(replica[0]["ref_text"], captions[2])

    def test_manual_dataset_multipliers_define_epoch_and_allow_zero(self) -> None:
        mixed = InterleavedPoseDataset(
            [("coco", _TinyDataset(6)), ("refhuman", _TinyDataset(4)), ("off", _TinyDataset(9))],
            weights={"coco": 3.0, "refhuman": 0.5, "off": 0.0},
            seed=1,
        )
        self.assertEqual(mixed.multipliers, [3.0, 0.5, 0.0])
        self.assertEqual(mixed.weights, [18, 2, 0])
        self.assertEqual(len(mixed), 20)

    def test_fractional_multiplier_continues_with_next_half(self) -> None:
        mixed = InterleavedPoseDataset(
            [("refhuman", _CostDataset([(224, 224)] * 8))],
            weights={"refhuman": 0.5},
            seed=1,
        )
        sampler = HomogeneousDatasetBatchSampler(
            mixed,
            batch_size=1,
            seed=19,
            rank=0,
            world_size=1,
            shuffle=False,
            fill_last=False,
        )
        sampler.set_epoch(0)
        first = [mixed[index]["record_index"] for batch in sampler for index in batch]
        sampler.set_epoch(1)
        second = [mixed[index]["record_index"] for batch in sampler for index in batch]
        self.assertEqual(len(first), 4)
        self.assertEqual(len(second), 4)
        self.assertEqual(set(first).intersection(second), set())
        self.assertEqual(set(first).union(second), set(range(8)))

    def test_fractional_multiplier_keeps_odd_halves_disjoint(self) -> None:
        mixed = InterleavedPoseDataset(
            [("refhuman", _CostDataset([(224, 224)] * 9))],
            weights={"refhuman": 0.5},
            seed=1,
        )
        sampler = HomogeneousDatasetBatchSampler(
            mixed,
            batch_size=1,
            seed=19,
            rank=0,
            world_size=1,
            shuffle=False,
            fill_last=False,
        )
        sampler.set_epoch(0)
        first = [mixed[index]["record_index"] for batch in sampler for index in batch]
        sampler.set_epoch(1)
        second = [mixed[index]["record_index"] for batch in sampler for index in batch]
        self.assertEqual(len(first), 5)
        self.assertEqual(len(second), 5)
        self.assertEqual(set(first).intersection(second), set())
        self.assertEqual(set(first).union(second), set(range(9)))

    def test_parse_dataset_mix_weights_accepts_decimals_and_rejects_negative(self) -> None:
        self.assertEqual(
            parse_dataset_mix_weights("coco:3,mpii:0.5,refhuman:0"),
            {"coco": 3.0, "mpii": 0.5, "refhuman": 0.0},
        )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            parse_dataset_mix_weights("coco:-0.5")

    def test_keypoint_decode_is_regression_only(self) -> None:
        outputs = {"keypoints": torch.rand(1, 1, len(UNION_KEYPOINTS), 3)}
        self.assertIs(apply_keypoint_decode_mode(outputs, "regression"), outputs)
        with self.assertRaisesRegex(ValueError, "only regression"):
            apply_keypoint_decode_mode(outputs, "fused")

    def test_box_denoising_builds_positive_and_negative_pairs(self) -> None:
        target = {
            "boxes": torch.tensor(
                [
                    [0.10, 0.10, 0.30, 0.70],
                    [0.50, 0.20, 0.80, 0.90],
                ]
            )
        }
        torch.manual_seed(7)
        dn = prepare_box_denoising(
            [target],
            torch.device("cpu"),
            max_queries=8,
            max_groups=2,
            image_size=800,
        )
        self.assertIsNotNone(dn)
        assert dn is not None
        self.assertEqual(tuple(dn["dn_boxes"].shape), (1, 8, 4))
        self.assertEqual(int(dn["dn_labels"].sum().item()), 4)
        self.assertEqual(int(dn["dn_box_mask"].sum().item()), 8)
        self.assertEqual(dn["dn_source_indices"].tolist(), [[0, 0, 1, 1, 0, 0, 1, 1]])
        positive = dn["dn_labels"].bool()
        self.assertTrue(torch.allclose(dn["dn_target_boxes"][positive], target["boxes"].repeat(2, 1)))

    def test_box_denoising_uses_only_matched_generated_queries(self) -> None:
        target = {
            "boxes": torch.tensor(
                [
                    [0.10, 0.10, 0.30, 0.70],
                    [0.50, 0.20, 0.80, 0.90],
                ]
            ),
            "matched_gt_indices": torch.tensor([0, -1]),
        }
        dn = prepare_box_denoising(
            [target],
            torch.device("cpu"),
            max_queries=4,
            max_groups=2,
            image_size=800,
        )
        self.assertIsNotNone(dn)
        assert dn is not None
        self.assertEqual(int(dn["dn_box_mask"].sum().item()), 4)
        self.assertEqual(dn["dn_source_indices"].tolist(), [[0, 0, 0, 0]])
        positive_targets = dn["dn_target_boxes"][dn["dn_labels"].bool()]
        self.assertTrue(torch.allclose(positive_targets, target["boxes"][:1].repeat(2, 1)))

    def test_keypoint_denoising_uses_ks_ranges_and_matched_queries(self) -> None:
        union = len(UNION_KEYPOINTS)
        keypoints = torch.zeros(2, union, 3)
        keypoints[..., :2] = 0.5
        keypoints[..., 2] = 1.0
        valid = torch.zeros(2, union, dtype=torch.bool)
        valid[:, 0] = True
        target = {
            "boxes": torch.tensor(
                [[0.10, 0.10, 0.90, 0.90], [0.20, 0.20, 0.80, 0.80]]
            ),
            "loss_boxes": torch.tensor(
                [[0.10, 0.10, 0.90, 0.90], [0.20, 0.20, 0.80, 0.80]]
            ),
            "loss_areas": torch.tensor([0.64, 0.36]),
            "keypoints": keypoints,
            "keypoint_valid": valid,
            "matched_gt_indices": torch.tensor([0, -1]),
        }
        torch.manual_seed(23)
        dn = prepare_keypoint_denoising(
            [target],
            torch.device("cpu"),
            max_queries=4,
            max_groups=2,
            image_size=800,
        )
        self.assertIsNotNone(dn)
        assert dn is not None
        self.assertEqual(tuple(dn["keypoint_dn_noisy_keypoints"].shape), (1, 4, union, 2))
        self.assertEqual(dn["keypoint_dn_labels"].tolist(), [[1.0, 0.0, 1.0, 0.0]])
        self.assertEqual(dn["keypoint_dn_source_indices"].tolist(), [[0, 0, 0, 0]])
        self.assertEqual(dn["keypoint_dn_group_ids"].tolist(), [[0, 0, 1, 1]])

        noisy = dn["keypoint_dn_noisy_keypoints"][0, :, 0]
        clean = dn["keypoint_dn_target_keypoints"][0, :, 0, :2]
        d2 = ((noisy - clean) ** 2).sum(dim=-1)
        variance = float((2.0 * UNION_SIGMAS[0]) ** 2)
        ks = torch.exp(-d2 / (2.0 * 0.64 * variance))
        positive = dn["keypoint_dn_labels"][0].bool()
        self.assertTrue(bool(((ks[positive] >= 0.5) & (ks[positive] <= 1.0)).all()))
        self.assertTrue(bool(((ks[~positive] >= 0.1) & (ks[~positive] <= 0.5)).all()))

    def test_pose_dn_pairs_with_positive_box_dn_by_source_and_group(self) -> None:
        union = len(UNION_KEYPOINTS)
        keypoints = torch.zeros(1, union, 3)
        keypoints[..., :2] = 0.5
        valid = torch.zeros(1, union, dtype=torch.bool)
        valid[:, 0] = True
        target = {
            "boxes": torch.tensor([[0.1, 0.1, 0.9, 0.9]]),
            "keypoints": keypoints,
            "keypoint_valid": valid,
        }
        box_dn = prepare_box_denoising(
            [target], torch.device("cpu"), max_queries=4, max_groups=2
        )
        pose_dn = prepare_keypoint_denoising(
            [target], torch.device("cpu"), max_queries=4, max_groups=2
        )
        paired = pair_keypoint_denoising_with_box_denoising(box_dn, pose_dn)
        self.assertIsNotNone(paired)
        assert paired is not None
        self.assertEqual(
            paired["keypoint_dn_box_query_indices"].tolist(),
            [[0, 0, 2, 2]],
        )
        self.assertTrue(bool(paired["keypoint_dn_mask"].all()))

    def test_keypoint_dn_negative_has_no_coordinate_loss(self) -> None:
        union = len(UNION_KEYPOINTS)
        target = torch.zeros(1, 2, union, 3)
        target[..., :2] = 0.5
        valid = torch.zeros(1, 2, union, dtype=torch.bool)
        valid[..., 0] = True
        pred = target.clone()
        pred[..., 2] = 0.5
        pred[:, 1, 0, :2] = torch.tensor([0.0, 0.0])
        pred.requires_grad_(True)
        outputs = {
            "keypoint_dn_keypoints": pred,
            "keypoint_dn_mask": torch.tensor([[True, True]]),
            "keypoint_dn_labels": torch.tensor([[1.0, 0.0]]),
            "keypoint_dn_target_keypoints": target,
            "keypoint_dn_target_valid": valid,
            "keypoint_dn_target_boxes": torch.tensor(
                [[[0.1, 0.1, 0.9, 0.9], [0.1, 0.1, 0.9, 0.9]]]
            ),
            "keypoint_dn_target_areas": torch.tensor([[0.64, 0.64]]),
            "keypoint_dn_confidence_logits": torch.zeros(1, 2, union),
            "keypoint_dn_pose_quality_logits": torch.zeros(1, 2),
            "person_confidence_head_available": True,
        }
        graph_anchor = pred.sum() * 0.0
        _, parts = compute_keypoint_denoising_loss(
            outputs,
            LossWeights(),
            graph_anchor,
        )
        self.assertAlmostEqual(float(parts["loss_keypoint_dn_coord"].detach()), 0.0, places=7)
        self.assertAlmostEqual(float(parts["loss_keypoint_dn_image_coord"].detach()), 0.0, places=7)
        self.assertAlmostEqual(float(parts["loss_keypoint_dn_oks"].detach()), 0.0, places=7)

    def test_standard_oks_loss_matches_evaluator_kernel(self) -> None:
        pred = torch.tensor([[[0.60, 0.50, 0.0]]])
        target = torch.tensor([[[0.50, 0.50, 0.0]]])
        valid = torch.tensor([[True]])
        area = torch.tensor([0.25])
        sigma = torch.tensor([0.05])
        actual = per_joint_oks_loss(pred, target, valid, area, sigma)
        d2 = torch.tensor(0.10**2)
        variance = torch.tensor((2.0 * 0.05) ** 2)
        expected = 1.0 - torch.exp(-d2 / (2.0 * area[0] * variance))
        self.assertTrue(torch.allclose(actual[0, 0], expected, atol=1e-7))

    def test_final_pose_ignores_legacy_coord_and_duplicate_last_refine(self) -> None:
        union = len(UNION_KEYPOINTS)
        target_keypoints = torch.zeros(1, union, 3)
        target_keypoints[..., :2] = 0.5
        target_valid = torch.zeros(1, union, dtype=torch.bool)
        target_valid[:, 0] = True
        final = target_keypoints.view(1, 1, union, 3).clone()
        final[:, :, 0, :2] = 0.7
        auxiliary = final.clone()
        auxiliary[:, :, 0, :2] = 0.6
        outputs = {
            "keypoints": final,
            "refine_keypoints": [auxiliary, final.clone()],
            "box_mask": torch.tensor([[True]]),
            "keypoint_valid_mask": torch.ones(1, union, dtype=torch.bool),
        }
        target = {
            "boxes": torch.tensor([[0.1, 0.1, 0.9, 0.9]]),
            "loss_boxes": torch.tensor([[0.1, 0.1, 0.9, 0.9]]),
            "loss_areas": torch.tensor([0.64]),
            "keypoints": target_keypoints,
            "keypoint_valid": target_valid,
        }
        common = dict(
            oks=0.0,
            coord=999.0,
            image_coord=0.0,
            keypoint_confidence=0.0,
            person_confidence=0.0,
            ref_match=0.0,
            hard_joint=0.0,
            box_objectness=0.0,
            box_l1=0.0,
            box_giou=0.0,
            box_relative=0.0,
            box_dn=0.0,
            keypoint_dn=0.0,
            coarse_coord=0.0,
            deform_coord=0.0,
        )
        loss, parts = compute_pose_losses(
            outputs,
            [target],
            torch.tensor([0]),
            LossWeights(**common, refine_coords=(2.0, 999.0)),
        )
        self.assertGreater(float(loss.detach()), 0.0)
        self.assertNotIn("loss_coord_refine_2", parts)
        no_aux_loss, no_aux_parts = compute_pose_losses(
            outputs,
            [target],
            torch.tensor([0]),
            LossWeights(**common, refine_coords=(0.0, 999.0)),
        )
        self.assertGreater(float(no_aux_parts["loss_coord"].detach()), 0.0)
        self.assertAlmostEqual(float(no_aux_loss.detach()), 0.0, places=7)

    def test_pose_dn_uses_only_prefinal_refine_coordinate_auxiliary(self) -> None:
        union = len(UNION_KEYPOINTS)
        target = torch.zeros(1, 1, union, 3)
        target[..., :2] = 0.5
        valid = torch.zeros(1, 1, union, dtype=torch.bool)
        valid[..., 0] = True
        final = target.clone()
        final[..., 0, :2] = 0.7
        auxiliary = final.clone()
        auxiliary[..., 0, :2] = 0.6
        outputs = {
            "keypoint_dn_keypoints": final,
            "keypoint_dn_refine_keypoints": [auxiliary, final.clone()],
            "keypoint_dn_mask": torch.tensor([[True]]),
            "keypoint_dn_labels": torch.tensor([[1.0]]),
            "keypoint_dn_target_keypoints": target,
            "keypoint_dn_target_valid": valid,
            "keypoint_dn_target_boxes": torch.tensor([[[0.1, 0.1, 0.9, 0.9]]]),
            "keypoint_dn_target_areas": torch.tensor([[0.64]]),
        }
        graph_anchor = final.sum() * 0.0
        common = dict(
            oks=0.0,
            coord=999.0,
            image_coord=0.0,
            keypoint_confidence=0.0,
            person_confidence=0.0,
            coarse_coord=0.0,
            deform_coord=0.0,
        )
        auxiliary_loss, _ = compute_keypoint_denoising_loss(
            outputs,
            LossWeights(**common, refine_coords=(2.0, 999.0)),
            graph_anchor,
        )
        self.assertGreater(float(auxiliary_loss.detach()), 0.0)
        no_aux_loss, parts = compute_keypoint_denoising_loss(
            outputs,
            LossWeights(**common, refine_coords=(0.0, 999.0)),
            graph_anchor,
        )
        self.assertGreater(float(parts["loss_keypoint_dn_coord"].detach()), 0.0)
        self.assertAlmostEqual(float(no_aux_loss.detach()), 0.0, places=7)

    def test_joint_count_does_not_change_per_instance_loss(self) -> None:
        def run(valid_count: int) -> float:
            union = len(UNION_KEYPOINTS)
            pred = torch.zeros(1, 1, union, 3)
            pred[..., :2] = 0.51
            pred[..., 2] = 0.5
            gt = torch.zeros(1, union, 3)
            gt[..., :2] = 0.50
            valid = torch.zeros(1, union, dtype=torch.bool)
            valid[:, :valid_count] = True
            target = {
                "boxes": torch.tensor([[0.1, 0.1, 0.9, 0.9]]),
                "loss_boxes": torch.tensor([[0.1, 0.1, 0.9, 0.9]]),
                "loss_areas": torch.tensor([0.64]),
                "keypoints": gt,
                "keypoint_valid": valid,
                "visibility_valid": torch.zeros_like(valid),
            }
            outputs = {
                "keypoints": pred,
                "box_mask": torch.tensor([[True]]),
                "pose_boxes": torch.tensor([[[0.1, 0.1, 0.9, 0.9]]]),
                "keypoint_valid_mask": torch.ones(1, union, dtype=torch.bool),
                "schema_joint_indices": torch.arange(union).view(1, union),
                "schema_joint_valid": torch.ones(1, union, dtype=torch.bool),
            }
            _, parts = compute_pose_losses(
                outputs,
                [target],
                torch.tensor([0]),
                LossWeights(
                    oks=0.0,
                    coord=0.0,
                    image_coord=1.0,
                    vis=0.0,
                ),
            )
            return float(parts["loss_image_coord"])

        self.assertAlmostEqual(run(14), run(17), places=7)

    def test_person_confidence_uses_unmatched_generated_boxes_as_negatives(self) -> None:
        union = len(UNION_KEYPOINTS)
        logits = torch.zeros(1, 2, requires_grad=True)
        pred = torch.zeros(1, 2, union, 3)
        pred[0, 0, 0, :2] = torch.tensor([0.5, 0.5])
        target_keypoints = torch.zeros(2, union, 3)
        target_keypoints[0, 0, :2] = torch.tensor([0.5, 0.5])
        valid = torch.zeros(2, union, dtype=torch.bool)
        valid[0, 0] = True
        outputs = {
            "person_logits": logits,
            "box_mask": torch.tensor([[True, True]]),
            "keypoints": pred,
        }
        target = {
            "boxes": torch.tensor(
                [[0.0, 0.0, 1.0, 1.0], [0.1, 0.1, 0.2, 0.2]]
            ),
            "loss_areas": torch.tensor([1.0, 0.01]),
            "keypoints": target_keypoints,
            "keypoint_valid": valid,
            "matched_gt_indices": torch.tensor([0, -1]),
        }
        loss, parts = compute_person_confidence_rescue_loss(outputs, [target])
        self.assertAlmostEqual(float(parts["person_confidence_instances"]), 2.0)
        self.assertAlmostEqual(float(parts["person_quality_target_mean"]), 0.5, places=6)
        loss.backward()
        self.assertLess(float(logits.grad[0, 0]), 0.0)
        self.assertGreater(float(logits.grad[0, 1]), 0.0)

    def test_main_pose_loss_includes_person_confidence_quality(self) -> None:
        union = len(UNION_KEYPOINTS)
        logits = torch.zeros(1, 2, requires_grad=True)
        pred = torch.zeros(1, 2, union, 3)
        pred[0, 0, 0, :2] = torch.tensor([0.5, 0.5])
        target_keypoints = torch.zeros(2, union, 3)
        target_keypoints[0, 0, :2] = torch.tensor([0.5, 0.5])
        valid = torch.zeros(2, union, dtype=torch.bool)
        valid[0, 0] = True
        outputs = {
            "person_logits": logits,
            "box_mask": torch.tensor([[True, True]]),
            "keypoints": pred,
            "keypoint_valid_mask": torch.ones(1, union, dtype=torch.bool),
        }
        target = {
            "boxes": torch.tensor(
                [[0.0, 0.0, 1.0, 1.0], [0.1, 0.1, 0.2, 0.2]]
            ),
            "loss_areas": torch.tensor([1.0, 0.01]),
            "keypoints": target_keypoints,
            "keypoint_valid": valid,
            "matched_gt_indices": torch.tensor([0, -1]),
        }
        loss, parts = compute_pose_losses(
            outputs,
            [target],
            torch.tensor([0]),
            LossWeights(
                oks=0.0,
                coord=0.0,
                image_coord=0.0,
                keypoint_confidence=0.0,
                person_confidence=1.0,
            ),
        )
        self.assertIn("loss_person_confidence", parts)
        self.assertAlmostEqual(
            float(parts["loss_total"].detach()),
            float(parts["loss_person_confidence"].detach()),
            places=6,
        )
        loss.backward()
        self.assertLess(float(logits.grad[0, 0]), 0.0)
        self.assertGreater(float(logits.grad[0, 1]), 0.0)

    def test_missing_gt_joints_do_not_become_confidence_negatives(self) -> None:
        union = len(UNION_KEYPOINTS)
        target_keypoints = torch.zeros(1, union, 3)
        target_keypoints[0, 0, :2] = torch.tensor([0.5, 0.5])
        valid = torch.zeros(1, union, dtype=torch.bool)
        valid[0, 0] = True
        target = {
            "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            "loss_areas": torch.tensor([1.0]),
            "keypoints": target_keypoints,
            "keypoint_valid": valid,
        }

        def run(invalid_logit: float) -> float:
            pred = torch.zeros(1, 1, union, 3)
            pred[0, 0, 0, :2] = torch.tensor([0.5, 0.5])
            confidence_logits = torch.full((1, 1, union), invalid_logit)
            confidence_logits[0, 0, 0] = 0.0
            outputs = {
                "person_logits": torch.zeros(1, 1),
                "box_mask": torch.tensor([[True]]),
                "keypoints": pred,
                "keypoint_valid_mask": torch.ones(1, union, dtype=torch.bool),
                "keypoint_confidence_logits": confidence_logits,
            }
            _, parts = compute_pose_losses(
                outputs,
                [target],
                torch.tensor([0]),
                LossWeights(
                    oks=0.0,
                    coord=0.0,
                    image_coord=0.0,
                    keypoint_confidence=1.0,
                    person_confidence=0.0,
                ),
            )
            return float(parts["loss_keypoint_confidence"])

        self.assertAlmostEqual(run(-100.0), run(100.0), places=6)

    def test_standard_roi_position_encoding_separates_spatial_tokens(self) -> None:
        encoding = SinePositionEncoding(32)(4, 4, torch.device("cpu"))
        adjacent_similarity = torch.nn.functional.cosine_similarity(
            encoding[0], encoding[1], dim=0
        )
        corner_similarity = torch.nn.functional.cosine_similarity(
            encoding[0], encoding[-1], dim=0
        )
        self.assertLess(float(adjacent_similarity), 0.99)
        self.assertLess(float(corner_similarity), float(adjacent_similarity))
        self.assertGreaterEqual(int(torch.linalg.matrix_rank(encoding)), 6)

    def test_group_pose_dn_attention_mask_is_asymmetric(self) -> None:
        model = QwenPoseModel(
            QwenPoseConfig(
                hidden_dim=32,
                external_dim=16,
                pose_decoder_layers=1,
                refinement_steps=1,
                decoder_heads=4,
                pose_roi_size=4,
                pose_feature_channels=8,
                pose_coordinate_init="box_center",
            )
        )
        mask = model._build_group_pose_attention_mask(
            box_mask=torch.tensor([[True, True, True, True, True]]),
            dn_query_mask=torch.tensor([[False, True, True, True, True]]),
            dn_group_ids=torch.tensor([[-1, 0, 0, 1, 1]]),
            num_roles=2,
        )
        assert mask is not None
        first_role_first_head = mask[0]
        self.assertTrue(bool(first_role_first_head[0, 1]))
        self.assertFalse(bool(first_role_first_head[1, 0]))
        self.assertFalse(bool(first_role_first_head[1, 2]))
        self.assertTrue(bool(first_role_first_head[1, 3]))
        self.assertFalse(bool(torch.diagonal(first_role_first_head).any()))

    def test_anatomical_dynamic_reference_drives_grouped_decoder_and_gradient(self) -> None:
        model = QwenPoseModel(
            QwenPoseConfig(
                hidden_dim=32,
                external_dim=16,
                pose_decoder_layers=3,
                refinement_steps=1,
                human_decoder_layers=1,
                decoder_heads=4,
                box_condition_scale=1.0,
                pose_roi_size=4,
                use_refinement=False,
                pose_feature_channels=8,
                pose_coordinate_init="anatomical_dynamic",
                schema_joint_priors_path="configs/schema_joint_priors.json",
            )
        ).train()
        schema_id = SCHEMA_TO_ID["COCO17"]
        active = SCHEMA_INDICES["COCO17"]
        box = torch.tensor([[[0.1, 0.2, 0.9, 0.8]]])
        outputs = model(
            torch.tensor([schema_id]),
            torch.tensor([0]),
            external_feature_map=torch.randn(1, 16, 8, 8),
            external_text_embed=torch.zeros(1, 16),
            target_boxes=box,
            target_box_mask=torch.tensor([[True]]),
            pose_condition_box_mode="input",
        )
        priors = build_schema_joint_priors("configs/schema_joint_priors.json")[
            schema_id, active
        ]
        expected = box[0, 0, :2] + priors * (box[0, 0, 2:] - box[0, 0, :2])
        self.assertEqual(len(outputs["decoder_keypoints"]), 3)
        for decoder_output in outputs["decoder_keypoints"]:
            torch.testing.assert_close(
                decoder_output[0, 0, active, :2], expected, atol=1e-6, rtol=0.0
            )

        union = len(UNION_KEYPOINTS)
        target_keypoints = torch.zeros(1, union, 3)
        target_valid = torch.zeros(1, union, dtype=torch.bool)
        target_keypoints[0, active, :2] = (expected - 0.05).clamp(0.0, 1.0)
        target_keypoints[0, active, 2] = 1.0
        target_valid[0, active] = True
        target = {
            "boxes": box[0],
            "loss_boxes": box[0],
            "loss_areas": torch.tensor([0.48]),
            "keypoints": target_keypoints,
            "keypoint_valid": target_valid,
            "visibility_valid": target_valid.clone(),
        }
        loss, _ = compute_pose_losses(
            outputs,
            [target],
            torch.tensor([0]),
            LossWeights(
                oks=0.0,
                coord=0.0,
                image_coord=0.0,
                keypoint_confidence=0.0,
                person_confidence=0.0,
                ref_match=0.0,
                box_objectness=0.0,
                box_l1=0.0,
                box_giou=0.0,
                box_relative=0.0,
                box_dn=0.0,
                keypoint_dn=0.0,
                decoder_coords=(1.0,),
                coarse_coord=0.0,
                deform_coord=0.0,
                refine_coords=(),
            ),
        )
        loss.backward()
        assert model.reference_offset_head is not None
        reference_output = model.reference_offset_head.net[-1]
        self.assertIsInstance(reference_output, torch.nn.Linear)
        self.assertIsNotNone(reference_output.weight.grad)
        self.assertGreater(int(torch.count_nonzero(reference_output.weight.grad)), 0)

    def test_refinement_uses_box_relative_inverse_sigmoid_update(self) -> None:
        model = QwenPoseModel(
            QwenPoseConfig(
                hidden_dim=32,
                external_dim=16,
                pose_decoder_layers=1,
                refinement_steps=1,
                human_decoder_layers=1,
                decoder_heads=4,
                box_condition_scale=1.0,
                pose_roi_size=4,
                use_refinement=True,
                pose_feature_channels=8,
                pose_coordinate_init="box_center",
            )
        ).eval()
        assert model.refine_heads is not None
        refine_output = model.refine_heads[0].net[-1]
        self.assertIsInstance(refine_output, torch.nn.Linear)
        with torch.no_grad():
            refine_output.bias.copy_(torch.tensor([1.0, -1.0]))
            outputs = model(
                torch.tensor([SCHEMA_TO_ID["COCO17"]]),
                torch.tensor([0]),
                external_feature_map=torch.randn(1, 16, 8, 8),
                external_text_embed=torch.zeros(1, 16),
                target_boxes=torch.tensor([[[0.1, 0.2, 0.9, 0.8]]]),
                target_box_mask=torch.tensor([[True]]),
                pose_condition_box_mode="input",
            )
        expected_rel = torch.sigmoid(torch.tanh(torch.tensor([1.0, -1.0])) * 0.75)
        expected_xy = torch.tensor([0.1, 0.2]) + expected_rel * torch.tensor([0.8, 0.6])
        active = SCHEMA_INDICES["COCO17"]
        torch.testing.assert_close(
            outputs["keypoints"][0, 0, active, :2],
            expected_xy.expand(int(active.numel()), -1),
            atol=1e-5,
            rtol=0.0,
        )

    def test_schema_prior_buffer_is_checkpoint_self_contained(self) -> None:
        config = QwenPoseConfig(
            hidden_dim=32,
            external_dim=16,
            pose_decoder_layers=1,
            refinement_steps=1,
            decoder_heads=4,
            pose_roi_size=4,
            pose_feature_channels=8,
            pose_coordinate_init="schema_prior",
            schema_joint_priors_path="configs/schema_joint_priors.json",
        )
        source = QwenPoseModel(config)
        state = source.state_dict()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            target = QwenPoseModel(
                QwenPoseConfig(
                    **{
                        **config.__dict__,
                        "schema_joint_priors_path": "/missing/schema_joint_priors.json",
                    }
                )
            )
        target.load_state_dict(state, strict=True)
        self.assertTrue(
            torch.equal(source.schema_joint_priors, target.schema_joint_priors)
        )

    def test_box_center_mode_has_no_fixed_skeleton_checkpoint_tensor(self) -> None:
        model = QwenPoseModel(
            QwenPoseConfig(
                hidden_dim=32,
                external_dim=16,
                pose_decoder_layers=1,
                refinement_steps=1,
                decoder_heads=4,
                pose_roi_size=4,
                pose_feature_channels=8,
                pose_coordinate_init="box_center",
                schema_joint_priors_path="/not/read/in/box_center/mode.json",
            )
        )
        self.assertIsNone(model.schema_joint_priors)
        self.assertNotIn("schema_joint_priors", model.state_dict())

    def test_learned_spread_is_dispersed_trainable_and_non_anatomical(self) -> None:
        model = QwenPoseModel(
            QwenPoseConfig(
                hidden_dim=32,
                external_dim=16,
                pose_decoder_layers=1,
                refinement_steps=1,
                decoder_heads=4,
                pose_roi_size=4,
                pose_feature_channels=8,
                pose_coordinate_init="learned_spread",
                schema_joint_priors_path="/not/read/in/learned_spread/mode.json",
            )
        )
        expected = nonsemantic_joint_reference_points(len(UNION_KEYPOINTS))
        actual = model.joint_reference_logits.sigmoid().detach()
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
        self.assertGreater(float(actual.std(dim=0).min()), 0.1)
        self.assertGreaterEqual(float(actual.min()), 0.2)
        self.assertLessEqual(float(actual.max()), 0.8)
        self.assertTrue(model.joint_reference_logits.requires_grad)
        self.assertIsNone(model.schema_joint_priors)

    def test_learned_spread_reaches_coarse_output_and_receives_supervision(self) -> None:
        model = QwenPoseModel(
            QwenPoseConfig(
                hidden_dim=32,
                external_dim=16,
                pose_decoder_layers=1,
                refinement_steps=1,
                decoder_heads=4,
                box_condition_scale=1.0,
                pose_roi_size=4,
                use_refinement=False,
                pose_feature_channels=8,
                pose_coordinate_init="learned_spread",
            )
        ).train()
        union = len(UNION_KEYPOINTS)
        active = SCHEMA_INDICES["COCO17"]
        target_keypoints = torch.zeros(1, union, 3)
        target_valid = torch.zeros(1, union, dtype=torch.bool)
        target_keypoints[0, active, :2] = 0.1
        target_keypoints[0, active, 2] = 1.0
        target_valid[0, active] = True
        target = {
            "boxes": torch.tensor([[0.1, 0.2, 0.9, 0.8]]),
            "loss_boxes": torch.tensor([[0.1, 0.2, 0.9, 0.8]]),
            "loss_areas": torch.tensor([0.48]),
            "keypoints": target_keypoints,
            "keypoint_valid": target_valid,
            "visibility_valid": target_valid.clone(),
        }
        outputs = model(
            torch.tensor([SCHEMA_TO_ID["COCO17"]]),
            torch.tensor([0]),
            images=torch.rand(1, 3, 64, 64),
            external_feature_map=torch.randn(1, 16, 8, 8),
            external_text_embed=torch.zeros(1, 16),
            target_boxes=torch.tensor([[[0.1, 0.2, 0.9, 0.8]]]),
            target_box_mask=torch.tensor([[True]]),
        )
        coarse_xy = outputs["coarse_keypoints"][0, 0, active, :2]
        self.assertGreater(int(torch.unique(coarse_xy.detach(), dim=0).shape[0]), 8)
        loss, parts = compute_pose_losses(
            outputs,
            [target],
            torch.tensor([0]),
            LossWeights(
                oks=0.0,
                coord=0.0,
                image_coord=0.0,
                keypoint_confidence=0.0,
                person_confidence=0.0,
                box_objectness=0.0,
                box_l1=0.0,
                box_giou=0.0,
                box_relative=0.0,
                box_dn=0.0,
                keypoint_dn=0.0,
                coarse_coord=1.0,
                deform_coord=0.0,
            ),
        )
        loss.backward()
        self.assertGreater(float(parts["loss_coord_coarse"].detach()), 0.0)
        self.assertIsNotNone(model.joint_reference_logits.grad)
        self.assertGreater(
            int(torch.count_nonzero(model.joint_reference_logits.grad)),
            0,
        )

    def test_joint_soft_box_input_mode_opens_box_gradient(self) -> None:
        model = QwenPoseModel(
            QwenPoseConfig(
                hidden_dim=32,
                external_dim=16,
                human_decoder_layers=1,
                pose_decoder_layers=1,
                refinement_steps=1,
                decoder_heads=4,
                box_condition_scale=1.0,
                pose_roi_size=4,
                use_refinement=False,
                pose_feature_channels=8,
                pose_coordinate_init="learned_spread",
            )
        ).train()
        schema_ids = torch.tensor([SCHEMA_TO_ID["COCO17"]])
        task_ids = torch.tensor([1])
        images = torch.rand(1, 3, 64, 64)
        feature_map = torch.randn(1, 16, 8, 8)
        text_embed = torch.randn(1, 16)
        box_mask = torch.tensor([[True]])

        default_boxes = torch.tensor(
            [[[0.1, 0.2, 0.9, 0.8]]],
            requires_grad=True,
        )
        default_outputs = model(
            schema_ids,
            task_ids,
            images=images,
            external_feature_map=feature_map,
            external_text_embed=text_embed,
            target_boxes=default_boxes,
            target_box_mask=box_mask,
        )
        default_outputs["keypoints"].sum().backward()
        default_grad = default_boxes.grad
        self.assertTrue(
            default_grad is None or int(torch.count_nonzero(default_grad)) == 0
        )

        model.zero_grad(set_to_none=True)
        pbd_boxes = torch.tensor(
            [[[0.1, 0.2, 0.9, 0.8]]],
            requires_grad=True,
        )
        pbd_outputs = model(
            schema_ids,
            task_ids,
            images=images,
            external_feature_map=feature_map.detach(),
            external_text_embed=text_embed.detach(),
            target_boxes=pbd_boxes,
            target_box_mask=box_mask,
            pose_condition_box_mode="input",
        )
        pbd_outputs["keypoints"].sum().backward()
        self.assertIsNotNone(pbd_boxes.grad)
        self.assertGreater(int(torch.count_nonzero(pbd_boxes.grad)), 0)

    def test_box_center_deform_inherits_learned_coarse_reference(self) -> None:
        model = QwenPoseModel(
            QwenPoseConfig(
                hidden_dim=32,
                external_dim=16,
                pose_decoder_layers=1,
                refinement_steps=1,
                decoder_heads=4,
                box_condition_scale=1.0,
                pose_roi_size=4,
                use_refinement=False,
                pose_feature_channels=8,
                pose_coordinate_init="box_center",
            )
        ).eval()
        with torch.no_grad():
            coarse_output = model.coarse_xy_head.net[-1]
            self.assertIsInstance(coarse_output, torch.nn.Linear)
            coarse_output.bias.copy_(torch.logit(torch.tensor([0.25, 0.75])))
            outputs = model(
                torch.tensor([SCHEMA_TO_ID["COCO17"]]),
                torch.tensor([0]),
                images=torch.rand(1, 3, 64, 64),
                external_feature_map=torch.randn(1, 16, 8, 8),
                external_text_embed=torch.zeros(1, 16),
                target_boxes=torch.tensor([[[0.1, 0.2, 0.9, 0.8]]]),
                target_box_mask=torch.tensor([[True]]),
            )
        active = SCHEMA_INDICES["COCO17"]
        expected = torch.tensor([0.30, 0.65]).expand(int(active.numel()), -1)
        coarse_xy = outputs["coarse_keypoints"][0, 0, active, :2]
        deform_xy = outputs["deform_keypoints"][0, 0, active, :2]
        self.assertTrue(torch.allclose(coarse_xy, expected, atol=1e-6))
        self.assertTrue(torch.allclose(deform_xy, coarse_xy, atol=1e-6))

    def test_locate_pose_feature_stays_single_scale(self) -> None:
        model = QwenPoseModel(
            QwenPoseConfig(
                hidden_dim=32,
                external_dim=16,
                pose_decoder_layers=1,
                refinement_steps=1,
                human_decoder_layers=2,
                decoder_heads=4,
                pose_roi_size=4,
                pose_feature_channels=8,
                pose_coordinate_init="box_center",
                schema_joint_priors_path="configs/schema_joint_priors.json",
            )
        ).eval()
        with torch.no_grad():
            raw_batch = SpatialFeatureBatch.from_maps(
                [torch.rand(16, 5, 7), torch.rand(16, 3, 4)]
            )
            levels, spatial_shapes, valid_masks, roi_level, local_level = (
                model.build_locate_pose_features(raw_batch)
            )
        self.assertEqual([tuple(level.shape) for level in levels], [(2, 8, 5, 7)])
        self.assertEqual(spatial_shapes[0].tolist(), [[5, 7], [3, 4]])
        self.assertEqual(int(valid_masks[0][0].sum()), 35)
        self.assertEqual(int(valid_masks[0][1].sum()), 12)
        self.assertEqual(roi_level, 0)
        self.assertEqual(local_level, 0)
        self.assertEqual(float(levels[0][1, :, 3:, :].abs().sum()), 0.0)
        self.assertEqual(float(levels[0][1, :, :, 4:].abs().sum()), 0.0)
        parameter_names = [name for name, _ in model.named_parameters()]
        self.assertFalse(any("pyramid" in name for name in parameter_names))
        self.assertFalse(any("level_adapter" in name for name in parameter_names))
        self.assertFalse(hasattr(model, "spatial_injector"))
        self.assertFalse(any("simcc" in name.lower() for name, _ in model.named_parameters()))

    def test_native_grid_padding_does_not_change_another_sample(self) -> None:
        torch.manual_seed(23)
        model = QwenPoseModel(
            QwenPoseConfig(
                hidden_dim=32,
                external_dim=16,
                pose_decoder_layers=1,
                refinement_steps=1,
                human_decoder_layers=1,
                decoder_heads=4,
                pose_roi_size=4,
                pose_feature_channels=8,
                use_refinement=True,
                pose_coordinate_init="box_center",
            )
        ).eval()
        first_map = torch.randn(16, 5, 7)
        second_map = torch.randn(16, 9, 4)
        box = torch.tensor([[[0.1, 0.2, 0.9, 0.8]]])
        mask = torch.ones(1, 1, dtype=torch.bool)
        with torch.no_grad():
            single = model(
                schema_ids=torch.tensor([SCHEMA_TO_ID["COCO17"]]),
                task_ids=torch.tensor([0]),
                external_feature_map=SpatialFeatureBatch.from_maps([first_map]),
                external_text_embed=torch.zeros(1, 16),
                target_boxes=box,
                target_box_mask=mask,
            )
            batched = model(
                schema_ids=torch.tensor([SCHEMA_TO_ID["COCO17"], SCHEMA_TO_ID["COCO17"]]),
                task_ids=torch.tensor([0, 0]),
                external_feature_map=SpatialFeatureBatch.from_maps([first_map, second_map]),
                external_text_embed=torch.zeros(2, 16),
                target_boxes=box.expand(2, -1, -1),
                target_box_mask=mask.expand(2, -1),
            )
        torch.testing.assert_close(single["pred_boxes"][0], batched["pred_boxes"][0])
        torch.testing.assert_close(single["keypoints"][0], batched["keypoints"][0])

    def test_hierarchical_box_and_pose_dn_share_source_and_group(self) -> None:
        torch.manual_seed(11)
        model = QwenPoseModel(
            QwenPoseConfig(
                hidden_dim=32,
                external_dim=16,
                pose_decoder_layers=1,
                refinement_steps=1,
                human_decoder_layers=2,
                decoder_heads=4,
                pose_roi_size=4,
                pose_feature_channels=8,
                pose_coordinate_init="box_center",
                schema_joint_priors_path="configs/schema_joint_priors.json",
            )
        ).train()
        # DN samples must not depend on how many parameters the active visual
        # architecture happened to initialize before this point.
        torch.manual_seed(0)
        union = len(UNION_KEYPOINTS)
        keypoints = torch.zeros(1, union, 3)
        valid = torch.zeros(1, union, dtype=torch.bool)
        keypoints[0, 0] = torch.tensor([0.5, 0.5, 1.0])
        valid[0, 0] = True
        target = {
            "boxes": torch.tensor([[0.1, 0.1, 0.9, 0.9]]),
            "loss_boxes": torch.tensor([[0.1, 0.1, 0.9, 0.9]]),
            "loss_areas": torch.tensor([0.64]),
            "keypoints": keypoints,
            "keypoint_valid": valid,
            "visibility_valid": valid.clone(),
        }
        box_dn = prepare_box_denoising(
            [target], torch.device("cpu"), max_queries=4, max_groups=2
        )
        keypoint_dn = prepare_keypoint_denoising(
            [target], torch.device("cpu"), max_queries=4, max_groups=2
        )
        keypoint_dn = pair_keypoint_denoising_with_box_denoising(
            box_dn, keypoint_dn
        )
        assert box_dn is not None
        assert keypoint_dn is not None
        images = torch.rand(1, 3, 800, 800)
        external_feature_map = torch.randn(1, 16, 25, 25)
        external_text_embed = torch.zeros(1, 16)
        outputs = model(
            torch.tensor([SCHEMA_TO_ID["COCO17"]]),
            torch.tensor([0]),
            images=images,
            external_feature_map=external_feature_map,
            external_text_embed=external_text_embed,
            target_boxes=torch.tensor([[[0.1, 0.1, 0.9, 0.9]]]),
            target_box_mask=torch.tensor([[True]]),
            pose_condition_box_mode="input",
            **box_dn,
            **keypoint_dn,
        )
        self.assertEqual(tuple(outputs["pred_boxes"].shape), (1, 1, 4))
        self.assertEqual(tuple(outputs["dn_pred_boxes"].shape), (1, 4, 4))
        self.assertEqual(tuple(outputs["pred_keypoints"].shape[:2]), (1, 1))
        self.assertNotIn("dn_keypoints", outputs)
        self.assertEqual(tuple(outputs["keypoint_dn_keypoints"].shape[:2]), (1, 4))
        self.assertEqual(
            outputs["keypoint_dn_box_query_indices"].tolist(), [[0, 0, 2, 2]]
        )
        perturbed_keypoint_dn = dict(keypoint_dn)
        perturbed_keypoint_dn["keypoint_dn_noisy_keypoints"] = (
            1.0 - keypoint_dn["keypoint_dn_noisy_keypoints"]
        )
        perturbed_outputs = model(
            torch.tensor([SCHEMA_TO_ID["COCO17"]]),
            torch.tensor([0]),
            images=images,
            external_feature_map=external_feature_map,
            external_text_embed=external_text_embed,
            target_boxes=torch.tensor([[[0.1, 0.1, 0.9, 0.9]]]),
            target_box_mask=torch.tensor([[True]]),
            pose_condition_box_mode="input",
            **box_dn,
            **perturbed_keypoint_dn,
        )
        self.assertTrue(
            torch.allclose(
                outputs["pred_keypoints"],
                perturbed_outputs["pred_keypoints"],
                atol=1e-6,
            )
        )
        self.assertFalse(any("simcc" in key.lower() for key in outputs))
        active = SCHEMA_INDICES["COCO17"]
        expected_center = torch.full((int(active.numel()), 2), 0.5)
        self.assertTrue(
            torch.allclose(
                outputs["coarse_keypoints"][0, 0, active, :2],
                expected_center,
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                outputs["deform_keypoints"][0, 0, active, :2],
                expected_center,
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                outputs["keypoints"][0, 0, active, :2],
                expected_center,
                atol=1e-6,
            )
        )
        # Only the first COCO joint is annotated in this synthetic DN target;
        # undefined joints use the paired positive box-DN center rather than a
        # main-query box or an anatomical mean pose.
        paired_box = outputs["dn_pred_boxes"][0, 0]
        paired_center = ((paired_box[:2] + paired_box[2:]) * 0.5).expand(
            int(active.numel()) - 1, -1
        )
        self.assertTrue(
            torch.allclose(
                outputs["keypoint_dn_coarse_keypoints"][0, 0, active[1:], :2],
                paired_center,
                atol=1e-6,
            )
        )
        dn_only, _ = compute_keypoint_denoising_loss(
            outputs,
            LossWeights(person_confidence=0.0, keypoint_dn=1.0),
            outputs["keypoint_dn_graph_anchor"],
        )
        dn_only.backward(retain_graph=True)
        human_grad = model.human_box_heads[-1].net[-1].weight.grad
        self.assertTrue(
            human_grad is None or int(torch.count_nonzero(human_grad)) == 0
        )
        self.assertIsNotNone(model.keypoint_dn_type_embed.weight.grad)
        model.zero_grad(set_to_none=True)

        # Main pose supervision consumes the refined human box as geometry but
        # must not update the box decoder across the box->pose boundary.  Keep
        # this assertion beside the pose-DN boundary check so concatenating the
        # two query sets cannot accidentally reintroduce the gradient path.
        pose_only, _ = compute_pose_losses(
            outputs,
            [target],
            torch.tensor([0]),
            LossWeights(
                oks=0.0,
                coord=0.0,
                image_coord=1.0,
                vis=0.0,
                person_confidence=0.0,
                ref_match=0.0,
                box_objectness=0.0,
                box_l1=0.0,
                box_giou=0.0,
                box_relative=0.0,
                box_dn=0.0,
                keypoint_dn=0.0,
                coarse_coord=0.0,
                deform_coord=0.0,
                refine_coords=(),
            ),
        )
        pose_only.backward(retain_graph=True)
        human_grad = model.human_box_heads[-1].net[-1].weight.grad
        self.assertTrue(
            human_grad is None or int(torch.count_nonzero(human_grad)) == 0
        )
        self.assertIsNotNone(model.pose_xy_head.net[-1].weight.grad)
        model.zero_grad(set_to_none=True)

        loss, parts = compute_pose_losses(
            outputs,
            [target],
            torch.tensor([0]),
            LossWeights(
                oks=0.0,
                coord=0.0,
                image_coord=0.0,
                vis=0.1,
                person_confidence=0.0,
                keypoint_dn=1.0,
            ),
        )
        loss.backward()
        self.assertGreater(float(parts["loss_box_dn"].detach()), 0.0)
        self.assertGreater(float(parts["loss_keypoint_dn"].detach()), 0.0)
        self.assertIsNotNone(model.human_box_heads[-1].net[-1].weight.grad)
        self.assertIsNotNone(model.keypoint_dn_type_embed.weight.grad)

    def test_missing_keypoint_dn_keeps_type_embedding_in_loss_graph(self) -> None:
        model = QwenPoseModel(
            QwenPoseConfig(
                hidden_dim=32,
                external_dim=16,
                pose_decoder_layers=1,
                refinement_steps=1,
                human_decoder_layers=1,
                decoder_heads=4,
                pose_roi_size=4,
                use_refinement=False,
                pose_feature_channels=8,
                enable_keypoint_denoising=True,
            )
        ).train()
        union = len(UNION_KEYPOINTS)
        target = {
            "boxes": torch.tensor([[0.1, 0.1, 0.9, 0.9]]),
            "loss_boxes": torch.tensor([[0.1, 0.1, 0.9, 0.9]]),
            "loss_areas": torch.tensor([0.64]),
            "keypoints": torch.zeros(1, union, 3),
            "keypoint_valid": torch.zeros(1, union, dtype=torch.bool),
            "visibility_valid": torch.zeros(1, union, dtype=torch.bool),
        }
        outputs = model(
            torch.tensor([SCHEMA_TO_ID["COCO17"]]),
            torch.tensor([0]),
            images=torch.rand(1, 3, 64, 64),
            external_feature_map=torch.randn(1, 16, 8, 8),
            external_text_embed=torch.zeros(1, 16),
            target_boxes=torch.tensor([[[0.1, 0.1, 0.9, 0.9]]]),
            target_box_mask=torch.tensor([[True]]),
        )

        loss, parts = compute_pose_losses(
            outputs,
            [target],
            torch.tensor([0]),
            LossWeights(keypoint_dn=1.0),
        )
        loss.backward()

        self.assertEqual(float(parts["loss_keypoint_dn"].detach()), 0.0)
        self.assertIsNotNone(model.keypoint_dn_type_embed)
        self.assertIsNotNone(model.keypoint_dn_type_embed.weight.grad)
        self.assertEqual(
            int(torch.count_nonzero(model.keypoint_dn_type_embed.weight.grad)),
            0,
        )

    def test_coco_loader_keeps_zero_keypoint_people_and_skips_crowd(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "annotations").mkdir()
            (root / "train2017").mkdir()
            Image.new("RGB", (100, 80), (0, 0, 0)).save(root / "train2017" / "sample.jpg")
            visible = [0.0] * 51
            visible[0:3] = [50.0, 30.0, 2.0]
            payload = {
                "images": [{"id": 1, "file_name": "sample.jpg", "width": 100, "height": 80}],
                "annotations": [
                    {"id": 1, "image_id": 1, "bbox": [10, 10, 30, 50], "keypoints": visible, "num_keypoints": 1, "iscrowd": 0},
                    {"id": 2, "image_id": 1, "bbox": [50, 15, 20, 40], "keypoints": [0.0] * 51, "num_keypoints": 0, "iscrowd": 0},
                    {"id": 3, "image_id": 1, "bbox": [0, 0, 100, 80], "keypoints": [0.0] * 51, "num_keypoints": 0, "iscrowd": 1},
                ],
            }
            (root / "annotations" / "person_keypoints_train2017.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            records = load_coco_records(root, split="train2017")

        self.assertEqual(len(records), 1)
        self.assertEqual(int(records[0].boxes_xyxy.shape[0]), 2)
        self.assertEqual(records[0].keypoint_valid.any(dim=-1).tolist(), [True, False])

    def test_mpii_loader_keeps_all_annotated_people_for_detection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "annotations").mkdir()
            (root / "images").mkdir()
            Image.new("RGB", (120, 100), (0, 0, 0)).save(root / "images" / "sample.jpg")
            joints = [[30.0 + idx, 40.0 + idx] for idx in range(16)]
            payload = [
                {"image": "sample.jpg", "center": [45.0, 50.0], "scale": 0.4, "joints": joints, "joints_vis": [1] * 16},
                {"image": "sample.jpg", "center": [85.0, 50.0], "scale": 0.3, "joints": joints, "joints_vis": [0] * 16},
            ]
            (root / "annotations" / "mpii_train.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            records = load_mpii_records(root, split="train")

        self.assertEqual(len(records), 1)
        self.assertEqual(int(records[0].boxes_xyxy.shape[0]), 2)
        self.assertEqual(records[0].keypoint_valid.any(dim=-1).tolist(), [True, False])

    def test_aic_loader_keeps_human_boxes_without_pose_annotation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ann_dir = root / "ai_challenger_keypoint_train_annotations_20170909"
            image_dir = root / "ai_challenger_keypoint_train_20170902" / "keypoint_train_images_20170902"
            ann_dir.mkdir(parents=True)
            image_dir.mkdir(parents=True)
            Image.new("RGB", (100, 80), (0, 0, 0)).save(image_dir / "sample.jpg")
            annotation = [0.0] * 42
            annotation[0:3] = [30.0, 30.0, 1.0]
            payload = [
                {
                    "image_id": "sample",
                    "human_annotations": {
                        "human1": [10.0, 10.0, 40.0, 70.0],
                        "human2": [55.0, 10.0, 90.0, 70.0],
                    },
                    "keypoint_annotations": {"human1": annotation},
                }
            ]
            (ann_dir / "keypoint_train_annotations_20170909.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            records = load_aic_records(
                root,
                split="train",
                cache_dir=None,
                disable_cache=True,
                show_progress=False,
            )

        self.assertEqual(len(records), 1)
        self.assertEqual(int(records[0].boxes_xyxy.shape[0]), 2)
        self.assertEqual(records[0].keypoint_valid.any(dim=-1).tolist(), [True, False])

    def test_legacy_grounding_prompts_are_task_consistent(self) -> None:
        batch = {
            "prompts": [ALL_POSE_PROMPT, 'Locate a single person that matches the following description: "yellow shirt".'],
            "ref_texts": ["", "yellow shirt"],
            "task_ids": torch.tensor([0, 1]),
            "targets": [
                {
                    "width": 1000,
                    "height": 1000,
                    "boxes": torch.tensor([[0.6, 0.6, 0.8, 0.9], [0.1, 0.1, 0.3, 0.4]]),
                    "keypoints": torch.zeros(2, len(UNION_KEYPOINTS), 3),
                    "keypoint_valid": torch.zeros(2, len(UNION_KEYPOINTS), dtype=torch.bool),
                    "ref_target": torch.tensor(-1),
                },
                {
                    "width": 1000,
                    "height": 1000,
                    "boxes": torch.tensor([[0.2, 0.2, 0.5, 0.8], [0.7, 0.1, 0.9, 0.7]]),
                    "keypoints": torch.zeros(2, len(UNION_KEYPOINTS), 3),
                    "keypoint_valid": torch.zeros(2, len(UNION_KEYPOINTS), dtype=torch.bool),
                    "ref_target": torch.tensor(1),
                },
            ],
        }
        prompts = build_locate_generation_prompts(batch)
        self.assertEqual(prompts[0], ALL_POSE_PROMPT)
        self.assertEqual(
            prompts[1],
            'Locate the person that matches the following description: "yellow shirt".',
        )

    def test_batch_contract_accepts_legacy_mask_and_rejects_missing_mask(self) -> None:
        union = len(UNION_KEYPOINTS)
        target = {
            "boxes": torch.zeros(1, 4),
            "keypoints": torch.zeros(1, union, 3),
            "keypoint_mask": torch.ones(1, union, dtype=torch.bool),
            "ref_target": torch.tensor(-1),
            "dataset": "coco",
            "image_id": "legacy",
        }
        batch = {"targets": [target], "task_ids": torch.tensor([0])}
        validate_pose_batch_contract(batch)
        self.assertIs(target["keypoint_valid"], target["keypoint_mask"])

        missing = {
            "targets": [
                {
                    "boxes": torch.zeros(1, 4),
                    "keypoints": torch.zeros(1, union, 3),
                    "ref_target": torch.tensor(-1),
                    "dataset": "coco",
                    "image_id": "missing",
                }
            ],
            "task_ids": torch.tensor([0]),
        }
        with self.assertRaisesRegex(ValueError, "keypoint_valid"):
            validate_pose_batch_contract(missing)

    def test_locate_conditioning_keeps_unmatched_predictions_and_pose_masks(self) -> None:
        union = len(UNION_KEYPOINTS)
        valid = torch.zeros(2, union, dtype=torch.bool)
        valid[:, 0] = True
        keypoints = torch.zeros(2, union, 3)
        keypoints[0, 0] = torch.tensor([0.2, 0.3, 1.0])
        keypoints[1, 0] = torch.tensor([0.7, 0.3, 1.0])
        target = {
            "width": 1000,
            "height": 1000,
            "dataset": "coco",
            "schema": "COCO17",
            "image_id": "synthetic",
            "boxes": torch.tensor([[0.1, 0.1, 0.3, 0.5], [0.6, 0.1, 0.8, 0.5]]),
            "loss_boxes": torch.tensor([[0.1, 0.1, 0.3, 0.5], [0.6, 0.1, 0.8, 0.5]]),
            "loss_areas": torch.tensor([0.08, 0.08]),
            "keypoints": keypoints,
            "keypoint_valid": valid,
            "visibility_valid": valid.clone(),
            "box_context_scale": torch.ones(2),
            "box_jitter_scale": torch.zeros(2),
            "box_jitter_shift": torch.zeros(2),
            "ref_target": torch.tensor(-1),
        }
        response = (
            "<box><100><100><300><500></box>"
            "<box><600><100><800><500></box>"
            "<box><400><400><500><600></box>"
        )
        box_tensor, box_mask, aligned = prepare_locate_generated_box_conditioning_from_responses(
            [response],
            {"targets": [target], "task_ids": torch.tensor([0])},
            torch.device("cpu"),
            max_instances=80,
            match_iou_thresh=0.10,
            nms_iou_thresh=0.70,
            disable_pre_pose_nms=True,
        )
        self.assertEqual(int(box_mask.sum().item()), 3)
        self.assertEqual(tuple(box_tensor.shape), (1, 3, 4))
        self.assertEqual(int(aligned[0]["boxes"].shape[0]), 3)
        self.assertEqual(int((aligned[0]["matched_gt_indices"] >= 0).sum().item()), 2)
        unmatched = torch.nonzero(aligned[0]["matched_gt_indices"] < 0, as_tuple=False).flatten()
        self.assertEqual(int(unmatched.numel()), 1)
        self.assertFalse(bool(aligned[0]["keypoint_valid"][unmatched[0]].any().item()))
        self.assertNotIn("box_jitter_scale", aligned[0])
        self.assertNotIn("box_jitter_shift", aligned[0])

    def test_pre_pose_nms_can_be_disabled_for_overlapping_people(self) -> None:
        union = len(UNION_KEYPOINTS)
        target = {
            "width": 1000,
            "height": 1000,
            "dataset": "crowdpose",
            "schema": "CrowdPose14",
            "image_id": "overlap",
            "boxes": torch.tensor([[0.1, 0.1, 0.5, 0.8]]),
            "loss_boxes": torch.tensor([[0.1, 0.1, 0.5, 0.8]]),
            "loss_areas": torch.tensor([0.28]),
            "keypoints": torch.zeros(1, union, 3),
            "keypoint_valid": torch.zeros(1, union, dtype=torch.bool),
            "visibility_valid": torch.zeros(1, union, dtype=torch.bool),
            "box_context_scale": torch.ones(1),
            "box_jitter_scale": torch.zeros(1),
            "box_jitter_shift": torch.zeros(1),
            "ref_target": torch.tensor(-1),
        }
        response = "<box><100><100><500><800></box><box><120><110><520><810></box>"
        common = dict(
            responses=[response],
            batch={"targets": [target], "task_ids": torch.tensor([0])},
            device=torch.device("cpu"),
            max_instances=80,
            match_iou_thresh=0.10,
            nms_iou_thresh=0.70,
        )
        _, mask_without_nms, _ = prepare_locate_generated_box_conditioning_from_responses(
            **common, disable_pre_pose_nms=True
        )
        _, mask_with_nms, _ = prepare_locate_generated_box_conditioning_from_responses(
            **common, disable_pre_pose_nms=False
        )
        self.assertEqual(int(mask_without_nms.sum().item()), 2)
        self.assertEqual(int(mask_with_nms.sum().item()), 1)

    def test_zero_pose_queries_do_not_change_pose_loss_denominator(self) -> None:
        union = len(UNION_KEYPOINTS)
        valid = torch.zeros(2, union, dtype=torch.bool)
        valid[0, 0] = True
        target = {
            "boxes": torch.tensor([[0.1, 0.1, 0.4, 0.8], [0.6, 0.1, 0.9, 0.8]]),
            "loss_boxes": torch.tensor([[0.1, 0.1, 0.4, 0.8], [0.6, 0.1, 0.9, 0.8]]),
            "loss_areas": torch.tensor([0.21, 0.21]),
            "keypoints": torch.zeros(2, union, 3),
            "keypoint_valid": valid,
            "visibility_valid": torch.zeros_like(valid),
        }
        target["keypoints"][0, 0, :2] = 0.5

        def run(second_value: float) -> float:
            pred = torch.zeros(1, 2, union, 3)
            pred[0, 0, 0, :2] = 0.6
            pred[0, 1, :, :2] = second_value
            outputs = {
                "keypoints": pred,
                "box_mask": torch.tensor([[True, True]]),
                "pose_boxes": target["boxes"].unsqueeze(0),
                "keypoint_valid_mask": torch.ones(1, union, dtype=torch.bool),
                "schema_joint_indices": torch.arange(union).view(1, union),
                "schema_joint_valid": torch.ones(1, union, dtype=torch.bool),
            }
            _, parts = compute_pose_losses(
                outputs,
                [target],
                torch.tensor([0]),
                LossWeights(oks=0.0, coord=0.0, image_coord=1.0, vis=0.0),
            )
            return float(parts["loss_image_coord"])

        self.assertAlmostEqual(run(0.0), run(100.0), places=7)

    def test_refhuman_direct_grounding_keeps_one_target_box(self) -> None:
        union = len(UNION_KEYPOINTS)
        target = {
            "width": 1000,
            "height": 1000,
            "dataset": "refhuman",
            "schema": "COCO17",
            "image_id": "ref-1",
            "boxes": torch.tensor(
                [
                    [0.10, 0.10, 0.35, 0.85],
                    [0.55, 0.10, 0.85, 0.85],
                ]
            ),
            "loss_boxes": torch.tensor(
                [
                    [0.10, 0.10, 0.35, 0.85],
                    [0.55, 0.10, 0.85, 0.85],
                ]
            ),
            "loss_areas": torch.tensor([0.1875, 0.225]),
            "keypoints": torch.zeros(2, union, 3),
            "keypoint_valid": torch.zeros(2, union, dtype=torch.bool),
            "visibility_valid": torch.zeros(2, union, dtype=torch.bool),
            "box_context_scale": torch.ones(2),
            "ref_target": torch.tensor(1),
        }
        response = "<box><550><100><850><850></box>"
        boxes, mask, aligned = prepare_locate_generated_box_conditioning_from_responses(
            [response],
            {"targets": [target], "task_ids": torch.tensor([1])},
            torch.device("cpu"),
            max_instances=80,
            match_iou_thresh=0.10,
            nms_iou_thresh=0.70,
            disable_pre_pose_nms=True,
        )
        self.assertEqual(tuple(boxes.shape), (1, 1, 4))
        self.assertEqual(int(mask.sum().item()), 1)
        self.assertEqual(aligned[0]["matched_gt_indices"].tolist(), [1])
        self.assertEqual(int(aligned[0]["ref_target"].item()), 0)

    def test_refhuman_empty_direct_result_can_fallback_to_all_people(self) -> None:
        union = len(UNION_KEYPOINTS)
        target = {
            "width": 1000,
            "height": 1000,
            "dataset": "refhuman",
            "schema": "COCO17",
            "image_id": "ref-fallback",
            "boxes": torch.tensor(
                [[0.10, 0.10, 0.35, 0.85], [0.55, 0.10, 0.85, 0.85]]
            ),
            "loss_boxes": torch.tensor(
                [[0.10, 0.10, 0.35, 0.85], [0.55, 0.10, 0.85, 0.85]]
            ),
            "loss_areas": torch.tensor([0.1875, 0.225]),
            "keypoints": torch.zeros(2, union, 3),
            "keypoint_valid": torch.zeros(2, union, dtype=torch.bool),
            "visibility_valid": torch.zeros(2, union, dtype=torch.bool),
            "box_context_scale": torch.ones(2),
            "ref_target": torch.tensor(1),
        }
        response = (
            REFHUMAN_FALLBACK_MARKER
            + "\n<box><100><100><350><850></box>"
            + "<box><550><100><850><850></box>"
        )
        boxes, mask, aligned = prepare_locate_generated_box_conditioning_from_responses(
            [response],
            {"targets": [target], "task_ids": torch.tensor([1])},
            torch.device("cpu"),
            max_instances=80,
            match_iou_thresh=0.10,
            nms_iou_thresh=0.70,
            disable_pre_pose_nms=True,
        )
        self.assertEqual(tuple(boxes.shape), (1, 2, 4))
        self.assertEqual(int(mask.sum().item()), 2)
        self.assertEqual(aligned[0]["matched_gt_indices"].tolist(), [-1, 1])
        self.assertEqual(int(aligned[0]["ref_target"].item()), 1)

    def test_refhuman_box_refinement_safety_is_eval_local(self) -> None:
        input_boxes = torch.tensor(
            [[[0.10, 0.10, 0.40, 0.90]], [[0.10, 0.10, 0.40, 0.90]]]
        )
        drifted = torch.tensor(
            [[[0.60, 0.10, 0.90, 0.90]], [[0.60, 0.10, 0.90, 0.90]]]
        )
        safe, fallback = apply_refhuman_box_refinement_safety(
            drifted,
            input_boxes,
            torch.tensor([[True], [True]]),
            torch.tensor([1, 0]),
        )
        self.assertTrue(torch.equal(safe[0], input_boxes[0]))
        self.assertTrue(torch.equal(safe[1], drifted[1]))
        self.assertEqual(fallback.tolist(), [[True], [False]])

    def test_refhuman_match_loss_is_independent_from_pose_quality(self) -> None:
        outputs = {
            "ref_logits": torch.tensor([[-1.0, 3.0, 0.5]], requires_grad=True),
            "box_mask": torch.tensor([[True, True, True]]),
            "keypoints": torch.zeros(1, 3, len(UNION_KEYPOINTS), 3),
        }
        target = {"ref_target": torch.tensor(1)}
        loss, parts = compute_refhuman_match_loss(
            outputs,
            [target],
            torch.tensor([1]),
        )
        loss.backward()
        self.assertGreater(float(loss.detach()), 0.0)
        self.assertEqual(float(parts["ref_match_accuracy"]), 1.0)
        self.assertGreater(float(parts["ref_match_margin"]), 0.0)
        self.assertIsNotNone(outputs["ref_logits"].grad)

    def test_refhuman_prediction_ranks_by_ref_match_then_pose_quality(self) -> None:
        union = len(UNION_KEYPOINTS)
        outputs = {
            "person_logits": torch.logit(torch.tensor([[0.95, 0.30]])),
            "pose_quality_logits": torch.logit(torch.tensor([[0.90, 0.40]])),
            "person_confidence_head_available": True,
            "ref_logits": torch.tensor([[-2.0, 3.0]]),
            "box_mask": torch.tensor([[True, True]]),
            "boxes": torch.tensor(
                [[[0.1, 0.1, 0.4, 0.8], [0.6, 0.1, 0.9, 0.8]]]
            ),
            "pose_boxes": torch.tensor(
                [[[0.1, 0.1, 0.4, 0.8], [0.6, 0.1, 0.9, 0.8]]]
            ),
            "keypoints": torch.zeros(1, 2, union, 3),
            "keypoint_valid_mask": torch.ones(1, union, dtype=torch.bool),
        }
        batch = {
            "task_ids": torch.tensor([1]),
            "prompts": ['The downstream target description is: "right person".'],
            "image_paths": ["sample.jpg"],
            "targets": [
                {
                    "dataset": "refhuman",
                    "image_id": "1",
                    "schema": "COCO17",
                    "width": 100,
                    "height": 100,
                    "boxes": torch.zeros(0, 4),
                }
            ],
        }
        args = SimpleNamespace(
            score_threshold=0.0,
            max_predictions_per_image=10,
            post_pose_nms_iou_thresh=0.95,
            ref_pose_quality_alpha=0.25,
        )
        rows = tensor_to_prediction_rows(outputs, batch, args)
        prediction = rows[0]["predictions"][0]
        self.assertEqual(prediction["query"], 1)
        self.assertGreater(prediction["ref_score"], 0.9)
        self.assertAlmostEqual(prediction["pose_quality_score"], 0.4, places=6)

    def test_refhuman_direct_grounding_does_not_rerank_its_only_box(self) -> None:
        union = len(UNION_KEYPOINTS)
        outputs = {
            "person_logits": torch.logit(torch.tensor([[0.9]])),
            "pose_quality_logits": torch.logit(torch.tensor([[0.4]])),
            "person_confidence_head_available": True,
            "ref_logits": torch.tensor([[-3.0]]),
            "box_mask": torch.tensor([[True]]),
            "boxes": torch.tensor([[[0.1, 0.1, 0.4, 0.8]]]),
            "pose_boxes": torch.tensor([[[0.1, 0.1, 0.4, 0.8]]]),
            "keypoints": torch.zeros(1, 1, union, 3),
            "keypoint_valid_mask": torch.ones(1, union, dtype=torch.bool),
        }
        batch = {
            "task_ids": torch.tensor([1]),
            "prompts": ['Locate the person described as "left person".'],
            "image_paths": ["sample.jpg"],
            "targets": [
                {
                    "dataset": "refhuman",
                    "image_id": "direct-1",
                    "schema": "COCO17",
                    "width": 100,
                    "height": 100,
                    "boxes": torch.zeros(0, 4),
                }
            ],
        }
        args = SimpleNamespace(
            score_threshold=0.0,
            max_predictions_per_image=10,
            post_pose_nms_iou_thresh=0.95,
            ref_pose_quality_alpha=0.25,
        )
        prediction = tensor_to_prediction_rows(outputs, batch, args)[0]["predictions"][0]
        self.assertEqual(prediction["query"], 0)
        self.assertEqual(prediction["ref_grounding_mode"], "direct")
        self.assertAlmostEqual(prediction["score"], 0.4, places=6)
        self.assertLess(prediction["ref_score"], 0.1)

    def test_post_pose_nms_only_removes_near_duplicates(self) -> None:
        boxes = torch.tensor(
            [
                [0.10, 0.10, 0.50, 0.80],
                [0.11, 0.10, 0.51, 0.80],
                [0.30, 0.10, 0.70, 0.80],
            ]
        )
        scores = torch.tensor([0.9, 0.8, 0.7])
        keep = nms_box_indices_xyxy(boxes, scores, iou_thresh=0.95, max_boxes=10)
        self.assertEqual(keep, [0, 2])

    def test_prediction_rows_use_only_person_quality_for_ranking(self) -> None:
        union = len(UNION_KEYPOINTS)
        keypoints = torch.zeros(1, 2, union, 3)
        keypoints[0, 0, 0, 2] = 0.9
        keypoints[0, 1, 0, 2] = 0.2
        schema_valid = torch.zeros(1, union, dtype=torch.bool)
        schema_valid[0, 0] = True
        outputs = {
            "person_logits": torch.logit(torch.tensor([[0.2, 0.8]])),
            "person_confidence_head_available": True,
            "ref_logits": torch.zeros(1, 2),
            "box_mask": torch.tensor([[True, True]]),
            "boxes": torch.tensor([[[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]]),
            "pose_boxes": torch.tensor([[[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]]),
            "keypoints": keypoints,
            "keypoint_valid_mask": schema_valid,
        }
        batch = {
            "task_ids": torch.tensor([0]),
            "prompts": [ALL_POSE_PROMPT],
            "image_paths": ["sample.jpg"],
            "targets": [
                {
                    "dataset": "coco",
                    "image_id": "1",
                    "schema": "COCO17",
                    "width": 100,
                    "height": 100,
                    "boxes": torch.zeros(0, 4),
                }
            ],
        }
        args = SimpleNamespace(
            score_threshold=0.0,
            max_predictions_per_image=10,
            post_pose_nms_iou_thresh=0.95,
        )
        raw_boxes = [[[10.0, 10.0, 40.0, 80.0], [10.0, 10.0, 40.0, 80.0]]]
        rows = tensor_to_prediction_rows(
            outputs,
            batch,
            args,
            raw_boxes_abs=raw_boxes,
        )
        self.assertEqual(len(rows[0]["predictions"]), 1)
        prediction = rows[0]["predictions"][0]
        self.assertEqual(prediction["query"], 1)
        self.assertAlmostEqual(prediction["score"], 0.8, places=6)
        self.assertAlmostEqual(prediction["pose_score"], 0.2, places=6)
        self.assertEqual(prediction["bbox_2d"], [0.0, 0.0, 100.0, 100.0])
        self.assertEqual(prediction["input_bbox_2d"], raw_boxes[0][1])

    def test_prediction_rows_reject_missing_person_confidence_head(self) -> None:
        union = len(UNION_KEYPOINTS)
        outputs = {
            "person_logits": torch.full((1, 1), 10.0),
            "ref_logits": torch.zeros(1, 1),
            "box_mask": torch.tensor([[True]]),
            "boxes": torch.zeros(1, 1, 4),
            "pose_boxes": torch.zeros(1, 1, 4),
            "keypoints": torch.zeros(1, 1, union, 3),
            "keypoint_valid_mask": torch.ones(1, union, dtype=torch.bool),
        }
        batch = {
            "task_ids": torch.tensor([0]),
            "prompts": [ALL_POSE_PROMPT],
            "image_paths": ["sample.jpg"],
            "targets": [
                {
                    "dataset": "coco",
                    "image_id": "1",
                    "schema": "COCO17",
                    "width": 100,
                    "height": 100,
                    "boxes": torch.zeros(0, 4),
                }
            ],
        }
        args = SimpleNamespace(
            score_threshold=0.0,
            max_predictions_per_image=10,
            post_pose_nms_iou_thresh=0.95,
        )
        with self.assertRaisesRegex(RuntimeError, "person confidence head"):
            tensor_to_prediction_rows(outputs, batch, args)

    def test_gt_box_prediction_rows_skip_box_nms(self) -> None:
        union = len(UNION_KEYPOINTS)
        outputs = {
            "person_logits": torch.logit(torch.tensor([[0.8, 0.7]])),
            "person_confidence_head_available": True,
            "ref_logits": torch.zeros(1, 2),
            "box_mask": torch.tensor([[True, True]]),
            "boxes": torch.tensor([[[0.1, 0.1, 0.4, 0.8], [0.1, 0.1, 0.4, 0.8]]]),
            "pose_boxes": torch.tensor([[[0.1, 0.1, 0.4, 0.8], [0.1, 0.1, 0.4, 0.8]]]),
            "keypoints": torch.zeros(1, 2, union, 3),
            "keypoint_valid_mask": torch.ones(1, union, dtype=torch.bool),
        }
        batch = {
            "task_ids": torch.tensor([0]),
            "prompts": [ALL_POSE_PROMPT],
            "image_paths": ["sample.jpg"],
            "targets": [
                {
                    "dataset": "coco",
                    "image_id": "1",
                    "schema": "COCO17",
                    "width": 100,
                    "height": 100,
                    "boxes": torch.zeros(0, 4),
                }
            ],
        }
        args = SimpleNamespace(
            score_threshold=0.0,
            max_predictions_per_image=10,
            post_pose_nms_iou_thresh=0.95,
            box_source="gt",
        )
        rows = tensor_to_prediction_rows(outputs, batch, args)
        self.assertEqual(
            [prediction["query"] for prediction in rows[0]["predictions"]],
            [0, 1],
        )

    def test_coco_and_internal_metrics_require_canonical_score(self) -> None:
        union_keypoints = [[0.0, 0.0, 0.5] for _ in UNION_KEYPOINTS]
        rows = [
            {
                "dataset": "coco",
                "image_id": "1",
                "schema": "COCO17",
                "predictions": [
                    {
                        "person_score": 0.9,
                        "pose_score": 0.2,
                        "score": 0.37,
                        "keypoints": union_keypoints,
                    }
                ],
            }
        ]
        coco_results = predictions_to_coco_results(rows, {"1": 1})
        self.assertAlmostEqual(coco_results[0]["score"], 0.37, places=7)
        instances = prediction_rows_to_instances(rows)
        self.assertAlmostEqual(instances[0].score, 0.37, places=7)

        missing_score_rows = json.loads(json.dumps(rows))
        del missing_score_rows[0]["predictions"][0]["score"]
        with self.assertRaisesRegex(KeyError, "canonical instance score"):
            predictions_to_coco_results(missing_score_rows, {"1": 1})
        with self.assertRaisesRegex(KeyError, "canonical instance score"):
            prediction_rows_to_instances(missing_score_rows)

    def test_text_condition_only_affects_ref_pose(self) -> None:
        torch.manual_seed(7)
        model = QwenPoseModel(
            QwenPoseConfig(
                hidden_dim=32,
                external_dim=16,
                pose_decoder_layers=1,
                refinement_steps=1,
                decoder_heads=4,
                box_condition_scale=1.0,
                pose_roi_size=4,
                pose_feature_channels=8,
                enable_person_confidence_head=True,
                schema_joint_priors_path="configs/schema_joint_priors.json",
            )
        ).eval()
        self.assertFalse(model.schema_embed.weight.requires_grad)
        self.assertTrue(torch.equal(model.ref_visual_gates, torch.zeros_like(model.ref_visual_gates)))

        feature = torch.randn(1, 16, 8, 8)
        images = torch.rand(1, 3, 64, 64)
        text_a = torch.randn(1, 16)
        text_b = torch.randn(1, 16)
        boxes = torch.tensor([[[0.1, 0.1, 0.9, 0.9]]])
        mask = torch.tensor([[True]])
        schema = torch.tensor([SCHEMA_TO_ID["COCO17"]])

        with torch.no_grad():
            all_a = model(
                schema,
                torch.tensor([0]),
                images=images,
                external_feature_map=feature,
                external_text_embed=text_a,
                target_boxes=boxes,
                target_box_mask=mask,
            )["keypoints"]
            all_b = model(
                schema,
                torch.tensor([0]),
                images=images,
                external_feature_map=feature,
                external_text_embed=text_b,
                target_boxes=boxes,
                target_box_mask=mask,
            )["keypoints"]
            ref_a_outputs = model(
                schema,
                torch.tensor([1]),
                images=images,
                external_feature_map=feature,
                external_text_embed=text_a,
                target_boxes=boxes,
                target_box_mask=mask,
            )
            ref_b_outputs = model(
                schema,
                torch.tensor([1]),
                images=images,
                external_feature_map=feature,
                external_text_embed=text_b,
                target_boxes=boxes,
                target_box_mask=mask,
            )
            ref_a = ref_a_outputs["keypoints"]
            ref_b = ref_b_outputs["keypoints"]

        self.assertTrue(torch.equal(all_a, all_b))
        # Coordinate and confidence output layers deliberately start at zero,
        # so fresh predictions can match even though the RefHuman instance
        # representation is text-conditioned.
        self.assertTrue(torch.equal(ref_a, ref_b))
        self.assertFalse(
            torch.equal(ref_a_outputs["instance_emb"], ref_b_outputs["instance_emb"])
        )
        self.assertFalse(
            torch.equal(ref_a_outputs["ref_logits"], ref_b_outputs["ref_logits"])
        )
        self.assertTrue(ref_a_outputs["person_confidence_head_available"])
        self.assertTrue(torch.equal(ref_a_outputs["pred_keypoints"], ref_a_outputs["keypoints"]))
        self.assertTrue(torch.equal(ref_a_outputs["pred_boxes"], ref_a_outputs["boxes"]))
        self.assertEqual(tuple(ref_a_outputs["pred_logits"].shape), (1, 1, 1))
        self.assertEqual(
            ALL_POSE_PROMPT,
            "Locate all the instances that match the following description: person.",
        )


if __name__ == "__main__":
    unittest.main()
