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
    InterleavedPoseDataset,
    PoseAugmentConfig,
    PoseRecord,
    PoseRecordDataset,
    aic_to_union,
    load_aic_records,
    load_coco_records,
    load_mpii_records,
    mpii_boxes_from_center_scale,
    pose_collate,
    transform_pose_boxes,
    transform_pose_keypoints,
)
from qwenpose.eval_pose import tensor_to_prediction_rows
from qwenpose.eagle_lora import (
    EagleFeatureExtractor,
    _load_eagle_vision_projector_weights,
    build_eagle_inputs,
)
from qwenpose.losses import LossWeights, compute_pose_losses, simcc_box_loss
from qwenpose.metrics import _mpii_bbox_from_center_scale, targets_to_gt_instances
from qwenpose.model import QwenPoseConfig, QwenPoseModel, build_schema_joint_priors
from qwenpose.qwen_lora import QwenFeatureRefiner
from qwenpose.train_pose import (
    SCHEMA_POSE_EDGE_INDICES,
    HomogeneousDatasetBatchSampler,
    build_locate_generation_prompts,
    build_locate_grounding_responses,
    configure_backbone_train_scope,
    estimate_locate_vision_tokens,
    nms_box_indices_xyxy,
    parse_locate_bbox_response,
    prepare_locate_generated_box_conditioning_from_responses,
    save_pose_visualization,
    validate_pose_batch_contract,
)
from qwenpose.schemas import (
    SCHEMA_INDICES,
    SCHEMA_TO_ID,
    UNION_KEYPOINTS,
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
            output_size=4,
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
        self.assertEqual(tuple(feature_map.shape), (2, 8, 4, 4))
        self.assertEqual(tuple(text_embed.shape), (2, 8))
        self.assertEqual(float(text_embed.abs().sum()), 0.0)
        feature_map.square().mean().backward()
        self.assertIsNotNone(backbone.vision_model.proj.weight.grad)

    def test_fast_stage_backbone_scope_only_opens_vision_lora(self) -> None:
        model = _DummyBackbone()
        counts = configure_backbone_train_scope(model, "vision_lora")
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

    def test_simcc_uniform_baseline_is_log_bin_normalized(self) -> None:
        bins = 256
        logits_x = torch.zeros(1, 1, bins)
        logits_y = torch.zeros(1, 1, bins)
        gt_keypoints = torch.zeros(1, len(UNION_KEYPOINTS), 3)
        gt_keypoints[:, 0, :2] = 0.5
        gt_valid = torch.zeros(1, len(UNION_KEYPOINTS), dtype=torch.bool)
        gt_valid[:, 0] = True
        loss = simcc_box_loss(
            logits_x,
            logits_y,
            gt_keypoints,
            gt_valid,
            torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            torch.tensor([0]),
            torch.tensor([True]),
            sigma=2.0,
        )
        self.assertAlmostEqual(float(loss), 1.0, places=6)

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

    def test_schema_prior_buffer_is_checkpoint_self_contained(self) -> None:
        config = QwenPoseConfig(
            hidden_dim=32,
            external_dim=16,
            pose_decoder_layers=1,
            refinement_steps=1,
            decoder_heads=4,
            pose_roi_size=4,
            simcc_bins=0,
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

    def test_simcc_is_computed_only_for_final_refinement(self) -> None:
        torch.manual_seed(11)
        model = QwenPoseModel(
            QwenPoseConfig(
                hidden_dim=32,
                external_dim=16,
                pose_decoder_layers=1,
                refinement_steps=3,
                decoder_heads=4,
                dropout=0.0,
                box_condition_scale=1.0,
                pose_roi_size=4,
                simcc_bins=32,
                schema_joint_priors_path="configs/schema_joint_priors.json",
            )
        ).eval()
        original_simcc = model._simcc_logits
        call_count = 0

        def counted_simcc(tokens: torch.Tensor):
            nonlocal call_count
            call_count += 1
            return original_simcc(tokens)

        model._simcc_logits = counted_simcc  # type: ignore[method-assign]
        with torch.no_grad():
            outputs = model(
                torch.tensor([SCHEMA_TO_ID["COCO17"]]),
                torch.tensor([0]),
                external_feature_map=torch.randn(1, 16, 8, 8),
                external_text_embed=torch.zeros(1, 16),
                target_boxes=torch.tensor([[[0.1, 0.1, 0.9, 0.9]]]),
                target_box_mask=torch.tensor([[True]]),
            )

        self.assertEqual(call_count, 1)
        self.assertNotIn("simcc_coarse_x", outputs)
        self.assertNotIn("simcc_deform_x", outputs)
        self.assertEqual(outputs["simcc_refine_x"][:2], [None, None])
        self.assertEqual(outputs["simcc_refine_y"][:2], [None, None])
        self.assertTrue(torch.is_tensor(outputs["simcc_refine_x"][2]))
        self.assertTrue(torch.is_tensor(outputs["simcc_refine_y"][2]))

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

    def test_mpii_loader_keeps_all_annotated_people_for_box_lm(self) -> None:
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

    def test_grounding_prompts_and_box_lm_are_task_consistent(self) -> None:
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
            'Locate a single person that matches the following description: "yellow shirt".',
        )
        box_responses, point_responses = build_locate_grounding_responses(
            batch, max_instances=20, max_points=0
        )
        all_boxes = parse_locate_bbox_response(box_responses[0], max_instances=20)
        ref_boxes = parse_locate_bbox_response(box_responses[1], max_instances=20)
        self.assertEqual(int(all_boxes.shape[0]), 2)
        self.assertTrue(torch.allclose(all_boxes[0], torch.tensor([0.1, 0.1, 0.3, 0.4]), atol=1e-3))
        self.assertEqual(int(ref_boxes.shape[0]), 1)
        self.assertTrue(torch.allclose(ref_boxes[0], torch.tensor([0.7, 0.1, 0.9, 0.7]), atol=1e-3))
        self.assertEqual(point_responses, ["None", "None"])

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

    def test_box_only_grounding_does_not_require_pose_fields(self) -> None:
        batch = {
            "task_ids": torch.tensor([0]),
            "targets": [
                {
                    "width": 1000,
                    "height": 1000,
                    "boxes": torch.tensor([[0.1, 0.2, 0.4, 0.8]]),
                    "ref_target": torch.tensor(-1),
                }
            ],
        }
        box_responses, point_responses = build_locate_grounding_responses(
            batch, max_instances=20, max_points=0
        )
        self.assertNotEqual(box_responses, ["None"])
        self.assertEqual(point_responses, ["None"])

    def test_point_grounding_accepts_legacy_keypoint_mask_alias(self) -> None:
        union = len(UNION_KEYPOINTS)
        keypoints = torch.zeros(1, union, 3)
        keypoints[0, 0, :2] = torch.tensor([0.25, 0.5])
        legacy_mask = torch.zeros(1, union, dtype=torch.bool)
        legacy_mask[0, 0] = True
        batch = {
            "task_ids": torch.tensor([0]),
            "targets": [
                {
                    "width": 1000,
                    "height": 1000,
                    "boxes": torch.tensor([[0.1, 0.2, 0.4, 0.8]]),
                    "keypoints": keypoints,
                    "keypoint_mask": legacy_mask,
                    "ref_target": torch.tensor(-1),
                }
            ],
        }
        _, point_responses = build_locate_grounding_responses(
            batch, max_instances=20, max_points=1
        )
        self.assertNotEqual(point_responses, ["None"])

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

    def test_prediction_rows_use_raw_locate_boxes_and_schema_valid_scores(self) -> None:
        union = len(UNION_KEYPOINTS)
        keypoints = torch.zeros(1, 2, union, 3)
        keypoints[0, 0, 0, 2] = 0.9
        keypoints[0, 1, 0, 2] = 0.8
        keypoints[0, 1, 1:, 2] = 1.0
        schema_valid = torch.zeros(1, union, dtype=torch.bool)
        schema_valid[0, 0] = True
        outputs = {
            "person_logits": torch.full((1, 2), 10.0),
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
        self.assertEqual(rows[0]["predictions"][0]["query"], 0)
        self.assertEqual(rows[0]["predictions"][0]["bbox_2d"], raw_boxes[0][0])

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
                simcc_bins=0,
                schema_joint_priors_path="configs/schema_joint_priors.json",
            )
        ).eval()
        self.assertFalse(model.schema_embed.weight.requires_grad)

        feature = torch.randn(1, 16, 8, 8)
        text_a = torch.randn(1, 16)
        text_b = torch.randn(1, 16)
        boxes = torch.tensor([[[0.1, 0.1, 0.9, 0.9]]])
        mask = torch.tensor([[True]])
        schema = torch.tensor([SCHEMA_TO_ID["COCO17"]])

        with torch.no_grad():
            all_a = model(
                schema,
                torch.tensor([0]),
                external_feature_map=feature,
                external_text_embed=text_a,
                target_boxes=boxes,
                target_box_mask=mask,
            )["keypoints"]
            all_b = model(
                schema,
                torch.tensor([0]),
                external_feature_map=feature,
                external_text_embed=text_b,
                target_boxes=boxes,
                target_box_mask=mask,
            )["keypoints"]
            ref_a = model(
                schema,
                torch.tensor([1]),
                external_feature_map=feature,
                external_text_embed=text_a,
                target_boxes=boxes,
                target_box_mask=mask,
            )["keypoints"]
            ref_b = model(
                schema,
                torch.tensor([1]),
                external_feature_map=feature,
                external_text_embed=text_b,
                target_boxes=boxes,
                target_box_mask=mask,
            )["keypoints"]

        self.assertTrue(torch.equal(all_a, all_b))
        self.assertFalse(torch.equal(ref_a, ref_b))
        self.assertEqual(
            ALL_POSE_PROMPT,
            "Locate all the instances that match the following description: person.",
        )


if __name__ == "__main__":
    unittest.main()
