from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import warnings

from PIL import Image
import torch

from qwenpose.data import ALL_POSE_PROMPT, InterleavedPoseDataset, aic_to_union, mpii_boxes_from_center_scale
from qwenpose.eagle_lora import EagleFeatureExtractor
from qwenpose.losses import LossWeights, compute_pose_losses, simcc_box_loss
from qwenpose.metrics import _mpii_bbox_from_center_scale, targets_to_gt_instances
from qwenpose.model import QwenPoseConfig, QwenPoseModel, build_schema_joint_priors
from qwenpose.qwen_lora import QwenFeatureRefiner
from qwenpose.train_pose import (
    SCHEMA_POSE_EDGE_INDICES,
    configure_backbone_train_scope,
    save_pose_visualization,
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


class MultiDatasetPoseTests(unittest.TestCase):
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
            "input_ids": torch.zeros(2, 3, dtype=torch.long),
            "attention_mask": torch.ones(2, 3, dtype=torch.long),
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
            pixels = list(Image.open(output_path).convert("RGB").get_flattened_data())
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
        self.assertIn("Estimate the human pose", ALL_POSE_PROMPT)


if __name__ == "__main__":
    unittest.main()
