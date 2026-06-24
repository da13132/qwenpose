from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional for minimal envs.
    tqdm = None

from qwenpose.data import build_datasets, pose_collate
from qwenpose.losses import LossWeights, compute_pose_losses
from qwenpose.model import QwenPoseConfig, QwenPoseModel
from qwenpose.qwen_lora import (
    QwenLoRAConfig,
    build_qwen_inputs,
    load_qwen_model,
    load_qwen_with_lora,
    qwen_hidden_size,
)
from qwenpose.eagle_lora import (
    EagleLoRAConfig,
    build_eagle_inputs,
    load_eagle_with_lora,
    eagle_hidden_size,
)
from qwenpose.schemas import SCHEMA_INDICES, SCHEMA_KEYPOINTS, UNION_KEYPOINTS
from qwenpose.train_pose import (
    CHECKPOINT_PAYLOAD_NAME,
    QwenPoseTrainingModel,
    checkpoint_step,
    move_batch_to_device,
    prepare_box_conditioning,
    save_pose_visualization,
)


# COCO keypoint evaluation utilities
# COCO17 keypoint order (standard 17 joints)
COCO_KEYPOINT_NAMES = SCHEMA_KEYPOINTS["COCO17"]
COCO_KPT_INDICES = SCHEMA_INDICES["COCO17"]  # indices into UNION_KEYPOINTS


def _coco_cat_id() -> int:
    """COCO person category id."""
    return 1


def build_coco_gt_annotations(
    all_targets: list[dict],
) -> tuple[dict, dict]:
    """Build COCO-format ground truth from collected pose targets.

    Returns:
        coco_gt: dict with 'images', 'annotations', 'categories' for pycocotools.
        id_map: mapping from image_id string to integer image_id.
    """
    images = []
    annotations = []
    image_id_map: dict[str, int] = {}
    next_img_id = 0
    next_ann_id = 0

    for target in all_targets:
        image_id_str = str(target["image_id"])
        if image_id_str not in image_id_map:
            image_id_map[image_id_str] = next_img_id
            images.append({
                "id": next_img_id,
                "width": int(target["width"]),
                "height": int(target["height"]),
            })
            next_img_id += 1

        img_id = image_id_map[image_id_str]
        w = float(target["width"])
        h = float(target["height"])
        boxes = target["boxes"]  # [N, 4] normalized xyxy
        keypoints = target["keypoints"]  # [N, U, 3] normalized x, y, vis
        keypoint_valid = target["keypoint_valid"]  # [N, U] bool

        num_instances = boxes.shape[0]
        for inst_idx in range(num_instances):
            box = boxes[inst_idx].tolist()  # normalized xyxy
            x1 = box[0] * w
            y1 = box[1] * h
            bw = (box[2] - box[0]) * w
            bh = (box[3] - box[1]) * h

            # Extract COCO17 keypoints
            coco_kpts = []
            num_visible = 0
            for kpt_idx in COCO_KPT_INDICES.tolist():
                kpt_idx = int(kpt_idx)
                if keypoint_valid[inst_idx, kpt_idx]:
                    kx = float(keypoints[inst_idx, kpt_idx, 0]) * w
                    ky = float(keypoints[inst_idx, kpt_idx, 1]) * h
                    vis = 2  # labeled and visible
                    coco_kpts.extend([kx, ky, vis])
                    num_visible += 1
                else:
                    coco_kpts.extend([0.0, 0.0, 0])

            annotations.append({
                "id": next_ann_id,
                "image_id": img_id,
                "category_id": _coco_cat_id(),
                "keypoints": coco_kpts,
                "num_keypoints": num_visible,
                "bbox": [x1, y1, bw, bh],
                "area": bw * bh,
                "iscrowd": 0,
            })
            next_ann_id += 1

    coco_gt = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": _coco_cat_id(), "name": "person", "keypoints": COCO_KEYPOINT_NAMES, "skeleton": []}],
    }
    return coco_gt, image_id_map


def predictions_to_coco_results(
    predictions_rows: list[dict],
    image_id_map: dict[str, int],
) -> list[dict]:
    """Convert QwenPose predictions to COCO keypoint results format."""
    results = []
    for row in predictions_rows:
        image_id_str = str(row["image_id"])
        if image_id_str not in image_id_map:
            continue
        img_id = image_id_map[image_id_str]
        w = float(row.get("width", 0))
        h = float(row.get("height", 0))
        # If width/height not in row, try to get from first prediction's bbox
        for pred in row["predictions"]:
            person_score = float(pred["person_score"])
            all_kpts = pred["keypoints"]  # [U, 3] with x, y, vis in absolute coords

            # Extract COCO17 keypoints (already in absolute coords from tensor_to_prediction_rows)
            coco_kpts = []
            for kpt_idx in COCO_KPT_INDICES.tolist():
                kpt_idx = int(kpt_idx)
                kx = float(all_kpts[kpt_idx][0])
                ky = float(all_kpts[kpt_idx][1])
                kvis = float(all_kpts[kpt_idx][2])
                coco_kpts.extend([kx, ky, kvis])

            # Score: person_score * mean visible keypoint confidence
            vis_values = [float(all_kpts[int(ki)][2]) for ki in COCO_KPT_INDICES.tolist()]
            mean_vis = sum(vis_values) / max(len(vis_values), 1)
            score = person_score * max(mean_vis, 0.01)

            results.append({
                "image_id": img_id,
                "category_id": _coco_cat_id(),
                "keypoints": coco_kpts,
                "score": score,
            })
    return results


def compute_coco_keypoint_ap(
    coco_gt_dict: dict,
    coco_results: list[dict],
) -> dict[str, float]:
    """Compute COCO keypoint AP metrics using pycocotools.

    Returns dict with AP, AP50, AP75, AR, AR50, AR75, ARm, ARl, etc.
    """
    if not coco_results:
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "AR": 0.0, "AR50": 0.0, "AR75": 0.0}

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO()
    coco_gt.dataset = coco_gt_dict
    coco_gt.createIndex()

    coco_dt = coco_gt.loadRes(coco_results)

    # Keypoint evaluation
    coco_eval = COCOeval(coco_gt, coco_dt, "keypoints")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Extract metrics from coco_eval.stats
    # stats[0] = AP (IoU=0.50:0.95)
    # stats[1] = AP50 (IoU=0.50)
    # stats[2] = AP75 (IoU=0.75)
    # stats[3] = APm (medium)
    # stats[4] = APl (large)
    # stats[5] = AR (maxDets=20, IoU=0.50:0.95)
    # stats[6] = AR50 (maxDets=20, IoU=0.50)
    # stats[7] = AR75 (maxDets=20, IoU=0.75)
    # stats[8] = ARm (medium)
    # stats[9] = ARl (large)
    stats = coco_eval.stats
    metric_names = ["AP", "AP50", "AP75", "APm", "APl", "AR", "AR50", "AR75", "ARm", "ARl"]
    metrics = {}
    for i, name in enumerate(metric_names):
        if i < len(stats):
            metrics[name] = float(stats[i])
        else:
            metrics[name] = 0.0
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate QwenPose checkpoints.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, default=Path("datasets"))
    parser.add_argument("--datasets", type=str, default="coco,crowdpose,mpii,refhuman")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--max_instances", type=int, default=80)
    parser.add_argument("--max_samples_per_dataset", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--record_cache_dir", type=Path, default=Path(".cache/qwenpose_records"))
    parser.add_argument("--disable_record_cache", action="store_true")
    parser.add_argument("--disable_progress", action="store_true")

    parser.add_argument("--backbone", choices=["qwen3vl", "eagle"], default="qwen3vl")
    parser.add_argument("--qwen_model_path", type=str, default="weights/Qwen3-VL-4B-Instruct")
    parser.add_argument("--qwen_dtype", choices=["bfloat16", "float16", "float32", "auto", "none"], default="bfloat16")
    parser.add_argument("--qwen_attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--qwen_min_pixels", type=int, default=None)
    parser.add_argument("--qwen_max_pixels", type=int, default=None)
    parser.add_argument("--qwen_feature_size", type=int, default=32)
    parser.add_argument("--qwen_feature_refiner_layers", type=int, default=0)
    parser.add_argument("--qwen_feature_refiner_bottleneck_dim", type=int, default=256)
    parser.add_argument("--qwen_feature_refiner_init_scale", type=float, default=0.1)
    parser.add_argument("--qwen_lora_r", type=int, default=32)
    parser.add_argument("--qwen_lora_alpha", type=int, default=64)
    parser.add_argument("--qwen_lora_dropout", type=float, default=0.05)
    parser.add_argument("--qwen_vision_lora_r", type=int, default=16)
    parser.add_argument("--qwen_vision_lora_alpha", type=int, default=32)
    parser.add_argument("--qwen_vision_lora_dropout", type=float, default=0.05)
    # Eagle backbone options
    parser.add_argument("--eagle_model_path", type=str, default="weights/LocateAnything-3B")
    parser.add_argument("--eagle_dtype", choices=["bfloat16", "float16", "float32", "auto", "none"], default="bfloat16")
    parser.add_argument("--eagle_attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--eagle_min_pixels", type=int, default=None)
    parser.add_argument("--eagle_max_pixels", type=int, default=None)
    parser.add_argument("--eagle_feature_size", type=int, default=32)
    parser.add_argument("--eagle_feature_refiner_layers", type=int, default=0)
    parser.add_argument("--eagle_feature_refiner_bottleneck_dim", type=int, default=256)
    parser.add_argument("--eagle_feature_refiner_init_scale", type=float, default=0.1)
    parser.add_argument("--eagle_lora_r", type=int, default=32)
    parser.add_argument("--eagle_lora_alpha", type=int, default=64)
    parser.add_argument("--eagle_lora_dropout", type=float, default=0.05)
    parser.add_argument("--eagle_vision_lora_r", type=int, default=16)
    parser.add_argument("--eagle_vision_lora_alpha", type=int, default=32)
    parser.add_argument("--eagle_vision_lora_dropout", type=float, default=0.05)

    parser.add_argument("--hidden_dim", type=int, default=448)
    parser.add_argument("--pose_decoder_layers", type=int, default=1)
    parser.add_argument("--refinement_steps", type=int, default=3)
    parser.add_argument("--decoder_heads", type=int, default=8)
    parser.add_argument("--box_condition_scale", type=float, default=1.2)
    parser.add_argument("--pose_roi_size", type=int, default=16)
    parser.add_argument("--simcc_bins", type=int, default=128)
    parser.add_argument("--disable_uncertainty", action="store_true")
    parser.add_argument("--disable_refinement", action="store_true")
    parser.add_argument("--disable_aux_center", action="store_true")
    parser.add_argument("--disable_simcc", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--score_threshold", type=float, default=0.05)
    parser.add_argument("--max_predictions_per_image", type=int, default=100)
    parser.add_argument("--visualize_max_samples", type=int, default=100)
    parser.add_argument("--visualize_max_instances", type=int, default=8)

    parser.add_argument("--w_oks", type=float, default=1.0)
    parser.add_argument("--w_coord", type=float, default=2.0)
    parser.add_argument("--w_vis", type=float, default=0.5)
    parser.add_argument("--w_uncertainty", type=float, default=0.05)
    parser.add_argument("--w_aux_center", type=float, default=0.2)
    parser.add_argument("--w_hard_joint", type=float, default=0.0)
    parser.add_argument("--hard_joint_fraction", type=float, default=0.3)
    return parser.parse_args()


def resolve_checkpoint(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {path}")
    if (path / CHECKPOINT_PAYLOAD_NAME).is_file():
        return path / CHECKPOINT_PAYLOAD_NAME
    checkpoints: list[tuple[int, Path]] = []
    for child in list(path.glob("checkpoint-*")) + list(path.glob("checkpoint_step_*.pt")):
        step = checkpoint_step(child)
        if step is None:
            continue
        payload_path = child / CHECKPOINT_PAYLOAD_NAME if child.is_dir() else child
        if payload_path.is_file():
            checkpoints.append((step, payload_path))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint-* or checkpoint_step_*.pt found under: {path}")
    return sorted(checkpoints)[-1][1]


def load_eval_model(args: argparse.Namespace, checkpoint: dict, device: torch.device) -> tuple[QwenPoseTrainingModel, object | None]:
    backbone_model = None
    backbone_processor = None
    external_dim = None
    backbone_name = getattr(args, "backbone", "qwen3vl")
    backbone_merged = bool(checkpoint.get("backbone_merged", False))

    if backbone_name == "eagle":
        backbone_model, backbone_processor = load_eagle_with_lora(
            EagleLoRAConfig(
                model_path=args.eagle_model_path,
                lora_r=args.eagle_lora_r,
                lora_alpha=args.eagle_lora_alpha,
                lora_dropout=args.eagle_lora_dropout,
                vision_lora_r=args.eagle_vision_lora_r,
                vision_lora_alpha=args.eagle_vision_lora_alpha,
                vision_lora_dropout=args.eagle_vision_lora_dropout,
                dtype=args.eagle_dtype,
                attn_implementation=args.eagle_attn_implementation,
            )
        )
        if "qwen_trainable" in checkpoint:
            backbone_model.load_state_dict(checkpoint["qwen_trainable"], strict=False)
        backbone_model.to(device)
        backbone_model.eval()
        external_dim = eagle_hidden_size(backbone_model)
        default_feature_size = args.eagle_feature_size
        default_refiner_layers = args.eagle_feature_refiner_layers
        default_refiner_bottleneck_dim = args.eagle_feature_refiner_bottleneck_dim
        default_refiner_init_scale = args.eagle_feature_refiner_init_scale
    else:
        if backbone_merged:
            backbone_model, backbone_processor = load_qwen_model(
                args.qwen_model_path,
                dtype=args.qwen_dtype,
                attn_implementation=args.qwen_attn_implementation,
            )
        else:
            backbone_model, backbone_processor = load_qwen_with_lora(
                QwenLoRAConfig(
                    model_path=args.qwen_model_path,
                    lora_r=args.qwen_lora_r,
                    lora_alpha=args.qwen_lora_alpha,
                    lora_dropout=args.qwen_lora_dropout,
                    vision_lora_r=args.qwen_vision_lora_r,
                    vision_lora_alpha=args.qwen_vision_lora_alpha,
                    vision_lora_dropout=args.qwen_vision_lora_dropout,
                    dtype=args.qwen_dtype,
                    attn_implementation=args.qwen_attn_implementation,
                )
            )
        if not backbone_merged and "qwen_trainable" in checkpoint:
            backbone_model.load_state_dict(checkpoint["qwen_trainable"], strict=False)
        backbone_model.to(device)
        backbone_model.eval()
        external_dim = qwen_hidden_size(backbone_model)
        default_feature_size = args.qwen_feature_size
        default_refiner_layers = args.qwen_feature_refiner_layers
        default_refiner_bottleneck_dim = args.qwen_feature_refiner_bottleneck_dim
        default_refiner_init_scale = args.qwen_feature_refiner_init_scale

    saved_pose_config = checkpoint.get("pose_config")
    pose_config_kwargs = (
        {
            "hidden_dim": args.hidden_dim,
            "external_dim": external_dim,
            "pose_decoder_layers": args.pose_decoder_layers,
            "refinement_steps": args.refinement_steps,
            "decoder_heads": args.decoder_heads,
            "box_condition_scale": args.box_condition_scale,
            "pose_roi_size": args.pose_roi_size,
            "use_uncertainty": not args.disable_uncertainty,
            "use_refinement": not args.disable_refinement,
            "use_aux_center": not args.disable_aux_center,
            "use_simcc": not args.disable_simcc,
            "simcc_bins": args.simcc_bins,
        }
        if saved_pose_config is None
        else {
            key: saved_pose_config[key]
            for key in QwenPoseConfig.__dataclass_fields__
            if key in saved_pose_config
        }
    )
    pose_config_kwargs["external_dim"] = external_dim
    pose_model = QwenPoseModel(QwenPoseConfig(**pose_config_kwargs))
    pose_model.load_state_dict(checkpoint["model"], strict=True)
    refiner_config = checkpoint.get("qwen_feature_refiner_config", {})
    feature_config = checkpoint.get("qwen_feature_config", {})
    has_refiner_checkpoint = "qwen_feature_refiner" in checkpoint
    feature_size = int(feature_config.get("output_size", default_feature_size))
    refiner_layers = int(refiner_config.get("layers", default_refiner_layers)) if has_refiner_checkpoint else 0
    refiner_bottleneck_dim = int(refiner_config.get("bottleneck_dim", default_refiner_bottleneck_dim))
    refiner_init_scale = float(refiner_config.get("init_scale", default_refiner_init_scale))

    # Build feature extractor
    if backbone_name == "eagle":
        from qwenpose.eagle_lora import EagleFeatureExtractor
        backbone_extractor = EagleFeatureExtractor(
            backbone_model,
            output_size=feature_size,
            refiner_layers=refiner_layers,
            refiner_bottleneck_dim=refiner_bottleneck_dim,
            refiner_init_scale=refiner_init_scale,
        )
    else:
        from qwenpose.qwen_lora import QwenFeatureExtractor
        backbone_extractor = QwenFeatureExtractor(
            backbone_model,
            output_size=feature_size,
            refiner_layers=refiner_layers,
            refiner_bottleneck_dim=refiner_bottleneck_dim,
            refiner_init_scale=refiner_init_scale,
        )

    model = QwenPoseTrainingModel(
        pose_model=pose_model,
        backbone_model=backbone_model,
        backbone_extractor=backbone_extractor,
        backbone_name=backbone_name,
    ).to(device)
    if has_refiner_checkpoint:
        model.backbone_extractor.feature_refiner.load_state_dict(checkpoint["qwen_feature_refiner"], strict=True)
    model.eval()
    return model, backbone_processor


def tensor_to_prediction_rows(outputs: dict[str, torch.Tensor], batch: dict, args: argparse.Namespace) -> list[dict]:
    rows: list[dict] = []
    person_scores = outputs["person_logits"].sigmoid().detach().cpu()
    ref_logits = outputs["ref_logits"].detach().cpu()
    box_mask = outputs.get("box_mask")
    box_mask_cpu = box_mask.detach().cpu().bool() if box_mask is not None else torch.ones_like(person_scores, dtype=torch.bool)
    boxes = outputs["boxes"].detach().cpu()
    pose_boxes = outputs.get("pose_boxes", outputs["boxes"]).detach().cpu()
    keypoints = outputs["keypoints"].detach().cpu()
    for b, target in enumerate(batch["targets"]):
        width = float(target["width"])
        height = float(target["height"])
        task_id = int(batch["task_ids"][b].cpu().item())
        valid = torch.nonzero(box_mask_cpu[b], as_tuple=False).flatten()
        if task_id == 1:
            selected = [int(valid[0].item())] if valid.numel() > 0 else []
        else:
            keep = valid[person_scores[b, valid] >= args.score_threshold] if valid.numel() > 0 else valid
            if keep.numel() == 0 and valid.numel() > 0:
                keep = valid[:1]
            selected = keep[: args.max_predictions_per_image].tolist()

        predictions = []
        for query_idx in selected:
            box = boxes[b, query_idx].tolist()
            box_abs = [
                box[0] * width,
                box[1] * height,
                box[2] * width,
                box[3] * height,
            ]
            pose_box = pose_boxes[b, query_idx].tolist()
            pose_box_abs = [
                pose_box[0] * width,
                pose_box[1] * height,
                pose_box[2] * width,
                pose_box[3] * height,
            ]
            kp = keypoints[b, query_idx].clone()
            kp[:, 0] *= width
            kp[:, 1] *= height
            predictions.append(
                {
                    "query": query_idx,
                    "person_score": float(person_scores[b, query_idx].item()),
                    "ref_score": float(ref_logits[b, query_idx].sigmoid().item()),
                    "bbox_2d": box_abs,
                    "pose_bbox_2d": pose_box_abs,
                    "keypoints": kp.tolist(),
                }
            )
        rows.append(
            {
                "dataset": target["dataset"],
                "image_id": target["image_id"],
                "image_path": batch["image_paths"][b],
                "width": width,
                "height": height,
                "task_id": task_id,
                "schema": target["schema"],
                "prompt": batch["prompts"][b],
                "num_gt": int(target["boxes"].shape[0]),
                "predictions": predictions,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if args.box_condition_scale <= 0:
        raise ValueError("--box_condition_scale must be positive.")
    if args.pose_roi_size <= 1:
        raise ValueError("--pose_roi_size must be greater than 1.")
    if not 0.0 <= args.hard_joint_fraction <= 1.0:
        raise ValueError("--hard_joint_fraction must be in [0, 1].")
    if args.visualize_max_samples < 0:
        raise ValueError("--visualize_max_samples must be non-negative.")
    if args.visualize_max_instances <= 0:
        raise ValueError("--visualize_max_instances must be positive.")
    if args.qwen_min_pixels is not None and args.qwen_min_pixels <= 0:
        raise ValueError("--qwen_min_pixels must be positive when set.")
    if args.qwen_max_pixels is not None and args.qwen_max_pixels <= 0:
        raise ValueError("--qwen_max_pixels must be positive when set.")
    if (
        args.qwen_min_pixels is not None
        and args.qwen_max_pixels is not None
        and args.qwen_max_pixels < args.qwen_min_pixels
    ):
        raise ValueError("--qwen_max_pixels must be >= --qwen_min_pixels.")
    if args.eagle_min_pixels is not None and args.eagle_min_pixels <= 0:
        raise ValueError("--eagle_min_pixels must be positive when set.")
    if args.eagle_max_pixels is not None and args.eagle_max_pixels <= 0:
        raise ValueError("--eagle_max_pixels must be positive when set.")
    if (
        args.eagle_min_pixels is not None
        and args.eagle_max_pixels is not None
        and args.eagle_max_pixels < args.eagle_min_pixels
    ):
        raise ValueError("--eagle_max_pixels must be >= --eagle_min_pixels.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = resolve_checkpoint(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    device = torch.device(args.device)

    dataset_names = [name.strip() for name in args.datasets.split(",") if name.strip()]
    dataset = build_datasets(
        dataset_root=args.dataset_root,
        names=dataset_names,
        max_instances=args.max_instances,
        split=args.split,
        max_samples_per_dataset=args.max_samples_per_dataset,
        mixing_strategy="concat_shuffle",
        seed=42,
        record_cache_dir=args.record_cache_dir,
        disable_record_cache=args.disable_record_cache,
        show_progress=not args.disable_progress,
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": pose_collate,
        "drop_last": False,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(dataset, **loader_kwargs)
    model, qwen_processor = load_eval_model(args, checkpoint, device)
    weights = LossWeights(
        oks=args.w_oks,
        coord=args.w_coord,
        vis=args.w_vis,
        uncertainty=args.w_uncertainty,
        aux_center=args.w_aux_center,
        hard_joint=args.w_hard_joint,
        hard_joint_fraction=args.hard_joint_fraction,
    )

    totals: dict[str, float] = {}
    batches = 0
    samples = 0
    predictions_path = args.output_dir / "predictions.jsonl"
    visualization_dir = args.output_dir / "visualizations"
    if args.visualize_max_samples > 0:
        visualization_dir.mkdir(parents=True, exist_ok=True)
    visualized_samples = 0

    # Collect GT and predictions for AP computation
    all_gt_targets: list[dict] = []
    all_predictions_rows: list[dict] = []
    # Per-dataset tracking
    dataset_gt: dict[str, list[dict]] = defaultdict(list)
    dataset_preds: dict[str, list[dict]] = defaultdict(list)

    with predictions_path.open("w", encoding="utf-8") as f:
        with torch.inference_mode():
            progress_bar = None
            if not args.disable_progress and tqdm is not None:
                progress_bar = tqdm(
                    total=len(loader),
                    desc=f"eval {args.split}",
                    unit="batch",
                    dynamic_ncols=True,
                    mininterval=1.0,
                )
            last_iter_end = time.perf_counter()
            for batch in loader:
                batch_ready = time.perf_counter()
                data_time = batch_ready - last_iter_end
                prep_started = time.perf_counter()
                batch = move_batch_to_device(batch, device)
                target_boxes, target_box_mask, pose_targets = prepare_box_conditioning(
                    batch["targets"],
                    batch["task_ids"],
                    device,
                    max_instances=args.max_instances,
                )
                backbone_name = getattr(args, "backbone", "qwen3vl")
                if qwen_processor is None:
                    qwen_inputs = None
                elif backbone_name == "eagle":
                    qwen_inputs = build_eagle_inputs(
                        qwen_processor,
                        batch["image_paths"],
                        batch["prompts"],
                        device,
                        min_pixels=args.eagle_min_pixels,
                        max_pixels=args.eagle_max_pixels,
                    )
                else:
                    qwen_inputs = build_qwen_inputs(
                        qwen_processor,
                        batch["image_paths"],
                        batch["prompts"],
                        device,
                        min_pixels=args.qwen_min_pixels,
                        max_pixels=args.qwen_max_pixels,
                    )
                prep_time = time.perf_counter() - prep_started
                forward_started = time.perf_counter()
                outputs = model(
                    schema_ids=batch["schema_ids"],
                    task_ids=batch["task_ids"],
                    qwen_inputs=qwen_inputs,
                    target_boxes=target_boxes,
                    target_box_mask=target_box_mask,
                )
                _, loss_dict = compute_pose_losses(outputs, pose_targets, batch["task_ids"], weights)
                forward_time = time.perf_counter() - forward_started
                for key, value in loss_dict.items():
                    totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())

                rows = tensor_to_prediction_rows(outputs, {**batch, "targets": pose_targets}, args)
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    # Collect GT for AP
                    ds_name = row.get("dataset", "unknown")
                    target_for_gt = {
                        "image_id": row["image_id"],
                        "width": 0,
                        "height": 0,
                        "boxes": torch.zeros(0, 4),
                        "keypoints": torch.zeros(0, len(UNION_KEYPOINTS), 3),
                        "keypoint_valid": torch.zeros(0, len(UNION_KEYPOINTS), dtype=torch.bool),
                    }
                    # Find matching target from pose_targets
                    for pt in pose_targets:
                        if str(pt["image_id"]) == str(row["image_id"]):
                            target_for_gt = pt
                            break
                    all_gt_targets.append(target_for_gt)
                    dataset_gt[ds_name].append(target_for_gt)
                    all_predictions_rows.append(row)
                    dataset_preds[ds_name].append(row)

                if args.visualize_max_samples > 0 and visualized_samples < args.visualize_max_samples:
                    eval_batch = {**batch, "targets": pose_targets}
                    batch_size = len(batch["schema_ids"])
                    for local_idx in range(batch_size):
                        if visualized_samples >= args.visualize_max_samples:
                            break
                        vis_path = visualization_dir / f"eval_{samples + local_idx:06d}.jpg"
                        save_pose_visualization(
                            outputs,
                            eval_batch,
                            vis_path,
                            sample_idx=local_idx,
                            max_instances=args.visualize_max_instances,
                        )
                        visualized_samples += 1
                batches += 1
                samples += len(batch["schema_ids"])
                last_iter_end = time.perf_counter()
                if progress_bar is not None:
                    progress_bar.set_postfix(
                        {
                            "loss": f"{float(loss_dict['loss_total'].detach().cpu()):.3f}",
                            "samples": samples,
                            "data": f"{data_time:.2f}s",
                            "prep": f"{prep_time:.2f}s",
                            "fwd": f"{forward_time:.2f}s",
                        },
                        refresh=False,
                    )
                    progress_bar.update(1)
            if progress_bar is not None:
                progress_bar.close()

    # ── Compute COCO keypoint AP metrics ──────────────────────────────────
    print("\n" + "=" * 60)
    print("Computing COCO keypoint AP metrics...")
    print("=" * 60)

    ap_metrics_all: dict[str, float] = {}
    ap_metrics_per_dataset: dict[str, dict[str, float]] = {}

    # Overall AP
    if all_gt_targets and all_predictions_rows:
        coco_gt_dict, id_map = build_coco_gt_annotations(all_gt_targets)
        coco_results = predictions_to_coco_results(all_predictions_rows, id_map)
        if coco_results:
            print(f"\n[Overall] {len(coco_gt_dict['annotations'])} GT annotations, {len(coco_results)} predictions")
            ap_metrics_all = compute_coco_keypoint_ap(coco_gt_dict, coco_results)
            print(f"\nOverall AP metrics:")
            for k, v in ap_metrics_all.items():
                print(f"  {k}: {v:.4f}")
        else:
            print("[Overall] No valid predictions to evaluate.")
    else:
        print("[Overall] No GT or predictions collected.")

    # Per-dataset AP
    for ds_name in sorted(dataset_gt.keys()):
        gt_list = dataset_gt[ds_name]
        pred_list = dataset_preds[ds_name]
        if gt_list and pred_list:
            coco_gt_dict, id_map = build_coco_gt_annotations(gt_list)
            coco_results = predictions_to_coco_results(pred_list, id_map)
            if coco_results:
                print(f"\n[{ds_name}] {len(coco_gt_dict['annotations'])} GT annotations, {len(coco_results)} predictions")
                ds_metrics = compute_coco_keypoint_ap(coco_gt_dict, coco_results)
                ap_metrics_per_dataset[ds_name] = ds_metrics
                print(f"\n{ds_name} AP metrics:")
                for k, v in ds_metrics.items():
                    print(f"  {k}: {v:.4f}")
            else:
                print(f"[{ds_name}] No valid predictions.")
                ap_metrics_per_dataset[ds_name] = {}
        else:
            print(f"[{ds_name}] No GT or predictions.")
            ap_metrics_per_dataset[ds_name] = {}

    # ── Export predictions JSON ────────────────────────────────────────────
    predictions_json_path = args.output_dir / "predictions.json"
    predictions_json_path.write_text(
        json.dumps(all_predictions_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ── Summary ────────────────────────────────────────────────────────────
    summary = {
        "checkpoint": str(checkpoint_path),
        "datasets": dataset_names,
        "split": args.split,
        "samples": samples,
        "batches": batches,
        "avg_losses": {key: value / max(batches, 1) for key, value in totals.items()},
        "ap_metrics": ap_metrics_all,
        "ap_metrics_per_dataset": ap_metrics_per_dataset,
        "predictions_jsonl": str(predictions_path),
        "predictions_json": str(predictions_json_path),
        "visualizations_dir": str(visualization_dir) if args.visualize_max_samples > 0 else None,
        "visualized_samples": visualized_samples,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── Report ─────────────────────────────────────────────────────────────
    report = [
        "# QwenPose Eval Report",
        "",
        f"- checkpoint: `{checkpoint_path}`",
        f"- datasets: `{','.join(dataset_names)}`",
        f"- split: `{args.split}`",
        f"- samples: `{samples}`",
        f"- batches: `{batches}`",
        "",
        "## Average Losses",
        "",
    ]
    for key, value in summary["avg_losses"].items():
        report.append(f"- `{key}`: {value:.6f}")

    report.extend(["", "## COCO Keypoint AP (Overall)", ""])
    if ap_metrics_all:
        report.append("| Metric | Value |")
        report.append("|--------|-------|")
        for k, v in ap_metrics_all.items():
            report.append(f"| {k} | {v:.4f} |")
    else:
        report.append("No metrics available.")

    for ds_name, ds_metrics in sorted(ap_metrics_per_dataset.items()):
        report.extend(["", f"## COCO Keypoint AP ({ds_name})", ""])
        if ds_metrics:
            report.append("| Metric | Value |")
            report.append("|--------|-------|")
            for k, v in ds_metrics.items():
                report.append(f"| {k} | {v:.4f} |")
        else:
            report.append("No metrics available.")

    report.extend(["", f"Predictions JSONL: `{predictions_path}`", ""])
    report.extend([f"Predictions JSON: `{predictions_json_path}`", ""])
    if args.visualize_max_samples > 0:
        report.extend([f"Visualizations: `{visualization_dir}`", ""])
    (args.output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
