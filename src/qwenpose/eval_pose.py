from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional for minimal envs.
    tqdm = None

from qwenpose.data import build_datasets, pose_collate
from qwenpose.losses import LossWeights, compute_pose_losses
from qwenpose.model import (
    QwenPoseConfig,
    QwenPoseModel,
    apply_keypoint_decode_mode,
    topk_keypoint_confidence,
)
from qwenpose.qwen_lora import (
    QwenLoRAConfig,
    load_qwen_model,
    load_qwen_with_lora,
    qwen_hidden_size,
)
from qwenpose.eagle_lora import (
    EagleLoRAConfig,
    load_eagle_vision_only_with_lora,
    load_eagle_with_lora,
    eagle_hidden_size,
)
from qwenpose.schemas import SCHEMA_INDICES, SCHEMA_KEYPOINTS, UNION_KEYPOINTS
from qwenpose.metrics import compute_pose_metrics_from_targets
from qwenpose.train_pose import (
    CHECKPOINT_PAYLOAD_NAME,
    LocatePoseUnifiedConfig,
    LocatePoseUnifiedRuntime,
    QwenPoseTrainingModel,
    checkpoint_step,
    locate_boxes_abs_from_responses,
    move_batch_to_device,
    nms_box_indices_xyxy,
    prepare_box_conditioning,
    prepare_locate_generated_box_conditioning_from_responses,
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
        # Keypoints are already in absolute image coordinates.
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

            if "score" not in pred:
                raise KeyError(
                    "COCO prediction is missing the canonical instance score."
                )
            score = float(pred["score"])

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


def compute_official_coco_keypoint_ap(
    prediction_rows: list[dict],
    dataset_root: Path,
    split: str,
) -> dict[str, float]:
    """Evaluate exported instance scores against the official COCO GT file."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_split = split if split.endswith("2017") else f"{split}2017"
    annotation_path = (
        dataset_root / "coco" / "annotations" / f"person_keypoints_{coco_split}.json"
    )
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Official COCO keypoint annotations not found: {annotation_path}")

    results: list[dict] = []
    image_ids: set[int] = set()
    for row in prediction_rows:
        if row.get("dataset") != "coco":
            continue
        image_id = int(row["image_id"])
        image_ids.add(image_id)
        for prediction in row.get("predictions", []):
            union_keypoints = prediction["keypoints"]
            coco_keypoints: list[float] = []
            for union_idx in COCO_KPT_INDICES.tolist():
                x, y, visibility = union_keypoints[int(union_idx)][:3]
                coco_keypoints.extend([float(x), float(y), float(visibility)])
            results.append(
                {
                    "image_id": image_id,
                    "category_id": _coco_cat_id(),
                    "keypoints": coco_keypoints,
                    # One learned person/pose-quality score is used for ranking;
                    # per-keypoint quality is not multiplied into the COCO score.
                    "score": float(prediction["score"]),
                }
            )
    if not image_ids or not results:
        return {
            "AP": 0.0,
            "AP50": 0.0,
            "AP75": 0.0,
            "APm": 0.0,
            "APl": 0.0,
            "AR": 0.0,
            "AR50": 0.0,
            "AR75": 0.0,
            "ARm": 0.0,
            "ARl": 0.0,
        }

    coco_gt = COCO(str(annotation_path))
    coco_dt = coco_gt.loadRes(results)
    evaluator = COCOeval(coco_gt, coco_dt, "keypoints")
    evaluator.params.imgIds = sorted(image_ids)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    names = ["AP", "AP50", "AP75", "APm", "APl", "AR", "AR50", "AR75", "ARm", "ARl"]
    return {name: float(evaluator.stats[index]) for index, name in enumerate(names)}


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

    parser.add_argument("--backbone", choices=["qwen3vl", "eagle", "locatepose"], default="qwen3vl")
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
    parser.add_argument("--locate_model_path", "--eagle_model_path", dest="eagle_model_path", type=str, default="weights/LocateAnything-3B")
    parser.add_argument("--locate_dtype", "--eagle_dtype", dest="eagle_dtype", choices=["bfloat16", "float16", "float32", "auto", "none"], default="bfloat16")
    parser.add_argument("--locate_attn_implementation", "--eagle_attn_implementation", dest="eagle_attn_implementation", type=str, default="sdpa")
    parser.add_argument("--locate_image_token_limit", "--eagle_image_token_limit", dest="eagle_image_token_limit", type=int, default=None)
    parser.add_argument("--locate_batch_token_limit", "--eagle_batch_token_limit", dest="eagle_batch_token_limit", type=int, default=None)
    parser.add_argument("--locate_feature_size", "--eagle_feature_size", dest="eagle_feature_size", type=int, default=100)
    parser.add_argument("--locate_feature_refiner_layers", "--eagle_feature_refiner_layers", dest="eagle_feature_refiner_layers", type=int, default=0)
    parser.add_argument(
        "--locate_feature_refiner_bottleneck_dim",
        "--eagle_feature_refiner_bottleneck_dim",
        dest="eagle_feature_refiner_bottleneck_dim",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--locate_feature_refiner_init_scale",
        "--eagle_feature_refiner_init_scale",
        dest="eagle_feature_refiner_init_scale",
        type=float,
        default=0.1,
    )
    parser.add_argument("--locate_lora_r", "--eagle_lora_r", dest="eagle_lora_r", type=int, default=32)
    parser.add_argument("--locate_lora_alpha", "--eagle_lora_alpha", dest="eagle_lora_alpha", type=int, default=64)
    parser.add_argument("--locate_lora_dropout", "--eagle_lora_dropout", dest="eagle_lora_dropout", type=float, default=0.05)
    parser.add_argument("--locate_vision_lora_r", "--eagle_vision_lora_r", dest="eagle_vision_lora_r", type=int, default=16)
    parser.add_argument("--locate_vision_lora_alpha", "--eagle_vision_lora_alpha", dest="eagle_vision_lora_alpha", type=int, default=32)
    parser.add_argument("--locate_vision_lora_dropout", "--eagle_vision_lora_dropout", dest="eagle_vision_lora_dropout", type=float, default=0.05)

    parser.add_argument("--hidden_dim", type=int, default=448)
    parser.add_argument("--pose_decoder_layers", type=int, default=3)
    parser.add_argument("--refinement_steps", type=int, default=3)
    parser.add_argument("--decoder_heads", type=int, default=8)
    parser.add_argument("--box_condition_scale", type=float, default=1.25)
    parser.add_argument("--pose_roi_size", type=int, default=16)
    parser.add_argument(
        "--box_source",
        choices=["gt", "qwen_generate", "locate_generate", "person_queries"],
        default=None,
    )
    parser.add_argument("--qwen_box_max_new_tokens", type=int, default=4096)
    parser.add_argument("--locate_box_max_new_tokens", type=int, default=8192)
    parser.add_argument("--locate_generation_mode", choices=["fast", "slow", "hybrid"], default="hybrid")
    parser.add_argument("--locate_generation_backend", choices=["transformers", "vllm", "auto"], default="transformers")
    parser.add_argument("--single_pass_prompt", choices=["locate", "pose"], default="locate")
    parser.add_argument("--disable_single_pass_features", action="store_true")
    parser.add_argument("--disable_vllm_fallback", action="store_true")
    parser.add_argument("--gpu", type=str, default="")
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--vllm_cpu_offload_gb", type=float, default=0.0)
    parser.add_argument("--vllm_enforce_eager", action="store_true")
    parser.add_argument("--vllm_max_model_len", type=int, default=0)
    parser.add_argument("--vllm_batch_size", type=int, default=0)
    parser.add_argument("--vllm_max_num_seqs", type=int, default=0)
    parser.add_argument("--vllm_max_num_batched_tokens", type=int, default=0)
    parser.add_argument("--vllm_model_impl", choices=["auto", "transformers", "vllm"], default="auto")
    parser.add_argument("--vllm_lora_adapter", type=str, default="auto")
    parser.add_argument("--vllm_max_lora_rank", type=int, default=64)
    parser.add_argument("--vllm_trust_remote_code", action="store_true", default=True)
    parser.add_argument("--no_vllm_trust_remote_code", dest="vllm_trust_remote_code", action="store_false")
    parser.add_argument("--box_match_iou_thresh", type=float, default=0.10)
    parser.add_argument("--box_nms_iou_thresh", type=float, default=0.70)
    parser.add_argument(
        "--disable_pre_pose_nms",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--post_pose_nms_iou_thresh", type=float, default=0.95)
    parser.add_argument(
        "--ref_pose_quality_alpha",
        type=float,
        default=0.25,
        help="Exponent applied to pose quality when ranking RefHuman candidates.",
    )
    parser.add_argument(
        "--keypoint_decode_mode",
        choices=["regression"],
        default="regression",
        help="Final keypoint coordinates use the direct regression head.",
    )
    parser.add_argument(
        "--keypoint_decode_modes",
        type=str,
        default="",
        help="Compatibility option; only regression is supported.",
    )
    parser.add_argument("--disable_refinement", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.0,
        help=(
            "Pre-COCO filtering threshold. Official evaluation defaults to zero "
            "so pycocotools performs the ranking and per-image top-20 truncation."
        ),
    )
    parser.add_argument("--max_predictions_per_image", type=int, default=100)
    parser.add_argument("--visualize_max_samples", type=int, default=100)
    parser.add_argument("--visualize_max_instances", type=int, default=8)

    parser.add_argument("--w_oks", type=float, default=0.5)
    parser.add_argument("--w_coord", type=float, default=3.0)
    parser.add_argument(
        "--w_keypoint_confidence",
        "--w_vis",
        dest="w_keypoint_confidence",
        type=float,
        default=0.1,
    )
    parser.add_argument("--w_hard_joint", type=float, default=0.0)
    parser.add_argument("--hard_joint_fraction", type=float, default=0.2)
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
    feature_config = checkpoint.get("backbone_feature_config", checkpoint.get("qwen_feature_config", {}))
    saved_feature_source = str(feature_config.get("feature_source", "raw_visual"))
    saved_pose_config = checkpoint.get("pose_config") or {}
    prune_locate_generation = bool(
        feature_config.get(
            "generation_components_pruned",
            saved_pose_config.get("use_global_person_queries", False),
        )
    ) and str(getattr(args, "box_source", "")) != "locate_generate"

    if backbone_name == "eagle":
        eagle_loader = (
            load_eagle_vision_only_with_lora
            if saved_feature_source == "vision_only"
            else load_eagle_with_lora
        )
        backbone_model, backbone_processor = eagle_loader(
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
                prune_generation=prune_locate_generation,
            )
        )
        if "backbone_trainable" in checkpoint or "qwen_trainable" in checkpoint:
            backbone_model.load_state_dict(checkpoint["backbone_trainable"] if "backbone_trainable" in checkpoint else checkpoint["qwen_trainable"], strict=False)
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
        qwen_state = checkpoint.get("backbone_trainable", checkpoint.get("qwen_trainable"))
        if not backbone_merged and qwen_state is not None:
            backbone_model.load_state_dict(qwen_state, strict=False)
        backbone_model.to(device)
        backbone_model.eval()
        external_dim = qwen_hidden_size(backbone_model)
        default_feature_size = args.qwen_feature_size
        default_refiner_layers = args.qwen_feature_refiner_layers
        default_refiner_bottleneck_dim = args.qwen_feature_refiner_bottleneck_dim
        default_refiner_init_scale = args.qwen_feature_refiner_init_scale

    saved_pose_config = checkpoint.get("pose_config")
    if saved_pose_config is not None and (
        "pose_pyramid_channels" not in saved_pose_config
        or int(saved_pose_config.get("rgb_input_size", 0)) != 800
    ):
        raise ValueError(
            "This checkpoint predates the unified 800x800 pose pyramid and cannot be evaluated "
            "with the new architecture. Train a new Stage1 checkpoint first."
        )
    pose_config_kwargs = (
        {
            "hidden_dim": args.hidden_dim,
            "external_dim": external_dim,
            "pose_decoder_layers": args.pose_decoder_layers,
            "refinement_steps": args.refinement_steps,
            "decoder_heads": args.decoder_heads,
            "box_condition_scale": args.box_condition_scale,
            "pose_roi_size": args.pose_roi_size,
            "use_refinement": not args.disable_refinement,
        }
        if saved_pose_config is None
        else {
            key: saved_pose_config[key]
            for key in QwenPoseConfig.__dataclass_fields__
            if key in saved_pose_config
        }
    )
    # Keypoint DN is training-only. Old unified-pyramid checkpoints do not own
    # its tiny type embedding, so keep their inference graph parameter-exact.
    if saved_pose_config is not None and "enable_keypoint_denoising" not in saved_pose_config:
        pose_config_kwargs["enable_keypoint_denoising"] = False
    # Checkpoints created before the center-reference migration used the
    # persistent schema skeleton in both query PE and coordinate residuals.
    if saved_pose_config is not None and "pose_coordinate_init" not in saved_pose_config:
        pose_config_kwargs["pose_coordinate_init"] = "schema_prior"
    pose_config_kwargs["external_dim"] = external_dim
    pose_model = QwenPoseModel(QwenPoseConfig(**pose_config_kwargs))
    pose_model.load_state_dict(checkpoint["model"], strict=True)
    if pose_model.person_confidence_head is None:
        raise RuntimeError(
            "This checkpoint has no trained person confidence head. Official COCO "
            "pose AP requires one canonical instance score per pose; run the "
            "confidence-rescue migration before evaluation."
        )
    refiner_config = checkpoint.get("backbone_feature_refiner_config", checkpoint.get("qwen_feature_refiner_config", {}))
    has_refiner_checkpoint = "backbone_feature_refiner" in checkpoint or "qwen_feature_refiner" in checkpoint
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
            feature_source=saved_feature_source,
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
        model.backbone_extractor.feature_refiner.load_state_dict(checkpoint["backbone_feature_refiner"] if "backbone_feature_refiner" in checkpoint else checkpoint["qwen_feature_refiner"], strict=True)
    model.eval()
    return model, backbone_processor


def tensor_to_prediction_rows(
    outputs: dict[str, torch.Tensor],
    batch: dict,
    args: argparse.Namespace,
    raw_boxes_abs: list[list[list[float]]] | None = None,
) -> list[dict]:
    rows: list[dict] = []
    if not bool(outputs.get("person_confidence_head_available", False)):
        raise RuntimeError(
            "Official COCO pose evaluation requires a trained person confidence "
            "head. Refusing to rank poses with fixed person logits or an implicit "
            "keypoint-confidence fallback."
        )
    detection_scores = outputs["person_logits"].sigmoid().detach().cpu()
    ref_logits = outputs["ref_logits"].detach().cpu()
    pose_quality_logits = outputs.get("pose_quality_logits")
    pose_quality_scores = (
        pose_quality_logits.sigmoid().detach().cpu()
        if torch.is_tensor(pose_quality_logits)
        else detection_scores
    )
    ref_match_scores = ref_logits.sigmoid()
    ref_quality_alpha = max(float(getattr(args, "ref_pose_quality_alpha", 0.25)), 0.0)
    ref_final_scores = ref_match_scores * pose_quality_scores.clamp_min(1e-6).pow(
        ref_quality_alpha
    )
    box_mask = outputs.get("box_mask")
    box_mask_cpu = box_mask.detach().cpu().bool() if box_mask is not None else torch.ones_like(detection_scores, dtype=torch.bool)
    refinement_fallback = outputs.get("ref_box_refinement_fallback_mask")
    refinement_fallback_cpu = (
        refinement_fallback.detach().cpu().bool()
        if torch.is_tensor(refinement_fallback)
        else torch.zeros_like(box_mask_cpu)
    )
    boxes = outputs["boxes"].detach().cpu()
    pose_boxes = outputs.get("pose_boxes", outputs["boxes"]).detach().cpu()
    keypoints = outputs["keypoints"].detach().cpu()
    schema_valid_output = outputs.get("keypoint_valid_mask")
    if torch.is_tensor(schema_valid_output):
        pose_scores_all = topk_keypoint_confidence(
            keypoints,
            schema_valid_output.detach().cpu().bool(),
            fraction=0.5,
        ).cpu()
    else:
        pose_scores_all = keypoints[..., 2].mean(dim=-1)
    # Canonical COCO ranking score: one learned scalar per pose query. Joint
    # quality remains diagnostic metadata and is never multiplied into score.
    final_scores = detection_scores
    for b, target in enumerate(batch["targets"]):
        width = float(target["width"])
        height = float(target["height"])
        task_id = int(batch["task_ids"][b].cpu().item())
        detection_boxes = boxes[b].clone()
        sample_raw_boxes = (
            raw_boxes_abs[b]
            if raw_boxes_abs is not None and b < len(raw_boxes_abs)
            else []
        )
        valid = torch.nonzero(box_mask_cpu[b], as_tuple=False).flatten()
        direct_refhuman = task_id == 1 and valid.numel() == 1
        if task_id == 1:
            if valid.numel() > 0:
                best_local = int(torch.argmax(ref_final_scores[b, valid]).item())
                selected = [int(valid[best_local].item())]
            else:
                selected = []
        else:
            keep = valid[final_scores[b, valid] >= args.score_threshold] if valid.numel() > 0 else valid
            if keep.numel() == 0 and valid.numel() > 0:
                keep = valid[:1]
            nms_scores = final_scores[b, keep] if keep.numel() > 0 else torch.zeros(0)
            if str(getattr(args, "box_source", "")).lower() == "gt":
                # GT-box evaluation already has one query per annotated person;
                # box NMS could incorrectly remove distinct crowded instances.
                order = torch.argsort(nms_scores, descending=True)
                order = order[: max(int(args.max_predictions_per_image), 0)]
                selected = [int(keep[idx].item()) for idx in order.tolist()]
            else:
                kept_local = nms_box_indices_xyxy(
                    detection_boxes[keep],
                    nms_scores,
                    iou_thresh=float(getattr(args, "post_pose_nms_iou_thresh", 0.95)),
                    max_boxes=args.max_predictions_per_image,
                )
                selected = [int(keep[idx].item()) for idx in kept_local]

        predictions = []
        for query_idx in selected:
            box = detection_boxes[query_idx].tolist()
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
            prediction_score = (
                pose_quality_scores[b, query_idx]
                if direct_refhuman
                else ref_final_scores[b, query_idx]
                if task_id == 1
                else final_scores[b, query_idx]
            )
            prediction = {
                "query": query_idx,
                "person_score": float(detection_scores[b, query_idx].item()),
                "pose_score": float(pose_scores_all[b, query_idx].item()),
                "pose_quality_score": float(pose_quality_scores[b, query_idx].item()),
                "score": float(prediction_score.item()),
                "ref_score": float(ref_match_scores[b, query_idx].item()),
                "ref_grounding_mode": (
                    "direct" if direct_refhuman else "candidate_fallback"
                ) if task_id == 1 else None,
                "ref_box_refinement_fallback": bool(
                    refinement_fallback_cpu[b, query_idx].item()
                ),
                "bbox_2d": box_abs,
                "pose_bbox_2d": pose_box_abs,
                "keypoints": kp.tolist(),
            }
            if query_idx < len(sample_raw_boxes) and len(sample_raw_boxes[query_idx]) == 4:
                prediction["input_bbox_2d"] = [
                    float(value) for value in sample_raw_boxes[query_idx]
                ]
            predictions.append(prediction)
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
                "keypoint_decode_mode": str(getattr(args, "keypoint_decode_mode", "regression")),
                "num_gt": int(target["boxes"].shape[0]),
                "predictions": predictions,
            }
        )
    return rows


def target_to_metric_target(target: dict) -> dict:
    """Detach target tensors before keeping them for end-of-run metrics."""
    copied = dict(target)
    for key in ("boxes", "keypoints", "keypoint_valid", "ref_target"):
        value = copied.get(key)
        if isinstance(value, torch.Tensor):
            copied[key] = value.detach().cpu()
    return copied


def requested_keypoint_decode_modes(args: argparse.Namespace) -> list[str]:
    raw = str(getattr(args, "keypoint_decode_modes", "") or "").strip()
    values = raw.split(",") if raw else [str(args.keypoint_decode_mode)]
    modes: list[str] = []
    for value in values:
        mode = value.strip().lower()
        if mode != "regression":
            raise ValueError(
                f"Unsupported keypoint decode mode {mode!r}; only regression is available."
            )
        if mode not in modes:
            modes.append(mode)
    return modes


def dataset_records_in_order(dataset) -> list:
    if hasattr(dataset, "records"):
        return list(dataset.records)
    if isinstance(dataset, ConcatDataset):
        records = []
        for child in dataset.datasets:
            records.extend(dataset_records_in_order(child))
        return records
    if hasattr(dataset, "datasets"):
        records = []
        for child in dataset.datasets:
            records.extend(dataset_records_in_order(child))
        return records
    raise TypeError(f"Cannot extract PoseRecord list from dataset type {type(dataset).__name__}")


def main() -> None:
    args = parse_args()
    if args.backbone == "locatepose":
        args.backbone = "eagle"
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
    if args.qwen_box_max_new_tokens <= 0:
        raise ValueError("--qwen_box_max_new_tokens must be positive.")
    if args.locate_box_max_new_tokens <= 0:
        raise ValueError("--locate_box_max_new_tokens must be positive.")
    if not 0.0 <= args.box_match_iou_thresh <= 1.0:
        raise ValueError("--box_match_iou_thresh must be in [0, 1].")
    if not 0.0 <= args.box_nms_iou_thresh <= 1.0:
        raise ValueError("--box_nms_iou_thresh must be in [0, 1].")
    if not 0.0 <= args.post_pose_nms_iou_thresh <= 1.0:
        raise ValueError("--post_pose_nms_iou_thresh must be in [0, 1].")
    if args.ref_pose_quality_alpha < 0.0:
        raise ValueError("--ref_pose_quality_alpha must be non-negative.")
    if args.box_source == "qwen_generate" and args.backbone != "qwen3vl":
        raise ValueError("--box_source=qwen_generate currently requires --backbone qwen3vl.")
    if args.box_source == "locate_generate" and args.backbone != "eagle":
        raise ValueError("--box_source=locate_generate currently requires --backbone eagle/locatepose.")
    if args.vllm_tensor_parallel_size <= 0:
        raise ValueError("--vllm_tensor_parallel_size must be positive.")
    if not 0.0 < args.vllm_gpu_memory_utilization <= 1.0:
        raise ValueError("--vllm_gpu_memory_utilization must be in (0, 1].")
    if args.vllm_batch_size <= 0:
        args.vllm_batch_size = int(args.batch_size)
    if args.vllm_max_num_seqs <= 0:
        args.vllm_max_num_seqs = int(args.vllm_batch_size)
    if args.vllm_max_num_batched_tokens <= 0:
        args.vllm_max_num_batched_tokens = int(args.vllm_max_model_len) if int(args.vllm_max_model_len) > 0 else 2048
    if args.vllm_max_lora_rank <= 0:
        raise ValueError("--vllm_max_lora_rank must be positive.")
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
    if args.eagle_image_token_limit is not None and args.eagle_image_token_limit <= 0:
        raise ValueError("--locate_image_token_limit/--eagle_image_token_limit must be positive when set.")
    if args.eagle_batch_token_limit is not None and args.eagle_batch_token_limit <= 0:
        raise ValueError("--locate_batch_token_limit/--eagle_batch_token_limit must be positive when set.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = resolve_checkpoint(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if args.box_source is None:
        pose_config = checkpoint.get("pose_config") or {}
        args.box_source = (
            "person_queries"
            if bool(pose_config.get("use_global_person_queries", False))
            else ("locate_generate" if args.backbone == "eagle" else "qwen_generate")
        )
    device = torch.device(args.device)
    pose_image_size = int((checkpoint.get("pose_config") or {}).get("rgb_input_size", 640))

    dataset_names = [name.strip() for name in args.datasets.split(",") if name.strip()]
    dataset = build_datasets(
        dataset_root=args.dataset_root,
        names=dataset_names,
        max_instances=args.max_instances,
        image_size=pose_image_size,
        split=args.split,
        max_samples_per_dataset=args.max_samples_per_dataset,
        mixing_strategy="concat_shuffle",
        seed=42,
        record_cache_dir=args.record_cache_dir,
        disable_record_cache=args.disable_record_cache,
        show_progress=not args.disable_progress,
    )
    records_in_order = dataset_records_in_order(dataset)
    backend = str(args.locate_generation_backend or "transformers").lower()
    use_vllm_integrated = args.box_source == "locate_generate" and backend == "vllm"
    vllm_generator = None
    if use_vllm_integrated:
        print(
            "[Locate generation] backend=vllm integrated: LocateAnything+PoseHead in the custom vLLM model.",
            flush=True,
        )
        from qwenpose.infer_locatepose import VLLMLocateBBoxGenerator, apply_gpu_selection

        apply_gpu_selection(args)
        try:
            vllm_generator = VLLMLocateBBoxGenerator(args, checkpoint_path)
        except Exception as exc:
            if args.disable_vllm_fallback:
                raise
            print(
                "[Locate generation] integrated vLLM failed during initialization; falling back to transformers. "
                f"Reason: {type(exc).__name__}: {exc}",
                flush=True,
            )
            use_vllm_integrated = False
            backend = "transformers"
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
    model = None
    backbone_processor = None
    if not use_vllm_integrated:
        model, backbone_processor = load_eval_model(args, checkpoint, device)
    use_single_pass_features = (
        args.box_source == "locate_generate"
        and not use_vllm_integrated
        and backend in {"transformers", "auto"}
        and not args.disable_single_pass_features
    )
    if use_single_pass_features:
        print(
            f"[Locate generation] transformers single-pass features enabled; prompt={args.single_pass_prompt}",
            flush=True,
        )
    unified_config = LocatePoseUnifiedConfig.from_args(
        args,
        use_single_pass_features=use_single_pass_features,
        keep_unmatched_predictions=True,
    )
    unified_runtime = None
    if not use_vllm_integrated:
        unified_runtime = LocatePoseUnifiedRuntime(
            model,
            backbone_processor,
            device,
            backbone_name=args.backbone,
        )
    weights = LossWeights(
        oks=args.w_oks,
        coord=args.w_coord,
        keypoint_confidence=args.w_keypoint_confidence,
        hard_joint=args.w_hard_joint,
        hard_joint_fraction=args.hard_joint_fraction,
    )

    totals: dict[str, float] = {}
    batches = 0
    samples = 0
    decode_modes = requested_keypoint_decode_modes(args)
    sweep_enabled = len(decode_modes) > 1
    mode_output_dirs = {
        mode: args.output_dir / mode if sweep_enabled else args.output_dir
        for mode in decode_modes
    }
    predictions_paths = {mode: path / "predictions.jsonl" for mode, path in mode_output_dirs.items()}
    visualization_dirs = {mode: path / "visualizations" for mode, path in mode_output_dirs.items()}
    for path in mode_output_dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    if args.visualize_max_samples > 0:
        for path in visualization_dirs.values():
            path.mkdir(parents=True, exist_ok=True)
    visualized_samples = {mode: 0 for mode in decode_modes}

    # Collect GT and predictions for AP computation
    all_gt_targets: list[dict] = []
    all_predictions_rows = {mode: [] for mode in decode_modes}
    # Per-dataset tracking
    dataset_gt: dict[str, list[dict]] = defaultdict(list)
    actual_single_pass_features_used = False
    actual_vllm_integrated_features_used = False
    response_offset = 0

    try:
        with ExitStack() as stack:
            prediction_writers = {
                mode: stack.enter_context(path.open("w", encoding="utf-8"))
                for mode, path in predictions_paths.items()
            }
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
                    prep_time = time.perf_counter() - prep_started
                    forward_started = time.perf_counter()
                    raw_boxes_abs_for_rows: list[list[list[float]]] | None = None
                    if use_vllm_integrated:
                        if vllm_generator is None:
                            raise RuntimeError("Integrated vLLM generator is not initialized.")
                        from qwenpose.infer_locatepose import validate_vllm_locate_responses

                        batch_count = len(batch["image_paths"])
                        records_for_batch = records_in_order[response_offset : response_offset + batch_count]
                        response_offset += batch_count
                        responses, feature_map, text_embed = vllm_generator.generate_with_features(records_for_batch)
                        validate_vllm_locate_responses(records_for_batch, responses)
                        _, _, gt_targets_for_eval = prepare_box_conditioning(
                            batch["targets"],
                            batch["task_ids"],
                            device,
                            max_instances=unified_config.max_instances,
                        )
                        target_boxes, target_box_mask, pose_targets = prepare_locate_generated_box_conditioning_from_responses(
                            responses,
                            batch,
                            device,
                            max_instances=unified_config.max_instances,
                            match_iou_thresh=unified_config.box_match_iou_thresh,
                            nms_iou_thresh=unified_config.box_nms_iou_thresh,
                            disable_pre_pose_nms=unified_config.disable_pre_pose_nms,
                        )
                        outputs = vllm_generator.run_pose(
                            batch,
                            target_boxes,
                            target_box_mask,
                            feature_map,
                            text_embed,
                        )
                        raw_boxes_abs_for_rows = locate_boxes_abs_from_responses(
                            responses,
                            batch,
                            max_instances=unified_config.max_instances,
                            nms_iou_thresh=unified_config.box_nms_iou_thresh,
                            disable_pre_pose_nms=unified_config.disable_pre_pose_nms,
                        )
                        actual_vllm_integrated_features_used = True
                    else:
                        if unified_runtime is None:
                            raise RuntimeError("LocatePose runtime is not initialized.")
                        unified_result = unified_runtime.eval_batch(
                            batch,
                            unified_config,
                            box_source=args.box_source,
                        )
                        outputs = unified_result.outputs
                        pose_targets = unified_result.pose_targets or batch["targets"]
                        gt_targets_for_eval = unified_result.gt_targets or pose_targets
                        raw_boxes_abs_for_rows = unified_result.locate_boxes_abs
                        actual_single_pass_features_used = (
                            actual_single_pass_features_used or unified_result.used_single_pass_features
                        )
                    _, loss_dict = compute_pose_losses(outputs, pose_targets, batch["task_ids"], weights)
                    decoded_outputs = {
                        mode: apply_keypoint_decode_mode(outputs, mode=mode)
                        for mode in decode_modes
                    }
                    for mode in decode_modes:
                        decoded_outputs[mode] = dict(decoded_outputs[mode])
                        decoded_outputs[mode]["keypoint_decode_mode"] = mode
                    forward_time = time.perf_counter() - forward_started
                    for key, value in loss_dict.items():
                        totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())

                    rows_by_mode: dict[str, list[dict]] = {}
                    for mode in decode_modes:
                        args.keypoint_decode_mode = mode
                        rows_by_mode[mode] = tensor_to_prediction_rows(
                            decoded_outputs[mode],
                            {**batch, "targets": gt_targets_for_eval},
                            args,
                            raw_boxes_abs=raw_boxes_abs_for_rows,
                        )
                        for row in rows_by_mode[mode]:
                            prediction_writers[mode].write(json.dumps(row, ensure_ascii=False) + "\n")
                            all_predictions_rows[mode].append(row)

                    reference_rows = rows_by_mode[decode_modes[0]]
                    for local_idx, row in enumerate(reference_rows):
                        ds_name = row.get("dataset", "unknown")
                        target_for_gt = gt_targets_for_eval[local_idx] if local_idx < len(gt_targets_for_eval) else {
                            "dataset": ds_name,
                            "image_id": row["image_id"],
                            "schema": row.get("schema", ""),
                            "width": row.get("width", 0),
                            "height": row.get("height", 0),
                            "boxes": torch.zeros(0, 4),
                            "keypoints": torch.zeros(0, len(UNION_KEYPOINTS), 3),
                            "keypoint_valid": torch.zeros(0, len(UNION_KEYPOINTS), dtype=torch.bool),
                        }
                        target_for_gt = target_to_metric_target(target_for_gt)
                        all_gt_targets.append(target_for_gt)
                        dataset_gt[ds_name].append(target_for_gt)

                    if args.visualize_max_samples > 0:
                        eval_batch = {**batch, "targets": pose_targets}
                        batch_size = len(batch["schema_ids"])
                        for mode in decode_modes:
                            for local_idx in range(batch_size):
                                if visualized_samples[mode] >= args.visualize_max_samples:
                                    break
                                vis_path = visualization_dirs[mode] / f"eval_{samples + local_idx:06d}.jpg"
                                save_pose_visualization(
                                    decoded_outputs[mode],
                                    eval_batch,
                                    vis_path,
                                    sample_idx=local_idx,
                                    max_instances=args.visualize_max_instances,
                                )
                                visualized_samples[mode] += 1
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
    finally:
        if vllm_generator is not None:
            vllm_generator.close()

    # ── Compute dataset-specific pose metrics ─────────────────────────────
    print("\n" + "=" * 60)
    print("Computing dataset-specific pose metrics...")
    print("=" * 60)

    pose_metrics_by_mode = {
        mode: compute_pose_metrics_from_targets(
            all_predictions_rows[mode],
            all_gt_targets,
            dataset_root=args.dataset_root,
            split=args.split,
        )
        for mode in decode_modes
    }
    official_coco_metrics_by_mode: dict[str, dict[str, float]] = {}
    if "coco" in dataset_names:
        print("Computing official pycocotools COCO keypoint AP...")
        official_coco_metrics_by_mode = {
            mode: compute_official_coco_keypoint_ap(
                all_predictions_rows[mode],
                args.dataset_root,
                args.split,
            )
            for mode in decode_modes
        }
    print(json.dumps(pose_metrics_by_mode, ensure_ascii=False, indent=2))
    if official_coco_metrics_by_mode:
        print(json.dumps({"official_coco": official_coco_metrics_by_mode}, ensure_ascii=False, indent=2))

    predictions_json_paths = {
        mode: mode_output_dirs[mode] / "predictions.json"
        for mode in decode_modes
    }
    for mode in decode_modes:
        predictions_json_paths[mode].write_text(
            json.dumps(all_predictions_rows[mode], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    primary_mode = decode_modes[0]
    args.keypoint_decode_mode = primary_mode
    pose_metrics = pose_metrics_by_mode[primary_mode]
    ap_metrics_all = pose_metrics.get("overall_ap", {})
    ap_metrics_per_dataset = pose_metrics.get("per_dataset", {})
    predictions_path = predictions_paths[primary_mode]
    predictions_json_path = predictions_json_paths[primary_mode]
    visualization_dir = visualization_dirs[primary_mode]

    mode_summaries: dict[str, dict] = {}
    for mode in decode_modes:
        metrics = pose_metrics_by_mode[mode]
        mode_summary = {
            "checkpoint": str(checkpoint_path),
            "backbone": args.backbone,
            "box_source": args.box_source,
            "keypoint_decode_mode": mode,
            "datasets": dataset_names,
            "split": args.split,
            "samples": samples,
            "batches": batches,
            "pose_metrics": metrics,
            "ap_metrics": metrics.get("overall_ap", {}),
            "ap_metrics_per_dataset": metrics.get("per_dataset", {}),
            "official_coco_keypoint_metrics": official_coco_metrics_by_mode.get(mode),
            "predictions_jsonl": str(predictions_paths[mode]),
            "predictions_json": str(predictions_json_paths[mode]),
            "visualizations_dir": str(visualization_dirs[mode]) if args.visualize_max_samples > 0 else None,
            "visualized_samples": visualized_samples[mode],
        }
        mode_summaries[mode] = mode_summary
        (mode_output_dirs[mode] / "summary.json").write_text(
            json.dumps(mode_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ── Summary ────────────────────────────────────────────────────────────
    summary = {
        "checkpoint": str(checkpoint_path),
        "backbone": args.backbone,
        "box_source": args.box_source,
        "locate_generation_backend": args.locate_generation_backend,
        "vllm_integrated_features_used": actual_vllm_integrated_features_used,
        "single_pass_features_used": actual_single_pass_features_used,
        "single_pass_prompt": args.single_pass_prompt,
        "keypoint_decode_mode": args.keypoint_decode_mode,
        "datasets": dataset_names,
        "split": args.split,
        "samples": samples,
        "batches": batches,
        "avg_losses": {key: value / max(batches, 1) for key, value in totals.items()},
        "pose_metrics": pose_metrics,
        "ap_metrics": ap_metrics_all,
        "ap_metrics_per_dataset": ap_metrics_per_dataset,
        "official_coco_keypoint_metrics": official_coco_metrics_by_mode.get(primary_mode),
        "predictions_jsonl": str(predictions_path),
        "predictions_json": str(predictions_json_path),
        "visualizations_dir": str(visualization_dir) if args.visualize_max_samples > 0 else None,
        "visualized_samples": visualized_samples[primary_mode],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── Report ─────────────────────────────────────────────────────────────
    report = [
        "# Pose Eval Report",
        "",
        f"- checkpoint: `{checkpoint_path}`",
        f"- backbone: `{args.backbone}`",
        f"- box_source: `{args.box_source}`",
        f"- locate_generation_backend: `{args.locate_generation_backend}`",
        f"- vllm_integrated_features_used: `{actual_vllm_integrated_features_used}`",
        f"- single_pass_features_used: `{actual_single_pass_features_used}`",
        f"- single_pass_prompt: `{args.single_pass_prompt}`",
        f"- keypoint_decode_mode: `{args.keypoint_decode_mode}`",
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

    report.extend(["", "## Pose Metrics (Overall AP Datasets)", ""])
    if ap_metrics_all:
        report.append("| Metric | Value |")
        report.append("|--------|-------|")
        for k, v in ap_metrics_all.items():
            if isinstance(v, (int, float)):
                report.append(f"| {k} | {float(v):.4f} |")
    else:
        report.append("No metrics available.")

    official_coco_metrics = summary.get("official_coco_keypoint_metrics")
    if official_coco_metrics:
        report.extend(["", "## Official COCO Keypoint Metrics (pycocotools)", ""])
        report.append("| Metric | Value |")
        report.append("|--------|-------|")
        for key, value in official_coco_metrics.items():
            report.append(f"| {key} | {float(value):.4f} |")

    for ds_name, ds_metrics in sorted(ap_metrics_per_dataset.items()):
        report.extend(["", f"## Pose Metrics ({ds_name})", ""])
        if ds_metrics:
            report.append("| Metric | Value |")
            report.append("|--------|-------|")
            for k, v in ds_metrics.items():
                if isinstance(v, (int, float)):
                    report.append(f"| {k} | {float(v):.4f} |")
            if isinstance(ds_metrics.get("per_joint"), dict):
                report.extend(["", "| Joint | PCKh |", "|-------|------|"])
                for joint, value in ds_metrics["per_joint"].items():
                    report.append(f"| {joint} | {float(value):.4f} |")
        else:
            report.append("No metrics available.")

    report.extend(["", f"Predictions JSONL: `{predictions_path}`", ""])
    report.extend([f"Predictions JSON: `{predictions_json_path}`", ""])
    if args.visualize_max_samples > 0:
        report.extend([f"Visualizations: `{visualization_dir}`", ""])
    (args.output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    if sweep_enabled:
        sweep_summary = {
            "checkpoint": str(checkpoint_path),
            "box_source": args.box_source,
            "decode_modes": decode_modes,
            "modes": mode_summaries,
        }
        (args.output_dir / "decode_sweep_summary.json").write_text(
            json.dumps(sweep_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
