from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional for minimal envs.
    tqdm = None

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qwenpose.data import build_datasets, pose_collate
from qwenpose.losses import LossWeights, compute_pose_losses
from qwenpose.model import QwenPoseConfig, QwenPoseModel, count_trainable_parameters
from qwenpose.qwen_lora import (
    QwenFeatureExtractor,
    QwenLoRAConfig,
    build_qwen_inputs,
    build_qwen_lm_inputs,
    count_qwen_lora_parameters,
    get_qwen_base_model,
    load_qwen_with_existing_lora,
    load_qwen_with_lora,
    qwen_hidden_size,
    qwen_forward_kwargs,
    _apply_chat_template_batch,
    _build_user_messages,
    _load_rgb_images,
    _move_processor_tensors_to_device,
    _processor_image_kwargs,
)
from qwenpose.eagle_lora import (
    EagleFeatureExtractor,
    EagleLoRAConfig,
    build_eagle_inputs,
    count_eagle_lora_parameters,
    load_eagle_with_lora,
    eagle_hidden_size,
)
from qwenpose.schemas import ID_TO_SCHEMA, UNION_KEYPOINTS

try:
    from qwenpose.eagle_lora import build_eagle_lm_inputs
except ImportError:  # pragma: no cover - optional for the Qwen-focused public snapshot.
    build_eagle_lm_inputs = None


CHECKPOINT_PAYLOAD_NAME = "qwenpose_checkpoint.pt"
DEEPSPEED_TAG = "deepspeed"


@dataclass
class QwenInitializationSource:
    requested_path: str
    backbone_model_path: str
    source_kind: str = "base_model"
    pose_checkpoint_path: Path | None = None
    adapter_path: Path | None = None
    is_merged_backbone: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-stage ALL_POSE + REF_POSE training.")

    # ---------------------------------------------------------------------
    # Data section: one command mixes regular pose datasets and RefHuman.
    # ---------------------------------------------------------------------
    parser.add_argument("--dataset_root", type=Path, default=Path("datasets"))
    parser.add_argument(
        "--datasets",
        type=str,
        default="coco,mpii,crowdpose,refhuman",
        help="Comma-separated datasets. RefHuman contributes REF_POSE; others contribute ALL_POSE.",
    )
    parser.add_argument("--max_instances", type=int, default=80)
    parser.add_argument("--image_size", type=int, default=256, help="Fixed RGB tensor size for the lightweight pose visual branch.")
    parser.add_argument(
        "--disable_image_tensors",
        action="store_true",
        help="Disable loading fixed-size RGB tensors for the pose visual branch.",
    )
    parser.add_argument("--max_samples_per_dataset", type=int, default=None)
    parser.add_argument(
        "--refhuman_max_captions_per_instance",
        type=int,
        default=2,
        help="Maximum captions kept for each RefHuman person instance. Captions are randomly sampled once at startup according to --seed. Use 0 to keep all captions.",
    )
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--record_cache_dir", type=Path, default=Path(".cache/qwenpose_records"))
    parser.add_argument("--disable_record_cache", action="store_true")
    parser.add_argument(
        "--mixing_strategy",
        choices=["interleave", "concat_shuffle"],
        default="interleave",
        help="interleave rotates datasets by weights; concat_shuffle keeps the old natural ConcatDataset shuffle.",
    )
    parser.add_argument(
        "--dataset_mix_weights",
        type=str,
        default="auto",
        help="Use auto for size-proportional interleaving, or manual weights like coco:4,aic:1.",
    )
    parser.add_argument(
        "--disable_homogeneous_batches",
        action="store_true",
        help="Disable one-dataset-per-batch sampling for interleaved multi-dataset training.",
    )

    # ---------------------------------------------------------------------
    # Model section: Qwen3-VL provides image/text features with LoRA, and all
    # new pose modules are fully trainable by default.
    # ---------------------------------------------------------------------
    parser.add_argument("--hidden_dim", type=int, default=448)
    parser.add_argument("--backbone", choices=["qwen3vl", "eagle", "locatepose"], default="qwen3vl")
    parser.add_argument("--qwen_model_path", type=str, default="weights/Qwen3-VL-4B-Instruct")
    parser.add_argument("--qwen_dtype", choices=["bfloat16", "float16", "float32", "auto", "none"], default="bfloat16")
    parser.add_argument("--qwen_attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--qwen_gradient_checkpointing", action="store_true")
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
    parser.add_argument("--freeze_qwen", action="store_true")
    # Eagle/Embodied (LocateAnything-3B) backbone options
    parser.add_argument("--eagle_model_path", type=str, default="weights/LocateAnything-3B")
    parser.add_argument("--eagle_dtype", choices=["bfloat16", "float16", "float32", "auto", "none"], default="bfloat16")
    parser.add_argument("--eagle_attn_implementation", type=str, default="sdpa")
    parser.add_argument("--eagle_gradient_checkpointing", action="store_true")
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
    parser.add_argument("--freeze_eagle", action="store_true")
    parser.add_argument("--pose_decoder_layers", type=int, default=1)
    parser.add_argument("--refinement_steps", type=int, default=3)
    parser.add_argument("--decoder_heads", type=int, default=8)
    parser.add_argument("--box_condition_scale", type=float, default=1.2)
    parser.add_argument("--pose_roi_size", type=int, default=16)
    parser.add_argument("--disable_refinement", action="store_true")

    # ---------------------------------------------------------------------
    # Optimization section.
    # ---------------------------------------------------------------------
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/qwenpose_debug"))
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="Optional step cap. Use 0 to train for the requested number of epochs.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--qwen_lora_lr_scale", type=float, default=1.0)
    parser.add_argument("--qwen_vision_lr_scale", type=float, default=0.01)
    parser.add_argument("--locate_lr_scale", type=float, default=0.05)
    parser.add_argument("--locate_vision_scale", type=float, default=0.02)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=Path,
        default=None,
        help="Resume from a checkpoint-* directory, checkpoint_step_*.pt file, or a run directory.",
    )
    parser.add_argument("--disable_progress", action="store_true")
    parser.add_argument(
        "--sync_timing",
        action="store_true",
        help="Synchronize CUDA around timing sections for more accurate profiling logs.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--deepspeed_config", type=Path, default=None)
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", -1)))

    # ---------------------------------------------------------------------
    # Loss weights. Keep the training target clean: coord + OKS + vis are the
    # main pose losses, Stage2 may add bbox LM supervision, and hard_joint is
    # retained only as an off-by-default ablation knob.
    # ---------------------------------------------------------------------
    parser.add_argument("--w_oks", type=float, default=0.2)
    parser.add_argument("--w_coord", type=float, default=5.0)
    parser.add_argument("--w_vis", type=float, default=0.05)
    parser.add_argument("--w_lm", type=float, default=0.05)
    parser.add_argument("--w_hard_joint", type=float, default=0.0)
    parser.add_argument("--hard_joint_fraction", type=float, default=0.2)
    parser.add_argument("--box_jitter_scale", type=float, default=0.05)
    parser.add_argument("--box_jitter_shift", type=float, default=0.0)
    parser.add_argument(
        "--box_source",
        choices=["gt", "qwen_generate"],
        default="gt",
        help="Box conditions for PoseHead: GT/teacher-forced boxes or boxes generated by Qwen.",
    )
    parser.add_argument("--qwen_box_max_new_tokens", type=int, default=4096)
    parser.add_argument("--box_match_iou_thresh", type=float, default=0.10)
    parser.add_argument("--box_nms_iou_thresh", type=float, default=0.70)
    parser.add_argument("--w_locate_box_lm", type=float, default=0.0)
    parser.add_argument("--w_locate_point_lm", type=float, default=0.0)
    parser.add_argument("--locate_lm_loss_every", type=int, default=2)
    parser.add_argument("--locate_lm_max_instances", type=int, default=20)
    parser.add_argument("--locate_lm_max_points", type=int, default=8)
    parser.add_argument("--disable_locate_grounding_aux", action="store_true")

    # ---------------------------------------------------------------------
    # Validation/debug section.
    # ---------------------------------------------------------------------
    parser.add_argument("--dry_run_data", action="store_true", help="Only load data and print one batch.")
    parser.add_argument(
        "--disable_batch_trace",
        action="store_true",
        help="Disable per-batch JSONL tracing. Tracing is enabled by default for OOM/debugging.",
    )
    parser.add_argument(
        "--batch_trace_file",
        type=Path,
        default=None,
        help="Optional JSONL path for per-batch trace output. Defaults to output_dir/batch_trace_rank{rank}.jsonl.",
    )
    parser.add_argument("--lm_loss_every", type=int, default=1)
    parser.add_argument("--lm_max_answer_instances", type=int, default=10)
    parser.add_argument("--visualize_every", type=int, default=0, help="Save one training visualization every N optimizer steps. Use 0 to disable.")
    parser.add_argument("--visualize_max_instances", type=int, default=8)
    args = parser.parse_args()
    if args.backbone == "locatepose":
        args.backbone = "eagle"
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    if "images" in batch:
        batch["images"] = batch["images"].to(device, non_blocking=True)
    batch["schema_ids"] = batch["schema_ids"].to(device, non_blocking=True)
    batch["task_ids"] = batch["task_ids"].to(device, non_blocking=True)
    return batch


def summarize_batch(batch: dict) -> str:
    target_counts = [int(t["boxes"].shape[0]) for t in batch["targets"]]
    ref_targets = [int(t["ref_target"].item()) for t in batch["targets"]]
    return json.dumps(
        {
            "schema_ids": batch["schema_ids"].tolist(),
            "task_ids": batch["task_ids"].tolist(),
            "source_datasets": batch.get("source_datasets", []),
            "target_counts": target_counts,
            "ref_targets": ref_targets,
            "first_prompt": batch["prompts"][0],
        },
        ensure_ascii=False,
        indent=2,
    )


def _truncate_text(text: object, limit: int = 160) -> str:
    text = str(text or "")
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)] + "..."


def _cuda_memory_snapshot(device: torch.device) -> dict[str, float | int | None]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {
            "device_index": None,
            "memory_allocated_mb": None,
            "memory_reserved_mb": None,
            "max_memory_allocated_mb": None,
            "max_memory_reserved_mb": None,
        }
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    divisor = float(1024**2)
    return {
        "device_index": int(device_index),
        "memory_allocated_mb": round(torch.cuda.memory_allocated(device_index) / divisor, 2),
        "memory_reserved_mb": round(torch.cuda.memory_reserved(device_index) / divisor, 2),
        "max_memory_allocated_mb": round(torch.cuda.max_memory_allocated(device_index) / divisor, 2),
        "max_memory_reserved_mb": round(torch.cuda.max_memory_reserved(device_index) / divisor, 2),
    }


def _extract_multimodal_trace(qwen_inputs: dict[str, torch.Tensor] | None) -> dict[str, object]:
    if not qwen_inputs:
        return {}

    trace: dict[str, object] = {}
    input_ids = qwen_inputs.get("input_ids")
    attention_mask = qwen_inputs.get("attention_mask")
    pixel_values = qwen_inputs.get("pixel_values")

    if torch.is_tensor(input_ids):
        trace["input_ids_shape"] = list(input_ids.shape)
    if torch.is_tensor(attention_mask):
        trace["attention_lengths"] = [
            int(row.sum().item()) for row in attention_mask.detach().cpu()
        ]
    if torch.is_tensor(pixel_values):
        trace["pixel_values_shape"] = list(pixel_values.shape)
        if pixel_values.ndim >= 3:
            trace["pixel_batch_shapes"] = [list(pixel_values[idx].shape) for idx in range(pixel_values.shape[0])]

    image_grid_thw = qwen_inputs.get("image_grid_thw")
    if torch.is_tensor(image_grid_thw):
        rows = image_grid_thw.detach().cpu().tolist()
        trace["image_grid_key"] = "image_grid_thw"
        trace["image_grids"] = rows
        trace["vision_token_counts"] = [
            int(max(int(t), 1) * max(int(h), 1) * max(int(w), 1))
            for t, h, w in rows
        ]
        return trace

    image_grid_hws = qwen_inputs.get("image_grid_hws")
    if torch.is_tensor(image_grid_hws):
        rows = image_grid_hws.detach().cpu().tolist()
        trace["image_grid_key"] = "image_grid_hws"
        trace["image_grids"] = rows
        trace["vision_token_counts"] = [
            int(max(int(h), 1) * max(int(w), 1))
            for h, w in rows
        ]
    return trace


def build_batch_trace_record(
    *,
    batch: dict,
    pose_targets: list[dict[str, torch.Tensor]] | None = None,
    epoch: int,
    batch_idx: int,
    global_step: int,
    micro_step: int,
    grad_accum_steps: int,
    rank: int,
    local_rank: int,
    device: torch.device,
    stage: str,
    qwen_inputs: dict[str, torch.Tensor] | None = None,
    loss: float | None = None,
    loss_dict: dict[str, float] | None = None,
    did_update: bool | None = None,
    skip_batch: bool | None = None,
    error: str | None = None,
    data_time: float | None = None,
    prep_time: float | None = None,
    forward_time: float | None = None,
    step_time: float | None = None,
) -> dict[str, object]:
    task_name_map = {0: "ALL_POSE", 1: "REF_POSE"}
    sample_summaries: list[dict[str, object]] = []
    target_counts: list[int] = []
    selected_target_counts: list[int] = []
    for sample_idx, target in enumerate(batch["targets"]):
        selected_target = pose_targets[sample_idx] if pose_targets is not None and sample_idx < len(pose_targets) else target
        task_id = int(batch["task_ids"][sample_idx].detach().cpu().item())
        schema_id = int(batch["schema_ids"][sample_idx].detach().cpu().item())
        target_count = int(target["boxes"].shape[0])
        selected_target_count = int(selected_target["boxes"].shape[0])
        target_counts.append(target_count)
        selected_target_counts.append(selected_target_count)
        sample_summaries.append(
            {
                "sample_idx": sample_idx,
                "source_dataset": (
                    batch.get("source_datasets", [])[sample_idx]
                    if sample_idx < len(batch.get("source_datasets", []))
                    else target.get("dataset", "")
                ),
                "dataset": str(target.get("dataset", "")),
                "schema": str(target.get("schema", ID_TO_SCHEMA.get(schema_id, "unknown"))),
                "schema_id": schema_id,
                "task": task_name_map.get(task_id, str(task_id)),
                "task_id": task_id,
                "image_id": str(target.get("image_id", "")),
                "image_path": batch["image_paths"][sample_idx],
                "width": int(target["width"]),
                "height": int(target["height"]),
                "image_area": int(target["width"]) * int(target["height"]),
                "target_count": target_count,
                "selected_target_count": selected_target_count,
                "ref_target": int(target["ref_target"].detach().cpu().item()),
                "selected_ref_target": int(selected_target["ref_target"].detach().cpu().item()),
                "prompt_preview": _truncate_text(batch["prompts"][sample_idx], limit=200),
            }
        )

    record: dict[str, object] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rank": int(rank),
        "local_rank": int(local_rank),
        "epoch": int(epoch) + 1,
        "batch_idx": int(batch_idx),
        "global_step": int(global_step),
        "micro_step": int(micro_step),
        "grad_accum_steps": int(grad_accum_steps),
        "stage": stage,
        "batch_size": len(batch["targets"]),
        "source_datasets": list(batch.get("source_datasets", [])),
        "target_count_total": int(sum(target_counts)),
        "target_count_max": int(max(target_counts)) if target_counts else 0,
        "selected_target_count_total": int(sum(selected_target_counts)),
        "selected_target_count_max": int(max(selected_target_counts)) if selected_target_counts else 0,
        "samples": sample_summaries,
        "multimodal": _extract_multimodal_trace(qwen_inputs),
        "cuda": _cuda_memory_snapshot(device),
    }
    timings = {
        "data_time_s": None if data_time is None else round(float(data_time), 4),
        "prep_time_s": None if prep_time is None else round(float(prep_time), 4),
        "forward_time_s": None if forward_time is None else round(float(forward_time), 4),
        "step_time_s": None if step_time is None else round(float(step_time), 4),
    }
    if any(value is not None for value in timings.values()):
        record["timings"] = timings
    if loss is not None:
        record["loss"] = float(loss)
    if loss_dict:
        record["loss_dict"] = {key: float(value) for key, value in loss_dict.items()}
    if did_update is not None:
        record["did_update"] = bool(did_update)
    if skip_batch is not None:
        record["skip_batch"] = bool(skip_batch)
    if error:
        record["error"] = str(error)
    return record


def append_batch_trace(trace_handle, record: dict[str, object]) -> None:
    if trace_handle is None:
        return
    trace_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    trace_handle.flush()


def build_lm_responses(batch: dict, max_instances: int = 10) -> list[str]:
    responses = []
    for sample_idx, target in enumerate(batch["targets"]):
        width = float(target["width"])
        height = float(target["height"])
        boxes = target["boxes"].detach().cpu()
        task_id = int(batch["task_ids"][sample_idx].detach().cpu().item())
        if task_id == 1:
            ref_target = int(target["ref_target"].detach().cpu().item())
            instance_indices = [ref_target] if 0 <= ref_target < int(boxes.shape[0]) else []
        else:
            instance_indices = list(range(min(int(boxes.shape[0]), max_instances)))

        people = []
        for person_idx in instance_indices:
            box = boxes[person_idx].tolist()
            person = {
                "bbox_2d": [
                    round(float(box[0]) * width),
                    round(float(box[1]) * height),
                    round(float(box[2]) * width),
                    round(float(box[3]) * height),
                ],
                "label": "person",
            }
            people.append(person)

        if task_id == 1:
            payload = {"person": people[0] if people else None}
        else:
            payload = {"people": people}
        responses.append(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    return responses


def _locate_coord_token(value: float, upper: float) -> str:
    if upper <= 0:
        scaled = 0
    else:
        scaled = int(round(max(0.0, min(float(value), float(upper))) / float(upper) * 1000.0))
    scaled = max(0, min(scaled, 1000))
    return "<" + f"{scaled:03d}" + ">"


def build_locate_grounding_responses(
    batch: dict,
    max_instances: int = 20,
    max_points: int = 8,
) -> tuple[list[str], list[str]]:
    box_responses: list[str] = []
    point_responses: list[str] = []
    box_start = "<" + "box>"
    box_end = "<" + "/box>"
    for sample_idx, target in enumerate(batch["targets"]):
        width = float(target["width"])
        height = float(target["height"])
        boxes = target["boxes"].detach().cpu()
        keypoints = target["keypoints"].detach().cpu()
        keypoint_mask = target["keypoint_mask"].detach().cpu().bool()
        task_id = int(batch["task_ids"][sample_idx].detach().cpu().item())
        if task_id == 1:
            ref_target = int(target["ref_target"].detach().cpu().item())
            instance_indices = [ref_target] if 0 <= ref_target < int(boxes.shape[0]) else []
        else:
            instance_indices = list(range(min(int(boxes.shape[0]), max_instances)))
        box_chunks: list[str] = []
        point_chunks: list[str] = []
        for person_idx in instance_indices:
            box = boxes[person_idx].tolist()
            box_chunks.append(
                box_start
                + _locate_coord_token(box[0] * width, width)
                + _locate_coord_token(box[1] * height, height)
                + _locate_coord_token(box[2] * width, width)
                + _locate_coord_token(box[3] * height, height)
                + box_end
            )
            visible_indices = torch.nonzero(keypoint_mask[person_idx], as_tuple=False).flatten().tolist()
            for joint_idx in visible_indices[: max(0, int(max_points))]:
                xy = keypoints[person_idx, joint_idx].tolist()
                point_chunks.append(
                    box_start
                    + _locate_coord_token(xy[0] * width, width)
                    + _locate_coord_token(xy[1] * height, height)
                    + box_end
                )
        box_responses.append("".join(box_chunks) if box_chunks else "None")
        point_responses.append("".join(point_chunks) if point_chunks else "None")
    return box_responses, point_responses


def prepare_box_conditioning(
    targets: list[dict[str, torch.Tensor]],
    task_ids: torch.Tensor,
    device: torch.device,
    max_instances: int | None = None,
    box_jitter_scale: float = 0.0,
    box_jitter_shift: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, torch.Tensor]]]:
    selected_targets: list[dict[str, torch.Tensor]] = []
    selected_boxes: list[torch.Tensor] = []
    for sample_idx, target in enumerate(targets):
        boxes = target["boxes"]
        num_boxes = int(boxes.shape[0])
        task_id = int(task_ids[sample_idx].detach().cpu().item())
        if task_id == 1:
            ref_target = int(target["ref_target"].detach().cpu().item())
            indices = [ref_target] if 0 <= ref_target < num_boxes else []
        else:
            limit = num_boxes if max_instances is None else min(num_boxes, int(max_instances))
            indices = list(range(limit))
        index_tensor = torch.as_tensor(indices, dtype=torch.long)
        selected = dict(target)
        selected["boxes"] = target["boxes"][index_tensor].clone() if indices else target["boxes"][:0].clone()
        selected["keypoints"] = (
            target["keypoints"][index_tensor].clone() if indices else target["keypoints"][:0].clone()
        )
        selected["keypoint_valid"] = (
            target["keypoint_valid"][index_tensor].clone() if indices else target["keypoint_valid"][:0].clone()
        )
        selected["ref_target"] = torch.tensor(0 if task_id == 1 and indices else -1, dtype=torch.long)
        selected_targets.append(selected)
        condition_boxes = selected["boxes"]
        if condition_boxes.numel() > 0 and (box_jitter_scale > 0.0 or box_jitter_shift > 0.0):
            condition_boxes = jitter_boxes_xyxy(
                condition_boxes,
                scale_jitter=box_jitter_scale,
                shift_jitter=box_jitter_shift,
            )
        selected_boxes.append(condition_boxes)

    max_boxes = max([int(boxes.shape[0]) for boxes in selected_boxes] + [1])
    box_tensor = torch.zeros(len(selected_boxes), max_boxes, 4, dtype=torch.float32, device=device)
    box_mask = torch.zeros(len(selected_boxes), max_boxes, dtype=torch.bool, device=device)
    for sample_idx, boxes in enumerate(selected_boxes):
        n = int(boxes.shape[0])
        if n == 0:
            continue
        box_tensor[sample_idx, :n] = boxes.to(device=device, dtype=torch.float32)
        box_mask[sample_idx, :n] = True
    return box_tensor, box_mask, selected_targets


def jitter_boxes_xyxy(
    boxes: torch.Tensor,
    scale_jitter: float = 0.0,
    shift_jitter: float = 0.0,
) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes
    boxes = boxes.float()
    xy1 = boxes[:, :2]
    xy2 = boxes[:, 2:]
    wh = (xy2 - xy1).clamp(min=1e-4)
    center = (xy1 + xy2) * 0.5
    if shift_jitter > 0.0:
        center = center + (torch.rand_like(center) * 2.0 - 1.0) * wh * float(shift_jitter)
    if scale_jitter > 0.0:
        scale = 1.0 + (torch.rand(boxes.shape[0], 1, device=boxes.device) * 2.0 - 1.0) * float(scale_jitter)
        wh = wh * scale.clamp(min=0.5)
    jittered = torch.cat([center - wh * 0.5, center + wh * 0.5], dim=-1)
    return jittered.clamp(0.0, 1.0)


def box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((int(boxes1.shape[0]), int(boxes2.shape[0])))
    boxes1 = boxes1.float()
    boxes2 = boxes2.float()
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0.0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0.0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0.0))[:, None]
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0.0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0.0))[None, :]
    return inter / (area1 + area2 - inter).clamp(min=1e-8)


def nms_boxes_xyxy(boxes: torch.Tensor, iou_thresh: float, max_boxes: int) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)
    boxes = boxes.float().clamp(0.0, 1.0)
    max_boxes = max(int(max_boxes), 0)
    if max_boxes == 0:
        return boxes[:0]
    keep: list[int] = []
    for idx in range(int(boxes.shape[0])):
        if len(keep) >= max_boxes:
            break
        candidate = boxes[idx : idx + 1]
        if keep:
            overlaps = box_iou_xyxy(candidate, boxes[torch.as_tensor(keep, dtype=torch.long)]).flatten()
            if bool((overlaps > float(iou_thresh)).any().item()):
                continue
        keep.append(idx)
    if not keep:
        return boxes[:0]
    return boxes[torch.as_tensor(keep, dtype=torch.long)]


def greedy_match_boxes(
    pred_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_thresh: float,
) -> list[tuple[int, int, float]]:
    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        return []
    ious = box_iou_xyxy(pred_boxes.cpu(), gt_boxes.cpu())
    pairs: list[tuple[float, int, int]] = []
    for pred_idx in range(int(ious.shape[0])):
        for gt_idx in range(int(ious.shape[1])):
            iou = float(ious[pred_idx, gt_idx].item())
            if iou >= float(iou_thresh):
                pairs.append((iou, pred_idx, gt_idx))
    pairs.sort(reverse=True)
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for iou, pred_idx, gt_idx in pairs:
        if pred_idx in used_pred or gt_idx in used_gt:
            continue
        used_pred.add(pred_idx)
        used_gt.add(gt_idx)
        matches.append((pred_idx, gt_idx, iou))
    matches.sort(key=lambda item: item[0])
    return matches


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    while start >= 0:
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]
        start = text.find("{", start + 1)
    return None


def _coerce_bbox_numbers(value: object) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        return [float(value[idx]) for idx in range(4)]
    except (TypeError, ValueError):
        return None


def _normalize_generated_box(box: list[float], width: float, height: float) -> list[float] | None:
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) > 1.5:
        x1 /= max(float(width), 1.0)
        x2 /= max(float(width), 1.0)
        y1 /= max(float(height), 1.0)
        y2 /= max(float(height), 1.0)
    x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
    y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
    if (x2 - x1) * (y2 - y1) <= 1e-6:
        return None
    return [x1, y1, x2, y2]


def parse_qwen_bbox_response(
    text: str,
    width: float,
    height: float,
    task_id: int,
    max_instances: int,
) -> torch.Tensor:
    raw_boxes: list[list[float]] = []
    json_text = _extract_first_json_object(text)
    payload: object | None = None
    if json_text is not None:
        try:
            payload = json.loads(json_text)
        except json.JSONDecodeError:
            payload = None
    if isinstance(payload, dict):
        if task_id == 1:
            person = payload.get("person")
            if isinstance(person, dict):
                numbers = _coerce_bbox_numbers(person.get("bbox_2d"))
                if numbers is not None:
                    raw_boxes.append(numbers)
            elif isinstance(person, list):
                for item in person:
                    if isinstance(item, dict):
                        numbers = _coerce_bbox_numbers(item.get("bbox_2d"))
                        if numbers is not None:
                            raw_boxes.append(numbers)
        people = payload.get("people")
        if task_id != 1 or not raw_boxes:
            if isinstance(people, dict):
                people = [people]
            if isinstance(people, list):
                for item in people:
                    if not isinstance(item, dict):
                        continue
                    numbers = _coerce_bbox_numbers(item.get("bbox_2d"))
                    if numbers is not None:
                        raw_boxes.append(numbers)
    if not raw_boxes:
        for match in re.finditer(r"bbox_2d\s*[:=]\s*\[([^\]]+)\]", text):
            values = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", match.group(1))
            if len(values) >= 4:
                raw_boxes.append([float(value) for value in values[:4]])
    boxes: list[list[float]] = []
    for raw_box in raw_boxes[: max(int(max_instances), 0)]:
        normalized = _normalize_generated_box(raw_box, width, height)
        if normalized is not None:
            boxes.append(normalized)
    if not boxes:
        return torch.zeros(0, 4, dtype=torch.float32)
    return torch.tensor(boxes, dtype=torch.float32).clamp_(0.0, 1.0)


def build_qwen_generation_inputs(
    processor,
    image_paths: list[str],
    prompts: list[str],
    device: torch.device,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> dict[str, torch.Tensor]:
    images = _load_rgb_images(image_paths)
    messages_batch = [_build_user_messages(image_path, prompt) for image_path, prompt in zip(image_paths, prompts)]
    texts = _apply_chat_template_batch(processor, messages_batch, add_generation_prompt=True)
    inputs = processor(
        text=texts,
        images=images,
        padding=True,
        return_tensors="pt",
        **_processor_image_kwargs(min_pixels=min_pixels, max_pixels=max_pixels),
    )
    return _move_processor_tensors_to_device(inputs, device)


def generate_qwen_bbox_responses(
    training_model: torch.nn.Module,
    processor,
    batch: dict,
    device: torch.device,
    *,
    max_new_tokens: int,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> list[str]:
    module = unwrap_training_model(training_model)
    if module.backbone_name != "qwen3vl" or module.backbone_model is None:
        raise ValueError("--box_source=qwen_generate currently requires --backbone qwen3vl.")
    qwen_model = module.backbone_model
    inputs = build_qwen_generation_inputs(
        processor,
        batch["image_paths"],
        batch["prompts"],
        device,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    tokenizer = getattr(processor, "tokenizer", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    generation_kwargs = {
        "max_new_tokens": max(1, int(max_new_tokens)),
        "do_sample": False,
        "use_cache": True,
    }
    if pad_token_id is not None:
        generation_kwargs["pad_token_id"] = int(pad_token_id)
    if eos_token_id is not None:
        generation_kwargs["eos_token_id"] = int(eos_token_id)
    was_training = bool(qwen_model.training)
    qwen_model.eval()
    try:
        with torch.inference_mode():
            generated = qwen_model.generate(
                **qwen_forward_kwargs(inputs),
                **generation_kwargs,
            )
    finally:
        if was_training:
            qwen_model.train()
    input_length = int(inputs["input_ids"].shape[1])
    new_tokens = generated[:, input_length:]
    texts = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return [str(text).strip() for text in texts]


def prepare_qwen_generated_box_conditioning(
    training_model: torch.nn.Module,
    processor,
    batch: dict,
    device: torch.device,
    max_instances: int,
    *,
    max_new_tokens: int,
    match_iou_thresh: float,
    nms_iou_thresh: float,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    keep_unmatched_predictions: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, torch.Tensor]]]:
    if processor is None:
        raise ValueError("Qwen processor is required for --box_source=qwen_generate.")
    responses = generate_qwen_bbox_responses(
        training_model,
        processor,
        batch,
        device,
        max_new_tokens=max_new_tokens,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    selected_targets: list[dict[str, torch.Tensor]] = []
    selected_condition_boxes: list[torch.Tensor] = []
    for sample_idx, target in enumerate(batch["targets"]):
        task_id = int(batch["task_ids"][sample_idx].detach().cpu().item())
        width = float(target["width"])
        height = float(target["height"])
        gt_boxes_all = target["boxes"]
        num_gt = int(gt_boxes_all.shape[0])
        if task_id == 1:
            ref_target = int(target["ref_target"].detach().cpu().item())
            gt_indices = [ref_target] if 0 <= ref_target < num_gt else []
        else:
            gt_indices = list(range(min(num_gt, int(max_instances))))
        gt_index_tensor = torch.as_tensor(gt_indices, dtype=torch.long)
        gt_boxes = gt_boxes_all[gt_index_tensor].clone() if gt_indices else gt_boxes_all[:0].clone()

        pred_boxes = parse_qwen_bbox_response(
            responses[sample_idx] if sample_idx < len(responses) else "",
            width,
            height,
            task_id,
            max_instances=max_instances,
        )
        pred_boxes = nms_boxes_xyxy(pred_boxes, iou_thresh=nms_iou_thresh, max_boxes=max_instances)
        matches = greedy_match_boxes(pred_boxes, gt_boxes, iou_thresh=match_iou_thresh)
        pred_match_indices = [pred_idx for pred_idx, _, _ in matches]
        gt_match_indices = [gt_idx for _, gt_idx, _ in matches]
        pred_index_tensor = torch.as_tensor(pred_match_indices, dtype=torch.long)
        matched_gt_index_tensor = torch.as_tensor(gt_match_indices, dtype=torch.long)

        selected = dict(target)
        if matches:
            matched_pred_boxes = pred_boxes[pred_index_tensor].clone()
            matched_gt_boxes = gt_boxes[matched_gt_index_tensor].clone()
            matched_keypoints = target["keypoints"][gt_index_tensor[matched_gt_index_tensor]].clone()
            matched_keypoint_valid = target["keypoint_valid"][gt_index_tensor[matched_gt_index_tensor]].clone()
            if keep_unmatched_predictions:
                matched_pred_set = set(pred_match_indices)
                unmatched_pred_indices = [idx for idx in range(int(pred_boxes.shape[0])) if idx not in matched_pred_set]
                if unmatched_pred_indices:
                    unmatched_index_tensor = torch.as_tensor(unmatched_pred_indices, dtype=torch.long)
                    condition_boxes = torch.cat([matched_pred_boxes, pred_boxes[unmatched_index_tensor].clone()], dim=0)
                else:
                    condition_boxes = matched_pred_boxes
            else:
                condition_boxes = matched_pred_boxes
            selected_condition_boxes.append(condition_boxes[:max_instances].clone())
            selected["boxes"] = matched_gt_boxes
            selected["keypoints"] = matched_keypoints
            selected["keypoint_valid"] = matched_keypoint_valid
            selected["ref_target"] = torch.tensor(0 if task_id == 1 else -1, dtype=torch.long)
        else:
            if keep_unmatched_predictions and pred_boxes.numel() > 0:
                selected_condition_boxes.append(pred_boxes[:max_instances].clone())
            else:
                selected_condition_boxes.append(gt_boxes_all[:0].clone())
            selected["boxes"] = gt_boxes_all[:0].clone()
            selected["keypoints"] = target["keypoints"][:0].clone()
            selected["keypoint_valid"] = target["keypoint_valid"][:0].clone()
            selected["ref_target"] = torch.tensor(-1, dtype=torch.long)
        selected_targets.append(selected)

    max_boxes = max([int(boxes.shape[0]) for boxes in selected_condition_boxes] + [1])
    box_tensor = torch.zeros(len(selected_condition_boxes), max_boxes, 4, dtype=torch.float32, device=device)
    box_mask = torch.zeros(len(selected_condition_boxes), max_boxes, dtype=torch.bool, device=device)
    for sample_idx, boxes in enumerate(selected_condition_boxes):
        n = int(boxes.shape[0])
        if n == 0:
            continue
        box_tensor[sample_idx, :n] = boxes.to(device=device, dtype=torch.float32)
        box_mask[sample_idx, :n] = True
    return box_tensor, box_mask, selected_targets


class QwenPoseTrainingModel(torch.nn.Module):
    """Single trainable module for PoseHead plus backbone LoRA (Qwen3-VL or Eagle)."""

    def __init__(
        self,
        pose_model: QwenPoseModel,
        qwen_model: torch.nn.Module | None = None,
        qwen_feature_size: int = 32,
        qwen_feature_refiner_layers: int = 0,
        qwen_feature_refiner_bottleneck_dim: int = 256,
        qwen_feature_refiner_init_scale: float = 0.1,
        freeze_qwen: bool = False,
        backbone_model: torch.nn.Module | None = None,
        backbone_extractor: torch.nn.Module | None = None,
        backbone_name: str = "qwen3vl",
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.pose_model = pose_model
        self.backbone_name = backbone_name

        # Support both old (qwen_model) and new (backbone_model) interface
        if backbone_model is not None:
            self.backbone_model = backbone_model
            self.backbone_extractor = backbone_extractor
            self.freeze_backbone = bool(freeze_backbone)
        else:
            # Legacy qwen-only interface
            self.backbone_model = qwen_model
            self.freeze_backbone = bool(freeze_qwen)
            if qwen_model is not None:
                self.backbone_extractor = QwenFeatureExtractor(
                    qwen_model,
                    output_size=qwen_feature_size,
                    refiner_layers=qwen_feature_refiner_layers,
                    refiner_bottleneck_dim=qwen_feature_refiner_bottleneck_dim,
                    refiner_init_scale=qwen_feature_refiner_init_scale,
                )
            else:
                self.backbone_extractor = None

        if self.freeze_backbone and self.backbone_model is not None:
            for param in self.backbone_model.parameters():
                param.requires_grad = False

        # Aliases for backward compatibility with checkpoint save/load
        self.qwen_model = self.backbone_model
        self.qwen_extractor = self.backbone_extractor
        self.freeze_qwen = self.freeze_backbone

    def forward(
        self,
        schema_ids: torch.Tensor,
        task_ids: torch.Tensor,
        qwen_inputs: dict[str, torch.Tensor] | None = None,
        qwen_lm_inputs: dict[str, torch.Tensor] | None = None,
        target_boxes: torch.Tensor | None = None,
        target_box_mask: torch.Tensor | None = None,
        images: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        extra = {}
        lm_loss = None
        if self.backbone_extractor is not None:
            if self.backbone_name == "qwen3vl" and qwen_lm_inputs is not None and not self.freeze_backbone and qwen_inputs is None:
                hidden = self.backbone_extractor.run_backbone_hidden(
                    qwen_lm_inputs,
                    freeze_qwen=False,
                )
                external_feature_map, external_text_embed = self.backbone_extractor.project_hidden_state(
                    qwen_lm_inputs,
                    hidden,
                    text_keep_mask=qwen_lm_inputs.get("pose_text_mask"),
                )
                shift_labels = qwen_lm_inputs["labels"][:, 1:].contiguous()
                valid_label_mask = shift_labels.ne(-100)
                if valid_label_mask.any():
                    shift_hidden = hidden[:, :-1, :]
                    target_hidden = shift_hidden[valid_label_mask]
                    target_labels = shift_labels[valid_label_mask]
                    base_model = get_qwen_base_model(self.backbone_model)
                    target_logits = base_model.lm_head(target_hidden).float()
                    lm_loss = F.cross_entropy(target_logits, target_labels, reduction="mean")
                else:
                    lm_loss = hidden.new_zeros(())
            else:
                if qwen_inputs is None:
                    raise ValueError("backbone inputs must be provided when backbone is enabled.")
                # Both QwenFeatureExtractor and EagleFeatureExtractor share the same
                # forward signature: (inputs, freeze_backbone=bool)
                if self.backbone_name == "eagle":
                    external_feature_map, external_text_embed = self.backbone_extractor(
                        qwen_inputs,
                        freeze_eagle=self.freeze_backbone,
                    )
                else:
                    external_feature_map, external_text_embed = self.backbone_extractor(
                        qwen_inputs,
                        freeze_qwen=self.freeze_backbone,
                    )
                if qwen_lm_inputs is not None and not self.freeze_backbone:
                    if self.backbone_name == "eagle":
                        allowed = {"pixel_values", "image_grid_hws", "image_flags", "input_ids", "attention_mask"}
                        lm_forward_inputs = {key: value for key, value in qwen_lm_inputs.items() if key in allowed}
                    else:
                        lm_forward_inputs = qwen_forward_kwargs(qwen_lm_inputs)
                    lm_forward_inputs["labels"] = qwen_lm_inputs["labels"]
                    lm_outputs = self.backbone_model(**lm_forward_inputs, use_cache=False)
                    lm_loss = lm_outputs.loss
            extra = {
                "external_feature_map": external_feature_map,
                "external_text_embed": external_text_embed,
            }
        outputs = self.pose_model(
            schema_ids=schema_ids,
            task_ids=task_ids,
            images=images,
            target_boxes=target_boxes,
            target_box_mask=target_box_mask,
            **extra,
        )
        if lm_loss is not None:
            outputs["lm_loss"] = lm_loss
        return outputs


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def distributed_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def distributed_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def distributed_barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def update_progress_bar(progress_bar, postfix: dict[str, object]) -> None:
    if progress_bar is not None:
        progress_bar.set_postfix(postfix, refresh=False)


class HomogeneousDatasetBatchSampler:
    """Yield one-dataset-only batches for InterleavedPoseDataset.

    The underlying dataset still owns the deterministic per-source offset/stride
    mapping. This sampler only groups global indices so a micro batch never mixes
    COCO/AIC/MPII/CrowdPose/RefHuman schemas. Tail batches are filled by wrapping
    within the same dataset, which keeps batch size stable without cross-dataset
    mixing.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        *,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
        shuffle: bool = True,
        fill_last: bool = True,
    ) -> None:
        required = ("datasets", "names", "global_index_for_dataset_linear")
        if not all(hasattr(dataset, attr) for attr in required):
            raise TypeError("HomogeneousDatasetBatchSampler requires InterleavedPoseDataset-like input.")
        self.dataset = dataset
        self.batch_size = max(int(batch_size), 1)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = max(int(world_size), 1)
        self.shuffle = bool(shuffle)
        self.fill_last = bool(fill_last)
        self.epoch = 0
        self._cached_batches: list[list[int]] | None = None

    @staticmethod
    def _weighted_schedule(counts: list[int]) -> list[int]:
        total = sum(max(int(count), 0) for count in counts)
        if total <= 0:
            return []
        used = [0 for _ in counts]
        schedule: list[int] = []
        for step in range(total):
            candidates = [idx for idx, count in enumerate(counts) if used[idx] < count]
            best = max(
                candidates,
                key=lambda idx: (
                    ((step + 1) * counts[idx] / total) - used[idx],
                    counts[idx],
                    -idx,
                ),
            )
            schedule.append(best)
            used[best] += 1
        return schedule

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._cached_batches = None

    def _build_batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch * 1009)
        per_dataset_batches: list[list[list[int]]] = []
        batch_counts: list[int] = []
        inner_datasets = list(getattr(self.dataset, "datasets"))
        for dataset_idx, inner_dataset in enumerate(inner_datasets):
            n = len(inner_dataset)
            if n <= 0:
                per_dataset_batches.append([])
                batch_counts.append(0)
                continue
            local_linear = list(range(n))
            if self.shuffle:
                random.Random(self.seed + self.epoch * 1009 + dataset_idx * 9176).shuffle(local_linear)
            num_batches = math.ceil(n / self.batch_size)
            target_count = num_batches * self.batch_size if self.fill_last else n
            if self.fill_last and len(local_linear) < target_count:
                pad_source = local_linear or [0]
                extra = [pad_source[i % len(pad_source)] for i in range(target_count - len(local_linear))]
                local_linear = local_linear + extra
            batches = []
            for start in range(0, len(local_linear), self.batch_size):
                chunk = local_linear[start : start + self.batch_size]
                if not chunk:
                    continue
                if len(chunk) < self.batch_size and self.fill_last:
                    chunk = chunk + [chunk[i % len(chunk)] for i in range(self.batch_size - len(chunk))]
                batches.append([
                    self.dataset.global_index_for_dataset_linear(dataset_idx, value)
                    for value in chunk
                ])
            per_dataset_batches.append(batches)
            batch_counts.append(len(batches))
        schedule = self._weighted_schedule(batch_counts)
        cursors = [0 for _ in batch_counts]
        all_batches: list[list[int]] = []
        for dataset_idx in schedule:
            all_batches.append(per_dataset_batches[dataset_idx][cursors[dataset_idx]])
            cursors[dataset_idx] += 1
        if self.shuffle and all_batches:
            # Keep source proportions from the schedule, but jitter local neighbors
            # slightly so every epoch is not the same deterministic source order.
            window = max(1, min(8, len(all_batches)))
            jittered: list[list[int]] = []
            for start in range(0, len(all_batches), window):
                block = all_batches[start : start + window]
                rng.shuffle(block)
                jittered.extend(block)
            all_batches = jittered
        if self.world_size > 1 and all_batches:
            remainder = len(all_batches) % self.world_size
            if remainder:
                needed = self.world_size - remainder
                pad_batches = [all_batches[i % len(all_batches)] for i in range(needed)]
                all_batches.extend(pad_batches)
            all_batches = all_batches[self.rank :: self.world_size]
        return all_batches

    def __iter__(self):
        self._cached_batches = self._build_batches()
        yield from self._cached_batches

    def __len__(self) -> int:
        if self._cached_batches is None:
            self._cached_batches = self._build_batches()
        return len(self._cached_batches)


def build_progress_loss_postfix(
    loss_metrics: dict[str, float],
    weights: LossWeights,
) -> dict[str, str]:
    """Build tqdm/log postfix for the active clean pose/LM objectives."""

    def _enabled(weight: float, key: str, flag: bool = True) -> bool:
        return bool(flag) and float(weight) > 0.0 and key in loss_metrics

    def _add(postfix: dict[str, str], label: str, key: str, weight: float, flag: bool = True) -> None:
        if _enabled(weight, key, flag):
            postfix[label] = f"{float(loss_metrics.get(key, 0.0)):.3f}"

    postfix = {"loss": f"{float(loss_metrics.get('loss_total', 0.0)):.3f}"}
    _add(postfix, "oks", "loss_oks", weights.oks)
    _add(postfix, "coord", "loss_coord", weights.coord)
    _add(postfix, "hard", "loss_hard_joint", weights.hard_joint)
    _add(postfix, "vis", "loss_vis", weights.vis)
    _add(postfix, "lm", "loss_lm", weights.lm)
    return postfix


def progress_write(progress_bar, message: str) -> None:
    if progress_bar is not None:
        progress_bar.write(message)
    else:
        print(message)


def _safe_vis_tag(value: object, default: str = "unknown") -> str:
    text = str(value or default).strip().lower()
    text = re.sub(r"[^a-z0-9_.-]+", "_", text)
    return text.strip("_") or default


SCHEMA_POSE_EDGES: dict[str, list[tuple[str, str]]] = {
    "COCO17": [
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
        ("nose", "left_eye"),
        ("nose", "right_eye"),
        ("left_eye", "left_ear"),
        ("right_eye", "right_ear"),
    ],
    "AIC14": [
        ("head_top", "neck"),
        ("neck", "left_shoulder"),
        ("neck", "right_shoulder"),
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
    ],
    "MPII16": [
        ("head_top", "upper_neck"),
        ("upper_neck", "thorax"),
        ("thorax", "left_shoulder"),
        ("thorax", "right_shoulder"),
        ("thorax", "pelvis"),
        ("pelvis", "left_hip"),
        ("pelvis", "right_hip"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
    ],
    "CrowdPose14": [
        ("head_top", "neck"),
        ("neck", "left_shoulder"),
        ("neck", "right_shoulder"),
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
    ],
}
SCHEMA_POSE_EDGE_INDICES = {
    schema: [
        (UNION_KEYPOINTS.index(a), UNION_KEYPOINTS.index(b))
        for a, b in edges
        if a in UNION_KEYPOINTS and b in UNION_KEYPOINTS
    ]
    for schema, edges in SCHEMA_POSE_EDGES.items()
}
DEFAULT_POSE_EDGE_INDICES = sorted(
    {
        edge
        for schema_edges in SCHEMA_POSE_EDGE_INDICES.values()
        for edge in schema_edges
    }
)


def _draw_box(draw: ImageDraw.ImageDraw, box: torch.Tensor, width: int, height: int, color: tuple[int, int, int], label: str) -> None:
    x1, y1, x2, y2 = box.tolist()
    xy = [x1 * width, y1 * height, x2 * width, y2 * height]
    draw.rectangle(xy, outline=color, width=3)
    draw.text((xy[0] + 2, max(xy[1] - 14, 0)), label, fill=color)


def _draw_pose(
    draw: ImageDraw.ImageDraw,
    keypoints: torch.Tensor,
    valid: torch.Tensor,
    width: int,
    height: int,
    color: tuple[int, int, int],
    edge_indices: list[tuple[int, int]],
) -> None:
    points: dict[int, tuple[float, float]] = {}
    for idx in torch.nonzero(valid, as_tuple=False).flatten().tolist():
        x, y = keypoints[idx, :2].tolist()
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        px, py = x * width, y * height
        points[idx] = (px, py)
        r = 3
        draw.ellipse([px - r, py - r, px + r, py + r], fill=color)
    for a, b in edge_indices:
        if a in points and b in points:
            draw.line([points[a], points[b]], fill=color, width=2)


def save_pose_visualization(
    outputs: dict[str, torch.Tensor],
    batch: dict,
    output_path: Path,
    sample_idx: int = 0,
    max_instances: int = 8,
    score_threshold: float = 0.05,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_path = batch["image_paths"][sample_idx]
    with Image.open(image_path) as image:
        canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size

    boxes = outputs["boxes"][sample_idx].detach().float().cpu()
    pose_boxes = outputs.get("pose_boxes", outputs["boxes"])[sample_idx].detach().float().cpu()
    box_mask = outputs.get("box_mask")
    valid_boxes = (
        box_mask[sample_idx].detach().cpu().bool()
        if box_mask is not None
        else torch.ones(boxes.shape[0], dtype=torch.bool)
    )
    scores = outputs["person_logits"][sample_idx].detach().sigmoid().float().cpu()
    keypoints = outputs["keypoints"][sample_idx].detach().float().cpu()
    schema_valid = outputs["keypoint_valid_mask"][sample_idx].detach().cpu().bool()

    target = batch["targets"][sample_idx]
    source_datasets = batch.get("source_datasets", [])
    dataset_name = "unknown"
    if sample_idx < len(source_datasets) and source_datasets[sample_idx]:
        dataset_name = str(source_datasets[sample_idx])
    elif target.get("dataset"):
        dataset_name = str(target.get("dataset"))
    schema_name = str(target.get("schema", ""))
    if not schema_name:
        schema_ids = batch.get("schema_ids")
        if schema_ids is not None:
            schema_name = ID_TO_SCHEMA.get(int(schema_ids[sample_idx].detach().cpu().item()), "unknown")
        else:
            schema_name = "unknown"
    task_name = str(target.get("task", "")) or "unknown"
    edge_indices = SCHEMA_POSE_EDGE_INDICES.get(schema_name, DEFAULT_POSE_EDGE_INDICES)
    schema_keypoint_count = int(schema_valid.sum().item())
    gt_boxes = target["boxes"].detach().float().cpu()
    gt_keypoints = target["keypoints"].detach().float().cpu()
    gt_valid = target["keypoint_valid"].detach().cpu().bool()
    gt_visible = gt_valid & (gt_keypoints[..., 2] > 0.5)

    valid_indices = torch.nonzero(valid_boxes, as_tuple=False).flatten()
    if valid_indices.numel() > 0:
        ranked = valid_indices[torch.argsort(scores[valid_indices], descending=True)]
    else:
        ranked = torch.empty(0, dtype=torch.long)
    selected = ranked[: max(int(max_instances), 0)].tolist()

    for idx in selected:
        if scores[idx] < score_threshold and len(selected) > 1:
            continue
        if idx < gt_boxes.shape[0]:
            _draw_box(draw, gt_boxes[idx], width, height, (50, 200, 80), f"gt {idx}")
            _draw_pose(draw, gt_keypoints[idx], gt_visible[idx], width, height, (50, 200, 80), edge_indices)
        _draw_box(draw, boxes[idx], width, height, (255, 220, 0), f"box {idx}")
        _draw_box(draw, pose_boxes[idx], width, height, (255, 140, 0), f"pose box {idx}")
        pred_valid = schema_valid & (keypoints[idx, :, 2] >= score_threshold)
        _draw_pose(draw, keypoints[idx], pred_valid, width, height, (240, 70, 70), edge_indices)

    draw.rectangle([0, 0, min(width, 940), 44], fill=(0, 0, 0))
    draw.text(
        (4, 4),
        f"dataset={dataset_name} schema={schema_name} keypoints={schema_keypoint_count} task={task_name}",
        fill=(255, 255, 255),
    )
    draw.text(
        (4, 24),
        "green=GT, red=pred keypoints, yellow=input box, orange=expanded pose box",
        fill=(255, 255, 255),
    )
    canvas.save(output_path)


def sync_cuda_for_timing(enabled: bool, device: torch.device) -> None:
    if enabled and device.type == "cuda":
        torch.cuda.synchronize(device)


def unwrap_training_model(model: torch.nn.Module) -> QwenPoseTrainingModel:
    return getattr(model, "module", model)


def trainable_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def capture_rng_state() -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "torch": torch.random.get_rng_state().cpu(),
    }
    if torch.cuda.is_available():
        try:
            state["cuda"] = [rng.cpu() for rng in torch.cuda.get_rng_state_all()]
        except Exception:
            pass
    return state


def restore_rng_state(state: dict[str, object] | None) -> None:
    if not state:
        return
    python_state = state.get("python")
    if python_state is not None:
        random.setstate(python_state)
    torch_state = state.get("torch")
    if torch_state is not None:
        torch.random.set_rng_state(torch_state)
    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def normalize_resume_position(epoch: int, batch_in_epoch: int, batches_per_epoch: int) -> tuple[int, int]:
    epoch = max(int(epoch), 0)
    batch_in_epoch = max(int(batch_in_epoch), 0)
    batches_per_epoch = max(int(batches_per_epoch), 1)
    epoch += batch_in_epoch // batches_per_epoch
    batch_in_epoch = batch_in_epoch % batches_per_epoch
    return epoch, batch_in_epoch


def build_training_state(
    *,
    epoch: int,
    batch_in_epoch: int,
    batches_per_epoch: int,
    global_step: int,
    micro_step: int,
    grad_accum_steps: int,
    batch_size: int,
    world_size: int,
    dataset_names: list[str],
    mixing_strategy: str,
    split: str,
    loss_ema: float | None,
    loss_spike_count: int,
) -> dict[str, object]:
    resume_epoch, resume_batch_in_epoch = normalize_resume_position(epoch, batch_in_epoch, batches_per_epoch)
    return {
        "version": 1,
        "global_step": int(global_step),
        "micro_step": int(micro_step),
        "epoch": int(resume_epoch),
        "batch_in_epoch": int(resume_batch_in_epoch),
        "batches_per_epoch": int(max(batches_per_epoch, 1)),
        "grad_accum_steps": int(max(grad_accum_steps, 1)),
        "batch_size": int(max(batch_size, 1)),
        "world_size": int(max(world_size, 1)),
        "dataset_names": [str(name) for name in dataset_names],
        "mixing_strategy": str(mixing_strategy),
        "split": str(split),
        "loss_ema": None if loss_ema is None else float(loss_ema),
        "loss_spike_count": int(max(loss_spike_count, 0)),
    }


def infer_resume_state(
    training_state: dict[str, object] | None,
    *,
    global_step: int,
    batches_per_epoch: int,
    grad_accum_steps: int,
    batch_size: int,
    world_size: int,
    dataset_names: list[str],
    mixing_strategy: str,
    split: str,
) -> tuple[dict[str, object], bool]:
    if training_state:
        resume_epoch = int(training_state.get("epoch", 0))
        resume_batch_in_epoch = int(training_state.get("batch_in_epoch", 0))
        resume_micro_step = int(training_state.get("micro_step", global_step * max(int(grad_accum_steps), 1)))
        resume_loss_ema = training_state.get("loss_ema")
        if resume_loss_ema is not None:
            resume_loss_ema = float(resume_loss_ema)
        resume_loss_spike_count = int(training_state.get("loss_spike_count", 0))
        resume_epoch, resume_batch_in_epoch = normalize_resume_position(
            resume_epoch,
            resume_batch_in_epoch,
            batches_per_epoch,
        )
        merged_state = dict(training_state)
        merged_state.update(
            build_training_state(
                epoch=resume_epoch,
                batch_in_epoch=resume_batch_in_epoch,
                batches_per_epoch=batches_per_epoch,
                global_step=global_step,
                micro_step=resume_micro_step,
                grad_accum_steps=int(training_state.get("grad_accum_steps", grad_accum_steps)),
                batch_size=int(training_state.get("batch_size", batch_size)),
                world_size=int(training_state.get("world_size", world_size)),
                dataset_names=list(training_state.get("dataset_names", dataset_names)),
                mixing_strategy=str(training_state.get("mixing_strategy", mixing_strategy)),
                split=str(training_state.get("split", split)),
                loss_ema=resume_loss_ema,
                loss_spike_count=resume_loss_spike_count,
            )
        )
        return merged_state, False

    legacy_micro_step = int(global_step) * max(int(grad_accum_steps), 1)
    legacy_epoch, legacy_batch_in_epoch = normalize_resume_position(
        0,
        legacy_micro_step,
        batches_per_epoch,
    )
    legacy_state = build_training_state(
        epoch=legacy_epoch,
        batch_in_epoch=legacy_batch_in_epoch,
        batches_per_epoch=batches_per_epoch,
        global_step=global_step,
        micro_step=legacy_micro_step,
        grad_accum_steps=grad_accum_steps,
        batch_size=batch_size,
        world_size=world_size,
        dataset_names=dataset_names,
        mixing_strategy=mixing_strategy,
        split=split,
        loss_ema=None,
        loss_spike_count=0,
    )
    legacy_state["legacy_inferred"] = True
    return legacy_state, True


def validate_resume_state(
    resume_state: dict[str, object],
    *,
    current_batches_per_epoch: int,
    batch_size: int,
    world_size: int,
    dataset_names: list[str],
    mixing_strategy: str,
    split: str,
) -> None:
    resume_batch_in_epoch = int(resume_state.get("batch_in_epoch", 0))
    if resume_batch_in_epoch <= 0:
        return

    mismatches: list[str] = []
    saved_batches_per_epoch = int(resume_state.get("batches_per_epoch", current_batches_per_epoch))
    if saved_batches_per_epoch != int(current_batches_per_epoch):
        mismatches.append(f"batches_per_epoch saved={saved_batches_per_epoch} current={current_batches_per_epoch}")
    saved_batch_size = int(resume_state.get("batch_size", batch_size))
    if saved_batch_size != int(batch_size):
        mismatches.append(f"batch_size saved={saved_batch_size} current={batch_size}")
    saved_world_size = int(resume_state.get("world_size", world_size))
    if saved_world_size != int(world_size):
        mismatches.append(f"world_size saved={saved_world_size} current={world_size}")
    saved_mixing_strategy = str(resume_state.get("mixing_strategy", mixing_strategy))
    if saved_mixing_strategy != str(mixing_strategy):
        mismatches.append(
            f"mixing_strategy saved={saved_mixing_strategy} current={mixing_strategy}"
        )
    saved_split = str(resume_state.get("split", split))
    if saved_split != str(split):
        mismatches.append(f"split saved={saved_split} current={split}")
    saved_dataset_names = [str(name) for name in resume_state.get("dataset_names", dataset_names)]
    if saved_dataset_names != [str(name) for name in dataset_names]:
        mismatches.append(
            f"datasets saved={','.join(saved_dataset_names)} current={','.join(dataset_names)}"
        )
    if mismatches:
        raise ValueError(
            "Cannot safely resume from the middle of an epoch because the data pipeline changed: "
            + "; ".join(mismatches)
        )


def next_resume_position_after_batch(epoch: int, batch_index: int, batches_per_epoch: int) -> tuple[int, int]:
    return normalize_resume_position(epoch, batch_index + 1, batches_per_epoch)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    output_dir: Path,
    save_total_limit: int = 1,
    include_optimizer: bool = True,
    qwen_processor: object | None = None,
    save_deepspeed: bool = False,
    scaler: torch.amp.GradScaler | None = None,
    training_state: dict[str, object] | None = None,
) -> None:
    checkpoint_dir = output_dir / f"checkpoint-{step}"
    rng_state = capture_rng_state()
    client_state = {"step": int(step)}
    if training_state is not None:
        client_state["training_state"] = training_state
    client_state["rng_state"] = rng_state
    if is_main_process():
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    distributed_barrier()
    if save_deepspeed and hasattr(model, "save_checkpoint"):
        model.save_checkpoint(
            str(checkpoint_dir),
            tag=DEEPSPEED_TAG,
            client_state=client_state,
        )
    if not is_main_process():
        distributed_barrier()
        return
    module = unwrap_training_model(model)
    payload = {
        "step": step,
        "model": module.pose_model.state_dict(),
        "pose_config": asdict(module.pose_model.config),
        "backbone_name": str(getattr(module, "backbone_name", "qwen3vl")),
        "deepspeed_managed": not include_optimizer,
        "checkpoint_format": "qwenpose-v3",
        "rng_state": rng_state,
    }
    if optimizer is not None and include_optimizer:
        payload["optimizer"] = optimizer.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if training_state is not None:
        payload["training_state"] = training_state
    if module.qwen_model is not None:
        backbone_state = trainable_state_dict(module.qwen_model)
        payload["backbone_trainable"] = backbone_state
        payload["qwen_trainable"] = backbone_state
    if module.qwen_extractor is not None:
        feature_config = {
            "output_size": int(module.qwen_extractor.output_size),
        }
        payload["backbone_feature_config"] = feature_config
        payload["qwen_feature_config"] = feature_config
    if module.qwen_extractor is not None and module.qwen_extractor.feature_refiner.num_layers > 0:
        refiner = module.qwen_extractor.feature_refiner
        refiner_state = refiner.state_dict()
        refiner_config = {
            "layers": int(refiner.num_layers),
            "bottleneck_dim": int(refiner.bottleneck_dim),
            "init_scale": float(refiner.init_scale),
        }
        payload["backbone_feature_refiner"] = refiner_state
        payload["backbone_feature_refiner_config"] = refiner_config
        payload["qwen_feature_refiner"] = refiner_state
        payload["qwen_feature_refiner_config"] = refiner_config
    torch.save(payload, checkpoint_dir / CHECKPOINT_PAYLOAD_NAME)
    with (checkpoint_dir / "qwenpose_state.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "step": int(step),
                "checkpoint": str(checkpoint_dir),
                "payload": CHECKPOINT_PAYLOAD_NAME,
                "deepspeed_tag": DEEPSPEED_TAG if save_deepspeed else None,
                "training_state": training_state,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
        f.write("\n")
    if module.qwen_model is not None and hasattr(module.qwen_model, "save_pretrained"):
        backbone_name = str(getattr(module, "backbone_name", "qwen3vl"))
        adapter_dir_name = "backbone_lora_adapter" if backbone_name == "eagle" else "qwen_lora_adapter"
        adapter_dir = checkpoint_dir / adapter_dir_name
        module.qwen_model.save_pretrained(adapter_dir)
        if qwen_processor is not None:
            qwen_processor.save_pretrained(adapter_dir)
    prune_checkpoints(output_dir, save_total_limit)
    distributed_barrier()


def checkpoint_step(path: Path) -> int | None:
    if path.is_dir():
        match = re.search(r"checkpoint-(\d+)$", path.name)
    else:
        match = re.search(r"checkpoint_step_(\d+)\.pt$", path.name)
    if not match:
        return None
    return int(match.group(1))


def _load_local_torch_payload(path: Path) -> dict[str, object]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Expected checkpoint payload dict at {path}, but got {type(payload).__name__}.")
    return payload


def resolve_qwenpose_checkpoint_payload(path: Path) -> Path | None:
    if path.is_file():
        return path if path.name == CHECKPOINT_PAYLOAD_NAME else None
    if not path.exists() or not path.is_dir():
        return None

    search_roots = [path]
    for child_name in (
        "stage2_qwen_box_closed_loop",
        "stage3_qwen_box_closed_loop",  # legacy removed three-stage layout
        "stage2_teacher_forcing",       # legacy removed middle stage
        "stage2_qwen_lora_lm",          # legacy public snapshot name
        "stage1_freeze_qwen",
    ):
        child = path / child_name
        if child.is_dir():
            search_roots.append(child)

    best_payload: tuple[int, Path] | None = None
    for root in search_roots:
        direct_payload = root / CHECKPOINT_PAYLOAD_NAME
        if direct_payload.is_file():
            return direct_payload
        for candidate in list(root.glob("checkpoint-*")) + list(root.glob("checkpoint_step_*.pt")):
            step = checkpoint_step(candidate)
            if step is None:
                continue
            payload_path = candidate / CHECKPOINT_PAYLOAD_NAME if candidate.is_dir() else candidate
            if not payload_path.is_file():
                continue
            if best_payload is None or step >= best_payload[0]:
                best_payload = (step, payload_path)
    return None if best_payload is None else best_payload[1]


def resolve_qwen_lora_adapter_dir(requested_path: Path, checkpoint_payload_path: Path | None) -> Path | None:
    candidates: list[Path] = []
    if requested_path.is_dir():
        if (requested_path / "adapter_config.json").is_file():
            candidates.append(requested_path)
        candidates.append(requested_path / "qwen_lora_adapter")
        for stage_name in (
            "stage2_qwen_box_closed_loop",
            "stage3_qwen_box_closed_loop",  # legacy removed three-stage layout
            "stage2_teacher_forcing",       # legacy removed middle stage
            "stage2_qwen_lora_lm",          # legacy public snapshot name
        ):
            stage_dir = requested_path / stage_name
            if stage_dir.is_dir():
                candidates.append(stage_dir / "qwen_lora_adapter")
    if checkpoint_payload_path is not None:
        checkpoint_dir = checkpoint_payload_path.parent
        candidates.append(checkpoint_dir / "qwen_lora_adapter")
        candidates.append(checkpoint_dir.parent / "qwen_lora_adapter")

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except FileNotFoundError:
            resolved = candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.is_dir() and (candidate / "adapter_config.json").is_file():
            return candidate
    return None


def resolve_qwen_adapter_base_model_path(adapter_dir: Path) -> str:
    config_path = adapter_dir / "adapter_config.json"
    config = json.load(open(config_path, encoding="utf-8"))
    base_model_name_or_path = str(config.get("base_model_name_or_path", "")).strip()
    if not base_model_name_or_path:
        raise ValueError(f"adapter_config.json under {adapter_dir} has no base_model_name_or_path.")
    candidate = Path(base_model_name_or_path).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    relative_candidate = (adapter_dir / base_model_name_or_path).resolve()
    if relative_candidate.exists():
        return str(relative_candidate)
    return base_model_name_or_path


def detect_qwen_initialization_source(model_path: str) -> QwenInitializationSource:
    requested_path_str = str(model_path)
    requested_path = Path(requested_path_str).expanduser()
    if not requested_path.exists():
        return QwenInitializationSource(
            requested_path=requested_path_str,
            backbone_model_path=requested_path_str,
        )

    pose_checkpoint_path = resolve_qwenpose_checkpoint_payload(requested_path)
    adapter_path = resolve_qwen_lora_adapter_dir(requested_path, pose_checkpoint_path)
    has_hf_config = requested_path.is_dir() and (requested_path / "config.json").is_file()

    if has_hf_config:
        is_merged_backbone = False
        source_kind = "base_model"
        if pose_checkpoint_path is not None:
            payload = _load_local_torch_payload(pose_checkpoint_path)
            is_merged_backbone = bool(payload.get("backbone_merged", False))
            source_kind = "qwenpose_merged" if is_merged_backbone else "qwenpose_model_dir"
        return QwenInitializationSource(
            requested_path=requested_path_str,
            backbone_model_path=str(requested_path.resolve()),
            source_kind=source_kind,
            pose_checkpoint_path=pose_checkpoint_path,
            adapter_path=adapter_path,
            is_merged_backbone=is_merged_backbone,
        )

    if adapter_path is not None:
        source_kind = "qwen_lora_adapter"
        if pose_checkpoint_path is not None:
            source_kind = "qwenpose_checkpoint"
        return QwenInitializationSource(
            requested_path=requested_path_str,
            backbone_model_path=resolve_qwen_adapter_base_model_path(adapter_path),
            source_kind=source_kind,
            pose_checkpoint_path=pose_checkpoint_path,
            adapter_path=adapter_path,
            is_merged_backbone=False,
        )

    return QwenInitializationSource(
        requested_path=requested_path_str,
        backbone_model_path=requested_path_str,
        pose_checkpoint_path=pose_checkpoint_path,
        source_kind="base_model",
        adapter_path=None,
        is_merged_backbone=False,
    )


def prune_checkpoints(output_dir: Path, save_total_limit: int) -> None:
    if save_total_limit <= 0:
        return
    checkpoints: list[tuple[int, Path]] = []
    for path in list(output_dir.glob("checkpoint-*")) + list(output_dir.glob("checkpoint_step_*.pt")):
        step = checkpoint_step(path)
        if step is not None:
            checkpoints.append((step, path))
    for _, path in sorted(checkpoints)[:-save_total_limit]:
        if path.is_dir():
            import shutil

            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)


def resolve_training_checkpoint(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint path does not exist: {path}")
    if (path / CHECKPOINT_PAYLOAD_NAME).is_file() or (path / DEEPSPEED_TAG).exists():
        return path
    checkpoints: list[tuple[int, Path]] = []
    for candidate in list(path.glob("checkpoint-*")) + list(path.glob("checkpoint_step_*.pt")):
        step = checkpoint_step(candidate)
        if step is not None:
            checkpoints.append((step, candidate))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint-* or checkpoint_step_*.pt found in {path}")
    return sorted(checkpoints)[-1][1]


def load_training_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    checkpoint_path: Path,
    load_optimizer: bool = True,
    scaler: torch.amp.GradScaler | None = None,
    load_scaler: bool = True,
) -> tuple[int, bool, bool, Path, dict[str, object]]:
    resolved_path = resolve_training_checkpoint(checkpoint_path)
    payload_path = resolved_path / CHECKPOINT_PAYLOAD_NAME if resolved_path.is_dir() else resolved_path
    payload = torch.load(payload_path, map_location="cpu")
    module = unwrap_training_model(model)
    module.pose_model.load_state_dict(payload["model"])
    backbone_state = payload.get("backbone_trainable", payload.get("qwen_trainable"))
    if module.qwen_model is not None and backbone_state is not None:
        module.qwen_model.load_state_dict(backbone_state, strict=False)
    if module.qwen_extractor is not None and ("backbone_feature_refiner" in payload or "qwen_feature_refiner" in payload):
        module.qwen_extractor.feature_refiner.load_state_dict(payload["backbone_feature_refiner"] if "backbone_feature_refiner" in payload else payload["qwen_feature_refiner"], strict=True)
    optimizer_loaded = False
    if load_optimizer and optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
        optimizer_loaded = True
    scaler_loaded = False
    if load_scaler and scaler is not None and "scaler" in payload:
        scaler.load_state_dict(payload["scaler"])
        scaler_loaded = True
    return int(payload.get("step", 0)), optimizer_loaded, scaler_loaded, resolved_path, payload


def load_deepspeed_config(path: Path, args: argparse.Namespace, world_size: int) -> dict:
    config = json.load(open(path, encoding="utf-8"))
    config["train_micro_batch_size_per_gpu"] = int(args.batch_size)
    config["gradient_accumulation_steps"] = int(args.grad_accum_steps)
    config["train_batch_size"] = int(args.batch_size) * int(args.grad_accum_steps) * max(int(world_size), 1)
    config["gradient_clipping"] = float(args.grad_clip)
    # Determine dtype from the active backbone
    backbone_name = getattr(args, "backbone", "qwen3vl")
    if backbone_name == "eagle":
        dtype_str = getattr(args, "eagle_dtype", "bfloat16")
    else:
        dtype_str = getattr(args, "qwen_dtype", "bfloat16")
    if "bf16" in config:
        config["bf16"]["enabled"] = dtype_str == "bfloat16"
    if "fp16" in config:
        config["fp16"]["enabled"] = dtype_str == "float16"
    config.setdefault("steps_per_print", 100000)
    return config


def is_vision_parameter(name: str) -> bool:
    """Check if a parameter belongs to the vision encoder (Qwen or Eagle)."""
    return ".visual." in name or ".vision_model." in name or "qwen_model.visual." in name or "backbone_model.vision_model." in name


def build_optimizer_param_groups(
    model: torch.nn.Module,
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], dict[str, tuple[int, int, float]]]:
    grouped: dict[str, list[torch.nn.Parameter]] = {
        "pose": [],
        "backbone_lora": [],
        "backbone_vision_lora": [],
    }
    stats: dict[str, list[float]] = {
        "pose": [0, 0],
        "backbone_lora": [0, 0],
        "backbone_vision_lora": [0, 0],
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("backbone_model.") or name.startswith("qwen_model."):
            group_name = "backbone_vision_lora" if is_vision_parameter(name) else "backbone_lora"
        else:
            group_name = "pose"
        grouped[group_name].append(param)
        stats[group_name][0] += 1
        stats[group_name][1] += param.numel()

    backbone_name = getattr(args, "backbone", "qwen3vl")
    if backbone_name == "eagle":
        lora_lr_scale = getattr(args, "locate_lr_scale", args.qwen_lora_lr_scale)
        vision_lr_scale = getattr(args, "locate_vision_scale", args.qwen_vision_lr_scale)
    else:
        lora_lr_scale = args.qwen_lora_lr_scale
        vision_lr_scale = args.qwen_vision_lr_scale

    lrs = {
        "pose": args.lr,
        "backbone_lora": args.lr * lora_lr_scale,
        "backbone_vision_lora": args.lr * vision_lr_scale,
    }
    param_groups = [
        {"params": params, "lr": lrs[name]}
        for name, params in grouped.items()
        if params
    ]
    printable = {
        name: (int(values[0]), int(values[1]), float(lrs[name]))
        for name, values in stats.items()
        if values[0] > 0
    }
    return param_groups, printable


class CosineLRScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_steps: int,
        min_lr_ratio: float,
    ) -> None:
        self.optimizer = optimizer
        self.total_steps = max(int(total_steps), 1)
        self.warmup_steps = max(min(int(warmup_steps), self.total_steps), 0)
        self.min_lr_ratio = min(max(float(min_lr_ratio), 0.0), 1.0)
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.step_num = 0
        self._apply(self._scale(0))

    def _scale(self, step: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return max(float(step + 1) / float(self.warmup_steps), 1e-8)
        decay_steps = max(self.total_steps - self.warmup_steps, 1)
        progress = min(max(float(step - self.warmup_steps) / float(decay_steps), 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    def _apply(self, scale: float) -> None:
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * scale

    def step(self) -> None:
        self.step_num += 1
        self._apply(self._scale(self.step_num))

    def set_step(self, step: int) -> None:
        self.step_num = max(int(step), 0)
        self._apply(self._scale(self.step_num))


def build_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> CosineLRScheduler | None:
    if total_steps <= 0:
        return None
    return CosineLRScheduler(optimizer, total_steps, warmup_steps, min_lr_ratio)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.grad_accum_steps <= 0:
        raise ValueError("--grad_accum_steps must be positive.")
    if args.refhuman_max_captions_per_instance < 0:
        raise ValueError("--refhuman_max_captions_per_instance must be >= 0.")
    if args.box_condition_scale <= 0:
        raise ValueError("--box_condition_scale must be positive.")
    if args.pose_roi_size <= 1:
        raise ValueError("--pose_roi_size must be greater than 1.")
    if not 0.0 <= args.hard_joint_fraction <= 1.0:
        raise ValueError("--hard_joint_fraction must be in [0, 1].")
    if args.box_jitter_scale < 0.0 or args.box_jitter_shift < 0.0:
        raise ValueError("--box_jitter_scale/--box_jitter_shift must be non-negative.")
    if args.qwen_box_max_new_tokens <= 0:
        raise ValueError("--qwen_box_max_new_tokens must be positive.")
    if not 0.0 <= args.box_match_iou_thresh <= 1.0:
        raise ValueError("--box_match_iou_thresh must be in [0, 1].")
    if not 0.0 <= args.box_nms_iou_thresh <= 1.0:
        raise ValueError("--box_nms_iou_thresh must be in [0, 1].")
    if args.box_source == "qwen_generate" and args.backbone != "qwen3vl":
        raise ValueError("--box_source=qwen_generate currently requires --backbone qwen3vl.")
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
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    use_deepspeed = args.deepspeed_config is not None
    world_size = distributed_world_size()
    rank = distributed_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))

    if use_deepspeed and torch.cuda.is_available():
        if local_rank >= 0:
            torch.cuda.set_device(local_rank)
        device = torch.device("cuda", max(local_rank, 0))
    else:
        device = torch.device(args.device)

    trace_handle = None
    trace_path: Path | None = None
    if not args.disable_batch_trace:
        if args.batch_trace_file is None:
            trace_path = args.output_dir / "logs" / f"batch_trace_rank{rank}.jsonl"
        else:
            trace_path = args.batch_trace_file
            if world_size > 1:
                suffix = trace_path.suffix or ".jsonl"
                trace_path = trace_path.with_name(f"{trace_path.stem}_rank{rank}{suffix}")
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_handle = open(trace_path, "a", encoding="utf-8", buffering=1)
        if is_main_process():
            print(f"Per-batch trace enabled: {trace_path}")

    # ------------------------------------------------------------------
    # 1. Build one mixed dataset. This is the single training stage:
    #    ALL_POSE from pose datasets and REF_POSE from RefHuman.
    # ------------------------------------------------------------------
    dataset_names = [name.strip().lower() for name in args.datasets.replace("/", ",").split(",") if name.strip()]
    dataset = build_datasets(
        dataset_root=args.dataset_root,
        names=dataset_names,
        max_instances=args.max_instances,
        image_size=args.image_size,
        load_image_tensors=not args.disable_image_tensors,
        split=args.split,
        max_samples_per_dataset=args.max_samples_per_dataset,
        refhuman_max_captions_per_instance=args.refhuman_max_captions_per_instance,
        mixing_strategy=args.mixing_strategy,
        dataset_mix_weights=args.dataset_mix_weights,
        seed=args.seed,
        record_cache_dir=args.record_cache_dir,
        disable_record_cache=args.disable_record_cache,
        show_progress=not args.disable_progress,
    )
    use_homogeneous_batches = (
        args.mixing_strategy == "interleave"
        and not args.disable_homogeneous_batches
        and hasattr(dataset, "global_index_for_dataset_linear")
        and len(dataset_names) > 1
    )
    sampler = None
    batch_sampler = None
    if use_homogeneous_batches:
        batch_sampler = HomogeneousDatasetBatchSampler(
            dataset,
            args.batch_size,
            seed=args.seed,
            rank=rank,
            world_size=world_size,
            shuffle=True,
            fill_last=True,
        )
    else:
        use_shuffle_sampler = world_size > 1 or args.mixing_strategy == "concat_shuffle"
        sampler = (
            DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=args.mixing_strategy == "concat_shuffle",
                seed=args.seed,
            )
            if use_shuffle_sampler
            else None
        )
    loader_kwargs = {
        "num_workers": args.num_workers,
        # The collated batch only contains small CPU tensors plus image paths.
        # Images are opened later in the main process when building Qwen inputs,
        # so pinning here brings little benefit and has caused sporadic multi-rank
        # stalls in practice.
        "pin_memory": False,
        "collate_fn": pose_collate,
    }
    if batch_sampler is not None:
        loader_kwargs["batch_sampler"] = batch_sampler
    else:
        loader_kwargs.update(
            {
                "batch_size": args.batch_size,
                "shuffle": sampler is None and args.mixing_strategy == "concat_shuffle",
                "sampler": sampler,
                "drop_last": False,
            }
        )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(dataset, **loader_kwargs)

    if batch_sampler is not None:
        batch_sampler.set_epoch(0)
        if is_main_process():
            print(
                "Homogeneous dataset batches enabled: "
                f"one source per batch, batches_per_rank={len(batch_sampler)}"
            )
    if sampler is not None:
        sampler.set_epoch(0)
    first_batch_started = time.perf_counter()
    first_batch = next(iter(loader))
    first_batch_elapsed = time.perf_counter() - first_batch_started
    if is_main_process():
        print(f"First batch warmup ready in {first_batch_elapsed:.2f}s")
        print("Batch preview:")
        print(summarize_batch(first_batch))
    if args.dry_run_data:
        if trace_handle is not None:
            trace_handle.close()
        return

    # ------------------------------------------------------------------
    # 2. Build backbone with LoRA and the trainable pose model.
    #    Supports qwen3vl (Qwen3-VL-4B) and eagle (LocateAnything-3B).
    # ------------------------------------------------------------------
    backbone_model = None
    backbone_processor = None
    external_dim = None
    backbone_name = args.backbone
    qwen_init_source: QwenInitializationSource | None = None
    qwenpose_init_payload: dict[str, object] | None = None

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
                gradient_checkpointing=args.eagle_gradient_checkpointing,
            )
        )
        backbone_model.to(device)
        backbone_model.train()
        external_dim = eagle_hidden_size(backbone_model)
        bb_trainable, bb_total = count_eagle_lora_parameters(backbone_model)
        if is_main_process():
            print(f"Eagle LoRA trainable parameters: {bb_trainable:,} / {bb_total:,}")
    else:
        qwen_init_source = detect_qwen_initialization_source(args.qwen_model_path)
        if qwen_init_source.adapter_path is not None and not qwen_init_source.is_merged_backbone:
            backbone_model, backbone_processor = load_qwen_with_existing_lora(
                base_model_path=qwen_init_source.backbone_model_path,
                adapter_path=str(qwen_init_source.adapter_path),
                dtype=args.qwen_dtype,
                attn_implementation=args.qwen_attn_implementation,
                gradient_checkpointing=args.qwen_gradient_checkpointing,
            )
        else:
            backbone_model, backbone_processor = load_qwen_with_lora(
                QwenLoRAConfig(
                    model_path=qwen_init_source.backbone_model_path,
                    lora_r=args.qwen_lora_r,
                    lora_alpha=args.qwen_lora_alpha,
                    lora_dropout=args.qwen_lora_dropout,
                    vision_lora_r=args.qwen_vision_lora_r,
                    vision_lora_alpha=args.qwen_vision_lora_alpha,
                    vision_lora_dropout=args.qwen_vision_lora_dropout,
                    dtype=args.qwen_dtype,
                    attn_implementation=args.qwen_attn_implementation,
                    gradient_checkpointing=args.qwen_gradient_checkpointing,
                )
            )
        backbone_model.to(device)
        backbone_model.train()
        external_dim = qwen_hidden_size(backbone_model)
        bb_trainable, bb_total = count_qwen_lora_parameters(backbone_model)
        if is_main_process():
            print(f"Qwen LoRA trainable parameters: {bb_trainable:,} / {bb_total:,}")
            print(
                "Qwen init source: "
                f"kind={qwen_init_source.source_kind}, "
                f"requested={qwen_init_source.requested_path}, "
                f"backbone_model_path={qwen_init_source.backbone_model_path}"
            )
            if qwen_init_source.adapter_path is not None:
                print(f"Qwen init adapter path: {qwen_init_source.adapter_path}")
            if qwen_init_source.pose_checkpoint_path is not None:
                print(f"QwenPose extra-module init checkpoint: {qwen_init_source.pose_checkpoint_path}")
                qwenpose_init_payload = _load_local_torch_payload(qwen_init_source.pose_checkpoint_path)
        if qwenpose_init_payload is None and qwen_init_source.pose_checkpoint_path is not None:
            qwenpose_init_payload = _load_local_torch_payload(qwen_init_source.pose_checkpoint_path)

    saved_pose_config = qwenpose_init_payload.get("pose_config") if qwenpose_init_payload is not None else None
    if saved_pose_config is not None:
        model_config = QwenPoseConfig(
            hidden_dim=int(saved_pose_config.get("hidden_dim", args.hidden_dim)),
            external_dim=external_dim,
            pose_decoder_layers=int(saved_pose_config.get("pose_decoder_layers", args.pose_decoder_layers)),
            refinement_steps=int(saved_pose_config.get("refinement_steps", args.refinement_steps)),
            decoder_heads=int(saved_pose_config.get("decoder_heads", args.decoder_heads)),
            box_condition_scale=float(saved_pose_config.get("box_condition_scale", args.box_condition_scale)),
            pose_roi_size=int(saved_pose_config.get("pose_roi_size", args.pose_roi_size)),
            use_refinement=bool(saved_pose_config.get("use_refinement", not args.disable_refinement)),
        )
    else:
        model_config = QwenPoseConfig(
            hidden_dim=args.hidden_dim,
            external_dim=external_dim,
            pose_decoder_layers=args.pose_decoder_layers,
            refinement_steps=args.refinement_steps,
            decoder_heads=args.decoder_heads,
            box_condition_scale=args.box_condition_scale,
            pose_roi_size=args.pose_roi_size,
            use_refinement=not args.disable_refinement,
        )
    model = QwenPoseModel(model_config).to(device)
    trainable, total = count_trainable_parameters(model)
    if is_main_process():
        print(f"Pose module trainable parameters: {trainable:,} / {total:,}")
    # Select feature extractor params based on backbone
    if backbone_name == "eagle":
        feature_size = args.eagle_feature_size
        refiner_layers = args.eagle_feature_refiner_layers
        refiner_bottleneck_dim = args.eagle_feature_refiner_bottleneck_dim
        refiner_init_scale = args.eagle_feature_refiner_init_scale
        freeze_backbone = args.freeze_eagle
    else:
        saved_feature_config = qwenpose_init_payload.get("qwen_feature_config") if qwenpose_init_payload is not None else None
        saved_refiner_config = qwenpose_init_payload.get("qwen_feature_refiner_config") if qwenpose_init_payload is not None else None
        feature_size = int(saved_feature_config.get("output_size", args.qwen_feature_size)) if saved_feature_config else args.qwen_feature_size
        refiner_layers = int(saved_refiner_config.get("layers", args.qwen_feature_refiner_layers)) if saved_refiner_config else args.qwen_feature_refiner_layers
        refiner_bottleneck_dim = (
            int(saved_refiner_config.get("bottleneck_dim", args.qwen_feature_refiner_bottleneck_dim))
            if saved_refiner_config
            else args.qwen_feature_refiner_bottleneck_dim
        )
        refiner_init_scale = (
            float(saved_refiner_config.get("init_scale", args.qwen_feature_refiner_init_scale))
            if saved_refiner_config
            else args.qwen_feature_refiner_init_scale
        )
        freeze_backbone = args.freeze_qwen

    # Build feature extractor
    if backbone_name == "eagle":
        backbone_extractor = EagleFeatureExtractor(
            backbone_model,
            output_size=feature_size,
            refiner_layers=refiner_layers,
            refiner_bottleneck_dim=refiner_bottleneck_dim,
            refiner_init_scale=refiner_init_scale,
        )
    else:
        backbone_extractor = QwenFeatureExtractor(
            backbone_model,
            output_size=feature_size,
            refiner_layers=refiner_layers,
            refiner_bottleneck_dim=refiner_bottleneck_dim,
            refiner_init_scale=refiner_init_scale,
        )

    training_model = QwenPoseTrainingModel(
        pose_model=model,
        backbone_model=backbone_model,
        backbone_extractor=backbone_extractor,
        backbone_name=backbone_name,
        freeze_backbone=freeze_backbone,
    ).to(device)
    if backbone_name == "qwen3vl" and qwen_init_source is not None and qwen_init_source.pose_checkpoint_path is not None:
        _, _, _, init_checkpoint, _ = load_training_checkpoint(
            training_model,
            optimizer=None,
            checkpoint_path=qwen_init_source.pose_checkpoint_path,
            load_optimizer=False,
            scaler=None,
            load_scaler=False,
        )
        if is_main_process():
            print(f"Initialized QwenPose extra modules from {init_checkpoint}")
    if is_main_process():
        print(f"Backbone: {backbone_name}")
        print(f"Feature grid size: {feature_size}x{feature_size}")
        print(f"Freeze backbone/LoRA: {freeze_backbone}")
        print(f"Box condition scale: {model_config.box_condition_scale}")
        print(f"Pose ROI size: {model_config.pose_roi_size}x{model_config.pose_roi_size}")
        print(f"Box source: {args.box_source}")
        print(f"Box jitter: scale={args.box_jitter_scale}, shift={args.box_jitter_shift}")
        if args.box_source == "qwen_generate":
            print(
                "Qwen generated-box loop: "
                f"max_new_tokens={args.qwen_box_max_new_tokens}, "
                f"match_iou_thresh={args.box_match_iou_thresh}, "
                f"nms_iou_thresh={args.box_nms_iou_thresh}"
            )
        print(
            "Qwen feature refiner: "
            f"layers={refiner_layers}, "
            f"bottleneck_dim={refiner_bottleneck_dim}, "
            f"init_scale={refiner_init_scale}"
        )

    # ------------------------------------------------------------------
    # 3. Configure optimizer and box-conditioned pose losses.
    # ------------------------------------------------------------------
    optim_params, optim_group_stats = build_optimizer_param_groups(training_model, args)
    if is_main_process():
        for group_name, (tensor_count, param_count, group_lr) in optim_group_stats.items():
            print(
                f"Optimizer group {group_name}: "
                f"{param_count:,} params in {tensor_count} tensors, lr={group_lr:.3e}"
            )
    optimizer = torch.optim.AdamW(
        optim_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    active_model: torch.nn.Module = training_model
    include_optimizer_in_checkpoint = True
    if use_deepspeed:
        import deepspeed

        ds_config = load_deepspeed_config(args.deepspeed_config, args, world_size)
        active_model, optimizer, _, _ = deepspeed.initialize(
            model=training_model,
            model_parameters=optim_params,
            optimizer=optimizer,
            config=ds_config,
        )
        include_optimizer_in_checkpoint = False
    total_epochs = max(int(args.epochs), 1)
    max_steps = int(args.max_steps)
    steps_per_epoch = math.ceil(len(loader) / max(int(args.grad_accum_steps), 1))
    scheduler_total_steps = max_steps if max_steps > 0 else steps_per_epoch * total_epochs
    scheduler = build_cosine_scheduler(
        optimizer,
        total_steps=scheduler_total_steps,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
    )
    weights = LossWeights(
        oks=args.w_oks,
        coord=args.w_coord,
        vis=args.w_vis,
        lm=args.w_lm,
        hard_joint=args.w_hard_joint,
        hard_joint_fraction=args.hard_joint_fraction,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    global_step = 0
    micro_step = 0
    resume_epoch = 0
    resume_batch_in_epoch = 0
    resume_rng_state: dict[str, object] | None = None
    resume_state_inferred = False
    loss_ema: float | None = None
    loss_ema_decay: float = 0.98
    loss_spike_threshold: float = 10.0   # skip if loss > threshold * ema
    loss_spike_count: int = 0
    loss_spike_max: int = 50             # abort after this many consecutive spikes
    loss_abs_cap: float = 50.0           # absolute loss cap — skip any batch above this
    if args.resume_from_checkpoint is not None:
        resolved_checkpoint = resolve_training_checkpoint(args.resume_from_checkpoint)
        optimizer_loaded = False
        scaler_loaded = False
        used_deepspeed_checkpoint = False
        resume_container: dict[str, object] = {}
        if (
            use_deepspeed
            and resolved_checkpoint.is_dir()
            and (resolved_checkpoint / DEEPSPEED_TAG).exists()
            and hasattr(active_model, "load_checkpoint")
        ):
            _, client_state = active_model.load_checkpoint(
                str(resolved_checkpoint),
                tag=DEEPSPEED_TAG,
                load_optimizer_states=True,
                load_lr_scheduler_states=False,
            )
            global_step = int((client_state or {}).get("step", checkpoint_step(resolved_checkpoint) or 0))
            optimizer_loaded = True
            used_deepspeed_checkpoint = True
            resume_container = dict(client_state or {})
        else:
            global_step, optimizer_loaded, scaler_loaded, resolved_checkpoint, payload = load_training_checkpoint(
                active_model,
                optimizer if include_optimizer_in_checkpoint else None,
                resolved_checkpoint,
                load_optimizer=include_optimizer_in_checkpoint,
                scaler=scaler,
                load_scaler=not use_deepspeed,
            )
            resume_container = dict(payload)
        resume_state, resume_state_inferred = infer_resume_state(
            resume_container.get("training_state") if isinstance(resume_container, dict) else None,
            global_step=global_step,
            batches_per_epoch=len(loader),
            grad_accum_steps=int(args.grad_accum_steps),
            batch_size=int(args.batch_size),
            world_size=world_size,
            dataset_names=dataset_names,
            mixing_strategy=args.mixing_strategy,
            split=args.split,
        )
        validate_resume_state(
            resume_state,
            current_batches_per_epoch=len(loader),
            batch_size=int(args.batch_size),
            world_size=world_size,
            dataset_names=dataset_names,
            mixing_strategy=args.mixing_strategy,
            split=args.split,
        )
        micro_step = int(resume_state.get("micro_step", global_step * max(int(args.grad_accum_steps), 1)))
        resume_epoch = int(resume_state.get("epoch", 0))
        resume_batch_in_epoch = int(resume_state.get("batch_in_epoch", 0))
        resume_rng_state = resume_container.get("rng_state") if isinstance(resume_container, dict) else None
        loaded_loss_ema = resume_state.get("loss_ema")
        loss_ema = None if loaded_loss_ema is None else float(loaded_loss_ema)
        loss_spike_count = int(resume_state.get("loss_spike_count", 0))
        if scheduler is not None:
            scheduler.set_step(global_step)
        if is_main_process():
            optimizer_note = "restored" if optimizer_loaded else "not restored"
            if used_deepspeed_checkpoint:
                optimizer_note = "restored (DeepSpeed checkpoint)"
            elif not include_optimizer_in_checkpoint:
                optimizer_note = "not restored (DeepSpeed lightweight checkpoint)"
            scaler_note = "restored" if scaler_loaded else "not restored"
            if use_deepspeed:
                scaler_note = "managed by DeepSpeed"
            resume_note = "resume cursor restored from checkpoint metadata"
            if resume_state_inferred:
                resume_note = "resume cursor inferred from legacy step-only checkpoint"
            print(
                f"Resumed from {resolved_checkpoint} at global_step={global_step}; "
                f"next_epoch={resume_epoch + 1}, next_batch_in_epoch={resume_batch_in_epoch}/{len(loader)}, "
                f"optimizer={optimizer_note}, scaler={scaler_note}, {resume_note}."
            )

    # ------------------------------------------------------------------
    # 4. One-stage training loop. It does not separate ALL_POSE and REF_POSE
    #    phases; batches are sampled from the mixed dataset directly.
    # ------------------------------------------------------------------
    active_model.train()
    if not use_deepspeed:
        optimizer.zero_grad(set_to_none=True)
    stop_training = False
    resume_cursor_epoch = resume_epoch
    resume_cursor_batch_in_epoch = resume_batch_in_epoch
    if max_steps > 0 and global_step >= max_steps:
        stop_training = True
        if is_main_process():
            print(f"Resume step {global_step} already reached max_steps={max_steps}; skipping training loop.")
    elif resume_epoch >= total_epochs:
        stop_training = True
        if is_main_process():
            if max_steps > 0 and global_step < max_steps:
                print(
                    f"Resume cursor is already at epoch {resume_epoch + 1}, "
                    f"which is past requested total epochs={total_epochs}. "
                    "Increase --epochs if you want to keep training toward the new max_steps target."
                )
            else:
                print(
                    f"Resume cursor is already at epoch {resume_epoch + 1}, "
                    f"which is past requested total epochs={total_epochs}; skipping training loop."
                )
    resume_rng_state_pending = resume_rng_state
    for epoch in range(resume_epoch, total_epochs):
        if stop_training:
            break
        if batch_sampler is not None:
            batch_sampler.set_epoch(epoch)
        if sampler is not None:
            sampler.set_epoch(epoch)
        batches_to_skip = resume_batch_in_epoch if epoch == resume_epoch else 0
        if is_main_process():
            print(f"Starting epoch {epoch + 1}/{total_epochs}")
            if batches_to_skip > 0:
                print(
                    f"Skipping {batches_to_skip} already-processed batches before resuming "
                    f"epoch {epoch + 1}/{total_epochs}."
                )
        progress_bar = None
        if is_main_process() and not args.disable_progress and tqdm is not None:
            total_batches = len(loader)
            if max_steps > 0:
                remaining_updates = max(max_steps - global_step, 0)
                remaining_batches = remaining_updates * max(int(args.grad_accum_steps), 1)
                total_batches = min(total_batches, max(batches_to_skip + remaining_batches, batches_to_skip + 1))
            progress_bar = tqdm(
                total=total_batches,
                desc=f"epoch {epoch + 1}/{total_epochs}",
                unit="batch",
                dynamic_ncols=True,
                mininterval=1.0,
                initial=batches_to_skip,
            )
        last_iter_end = time.perf_counter()
        timing_sums = {"data": 0.0, "prep": 0.0, "fwd": 0.0, "bwd": 0.0, "micro": 0}
        batch_iterator = iter(loader)
        if batches_to_skip > 0:
            for skipped_idx in range(batches_to_skip):
                try:
                    next(batch_iterator)
                except StopIteration as exc:
                    raise RuntimeError(
                        f"Failed to skip resume batch offset {batches_to_skip} within epoch {epoch + 1}; "
                        "the current dataloader is shorter than the saved resume position."
                    ) from exc
        if resume_rng_state_pending is not None:
            restore_rng_state(resume_rng_state_pending)
            resume_rng_state_pending = None
        for batch_idx, batch in enumerate(batch_iterator, start=batches_to_skip):
            batch_ready = time.perf_counter()
            data_time = batch_ready - last_iter_end
            micro_step += 1
            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)
            qwen_inputs = None
            qwen_lm_inputs = None
            trace_qwen_inputs = None
            use_lm_loss = False
            pose_targets = batch.get("targets", [])
            prep_time = None
            forward_time = None
            step_time = None
            did_update = False
            skip_batch = None
            loss_metrics_for_trace: dict[str, float] | None = None
            trace_loss_val: float | None = None
            try:
                sync_cuda_for_timing(args.sync_timing, device)
                prep_started = time.perf_counter()
                batch = move_batch_to_device(batch, device)
                if args.box_source == "qwen_generate":
                    target_boxes, target_box_mask, pose_targets = prepare_qwen_generated_box_conditioning(
                        active_model,
                        backbone_processor,
                        batch,
                        device,
                        max_instances=args.max_instances,
                        max_new_tokens=args.qwen_box_max_new_tokens,
                        match_iou_thresh=args.box_match_iou_thresh,
                        nms_iou_thresh=args.box_nms_iou_thresh,
                        min_pixels=args.qwen_min_pixels,
                        max_pixels=args.qwen_max_pixels,
                    )
                else:
                    target_boxes, target_box_mask, pose_targets = prepare_box_conditioning(
                        batch["targets"],
                        batch["task_ids"],
                        device,
                        max_instances=args.max_instances,
                        box_jitter_scale=args.box_jitter_scale,
                        box_jitter_shift=args.box_jitter_shift,
                    )
                if backbone_processor is None:
                    qwen_inputs = None
                elif backbone_name == "eagle":
                    qwen_inputs = build_eagle_inputs(
                        backbone_processor,
                        batch["image_paths"],
                        batch["prompts"],
                        device,
                        min_pixels=args.eagle_min_pixels,
                        max_pixels=args.eagle_max_pixels,
                    )
                    locate_aux_weight = float(args.w_locate_box_lm) + float(args.w_locate_point_lm)
                    use_lm_loss = (
                        locate_aux_weight > 0.0
                        and not args.disable_locate_grounding_aux
                        and not args.freeze_eagle
                        and args.locate_lm_loss_every > 0
                        and micro_step % args.locate_lm_loss_every == 0
                    )
                    if use_lm_loss:
                        if build_eagle_lm_inputs is None:
                            raise RuntimeError(
                                "Locate grounding LM auxiliary training requires build_eagle_lm_inputs in qwenpose.eagle_lora."
                            )
                        box_responses, point_responses = build_locate_grounding_responses(
                            batch,
                            max_instances=args.locate_lm_max_instances,
                            max_points=args.locate_lm_max_points,
                        )
                        locate_responses = [
                            f"Boxes: {box_text}\nPoints: {point_text}"
                            for box_text, point_text in zip(box_responses, point_responses)
                        ]
                        qwen_lm_inputs = build_eagle_lm_inputs(
                            backbone_processor,
                            batch["image_paths"],
                            batch["prompts"],
                            locate_responses,
                            device,
                            min_pixels=args.eagle_min_pixels,
                            max_pixels=args.eagle_max_pixels,
                        )
                    trace_qwen_inputs = qwen_lm_inputs if qwen_lm_inputs is not None else qwen_inputs
                else:
                    use_lm_loss = (
                        args.w_lm > 0.0
                        and args.lm_loss_every > 0
                        and micro_step % args.lm_loss_every == 0
                    )
                    if use_lm_loss:
                        lm_responses = build_lm_responses(
                            batch,
                            max_instances=args.lm_max_answer_instances,
                        )
                        qwen_lm_inputs = build_qwen_lm_inputs(
                            backbone_processor,
                            batch["image_paths"],
                            batch["prompts"],
                            lm_responses,
                            device,
                            min_pixels=args.qwen_min_pixels,
                            max_pixels=args.qwen_max_pixels,
                        )
                        # Reuse the LM batch inputs for trace/profiling, and let the
                        # training module extract pose features from the same Qwen pass.
                        trace_qwen_inputs = qwen_lm_inputs
                    else:
                        qwen_inputs = build_qwen_inputs(
                            backbone_processor,
                            batch["image_paths"],
                            batch["prompts"],
                            device,
                            min_pixels=args.qwen_min_pixels,
                            max_pixels=args.qwen_max_pixels,
                        )
                        trace_qwen_inputs = qwen_inputs
                sync_cuda_for_timing(args.sync_timing, device)
                prep_time = time.perf_counter() - prep_started

                sync_cuda_for_timing(args.sync_timing, device)
                forward_started = time.perf_counter()
                with torch.amp.autocast("cuda", enabled=(not use_deepspeed) and args.amp and device.type == "cuda"):
                    outputs = active_model(
                        schema_ids=batch["schema_ids"],
                        task_ids=batch["task_ids"],
                        qwen_inputs=qwen_inputs,
                        qwen_lm_inputs=qwen_lm_inputs,
                        target_boxes=target_boxes,
                        target_box_mask=target_box_mask,
                        images=batch.get("images"),
                    )
                    loss, loss_dict = compute_pose_losses(outputs, pose_targets, batch["task_ids"], weights)
                    if "lm_loss" in outputs:
                        lm_loss = outputs["lm_loss"].float()
                        lm_weight = weights.lm
                        if backbone_name == "eagle":
                            lm_weight = float(args.w_locate_box_lm) + float(args.w_locate_point_lm)
                        loss = loss + lm_weight * lm_loss
                        loss_dict["loss_lm"] = lm_loss
                        loss_dict["loss_lm_weight"] = torch.as_tensor(lm_weight, device=loss.device)
                        loss_dict["loss_total"] = loss

                # --- Loss spike detection ---
                loss_val = float(loss.detach())
                trace_loss_val = loss_val
                skip_batch = False
                if not math.isfinite(loss_val) or loss_val > loss_abs_cap:
                    skip_batch = True
                elif loss_ema is not None and loss_val > loss_ema * loss_spike_threshold:
                    skip_batch = True
                if skip_batch:
                    loss_spike_count += 1
                    ema_str = f"{loss_ema:.4f}" if loss_ema is not None else "n/a"
                    if is_main_process():
                        progress_write(
                            progress_bar,
                            f"[SPIKE] step={global_step} micro={micro_step} loss={loss_val:.4f} "
                            f"ema={ema_str} consecutive={loss_spike_count} — skipping batch",
                        )
                    # Replace loss with zero so backward produces zero gradients.
                    loss = loss * 0.0
                    if loss_spike_count >= loss_spike_max:
                        if is_main_process():
                            print(f"[ABORT] {loss_spike_count} consecutive loss spikes; stopping training.")
                        stop_training = True
                else:
                    loss_spike_count = 0
                    loss_ema = (
                        loss_val
                        if loss_ema is None
                        else loss_ema * loss_ema_decay + loss_val * (1.0 - loss_ema_decay)
                    )

                sync_cuda_for_timing(args.sync_timing, device)
                forward_time = time.perf_counter() - forward_started

                sync_cuda_for_timing(args.sync_timing, device)
                step_started = time.perf_counter()
                if use_deepspeed:
                    boundary = active_model.is_gradient_accumulation_boundary()
                    active_model.backward(loss)
                    active_model.step()
                    did_update = bool(boundary)
                else:
                    scaler.scale(loss / args.grad_accum_steps).backward()
                    if micro_step % args.grad_accum_steps == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(training_model.parameters(), args.grad_clip)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        did_update = True
                sync_cuda_for_timing(args.sync_timing, device)
                step_time = time.perf_counter() - step_started
                loss_metrics_for_trace = {k: float(v.detach().cpu()) for k, v in loss_dict.items()}
            except Exception as exc:
                error_stage = "oom" if "out of memory" in str(exc).lower() else "error"
                append_batch_trace(
                    trace_handle,
                    build_batch_trace_record(
                        batch=batch,
                        pose_targets=pose_targets,
                        epoch=epoch,
                        batch_idx=batch_idx,
                        global_step=global_step,
                        micro_step=micro_step,
                        grad_accum_steps=int(args.grad_accum_steps),
                        rank=rank,
                        local_rank=local_rank,
                        device=device,
                        stage=error_stage,
                        qwen_inputs=trace_qwen_inputs,
                        loss=trace_loss_val,
                        did_update=False,
                        skip_batch=skip_batch,
                        error=str(exc),
                        data_time=data_time,
                        prep_time=prep_time,
                        forward_time=forward_time,
                        step_time=step_time,
                    ),
                )
                if error_stage == "oom" and device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if trace_handle is not None:
                    trace_handle.flush()
                raise
            last_iter_end = time.perf_counter()
            timing_sums["data"] += data_time
            timing_sums["prep"] += prep_time
            timing_sums["fwd"] += forward_time
            timing_sums["bwd"] += step_time
            timing_sums["micro"] += 1
            loss_metrics: dict[str, float] | None = None
            progress_loss_postfix: dict[str, str] | None = None
            if is_main_process():
                if progress_bar is not None:
                    loss_metrics = {k: float(v.detach().cpu()) for k, v in loss_dict.items()}
                    progress_loss_postfix = build_progress_loss_postfix(loss_metrics, weights)
                    update_progress_bar(
                        progress_bar,
                        {
                            "step": global_step + (1 if did_update else 0),
                            **progress_loss_postfix,
                            "src": ",".join(batch.get("source_datasets", [])[:2]),
                            "data": f"{data_time:.2f}s",
                            "prep": f"{prep_time:.2f}s",
                            "fwd": f"{forward_time:.2f}s",
                            "bwd": f"{step_time:.2f}s",
                        },
                    )
            append_batch_trace(
                trace_handle,
                build_batch_trace_record(
                    batch=batch,
                    pose_targets=pose_targets,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    global_step=global_step + (1 if did_update else 0),
                    micro_step=micro_step,
                    grad_accum_steps=int(args.grad_accum_steps),
                    rank=rank,
                    local_rank=local_rank,
                    device=device,
                    stage="completed",
                    qwen_inputs=trace_qwen_inputs,
                    loss=trace_loss_val,
                    loss_dict=loss_metrics_for_trace,
                    did_update=did_update,
                    skip_batch=skip_batch,
                    data_time=data_time,
                    prep_time=prep_time,
                    forward_time=forward_time,
                    step_time=step_time,
                ),
            )
            if progress_bar is not None:
                progress_bar.update(1)
            resume_cursor_epoch, resume_cursor_batch_in_epoch = next_resume_position_after_batch(
                epoch,
                batch_idx,
                len(loader),
            )

            if did_update:
                global_step += 1
                if scheduler is not None:
                    scheduler.step()
                if is_main_process() and (global_step % args.log_every == 0 or global_step == 1):
                    if loss_metrics is None:
                        loss_metrics = {k: float(v.detach().cpu()) for k, v in loss_dict.items()}
                    if progress_loss_postfix is None:
                        progress_loss_postfix = build_progress_loss_postfix(loss_metrics, weights)
                    current_lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else args.lr
                    timing_count = max(int(timing_sums["micro"]), 1)
                    timing_message = (
                        f"lr={current_lr:.3e} "
                        f"time_data={timing_sums['data'] / timing_count:.3f}s "
                        f"time_prep={timing_sums['prep'] / timing_count:.3f}s "
                        f"time_fwd={timing_sums['fwd'] / timing_count:.3f}s "
                        f"time_bwd={timing_sums['bwd'] / timing_count:.3f}s "
                        f"time_micro={timing_count}"
                    )
                    progress_write(
                        progress_bar,
                        f"step={global_step} micro_step={micro_step} "
                        + " ".join(f"{k}={v}" for k, v in progress_loss_postfix.items())
                        + " "
                        + timing_message,
                    )
                    timing_sums = {"data": 0.0, "prep": 0.0, "fwd": 0.0, "bwd": 0.0, "micro": 0}

                if is_main_process() and args.visualize_every > 0 and (
                    global_step % args.visualize_every == 0 or global_step == 1
                ):
                    vis_target = pose_targets[0]
                    vis_source_datasets = batch.get("source_datasets", [])
                    vis_dataset_name = (
                        vis_source_datasets[0]
                        if vis_source_datasets
                        else vis_target.get("dataset", "unknown")
                    )
                    vis_dataset = _safe_vis_tag(vis_dataset_name)
                    vis_schema = _safe_vis_tag(vis_target.get("schema", "unknown"))
                    vis_path = args.output_dir / "visualizations" / f"train_step_{global_step:08d}_{vis_dataset}_{vis_schema}.jpg"
                    try:
                        save_pose_visualization(
                            outputs,
                            {**batch, "targets": pose_targets},
                            vis_path,
                            sample_idx=0,
                            max_instances=args.visualize_max_instances,
                        )
                        progress_write(progress_bar, f"saved_visualization={vis_path}")
                    except Exception as vis_exc:
                        progress_write(
                            progress_bar,
                            f"[WARN] step={global_step} visualization skipped: {type(vis_exc).__name__}: {vis_exc}",
                        )

                if args.save_every > 0 and global_step % args.save_every == 0:
                    training_state = build_training_state(
                        epoch=resume_cursor_epoch,
                        batch_in_epoch=resume_cursor_batch_in_epoch,
                        batches_per_epoch=len(loader),
                        global_step=global_step,
                        micro_step=micro_step,
                        grad_accum_steps=int(args.grad_accum_steps),
                        batch_size=int(args.batch_size),
                        world_size=world_size,
                        dataset_names=dataset_names,
                        mixing_strategy=args.mixing_strategy,
                        split=args.split,
                        loss_ema=loss_ema,
                        loss_spike_count=loss_spike_count,
                    )
                    save_checkpoint(
                        active_model,
                        optimizer,
                        global_step,
                        args.output_dir,
                        save_total_limit=args.save_total_limit,
                        include_optimizer=include_optimizer_in_checkpoint,
                        qwen_processor=backbone_processor,
                        save_deepspeed=use_deepspeed,
                        scaler=scaler,
                        training_state=training_state,
                    )

                if max_steps > 0 and global_step >= max_steps:
                    stop_training = True
                    break
        if progress_bar is not None:
            progress_bar.close()
        if epoch == resume_epoch:
            resume_batch_in_epoch = 0
        if stop_training:
            break

    if not use_deepspeed and micro_step % args.grad_accum_steps != 0 and not stop_training:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(training_model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        if len(loader) > 0:
            resume_cursor_epoch, resume_cursor_batch_in_epoch = normalize_resume_position(
                resume_cursor_epoch,
                resume_cursor_batch_in_epoch,
                len(loader),
            )
        if scheduler is not None:
            scheduler.step()

    training_state = build_training_state(
        epoch=resume_cursor_epoch,
        batch_in_epoch=resume_cursor_batch_in_epoch,
        batches_per_epoch=len(loader),
        global_step=global_step,
        micro_step=micro_step,
        grad_accum_steps=int(args.grad_accum_steps),
        batch_size=int(args.batch_size),
        world_size=world_size,
        dataset_names=dataset_names,
        mixing_strategy=args.mixing_strategy,
        split=args.split,
        loss_ema=loss_ema,
        loss_spike_count=loss_spike_count,
    )
    save_checkpoint(
        active_model,
        optimizer,
        global_step,
        args.output_dir,
        save_total_limit=args.save_total_limit,
        include_optimizer=include_optimizer_in_checkpoint,
        qwen_processor=backbone_processor,
        save_deepspeed=use_deepspeed,
        scaler=scaler,
        training_state=training_state,
    )
    final_module = unwrap_training_model(active_model)
    if is_main_process() and final_module.backbone_model is not None and hasattr(final_module.backbone_model, "save_pretrained"):
        adapter_dir_name = f"{backbone_name}_lora_adapter"
        final_module.backbone_model.save_pretrained(args.output_dir / adapter_dir_name)
        if backbone_processor is not None:
            backbone_processor.save_pretrained(args.output_dir / adapter_dir_name)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    if trace_handle is not None:
        trace_handle.close()
    if is_main_process():
        print(f"Training finished at step {global_step}. Checkpoints saved to {args.output_dir}")


if __name__ == "__main__":
    main()
