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
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

try:
    from scipy.optimize import linear_sum_assignment as _scipy_linear_sum_assignment
except Exception:  # pragma: no cover - deterministic greedy fallback keeps minimal envs usable.
    _scipy_linear_sum_assignment = None
from torch.utils.data.distributed import DistributedSampler

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional for minimal envs.
    tqdm = None

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qwenpose.data import (
    DATASET_BOX_CONTEXT_SCALE,
    PoseAugmentConfig,
    build_datasets,
    pose_collate,
    set_pose_dataset_epoch,
)
from qwenpose.losses import (
    LossWeights,
    compute_person_confidence_quality_loss,
    compute_pose_losses,
)
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
    build_eagle_lm_inputs,
    count_eagle_lora_parameters,
    get_eagle_base_model,
    load_eagle_vision_only_with_lora,
    load_eagle_with_lora,
    eagle_hidden_size,
)
from qwenpose.schemas import ID_TO_SCHEMA, UNION_KEYPOINTS, UNION_SIGMAS


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
    parser.add_argument(
        "--image_size",
        type=int,
        default=800,
        help="Unified square RGB input size shared by all LocatePose stages.",
    )
    parser.add_argument(
        "--disable_image_tensors",
        action="store_true",
        help="Disable loading fixed-size RGB tensors for the pose visual branch.",
    )
    parser.add_argument("--max_samples_per_dataset", type=int, default=None)
    parser.add_argument(
        "--refhuman_max_captions_per_instance",
        type=int,
        default=1,
        help=(
            "Maximum captions used for each RefHuman person instance in one epoch. "
            "Captions follow a seed-randomized epoch rotation; use 0 to keep all "
            "captions as separate records."
        ),
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
        help=(
            "Per-epoch dataset traversal multipliers. Use auto for one traversal of every source, "
            "or values such as coco:3,mpii:3,crowdpose:3,refhuman:0.5. Zero disables a source; "
            "fractional sources continue from the next slice in following epochs."
        ),
    )
    parser.add_argument(
        "--disable_homogeneous_batches",
        action="store_true",
        help="Disable one-dataset-per-batch sampling for interleaved multi-dataset training.",
    )
    parser.add_argument(
        "--disable_vision_token_balancing",
        action="store_true",
        help="Disable cross-rank batching by estimated LocateAnything vision-token cost.",
    )
    parser.add_argument("--pose_augment", action="store_true", help="Enable synchronized image/box/keypoint augmentation.")
    parser.add_argument("--augment_flip_prob", type=float, default=0.5)
    parser.add_argument("--augment_affine_prob", type=float, default=0.8)
    parser.add_argument("--augment_rotate_degrees", type=float, default=15.0)
    parser.add_argument("--augment_scale_min", type=float, default=0.85)
    parser.add_argument("--augment_scale_max", type=float, default=1.15)
    parser.add_argument("--augment_translate_fraction", type=float, default=0.08)
    parser.add_argument("--augment_color_prob", type=float, default=0.8)
    parser.add_argument("--augment_brightness", type=float, default=0.20)
    parser.add_argument("--augment_contrast", type=float, default=0.20)
    parser.add_argument("--augment_saturation", type=float, default=0.20)
    parser.add_argument("--augment_hue", type=float, default=0.05)
    parser.add_argument("--augment_grayscale_prob", type=float, default=0.05)
    parser.add_argument("--augment_blur_prob", type=float, default=0.10)
    parser.add_argument("--augment_blur_sigma_min", type=float, default=0.10)
    parser.add_argument("--augment_blur_sigma_max", type=float, default=1.50)
    parser.add_argument("--augment_erase_prob", type=float, default=0.15)
    parser.add_argument("--augment_erase_area_min", type=float, default=0.02)
    parser.add_argument("--augment_erase_area_max", type=float, default=0.10)

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
    # LocatePose / LocateAnything-3B backbone options. The --eagle_* names are
    # kept as legacy aliases; scripts should prefer the --locate_* names.
    parser.add_argument("--locate_model_path", "--eagle_model_path", dest="eagle_model_path", type=str, default="weights/LocateAnything-3B")
    parser.add_argument(
        "--locate_dtype",
        "--eagle_dtype",
        dest="eagle_dtype",
        choices=["bfloat16", "float16", "float32", "auto", "none"],
        default="bfloat16",
    )
    parser.add_argument("--locate_attn_implementation", "--eagle_attn_implementation", dest="eagle_attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--locate_gradient_checkpointing", "--eagle_gradient_checkpointing", dest="eagle_gradient_checkpointing", action="store_true")
    parser.add_argument(
        "--locate_image_token_limit",
        "--eagle_image_token_limit",
        dest="eagle_image_token_limit",
        type=int,
        default=None,
        help="LocateAnything raw MoonViT patch-token limit per image.",
    )
    parser.add_argument(
        "--locate_batch_token_limit",
        "--eagle_batch_token_limit",
        dest="eagle_batch_token_limit",
        type=int,
        default=None,
        help="Maximum raw MoonViT patch-token budget for one local micro batch.",
    )
    parser.add_argument("--locate_feature_size", "--eagle_feature_size", dest="eagle_feature_size", type=int, default=100)
    parser.add_argument("--locate_feature_refiner_layers", "--eagle_feature_refiner_layers", dest="eagle_feature_refiner_layers", type=int, default=0)
    parser.add_argument("--locate_feature_refiner_bottleneck_dim", "--eagle_feature_refiner_bottleneck_dim", dest="eagle_feature_refiner_bottleneck_dim", type=int, default=256)
    parser.add_argument("--locate_feature_refiner_init_scale", "--eagle_feature_refiner_init_scale", dest="eagle_feature_refiner_init_scale", type=float, default=0.1)
    parser.add_argument("--locate_lora_r", "--eagle_lora_r", dest="eagle_lora_r", type=int, default=32)
    parser.add_argument("--locate_lora_alpha", "--eagle_lora_alpha", dest="eagle_lora_alpha", type=int, default=64)
    parser.add_argument("--locate_lora_dropout", "--eagle_lora_dropout", dest="eagle_lora_dropout", type=float, default=0.05)
    parser.add_argument("--locate_vision_lora_r", "--eagle_vision_lora_r", dest="eagle_vision_lora_r", type=int, default=16)
    parser.add_argument("--locate_vision_lora_alpha", "--eagle_vision_lora_alpha", dest="eagle_vision_lora_alpha", type=int, default=32)
    parser.add_argument("--locate_vision_lora_dropout", "--eagle_vision_lora_dropout", dest="eagle_vision_lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--locate_feature_source",
        choices=["vision_only", "raw_visual"],
        default="raw_visual",
        help=(
            "vision_only loads only MoonViT; raw_visual loads the full model and LLM text path "
            "while feeding the same raw MoonViT map to PoseHead."
        ),
    )
    parser.add_argument(
        "--prune_locate_generation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Skip LocateAnything's duplicate lm_head checkpoint tensor, remove the vocabulary "
            "projection, and disable KV-cache generation. Use only when coordinates come from "
            "the person-query detection head."
        ),
    )
    parser.add_argument(
        "--locate_train_scope",
        choices=["frozen", "vision_lora", "all_lora", "selective_lora"],
        default="all_lora",
        help="Select which LocateAnything adapters receive gradients.",
    )
    parser.add_argument(
        "--locate_llm_layers",
        type=str,
        default="32-35",
        help="LLM layer ranges enabled by selective_lora, e.g. 32-35.",
    )
    parser.add_argument(
        "--locate_vision_layers",
        type=str,
        default="15-26",
        help="MoonViT block ranges enabled by selective_lora, e.g. 15-26.",
    )
    parser.add_argument(
        "--locate_llm_modules",
        type=str,
        default="q_proj,v_proj",
        help="Comma-separated LLM LoRA projections enabled by selective_lora.",
    )
    parser.add_argument(
        "--locate_vision_modules",
        type=str,
        default="wqkv,wo,fc0,fc1",
        help="Comma-separated MoonViT LoRA projections enabled by selective_lora.",
    )
    parser.add_argument("--train_locate_projector", action="store_true")
    parser.add_argument("--freeze_locate", "--freeze_eagle", dest="freeze_eagle", action="store_true")
    parser.add_argument("--pose_decoder_layers", type=int, default=3)
    parser.add_argument("--refinement_steps", type=int, default=3)
    parser.add_argument("--human_decoder_layers", type=int, default=2)
    parser.add_argument("--decoder_heads", type=int, default=8)
    parser.add_argument("--pose_dropout", type=float, default=0.0)
    parser.add_argument("--box_condition_scale", type=float, default=1.25)
    parser.add_argument(
        "--pose_coordinate_init",
        choices=("learned_spread", "box_center", "schema_prior"),
        default="learned_spread",
        help=(
            "Main-pose coordinate reference. learned_spread uses trainable, "
            "non-anatomical dispersed anchors; box_center is an ablation and "
            "schema_prior is retained only for legacy checkpoint evaluation."
        ),
    )
    parser.add_argument(
        "--schema_joint_priors_path",
        type=str,
        default="configs/schema_joint_priors.json",
        help="Legacy schema-prior JSON; used only by --pose_coordinate_init=schema_prior.",
    )
    parser.add_argument("--pose_roi_size", type=int, default=16)
    parser.add_argument("--pose_pyramid_channels", type=int, default=128)
    parser.add_argument("--pose_pyramid_blocks", type=int, default=3)
    parser.add_argument("--deformable_points", type=int, default=4)
    parser.add_argument("--deformable_min_radius_cells", type=float, default=2.0)
    parser.add_argument(
        "--ref_text_scale",
        type=float,
        default=0.2,
        help="Scale applied when RefHuman text conditions human and joint queries.",
    )
    parser.add_argument(
        "--disable_ref_visual_modulation",
        action="store_true",
        help="Disable zero-initialized RefHuman text FiLM modulation on P2/P3/P4.",
    )
    parser.add_argument(
        "--disable_box_denoising",
        action="store_true",
        help="Disable training-only positive/negative box denoising queries.",
    )
    parser.add_argument("--max_dn_queries", type=int, default=96)
    parser.add_argument("--max_dn_groups", type=int, default=4)
    parser.add_argument("--dn_positive_noise", type=float, default=0.40)
    parser.add_argument("--dn_negative_noise", type=float, default=1.00)
    parser.add_argument(
        "--disable_keypoint_denoising",
        action="store_true",
        help="Disable training-only box-conditioned OKS keypoint denoising.",
    )
    parser.add_argument("--max_keypoint_dn_queries", type=int, default=16)
    parser.add_argument("--max_keypoint_dn_groups", type=int, default=2)
    parser.add_argument("--keypoint_dn_positive_ks_min", type=float, default=0.5)
    parser.add_argument("--keypoint_dn_positive_ks_max", type=float, default=1.0)
    parser.add_argument("--keypoint_dn_negative_ks_min", type=float, default=0.1)
    parser.add_argument("--keypoint_dn_negative_ks_max", type=float, default=0.5)
    parser.add_argument("--disable_refinement", action="store_true")
    parser.add_argument(
        "--person_confidence_rescue",
        action="store_true",
        help=(
            "Rebuild the checkpoint-14000 legacy PoseHead, freeze every existing "
            "weight, and train only a YOLO-style instance confidence head."
        ),
    )
    parser.add_argument(
        "--legacy_checkpoint_compat",
        action="store_true",
        help=(
            "Use the checkpoint-14000 legacy PoseHead graph (legacy RGB/refinement "
            "path and visibility head) without enabling rescue-only freezing."
        ),
    )
    parser.add_argument(
        "--enable_person_confidence_head",
        action="store_true",
        help=(
            "Enable the learned YOLO-style instance confidence head during normal "
            "pose training without freezing the rest of the model."
        ),
    )
    parser.add_argument(
        "--init_from_checkpoint",
        type=Path,
        default=None,
        help="Weight-only initialization checkpoint used by confidence rescue.",
    )

    # ---------------------------------------------------------------------
    # Optimization section.
    # ---------------------------------------------------------------------
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/qwenpose_debug"))
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help=(
            "Number of training epochs. Set to 0 together with --max_steps > 0 "
            "for optimizer-step-only training without an epoch cap."
        ),
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="Optional step cap. Use 0 to train for the requested number of epochs.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--qwen_lora_lr_scale", type=float, default=1.0)
    parser.add_argument("--qwen_vision_lr_scale", type=float, default=0.01)
    parser.add_argument("--locate_vision_scale", type=float, default=0.10)
    parser.add_argument("--locate_llm_scale", type=float, default=0.01)
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
    parser.add_argument("--w_oks", type=float, default=0.5)
    parser.add_argument("--w_coord", type=float, default=3.0)
    parser.add_argument("--w_image_coord", type=float, default=5.0)
    parser.add_argument(
        "--w_keypoint_confidence",
        "--w_vis",
        dest="w_keypoint_confidence",
        type=float,
        default=0.1,
        help="Weight for evaluator-aligned per-keypoint localization confidence.",
    )
    parser.add_argument(
        "--w_person_confidence",
        type=float,
        default=0.0,
        help=(
            "Weight for evaluator-aligned instance OKS confidence. Generated boxes "
            "that do not match a GT person are zero-quality negatives."
        ),
    )
    parser.add_argument(
        "--w_ref_match",
        type=float,
        default=1.0,
        help="Weight for RefHuman expression-to-candidate cross-entropy.",
    )
    parser.add_argument("--w_lm", type=float, default=0.05)
    parser.add_argument("--w_hard_joint", type=float, default=0.0)
    parser.add_argument("--hard_joint_fraction", type=float, default=0.2)
    parser.add_argument("--w_coarse_coord", type=float, default=0.5)
    parser.add_argument("--w_deform_coord", type=float, default=0.75)
    parser.add_argument("--w_refine_coords", type=str, default="0.75,1.0,1.25")
    parser.add_argument("--w_box_objectness", type=float, default=1.0)
    parser.add_argument("--w_box_l1", type=float, default=5.0)
    parser.add_argument("--w_box_giou", type=float, default=2.0)
    parser.add_argument("--w_box_relative", type=float, default=1.0)
    parser.add_argument("--w_box_dn", type=float, default=1.0)
    parser.add_argument("--w_keypoint_dn", type=float, default=1.0)
    parser.add_argument(
        "--box_jitter_scale",
        type=float,
        default=0.0,
        help="Fallback jitter for records without per-dataset jitter metadata.",
    )
    parser.add_argument(
        "--box_jitter_shift",
        type=float,
        default=0.0,
        help="Fallback shift jitter for records without per-dataset metadata.",
    )
    parser.add_argument(
        "--box_source",
        choices=["gt", "qwen_generate", "locate_generate", "person_queries"],
        default="gt",
        help=(
            "Box conditions for PoseHead: GT/generated boxes, or unified learned "
            "person queries that detect every person before RefHuman selection."
        ),
    )
    parser.add_argument(
        "--num_person_queries",
        type=int,
        default=80,
        help="Number of global detection/pose candidates for --box_source=person_queries.",
    )
    parser.add_argument("--qwen_box_max_new_tokens", type=int, default=4096)
    parser.add_argument("--locate_box_max_new_tokens", type=int, default=512)
    parser.add_argument(
        "--locate_generation_mode",
        choices=["fast", "slow", "hybrid"],
        default="hybrid",
    )
    parser.add_argument("--box_match_iou_thresh", type=float, default=0.10)
    parser.add_argument("--box_nms_iou_thresh", type=float, default=0.70)
    parser.add_argument(
        "--disable_pre_pose_nms",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep all parsed LocateAnything boxes until PoseHead processing.",
    )
    parser.add_argument(
        "--locate_generate_refhuman_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "During training, use LocateAnything generated boxes for RefHuman and "
            "GT boxes for regular pose datasets. This restores the last stable LLM-box recipe."
        ),
    )
    parser.add_argument("--w_locate_box_lm", type=float, default=0.0)
    parser.add_argument("--locate_lm_loss_every", type=int, default=1)
    parser.add_argument("--locate_lm_max_instances", type=int, default=20)
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
    parser.add_argument(
        "--visualize_min_gt_area_ratio",
        type=float,
        default=0.005,
        help="Skip training visualizations whose largest pose-annotated person is smaller than this image-area ratio.",
    )
    args = parser.parse_args()
    if args.backbone == "locatepose":
        args.backbone = "eagle"
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_float_list(value: str | list[float] | tuple[float, ...]) -> tuple[float, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(float(item) for item in value)
    text = str(value).strip()
    if not text:
        return ()
    return tuple(float(item.strip()) for item in text.split(",") if item.strip())


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    if "images" in batch:
        batch["images"] = batch["images"].to(device, non_blocking=True)
    batch["schema_ids"] = batch["schema_ids"].to(device, non_blocking=True)
    batch["task_ids"] = batch["task_ids"].to(device, non_blocking=True)
    return batch


def validate_pose_batch_contract(batch: dict) -> None:
    """Validate the canonical pose-target schema before loading large models."""
    targets = batch.get("targets")
    task_ids = batch.get("task_ids")
    if not isinstance(targets, list) or not torch.is_tensor(task_ids):
        raise TypeError("Pose batch requires a target list and tensor task_ids.")
    if len(targets) != int(task_ids.numel()):
        raise ValueError(
            f"Pose batch size mismatch: targets={len(targets)} task_ids={int(task_ids.numel())}."
        )

    union_count = len(UNION_KEYPOINTS)
    for sample_idx, target in enumerate(targets):
        if not isinstance(target, dict):
            raise TypeError(f"Pose target {sample_idx} must be a dict.")
        context = (
            f"sample={sample_idx} dataset={target.get('dataset', '')!r} "
            f"image_id={target.get('image_id', '')!r}"
        )
        boxes = target.get("boxes")
        keypoints = target.get("keypoints")
        keypoint_valid = target.get("keypoint_valid")
        if keypoint_valid is None and torch.is_tensor(target.get("keypoint_mask")):
            keypoint_valid = target["keypoint_mask"]
            target["keypoint_valid"] = keypoint_valid
        if not torch.is_tensor(boxes) or boxes.ndim != 2 or boxes.shape[-1] != 4:
            raise ValueError(f"Invalid boxes for {context}; expected [N,4].")
        if (
            not torch.is_tensor(keypoints)
            or keypoints.ndim != 3
            or keypoints.shape[1:] != (union_count, 3)
        ):
            raise ValueError(
                f"Invalid keypoints for {context}; expected [N,{union_count},3]."
            )
        if (
            not torch.is_tensor(keypoint_valid)
            or keypoint_valid.ndim != 2
            or keypoint_valid.shape[1] != union_count
        ):
            available = ", ".join(sorted(str(key) for key in target.keys()))
            raise ValueError(
                f"Invalid or missing keypoint_valid for {context}; expected "
                f"[N,{union_count}]. Available keys: {available}"
            )
        instance_count = int(boxes.shape[0])
        if int(keypoints.shape[0]) != instance_count or int(keypoint_valid.shape[0]) != instance_count:
            raise ValueError(
                f"Instance count mismatch for {context}: boxes={instance_count}, "
                f"keypoints={int(keypoints.shape[0])}, "
                f"keypoint_valid={int(keypoint_valid.shape[0])}."
            )
        visibility_valid = target.get("visibility_valid")
        if visibility_valid is not None and (
            not torch.is_tensor(visibility_valid)
            or tuple(visibility_valid.shape) != tuple(keypoint_valid.shape)
        ):
            raise ValueError(
                f"Invalid visibility_valid for {context}; expected "
                f"shape {tuple(keypoint_valid.shape)}."
            )
        ref_target = target.get("ref_target")
        task_id = int(task_ids[sample_idx].detach().cpu().item())
        if task_id == 1:
            if not torch.is_tensor(ref_target) or ref_target.numel() != 1:
                raise ValueError(f"REF_POSE target requires scalar ref_target for {context}.")
            ref_index = int(ref_target.detach().cpu().item())
            if not 0 <= ref_index < instance_count:
                raise ValueError(
                    f"REF_POSE ref_target={ref_index} is out of range for "
                    f"{instance_count} instances ({context})."
                )


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


def _spatially_sorted_instance_indices(
    boxes: torch.Tensor,
    indices: list[int],
    max_instances: int,
) -> list[int]:
    """Return a deterministic top-to-bottom, then left-to-right instance order."""
    limited = [int(idx) for idx in indices if 0 <= int(idx) < int(boxes.shape[0])]
    limited.sort(
        key=lambda idx: (
            float((boxes[idx, 1] + boxes[idx, 3]).item()) * 0.5,
            float((boxes[idx, 0] + boxes[idx, 2]).item()) * 0.5,
        )
    )
    return limited[: max(int(max_instances), 0)]


def build_lm_responses(batch: dict, max_instances: int = 10) -> list[str]:
    responses = []
    for sample_idx, target in enumerate(batch["targets"]):
        width = float(target["width"])
        height = float(target["height"])
        boxes = target["boxes"].detach().cpu()
        task_id = int(batch["task_ids"][sample_idx].detach().cpu().item())
        instance_indices = _spatially_sorted_instance_indices(
            boxes,
            list(range(int(boxes.shape[0]))),
            max_instances,
        )

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

        payload = {"people": people}
        responses.append(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    return responses


def _locate_coord_token(value: float, upper: float) -> str:
    scaled = 0 if upper <= 0 else int(round(max(0.0, min(float(value), float(upper))) / float(upper) * 1000.0))
    return "<" + f"{max(0, min(scaled, 1000)):03d}" + ">"


def build_locate_grounding_responses(
    batch: dict,
    max_instances: int = 20,
) -> list[str]:
    """Encode GT person boxes in LocateAnything's native coordinate-token form."""
    responses: list[str] = []
    for sample_idx, target in enumerate(batch["targets"]):
        width = float(target["width"])
        height = float(target["height"])
        boxes = target["boxes"].detach().cpu()
        task_id = int(batch["task_ids"][sample_idx].detach().cpu().item())
        if task_id == 1:
            ref_target = int(target["ref_target"].detach().cpu().item())
            instance_indices = [ref_target] if 0 <= ref_target < int(boxes.shape[0]) else []
        else:
            instance_indices = _spatially_sorted_instance_indices(
                boxes,
                list(range(int(boxes.shape[0]))),
                max_instances,
            )
        chunks: list[str] = []
        for person_idx in instance_indices:
            x1, y1, x2, y2 = boxes[person_idx].tolist()
            chunks.append(
                "<box>"
                + _locate_coord_token(x1 * width, width)
                + _locate_coord_token(y1 * height, height)
                + _locate_coord_token(x2 * width, width)
                + _locate_coord_token(y2 * height, height)
                + "</box>"
            )
        responses.append("".join(chunks) if chunks else "None")
    return responses


def _select_target_instances(
    target: dict[str, torch.Tensor],
    indices: list[int],
    *,
    task_id: int,
) -> dict[str, torch.Tensor]:
    index_tensor = torch.as_tensor(indices, dtype=torch.long)
    selected = dict(target)
    instance_fields = (
        "boxes",
        "loss_boxes",
        "loss_areas",
        "keypoints",
        "keypoint_valid",
        "visibility_valid",
        "box_context_scale",
        "box_jitter_scale",
        "box_jitter_shift",
    )
    for key in instance_fields:
        if key not in target:
            continue
        selected[key] = target[key][index_tensor].clone() if indices else target[key][:0].clone()
    selected["ref_target"] = torch.tensor(0 if task_id == 1 and indices else -1, dtype=torch.long)
    return selected


def expand_boxes_xyxy_per_box(
    boxes: torch.Tensor,
    scale: float | torch.Tensor,
) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes
    scale_tensor = torch.as_tensor(scale, device=boxes.device, dtype=boxes.dtype)
    if scale_tensor.ndim == 0:
        scale_tensor = scale_tensor.expand(boxes.shape[0])
    scale_tensor = scale_tensor.reshape(-1, 1).clamp(min=1e-4)
    if int(scale_tensor.shape[0]) != int(boxes.shape[0]):
        raise ValueError("Per-box context scale must match the number of boxes.")
    center = (boxes[:, :2] + boxes[:, 2:]) * 0.5
    wh = (boxes[:, 2:] - boxes[:, :2]).clamp(min=1e-4) * scale_tensor
    return torch.cat([center - wh * 0.5, center + wh * 0.5], dim=-1).clamp(0.0, 1.0)


def _context_scale_for_indices(
    target: dict[str, torch.Tensor],
    indices: list[int],
    count: int,
) -> torch.Tensor:
    if count <= 0:
        return torch.zeros(0, dtype=torch.float32)
    values = target.get("box_context_scale")
    if values is None or int(values.numel()) == 0:
        dataset_name = str(target.get("dataset", "")).lower()
        fallback = DATASET_BOX_CONTEXT_SCALE.get(dataset_name, 1.0)
        return torch.full((count,), float(fallback), dtype=torch.float32)
    if indices:
        index_tensor = torch.as_tensor(indices, dtype=torch.long)
        return values[index_tensor].clone().float()
    return torch.full((count,), float(values.flatten()[0].item()), dtype=torch.float32)


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

        selected = _select_target_instances(target, indices, task_id=task_id)
        selected_targets.append(selected)
        condition_boxes = selected["boxes"]
        if condition_boxes.numel() > 0:
            scale_jitter = selected.get(
                "box_jitter_scale",
                torch.full((condition_boxes.shape[0],), float(box_jitter_scale)),
            )
            shift_jitter = selected.get(
                "box_jitter_shift",
                torch.full((condition_boxes.shape[0],), float(box_jitter_shift)),
            )
            condition_boxes = jitter_boxes_xyxy(
                condition_boxes,
                scale_jitter=scale_jitter,
                shift_jitter=shift_jitter,
            )
            context_scale = selected.get(
                "box_context_scale",
                torch.ones(condition_boxes.shape[0], dtype=condition_boxes.dtype),
            )
            condition_boxes = expand_boxes_xyxy_per_box(condition_boxes, context_scale)
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


def prepare_person_query_conditioning(
    targets: list[dict[str, torch.Tensor]],
    task_ids: torch.Tensor,
    device: torch.device,
    max_instances: int,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, torch.Tensor]]]:
    """Keep every GT person for set prediction while supplying no box prompt.

    The model replaces the returned placeholder with its learned global person
    queries.  RefHuman keeps all people in the image (not just the referred
    person), because its supervision is a candidate-selection loss.
    """
    selected_targets: list[dict[str, torch.Tensor]] = []
    limit = max(int(max_instances), 1)
    for sample_idx, target in enumerate(targets):
        count = int(target["boxes"].shape[0])
        indices = list(range(min(count, limit)))
        task_id = int(task_ids[sample_idx].detach().cpu().item())
        if task_id == 1 and count > limit:
            ref_target = int(target["ref_target"].detach().cpu().item())
            if 0 <= ref_target < count and ref_target not in indices:
                indices[-1] = ref_target
        selected = _select_target_instances(target, indices, task_id=0)
        ref_target = int(target["ref_target"].detach().cpu().item()) if task_id == 1 else -1
        selected["ref_target"] = torch.tensor(
            indices.index(ref_target) if ref_target in indices else -1,
            dtype=torch.long,
        )
        selected_targets.append(selected)

    batch_size = len(selected_targets)
    placeholder_boxes = torch.zeros(batch_size, 1, 4, dtype=torch.float32, device=device)
    placeholder_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
    return placeholder_boxes, placeholder_mask, selected_targets


def jitter_boxes_xyxy(
    boxes: torch.Tensor,
    scale_jitter: float | torch.Tensor = 0.0,
    shift_jitter: float | torch.Tensor = 0.0,
) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes
    boxes = boxes.float()
    xy1 = boxes[:, :2]
    xy2 = boxes[:, 2:]
    wh = (xy2 - xy1).clamp(min=1e-4)
    center = (xy1 + xy2) * 0.5

    scale_jitter_tensor = torch.as_tensor(
        scale_jitter, device=boxes.device, dtype=boxes.dtype
    )
    shift_jitter_tensor = torch.as_tensor(
        shift_jitter, device=boxes.device, dtype=boxes.dtype
    )
    if scale_jitter_tensor.ndim == 0:
        scale_jitter_tensor = scale_jitter_tensor.expand(boxes.shape[0])
    if shift_jitter_tensor.ndim == 0:
        shift_jitter_tensor = shift_jitter_tensor.expand(boxes.shape[0])
    scale_jitter_tensor = scale_jitter_tensor.reshape(-1, 1).clamp(min=0.0)
    shift_jitter_tensor = shift_jitter_tensor.reshape(-1, 1).clamp(min=0.0)
    if (
        int(scale_jitter_tensor.shape[0]) != int(boxes.shape[0])
        or int(shift_jitter_tensor.shape[0]) != int(boxes.shape[0])
    ):
        raise ValueError("Per-box jitter tensors must match the number of boxes.")

    center = center + (torch.rand_like(center) * 2.0 - 1.0) * wh * shift_jitter_tensor
    random_scale = 1.0 + (
        torch.rand(boxes.shape[0], 1, device=boxes.device, dtype=boxes.dtype) * 2.0 - 1.0
    ) * scale_jitter_tensor
    wh = wh * random_scale.clamp(min=0.5)
    jittered = torch.cat([center - wh * 0.5, center + wh * 0.5], dim=-1)
    return jittered.clamp(0.0, 1.0)


def prepare_box_denoising(
    targets: list[dict[str, torch.Tensor]],
    device: torch.device,
    *,
    max_queries: int = 96,
    max_groups: int = 4,
    positive_noise: float = 0.40,
    negative_noise: float = 1.00,
    image_size: int = 800,
) -> dict[str, torch.Tensor] | None:
    """Build DN-DETR/DINO-style positive and negative box queries.

    Positive queries reconstruct their clean GT box. Negative queries are moved
    far enough from the source person to supervise background objectness only.
    """
    max_queries = max(int(max_queries), 0)
    max_groups = max(int(max_groups), 1)
    if max_queries < 2:
        return None
    minimum_center_shift = 2.0 / max(float(image_size), 1.0)
    minimum_box_size = 4.0 / max(float(image_size), 1.0)

    per_sample: list[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ] = []
    for target in targets:
        clean_boxes = target["boxes"].to(device=device, dtype=torch.float32)
        clean_source_indices = torch.arange(
            clean_boxes.shape[0], device=device, dtype=torch.long
        )
        if clean_boxes.numel() == 0:
            per_sample.append(
                (
                    clean_boxes.new_zeros((0, 4)),
                    clean_boxes.new_zeros((0,)),
                    clean_boxes.new_zeros((0, 4)),
                    torch.zeros(0, device=device, dtype=torch.long),
                    torch.zeros(0, device=device, dtype=torch.long),
                )
            )
            continue
        matched = target.get("matched_gt_indices")
        if isinstance(matched, torch.Tensor):
            positive_source = matched.to(device=device)[: clean_boxes.shape[0]].ge(0)
            clean_boxes = clean_boxes[positive_source]
            clean_source_indices = clean_source_indices[positive_source]
        wh = (clean_boxes[:, 2:] - clean_boxes[:, :2]).clamp(min=minimum_box_size)
        valid = (wh[:, 0] > 0.0) & (wh[:, 1] > 0.0)
        clean_boxes = clean_boxes[valid]
        clean_source_indices = clean_source_indices[valid]
        if clean_boxes.numel() == 0:
            per_sample.append(
                (
                    clean_boxes.new_zeros((0, 4)),
                    clean_boxes.new_zeros((0,)),
                    clean_boxes.new_zeros((0, 4)),
                    torch.zeros(0, device=device, dtype=torch.long),
                    torch.zeros(0, device=device, dtype=torch.long),
                )
            )
            continue
        max_gt = max(max_queries // 2, 1)
        clean_boxes = clean_boxes[:max_gt]
        clean_source_indices = clean_source_indices[:max_gt]
        n = int(clean_boxes.shape[0])
        groups = min(max_groups, max(max_queries // max(2 * n, 1), 1))
        noisy_boxes: list[torch.Tensor] = []
        labels: list[float] = []
        target_boxes: list[torch.Tensor] = []
        group_ids: list[int] = []
        source_indices: list[int] = []

        for group_idx in range(groups):
            for clean, source_idx_tensor in zip(clean_boxes, clean_source_indices):
                source_idx = int(source_idx_tensor.item())
                xy1, xy2 = clean[:2], clean[2:]
                box_wh = (xy2 - xy1).clamp(min=minimum_box_size)
                center = (xy1 + xy2) * 0.5

                center_scale = max(float(positive_noise) * 0.5, 0.0)
                center_shift = (torch.rand(2, device=device) * 2.0 - 1.0) * box_wh * center_scale
                center_shift = torch.where(
                    center_shift.abs() < minimum_center_shift,
                    center_shift.sign().masked_fill(center_shift.eq(0), 1.0) * minimum_center_shift,
                    center_shift,
                )
                size_scale = 1.0 + (torch.rand(2, device=device) * 2.0 - 1.0) * float(positive_noise)
                positive_wh = (box_wh * size_scale.clamp(min=0.5)).clamp(min=minimum_box_size)
                positive_center = center + center_shift
                positive_box = torch.cat(
                    [positive_center - positive_wh * 0.5, positive_center + positive_wh * 0.5]
                ).clamp(0.0, 1.0)
                noisy_boxes.append(positive_box)
                labels.append(1.0)
                target_boxes.append(clean)
                group_ids.append(group_idx)
                source_indices.append(source_idx)

                negative_box = positive_box
                best_negative_iou = float("inf")
                for _ in range(10):
                    direction = torch.where(
                        torch.rand(2, device=device) > 0.5,
                        torch.ones(2, device=device),
                        -torch.ones(2, device=device),
                    )
                    distance = 0.20 + 0.30 * torch.rand(2, device=device)
                    negative_center = center + direction * distance * box_wh * max(float(negative_noise), 0.0)
                    negative_scale = 1.0 + (
                        torch.rand(2, device=device) * 2.0 - 1.0
                    ) * max(float(negative_noise), 0.0)
                    negative_wh = (box_wh * negative_scale.clamp(min=0.4, max=2.0)).clamp(
                        min=minimum_box_size
                    )
                    candidate = torch.cat(
                        [negative_center - negative_wh * 0.5, negative_center + negative_wh * 0.5]
                    ).clamp(0.0, 1.0)
                    candidate_iou = float(
                        box_iou_xyxy(candidate.view(1, 4), clean.view(1, 4))[0, 0]
                    )
                    if candidate_iou < best_negative_iou:
                        best_negative_iou = candidate_iou
                        negative_box = candidate
                    if candidate_iou < 0.30:
                        break
                noisy_boxes.append(negative_box)
                labels.append(0.0)
                target_boxes.append(clean)
                group_ids.append(group_idx)
                source_indices.append(source_idx)

        sample_boxes = torch.stack(noisy_boxes[:max_queries], dim=0)
        sample_labels = torch.tensor(labels[:max_queries], device=device, dtype=torch.float32)
        sample_targets = torch.stack(target_boxes[:max_queries], dim=0)
        sample_groups = torch.tensor(group_ids[:max_queries], device=device, dtype=torch.long)
        sample_sources = torch.tensor(
            source_indices[:max_queries], device=device, dtype=torch.long
        )
        per_sample.append(
            (sample_boxes, sample_labels, sample_targets, sample_groups, sample_sources)
        )

    padded_count = max([int(item[0].shape[0]) for item in per_sample] + [0])
    if padded_count <= 0:
        return None
    boxes = torch.zeros(len(per_sample), padded_count, 4, device=device, dtype=torch.float32)
    mask = torch.zeros(len(per_sample), padded_count, device=device, dtype=torch.bool)
    labels = torch.zeros(len(per_sample), padded_count, device=device, dtype=torch.float32)
    target_boxes = torch.zeros(len(per_sample), padded_count, 4, device=device, dtype=torch.float32)
    group_ids = torch.full(
        (len(per_sample), padded_count),
        -1,
        device=device,
        dtype=torch.long,
    )
    source_indices = torch.full_like(group_ids, -1)
    for batch_idx, (
        sample_boxes,
        sample_labels,
        sample_targets,
        sample_groups,
        sample_sources,
    ) in enumerate(per_sample):
        count = int(sample_boxes.shape[0])
        if count <= 0:
            continue
        boxes[batch_idx, :count] = sample_boxes
        labels[batch_idx, :count] = sample_labels
        target_boxes[batch_idx, :count] = sample_targets
        group_ids[batch_idx, :count] = sample_groups
        source_indices[batch_idx, :count] = sample_sources
        mask[batch_idx, :count] = True
    return {
        "dn_boxes": boxes,
        "dn_box_mask": mask,
        "dn_labels": labels,
        "dn_target_boxes": target_boxes,
        "dn_group_ids": group_ids,
        "dn_source_indices": source_indices,
    }


def prepare_keypoint_denoising(
    targets: list[dict[str, torch.Tensor]],
    device: torch.device,
    *,
    max_queries: int = 16,
    max_groups: int = 2,
    positive_ks_min: float = 0.5,
    positive_ks_max: float = 1.0,
    negative_ks_min: float = 0.1,
    negative_ks_max: float = 0.5,
    image_size: int = 800,
) -> dict[str, torch.Tensor] | None:
    """Build box-conditioned DETRPose-style positive/negative pose queries.

    A query here is one complete person skeleton, not one joint token. Positive
    skeletons reconstruct clean GT joints. Negative skeletons receive only
    confidence/quality supervision in the loss. In external-box modes, only
    successfully matched proposals receive pose supervision.
    """
    max_queries = max(int(max_queries), 0)
    max_groups = max(int(max_groups), 1)
    if max_queries < 2:
        return None
    if not (0.0 < positive_ks_min <= positive_ks_max <= 1.0):
        raise ValueError("Positive keypoint-DN KS range must lie in (0, 1].")
    if not (0.0 < negative_ks_min <= negative_ks_max <= 1.0):
        raise ValueError("Negative keypoint-DN KS range must lie in (0, 1].")

    union_count = len(UNION_KEYPOINTS)
    sigmas = UNION_SIGMAS.to(device=device, dtype=torch.float32).view(1, -1)
    variances = (2.0 * sigmas).square()
    pixel_scale = max(float(image_size), 1.0)
    eps = torch.finfo(torch.float32).eps
    per_sample: list[dict[str, torch.Tensor]] = []

    for target in targets:
        clean_keypoints = target["keypoints"].to(device=device, dtype=torch.float32)
        clean_valid = target["keypoint_valid"].to(device=device).bool()
        clean_boxes = target.get("loss_boxes", target["boxes"]).to(
            device=device, dtype=torch.float32
        )
        default_areas = (
            (clean_boxes[:, 2] - clean_boxes[:, 0]).clamp(min=0.0)
            * (clean_boxes[:, 3] - clean_boxes[:, 1]).clamp(min=0.0)
        )
        clean_areas = target.get("loss_areas", default_areas).to(
            device=device, dtype=torch.float32
        )
        candidate = clean_valid.any(dim=-1)
        matched = target.get("matched_gt_indices")
        if isinstance(matched, torch.Tensor):
            candidate = candidate & matched.to(device=device).ge(0)
        indices = torch.nonzero(candidate, as_tuple=False).flatten()
        max_people = max(max_queries // 2, 1)
        indices = indices[:max_people]
        n = int(indices.numel())
        if n <= 0:
            per_sample.append({
                "noisy": torch.zeros(0, union_count, 2, device=device),
                "labels": torch.zeros(0, device=device),
                "targets": torch.zeros(0, union_count, 3, device=device),
                "valid": torch.zeros(0, union_count, device=device, dtype=torch.bool),
                "boxes": torch.zeros(0, 4, device=device),
                "areas": torch.zeros(0, device=device),
                "sources": torch.zeros(0, device=device, dtype=torch.long),
                "groups": torch.zeros(0, device=device, dtype=torch.long),
            })
            continue

        groups = min(max_groups, max(max_queries // max(2 * n, 1), 1))
        sample_noisy: list[torch.Tensor] = []
        sample_labels: list[float] = []
        sample_targets: list[torch.Tensor] = []
        sample_valid: list[torch.Tensor] = []
        sample_boxes: list[torch.Tensor] = []
        sample_areas: list[torch.Tensor] = []
        sample_sources: list[int] = []
        sample_groups: list[int] = []

        for group_idx in range(groups):
            for label, ks_min, ks_max in (
                (1.0, positive_ks_min, positive_ks_max),
                (0.0, negative_ks_min, negative_ks_max),
            ):
                for source_idx_tensor in indices:
                    source_idx = int(source_idx_tensor.item())
                    target_keypoints = clean_keypoints[source_idx]
                    valid = clean_valid[source_idx]
                    area_pixels = clean_areas[source_idx].clamp(min=eps) * pixel_scale * pixel_scale
                    ks = torch.empty(union_count, device=device).uniform_(
                        float(ks_min), float(ks_max)
                    ).clamp(min=eps, max=1.0)
                    radius_pixels = torch.sqrt(
                        -2.0 * area_pixels * variances[0] * torch.log(ks)
                    )
                    theta = torch.rand(union_count, device=device) * (2.0 * math.pi)
                    direction = torch.stack([theta.cos(), theta.sin()], dim=-1)
                    noisy_xy = target_keypoints[..., :2] + (
                        radius_pixels[:, None] / pixel_scale
                    ) * direction
                    noisy_xy = torch.where(
                        valid[:, None], noisy_xy, target_keypoints[..., :2]
                    ).clamp(0.0, 1.0)
                    sample_noisy.append(noisy_xy)
                    sample_labels.append(label)
                    sample_targets.append(target_keypoints)
                    sample_valid.append(valid)
                    sample_boxes.append(clean_boxes[source_idx])
                    sample_areas.append(clean_areas[source_idx])
                    sample_sources.append(source_idx)
                    sample_groups.append(group_idx)

        count = min(len(sample_noisy), max_queries)
        per_sample.append({
            "noisy": torch.stack(sample_noisy[:count]),
            "labels": torch.tensor(sample_labels[:count], device=device),
            "targets": torch.stack(sample_targets[:count]),
            "valid": torch.stack(sample_valid[:count]),
            "boxes": torch.stack(sample_boxes[:count]),
            "areas": torch.stack(sample_areas[:count]),
            "sources": torch.tensor(sample_sources[:count], device=device, dtype=torch.long),
            "groups": torch.tensor(sample_groups[:count], device=device, dtype=torch.long),
        })

    padded_count = max([int(sample["noisy"].shape[0]) for sample in per_sample] + [0])
    if padded_count <= 0:
        return None
    batch_size = len(per_sample)
    noisy = torch.zeros(batch_size, padded_count, union_count, 2, device=device)
    mask = torch.zeros(batch_size, padded_count, device=device, dtype=torch.bool)
    labels = torch.zeros(batch_size, padded_count, device=device)
    target_keypoints = torch.zeros(batch_size, padded_count, union_count, 3, device=device)
    target_valid = torch.zeros(
        batch_size, padded_count, union_count, device=device, dtype=torch.bool
    )
    target_boxes = torch.zeros(batch_size, padded_count, 4, device=device)
    target_areas = torch.zeros(batch_size, padded_count, device=device)
    source_indices = torch.full(
        (batch_size, padded_count), -1, device=device, dtype=torch.long
    )
    group_ids = torch.full(
        (batch_size, padded_count), -1, device=device, dtype=torch.long
    )
    for batch_idx, sample in enumerate(per_sample):
        count = int(sample["noisy"].shape[0])
        if count <= 0:
            continue
        noisy[batch_idx, :count] = sample["noisy"]
        labels[batch_idx, :count] = sample["labels"]
        target_keypoints[batch_idx, :count] = sample["targets"]
        target_valid[batch_idx, :count] = sample["valid"]
        target_boxes[batch_idx, :count] = sample["boxes"]
        target_areas[batch_idx, :count] = sample["areas"]
        source_indices[batch_idx, :count] = sample["sources"]
        group_ids[batch_idx, :count] = sample["groups"]
        mask[batch_idx, :count] = True
    return {
        "keypoint_dn_noisy_keypoints": noisy,
        "keypoint_dn_mask": mask,
        "keypoint_dn_labels": labels,
        "keypoint_dn_target_keypoints": target_keypoints,
        "keypoint_dn_target_valid": target_valid,
        "keypoint_dn_target_boxes": target_boxes,
        "keypoint_dn_target_areas": target_areas,
        "keypoint_dn_source_indices": source_indices,
        "keypoint_dn_group_ids": group_ids,
    }


def pair_keypoint_denoising_with_box_denoising(
    box_dn: dict[str, torch.Tensor] | None,
    keypoint_dn: dict[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor] | None:
    """Attach each pose-DN skeleton to its matching positive box-DN query.

    Pairing is exact on ``(batch, source person, DN group)``. Both the positive
    and negative skeleton for a person use the same positive noisy-box query;
    negative skeletons remain quality-only examples in the pose-DN loss. Any
    skeleton without a positive box-DN partner is masked out instead of falling
    back to a main prediction query.
    """

    if box_dn is None or keypoint_dn is None:
        return None
    required_box = (
        "dn_box_mask",
        "dn_labels",
        "dn_group_ids",
        "dn_source_indices",
    )
    required_pose = (
        "keypoint_dn_mask",
        "keypoint_dn_source_indices",
        "keypoint_dn_group_ids",
    )
    if not all(torch.is_tensor(box_dn.get(key)) for key in required_box):
        return None
    if not all(torch.is_tensor(keypoint_dn.get(key)) for key in required_pose):
        return None

    box_mask = box_dn["dn_box_mask"].bool()
    box_positive = box_dn["dn_labels"].gt(0.5)
    box_groups = box_dn["dn_group_ids"].long()
    box_sources = box_dn["dn_source_indices"].long()
    pose_mask = keypoint_dn["keypoint_dn_mask"].bool()
    pose_groups = keypoint_dn["keypoint_dn_group_ids"].long()
    pose_sources = keypoint_dn["keypoint_dn_source_indices"].long()
    if box_mask.shape[0] != pose_mask.shape[0]:
        raise ValueError("box-DN and pose-DN batch sizes must match for pairing.")

    box_query_indices = torch.full_like(pose_sources, -1)
    for batch_idx in range(int(pose_mask.shape[0])):
        for pose_idx in torch.nonzero(pose_mask[batch_idx], as_tuple=False).flatten():
            source = pose_sources[batch_idx, pose_idx]
            group = pose_groups[batch_idx, pose_idx]
            candidates = (
                box_mask[batch_idx]
                & box_positive[batch_idx]
                & box_sources[batch_idx].eq(source)
                & box_groups[batch_idx].eq(group)
            )
            candidate_indices = torch.nonzero(candidates, as_tuple=False).flatten()
            if candidate_indices.numel() > 0:
                box_query_indices[batch_idx, pose_idx] = candidate_indices[0]

    paired_mask = pose_mask & box_query_indices.ge(0)
    if not paired_mask.any():
        return None
    paired = dict(keypoint_dn)
    paired["keypoint_dn_mask"] = paired_mask
    paired["keypoint_dn_box_query_indices"] = box_query_indices
    return paired


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


def nms_box_indices_xyxy(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_thresh: float,
    max_boxes: int,
) -> list[int]:
    """Score-ordered NMS that returns indices into the original box tensor."""
    if boxes.numel() == 0 or max(int(max_boxes), 0) == 0:
        return []
    boxes_cpu = boxes.detach().cpu().float().clamp(0.0, 1.0)
    scores_cpu = scores.detach().cpu().float().reshape(-1)
    order = torch.argsort(scores_cpu, descending=True).tolist()
    keep: list[int] = []
    for idx in order:
        if len(keep) >= int(max_boxes):
            break
        if keep:
            overlaps = box_iou_xyxy(
                boxes_cpu[idx : idx + 1],
                boxes_cpu[torch.as_tensor(keep, dtype=torch.long)],
            ).flatten()
            if bool((overlaps > float(iou_thresh)).any().item()):
                continue
        keep.append(int(idx))
    return keep


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


def hungarian_match_boxes(
    pred_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_thresh: float,
) -> list[tuple[int, int, float]]:
    """Globally match predicted and GT boxes, then reject very-low-IoU pairs."""
    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        return []
    pred = pred_boxes.detach().cpu().float()
    gt = gt_boxes.detach().cpu().float()
    ious = box_iou_xyxy(pred, gt)
    pred_center = (pred[:, :2] + pred[:, 2:]) * 0.5
    gt_center = (gt[:, :2] + gt[:, 2:]) * 0.5
    pred_wh = (pred[:, 2:] - pred[:, :2]).clamp(min=1e-4).log()
    gt_wh = (gt[:, 2:] - gt[:, :2]).clamp(min=1e-4).log()
    center_cost = torch.cdist(pred_center, gt_center, p=1)
    size_cost = torch.cdist(pred_wh, gt_wh, p=1)
    cost = (1.0 - ious) + 0.5 * center_cost + 0.25 * size_cost
    if not bool(torch.isfinite(cost).all().item()) or _scipy_linear_sum_assignment is None:
        return greedy_match_boxes(pred, gt, iou_thresh=iou_thresh)
    pred_indices, gt_indices = _scipy_linear_sum_assignment(cost.numpy())
    matches: list[tuple[int, int, float]] = []
    for pred_idx, gt_idx in zip(pred_indices.tolist(), gt_indices.tolist()):
        iou = float(ious[pred_idx, gt_idx].item())
        if iou >= float(iou_thresh):
            matches.append((int(pred_idx), int(gt_idx), iou))
    matches.sort(key=lambda item: item[0])
    return matches


def align_target_to_predictions(
    target: dict[str, Any],
    pred_boxes: torch.Tensor,
    gt_indices: list[int],
    matches: list[tuple[int, int, float]],
    *,
    task_id: int,
) -> dict[str, Any]:
    """Build a query-aligned target; unmatched queries carry no pose supervision."""
    selected = dict(target)
    count = int(pred_boxes.shape[0])
    target_device = target["boxes"].device
    pred_boxes = pred_boxes.detach().to(
        device=target_device,
        dtype=target["boxes"].dtype,
    )
    pred_areas = (
        (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=0.0)
        * (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=0.0)
    ).clamp(min=1e-8)

    selected["boxes"] = pred_boxes.clone()
    selected["loss_boxes"] = pred_boxes.clone()
    selected["loss_areas"] = pred_areas.clone()
    selected.pop("box_jitter_scale", None)
    selected.pop("box_jitter_shift", None)
    for key in ("keypoints", "keypoint_valid", "visibility_valid"):
        template = target[key]
        selected[key] = template.new_zeros((count, *template.shape[1:]))

    fallback_context = _context_scale_for_indices(target, [], count)
    selected["box_context_scale"] = fallback_context.to(
        device=target_device,
        dtype=torch.float32,
    )
    matched_gt_indices = torch.full(
        (count,), -1, device=target_device, dtype=torch.long
    )
    match_ious = torch.zeros(count, device=target_device, dtype=torch.float32)

    instance_fields = (
        "boxes",
        "loss_boxes",
        "loss_areas",
        "keypoints",
        "keypoint_valid",
        "visibility_valid",
        "box_context_scale",
    )
    ref_query = -1
    for pred_idx, local_gt_idx, iou in matches:
        if not (0 <= local_gt_idx < len(gt_indices) and 0 <= pred_idx < count):
            continue
        original_gt_idx = int(gt_indices[local_gt_idx])
        for key in instance_fields:
            if key in target:
                selected[key][pred_idx] = target[key][original_gt_idx]
        matched_gt_indices[pred_idx] = original_gt_idx
        match_ious[pred_idx] = float(iou)
        if task_id == 1:
            original_ref_target = int(target["ref_target"].detach().cpu().item())
            if original_gt_idx == original_ref_target:
                ref_query = pred_idx

    selected["matched_gt_indices"] = matched_gt_indices
    selected["match_ious"] = match_ious
    selected["ref_target"] = torch.tensor(
        ref_query, device=target_device, dtype=torch.long
    )
    return selected


def align_targets_to_person_queries(
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, Any]],
    task_ids: torch.Tensor,
) -> list[dict[str, Any]]:
    """Hungarian-align global person queries to all GT people in each image.

    Matching is intentionally accepted at any IoU during training.  Early in
    training the learned references are only a spatial grid; rejecting those
    matches would remove the box regression signal needed to turn them into a
    detector.  Unmatched queries remain explicit background negatives.
    """
    predicted = outputs["pred_boxes"].detach()
    query_mask = outputs.get("box_mask")
    aligned: list[dict[str, Any]] = []
    for batch_idx, target in enumerate(targets):
        if query_mask is None:
            valid_queries = torch.ones(
                predicted.shape[1], device=predicted.device, dtype=torch.bool
            )
        else:
            valid_queries = query_mask[batch_idx].detach().bool()
        query_indices = torch.nonzero(valid_queries, as_tuple=False).flatten()
        sample_predictions = predicted[batch_idx, query_indices]
        gt_boxes = target.get("loss_boxes", target["boxes"])
        gt_indices = list(range(int(gt_boxes.shape[0])))
        local_matches = hungarian_match_boxes(
            sample_predictions,
            gt_boxes,
            iou_thresh=0.0,
        )
        matches = [
            (int(query_indices[pred_idx].item()), gt_idx, iou)
            for pred_idx, gt_idx, iou in local_matches
        ]
        aligned.append(
            align_target_to_predictions(
                target,
                predicted[batch_idx],
                gt_indices,
                matches,
                task_id=int(task_ids[batch_idx].detach().cpu().item()),
            )
        )
    return aligned


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

        if matches:
            matched_original_indices = gt_index_tensor[matched_gt_index_tensor].tolist()
            selected = _select_target_instances(
                target, matched_original_indices, task_id=task_id
            )
            matched_pred_boxes = pred_boxes[pred_index_tensor].clone()
            matched_context = selected.get(
                "box_context_scale",
                torch.ones(matched_pred_boxes.shape[0], dtype=matched_pred_boxes.dtype),
            )
            matched_pred_boxes = expand_boxes_xyxy_per_box(
                matched_pred_boxes, matched_context
            )
            if keep_unmatched_predictions:
                matched_pred_set = set(pred_match_indices)
                unmatched_pred_indices = [
                    idx
                    for idx in range(int(pred_boxes.shape[0]))
                    if idx not in matched_pred_set
                ]
                if unmatched_pred_indices:
                    unmatched_index_tensor = torch.as_tensor(
                        unmatched_pred_indices, dtype=torch.long
                    )
                    unmatched_boxes = pred_boxes[unmatched_index_tensor].clone()
                    unmatched_context = _context_scale_for_indices(
                        target, [], int(unmatched_boxes.shape[0])
                    )
                    unmatched_boxes = expand_boxes_xyxy_per_box(
                        unmatched_boxes, unmatched_context
                    )
                    condition_boxes = torch.cat(
                        [matched_pred_boxes, unmatched_boxes], dim=0
                    )
                else:
                    condition_boxes = matched_pred_boxes
            else:
                condition_boxes = matched_pred_boxes
            selected_condition_boxes.append(condition_boxes[:max_instances].clone())
        else:
            selected = _select_target_instances(target, [], task_id=task_id)
            if keep_unmatched_predictions and pred_boxes.numel() > 0:
                unmatched_boxes = pred_boxes[:max_instances].clone()
                unmatched_context = _context_scale_for_indices(
                    target, [], int(unmatched_boxes.shape[0])
                )
                selected_condition_boxes.append(
                    expand_boxes_xyxy_per_box(unmatched_boxes, unmatched_context)
                )
            else:
                selected_condition_boxes.append(gt_boxes_all[:0].clone())
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


def _extract_ref_description_from_prompt(prompt: str) -> str:
    quoted = re.findall(r'"([^"]+)"', str(prompt))
    if quoted:
        return quoted[0].strip()
    marker = "description:"
    lowered = str(prompt).lower()
    idx = lowered.find(marker)
    if idx >= 0:
        fragment = str(prompt)[idx + len(marker) :].strip()
        fragment = fragment.split(".", 1)[0].strip()
        if fragment:
            return fragment
    return "person"


REFHUMAN_FALLBACK_MARKER = "<refhuman_all_person_fallback>"


def build_refhuman_locate_prompt(ref_text: str) -> str:
    ref_text = str(ref_text).strip() or "person"
    return f'Locate the person that matches the following description: "{ref_text}".'


def build_refhuman_fallback_prompt(ref_text: str) -> str:
    ref_text = str(ref_text).strip() or "person"
    return (
        "Locate all visible people in the image and return every person box. "
        f'The downstream target description is: "{ref_text}".'
    )


def build_locate_generation_prompts(batch: dict) -> list[str]:
    prompts: list[str] = []
    ref_texts = batch.get("ref_texts") or [""] * len(batch.get("prompts", []))
    for sample_idx, source_prompt in enumerate(batch.get("prompts", [])):
        task_id = int(batch["task_ids"][sample_idx].detach().cpu().item())
        if task_id == 1:
            ref_text = str(ref_texts[sample_idx]).strip() if sample_idx < len(ref_texts) else ""
            if not ref_text:
                ref_text = _extract_ref_description_from_prompt(str(source_prompt))
            prompts.append(build_refhuman_locate_prompt(ref_text))
        else:
            prompts.append("Locate all the instances that match the following description: person.")
    return prompts


def parse_locate_bbox_response(text: str, max_instances: int) -> torch.Tensor:
    raw_boxes: list[list[float]] = []
    # LocateAnything normally decodes boxes as
    #   <box><010><020><900><950></box>
    # but some tokenizer/version combinations may introduce spaces or bare
    # comma-separated numbers inside the box block. Parse by block first, then
    # accept exactly four coordinates; two-coordinate point blocks are ignored.
    box_block_pattern = re.compile(r"<\s*box\s*>\s*(.*?)\s*<\s*/\s*box\s*>", re.DOTALL)
    coord_token_pattern = re.compile(r"<\s*([0-9]{1,4})\s*>")
    bare_number_pattern = re.compile(r"(?<![A-Za-z])[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
    for block_match in box_block_pattern.finditer(str(text)):
        block = block_match.group(1)
        token_values = coord_token_pattern.findall(block)
        values = token_values if len(token_values) == 4 else bare_number_pattern.findall(block)
        if len(values) != 4:
            continue
        raw_boxes.append([float(value) for value in values[:4]])
        if len(raw_boxes) >= max(int(max_instances), 0):
            break
    boxes: list[list[float]] = []
    for raw_box in raw_boxes:
        normalized = _normalize_generated_box(raw_box, width=1000.0, height=1000.0)
        if normalized is not None:
            boxes.append(normalized)
    if not boxes:
        return torch.zeros(0, 4, dtype=torch.float32)
    return torch.tensor(boxes, dtype=torch.float32).clamp_(0.0, 1.0)


def parse_locate_boxes_for_task(
    response: str,
    *,
    task_id: int,
    max_instances: int,
    nms_iou_thresh: float,
    disable_pre_pose_nms: bool,
) -> torch.Tensor:
    """Parse one Locate response with the exact train/eval/infer filtering policy."""
    is_refhuman_fallback = int(task_id) == 1 and REFHUMAN_FALLBACK_MARKER in str(response)
    boxes = parse_locate_bbox_response(response, max_instances=max_instances)
    if int(task_id) == 1 and not is_refhuman_fallback:
        # The direct RefHuman grounding contract is one expression -> one box.
        # If a model emits extras, generation order is its grounding decision.
        return boxes[:1]
    if disable_pre_pose_nms:
        boxes = boxes[:max_instances]
    else:
        boxes = nms_boxes_xyxy(
            boxes,
            iou_thresh=nms_iou_thresh,
            max_boxes=max_instances,
        )
    return boxes


def locate_boxes_abs_from_responses(
    responses: list[str],
    batch: dict,
    *,
    max_instances: int,
    nms_iou_thresh: float,
    disable_pre_pose_nms: bool,
) -> list[list[list[float]]]:
    """Return raw Locate boxes in image pixels, before PoseHead context expansion."""
    output: list[list[list[float]]] = []
    for sample_idx, target in enumerate(batch["targets"]):
        task_id = int(batch["task_ids"][sample_idx].detach().cpu().item())
        response = responses[sample_idx] if sample_idx < len(responses) else ""
        boxes = parse_locate_boxes_for_task(
            response,
            task_id=task_id,
            max_instances=max_instances,
            nms_iou_thresh=nms_iou_thresh,
            disable_pre_pose_nms=disable_pre_pose_nms,
        )
        width = float(target["width"])
        height = float(target["height"])
        output.append(
            [
                [
                    float(box[0]) * width,
                    float(box[1]) * height,
                    float(box[2]) * width,
                    float(box[3]) * height,
                ]
                for box in boxes.detach().cpu().tolist()
            ]
        )
    return output


def _normalize_locate_generate_output(response: object) -> str:
    if isinstance(response, tuple) and response:
        response = response[0]
    if isinstance(response, list):
        return str(response[0]).strip() if response else ""
    return str(response).strip()


def locate_generation_token_budget(
    requested_max_new_tokens: int,
    max_instances: int,
    *,
    task_id: int,
) -> int:
    """Cap decoding to the number of box tokens the task can actually use.

    A Locate box uses six structural/coordinate tokens. The extra allowance
    covers termination and an occasional hybrid-mode recovery token without
    permitting a malformed response to decode to the full model context.
    """

    expected_instances = 1 if int(task_id) == 1 else max(int(max_instances), 1)
    task_budget = 8 * expected_instances + 16
    return max(8, min(max(int(requested_max_new_tokens), 1), task_budget))


def generate_locate_bbox_responses(
    training_model: torch.nn.Module,
    processor,
    batch: dict,
    device: torch.device,
    *,
    max_instances: int,
    max_new_tokens: int,
    generation_mode: str,
    image_token_limit: int | None = None,
) -> list[str]:
    module = unwrap_training_model(training_model)
    if module.backbone_name != "eagle" or module.backbone_model is None:
        raise ValueError("--box_source=locate_generate currently requires --backbone eagle/locatepose.")
    if processor is None:
        raise ValueError("LocateAnything processor is required for --box_source=locate_generate.")
    locate_model = get_eagle_base_model(module.backbone_model)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        tokenizer = getattr(locate_model, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("LocateAnything tokenizer is required for --box_source=locate_generate.")
    prompts = build_locate_generation_prompts(batch)
    was_training = bool(locate_model.training)
    locate_model.eval()
    responses: list[str] = []
    try:
        with torch.inference_mode():
            vision_images = batch.get("vision_images")
            for sample_idx, (image_path, prompt) in enumerate(zip(batch["image_paths"], prompts)):
                inputs = build_eagle_inputs(
                    processor,
                    [image_path],
                    [prompt],
                    device,
                    image_token_limit=image_token_limit,
                    image_tensors=None if vision_images is None else [vision_images[sample_idx]],
                )
                task_id = int(batch["task_ids"][sample_idx].detach().cpu().item())
                sample_max_new_tokens = locate_generation_token_budget(
                    max_new_tokens,
                    max_instances,
                    task_id=task_id,
                )
                response = locate_model.generate(
                    pixel_values=inputs["pixel_values"],
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    image_grid_hws=inputs.get("image_grid_hws"),
                    tokenizer=tokenizer,
                    max_new_tokens=sample_max_new_tokens,
                    use_cache=True,
                    generation_mode=str(generation_mode),
                    do_sample=False,
                    verbose=False,
                )
                response_text = _normalize_locate_generate_output(response)
                if task_id == 1 and parse_locate_bbox_response(response_text, max_instances=1).numel() == 0:
                    ref_texts = batch.get("ref_texts") or []
                    ref_text = str(ref_texts[sample_idx]).strip() if sample_idx < len(ref_texts) else ""
                    if not ref_text:
                        ref_text = _extract_ref_description_from_prompt(str(batch["prompts"][sample_idx]))
                    fallback_inputs = build_eagle_inputs(
                        processor,
                        [image_path],
                        [build_refhuman_fallback_prompt(ref_text)],
                        device,
                        image_token_limit=image_token_limit,
                        image_tensors=None if vision_images is None else [vision_images[sample_idx]],
                    )
                    fallback_response = locate_model.generate(
                        pixel_values=fallback_inputs["pixel_values"],
                        input_ids=fallback_inputs["input_ids"],
                        attention_mask=fallback_inputs.get("attention_mask"),
                        image_grid_hws=fallback_inputs.get("image_grid_hws"),
                        tokenizer=tokenizer,
                        max_new_tokens=locate_generation_token_budget(
                            max_new_tokens,
                            max_instances,
                            task_id=0,
                        ),
                        use_cache=True,
                        generation_mode=str(generation_mode),
                        do_sample=False,
                        verbose=False,
                    )
                    response_text = (
                        REFHUMAN_FALLBACK_MARKER
                        + "\n"
                        + _normalize_locate_generate_output(fallback_response)
                    )
                responses.append(response_text)
    finally:
        if was_training:
            locate_model.train()
    return responses


def prepare_locate_generated_box_conditioning_from_responses(
    responses: list[str],
    batch: dict,
    device: torch.device,
    max_instances: int,
    *,
    match_iou_thresh: float,
    nms_iou_thresh: float,
    disable_pre_pose_nms: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, torch.Tensor]]]:
    selected_targets: list[dict[str, Any]] = []
    selected_condition_boxes: list[torch.Tensor] = []
    for sample_idx, target in enumerate(batch["targets"]):
        task_id = int(batch["task_ids"][sample_idx].detach().cpu().item())
        gt_boxes_all = target["boxes"]
        num_gt = int(gt_boxes_all.shape[0])
        if task_id == 1:
            ref_target = int(target["ref_target"].detach().cpu().item())
            gt_indices = [ref_target] if 0 <= ref_target < num_gt else []
        else:
            gt_indices = list(range(min(num_gt, int(max_instances))))
        gt_index_tensor = torch.as_tensor(gt_indices, dtype=torch.long)
        gt_boxes = gt_boxes_all[gt_index_tensor].clone() if gt_indices else gt_boxes_all[:0].clone()

        pred_boxes = parse_locate_boxes_for_task(
            responses[sample_idx] if sample_idx < len(responses) else "",
            task_id=task_id,
            max_instances=max_instances,
            nms_iou_thresh=nms_iou_thresh,
            disable_pre_pose_nms=disable_pre_pose_nms,
        )

        matches = hungarian_match_boxes(pred_boxes, gt_boxes, iou_thresh=match_iou_thresh)
        selected = align_target_to_predictions(
            target,
            pred_boxes,
            gt_indices,
            matches,
            task_id=task_id,
        )
        context_scale = selected.get(
            "box_context_scale",
            torch.ones(pred_boxes.shape[0], dtype=pred_boxes.dtype),
        )
        condition_boxes = expand_boxes_xyxy_per_box(pred_boxes, context_scale)
        selected_condition_boxes.append(condition_boxes[:max_instances].clone())
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


def generate_locate_bbox_responses_with_features(
    training_model: torch.nn.Module,
    processor,
    batch: dict,
    device: torch.device,
    *,
    max_instances: int,
    max_new_tokens: int,
    generation_mode: str,
    image_token_limit: int | None = None,
    single_pass_prompt: str = "locate",
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    module = unwrap_training_model(training_model)
    if module.backbone_name != "eagle" or module.backbone_model is None or module.backbone_extractor is None:
        raise ValueError("cached LocateAnything generation requires --backbone eagle/locatepose.")
    if processor is None:
        raise ValueError("LocateAnything processor is required for cached LocateAnything generation.")
    locate_model = get_eagle_base_model(module.backbone_model)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        tokenizer = getattr(locate_model, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("LocateAnything tokenizer is required for cached LocateAnything generation.")

    if str(single_pass_prompt).strip().lower() == "pose":
        prompts = [str(prompt) for prompt in batch.get("prompts", [])]
    else:
        prompts = build_locate_generation_prompts(batch)

    was_training = bool(locate_model.training)
    locate_model.eval()
    responses: list[str] = []
    feature_maps: list[torch.Tensor] = []
    text_embeds: list[torch.Tensor] = []
    try:
        vision_images = batch.get("vision_images")
        for sample_idx, (image_path, prompt) in enumerate(zip(batch["image_paths"], prompts)):
            inputs = build_eagle_inputs(
                processor,
                [image_path],
                [prompt],
                device,
                image_token_limit=image_token_limit,
                image_tensors=None if vision_images is None else [vision_images[sample_idx]],
            )
            response, feature_map, text_embed = module.backbone_extractor.generate_response_with_cached_features(
                inputs,
                tokenizer,
                max_new_tokens=locate_generation_token_budget(
                    max_new_tokens,
                    max_instances,
                    task_id=int(batch["task_ids"][sample_idx].detach().cpu().item()),
                ),
                generation_mode=generation_mode,
                do_sample=False,
                temperature=0,
            )
            task_id = int(batch["task_ids"][sample_idx].detach().cpu().item())
            if task_id == 1 and parse_locate_bbox_response(response, max_instances=1).numel() == 0:
                ref_texts = batch.get("ref_texts") or []
                ref_text = str(ref_texts[sample_idx]).strip() if sample_idx < len(ref_texts) else ""
                if not ref_text:
                    ref_text = _extract_ref_description_from_prompt(str(batch["prompts"][sample_idx]))
                fallback_inputs = build_eagle_inputs(
                    processor,
                    [image_path],
                    [build_refhuman_fallback_prompt(ref_text)],
                    device,
                    image_token_limit=image_token_limit,
                    image_tensors=None if vision_images is None else [vision_images[sample_idx]],
                )
                fallback_response, _, _ = module.backbone_extractor.generate_response_with_cached_features(
                    fallback_inputs,
                    tokenizer,
                    max_new_tokens=locate_generation_token_budget(
                        max_new_tokens,
                        max_instances,
                        task_id=0,
                    ),
                    generation_mode=generation_mode,
                    do_sample=False,
                    temperature=0,
                )
                response = REFHUMAN_FALLBACK_MARKER + "\n" + str(fallback_response)
            responses.append(response)
            feature_maps.append(feature_map)
            text_embeds.append(text_embed)
    finally:
        if was_training:
            locate_model.train()

    return responses, torch.cat(feature_maps, dim=0), torch.cat(text_embeds, dim=0)


def parse_layer_selection(spec: str) -> set[int]:
    """Parse comma-separated layer indices/ranges such as ``15-26,30``."""
    selected: set[int] = set()
    for raw_token in str(spec).split(","):
        token = raw_token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if start < 0 or end < start:
                raise ValueError(f"Invalid layer range: {token!r}")
            selected.update(range(start, end + 1))
        else:
            index = int(token)
            if index < 0:
                raise ValueError(f"Invalid layer index: {token!r}")
            selected.add(index)
    return selected


def parse_module_selection(spec: str) -> set[str]:
    return {token.strip() for token in str(spec).split(",") if token.strip()}


def _adapter_layer_index(name: str, *, vision: bool) -> int | None:
    pattern = (
        r"vision_model(?:\.[^.]+)*\.(?:blocks|layers)\.(\d+)\."
        if vision
        else r"language_model(?:\.[^.]+)*\.layers\.(\d+)\."
    )
    match = re.search(pattern, name)
    return int(match.group(1)) if match else None


def _adapter_projection_name(name: str) -> str | None:
    match = re.search(r"\.([A-Za-z0-9_]+)\.lora_[AB](?:\.|$)", name)
    return match.group(1) if match else None


def configure_backbone_train_scope(
    model: torch.nn.Module | None,
    scope: str,
    *,
    train_projector: bool = False,
    llm_layers: str = "32-35",
    vision_layers: str = "15-26",
    llm_modules: str = "q_proj,v_proj",
    vision_modules: str = "wqkv,wo,fc0,fc1",
) -> dict[str, int]:
    """Enable exactly the requested pretrained-backbone adapter parameters."""
    scope = str(scope)
    if scope not in {"frozen", "vision_lora", "all_lora", "selective_lora"}:
        raise ValueError(f"Unsupported backbone train scope: {scope!r}")
    selected_llm_layers = parse_layer_selection(llm_layers)
    selected_vision_layers = parse_layer_selection(vision_layers)
    selected_llm_modules = parse_module_selection(llm_modules)
    selected_vision_modules = parse_module_selection(vision_modules)
    if scope == "selective_lora":
        if not selected_llm_layers or not selected_vision_layers:
            raise ValueError("selective_lora requires non-empty LLM and vision layer selections.")
        if not selected_llm_modules or not selected_vision_modules:
            raise ValueError("selective_lora requires non-empty LLM and vision module selections.")

    counts = {"vision_lora": 0, "language_lora": 0, "projector": 0}
    if model is None:
        return counts
    for name, param in model.named_parameters():
        is_lora = "lora_" in name
        is_vision = is_vision_parameter(name)
        is_projector = ".mlp1." in name or name.startswith("mlp1.")
        enabled = False
        if scope == "vision_lora":
            enabled = is_lora and is_vision
        elif scope == "all_lora":
            enabled = is_lora or (train_projector and is_projector)
        elif scope == "selective_lora" and is_lora:
            layer_index = _adapter_layer_index(name, vision=is_vision)
            projection_name = _adapter_projection_name(name)
            if is_vision:
                enabled = (
                    layer_index in selected_vision_layers
                    and projection_name in selected_vision_modules
                )
            else:
                enabled = (
                    layer_index in selected_llm_layers
                    and projection_name in selected_llm_modules
                )
        param.requires_grad = bool(enabled)
        if not enabled:
            continue
        if is_projector:
            counts["projector"] += param.numel()
        elif is_vision:
            counts["vision_lora"] += param.numel()
        else:
            counts["language_lora"] += param.numel()
    if scope == "selective_lora":
        if counts["vision_lora"] == 0:
            raise RuntimeError(
                "selective_lora matched no vision LoRA parameters; check layer/module names."
            )
        if counts["language_lora"] == 0:
            raise RuntimeError(
                "selective_lora matched no language LoRA parameters; check layer/module names."
            )
    return counts


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
        backbone_train_scope: str = "all_lora",
        train_backbone_projector: bool = False,
        backbone_llm_layers: str = "32-35",
        backbone_vision_layers: str = "15-26",
        backbone_llm_modules: str = "q_proj,v_proj",
        backbone_vision_modules: str = "wqkv,wo,fc0,fc1",
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

        requested_scope = "frozen" if self.freeze_backbone else str(backbone_train_scope)
        self.backbone_train_scope = requested_scope
        self.backbone_llm_layers = str(backbone_llm_layers)
        self.backbone_vision_layers = str(backbone_vision_layers)
        self.backbone_llm_modules = str(backbone_llm_modules)
        self.backbone_vision_modules = str(backbone_vision_modules)
        self.backbone_trainable_counts = configure_backbone_train_scope(
            self.backbone_model,
            requested_scope,
            train_projector=train_backbone_projector,
            llm_layers=self.backbone_llm_layers,
            vision_layers=self.backbone_vision_layers,
            llm_modules=self.backbone_llm_modules,
            vision_modules=self.backbone_vision_modules,
        )
        self.freeze_backbone = requested_scope == "frozen"

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
        dn_boxes: torch.Tensor | None = None,
        dn_box_mask: torch.Tensor | None = None,
        dn_labels: torch.Tensor | None = None,
        dn_target_boxes: torch.Tensor | None = None,
        dn_group_ids: torch.Tensor | None = None,
        dn_source_indices: torch.Tensor | None = None,
        keypoint_dn_noisy_keypoints: torch.Tensor | None = None,
        keypoint_dn_mask: torch.Tensor | None = None,
        keypoint_dn_labels: torch.Tensor | None = None,
        keypoint_dn_target_keypoints: torch.Tensor | None = None,
        keypoint_dn_target_valid: torch.Tensor | None = None,
        keypoint_dn_target_boxes: torch.Tensor | None = None,
        keypoint_dn_target_areas: torch.Tensor | None = None,
        keypoint_dn_source_indices: torch.Tensor | None = None,
        keypoint_dn_group_ids: torch.Tensor | None = None,
        keypoint_dn_box_query_indices: torch.Tensor | None = None,
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
                        require_text=bool(task_ids.eq(1).any().item()),
                    )
                    if qwen_lm_inputs is not None and not self.freeze_backbone:
                        # LocateAnything's custom top-level forward performs an
                        # in-place image-token write which is incompatible with
                        # PEFT's input-gradient hook.  Reuse the extractor's
                        # clone-based injection path, then project only response
                        # tokens through lm_head instead of materializing full
                        # vocabulary logits for the whole prompt.
                        (
                            locate_base_model,
                            lm_input_ids,
                            lm_attention_mask,
                            lm_pixel_values,
                            lm_image_grid_hws,
                        ) = self.backbone_extractor._prepare_locate_inputs(qwen_lm_inputs)
                        _, _, projected_visual_tokens = self.backbone_extractor.run_vision_tokens(
                            lm_pixel_values,
                            lm_image_grid_hws,
                        )
                        lm_hidden = self.backbone_extractor.run_language_hidden(
                            lm_input_ids,
                            lm_attention_mask,
                            projected_visual_tokens,
                        )
                        shift_labels = qwen_lm_inputs["labels"][:, 1:].contiguous()
                        valid_label_mask = shift_labels.ne(-100)
                        if valid_label_mask.any():
                            target_hidden = lm_hidden[:, :-1, :][valid_label_mask]
                            target_labels = shift_labels[valid_label_mask]
                            target_logits = locate_base_model.language_model.lm_head(
                                target_hidden
                            ).float()
                            lm_loss = F.cross_entropy(
                                target_logits,
                                target_labels,
                                reduction="mean",
                            )
                        else:
                            lm_loss = lm_hidden.new_zeros(())
                else:
                    external_feature_map, external_text_embed = self.backbone_extractor(
                        qwen_inputs,
                        freeze_qwen=self.freeze_backbone,
                    )
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
            dn_boxes=dn_boxes,
            dn_box_mask=dn_box_mask,
            dn_labels=dn_labels,
            dn_target_boxes=dn_target_boxes,
            dn_group_ids=dn_group_ids,
            dn_source_indices=dn_source_indices,
            keypoint_dn_noisy_keypoints=keypoint_dn_noisy_keypoints,
            keypoint_dn_mask=keypoint_dn_mask,
            keypoint_dn_labels=keypoint_dn_labels,
            keypoint_dn_target_keypoints=keypoint_dn_target_keypoints,
            keypoint_dn_target_valid=keypoint_dn_target_valid,
            keypoint_dn_target_boxes=keypoint_dn_target_boxes,
            keypoint_dn_target_areas=keypoint_dn_target_areas,
            keypoint_dn_source_indices=keypoint_dn_source_indices,
            keypoint_dn_group_ids=keypoint_dn_group_ids,
            keypoint_dn_box_query_indices=keypoint_dn_box_query_indices,
            **extra,
        )
        if lm_loss is not None:
            outputs["lm_loss"] = lm_loss
        return outputs


@dataclass
class LocatePoseUnifiedConfig:
    max_instances: int = 80
    qwen_box_max_new_tokens: int = 4096
    locate_box_max_new_tokens: int = 512
    locate_generation_mode: str = "hybrid"
    box_match_iou_thresh: float = 0.10
    box_nms_iou_thresh: float = 0.70
    disable_pre_pose_nms: bool = True
    locate_generate_refhuman_only: bool = True
    qwen_min_pixels: int | None = None
    qwen_max_pixels: int | None = None
    eagle_image_token_limit: int | None = None
    eagle_batch_token_limit: int | None = None
    single_pass_prompt: str = "locate"
    use_single_pass_features: bool = False
    keep_unmatched_predictions: bool = False
    box_jitter_scale: float = 0.0
    box_jitter_shift: float = 0.0

    @classmethod
    def from_args(
        cls,
        args: argparse.Namespace,
        *,
        use_single_pass_features: bool | None = None,
        keep_unmatched_predictions: bool | None = None,
    ) -> "LocatePoseUnifiedConfig":
        return cls(
            max_instances=int(getattr(args, "max_instances", 80)),
            qwen_box_max_new_tokens=int(getattr(args, "qwen_box_max_new_tokens", 4096)),
            locate_box_max_new_tokens=int(getattr(args, "locate_box_max_new_tokens", 512)),
            locate_generation_mode=str(getattr(args, "locate_generation_mode", "hybrid")),
            box_match_iou_thresh=float(getattr(args, "box_match_iou_thresh", 0.10)),
            box_nms_iou_thresh=float(getattr(args, "box_nms_iou_thresh", 0.70)),
            disable_pre_pose_nms=bool(getattr(args, "disable_pre_pose_nms", True)),
            locate_generate_refhuman_only=bool(
                getattr(args, "locate_generate_refhuman_only", True)
            ),
            qwen_min_pixels=getattr(args, "qwen_min_pixels", None),
            qwen_max_pixels=getattr(args, "qwen_max_pixels", None),
            eagle_image_token_limit=getattr(args, "eagle_image_token_limit", None),
            eagle_batch_token_limit=getattr(args, "eagle_batch_token_limit", None),
            single_pass_prompt=str(getattr(args, "single_pass_prompt", "locate")),
            use_single_pass_features=(
                bool(getattr(args, "use_single_pass_features", False))
                if use_single_pass_features is None
                else bool(use_single_pass_features)
            ),
            keep_unmatched_predictions=(
                bool(getattr(args, "keep_unmatched_predictions", False))
                if keep_unmatched_predictions is None
                else bool(keep_unmatched_predictions)
            ),
            box_jitter_scale=float(getattr(args, "box_jitter_scale", 0.0)),
            box_jitter_shift=float(getattr(args, "box_jitter_shift", 0.0)),
        )


@dataclass
class LocatePoseUnifiedResult:
    outputs: dict[str, torch.Tensor]
    target_boxes: torch.Tensor
    target_box_mask: torch.Tensor
    pose_targets: list[dict[str, torch.Tensor]] | None = None
    gt_targets: list[dict[str, torch.Tensor]] | None = None
    locate_responses: list[str] | None = None
    locate_boxes_abs: list[list[list[float]]] | None = None
    external_feature_map: torch.Tensor | None = None
    external_text_embed: torch.Tensor | None = None
    qwen_inputs: dict[str, torch.Tensor] | None = None
    used_single_pass_features: bool = False


class LocatePoseUnifiedRuntime:
    """End-to-end LocatePose runner: box generation, feature reuse, PoseHead."""

    def __init__(
        self,
        model: torch.nn.Module,
        processor: Any | None,
        device: torch.device,
        *,
        backbone_name: str | None = None,
    ) -> None:
        self.model = model
        self.processor = processor
        self.device = device
        self.backbone_name = str(backbone_name or getattr(unwrap_training_model(model), "backbone_name", "qwen3vl"))
        if self.backbone_name == "locatepose":
            self.backbone_name = "eagle"

    @property
    def module(self) -> QwenPoseTrainingModel:
        return unwrap_training_model(self.model)

    def build_backbone_inputs(self, batch: dict, config: LocatePoseUnifiedConfig) -> dict[str, torch.Tensor] | None:
        if self.processor is None:
            return None
        if self.backbone_name == "eagle":
            vision_only = bool(getattr(self.processor, "_qwenpose_vision_only", False))
            return build_eagle_inputs(
                self.processor,
                batch["image_paths"],
                None if vision_only else batch["prompts"],
                self.device,
                image_token_limit=config.eagle_image_token_limit,
                batch_token_limit=config.eagle_batch_token_limit,
                image_tensors=batch.get("vision_images"),
            )
        return build_qwen_inputs(
            self.processor,
            batch["image_paths"],
            batch["prompts"],
            self.device,
            min_pixels=config.qwen_min_pixels,
            max_pixels=config.qwen_max_pixels,
        )

    def generate_locate_responses(
        self,
        batch: dict,
        config: LocatePoseUnifiedConfig,
    ) -> tuple[list[str], torch.Tensor | None, torch.Tensor | None]:
        if config.use_single_pass_features:
            responses, feature_map, text_embed = generate_locate_bbox_responses_with_features(
                self.model,
                self.processor,
                batch,
                self.device,
                max_instances=config.max_instances,
                max_new_tokens=config.locate_box_max_new_tokens,
                generation_mode=config.locate_generation_mode,
                image_token_limit=config.eagle_image_token_limit,
                single_pass_prompt=config.single_pass_prompt,
            )
            return responses, feature_map, text_embed
        responses = generate_locate_bbox_responses(
            self.model,
            self.processor,
            batch,
            self.device,
            max_instances=config.max_instances,
            max_new_tokens=config.locate_box_max_new_tokens,
            generation_mode=config.locate_generation_mode,
            image_token_limit=config.eagle_image_token_limit,
        )
        return responses, None, None

    def condition_inference_from_locate_responses(
        self,
        responses: list[str],
        batch: dict,
        config: LocatePoseUnifiedConfig,
    ) -> tuple[torch.Tensor, torch.Tensor, list[list[list[float]]]]:
        selected_boxes: list[torch.Tensor] = []
        locate_boxes_abs: list[list[list[float]]] = []
        for sample_idx, response in enumerate(responses):
            task_id = int(batch["task_ids"][sample_idx].detach().cpu().item())
            boxes = parse_locate_boxes_for_task(
                response,
                task_id=task_id,
                max_instances=config.max_instances,
                nms_iou_thresh=config.box_nms_iou_thresh,
                disable_pre_pose_nms=config.disable_pre_pose_nms,
            )

            target = batch["targets"][sample_idx]
            width = float(target["width"])
            height = float(target["height"])
            locate_boxes_abs.append(
                [
                    [
                        float(box[0]) * width,
                        float(box[1]) * height,
                        float(box[2]) * width,
                        float(box[3]) * height,
                    ]
                    for box in boxes.detach().cpu().tolist()
                ]
            )
            context_scale = _context_scale_for_indices(
                target, [], int(boxes.shape[0])
            )
            selected_boxes.append(
                expand_boxes_xyxy_per_box(boxes, context_scale)
            )

        max_boxes = max([int(boxes.shape[0]) for boxes in selected_boxes] + [1])
        box_tensor = torch.zeros(len(selected_boxes), max_boxes, 4, dtype=torch.float32, device=self.device)
        box_mask = torch.zeros(len(selected_boxes), max_boxes, dtype=torch.bool, device=self.device)
        for sample_idx, boxes in enumerate(selected_boxes):
            n = int(boxes.shape[0])
            if n <= 0:
                continue
            box_tensor[sample_idx, :n] = boxes.to(device=self.device, dtype=torch.float32)
            box_mask[sample_idx, :n] = True
        return box_tensor, box_mask, locate_boxes_abs

    def forward_pose(
        self,
        batch: dict,
        target_boxes: torch.Tensor,
        target_box_mask: torch.Tensor,
        config: LocatePoseUnifiedConfig,
        *,
        external_feature_map: torch.Tensor | None = None,
        external_text_embed: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor] | None]:
        if external_feature_map is not None and external_text_embed is not None:
            outputs = self.module.pose_model(
                schema_ids=batch["schema_ids"],
                task_ids=batch["task_ids"],
                external_feature_map=external_feature_map,
                external_text_embed=external_text_embed,
                target_boxes=target_boxes,
                target_box_mask=target_box_mask,
                images=batch.get("images"),
            )
            return outputs, None
        backbone_inputs = self.build_backbone_inputs(batch, config)
        outputs = self.model(
            schema_ids=batch["schema_ids"],
            task_ids=batch["task_ids"],
            qwen_inputs=backbone_inputs,
            target_boxes=target_boxes,
            target_box_mask=target_box_mask,
            images=batch.get("images"),
        )
        return outputs, backbone_inputs

    def infer_batch(
        self,
        batch: dict,
        config: LocatePoseUnifiedConfig,
        *,
        precomputed_locate_responses: list[str] | None = None,
    ) -> LocatePoseUnifiedResult:
        if bool(self.module.pose_model.config.use_global_person_queries):
            batch_size = int(batch["schema_ids"].shape[0])
            placeholder_boxes = torch.zeros(
                batch_size, 1, 4, device=self.device, dtype=torch.float32
            )
            placeholder_mask = torch.zeros(
                batch_size, 1, device=self.device, dtype=torch.bool
            )
            outputs, qwen_inputs = self.forward_pose(
                batch,
                placeholder_boxes,
                placeholder_mask,
                config,
            )
            predicted_boxes = outputs["pred_boxes"]
            predicted_mask = outputs["box_mask"]
            predicted_abs: list[list[list[float]]] = []
            for sample_idx, target in enumerate(batch["targets"]):
                width = float(target["width"])
                height = float(target["height"])
                sample_boxes = predicted_boxes[sample_idx][predicted_mask[sample_idx]]
                predicted_abs.append(
                    [
                        [
                            float(box[0]) * width,
                            float(box[1]) * height,
                            float(box[2]) * width,
                            float(box[3]) * height,
                        ]
                        for box in sample_boxes.detach().cpu().tolist()
                    ]
                )
            return LocatePoseUnifiedResult(
                outputs=outputs,
                target_boxes=predicted_boxes,
                target_box_mask=predicted_mask,
                locate_responses=None,
                locate_boxes_abs=predicted_abs,
                qwen_inputs=qwen_inputs,
                used_single_pass_features=True,
            )
        external_feature_map = None
        external_text_embed = None
        if precomputed_locate_responses is None:
            responses, external_feature_map, external_text_embed = self.generate_locate_responses(batch, config)
        else:
            responses = precomputed_locate_responses
        target_boxes, target_box_mask, locate_boxes_abs = self.condition_inference_from_locate_responses(
            responses,
            batch,
            config,
        )
        outputs, qwen_inputs = self.forward_pose(
            batch,
            target_boxes,
            target_box_mask,
            config,
            external_feature_map=external_feature_map,
            external_text_embed=external_text_embed,
        )
        return LocatePoseUnifiedResult(
            outputs=outputs,
            target_boxes=target_boxes,
            target_box_mask=target_box_mask,
            locate_responses=responses,
            locate_boxes_abs=locate_boxes_abs,
            external_feature_map=external_feature_map,
            external_text_embed=external_text_embed,
            qwen_inputs=qwen_inputs,
            used_single_pass_features=external_feature_map is not None and external_text_embed is not None,
        )

    def prepare_training_conditioning(
        self,
        batch: dict,
        config: LocatePoseUnifiedConfig,
        *,
        box_source: str,
    ) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, torch.Tensor]]]:
        if box_source == "person_queries":
            return prepare_person_query_conditioning(
                batch["targets"],
                batch["task_ids"],
                self.device,
                max_instances=config.max_instances,
            )
        if box_source == "qwen_generate":
            return prepare_qwen_generated_box_conditioning(
                self.model,
                self.processor,
                batch,
                self.device,
                max_instances=config.max_instances,
                max_new_tokens=config.qwen_box_max_new_tokens,
                match_iou_thresh=config.box_match_iou_thresh,
                nms_iou_thresh=config.box_nms_iou_thresh,
                min_pixels=config.qwen_min_pixels,
                max_pixels=config.qwen_max_pixels,
            )
        if box_source == "locate_generate":
            # The restored recipe closed the loop only for RefHuman. Regular
            # pose datasets keep clean GT box conditioning for PoseHead.
            if config.locate_generate_refhuman_only and not bool(
                batch["task_ids"].eq(1).any().item()
            ):
                return prepare_box_conditioning(
                    batch["targets"],
                    batch["task_ids"],
                    self.device,
                    max_instances=config.max_instances,
                    box_jitter_scale=config.box_jitter_scale,
                    box_jitter_shift=config.box_jitter_shift,
                )
            responses = generate_locate_bbox_responses(
                self.model,
                self.processor,
                batch,
                self.device,
                max_instances=config.max_instances,
                max_new_tokens=config.locate_box_max_new_tokens,
                generation_mode=config.locate_generation_mode,
                image_token_limit=config.eagle_image_token_limit,
            )
            return prepare_locate_generated_box_conditioning_from_responses(
                responses,
                batch,
                self.device,
                max_instances=config.max_instances,
                match_iou_thresh=config.box_match_iou_thresh,
                nms_iou_thresh=config.box_nms_iou_thresh,
                disable_pre_pose_nms=config.disable_pre_pose_nms,
            )
        return prepare_box_conditioning(
            batch["targets"],
            batch["task_ids"],
            self.device,
            max_instances=config.max_instances,
            box_jitter_scale=config.box_jitter_scale,
            box_jitter_shift=config.box_jitter_shift,
        )

    def eval_batch(
        self,
        batch: dict,
        config: LocatePoseUnifiedConfig,
        *,
        box_source: str,
        precomputed_locate_responses: list[str] | None = None,
    ) -> LocatePoseUnifiedResult:
        _, _, gt_targets_for_eval = prepare_box_conditioning(
            batch["targets"],
            batch["task_ids"],
            self.device,
            max_instances=config.max_instances,
        )
        external_feature_map = None
        external_text_embed = None
        responses: list[str] | None = None
        locate_boxes_abs: list[list[list[float]]] | None = None
        if box_source == "person_queries":
            target_boxes, target_box_mask, pose_targets = prepare_person_query_conditioning(
                batch["targets"],
                batch["task_ids"],
                self.device,
                max_instances=config.max_instances,
            )
        elif box_source == "qwen_generate":
            target_boxes, target_box_mask, pose_targets = prepare_qwen_generated_box_conditioning(
                self.model,
                self.processor,
                batch,
                self.device,
                max_instances=config.max_instances,
                max_new_tokens=config.qwen_box_max_new_tokens,
                match_iou_thresh=config.box_match_iou_thresh,
                nms_iou_thresh=config.box_nms_iou_thresh,
                min_pixels=config.qwen_min_pixels,
                max_pixels=config.qwen_max_pixels,
                keep_unmatched_predictions=config.keep_unmatched_predictions,
            )
        elif box_source == "locate_generate":
            if precomputed_locate_responses is not None:
                responses = precomputed_locate_responses
                target_boxes, target_box_mask, pose_targets = prepare_locate_generated_box_conditioning_from_responses(
                    responses,
                    batch,
                    self.device,
                    max_instances=config.max_instances,
                    match_iou_thresh=config.box_match_iou_thresh,
                    nms_iou_thresh=config.box_nms_iou_thresh,
                    disable_pre_pose_nms=config.disable_pre_pose_nms,
                )
            elif config.use_single_pass_features:
                responses, external_feature_map, external_text_embed = self.generate_locate_responses(batch, config)
                target_boxes, target_box_mask, pose_targets = prepare_locate_generated_box_conditioning_from_responses(
                    responses,
                    batch,
                    self.device,
                    max_instances=config.max_instances,
                    match_iou_thresh=config.box_match_iou_thresh,
                    nms_iou_thresh=config.box_nms_iou_thresh,
                    disable_pre_pose_nms=config.disable_pre_pose_nms,
                )
            else:
                responses, _, _ = self.generate_locate_responses(batch, config)
                target_boxes, target_box_mask, pose_targets = prepare_locate_generated_box_conditioning_from_responses(
                    responses,
                    batch,
                    self.device,
                    max_instances=config.max_instances,
                    match_iou_thresh=config.box_match_iou_thresh,
                    nms_iou_thresh=config.box_nms_iou_thresh,
                    disable_pre_pose_nms=config.disable_pre_pose_nms,
                )
            locate_boxes_abs = locate_boxes_abs_from_responses(
                responses or [],
                batch,
                max_instances=config.max_instances,
                nms_iou_thresh=config.box_nms_iou_thresh,
                disable_pre_pose_nms=config.disable_pre_pose_nms,
            )
        else:
            target_boxes, target_box_mask, pose_targets = prepare_box_conditioning(
                batch["targets"],
                batch["task_ids"],
                self.device,
                max_instances=config.max_instances,
                box_jitter_scale=config.box_jitter_scale,
                box_jitter_shift=config.box_jitter_shift,
            )
        outputs, qwen_inputs = self.forward_pose(
            batch,
            target_boxes,
            target_box_mask,
            config,
            external_feature_map=external_feature_map,
            external_text_embed=external_text_embed,
        )
        if box_source == "person_queries":
            pose_targets = align_targets_to_person_queries(
                outputs,
                pose_targets,
                batch["task_ids"],
            )
        return LocatePoseUnifiedResult(
            outputs=outputs,
            target_boxes=target_boxes,
            target_box_mask=target_box_mask,
            pose_targets=pose_targets,
            gt_targets=gt_targets_for_eval,
            locate_responses=responses,
            locate_boxes_abs=locate_boxes_abs,
            external_feature_map=external_feature_map,
            external_text_embed=external_text_embed,
            qwen_inputs=qwen_inputs,
            used_single_pass_features=external_feature_map is not None and external_text_embed is not None,
        )


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def distributed_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def distributed_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def distributed_barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def distributed_any(value: bool, device: torch.device) -> bool:
    """Return True on every rank when any rank reports True."""
    flag = torch.tensor(int(bool(value)), device=device, dtype=torch.int32)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
    return bool(flag.item())


def iter_named_floating_tensors(value, prefix: str = ""):
    """Yield floating tensors from nested model outputs without detaching them."""
    if torch.is_tensor(value):
        if torch.is_floating_point(value):
            yield prefix or "tensor", value
        return
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from iter_named_floating_tensors(item, child)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]" if prefix else f"[{index}]"
            yield from iter_named_floating_tensors(item, child)


def synchronized_finite_check(
    named_tensors,
    device: torch.device,
    *,
    max_bad_names: int = 4,
) -> tuple[bool, list[str]]:
    """Check tensors locally and synchronize the result across all ranks."""
    items = [(name, tensor) for name, tensor in named_tensors if tensor is not None]
    local_ok = torch.ones((), device=device, dtype=torch.int32)
    for _, tensor in items:
        tensor_ok = torch.isfinite(tensor.detach()).all().to(device=device, dtype=torch.int32)
        local_ok = torch.minimum(local_ok, tensor_ok)
    global_ok = local_ok.clone()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(global_ok, op=torch.distributed.ReduceOp.MIN)
    if bool(global_ok.item()):
        return True, []
    bad_names: list[str] = []
    if not bool(local_ok.item()):
        for name, tensor in items:
            if not bool(torch.isfinite(tensor.detach()).all().item()):
                bad_names.append(name)
                if len(bad_names) >= max_bad_names:
                    break
    return False, bad_names


def deepspeed_partition_nonfinite_names(
    active_model: torch.nn.Module,
    training_model: torch.nn.Module,
    device: torch.device,
    *,
    max_bad_names_per_rank: int = 8,
) -> list[str]:
    """Name non-finite ZeRO-1/2 partition gradients before ``step`` clears them.

    ZeRO-2 moves reduced gradients out of ``Parameter.grad`` into
    ``optimizer.averaged_gradients`` during backward.  Consequently a normal
    parameter-gradient scan can report no offending name even though
    DeepSpeed correctly rejects the update.  The lists in
    ``averaged_gradients`` follow ``params_in_partition``; use that association
    here and gather the small set of names from every data-parallel rank.
    """
    optimizer = getattr(active_model, "optimizer", None)
    averaged_gradients = getattr(optimizer, "averaged_gradients", None)
    params_in_partition = getattr(optimizer, "params_in_partition", None)
    if not isinstance(averaged_gradients, dict) or params_in_partition is None:
        return []

    parameter_names = {id(parameter): name for name, parameter in training_model.named_parameters()}
    local_bad: list[str] = []
    for group_index, partition_parameters in enumerate(params_in_partition):
        partition_gradients = averaged_gradients.get(group_index)
        if partition_gradients is None:
            continue
        for parameter, gradient in zip(partition_parameters, partition_gradients):
            if gradient is None or bool(torch.isfinite(gradient.detach()).all().item()):
                continue
            name = parameter_names.get(id(parameter), f"optimizer_group_{group_index}.unknown_parameter")
            local_bad.append(name)
            if len(local_bad) >= max_bad_names_per_rank:
                break
        if len(local_bad) >= max_bad_names_per_rank:
            break

    any_bad = distributed_any(bool(local_bad), device)
    if not any_bad:
        return []
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        gathered: list[list[str] | None] = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, local_bad)
        return [
            f"rank{rank}:{name}"
            for rank, names in enumerate(gathered)
            for name in (names or [])
        ]
    return local_bad


def update_progress_bar(progress_bar, postfix: dict[str, object]) -> None:
    if progress_bar is not None:
        progress_bar.set_postfix(postfix, refresh=False)


def _format_loss_float(value: float, digits: int = 3) -> str:
    value = float(value)
    if not math.isfinite(value):
        return str(value)
    return f"{value:.{digits}f}"


def _format_loss_weight(value: float) -> str:
    value = float(value)
    if not math.isfinite(value):
        return str(value)
    return f"{value:.4g}"


def estimate_locate_vision_tokens(
    width: int,
    height: int,
    image_token_limit: int | None,
    *,
    patch_size: int = 14,
    merge_kernel_size: tuple[int, int] = (2, 2),
) -> int:
    """Match LocateAnything image preprocessing closely enough for bucketing."""
    width = max(int(width), 1)
    height = max(int(height), 1)
    patch_size = max(int(patch_size), 1)
    raw_w = max(width // patch_size, 1)
    raw_h = max(height // patch_size, 1)
    raw_tokens = raw_w * raw_h
    if image_token_limit is not None and int(image_token_limit) > 0 and raw_tokens > int(image_token_limit):
        scale = math.sqrt(float(image_token_limit) / float(raw_tokens))
        width = max(int(width * scale), 1)
        height = max(int(height * scale), 1)
    pad_h = max(int(merge_kernel_size[0]), 1) * patch_size
    pad_w = max(int(merge_kernel_size[1]), 1) * patch_size
    target_w = math.ceil(width / pad_w) * pad_w
    target_h = math.ceil(height / pad_h) * pad_h
    return max((target_w // patch_size) * (target_h // patch_size), 1)


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
        balance_vision_tokens: bool = False,
        vision_token_limit: int | None = None,
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
        self.balance_vision_tokens = bool(balance_vision_tokens)
        self.vision_token_limit = None if vision_token_limit is None else int(vision_token_limit)
        self.epoch = 0
        self.start_batch = 0
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
        self.start_batch = 0
        self._cached_batches = None
        set_pose_dataset_epoch(self.dataset, self.epoch)

    def set_start_batch(self, start_batch: int) -> None:
        """Start iteration at a batch offset without materializing dataset samples."""
        self.start_batch = max(int(start_batch), 0)

    def _sample_cost(self, dataset_idx: int, local_linear: int) -> int:
        """Estimate activation cost without opening the image file."""
        inner_datasets = list(getattr(self.dataset, "datasets"))
        inner_dataset = inner_datasets[dataset_idx]
        records = getattr(inner_dataset, "records", None)
        offsets = getattr(self.dataset, "offsets", None)
        strides = getattr(self.dataset, "strides", None)
        if records is None or offsets is None or strides is None or not records:
            return 1
        local_index = (
            int(offsets[dataset_idx]) + int(local_linear) * int(strides[dataset_idx])
        ) % len(records)
        record = records[local_index]
        token_cost = estimate_locate_vision_tokens(
            int(getattr(record, "width", 1)),
            int(getattr(record, "height", 1)),
            self.vision_token_limit,
        )
        boxes = getattr(record, "boxes_xyxy", None)
        instance_count = 0 if boxes is None else min(
            int(boxes.shape[0]),
            int(getattr(inner_dataset, "max_instances", 80)),
        )
        # Pose queries matter, but traces show vision tokens dominate peak memory.
        return int(token_cost + instance_count * 32)

    def _build_batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch * 1009)
        per_dataset_batches: list[list[list[int]]] = []
        batch_counts: list[int] = []
        inner_datasets = list(getattr(self.dataset, "datasets"))
        global_batch_size = self.batch_size * self.world_size
        for dataset_idx, inner_dataset in enumerate(inner_datasets):
            n = (
                int(self.dataset.sample_count_for_epoch(dataset_idx, self.epoch))
                if hasattr(self.dataset, "sample_count_for_epoch")
                else len(inner_dataset)
            )
            if n <= 0:
                per_dataset_batches.append([])
                batch_counts.append(0)
                continue

            if hasattr(self.dataset, "sample_linear_indices_for_epoch"):
                local_linear = list(
                    self.dataset.sample_linear_indices_for_epoch(dataset_idx, self.epoch)
                )
            else:
                start_offset = (
                    int(self.dataset.sample_start_for_epoch(dataset_idx, self.epoch))
                    if hasattr(self.dataset, "sample_start_for_epoch")
                    else 0
                )
                local_linear = list(range(start_offset, start_offset + n))
            dataset_rng = random.Random(self.seed + self.epoch * 1009 + dataset_idx * 9176)
            sample_costs: dict[int, int] = {}
            if self.balance_vision_tokens:
                sample_costs = {
                    value: self._sample_cost(dataset_idx, value)
                    for value in local_linear
                }
                # Random tie-breaking changes neighboring samples each epoch while
                # retaining length bucketing by the dominant vision-token cost.
                decorated = [
                    (sample_costs[value], dataset_rng.random(), value)
                    for value in local_linear
                ]
                decorated.sort(key=lambda item: (item[0], item[1]))
                local_linear = [item[2] for item in decorated]
            elif self.shuffle:
                dataset_rng.shuffle(local_linear)

            num_global_batches = math.ceil(n / global_batch_size)
            target_count = num_global_batches * global_batch_size if self.fill_last else n
            if self.fill_last and len(local_linear) < target_count:
                pad_source = local_linear or [0]
                local_linear.extend(
                    pad_source[i % len(pad_source)]
                    for i in range(target_count - len(local_linear))
                )

            rank_batches: list[list[int]] = []
            for start in range(0, len(local_linear), global_batch_size):
                global_chunk = local_linear[start : start + global_batch_size]
                if not global_chunk:
                    continue
                if len(global_chunk) < global_batch_size and self.fill_last:
                    global_chunk = global_chunk + [
                        global_chunk[i % len(global_chunk)]
                        for i in range(global_batch_size - len(global_chunk))
                    ]

                if self.balance_vision_tokens:
                    assignments = [[] for _ in range(self.world_size)]
                    assignment_costs = [0 for _ in range(self.world_size)]
                    ordered = sorted(
                        global_chunk,
                        key=lambda value: sample_costs[value],
                        reverse=True,
                    )
                    for value in ordered:
                        candidates = [
                            rank_idx
                            for rank_idx in range(self.world_size)
                            if len(assignments[rank_idx]) < self.batch_size
                        ]
                        rank_idx = min(
                            candidates,
                            key=lambda idx: (assignment_costs[idx], len(assignments[idx]), idx),
                        )
                        assignments[rank_idx].append(value)
                        assignment_costs[rank_idx] += sample_costs[value]
                else:
                    assignments = [
                        global_chunk[
                            rank_idx * self.batch_size : (rank_idx + 1) * self.batch_size
                        ]
                        for rank_idx in range(self.world_size)
                    ]

                local_batch = assignments[self.rank]
                if len(local_batch) < self.batch_size and not self.fill_last:
                    continue
                rank_batches.append([
                    self.dataset.global_index_for_dataset_linear(dataset_idx, value)
                    for value in local_batch
                ])

            if self.shuffle and rank_batches:
                dataset_rng.shuffle(rank_batches)
            per_dataset_batches.append(rank_batches)
            batch_counts.append(len(rank_batches))

        schedule = self._weighted_schedule(batch_counts)
        cursors = [0 for _ in batch_counts]
        all_batches: list[list[int]] = []
        for dataset_idx in schedule:
            all_batches.append(per_dataset_batches[dataset_idx][cursors[dataset_idx]])
            cursors[dataset_idx] += 1
        if self.shuffle and all_batches:
            # Every rank uses the same deterministic block permutation, so source
            # datasets and cost buckets stay aligned at each distributed step.
            window = max(1, min(8, len(all_batches)))
            jittered: list[list[int]] = []
            for start in range(0, len(all_batches), window):
                block = all_batches[start : start + window]
                rng.shuffle(block)
                jittered.extend(block)
            all_batches = jittered
        return all_batches

    def __iter__(self):
        self._cached_batches = self._build_batches()
        yield from self._cached_batches[self.start_batch :]

    def __len__(self) -> int:
        if self._cached_batches is None:
            self._cached_batches = self._build_batches()
        return len(self._cached_batches)


def compute_pose_diagnostics(
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    task_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute dataset-comparable diagnostics without affecting gradients."""
    device = outputs["keypoints"].device
    box_mask = outputs["box_mask"].to(device).bool()
    image_error_sum = torch.zeros((), device=device)
    box_error_sum = torch.zeros((), device=device)
    valid_joint_count = torch.zeros((), device=device)
    clipped_joint_count = torch.zeros((), device=device)
    mpii_area_ratio_sum = torch.zeros((), device=device)
    mpii_area_ratio_count = 0
    ref_total = 0
    ref_matched = 0

    with torch.no_grad():
        for sample_idx, target in enumerate(targets):
            valid_queries = torch.nonzero(
                box_mask[sample_idx], as_tuple=False
            ).flatten()
            target_count = int(target["boxes"].shape[0])
            n = min(int(valid_queries.numel()), target_count)
            is_ref = int(task_ids[sample_idx].detach().cpu().item()) == 1
            if is_ref:
                ref_total += 1
                ref_target = target.get("ref_target")
                if isinstance(ref_target, torch.Tensor) and ref_target.numel() == 1:
                    ref_matched += int(int(ref_target.detach().cpu().item()) >= 0)
                else:
                    ref_matched += 0
            if n <= 0:
                continue

            queries = valid_queries[:n]
            gt = target["keypoints"].to(device)[:n, :, :2]
            valid = target["keypoint_valid"].to(device)[:n].bool()
            supervised_people = valid.any(dim=-1)
            if not supervised_people.any():
                continue
            queries = queries[supervised_people]
            gt = gt[supervised_people]
            valid = valid[supervised_people]
            pred = outputs["keypoints"][sample_idx, queries, :, :2]
            valid_f = valid.float()
            joint_count = valid_f.sum()
            if joint_count <= 0:
                continue

            image_joint_error = (pred - gt).abs().mean(dim=-1)
            image_error_sum = image_error_sum + (image_joint_error * valid_f).sum()

            loss_boxes = target.get("loss_boxes", target["boxes"]).to(device)[:n][supervised_people]
            loss_wh = (loss_boxes[:, 2:] - loss_boxes[:, :2]).clamp(min=1e-4)
            box_joint_error = (
                (pred - gt).abs() / loss_wh[:, None, :]
            ).mean(dim=-1)
            box_error_sum = box_error_sum + (box_joint_error * valid_f).sum()

            pose_boxes = outputs["pose_boxes"][sample_idx, queries]
            pose_wh = (pose_boxes[:, 2:] - pose_boxes[:, :2]).clamp(min=1e-4)
            relative_gt = (
                gt - pose_boxes[:, None, :2]
            ) / pose_wh[:, None, :]
            clipped = (
                (relative_gt < 0.0) | (relative_gt > 1.0)
            ).any(dim=-1) & valid
            clipped_joint_count = clipped_joint_count + clipped.float().sum()
            valid_joint_count = valid_joint_count + joint_count

            if str(target.get("dataset", "")).lower() == "mpii":
                condition_boxes = target["boxes"].to(device)[:n][supervised_people]
                condition_wh = (
                    condition_boxes[:, 2:] - condition_boxes[:, :2]
                ).clamp(min=0.0)
                loss_wh_area = (
                    loss_boxes[:, 2:] - loss_boxes[:, :2]
                ).clamp(min=1e-6)
                ratio = (
                    condition_wh[:, 0] * condition_wh[:, 1]
                ) / (loss_wh_area[:, 0] * loss_wh_area[:, 1]).clamp(min=1e-8)
                mpii_area_ratio_sum = mpii_area_ratio_sum + ratio.sum()
                mpii_area_ratio_count += int(ratio.numel())

    denom = valid_joint_count.clamp(min=1.0)
    diagnostics = {
        "metric_image_mae": image_error_sum / denom,
        "metric_box_relative_mae": box_error_sum / denom,
        "metric_pose_box_clipped_ratio": clipped_joint_count / denom,
        "metric_valid_joint_count": valid_joint_count,
    }
    if mpii_area_ratio_count > 0:
        diagnostics["metric_mpii_condition_base_area_ratio"] = (
            mpii_area_ratio_sum / mpii_area_ratio_count
        )
    if ref_total > 0:
        diagnostics["metric_refhuman_match_rate"] = torch.tensor(
            ref_matched / ref_total, device=device, dtype=torch.float32
        )
    return diagnostics


def update_dataset_metric_ema(
    state: dict[str, dict[str, float]],
    dataset_name: str,
    metrics: dict[str, float],
    *,
    decay: float = 0.95,
) -> None:
    dataset_state = state.setdefault(dataset_name, {})
    tracked_keys = (
        "loss_total",
        "loss_coord",
        "loss_image_coord",
        "loss_oks",
        "loss_keypoint_confidence",
        "loss_person_confidence",
        "loss_ref_match",
        "ref_match_accuracy",
        "ref_match_margin",
        "person_quality_target_mean",
        "person_confidence_mean",
        "person_confidence_std",
        "metric_image_mae",
        "metric_box_relative_mae",
        "metric_pose_box_clipped_ratio",
        "metric_valid_joint_count",
        "metric_mpii_condition_base_area_ratio",
        "metric_refhuman_match_rate",
    )
    for key in tracked_keys:
        if key not in metrics or not math.isfinite(float(metrics[key])):
            continue
        value = float(metrics[key])
        if key not in dataset_state:
            dataset_state[key] = value
        else:
            dataset_state[key] = decay * dataset_state[key] + (1.0 - decay) * value


def format_dataset_metric_ema(
    state: dict[str, dict[str, float]],
) -> list[str]:
    messages: list[str] = []
    for dataset_name in sorted(state):
        metrics = state[dataset_name]
        fields = []
        for key, label in (
            ("loss_total", "loss"),
            ("loss_coord", "coord"),
            ("loss_image_coord", "img_loss"),
            ("metric_image_mae", "img_mae"),
            ("metric_box_relative_mae", "box_mae"),
            ("metric_pose_box_clipped_ratio", "clip"),
            ("metric_refhuman_match_rate", "proposal_match"),
            ("ref_match_accuracy", "ref_acc"),
            ("ref_match_margin", "ref_margin"),
            ("metric_mpii_condition_base_area_ratio", "area_ratio"),
        ):
            if key in metrics:
                fields.append(f"{label}={metrics[key]:.4f}")
        messages.append(f"dataset_ema dataset={dataset_name} " + " ".join(fields))
    return messages


def build_progress_loss_postfix(
    loss_metrics: dict[str, float],
    weights: LossWeights,
) -> dict[str, str]:
    """Build a compact tqdm postfix from weighted head contributions.

    ``pose`` and ``box`` are the complete non-DN regression-head objectives.
    Their denoising objectives are deliberately reported as ``posedn`` and
    ``boxdn`` so a large DN auxiliary cannot be mistaken for main-head error.
    """

    group_totals = _weighted_loss_group_totals(loss_metrics, weights)
    postfix = {"loss": _format_loss_float(loss_metrics.get("loss_total", 0.0))}
    for group, label in (
        ("pose", "pose"),
        ("pose_dn", "posedn"),
        ("box", "box"),
        ("box_dn", "boxdn"),
    ):
        if group_totals.get(group, 0.0) > 0.0:
            postfix[label] = _format_loss_float(group_totals[group])
    if "loss_person_confidence" in loss_metrics:
        # Quality target statistics remain diagnostics rather than additional
        # loss fields; the weighted BCE itself is already included in pose.
        postfix["q"] = _format_loss_float(
            loss_metrics.get("person_quality_target_mean", 0.0)
        )
        postfix["sstd"] = _format_loss_float(
            loss_metrics.get("person_confidence_std", 0.0)
        )
    if group_totals.get("ref", 0.0) > 0.0:
        postfix["ref"] = _format_loss_float(group_totals["ref"])
        postfix["racc"] = _format_loss_float(
            loss_metrics.get("ref_match_accuracy", 0.0)
        )
    if group_totals.get("lm", 0.0) > 0.0:
        postfix["lm"] = _format_loss_float(group_totals["lm"])
        postfix["lmraw"] = _format_loss_float(loss_metrics.get("loss_lm", 0.0))
    return postfix


def _weighted_loss_items(
    loss_metrics: dict[str, float],
    weights: LossWeights,
) -> list[tuple[str, str, float, float, float]]:
    items: list[tuple[str, str, float, float, float]] = []

    def add(group: str, label: str, key: str, weight: float) -> None:
        weight = float(weight)
        if weight <= 0.0 or key not in loss_metrics:
            return
        raw = float(loss_metrics.get(key, 0.0))
        items.append((group, label, raw, weight, raw * weight))

    add("pose", "oks", "loss_oks", weights.oks)
    add("pose", "coord", "loss_coord", weights.coord)
    add("pose", "img", "loss_image_coord", weights.image_coord)
    add("pose", "hard", "loss_hard_joint", weights.hard_joint)
    confidence_weight = (
        weights.keypoint_confidence if weights.vis is None else float(weights.vis)
    )
    add(
        "pose",
        "conf",
        "loss_keypoint_confidence",
        confidence_weight,
    )
    add("pose", "pconf", "loss_person_confidence", weights.person_confidence)
    add("ref", "match", "loss_ref_match", weights.ref_match)
    add(
        "pose",
        "coarse_oks",
        "loss_oks_coarse",
        weights.coarse_coord * weights.oks,
    )
    add(
        "pose",
        "coarse_coord",
        "loss_coord_coarse",
        weights.coarse_coord * weights.coord,
    )
    add(
        "pose",
        "coarse_img",
        "loss_image_coord_coarse",
        weights.coarse_coord * weights.image_coord,
    )
    add("pose", "deform", "loss_coord_deform", weights.deform_coord)
    for refine_idx, refine_weight in enumerate(parse_float_list(weights.refine_coords), start=1):
        add("pose", f"ref{refine_idx}", f"loss_coord_refine_{refine_idx}", refine_weight)
    add("box", "obj", "loss_box_objectness", weights.box_objectness)
    add("box", "l1", "loss_box_l1", weights.box_l1)
    add("box", "giou", "loss_box_giou", weights.box_giou)
    add("box", "rel", "loss_box_relative", weights.box_relative)
    add("box_dn", "total", "loss_box_dn", weights.box_dn)
    # loss_keypoint_dn already contains its internal pose/confidence/deep-
    # supervision weights. Apply only the outer DN weight here so the displayed
    # value is exactly its contribution to loss_total.
    add("pose_dn", "total", "loss_keypoint_dn", weights.keypoint_dn)
    add("lm", "lm", "loss_lm", float(loss_metrics.get("loss_lm_weight", weights.lm)))
    return items


def _weighted_loss_group_totals(
    loss_metrics: dict[str, float],
    weights: LossWeights,
) -> dict[str, float]:
    totals: dict[str, float] = {}
    for group, _label, _raw, _weight, weighted in _weighted_loss_items(loss_metrics, weights):
        totals[group] = totals.get(group, 0.0) + weighted
    return totals


def build_detailed_loss_message(
    loss_metrics: dict[str, float],
    weights: LossWeights,
) -> str:
    if "loss_person_confidence" in loss_metrics and "loss_oks" not in loss_metrics:
        return (
            "loss_detail "
            f"total={_format_loss_float(loss_metrics.get('loss_total', 0.0))} "
            f"person_conf={_format_loss_float(loss_metrics['loss_person_confidence'])} "
            f"target_mean={_format_loss_float(loss_metrics.get('person_quality_target_mean', 0.0))} "
            f"target_std={_format_loss_float(loss_metrics.get('person_quality_target_std', 0.0))} "
            f"score_mean={_format_loss_float(loss_metrics.get('person_confidence_mean', 0.0))} "
            f"score_std={_format_loss_float(loss_metrics.get('person_confidence_std', 0.0))}"
        )
    items = _weighted_loss_items(loss_metrics, weights)
    group_totals = _weighted_loss_group_totals(loss_metrics, weights)
    group_names = {
        "pose": "pose",
        "pose_dn": "posedn",
        "box": "box",
        "box_dn": "boxdn",
        "ref": "ref",
        "lm": "lm",
    }
    summary = [
        f"total={_format_loss_float(loss_metrics.get('loss_total', 0.0))}",
        *[
            f"{name}={_format_loss_float(group_totals[group])}"
            for group, name in group_names.items()
            if group_totals.get(group, 0.0) > 0.0
        ],
    ]
    groups: list[str] = []
    for group, name in group_names.items():
        group_items = [item for item in items if item[0] == group]
        if not group_items:
            continue
        detail = " ".join(
            f"{label}={_format_loss_float(weighted)}"
            f"(raw={_format_loss_float(raw)},w={_format_loss_weight(weight)})"
            for _group, label, raw, weight, weighted in group_items
        )
        groups.append(f"{name}: {detail}")
    return "loss_detail " + " ".join(summary) + "; " + "; ".join(groups)


def format_loss_weights(weights: LossWeights) -> str:
    coord_refine = ",".join(_format_loss_weight(value) for value in parse_float_list(weights.refine_coords)) or "none"
    return (
        "Loss weights: "
        f"oks={_format_loss_weight(weights.oks)} "
        f"coord={_format_loss_weight(weights.coord)} "
        f"image_coord={_format_loss_weight(weights.image_coord)} "
        f"keypoint_confidence={_format_loss_weight(weights.keypoint_confidence if weights.vis is None else weights.vis)} "
        f"person_confidence={_format_loss_weight(weights.person_confidence)} "
        f"ref_match={_format_loss_weight(weights.ref_match)} "
        f"hard={_format_loss_weight(weights.hard_joint)} "
        f"coord_aux(coarse_full_objective_scale={_format_loss_weight(weights.coarse_coord)},"
        f"deform={_format_loss_weight(weights.deform_coord)},refine={coord_refine}) "
        f"box(obj={_format_loss_weight(weights.box_objectness)},"
        f"l1={_format_loss_weight(weights.box_l1)},"
        f"giou={_format_loss_weight(weights.box_giou)},"
        f"relative={_format_loss_weight(weights.box_relative)},"
        f"dn={_format_loss_weight(weights.box_dn)}) "
        f"keypoint_dn={_format_loss_weight(weights.keypoint_dn)} "
        f"lm={_format_loss_weight(weights.lm)}"
    )


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
        ("crowdpose_head", "neck"),
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


def select_informative_visualization_sample(
    outputs: dict[str, torch.Tensor],
    batch: dict,
    *,
    min_gt_area_ratio: float = 0.005,
    min_valid_keypoints: int = 3,
) -> int | None:
    """Choose a batch item that will produce a readable pose visualization."""

    box_mask = outputs.get("box_mask")
    best_sample: int | None = None
    best_score = -1.0
    for sample_idx, target in enumerate(batch.get("targets", [])):
        gt_boxes = target.get("boxes")
        gt_valid = target.get("keypoint_valid")
        if not torch.is_tensor(gt_boxes) or not torch.is_tensor(gt_valid):
            continue
        if gt_boxes.numel() == 0 or gt_valid.numel() == 0:
            continue
        valid_counts = gt_valid.detach().cpu().bool().sum(dim=-1)
        boxes = gt_boxes.detach().float().cpu()
        areas = (
            (boxes[:, 2] - boxes[:, 0]).clamp_min(0.0)
            * (boxes[:, 3] - boxes[:, 1]).clamp_min(0.0)
        )
        informative = (valid_counts >= int(min_valid_keypoints)) & (
            areas >= float(min_gt_area_ratio)
        )
        if not bool(informative.any()):
            continue
        if torch.is_tensor(box_mask):
            if sample_idx >= int(box_mask.shape[0]) or not bool(
                box_mask[sample_idx].detach().bool().any().item()
            ):
                continue
        sample_score = float(areas[informative].max().item())
        if sample_score > best_score:
            best_score = sample_score
            best_sample = sample_idx
    return best_sample


def save_pose_visualization(
    outputs: dict[str, torch.Tensor],
    batch: dict,
    output_path: Path,
    sample_idx: int = 0,
    max_instances: int = 8,
    score_threshold: float = 0.05,
    draw_all_schema_keypoints: bool = False,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vision_images = batch.get("vision_images")
    if vision_images is not None and sample_idx < len(vision_images):
        tensor = vision_images[sample_idx].detach().cpu()
        if tensor.dtype != torch.uint8:
            tensor = tensor.clamp(0, 255).to(torch.uint8)
        canvas = Image.fromarray(
            tensor.permute(1, 2, 0).contiguous().numpy(),
            mode="RGB",
        )
    else:
        image_path = batch["image_paths"][sample_idx]
        with Image.open(image_path) as image:
            canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size

    # Keep training visualizations deliberately minimal. ``boxes`` is the box
    # regression head output; input Locate/GT boxes and the expanded ROI boxes
    # remain available in tensors/metrics but are not drawn over the result.
    boxes = outputs["boxes"][sample_idx].detach().float().cpu()
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
    task_ids = batch.get("task_ids")
    task_id = (
        int(task_ids[sample_idx].detach().cpu().item())
        if torch.is_tensor(task_ids) and sample_idx < int(task_ids.numel())
        else -1
    )
    task_name = str(target.get("task", "")) or (
        "REF_POSE" if task_id == 1 else "ALL_POSE" if task_id == 0 else "unknown"
    )
    ref_texts = batch.get("ref_texts") or []
    ref_description = (
        str(ref_texts[sample_idx]).strip()
        if sample_idx < len(ref_texts)
        else ""
    )
    if task_name == "REF_POSE" and not ref_description:
        prompts = batch.get("prompts") or []
        if sample_idx < len(prompts):
            ref_description = _extract_ref_description_from_prompt(str(prompts[sample_idx]))
    edge_indices = SCHEMA_POSE_EDGE_INDICES.get(schema_name, DEFAULT_POSE_EDGE_INDICES)
    schema_keypoint_count = int(schema_valid.sum().item())
    gt_boxes = target["boxes"].detach().float().cpu()
    gt_keypoints = target["keypoints"].detach().float().cpu()
    gt_valid = target["keypoint_valid"].detach().cpu().bool()
    gt_draw_valid = (
        gt_valid
        if draw_all_schema_keypoints
        else gt_valid & (gt_keypoints[..., 2] > 0.5)
    )
    matched_gt_indices = target.get("matched_gt_indices")
    matched_gt_indices = (
        matched_gt_indices.detach().cpu().long()
        if torch.is_tensor(matched_gt_indices)
        else None
    )

    valid_indices = torch.nonzero(valid_boxes, as_tuple=False).flatten()
    if valid_indices.numel() > 0:
        ranked = valid_indices[torch.argsort(scores[valid_indices], descending=True)]
    else:
        ranked = torch.empty(0, dtype=torch.long)

    # Training-time person quality is intentionally low while pose coordinates
    # are still inaccurate. Do not let the prediction score threshold hide the
    # matched GT as well: matched queries are always selected and visualized,
    # followed by the highest-scoring unmatched predictions.
    if matched_gt_indices is not None:
        matched_mask = torch.zeros_like(valid_boxes)
        matched_count = min(
            int(matched_gt_indices.numel()),
            int(matched_mask.numel()),
        )
        matched_mask[:matched_count] = matched_gt_indices[:matched_count].ge(0)
    else:
        matched_mask = torch.arange(boxes.shape[0]) < int(gt_boxes.shape[0])
    matched_indices = torch.nonzero(valid_boxes & matched_mask, as_tuple=False).flatten()
    if matched_indices.numel() > 0:
        matched_areas = (
            (gt_boxes[matched_indices, 2] - gt_boxes[matched_indices, 0]).clamp_min(0.0)
            * (gt_boxes[matched_indices, 3] - gt_boxes[matched_indices, 1]).clamp_min(0.0)
        )
        matched_indices = matched_indices[
            torch.argsort(matched_areas, descending=True)
        ]

    selected: list[int] = []
    selected_set: set[int] = set()
    for idx in torch.cat((matched_indices, ranked)).tolist():
        idx = int(idx)
        if idx in selected_set:
            continue
        selected.append(idx)
        selected_set.add(idx)
        if len(selected) >= max(int(max_instances), 0):
            break

    for selection_rank, idx in enumerate(selected):
        has_gt = bool(matched_mask[idx].item())
        if has_gt:
            _draw_box(draw, gt_boxes[idx], width, height, (50, 200, 80), f"GT {idx}")
            _draw_pose(draw, gt_keypoints[idx], gt_draw_valid[idx], width, height, (50, 200, 80), edge_indices)

        # Always show predictions paired with GT, and keep at least the top
        # prediction visible for samples without a matched target. The score
        # threshold only suppresses additional unmatched low-confidence boxes.
        if not has_gt and scores[idx] < score_threshold and selection_rank > 0:
            continue
        _draw_box(
            draw,
            boxes[idx],
            width,
            height,
            (80, 180, 255),
            f"Pred {idx} score={float(scores[idx]):.2f}",
        )
        predicted_valid = (
            schema_valid
            if draw_all_schema_keypoints
            else schema_valid & (keypoints[idx, :, 2] >= score_threshold)
        )
        _draw_pose(draw, keypoints[idx], predicted_valid, width, height, (240, 70, 70), edge_indices)

    header_width = min(width, 1100)
    is_refhuman = task_name == "REF_POSE" or bool(ref_description)
    header_height = 64 if is_refhuman else 44
    draw.rectangle([0, 0, header_width, header_height], fill=(0, 0, 0))
    draw.text(
        (4, 4),
        f"dataset={dataset_name} schema={schema_name} keypoints={schema_keypoint_count} task={task_name}",
        fill=(255, 255, 255),
    )
    draw.text(
        (4, 24),
        "green=GT box/pose | blue=predicted box | red=predicted pose",
        fill=(255, 255, 255),
    )
    if is_refhuman:
        description = " ".join(ref_description.split()) or "person"
        # The default PIL font is roughly seven pixels per ASCII character.
        # Truncate long RefHuman captions rather than covering the image.
        max_chars = max((header_width - 12) // 7, 8)
        if len(description) > max_chars:
            description = description[: max(max_chars - 3, 1)].rstrip() + "..."
        draw.text((4, 44), f"ref: {description}", fill=(255, 230, 120))
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
        backbone_base = (
            get_eagle_base_model(module.qwen_model)
            if str(getattr(module, "backbone_name", "")) == "eagle" and module.qwen_model is not None
            else None
        )
        feature_config = {
            "output_size": int(module.qwen_extractor.output_size),
            "feature_source": str(getattr(module.qwen_extractor, "feature_source", "raw_visual")),
            "backbone_train_scope": str(getattr(module, "backbone_train_scope", "all_lora")),
            "backbone_llm_layers": str(getattr(module, "backbone_llm_layers", "")),
            "backbone_vision_layers": str(getattr(module, "backbone_vision_layers", "")),
            "backbone_llm_modules": str(getattr(module, "backbone_llm_modules", "")),
            "backbone_vision_modules": str(getattr(module, "backbone_vision_modules", "")),
            "backbone_load_mode": (
                "vision_tower_only"
                if bool(getattr(backbone_base, "is_vision_only_backbone", False))
                else "full_multimodal"
            ),
            "generation_components_pruned": bool(
                getattr(backbone_base, "generation_components_pruned", False)
            ),
        }
        payload["backbone_feature_config"] = feature_config
        payload["qwen_feature_config"] = feature_config
        extractor_state = {
            key: value.detach().cpu()
            for key, value in module.qwen_extractor.state_dict().items()
            if not key.startswith(("qwen_model.", "eagle_model."))
        }
        payload["backbone_extractor"] = extractor_state
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
    if module.qwen_extractor is not None and "backbone_extractor" in payload:
        module.qwen_extractor.load_state_dict(payload["backbone_extractor"], strict=False)
    elif module.qwen_extractor is not None and ("backbone_feature_refiner" in payload or "qwen_feature_refiner" in payload):
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


def load_person_confidence_rescue_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
) -> tuple[Path, dict[str, object]]:
    """Strictly restore every legacy tensor and initialize only the new head."""
    resolved_path = resolve_training_checkpoint(checkpoint_path)
    payload_path = resolved_path / CHECKPOINT_PAYLOAD_NAME if resolved_path.is_dir() else resolved_path
    payload = torch.load(payload_path, map_location="cpu")
    module = unwrap_training_model(model)
    incompatible = module.pose_model.load_state_dict(payload["model"], strict=False)
    expected_missing = {
        f"person_confidence_head.{key}"
        for key in module.pose_model.person_confidence_head.state_dict()
    }
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    if missing != expected_missing or unexpected:
        raise RuntimeError(
            "Legacy rescue checkpoint is not architecture-compatible: "
            f"missing={sorted(missing)}, expected_missing={sorted(expected_missing)}, "
            f"unexpected={sorted(unexpected)}"
        )

    backbone_state = payload.get("backbone_trainable", payload.get("qwen_trainable"))
    if module.qwen_model is not None and backbone_state is not None:
        module.qwen_model.load_state_dict(backbone_state, strict=False)
    if module.qwen_extractor is not None and "backbone_extractor" in payload:
        module.qwen_extractor.load_state_dict(payload["backbone_extractor"], strict=False)
    if module.qwen_extractor is not None and (
        "backbone_feature_refiner" in payload or "qwen_feature_refiner" in payload
    ):
        refiner_state = (
            payload["backbone_feature_refiner"]
            if "backbone_feature_refiner" in payload
            else payload["qwen_feature_refiner"]
        )
        module.qwen_extractor.feature_refiner.load_state_dict(refiner_state, strict=True)

    module.pose_model.initialize_person_confidence_from_visibility()
    for parameter in module.parameters():
        parameter.requires_grad = False
    for parameter in module.pose_model.person_confidence_head.parameters():
        parameter.requires_grad = True
    return resolved_path, payload


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
        if config["bf16"]["enabled"]:
            # DeepSpeed defaults this to false for bf16.  With ZeRO-1/2 the
            # reduced/partitioned gradients live in ``averaged_gradients`` and
            # are therefore not completely covered by scanning Parameter.grad
            # from the training loop.  Enabling the optimizer-side check makes
            # DeepSpeed reject the update before a single bad rank can poison
            # the fp32 master weights during all-gather.
            config["bf16"]["check_grad_overflow"] = True
    if "fp16" in config:
        config["fp16"]["enabled"] = dtype_str == "float16"
    config.setdefault("steps_per_print", 100000)
    return config


def is_vision_parameter(name: str) -> bool:
    """Check if a parameter belongs to the vision encoder (Qwen or Eagle)."""
    return (
        name.startswith(("visual.", "vision_model."))
        or ".visual." in name
        or ".vision_model." in name
        or "qwen_model.visual." in name
        or "backbone_model.vision_model." in name
    )


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
    backbone_name = getattr(args, "backbone", "qwen3vl")
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

    if backbone_name == "eagle":
        lora_lr_scale = getattr(args, "locate_llm_scale", 0.01)
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
        {
            "params": params,
            "lr": lrs[name],
            "weight_decay": args.weight_decay if name == "pose" else 0.0,
        }
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
    sharing_strategy = os.environ.get("QWENPOSE_MP_SHARING_STRATEGY", "file_system").strip()
    if sharing_strategy not in {"file_descriptor", "file_system"}:
        raise ValueError(
            "QWENPOSE_MP_SHARING_STRATEGY must be file_descriptor or file_system, "
            f"got {sharing_strategy!r}."
        )
    torch.multiprocessing.set_sharing_strategy(sharing_strategy)
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.grad_accum_steps <= 0:
        raise ValueError("--grad_accum_steps must be positive.")
    if args.epochs < 0:
        raise ValueError("--epochs must be non-negative.")
    if args.max_steps < 0:
        raise ValueError("--max_steps must be non-negative.")
    if args.epochs == 0 and args.max_steps == 0:
        raise ValueError("At least one of --epochs or --max_steps must be positive.")
    if args.refhuman_max_captions_per_instance < 0:
        raise ValueError("--refhuman_max_captions_per_instance must be >= 0.")
    if args.box_condition_scale <= 0:
        raise ValueError("--box_condition_scale must be positive.")
    if args.image_size != 800:
        raise ValueError("The unified LocatePose architecture requires --image_size 800.")
    if args.pose_pyramid_channels <= 0 or args.pose_pyramid_blocks <= 0:
        raise ValueError("Pose pyramid channels and block count must be positive.")
    if args.human_decoder_layers <= 0 or args.pose_decoder_layers <= 0:
        raise ValueError("Human and pose decoder layer counts must be positive.")
    if args.deformable_points <= 0 or args.deformable_min_radius_cells <= 0.0:
        raise ValueError("Deformable sampling points and minimum radius must be positive.")
    if args.max_dn_queries < 0 or args.max_dn_groups <= 0:
        raise ValueError("DN query count must be non-negative and DN groups positive.")
    if args.dn_positive_noise < 0.0 or args.dn_negative_noise < 0.0:
        raise ValueError("DN noise scales must be non-negative.")
    if args.max_keypoint_dn_queries < 0 or args.max_keypoint_dn_groups <= 0:
        raise ValueError("Keypoint-DN query count must be non-negative and groups positive.")
    for range_name, lower, upper in (
        (
            "positive",
            args.keypoint_dn_positive_ks_min,
            args.keypoint_dn_positive_ks_max,
        ),
        (
            "negative",
            args.keypoint_dn_negative_ks_min,
            args.keypoint_dn_negative_ks_max,
        ),
    ):
        if not 0.0 < lower <= upper <= 1.0:
            raise ValueError(
                f"Keypoint-DN {range_name} KS range must satisfy 0 < min <= max <= 1."
            )
    if args.legacy_checkpoint_compat or args.person_confidence_rescue:
        raise ValueError(
            "Legacy checkpoint compatibility/rescue is not available after replacing the RGB branches. "
            "Train a new Stage1 checkpoint."
        )
    if not 0.0 <= args.pose_dropout < 1.0:
        raise ValueError("--pose_dropout must be in [0, 1).")
    if (
        args.pose_coordinate_init == "schema_prior"
        and args.schema_joint_priors_path
        and not Path(args.schema_joint_priors_path).is_file()
    ):
        raise FileNotFoundError(
            f"--schema_joint_priors_path does not exist: {args.schema_joint_priors_path}"
        )
    if args.pose_roi_size <= 1:
        raise ValueError("--pose_roi_size must be greater than 1.")
    if not 0.0 <= args.visualize_min_gt_area_ratio <= 1.0:
        raise ValueError("--visualize_min_gt_area_ratio must be in [0, 1].")
    if min(
        args.w_box_objectness,
        args.w_box_l1,
        args.w_box_giou,
        args.w_box_relative,
        args.w_box_dn,
    ) < 0.0:
        raise ValueError("All box refinement and denoising loss weights must be non-negative.")
    if args.w_keypoint_confidence < 0.0:
        raise ValueError("--w_keypoint_confidence must be non-negative.")
    if args.w_keypoint_dn < 0.0:
        raise ValueError("--w_keypoint_dn must be non-negative.")
    if args.w_person_confidence < 0.0:
        raise ValueError("--w_person_confidence must be non-negative.")
    if args.w_ref_match < 0.0:
        raise ValueError("--w_ref_match must be non-negative.")
    if args.w_locate_box_lm < 0.0:
        raise ValueError("--w_locate_box_lm must be non-negative.")
    if args.locate_box_max_new_tokens <= 0:
        raise ValueError("--locate_box_max_new_tokens must be positive.")
    if args.locate_lm_loss_every <= 0:
        raise ValueError("--locate_lm_loss_every must be positive.")
    if args.locate_lm_max_instances <= 0:
        raise ValueError("--locate_lm_max_instances must be positive.")
    if args.ref_text_scale < 0.0:
        raise ValueError("--ref_text_scale must be non-negative.")
    if args.locate_vision_scale < 0.0 or args.locate_llm_scale < 0.0:
        raise ValueError("--locate_vision_scale/--locate_llm_scale must be non-negative.")
    if args.locate_train_scope == "selective_lora":
        llm_layers = parse_layer_selection(args.locate_llm_layers)
        vision_layers = parse_layer_selection(args.locate_vision_layers)
        llm_modules = parse_module_selection(args.locate_llm_modules)
        vision_modules = parse_module_selection(args.locate_vision_modules)
        if not llm_layers or max(llm_layers) >= 36:
            raise ValueError("--locate_llm_layers must select Qwen2.5 layers in [0, 35].")
        if not vision_layers or max(vision_layers) >= 27:
            raise ValueError("--locate_vision_layers must select MoonViT blocks in [0, 26].")
        if not llm_modules or not vision_modules:
            raise ValueError("Selective LoRA module selections must be non-empty.")
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
    if args.prune_locate_generation and args.backbone != "eagle":
        raise ValueError("--prune_locate_generation is only supported by LocateAnything/eagle.")
    if args.box_source == "locate_generate" and args.backbone != "eagle":
        raise ValueError("--box_source=locate_generate requires --backbone eagle/locatepose.")
    if args.prune_locate_generation and (
        args.box_source == "locate_generate" or args.w_locate_box_lm > 0.0
    ):
        raise ValueError(
            "LocateAnything generation/LM supervision requires its lm_head; "
            "use --no-prune_locate_generation."
        )
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
    if args.eagle_image_token_limit is not None and args.eagle_image_token_limit <= 0:
        raise ValueError("--locate_image_token_limit/--eagle_image_token_limit must be positive when set.")
    if args.eagle_batch_token_limit is not None and args.eagle_batch_token_limit <= 0:
        raise ValueError("--locate_batch_token_limit/--eagle_batch_token_limit must be positive when set.")
    probability_args = {
        "augment_flip_prob": args.augment_flip_prob,
        "augment_affine_prob": args.augment_affine_prob,
        "augment_color_prob": args.augment_color_prob,
        "augment_grayscale_prob": args.augment_grayscale_prob,
        "augment_blur_prob": args.augment_blur_prob,
        "augment_erase_prob": args.augment_erase_prob,
    }
    for name, value in probability_args.items():
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"--{name} must be in [0, 1].")
    if args.augment_scale_min <= 0.0 or args.augment_scale_max < args.augment_scale_min:
        raise ValueError("--augment_scale_min/max must satisfy 0 < min <= max.")
    if args.augment_rotate_degrees < 0.0 or args.augment_translate_fraction < 0.0:
        raise ValueError("--augment_rotate_degrees and --augment_translate_fraction must be non-negative.")
    if min(args.augment_brightness, args.augment_contrast, args.augment_saturation, args.augment_hue) < 0.0:
        raise ValueError("Color augmentation magnitudes must be non-negative.")
    if args.augment_hue > 0.5:
        raise ValueError("--augment_hue must not exceed 0.5.")
    if args.augment_blur_sigma_min < 0.0 or args.augment_blur_sigma_max < args.augment_blur_sigma_min:
        raise ValueError("--augment_blur_sigma_min/max must satisfy 0 <= min <= max.")
    if not 0.0 <= args.augment_erase_area_min <= args.augment_erase_area_max <= 1.0:
        raise ValueError("--augment_erase_area_min/max must satisfy 0 <= min <= max <= 1.")
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
    if args.locate_feature_source == "vision_only" and "refhuman" in dataset_names:
        raise ValueError(
            "--locate_feature_source=vision_only cannot train RefHuman text conditioning. "
            "Use pose-only datasets in Stage 1 and add RefHuman in raw_visual Stage 3."
        )
    if args.pose_augment and args.locate_feature_source != "vision_only":
        raise ValueError(
            "--pose_augment is restricted to vision-only Stage 1 so geometric transforms "
            "cannot contradict RefHuman left/right language in the full-model stages."
        )
    augment_config = PoseAugmentConfig(
        enabled=bool(args.pose_augment),
        horizontal_flip_prob=float(args.augment_flip_prob),
        affine_prob=float(args.augment_affine_prob),
        rotate_degrees=float(args.augment_rotate_degrees),
        scale_min=float(args.augment_scale_min),
        scale_max=float(args.augment_scale_max),
        translate_fraction=float(args.augment_translate_fraction),
        color_prob=float(args.augment_color_prob),
        brightness=float(args.augment_brightness),
        contrast=float(args.augment_contrast),
        saturation=float(args.augment_saturation),
        hue=float(args.augment_hue),
        grayscale_prob=float(args.augment_grayscale_prob),
        blur_prob=float(args.augment_blur_prob),
        blur_sigma_min=float(args.augment_blur_sigma_min),
        blur_sigma_max=float(args.augment_blur_sigma_max),
        erase_prob=float(args.augment_erase_prob),
        erase_area_min=float(args.augment_erase_area_min),
        erase_area_max=float(args.augment_erase_area_max),
    )
    dataset = build_datasets(
        dataset_root=args.dataset_root,
        names=dataset_names,
        max_instances=args.max_instances,
        image_size=args.image_size,
        load_image_tensors=not args.disable_image_tensors,
        # Only materialize and transfer a full-resolution image tensor when
        # augmentation changes the pixels. Without augmentation, reopening the
        # path in the main process is faster and avoids large worker IPC payloads.
        load_vision_images=args.backbone == "eagle" and bool(args.pose_augment),
        augment_config=augment_config,
        use_prompts=args.locate_feature_source != "vision_only",
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
        balance_vision_tokens = (
            args.backbone in {"eagle", "locatepose"}
            and not args.disable_vision_token_balancing
        )
        sampler_token_limit = args.eagle_image_token_limit
        if args.eagle_batch_token_limit is not None and int(args.eagle_batch_token_limit) > 0:
            per_image_limit = max(
                int(int(args.eagle_batch_token_limit) / max(int(args.batch_size), 1) * 0.875),
                64,
            )
            sampler_token_limit = (
                per_image_limit
                if sampler_token_limit is None
                else min(int(sampler_token_limit), per_image_limit)
            )
        batch_sampler = HomogeneousDatasetBatchSampler(
            dataset,
            args.batch_size,
            seed=args.seed,
            rank=rank,
            world_size=world_size,
            shuffle=True,
            fill_last=True,
            balance_vision_tokens=balance_vision_tokens,
            vision_token_limit=sampler_token_limit,
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
        # Augmented Locate batches may contain original-resolution uint8 tensors;
        # pinning those variable-size tensors brings little benefit and has caused
        # sporadic multi-rank stalls in practice.
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
            source_batch_counts = [
                math.ceil(
                    (
                        dataset.sample_count_for_epoch(dataset_idx, 0)
                        if hasattr(dataset, "sample_count_for_epoch")
                        else len(inner_dataset)
                    )
                    / max(int(args.batch_size) * max(int(world_size), 1), 1)
                )
                for dataset_idx, inner_dataset in enumerate(getattr(dataset, "datasets", []))
            ]
            total_source_batches = max(sum(source_batch_counts), 1)
            source_batch_desc = ",".join(
                f"{name}:{count}({count / total_source_batches:.1%})"
                for name, count in zip(
                    getattr(dataset, "names", []), source_batch_counts
                )
            )
            print(
                "Homogeneous dataset batches enabled: "
                f"one source per batch, batches_per_rank={len(batch_sampler)}, "
                f"vision_token_balancing={batch_sampler.balance_vision_tokens}, "
                f"sampler_token_limit={batch_sampler.vision_token_limit}, "
                f"global_source_batches=[{source_batch_desc}]"
            )
    if sampler is not None:
        sampler.set_epoch(0)
    first_batch_started = time.perf_counter()
    first_batch = next(iter(loader))
    validate_pose_batch_contract(first_batch)
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
        eagle_loader = (
            load_eagle_vision_only_with_lora
            if args.locate_feature_source == "vision_only"
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
                gradient_checkpointing=args.eagle_gradient_checkpointing,
                prune_generation=bool(args.prune_locate_generation),
            )
        )
        backbone_model.to(device)
        backbone_model.train()
        external_dim = eagle_hidden_size(backbone_model)
        bb_trainable, bb_total = count_eagle_lora_parameters(backbone_model)
        if is_main_process():
            eagle_base = get_eagle_base_model(backbone_model)
            load_mode = (
                "vision_tower_only"
                if bool(getattr(eagle_base, "is_vision_only_backbone", False))
                else "full_multimodal"
            )
            print(f"Eagle load mode: {load_mode}")
            print(
                "Eagle generation components: "
                + (
                    "pruned (lm_head checkpoint tensor skipped; KV cache disabled)"
                    if bool(getattr(eagle_base, "generation_components_pruned", False))
                    else "available"
                )
            )
            print(f"Eagle LoRA parameters available before stage scope: {bb_trainable:,} / {bb_total:,}")
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

    if args.init_from_checkpoint is not None:
        init_payload_path = resolve_training_checkpoint(args.init_from_checkpoint)
        if init_payload_path.is_dir():
            init_payload_path = init_payload_path / CHECKPOINT_PAYLOAD_NAME
        qwenpose_init_payload = _load_local_torch_payload(init_payload_path)
    saved_pose_config = qwenpose_init_payload.get("pose_config") if qwenpose_init_payload is not None else None
    if saved_pose_config is not None:
        if "pose_pyramid_channels" not in saved_pose_config or int(saved_pose_config.get("rgb_input_size", 0)) != 800:
            raise ValueError(
                "The initialization checkpoint predates the unified 800x800 pose pyramid and cannot be loaded. "
                "Train a new Stage1 checkpoint, then use it for Stage2/Stage3."
            )
        if (
            "enable_keypoint_denoising" not in saved_pose_config
            and not args.disable_keypoint_denoising
        ):
            raise ValueError(
                "The initialization checkpoint predates keypoint denoising. "
                "For fully matched Stage1/Stage2 training, retrain Stage1 with "
                "the current script. To intentionally keep the legacy graph, "
                "pass --disable_keypoint_denoising."
            )
        saved_coordinate_init = str(
            saved_pose_config.get("pose_coordinate_init", "schema_prior")
        ).strip().lower()
        if saved_coordinate_init != args.pose_coordinate_init:
            raise ValueError(
                "Pose coordinate initialization must match between stages/checkpoints: "
                f"checkpoint={saved_coordinate_init}, requested={args.pose_coordinate_init}. "
                "Retrain Stage1 with learned_spread for the new image-conditioned path; "
                "schema_prior is legacy-only."
            )
        model_config = QwenPoseConfig(
            hidden_dim=int(saved_pose_config.get("hidden_dim", args.hidden_dim)),
            external_dim=external_dim,
            pose_decoder_layers=int(saved_pose_config.get("pose_decoder_layers", args.pose_decoder_layers)),
            refinement_steps=int(saved_pose_config.get("refinement_steps", args.refinement_steps)),
            decoder_heads=int(saved_pose_config.get("decoder_heads", args.decoder_heads)),
            dropout=float(saved_pose_config.get("dropout", args.pose_dropout)),
            box_condition_scale=float(saved_pose_config.get("box_condition_scale", args.box_condition_scale)),
            pose_roi_size=int(saved_pose_config.get("pose_roi_size", args.pose_roi_size)),
            use_refinement=bool(saved_pose_config.get("use_refinement", not args.disable_refinement)),
            rgb_input_size=int(saved_pose_config.get("rgb_input_size", 800)),
            pose_pyramid_channels=int(saved_pose_config.get("pose_pyramid_channels", args.pose_pyramid_channels)),
            pose_pyramid_blocks=int(saved_pose_config.get("pose_pyramid_blocks", args.pose_pyramid_blocks)),
            human_decoder_layers=int(saved_pose_config.get("human_decoder_layers", args.human_decoder_layers)),
            deformable_points=int(saved_pose_config.get("deformable_points", args.deformable_points)),
            deformable_min_radius_cells=float(
                saved_pose_config.get("deformable_min_radius_cells", args.deformable_min_radius_cells)
            ),
            enable_box_denoising=bool(
                saved_pose_config.get("enable_box_denoising", not args.disable_box_denoising)
            ),
            enable_keypoint_denoising=bool(
                saved_pose_config.get(
                    "enable_keypoint_denoising", not args.disable_keypoint_denoising
                )
            ),
            ref_text_scale=float(
                saved_pose_config.get("ref_text_scale", args.ref_text_scale)
            ),
            enable_ref_visual_modulation=bool(
                saved_pose_config.get(
                    "enable_ref_visual_modulation",
                    not args.disable_ref_visual_modulation,
                )
            ),
            legacy_checkpoint_compat=False,
            enable_person_confidence_head=bool(
                saved_pose_config.get("enable_person_confidence_head", True)
            ),
            person_confidence_rescue=False,
            use_global_person_queries=bool(
                saved_pose_config.get("use_global_person_queries", False)
            ),
            num_person_queries=int(
                saved_pose_config.get("num_person_queries", args.num_person_queries)
            ),
            pose_coordinate_init=saved_coordinate_init,
            schema_joint_priors_path=str(
                saved_pose_config.get(
                    "schema_joint_priors_path", args.schema_joint_priors_path
                )
            ),
        )
    else:
        model_config = QwenPoseConfig(
            hidden_dim=args.hidden_dim,
            external_dim=external_dim,
            pose_decoder_layers=args.pose_decoder_layers,
            refinement_steps=args.refinement_steps,
            decoder_heads=args.decoder_heads,
            dropout=args.pose_dropout,
            box_condition_scale=args.box_condition_scale,
            pose_roi_size=args.pose_roi_size,
            use_refinement=not args.disable_refinement,
            rgb_input_size=args.image_size,
            pose_pyramid_channels=args.pose_pyramid_channels,
            pose_pyramid_blocks=args.pose_pyramid_blocks,
            human_decoder_layers=args.human_decoder_layers,
            deformable_points=args.deformable_points,
            deformable_min_radius_cells=args.deformable_min_radius_cells,
            enable_box_denoising=not args.disable_box_denoising,
            enable_keypoint_denoising=not args.disable_keypoint_denoising,
            ref_text_scale=args.ref_text_scale,
            enable_ref_visual_modulation=not args.disable_ref_visual_modulation,
            legacy_checkpoint_compat=False,
            enable_person_confidence_head=bool(args.enable_person_confidence_head),
            person_confidence_rescue=False,
            use_global_person_queries=args.box_source == "person_queries",
            num_person_queries=args.num_person_queries,
            pose_coordinate_init=args.pose_coordinate_init,
            schema_joint_priors_path=args.schema_joint_priors_path,
        )
    requested_person_queries = args.box_source == "person_queries"
    if bool(model_config.use_global_person_queries) != requested_person_queries:
        raise ValueError(
            "Initialization checkpoint and --box_source disagree about global person queries: "
            f"checkpoint={model_config.use_global_person_queries}, box_source={args.box_source}."
        )
    if requested_person_queries and int(model_config.num_person_queries) != int(args.num_person_queries):
        raise ValueError(
            "--num_person_queries must match the initialization checkpoint: "
            f"checkpoint={model_config.num_person_queries}, requested={args.num_person_queries}."
        )
    person_head_enabled = bool(
        model_config.enable_person_confidence_head
        or model_config.person_confidence_rescue
    )
    if not args.person_confidence_rescue and person_head_enabled and args.w_person_confidence <= 0.0:
        raise ValueError(
            "A person confidence head is enabled but --w_person_confidence is zero. "
            "Refusing to train a head that would later become the official COCO ranking score."
        )
    if args.w_person_confidence > 0.0 and not person_head_enabled:
        raise ValueError(
            "--w_person_confidence > 0 requires an enabled person confidence head."
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
            feature_source=args.locate_feature_source,
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
        backbone_train_scope=args.locate_train_scope,
        train_backbone_projector=args.train_locate_projector,
        backbone_llm_layers=args.locate_llm_layers,
        backbone_vision_layers=args.locate_vision_layers,
        backbone_llm_modules=args.locate_llm_modules,
        backbone_vision_modules=args.locate_vision_modules,
    ).to(device)
    if args.person_confidence_rescue:
        if args.init_from_checkpoint is None:
            raise ValueError("--person_confidence_rescue requires --init_from_checkpoint.")
        if args.resume_from_checkpoint is not None:
            raise ValueError(
                "Start confidence rescue with --init_from_checkpoint; reserve "
                "--resume_from_checkpoint for checkpoints created by the rescue run."
            )
        if args.box_source != "gt":
            raise ValueError("Initial confidence rescue is defined for --box_source gt only.")
        rescue_checkpoint, _ = load_person_confidence_rescue_checkpoint(
            training_model,
            args.init_from_checkpoint,
        )
        rescue_trainable, rescue_total = count_trainable_parameters(training_model)
        expected_trainable = sum(
            parameter.numel()
            for parameter in training_model.pose_model.person_confidence_head.parameters()
        )
        if rescue_trainable != expected_trainable:
            raise RuntimeError(
                "Confidence rescue freeze invariant failed: "
                f"trainable={rescue_trainable:,}, expected={expected_trainable:,}."
            )
        if is_main_process():
            print(
                "Initialized legacy confidence rescue from "
                f"{rescue_checkpoint}; trainable={rescue_trainable:,}/{rescue_total:,} "
                "(person_confidence_head only)."
            )
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
        if backbone_name == "eagle":
            print(
                "Locate image budget: "
                f"image_token_limit={args.eagle_image_token_limit}, "
                f"batch_token_limit={args.eagle_batch_token_limit}"
            )
        print(f"Backbone train scope: {training_model.backbone_train_scope}")
        print(f"Backbone trainable counts: {training_model.backbone_trainable_counts}")
        if backbone_name == "eagle":
            print(f"Locate feature source: {args.locate_feature_source}")
        print(f"Pose dropout: {model_config.dropout}")
        print(
            "Pose augmentation: "
            f"enabled={args.pose_augment}, flip={args.augment_flip_prob}, "
            f"affine={args.augment_affine_prob}, rotate=±{args.augment_rotate_degrees}, "
            f"scale=[{args.augment_scale_min},{args.augment_scale_max}], "
            f"translate=±{args.augment_translate_fraction}"
        )
        print(f"Box condition scale: {model_config.box_condition_scale}")
        print(f"Pose ROI size: {model_config.pose_roi_size}x{model_config.pose_roi_size}")
        print(
            "Unified pose pyramid: "
            f"input={model_config.rgb_input_size}x{model_config.rgb_input_size}, "
            f"channels={model_config.pose_pyramid_channels}, "
            f"grids={model_config.rgb_input_size // 4}x{model_config.rgb_input_size // 4}/"
            f"{model_config.rgb_input_size // 8}x{model_config.rgb_input_size // 8}/"
            f"{model_config.rgb_input_size // 16}x{model_config.rgb_input_size // 16}, "
            f"blocks={model_config.pose_pyramid_blocks}, "
            f"human_decoder_layers={model_config.human_decoder_layers}, "
            f"pose_decoder_layers={model_config.pose_decoder_layers}"
        )
        print(
            "Box denoising: "
            f"enabled={model_config.enable_box_denoising}, "
            f"max_queries={args.max_dn_queries}, groups={args.max_dn_groups}, "
            f"positive_noise={args.dn_positive_noise}, negative_noise={args.dn_negative_noise}"
        )
        print(
            "Keypoint denoising: "
            f"enabled={model_config.enable_keypoint_denoising}, "
            f"max_queries={args.max_keypoint_dn_queries}, "
            f"groups={args.max_keypoint_dn_groups}, "
            f"positive_ks=[{args.keypoint_dn_positive_ks_min},{args.keypoint_dn_positive_ks_max}], "
            f"negative_ks=[{args.keypoint_dn_negative_ks_min},{args.keypoint_dn_negative_ks_max}], "
            f"weight={args.w_keypoint_dn}"
        )
        if model_config.pose_coordinate_init == "learned_spread":
            coordinate_message = (
                "mode=learned_spread, main_reference=trainable_nonsemantic_halton, "
                "deform_reference=detached_coarse"
            )
        elif model_config.pose_coordinate_init == "box_center":
            coordinate_message = (
                "mode=box_center (ablation), main_reference=box_center, "
                "deform_reference=detached_coarse"
            )
        else:
            coordinate_message = "mode=schema_prior (legacy)"
        print(f"Pose coordinate initialization: {coordinate_message}")
        print(
            "RefHuman conditioning: "
            f"text_scale={model_config.ref_text_scale}, "
            f"visual_modulation={model_config.enable_ref_visual_modulation and not model_config.use_global_person_queries}, "
            f"match_loss_weight={args.w_ref_match}"
        )
        print(f"Box source: {args.box_source}")
        if args.box_source != "person_queries":
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
    unified_config = LocatePoseUnifiedConfig.from_args(
        args,
        use_single_pass_features=False,
    )
    unified_runtime = LocatePoseUnifiedRuntime(
        active_model,
        backbone_processor,
        device,
        backbone_name=backbone_name,
    )
    max_steps = int(args.max_steps)
    steps_per_epoch = math.ceil(len(loader) / max(int(args.grad_accum_steps), 1))
    if int(args.epochs) > 0:
        total_epochs = int(args.epochs)
        step_only_training = False
    else:
        # Optimizer-step-only mode. The accumulation cursor is global across
        # epoch boundaries, so derive a conservative epoch count from micro
        # batches and let max_steps stop the loop exactly at the requested step.
        total_epochs = max(
            math.ceil(
                max_steps * max(int(args.grad_accum_steps), 1)
                / max(len(loader), 1)
            ),
            1,
        )
        step_only_training = True
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
        image_coord=args.w_image_coord,
        keypoint_confidence=args.w_keypoint_confidence,
        person_confidence=args.w_person_confidence,
        ref_match=args.w_ref_match,
        lm=(0.0 if backbone_name == "eagle" else args.w_lm),
        hard_joint=args.w_hard_joint,
        hard_joint_fraction=args.hard_joint_fraction,
        box_objectness=args.w_box_objectness,
        box_l1=args.w_box_l1,
        box_giou=args.w_box_giou,
        box_relative=args.w_box_relative,
        box_dn=args.w_box_dn,
        keypoint_dn=args.w_keypoint_dn,
        coarse_coord=args.w_coarse_coord,
        deform_coord=args.w_deform_coord,
        refine_coords=parse_float_list(args.w_refine_coords),
    )
    if is_main_process():
        duration = (
            f"optimizer_steps={max_steps}, derived_epoch_limit={total_epochs}"
            if step_only_training
            else f"epochs={total_epochs}, max_steps={max_steps or 'disabled'}"
        )
        print(f"Training duration: {duration}")
        print(format_loss_weights(weights))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    global_step = 0
    micro_step = 0
    last_saved_step = -1
    resume_epoch = 0
    resume_batch_in_epoch = 0
    resume_rng_state: dict[str, object] | None = None
    resume_state_inferred = False
    loss_ema: float | None = None
    loss_ema_decay: float = 0.98
    dataset_metric_ema: dict[str, dict[str, float]] = {}
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
    grad_diagnostics = os.environ.get("QWENPOSE_GRAD_DIAGNOSTICS", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if grad_diagnostics and is_main_process():
        print("Per-micro-step ZeRO/output gradient diagnostics enabled.")
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
        set_pose_dataset_epoch(dataset, epoch)
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
                desc=f"Epoch {epoch + 1:02d}/{total_epochs}",
                unit="batch",
                dynamic_ncols=True,
                mininterval=0.5,
                smoothing=0.05,
                initial=batches_to_skip,
                bar_format=(
                    "{desc} {percentage:3.0f}%|{bar:28}| {n_fmt}/{total_fmt} "
                    "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
                ),
            )
        last_iter_end = time.perf_counter()
        timing_sums = {"data": 0.0, "prep": 0.0, "fwd": 0.0, "bwd": 0.0, "micro": 0}
        if batch_sampler is not None:
            batch_sampler.set_start_batch(batches_to_skip)
        batch_iterator = iter(loader)
        if batches_to_skip > 0 and batch_sampler is None:
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
                target_boxes, target_box_mask, pose_targets = unified_runtime.prepare_training_conditioning(
                    batch,
                    unified_config,
                    box_source=args.box_source,
                )
                dn_batch: dict[str, torch.Tensor] = {}
                box_dn_batch: dict[str, torch.Tensor] | None = None
                if not args.disable_box_denoising:
                    box_dn_batch = prepare_box_denoising(
                        pose_targets,
                        device,
                        max_queries=args.max_dn_queries,
                        max_groups=args.max_dn_groups,
                        positive_noise=args.dn_positive_noise,
                        negative_noise=args.dn_negative_noise,
                        image_size=args.image_size,
                    )
                    if box_dn_batch is not None:
                        dn_batch.update(box_dn_batch)
                if not args.disable_keypoint_denoising and box_dn_batch is not None:
                    keypoint_dn_batch = prepare_keypoint_denoising(
                        pose_targets,
                        device,
                        max_queries=args.max_keypoint_dn_queries,
                        max_groups=args.max_keypoint_dn_groups,
                        positive_ks_min=args.keypoint_dn_positive_ks_min,
                        positive_ks_max=args.keypoint_dn_positive_ks_max,
                        negative_ks_min=args.keypoint_dn_negative_ks_min,
                        negative_ks_max=args.keypoint_dn_negative_ks_max,
                        image_size=args.image_size,
                    )
                    keypoint_dn_batch = pair_keypoint_denoising_with_box_denoising(
                        box_dn_batch,
                        keypoint_dn_batch,
                    )
                    if keypoint_dn_batch is not None:
                        dn_batch.update(keypoint_dn_batch)
                if backbone_processor is None:
                    qwen_inputs = None
                elif backbone_name == "eagle":
                    qwen_inputs = build_eagle_inputs(
                        backbone_processor,
                        batch["image_paths"],
                        None if args.locate_feature_source == "vision_only" else batch["prompts"],
                        device,
                        image_token_limit=args.eagle_image_token_limit,
                        batch_token_limit=args.eagle_batch_token_limit,
                        image_tensors=batch.get("vision_images"),
                    )
                    use_lm_loss = (
                        args.w_locate_box_lm > 0.0
                        and not args.disable_locate_grounding_aux
                        and not args.freeze_eagle
                        and args.locate_lm_loss_every > 0
                        and micro_step % args.locate_lm_loss_every == 0
                    )
                    if use_lm_loss:
                        locate_responses = build_locate_grounding_responses(
                            batch,
                            max_instances=args.locate_lm_max_instances,
                        )
                        qwen_lm_inputs = build_eagle_lm_inputs(
                            backbone_processor,
                            batch["image_paths"],
                            build_locate_generation_prompts(batch),
                            locate_responses,
                            device,
                            image_token_limit=args.eagle_image_token_limit,
                            image_tensors=batch.get("vision_images"),
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
                        **dn_batch,
                    )
                    outputs_finite, bad_output_names = synchronized_finite_check(
                        iter_named_floating_tensors(outputs),
                        device,
                    )
                    if not outputs_finite:
                        local_detail = ", ".join(bad_output_names) if bad_output_names else "nonfinite values reported by another rank"
                        raise FloatingPointError(
                            f"Non-finite model output before loss at step={global_step} micro={micro_step}: {local_detail}"
                        )
                    if args.box_source == "person_queries":
                        pose_targets = align_targets_to_person_queries(
                            outputs,
                            pose_targets,
                            batch["task_ids"],
                        )
                    diagnostic_output_tensors: list[tuple[str, torch.Tensor]] = []
                    if grad_diagnostics:
                        diagnostic_output_tensors = [
                            (name, tensor)
                            for name, tensor in iter_named_floating_tensors(outputs)
                            if tensor.requires_grad
                        ]
                        for _, tensor in diagnostic_output_tensors:
                            tensor.retain_grad()
                    if args.person_confidence_rescue:
                        loss, loss_dict = compute_person_confidence_quality_loss(
                            outputs,
                            pose_targets,
                        )
                    else:
                        loss, loss_dict = compute_pose_losses(
                            outputs,
                            pose_targets,
                            batch["task_ids"],
                            weights,
                        )
                    loss_dict.update(
                        compute_pose_diagnostics(outputs, pose_targets, batch["task_ids"])
                    )
                    if "lm_loss" in outputs:
                        lm_loss = outputs["lm_loss"].float()
                        lm_weight = (
                            float(args.w_locate_box_lm)
                            if backbone_name == "eagle"
                            else weights.lm
                        )
                        loss = loss + lm_weight * lm_loss
                        loss_dict["loss_lm"] = lm_loss
                        loss_dict["loss_lm_weight"] = torch.as_tensor(lm_weight, device=loss.device)
                        loss_dict["loss_total"] = loss
                        if is_main_process():
                            progress_write(
                                progress_bar,
                                f"{'locate_box_lm' if backbone_name == 'eagle' else 'qwen_lm'} micro_step={micro_step} "
                                f"raw={float(lm_loss.detach()):.6f} "
                                f"weight={lm_weight:g} "
                                f"contribution={float((lm_weight * lm_loss).detach()):.6f}",
                            )

                # --- Synchronized loss spike detection ---
                loss_val = float(loss.detach())
                trace_loss_val = loss_val
                local_skip_batch = not math.isfinite(loss_val) or loss_val > loss_abs_cap
                if not local_skip_batch and loss_ema is not None:
                    local_skip_batch = loss_val > loss_ema * loss_spike_threshold
                skip_batch = distributed_any(local_skip_batch, device)
                if skip_batch:
                    loss_spike_count += 1
                    ema_str = f"{loss_ema:.4f}" if loss_ema is not None else "n/a"
                    if is_main_process():
                        progress_write(
                            progress_bar,
                            f"[SPIKE] step={global_step} micro={micro_step} loss={loss_val:.4f} "
                            f"ema={ema_str} consecutive={loss_spike_count} — optimizer step skipped on all ranks",
                        )
                    if loss_spike_count >= loss_spike_max:
                        if is_main_process():
                            print(f"[ABORT] {loss_spike_count} consecutive loss spikes; stopping training.")
                        stop_training = True
                else:
                    loss_ema = (
                        loss_val
                        if loss_ema is None
                        else loss_ema * loss_ema_decay + loss_val * (1.0 - loss_ema_decay)
                    )

                sync_cuda_for_timing(args.sync_timing, device)
                forward_time = time.perf_counter() - forward_started

                sync_cuda_for_timing(args.sync_timing, device)
                step_started = time.perf_counter()
                if skip_batch:
                    if use_deepspeed:
                        active_model.zero_grad()
                    else:
                        optimizer.zero_grad(set_to_none=True)
                elif use_deepspeed:
                    boundary = active_model.is_gradient_accumulation_boundary()
                    active_model.backward(loss)
                    zero_bad_gradient_names = (
                        deepspeed_partition_nonfinite_names(
                            active_model,
                            training_model,
                            device,
                        )
                        if boundary or grad_diagnostics
                        else []
                    )
                    if grad_diagnostics:
                        output_gradients_finite, bad_output_gradient_names = synchronized_finite_check(
                            (
                                (name, tensor.grad)
                                for name, tensor in diagnostic_output_tensors
                                if tensor.grad is not None
                            ),
                            device,
                            max_bad_names=12,
                        )
                        if zero_bad_gradient_names or not output_gradients_finite:
                            source_names = ",".join(
                                str(target.get("dataset", "unknown"))
                                for target in pose_targets
                            )
                            if is_main_process():
                                progress_write(
                                    progress_bar,
                                    f"[GRAD_DIAGNOSTIC] step={global_step} micro={micro_step} "
                                    f"boundary={int(boundary)} src={source_names} lm={int(use_lm_loss)} "
                                    f"bad_outputs={','.join(bad_output_gradient_names) or 'none'} "
                                    f"bad_zero={','.join(zero_bad_gradient_names) or 'none'}",
                                )
                    gradients_finite, bad_gradient_names = synchronized_finite_check(
                        (
                            (name, parameter.grad)
                            for name, parameter in training_model.named_parameters()
                            if parameter.requires_grad and parameter.grad is not None
                        ),
                        device,
                    )
                    # Always let DeepSpeed finish the micro step.  In bf16
                    # ZeRO-2 its optimizer-side overflow check inspects the
                    # actual reduced/partitioned gradients and, on overflow,
                    # clears both Parameter.grad and averaged_gradients without
                    # touching the fp32 master weights.  Calling zero_grad()
                    # here would only clear the former and could leave a bad
                    # ZeRO gradient buffered for the next update.
                    active_model.step()
                    optimizer_overflow = bool(
                        boundary
                        and getattr(getattr(active_model, "optimizer", None), "overflow", False)
                    )
                    if optimizer_overflow:
                        skip_batch = True
                        loss_spike_count += 1
                        if is_main_process():
                            local_detail = (
                                ", ".join(zero_bad_gradient_names or bad_gradient_names)
                                if zero_bad_gradient_names or bad_gradient_names
                                else "nonfinite reduced/partitioned ZeRO gradients"
                            )
                            progress_write(
                                progress_bar,
                                f"[NONFINITE_GRAD] step={global_step} micro={micro_step} "
                                f"{local_detail} — DeepSpeed rejected the optimizer update on all ranks",
                            )
                        if loss_spike_count >= loss_spike_max:
                            stop_training = True
                    elif boundary and not gradients_finite:
                        # This should be unreachable with bf16
                        # check_grad_overflow enabled.  Fail before incrementing
                        # our scheduler/global-step counters if a future
                        # DeepSpeed version no longer honors that contract.
                        local_detail = (
                            ", ".join(bad_gradient_names)
                            if bad_gradient_names
                            else "nonfinite gradients reported by another rank"
                        )
                        raise FloatingPointError(
                            "DeepSpeed did not reject non-finite gradients at "
                            f"step={global_step} micro={micro_step}: {local_detail}"
                        )
                    else:
                        loss_spike_count = 0
                        did_update = bool(boundary)
                else:
                    scaler.scale(loss / args.grad_accum_steps).backward()
                    if micro_step % args.grad_accum_steps == 0:
                        scaler.unscale_(optimizer)
                        gradients_finite, bad_gradient_names = synchronized_finite_check(
                            (
                                (name, parameter.grad)
                                for name, parameter in training_model.named_parameters()
                                if parameter.requires_grad and parameter.grad is not None
                            ),
                            device,
                        )
                        if not gradients_finite:
                            skip_batch = True
                            loss_spike_count += 1
                            optimizer.zero_grad(set_to_none=True)
                            scaler.update()
                            if is_main_process():
                                local_detail = ", ".join(bad_gradient_names) if bad_gradient_names else "nonfinite gradients reported by another rank"
                                progress_write(
                                    progress_bar,
                                    f"[NONFINITE_GRAD] step={global_step} micro={micro_step} {local_detail} — optimizer step skipped on all ranks",
                                )
                            if loss_spike_count >= loss_spike_max:
                                stop_training = True
                        else:
                            loss_spike_count = 0
                            torch.nn.utils.clip_grad_norm_(training_model.parameters(), args.grad_clip)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad(set_to_none=True)
                            did_update = True
                if did_update:
                    parameters_finite, bad_parameter_names = synchronized_finite_check(
                        (
                            (name, parameter)
                            for name, parameter in training_model.named_parameters()
                            if parameter.requires_grad
                        ),
                        device,
                    )
                    if not parameters_finite:
                        local_detail = ", ".join(bad_parameter_names) if bad_parameter_names else "nonfinite parameters reported by another rank"
                        raise FloatingPointError(
                            f"Optimizer produced non-finite parameters at step={global_step} micro={micro_step}: {local_detail}"
                        )
                sync_cuda_for_timing(args.sync_timing, device)
                step_time = time.perf_counter() - step_started
                loss_metrics_for_trace = {k: float(v.detach().cpu()) for k, v in loss_dict.items()}
                if is_main_process():
                    source_names = batch.get("source_datasets", [])
                    dataset_name = str(source_names[0] if source_names else "unknown").lower()
                    update_dataset_metric_ema(
                        dataset_metric_ema,
                        dataset_name,
                        loss_metrics_for_trace,
                    )
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
                    progress_write(progress_bar, "  " + build_detailed_loss_message(loss_metrics, weights))
                    for dataset_message in format_dataset_metric_ema(dataset_metric_ema):
                        progress_write(progress_bar, "  " + dataset_message)
                    timing_sums = {"data": 0.0, "prep": 0.0, "fwd": 0.0, "bwd": 0.0, "micro": 0}

                if is_main_process() and args.visualize_every > 0 and (
                    global_step % args.visualize_every == 0 or global_step == 1
                ):
                    visualization_batch = {**batch, "targets": pose_targets}
                    vis_sample_idx = select_informative_visualization_sample(
                        outputs,
                        visualization_batch,
                        min_gt_area_ratio=args.visualize_min_gt_area_ratio,
                    )
                    if vis_sample_idx is None:
                        progress_write(
                            progress_bar,
                            f"visualization_skipped=step_{global_step}:no_readable_pose_target",
                        )
                    else:
                        vis_target = pose_targets[vis_sample_idx]
                        vis_source_datasets = batch.get("source_datasets", [])
                        vis_dataset_name = (
                            vis_source_datasets[vis_sample_idx]
                            if vis_sample_idx < len(vis_source_datasets)
                            else vis_target.get("dataset", "unknown")
                        )
                        vis_dataset = _safe_vis_tag(vis_dataset_name)
                        vis_schema = _safe_vis_tag(vis_target.get("schema", "unknown"))
                        vis_path = args.output_dir / "visualizations" / f"train_step_{global_step:08d}_{vis_dataset}_{vis_schema}.jpg"
                        try:
                            save_pose_visualization(
                                outputs,
                                visualization_batch,
                                vis_path,
                                sample_idx=vis_sample_idx,
                                max_instances=args.visualize_max_instances,
                                draw_all_schema_keypoints=True,
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
                    last_saved_step = global_step

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
        gradients_finite, bad_gradient_names = synchronized_finite_check(
            (
                (name, parameter.grad)
                for name, parameter in training_model.named_parameters()
                if parameter.requires_grad and parameter.grad is not None
            ),
            device,
        )
        if not gradients_finite:
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            if is_main_process():
                local_detail = ", ".join(bad_gradient_names) if bad_gradient_names else "nonfinite gradients reported by another rank"
                print(f"[NONFINITE_GRAD] final partial accumulation skipped: {local_detail}")
        else:
            torch.nn.utils.clip_grad_norm_(training_model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            parameters_finite, bad_parameter_names = synchronized_finite_check(
                (
                    (name, parameter)
                    for name, parameter in training_model.named_parameters()
                    if parameter.requires_grad
                ),
                device,
            )
            if not parameters_finite:
                local_detail = ", ".join(bad_parameter_names) if bad_parameter_names else "nonfinite parameters reported by another rank"
                raise FloatingPointError(
                    f"Optimizer produced non-finite parameters in final partial accumulation: {local_detail}"
                )
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
    if last_saved_step != global_step:
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
    try:
        main()
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            try:
                torch.distributed.destroy_process_group()
            except Exception:
                pass
