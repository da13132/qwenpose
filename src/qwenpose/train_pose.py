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
    ALL_POSE_PROMPT,
    DATASET_BOX_CONTEXT_SCALE,
    PoseAugmentConfig,
    build_refhuman_locate_prompt,
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
from qwenpose.spatial_features import SpatialFeatureBatch
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
        help="Reference pixel scale used by box/keypoint denoising.",
    )
    parser.add_argument(
        "--letterbox_size",
        type=int,
        default=800,
        help=(
            "Resize the long image side to this value and center-pad the short side "
            "to a fixed square canvas. Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--letterbox_fill",
        type=int,
        default=127,
        help="RGB gray value used for fixed-square letterbox padding.",
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
    parser.add_argument(
        "--prompt_embedding_cache",
        "--refhuman_text_embedding_cache",
        dest="refhuman_text_embedding_cache",
        type=Path,
        default=None,
        help=(
            "Full-prompt frozen LocateAnything token/pooled cache used by vision-only "
            "Stage1. Build with scripts/cache_refhuman_text_embeddings.py."
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
    parser.add_argument(
        "--disable_spatial_shape_bucketing",
        action="store_true",
        help="Disable native-grid area/aspect bucketing inside homogeneous batches.",
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
    parser.add_argument(
        "--locate_feature_size",
        "--eagle_feature_size",
        dest="eagle_feature_size",
        type=int,
        default=None,
        help="Deprecated compatibility option; Locate uses native variable grids.",
    )
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
        choices=[
            "frozen",
            "vision_lora",
            "llm_lora",
            "all_lora",
            "selective_vision_lora",
            "selective_llm_lora",
            "selective_lora",
        ],
        default="all_lora",
        help=(
            "Select which LocateAnything adapters receive gradients. The selective_vision_lora "
            "and selective_llm_lora modes are intended for the decoupled three-stage recipe."
        ),
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
        default="0-26",
        help="MoonViT block ranges enabled by selective_lora; defaults to all 27 blocks.",
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
    parser.add_argument(
        "--train_locate_projector",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Fully train LocateAnything's visual projector (mlp1) in every non-frozen "
            "backbone scope. Use --no-train_locate_projector to keep it frozen."
        ),
    )
    parser.add_argument("--freeze_locate", "--freeze_eagle", dest="freeze_eagle", action="store_true")
    parser.add_argument(
        "--freeze_pose",
        action="store_true",
        help="Freeze every LocatePose/PoseHead parameter while keeping checkpoint weights intact.",
    )
    parser.add_argument(
        "--locate_grounding_only",
        action="store_true",
        help=(
            "Run only LocateAnything grounding LM supervision and skip PoseHead forward. "
            "This is the low-memory Stage-2 path of the decoupled recipe."
        ),
    )
    parser.add_argument("--pose_decoder_layers", type=int, default=3)
    parser.add_argument(
        "--refinement_steps",
        type=int,
        default=1,
        help="Final P2 local enhancement steps; DETRPose uses exactly one.",
    )
    parser.add_argument("--decoder_heads", type=int, default=8)
    parser.add_argument("--pose_dropout", type=float, default=0.0)
    parser.add_argument(
        "--box_condition_scale",
        type=float,
        default=1.15,
        help="Context expansion used by multiscale box pooling and pose attention; joint initialization stays on the tight box.",
    )
    parser.add_argument(
        "--pose_coordinate_init",
        choices=(
            "anatomical_dynamic",
            "learned_spread",
            "box_center",
            "schema_prior",
        ),
        default="anatomical_dynamic",
        help=(
            "Main-pose coordinate reference. anatomical_dynamic starts from a "
            "schema prior and predicts an instance-conditioned residual; the "
            "other modes are retained for ablation/legacy evaluation."
        ),
    )
    parser.add_argument(
        "--dynamic_reference_offset_scale",
        type=float,
        default=1.5,
        help="Maximum logit-space residual scale for anatomical dynamic references.",
    )
    parser.add_argument(
        "--schema_joint_priors_path",
        type=str,
        default="configs/schema_joint_priors.json",
        help="Schema prior JSON used by anatomical_dynamic and schema_prior modes.",
    )
    parser.add_argument("--pose_feature_channels", type=int, default=256)
    parser.add_argument("--deformable_points", type=int, default=4)
    parser.add_argument("--deformable_min_radius_cells", type=float, default=2.0)
    parser.add_argument(
        "--ref_text_scale",
        type=float,
        default=0.2,
        help="Scale applied when RefHuman text conditions human and joint queries.",
    )
    parser.add_argument(
        "--disable_keypoint_denoising",
        action="store_true",
        help="Disable training-only box-conditioned OKS keypoint denoising.",
    )
    # DETRPose passes dn_number=20 and creates one positive plus one negative
    # skeleton per group. This code counts the resulting skeleton queries, so
    # the equivalent common-case budget is 40 queries and 20 groups.
    parser.add_argument("--max_keypoint_dn_queries", type=int, default=40)
    parser.add_argument("--max_keypoint_dn_groups", type=int, default=20)
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
        action=argparse.BooleanOptionalAction,
        default=True,
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
    # Loss weights. The final pose uses image-coordinate SmoothL1, evaluator-
    # aligned OKS and keypoint quality confidence. Box-normalized coordinates
    # are reserved for intermediate-stage auxiliary supervision.
    # ---------------------------------------------------------------------
    parser.add_argument("--w_oks", type=float, default=0.5)
    parser.add_argument(
        "--w_coord",
        type=float,
        default=0.0,
        help=(
            "Deprecated compatibility option; final pose no longer uses a "
            "box-normalized coordinate loss."
        ),
    )
    parser.add_argument("--w_image_coord", type=float, default=5.0)
    parser.add_argument(
        "--w_keypoint_confidence",
        "--w_vis",
        dest="w_keypoint_confidence",
        type=float,
        default=0.1,
        help="Weight for matched per-keypoint presence/visibility BCE.",
    )
    parser.add_argument(
        "--w_keypoint_quality",
        type=float,
        default=0.1,
        help="Weight for direct per-keypoint localization-quality supervision.",
    )
    parser.add_argument(
        "--w_person_confidence",
        type=float,
        default=1.0,
        help=(
            "Weight for the direct pose AP logit trained against detached OKS. "
            "Unmatched generated queries are zero-quality negatives."
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
    parser.add_argument(
        "--w_decoder_coords",
        type=str,
        default="0.25,0.5,0.75",
        help="Comma-separated box-normalized auxiliary weights for grouped decoder layers.",
    )
    parser.add_argument("--w_coarse_coord", type=float, default=0.5)
    parser.add_argument("--w_deform_coord", type=float, default=0.75)
    parser.add_argument(
        "--w_refine_coords",
        type=str,
        default="0.75,1.0",
        help=(
            "Comma-separated box-normalized coordinate weights for refinement "
            "outputs before the final prediction."
        ),
    )
    parser.add_argument("--w_box_objectness", type=float, default=1.0)
    parser.add_argument(
        "--w_box_quality",
        type=float,
        default=1.0,
        help="Weight for the direct person/bbox AP logit trained against detached IoU.",
    )
    parser.add_argument("--w_box_l1", type=float, default=5.0)
    parser.add_argument("--w_box_giou", type=float, default=2.0)
    parser.add_argument("--w_box_relative", type=float, default=1.0)
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
        default="person_queries",
        help=(
            "Optional external box priors for a pose-set model; person_queries uses only "
            "person queries that detect every person before RefHuman selection."
        ),
    )
    parser.add_argument(
        "--locate_proxy_probability",
        type=float,
        default=0.0,
        help=(
            "Probability that vision-only person-query training injects jittered clean-GT "
            "boxes as Locate-style external priors. The clean GT remains the loss target."
        ),
    )
    parser.add_argument(
        "--locate_proxy_center_noise",
        type=float,
        default=0.03,
        help="Stddev of Stage1 Locate-proxy center noise, relative to GT box width/height.",
    )
    parser.add_argument(
        "--locate_proxy_scale_noise",
        type=float,
        default=0.06,
        help="Stddev of Stage1 Locate-proxy log-width/log-height noise.",
    )
    parser.add_argument(
        "--locate_proxy_miss_probability",
        type=float,
        default=0.5,
        help=(
            "For ordinary multi-person images, probability of dropping an adaptive "
            "1-3 GT-derived external boxes while retaining at least one person."
        ),
    )
    parser.add_argument(
        "--locate_proxy_duplicate_probability",
        type=float,
        default=0.5,
        help=(
            "For ordinary multi-person images, probability of adding an adaptive "
            "1-3 unmatched duplicate detections."
        ),
    )
    parser.add_argument(
        "--num_person_queries",
        type=int,
        default=60,
        help="Number of internal pose-set candidates; defaults to 60.",
    )
    parser.add_argument(
        "--num_ref_queries",
        type=int,
        default=4,
        help="RefHuman text-conditioned candidates reserved inside the pose groups.",
    )
    parser.add_argument(
        "--multiscale_encoder_layers",
        type=int,
        default=2,
        help="Number of trainable native-grid P2/P3/P4 deformable encoder layers.",
    )
    parser.add_argument(
        "--multiscale_encoder_points",
        type=int,
        default=4,
        help="Sampling points per level in each multi-scale encoder layer.",
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
            "Generate LocateAnything boxes only for RefHuman; when disabled, generate "
            "boxes for every pose dataset."
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
        "--visualize_keypoint_visibility_threshold",
        type=float,
        default=0.5,
        help="Only draw predicted joints whose learned visibility probability reaches this threshold.",
    )
    parser.add_argument(
        "--visualize_keypoint_quality_threshold",
        type=float,
        default=0.3,
        help="Only draw predicted joints whose learned localization quality reaches this threshold.",
    )
    parser.add_argument(
        "--visualize_nms_iou_thresh",
        type=float,
        default=0.65,
        help="Class-agnostic proposal-box NMS IoU used by raw training visualizations.",
    )
    parser.add_argument(
        "--visualize_objectness_threshold",
        type=float,
        default=0.05,
        help=(
            "Final person-box score threshold after NMS. The score is "
            "sigmoid(person_class) * sigmoid(box_quality); all-filtered ALL_POSE "
            "samples intentionally display no prediction."
        ),
    )
    parser.add_argument(
        "--visualize_pose_threshold",
        type=float,
        default=0.05,
        help=(
            "Minimum sigmoid(person_class) * sigmoid(pose_quality) required "
            "before drawing a predicted skeleton."
        ),
    )
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
    if torch.is_tensor(batch.get("cached_text_embeddings")):
        batch["cached_text_embeddings"] = batch["cached_text_embeddings"].to(
            device, non_blocking=True
        )
    if torch.is_tensor(batch.get("cached_text_embedding_mask")):
        batch["cached_text_embedding_mask"] = batch[
            "cached_text_embedding_mask"
        ].to(device, non_blocking=True)
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
                "locate_direct_grounding_failed": bool(
                    selected_target.get(
                        "locate_direct_grounding_failed",
                        torch.tensor(False),
                    ).detach().cpu().item()
                ),
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
    # LocateAnything registers coordinate tokens as <0> ... <1000>, without
    # zero padding. Formatting <029> silently splits into ordinary text tokens
    # and breaks native six-token box supervision.
    return "<" + str(max(0, min(scaled, 1000))) + ">"


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
    index_tensor = torch.as_tensor(
        indices,
        dtype=torch.long,
        device=target["boxes"].device,
    )
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
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=values.device)
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


def jitter_locate_proxy_boxes_xyxy(
    boxes: torch.Tensor,
    *,
    center_noise: float,
    scale_noise: float,
) -> torch.Tensor:
    """Apply independent Gaussian center/log-size noise to clean GT boxes."""
    if boxes.numel() == 0:
        return boxes
    boxes = boxes.float()
    center = (boxes[:, :2] + boxes[:, 2:]) * 0.5
    wh = (boxes[:, 2:] - boxes[:, :2]).clamp(min=1e-4)
    center = center + torch.randn_like(center) * wh * max(float(center_noise), 0.0)
    wh = wh * torch.exp(
        torch.randn_like(wh) * max(float(scale_noise), 0.0)
    )
    wh = wh.clamp(min=1e-4, max=1.0)
    jittered = torch.cat([center - 0.5 * wh, center + 0.5 * wh], dim=-1)
    return jittered.clamp(0.0, 1.0)


def prepare_locate_proxy_conditioning(
    targets: list[dict[str, torch.Tensor]],
    task_ids: torch.Tensor,
    device: torch.device,
    *,
    max_instances: int,
    center_noise: float,
    scale_noise: float,
    miss_probability: float = 0.5,
    duplicate_probability: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, torch.Tensor]]]:
    """Build noisy Stage1 Locate-style detections with explicit GT identity.

    Every retained real detection receives center/log-size noise. Ordinary
    multi-person samples independently simulate missed and duplicate detections.
    The number of affected boxes scales with crowd size: at most one for a
    three-person image and at most three for a ten-person image. RefHuman keeps
    exactly its referred person so language identity cannot be corrupted.
    """
    clean_targets = prepare_person_query_conditioning(
        targets, task_ids, device, max_instances=max_instances
    )[2]
    selected_boxes: list[torch.Tensor] = []
    for sample_idx, clean_target in enumerate(clean_targets):
        clean_boxes = clean_target["boxes"]
        count = int(clean_boxes.shape[0])
        task_id = int(task_ids[sample_idx].detach().cpu().item())
        if task_id == 1:
            ref_target = int(clean_target["ref_target"].detach().cpu().item())
            retained_indices = [ref_target] if 0 <= ref_target < count else []
        else:
            retained_indices = list(range(count))

        missed_count = 0
        if (
            task_id != 1
            and len(retained_indices) > 1
            and random.random() < max(0.0, min(float(miss_probability), 1.0))
        ):
            max_missed = min(
                3,
                len(retained_indices) - 1,
                max(1, (len(retained_indices) + 2) // 3),
            )
            missed_count = random.randint(1, max_missed)
            missed = set(random.sample(retained_indices, missed_count))
            retained_indices = [idx for idx in retained_indices if idx not in missed]

        proxy_parts: list[torch.Tensor] = []
        source_gt_indices: list[int] = []
        if retained_indices:
            retained_tensor = torch.as_tensor(
                retained_indices, device=clean_boxes.device, dtype=torch.long
            )
            proxy_parts.append(
                jitter_locate_proxy_boxes_xyxy(
                    clean_boxes[retained_tensor],
                    center_noise=center_noise,
                    scale_noise=scale_noise,
                )
            )
            source_gt_indices.extend(retained_indices)

        duplicate_count = 0
        if (
            task_id != 1
            and count > 1
            and retained_indices
            and random.random()
            < max(0.0, min(float(duplicate_probability), 1.0))
        ):
            max_duplicates = min(
                3,
                max(1, (count + 2) // 3),
                max(max(int(max_instances), 1) - len(retained_indices), 0),
            )
            if max_duplicates > 0:
                duplicate_count = random.randint(1, max_duplicates)
                duplicate_sources = [
                    random.choice(retained_indices) for _ in range(duplicate_count)
                ]
                duplicate_tensor = torch.as_tensor(
                    duplicate_sources, device=clean_boxes.device, dtype=torch.long
                )
                proxy_parts.append(
                    jitter_locate_proxy_boxes_xyxy(
                        clean_boxes[duplicate_tensor],
                        center_noise=max(float(center_noise) * 1.5, 0.05),
                        scale_noise=max(float(scale_noise) * 1.5, 0.10),
                    )
                )
                # Duplicate detections are intentional false positives. They
                # receive confidence/quality supervision but no pose/box target.
                source_gt_indices.extend([-1] * duplicate_count)

        proxy = (
            torch.cat(proxy_parts, dim=0)
            if proxy_parts
            else clean_boxes[:0].clone()
        )
        selected_boxes.append(proxy)
        clean_target["locate_proxy_active"] = torch.tensor(
            True, device=clean_boxes.device, dtype=torch.bool
        )
        clean_target["locate_proxy_gt_indices"] = torch.tensor(
            source_gt_indices, device=clean_boxes.device, dtype=torch.long
        )
        clean_target["locate_proxy_missed_count"] = torch.tensor(
            missed_count, device=clean_boxes.device, dtype=torch.long
        )
        clean_target["locate_proxy_duplicate_count"] = torch.tensor(
            duplicate_count, device=clean_boxes.device, dtype=torch.long
        )

    max_boxes = max([int(boxes.shape[0]) for boxes in selected_boxes] + [1])
    box_tensor = torch.zeros(
        len(selected_boxes), max_boxes, 4, dtype=torch.float32, device=device
    )
    box_mask = torch.zeros(
        len(selected_boxes), max_boxes, dtype=torch.bool, device=device
    )
    for sample_idx, boxes in enumerate(selected_boxes):
        n = int(boxes.shape[0])
        if n > 0:
            box_tensor[sample_idx, :n] = boxes.to(device=device, dtype=torch.float32)
            box_mask[sample_idx, :n] = True
    return box_tensor, box_mask, clean_targets


def prepare_keypoint_denoising(
    targets: list[dict[str, torch.Tensor]],
    device: torch.device,
    *,
    max_queries: int = 40,
    max_groups: int = 20,
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
        proxy_sources = target.get("locate_proxy_gt_indices")
        if isinstance(proxy_sources, torch.Tensor):
            retained = proxy_sources.to(device=device, dtype=torch.long)
            retained = retained[(retained >= 0) & (retained < candidate.numel())]
            retained_mask = torch.zeros_like(candidate)
            if retained.numel() > 0:
                retained_mask[retained.unique()] = True
            candidate = candidate & retained_mask
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


def hungarian_match_poses(
    pred_keypoints: torch.Tensor,
    pred_logits: torch.Tensor,
    gt_keypoints: torch.Tensor,
    gt_valid: torch.Tensor,
    gt_areas: torch.Tensor,
    pred_boxes: torch.Tensor | None = None,
    gt_boxes: torch.Tensor | None = None,
) -> list[tuple[int, int, float]]:
    """Match pose groups by pose quality plus a modest proposal-box prior."""
    if pred_keypoints.numel() == 0 or gt_keypoints.numel() == 0:
        return []
    pred_xy = pred_keypoints.detach().cpu().float()[..., :2]
    gt_xy = gt_keypoints.detach().cpu().float()[..., :2]
    valid = gt_valid.detach().cpu().bool()
    areas = gt_areas.detach().cpu().float().clamp(min=1e-8)
    logits = pred_logits.detach().cpu().float().reshape(-1)
    delta = pred_xy[:, None] - gt_xy[None]
    valid_f = valid[None].float()
    valid_count = valid_f.sum(dim=-1).clamp(min=1.0)
    image_l1 = (delta.abs().sum(dim=-1) * valid_f).sum(dim=-1) / valid_count
    sigmas = UNION_SIGMAS.detach().cpu().float().view(1, 1, -1)
    variance = (2.0 * sigmas).square().clamp(min=1e-8)
    oks_per_joint = torch.exp(
        -delta.square().sum(dim=-1)
        / (2.0 * areas.view(1, -1, 1) * variance)
    )
    oks = (oks_per_joint * valid_f).sum(dim=-1) / valid_count
    quality_cost = -logits.sigmoid()[:, None]
    cost = 5.0 * image_l1 + 2.0 * (1.0 - oks) + 2.0 * quality_cost
    if (
        torch.is_tensor(pred_boxes)
        and torch.is_tensor(gt_boxes)
        and int(pred_boxes.shape[0]) == int(pred_xy.shape[0])
        and int(gt_boxes.shape[0]) == int(gt_xy.shape[0])
    ):
        proposal_boxes = pred_boxes.detach().cpu().float().clamp(0.0, 1.0)
        target_boxes = gt_boxes.detach().cpu().float().clamp(0.0, 1.0)
        box_l1 = torch.cdist(proposal_boxes, target_boxes, p=1)

        left_top = torch.maximum(
            proposal_boxes[:, None, :2], target_boxes[None, :, :2]
        )
        right_bottom = torch.minimum(
            proposal_boxes[:, None, 2:], target_boxes[None, :, 2:]
        )
        intersection_wh = (right_bottom - left_top).clamp(min=0.0)
        intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
        proposal_area = (
            (proposal_boxes[:, 2] - proposal_boxes[:, 0]).clamp(min=0.0)
            * (proposal_boxes[:, 3] - proposal_boxes[:, 1]).clamp(min=0.0)
        )
        target_area = (
            (target_boxes[:, 2] - target_boxes[:, 0]).clamp(min=0.0)
            * (target_boxes[:, 3] - target_boxes[:, 1]).clamp(min=0.0)
        )
        union = proposal_area[:, None] + target_area[None, :] - intersection
        iou = intersection / union.clamp(min=1e-8)
        enclosing_left_top = torch.minimum(
            proposal_boxes[:, None, :2], target_boxes[None, :, :2]
        )
        enclosing_right_bottom = torch.maximum(
            proposal_boxes[:, None, 2:], target_boxes[None, :, 2:]
        )
        enclosing_wh = (enclosing_right_bottom - enclosing_left_top).clamp(min=0.0)
        enclosing_area = enclosing_wh[..., 0] * enclosing_wh[..., 1]
        giou = iou - (enclosing_area - union) / enclosing_area.clamp(min=1e-8)
        # Pose remains the primary assignment signal.  Box geometry only
        # stabilizes early proposal learning and resolves nearby-person ties.
        cost = cost + box_l1 + (1.0 - giou)
    # GT instances with no annotated joints cannot supervise a pose match.
    invalid_gt = ~valid.any(dim=-1)
    if invalid_gt.any():
        cost[:, invalid_gt] = 1e6
    if not bool(torch.isfinite(cost).all().item()):
        return []
    if _scipy_linear_sum_assignment is not None:
        pred_indices, gt_indices = _scipy_linear_sum_assignment(cost.numpy())
    else:
        remaining_pred = set(range(int(cost.shape[0])))
        remaining_gt = set(range(int(cost.shape[1])))
        pred_indices, gt_indices = [], []
        while remaining_pred and remaining_gt:
            best = min(
                (
                    (float(cost[pred_idx, gt_idx]), pred_idx, gt_idx)
                    for pred_idx in remaining_pred
                    for gt_idx in remaining_gt
                ),
                key=lambda row: row[0],
            )
            _, pred_idx, gt_idx = best
            pred_indices.append(pred_idx)
            gt_indices.append(gt_idx)
            remaining_pred.remove(pred_idx)
            remaining_gt.remove(gt_idx)
    matches: list[tuple[int, int, float]] = []
    for pred_idx, gt_idx in zip(list(pred_indices), list(gt_indices)):
        if not bool(valid[int(gt_idx)].any().item()):
            continue
        matches.append((int(pred_idx), int(gt_idx), float(oks[int(pred_idx), int(gt_idx)])))
    matches.sort(key=lambda row: row[0])
    return matches


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
    proposal_boxes = outputs.get("input_boxes")
    if not torch.is_tensor(proposal_boxes):
        proposal_boxes = predicted
    else:
        proposal_boxes = proposal_boxes.detach()
    predicted_keypoints = outputs.get("pred_keypoints", outputs.get("keypoints"))
    # Matching/top-Q uses auxiliary proposal objectness, never either public AP
    # score. This keeps assignment stable while box and pose quality calibrate
    # independently against their detached IoU/OKS targets.
    predicted_logits = outputs.get(
        "proposal_objectness_logits_aux",
        outputs.get("person_class_logits", outputs.get("person_logits")),
    )
    pose_set_prediction = bool(
        torch.is_tensor(outputs.get("pose_set_prediction"))
        and bool(outputs["pose_set_prediction"].detach().item())
    )
    query_mask = outputs.get("box_mask")
    proposal_source_ids = outputs.get("proposal_source_ids")
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
        proxy_gt_indices = target.get("locate_proxy_gt_indices")
        if (
            torch.is_tensor(proxy_gt_indices)
            and torch.is_tensor(proposal_source_ids)
            and bool(target.get("locate_proxy_active", False))
        ):
            # Stage1 external boxes carry their source identity explicitly.
            # Never let an immature pose prediction remap a referred/noisy box
            # to another person. Missed GT people remain missed; intentional
            # duplicate detections carry source -1 and remain background.
            external_queries = torch.nonzero(
                valid_queries & proposal_source_ids[batch_idx].detach().eq(1),
                as_tuple=False,
            ).flatten()
            source_indices = proxy_gt_indices.to(
                device=predicted.device, dtype=torch.long
            )
            fixed_count = min(
                int(external_queries.numel()), int(source_indices.numel())
            )
            matches: list[tuple[int, int, float]] = []
            for external_idx in range(fixed_count):
                query_idx = int(external_queries[external_idx].item())
                gt_idx = int(source_indices[external_idx].item())
                if not 0 <= gt_idx < int(gt_boxes.shape[0]):
                    continue
                iou = float(
                    box_iou_xyxy(
                        proposal_boxes[batch_idx, query_idx : query_idx + 1].float(),
                        target["boxes"][gt_idx : gt_idx + 1]
                        .to(device=proposal_boxes.device)
                        .float(),
                    )[0, 0].detach().item()
                )
                matches.append((query_idx, gt_idx, iou))
            aligned.append(
                align_target_to_predictions(
                    target,
                    predicted[batch_idx],
                    gt_indices,
                    matches,
                    task_id=int(task_ids[batch_idx].detach().cpu().item()),
                )
            )
            continue
        if (
            pose_set_prediction
            and torch.is_tensor(predicted_keypoints)
            and torch.is_tensor(predicted_logits)
        ):
            sample_logits = predicted_logits[batch_idx, query_indices]
            loss_areas = target.get("loss_areas")
            if not torch.is_tensor(loss_areas):
                loss_areas = (
                    (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=0.0)
                    * (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=0.0)
                )
            local_matches = hungarian_match_poses(
                predicted_keypoints[batch_idx, query_indices],
                sample_logits,
                target["keypoints"],
                target["keypoint_valid"],
                loss_areas,
                pred_boxes=proposal_boxes[batch_idx, query_indices],
                gt_boxes=target["boxes"],
            )
            # Some detection annotations contain a valid human box but no pose
            # labels.  They cannot enter pose matching, yet should still train
            # proposal objectness/L1/GIoU. Match them by box using only queries
            # left over from the pose assignment.
            box_only_gt = torch.nonzero(
                ~target["keypoint_valid"].bool().any(dim=-1),
                as_tuple=False,
            ).flatten()
            used_predictions = {pred_idx for pred_idx, _, _ in local_matches}
            remaining_predictions = torch.tensor(
                [
                    pred_idx
                    for pred_idx in range(int(query_indices.numel()))
                    if pred_idx not in used_predictions
                ],
                device=proposal_boxes.device,
                dtype=torch.long,
            )
            if box_only_gt.numel() > 0 and remaining_predictions.numel() > 0:
                box_only_matches = hungarian_match_boxes(
                    proposal_boxes[batch_idx, query_indices[remaining_predictions]],
                    target["boxes"].to(device=proposal_boxes.device)[box_only_gt],
                    iou_thresh=0.0,
                )
                local_matches.extend(
                    (
                        int(remaining_predictions[pred_idx].item()),
                        int(box_only_gt[gt_idx].item()),
                        iou,
                    )
                    for pred_idx, gt_idx, iou in box_only_matches
                )
                local_matches.sort(key=lambda row: row[0])
        else:
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
            prompts.append(ALL_POSE_PROMPT)
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
    gt_ref_fallback_on_failure: bool = False,
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
        gt_index_tensor = torch.as_tensor(
            gt_indices,
            dtype=torch.long,
            device=gt_boxes_all.device,
        )
        gt_boxes = gt_boxes_all[gt_index_tensor].clone() if gt_indices else gt_boxes_all[:0].clone()

        pred_boxes = parse_locate_boxes_for_task(
            responses[sample_idx] if sample_idx < len(responses) else "",
            task_id=task_id,
            max_instances=max_instances,
            nms_iou_thresh=nms_iou_thresh,
            disable_pre_pose_nms=disable_pre_pose_nms,
        )

        direct_grounding_failed = bool(task_id == 1 and pred_boxes.numel() == 0)
        if gt_ref_fallback_on_failure and direct_grounding_failed and gt_boxes.numel() > 0:
            pred_boxes = gt_boxes.clone()
            matches = [(0, 0, 1.0)]
        else:
            matches = hungarian_match_boxes(pred_boxes, gt_boxes, iou_thresh=match_iou_thresh)
        selected = align_target_to_predictions(
            target,
            pred_boxes,
            gt_indices,
            matches,
            task_id=task_id,
        )
        selected["locate_direct_grounding_failed"] = torch.tensor(
            direct_grounding_failed,
            device=target["boxes"].device,
            dtype=torch.bool,
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
) -> tuple[list[str], SpatialFeatureBatch, torch.Tensor]:
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
    feature_maps: list[SpatialFeatureBatch] = []
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
            responses.append(response)
            feature_maps.append(feature_map)
            text_embeds.append(text_embed)
    finally:
        if was_training:
            locate_model.train()

    return responses, SpatialFeatureBatch.concatenate(feature_maps), torch.cat(text_embeds, dim=0)


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
    train_projector: bool = True,
    llm_layers: str = "32-35",
    vision_layers: str = "0-26",
    llm_modules: str = "q_proj,v_proj",
    vision_modules: str = "wqkv,wo,fc0,fc1",
) -> dict[str, int]:
    """Enable exactly the requested pretrained-backbone adapter parameters."""
    scope = str(scope)
    valid_scopes = {
        "frozen",
        "vision_lora",
        "llm_lora",
        "all_lora",
        "selective_vision_lora",
        "selective_llm_lora",
        "selective_lora",
    }
    if scope not in valid_scopes:
        raise ValueError(f"Unsupported backbone train scope: {scope!r}")
    selected_llm_layers = parse_layer_selection(llm_layers)
    selected_vision_layers = parse_layer_selection(vision_layers)
    selected_llm_modules = parse_module_selection(llm_modules)
    selected_vision_modules = parse_module_selection(vision_modules)
    if scope in {"selective_lora", "selective_llm_lora"}:
        if not selected_llm_layers or not selected_llm_modules:
            raise ValueError(f"{scope} requires non-empty LLM layer/module selections.")
    if scope in {"selective_lora", "selective_vision_lora"}:
        if not selected_vision_layers or not selected_vision_modules:
            raise ValueError(f"{scope} requires non-empty vision layer/module selections.")

    counts = {"vision_lora": 0, "language_lora": 0, "projector": 0}
    if model is None:
        return counts
    for name, param in model.named_parameters():
        is_lora = "lora_" in name
        is_vision = is_vision_parameter(name)
        is_projector = ".mlp1." in name or name.startswith("mlp1.")
        enabled = bool(scope != "frozen" and train_projector and is_projector)
        if not enabled and scope == "vision_lora":
            enabled = is_lora and is_vision
        elif not enabled and scope == "llm_lora":
            enabled = is_lora and not is_vision
        elif not enabled and scope == "all_lora":
            enabled = is_lora
        elif not enabled and scope in {
            "selective_vision_lora",
            "selective_llm_lora",
            "selective_lora",
        } and is_lora:
            layer_index = _adapter_layer_index(name, vision=is_vision)
            projection_name = _adapter_projection_name(name)
            if is_vision and scope in {"selective_vision_lora", "selective_lora"}:
                enabled = (
                    layer_index in selected_vision_layers
                    and projection_name in selected_vision_modules
                )
            elif not is_vision and scope in {"selective_llm_lora", "selective_lora"}:
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
    if scope in {"selective_lora", "selective_vision_lora"} and counts["vision_lora"] == 0:
        raise RuntimeError(
            f"{scope} matched no vision LoRA parameters; check layer/module names."
        )
    if scope in {"selective_lora", "selective_llm_lora"} and counts["language_lora"] == 0:
        raise RuntimeError(
            f"{scope} matched no language LoRA parameters; check layer/module names."
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
        train_backbone_projector: bool = True,
        backbone_llm_layers: str = "32-35",
        backbone_vision_layers: str = "0-26",
        backbone_llm_modules: str = "q_proj,v_proj",
        backbone_vision_modules: str = "wqkv,wo,fc0,fc1",
        pose_condition_box_mode: str = "refined_detached",
    ) -> None:
        super().__init__()
        self.pose_model = pose_model
        self.backbone_name = backbone_name
        if pose_condition_box_mode not in {"input", "refined_detached"}:
            raise ValueError(
                "pose_condition_box_mode must be input or refined_detached, "
                f"got {pose_condition_box_mode!r}."
            )
        self.pose_condition_box_mode = pose_condition_box_mode

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
        run_pose: bool = True,
        target_boxes: torch.Tensor | None = None,
        target_box_mask: torch.Tensor | None = None,
        images: torch.Tensor | None = None,
        cached_text_embed: torch.Tensor | None = None,
        cached_text_mask: torch.Tensor | None = None,
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
        projected_vit_cache: list[torch.Tensor] | None = None

        # Stage 2 of the decoupled recipe trains only LocateAnything grounding.
        # It intentionally avoids PoseHead and avoids a second multimodal pass:
        # the teacher-forcing batch supplies both MoonViT tokens and LLM labels.
        if not bool(run_pose):
            if self.backbone_name != "eagle" or self.backbone_extractor is None:
                raise ValueError("Grounding-only forward requires the LocateAnything backbone.")
            if qwen_lm_inputs is None:
                raise ValueError("Grounding-only forward requires qwen_lm_inputs.")
            if self.freeze_backbone:
                raise ValueError("Grounding-only forward requires trainable LLM adapters.")
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
            outputs: dict[str, torch.Tensor] = {}
            if valid_label_mask.any():
                target_hidden = lm_hidden[:, :-1, :][valid_label_mask]
                target_labels = shift_labels[valid_label_mask]
                target_logits = locate_base_model.language_model.lm_head(
                    target_hidden
                ).float()
                outputs["lm_loss"] = F.cross_entropy(
                    target_logits,
                    target_labels,
                    reduction="mean",
                )
            else:
                outputs["lm_loss"] = lm_hidden.sum() * 0.0
            return outputs

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
                    require_text = bool(task_ids.eq(1).any().item())
                    if qwen_lm_inputs is not None:
                        (
                            external_feature_map,
                            external_text_embed,
                            projected_vit_cache,
                        ) = self.backbone_extractor.forward_with_vision_cache(
                            qwen_inputs,
                            freeze_eagle=self.freeze_backbone,
                            require_text=require_text,
                        )
                    else:
                        external_feature_map, external_text_embed = self.backbone_extractor(
                            qwen_inputs,
                            freeze_eagle=self.freeze_backbone,
                            require_text=require_text,
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
                        projected_visual_tokens = None
                        if projected_vit_cache is not None:
                            cached_visual_tokens = torch.cat(projected_vit_cache, dim=0)
                            expected_image_tokens = int(
                                lm_input_ids.eq(int(locate_base_model.image_token_index)).sum().item()
                            )
                            if int(cached_visual_tokens.shape[0]) == expected_image_tokens:
                                projected_visual_tokens = cached_visual_tokens
                        if projected_visual_tokens is None:
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
            cached_text_embed=cached_text_embed,
            cached_text_mask=cached_text_mask,
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
            pose_condition_box_mode=self.pose_condition_box_mode,
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
    locate_proxy_probability: float = 0.0
    locate_proxy_center_noise: float = 0.03
    locate_proxy_scale_noise: float = 0.06
    locate_proxy_miss_probability: float = 0.5
    locate_proxy_duplicate_probability: float = 0.5

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
            locate_proxy_probability=float(
                getattr(args, "locate_proxy_probability", 0.0)
            ),
            locate_proxy_center_noise=float(
                getattr(args, "locate_proxy_center_noise", 0.03)
            ),
            locate_proxy_scale_noise=float(
                getattr(args, "locate_proxy_scale_noise", 0.06)
            ),
            locate_proxy_miss_probability=float(
                getattr(args, "locate_proxy_miss_probability", 0.5)
            ),
            locate_proxy_duplicate_probability=float(
                getattr(args, "locate_proxy_duplicate_probability", 0.5)
            ),
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
        use_pose_set = bool(self.module.pose_model.config.use_global_person_queries)
        needs_locate_prior = (
            precomputed_locate_responses is not None
            or bool(batch["task_ids"].eq(1).any().item())
        )
        if use_pose_set and not needs_locate_prior:
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
        pose_set_architecture = bool(
            getattr(self.module.pose_model.config, "use_detrpose_architecture", False)
        )

        def all_pose_targets() -> list[dict[str, torch.Tensor]]:
            return prepare_person_query_conditioning(
                batch["targets"],
                batch["task_ids"],
                self.device,
                max_instances=config.max_instances,
            )[2]

        if box_source == "person_queries":
            if (
                config.locate_proxy_probability > 0.0
                and random.random() < config.locate_proxy_probability
            ):
                return prepare_locate_proxy_conditioning(
                    batch["targets"],
                    batch["task_ids"],
                    self.device,
                    max_instances=config.max_instances,
                    center_noise=config.locate_proxy_center_noise,
                    scale_noise=config.locate_proxy_scale_noise,
                    miss_probability=config.locate_proxy_miss_probability,
                    duplicate_probability=config.locate_proxy_duplicate_probability,
                )
            return prepare_person_query_conditioning(
                batch["targets"],
                batch["task_ids"],
                self.device,
                max_instances=config.max_instances,
            )
        if box_source == "qwen_generate":
            boxes, mask, selected_targets = prepare_qwen_generated_box_conditioning(
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
            return boxes, mask, all_pose_targets() if pose_set_architecture else selected_targets
        if box_source == "locate_generate":
            # In the no-ROI pose-set graph, ordinary multi-person samples use
            # only internal proposals. Locate boxes are an optional RefHuman
            # prior, never a reason to fall back to GT boxes.
            if config.locate_generate_refhuman_only and not bool(
                batch["task_ids"].eq(1).any().item()
            ):
                if pose_set_architecture:
                    return prepare_person_query_conditioning(
                        batch["targets"],
                        batch["task_ids"],
                        self.device,
                        max_instances=config.max_instances,
                    )
                boxes, mask, selected_targets = prepare_box_conditioning(
                    batch["targets"],
                    batch["task_ids"],
                    self.device,
                    max_instances=config.max_instances,
                    box_jitter_scale=config.box_jitter_scale,
                    box_jitter_shift=config.box_jitter_shift,
                )
                return boxes, mask, selected_targets
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
            boxes, mask, selected_targets = prepare_locate_generated_box_conditioning_from_responses(
                responses,
                batch,
                self.device,
                max_instances=config.max_instances,
                match_iou_thresh=config.box_match_iou_thresh,
                nms_iou_thresh=config.box_nms_iou_thresh,
                disable_pre_pose_nms=config.disable_pre_pose_nms,
                gt_ref_fallback_on_failure=False,
            )
            return boxes, mask, all_pose_targets() if pose_set_architecture else selected_targets
        boxes, mask, selected_targets = prepare_box_conditioning(
            batch["targets"],
            batch["task_ids"],
            self.device,
            max_instances=config.max_instances,
            box_jitter_scale=config.box_jitter_scale,
            box_jitter_shift=config.box_jitter_shift,
        )
        return boxes, mask, all_pose_targets() if pose_set_architecture else selected_targets

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
            no_external_locate = (
                bool(getattr(self.module.pose_model.config, "use_detrpose_architecture", False))
                and config.locate_generate_refhuman_only
                and not bool(batch["task_ids"].eq(1).any().item())
            )
            if no_external_locate:
                target_boxes, target_box_mask, pose_targets = prepare_person_query_conditioning(
                    batch["targets"],
                    batch["task_ids"],
                    self.device,
                    max_instances=config.max_instances,
                )
            else:
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
        if bool(getattr(self.module.pose_model.config, "use_detrpose_architecture", False)):
            pose_targets = prepare_person_query_conditioning(
                batch["targets"],
                batch["task_ids"],
                self.device,
                max_instances=config.max_instances,
            )[2]
        outputs, qwen_inputs = self.forward_pose(
            batch,
            target_boxes,
            target_box_mask,
            config,
            external_feature_map=external_feature_map,
            external_text_embed=external_text_embed,
        )
        if (
            torch.is_tensor(outputs.get("pose_set_prediction"))
            and bool(outputs["pose_set_prediction"].detach().item())
        ):
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


def estimate_locate_merged_grid(
    width: int,
    height: int,
    image_token_limit: int | None,
    *,
    patch_size: int = 14,
    merge_kernel_size: tuple[int, int] = (2, 2),
) -> tuple[int, int]:
    """Return LocateAnything's native post-merge (height, width)."""
    width = max(int(width), 1)
    height = max(int(height), 1)
    raw_tokens = max(width // patch_size, 1) * max(height // patch_size, 1)
    if image_token_limit is not None and int(image_token_limit) > 0 and raw_tokens > int(image_token_limit):
        scale = math.sqrt(float(image_token_limit) / float(raw_tokens))
        width = max(int(width * scale), 1)
        height = max(int(height * scale), 1)
    merged_h_pixels = max(int(merge_kernel_size[0]), 1) * max(int(patch_size), 1)
    merged_w_pixels = max(int(merge_kernel_size[1]), 1) * max(int(patch_size), 1)
    return (
        max(int(math.ceil(height / merged_h_pixels)), 1),
        max(int(math.ceil(width / merged_w_pixels)), 1),
    )


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
        bucket_spatial_shapes: bool = False,
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
        self.bucket_spatial_shapes = bool(bucket_spatial_shapes)
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

    def _sample_spatial_bucket(self, dataset_idx: int, local_linear: int) -> tuple[int, int]:
        inner_dataset = list(getattr(self.dataset, "datasets"))[dataset_idx]
        records = getattr(inner_dataset, "records", None)
        offsets = getattr(self.dataset, "offsets", None)
        strides = getattr(self.dataset, "strides", None)
        if records is None or offsets is None or strides is None or not records:
            return 0, 0
        local_index = (
            int(offsets[dataset_idx]) + int(local_linear) * int(strides[dataset_idx])
        ) % len(records)
        record = records[local_index]
        grid_h, grid_w = estimate_locate_merged_grid(
            int(getattr(record, "width", 1)),
            int(getattr(record, "height", 1)),
            self.vision_token_limit,
        )
        area = grid_h * grid_w
        area_bucket = 0 if area <= 576 else (1 if area <= 800 else 2)
        aspect = float(grid_w) / max(float(grid_h), 1.0)
        aspect_bucket = (
            0
            if aspect < 0.8
            else (1 if aspect <= 1.25 else (2 if aspect < 2.0 else 3))
        )
        return area_bucket, aspect_bucket

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
                sample_buckets = (
                    {
                        value: self._sample_spatial_bucket(dataset_idx, value)
                        for value in local_linear
                    }
                    if self.bucket_spatial_shapes
                    else {}
                )
                # Random tie-breaking changes neighboring samples each epoch while
                # retaining length bucketing by the dominant vision-token cost.
                decorated = [
                    (
                        sample_buckets.get(value, (0, 0)),
                        sample_costs[value],
                        dataset_rng.random(),
                        value,
                    )
                    for value in local_linear
                ]
                decorated.sort(key=lambda item: (item[0], item[1], item[2]))
                local_linear = [item[3] for item in decorated]
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

            # Diagnose whether final keypoints fall outside the independently
            # regressed person box.  The old keypoint envelope made this metric
            # tautologically zero and hid box/pose disagreement.
            pose_boxes = outputs["pred_boxes"][sample_idx, queries]
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
        "loss_keypoint_quality",
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

    ``pose`` and ``box`` are the complete regression-head objectives; keypoint
    denoising is reported separately as ``posedn``.
    """

    group_totals = _weighted_loss_group_totals(loss_metrics, weights)
    postfix = {"loss": _format_loss_float(loss_metrics.get("loss_total", 0.0))}
    for group, label in (
        ("pose", "pose"),
        ("pose_dn", "posedn"),
        ("box", "box"),
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
    add("pose", "img", "loss_image_coord", weights.image_coord)
    add("pose", "hard", "loss_hard_joint", weights.hard_joint)
    confidence_weight = (
        weights.keypoint_confidence if weights.vis is None else float(weights.vis)
    )
    add(
        "pose",
        "vis",
        "loss_keypoint_confidence",
        confidence_weight,
    )
    add("pose", "jointq", "loss_keypoint_quality", weights.keypoint_quality)
    add("pose", "pscore", "loss_person_confidence", weights.person_confidence)
    add("ref", "match", "loss_ref_match", weights.ref_match)
    for decoder_idx, decoder_weight in enumerate(
        parse_float_list(weights.decoder_coords), start=1
    ):
        add(
            "pose",
            f"dec{decoder_idx}",
            f"loss_coord_decoder_{decoder_idx}",
            decoder_weight,
        )
    add("pose", "coarse", "loss_coord_coarse", weights.coarse_coord)
    add("pose", "deform", "loss_coord_deform", weights.deform_coord)
    for refine_idx, refine_weight in enumerate(parse_float_list(weights.refine_coords), start=1):
        add("pose", f"ref{refine_idx}", f"loss_coord_refine_{refine_idx}", refine_weight)
    add("box", "obj", "loss_box_objectness", weights.box_objectness)
    add("box", "quality", "loss_box_quality", weights.box_quality)
    add("box", "l1", "loss_box_l1", weights.box_l1)
    add("box", "giou", "loss_box_giou", weights.box_giou)
    add("box", "rel", "loss_box_relative", weights.box_relative)
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
    coord_decoder = ",".join(
        _format_loss_weight(value) for value in parse_float_list(weights.decoder_coords)
    ) or "none"
    coord_refine = ",".join(_format_loss_weight(value) for value in parse_float_list(weights.refine_coords)) or "none"
    return (
        "Loss weights: "
        f"oks={_format_loss_weight(weights.oks)} "
        f"image_coord={_format_loss_weight(weights.image_coord)} "
        f"keypoint_confidence={_format_loss_weight(weights.keypoint_confidence if weights.vis is None else weights.vis)} "
        f"keypoint_quality={_format_loss_weight(weights.keypoint_quality)} "
        f"person_confidence={_format_loss_weight(weights.person_confidence)} "
        f"ref_match={_format_loss_weight(weights.ref_match)} "
        f"hard={_format_loss_weight(weights.hard_joint)} "
        f"coord_aux(decoder={coord_decoder},coarse={_format_loss_weight(weights.coarse_coord)},"
        f"deform={_format_loss_weight(weights.deform_coord)},refine={coord_refine}) "
        f"box(obj={_format_loss_weight(weights.box_objectness)},"
        f"quality={_format_loss_weight(weights.box_quality)},"
        f"l1={_format_loss_weight(weights.box_l1)},"
        f"giou={_format_loss_weight(weights.box_giou)},"
        f"relative={_format_loss_weight(weights.box_relative)}) "
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


def _draw_box(
    draw: ImageDraw.ImageDraw,
    box: torch.Tensor,
    width: int,
    height: int,
    color: tuple[int, int, int],
    label: str,
    *,
    line_width: int = 3,
) -> None:
    x1, y1, x2, y2 = box.tolist()
    xy = [x1 * width, y1 * height, x2 * width, y2 * height]
    draw.rectangle(xy, outline=color, width=max(int(line_width), 1))
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
    keypoint_visibility_threshold: float = 0.5,
    keypoint_quality_threshold: float = 0.3,
    draw_all_schema_keypoints: bool = False,
    prediction_row: dict[str, Any] | None = None,
    ref_pose_quality_alpha: float = 0.25,
    proposal_nms_iou_thresh: float = 0.65,
    proposal_objectness_threshold: float = 0.05,
    proposal_pose_threshold: float = 0.05,
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

    keypoints = outputs["keypoints"][sample_idx].detach().float().cpu()
    num_queries = int(keypoints.shape[0])
    canonical_boxes = outputs.get("pred_boxes", outputs.get("boxes"))
    if not torch.is_tensor(canonical_boxes):
        raise ValueError("Visualization requires canonical pred_boxes.")
    boxes = canonical_boxes[sample_idx, :num_queries].detach().float().cpu()
    box_mask = outputs.get("box_mask")
    valid_boxes = (
        box_mask[sample_idx, :num_queries].detach().cpu().bool()
        if box_mask is not None
        else torch.ones(boxes.shape[0], dtype=torch.bool)
    )
    person_class_logits = outputs.get("person_class_logits")
    if not torch.is_tensor(person_class_logits):
        person_class_logits = outputs["person_logits"]
    person_scores = person_class_logits[
        sample_idx, :num_queries
    ].detach().sigmoid().float().cpu()
    pred_box_logits = outputs.get("pred_box_logits")
    box_quality_logits = outputs.get("box_quality_logits")
    box_quality_scores = (
        box_quality_logits[sample_idx, :num_queries].detach().sigmoid().float().cpu()
        if torch.is_tensor(box_quality_logits)
        else torch.ones_like(person_scores)
    )
    pose_quality_logits = outputs.get("pose_quality_logits")
    pose_quality_scores = (
        pose_quality_logits[sample_idx, :num_queries].detach().sigmoid().float().cpu()
        if torch.is_tensor(pose_quality_logits)
        else torch.ones_like(person_scores)
    )
    box_scores = (
        pred_box_logits[sample_idx, :num_queries].detach().sigmoid().float().cpu()
        if torch.is_tensor(pred_box_logits)
        else person_scores * box_quality_scores
    )
    pred_pose_logits = outputs.get("pred_pose_logits")
    pose_scores = (
        pred_pose_logits[sample_idx, :num_queries].detach().sigmoid().float().cpu()
        if torch.is_tensor(pred_pose_logits)
        else person_scores * pose_quality_scores
    )
    ref_logits = outputs.get("ref_logits")
    sample_ref_logits = (
        ref_logits[sample_idx, :num_queries].detach().float().cpu()
        if torch.is_tensor(ref_logits)
        else person_scores.new_zeros(person_scores.shape)
    )
    ref_scores = sample_ref_logits.sigmoid()
    schema_valid = outputs["keypoint_valid_mask"][sample_idx].detach().cpu().bool()
    pose_lqe_joint_logits = outputs.get("pose_lqe_joint_logits")
    keypoint_quality_scores = (
        pose_lqe_joint_logits[sample_idx, :num_queries]
        .detach()
        .sigmoid()
        .float()
        .cpu()
        if torch.is_tensor(pose_lqe_joint_logits)
        else torch.ones_like(keypoints[..., 2])
    )

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

    is_refhuman = task_name == "REF_POSE" or bool(ref_description)
    valid_indices = torch.nonzero(valid_boxes, as_tuple=False).flatten()
    proposal_source_ids = outputs.get("proposal_source_ids")
    sample_source_ids = (
        proposal_source_ids[sample_idx, :num_queries].detach().cpu().long()
        if torch.is_tensor(proposal_source_ids)
        else torch.zeros(num_queries, dtype=torch.long)
    )
    input_boxes_tensor = outputs.get("input_boxes")
    input_boxes = (
        input_boxes_tensor[sample_idx, :num_queries].detach().float().cpu()
        if torch.is_tensor(input_boxes_tensor)
        else boxes
    )

    ref_probabilities = torch.zeros_like(ref_scores)
    if valid_indices.numel() > 0:
        ref_probabilities[valid_indices] = torch.softmax(
            sample_ref_logits[valid_indices], dim=0
        )
    quality_alpha = max(float(ref_pose_quality_alpha), 0.0)
    ref_final_scores = ref_probabilities * pose_scores.clamp_min(1e-8).pow(
        quality_alpha
    )

    selected_boxes: list[int] = []
    selected_poses: list[int] = []
    box_nms_survivor_count = 0
    pose_nms_survivor_count = 0
    if prediction_row is not None:
        serialized = []
        for prediction in prediction_row.get("predictions", []):
            query_idx = int(prediction.get("query", -1))
            if 0 <= query_idx < num_queries and bool(valid_boxes[query_idx]):
                serialized.append(query_idx)
        if is_refhuman and serialized:
            best = max(serialized, key=lambda idx: float(ref_final_scores[idx]))
            selected_boxes = [best]
            selected_poses = [best]
        else:
            box_ranked = sorted(
                serialized, key=lambda idx: float(box_scores[idx]), reverse=True
            )
            pose_ranked = sorted(
                serialized, key=lambda idx: float(pose_scores[idx]), reverse=True
            )
            selected_boxes = [
                idx for idx in box_ranked
                if float(box_scores[idx]) >= float(proposal_objectness_threshold)
            ][: max(int(max_instances), 0)]
            selected_poses = [
                idx for idx in pose_ranked
                if float(pose_scores[idx]) >= float(proposal_pose_threshold)
            ][: max(int(max_instances), 0)]
    elif is_refhuman and valid_indices.numel() > 0:
        best = int(valid_indices[torch.argmax(ref_final_scores[valid_indices])].item())
        selected_boxes = [best]
        selected_poses = [best]
        box_nms_survivor_count = 1
        pose_nms_survivor_count = 1
    elif valid_indices.numel() > 0:
        box_nms_local = nms_box_indices_xyxy(
            boxes[valid_indices],
            box_scores[valid_indices],
            iou_thresh=float(proposal_nms_iou_thresh),
            max_boxes=max(int(valid_indices.numel()), 1),
        )
        pose_nms_local = nms_box_indices_xyxy(
            boxes[valid_indices],
            pose_scores[valid_indices],
            iou_thresh=float(proposal_nms_iou_thresh),
            max_boxes=max(int(valid_indices.numel()), 1),
        )
        box_nms_indices = valid_indices[
            torch.as_tensor(box_nms_local, dtype=torch.long)
        ] if box_nms_local else valid_indices[:0]
        pose_nms_indices = valid_indices[
            torch.as_tensor(pose_nms_local, dtype=torch.long)
        ] if pose_nms_local else valid_indices[:0]
        box_nms_survivor_count = int(box_nms_indices.numel())
        pose_nms_survivor_count = int(pose_nms_indices.numel())
        selected_boxes = [
            int(idx) for idx in box_nms_indices[
                box_scores[box_nms_indices] >= float(proposal_objectness_threshold)
            ][: max(int(max_instances), 0)].tolist()
        ]
        selected_poses = [
            int(idx) for idx in pose_nms_indices[
                pose_scores[pose_nms_indices] >= float(proposal_pose_threshold)
            ][: max(int(max_instances), 0)].tolist()
        ]

    selected_union = list(dict.fromkeys(selected_boxes + selected_poses))

    # Draw GT independently from predictions. Aligned training/eval targets
    # store GT at the matched query slot; raw inference targets store ordinary
    # instance rows. RefHuman highlights only the referred person in green and
    # keeps at most five distractor boxes in unobtrusive gray.
    gt_entries: list[tuple[int, int, int | None]] = []
    if matched_gt_indices is not None:
        for query_idx in torch.nonzero(matched_gt_indices.ge(0), as_tuple=False).flatten().tolist():
            if query_idx >= int(gt_boxes.shape[0]):
                continue
            gt_entries.append((query_idx, int(matched_gt_indices[query_idx].item()), query_idx))
    else:
        gt_entries = [(gt_idx, gt_idx, None) for gt_idx in range(int(gt_boxes.shape[0]))]

    ref_value = target.get("ref_target", -1)
    ref_target = int(ref_value.detach().cpu().item()) if torch.is_tensor(ref_value) else int(ref_value)
    distractors_drawn = 0
    for target_idx, original_gt_idx, query_idx in gt_entries:
        referred = is_refhuman and (
            (query_idx is not None and query_idx == ref_target)
            or (query_idx is None and original_gt_idx == ref_target)
        )
        if is_refhuman and not referred:
            if distractors_drawn >= 5:
                continue
            _draw_box(
                draw,
                gt_boxes[target_idx],
                width,
                height,
                (145, 145, 145),
                f"other GT {original_gt_idx}",
                line_width=2,
            )
            distractors_drawn += 1
            continue
        color = (50, 200, 80)
        label = f"REF GT {original_gt_idx}" if is_refhuman else f"GT {original_gt_idx}"
        _draw_box(draw, gt_boxes[target_idx], width, height, color, label)
        _draw_pose(
            draw,
            gt_keypoints[target_idx],
            gt_draw_valid[target_idx],
            width,
            height,
            color,
            edge_indices,
        )

    # External source 1 is either a Stage1 noisy-GT Locate proxy or a real
    # LocateAnything box. Draw the input prior separately from the refined blue
    # prediction so visualizations cannot be mistaken for direct GT copying.
    selected_external_inputs = [
        idx for idx in selected_union if int(sample_source_ids[idx].item()) == 1
    ]
    for idx in selected_external_inputs:
        _draw_box(
            draw,
            input_boxes[idx],
            width,
            height,
            (255, 165, 40),
            f"Locate input Q{idx}",
            line_width=2,
        )

    for idx in selected_boxes:
        if is_refhuman:
            label = (
                f"Q{idx}{' ext-refined' if int(sample_source_ids[idx]) == 1 else ''} "
                f"ref={float(ref_final_scores[idx]):.2f} "
                f"box={float(box_scores[idx]):.2f} pose={float(pose_scores[idx]):.2f}"
            )
        else:
            label = (
                f"Q{idx}{' ext-refined' if int(sample_source_ids[idx]) == 1 else ''} "
                f"box={float(box_scores[idx]):.2f} "
                f"pose={float(pose_scores[idx]):.2f}"
            )
        _draw_box(
            draw,
            boxes[idx],
            width,
            height,
            (80, 180, 255),
            label,
        )

    for idx in selected_poses:
        predicted_valid = (
            schema_valid
            if draw_all_schema_keypoints
            else schema_valid
            & (keypoints[idx, :, 2] > float(keypoint_visibility_threshold))
            & (keypoint_quality_scores[idx] > float(keypoint_quality_threshold))
        )
        _draw_pose(
            draw,
            keypoints[idx],
            predicted_valid,
            width,
            height,
            (240, 70, 70),
            edge_indices,
        )
        visible_indices = torch.nonzero(predicted_valid, as_tuple=False).flatten()
        if visible_indices.numel() > 0:
            label_joint = int(visible_indices[0].item())
            label_x = int(float(keypoints[idx, label_joint, 0]) * width) + 3
            label_y = int(float(keypoints[idx, label_joint, 1]) * height) + 3
            draw.text(
                (label_x, label_y),
                f"P Q{idx}={float(pose_scores[idx]):.2f}",
                fill=(255, 120, 120),
            )

    header_width = min(width, 1100)
    header_height = 84 if is_refhuman else 64
    draw.rectangle([0, 0, header_width, header_height], fill=(0, 0, 0))
    draw.text(
        (4, 4),
        f"dataset={dataset_name} schema={schema_name} keypoints={schema_keypoint_count} task={task_name}",
        fill=(255, 255, 255),
    )
    draw.text(
        (4, 24),
        (
            "green=REF GT | gray=other GT | orange=Locate input | blue=refined box | red=pose"
            if is_refhuman
            else "green=GT | orange=Locate input | blue=pred/refined box | red=pose"
        ),
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
    status_y = 64 if is_refhuman else 44
    best_box_score = (
        float(box_scores[valid_indices].max().item())
        if valid_indices.numel() > 0
        else 0.0
    )
    best_pose_score = (
        float(pose_scores[valid_indices].max().item())
        if valid_indices.numel() > 0
        else 0.0
    )
    if prediction_row is None:
        status = (
            f"box={len(selected_boxes)}(nms={box_nms_survivor_count},best={best_box_score:.3f}) "
            f"pose={len(selected_poses)}(nms={pose_nms_survivor_count},best={best_pose_score:.3f})"
        )
        if not is_refhuman:
            status += (
                f" thresholds={float(proposal_objectness_threshold):.3f}/"
                f"{float(proposal_pose_threshold):.3f}"
            )
    else:
        status = (
            f"serialized box={len(selected_boxes)} pose={len(selected_poses)}"
        )
    draw.text((4, status_y), status, fill=(180, 220, 255))
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


def backbone_adapter_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Save the complete cross-stage Locate adapter state, including frozen LoRA.

    Stage 2 freezes the vision LoRA learned in Stage 1 while training only LLM
    LoRA. Saving only ``requires_grad`` tensors would silently drop the vision
    adapter before Stage 3, so keep every LoRA tensor plus the small mlp1
    projector state used by both LocateAnything and LocatePose.
    """
    return {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if (
            "lora_" in name
            or ".mlp1." in name
            or name.startswith("mlp1.")
            or param.requires_grad
        )
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
        adapter_state = backbone_adapter_state_dict(module.qwen_model)
        payload["backbone_trainable"] = backbone_state
        payload["backbone_adapter"] = adapter_state
        payload["qwen_trainable"] = backbone_state
    if module.qwen_extractor is not None:
        backbone_base = (
            get_eagle_base_model(module.qwen_model)
            if str(getattr(module, "backbone_name", "")) == "eagle" and module.qwen_model is not None
            else None
        )
        feature_config = {
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
        if str(getattr(module, "backbone_name", "")) == "eagle":
            feature_config["native_spatial_features"] = True
        else:
            feature_config["output_size"] = int(module.qwen_extractor.output_size)
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
    source_model_state = payload["model"]
    if bool(payload.get("allow_partial_model_init", False)):
        current_state = module.pose_model.state_dict()
        compatible_state = {
            key: value
            for key, value in source_model_state.items()
            if key in current_state and tuple(value.shape) == tuple(current_state[key].shape)
        }
        incompatible = module.pose_model.load_state_dict(compatible_state, strict=False)
        payload["partial_model_init_report"] = {
            "loaded_tensors": len(compatible_state),
            "source_tensors": len(source_model_state),
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
        }
        if is_main_process():
            print(
                "Compatible weight-only PoseHead initialization: "
                f"loaded={len(compatible_state)}/{len(source_model_state)}, "
                f"new_or_changed={len(incompatible.missing_keys)}."
            )
    else:
        module.pose_model.load_state_dict(source_model_state)
    backbone_state = payload.get(
        "backbone_adapter",
        payload.get("backbone_trainable", payload.get("qwen_trainable")),
    )
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

    backbone_state = payload.get(
        "backbone_adapter",
        payload.get("backbone_trainable", payload.get("qwen_trainable")),
    )
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


def is_visual_projector_parameter(name: str) -> bool:
    """Check if a parameter belongs to LocateAnything's full visual projector."""
    return (
        name.startswith("mlp1.")
        or ".mlp1." in name
        or name.startswith("backbone_model.mlp1.")
        or name.startswith("qwen_model.mlp1.")
    )


def build_optimizer_param_groups(
    model: torch.nn.Module,
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], dict[str, tuple[int, int, float]]]:
    grouped: dict[str, list[torch.nn.Parameter]] = {
        "pose": [],
        "backbone_lora": [],
        "backbone_vision_lora": [],
        "backbone_projector": [],
    }
    stats: dict[str, list[float]] = {
        "pose": [0, 0],
        "backbone_lora": [0, 0],
        "backbone_vision_lora": [0, 0],
        "backbone_projector": [0, 0],
    }
    backbone_name = getattr(args, "backbone", "qwen3vl")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("backbone_model.") or name.startswith("qwen_model."):
            if is_visual_projector_parameter(name):
                group_name = "backbone_projector"
            elif is_vision_parameter(name):
                group_name = "backbone_vision_lora"
            else:
                group_name = "backbone_lora"
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
        "backbone_projector": args.lr * vision_lr_scale,
    }
    param_groups = [
        {
            "params": params,
            "lr": lrs[name],
            "weight_decay": (
                args.weight_decay if name in {"pose", "backbone_projector"} else 0.0
            ),
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
    if args.dynamic_reference_offset_scale <= 0.0:
        raise ValueError("--dynamic_reference_offset_scale must be positive.")
    if args.pose_feature_channels <= 0:
        raise ValueError("Pose feature channels must be positive.")
    if args.decoder_heads <= 0 or args.hidden_dim % args.decoder_heads != 0:
        raise ValueError("--hidden_dim must be divisible by positive --decoder_heads.")
    if args.pose_decoder_layers <= 0:
        raise ValueError("Pose decoder layer count must be positive.")
    if args.deformable_points <= 0 or args.deformable_min_radius_cells <= 0.0:
        raise ValueError("Deformable sampling points and minimum radius must be positive.")
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
        args.pose_coordinate_init in {"anatomical_dynamic", "schema_prior"}
        and args.schema_joint_priors_path
        and not Path(args.schema_joint_priors_path).is_file()
    ):
        raise FileNotFoundError(
            f"--schema_joint_priors_path does not exist: {args.schema_joint_priors_path}"
        )
    decoder_coord_weights = parse_float_list(args.w_decoder_coords)
    if any(weight < 0.0 for weight in decoder_coord_weights):
        raise ValueError("--w_decoder_coords weights must be non-negative.")
    if args.letterbox_size < 0:
        raise ValueError("--letterbox_size must be non-negative.")
    if not 0 <= args.letterbox_fill <= 255:
        raise ValueError("--letterbox_fill must be in [0, 255].")
    if not 0.0 <= args.visualize_min_gt_area_ratio <= 1.0:
        raise ValueError("--visualize_min_gt_area_ratio must be in [0, 1].")
    if not 0.0 <= args.visualize_nms_iou_thresh <= 1.0:
        raise ValueError("--visualize_nms_iou_thresh must be in [0, 1].")
    if not 0.0 <= args.visualize_objectness_threshold <= 1.0:
        raise ValueError("--visualize_objectness_threshold must be in [0, 1].")
    if not 0.0 <= args.visualize_pose_threshold <= 1.0:
        raise ValueError("--visualize_pose_threshold must be in [0, 1].")
    if not 0.0 <= args.visualize_keypoint_visibility_threshold <= 1.0:
        raise ValueError(
            "--visualize_keypoint_visibility_threshold must be in [0, 1]."
        )
    if not 0.0 <= args.visualize_keypoint_quality_threshold <= 1.0:
        raise ValueError(
            "--visualize_keypoint_quality_threshold must be in [0, 1]."
        )
    if not 0.0 <= args.locate_proxy_probability <= 1.0:
        raise ValueError("--locate_proxy_probability must be in [0, 1].")
    if args.locate_proxy_center_noise < 0.0 or args.locate_proxy_scale_noise < 0.0:
        raise ValueError("Locate proxy noise scales must be non-negative.")
    if not 0.0 <= args.locate_proxy_miss_probability <= 1.0:
        raise ValueError("--locate_proxy_miss_probability must be in [0, 1].")
    if not 0.0 <= args.locate_proxy_duplicate_probability <= 1.0:
        raise ValueError("--locate_proxy_duplicate_probability must be in [0, 1].")
    if args.locate_proxy_probability > 0.0 and args.box_source != "person_queries":
        raise ValueError(
            "--locate_proxy_probability is only valid with --box_source=person_queries."
        )
    if min(
        args.w_box_objectness,
        args.w_box_quality,
        args.w_box_l1,
        args.w_box_giou,
        args.w_box_relative,
    ) < 0.0:
        raise ValueError("All box refinement loss weights must be non-negative.")
    if args.w_keypoint_confidence < 0.0:
        raise ValueError("--w_keypoint_confidence must be non-negative.")
    if args.w_keypoint_quality < 0.0:
        raise ValueError("--w_keypoint_quality must be non-negative.")
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
    selective_llm = args.locate_train_scope in {"selective_llm_lora", "selective_lora"}
    selective_vision = args.locate_train_scope in {"selective_vision_lora", "selective_lora"}
    if selective_llm or selective_vision:
        llm_layers = parse_layer_selection(args.locate_llm_layers)
        vision_layers = parse_layer_selection(args.locate_vision_layers)
        llm_modules = parse_module_selection(args.locate_llm_modules)
        vision_modules = parse_module_selection(args.locate_vision_modules)
        if selective_llm and (not llm_layers or max(llm_layers) >= 36):
            raise ValueError("--locate_llm_layers must select Qwen2.5 layers in [0, 35].")
        if selective_vision and (not vision_layers or max(vision_layers) >= 27):
            raise ValueError("--locate_vision_layers must select MoonViT blocks in [0, 26].")
        if selective_llm and not llm_modules:
            raise ValueError("Selective LLM LoRA module selection must be non-empty.")
        if selective_vision and not vision_modules:
            raise ValueError("Selective vision LoRA module selection must be non-empty.")
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
    if args.locate_grounding_only:
        if args.backbone != "eagle":
            raise ValueError("--locate_grounding_only is supported only by LocateAnything/eagle.")
        if not args.freeze_pose:
            raise ValueError("--locate_grounding_only requires --freeze_pose.")
        if args.freeze_eagle or args.locate_train_scope == "frozen":
            raise ValueError("Grounding-only training requires trainable LocateAnything LLM adapters.")
        if args.locate_feature_source != "raw_visual":
            raise ValueError("Grounding-only training requires --locate_feature_source=raw_visual.")
        if args.disable_locate_grounding_aux:
            raise ValueError("Grounding-only training cannot disable Locate grounding supervision.")
        if args.w_locate_box_lm <= 0.0:
            raise ValueError("Grounding-only training requires a positive LM loss weight.")
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
        load_image_tensors=False,
        # Fixed-size letterboxing changes both pixels and annotation geometry, so
        # every LocateAnything stage consumes the materialized 800x800 tensor.
        load_vision_images=(
            args.backbone == "eagle"
            and (bool(args.pose_augment) or int(args.letterbox_size) > 0)
        ),
        augment_config=augment_config,
        use_prompts=args.locate_feature_source != "vision_only",
        letterbox_size=(
            int(args.letterbox_size)
            if args.backbone == "eagle" and int(args.letterbox_size) > 0
            else None
        ),
        letterbox_fill=int(args.letterbox_fill),
        split=args.split,
        max_samples_per_dataset=args.max_samples_per_dataset,
        refhuman_max_captions_per_instance=args.refhuman_max_captions_per_instance,
        mixing_strategy=args.mixing_strategy,
        dataset_mix_weights=args.dataset_mix_weights,
        seed=args.seed,
        record_cache_dir=args.record_cache_dir,
        disable_record_cache=args.disable_record_cache,
        show_progress=not args.disable_progress,
        refhuman_text_embedding_cache=args.refhuman_text_embedding_cache,
        require_cached_refhuman_text=args.locate_feature_source == "vision_only",
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
            bucket_spatial_shapes=(
                balance_vision_tokens and not args.disable_spatial_shape_bucketing
            ),
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
                f"spatial_shape_bucketing={batch_sampler.bucket_spatial_shapes}, "
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
    high_res_external_dim = 0
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
        eagle_base = get_eagle_base_model(backbone_model)
        vision_config = getattr(getattr(eagle_base, "vision_model", None), "config", None)
        high_res_external_dim = int(
            getattr(vision_config, "hidden_size", 1152)
        )
        bb_trainable, bb_total = count_eagle_lora_parameters(backbone_model)
        if is_main_process():
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
        if saved_pose_config.get("use_native_spatial_features") is not True:
            raise ValueError(
                "The initialization checkpoint predates native-grid Locate pose features. "
                "Train a new Stage1 checkpoint, then use it for Stage2."
            )
        if saved_pose_config.get("use_detrpose_architecture") is not True:
            raise ValueError(
                "The initialization checkpoint uses the retired ROI pose graph. "
                "The no-ROI DETRPose architecture requires a fresh Stage1 checkpoint."
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
                "Architecture-changing coordinate modes require a new matching Stage1 run."
            )
        model_config = QwenPoseConfig(
            hidden_dim=int(saved_pose_config.get("hidden_dim", args.hidden_dim)),
            external_dim=external_dim,
            high_res_external_dim=int(
                saved_pose_config.get("high_res_external_dim", high_res_external_dim)
            ),
            pose_decoder_layers=int(saved_pose_config.get("pose_decoder_layers", args.pose_decoder_layers)),
            refinement_steps=int(saved_pose_config.get("refinement_steps", args.refinement_steps)),
            decoder_heads=int(saved_pose_config.get("decoder_heads", args.decoder_heads)),
            dropout=float(saved_pose_config.get("dropout", args.pose_dropout)),
            box_condition_scale=float(saved_pose_config.get("box_condition_scale", args.box_condition_scale)),
            use_refinement=bool(saved_pose_config.get("use_refinement", not args.disable_refinement)),
            pose_feature_channels=int(saved_pose_config["pose_feature_channels"]),
            deformable_points=int(saved_pose_config.get("deformable_points", args.deformable_points)),
            deformable_min_radius_cells=float(
                saved_pose_config.get("deformable_min_radius_cells", args.deformable_min_radius_cells)
            ),
            deformable_scale_prior_strength=float(
                saved_pose_config.get("deformable_scale_prior_strength", 0.5)
            ),
            deformable_scale_prior_center_cells=float(
                saved_pose_config.get("deformable_scale_prior_center_cells", 6.0)
            ),
            deformable_scale_prior_temperature=float(
                saved_pose_config.get("deformable_scale_prior_temperature", 1.5)
            ),
            enable_keypoint_denoising=bool(
                saved_pose_config.get(
                    "enable_keypoint_denoising", not args.disable_keypoint_denoising
                )
            ),
            ref_text_scale=float(
                saved_pose_config.get("ref_text_scale", args.ref_text_scale)
            ),
            legacy_checkpoint_compat=False,
            enable_person_confidence_head=bool(
                saved_pose_config.get("enable_person_confidence_head", True)
            ),
            person_confidence_rescue=False,
            use_global_person_queries=True,
            num_person_queries=int(
                saved_pose_config.get("num_person_queries", args.num_person_queries)
            ),
            num_ref_queries=int(
                saved_pose_config.get("num_ref_queries", args.num_ref_queries)
            ),
            multiscale_encoder_layers=int(
                saved_pose_config.get(
                    "multiscale_encoder_layers", args.multiscale_encoder_layers
                )
            ),
            multiscale_encoder_points=int(
                saved_pose_config.get(
                    "multiscale_encoder_points", args.multiscale_encoder_points
                )
            ),
            use_detrpose_architecture=True,
            pose_coordinate_init=saved_coordinate_init,
            schema_joint_priors_path=str(
                saved_pose_config.get(
                    "schema_joint_priors_path", args.schema_joint_priors_path
                )
            ),
            dynamic_reference_offset_scale=float(
                saved_pose_config.get(
                    "dynamic_reference_offset_scale",
                    args.dynamic_reference_offset_scale,
                )
            ),
        )
    else:
        model_config = QwenPoseConfig(
            hidden_dim=args.hidden_dim,
            external_dim=external_dim,
            high_res_external_dim=high_res_external_dim,
            pose_decoder_layers=args.pose_decoder_layers,
            refinement_steps=args.refinement_steps,
            decoder_heads=args.decoder_heads,
            dropout=args.pose_dropout,
            box_condition_scale=args.box_condition_scale,
            use_refinement=not args.disable_refinement,
            pose_feature_channels=args.pose_feature_channels,
            deformable_points=args.deformable_points,
            deformable_min_radius_cells=args.deformable_min_radius_cells,
            deformable_scale_prior_strength=0.5,
            deformable_scale_prior_center_cells=6.0,
            deformable_scale_prior_temperature=1.5,
            enable_keypoint_denoising=not args.disable_keypoint_denoising,
            ref_text_scale=args.ref_text_scale,
            legacy_checkpoint_compat=False,
            enable_person_confidence_head=True,
            person_confidence_rescue=False,
            use_global_person_queries=True,
            num_person_queries=args.num_person_queries,
            num_ref_queries=args.num_ref_queries,
            multiscale_encoder_layers=args.multiscale_encoder_layers,
            multiscale_encoder_points=args.multiscale_encoder_points,
            use_detrpose_architecture=True,
            pose_coordinate_init=args.pose_coordinate_init,
            schema_joint_priors_path=args.schema_joint_priors_path,
            dynamic_reference_offset_scale=args.dynamic_reference_offset_scale,
        )
    if len(decoder_coord_weights) != int(model_config.pose_decoder_layers):
        raise ValueError(
            "--w_decoder_coords must provide exactly one weight per actual pose "
            "decoder layer: "
            f"layers={model_config.pose_decoder_layers}, weights={decoder_coord_weights}."
        )
    requested_person_queries = True
    if int(model_config.num_person_queries) != int(args.num_person_queries):
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
    if args.freeze_pose:
        for parameter in model.parameters():
            parameter.requires_grad = False
    trainable, total = count_trainable_parameters(model)
    if is_main_process():
        print(f"Pose module trainable parameters: {trainable:,} / {total:,}")
    # Select feature extractor params based on backbone
    if backbone_name == "eagle":
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
        pose_condition_box_mode=(
            "input" if args.box_source == "gt" else "refined_detached"
        ),
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

    # DeepSpeed creates FP32 optimizer master parameters from the model values
    # present at ``deepspeed.initialize`` time.  Loading a lightweight
    # (model-only) checkpoint after that point updates the module parameters but
    # leaves those master copies stale; the first optimizer step then writes the
    # pre-checkpoint/random values back into the model.  Full DeepSpeed
    # checkpoints must still be restored through ``engine.load_checkpoint``, but
    # model-only resumes have to be applied before the optimizer is constructed.
    preloaded_deepspeed_resume: tuple[int, Path, dict[str, object]] | None = None
    if use_deepspeed and args.resume_from_checkpoint is not None:
        candidate_checkpoint = resolve_training_checkpoint(args.resume_from_checkpoint)
        has_deepspeed_state = (
            candidate_checkpoint.is_dir()
            and (candidate_checkpoint / DEEPSPEED_TAG).exists()
        )
        if not has_deepspeed_state:
            (
                preloaded_step,
                _,
                _,
                preloaded_checkpoint,
                preloaded_payload,
            ) = load_training_checkpoint(
                training_model,
                optimizer=None,
                checkpoint_path=candidate_checkpoint,
                load_optimizer=False,
                scaler=None,
                load_scaler=False,
            )
            preloaded_deepspeed_resume = (
                preloaded_step,
                preloaded_checkpoint,
                preloaded_payload,
            )
            if is_main_process():
                print(
                    "Preloaded lightweight checkpoint before DeepSpeed optimizer "
                    f"initialization: {preloaded_checkpoint}"
                )
    if is_main_process():
        print(f"Backbone: {backbone_name}")
        print(
            "Feature grid size: native_dynamic"
            if backbone_name == "eagle"
            else f"Feature grid size: {feature_size}x{feature_size}"
        )
        print(
            "Model dimensions: "
            f"hidden_dim={model_config.hidden_dim}, "
            f"external_dim={model_config.external_dim}, "
            f"decoder_heads={model_config.decoder_heads}, "
            f"pose_decoder_layers={model_config.pose_decoder_layers}, "
            f"refinement_steps={model_config.refinement_steps}, "
            f"dropout={model_config.dropout}"
        )
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
        print(f"Box context scale: {model_config.box_condition_scale}")
        if backbone_name == "eagle":
            print(
                "Locate pose feature: "
                f"channels={model_config.pose_feature_channels}, grid=native_dynamic, "
                f"pose_decoder_layers={model_config.pose_decoder_layers}"
            )
        else:
            print(
                "Qwen pose feature: "
                f"channels={model_config.pose_feature_channels}, "
                f"grid={int(args.qwen_feature_size)}x{int(args.qwen_feature_size)}, "
                f"pose_decoder_layers={model_config.pose_decoder_layers}"
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
        if model_config.pose_coordinate_init == "anatomical_dynamic":
            coordinate_message = (
                "mode=anatomical_dynamic, main_reference=schema_prior+instance_offset, "
                "decoder=grouped_iterative"
            )
        elif model_config.pose_coordinate_init == "learned_spread":
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
        keypoint_quality=args.w_keypoint_quality,
        person_confidence=args.w_person_confidence,
        ref_match=args.w_ref_match,
        lm=(0.0 if backbone_name == "eagle" else args.w_lm),
        hard_joint=args.w_hard_joint,
        hard_joint_fraction=args.hard_joint_fraction,
        box_objectness=args.w_box_objectness,
        box_quality=args.w_box_quality,
        box_l1=args.w_box_l1,
        box_giou=args.w_box_giou,
        box_relative=args.w_box_relative,
        keypoint_dn=args.w_keypoint_dn,
        decoder_coords=parse_float_list(args.w_decoder_coords),
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
        print(
            "Locate grounding loss: "
            f"box_lm={args.w_locate_box_lm}, "
            f"lm_every_micro_steps={args.locate_lm_loss_every}"
        )
        print(
            "Training/logging intervals: "
            f"world_size={world_size}, batch_per_gpu={args.batch_size}, "
            f"grad_accum={args.grad_accum_steps}, "
            f"effective_global_batch={world_size * args.batch_size * args.grad_accum_steps}, "
            f"warmup_steps={args.warmup_steps}, log_every={args.log_every}, "
            f"save_every={args.save_every}, save_total_limit={args.save_total_limit}, "
            f"visualize_every={args.visualize_every}, "
            f"visualize_max_instances={args.visualize_max_instances}, "
            f"visualize_min_gt_area_ratio={args.visualize_min_gt_area_ratio}, "
            f"visualize_nms_iou={args.visualize_nms_iou_thresh}, "
            f"visualize_objectness_threshold={args.visualize_objectness_threshold}, "
            f"visualize_pose_threshold={args.visualize_pose_threshold}, "
            f"batch_trace={not args.disable_batch_trace}"
        )
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
        elif preloaded_deepspeed_resume is not None:
            global_step, resolved_checkpoint, payload = preloaded_deepspeed_resume
            resume_container = dict(payload)
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
                if (
                    not args.locate_grounding_only
                    and not args.disable_keypoint_denoising
                ):
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
                    if keypoint_dn_batch is not None:
                        dn_batch.update(keypoint_dn_batch)
                if backbone_processor is None:
                    qwen_inputs = None
                elif backbone_name == "eagle":
                    if not args.locate_grounding_only:
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
                    if args.locate_grounding_only or use_lm_loss:
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
                            batch_token_limit=args.eagle_batch_token_limit,
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
                        run_pose=not args.locate_grounding_only,
                        target_boxes=target_boxes,
                        target_box_mask=target_box_mask,
                        images=batch.get("images"),
                        cached_text_embed=(
                            batch.get("cached_text_embeddings")
                            if torch.is_tensor(batch.get("cached_text_embeddings"))
                            and int(batch["cached_text_embeddings"].shape[-1]) > 0
                            else None
                        ),
                        cached_text_mask=batch.get("cached_text_embedding_mask"),
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
                    if (
                        not args.locate_grounding_only
                        and torch.is_tensor(outputs.get("pose_set_prediction"))
                        and bool(outputs["pose_set_prediction"].detach().item())
                    ):
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
                    if args.locate_grounding_only:
                        floating_outputs = [
                            tensor for _, tensor in iter_named_floating_tensors(outputs)
                        ]
                        if not floating_outputs:
                            raise RuntimeError("Grounding-only forward returned no floating tensors.")
                        loss = sum(tensor.sum() * 0.0 for tensor in floating_outputs)
                        loss_dict = {"loss_total": loss}
                    elif args.person_confidence_rescue:
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
                    if not args.locate_grounding_only:
                        loss_dict.update(
                            compute_pose_diagnostics(outputs, pose_targets, batch["task_ids"])
                        )
                        direct_failure_flags = [
                            target["locate_direct_grounding_failed"].float()
                            for target in pose_targets
                            if torch.is_tensor(target.get("locate_direct_grounding_failed"))
                        ]
                        if direct_failure_flags:
                            loss_dict["locate_direct_grounding_failure_rate"] = torch.stack(
                                direct_failure_flags
                            ).mean().to(device=loss.device)
                        proxy_flags = [
                            target["locate_proxy_active"].float()
                            for target in pose_targets
                            if torch.is_tensor(target.get("locate_proxy_active"))
                        ]
                        loss_dict["locate_proxy_rate"] = (
                            torch.stack(proxy_flags).mean().to(device=loss.device)
                            if proxy_flags
                            else torch.zeros((), device=loss.device)
                        )
                    if "lm_loss" in outputs:
                        lm_loss = outputs["lm_loss"].float()
                        lm_weight = (
                            (float(args.w_locate_box_lm) if use_lm_loss else 0.0)
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
                    loss_dict["loss_total"] = loss

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
                                keypoint_visibility_threshold=args.visualize_keypoint_visibility_threshold,
                                keypoint_quality_threshold=args.visualize_keypoint_quality_threshold,
                                draw_all_schema_keypoints=False,
                                proposal_nms_iou_thresh=args.visualize_nms_iou_thresh,
                                proposal_objectness_threshold=args.visualize_objectness_threshold,
                                proposal_pose_threshold=args.visualize_pose_threshold,
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
