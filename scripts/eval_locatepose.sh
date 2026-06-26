#!/usr/bin/env bash
set -Eeuo pipefail

###############################################################################
# LocatePose 验证脚本
#
# 默认行为：
#   1. 自动定位 outputs/locatepose 下最近一次 run；
#   2. 优先选择 stage2_locate_box_closed_loop 的最新 checkpoint；
#   3. 使用 LocateAnything 生成框闭环评估 PoseHead；
#   4. 导出 summary.json、predictions.jsonl、predictions.json、report.md。
#
# 如需查看 GT-box-conditioned 上限，可设置 BOX_SOURCE=gt。
###############################################################################

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH_REL="scripts/$(basename "${BASH_SOURCE[0]}")"

resolve_default_python() {
  if [[ -x "${PROJECT_ROOT}/envs/qwenpose/bin/python" ]]; then
    printf '%s\n' "${PROJECT_ROOT}/envs/qwenpose/bin/python"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi
  echo "No Python interpreter found. Set PYTHON=/path/to/python before running ${SCRIPT_PATH_REL}." >&2
  exit 1
}

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

DEFAULT_PYTHON="$(resolve_default_python)"
PYTHON="${PYTHON:-${DEFAULT_PYTHON}}"

resolve_latest_train_dir() {
  local root="$1"
  local latest=""
  if [[ -d "${root}" ]]; then
    latest="$(find "${root}" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
  fi
  if [[ -n "${latest}" ]]; then
    printf '%s\n' "${latest}"
  else
    printf '%s\n' "${root}"
  fi
}

dir_has_checkpoints() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    return 1
  fi
  if [[ -f "${path}/qwenpose_checkpoint.pt" || -e "${path}/deepspeed" ]]; then
    return 0
  fi
  find "${path}" -maxdepth 1 \( -name 'checkpoint-*' -o -name 'checkpoint_step_*.pt' \) -print -quit | grep -q .
}

resolve_default_checkpoint_target() {
  local run_dir="$1"
  local stage2_dir="${run_dir}/stage2_locate_box_closed_loop"
  local stage1_dir="${run_dir}/stage1_freeze_locate_gt_box"
  if dir_has_checkpoints "${stage2_dir}"; then
    printf '%s\n' "${stage2_dir}"
  elif dir_has_checkpoints "${stage1_dir}"; then
    printf '%s\n' "${stage1_dir}"
  else
    printf '%s\n' "${run_dir}"
  fi
}

EVAL_TS="${EVAL_TS:-$(date +%Y%m%d-%H%M%S)}"
TRAIN_OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT:-outputs/locatepose}"
DEFAULT_TRAIN_OUTPUT_DIR="$(resolve_latest_train_dir "${TRAIN_OUTPUT_ROOT}")"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-${DEFAULT_TRAIN_OUTPUT_DIR}}"
DEFAULT_CHECKPOINT_TARGET="$(resolve_default_checkpoint_target "${TRAIN_OUTPUT_DIR}")"
CHECKPOINT="${CHECKPOINT:-${1:-${DEFAULT_CHECKPOINT_TARGET}}}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${TRAIN_OUTPUT_DIR}/eval_locatepose_${EVAL_TS}}"

DATASET_ROOT="${DATASET_ROOT:-datasets}"
DATASETS="${DATASETS:-coco}"
SPLIT="${SPLIT:-val}"
MAX_SAMPLES_PER_DATASET="${MAX_SAMPLES_PER_DATASET:-}"
RECORD_CACHE_DIR="${RECORD_CACHE_DIR:-.cache/qwenpose_records}"
DISABLE_RECORD_CACHE="${DISABLE_RECORD_CACHE:-0}"
PROGRESS_BAR="${PROGRESS_BAR:-1}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_INSTANCES="${MAX_INSTANCES:-80}"

LOCATE_MODEL_PATH="${LOCATE_MODEL_PATH:-weights/LocateAnything-3B}"
LOCATE_DTYPE="${LOCATE_DTYPE:-bfloat16}"
LOCATE_ATTN_IMPLEMENTATION="${LOCATE_ATTN_IMPLEMENTATION:-sdpa}"
LOCATE_MIN_PIXELS="${LOCATE_MIN_PIXELS:-}"
LOCATE_MAX_PIXELS="${LOCATE_MAX_PIXELS:-}"
LOCATE_IMAGE_TOKEN_LIMIT="${LOCATE_IMAGE_TOKEN_LIMIT:-4096}"
LOCATE_FEATURE_SIZE="${LOCATE_FEATURE_SIZE:-64}"
LOCATE_FEATURE_REFINER_LAYERS="${LOCATE_FEATURE_REFINER_LAYERS:-2}"
LOCATE_FEATURE_REFINER_BOTTLENECK_DIM="${LOCATE_FEATURE_REFINER_BOTTLENECK_DIM:-256}"
LOCATE_FEATURE_REFINER_INIT_SCALE="${LOCATE_FEATURE_REFINER_INIT_SCALE:-0.1}"
LOCATE_LORA_R="${LOCATE_LORA_R:-32}"
LOCATE_LORA_ALPHA="${LOCATE_LORA_ALPHA:-64}"
LOCATE_LORA_DROPOUT="${LOCATE_LORA_DROPOUT:-0.05}"
LOCATE_VISION_LORA_R="${LOCATE_VISION_LORA_R:-16}"
LOCATE_VISION_LORA_ALPHA="${LOCATE_VISION_LORA_ALPHA:-32}"
LOCATE_VISION_LORA_DROPOUT="${LOCATE_VISION_LORA_DROPOUT:-0.05}"

HIDDEN_DIM="${HIDDEN_DIM:-448}"
POSE_DECODER_LAYERS="${POSE_DECODER_LAYERS:-3}"
REFINEMENT_STEPS="${REFINEMENT_STEPS:-3}"
DECODER_HEADS="${DECODER_HEADS:-8}"
BOX_CONDITION_SCALE="${BOX_CONDITION_SCALE:-1.2}"
POSE_ROI_SIZE="${POSE_ROI_SIZE:-16}"

BOX_SOURCE="${BOX_SOURCE:-locate_generate}"
LOCATE_GENERATION_MODE="${LOCATE_GENERATION_MODE:-hybrid}"
LOCATE_BOX_MAX_NEW_TOKENS="${LOCATE_BOX_MAX_NEW_TOKENS:-8192}"
BOX_MATCH_IOU_THRESH="${BOX_MATCH_IOU_THRESH:-0.10}"
BOX_NMS_IOU_THRESH="${BOX_NMS_IOU_THRESH:-0.70}"

VISUALIZE_MAX_SAMPLES="${VISUALIZE_MAX_SAMPLES:-100}"
VISUALIZE_MAX_INSTANCES="${VISUALIZE_MAX_INSTANCES:-8}"
SCORE_THRESHOLD="${SCORE_THRESHOLD:-0.05}"
MAX_PREDICTIONS_PER_IMAGE="${MAX_PREDICTIONS_PER_IMAGE:-100}"

W_OKS="${W_OKS:-0.2}"
W_COORD="${W_COORD:-5.0}"
W_VIS="${W_VIS:-0.05}"
W_HARD_JOINT="${W_HARD_JOINT:-0.0}"
HARD_JOINT_FRACTION="${HARD_JOINT_FRACTION:-0.2}"

args=(
  --checkpoint "${CHECKPOINT}"
  --backbone locatepose
  --dataset_root "${DATASET_ROOT}"
  --datasets "${DATASETS}"
  --split "${SPLIT}"
  --output_dir "${EVAL_OUTPUT_DIR}"
  --device "${DEVICE}"
  --num_workers "${NUM_WORKERS}"
  --prefetch_factor "${PREFETCH_FACTOR}"
  --batch_size "${BATCH_SIZE}"
  --max_instances "${MAX_INSTANCES}"
  --record_cache_dir "${RECORD_CACHE_DIR}"
  --locate_model_path "${LOCATE_MODEL_PATH}"
  --locate_dtype "${LOCATE_DTYPE}"
  --locate_attn_implementation "${LOCATE_ATTN_IMPLEMENTATION}"
  --locate_image_token_limit "${LOCATE_IMAGE_TOKEN_LIMIT}"
  --locate_feature_size "${LOCATE_FEATURE_SIZE}"
  --locate_feature_refiner_layers "${LOCATE_FEATURE_REFINER_LAYERS}"
  --locate_feature_refiner_bottleneck_dim "${LOCATE_FEATURE_REFINER_BOTTLENECK_DIM}"
  --locate_feature_refiner_init_scale "${LOCATE_FEATURE_REFINER_INIT_SCALE}"
  --locate_lora_r "${LOCATE_LORA_R}"
  --locate_lora_alpha "${LOCATE_LORA_ALPHA}"
  --locate_lora_dropout "${LOCATE_LORA_DROPOUT}"
  --locate_vision_lora_r "${LOCATE_VISION_LORA_R}"
  --locate_vision_lora_alpha "${LOCATE_VISION_LORA_ALPHA}"
  --locate_vision_lora_dropout "${LOCATE_VISION_LORA_DROPOUT}"
  --hidden_dim "${HIDDEN_DIM}"
  --pose_decoder_layers "${POSE_DECODER_LAYERS}"
  --refinement_steps "${REFINEMENT_STEPS}"
  --decoder_heads "${DECODER_HEADS}"
  --box_condition_scale "${BOX_CONDITION_SCALE}"
  --pose_roi_size "${POSE_ROI_SIZE}"
  --box_source "${BOX_SOURCE}"
  --locate_generation_mode "${LOCATE_GENERATION_MODE}"
  --locate_box_max_new_tokens "${LOCATE_BOX_MAX_NEW_TOKENS}"
  --box_match_iou_thresh "${BOX_MATCH_IOU_THRESH}"
  --box_nms_iou_thresh "${BOX_NMS_IOU_THRESH}"
  --score_threshold "${SCORE_THRESHOLD}"
  --max_predictions_per_image "${MAX_PREDICTIONS_PER_IMAGE}"
  --visualize_max_samples "${VISUALIZE_MAX_SAMPLES}"
  --visualize_max_instances "${VISUALIZE_MAX_INSTANCES}"
  --w_oks "${W_OKS}"
  --w_coord "${W_COORD}"
  --w_vis "${W_VIS}"
  --w_hard_joint "${W_HARD_JOINT}"
  --hard_joint_fraction "${HARD_JOINT_FRACTION}"
)

[[ -n "${MAX_SAMPLES_PER_DATASET}" ]] && args+=(--max_samples_per_dataset "${MAX_SAMPLES_PER_DATASET}")
[[ -n "${LOCATE_MIN_PIXELS}" ]] && args+=(--locate_min_pixels "${LOCATE_MIN_PIXELS}")
[[ -n "${LOCATE_MAX_PIXELS}" ]] && args+=(--locate_max_pixels "${LOCATE_MAX_PIXELS}")
[[ "${DISABLE_RECORD_CACHE}" == "1" ]] && args+=(--disable_record_cache)
[[ "${PROGRESS_BAR}" == "0" ]] && args+=(--disable_progress)

"${PYTHON}" -m qwenpose.eval_pose "${args[@]}"
