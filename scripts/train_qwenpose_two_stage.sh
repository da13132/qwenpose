#!/usr/bin/env bash
set -Eeuo pipefail

###############################################################################
# QwenPose 两阶段训练脚本
#
# Stage 1 / GT-box pose warmup:
#   冻结 Qwen，PoseHead 使用 GT box，默认训练 5 epoch。
# Stage 2 / Closed-loop Qwen-box training:
#   Qwen generate bbox JSON，解析后作为 PoseHead 条件框；GT box 只用于匹配和监督。
###############################################################################

DEFAULT_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"
SCRIPT_PATH_REL="scripts/$(basename "${BASH_SOURCE[0]}")"

print_usage() {
  cat <<EOF
Usage:
  ${SCRIPT_PATH_REL} [--resume <checkpoint_or_run_dir>] [--VAR VALUE|--VAR=VALUE]...

Options:
  --resume PATH   Resume from a run dir, stage dir, checkpoint dir, or checkpoint file.
                  A run dir prefers stage2_qwen_box_closed_loop; if only stage1 exists,
                  stage2 is initialized from stage1 weights.
  --VAR VALUE     Override any script variable. Supports ALL_CAPS, snake_case, and kebab-case.
  --VAR=VALUE     Same as above, using inline assignment.
  -h, --help      Show this help message.
EOF
}

normalize_cli_var_name() {
  local raw_name="$1"
  local normalized="${raw_name//-/_}"
  printf '%s\n' "${normalized^^}"
}

is_cli_var_name() {
  local normalized_name
  normalized_name="$(normalize_cli_var_name "$1")"
  [[ "${normalized_name}" =~ ^[A-Z][A-Z0-9_]*$ ]]
}

set_cli_var() {
  local raw_name="$1"
  local value="$2"
  local name
  name="$(normalize_cli_var_name "${raw_name}")"
  if ! is_cli_var_name "${raw_name}"; then
    echo "Unsupported argument: --${raw_name}" >&2
    print_usage >&2
    exit 1
  fi
  printf -v "${name}" '%s' "${value}"
  export "${name}"
}

CLI_RESUME_PATH="${CLI_RESUME_PATH:-}"
while (($# > 0)); do
  case "$1" in
    --resume)
      shift
      if (($# == 0)); then
        echo "--resume requires a path argument." >&2
        exit 1
      fi
      CLI_RESUME_PATH="$1"
      ;;
    --resume=*)
      CLI_RESUME_PATH="${1#*=}"
      if [[ -z "${CLI_RESUME_PATH}" ]]; then
        echo "--resume requires a non-empty path argument." >&2
        exit 1
      fi
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    --*=*)
      cli_name="${1%%=*}"
      cli_value="${1#*=}"
      cli_name="${cli_name#--}"
      set_cli_var "${cli_name}" "${cli_value}"
      ;;
    --*)
      cli_name="${1#--}"
      shift
      if (($# == 0)); then
        echo "--${cli_name} requires a value argument." >&2
        exit 1
      fi
      set_cli_var "${cli_name}" "$1"
      ;;
    *)
      echo "Unsupported argument: $1" >&2
      print_usage >&2
      exit 1
      ;;
  esac
  shift
done

PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"

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

resolve_default_torchrun() {
  if [[ -x "${PROJECT_ROOT}/envs/qwenpose/bin/torchrun" ]]; then
    printf '%s\n' "${PROJECT_ROOT}/envs/qwenpose/bin/torchrun"
    return 0
  fi
  if command -v torchrun >/dev/null 2>&1; then
    command -v torchrun
    return 0
  fi
  printf '\n'
}

DEFAULT_PYTHON="$(resolve_default_python)"
PYTHON="${PYTHON:-${DEFAULT_PYTHON}}"
DEFAULT_TORCHRUN="$(resolve_default_torchrun)"
TORCHRUN="${TORCHRUN:-${DEFAULT_TORCHRUN}}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

resolve_cli_resume_context() {
  "${PYTHON}" - "$1" <<'PY'
from __future__ import annotations
import shlex
import sys
from pathlib import Path

CHECKPOINT_PAYLOAD_NAME = "qwenpose_checkpoint.pt"
STAGE1_NAME = "stage1_freeze_qwen"
STAGE2_NAMES = (
    "stage2_qwen_box_closed_loop",
    "stage3_qwen_box_closed_loop",  # legacy from the removed three-stage layout
    "stage2_teacher_forcing",       # legacy middle-stage checkpoints can initialize closed-loop stage2
    "stage2_qwen_lora_lm",          # legacy public snapshot name
)

def shell_assign(name: str, value: str) -> str:
    return f"{name}={shlex.quote(value)}"

def has_checkpoint_payload(path: Path) -> bool:
    if path.is_file():
        return path.name == CHECKPOINT_PAYLOAD_NAME or path.name.startswith("checkpoint_step_")
    if not path.is_dir():
        return False
    if (path / CHECKPOINT_PAYLOAD_NAME).is_file() or (path / "deepspeed").exists():
        return True
    return any(path.glob("checkpoint-*")) or any(path.glob("checkpoint_step_*.pt"))

def has_direct_checkpoint_children(path: Path) -> bool:
    return path.is_dir() and (any(path.glob("checkpoint-*")) or any(path.glob("checkpoint_step_*.pt")))

def latest_log_file(run_dir: Path) -> str:
    log_dir = run_dir / "logs"
    if not log_dir.is_dir():
        return ""
    logs = sorted(log_dir.glob("train_*.log"))
    return str(logs[-1]) if logs else ""

def first_existing_stage(root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = root / name
        if candidate.exists():
            return candidate
    return None

target = Path(sys.argv[1]).expanduser().resolve()
if not target.exists():
    raise FileNotFoundError(f"Resume path not found: {target}")

run_dir = target
stage1_dir = ""
stage2_dir = ""
stage2_resume = "none"
stage2_init = ""
run_stage1 = "0"
run_stage2 = "1"

def set_from_run(root: Path) -> None:
    global run_dir, stage1_dir, stage2_dir, stage2_resume, stage2_init, run_stage1, run_stage2
    run_dir = root
    s1 = root / STAGE1_NAME
    s2 = first_existing_stage(root, STAGE2_NAMES)
    stage1_dir = str(s1) if s1.exists() else ""
    stage2_dir = str(s2) if s2 is not None else str(root / STAGE2_NAMES[0])
    run_stage1 = "0"
    run_stage2 = "1"
    if s2 is not None and has_checkpoint_payload(s2):
        # Final closed-loop stage checkpoints resume directly. Legacy teacher-forcing stage dirs initialize stage2.
        if s2.name in ("stage2_qwen_box_closed_loop", "stage3_qwen_box_closed_loop"):
            stage2_resume = str(s2)
        else:
            stage2_init = str(s2)
    elif s1.exists() and has_checkpoint_payload(s1):
        stage2_init = str(s1)

if target.is_dir() and ((target / STAGE1_NAME).is_dir() or first_existing_stage(target, STAGE2_NAMES) is not None):
    set_from_run(target)
elif target.is_dir() and target.name in STAGE2_NAMES:
    run_dir = target.parent
    stage1_candidate = run_dir / STAGE1_NAME
    stage1_dir = str(stage1_candidate) if stage1_candidate.exists() else ""
    stage2_dir = str(target)
    if target.name in ("stage2_qwen_box_closed_loop", "stage3_qwen_box_closed_loop"):
        stage2_resume = str(target)
    else:
        stage2_init = str(target)
elif target.parent.name in STAGE2_NAMES:
    run_dir = target.parent.parent
    stage1_candidate = run_dir / STAGE1_NAME
    stage1_dir = str(stage1_candidate) if stage1_candidate.exists() else ""
    stage2_dir = str(target.parent)
    if target.parent.name in ("stage2_qwen_box_closed_loop", "stage3_qwen_box_closed_loop"):
        stage2_resume = str(target)
    else:
        stage2_init = str(target)
elif target.is_dir() and target.name == STAGE1_NAME:
    run_dir = target.parent
    stage1_dir = str(target)
    stage2_dir = str(run_dir / STAGE2_NAMES[0])
    stage2_init = str(target)
elif target.parent.name == STAGE1_NAME:
    run_dir = target.parent.parent
    stage1_dir = str(target.parent)
    stage2_dir = str(run_dir / STAGE2_NAMES[0])
    stage2_init = str(target.parent)
elif target.parent.name.startswith("checkpoint-") and target.parent.parent.name in STAGE2_NAMES:
    run_dir = target.parent.parent.parent
    stage1_candidate = run_dir / STAGE1_NAME
    stage1_dir = str(stage1_candidate) if stage1_candidate.exists() else ""
    stage2_dir = str(target.parent.parent)
    if target.parent.parent.name in ("stage2_qwen_box_closed_loop", "stage3_qwen_box_closed_loop"):
        stage2_resume = str(target)
    else:
        stage2_init = str(target.parent.parent)
elif target.parent.name.startswith("checkpoint-") and target.parent.parent.name == STAGE1_NAME:
    run_dir = target.parent.parent.parent
    stage1_dir = str(target.parent.parent)
    stage2_dir = str(run_dir / STAGE2_NAMES[0])
    stage2_init = str(target.parent.parent)
elif target.is_dir() and has_direct_checkpoint_children(target):
    run_dir = target
    stage2_dir = str(target)
    stage2_resume = str(target)
elif target.is_dir() and has_checkpoint_payload(target):
    run_dir = target.parent
    stage2_dir = str(target)
    stage2_resume = str(target)
elif target.is_file():
    run_dir = target.parent
    stage2_dir = str(run_dir)
    stage2_resume = str(target)
else:
    raise ValueError("Unsupported resume path layout. Expected a run dir, stage dir, checkpoint dir, or qwenpose_checkpoint.pt file.")

print(shell_assign("RESUME_RESOLVED_RUN_DIR", str(run_dir)))
print(shell_assign("RESUME_RESOLVED_STAGE1_OUTPUT_DIR", stage1_dir))
print(shell_assign("RESUME_RESOLVED_STAGE2_OUTPUT_DIR", stage2_dir))
print(shell_assign("RESUME_RESOLVED_STAGE2_RESUME", stage2_resume))
print(shell_assign("RESUME_RESOLVED_STAGE2_INIT_CHECKPOINT", stage2_init))
print(shell_assign("RESUME_RESOLVED_RUN_STAGE1", run_stage1))
print(shell_assign("RESUME_RESOLVED_RUN_STAGE2", run_stage2))
print(shell_assign("RESUME_RESOLVED_APPEND_LOG_FILE", latest_log_file(run_dir)))
PY
}

if [[ -n "${CLI_RESUME_PATH}" ]]; then
  eval "$(resolve_cli_resume_context "${CLI_RESUME_PATH}")"
  if [[ -n "${RESUME_RESOLVED_RUN_DIR:-}" ]]; then
    if [[ ! -v OUTPUT_DIR ]]; then OUTPUT_DIR="${RESUME_RESOLVED_RUN_DIR}"; fi
    if [[ ! -v OUTPUT_ROOT ]]; then OUTPUT_ROOT="$(dirname "${RESUME_RESOLVED_RUN_DIR}")"; fi
    if [[ ! -v RUN_NAME ]]; then RUN_NAME="$(basename "${RESUME_RESOLVED_RUN_DIR}")"; fi
    if [[ ! -v LOG_DIR ]]; then LOG_DIR="${RESUME_RESOLVED_RUN_DIR}/logs"; fi
    if [[ ! -v TRAIN_LOG_FILE && -n "${RESUME_RESOLVED_APPEND_LOG_FILE:-}" ]]; then TRAIN_LOG_FILE="${RESUME_RESOLVED_APPEND_LOG_FILE}"; fi
    if [[ ! -v RUN_STAGE1 && -n "${RESUME_RESOLVED_RUN_STAGE1:-}" ]]; then RUN_STAGE1="${RESUME_RESOLVED_RUN_STAGE1}"; fi
    if [[ ! -v RUN_STAGE2 && -n "${RESUME_RESOLVED_RUN_STAGE2:-}" ]]; then RUN_STAGE2="${RESUME_RESOLVED_RUN_STAGE2}"; fi
    if [[ ! -v STAGE1_OUTPUT_DIR && -n "${RESUME_RESOLVED_STAGE1_OUTPUT_DIR:-}" ]]; then STAGE1_OUTPUT_DIR="${RESUME_RESOLVED_STAGE1_OUTPUT_DIR}"; fi
    if [[ ! -v STAGE2_OUTPUT_DIR && -n "${RESUME_RESOLVED_STAGE2_OUTPUT_DIR:-}" ]]; then STAGE2_OUTPUT_DIR="${RESUME_RESOLVED_STAGE2_OUTPUT_DIR}"; fi
    if [[ ! -v STAGE2_INIT_CHECKPOINT && -n "${RESUME_RESOLVED_STAGE2_INIT_CHECKPOINT:-}" ]]; then STAGE2_INIT_CHECKPOINT="${RESUME_RESOLVED_STAGE2_INIT_CHECKPOINT}"; fi
    if [[ ! -v STAGE2_RESUME_FROM_CHECKPOINT && -n "${RESUME_RESOLVED_STAGE2_RESUME:-}" ]]; then STAGE2_RESUME_FROM_CHECKPOINT="${RESUME_RESOLVED_STAGE2_RESUME}"; fi
  fi
fi

###############################################################################
# Defaults
###############################################################################

RUN_TS="${RUN_TS:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/qwenpose_two_stage_qwen}"
RUN_NAME_BASE="${RUN_NAME:-qwenpose-two-stage-qwen3vl-lora}"
if [[ "${RUN_NAME_BASE}" =~ [0-9]{8}-[0-9]{6}$ ]]; then
  RUN_NAME="${RUN_NAME_BASE}"
else
  RUN_NAME="${RUN_NAME_BASE}-${RUN_TS}"
fi
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"
LOG_TS="${LOG_TS:-${RUN_TS}}"
TRAIN_LOG_FILE="${TRAIN_LOG_FILE:-${LOG_DIR}/train_${LOG_TS}.log}"

mkdir -p "${LOG_DIR}" "$(dirname "${TRAIN_LOG_FILE}")"
touch "${TRAIN_LOG_FILE}"
exec > >(tee -a "${TRAIN_LOG_FILE}") 2>&1
echo "Logging all stdout/stderr to ${TRAIN_LOG_FILE}"
trap 'status=$?; echo "[ERROR] ${BASH_SOURCE[0]}:${LINENO}: ${BASH_COMMAND} exited with status ${status}" >&2' ERR
trap 'status=$?; echo "========== qwenpose two-stage train exit status ${status} at $(date -Is) =========="; exit ${status}' EXIT

ZERO_STAGE="${ZERO_STAGE:-zero2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}"
export TORCH_FR_BUFFER_SIZE="${TORCH_FR_BUFFER_SIZE:-200000}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
export TORCH_NCCL_DESYNC_DEBUG="${TORCH_NCCL_DESYNC_DEBUG:-1}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"

case "${ZERO_STAGE}" in
  zero2) DEFAULT_DEEPSPEED_CONFIG="${PROJECT_ROOT}/scripts/zero2.json" ;;
  zero3) DEFAULT_DEEPSPEED_CONFIG="${PROJECT_ROOT}/scripts/zero3.json" ;;
  zero3_offload) DEFAULT_DEEPSPEED_CONFIG="${PROJECT_ROOT}/scripts/zero3_offload.json" ;;
  none) DEFAULT_DEEPSPEED_CONFIG="" ;;
  *) echo "Unsupported ZERO_STAGE=${ZERO_STAGE}. Use zero2, zero3, zero3_offload, or none." >&2; exit 1 ;;
esac
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${DEFAULT_DEEPSPEED_CONFIG}}"

DATASET_ROOT="${DATASET_ROOT:-datasets}"
SPLIT="${SPLIT:-train}"
MIXING_STRATEGY="${MIXING_STRATEGY:-interleave}"
DATASET_MIX_WEIGHTS="${DATASET_MIX_WEIGHTS:-auto}"
MAX_INSTANCES="${MAX_INSTANCES:-80}"
MAX_SAMPLES_PER_DATASET="${MAX_SAMPLES_PER_DATASET:-}"
RECORD_CACHE_DIR="${RECORD_CACHE_DIR:-.cache/qwenpose_records}"
DISABLE_RECORD_CACHE="${DISABLE_RECORD_CACHE:-0}"

QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-weights/Qwen3-VL-4B-Instruct}"
QWEN_DTYPE="${QWEN_DTYPE:-bfloat16}"
QWEN_ATTN_IMPLEMENTATION="${QWEN_ATTN_IMPLEMENTATION:-flash_attention_2}"
QWEN_GRADIENT_CHECKPOINTING="${QWEN_GRADIENT_CHECKPOINTING:-1}"
QWEN_MIN_PIXELS="${QWEN_MIN_PIXELS:-}"
QWEN_MAX_PIXELS="${QWEN_MAX_PIXELS:-}"
QWEN_FEATURE_SIZE="${QWEN_FEATURE_SIZE:-64}"
QWEN_FEATURE_REFINER_LAYERS="${QWEN_FEATURE_REFINER_LAYERS:-1}"
QWEN_FEATURE_REFINER_BOTTLENECK_DIM="${QWEN_FEATURE_REFINER_BOTTLENECK_DIM:-256}"
QWEN_FEATURE_REFINER_INIT_SCALE="${QWEN_FEATURE_REFINER_INIT_SCALE:-0.1}"
HIDDEN_DIM="${HIDDEN_DIM:-448}"
POSE_DECODER_LAYERS="${POSE_DECODER_LAYERS:-3}"
REFINEMENT_STEPS="${REFINEMENT_STEPS:-3}"
BOX_CONDITION_SCALE="${BOX_CONDITION_SCALE:-1.2}"
POSE_ROI_SIZE="${POSE_ROI_SIZE:-16}"
DECODER_HEADS="${DECODER_HEADS:-8}"
QWEN_LORA_R="${QWEN_LORA_R:-32}"
QWEN_LORA_ALPHA="${QWEN_LORA_ALPHA:-64}"
QWEN_LORA_DROPOUT="${QWEN_LORA_DROPOUT:-0.05}"
QWEN_VISION_LORA_R="${QWEN_VISION_LORA_R:-16}"
QWEN_VISION_LORA_ALPHA="${QWEN_VISION_LORA_ALPHA:-32}"
QWEN_VISION_LORA_DROPOUT="${QWEN_VISION_LORA_DROPOUT:-0.05}"

BATCH_SIZE="${BATCH_SIZE:-}"
LR="${LR:-2e-4}"
QWEN_LORA_LR_SCALE="${QWEN_LORA_LR_SCALE:-0.05}"
QWEN_VISION_LR_SCALE="${QWEN_VISION_LR_SCALE:-0.02}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
DEVICE="${DEVICE:-cuda}"
AMP="${AMP:-0}"
LOG_EVERY="${LOG_EVERY:-10}"
VISUALIZE_EVERY="${VISUALIZE_EVERY:-10}"
VISUALIZE_MAX_INSTANCES="${VISUALIZE_MAX_INSTANCES:-8}"
SYNC_TIMING="${SYNC_TIMING:-0}"
SAVE_EVERY="${SAVE_EVERY:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
SEED="${SEED:-42}"

W_OKS="${W_OKS:-0.2}"
W_COORD="${W_COORD:-5.0}"
W_VIS="${W_VIS:-0.05}"
W_HARD_JOINT="${W_HARD_JOINT:-0}"
HARD_JOINT_FRACTION="${HARD_JOINT_FRACTION:-0.2}"
W_LM="${W_LM:-0.05}"
LM_LOSS_EVERY="${LM_LOSS_EVERY:-2}"
LM_MAX_ANSWER_INSTANCES="${LM_MAX_ANSWER_INSTANCES:-${MAX_INSTANCES}}"
QWEN_BOX_MAX_NEW_TOKENS="${QWEN_BOX_MAX_NEW_TOKENS:-4096}"
BOX_MATCH_IOU_THRESH="${BOX_MATCH_IOU_THRESH:-0.10}"
BOX_NMS_IOU_THRESH="${BOX_NMS_IOU_THRESH:-0.70}"

DRY_RUN_DATA="${DRY_RUN_DATA:-0}"
PROGRESS_BAR="${PROGRESS_BAR:-1}"
DISABLE_REFINEMENT="${DISABLE_REFINEMENT:-0}"
DISABLE_HOMOGENEOUS_BATCHES="${DISABLE_HOMOGENEOUS_BATCHES:-0}"
DISABLE_BATCH_TRACE="${DISABLE_BATCH_TRACE:-0}"

RUN_STAGE1="${RUN_STAGE1:-1}"
RUN_STAGE2="${RUN_STAGE2:-1}"
STAGE1_OUTPUT_DIR="${STAGE1_OUTPUT_DIR:-${OUTPUT_DIR}/stage1_freeze_qwen}"
STAGE2_OUTPUT_DIR="${STAGE2_OUTPUT_DIR:-${OUTPUT_DIR}/stage2_qwen_box_closed_loop}"
STAGE2_INIT_WEIGHTS_DIR="${STAGE2_INIT_WEIGHTS_DIR:-${OUTPUT_DIR}/stage2_init_weights}"

STAGE1_TRAIN_DATASETS="${STAGE1_TRAIN_DATASETS:-coco}"
STAGE2_TRAIN_DATASETS="${STAGE2_TRAIN_DATASETS:-${STAGE1_TRAIN_DATASETS}}"
STAGE1_MIXING_STRATEGY="${STAGE1_MIXING_STRATEGY:-${MIXING_STRATEGY}}"
STAGE2_MIXING_STRATEGY="${STAGE2_MIXING_STRATEGY:-${MIXING_STRATEGY}}"
STAGE1_DATASET_MIX_WEIGHTS="${STAGE1_DATASET_MIX_WEIGHTS:-${DATASET_MIX_WEIGHTS}}"
STAGE2_DATASET_MIX_WEIGHTS="${STAGE2_DATASET_MIX_WEIGHTS:-${DATASET_MIX_WEIGHTS}}"
STAGE1_MAX_SAMPLES_PER_DATASET="${STAGE1_MAX_SAMPLES_PER_DATASET:-${MAX_SAMPLES_PER_DATASET}}"
STAGE2_MAX_SAMPLES_PER_DATASET="${STAGE2_MAX_SAMPLES_PER_DATASET:-${MAX_SAMPLES_PER_DATASET}}"
STAGE1_REFHUMAN_MAX_CAPTIONS_PER_INSTANCE="${STAGE1_REFHUMAN_MAX_CAPTIONS_PER_INSTANCE:-${REFHUMAN_MAX_CAPTIONS_PER_INSTANCE:-1}}"
STAGE2_REFHUMAN_MAX_CAPTIONS_PER_INSTANCE="${STAGE2_REFHUMAN_MAX_CAPTIONS_PER_INSTANCE:-${REFHUMAN_MAX_CAPTIONS_PER_INSTANCE:-1}}"

STAGE1_EPOCHS="${STAGE1_EPOCHS:-30}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-12}"
STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-${BATCH_SIZE:-4}}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-${BATCH_SIZE:-1}}"
STAGE1_GRAD_ACCUM_STEPS="${STAGE1_GRAD_ACCUM_STEPS:-2}"
STAGE2_GRAD_ACCUM_STEPS="${STAGE2_GRAD_ACCUM_STEPS:-8}"
STAGE1_MAX_STEPS="${STAGE1_MAX_STEPS:-0}"
STAGE2_MAX_STEPS="${STAGE2_MAX_STEPS:-0}"
STAGE1_FREEZE_QWEN="${STAGE1_FREEZE_QWEN:-1}"
STAGE2_FREEZE_QWEN="${STAGE2_FREEZE_QWEN:-0}"
STAGE1_W_LM="${STAGE1_W_LM:-0}"
STAGE2_W_LM="${STAGE2_W_LM:-0.2}"
STAGE1_LM_LOSS_EVERY="${STAGE1_LM_LOSS_EVERY:-0}"
STAGE2_LM_LOSS_EVERY="${STAGE2_LM_LOSS_EVERY:-1}"
STAGE1_BOX_SOURCE="${STAGE1_BOX_SOURCE:-gt}"
STAGE2_BOX_SOURCE="${STAGE2_BOX_SOURCE:-qwen_generate}"
STAGE1_BOX_JITTER_SCALE="${STAGE1_BOX_JITTER_SCALE:-0.0}"
STAGE1_BOX_JITTER_SHIFT="${STAGE1_BOX_JITTER_SHIFT:-0.0}"
STAGE2_BOX_JITTER_SCALE="${STAGE2_BOX_JITTER_SCALE:-0.0}"
STAGE2_BOX_JITTER_SHIFT="${STAGE2_BOX_JITTER_SHIFT:-0.0}"
STAGE1_QWEN_BOX_MAX_NEW_TOKENS="${STAGE1_QWEN_BOX_MAX_NEW_TOKENS:-${QWEN_BOX_MAX_NEW_TOKENS}}"
STAGE2_QWEN_BOX_MAX_NEW_TOKENS="${STAGE2_QWEN_BOX_MAX_NEW_TOKENS:-${QWEN_BOX_MAX_NEW_TOKENS}}"
STAGE1_BOX_MATCH_IOU_THRESH="${STAGE1_BOX_MATCH_IOU_THRESH:-${BOX_MATCH_IOU_THRESH}}"
STAGE2_BOX_MATCH_IOU_THRESH="${STAGE2_BOX_MATCH_IOU_THRESH:-${BOX_MATCH_IOU_THRESH}}"
STAGE1_BOX_NMS_IOU_THRESH="${STAGE1_BOX_NMS_IOU_THRESH:-${BOX_NMS_IOU_THRESH}}"
STAGE2_BOX_NMS_IOU_THRESH="${STAGE2_BOX_NMS_IOU_THRESH:-${BOX_NMS_IOU_THRESH}}"
STAGE1_LR="${STAGE1_LR:-${LR}}"
STAGE2_LR="${STAGE2_LR:-5e-5}"
STAGE1_QWEN_LORA_LR_SCALE="${STAGE1_QWEN_LORA_LR_SCALE:-${QWEN_LORA_LR_SCALE}}"
STAGE2_QWEN_LORA_LR_SCALE="${STAGE2_QWEN_LORA_LR_SCALE:-${QWEN_LORA_LR_SCALE}}"
STAGE1_QWEN_VISION_LR_SCALE="${STAGE1_QWEN_VISION_LR_SCALE:-${QWEN_VISION_LR_SCALE}}"
STAGE2_QWEN_VISION_LR_SCALE="${STAGE2_QWEN_VISION_LR_SCALE:-${QWEN_VISION_LR_SCALE}}"
STAGE1_WARMUP_STEPS="${STAGE1_WARMUP_STEPS:-${WARMUP_STEPS}}"
STAGE2_WARMUP_STEPS="${STAGE2_WARMUP_STEPS:-${WARMUP_STEPS}}"
STAGE1_MIN_LR_RATIO="${STAGE1_MIN_LR_RATIO:-${MIN_LR_RATIO}}"
STAGE2_MIN_LR_RATIO="${STAGE2_MIN_LR_RATIO:-${MIN_LR_RATIO}}"
STAGE1_NUM_WORKERS="${STAGE1_NUM_WORKERS:-${NUM_WORKERS}}"
STAGE2_NUM_WORKERS="${STAGE2_NUM_WORKERS:-${NUM_WORKERS}}"
STAGE1_PREFETCH_FACTOR="${STAGE1_PREFETCH_FACTOR:-${PREFETCH_FACTOR}}"
STAGE2_PREFETCH_FACTOR="${STAGE2_PREFETCH_FACTOR:-${PREFETCH_FACTOR}}"
STAGE1_SAVE_EVERY="${STAGE1_SAVE_EVERY:-1000}"
STAGE2_SAVE_EVERY="${STAGE2_SAVE_EVERY:-${SAVE_EVERY}}"
STAGE1_SAVE_TOTAL_LIMIT="${STAGE1_SAVE_TOTAL_LIMIT:-${SAVE_TOTAL_LIMIT}}"
STAGE2_SAVE_TOTAL_LIMIT="${STAGE2_SAVE_TOTAL_LIMIT:-${SAVE_TOTAL_LIMIT}}"
STAGE1_VISUALIZE_EVERY="${STAGE1_VISUALIZE_EVERY:-10}"
STAGE2_VISUALIZE_EVERY="${STAGE2_VISUALIZE_EVERY:-${VISUALIZE_EVERY}}"
STAGE1_DISABLE_BATCH_TRACE="${STAGE1_DISABLE_BATCH_TRACE:-1}"
STAGE2_DISABLE_BATCH_TRACE="${STAGE2_DISABLE_BATCH_TRACE:-${DISABLE_BATCH_TRACE}}"
STAGE1_SEED="${STAGE1_SEED:-${SEED}}"
STAGE2_SEED="${STAGE2_SEED:-${SEED}}"
STAGE1_RESUME_FROM_CHECKPOINT="${STAGE1_RESUME_FROM_CHECKPOINT:-none}"
STAGE2_RESUME_FROM_CHECKPOINT="${STAGE2_RESUME_FROM_CHECKPOINT:-none}"
STAGE2_INIT_CHECKPOINT="${STAGE2_INIT_CHECKPOINT:-}"
STAGE2_INIT_FROM_STAGE1="${STAGE2_INIT_FROM_STAGE1:-1}"
MERGE_FINAL_WEIGHTS="${MERGE_FINAL_WEIGHTS:-0}"
MERGED_WEIGHTS_ROOT="${MERGED_WEIGHTS_ROOT:-weights}"
MERGED_WEIGHTS_DIR="${MERGED_WEIGHTS_DIR:-${MERGED_WEIGHTS_ROOT}/${RUN_NAME}-merged-${RUN_TS}}"

if [[ -n "${CLI_RESUME_PATH}" ]]; then
  if [[ "${STAGE2_RESUME_FROM_CHECKPOINT}" == "none" && -n "${RESUME_RESOLVED_STAGE2_RESUME:-}" ]]; then
    STAGE2_RESUME_FROM_CHECKPOINT="${RESUME_RESOLVED_STAGE2_RESUME}"
  fi
  if [[ -z "${STAGE2_INIT_CHECKPOINT}" && -n "${RESUME_RESOLVED_STAGE2_INIT_CHECKPOINT:-}" ]]; then
    STAGE2_INIT_CHECKPOINT="${RESUME_RESOLVED_STAGE2_INIT_CHECKPOINT}"
  fi
fi

###############################################################################
# Validation helpers
###############################################################################

require_positive_int() { local name="$1" value="$2"; if ! [[ "${value}" =~ ^[0-9]+$ ]] || (( value <= 0 )); then echo "${name} must be a positive integer, got: ${value}" >&2; exit 1; fi; }
require_nonnegative_int() { local name="$1" value="$2"; if ! [[ "${value}" =~ ^[0-9]+$ ]]; then echo "${name} must be a non-negative integer, got: ${value}" >&2; exit 1; fi; }
require_bool() { local name="$1" value="$2"; if [[ "${value}" != "0" && "${value}" != "1" ]]; then echo "${name} must be 0 or 1, got: ${value}" >&2; exit 1; fi; }

resume_target_has_checkpoint() {
  local resume_path="$1"
  if [[ -f "${resume_path}" ]]; then return 0; fi
  if [[ ! -d "${resume_path}" ]]; then return 1; fi
  if [[ -f "${resume_path}/qwenpose_checkpoint.pt" || -e "${resume_path}/deepspeed" ]]; then return 0; fi
  find "${resume_path}" -maxdepth 1 \( -name 'checkpoint-*' -o -name 'checkpoint_step_*.pt' \) -print -quit | grep -q .
}

check_optional_checkpoint() {
  local name="$1" path="$2"
  if [[ "${path}" == "none" || -z "${path}" ]]; then return 0; fi
  if [[ ! -e "${path}" ]]; then echo "${name} path does not exist: ${path}" >&2; exit 1; fi
  if ! resume_target_has_checkpoint "${path}"; then echo "${name} has no checkpoint payload: ${path}" >&2; exit 1; fi
}

VISIBLE_GPU_COUNT="$("${PYTHON}" - <<'PY'
import os
visible = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
print(len(visible) or 1)
PY
)"

for spec in \
  "NPROC_PER_NODE:${NPROC_PER_NODE}" "VISIBLE_GPU_COUNT:${VISIBLE_GPU_COUNT}" \
  "STAGE1_BATCH_SIZE:${STAGE1_BATCH_SIZE}" "STAGE2_BATCH_SIZE:${STAGE2_BATCH_SIZE}" \
  "STAGE1_GRAD_ACCUM_STEPS:${STAGE1_GRAD_ACCUM_STEPS}" "STAGE2_GRAD_ACCUM_STEPS:${STAGE2_GRAD_ACCUM_STEPS}" \
  "STAGE1_EPOCHS:${STAGE1_EPOCHS}" "STAGE2_EPOCHS:${STAGE2_EPOCHS}" \
  "REFINEMENT_STEPS:${REFINEMENT_STEPS}" "POSE_ROI_SIZE:${POSE_ROI_SIZE}" \
  "STAGE1_SAVE_EVERY:${STAGE1_SAVE_EVERY}" "STAGE2_SAVE_EVERY:${STAGE2_SAVE_EVERY}" \
  "STAGE1_SAVE_TOTAL_LIMIT:${STAGE1_SAVE_TOTAL_LIMIT}" "STAGE2_SAVE_TOTAL_LIMIT:${STAGE2_SAVE_TOTAL_LIMIT}" \
  "VISUALIZE_MAX_INSTANCES:${VISUALIZE_MAX_INSTANCES}" "LM_MAX_ANSWER_INSTANCES:${LM_MAX_ANSWER_INSTANCES}" \
  "QWEN_BOX_MAX_NEW_TOKENS:${QWEN_BOX_MAX_NEW_TOKENS}" \
  "STAGE1_QWEN_BOX_MAX_NEW_TOKENS:${STAGE1_QWEN_BOX_MAX_NEW_TOKENS}" \
  "STAGE2_QWEN_BOX_MAX_NEW_TOKENS:${STAGE2_QWEN_BOX_MAX_NEW_TOKENS}"; do
  require_positive_int "${spec%%:*}" "${spec#*:}"
done

for spec in \
  "STAGE1_MAX_STEPS:${STAGE1_MAX_STEPS}" "STAGE2_MAX_STEPS:${STAGE2_MAX_STEPS}" \
  "STAGE1_NUM_WORKERS:${STAGE1_NUM_WORKERS}" "STAGE2_NUM_WORKERS:${STAGE2_NUM_WORKERS}" \
  "STAGE1_VISUALIZE_EVERY:${STAGE1_VISUALIZE_EVERY}" "STAGE2_VISUALIZE_EVERY:${STAGE2_VISUALIZE_EVERY}" \
  "STAGE1_LM_LOSS_EVERY:${STAGE1_LM_LOSS_EVERY}" "STAGE2_LM_LOSS_EVERY:${STAGE2_LM_LOSS_EVERY}" \
  "STAGE1_REFHUMAN_MAX_CAPTIONS_PER_INSTANCE:${STAGE1_REFHUMAN_MAX_CAPTIONS_PER_INSTANCE}" \
  "STAGE2_REFHUMAN_MAX_CAPTIONS_PER_INSTANCE:${STAGE2_REFHUMAN_MAX_CAPTIONS_PER_INSTANCE}"; do
  require_nonnegative_int "${spec%%:*}" "${spec#*:}"
done

for spec in RUN_STAGE1 RUN_STAGE2 STAGE1_FREEZE_QWEN STAGE2_FREEZE_QWEN STAGE2_INIT_FROM_STAGE1 MERGE_FINAL_WEIGHTS QWEN_GRADIENT_CHECKPOINTING AMP DRY_RUN_DATA PROGRESS_BAR SYNC_TIMING DISABLE_RECORD_CACHE DISABLE_REFINEMENT DISABLE_HOMOGENEOUS_BATCHES DISABLE_BATCH_TRACE STAGE1_DISABLE_BATCH_TRACE STAGE2_DISABLE_BATCH_TRACE; do
  require_bool "${spec}" "${!spec}"
done

if (( NPROC_PER_NODE > VISIBLE_GPU_COUNT )); then echo "NPROC_PER_NODE=${NPROC_PER_NODE} exceeds visible GPUs (${CUDA_VISIBLE_DEVICES}; count=${VISIBLE_GPU_COUNT})." >&2; exit 1; fi
if [[ -n "${QWEN_MIN_PIXELS}" ]]; then require_positive_int QWEN_MIN_PIXELS "${QWEN_MIN_PIXELS}"; fi
if [[ -n "${QWEN_MAX_PIXELS}" ]]; then require_positive_int QWEN_MAX_PIXELS "${QWEN_MAX_PIXELS}"; fi
if [[ -n "${QWEN_MIN_PIXELS}" && -n "${QWEN_MAX_PIXELS}" ]] && (( QWEN_MAX_PIXELS < QWEN_MIN_PIXELS )); then echo "QWEN_MAX_PIXELS=${QWEN_MAX_PIXELS} must be >= QWEN_MIN_PIXELS=${QWEN_MIN_PIXELS}." >&2; exit 1; fi
if [[ "${DEVICE}" != "cuda" && "${ZERO_STAGE}" != "none" ]]; then echo "DEVICE=${DEVICE} cannot use DeepSpeed ${ZERO_STAGE}. Use ZERO_STAGE=none for CPU debugging." >&2; exit 1; fi
if [[ "${RUN_STAGE2}" == "1" && "${STAGE2_BOX_SOURCE}" == "qwen_generate" && "${ZERO_STAGE}" != "zero2" && "${ZERO_STAGE}" != "none" ]]; then echo "Stage 2 qwen_generate calls model.generate during training and currently supports ZERO_STAGE=zero2 or none. Got ZERO_STAGE=${ZERO_STAGE}." >&2; exit 1; fi
if [[ ! -e "${QWEN_MODEL_PATH}" ]]; then echo "QWEN_MODEL_PATH not found: ${QWEN_MODEL_PATH}" >&2; exit 1; fi
if [[ -n "${DEEPSPEED_CONFIG}" && ! -f "${DEEPSPEED_CONFIG}" ]]; then echo "DEEPSPEED_CONFIG not found: ${DEEPSPEED_CONFIG}" >&2; exit 1; fi

check_optional_checkpoint STAGE1_RESUME_FROM_CHECKPOINT "${STAGE1_RESUME_FROM_CHECKPOINT}"
check_optional_checkpoint STAGE2_RESUME_FROM_CHECKPOINT "${STAGE2_RESUME_FROM_CHECKPOINT}"
check_optional_checkpoint STAGE2_INIT_CHECKPOINT "${STAGE2_INIT_CHECKPOINT}"

prepare_weights_only_checkpoint() {
  local source_path="$1" dest_dir="$2"
  if [[ -z "${source_path}" || -z "${dest_dir}" || "${dest_dir}" == "/" ]]; then echo "Invalid weight-only checkpoint arguments: source=${source_path}, dest=${dest_dir}" >&2; exit 1; fi
  if ! resume_target_has_checkpoint "${source_path}"; then echo "Cannot initialize stage 2; no checkpoint found in ${source_path}" >&2; exit 1; fi
  rm -rf "${dest_dir}"
  mkdir -p "${dest_dir}"
  "${PYTHON}" - "${source_path}" "${dest_dir}" <<'PY'
import json
import re
import shutil
import sys
from pathlib import Path
import torch
CHECKPOINT_PAYLOAD_NAME = "qwenpose_checkpoint.pt"
source = Path(sys.argv[1])
dest = Path(sys.argv[2])
def checkpoint_step(path: Path) -> int | None:
    match = re.search(r"checkpoint-(\d+)$", path.name) if path.is_dir() else re.search(r"checkpoint_step_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else None
def resolve(path: Path) -> Path:
    if path.is_file():
        return path
    if (path / CHECKPOINT_PAYLOAD_NAME).is_file():
        return path
    candidates = []
    for candidate in list(path.glob("checkpoint-*")) + list(path.glob("checkpoint_step_*.pt")):
        step = checkpoint_step(candidate)
        if step is not None:
            candidates.append((step, candidate))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-* or checkpoint_step_*.pt found in {path}")
    return sorted(candidates)[-1][1]
resolved = resolve(source)
payload_path = resolved / CHECKPOINT_PAYLOAD_NAME if resolved.is_dir() else resolved
try:
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
except TypeError:
    payload = torch.load(payload_path, map_location="cpu")
for key in ("optimizer", "scaler", "training_state", "rng_state"):
    payload.pop(key, None)
payload["step"] = 0
payload["deepspeed_managed"] = False
payload["stage2_weight_only_init_from"] = str(resolved)
out = dest / "checkpoint-0"
out.mkdir(parents=True, exist_ok=True)
torch.save(payload, out / CHECKPOINT_PAYLOAD_NAME)
with (out / "qwenpose_state.json").open("w", encoding="utf-8") as f:
    json.dump({"step": 0, "checkpoint": str(out), "payload": CHECKPOINT_PAYLOAD_NAME, "deepspeed_tag": None, "training_state": None, "stage2_weight_only_init_from": str(resolved)}, f, indent=2, ensure_ascii=False)
    f.write("\n")
adapter_src = resolved / "qwen_lora_adapter" if resolved.is_dir() else None
if adapter_src is not None and adapter_src.is_dir():
    shutil.copytree(adapter_src, out / "qwen_lora_adapter", dirs_exist_ok=True)
print(out)
PY
}

merge_full_weights() {
  local checkpoint_source="$1" merged_dir="$2"
  if [[ "${MERGE_FINAL_WEIGHTS}" != "1" ]]; then return 0; fi
  mkdir -p "$(dirname "${merged_dir}")"
  echo "Merging final checkpoint from ${checkpoint_source} into full weights: ${merged_dir}"
  "${PYTHON}" -m qwenpose.merge_full_weights --checkpoint "${checkpoint_source}" --base_model_path "${QWEN_MODEL_PATH}" --output_dir "${merged_dir}" --qwen_dtype "${QWEN_DTYPE}" --qwen_attn_implementation "${QWEN_ATTN_IMPLEMENTATION}" --overwrite
}

run_train_pose() {
  if [[ "${ZERO_STAGE}" == "none" ]]; then
    "${PYTHON}" -m qwenpose.train_pose "$@"
  elif [[ -n "${TORCHRUN}" ]]; then
    "${TORCHRUN}" --nproc_per_node "${NPROC_PER_NODE}" --master_addr "${MASTER_ADDR}" --master_port "${MASTER_PORT}" "${PROJECT_ROOT}/src/qwenpose/train_pose.py" "$@"
  else
    "${PYTHON}" -m torch.distributed.run --nproc_per_node "${NPROC_PER_NODE}" --master_addr "${MASTER_ADDR}" --master_port "${MASTER_PORT}" "${PROJECT_ROOT}/src/qwenpose/train_pose.py" "$@"
  fi
}

run_stage() {
  local stage_label="$1" output_dir="$2" datasets="$3" batch_size="$4" grad_accum_steps="$5" epochs="$6" max_steps="$7" freeze_qwen="$8" w_lm="$9"
  local lm_loss_every="${10}" refhuman_max_captions="${11}" mixing_strategy="${12}" dataset_mix_weights="${13}" max_samples_per_dataset="${14}"
  local lr="${15}" qwen_lora_lr_scale="${16}" qwen_vision_lr_scale="${17}" warmup_steps="${18}" min_lr_ratio="${19}" num_workers="${20}" prefetch_factor="${21}"
  local save_every="${22}" save_total_limit="${23}" visualize_every="${24}" seed="${25}" resume_arg="${26}" disable_batch_trace="${27}"
  local box_source="${28}" box_jitter_scale="${29}" box_jitter_shift="${30}" qwen_box_max_new_tokens="${31}" box_match_iou_thresh="${32}" box_nms_iou_thresh="${33}"
  local effective_batch=$((NPROC_PER_NODE * batch_size * grad_accum_steps))
  local args=(
    --dataset_root "${DATASET_ROOT}" --datasets "${datasets}" --split "${SPLIT}" --mixing_strategy "${mixing_strategy}" --dataset_mix_weights "${dataset_mix_weights}" --max_instances "${MAX_INSTANCES}" --refhuman_max_captions_per_instance "${refhuman_max_captions}" --record_cache_dir "${RECORD_CACHE_DIR}"
    --hidden_dim "${HIDDEN_DIM}" --backbone "qwen3vl" --qwen_model_path "${QWEN_MODEL_PATH}" --qwen_dtype "${QWEN_DTYPE}" --qwen_attn_implementation "${QWEN_ATTN_IMPLEMENTATION}" --qwen_feature_size "${QWEN_FEATURE_SIZE}" --qwen_feature_refiner_layers "${QWEN_FEATURE_REFINER_LAYERS}" --qwen_feature_refiner_bottleneck_dim "${QWEN_FEATURE_REFINER_BOTTLENECK_DIM}" --qwen_feature_refiner_init_scale "${QWEN_FEATURE_REFINER_INIT_SCALE}" --qwen_lora_r "${QWEN_LORA_R}" --qwen_lora_alpha "${QWEN_LORA_ALPHA}" --qwen_lora_dropout "${QWEN_LORA_DROPOUT}" --qwen_vision_lora_r "${QWEN_VISION_LORA_R}" --qwen_vision_lora_alpha "${QWEN_VISION_LORA_ALPHA}" --qwen_vision_lora_dropout "${QWEN_VISION_LORA_DROPOUT}" --pose_decoder_layers "${POSE_DECODER_LAYERS}" --refinement_steps "${REFINEMENT_STEPS}" --box_condition_scale "${BOX_CONDITION_SCALE}" --pose_roi_size "${POSE_ROI_SIZE}" --decoder_heads "${DECODER_HEADS}"
    --output_dir "${output_dir}" --batch_size "${batch_size}" --grad_accum_steps "${grad_accum_steps}" --epochs "${epochs}" --max_steps "${max_steps}" --lr "${lr}" --qwen_lora_lr_scale "${qwen_lora_lr_scale}" --qwen_vision_lr_scale "${qwen_vision_lr_scale}" --weight_decay "${WEIGHT_DECAY}" --grad_clip "${GRAD_CLIP}" --warmup_steps "${warmup_steps}" --min_lr_ratio "${min_lr_ratio}" --num_workers "${num_workers}" --prefetch_factor "${prefetch_factor}" --device "${DEVICE}" --log_every "${LOG_EVERY}" --visualize_every "${visualize_every}" --visualize_max_instances "${VISUALIZE_MAX_INSTANCES}" --save_every "${save_every}" --save_total_limit "${save_total_limit}" --seed "${seed}"
    --w_oks "${W_OKS}" --w_coord "${W_COORD}" --w_vis "${W_VIS}" --w_hard_joint "${W_HARD_JOINT}" --hard_joint_fraction "${HARD_JOINT_FRACTION}" --w_lm "${w_lm}" --lm_loss_every "${lm_loss_every}" --lm_max_answer_instances "${LM_MAX_ANSWER_INSTANCES}" --box_source "${box_source}" --box_jitter_scale "${box_jitter_scale}" --box_jitter_shift "${box_jitter_shift}" --qwen_box_max_new_tokens "${qwen_box_max_new_tokens}" --box_match_iou_thresh "${box_match_iou_thresh}" --box_nms_iou_thresh "${box_nms_iou_thresh}"
  )
  if [[ -n "${max_samples_per_dataset}" ]]; then args+=(--max_samples_per_dataset "${max_samples_per_dataset}"); fi
  if [[ "${resume_arg}" != "none" && -n "${resume_arg}" ]]; then args+=(--resume_from_checkpoint "${resume_arg}"); fi
  if [[ -n "${QWEN_MIN_PIXELS}" ]]; then args+=(--qwen_min_pixels "${QWEN_MIN_PIXELS}"); fi
  if [[ -n "${QWEN_MAX_PIXELS}" ]]; then args+=(--qwen_max_pixels "${QWEN_MAX_PIXELS}"); fi
  if [[ "${AMP}" == "1" ]]; then args+=(--amp); fi
  if [[ "${QWEN_GRADIENT_CHECKPOINTING}" == "1" ]]; then args+=(--qwen_gradient_checkpointing); fi
  if [[ "${freeze_qwen}" == "1" ]]; then args+=(--freeze_qwen); fi
  if [[ "${DRY_RUN_DATA}" == "1" ]]; then args+=(--dry_run_data); fi
  if [[ "${PROGRESS_BAR}" == "0" ]]; then args+=(--disable_progress); fi
  if [[ "${SYNC_TIMING}" == "1" ]]; then args+=(--sync_timing); fi
  if [[ "${DISABLE_RECORD_CACHE}" == "1" ]]; then args+=(--disable_record_cache); fi
  if [[ "${DISABLE_REFINEMENT}" == "1" ]]; then args+=(--disable_refinement); fi
  if [[ "${DISABLE_HOMOGENEOUS_BATCHES}" == "1" ]]; then args+=(--disable_homogeneous_batches); fi
  if [[ "${disable_batch_trace}" == "1" ]]; then args+=(--disable_batch_trace); fi
  if [[ -n "${DEEPSPEED_CONFIG}" ]]; then args+=(--deepspeed_config "${DEEPSPEED_CONFIG}"); fi

  echo "================ QwenPose ${stage_label} 配置 ================"
  echo "OUTPUT_DIR=${output_dir}"
  echo "DATASETS=${datasets}"
  echo "ZERO_STAGE=${ZERO_STAGE}"
  echo "DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG}"
  echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "BATCH_SIZE=${batch_size}"
  echo "GRAD_ACCUM_STEPS=${grad_accum_steps}"
  echo "EFFECTIVE_BATCH=${effective_batch}"
  echo "EPOCHS=${epochs}"
  echo "MAX_STEPS=${max_steps}"
  echo "FREEZE_QWEN=${freeze_qwen}"
  echo "W_LM=${w_lm}"
  echo "LM_LOSS_EVERY=${lm_loss_every}"
  echo "BOX_SOURCE=${box_source}"
  echo "BOX_JITTER_SCALE=${box_jitter_scale}"
  echo "BOX_JITTER_SHIFT=${box_jitter_shift}"
  echo "QWEN_BOX_MAX_NEW_TOKENS=${qwen_box_max_new_tokens}"
  echo "BOX_MATCH_IOU_THRESH=${box_match_iou_thresh}"
  echo "BOX_NMS_IOU_THRESH=${box_nms_iou_thresh}"
  echo "LR=${lr}"
  echo "QWEN_LORA_LR_SCALE=${qwen_lora_lr_scale}"
  echo "QWEN_VISION_LR_SCALE=${qwen_vision_lr_scale}"
  echo "RESUME_ARG=${resume_arg}"
  echo "===================================================="
  run_train_pose "${args[@]}"
}

###############################################################################
# Launch two stages
###############################################################################

echo "================ QwenPose two-stage run ================"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "RUN_NAME=${RUN_NAME}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "STAGE1_OUTPUT_DIR=${STAGE1_OUTPUT_DIR}"
echo "STAGE2_OUTPUT_DIR=${STAGE2_OUTPUT_DIR}"
echo "RUN_STAGE1=${RUN_STAGE1}"
echo "RUN_STAGE2=${RUN_STAGE2}"
echo "STAGE2_INIT_FROM_STAGE1=${STAGE2_INIT_FROM_STAGE1}"
echo "STAGE2_INIT_CHECKPOINT=${STAGE2_INIT_CHECKPOINT}"
echo "STAGE2_RESUME_FROM_CHECKPOINT=${STAGE2_RESUME_FROM_CHECKPOINT}"
echo "MERGE_FINAL_WEIGHTS=${MERGE_FINAL_WEIGHTS}"
echo "MERGED_WEIGHTS_DIR=${MERGED_WEIGHTS_DIR}"
echo "TRAIN_LOG_FILE=${TRAIN_LOG_FILE}"
echo "========================================================="

if [[ "${RUN_STAGE1}" == "1" ]]; then
  run_stage \
    "Stage 1 / GT-box pose warmup" \
    "${STAGE1_OUTPUT_DIR}" "${STAGE1_TRAIN_DATASETS}" "${STAGE1_BATCH_SIZE}" "${STAGE1_GRAD_ACCUM_STEPS}" "${STAGE1_EPOCHS}" "${STAGE1_MAX_STEPS}" "${STAGE1_FREEZE_QWEN}" "${STAGE1_W_LM}" "${STAGE1_LM_LOSS_EVERY}" "${STAGE1_REFHUMAN_MAX_CAPTIONS_PER_INSTANCE}" "${STAGE1_MIXING_STRATEGY}" "${STAGE1_DATASET_MIX_WEIGHTS}" "${STAGE1_MAX_SAMPLES_PER_DATASET}" "${STAGE1_LR}" "${STAGE1_QWEN_LORA_LR_SCALE}" "${STAGE1_QWEN_VISION_LR_SCALE}" "${STAGE1_WARMUP_STEPS}" "${STAGE1_MIN_LR_RATIO}" "${STAGE1_NUM_WORKERS}" "${STAGE1_PREFETCH_FACTOR}" "${STAGE1_SAVE_EVERY}" "${STAGE1_SAVE_TOTAL_LIMIT}" "${STAGE1_VISUALIZE_EVERY}" "${STAGE1_SEED}" "${STAGE1_RESUME_FROM_CHECKPOINT}" "${STAGE1_DISABLE_BATCH_TRACE}" "${STAGE1_BOX_SOURCE}" "${STAGE1_BOX_JITTER_SCALE}" "${STAGE1_BOX_JITTER_SHIFT}" "${STAGE1_QWEN_BOX_MAX_NEW_TOKENS}" "${STAGE1_BOX_MATCH_IOU_THRESH}" "${STAGE1_BOX_NMS_IOU_THRESH}"
else
  echo "Skipping stage 1 because RUN_STAGE1=0"
fi

if [[ "${RUN_STAGE2}" == "1" ]]; then
  stage2_resume_arg="${STAGE2_RESUME_FROM_CHECKPOINT}"
  if [[ "${stage2_resume_arg}" == "none" || -z "${stage2_resume_arg}" ]]; then
    if [[ "${DRY_RUN_DATA}" == "1" ]]; then
      stage2_resume_arg="none"
    elif [[ -n "${STAGE2_INIT_CHECKPOINT}" ]]; then
      echo "Preparing stage 2 weight-only init from STAGE2_INIT_CHECKPOINT=${STAGE2_INIT_CHECKPOINT}"
      stage2_resume_arg="$(prepare_weights_only_checkpoint "${STAGE2_INIT_CHECKPOINT}" "${STAGE2_INIT_WEIGHTS_DIR}")"
    elif [[ "${STAGE2_INIT_FROM_STAGE1}" == "1" ]]; then
      echo "Preparing stage 2 weight-only init from stage 1 output: ${STAGE1_OUTPUT_DIR}"
      stage2_resume_arg="$(prepare_weights_only_checkpoint "${STAGE1_OUTPUT_DIR}" "${STAGE2_INIT_WEIGHTS_DIR}")"
    else
      stage2_resume_arg="none"
      echo "Stage 2 will start from base Qwen + newly initialized pose modules because STAGE2_INIT_FROM_STAGE1=0."
    fi
  else
    echo "Stage 2 will resume checkpoint state from ${stage2_resume_arg}"
  fi

  run_stage \
    "Stage 2 / Closed-loop Qwen-box training" \
    "${STAGE2_OUTPUT_DIR}" "${STAGE2_TRAIN_DATASETS}" "${STAGE2_BATCH_SIZE}" "${STAGE2_GRAD_ACCUM_STEPS}" "${STAGE2_EPOCHS}" "${STAGE2_MAX_STEPS}" "${STAGE2_FREEZE_QWEN}" "${STAGE2_W_LM}" "${STAGE2_LM_LOSS_EVERY}" "${STAGE2_REFHUMAN_MAX_CAPTIONS_PER_INSTANCE}" "${STAGE2_MIXING_STRATEGY}" "${STAGE2_DATASET_MIX_WEIGHTS}" "${STAGE2_MAX_SAMPLES_PER_DATASET}" "${STAGE2_LR}" "${STAGE2_QWEN_LORA_LR_SCALE}" "${STAGE2_QWEN_VISION_LR_SCALE}" "${STAGE2_WARMUP_STEPS}" "${STAGE2_MIN_LR_RATIO}" "${STAGE2_NUM_WORKERS}" "${STAGE2_PREFETCH_FACTOR}" "${STAGE2_SAVE_EVERY}" "${STAGE2_SAVE_TOTAL_LIMIT}" "${STAGE2_VISUALIZE_EVERY}" "${STAGE2_SEED}" "${stage2_resume_arg}" "${STAGE2_DISABLE_BATCH_TRACE}" "${STAGE2_BOX_SOURCE}" "${STAGE2_BOX_JITTER_SCALE}" "${STAGE2_BOX_JITTER_SHIFT}" "${STAGE2_QWEN_BOX_MAX_NEW_TOKENS}" "${STAGE2_BOX_MATCH_IOU_THRESH}" "${STAGE2_BOX_NMS_IOU_THRESH}"

  if [[ "${DRY_RUN_DATA}" != "1" && "${MERGE_FINAL_WEIGHTS}" == "1" ]]; then
    merge_full_weights "${STAGE2_OUTPUT_DIR}" "${MERGED_WEIGHTS_DIR}"
  fi
else
  echo "Skipping stage 2 because RUN_STAGE2=0"
fi
