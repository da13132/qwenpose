#!/usr/bin/env bash
set -Eeuo pipefail

###############################################################################
# LocatePose 两阶段训练脚本
#
# 目标模型：LocateAnything-3B + QwenPose PoseHead。
#
# Stage 1 / GT-box pose warmup：
#   冻结 LocateAnything，只训练 PoseHead / 特征 refiner 等 pose 相关模块；
#   PoseHead 输入 GT box，当前默认训练 30 epoch。
#
# Stage 2 / closed-loop Locate-box training：
#   解冻 LocateAnything LoRA / vision LoRA / projector 等可训练适配参数；
#   先让 LocateAnything generate 人体框，再把生成框喂给 PoseHead；
#   GT box 只用于匹配和监督，不再作为 PoseHead 输入。
#
# 命名说明：
#   脚本层统一使用 LOCATE_* / LocatePose 命名；Python 端已经支持
#   --locate_* 参数别名，脚本层不再暴露旧 backend 命名。
###############################################################################

###############################################################################
# 命令行参数解析
###############################################################################

# DEFAULT_PROJECT_ROOT：默认项目根目录，取当前脚本所在目录的上一级。
DEFAULT_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# PROJECT_ROOT：项目根目录；如果从其他目录启动脚本，也会 cd 到这里。
PROJECT_ROOT="${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"
# SCRIPT_PATH_REL：用于 help 信息展示的脚本相对路径。
SCRIPT_PATH_REL="scripts/$(basename "${BASH_SOURCE[0]}")"

print_usage() {
  cat <<EOF
Usage:
  ${SCRIPT_PATH_REL} [--resume <checkpoint_or_run_dir>] [--VAR VALUE|--VAR=VALUE]...

Options:
  --resume PATH   Resume from a run dir, stage dir, checkpoint dir, or checkpoint file.
                  A full run dir prefers stage2_locate_box_closed_loop; if only stage1
                  exists, stage2 is initialized from stage1 weights.
  --VAR VALUE     Override any script variable. Supports ALL_CAPS, snake_case, and kebab-case.
  --VAR=VALUE     Same as above, using inline assignment.
  -h, --help      Show this help message.

Stages:
  stage1_freeze_locate_gt_box       freeze LocateAnything, use GT boxes
  stage2_locate_box_closed_loop     unfreeze Locate adapters, use Locate-generated boxes
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

# CLI_RESUME_PATH：命令行 --resume 传入的路径，稍后会做阶段感知解析。
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
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

###############################################################################
# Python / torchrun 自动发现
###############################################################################

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

# PYTHON：训练使用的 Python。优先项目内 envs/qwenpose/bin/python。
DEFAULT_PYTHON="$(resolve_default_python)"
PYTHON="${PYTHON:-${DEFAULT_PYTHON}}"
# TORCHRUN：多卡 / DeepSpeed 启动器。找不到时单进程 ZERO_STAGE=none 仍可运行。
DEFAULT_TORCHRUN="$(resolve_default_torchrun)"
TORCHRUN="${TORCHRUN:-${DEFAULT_TORCHRUN}}"

###############################################################################
# --resume 阶段感知解析
###############################################################################

resolve_cli_resume_context() {
  "${PYTHON}" - "$1" <<'PY'
from __future__ import annotations
import shlex
import sys
from pathlib import Path

CHECKPOINT_PAYLOAD_NAME = "qwenpose_checkpoint.pt"
STAGE1_NAME = "stage1_freeze_locate_gt_box"
STAGE2_NAME = "stage2_locate_box_closed_loop"

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

def latest_log_file(run_dir: Path) -> str:
    log_dir = run_dir / "logs"
    if not log_dir.is_dir():
        return ""
    logs = sorted(log_dir.glob("train_*.log"))
    return str(logs[-1]) if logs else ""

target = Path(sys.argv[1]).expanduser().resolve()
if not target.exists():
    raise FileNotFoundError(f"Resume path not found: {target}")

run_dir = target
stage1_dir = ""
stage2_dir = ""
stage1_resume = "none"
stage2_resume = "none"
stage2_init = ""
run_stage1 = "0"
run_stage2 = "1"

if target.is_dir() and ((target / STAGE1_NAME).is_dir() or (target / STAGE2_NAME).is_dir()):
    run_dir = target
    s1 = run_dir / STAGE1_NAME
    s2 = run_dir / STAGE2_NAME
    stage1_dir = str(s1) if s1.exists() else ""
    stage2_dir = str(s2)
    if s2.exists() and has_checkpoint_payload(s2):
        stage2_resume = str(s2)
    elif s1.exists() and has_checkpoint_payload(s1):
        stage2_init = str(s1)
elif target.is_dir() and target.name == STAGE2_NAME:
    run_dir = target.parent
    stage1_candidate = run_dir / STAGE1_NAME
    stage1_dir = str(stage1_candidate) if stage1_candidate.exists() else ""
    stage2_dir = str(target)
    stage2_resume = str(target)
elif target.parent.name == STAGE2_NAME:
    run_dir = target.parent.parent
    stage1_candidate = run_dir / STAGE1_NAME
    stage1_dir = str(stage1_candidate) if stage1_candidate.exists() else ""
    stage2_dir = str(target.parent)
    stage2_resume = str(target)
elif target.parent.name.startswith("checkpoint-") and target.parent.parent.name == STAGE2_NAME:
    run_dir = target.parent.parent.parent
    stage1_candidate = run_dir / STAGE1_NAME
    stage1_dir = str(stage1_candidate) if stage1_candidate.exists() else ""
    stage2_dir = str(target.parent.parent)
    stage2_resume = str(target)
elif target.is_dir() and target.name == STAGE1_NAME:
    run_dir = target.parent
    stage1_dir = str(target)
    stage2_dir = str(run_dir / STAGE2_NAME)
    stage2_init = str(target)
elif target.parent.name == STAGE1_NAME:
    run_dir = target.parent.parent
    stage1_dir = str(target.parent)
    stage2_dir = str(run_dir / STAGE2_NAME)
    stage2_init = str(target.parent)
elif target.parent.name.startswith("checkpoint-") and target.parent.parent.name == STAGE1_NAME:
    run_dir = target.parent.parent.parent
    stage1_dir = str(target.parent.parent)
    stage2_dir = str(run_dir / STAGE2_NAME)
    stage2_init = str(target.parent.parent)
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
print(shell_assign("RESUME_RESOLVED_STAGE1_RESUME", stage1_resume))
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
    if [[ ! -v STAGE1_RESUME_FROM_CHECKPOINT && -n "${RESUME_RESOLVED_STAGE1_RESUME:-}" ]]; then STAGE1_RESUME_FROM_CHECKPOINT="${RESUME_RESOLVED_STAGE1_RESUME}"; fi
    if [[ ! -v STAGE2_RESUME_FROM_CHECKPOINT && -n "${RESUME_RESOLVED_STAGE2_RESUME:-}" ]]; then STAGE2_RESUME_FROM_CHECKPOINT="${RESUME_RESOLVED_STAGE2_RESUME}"; fi
    if [[ ! -v STAGE2_INIT_CHECKPOINT && -n "${RESUME_RESOLVED_STAGE2_INIT_CHECKPOINT:-}" ]]; then STAGE2_INIT_CHECKPOINT="${RESUME_RESOLVED_STAGE2_INIT_CHECKPOINT}"; fi
  fi
fi

###############################################################################
# 输出目录与日志参数
###############################################################################

# RUN_TS：本次训练 run 的时间戳，默认精确到秒，用于 run 名和日志名。
RUN_TS="${RUN_TS:-$(date +%Y%m%d-%H%M%S)}"
# OUTPUT_ROOT：LocatePose 训练输出根目录。每次 run 默认写到它下面。
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/locatepose}"
# RUN_NAME_BASE：run 名基础部分；如果用户已传完整时间戳名，则不重复追加时间戳。
RUN_NAME_BASE="${RUN_NAME:-locatepose-locateanything-3b}"
if [[ "${RUN_NAME_BASE}" =~ [0-9]{8}-[0-9]{6}$ ]]; then
  RUN_NAME="${RUN_NAME_BASE}"
else
  RUN_NAME="${RUN_NAME_BASE}-${RUN_TS}"
fi
# OUTPUT_DIR：本次两阶段 run 的总目录，stage1/stage2 默认都放到它下面。
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"
# LOG_DIR：训练日志目录。
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"
# TRAIN_LOG_FILE：完整 stdout/stderr 日志文件。
TRAIN_LOG_FILE="${TRAIN_LOG_FILE:-${LOG_DIR}/train_${RUN_TS}.log}"

mkdir -p "${LOG_DIR}" "$(dirname "${TRAIN_LOG_FILE}")"
touch "${TRAIN_LOG_FILE}"
exec > >(tee -a "${TRAIN_LOG_FILE}") 2>&1
echo "Logging all stdout/stderr to ${TRAIN_LOG_FILE}"
trap 'status=$?; echo "[ERROR] ${BASH_SOURCE[0]}:${LINENO}: ${BASH_COMMAND} exited with status ${status}" >&2' ERR
trap 'status=$?; echo "========== LocatePose two-stage train exit status ${status} at $(date -Is) =========="; exit ${status}' EXIT

###############################################################################
# 分布式 / DeepSpeed / CUDA 环境参数
###############################################################################

# ZERO_STAGE：DeepSpeed 预设。zero2 是默认推荐；none 用于单进程 CPU/GPU 调试。
ZERO_STAGE="${ZERO_STAGE:-zero2}"
case "${ZERO_STAGE}" in
  zero2) DEFAULT_DEEPSPEED_CONFIG="${PROJECT_ROOT}/scripts/zero2.json" ;;
  zero3) DEFAULT_DEEPSPEED_CONFIG="${PROJECT_ROOT}/scripts/zero3.json" ;;
  zero3_offload) DEFAULT_DEEPSPEED_CONFIG="${PROJECT_ROOT}/scripts/zero3_offload.json" ;;
  none) DEFAULT_DEEPSPEED_CONFIG="" ;;
  *) echo "Unsupported ZERO_STAGE=${ZERO_STAGE}. Use zero2, zero3, zero3_offload, or none." >&2; exit 1 ;;
esac
# DEEPSPEED_CONFIG：DeepSpeed JSON 配置路径；ZERO_STAGE=none 时为空。
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${DEFAULT_DEEPSPEED_CONFIG}}"
# CUDA_VISIBLE_DEVICES：可见 GPU 列表。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
# NPROC_PER_NODE：每个节点启动的训练进程数，一般等于可见 GPU 数。
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
# MASTER_ADDR：torch.distributed 主节点地址。
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
# MASTER_PORT：torch.distributed 端口；默认随机选一个高位端口。
export MASTER_PORT="${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}"
# TORCH_FR_BUFFER_SIZE：NCCL flight recorder buffer，便于排查分布式 hang。
export TORCH_FR_BUFFER_SIZE="${TORCH_FR_BUFFER_SIZE:-200000}"
# TORCH_NCCL_DUMP_ON_TIMEOUT：NCCL 超时时 dump 调试信息。
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
# TORCH_NCCL_DESYNC_DEBUG：打开 NCCL desync 调试。
export TORCH_NCCL_DESYNC_DEBUG="${TORCH_NCCL_DESYNC_DEBUG:-1}"
# TORCH_DISTRIBUTED_DEBUG：torch distributed debug 级别。
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
# PYTORCH_CUDA_ALLOC_CONF：CUDA allocator 配置；默认启用 expandable_segments 降低碎片问题。
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# WANDB_DISABLED：默认关闭 wandb，避免公共训练脚本误上传。
export WANDB_DISABLED="${WANDB_DISABLED:-true}"

###############################################################################
# 数据集与 DataLoader 参数
###############################################################################

# DATASET_ROOT：数据集根目录，内部应包含 coco/mpii/crowdpose/refhuman 等子目录。
DATASET_ROOT="${DATASET_ROOT:-datasets}"
# SPLIT：训练 split。COCO 会映射到 train2017 等内部 split。
SPLIT="${SPLIT:-train}"
# MIXING_STRATEGY：多数据集混合方式。interleave 会按数据集交错采样。
MIXING_STRATEGY="${MIXING_STRATEGY:-interleave}"
# DATASET_MIX_WEIGHTS：全局数据集采样权重；stage 可单独覆盖。
DATASET_MIX_WEIGHTS="${DATASET_MIX_WEIGHTS:-auto}"
# MAX_INSTANCES：每张图最多保留/训练的人体实例数。
MAX_INSTANCES="${MAX_INSTANCES:-80}"
# MAX_SAMPLES_PER_DATASET：每个数据集最多样本数；空值表示不截断。
MAX_SAMPLES_PER_DATASET="${MAX_SAMPLES_PER_DATASET:-}"
# REFHUMAN_MAX_CAPTIONS_PER_INSTANCE：RefHuman 每个人最多使用多少条文本描述。
REFHUMAN_MAX_CAPTIONS_PER_INSTANCE="${REFHUMAN_MAX_CAPTIONS_PER_INSTANCE:-1}"
# RECORD_CACHE_DIR：样本 record 缓存目录，加速重复启动。
RECORD_CACHE_DIR="${RECORD_CACHE_DIR:-.cache/qwenpose_records}"
# DISABLE_RECORD_CACHE：是否禁用 record 缓存。1 表示每次重新解析数据集。
DISABLE_RECORD_CACHE="${DISABLE_RECORD_CACHE:-0}"
# NUM_WORKERS：DataLoader worker 数。默认 0 更稳，避免多进程 PIL/IO 问题。
NUM_WORKERS="${NUM_WORKERS:-0}"
# PREFETCH_FACTOR：DataLoader prefetch factor，仅 NUM_WORKERS>0 时生效。
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
# DISABLE_HOMOGENEOUS_BATCHES：是否关闭同数据集 batch 采样。0 表示启用同源 batch。
DISABLE_HOMOGENEOUS_BATCHES="${DISABLE_HOMOGENEOUS_BATCHES:-0}"

###############################################################################
# LocateAnything backbone 参数
###############################################################################

# LOCATE_MODEL_PATH：LocateAnything-3B 权重目录。
LOCATE_MODEL_PATH="${LOCATE_MODEL_PATH:-weights/LocateAnything-3B}"
# LOCATE_DTYPE：LocateAnything 加载 dtype。bfloat16 是默认推荐。
LOCATE_DTYPE="${LOCATE_DTYPE:-bfloat16}"
# LOCATE_ATTN_IMPLEMENTATION：Locate vision tower 使用 flash_attention_2 full-batch varlen attention。
LOCATE_ATTN_IMPLEMENTATION="${LOCATE_ATTN_IMPLEMENTATION:-flash_attention_2}"
# LOCATE_GRADIENT_CHECKPOINTING：是否启用 LocateAnything gradient checkpointing 以省显存。
LOCATE_GRADIENT_CHECKPOINTING="${LOCATE_GRADIENT_CHECKPOINTING:-1}"
# LOCATE_MIN_PIXELS：保留为兼容参数；LocateAnything processor 实际不识别 Qwen-style min_pixels。
LOCATE_MIN_PIXELS="${LOCATE_MIN_PIXELS:-}"
# LOCATE_MAX_PIXELS：兼容参数；默认不使用它作为核心控制。
LOCATE_MAX_PIXELS="${LOCATE_MAX_PIXELS:-}"
# LOCATE_IMAGE_TOKEN_LIMIT：每张图 raw MoonViT patch token 上限；默认不压当前常规样本，只挡极端大图。
LOCATE_IMAGE_TOKEN_LIMIT="${LOCATE_IMAGE_TOKEN_LIMIT:-4096}"
# LOCATE_FEATURE_SIZE：从 LocateAnything token 特征投影出的空间特征图边长。
LOCATE_FEATURE_SIZE="${LOCATE_FEATURE_SIZE:-64}"
# LOCATE_FEATURE_REFINER_LAYERS：Locate 特征 refiner 层数。
LOCATE_FEATURE_REFINER_LAYERS="${LOCATE_FEATURE_REFINER_LAYERS:-2}"
# LOCATE_FEATURE_REFINER_BOTTLENECK_DIM：Locate 特征 refiner bottleneck 隐藏维度。
LOCATE_FEATURE_REFINER_BOTTLENECK_DIM="${LOCATE_FEATURE_REFINER_BOTTLENECK_DIM:-256}"
# LOCATE_FEATURE_REFINER_INIT_SCALE：refiner 残差初始化尺度，越小越稳。
LOCATE_FEATURE_REFINER_INIT_SCALE="${LOCATE_FEATURE_REFINER_INIT_SCALE:-0.1}"
# LOCATE_LORA_R：LocateAnything 语言/主干 LoRA rank。
LOCATE_LORA_R="${LOCATE_LORA_R:-32}"
# LOCATE_LORA_ALPHA：LocateAnything 语言/主干 LoRA alpha。
LOCATE_LORA_ALPHA="${LOCATE_LORA_ALPHA:-64}"
# LOCATE_LORA_DROPOUT：LocateAnything 语言/主干 LoRA dropout。
LOCATE_LORA_DROPOUT="${LOCATE_LORA_DROPOUT:-0.05}"
# LOCATE_VISION_LORA_R：LocateAnything vision 分支 LoRA rank。
LOCATE_VISION_LORA_R="${LOCATE_VISION_LORA_R:-16}"
# LOCATE_VISION_LORA_ALPHA：LocateAnything vision 分支 LoRA alpha。
LOCATE_VISION_LORA_ALPHA="${LOCATE_VISION_LORA_ALPHA:-32}"
# LOCATE_VISION_LORA_DROPOUT：LocateAnything vision 分支 LoRA dropout。
LOCATE_VISION_LORA_DROPOUT="${LOCATE_VISION_LORA_DROPOUT:-0.05}"

###############################################################################
# PoseHead 结构参数
###############################################################################

# HIDDEN_DIM：PoseHead 内部 hidden dimension。
HIDDEN_DIM="${HIDDEN_DIM:-448}"
# POSE_DECODER_LAYERS：Pose decoder 层数。
POSE_DECODER_LAYERS="${POSE_DECODER_LAYERS:-3}"
# REFINEMENT_STEPS：关键点迭代 refinement 步数。
REFINEMENT_STEPS="${REFINEMENT_STEPS:-3}"
# DECODER_HEADS：pose decoder attention head 数。
DECODER_HEADS="${DECODER_HEADS:-8}"
# BOX_CONDITION_SCALE：PoseHead 使用条件框前的放大比例，给关键点留上下文。
BOX_CONDITION_SCALE="${BOX_CONDITION_SCALE:-1.2}"
# POSE_ROI_SIZE：每个 box 从 Locate feature map 上采样的 ROI 特征边长。
POSE_ROI_SIZE="${POSE_ROI_SIZE:-16}"
# DISABLE_REFINEMENT：是否关闭 keypoint refinement。
DISABLE_REFINEMENT="${DISABLE_REFINEMENT:-0}"

###############################################################################
# 优化器 / 学习率 / 训练控制参数
###############################################################################

# LR：全局基础学习率；stage 可用 STAGE*_LR 单独覆盖。
LR="${LR:-2e-4}"
# LOCATE_LR_SCALE：LocateAnything LoRA 参数学习率相对 LR 的倍率。
LOCATE_LR_SCALE="${LOCATE_LR_SCALE:-0.05}"
# LOCATE_VISION_SCALE：LocateAnything vision LoRA 参数学习率相对 LR 的倍率。
LOCATE_VISION_SCALE="${LOCATE_VISION_SCALE:-0.02}"
# LOCATE_PROJECTOR_SCALE：LocateAnything projector 参数学习率相对 LR 的倍率。
LOCATE_PROJECTOR_SCALE="${LOCATE_PROJECTOR_SCALE:-0.05}"
# WEIGHT_DECAY：AdamW weight decay。
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
# GRAD_CLIP：梯度裁剪范数。
GRAD_CLIP="${GRAD_CLIP:-1.0}"
# WARMUP_STEPS：scheduler warmup step 数。
WARMUP_STEPS="${WARMUP_STEPS:-100}"
# MIN_LR_RATIO：cosine scheduler 最低学习率比例。
MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}"
# DEVICE：训练设备。cuda 为正式训练，cpu 主要用于 dry-run/debug。
DEVICE="${DEVICE:-cuda}"
# AMP：是否启用 torch AMP。DeepSpeed 模式下通常由 ds config 管理。
AMP="${AMP:-0}"
# LOG_EVERY：训练日志打印间隔。
LOG_EVERY="${LOG_EVERY:-10}"
# SAVE_EVERY：checkpoint 保存间隔。
SAVE_EVERY="${SAVE_EVERY:-500}"
# SAVE_TOTAL_LIMIT：每个 stage 最多保留多少个 checkpoint。
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
# SEED：随机种子。
SEED="${SEED:-42}"
# VISUALIZE_EVERY：训练可视化保存间隔，0 表示关闭。
VISUALIZE_EVERY="${VISUALIZE_EVERY:-10}"
# VISUALIZE_MAX_INSTANCES：每张可视化最多绘制实例数。
VISUALIZE_MAX_INSTANCES="${VISUALIZE_MAX_INSTANCES:-8}"
# DRY_RUN_DATA：只解析数据和打印首个 batch，不加载模型、不训练。
DRY_RUN_DATA="${DRY_RUN_DATA:-0}"
# PROGRESS_BAR：是否显示 tqdm 进度条。0 表示关闭。
PROGRESS_BAR="${PROGRESS_BAR:-1}"
# SYNC_TIMING：是否同步 CUDA 后再记录耗时，便于 profiling 但会变慢。
SYNC_TIMING="${SYNC_TIMING:-0}"
# DISABLE_BATCH_TRACE：是否关闭 batch JSONL trace。0 表示保留 trace 方便排查。
DISABLE_BATCH_TRACE="${DISABLE_BATCH_TRACE:-0}"

###############################################################################
# Loss 权重参数
###############################################################################

# W_OKS：OKS loss 权重。
W_OKS="${W_OKS:-0.2}"
# W_COORD：归一化坐标回归 loss 权重。
W_COORD="${W_COORD:-5.0}"
# W_VIS：关键点可见性 BCE loss 权重。
W_VIS="${W_VIS:-0.05}"
# W_HARD_JOINT：hard-joint mining loss 权重，默认关闭。
W_HARD_JOINT="${W_HARD_JOINT:-0.0}"
# HARD_JOINT_FRACTION：hard-joint mining 选取最难关节点比例。
HARD_JOINT_FRACTION="${HARD_JOINT_FRACTION:-0.2}"

###############################################################################
# Locate 生成框 / grounding 辅助监督参数
###############################################################################

# LOCATE_GENERATION_MODE：LocateAnything generate 模式。hybrid 兼顾速度和稳定性。
LOCATE_GENERATION_MODE="${LOCATE_GENERATION_MODE:-hybrid}"
# LOCATE_BOX_MAX_NEW_TOKENS：LocateAnything 生成框文本的最大新 token 数。
LOCATE_BOX_MAX_NEW_TOKENS="${LOCATE_BOX_MAX_NEW_TOKENS:-8192}"
# BOX_MATCH_IOU_THRESH：生成框和 GT 框匹配的 IoU 阈值。
BOX_MATCH_IOU_THRESH="${BOX_MATCH_IOU_THRESH:-0.10}"
# BOX_NMS_IOU_THRESH：生成框 NMS 阈值。
BOX_NMS_IOU_THRESH="${BOX_NMS_IOU_THRESH:-0.70}"
# LOCATE_LM_LOSS_EVERY：Locate grounding LM 辅助 loss 计算频率。
LOCATE_LM_LOSS_EVERY="${LOCATE_LM_LOSS_EVERY:-2}"
# LOCATE_LM_MAX_INSTANCES：Locate bbox LM answer 中最多监督实例数。
LOCATE_LM_MAX_INSTANCES="${LOCATE_LM_MAX_INSTANCES:-20}"
# LOCATE_LM_MAX_POINTS：Locate point LM answer 中每实例最多监督点数。
LOCATE_LM_MAX_POINTS="${LOCATE_LM_MAX_POINTS:-8}"
# DISABLE_LOCATE_GROUNDING_AUX：是否关闭 Locate grounding 辅助监督。
DISABLE_LOCATE_GROUNDING_AUX="${DISABLE_LOCATE_GROUNDING_AUX:-0}"

###############################################################################
# 两阶段开关与 stage-specific 参数
###############################################################################

# RUN_STAGE1：是否执行 stage1。1 执行；0 跳过。
RUN_STAGE1="${RUN_STAGE1:-1}"
# RUN_STAGE2：是否执行 stage2。1 执行；0 跳过。
RUN_STAGE2="${RUN_STAGE2:-1}"
# STAGE1_OUTPUT_DIR：stage1 输出目录。
STAGE1_OUTPUT_DIR="${STAGE1_OUTPUT_DIR:-${OUTPUT_DIR}/stage1_freeze_locate_gt_box}"
# STAGE2_OUTPUT_DIR：stage2 输出目录。
STAGE2_OUTPUT_DIR="${STAGE2_OUTPUT_DIR:-${OUTPUT_DIR}/stage2_locate_box_closed_loop}"
# STAGE2_INIT_WEIGHTS_DIR：stage2 weight-only 初始化 checkpoint 临时目录。
STAGE2_INIT_WEIGHTS_DIR="${STAGE2_INIT_WEIGHTS_DIR:-${OUTPUT_DIR}/stage2_init_weights}"
# STAGE1_TRAIN_DATASETS：stage1 训练数据集。
STAGE1_TRAIN_DATASETS="${STAGE1_TRAIN_DATASETS:-coco}"
# STAGE2_TRAIN_DATASETS：stage2 训练数据集。
STAGE2_TRAIN_DATASETS="${STAGE2_TRAIN_DATASETS:-coco,mpii,crowdpose,refhuman}"
# STAGE1_DATASET_MIX_WEIGHTS：stage1 数据集采样权重。
STAGE1_DATASET_MIX_WEIGHTS="${STAGE1_DATASET_MIX_WEIGHTS:-${DATASET_MIX_WEIGHTS}}"
# STAGE2_DATASET_MIX_WEIGHTS：stage2 数据集采样权重。
STAGE2_DATASET_MIX_WEIGHTS="${STAGE2_DATASET_MIX_WEIGHTS:-coco:2,mpii:1,crowdpose:2,refhuman:1}"
# STAGE1_EPOCHS：stage1 epoch 数。
STAGE1_EPOCHS="${STAGE1_EPOCHS:-30}"
# STAGE2_EPOCHS：stage2 epoch 数。
STAGE2_EPOCHS="${STAGE2_EPOCHS:-12}"
# STAGE1_BATCH_SIZE：stage1 每卡 micro batch size；Locate vision 走 flash_attention_2 full-batch forward。
STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-16}"
# STAGE2_BATCH_SIZE：stage2 每卡 micro batch size；locate_generate 逐图生成，默认 1。
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-1}"
# STAGE1_GRAD_ACCUM_STEPS：stage1 梯度累积步数。
STAGE1_GRAD_ACCUM_STEPS="${STAGE1_GRAD_ACCUM_STEPS:-1}"
# STAGE2_GRAD_ACCUM_STEPS：stage2 梯度累积步数。
STAGE2_GRAD_ACCUM_STEPS="${STAGE2_GRAD_ACCUM_STEPS:-8}"
# STAGE1_LR：stage1 基础学习率。
STAGE1_LR="${STAGE1_LR:-2e-4}"
# STAGE2_LR：stage2 基础学习率。
STAGE2_LR="${STAGE2_LR:-5e-5}"
# STAGE1_MAX_STEPS：stage1 最大 optimizer step；0 表示按 epoch 跑满。
STAGE1_MAX_STEPS="${STAGE1_MAX_STEPS:-${MAX_STEPS:-0}}"
# STAGE2_MAX_STEPS：stage2 最大 optimizer step；0 表示按 epoch 跑满。
STAGE2_MAX_STEPS="${STAGE2_MAX_STEPS:-0}"
# STAGE1_FREEZE_LOCATE：stage1 是否冻结 LocateAnything。
STAGE1_FREEZE_LOCATE="${STAGE1_FREEZE_LOCATE:-1}"
# STAGE2_FREEZE_LOCATE：stage2 是否冻结 LocateAnything。
STAGE2_FREEZE_LOCATE="${STAGE2_FREEZE_LOCATE:-0}"
# STAGE1_BOX_SOURCE：stage1 条件框来源，默认 GT box。
STAGE1_BOX_SOURCE="${STAGE1_BOX_SOURCE:-gt}"
# STAGE2_BOX_SOURCE：stage2 条件框来源，默认 LocateAnything 生成框。
STAGE2_BOX_SOURCE="${STAGE2_BOX_SOURCE:-locate_generate}"
# STAGE1_BOX_JITTER_SCALE：stage1 条件框缩放扰动。
STAGE1_BOX_JITTER_SCALE="${STAGE1_BOX_JITTER_SCALE:-0.0}"
# STAGE1_BOX_JITTER_SHIFT：stage1 条件框中心平移扰动。
STAGE1_BOX_JITTER_SHIFT="${STAGE1_BOX_JITTER_SHIFT:-0.0}"
# STAGE2_BOX_JITTER_SCALE：stage2 生成框不再额外 jitter。
STAGE2_BOX_JITTER_SCALE="${STAGE2_BOX_JITTER_SCALE:-0.0}"
# STAGE2_BOX_JITTER_SHIFT：stage2 生成框不再额外 jitter。
STAGE2_BOX_JITTER_SHIFT="${STAGE2_BOX_JITTER_SHIFT:-0.0}"
# STAGE1_W_LOCATE_BOX_LM：stage1 Locate bbox LM 辅助 loss 权重，默认关闭。
STAGE1_W_LOCATE_BOX_LM="${STAGE1_W_LOCATE_BOX_LM:-0}"
# STAGE1_W_LOCATE_POINT_LM：stage1 Locate point LM 辅助 loss 权重，默认关闭。
STAGE1_W_LOCATE_POINT_LM="${STAGE1_W_LOCATE_POINT_LM:-0}"
# STAGE2_W_LOCATE_BOX_LM：stage2 Locate bbox LM 辅助 loss 权重。
STAGE2_W_LOCATE_BOX_LM="${STAGE2_W_LOCATE_BOX_LM:-0.04}"
# STAGE2_W_LOCATE_POINT_LM：stage2 Locate point LM 辅助 loss 权重。
STAGE2_W_LOCATE_POINT_LM="${STAGE2_W_LOCATE_POINT_LM:-0.01}"
# STAGE1_RESUME_FROM_CHECKPOINT：stage1 断点续训路径。none 表示不续训。
STAGE1_RESUME_FROM_CHECKPOINT="${STAGE1_RESUME_FROM_CHECKPOINT:-none}"
# STAGE2_RESUME_FROM_CHECKPOINT：stage2 断点续训路径。none 表示不续训。
STAGE2_RESUME_FROM_CHECKPOINT="${STAGE2_RESUME_FROM_CHECKPOINT:-none}"
# STAGE2_INIT_CHECKPOINT：stage2 显式 weight-only 初始化来源。
STAGE2_INIT_CHECKPOINT="${STAGE2_INIT_CHECKPOINT:-}"
# STAGE2_INIT_FROM_STAGE1：stage2 是否默认从 stage1 输出初始化。
STAGE2_INIT_FROM_STAGE1="${STAGE2_INIT_FROM_STAGE1:-1}"
# MERGE_FINAL_WEIGHTS：LocatePose 当前不自动合并 LocateAnything 完整权重；设 1 只提示。
MERGE_FINAL_WEIGHTS="${MERGE_FINAL_WEIGHTS:-0}"

if [[ -n "${CLI_RESUME_PATH}" ]]; then
  if [[ "${STAGE2_RESUME_FROM_CHECKPOINT}" == "none" && -n "${RESUME_RESOLVED_STAGE2_RESUME:-}" ]]; then
    STAGE2_RESUME_FROM_CHECKPOINT="${RESUME_RESOLVED_STAGE2_RESUME}"
  fi
  if [[ -z "${STAGE2_INIT_CHECKPOINT}" && -n "${RESUME_RESOLVED_STAGE2_INIT_CHECKPOINT:-}" ]]; then
    STAGE2_INIT_CHECKPOINT="${RESUME_RESOLVED_STAGE2_INIT_CHECKPOINT}"
  fi
fi

###############################################################################
# 参数校验
###############################################################################

require_positive_int() { local name="$1" value="$2"; if ! [[ "${value}" =~ ^[0-9]+$ ]] || (( value <= 0 )); then echo "${name} must be a positive integer, got: ${value}" >&2; exit 1; fi; }
require_nonnegative_int() { local name="$1" value="$2"; if ! [[ "${value}" =~ ^[0-9]+$ ]]; then echo "${name} must be a non-negative integer, got: ${value}" >&2; exit 1; fi; }
require_bool() { local name="$1" value="$2"; if [[ "${value}" != "0" && "${value}" != "1" ]]; then echo "${name} must be 0 or 1, got: ${value}" >&2; exit 1; fi; }

for spec in \
  "NPROC_PER_NODE:${NPROC_PER_NODE}" \
  "MAX_INSTANCES:${MAX_INSTANCES}" \
  "STAGE1_BATCH_SIZE:${STAGE1_BATCH_SIZE}" \
  "STAGE2_BATCH_SIZE:${STAGE2_BATCH_SIZE}" \
  "STAGE1_GRAD_ACCUM_STEPS:${STAGE1_GRAD_ACCUM_STEPS}" \
  "STAGE2_GRAD_ACCUM_STEPS:${STAGE2_GRAD_ACCUM_STEPS}" \
  "STAGE1_EPOCHS:${STAGE1_EPOCHS}" \
  "STAGE2_EPOCHS:${STAGE2_EPOCHS}" \
  "REFINEMENT_STEPS:${REFINEMENT_STEPS}" \
  "POSE_ROI_SIZE:${POSE_ROI_SIZE}" \
  "VISUALIZE_MAX_INSTANCES:${VISUALIZE_MAX_INSTANCES}" \
  "LOCATE_BOX_MAX_NEW_TOKENS:${LOCATE_BOX_MAX_NEW_TOKENS}" \
  "LOCATE_LM_MAX_INSTANCES:${LOCATE_LM_MAX_INSTANCES}" \
  "LOCATE_LM_MAX_POINTS:${LOCATE_LM_MAX_POINTS}"; do
  require_positive_int "${spec%%:*}" "${spec#*:}"
done

for spec in \
  "STAGE1_MAX_STEPS:${STAGE1_MAX_STEPS}" \
  "STAGE2_MAX_STEPS:${STAGE2_MAX_STEPS}" \
  "NUM_WORKERS:${NUM_WORKERS}" \
  "VISUALIZE_EVERY:${VISUALIZE_EVERY}" \
  "LOCATE_LM_LOSS_EVERY:${LOCATE_LM_LOSS_EVERY}"; do
  require_nonnegative_int "${spec%%:*}" "${spec#*:}"
done

for spec in \
  RUN_STAGE1 RUN_STAGE2 STAGE1_FREEZE_LOCATE STAGE2_FREEZE_LOCATE STAGE2_INIT_FROM_STAGE1 \
  MERGE_FINAL_WEIGHTS LOCATE_GRADIENT_CHECKPOINTING AMP DRY_RUN_DATA PROGRESS_BAR SYNC_TIMING \
  DISABLE_BATCH_TRACE DISABLE_HOMOGENEOUS_BATCHES DISABLE_REFINEMENT DISABLE_LOCATE_GROUNDING_AUX \
  DISABLE_RECORD_CACHE; do
  require_bool "${spec}" "${!spec}"
done

if [[ "${LOCATE_GENERATION_MODE}" != "fast" && "${LOCATE_GENERATION_MODE}" != "slow" && "${LOCATE_GENERATION_MODE}" != "hybrid" ]]; then
  echo "LOCATE_GENERATION_MODE must be fast, slow, or hybrid, got: ${LOCATE_GENERATION_MODE}" >&2
  exit 1
fi
if [[ "${STAGE1_BOX_SOURCE}" != "gt" && "${STAGE1_BOX_SOURCE}" != "locate_generate" ]]; then
  echo "STAGE1_BOX_SOURCE must be gt or locate_generate, got: ${STAGE1_BOX_SOURCE}" >&2
  exit 1
fi
if [[ "${STAGE2_BOX_SOURCE}" != "gt" && "${STAGE2_BOX_SOURCE}" != "locate_generate" ]]; then
  echo "STAGE2_BOX_SOURCE must be gt or locate_generate, got: ${STAGE2_BOX_SOURCE}" >&2
  exit 1
fi
if [[ "${DEVICE}" != "cuda" && "${ZERO_STAGE}" != "none" ]]; then
  echo "DEVICE=${DEVICE} cannot use DeepSpeed ${ZERO_STAGE}. Use ZERO_STAGE=none for CPU debugging." >&2
  exit 1
fi
if [[ "${RUN_STAGE2}" == "1" && "${STAGE2_BOX_SOURCE}" == "locate_generate" && "${ZERO_STAGE}" != "zero2" && "${ZERO_STAGE}" != "none" ]]; then
  echo "Stage 2 locate_generate calls model.generate during training and currently supports ZERO_STAGE=zero2 or none. Got ZERO_STAGE=${ZERO_STAGE}." >&2
  exit 1
fi
if [[ ! -e "${LOCATE_MODEL_PATH}" ]]; then
  echo "LOCATE_MODEL_PATH not found: ${LOCATE_MODEL_PATH}" >&2
  exit 1
fi
if [[ -n "${DEEPSPEED_CONFIG}" && ! -f "${DEEPSPEED_CONFIG}" ]]; then
  echo "DEEPSPEED_CONFIG not found: ${DEEPSPEED_CONFIG}" >&2
  exit 1
fi
if [[ -n "${LOCATE_MIN_PIXELS}" ]]; then require_positive_int LOCATE_MIN_PIXELS "${LOCATE_MIN_PIXELS}"; fi
if [[ -n "${LOCATE_MAX_PIXELS}" ]]; then require_positive_int LOCATE_MAX_PIXELS "${LOCATE_MAX_PIXELS}"; fi
if [[ -n "${LOCATE_MIN_PIXELS}" && -n "${LOCATE_MAX_PIXELS}" ]] && (( LOCATE_MAX_PIXELS < LOCATE_MIN_PIXELS )); then
  echo "LOCATE_MAX_PIXELS=${LOCATE_MAX_PIXELS} must be >= LOCATE_MIN_PIXELS=${LOCATE_MIN_PIXELS}." >&2
  exit 1
fi
if (( NPROC_PER_NODE > 1 )) || [[ "${ZERO_STAGE}" != "none" ]]; then
  [[ -n "${TORCHRUN}" ]] || { echo "torchrun not found. Set TORCHRUN=/path/to/torchrun or use ZERO_STAGE=none NPROC_PER_NODE=1." >&2; exit 1; }
fi

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

check_optional_checkpoint STAGE1_RESUME_FROM_CHECKPOINT "${STAGE1_RESUME_FROM_CHECKPOINT}"
check_optional_checkpoint STAGE2_RESUME_FROM_CHECKPOINT "${STAGE2_RESUME_FROM_CHECKPOINT}"
check_optional_checkpoint STAGE2_INIT_CHECKPOINT "${STAGE2_INIT_CHECKPOINT}"

###############################################################################
# checkpoint 初始化与启动函数
###############################################################################

prepare_weights_only_checkpoint() {
  local source_path="$1"
  local dest_dir="$2"
  if [[ -z "${source_path}" || -z "${dest_dir}" || "${dest_dir}" == "/" ]]; then
    echo "Invalid weight-only checkpoint arguments: source=${source_path}, dest=${dest_dir}" >&2
    exit 1
  fi
  if ! resume_target_has_checkpoint "${source_path}"; then
    echo "Cannot initialize stage 2; no checkpoint found in ${source_path}" >&2
    exit 1
  fi
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
source = Path(sys.argv[1]).expanduser().resolve()
dest = Path(sys.argv[2]).expanduser().resolve()

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
for adapter_name in ("backbone_lora_adapter", "qwen_lora_adapter"):
    adapter_src = resolved / adapter_name if resolved.is_dir() else None
    if adapter_src is not None and adapter_src.is_dir():
        shutil.copytree(adapter_src, out / adapter_name, dirs_exist_ok=True)
print(out)
PY
}

add_opt() {
  local -n arr_ref="$1"
  local flag="$2"
  local value="$3"
  if [[ -n "${value}" && "${value}" != "none" ]]; then
    arr_ref+=("${flag}" "${value}")
  fi
  return 0
}

run_train_pose() {
  local -n arr="$1"
  if [[ "${ZERO_STAGE}" == "none" && "${NPROC_PER_NODE}" == "1" ]]; then
    "${PYTHON}" -m qwenpose.train_pose "${arr[@]}"
  elif [[ -n "${TORCHRUN}" ]]; then
    "${TORCHRUN}" \
      --nproc_per_node "${NPROC_PER_NODE}" \
      --master_addr "${MASTER_ADDR}" \
      --master_port "${MASTER_PORT}" \
      "${PROJECT_ROOT}/src/qwenpose/train_pose.py" \
      "${arr[@]}"
  else
    "${PYTHON}" -m torch.distributed.run \
      --nproc_per_node "${NPROC_PER_NODE}" \
      --master_addr "${MASTER_ADDR}" \
      --master_port "${MASTER_PORT}" \
      "${PROJECT_ROOT}/src/qwenpose/train_pose.py" \
      "${arr[@]}"
  fi
}

common_args() {
  local -n a="$1"
  a+=(--backbone locatepose)
  a+=(--dataset_root "${DATASET_ROOT}" --split "${SPLIT}" --max_instances "${MAX_INSTANCES}")
  a+=(--refhuman_max_captions_per_instance "${REFHUMAN_MAX_CAPTIONS_PER_INSTANCE}")
  a+=(--num_workers "${NUM_WORKERS}" --prefetch_factor "${PREFETCH_FACTOR}")
  a+=(--mixing_strategy "${MIXING_STRATEGY}" --record_cache_dir "${RECORD_CACHE_DIR}")
  a+=(--locate_model_path "${LOCATE_MODEL_PATH}" --locate_dtype "${LOCATE_DTYPE}" --locate_attn_implementation "${LOCATE_ATTN_IMPLEMENTATION}")
  add_opt a --locate_min_pixels "${LOCATE_MIN_PIXELS}"
  add_opt a --locate_max_pixels "${LOCATE_MAX_PIXELS}"
  add_opt a --locate_image_token_limit "${LOCATE_IMAGE_TOKEN_LIMIT}"
  a+=(--locate_feature_size "${LOCATE_FEATURE_SIZE}" --locate_feature_refiner_layers "${LOCATE_FEATURE_REFINER_LAYERS}")
  a+=(--locate_feature_refiner_bottleneck_dim "${LOCATE_FEATURE_REFINER_BOTTLENECK_DIM}" --locate_feature_refiner_init_scale "${LOCATE_FEATURE_REFINER_INIT_SCALE}")
  a+=(--locate_lora_r "${LOCATE_LORA_R}" --locate_lora_alpha "${LOCATE_LORA_ALPHA}" --locate_lora_dropout "${LOCATE_LORA_DROPOUT}")
  a+=(--locate_vision_lora_r "${LOCATE_VISION_LORA_R}" --locate_vision_lora_alpha "${LOCATE_VISION_LORA_ALPHA}" --locate_vision_lora_dropout "${LOCATE_VISION_LORA_DROPOUT}")
  a+=(--hidden_dim "${HIDDEN_DIM}" --pose_decoder_layers "${POSE_DECODER_LAYERS}" --refinement_steps "${REFINEMENT_STEPS}" --decoder_heads "${DECODER_HEADS}")
  a+=(--box_condition_scale "${BOX_CONDITION_SCALE}" --pose_roi_size "${POSE_ROI_SIZE}")
  a+=(--locate_lr_scale "${LOCATE_LR_SCALE}" --locate_vision_scale "${LOCATE_VISION_SCALE}" --locate_projector_scale "${LOCATE_PROJECTOR_SCALE}")
  a+=(--locate_generation_mode "${LOCATE_GENERATION_MODE}" --locate_box_max_new_tokens "${LOCATE_BOX_MAX_NEW_TOKENS}")
  a+=(--box_match_iou_thresh "${BOX_MATCH_IOU_THRESH}" --box_nms_iou_thresh "${BOX_NMS_IOU_THRESH}")
  a+=(--weight_decay "${WEIGHT_DECAY}" --grad_clip "${GRAD_CLIP}" --warmup_steps "${WARMUP_STEPS}" --min_lr_ratio "${MIN_LR_RATIO}")
  a+=(--log_every "${LOG_EVERY}" --save_every "${SAVE_EVERY}" --save_total_limit "${SAVE_TOTAL_LIMIT}" --seed "${SEED}" --device "${DEVICE}")
  a+=(--w_oks "${W_OKS}" --w_coord "${W_COORD}" --w_vis "${W_VIS}")
  a+=(--w_hard_joint "${W_HARD_JOINT}" --hard_joint_fraction "${HARD_JOINT_FRACTION}")
  a+=(--visualize_every "${VISUALIZE_EVERY}" --visualize_max_instances "${VISUALIZE_MAX_INSTANCES}")
  add_opt a --max_samples_per_dataset "${MAX_SAMPLES_PER_DATASET}"
  add_opt a --deepspeed_config "${DEEPSPEED_CONFIG}"
  [[ "${LOCATE_GRADIENT_CHECKPOINTING}" == "1" ]] && a+=(--locate_gradient_checkpointing)
  [[ "${DISABLE_RECORD_CACHE}" == "1" ]] && a+=(--disable_record_cache)
  [[ "${AMP}" == "1" ]] && a+=(--amp)
  [[ "${SYNC_TIMING}" == "1" ]] && a+=(--sync_timing)
  [[ "${PROGRESS_BAR}" == "0" ]] && a+=(--disable_progress)
  [[ "${DRY_RUN_DATA}" == "1" ]] && a+=(--dry_run_data)
  [[ "${DISABLE_BATCH_TRACE}" == "1" ]] && a+=(--disable_batch_trace)
  [[ "${DISABLE_HOMOGENEOUS_BATCHES}" == "1" ]] && a+=(--disable_homogeneous_batches)
  [[ "${DISABLE_REFINEMENT}" == "1" ]] && a+=(--disable_refinement)
  [[ "${DISABLE_LOCATE_GROUNDING_AUX}" == "1" ]] && a+=(--disable_locate_grounding_aux)
  return 0
}

run_stage() {
  local stage_label="$1"
  local output_dir="$2"
  local datasets="$3"
  local dataset_weights="$4"
  local epochs="$5"
  local batch_size="$6"
  local grad_accum_steps="$7"
  local lr="$8"
  local max_steps="$9"
  local freeze_locate="${10}"
  local box_source="${11}"
  local jitter_scale="${12}"
  local jitter_shift="${13}"
  local w_box_lm="${14}"
  local w_point_lm="${15}"
  local resume_arg="${16}"

  local effective_batch=$((NPROC_PER_NODE * batch_size * grad_accum_steps))
  mkdir -p "${output_dir}" "${output_dir}/logs"
  local args=()
  common_args args
  args+=(--datasets "${datasets}" --dataset_mix_weights "${dataset_weights}" --output_dir "${output_dir}")
  args+=(--epochs "${epochs}" --batch_size "${batch_size}" --grad_accum_steps "${grad_accum_steps}" --lr "${lr}" --max_steps "${max_steps}")
  args+=(--box_source "${box_source}" --box_jitter_scale "${jitter_scale}" --box_jitter_shift "${jitter_shift}")
  args+=(--w_locate_box_lm "${w_box_lm}" --w_locate_point_lm "${w_point_lm}" --locate_lm_loss_every "${LOCATE_LM_LOSS_EVERY}")
  args+=(--locate_lm_max_instances "${LOCATE_LM_MAX_INSTANCES}" --locate_lm_max_points "${LOCATE_LM_MAX_POINTS}")
  [[ "${freeze_locate}" == "1" ]] && args+=(--freeze_locate)
  add_opt args --resume_from_checkpoint "${resume_arg}"

  echo "================ LocatePose ${stage_label} 配置 ================"
  echo "OUTPUT_DIR=${output_dir}"
  echo "DATASETS=${datasets}"
  echo "DATASET_MIX_WEIGHTS=${dataset_weights}"
  echo "ZERO_STAGE=${ZERO_STAGE}"
  echo "DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG}"
  echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "BATCH_SIZE=${batch_size}"
  echo "GRAD_ACCUM_STEPS=${grad_accum_steps}"
  echo "EFFECTIVE_BATCH=${effective_batch}"
  echo "EPOCHS=${epochs}"
  echo "MAX_STEPS=${max_steps}"
  echo "FREEZE_LOCATE=${freeze_locate}"
  echo "LOCATE_ATTN_IMPLEMENTATION=${LOCATE_ATTN_IMPLEMENTATION}"
  echo "LOCATE_MAX_PIXELS=${LOCATE_MAX_PIXELS}"
  echo "LOCATE_IMAGE_TOKEN_LIMIT=${LOCATE_IMAGE_TOKEN_LIMIT}"
  echo "BOX_SOURCE=${box_source}"
  echo "BOX_JITTER_SCALE=${jitter_scale}"
  echo "BOX_JITTER_SHIFT=${jitter_shift}"
  echo "W_LOCATE_BOX_LM=${w_box_lm}"
  echo "W_LOCATE_POINT_LM=${w_point_lm}"
  echo "LOCATE_GENERATION_MODE=${LOCATE_GENERATION_MODE}"
  echo "LOCATE_BOX_MAX_NEW_TOKENS=${LOCATE_BOX_MAX_NEW_TOKENS}"
  echo "BOX_MATCH_IOU_THRESH=${BOX_MATCH_IOU_THRESH}"
  echo "BOX_NMS_IOU_THRESH=${BOX_NMS_IOU_THRESH}"
  echo "LR=${lr}"
  echo "LOCATE_LR_SCALE=${LOCATE_LR_SCALE}"
  echo "LOCATE_VISION_SCALE=${LOCATE_VISION_SCALE}"
  echo "LOCATE_PROJECTOR_SCALE=${LOCATE_PROJECTOR_SCALE}"
  echo "RESUME_ARG=${resume_arg}"
  echo "===================================================="
  run_train_pose args
}

###############################################################################
# 启动两阶段训练
###############################################################################

echo "================ LocatePose two-stage run ================"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "PYTHON=${PYTHON}"
echo "TORCHRUN=${TORCHRUN}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "RUN_NAME=${RUN_NAME}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "STAGE1_OUTPUT_DIR=${STAGE1_OUTPUT_DIR}"
echo "STAGE2_OUTPUT_DIR=${STAGE2_OUTPUT_DIR}"
echo "RUN_STAGE1=${RUN_STAGE1}"
echo "RUN_STAGE2=${RUN_STAGE2}"
echo "LOCATE_MODEL_PATH=${LOCATE_MODEL_PATH}"
echo "LOCATE_ATTN_IMPLEMENTATION=${LOCATE_ATTN_IMPLEMENTATION}"
echo "LOCATE_MIN_PIXELS=${LOCATE_MIN_PIXELS}"
echo "LOCATE_MAX_PIXELS=${LOCATE_MAX_PIXELS}"
echo "TRAIN_LOG_FILE=${TRAIN_LOG_FILE}"
echo "=========================================================="

last_stage_output=""
if [[ "${RUN_STAGE1}" == "1" ]]; then
  stage1_resume_arg=""
  [[ "${STAGE1_RESUME_FROM_CHECKPOINT}" != "none" ]] && stage1_resume_arg="${STAGE1_RESUME_FROM_CHECKPOINT}"
  run_stage \
    "Stage 1 / freeze Locate GT-box warmup" \
    "${STAGE1_OUTPUT_DIR}" \
    "${STAGE1_TRAIN_DATASETS}" \
    "${STAGE1_DATASET_MIX_WEIGHTS}" \
    "${STAGE1_EPOCHS}" \
    "${STAGE1_BATCH_SIZE}" \
    "${STAGE1_GRAD_ACCUM_STEPS}" \
    "${STAGE1_LR}" \
    "${STAGE1_MAX_STEPS}" \
    "${STAGE1_FREEZE_LOCATE}" \
    "${STAGE1_BOX_SOURCE}" \
    "${STAGE1_BOX_JITTER_SCALE}" \
    "${STAGE1_BOX_JITTER_SHIFT}" \
    "${STAGE1_W_LOCATE_BOX_LM}" \
    "${STAGE1_W_LOCATE_POINT_LM}" \
    "${stage1_resume_arg}"
  last_stage_output="${STAGE1_OUTPUT_DIR}"
else
  echo "Skipping stage 1 because RUN_STAGE1=0"
fi

if [[ "${RUN_STAGE2}" == "1" ]]; then
  stage2_resume_arg="${STAGE2_RESUME_FROM_CHECKPOINT}"
  if [[ "${stage2_resume_arg}" == "none" || -z "${stage2_resume_arg}" ]]; then
    if [[ "${DRY_RUN_DATA}" == "1" ]]; then
      stage2_resume_arg=""
    elif [[ -n "${STAGE2_INIT_CHECKPOINT}" ]]; then
      echo "Preparing stage 2 weight-only init from STAGE2_INIT_CHECKPOINT=${STAGE2_INIT_CHECKPOINT}"
      stage2_resume_arg="$(prepare_weights_only_checkpoint "${STAGE2_INIT_CHECKPOINT}" "${STAGE2_INIT_WEIGHTS_DIR}")"
    elif [[ "${STAGE2_INIT_FROM_STAGE1}" == "1" && -n "${last_stage_output}" ]]; then
      echo "Preparing stage 2 weight-only init from stage 1 output: ${last_stage_output}"
      stage2_resume_arg="$(prepare_weights_only_checkpoint "${last_stage_output}" "${STAGE2_INIT_WEIGHTS_DIR}")"
    else
      stage2_resume_arg=""
      echo "Stage 2 will start from base LocateAnything + newly initialized pose modules."
    fi
  else
    echo "Stage 2 will resume checkpoint state from ${stage2_resume_arg}"
  fi

  run_stage \
    "Stage 2 / closed-loop Locate-box training" \
    "${STAGE2_OUTPUT_DIR}" \
    "${STAGE2_TRAIN_DATASETS}" \
    "${STAGE2_DATASET_MIX_WEIGHTS}" \
    "${STAGE2_EPOCHS}" \
    "${STAGE2_BATCH_SIZE}" \
    "${STAGE2_GRAD_ACCUM_STEPS}" \
    "${STAGE2_LR}" \
    "${STAGE2_MAX_STEPS}" \
    "${STAGE2_FREEZE_LOCATE}" \
    "${STAGE2_BOX_SOURCE}" \
    "${STAGE2_BOX_JITTER_SCALE}" \
    "${STAGE2_BOX_JITTER_SHIFT}" \
    "${STAGE2_W_LOCATE_BOX_LM}" \
    "${STAGE2_W_LOCATE_POINT_LM}" \
    "${stage2_resume_arg}"
  last_stage_output="${STAGE2_OUTPUT_DIR}"
else
  echo "Skipping stage 2 because RUN_STAGE2=0"
fi

if [[ "${MERGE_FINAL_WEIGHTS}" == "1" ]]; then
  echo "MERGE_FINAL_WEIGHTS=1 requested; LocatePose full merge is not enabled in this public script yet." >&2
fi

echo "LocatePose finished. final_stage=${last_stage_output:-none}"
