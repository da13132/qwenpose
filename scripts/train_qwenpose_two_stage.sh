#!/usr/bin/env bash
set -Eeuo pipefail

###############################################################################
# QwenPose 两阶段训练脚本
#
# train_qwenpose_two_stage.sh：
#   先冻结 Qwen 做 Pose warmup，再打开 Qwen LoRA + LM 辅助损失做短程联合微调。
#
# 所有脚本变量都支持通过命令行覆盖，命令行解析发生在默认值赋值之前，因此下面任意
# 大写变量均可用以下三种写法传入：
#   1. --变量名 值，例如：--STAGE1_BATCH_SIZE 4
#   2. --变量名=值，例如：--STAGE2_EPOCHS=1
#   3. --kebab-case-name 值，例如：--stage2-train-datasets coco,refhuman
# 脚本会把 kebab-case / snake_case 统一转换为大写下划线变量名。
#
# Stage 1 / FREEZE_QWEN warmup：
#   - 默认冻结 Qwen 主体与 LoRA，只训练 RGB 视觉分支、Qwen feature refiner 和 PoseHead。
#   - 默认数据：coco。
#   - 默认使用稳定 warmup 组合：per-GPU micro batch size = 4，grad accum = 2。
#   - 默认关闭 LM 辅助损失，只保留 coord + OKS + vis pose 监督。
#   - 默认关闭 stage1 batch trace，减少频繁 I/O 对吞吐的干扰。
#
# Stage 2 / Qwen LoRA + bbox LM：
#   - 默认打开 Qwen LoRA，使 Qwen 主体中的 LoRA 参数可训练。
#   - 默认启用低权重 bbox JSON LM 监督：W_LM=0.05，LM_LOSS_EVERY=2。
#   - 默认数据：coco。
#   - 默认训练 1 epoch，per-GPU micro batch size = 1，grad accum = 8。
#   - 默认 REFHUMAN_MAX_CAPTIONS_PER_INSTANCE="${REFHUMAN_MAX_CAPTIONS_PER_INSTANCE:-1}"。
#
# 示例：
#   scripts/train_qwenpose_two_stage.sh \
#     --CUDA_VISIBLE_DEVICES 0,1 \
#     --NPROC_PER_NODE 2 \
#     --STAGE1_EPOCHS 2 \
#     --STAGE1_BATCH_SIZE 4 \
#     --STAGE2_EPOCHS 1 \
#     --STAGE2_BATCH_SIZE 1
###############################################################################

DEFAULT_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"
SCRIPT_PATH_REL="scripts/$(basename "${BASH_SOURCE[0]}")"

print_usage() {
  cat <<EOF
Usage:
  ${SCRIPT_PATH_REL} [--resume <checkpoint_or_run_dir>] [--VAR VALUE|--VAR=VALUE]...

Options:
  --resume PATH   Resume stage 2 as a true checkpoint resume.
                  For stage 1 use --STAGE1_RESUME_FROM_CHECKPOINT PATH.
                  For stage 2 weight-only init use --STAGE2_INIT_CHECKPOINT PATH.
  --VAR VALUE     Override any script variable. Supports ALL_CAPS, snake_case,
                  and kebab-case spellings for the same option.
                  Examples:
                    --CUDA_VISIBLE_DEVICES 0,1
                    --NPROC_PER_NODE 2
                    --STAGE1_BATCH_SIZE 4
                    --stage2-train-datasets coco
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

# CLI_RESUME_PATH：--resume 的专用别名，只用于 stage 2 的真实断点续训。
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

# PROJECT_ROOT：CLI 覆盖后重新归一化为绝对路径，避免相对路径在 cd 后歧义。
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

# PYTHON：训练环境中的 Python 解释器。优先使用项目内 envs/qwenpose/bin/python，否则回退到当前环境。
DEFAULT_PYTHON="$(resolve_default_python)"
PYTHON="${PYTHON:-${DEFAULT_PYTHON}}"

# TORCHRUN：分布式启动器。优先使用项目内 envs/qwenpose/bin/torchrun，否则回退到 PATH 里的 torchrun。
DEFAULT_TORCHRUN="$(resolve_default_torchrun)"
TORCHRUN="${TORCHRUN:-${DEFAULT_TORCHRUN}}"

cd "${PROJECT_ROOT}"

# PYTHONPATH：优先加载当前项目 src 下的 qwenpose 包，避免误用系统环境中的旧包。
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

resolve_cli_resume_context() {
  "${PYTHON}" - "$1" <<'PY'
from __future__ import annotations

import shlex
import sys
from pathlib import Path

CHECKPOINT_PAYLOAD_NAME = "qwenpose_checkpoint.pt"


def shell_assign(name: str, value: str) -> str:
    return f"{name}={shlex.quote(value)}"


def has_checkpoint_payload(path: Path) -> bool:
    if path.is_file():
        return path.name == CHECKPOINT_PAYLOAD_NAME or path.name.startswith("checkpoint_step_")
    if not path.is_dir():
        return False
    if (path / CHECKPOINT_PAYLOAD_NAME).is_file() or (path / "deepspeed").exists():
        return True
    for pattern in ("checkpoint-*", "checkpoint_step_*.pt"):
        if any(path.glob(pattern)):
            return True
    return False


def has_direct_checkpoint_children(path: Path) -> bool:
    if not path.is_dir():
        return False
    for pattern in ("checkpoint-*", "checkpoint_step_*.pt"):
        if any(path.glob(pattern)):
            return True
    return False


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
stage2_resume = "none"
stage2_init = ""
run_stage1 = "0"
run_stage2 = "1"

if target.is_dir() and (target / "stage2_qwen_lora_lm").is_dir():
    run_dir = target
    stage1_candidate = run_dir / "stage1_freeze_qwen"
    stage2_candidate = run_dir / "stage2_qwen_lora_lm"
    stage1_dir = str(stage1_candidate) if stage1_candidate.exists() else ""
    stage2_dir = str(stage2_candidate)
    if has_checkpoint_payload(stage2_candidate):
        stage2_resume = str(stage2_candidate)
    elif stage1_candidate.exists() and has_checkpoint_payload(stage1_candidate):
        stage2_init = str(stage1_candidate)
elif target.is_dir() and target.name == "stage2_qwen_lora_lm":
    stage2_dir = str(target)
    run_dir = target.parent
    stage1_candidate = run_dir / "stage1_freeze_qwen"
    stage1_dir = str(stage1_candidate) if stage1_candidate.exists() else ""
    stage2_resume = str(target)
elif target.is_dir() and target.name == "stage1_freeze_qwen":
    run_dir = target.parent
    stage1_dir = str(target)
    stage2_dir = str(run_dir / "stage2_qwen_lora_lm")
    stage2_init = str(target)
elif target.parent.name == "stage2_qwen_lora_lm":
    stage2_dir = str(target.parent)
    run_dir = target.parent.parent
    stage1_candidate = run_dir / "stage1_freeze_qwen"
    stage1_dir = str(stage1_candidate) if stage1_candidate.exists() else ""
    stage2_resume = str(target)
elif target.parent.name == "stage1_freeze_qwen":
    run_dir = target.parent.parent
    stage1_dir = str(target.parent)
    stage2_dir = str(run_dir / "stage2_qwen_lora_lm")
    stage2_init = str(target.parent)
elif target.is_dir() and has_direct_checkpoint_children(target):
    run_dir = target
    stage2_dir = str(target)
    stage2_resume = str(target)
elif target.is_dir() and has_checkpoint_payload(target):
    run_dir = target.parent
    stage2_dir = str(run_dir)
    stage2_resume = str(target)
elif target.is_file():
    run_dir = target.parent
    stage2_dir = str(run_dir)
    stage2_resume = str(target)
else:
    raise ValueError(
        "Unsupported resume path layout. Expected a run dir, stage dir, checkpoint dir, or qwenpose_checkpoint.pt file."
    )

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
    if [[ ! -v OUTPUT_DIR ]]; then
      OUTPUT_DIR="${RESUME_RESOLVED_RUN_DIR}"
    fi
    if [[ ! -v OUTPUT_ROOT ]]; then
      OUTPUT_ROOT="$(dirname "${RESUME_RESOLVED_RUN_DIR}")"
    fi
    if [[ ! -v RUN_NAME ]]; then
      RUN_NAME="$(basename "${RESUME_RESOLVED_RUN_DIR}")"
    fi
    if [[ ! -v LOG_DIR ]]; then
      LOG_DIR="${RESUME_RESOLVED_RUN_DIR}/logs"
    fi
    if [[ ! -v TRAIN_LOG_FILE && -n "${RESUME_RESOLVED_APPEND_LOG_FILE:-}" ]]; then
      TRAIN_LOG_FILE="${RESUME_RESOLVED_APPEND_LOG_FILE}"
    fi
    if [[ ! -v RUN_STAGE1 && -n "${RESUME_RESOLVED_RUN_STAGE1:-}" ]]; then
      RUN_STAGE1="${RESUME_RESOLVED_RUN_STAGE1}"
    fi
    if [[ ! -v RUN_STAGE2 && -n "${RESUME_RESOLVED_RUN_STAGE2:-}" ]]; then
      RUN_STAGE2="${RESUME_RESOLVED_RUN_STAGE2}"
    fi
    if [[ ! -v STAGE1_OUTPUT_DIR && -n "${RESUME_RESOLVED_STAGE1_OUTPUT_DIR:-}" ]]; then
      STAGE1_OUTPUT_DIR="${RESUME_RESOLVED_STAGE1_OUTPUT_DIR}"
    fi
    if [[ ! -v STAGE2_OUTPUT_DIR && -n "${RESUME_RESOLVED_STAGE2_OUTPUT_DIR:-}" ]]; then
      STAGE2_OUTPUT_DIR="${RESUME_RESOLVED_STAGE2_OUTPUT_DIR}"
    fi
    if [[ ! -v STAGE2_INIT_CHECKPOINT && -n "${RESUME_RESOLVED_STAGE2_INIT_CHECKPOINT:-}" ]]; then
      STAGE2_INIT_CHECKPOINT="${RESUME_RESOLVED_STAGE2_INIT_CHECKPOINT}"
    fi
    if [[ ! -v STAGE2_RESUME_FROM_CHECKPOINT && -n "${RESUME_RESOLVED_STAGE2_RESUME:-}" ]]; then
      STAGE2_RESUME_FROM_CHECKPOINT="${RESUME_RESOLVED_STAGE2_RESUME}"
    fi
  fi
fi

###############################################################################
# 日志和输出目录参数
###############################################################################

# RUN_TS：本次 run 的时间戳。默认精确到秒；可通过 --RUN_TS 固定，便于复现实验路径。
RUN_TS="${RUN_TS:-$(date +%Y%m%d-%H%M%S)}"

# OUTPUT_ROOT：两阶段训练输出根目录。默认写入 outputs/qwenpose_two_stage_qwen。
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/qwenpose_two_stage_qwen}"

# RUN_NAME：本次 run 的目录名。默认由 RUN_NAME_BASE + RUN_TS 组成。
# 如果传入的 RUN_NAME 已经以 YYYYMMDD-HHMMSS 结尾，则不会重复追加时间戳。
RUN_NAME_BASE="${RUN_NAME:-qwenpose-two-stage-qwen3vl-lora}"
if [[ "${RUN_NAME_BASE}" =~ [0-9]{8}-[0-9]{6}$ ]]; then
  RUN_NAME="${RUN_NAME_BASE}"
else
  RUN_NAME="${RUN_NAME_BASE}-${RUN_TS}"
fi

# OUTPUT_DIR：本次两阶段 run 的总目录。stage1/stage2 默认会放在这个目录下。
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"

# LOG_DIR：总日志目录。默认位于 OUTPUT_DIR/logs。
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"

# LOG_TS：日志文件时间戳。默认复用 RUN_TS，保证同一次运行目录和日志文件一致。
LOG_TS="${LOG_TS:-${RUN_TS}}"

# TRAIN_LOG_FILE：完整 stdout/stderr 日志文件。所有 stage 的输出都会 tee 到这个文件。
TRAIN_LOG_FILE="${TRAIN_LOG_FILE:-${LOG_DIR}/train_${LOG_TS}.log}"

mkdir -p "${LOG_DIR}" "$(dirname "${TRAIN_LOG_FILE}")"
touch "${TRAIN_LOG_FILE}"
exec > >(tee -a "${TRAIN_LOG_FILE}") 2>&1
echo "Logging all stdout/stderr to ${TRAIN_LOG_FILE}"
trap 'status=$?; echo "[ERROR] ${BASH_SOURCE[0]}:${LINENO}: ${BASH_COMMAND} exited with status ${status}" >&2' ERR
trap 'status=$?; echo "========== qwenpose two-stage train exit status ${status} at $(date -Is) =========="; exit ${status}' EXIT

###############################################################################
# 分布式和 DeepSpeed 参数
###############################################################################

# ZERO_STAGE：DeepSpeed ZeRO 策略。zero2 默认推荐；zero3 更省显存；zero3_offload 更省显存但慢；none 用于调试。
ZERO_STAGE="${ZERO_STAGE:-zero2}"

# CUDA_VISIBLE_DEVICES：可见 GPU 列表。默认使用 0,1；单卡可传 --CUDA_VISIBLE_DEVICES 0。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

# NPROC_PER_NODE：本机启动的训练进程数，通常等于可见 GPU 数。
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

# MASTER_ADDR：torch distributed 主节点地址。单机训练保持 127.0.0.1 即可。
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

# MASTER_PORT：torch distributed 通信端口。默认随机选 20001-29999，避免多实验冲突。
export MASTER_PORT="${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}"

# TORCH_FR_BUFFER_SIZE：Torch flight recorder buffer，用于 NCCL/分布式超时诊断。
export TORCH_FR_BUFFER_SIZE="${TORCH_FR_BUFFER_SIZE:-200000}"

# TORCH_NCCL_DUMP_ON_TIMEOUT：NCCL 超时时是否 dump 调试信息；1 表示开启。
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"

# TORCH_NCCL_DESYNC_DEBUG：NCCL desync 调试开关，多卡卡死时有助定位。
export TORCH_NCCL_DESYNC_DEBUG="${TORCH_NCCL_DESYNC_DEBUG:-1}"

# TORCH_DISTRIBUTED_DEBUG：PyTorch 分布式日志级别；DETAIL 更详细但日志更长。
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"

# PYTORCH_CUDA_ALLOC_CONF：CUDA allocator 配置；expandable_segments 可缓解显存碎片化 OOM。
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# WANDB_DISABLED：默认关闭 wandb，避免服务器环境未登录导致训练阻塞。
export WANDB_DISABLED="${WANDB_DISABLED:-true}"

case "${ZERO_STAGE}" in
  zero2) DEFAULT_DEEPSPEED_CONFIG="${PROJECT_ROOT}/scripts/zero2.json" ;;
  zero3) DEFAULT_DEEPSPEED_CONFIG="${PROJECT_ROOT}/scripts/zero3.json" ;;
  zero3_offload) DEFAULT_DEEPSPEED_CONFIG="${PROJECT_ROOT}/scripts/zero3_offload.json" ;;
  none) DEFAULT_DEEPSPEED_CONFIG="" ;;
  *)
    echo "Unsupported ZERO_STAGE=${ZERO_STAGE}. Use zero2, zero3, zero3_offload, or none." >&2
    exit 1
    ;;
esac

# DEEPSPEED_CONFIG：DeepSpeed JSON 配置路径。默认由 ZERO_STAGE 自动选择；ZERO_STAGE=none 时为空。
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${DEFAULT_DEEPSPEED_CONFIG}}"

###############################################################################
# 通用数据参数
###############################################################################

# DATASET_ROOT：所有数据集根目录。Python 训练入口会在该目录下查找 coco/aic/mpii/crowdpose/refhuman 等数据。
DATASET_ROOT="${DATASET_ROOT:-datasets}"

# SPLIT：训练使用的数据 split。默认 train；调试数据解析时可传 --SPLIT val。
SPLIT="${SPLIT:-train}"

# MIXING_STRATEGY：默认多数据集混合策略。interleave 按权重轮转；concat_shuffle 拼接后整体 shuffle。
MIXING_STRATEGY="${MIXING_STRATEGY:-interleave}"

# DATASET_MIX_WEIGHTS：默认数据集采样权重。auto 按样本量自动分配；也可传 coco:4,mpii:1。
DATASET_MIX_WEIGHTS="${DATASET_MIX_WEIGHTS:-auto}"

# MAX_INSTANCES：单张图最多保留的人体实例数。拥挤图超出后截断，以控制显存和匹配成本。
MAX_INSTANCES="${MAX_INSTANCES:-80}"

# MAX_SAMPLES_PER_DATASET：每个数据集最多加载的样本数。空值表示全量；调试可设 1/10/100。
MAX_SAMPLES_PER_DATASET="${MAX_SAMPLES_PER_DATASET:-}"

# RECORD_CACHE_DIR：PoseRecord 标注缓存目录。首次构建后复用缓存可加速后续启动。
RECORD_CACHE_DIR="${RECORD_CACHE_DIR:-.cache/qwenpose_records}"

# DISABLE_RECORD_CACHE：是否禁用标注缓存。1 表示强制重新解析原始标注文件。
DISABLE_RECORD_CACHE="${DISABLE_RECORD_CACHE:-0}"

###############################################################################
# 通用 Qwen 和 Pose 模型参数
###############################################################################

# QWEN_MODEL_PATH：本地 Qwen3-VL 权重目录。正式训练必须存在。
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-weights/Qwen3-VL-4B-Instruct}"
# QWEN_DTYPE：Qwen3-VL 加载精度，可选 bfloat16/float16/float32/auto/none。
QWEN_DTYPE="${QWEN_DTYPE:-bfloat16}"
# QWEN_ATTN_IMPLEMENTATION：Qwen attention 后端。flash_attention_2 更省显存；无 flash-attn 可设 sdpa。
QWEN_ATTN_IMPLEMENTATION="${QWEN_ATTN_IMPLEMENTATION:-flash_attention_2}"
# QWEN_GRADIENT_CHECKPOINTING：是否开启 Qwen 梯度检查点。1 用计算换显存。
QWEN_GRADIENT_CHECKPOINTING="${QWEN_GRADIENT_CHECKPOINTING:-1}"
# QWEN_MIN_PIXELS：可选覆盖 Qwen processor 最小像素数。空值表示默认。
QWEN_MIN_PIXELS="${QWEN_MIN_PIXELS:-}"
# QWEN_MAX_PIXELS：可选覆盖 Qwen processor 最大像素数，用于限制视觉 token 数和显存。
QWEN_MAX_PIXELS="${QWEN_MAX_PIXELS:-}"
# QWEN_FEATURE_SIZE：Qwen image tokens 汇聚后给 PoseHead 的固定空间网格边长。
QWEN_FEATURE_SIZE="${QWEN_FEATURE_SIZE:-64}"
# QWEN_FEATURE_REFINER_LAYERS：Qwen feature refiner 残差细化层数。默认 1，避免 stage1 过度复杂。
QWEN_FEATURE_REFINER_LAYERS="${QWEN_FEATURE_REFINER_LAYERS:-1}"
# QWEN_FEATURE_REFINER_BOTTLENECK_DIM：feature refiner bottleneck 通道数，影响新增参数和计算量。
QWEN_FEATURE_REFINER_BOTTLENECK_DIM="${QWEN_FEATURE_REFINER_BOTTLENECK_DIM:-256}"
# QWEN_FEATURE_REFINER_INIT_SCALE：feature refiner 残差初始强度，较小值有助于 warmup 稳定。
QWEN_FEATURE_REFINER_INIT_SCALE="${QWEN_FEATURE_REFINER_INIT_SCALE:-0.1}"
# HIDDEN_DIM：Pose decoder 隐藏维度，需要能被 DECODER_HEADS 整除。
HIDDEN_DIM="${HIDDEN_DIM:-448}"
# POSE_DECODER_LAYERS：关键点 token decoder 层数。更大表达力更强，但更慢/更占显存。
POSE_DECODER_LAYERS="${POSE_DECODER_LAYERS:-3}"
# REFINEMENT_STEPS：关键点局部残差细化轮数。
REFINEMENT_STEPS="${REFINEMENT_STEPS:-3}"
# BOX_CONDITION_SCALE：传给 PoseHead 的 bbox 条件框放大倍数。
BOX_CONDITION_SCALE="${BOX_CONDITION_SCALE:-1.2}"
# POSE_ROI_SIZE：每个人体 bbox 从 Qwen grid 上采样出的 ROI feature 空间边长。
POSE_ROI_SIZE="${POSE_ROI_SIZE:-16}"
# DECODER_HEADS：Pose decoder 多头注意力头数，必须整除 HIDDEN_DIM。
DECODER_HEADS="${DECODER_HEADS:-8}"
# QWEN_LORA_R：Qwen LLM LoRA rank，越大可训练容量越强，但显存和参数量也更高。
QWEN_LORA_R="${QWEN_LORA_R:-32}"
# QWEN_LORA_ALPHA：Qwen LLM LoRA alpha 缩放系数，通常设为 rank 的 2 倍左右。
QWEN_LORA_ALPHA="${QWEN_LORA_ALPHA:-64}"
# QWEN_LORA_DROPOUT：Qwen LLM LoRA dropout，数据较少时可增加正则。
QWEN_LORA_DROPOUT="${QWEN_LORA_DROPOUT:-0.05}"
# QWEN_VISION_LORA_R：Qwen 视觉塔 LoRA rank，用于视觉编码部分低秩适配。
QWEN_VISION_LORA_R="${QWEN_VISION_LORA_R:-16}"
# QWEN_VISION_LORA_ALPHA：Qwen 视觉塔 LoRA alpha 缩放系数。
QWEN_VISION_LORA_ALPHA="${QWEN_VISION_LORA_ALPHA:-32}"
# QWEN_VISION_LORA_DROPOUT：Qwen 视觉塔 LoRA dropout。
QWEN_VISION_LORA_DROPOUT="${QWEN_VISION_LORA_DROPOUT:-0.05}"

###############################################################################
# 通用训练超参数
###############################################################################

# BATCH_SIZE：可选全局 per-GPU micro batch size。留空时 stage1 默认 4、stage2 默认 1。
BATCH_SIZE="${BATCH_SIZE:-}"
# GRAD_ACCUM_STEPS：全局梯度累积步数。stage 未单独覆盖时继承它。
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
# MAX_STEPS：全局最大 optimizer step 截断。0 表示不截断，按 epoch 完整训练。
MAX_STEPS="${MAX_STEPS:-0}"
# LR：全局基础学习率。Pose/refiner 使用 LR；Qwen LoRA 学习率由 scale 决定。
LR="${LR:-2e-4}"
# QWEN_LORA_LR_SCALE：LLM LoRA 相对 LR 的倍率，实际 lr = LR * QWEN_LORA_LR_SCALE。
QWEN_LORA_LR_SCALE="${QWEN_LORA_LR_SCALE:-0.05}"
# QWEN_VISION_LR_SCALE：视觉塔 LoRA 相对 LR 的倍率，实际 lr = LR * QWEN_VISION_LR_SCALE。
QWEN_VISION_LR_SCALE="${QWEN_VISION_LR_SCALE:-0.02}"
# WEIGHT_DECAY：AdamW 权重衰减。
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
# GRAD_CLIP：梯度裁剪阈值，防止异常 batch 导致梯度爆炸。
GRAD_CLIP="${GRAD_CLIP:-1.0}"
# WARMUP_STEPS：cosine learning-rate schedule 的 warmup optimizer step 数。
WARMUP_STEPS="${WARMUP_STEPS:-100}"
# MIN_LR_RATIO：cosine schedule 最低学习率相对基础学习率的比例。
MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}"
# NUM_WORKERS：DataLoader worker 数。
# 这里默认设为 0，因为本项目的 DataLoader 只搬运已缓存的标注/张量和图片路径，
# 真正的图片解码发生在主进程里的 Qwen processor。多 worker 在这个项目上收益很小，
# 反而在 torchrun + DeepSpeed 下出现过 rank0 长时间卡在取 batch、随后触发 all_reduce
# 超时的情况。需要更高吞吐时仍可手动覆盖。
NUM_WORKERS="${NUM_WORKERS:-0}"
# PREFETCH_FACTOR：每个 DataLoader worker 预取 batch 数；NUM_WORKERS>0 时生效。
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
# DEVICE：训练设备。正式训练通常为 cuda；CPU 仅适合极小调试。
DEVICE="${DEVICE:-cuda}"
# AMP：是否启用 PyTorch autocast AMP。DeepSpeed bf16 训练通常无需额外开启。
AMP="${AMP:-0}"
# LOG_EVERY：每多少个 optimizer step 打印一次训练 loss 日志。
LOG_EVERY="${LOG_EVERY:-10}"
# VISUALIZE_EVERY：全局训练可视化保存间隔。0 表示关闭。
VISUALIZE_EVERY="${VISUALIZE_EVERY:-10}"
# VISUALIZE_MAX_INSTANCES：单张可视化图最多绘制的人体实例数。
VISUALIZE_MAX_INSTANCES="${VISUALIZE_MAX_INSTANCES:-8}"
# SYNC_TIMING：是否在计时点同步 CUDA。1 更准确但更慢，适合 profiling。
SYNC_TIMING="${SYNC_TIMING:-0}"
# SAVE_EVERY：全局 checkpoint 保存间隔，单位为 optimizer step。
SAVE_EVERY="${SAVE_EVERY:-500}"
# SAVE_TOTAL_LIMIT：全局最多保留 checkpoint 数量，超出后自动清理旧 checkpoint。
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
# SEED：全局随机种子，影响数据顺序、caption 抽样、初始化等。
SEED="${SEED:-42}"

###############################################################################
# 通用 loss 参数
###############################################################################

# W_OKS：OKS loss 权重，作为关键点几何定位监督。
W_OKS="${W_OKS:-0.2}"
# W_COORD：关键点坐标回归 loss 权重，pose 定位主监督。
W_COORD="${W_COORD:-5.0}"
# W_VIS：关键点可见性/有效性 BCE loss 权重。
W_VIS="${W_VIS:-0.05}"
# W_HARD_JOINT：hard keypoint mining loss 权重。保留给后续实验，默认关闭。
W_HARD_JOINT="${W_HARD_JOINT:-0}"
# HARD_JOINT_FRACTION：hard mining 选择的可见关键点比例。
HARD_JOINT_FRACTION="${HARD_JOINT_FRACTION:-0.2}"
# W_LM：全局 LM bbox 监督权重。stage1 默认强制为 0，stage2 默认继承它。
W_LM="${W_LM:-0.05}"
# LM_LOSS_EVERY：全局 LM loss 计算频率，单位为 micro batch。stage2 默认隔步计算。
LM_LOSS_EVERY="${LM_LOSS_EVERY:-2}"
# LM_MAX_ANSWER_INSTANCES：ALL_POSE LM answer 中最多监督的人体实例数。
LM_MAX_ANSWER_INSTANCES="${LM_MAX_ANSWER_INSTANCES:-10}"
# BOX_JITTER_SCALE：训练时对条件框宽高的随机缩放扰动强度。
BOX_JITTER_SCALE="${BOX_JITTER_SCALE:-0.0}"
# BOX_JITTER_SHIFT：训练时对条件框中心的随机平移扰动强度。
BOX_JITTER_SHIFT="${BOX_JITTER_SHIFT:-0.0}"

###############################################################################
# 通用开关参数
###############################################################################

# DRY_RUN_DATA：1 表示只构建数据集并预览 batch，不进入训练。
DRY_RUN_DATA="${DRY_RUN_DATA:-0}"
# PROGRESS_BAR：1 显示 tqdm 进度条；0 关闭动态进度条。
PROGRESS_BAR="${PROGRESS_BAR:-1}"
# DISABLE_REFINEMENT：1 表示关闭 decoder 后关键点 refinement 分支。
DISABLE_REFINEMENT="${DISABLE_REFINEMENT:-0}"
# DISABLE_HOMOGENEOUS_BATCHES：1 表示允许一个 batch 混合多个数据集；默认 0 表示单 batch 单数据集。
DISABLE_HOMOGENEOUS_BATCHES="${DISABLE_HOMOGENEOUS_BATCHES:-0}"
# DISABLE_BATCH_TRACE：1 表示关闭每个 batch 的 JSONL trace。
DISABLE_BATCH_TRACE="${DISABLE_BATCH_TRACE:-0}"

###############################################################################
# Stage 独立参数
###############################################################################

# RUN_STAGE1：是否执行 stage1。1 执行；0 跳过。
RUN_STAGE1="${RUN_STAGE1:-1}"
# RUN_STAGE2：是否执行 stage2。1 执行；0 跳过。
RUN_STAGE2="${RUN_STAGE2:-1}"
# STAGE1_OUTPUT_DIR：stage1 checkpoint、可视化和 batch trace 的输出目录。
STAGE1_OUTPUT_DIR="${STAGE1_OUTPUT_DIR:-${OUTPUT_DIR}/stage1_freeze_qwen}"
# STAGE2_OUTPUT_DIR：stage2 checkpoint、可视化和 batch trace 的输出目录。
STAGE2_OUTPUT_DIR="${STAGE2_OUTPUT_DIR:-${OUTPUT_DIR}/stage2_qwen_lora_lm}"
# STAGE2_INIT_WEIGHTS_DIR：自动生成的 stage2 weight-only init checkpoint 存放目录。
STAGE2_INIT_WEIGHTS_DIR="${STAGE2_INIT_WEIGHTS_DIR:-${OUTPUT_DIR}/stage2_init_weights}"
# STAGE1_TRAIN_DATASETS：stage1 训练数据集列表，逗号分隔。 # coco,mpii,crowdpose,aic,refhuman #
STAGE1_TRAIN_DATASETS="${STAGE1_TRAIN_DATASETS:-coco}"
# STAGE2_TRAIN_DATASETS：stage2 训练数据集列表，逗号分隔。
STAGE2_TRAIN_DATASETS="${STAGE2_TRAIN_DATASETS:-coco}"
# STAGE1_MIXING_STRATEGY：stage1 多数据集混合策略；默认继承 MIXING_STRATEGY。
STAGE1_MIXING_STRATEGY="${STAGE1_MIXING_STRATEGY:-${MIXING_STRATEGY}}"
# STAGE2_MIXING_STRATEGY：stage2 多数据集混合策略；默认继承 MIXING_STRATEGY。
STAGE2_MIXING_STRATEGY="${STAGE2_MIXING_STRATEGY:-${MIXING_STRATEGY}}"
# STAGE1_DATASET_MIX_WEIGHTS：stage1 数据集采样权重；默认继承 DATASET_MIX_WEIGHTS。
STAGE1_DATASET_MIX_WEIGHTS="${STAGE1_DATASET_MIX_WEIGHTS:-${DATASET_MIX_WEIGHTS}}"
# STAGE2_DATASET_MIX_WEIGHTS：stage2 数据集采样权重；默认继承 DATASET_MIX_WEIGHTS。
STAGE2_DATASET_MIX_WEIGHTS="${STAGE2_DATASET_MIX_WEIGHTS:-${DATASET_MIX_WEIGHTS}}"
# STAGE1_MAX_SAMPLES_PER_DATASET：stage1 每个数据集最大样本数。空值表示全量。
STAGE1_MAX_SAMPLES_PER_DATASET="${STAGE1_MAX_SAMPLES_PER_DATASET:-${MAX_SAMPLES_PER_DATASET}}"
# STAGE2_MAX_SAMPLES_PER_DATASET：stage2 每个数据集最大样本数。空值表示全量。
STAGE2_MAX_SAMPLES_PER_DATASET="${STAGE2_MAX_SAMPLES_PER_DATASET:-${MAX_SAMPLES_PER_DATASET}}"
# STAGE1_REFHUMAN_MAX_CAPTIONS_PER_INSTANCE：stage1 RefHuman 单实例最多 caption 数。
STAGE1_REFHUMAN_MAX_CAPTIONS_PER_INSTANCE="${STAGE1_REFHUMAN_MAX_CAPTIONS_PER_INSTANCE:-${REFHUMAN_MAX_CAPTIONS_PER_INSTANCE:-1}}"
# STAGE2_REFHUMAN_MAX_CAPTIONS_PER_INSTANCE：stage2 RefHuman 单实例最多 caption 数。默认 1。
STAGE2_REFHUMAN_MAX_CAPTIONS_PER_INSTANCE="${STAGE2_REFHUMAN_MAX_CAPTIONS_PER_INSTANCE:-${REFHUMAN_MAX_CAPTIONS_PER_INSTANCE:-1}}"
# STAGE1_EPOCHS：stage1 训练 epoch 数。默认 2，用于 RGB 视觉分支 pose warmup。
STAGE1_EPOCHS="${STAGE1_EPOCHS:-2}"
# STAGE2_EPOCHS：stage2 训练 epoch 数。默认 1，只做 LoRA + bbox LM 适配。
STAGE2_EPOCHS="${STAGE2_EPOCHS:-1}"
# STAGE1_BATCH_SIZE：stage1 每张 GPU 的 micro batch size。默认 4。
STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-${BATCH_SIZE:-4}}"
# STAGE2_BATCH_SIZE：stage2 每张 GPU 的 micro batch size。默认 1。
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-${BATCH_SIZE:-1}}"
# STAGE1_GRAD_ACCUM_STEPS：stage1 梯度累积步数。默认 2。
STAGE1_GRAD_ACCUM_STEPS="${STAGE1_GRAD_ACCUM_STEPS:-2}"
# STAGE2_GRAD_ACCUM_STEPS：stage2 梯度累积步数。默认 8。
STAGE2_GRAD_ACCUM_STEPS="${STAGE2_GRAD_ACCUM_STEPS:-8}"
# STAGE1_MAX_STEPS：stage1 最大 optimizer step 截断。默认 7000。
STAGE1_MAX_STEPS="${STAGE1_MAX_STEPS:-7000}"
# STAGE2_MAX_STEPS：stage2 最大 optimizer step 截断。默认 3000。
STAGE2_MAX_STEPS="${STAGE2_MAX_STEPS:-3000}"
# STAGE1_FREEZE_QWEN：stage1 是否冻结 Qwen 主体/LoRA。默认 1。
STAGE1_FREEZE_QWEN="${STAGE1_FREEZE_QWEN:-1}"
# STAGE2_FREEZE_QWEN：stage2 是否冻结 Qwen 主体/LoRA。默认 0。
STAGE2_FREEZE_QWEN="${STAGE2_FREEZE_QWEN:-0}"
# STAGE1_W_LM：stage1 LM 辅助损失权重。默认 0。
STAGE1_W_LM="${STAGE1_W_LM:-0}"
# STAGE2_W_LM：stage2 LM 辅助损失权重。默认继承 W_LM。
STAGE2_W_LM="${STAGE2_W_LM:-${W_LM}}"
# STAGE1_LM_LOSS_EVERY：stage1 LM loss 计算频率。默认 0。
STAGE1_LM_LOSS_EVERY="${STAGE1_LM_LOSS_EVERY:-0}"
# STAGE2_LM_LOSS_EVERY：stage2 LM loss 计算频率。默认继承 LM_LOSS_EVERY。
STAGE2_LM_LOSS_EVERY="${STAGE2_LM_LOSS_EVERY:-${LM_LOSS_EVERY}}"
# STAGE1_LR：stage1 基础学习率。默认继承 LR。
STAGE1_LR="${STAGE1_LR:-${LR}}"
# STAGE2_LR：stage2 基础学习率。默认 1e-4。
STAGE2_LR="${STAGE2_LR:-1e-4}"
# STAGE1_QWEN_LORA_LR_SCALE：stage1 LLM LoRA lr scale；冻结 Qwen 时通常不生效。
STAGE1_QWEN_LORA_LR_SCALE="${STAGE1_QWEN_LORA_LR_SCALE:-${QWEN_LORA_LR_SCALE}}"
# STAGE2_QWEN_LORA_LR_SCALE：stage2 LLM LoRA lr scale。
STAGE2_QWEN_LORA_LR_SCALE="${STAGE2_QWEN_LORA_LR_SCALE:-${QWEN_LORA_LR_SCALE}}"
# STAGE1_QWEN_VISION_LR_SCALE：stage1 vision LoRA lr scale；冻结 Qwen 时通常不生效。
STAGE1_QWEN_VISION_LR_SCALE="${STAGE1_QWEN_VISION_LR_SCALE:-${QWEN_VISION_LR_SCALE}}"
# STAGE2_QWEN_VISION_LR_SCALE：stage2 vision LoRA lr scale。
STAGE2_QWEN_VISION_LR_SCALE="${STAGE2_QWEN_VISION_LR_SCALE:-${QWEN_VISION_LR_SCALE}}"
# STAGE1_WARMUP_STEPS：stage1 warmup step 数。
STAGE1_WARMUP_STEPS="${STAGE1_WARMUP_STEPS:-${WARMUP_STEPS}}"
# STAGE2_WARMUP_STEPS：stage2 warmup step 数。
STAGE2_WARMUP_STEPS="${STAGE2_WARMUP_STEPS:-${WARMUP_STEPS}}"
# STAGE1_MIN_LR_RATIO：stage1 cosine scheduler 最低学习率比例。
STAGE1_MIN_LR_RATIO="${STAGE1_MIN_LR_RATIO:-${MIN_LR_RATIO}}"
# STAGE2_MIN_LR_RATIO：stage2 cosine scheduler 最低学习率比例。
STAGE2_MIN_LR_RATIO="${STAGE2_MIN_LR_RATIO:-${MIN_LR_RATIO}}"
# STAGE1_NUM_WORKERS：stage1 DataLoader worker 数。
STAGE1_NUM_WORKERS="${STAGE1_NUM_WORKERS:-${NUM_WORKERS}}"
# STAGE2_NUM_WORKERS：stage2 DataLoader worker 数。
STAGE2_NUM_WORKERS="${STAGE2_NUM_WORKERS:-${NUM_WORKERS}}"
# STAGE1_PREFETCH_FACTOR：stage1 DataLoader prefetch factor。
STAGE1_PREFETCH_FACTOR="${STAGE1_PREFETCH_FACTOR:-${PREFETCH_FACTOR}}"
# STAGE2_PREFETCH_FACTOR：stage2 DataLoader prefetch factor。
STAGE2_PREFETCH_FACTOR="${STAGE2_PREFETCH_FACTOR:-${PREFETCH_FACTOR}}"
# STAGE1_SAVE_EVERY：stage1 checkpoint 保存间隔。默认放宽到 1000，减少 warmup 阶段频繁落盘。
STAGE1_SAVE_EVERY="${STAGE1_SAVE_EVERY:-1000}"
# STAGE2_SAVE_EVERY：stage2 checkpoint 保存间隔。
STAGE2_SAVE_EVERY="${STAGE2_SAVE_EVERY:-${SAVE_EVERY}}"
# STAGE1_SAVE_TOTAL_LIMIT：stage1 最多保留 checkpoint 数量。
STAGE1_SAVE_TOTAL_LIMIT="${STAGE1_SAVE_TOTAL_LIMIT:-${SAVE_TOTAL_LIMIT}}"
# STAGE2_SAVE_TOTAL_LIMIT：stage2 最多保留 checkpoint 数量。
STAGE2_SAVE_TOTAL_LIMIT="${STAGE2_SAVE_TOTAL_LIMIT:-${SAVE_TOTAL_LIMIT}}"
# STAGE1_VISUALIZE_EVERY：stage1 可视化保存间隔。默认 10，便于观察 warmup 变化。
STAGE1_VISUALIZE_EVERY="${STAGE1_VISUALIZE_EVERY:-10}"
# STAGE2_VISUALIZE_EVERY：stage2 可视化保存间隔。0 表示关闭。
STAGE2_VISUALIZE_EVERY="${STAGE2_VISUALIZE_EVERY:-${VISUALIZE_EVERY}}"
# STAGE1_DISABLE_BATCH_TRACE：stage1 是否关闭每 batch JSONL trace。默认 1，优先吞吐。
STAGE1_DISABLE_BATCH_TRACE="${STAGE1_DISABLE_BATCH_TRACE:-1}"
# STAGE2_DISABLE_BATCH_TRACE：stage2 是否关闭每 batch JSONL trace。默认继承全局开关。
STAGE2_DISABLE_BATCH_TRACE="${STAGE2_DISABLE_BATCH_TRACE:-${DISABLE_BATCH_TRACE}}"
# STAGE1_SEED：stage1 随机种子。
STAGE1_SEED="${STAGE1_SEED:-${SEED}}"
# STAGE2_SEED：stage2 随机种子。
STAGE2_SEED="${STAGE2_SEED:-${SEED}}"
# STAGE1_RESUME_FROM_CHECKPOINT：stage1 真实断点续训路径。none 表示不续训。
STAGE1_RESUME_FROM_CHECKPOINT="${STAGE1_RESUME_FROM_CHECKPOINT:-none}"
# STAGE2_RESUME_FROM_CHECKPOINT：stage2 真实断点续训路径。none 表示不续训。
STAGE2_RESUME_FROM_CHECKPOINT="${STAGE2_RESUME_FROM_CHECKPOINT:-none}"
# STAGE2_INIT_CHECKPOINT：stage2 显式 weight-only 初始化来源，只加载模型权重，不恢复训练游标。
STAGE2_INIT_CHECKPOINT="${STAGE2_INIT_CHECKPOINT:-}"
# STAGE2_INIT_FROM_STAGE1：stage2 是否默认从 stage1 输出生成 weight-only init。1 表示是。
STAGE2_INIT_FROM_STAGE1="${STAGE2_INIT_FROM_STAGE1:-1}"
# MERGE_FINAL_WEIGHTS：stage2 正常结束后是否自动合并 LoRA 为完整发布权重。
MERGE_FINAL_WEIGHTS="${MERGE_FINAL_WEIGHTS:-0}"
# MERGED_WEIGHTS_ROOT：自动合并后的完整权重输出根目录。
MERGED_WEIGHTS_ROOT="${MERGED_WEIGHTS_ROOT:-weights}"
# MERGED_WEIGHTS_DIR：自动合并后的完整权重目录，默认带 run 名和当前时间戳。
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
# 参数校验和工具函数
###############################################################################

require_positive_int() {
  local name="$1"
  local value="$2"
  if ! [[ "${value}" =~ ^[0-9]+$ ]] || (( value <= 0 )); then
    echo "${name} must be a positive integer, got: ${value}" >&2
    exit 1
  fi
}

require_nonnegative_int() {
  local name="$1"
  local value="$2"
  if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
    echo "${name} must be a non-negative integer, got: ${value}" >&2
    exit 1
  fi
}

require_bool() {
  local name="$1"
  local value="$2"
  if [[ "${value}" != "0" && "${value}" != "1" ]]; then
    echo "${name} must be 0 or 1, got: ${value}" >&2
    exit 1
  fi
}

resume_target_has_checkpoint() {
  local resume_path="$1"
  if [[ -f "${resume_path}" ]]; then
    return 0
  fi
  if [[ ! -d "${resume_path}" ]]; then
    return 1
  fi
  if [[ -f "${resume_path}/qwenpose_checkpoint.pt" || -e "${resume_path}/deepspeed" ]]; then
    return 0
  fi
  find "${resume_path}" -maxdepth 1 \( -name 'checkpoint-*' -o -name 'checkpoint_step_*.pt' \) -print -quit | grep -q .
}

check_optional_checkpoint() {
  local name="$1"
  local path="$2"
  if [[ "${path}" == "none" || -z "${path}" ]]; then
    return 0
  fi
  if [[ ! -e "${path}" ]]; then
    echo "${name} path does not exist: ${path}" >&2
    exit 1
  fi
  if ! resume_target_has_checkpoint "${path}"; then
    echo "${name} has no checkpoint payload: ${path}" >&2
    exit 1
  fi
}

VISIBLE_GPU_COUNT="$("${PYTHON}" - <<'PY'
import os
visible = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
print(len(visible) or 1)
PY
)"

require_positive_int NPROC_PER_NODE "${NPROC_PER_NODE}"
require_positive_int VISIBLE_GPU_COUNT "${VISIBLE_GPU_COUNT}"
require_bool RUN_STAGE1 "${RUN_STAGE1}"
require_bool RUN_STAGE2 "${RUN_STAGE2}"
require_bool STAGE1_FREEZE_QWEN "${STAGE1_FREEZE_QWEN}"
require_bool STAGE2_FREEZE_QWEN "${STAGE2_FREEZE_QWEN}"
require_bool STAGE2_INIT_FROM_STAGE1 "${STAGE2_INIT_FROM_STAGE1}"
require_bool MERGE_FINAL_WEIGHTS "${MERGE_FINAL_WEIGHTS}"
require_bool QWEN_GRADIENT_CHECKPOINTING "${QWEN_GRADIENT_CHECKPOINTING}"
require_bool AMP "${AMP}"
require_bool DRY_RUN_DATA "${DRY_RUN_DATA}"
require_bool PROGRESS_BAR "${PROGRESS_BAR}"
require_bool SYNC_TIMING "${SYNC_TIMING}"
require_bool DISABLE_RECORD_CACHE "${DISABLE_RECORD_CACHE}"
require_bool DISABLE_REFINEMENT "${DISABLE_REFINEMENT}"
require_bool DISABLE_HOMOGENEOUS_BATCHES "${DISABLE_HOMOGENEOUS_BATCHES}"
require_bool DISABLE_BATCH_TRACE "${DISABLE_BATCH_TRACE}"
require_bool STAGE1_DISABLE_BATCH_TRACE "${STAGE1_DISABLE_BATCH_TRACE}"
require_bool STAGE2_DISABLE_BATCH_TRACE "${STAGE2_DISABLE_BATCH_TRACE}"

if (( NPROC_PER_NODE > VISIBLE_GPU_COUNT )); then
  echo "NPROC_PER_NODE=${NPROC_PER_NODE} exceeds visible GPUs (${CUDA_VISIBLE_DEVICES}; count=${VISIBLE_GPU_COUNT})." >&2
  exit 1
fi

for spec in \
  "STAGE1_BATCH_SIZE:${STAGE1_BATCH_SIZE}" \
  "STAGE2_BATCH_SIZE:${STAGE2_BATCH_SIZE}" \
  "STAGE1_GRAD_ACCUM_STEPS:${STAGE1_GRAD_ACCUM_STEPS}" \
  "STAGE2_GRAD_ACCUM_STEPS:${STAGE2_GRAD_ACCUM_STEPS}" \
  "STAGE1_EPOCHS:${STAGE1_EPOCHS}" \
  "STAGE2_EPOCHS:${STAGE2_EPOCHS}" \
  "REFINEMENT_STEPS:${REFINEMENT_STEPS}" \
  "POSE_ROI_SIZE:${POSE_ROI_SIZE}" \
  "STAGE1_SAVE_EVERY:${STAGE1_SAVE_EVERY}" \
  "STAGE2_SAVE_EVERY:${STAGE2_SAVE_EVERY}" \
  "STAGE1_SAVE_TOTAL_LIMIT:${STAGE1_SAVE_TOTAL_LIMIT}" \
  "STAGE2_SAVE_TOTAL_LIMIT:${STAGE2_SAVE_TOTAL_LIMIT}" \
  "VISUALIZE_MAX_INSTANCES:${VISUALIZE_MAX_INSTANCES}" \
  "LM_MAX_ANSWER_INSTANCES:${LM_MAX_ANSWER_INSTANCES}"; do
  require_positive_int "${spec%%:*}" "${spec#*:}"
done

for spec in \
  "STAGE1_MAX_STEPS:${STAGE1_MAX_STEPS}" \
  "STAGE2_MAX_STEPS:${STAGE2_MAX_STEPS}" \
  "STAGE1_NUM_WORKERS:${STAGE1_NUM_WORKERS}" \
  "STAGE2_NUM_WORKERS:${STAGE2_NUM_WORKERS}" \
  "STAGE1_VISUALIZE_EVERY:${STAGE1_VISUALIZE_EVERY}" \
  "STAGE2_VISUALIZE_EVERY:${STAGE2_VISUALIZE_EVERY}" \
  "STAGE1_LM_LOSS_EVERY:${STAGE1_LM_LOSS_EVERY}" \
  "STAGE2_LM_LOSS_EVERY:${STAGE2_LM_LOSS_EVERY}" \
  "STAGE1_REFHUMAN_MAX_CAPTIONS_PER_INSTANCE:${STAGE1_REFHUMAN_MAX_CAPTIONS_PER_INSTANCE}" \
  "STAGE2_REFHUMAN_MAX_CAPTIONS_PER_INSTANCE:${STAGE2_REFHUMAN_MAX_CAPTIONS_PER_INSTANCE}"; do
  require_nonnegative_int "${spec%%:*}" "${spec#*:}"
done

if [[ -n "${QWEN_MIN_PIXELS}" ]]; then
  require_positive_int QWEN_MIN_PIXELS "${QWEN_MIN_PIXELS}"
fi
if [[ -n "${QWEN_MAX_PIXELS}" ]]; then
  require_positive_int QWEN_MAX_PIXELS "${QWEN_MAX_PIXELS}"
fi
if [[ -n "${QWEN_MIN_PIXELS}" && -n "${QWEN_MAX_PIXELS}" ]] && (( QWEN_MAX_PIXELS < QWEN_MIN_PIXELS )); then
  echo "QWEN_MAX_PIXELS=${QWEN_MAX_PIXELS} must be >= QWEN_MIN_PIXELS=${QWEN_MIN_PIXELS}." >&2
  exit 1
fi

if [[ "${DEVICE}" != "cuda" && "${ZERO_STAGE}" != "none" ]]; then
  echo "DEVICE=${DEVICE} cannot use DeepSpeed ${ZERO_STAGE}. Use ZERO_STAGE=none for CPU debugging." >&2
  exit 1
fi
if [[ ! -e "${QWEN_MODEL_PATH}" ]]; then
  echo "QWEN_MODEL_PATH not found: ${QWEN_MODEL_PATH}" >&2
  exit 1
fi
if [[ -n "${DEEPSPEED_CONFIG}" && ! -f "${DEEPSPEED_CONFIG}" ]]; then
  echo "DEEPSPEED_CONFIG not found: ${DEEPSPEED_CONFIG}" >&2
  exit 1
fi

check_optional_checkpoint STAGE1_RESUME_FROM_CHECKPOINT "${STAGE1_RESUME_FROM_CHECKPOINT}"
check_optional_checkpoint STAGE2_RESUME_FROM_CHECKPOINT "${STAGE2_RESUME_FROM_CHECKPOINT}"
check_optional_checkpoint STAGE2_INIT_CHECKPOINT "${STAGE2_INIT_CHECKPOINT}"

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
source = Path(sys.argv[1])
dest = Path(sys.argv[2])

def checkpoint_step(path: Path) -> int | None:
    if path.is_dir():
        match = re.search(r"checkpoint-(\d+)$", path.name)
    else:
        match = re.search(r"checkpoint_step_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else None

def resolve(path: Path) -> Path:
    if path.is_file():
        return path
    if (path / CHECKPOINT_PAYLOAD_NAME).is_file():
        return path
    candidates: list[tuple[int, Path]] = []
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
    json.dump(
        {
            "step": 0,
            "checkpoint": str(out),
            "payload": CHECKPOINT_PAYLOAD_NAME,
            "deepspeed_tag": None,
            "training_state": None,
            "stage2_weight_only_init_from": str(resolved),
        },
        f,
        indent=2,
        ensure_ascii=False,
    )
    f.write("\n")

adapter_src = resolved / "qwen_lora_adapter" if resolved.is_dir() else None
if adapter_src is not None and adapter_src.is_dir():
    shutil.copytree(adapter_src, out / "qwen_lora_adapter", dirs_exist_ok=True)

print(out)
PY
}

merge_full_weights() {
  local checkpoint_source="$1"
  local merged_dir="$2"
  if [[ "${MERGE_FINAL_WEIGHTS}" != "1" ]]; then
    return 0
  fi
  if [[ -z "${checkpoint_source}" || -z "${merged_dir}" ]]; then
    echo "Invalid merge_full_weights arguments: checkpoint_source=${checkpoint_source}, merged_dir=${merged_dir}" >&2
    exit 1
  fi
  mkdir -p "$(dirname "${merged_dir}")"
  echo "Merging final checkpoint from ${checkpoint_source} into full weights: ${merged_dir}"
  "${PYTHON}" -m qwenpose.merge_full_weights \
    --checkpoint "${checkpoint_source}" \
    --base_model_path "${QWEN_MODEL_PATH}" \
    --output_dir "${merged_dir}" \
    --qwen_dtype "${QWEN_DTYPE}" \
    --qwen_attn_implementation "${QWEN_ATTN_IMPLEMENTATION}" \
    --overwrite
}

run_train_pose() {
  if [[ "${ZERO_STAGE}" == "none" ]]; then
    "${PYTHON}" -m qwenpose.train_pose "$@"
  elif [[ -n "${TORCHRUN}" ]]; then
    "${TORCHRUN}" \
      --nproc_per_node "${NPROC_PER_NODE}" \
      --master_addr "${MASTER_ADDR}" \
      --master_port "${MASTER_PORT}" \
      "${PROJECT_ROOT}/src/qwenpose/train_pose.py" \
      "$@"
  else
    "${PYTHON}" -m torch.distributed.run \
      --nproc_per_node "${NPROC_PER_NODE}" \
      --master_addr "${MASTER_ADDR}" \
      --master_port "${MASTER_PORT}" \
      "${PROJECT_ROOT}/src/qwenpose/train_pose.py" \
      "$@"
  fi
}

run_stage() {
  local stage_label="$1"
  local output_dir="$2"
  local datasets="$3"
  local batch_size="$4"
  local grad_accum_steps="$5"
  local epochs="$6"
  local max_steps="$7"
  local freeze_qwen="$8"
  local w_lm="$9"
  local lm_loss_every="${10}"
  local refhuman_max_captions="${11}"
  local mixing_strategy="${12}"
  local dataset_mix_weights="${13}"
  local max_samples_per_dataset="${14}"
  local lr="${15}"
  local qwen_lora_lr_scale="${16}"
  local qwen_vision_lr_scale="${17}"
  local warmup_steps="${18}"
  local min_lr_ratio="${19}"
  local num_workers="${20}"
  local prefetch_factor="${21}"
  local save_every="${22}"
  local save_total_limit="${23}"
  local visualize_every="${24}"
  local seed="${25}"
  local resume_arg="${26}"
  local disable_batch_trace="${27}"

  local effective_batch=$((NPROC_PER_NODE * batch_size * grad_accum_steps))
  local args=(
    --dataset_root "${DATASET_ROOT}"
    --datasets "${datasets}"
    --split "${SPLIT}"
    --mixing_strategy "${mixing_strategy}"
    --dataset_mix_weights "${dataset_mix_weights}"
    --max_instances "${MAX_INSTANCES}"
    --refhuman_max_captions_per_instance "${refhuman_max_captions}"
    --record_cache_dir "${RECORD_CACHE_DIR}"

    --hidden_dim "${HIDDEN_DIM}"
    --backbone "qwen3vl"
    --qwen_model_path "${QWEN_MODEL_PATH}"
    --qwen_dtype "${QWEN_DTYPE}"
    --qwen_attn_implementation "${QWEN_ATTN_IMPLEMENTATION}"
    --qwen_feature_size "${QWEN_FEATURE_SIZE}"
    --qwen_feature_refiner_layers "${QWEN_FEATURE_REFINER_LAYERS}"
    --qwen_feature_refiner_bottleneck_dim "${QWEN_FEATURE_REFINER_BOTTLENECK_DIM}"
    --qwen_feature_refiner_init_scale "${QWEN_FEATURE_REFINER_INIT_SCALE}"
    --qwen_lora_r "${QWEN_LORA_R}"
    --qwen_lora_alpha "${QWEN_LORA_ALPHA}"
    --qwen_lora_dropout "${QWEN_LORA_DROPOUT}"
    --qwen_vision_lora_r "${QWEN_VISION_LORA_R}"
    --qwen_vision_lora_alpha "${QWEN_VISION_LORA_ALPHA}"
    --qwen_vision_lora_dropout "${QWEN_VISION_LORA_DROPOUT}"
    --pose_decoder_layers "${POSE_DECODER_LAYERS}"
    --refinement_steps "${REFINEMENT_STEPS}"
    --box_condition_scale "${BOX_CONDITION_SCALE}"
    --pose_roi_size "${POSE_ROI_SIZE}"
    --decoder_heads "${DECODER_HEADS}"

    --output_dir "${output_dir}"
    --batch_size "${batch_size}"
    --grad_accum_steps "${grad_accum_steps}"
    --epochs "${epochs}"
    --max_steps "${max_steps}"
    --lr "${lr}"
    --qwen_lora_lr_scale "${qwen_lora_lr_scale}"
    --qwen_vision_lr_scale "${qwen_vision_lr_scale}"
    --weight_decay "${WEIGHT_DECAY}"
    --grad_clip "${GRAD_CLIP}"
    --warmup_steps "${warmup_steps}"
    --min_lr_ratio "${min_lr_ratio}"
    --num_workers "${num_workers}"
    --prefetch_factor "${prefetch_factor}"
    --device "${DEVICE}"
    --log_every "${LOG_EVERY}"
    --visualize_every "${visualize_every}"
    --visualize_max_instances "${VISUALIZE_MAX_INSTANCES}"
    --save_every "${save_every}"
    --save_total_limit "${save_total_limit}"
    --seed "${seed}"

    --w_oks "${W_OKS}"
    --w_coord "${W_COORD}"
    --w_vis "${W_VIS}"
    --w_hard_joint "${W_HARD_JOINT}"
    --hard_joint_fraction "${HARD_JOINT_FRACTION}"
    --w_lm "${w_lm}"
    --lm_loss_every "${lm_loss_every}"
    --lm_max_answer_instances "${LM_MAX_ANSWER_INSTANCES}"
    --box_jitter_scale "${BOX_JITTER_SCALE}"
    --box_jitter_shift "${BOX_JITTER_SHIFT}"
  )

  if [[ -n "${max_samples_per_dataset}" ]]; then
    args+=(--max_samples_per_dataset "${max_samples_per_dataset}")
  fi
  if [[ "${resume_arg}" != "none" && -n "${resume_arg}" ]]; then
    args+=(--resume_from_checkpoint "${resume_arg}")
  fi
  if [[ -n "${QWEN_MIN_PIXELS}" ]]; then
    args+=(--qwen_min_pixels "${QWEN_MIN_PIXELS}")
  fi
  if [[ -n "${QWEN_MAX_PIXELS}" ]]; then
    args+=(--qwen_max_pixels "${QWEN_MAX_PIXELS}")
  fi
  if [[ "${AMP}" == "1" ]]; then
    args+=(--amp)
  fi
  if [[ "${QWEN_GRADIENT_CHECKPOINTING}" == "1" ]]; then
    args+=(--qwen_gradient_checkpointing)
  fi
  if [[ "${freeze_qwen}" == "1" ]]; then
    args+=(--freeze_qwen)
  fi
  if [[ "${DRY_RUN_DATA}" == "1" ]]; then
    args+=(--dry_run_data)
  fi
  if [[ "${PROGRESS_BAR}" == "0" ]]; then
    args+=(--disable_progress)
  fi
  if [[ "${SYNC_TIMING}" == "1" ]]; then
    args+=(--sync_timing)
  fi
  if [[ "${DISABLE_RECORD_CACHE}" == "1" ]]; then
    args+=(--disable_record_cache)
  fi
  if [[ "${DISABLE_REFINEMENT}" == "1" ]]; then
    args+=(--disable_refinement)
  fi
  if [[ "${DISABLE_HOMOGENEOUS_BATCHES}" == "1" ]]; then
    args+=(--disable_homogeneous_batches)
  fi
  if [[ "${disable_batch_trace}" == "1" ]]; then
    args+=(--disable_batch_trace)
  fi
  if [[ -n "${DEEPSPEED_CONFIG}" ]]; then
    args+=(--deepspeed_config "${DEEPSPEED_CONFIG}")
  fi

  echo "================ QwenPose ${stage_label} 配置 ================"
  echo "PROJECT_ROOT=${PROJECT_ROOT}"
  echo "PYTHON=${PYTHON}"
  echo "TORCHRUN=${TORCHRUN:-${PYTHON} -m torch.distributed.run}"
  echo "QWEN_MODEL_PATH=${QWEN_MODEL_PATH}"
  echo "QWEN_ATTN_IMPLEMENTATION=${QWEN_ATTN_IMPLEMENTATION}"
  echo "QWEN_GRADIENT_CHECKPOINTING=${QWEN_GRADIENT_CHECKPOINTING}"
  echo "QWEN_MIN_PIXELS=${QWEN_MIN_PIXELS}"
  echo "QWEN_MAX_PIXELS=${QWEN_MAX_PIXELS}"
  echo "DATASETS=${datasets}"
  echo "SPLIT=${SPLIT}"
  echo "MIXING_STRATEGY=${mixing_strategy}"
  echo "DATASET_MIX_WEIGHTS=${dataset_mix_weights}"
  echo "MAX_SAMPLES_PER_DATASET=${max_samples_per_dataset}"
  echo "REFHUMAN_MAX_CAPTIONS_PER_INSTANCE=${refhuman_max_captions}"
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
  echo "LR=${lr}"
  echo "QWEN_LORA_LR_SCALE=${qwen_lora_lr_scale}"
  echo "QWEN_VISION_LR_SCALE=${qwen_vision_lr_scale}"
  echo "WARMUP_STEPS=${warmup_steps}"
  echo "MIN_LR_RATIO=${min_lr_ratio}"
  echo "NUM_WORKERS=${num_workers}"
  echo "PREFETCH_FACTOR=${prefetch_factor}"
  echo "SAVE_EVERY=${save_every}"
  echo "SAVE_TOTAL_LIMIT=${save_total_limit}"
  echo "VISUALIZE_EVERY=${visualize_every}"
  echo "DISABLE_BATCH_TRACE=${disable_batch_trace}"
  echo "SEED=${seed}"
  echo "RESUME_ARG=${resume_arg}"
  echo "OUTPUT_DIR=${output_dir}"
  echo "TRAIN_LOG_FILE=${TRAIN_LOG_FILE}"
  echo "===================================================="

  run_train_pose "${args[@]}"
}

###############################################################################
# 启动两阶段训练
###############################################################################

echo "================ QwenPose two-stage run ================"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "RUN_NAME=${RUN_NAME}"
echo "RUN_TS=${RUN_TS}"
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
    "Stage 1 / FREEZE_QWEN warmup" \
    "${STAGE1_OUTPUT_DIR}" \
    "${STAGE1_TRAIN_DATASETS}" \
    "${STAGE1_BATCH_SIZE}" \
    "${STAGE1_GRAD_ACCUM_STEPS}" \
    "${STAGE1_EPOCHS}" \
    "${STAGE1_MAX_STEPS}" \
    "${STAGE1_FREEZE_QWEN}" \
    "${STAGE1_W_LM}" \
    "${STAGE1_LM_LOSS_EVERY}" \
    "${STAGE1_REFHUMAN_MAX_CAPTIONS_PER_INSTANCE}" \
    "${STAGE1_MIXING_STRATEGY}" \
    "${STAGE1_DATASET_MIX_WEIGHTS}" \
    "${STAGE1_MAX_SAMPLES_PER_DATASET}" \
    "${STAGE1_LR}" \
    "${STAGE1_QWEN_LORA_LR_SCALE}" \
    "${STAGE1_QWEN_VISION_LR_SCALE}" \
    "${STAGE1_WARMUP_STEPS}" \
    "${STAGE1_MIN_LR_RATIO}" \
    "${STAGE1_NUM_WORKERS}" \
    "${STAGE1_PREFETCH_FACTOR}" \
    "${STAGE1_SAVE_EVERY}" \
    "${STAGE1_SAVE_TOTAL_LIMIT}" \
    "${STAGE1_VISUALIZE_EVERY}" \
    "${STAGE1_SEED}" \
    "${STAGE1_RESUME_FROM_CHECKPOINT}" \
    "${STAGE1_DISABLE_BATCH_TRACE}"
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
    "Stage 2 / Qwen LoRA + bbox LM" \
    "${STAGE2_OUTPUT_DIR}" \
    "${STAGE2_TRAIN_DATASETS}" \
    "${STAGE2_BATCH_SIZE}" \
    "${STAGE2_GRAD_ACCUM_STEPS}" \
    "${STAGE2_EPOCHS}" \
    "${STAGE2_MAX_STEPS}" \
    "${STAGE2_FREEZE_QWEN}" \
    "${STAGE2_W_LM}" \
    "${STAGE2_LM_LOSS_EVERY}" \
    "${STAGE2_REFHUMAN_MAX_CAPTIONS_PER_INSTANCE}" \
    "${STAGE2_MIXING_STRATEGY}" \
    "${STAGE2_DATASET_MIX_WEIGHTS}" \
    "${STAGE2_MAX_SAMPLES_PER_DATASET}" \
    "${STAGE2_LR}" \
    "${STAGE2_QWEN_LORA_LR_SCALE}" \
    "${STAGE2_QWEN_VISION_LR_SCALE}" \
    "${STAGE2_WARMUP_STEPS}" \
    "${STAGE2_MIN_LR_RATIO}" \
    "${STAGE2_NUM_WORKERS}" \
    "${STAGE2_PREFETCH_FACTOR}" \
    "${STAGE2_SAVE_EVERY}" \
    "${STAGE2_SAVE_TOTAL_LIMIT}" \
    "${STAGE2_VISUALIZE_EVERY}" \
    "${STAGE2_SEED}" \
    "${stage2_resume_arg}" \
    "${STAGE2_DISABLE_BATCH_TRACE}"

  if [[ "${DRY_RUN_DATA}" != "1" && "${MERGE_FINAL_WEIGHTS}" == "1" ]]; then
    merge_full_weights "${STAGE2_OUTPUT_DIR}" "${MERGED_WEIGHTS_DIR}"
  fi
else
  echo "Skipping stage 2 because RUN_STAGE2=0"
fi
