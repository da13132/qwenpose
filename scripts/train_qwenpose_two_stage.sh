#!/usr/bin/env bash
# 开启严格模式：
# -e  任一命令失败立即退出；
# -E  让 ERR trap 在函数/子 shell 中也能生效；
# -u  访问未定义变量时报错；
# -o pipefail 管道中任一命令失败即整条管道失败。
set -Eeuo pipefail

###############################################################################
# QwenPose 两阶段训练脚本
#
# Stage 1 / GT-box pose warmup:
#   冻结 Qwen，PoseHead 使用 GT box，当前默认训练 30 epoch。
# Stage 2 / Closed-loop Qwen-box training:
#   Qwen generate bbox JSON，解析后作为 PoseHead 条件框；GT box 只用于匹配和监督。
###############################################################################

# DEFAULT_PROJECT_ROOT：脚本所在目录的上一级，也就是默认项目根目录。
DEFAULT_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# PROJECT_ROOT：允许外部先导出 PROJECT_ROOT 覆盖默认项目根目录。
PROJECT_ROOT="${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"
# SCRIPT_PATH_REL：用于帮助信息里的相对脚本名，避免打印一长串绝对路径。
SCRIPT_PATH_REL="scripts/$(basename "${BASH_SOURCE[0]}")"

print_usage() {
  # 这里说明两种常见调用方式：
  # 1. 直接启动一个新的两阶段训练；
  # 2. 通过 --resume 从已有 run/stage/checkpoint 继续训练。
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
  # 把命令行里的 kebab-case 统一转成 shell 变量可接受的 ALL_CAPS。
  # 例如 --stage2-batch-size 会被规范成 STAGE2_BATCH_SIZE。
  local raw_name="$1"
  local normalized="${raw_name//-/_}"
  printf '%s\n' "${normalized^^}"
}

is_cli_var_name() {
  # 只允许 A-Z、数字和下划线，且首字符必须是字母，避免用户注入非法变量名。
  local normalized_name
  normalized_name="$(normalize_cli_var_name "$1")"
  [[ "${normalized_name}" =~ ^[A-Z][A-Z0-9_]*$ ]]
}

set_cli_var() {
  # 把命令行里的 --VAR VALUE / --var=value 统一写进当前 shell 变量并 export，
  # 这样后面的默认值逻辑和子进程都能直接读取到。
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

# CLI_RESUME_PATH：记录用户通过 --resume 指定的恢复入口，可以是 run/stage/checkpoint/file。
CLI_RESUME_PATH="${CLI_RESUME_PATH:-}"
while (($# > 0)); do
  case "$1" in
    --resume)
      # 处理形如 --resume PATH 的写法。
      shift
      if (($# == 0)); then
        echo "--resume requires a path argument." >&2
        exit 1
      fi
      CLI_RESUME_PATH="$1"
      ;;
    --resume=*)
      # 处理形如 --resume=PATH 的写法。
      CLI_RESUME_PATH="${1#*=}"
      if [[ -z "${CLI_RESUME_PATH}" ]]; then
        echo "--resume requires a non-empty path argument." >&2
        exit 1
      fi
      ;;
    -h|--help)
      # 显示帮助并直接退出，不进入任何训练逻辑。
      print_usage
      exit 0
      ;;
    --*=*)
      # 处理形如 --stage2_lr=5e-5 的内联赋值形式。
      cli_name="${1%%=*}"
      cli_value="${1#*=}"
      cli_name="${cli_name#--}"
      set_cli_var "${cli_name}" "${cli_value}"
      ;;
    --*)
      # 处理形如 --stage2_lr 5e-5 的分离赋值形式。
      cli_name="${1#--}"
      shift
      if (($# == 0)); then
        echo "--${cli_name} requires a value argument." >&2
        exit 1
      fi
      set_cli_var "${cli_name}" "$1"
      ;;
    *)
      # 脚本不接受位置参数，遇到未知输入直接报错，避免悄悄忽略。
      echo "Unsupported argument: $1" >&2
      print_usage >&2
      exit 1
      ;;
  esac
  shift
done

# 把项目根目录解析成绝对路径，后续切目录/拼路径都以它为准。
PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"

resolve_default_python() {
  # 优先使用项目内的独立环境，保证 torch/qwenpose 依赖版本最稳定。
  if [[ -x "${PROJECT_ROOT}/envs/qwenpose/bin/python" ]]; then
    printf '%s\n' "${PROJECT_ROOT}/envs/qwenpose/bin/python"
    return 0
  fi
  # 其次回退到系统 python3。
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  # 最后再尝试 python。
  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi
  echo "No Python interpreter found. Set PYTHON=/path/to/python before running ${SCRIPT_PATH_REL}." >&2
  exit 1
}

resolve_default_torchrun() {
  # torchrun 仅在分布式/DeepSpeed 启动时使用；
  # 如果环境里没有，也会在后面回退到 python -m torch.distributed.run。
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

# DEFAULT_PYTHON / PYTHON：训练和辅助 Python 逻辑统一使用的解释器。
DEFAULT_PYTHON="$(resolve_default_python)"
PYTHON="${PYTHON:-${DEFAULT_PYTHON}}"
# DEFAULT_TORCHRUN / TORCHRUN：分布式启动器路径，可被外部 TORCHRUN 覆盖。
DEFAULT_TORCHRUN="$(resolve_default_torchrun)"
TORCHRUN="${TORCHRUN:-${DEFAULT_TORCHRUN}}"

# 进入项目根目录，并把 src 加入 PYTHONPATH，确保 `python -m qwenpose.*` 可导入本地源码。
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

resolve_cli_resume_context() {
  # 这里借助一小段 Python 来做复杂路径推断：
  # 用户可能传 run 目录、stage 目录、checkpoint 目录，甚至单个 pt 文件，
  # 用 Python 的 pathlib 处理会比纯 bash 更稳、更容易兼容历史目录结构。
  "${PYTHON}" - "$1" <<'PY'
from __future__ import annotations
import shlex
import sys
from pathlib import Path

# qwenpose 自定义单文件 checkpoint 名称。
CHECKPOINT_PAYLOAD_NAME = "qwenpose_checkpoint.pt"
# 当前两阶段脚本里 stage1 的标准目录名。
STAGE1_NAME = "stage1_freeze_qwen"
# 兼容历史版本里出现过的 stage2/stage3 目录名，便于老实验继续恢复。
STAGE2_NAMES = (
    "stage2_qwen_box_closed_loop",
    "stage3_qwen_box_closed_loop",  # legacy from the removed three-stage layout
    "stage2_teacher_forcing",       # legacy middle-stage checkpoints can initialize closed-loop stage2
    "stage2_qwen_lora_lm",          # legacy public snapshot name
)

def shell_assign(name: str, value: str) -> str:
    # 生成可被 bash `eval` 安全接收的 NAME='value' 赋值语句。
    return f"{name}={shlex.quote(value)}"

def has_checkpoint_payload(path: Path) -> bool:
    # 判断一个路径是否“像是一个可恢复 checkpoint”：
    # 既支持单文件权重，也支持 stage 目录 / DeepSpeed 目录。
    if path.is_file():
        return path.name == CHECKPOINT_PAYLOAD_NAME or path.name.startswith("checkpoint_step_")
    if not path.is_dir():
        return False
    if (path / CHECKPOINT_PAYLOAD_NAME).is_file() or (path / "deepspeed").exists():
        return True
    return any(path.glob("checkpoint-*")) or any(path.glob("checkpoint_step_*.pt"))

def has_direct_checkpoint_children(path: Path) -> bool:
    # 判断一个目录的一级子目录/子文件中是否直接带 checkpoint-*。
    # 这用于区分“run 根目录”和“某个 stage 输出目录”。
    return path.is_dir() and (any(path.glob("checkpoint-*")) or any(path.glob("checkpoint_step_*.pt")))

def latest_log_file(run_dir: Path) -> str:
    # 恢复训练时，优先沿用该 run 下最新的日志文件继续追加输出。
    log_dir = run_dir / "logs"
    if not log_dir.is_dir():
        return ""
    logs = sorted(log_dir.glob("train_*.log"))
    return str(logs[-1]) if logs else ""

def first_existing_stage(root: Path, names: tuple[str, ...]) -> Path | None:
    # 按优先顺序找到第一个存在的 stage 目录。
    for name in names:
        candidate = root / name
        if candidate.exists():
            return candidate
    return None

# target：用户传入的恢复路径，先展开 ~ 再转绝对路径。
target = Path(sys.argv[1]).expanduser().resolve()
if not target.exists():
    raise FileNotFoundError(f"Resume path not found: {target}")

# 下面这些值会回写给 bash，用来自动补全输出目录、阶段目录、恢复策略等变量。
run_dir = target
stage1_dir = ""
stage2_dir = ""
stage2_resume = "none"
stage2_init = ""
run_stage1 = "0"
run_stage2 = "1"

def set_from_run(root: Path) -> None:
    # 用户给的是一个完整 run 根目录时：
    # - 默认不重跑 stage1；
    # - 优先恢复/初始化 stage2；
    # - 如果 stage2 不存在但 stage1 有权重，则允许 stage2 从 stage1 初始化。
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

# 下面按用户给定路径的“层级形态”逐一判断。
# 支持的输入包括：
# - run 根目录
# - stage1/stage2 目录
# - checkpoint-* 目录
# - qwenpose_checkpoint.pt / checkpoint_step_*.pt 单文件
if target.is_dir() and ((target / STAGE1_NAME).is_dir() or first_existing_stage(target, STAGE2_NAMES) is not None):
    set_from_run(target)
elif target.is_dir() and target.name in STAGE2_NAMES:
    # 直接指向某个 stage2 目录：最终闭环 stage 用 resume，历史 teacher-forcing stage 用 init。
    run_dir = target.parent
    stage1_candidate = run_dir / STAGE1_NAME
    stage1_dir = str(stage1_candidate) if stage1_candidate.exists() else ""
    stage2_dir = str(target)
    if target.name in ("stage2_qwen_box_closed_loop", "stage3_qwen_box_closed_loop"):
        stage2_resume = str(target)
    else:
        stage2_init = str(target)
elif target.parent.name in STAGE2_NAMES:
    # 指向 stage2 目录下的某个子路径（例如某个 checkpoint-* 或权重文件）。
    run_dir = target.parent.parent
    stage1_candidate = run_dir / STAGE1_NAME
    stage1_dir = str(stage1_candidate) if stage1_candidate.exists() else ""
    stage2_dir = str(target.parent)
    if target.parent.name in ("stage2_qwen_box_closed_loop", "stage3_qwen_box_closed_loop"):
        stage2_resume = str(target)
    else:
        stage2_init = str(target)
elif target.is_dir() and target.name == STAGE1_NAME:
    # 直接传 stage1 目录：说明 stage2 还没开跑，后续应从 stage1 初始化 stage2。
    run_dir = target.parent
    stage1_dir = str(target)
    stage2_dir = str(run_dir / STAGE2_NAMES[0])
    stage2_init = str(target)
elif target.parent.name == STAGE1_NAME:
    # 传的是 stage1 目录内部的子路径/某个 checkpoint。
    run_dir = target.parent.parent
    stage1_dir = str(target.parent)
    stage2_dir = str(run_dir / STAGE2_NAMES[0])
    stage2_init = str(target.parent)
elif target.parent.name.startswith("checkpoint-") and target.parent.parent.name in STAGE2_NAMES:
    # 传的是 stage2/checkpoint-* 里的某个文件。
    run_dir = target.parent.parent.parent
    stage1_candidate = run_dir / STAGE1_NAME
    stage1_dir = str(stage1_candidate) if stage1_candidate.exists() else ""
    stage2_dir = str(target.parent.parent)
    if target.parent.parent.name in ("stage2_qwen_box_closed_loop", "stage3_qwen_box_closed_loop"):
        stage2_resume = str(target)
    else:
        stage2_init = str(target.parent.parent)
elif target.parent.name.startswith("checkpoint-") and target.parent.parent.name == STAGE1_NAME:
    # 传的是 stage1/checkpoint-* 里的某个文件，用它给 stage2 做初始化。
    run_dir = target.parent.parent.parent
    stage1_dir = str(target.parent.parent)
    stage2_dir = str(run_dir / STAGE2_NAMES[0])
    stage2_init = str(target.parent.parent)
elif target.is_dir() and has_direct_checkpoint_children(target):
    # 目录本身就是一个“可直接 resume 的 checkpoint 容器”。
    run_dir = target
    stage2_dir = str(target)
    stage2_resume = str(target)
elif target.is_dir() and has_checkpoint_payload(target):
    # 目录里已经带有 qwenpose_checkpoint.pt 或 deepspeed 状态，按 stage2 resume 处理。
    run_dir = target.parent
    stage2_dir = str(target)
    stage2_resume = str(target)
elif target.is_file():
    # 单文件 checkpoint 直接恢复到其父目录。
    run_dir = target.parent
    stage2_dir = str(run_dir)
    stage2_resume = str(target)
else:
    raise ValueError("Unsupported resume path layout. Expected a run dir, stage dir, checkpoint dir, or qwenpose_checkpoint.pt file.")

# 逐项打印给 bash，让外层脚本可以 `eval` 回收这些推断结果。
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
  # 让 Python 先把恢复路径“翻译”成当前脚本可直接使用的一组 shell 变量。
  eval "$(resolve_cli_resume_context "${CLI_RESUME_PATH}")"
  if [[ -n "${RESUME_RESOLVED_RUN_DIR:-}" ]]; then
    # 只有用户没有手工覆盖对应变量时，才采用自动推断值；
    # 这样脚本仍然保留“自动恢复 + 局部手动覆写”的灵活性。
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

# RUN_TS：默认 run 时间戳，既用于目录命名，也用于日志文件名去重。
RUN_TS="${RUN_TS:-$(date +%Y%m%d-%H%M%S)}"
# OUTPUT_ROOT：所有两阶段训练实验的父目录。
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/qwenpose_two_stage_qwen}"
# RUN_NAME_BASE：实验名主体；如果用户传入的 RUN_NAME 已经自带时间戳，则直接沿用。
RUN_NAME_BASE="${RUN_NAME:-qwenpose-two-stage-qwen3vl-lora}"
if [[ "${RUN_NAME_BASE}" =~ [0-9]{8}-[0-9]{6}$ ]]; then
  # 已经带类似 20260625-120000 这样的结尾时，不再重复追加时间戳。
  RUN_NAME="${RUN_NAME_BASE}"
else
  # 默认把基础名和时间戳拼在一起，避免多次运行时目录冲突。
  RUN_NAME="${RUN_NAME_BASE}-${RUN_TS}"
fi
# OUTPUT_DIR：本次 run 的总输出目录，stage1/stage2 默认都落在这里下面。
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"
# LOG_DIR：日志目录，默认放在 run 目录下的 logs/。
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"
# LOG_TS：日志时间戳，默认与 RUN_TS 一致，但也支持单独覆盖。
LOG_TS="${LOG_TS:-${RUN_TS}}"
# TRAIN_LOG_FILE：汇总 stdout/stderr 的训练日志文件。
TRAIN_LOG_FILE="${TRAIN_LOG_FILE:-${LOG_DIR}/train_${LOG_TS}.log}"

# 预先创建日志目录，避免 tee/touch 因父目录不存在而失败。
mkdir -p "${LOG_DIR}" "$(dirname "${TRAIN_LOG_FILE}")"
# 先创建一个空日志文件，确保追加模式有明确目标。
touch "${TRAIN_LOG_FILE}"
# 把当前 shell 之后的标准输出和标准错误都重定向到 tee：
# 终端里能实时看到，同时会追加写入日志文件。
exec > >(tee -a "${TRAIN_LOG_FILE}") 2>&1
echo "Logging all stdout/stderr to ${TRAIN_LOG_FILE}"
# ERR trap：一旦任一命令报错，打印失败行号和具体命令，方便排查。
trap 'status=$?; echo "[ERROR] ${BASH_SOURCE[0]}:${LINENO}: ${BASH_COMMAND} exited with status ${status}" >&2' ERR
# EXIT trap：不管成功还是失败，脚本结束时都打印退出码和结束时间。
trap 'status=$?; echo "========== qwenpose two-stage train exit status ${status} at $(date -Is) =========="; exit ${status}' EXIT

# ZERO_STAGE：控制是否启用 DeepSpeed，以及启用哪种 ZeRO 策略。
ZERO_STAGE="${ZERO_STAGE:-zero2}"
# CUDA_VISIBLE_DEVICES：指定当前脚本可见的 GPU 列表。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
# NPROC_PER_NODE：单机启动多少个训练进程，通常等于使用的 GPU 数量。
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
# MASTER_ADDR：分布式主节点地址；单机训练时一般就是回环地址。
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
# MASTER_PORT：分布式通信端口，默认随机挑一个 20001-29999 的端口，减少端口冲突。
export MASTER_PORT="${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}"
# TORCH_FR_BUFFER_SIZE：PyTorch 内部某些分布式数据缓冲大小，保留较大值以降低通信问题概率。
export TORCH_FR_BUFFER_SIZE="${TORCH_FR_BUFFER_SIZE:-200000}"
# TORCH_NCCL_DUMP_ON_TIMEOUT：NCCL 超时时输出更多诊断信息。
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
# TORCH_NCCL_DESYNC_DEBUG：启用 NCCL 反同步调试信息，定位卡死时更有帮助。
export TORCH_NCCL_DESYNC_DEBUG="${TORCH_NCCL_DESYNC_DEBUG:-1}"
# TORCH_DISTRIBUTED_DEBUG：让分布式子系统输出更详细日志。
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
# PYTORCH_CUDA_ALLOC_CONF：启用可扩展显存段，减少显存碎片。
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# WANDB_DISABLED：默认禁用 wandb，避免脚本在无网络/未登录环境里额外依赖外部服务。
export WANDB_DISABLED="${WANDB_DISABLED:-true}"

# 根据 ZERO_STAGE 自动选中对应的 DeepSpeed 配置文件。
case "${ZERO_STAGE}" in
  zero2) DEFAULT_DEEPSPEED_CONFIG="${PROJECT_ROOT}/scripts/zero2.json" ;;
  zero3) DEFAULT_DEEPSPEED_CONFIG="${PROJECT_ROOT}/scripts/zero3.json" ;;
  zero3_offload) DEFAULT_DEEPSPEED_CONFIG="${PROJECT_ROOT}/scripts/zero3_offload.json" ;;
  none) DEFAULT_DEEPSPEED_CONFIG="" ;;
  *) echo "Unsupported ZERO_STAGE=${ZERO_STAGE}. Use zero2, zero3, zero3_offload, or none." >&2; exit 1 ;;
esac
# DEEPSPEED_CONFIG：允许手动覆盖自动选择出的 DeepSpeed 配置。
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${DEFAULT_DEEPSPEED_CONFIG}}"

# DATASET_ROOT：所有数据集的根目录。
DATASET_ROOT="${DATASET_ROOT:-datasets}"
# SPLIT：读取数据集的哪个 split，训练时默认用 train。
SPLIT="${SPLIT:-train}"
# MIXING_STRATEGY：多数据集混合方式，例如 interleave 表示交替采样。
MIXING_STRATEGY="${MIXING_STRATEGY:-interleave}"
# MAX_INSTANCES：单张图最多保留多少个人体实例，防止极端图像拖慢训练。
MAX_INSTANCES="${MAX_INSTANCES:-80}"
# MAX_SAMPLES_PER_DATASET：可选的每个数据集样本上限；为空表示不截断。
MAX_SAMPLES_PER_DATASET="${MAX_SAMPLES_PER_DATASET:-}"
# RECORD_CACHE_DIR：样本索引/解析结果缓存目录，加快重复启动。
RECORD_CACHE_DIR="${RECORD_CACHE_DIR:-.cache/qwenpose_records}"
# DISABLE_RECORD_CACHE：是否禁用上述样本缓存；1 表示关闭缓存。
DISABLE_RECORD_CACHE="${DISABLE_RECORD_CACHE:-0}"

# QWEN_MODEL_PATH：Qwen3-VL 基座模型的本地路径。
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-weights/Qwen3-VL-4B-Instruct}"
# QWEN_DTYPE：加载 Qwen 权重时使用的数据类型。
QWEN_DTYPE="${QWEN_DTYPE:-bfloat16}"
# QWEN_ATTN_IMPLEMENTATION：Qwen 注意力实现，默认用 flash_attention_2 以提升速度和显存效率。
QWEN_ATTN_IMPLEMENTATION="${QWEN_ATTN_IMPLEMENTATION:-flash_attention_2}"
# QWEN_GRADIENT_CHECKPOINTING：是否启用 Qwen 的梯度检查点，1 可省显存，代价是更慢。
QWEN_GRADIENT_CHECKPOINTING="${QWEN_GRADIENT_CHECKPOINTING:-1}"
# QWEN_MIN_PIXELS：输入到 Qwen-VL 前允许的最小像素规模；留空表示沿用模型默认行为。
QWEN_MIN_PIXELS="${QWEN_MIN_PIXELS:-}"
# QWEN_MAX_PIXELS：输入到 Qwen-VL 前允许的最大像素规模；留空表示不额外限制。
QWEN_MAX_PIXELS="${QWEN_MAX_PIXELS:-}"
# QWEN_FEATURE_SIZE：从 Qwen 特征映射到 PoseHead 时的特征尺寸。
QWEN_FEATURE_SIZE="${QWEN_FEATURE_SIZE:-64}"
# QWEN_FEATURE_REFINER_LAYERS：Qwen 特征精炼层数。
QWEN_FEATURE_REFINER_LAYERS="${QWEN_FEATURE_REFINER_LAYERS:-1}"
# QWEN_FEATURE_REFINER_BOTTLENECK_DIM：特征精炼模块的瓶颈维度。
QWEN_FEATURE_REFINER_BOTTLENECK_DIM="${QWEN_FEATURE_REFINER_BOTTLENECK_DIM:-256}"
# QWEN_FEATURE_REFINER_INIT_SCALE：特征精炼模块初始化缩放系数，避免一开始扰动过大。
QWEN_FEATURE_REFINER_INIT_SCALE="${QWEN_FEATURE_REFINER_INIT_SCALE:-0.1}"
# HIDDEN_DIM：Pose 解码器隐藏维度。
HIDDEN_DIM="${HIDDEN_DIM:-448}"
# POSE_DECODER_LAYERS：Pose 解码器层数。
POSE_DECODER_LAYERS="${POSE_DECODER_LAYERS:-3}"
# REFINEMENT_STEPS：关键点精修迭代步数。
REFINEMENT_STEPS="${REFINEMENT_STEPS:-3}"
# BOX_CONDITION_SCALE：条件框扩张倍率，用于给 ROI/条件提示留一定上下文。
BOX_CONDITION_SCALE="${BOX_CONDITION_SCALE:-1.2}"
# POSE_ROI_SIZE：Pose 分支提取 ROI 的空间尺寸。
POSE_ROI_SIZE="${POSE_ROI_SIZE:-16}"
# DECODER_HEADS：Transformer decoder 的注意力头数。
DECODER_HEADS="${DECODER_HEADS:-8}"
# QWEN_LORA_R：语言侧 LoRA 秩。
QWEN_LORA_R="${QWEN_LORA_R:-32}"
# QWEN_LORA_ALPHA：语言侧 LoRA 缩放系数。
QWEN_LORA_ALPHA="${QWEN_LORA_ALPHA:-64}"
# QWEN_LORA_DROPOUT：语言侧 LoRA dropout。
QWEN_LORA_DROPOUT="${QWEN_LORA_DROPOUT:-0.05}"
# QWEN_VISION_LORA_R：视觉侧 LoRA 秩。
QWEN_VISION_LORA_R="${QWEN_VISION_LORA_R:-16}"
# QWEN_VISION_LORA_ALPHA：视觉侧 LoRA 缩放系数。
QWEN_VISION_LORA_ALPHA="${QWEN_VISION_LORA_ALPHA:-32}"
# QWEN_VISION_LORA_DROPOUT：视觉侧 LoRA dropout。
QWEN_VISION_LORA_DROPOUT="${QWEN_VISION_LORA_DROPOUT:-0.05}"

# BATCH_SIZE：公共 batch_size 占位符；若设置了它，stage1/stage2 会按各自默认逻辑继承。
BATCH_SIZE="${BATCH_SIZE:-}"
# LR：公共基础学习率，stage1 默认继承它。
LR="${LR:-2e-4}"
# QWEN_LORA_LR_SCALE：语言侧 LoRA 相对于主学习率的倍率。
QWEN_LORA_LR_SCALE="${QWEN_LORA_LR_SCALE:-0.05}"
# QWEN_VISION_LR_SCALE：视觉侧 LoRA 相对于主学习率的倍率。
QWEN_VISION_LR_SCALE="${QWEN_VISION_LR_SCALE:-0.02}"
# WEIGHT_DECAY：权重衰减系数。
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
# GRAD_CLIP：梯度裁剪阈值。
GRAD_CLIP="${GRAD_CLIP:-1.0}"
# WARMUP_STEPS：学习率预热步数。
WARMUP_STEPS="${WARMUP_STEPS:-100}"
# MIN_LR_RATIO：余弦/衰减调度到末尾时的最小学习率比例。
MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}"
# NUM_WORKERS：DataLoader 的 worker 数。
NUM_WORKERS="${NUM_WORKERS:-0}"
# PREFETCH_FACTOR：每个 worker 预取 batch 数；worker=0 时通常不会实际生效。
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
# DEVICE：训练设备；正常训练是 cuda，调试时也可设 cpu。
DEVICE="${DEVICE:-cuda}"
# AMP：是否启用自动混合精度；1 表示开启。
AMP="${AMP:-0}"
# LOG_EVERY：每多少 step 打印一次训练日志。
LOG_EVERY="${LOG_EVERY:-10}"
# VISUALIZE_EVERY：每多少 step 进行一次可视化输出。
VISUALIZE_EVERY="${VISUALIZE_EVERY:-10}"
# VISUALIZE_MAX_INSTANCES：单次可视化最多画多少个实例。
VISUALIZE_MAX_INSTANCES="${VISUALIZE_MAX_INSTANCES:-8}"
# SYNC_TIMING：是否在统计耗时时强制同步 CUDA，1 更准确但会变慢。
SYNC_TIMING="${SYNC_TIMING:-0}"
# SAVE_EVERY：默认保存 checkpoint 的间隔步数。
SAVE_EVERY="${SAVE_EVERY:-500}"
# SAVE_TOTAL_LIMIT：每个阶段最多保留多少个 checkpoint。
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
# SEED：随机种子。
SEED="${SEED:-42}"

# W_OKS：OKS 损失/指标分支的权重。
W_OKS="${W_OKS:-0.2}"
# W_COORD：关键点坐标回归损失权重。
W_COORD="${W_COORD:-5.0}"
# W_VIS：关键点可见性损失权重。
W_VIS="${W_VIS:-0.05}"
# W_HARD_JOINT：困难关键点重加权损失权重。
W_HARD_JOINT="${W_HARD_JOINT:-0}"
# HARD_JOINT_FRACTION：每个 batch 里认定为“困难关键点”的比例。
HARD_JOINT_FRACTION="${HARD_JOINT_FRACTION:-0.2}"
# W_LM：语言模型损失总权重。
W_LM="${W_LM:-0.05}"
# LM_LOSS_EVERY：每隔多少 step 才计算一次 LM loss，降低开销。
LM_LOSS_EVERY="${LM_LOSS_EVERY:-2}"
# LM_MAX_ANSWER_INSTANCES：LM 生成答案时最多考虑多少个实例，默认跟 MAX_INSTANCES 一致。
LM_MAX_ANSWER_INSTANCES="${LM_MAX_ANSWER_INSTANCES:-${MAX_INSTANCES}}"
# QWEN_BOX_MAX_NEW_TOKENS：Qwen 生成 bbox JSON 时允许的新 token 上限。
QWEN_BOX_MAX_NEW_TOKENS="${QWEN_BOX_MAX_NEW_TOKENS:-4096}"
# BOX_MATCH_IOU_THRESH：Qwen 预测框与 GT 匹配的 IoU 阈值。
BOX_MATCH_IOU_THRESH="${BOX_MATCH_IOU_THRESH:-0.10}"
# BOX_NMS_IOU_THRESH：Qwen 多框输出做 NMS 时的 IoU 阈值。
BOX_NMS_IOU_THRESH="${BOX_NMS_IOU_THRESH:-0.70}"

# DRY_RUN_DATA：只走数据/前向链路检查，不做完整训练的开关。
DRY_RUN_DATA="${DRY_RUN_DATA:-0}"
# PROGRESS_BAR：是否显示 tqdm 进度条；0 表示关闭。
PROGRESS_BAR="${PROGRESS_BAR:-1}"
# DISABLE_REFINEMENT：是否关闭关键点 refinement 模块。
DISABLE_REFINEMENT="${DISABLE_REFINEMENT:-0}"
# DISABLE_HOMOGENEOUS_BATCHES：是否关闭“同质 batch”采样策略。
DISABLE_HOMOGENEOUS_BATCHES="${DISABLE_HOMOGENEOUS_BATCHES:-0}"
# DISABLE_BATCH_TRACE：是否关闭 batch trace 调试信息。
DISABLE_BATCH_TRACE="${DISABLE_BATCH_TRACE:-0}"

# RUN_STAGE1：是否执行第一阶段。
RUN_STAGE1="${RUN_STAGE1:-1}"
# RUN_STAGE2：是否执行第二阶段。
RUN_STAGE2="${RUN_STAGE2:-1}"
# STAGE1_OUTPUT_DIR：第一阶段输出目录。
STAGE1_OUTPUT_DIR="${STAGE1_OUTPUT_DIR:-${OUTPUT_DIR}/stage1_freeze_qwen}"
# STAGE2_OUTPUT_DIR：第二阶段输出目录。
STAGE2_OUTPUT_DIR="${STAGE2_OUTPUT_DIR:-${OUTPUT_DIR}/stage2_qwen_box_closed_loop}"
# STAGE2_INIT_WEIGHTS_DIR：把初始化用 checkpoint 裁成“仅权重版本”后放置的临时目录。
STAGE2_INIT_WEIGHTS_DIR="${STAGE2_INIT_WEIGHTS_DIR:-${OUTPUT_DIR}/stage2_init_weights}"

# STAGE1_TRAIN_DATASETS：第一阶段使用的数据集列表。# coco,mpii,crowdpose,aic,refhuman
STAGE1_TRAIN_DATASETS="${STAGE1_TRAIN_DATASETS:-coco}"
# STAGE2_TRAIN_DATASETS：第二阶段使用的数据集列表，默认继承第一阶段。
STAGE2_TRAIN_DATASETS="${STAGE2_TRAIN_DATASETS:-${STAGE1_TRAIN_DATASETS}}"
# STAGE1_MIXING_STRATEGY：第一阶段多数据集混合策略。
STAGE1_MIXING_STRATEGY="${STAGE1_MIXING_STRATEGY:-${MIXING_STRATEGY}}"
# STAGE2_MIXING_STRATEGY：第二阶段多数据集混合策略。
STAGE2_MIXING_STRATEGY="${STAGE2_MIXING_STRATEGY:-${MIXING_STRATEGY}}"
# STAGE1_MAX_SAMPLES_PER_DATASET：第一阶段每个数据集的样本截断上限。
STAGE1_MAX_SAMPLES_PER_DATASET="${STAGE1_MAX_SAMPLES_PER_DATASET:-${MAX_SAMPLES_PER_DATASET}}"
# STAGE2_MAX_SAMPLES_PER_DATASET：第二阶段每个数据集的样本截断上限。
STAGE2_MAX_SAMPLES_PER_DATASET="${STAGE2_MAX_SAMPLES_PER_DATASET:-${MAX_SAMPLES_PER_DATASET}}"
# STAGE1_REFHUMAN_MAX_CAPTIONS_PER_INSTANCE：第一阶段 RefHuman 每个实例最多采样多少条 caption。
STAGE1_REFHUMAN_MAX_CAPTIONS_PER_INSTANCE="${STAGE1_REFHUMAN_MAX_CAPTIONS_PER_INSTANCE:-${REFHUMAN_MAX_CAPTIONS_PER_INSTANCE:-1}}"
# STAGE2_REFHUMAN_MAX_CAPTIONS_PER_INSTANCE：第二阶段 RefHuman 每个实例最多采样多少条 caption。
STAGE2_REFHUMAN_MAX_CAPTIONS_PER_INSTANCE="${STAGE2_REFHUMAN_MAX_CAPTIONS_PER_INSTANCE:-${REFHUMAN_MAX_CAPTIONS_PER_INSTANCE:-1}}"

# STAGE1_EPOCHS：第一阶段训练轮数。
STAGE1_EPOCHS="${STAGE1_EPOCHS:-30}"
# STAGE2_EPOCHS：第二阶段训练轮数。
STAGE2_EPOCHS="${STAGE2_EPOCHS:-12}"
# STAGE1_BATCH_SIZE：第一阶段每卡 batch size；默认比第二阶段更大，因为 Qwen 冻结且 box 来源更简单。
STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-${BATCH_SIZE:-16}}"
# STAGE2_BATCH_SIZE：第二阶段每卡 batch size；默认更小，因为闭环 generate 更吃显存/算力。
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-${BATCH_SIZE:-1}}"
# STAGE1_GRAD_ACCUM_STEPS：第一阶段梯度累积步数。
STAGE1_GRAD_ACCUM_STEPS="${STAGE1_GRAD_ACCUM_STEPS:-2}"
# STAGE2_GRAD_ACCUM_STEPS：第二阶段梯度累积步数。
STAGE2_GRAD_ACCUM_STEPS="${STAGE2_GRAD_ACCUM_STEPS:-8}"
# STAGE1_MAX_STEPS：第一阶段最大 step 数；>0 时通常优先作为硬上限。
STAGE1_MAX_STEPS="${STAGE1_MAX_STEPS:-0}"
# STAGE2_MAX_STEPS：第二阶段最大 step 数；0 常用于表示“不额外限制，只按 epoch 结束”。
STAGE2_MAX_STEPS="${STAGE2_MAX_STEPS:-0}"
# STAGE1_FREEZE_QWEN：第一阶段是否冻结 Qwen 主体。
STAGE1_FREEZE_QWEN="${STAGE1_FREEZE_QWEN:-1}"
# STAGE2_FREEZE_QWEN：第二阶段是否冻结 Qwen 主体；默认不冻结，让 Qwen 学会闭环框生成。
STAGE2_FREEZE_QWEN="${STAGE2_FREEZE_QWEN:-0}"
# STAGE1_W_LM：第一阶段 LM loss 权重，默认关闭。
STAGE1_W_LM="${STAGE1_W_LM:-0}"
# STAGE2_W_LM：第二阶段 LM loss 权重，默认开启。
STAGE2_W_LM="${STAGE2_W_LM:-0.2}"
# STAGE1_LM_LOSS_EVERY：第一阶段隔多少步算一次 LM loss；0 表示不算。
STAGE1_LM_LOSS_EVERY="${STAGE1_LM_LOSS_EVERY:-0}"
# STAGE2_LM_LOSS_EVERY：第二阶段隔多少步算一次 LM loss。
STAGE2_LM_LOSS_EVERY="${STAGE2_LM_LOSS_EVERY:-1}"
# STAGE1_BOX_SOURCE：第一阶段的条件框来源；默认直接使用 GT。
STAGE1_BOX_SOURCE="${STAGE1_BOX_SOURCE:-gt}"
# STAGE2_BOX_SOURCE：第二阶段的条件框来源；默认用 Qwen 生成的框。
STAGE2_BOX_SOURCE="${STAGE2_BOX_SOURCE:-qwen_generate}"
# STAGE1_BOX_JITTER_SCALE：第一阶段对条件框尺度做随机扰动的幅度。
STAGE1_BOX_JITTER_SCALE="${STAGE1_BOX_JITTER_SCALE:-0.0}"
# STAGE1_BOX_JITTER_SHIFT：第一阶段对条件框中心做随机平移的幅度。
STAGE1_BOX_JITTER_SHIFT="${STAGE1_BOX_JITTER_SHIFT:-0.0}"
# STAGE2_BOX_JITTER_SCALE：第二阶段对条件框尺度做随机扰动的幅度。
STAGE2_BOX_JITTER_SCALE="${STAGE2_BOX_JITTER_SCALE:-0.0}"
# STAGE2_BOX_JITTER_SHIFT：第二阶段对条件框中心做随机平移的幅度。
STAGE2_BOX_JITTER_SHIFT="${STAGE2_BOX_JITTER_SHIFT:-0.0}"
# STAGE1_QWEN_BOX_MAX_NEW_TOKENS：第一阶段若启用 Qwen 产框时，生成 token 上限。
STAGE1_QWEN_BOX_MAX_NEW_TOKENS="${STAGE1_QWEN_BOX_MAX_NEW_TOKENS:-${QWEN_BOX_MAX_NEW_TOKENS}}"
# STAGE2_QWEN_BOX_MAX_NEW_TOKENS：第二阶段 Qwen 产框生成 token 上限。
STAGE2_QWEN_BOX_MAX_NEW_TOKENS="${STAGE2_QWEN_BOX_MAX_NEW_TOKENS:-${QWEN_BOX_MAX_NEW_TOKENS}}"
# STAGE1_BOX_MATCH_IOU_THRESH：第一阶段框匹配 IoU 阈值。
STAGE1_BOX_MATCH_IOU_THRESH="${STAGE1_BOX_MATCH_IOU_THRESH:-${BOX_MATCH_IOU_THRESH}}"
# STAGE2_BOX_MATCH_IOU_THRESH：第二阶段框匹配 IoU 阈值。
STAGE2_BOX_MATCH_IOU_THRESH="${STAGE2_BOX_MATCH_IOU_THRESH:-${BOX_MATCH_IOU_THRESH}}"
# STAGE1_BOX_NMS_IOU_THRESH：第一阶段框 NMS 阈值。
STAGE1_BOX_NMS_IOU_THRESH="${STAGE1_BOX_NMS_IOU_THRESH:-${BOX_NMS_IOU_THRESH}}"
# STAGE2_BOX_NMS_IOU_THRESH：第二阶段框 NMS 阈值。
STAGE2_BOX_NMS_IOU_THRESH="${STAGE2_BOX_NMS_IOU_THRESH:-${BOX_NMS_IOU_THRESH}}"
# STAGE1_LR：第一阶段主学习率。
STAGE1_LR="${STAGE1_LR:-${LR}}"
# STAGE2_LR：第二阶段主学习率，默认比 stage1 更小。
STAGE2_LR="${STAGE2_LR:-5e-5}"
# STAGE1_QWEN_LORA_LR_SCALE：第一阶段语言侧 LoRA 学习率倍率。
STAGE1_QWEN_LORA_LR_SCALE="${STAGE1_QWEN_LORA_LR_SCALE:-${QWEN_LORA_LR_SCALE}}"
# STAGE2_QWEN_LORA_LR_SCALE：第二阶段语言侧 LoRA 学习率倍率。
STAGE2_QWEN_LORA_LR_SCALE="${STAGE2_QWEN_LORA_LR_SCALE:-${QWEN_LORA_LR_SCALE}}"
# STAGE1_QWEN_VISION_LR_SCALE：第一阶段视觉侧 LoRA 学习率倍率。
STAGE1_QWEN_VISION_LR_SCALE="${STAGE1_QWEN_VISION_LR_SCALE:-${QWEN_VISION_LR_SCALE}}"
# STAGE2_QWEN_VISION_LR_SCALE：第二阶段视觉侧 LoRA 学习率倍率。
STAGE2_QWEN_VISION_LR_SCALE="${STAGE2_QWEN_VISION_LR_SCALE:-${QWEN_VISION_LR_SCALE}}"
# STAGE1_WARMUP_STEPS：第一阶段预热步数。
STAGE1_WARMUP_STEPS="${STAGE1_WARMUP_STEPS:-${WARMUP_STEPS}}"
# STAGE2_WARMUP_STEPS：第二阶段预热步数。
STAGE2_WARMUP_STEPS="${STAGE2_WARMUP_STEPS:-${WARMUP_STEPS}}"
# STAGE1_MIN_LR_RATIO：第一阶段最小学习率比例。
STAGE1_MIN_LR_RATIO="${STAGE1_MIN_LR_RATIO:-${MIN_LR_RATIO}}"
# STAGE2_MIN_LR_RATIO：第二阶段最小学习率比例。
STAGE2_MIN_LR_RATIO="${STAGE2_MIN_LR_RATIO:-${MIN_LR_RATIO}}"
# STAGE1_NUM_WORKERS：第一阶段 DataLoader worker 数。
STAGE1_NUM_WORKERS="${STAGE1_NUM_WORKERS:-${NUM_WORKERS}}"
# STAGE2_NUM_WORKERS：第二阶段 DataLoader worker 数。
STAGE2_NUM_WORKERS="${STAGE2_NUM_WORKERS:-${NUM_WORKERS}}"
# STAGE1_PREFETCH_FACTOR：第一阶段 DataLoader 预取系数。
STAGE1_PREFETCH_FACTOR="${STAGE1_PREFETCH_FACTOR:-${PREFETCH_FACTOR}}"
# STAGE2_PREFETCH_FACTOR：第二阶段 DataLoader 预取系数。
STAGE2_PREFETCH_FACTOR="${STAGE2_PREFETCH_FACTOR:-${PREFETCH_FACTOR}}"
# STAGE1_SAVE_EVERY：第一阶段 checkpoint 保存间隔。
STAGE1_SAVE_EVERY="${STAGE1_SAVE_EVERY:-1000}"
# STAGE2_SAVE_EVERY：第二阶段 checkpoint 保存间隔。
STAGE2_SAVE_EVERY="${STAGE2_SAVE_EVERY:-${SAVE_EVERY}}"
# STAGE1_SAVE_TOTAL_LIMIT：第一阶段最多保留 checkpoint 数。
STAGE1_SAVE_TOTAL_LIMIT="${STAGE1_SAVE_TOTAL_LIMIT:-${SAVE_TOTAL_LIMIT}}"
# STAGE2_SAVE_TOTAL_LIMIT：第二阶段最多保留 checkpoint 数。
STAGE2_SAVE_TOTAL_LIMIT="${STAGE2_SAVE_TOTAL_LIMIT:-${SAVE_TOTAL_LIMIT}}"
# STAGE1_VISUALIZE_EVERY：第一阶段可视化频率。
STAGE1_VISUALIZE_EVERY="${STAGE1_VISUALIZE_EVERY:-10}"
# STAGE2_VISUALIZE_EVERY：第二阶段可视化频率。
STAGE2_VISUALIZE_EVERY="${STAGE2_VISUALIZE_EVERY:-${VISUALIZE_EVERY}}"
# STAGE1_DISABLE_BATCH_TRACE：第一阶段默认关闭 batch trace，减少日志噪音。
STAGE1_DISABLE_BATCH_TRACE="${STAGE1_DISABLE_BATCH_TRACE:-1}"
# STAGE2_DISABLE_BATCH_TRACE：第二阶段是否关闭 batch trace。
STAGE2_DISABLE_BATCH_TRACE="${STAGE2_DISABLE_BATCH_TRACE:-${DISABLE_BATCH_TRACE}}"
# STAGE1_SEED：第一阶段随机种子。
STAGE1_SEED="${STAGE1_SEED:-${SEED}}"
# STAGE2_SEED：第二阶段随机种子。
STAGE2_SEED="${STAGE2_SEED:-${SEED}}"
# STAGE1_RESUME_FROM_CHECKPOINT：第一阶段恢复入口；none 表示从头开始。
STAGE1_RESUME_FROM_CHECKPOINT="${STAGE1_RESUME_FROM_CHECKPOINT:-none}"
# STAGE2_RESUME_FROM_CHECKPOINT：第二阶段恢复入口；none 表示不直接续训。
STAGE2_RESUME_FROM_CHECKPOINT="${STAGE2_RESUME_FROM_CHECKPOINT:-none}"
# STAGE2_INIT_CHECKPOINT：第二阶段初始化权重来源，可来自任意兼容 checkpoint。
STAGE2_INIT_CHECKPOINT="${STAGE2_INIT_CHECKPOINT:-}"
# STAGE2_INIT_FROM_STAGE1：若 stage2 没有显式 resume/init，是否默认从 stage1 权重初始化。
STAGE2_INIT_FROM_STAGE1="${STAGE2_INIT_FROM_STAGE1:-1}"
# MERGE_FINAL_WEIGHTS：训练结束后是否把 LoRA/增量权重并回完整模型。
MERGE_FINAL_WEIGHTS="${MERGE_FINAL_WEIGHTS:-0}"
# MERGED_WEIGHTS_ROOT：合并后完整权重的父目录。
MERGED_WEIGHTS_ROOT="${MERGED_WEIGHTS_ROOT:-weights}"
# MERGED_WEIGHTS_DIR：最终导出的完整权重目录。
MERGED_WEIGHTS_DIR="${MERGED_WEIGHTS_DIR:-${MERGED_WEIGHTS_ROOT}/${RUN_NAME}-merged-${RUN_TS}}"

if [[ -n "${CLI_RESUME_PATH}" ]]; then
  # 如果用户启用了 --resume，而当前脚本变量又还没被显式覆盖，
  # 那么这里把自动推断出的 stage2 resume/init 路径补回来。
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

# require_positive_int：校验某个变量必须是正整数，常用于 batch/epoch/save_every 等参数。
require_positive_int() { local name="$1" value="$2"; if ! [[ "${value}" =~ ^[0-9]+$ ]] || (( value <= 0 )); then echo "${name} must be a positive integer, got: ${value}" >&2; exit 1; fi; }
# require_nonnegative_int：校验变量必须是非负整数，允许 0 作为“关闭/不限”的语义。
require_nonnegative_int() { local name="$1" value="$2"; if ! [[ "${value}" =~ ^[0-9]+$ ]]; then echo "${name} must be a non-negative integer, got: ${value}" >&2; exit 1; fi; }
# require_bool：统一约束脚本里的布尔开关只能取 0 或 1。
require_bool() { local name="$1" value="$2"; if [[ "${value}" != "0" && "${value}" != "1" ]]; then echo "${name} must be 0 or 1, got: ${value}" >&2; exit 1; fi; }

resume_target_has_checkpoint() {
  # 判断一个路径下是否存在脚本可识别的 checkpoint 载荷。
  # 这里同时兼容：
  # - 单文件 checkpoint
  # - 目录中的 qwenpose_checkpoint.pt
  # - DeepSpeed 目录
  # - checkpoint-* / checkpoint_step_*.pt
  local resume_path="$1"
  if [[ -f "${resume_path}" ]]; then return 0; fi
  if [[ ! -d "${resume_path}" ]]; then return 1; fi
  if [[ -f "${resume_path}/qwenpose_checkpoint.pt" || -e "${resume_path}/deepspeed" ]]; then return 0; fi
  find "${resume_path}" -maxdepth 1 \( -name 'checkpoint-*' -o -name 'checkpoint_step_*.pt' \) -print -quit | grep -q .
}

check_optional_checkpoint() {
  # 某些 checkpoint 参数是可选的；未传或传 none 时直接跳过。
  # 一旦传了有效路径，就强制检查该路径是否真的包含可恢复权重。
  local name="$1" path="$2"
  if [[ "${path}" == "none" || -z "${path}" ]]; then return 0; fi
  if [[ ! -e "${path}" ]]; then echo "${name} path does not exist: ${path}" >&2; exit 1; fi
  if ! resume_target_has_checkpoint "${path}"; then echo "${name} has no checkpoint payload: ${path}" >&2; exit 1; fi
}

# 用 Python 数一下当前可见 GPU 数量，后面会和 NPROC_PER_NODE 做一致性校验。
VISIBLE_GPU_COUNT="$("${PYTHON}" - <<'PY'
import os
visible = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
print(len(visible) or 1)
PY
)"

# 第一轮：要求这些变量必须是严格正整数，因为 0 会让训练语义不成立。
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

# 第二轮：允许 0 的整型参数，例如 max_steps=0 / lm_loss_every=0 代表某种关闭语义。
for spec in \
  "STAGE1_MAX_STEPS:${STAGE1_MAX_STEPS}" "STAGE2_MAX_STEPS:${STAGE2_MAX_STEPS}" \
  "STAGE1_NUM_WORKERS:${STAGE1_NUM_WORKERS}" "STAGE2_NUM_WORKERS:${STAGE2_NUM_WORKERS}" \
  "STAGE1_VISUALIZE_EVERY:${STAGE1_VISUALIZE_EVERY}" "STAGE2_VISUALIZE_EVERY:${STAGE2_VISUALIZE_EVERY}" \
  "STAGE1_LM_LOSS_EVERY:${STAGE1_LM_LOSS_EVERY}" "STAGE2_LM_LOSS_EVERY:${STAGE2_LM_LOSS_EVERY}" \
  "STAGE1_REFHUMAN_MAX_CAPTIONS_PER_INSTANCE:${STAGE1_REFHUMAN_MAX_CAPTIONS_PER_INSTANCE}" \
  "STAGE2_REFHUMAN_MAX_CAPTIONS_PER_INSTANCE:${STAGE2_REFHUMAN_MAX_CAPTIONS_PER_INSTANCE}"; do
  require_nonnegative_int "${spec%%:*}" "${spec#*:}"
done

# 第三轮：统一检查所有脚本布尔开关都只取 0/1。
for spec in RUN_STAGE1 RUN_STAGE2 STAGE1_FREEZE_QWEN STAGE2_FREEZE_QWEN STAGE2_INIT_FROM_STAGE1 MERGE_FINAL_WEIGHTS QWEN_GRADIENT_CHECKPOINTING AMP DRY_RUN_DATA PROGRESS_BAR SYNC_TIMING DISABLE_RECORD_CACHE DISABLE_REFINEMENT DISABLE_HOMOGENEOUS_BATCHES DISABLE_BATCH_TRACE STAGE1_DISABLE_BATCH_TRACE STAGE2_DISABLE_BATCH_TRACE; do
  require_bool "${spec}" "${!spec}"
done

# 进程数不能超过可见 GPU 数量，否则分布式启动必然失败。
if (( NPROC_PER_NODE > VISIBLE_GPU_COUNT )); then echo "NPROC_PER_NODE=${NPROC_PER_NODE} exceeds visible GPUs (${CUDA_VISIBLE_DEVICES}; count=${VISIBLE_GPU_COUNT})." >&2; exit 1; fi
# Qwen 像素上下限如果用户显式设置了，也要是正整数。
if [[ -n "${QWEN_MIN_PIXELS}" ]]; then require_positive_int QWEN_MIN_PIXELS "${QWEN_MIN_PIXELS}"; fi
if [[ -n "${QWEN_MAX_PIXELS}" ]]; then require_positive_int QWEN_MAX_PIXELS "${QWEN_MAX_PIXELS}"; fi
# 同时设置时，最大像素不能小于最小像素。
if [[ -n "${QWEN_MIN_PIXELS}" && -n "${QWEN_MAX_PIXELS}" ]] && (( QWEN_MAX_PIXELS < QWEN_MIN_PIXELS )); then echo "QWEN_MAX_PIXELS=${QWEN_MAX_PIXELS} must be >= QWEN_MIN_PIXELS=${QWEN_MIN_PIXELS}." >&2; exit 1; fi
# CPU 调试模式下不能再强行启用 DeepSpeed。
if [[ "${DEVICE}" != "cuda" && "${ZERO_STAGE}" != "none" ]]; then echo "DEVICE=${DEVICE} cannot use DeepSpeed ${ZERO_STAGE}. Use ZERO_STAGE=none for CPU debugging." >&2; exit 1; fi
# 目前 stage2 的 qwen_generate 训练路径只验证过 zero2/none。
if [[ "${RUN_STAGE2}" == "1" && "${STAGE2_BOX_SOURCE}" == "qwen_generate" && "${ZERO_STAGE}" != "zero2" && "${ZERO_STAGE}" != "none" ]]; then echo "Stage 2 qwen_generate calls model.generate during training and currently supports ZERO_STAGE=zero2 or none. Got ZERO_STAGE=${ZERO_STAGE}." >&2; exit 1; fi
# 基座模型和 DeepSpeed 配置文件必须真实存在。
if [[ ! -e "${QWEN_MODEL_PATH}" ]]; then echo "QWEN_MODEL_PATH not found: ${QWEN_MODEL_PATH}" >&2; exit 1; fi
if [[ -n "${DEEPSPEED_CONFIG}" && ! -f "${DEEPSPEED_CONFIG}" ]]; then echo "DEEPSPEED_CONFIG not found: ${DEEPSPEED_CONFIG}" >&2; exit 1; fi

# 对三个“可选 checkpoint 输入”做存在性与结构校验。
check_optional_checkpoint STAGE1_RESUME_FROM_CHECKPOINT "${STAGE1_RESUME_FROM_CHECKPOINT}"
check_optional_checkpoint STAGE2_RESUME_FROM_CHECKPOINT "${STAGE2_RESUME_FROM_CHECKPOINT}"
check_optional_checkpoint STAGE2_INIT_CHECKPOINT "${STAGE2_INIT_CHECKPOINT}"

prepare_weights_only_checkpoint() {
  # 这个辅助函数用于“只拿权重初始化 stage2，而不是完整续训”。
  # 它会把源 checkpoint 裁成一个 step=0 的新 checkpoint：
  # - 保留模型权重 / LoRA adapter
  # - 去掉优化器、scaler、随机数状态等训练态信息
  # 这样 stage2 会像“加载预训练权重”一样开始，而不是接着原状态往下跑。
  local source_path="$1" dest_dir="$2"
  if [[ -z "${source_path}" || -z "${dest_dir}" || "${dest_dir}" == "/" ]]; then echo "Invalid weight-only checkpoint arguments: source=${source_path}, dest=${dest_dir}" >&2; exit 1; fi
  if ! resume_target_has_checkpoint "${source_path}"; then echo "Cannot initialize stage 2; no checkpoint found in ${source_path}" >&2; exit 1; fi
  # 每次都重建目标目录，保证初始化权重目录干净、不混入旧文件。
  rm -rf "${dest_dir}"
  mkdir -p "${dest_dir}"
  # 下面用 Python 做 checkpoint 裁剪和 adapter 拷贝。
  "${PYTHON}" - "${source_path}" "${dest_dir}" <<'PY'
import json
import re
import shutil
import sys
from pathlib import Path
import torch

# 与 bash 侧保持一致的自定义 checkpoint 文件名。
CHECKPOINT_PAYLOAD_NAME = "qwenpose_checkpoint.pt"
source = Path(sys.argv[1])
dest = Path(sys.argv[2])

def checkpoint_step(path: Path) -> int | None:
    # 从 checkpoint-123 / checkpoint_step_123.pt 中提取 step 数，便于找“最新一个”。
    match = re.search(r"checkpoint-(\d+)$", path.name) if path.is_dir() else re.search(r"checkpoint_step_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else None

def resolve(path: Path) -> Path:
    # 解析用户传入路径真正对应的 checkpoint：
    # - 文件就直接返回
    # - 目录里有 qwenpose_checkpoint.pt 就返回目录本身
    # - 否则从 checkpoint-* / checkpoint_step_*.pt 中挑 step 最大的那个
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
# payload_path：真正要 torch.load 的载荷文件路径。
payload_path = resolved / CHECKPOINT_PAYLOAD_NAME if resolved.is_dir() else resolved
try:
    # 新版 torch 支持显式传 weights_only=False，以兼容包含训练状态的旧载荷。
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
except TypeError:
    # 老版本 torch 没有 weights_only 参数，回退到兼容写法。
    payload = torch.load(payload_path, map_location="cpu")

# 删除“完整续训”才需要的训练状态字段，让 stage2 以 step=0 的新训练启动。
for key in ("optimizer", "scaler", "training_state", "rng_state"):
    payload.pop(key, None)
payload["step"] = 0
payload["deepspeed_managed"] = False
payload["stage2_weight_only_init_from"] = str(resolved)

# 统一输出成 checkpoint-0 目录，和训练脚本的恢复逻辑保持一致。
out = dest / "checkpoint-0"
out.mkdir(parents=True, exist_ok=True)
torch.save(payload, out / CHECKPOINT_PAYLOAD_NAME)
with (out / "qwenpose_state.json").open("w", encoding="utf-8") as f:
    # 额外写一份 JSON 元数据，便于后续脚本快速读取来源信息。
    json.dump({"step": 0, "checkpoint": str(out), "payload": CHECKPOINT_PAYLOAD_NAME, "deepspeed_tag": None, "training_state": None, "stage2_weight_only_init_from": str(resolved)}, f, indent=2, ensure_ascii=False)
    f.write("\n")
adapter_src = resolved / "qwen_lora_adapter" if resolved.is_dir() else None
if adapter_src is not None and adapter_src.is_dir():
    # 如果源 checkpoint 带 LoRA adapter，也一起复制，避免 stage2 初始化丢失 adapter 权重。
    shutil.copytree(adapter_src, out / "qwen_lora_adapter", dirs_exist_ok=True)
# 最后把新 checkpoint 目录打印给 bash，供命令替换接收。
print(out)
PY
}

merge_full_weights() {
  # 可选收尾步骤：把 stage2 最终 checkpoint 与基础 Qwen 权重合并成一份完整可直接推理的模型目录。
  local checkpoint_source="$1" merged_dir="$2"
  if [[ "${MERGE_FINAL_WEIGHTS}" != "1" ]]; then return 0; fi
  mkdir -p "$(dirname "${merged_dir}")"
  echo "Merging final checkpoint from ${checkpoint_source} into full weights: ${merged_dir}"
  "${PYTHON}" -m qwenpose.merge_full_weights --checkpoint "${checkpoint_source}" --base_model_path "${QWEN_MODEL_PATH}" --output_dir "${merged_dir}" --qwen_dtype "${QWEN_DTYPE}" --qwen_attn_implementation "${QWEN_ATTN_IMPLEMENTATION}" --overwrite
}

run_train_pose() {
  # 按 ZERO_STAGE 决定具体怎么启动训练：
  # 1. none：单进程直接 `python -m qwenpose.train_pose`
  # 2. 有 torchrun：优先用 torchrun 拉起分布式
  # 3. 否则回退到 `python -m torch.distributed.run`
  if [[ "${ZERO_STAGE}" == "none" ]]; then
    "${PYTHON}" -m qwenpose.train_pose "$@"
  elif [[ -n "${TORCHRUN}" ]]; then
    "${TORCHRUN}" --nproc_per_node "${NPROC_PER_NODE}" --master_addr "${MASTER_ADDR}" --master_port "${MASTER_PORT}" "${PROJECT_ROOT}/src/qwenpose/train_pose.py" "$@"
  else
    "${PYTHON}" -m torch.distributed.run --nproc_per_node "${NPROC_PER_NODE}" --master_addr "${MASTER_ADDR}" --master_port "${MASTER_PORT}" "${PROJECT_ROOT}/src/qwenpose/train_pose.py" "$@"
  fi
}

run_stage() {
  # run_stage 是整个脚本的“单阶段执行器”。
  # 它把 stage1/stage2 的一组位置参数统一翻译为 train_pose.py 的命令行参数。
  #
  # 各位置参数含义如下：
  # $1  stage_label                阶段名称，仅用于日志打印
  # $2  output_dir                 当前阶段输出目录
  # $3  datasets                   当前阶段训练数据集列表
  # $4  batch_size                 当前阶段每卡 batch size
  # $5  grad_accum_steps           当前阶段梯度累积步数
  # $6  epochs                     当前阶段 epoch 数
  # $7  max_steps                  当前阶段最大 step 数
  # $8  freeze_qwen                是否冻结 Qwen 主体
  # $9  w_lm                       语言模型损失权重
  # $10 lm_loss_every              LM loss 计算频率
  # $11 refhuman_max_captions      RefHuman 每实例 caption 上限
  # $12 mixing_strategy            多数据集混合策略
  # $13 max_samples_per_dataset    每数据集样本上限
  # $14 lr                         当前阶段基础学习率
  # $15 qwen_lora_lr_scale         文本侧 LoRA 学习率倍率
  # $16 qwen_vision_lr_scale       视觉侧 LoRA 学习率倍率
  # $17 warmup_steps               预热步数
  # $18 min_lr_ratio               最小学习率比例
  # $19 num_workers                DataLoader worker 数
  # $20 prefetch_factor            DataLoader 预取系数
  # $21 save_every                 checkpoint 保存间隔
  # $22 save_total_limit           checkpoint 保留上限
  # $23 visualize_every            可视化频率
  # $24 seed                       随机种子
  # $25 resume_arg                 恢复入口
  # $26 disable_batch_trace        是否关闭 batch trace
  # $27 box_source                 条件框来源
  # $28 box_jitter_scale           框尺度扰动
  # $29 box_jitter_shift           框平移扰动
  # $30 qwen_box_max_new_tokens    Qwen 生成 bbox 的 token 上限
  # $31 box_match_iou_thresh       框匹配 IoU 阈值
  # $32 box_nms_iou_thresh         框 NMS IoU 阈值
  local stage_label="$1" output_dir="$2" datasets="$3" batch_size="$4" grad_accum_steps="$5" epochs="$6" max_steps="$7" freeze_qwen="$8" w_lm="$9"
  local lm_loss_every="${10}" refhuman_max_captions="${11}" mixing_strategy="${12}" max_samples_per_dataset="${13}"
  local lr="${14}" qwen_lora_lr_scale="${15}" qwen_vision_lr_scale="${16}" warmup_steps="${17}" min_lr_ratio="${18}" num_workers="${19}" prefetch_factor="${20}"
  local save_every="${21}" save_total_limit="${22}" visualize_every="${23}" seed="${24}" resume_arg="${25}" disable_batch_trace="${26}"
  local box_source="${27}" box_jitter_scale="${28}" box_jitter_shift="${29}" qwen_box_max_new_tokens="${30}" box_match_iou_thresh="${31}" box_nms_iou_thresh="${32}"
  # effective_batch：实际总 batch = 进程数 * 每卡 batch * 梯度累积。
  local effective_batch=$((NPROC_PER_NODE * batch_size * grad_accum_steps))
  local args=(
    # 数据集相关参数。
    --dataset_root "${DATASET_ROOT}"                    # 数据集根目录。
    --datasets "${datasets}"                            # 当前阶段使用的数据集列表。
    --split "${SPLIT}"                                  # 读取的数据切分。
    --mixing_strategy "${mixing_strategy}"              # 多数据集采样/混合策略。
    --max_instances "${MAX_INSTANCES}"                  # 单图最多保留的人体实例数。
    --refhuman_max_captions_per_instance "${refhuman_max_captions}"  # RefHuman 每实例最多使用多少条 caption。
    --record_cache_dir "${RECORD_CACHE_DIR}"            # 样本记录缓存目录。

    # 模型结构与 Qwen 相关参数。
    --hidden_dim "${HIDDEN_DIM}"                        # Pose 解码器隐藏维度。
    --backbone "qwen3vl"                                # 固定指定当前脚本使用的视觉语言骨干。
    --qwen_model_path "${QWEN_MODEL_PATH}"              # Qwen 基座模型路径。
    --qwen_dtype "${QWEN_DTYPE}"                        # Qwen 权重 dtype。
    --qwen_attn_implementation "${QWEN_ATTN_IMPLEMENTATION}"  # 注意力实现。
    --qwen_feature_size "${QWEN_FEATURE_SIZE}"          # 送入 PoseHead 的 Qwen 特征尺寸。
    --qwen_feature_refiner_layers "${QWEN_FEATURE_REFINER_LAYERS}"  # 特征精炼层数。
    --qwen_feature_refiner_bottleneck_dim "${QWEN_FEATURE_REFINER_BOTTLENECK_DIM}"  # 特征精炼瓶颈维度。
    --qwen_feature_refiner_init_scale "${QWEN_FEATURE_REFINER_INIT_SCALE}"  # 特征精炼初始化缩放。
    --qwen_lora_r "${QWEN_LORA_R}"                      # 文本侧 LoRA 秩。
    --qwen_lora_alpha "${QWEN_LORA_ALPHA}"              # 文本侧 LoRA alpha。
    --qwen_lora_dropout "${QWEN_LORA_DROPOUT}"          # 文本侧 LoRA dropout。
    --qwen_vision_lora_r "${QWEN_VISION_LORA_R}"        # 视觉侧 LoRA 秩。
    --qwen_vision_lora_alpha "${QWEN_VISION_LORA_ALPHA}"  # 视觉侧 LoRA alpha。
    --qwen_vision_lora_dropout "${QWEN_VISION_LORA_DROPOUT}"  # 视觉侧 LoRA dropout。
    --pose_decoder_layers "${POSE_DECODER_LAYERS}"      # Pose decoder 层数。
    --refinement_steps "${REFINEMENT_STEPS}"            # 关键点 refinement 迭代次数。
    --box_condition_scale "${BOX_CONDITION_SCALE}"      # 条件框扩张系数。
    --pose_roi_size "${POSE_ROI_SIZE}"                  # Pose ROI 输出尺寸。
    --decoder_heads "${DECODER_HEADS}"                  # decoder 注意力头数。

    # 训练循环与优化器相关参数。
    --output_dir "${output_dir}"                        # 当前阶段输出目录。
    --batch_size "${batch_size}"                        # 每卡 batch size。
    --grad_accum_steps "${grad_accum_steps}"            # 梯度累积步数。
    --epochs "${epochs}"                                # epoch 数。
    --max_steps "${max_steps}"                          # 最大 step 数。
    --lr "${lr}"                                        # 主学习率。
    --qwen_lora_lr_scale "${qwen_lora_lr_scale}"        # 文本侧 LoRA 学习率倍率。
    --qwen_vision_lr_scale "${qwen_vision_lr_scale}"    # 视觉侧 LoRA 学习率倍率。
    --weight_decay "${WEIGHT_DECAY}"                    # 权重衰减。
    --grad_clip "${GRAD_CLIP}"                          # 梯度裁剪阈值。
    --warmup_steps "${warmup_steps}"                    # 预热步数。
    --min_lr_ratio "${min_lr_ratio}"                    # 最小学习率比例。
    --num_workers "${num_workers}"                      # DataLoader worker 数。
    --prefetch_factor "${prefetch_factor}"              # DataLoader 预取系数。
    --device "${DEVICE}"                                # 训练设备。
    --log_every "${LOG_EVERY}"                          # 日志打印频率。
    --visualize_every "${visualize_every}"              # 可视化频率。
    --visualize_max_instances "${VISUALIZE_MAX_INSTANCES}"  # 单次可视化最多实例数。
    --save_every "${save_every}"                        # checkpoint 保存间隔。
    --save_total_limit "${save_total_limit}"            # checkpoint 保留上限。
    --seed "${seed}"                                    # 随机种子。

    # 损失项与 bbox 条件相关参数。
    --w_oks "${W_OKS}"                                  # OKS 损失权重。
    --w_coord "${W_COORD}"                              # 坐标损失权重。
    --w_vis "${W_VIS}"                                  # 可见性损失权重。
    --w_hard_joint "${W_HARD_JOINT}"                    # 困难关键点损失权重。
    --hard_joint_fraction "${HARD_JOINT_FRACTION}"      # 困难关键点比例。
    --w_lm "${w_lm}"                                    # 语言模型损失权重。
    --lm_loss_every "${lm_loss_every}"                  # LM loss 计算频率。
    --lm_max_answer_instances "${LM_MAX_ANSWER_INSTANCES}"  # LM 生成答案时最多保留的实例数。
    --box_source "${box_source}"                        # 条件框来源：GT / Qwen 生成等。
    --box_jitter_scale "${box_jitter_scale}"            # 条件框尺度扰动。
    --box_jitter_shift "${box_jitter_shift}"            # 条件框平移扰动。
    --qwen_box_max_new_tokens "${qwen_box_max_new_tokens}"  # Qwen bbox 生成 token 上限。
    --box_match_iou_thresh "${box_match_iou_thresh}"    # 预测框与 GT 的匹配阈值。
    --box_nms_iou_thresh "${box_nms_iou_thresh}"        # 预测框 NMS 阈值。
  )
  # 这些参数只有在用户显式启用/提供时才追加，避免把“空值”传进 Python。
  if [[ -n "${max_samples_per_dataset}" ]]; then args+=(--max_samples_per_dataset "${max_samples_per_dataset}"); fi   # 每数据集样本截断上限。
  if [[ "${resume_arg}" != "none" && -n "${resume_arg}" ]]; then args+=(--resume_from_checkpoint "${resume_arg}"); fi  # 当前阶段恢复入口。
  if [[ -n "${QWEN_MIN_PIXELS}" ]]; then args+=(--qwen_min_pixels "${QWEN_MIN_PIXELS}"); fi                           # Qwen 输入最小像素。
  if [[ -n "${QWEN_MAX_PIXELS}" ]]; then args+=(--qwen_max_pixels "${QWEN_MAX_PIXELS}"); fi                           # Qwen 输入最大像素。
  if [[ "${AMP}" == "1" ]]; then args+=(--amp); fi                                                                     # 开启自动混合精度。
  if [[ "${QWEN_GRADIENT_CHECKPOINTING}" == "1" ]]; then args+=(--qwen_gradient_checkpointing); fi                    # 开启 Qwen 梯度检查点。
  if [[ "${freeze_qwen}" == "1" ]]; then args+=(--freeze_qwen); fi                                                     # 冻结 Qwen 主体参数。
  if [[ "${DRY_RUN_DATA}" == "1" ]]; then args+=(--dry_run_data); fi                                                   # 仅做数据/前向链路检查。
  if [[ "${PROGRESS_BAR}" == "0" ]]; then args+=(--disable_progress); fi                                               # 关闭进度条。
  if [[ "${SYNC_TIMING}" == "1" ]]; then args+=(--sync_timing); fi                                                     # 计时前强制同步 CUDA。
  if [[ "${DISABLE_RECORD_CACHE}" == "1" ]]; then args+=(--disable_record_cache); fi                                   # 禁用样本记录缓存。
  if [[ "${DISABLE_REFINEMENT}" == "1" ]]; then args+=(--disable_refinement); fi                                       # 关闭关键点 refinement。
  if [[ "${DISABLE_HOMOGENEOUS_BATCHES}" == "1" ]]; then args+=(--disable_homogeneous_batches); fi                    # 关闭同质 batch。
  if [[ "${disable_batch_trace}" == "1" ]]; then args+=(--disable_batch_trace); fi                                     # 关闭 batch trace。
  if [[ -n "${DEEPSPEED_CONFIG}" ]]; then args+=(--deepspeed_config "${DEEPSPEED_CONFIG}"); fi                         # DeepSpeed 配置文件。

  # 先把本阶段真正生效的关键配置打印出来，便于日志里直接核对实验设置。
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
  # 真正启动训练。
  run_train_pose "${args[@]}"
}

###############################################################################
# Launch two stages
###############################################################################

# 先输出一次总览，方便日志开头就确认本次 run 的整体结构。
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
  # 第一阶段：
  # - 默认使用 GT box；
  # - 默认冻结 Qwen；
  # - 主要把 PoseHead / 条件框到关键点的链路先 warm up 稳定。
  run_stage \
    "Stage 1 / GT-box pose warmup" \
    "${STAGE1_OUTPUT_DIR}" "${STAGE1_TRAIN_DATASETS}" "${STAGE1_BATCH_SIZE}" "${STAGE1_GRAD_ACCUM_STEPS}" "${STAGE1_EPOCHS}" "${STAGE1_MAX_STEPS}" "${STAGE1_FREEZE_QWEN}" "${STAGE1_W_LM}" "${STAGE1_LM_LOSS_EVERY}" "${STAGE1_REFHUMAN_MAX_CAPTIONS_PER_INSTANCE}" "${STAGE1_MIXING_STRATEGY}" "${STAGE1_MAX_SAMPLES_PER_DATASET}" "${STAGE1_LR}" "${STAGE1_QWEN_LORA_LR_SCALE}" "${STAGE1_QWEN_VISION_LR_SCALE}" "${STAGE1_WARMUP_STEPS}" "${STAGE1_MIN_LR_RATIO}" "${STAGE1_NUM_WORKERS}" "${STAGE1_PREFETCH_FACTOR}" "${STAGE1_SAVE_EVERY}" "${STAGE1_SAVE_TOTAL_LIMIT}" "${STAGE1_VISUALIZE_EVERY}" "${STAGE1_SEED}" "${STAGE1_RESUME_FROM_CHECKPOINT}" "${STAGE1_DISABLE_BATCH_TRACE}" "${STAGE1_BOX_SOURCE}" "${STAGE1_BOX_JITTER_SCALE}" "${STAGE1_BOX_JITTER_SHIFT}" "${STAGE1_QWEN_BOX_MAX_NEW_TOKENS}" "${STAGE1_BOX_MATCH_IOU_THRESH}" "${STAGE1_BOX_NMS_IOU_THRESH}"
else
  # RUN_STAGE1=0 时跳过第一阶段，常见于只补跑 stage2 的场景。
  echo "Skipping stage 1 because RUN_STAGE1=0"
fi

if [[ "${RUN_STAGE2}" == "1" ]]; then
  # stage2_resume_arg：本阶段最终真正传给 train_pose 的恢复入口。
  # 它可能来自：
  # 1. 用户显式传入的 STAGE2_RESUME_FROM_CHECKPOINT
  # 2. STAGE2_INIT_CHECKPOINT 裁出来的仅权重初始化 checkpoint
  # 3. stage1 输出裁出来的仅权重初始化 checkpoint
  stage2_resume_arg="${STAGE2_RESUME_FROM_CHECKPOINT}"
  if [[ "${stage2_resume_arg}" == "none" || -z "${stage2_resume_arg}" ]]; then
    if [[ "${DRY_RUN_DATA}" == "1" ]]; then
      # dry-run 时不需要也不应该额外准备初始化 checkpoint。
      stage2_resume_arg="none"
    elif [[ -n "${STAGE2_INIT_CHECKPOINT}" ]]; then
      # 优先使用用户显式指定的 stage2 初始化来源。
      echo "Preparing stage 2 weight-only init from STAGE2_INIT_CHECKPOINT=${STAGE2_INIT_CHECKPOINT}"
      stage2_resume_arg="$(prepare_weights_only_checkpoint "${STAGE2_INIT_CHECKPOINT}" "${STAGE2_INIT_WEIGHTS_DIR}")"
    elif [[ "${STAGE2_INIT_FROM_STAGE1}" == "1" ]]; then
      # 其次按默认策略：从 stage1 输出裁出一个“仅权重初始化 checkpoint”给 stage2。
      echo "Preparing stage 2 weight-only init from stage 1 output: ${STAGE1_OUTPUT_DIR}"
      stage2_resume_arg="$(prepare_weights_only_checkpoint "${STAGE1_OUTPUT_DIR}" "${STAGE2_INIT_WEIGHTS_DIR}")"
    else
      # 如果既没有显式 init checkpoint，也不允许从 stage1 初始化，那就完全从基础模型起跑。
      stage2_resume_arg="none"
      echo "Stage 2 will start from base Qwen + newly initialized pose modules because STAGE2_INIT_FROM_STAGE1=0."
    fi
  else
    # 用户已经提供了完整恢复路径时，stage2 将直接续训，不做裁权重初始化。
    echo "Stage 2 will resume checkpoint state from ${stage2_resume_arg}"
  fi

  # 第二阶段：
  # - 默认启用 Qwen 生成框；
  # - 默认解冻 Qwen；
  # - 让语言模型与姿态分支在闭环条件下联合训练。
  run_stage \
    "Stage 2 / Closed-loop Qwen-box training" \
    "${STAGE2_OUTPUT_DIR}" "${STAGE2_TRAIN_DATASETS}" "${STAGE2_BATCH_SIZE}" "${STAGE2_GRAD_ACCUM_STEPS}" "${STAGE2_EPOCHS}" "${STAGE2_MAX_STEPS}" "${STAGE2_FREEZE_QWEN}" "${STAGE2_W_LM}" "${STAGE2_LM_LOSS_EVERY}" "${STAGE2_REFHUMAN_MAX_CAPTIONS_PER_INSTANCE}" "${STAGE2_MIXING_STRATEGY}" "${STAGE2_MAX_SAMPLES_PER_DATASET}" "${STAGE2_LR}" "${STAGE2_QWEN_LORA_LR_SCALE}" "${STAGE2_QWEN_VISION_LR_SCALE}" "${STAGE2_WARMUP_STEPS}" "${STAGE2_MIN_LR_RATIO}" "${STAGE2_NUM_WORKERS}" "${STAGE2_PREFETCH_FACTOR}" "${STAGE2_SAVE_EVERY}" "${STAGE2_SAVE_TOTAL_LIMIT}" "${STAGE2_VISUALIZE_EVERY}" "${STAGE2_SEED}" "${stage2_resume_arg}" "${STAGE2_DISABLE_BATCH_TRACE}" "${STAGE2_BOX_SOURCE}" "${STAGE2_BOX_JITTER_SCALE}" "${STAGE2_BOX_JITTER_SHIFT}" "${STAGE2_QWEN_BOX_MAX_NEW_TOKENS}" "${STAGE2_BOX_MATCH_IOU_THRESH}" "${STAGE2_BOX_NMS_IOU_THRESH}"

  if [[ "${DRY_RUN_DATA}" != "1" && "${MERGE_FINAL_WEIGHTS}" == "1" ]]; then
    # 可选导出：把最终 stage2 权重合并成一份完整模型目录，便于独立部署/推理。
    merge_full_weights "${STAGE2_OUTPUT_DIR}" "${MERGED_WEIGHTS_DIR}"
  fi
else
  # RUN_STAGE2=0 时跳过第二阶段，常见于只训练/调试 stage1 的场景。
  echo "Skipping stage 2 because RUN_STAGE2=0"
fi
