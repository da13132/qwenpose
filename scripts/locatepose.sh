#!/usr/bin/env bash
set -Eeuo pipefail

# Long training runs must not continue reading a script file that may be edited
# while the run is still active. Execute an immutable /tmp snapshot so the stage1
# -> stage2 tail cannot see a half-updated script and fail with an EOF quote
# parser error after hours of training.
if [[ -z "${LOCATEPOSE_SCRIPT_SNAPSHOT:-}" ]]; then
  original_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  # PROJECT_ROOT：传递给不可变脚本快照的项目根目录；允许从任意工作目录启动训练。
  export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${original_script_dir}/.." && pwd)}"
  # LOCATEPOSE_SCRIPT_PATH_REL：原始脚本相对路径，仅用于 help/报错展示，避免 /tmp 快照路径干扰用户。
  export LOCATEPOSE_SCRIPT_PATH_REL="${LOCATEPOSE_SCRIPT_PATH_REL:-scripts/$(basename "${BASH_SOURCE[0]}")}"
  snapshot_script="${TMPDIR:-/tmp}/locatepose.$$.sh"
  cp -- "${BASH_SOURCE[0]}" "${snapshot_script}"
  export LOCATEPOSE_SCRIPT_SNAPSHOT=1
  exec bash "${snapshot_script}" "$@"
fi

###############################################################################
# LocatePose 两阶段训练脚本
#
# 目标模型：LocateAnything-3B + QwenPose PoseHead。
#
# Stage 1 / frozen-backbone unified person-query warmup：
#   加载 MoonViT、投影器和 RefHuman 文本 decoder，但裁掉自回归 lm_head/KV cache；
#   默认冻结 Locate 主干，从第一步联合训练检测、指代匹配和 PoseHead。
#
# Stage 2 / selective Locate + RefHuman adaptation：
#   合并原 Stage 2/3，同时使用 COCO/MPII/CrowdPose/RefHuman，解冻少量 MoonViT 与 LLM LoRA；
#   人体框由 person queries 单次前向输出，不恢复自回归坐标生成组件。
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
SCRIPT_PATH_REL="${LOCATEPOSE_SCRIPT_PATH_REL:-scripts/$(basename "${BASH_SOURCE[0]}")}"

print_usage() {
  cat <<EOF
Usage:
  ${SCRIPT_PATH_REL} [stage1|stage2|all]... [--resume PATH] [--VAR VALUE|--VAR=VALUE]...

Examples:
  ${SCRIPT_PATH_REL} stage1
  ${SCRIPT_PATH_REL} stage2 --STAGE2_INIT_CHECKPOINT /path/to/stage1
  ${SCRIPT_PATH_REL} stage1,stage2
  ${SCRIPT_PATH_REL} all

Options:
  --resume PATH   Resume from a run dir, stage dir, checkpoint dir, or checkpoint file.
                  A full run dir prefers the merged stage2 checkpoint, while legacy
                  stage3_refhuman_closed_loop checkpoints remain resumable.
  --VAR VALUE     Override any script variable. Supports ALL_CAPS, snake_case, and kebab-case.
  --VAR=VALUE     Same as above, using inline assignment.
  -h, --help      Show this help message.

Stages:
  stage1  frozen LocateAnything, all four datasets, train unified detection/pose/ref heads
  stage2  selective vision/LLM LoRA + pose/fusion/ref-match parameters, all four datasets
  stage3  legacy alias for the merged stage2
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
  case "${name}" in
    CUDA_VISIBLE_DEVICES|LOCATEPOSE_CUDA_VISIBLE_DEVICES|NPROC_PER_NODE)
      echo "Removed GPU override option: --${raw_name}" >&2
      echo "scripts/locatepose.sh forcibly uses physical GPUs 1,2,3 with three processes." >&2
      exit 1
      ;;
    LOCATE_MIN_PIXELS|LOCATE_MAX_PIXELS|LOCATE_FEATURE_SIZE|POSE_IMAGE_SIZE|ENABLE_REF_VISUAL_MODULATION|LR|MAX_STEPS|W_VIS|MERGE_FINAL_WEIGHTS|KEYPOINT_DECODE_MODE|\
    STAGE1_FREEZE_LOCATE|STAGE1_LOCATE_TRAIN_SCOPE|STAGE1_LOCATE_GRADIENT_CHECKPOINTING|STAGE1_LOCATE_FEATURE_SOURCE|STAGE2_LOCATE_FEATURE_SOURCE|\
    STAGE1_BOX_SOURCE|STAGE2_BOX_SOURCE|STAGE1_BOX_JITTER_SCALE|STAGE1_BOX_JITTER_SHIFT|STAGE2_BOX_JITTER_SCALE|STAGE2_BOX_JITTER_SHIFT|\
    STAGE1_POSE_AUGMENT|STAGE2_POSE_AUGMENT|AUGMENT_*|LOCATE_GENERATION_MODE|LOCATE_BOX_MAX_NEW_TOKENS|LOCATE_GENERATE_REFHUMAN_ONLY|\
    BOX_MATCH_IOU_THRESH|BOX_NMS_IOU_THRESH|DISABLE_PRE_POSE_NMS|POST_POSE_NMS_IOU_THRESH|DISABLE_LOCATE_GROUNDING_AUX|\
    LOCATE_LM_*|STAGE1_W_LOCATE_BOX_LM|STAGE2_W_LOCATE_BOX_LM|STAGE1_W_LOCATE_POINT_LM|STAGE2_W_LOCATE_POINT_LM)
      echo "Removed LocatePose training option: --${raw_name}" >&2
      echo "The current training path is fixed to raw_visual + person_queries and has no Locate coordinate-generation/teacher-forcing branch." >&2
      exit 1
      ;;
  esac
  printf -v "${name}" '%s' "${value}"
  export "${name}"
}

# CLI_RESUME_PATH：命令行 --resume 传入的路径，稍后会做阶段感知解析。
CLI_RESUME_PATH="${CLI_RESUME_PATH:-}"
# CLI_STAGE_SELECTION：位置参数指定的可组合阶段列表。
CLI_STAGE_SELECTION="${CLI_STAGE_SELECTION:-}"
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
    stage1|stage2|stage3|all|*,*)
      CLI_STAGE_SELECTION="${CLI_STAGE_SELECTION:+${CLI_STAGE_SELECTION},}$1"
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

# DEFAULT_PYTHON：自动探测结果，优先项目虚拟环境，其次系统 python3/python；通常无需手动设置。
DEFAULT_PYTHON="$(resolve_default_python)"
# PYTHON：实际执行训练与校验的解释器路径；需要固定依赖环境时可显式覆盖为绝对路径。
PYTHON="${PYTHON:-${DEFAULT_PYTHON}}"
# DEFAULT_TORCHRUN：自动探测到的 torchrun；为空时只允许 ZERO_STAGE=none 的单进程运行。
DEFAULT_TORCHRUN="$(resolve_default_torchrun)"
# TORCHRUN：多卡和 DeepSpeed 启动器路径；正式多 GPU 训练应指向与 PYTHON 同一环境中的 torchrun。
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
STAGE1_NAME = "stage1_freeze_locate_person_queries"
STAGE2_NAME = "stage2_unfreeze_locate_person_queries"
LEGACY_STAGE1_NAME = "stage1_freeze_locate_gt_box"
LEGACY_STAGE2_NAME = "stage2_locate_box_closed_loop"
STAGE3_NAME = "stage3_refhuman_closed_loop"

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

def checkpoint_step(path: Path) -> int:
    import re

    if path.is_dir():
        match = re.search(r"checkpoint-(\d+)$", path.name)
    else:
        match = re.search(r"checkpoint_step_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1

def latest_checkpoint(path: Path) -> str:
    if path.is_file() or (path / CHECKPOINT_PAYLOAD_NAME).is_file() or (path / "deepspeed").exists():
        return str(path)
    candidates = [candidate for candidate in list(path.glob("checkpoint-*")) + list(path.glob("checkpoint_step_*.pt")) if checkpoint_step(candidate) >= 0]
    return str(sorted(candidates, key=checkpoint_step)[-1]) if candidates else ""

def stage_has_activity(path: Path) -> bool:
    if not path.is_dir():
        return False
    if has_checkpoint_payload(path):
        return True
    return any(path.iterdir())

def latest_log_file(run_dir: Path) -> str:
    log_dir = run_dir / "logs"
    if not log_dir.is_dir():
        return ""
    logs = sorted(log_dir.glob("train_*.log"))
    return str(logs[-1]) if logs else ""

def checkpoint_training_state(path_value: str) -> dict[str, object]:
    """Read the lightweight resume cursor without loading the model tensors."""
    if not path_value or path_value == "none":
        return {}
    path = Path(path_value)
    if path.is_file():
        state_file = path.parent / "qwenpose_state.json"
    else:
        state_file = path / "qwenpose_state.json"
    if not state_file.is_file():
        return {}
    import json

    payload = json.loads(state_file.read_text(encoding="utf-8"))
    state = payload.get("training_state")
    return state if isinstance(state, dict) else {}

def print_resume_defaults(stage: str, checkpoint: str) -> None:
    state = checkpoint_training_state(checkpoint)
    if not state:
        return
    prefix = f"RESUME_RESOLVED_{stage}"
    scalar_fields = {
        "BATCH_SIZE": "batch_size",
        "GRAD_ACCUM_STEPS": "grad_accum_steps",
        "WORLD_SIZE": "world_size",
        "MIXING_STRATEGY": "mixing_strategy",
        "SPLIT": "split",
        "GLOBAL_STEP": "global_step",
        "EPOCH": "epoch",
        "BATCH_IN_EPOCH": "batch_in_epoch",
        "BATCHES_PER_EPOCH": "batches_per_epoch",
    }
    for suffix, key in scalar_fields.items():
        if key in state:
            print(shell_assign(f"{prefix}_{suffix}", str(state[key])))
    datasets = state.get("dataset_names")
    if isinstance(datasets, list) and datasets:
        print(shell_assign(f"{prefix}_TRAIN_DATASETS", ",".join(str(name) for name in datasets)))

target = Path(sys.argv[1]).expanduser().resolve()
if not target.exists():
    raise FileNotFoundError(f"Resume path not found: {target}")

# Keep old runs resumable while making the unified person-query directories the
# default for new runs and checkpoints.
target_parts = set(target.parts)
legacy_run = (
    LEGACY_STAGE1_NAME in target_parts
    or LEGACY_STAGE2_NAME in target_parts
    or (
        target.is_dir()
        and not (target / STAGE1_NAME).exists()
        and not (target / STAGE2_NAME).exists()
        and ((target / LEGACY_STAGE1_NAME).exists() or (target / LEGACY_STAGE2_NAME).exists())
    )
)
if legacy_run:
    STAGE1_NAME = LEGACY_STAGE1_NAME
    STAGE2_NAME = LEGACY_STAGE2_NAME

run_dir = target
stage1_dir = ""
stage2_dir = ""
stage3_dir = ""
stage1_resume = "none"
stage2_resume = "none"
stage3_resume = "none"
stage2_init = ""
stage3_init = ""
run_stage1 = "0"
run_stage2 = "0"
run_stage3 = "1"

if target.is_dir() and any((target / name).is_dir() for name in (STAGE1_NAME, STAGE2_NAME, STAGE3_NAME)):
    run_dir = target
    s1 = run_dir / STAGE1_NAME
    s2 = run_dir / STAGE2_NAME
    s3 = run_dir / STAGE3_NAME
    stage1_dir = str(s1)
    stage2_dir = str(s2)
    stage3_dir = str(s3)
    if s3.exists() and has_checkpoint_payload(s3):
        stage3_resume = latest_checkpoint(s3)
    elif s2.exists() and has_checkpoint_payload(s2):
        stage2_resume = latest_checkpoint(s2)
        run_stage2 = "1"
        run_stage3 = "1"
    elif s1.exists() and has_checkpoint_payload(s1):
        stage1_resume = latest_checkpoint(s1)
        run_stage1 = "1"
        run_stage2 = "1"
        run_stage3 = "1"
elif target.is_dir() and target.name == STAGE3_NAME:
    run_dir = target.parent
    stage1_candidate = run_dir / STAGE1_NAME
    stage2_candidate = run_dir / STAGE2_NAME
    s3_init = run_dir / "stage3_init_weights"
    stage1_dir = str(stage1_candidate)
    stage2_dir = str(stage2_candidate)
    stage3_dir = str(target)
    if has_checkpoint_payload(target):
        stage3_resume = latest_checkpoint(target)
    elif has_checkpoint_payload(s3_init):
        stage3_resume = latest_checkpoint(s3_init)
    elif stage2_candidate.exists() and has_checkpoint_payload(stage2_candidate):
        stage3_init = str(stage2_candidate)
    else:
        stage3_resume = str(target)
elif target.parent.name == STAGE3_NAME:
    run_dir = target.parent.parent
    stage1_dir = str(run_dir / STAGE1_NAME)
    stage2_dir = str(run_dir / STAGE2_NAME)
    stage3_dir = str(target.parent)
    stage3_resume = str(target)
elif target.parent.name.startswith("checkpoint-") and target.parent.parent.name == STAGE3_NAME:
    run_dir = target.parent.parent.parent
    stage1_dir = str(run_dir / STAGE1_NAME)
    stage2_dir = str(run_dir / STAGE2_NAME)
    stage3_dir = str(target.parent.parent)
    stage3_resume = str(target)
elif target.is_dir() and target.name == STAGE2_NAME:
    run_dir = target.parent
    stage1_candidate = run_dir / STAGE1_NAME
    s2_init = run_dir / "stage2_init_weights"
    stage1_dir = str(stage1_candidate) if stage1_candidate.exists() else ""
    stage2_dir = str(target)
    stage3_dir = str(run_dir / STAGE3_NAME)
    run_stage2 = "1"
    run_stage3 = "1"
    if has_checkpoint_payload(target):
        stage2_resume = latest_checkpoint(target)
    elif has_checkpoint_payload(s2_init):
        stage2_resume = latest_checkpoint(s2_init)
    elif stage1_candidate.exists() and has_checkpoint_payload(stage1_candidate):
        stage2_init = str(stage1_candidate)
    else:
        stage2_resume = str(target)
elif target.parent.name == STAGE2_NAME:
    run_dir = target.parent.parent
    stage1_candidate = run_dir / STAGE1_NAME
    stage1_dir = str(stage1_candidate) if stage1_candidate.exists() else ""
    stage2_dir = str(target.parent)
    stage3_dir = str(run_dir / STAGE3_NAME)
    stage2_resume = str(target)
    run_stage2 = "1"
    run_stage3 = "1"
elif target.parent.name.startswith("checkpoint-") and target.parent.parent.name == STAGE2_NAME:
    run_dir = target.parent.parent.parent
    stage1_candidate = run_dir / STAGE1_NAME
    stage1_dir = str(stage1_candidate) if stage1_candidate.exists() else ""
    stage2_dir = str(target.parent.parent)
    stage3_dir = str(run_dir / STAGE3_NAME)
    stage2_resume = str(target)
    run_stage2 = "1"
    run_stage3 = "1"
elif target.is_dir() and target.name == STAGE1_NAME:
    run_dir = target.parent
    stage1_dir = str(target)
    stage2_dir = str(run_dir / STAGE2_NAME)
    stage3_dir = str(run_dir / STAGE3_NAME)
    stage1_resume = latest_checkpoint(target)
    run_stage1 = "1"
    run_stage2 = "1"
    run_stage3 = "1"
elif target.parent.name == STAGE1_NAME:
    run_dir = target.parent.parent
    stage1_dir = str(target.parent)
    stage2_dir = str(run_dir / STAGE2_NAME)
    stage3_dir = str(run_dir / STAGE3_NAME)
    stage1_resume = str(target)
    run_stage1 = "1"
    run_stage2 = "1"
    run_stage3 = "1"
elif target.parent.name.startswith("checkpoint-") and target.parent.parent.name == STAGE1_NAME:
    run_dir = target.parent.parent.parent
    stage1_dir = str(target.parent.parent)
    stage2_dir = str(run_dir / STAGE2_NAME)
    stage3_dir = str(run_dir / STAGE3_NAME)
    stage1_resume = str(target)
    run_stage1 = "1"
    run_stage2 = "1"
    run_stage3 = "1"
elif target.is_dir() and has_checkpoint_payload(target):
    run_dir = target.parent
    stage2_dir = str(target)
    stage3_dir = str(run_dir / STAGE3_NAME)
    stage2_resume = str(target)
    run_stage2 = "1"
    run_stage3 = "1"
elif target.is_file():
    run_dir = target.parent
    stage2_dir = str(run_dir)
    stage3_dir = str(run_dir / STAGE3_NAME)
    stage2_resume = str(target)
    run_stage2 = "1"
    run_stage3 = "1"
else:
    raise ValueError("Unsupported resume path layout. Expected a run dir, stage dir, checkpoint dir, or qwenpose_checkpoint.pt file.")

print(shell_assign("RESUME_RESOLVED_RUN_DIR", str(run_dir)))
print(shell_assign("RESUME_RESOLVED_STAGE1_OUTPUT_DIR", stage1_dir))
print(shell_assign("RESUME_RESOLVED_STAGE2_OUTPUT_DIR", stage2_dir))
print(shell_assign("RESUME_RESOLVED_STAGE3_OUTPUT_DIR", stage3_dir))
print(shell_assign("RESUME_RESOLVED_STAGE1_RESUME", stage1_resume))
print(shell_assign("RESUME_RESOLVED_STAGE2_RESUME", stage2_resume))
print(shell_assign("RESUME_RESOLVED_STAGE3_RESUME", stage3_resume))
print(shell_assign("RESUME_RESOLVED_STAGE2_INIT_CHECKPOINT", stage2_init))
print(shell_assign("RESUME_RESOLVED_STAGE3_INIT_CHECKPOINT", stage3_init))
print(shell_assign("RESUME_RESOLVED_RUN_STAGE1", run_stage1))
print(shell_assign("RESUME_RESOLVED_RUN_STAGE2", run_stage2))
print(shell_assign("RESUME_RESOLVED_RUN_STAGE3", run_stage3))
print(shell_assign("RESUME_RESOLVED_APPEND_LOG_FILE", latest_log_file(run_dir)))
print_resume_defaults("STAGE1", stage1_resume)
print_resume_defaults("STAGE2", stage2_resume)
print_resume_defaults("STAGE3", stage3_resume)
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
    if [[ ! -v RUN_STAGE2 ]]; then
      if [[ "${RESUME_RESOLVED_RUN_STAGE3:-0}" == "1" ]]; then RUN_STAGE2=1
      elif [[ -n "${RESUME_RESOLVED_RUN_STAGE2:-}" ]]; then RUN_STAGE2="${RESUME_RESOLVED_RUN_STAGE2}"
      fi
    fi
    if [[ ! -v STAGE1_OUTPUT_DIR && -n "${RESUME_RESOLVED_STAGE1_OUTPUT_DIR:-}" ]]; then STAGE1_OUTPUT_DIR="${RESUME_RESOLVED_STAGE1_OUTPUT_DIR}"; fi
    if [[ ! -v STAGE2_OUTPUT_DIR ]]; then
      if [[ -n "${RESUME_RESOLVED_STAGE3_RESUME:-}" && "${RESUME_RESOLVED_STAGE3_RESUME}" != "none" ]]; then STAGE2_OUTPUT_DIR="${RESUME_RESOLVED_STAGE3_OUTPUT_DIR}"
      elif [[ -n "${RESUME_RESOLVED_STAGE2_OUTPUT_DIR:-}" ]]; then STAGE2_OUTPUT_DIR="${RESUME_RESOLVED_STAGE2_OUTPUT_DIR}"
      fi
    fi
    if [[ ! -v STAGE1_RESUME_FROM_CHECKPOINT && -n "${RESUME_RESOLVED_STAGE1_RESUME:-}" ]]; then STAGE1_RESUME_FROM_CHECKPOINT="${RESUME_RESOLVED_STAGE1_RESUME}"; fi
    if [[ ! -v STAGE2_RESUME_FROM_CHECKPOINT ]]; then
      if [[ -n "${RESUME_RESOLVED_STAGE3_RESUME:-}" && "${RESUME_RESOLVED_STAGE3_RESUME}" != "none" ]]; then STAGE2_RESUME_FROM_CHECKPOINT="${RESUME_RESOLVED_STAGE3_RESUME}"
      elif [[ -n "${RESUME_RESOLVED_STAGE2_RESUME:-}" ]]; then STAGE2_RESUME_FROM_CHECKPOINT="${RESUME_RESOLVED_STAGE2_RESUME}"
      fi
    fi
    if [[ ! -v STAGE2_INIT_CHECKPOINT && -n "${RESUME_RESOLVED_STAGE2_INIT_CHECKPOINT:-}" ]]; then STAGE2_INIT_CHECKPOINT="${RESUME_RESOLVED_STAGE2_INIT_CHECKPOINT}"; fi
    # Mid-epoch continuation requires the same sampler geometry. Restore it
    # from qwenpose_state.json unless the caller explicitly supplied an
    # override; explicit incompatible overrides are still rejected by Python.
    if [[ ! -v STAGE1_BATCH_SIZE && -n "${RESUME_RESOLVED_STAGE1_BATCH_SIZE:-}" ]]; then STAGE1_BATCH_SIZE="${RESUME_RESOLVED_STAGE1_BATCH_SIZE}"; fi
    if [[ ! -v STAGE1_GRAD_ACCUM_STEPS && -n "${RESUME_RESOLVED_STAGE1_GRAD_ACCUM_STEPS:-}" ]]; then STAGE1_GRAD_ACCUM_STEPS="${RESUME_RESOLVED_STAGE1_GRAD_ACCUM_STEPS}"; fi
    if [[ ! -v STAGE1_TRAIN_DATASETS && -n "${RESUME_RESOLVED_STAGE1_TRAIN_DATASETS:-}" ]]; then STAGE1_TRAIN_DATASETS="${RESUME_RESOLVED_STAGE1_TRAIN_DATASETS}"; fi
    if [[ ! -v STAGE2_BATCH_SIZE && -n "${RESUME_RESOLVED_STAGE2_BATCH_SIZE:-}" ]]; then STAGE2_BATCH_SIZE="${RESUME_RESOLVED_STAGE2_BATCH_SIZE}"; fi
    if [[ ! -v STAGE2_GRAD_ACCUM_STEPS && -n "${RESUME_RESOLVED_STAGE2_GRAD_ACCUM_STEPS:-}" ]]; then STAGE2_GRAD_ACCUM_STEPS="${RESUME_RESOLVED_STAGE2_GRAD_ACCUM_STEPS}"; fi
    if [[ ! -v STAGE2_TRAIN_DATASETS ]]; then
      if [[ -n "${RESUME_RESOLVED_STAGE3_TRAIN_DATASETS:-}" ]]; then STAGE2_TRAIN_DATASETS="${RESUME_RESOLVED_STAGE3_TRAIN_DATASETS}"
      elif [[ -n "${RESUME_RESOLVED_STAGE2_TRAIN_DATASETS:-}" ]]; then STAGE2_TRAIN_DATASETS="${RESUME_RESOLVED_STAGE2_TRAIN_DATASETS}"
      fi
    fi
    if [[ ! -v STAGE2_BATCH_SIZE && -n "${RESUME_RESOLVED_STAGE3_BATCH_SIZE:-}" ]]; then STAGE2_BATCH_SIZE="${RESUME_RESOLVED_STAGE3_BATCH_SIZE}"; fi
    if [[ ! -v STAGE2_GRAD_ACCUM_STEPS && -n "${RESUME_RESOLVED_STAGE3_GRAD_ACCUM_STEPS:-}" ]]; then STAGE2_GRAD_ACCUM_STEPS="${RESUME_RESOLVED_STAGE3_GRAD_ACCUM_STEPS}"; fi
    resume_mixing_strategy="${RESUME_RESOLVED_STAGE3_MIXING_STRATEGY:-${RESUME_RESOLVED_STAGE2_MIXING_STRATEGY:-${RESUME_RESOLVED_STAGE1_MIXING_STRATEGY:-}}}"
    resume_split="${RESUME_RESOLVED_STAGE3_SPLIT:-${RESUME_RESOLVED_STAGE2_SPLIT:-${RESUME_RESOLVED_STAGE1_SPLIT:-}}}"
    if [[ ! -v MIXING_STRATEGY && -n "${resume_mixing_strategy}" ]]; then MIXING_STRATEGY="${resume_mixing_strategy}"; fi
    if [[ ! -v SPLIT && -n "${resume_split}" ]]; then SPLIT="${resume_split}"; fi
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
# 物理 GPU 固定为 1、2、3。这里故意覆盖调用环境中已有的 CUDA_VISIBLE_DEVICES。
export CUDA_VISIBLE_DEVICES="1,2"
# 每个物理 GPU 启动一个训练进程；不读取外部 NPROC_PER_NODE。
export NPROC_PER_NODE=2
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
# OPEN_FILE_LIMIT：大 batch + 多 worker 会为共享 tensor 创建大量句柄；提升 tmux 子进程的 soft nofile。
OPEN_FILE_LIMIT="${OPEN_FILE_LIMIT:-65536}"
HARD_OPEN_FILE_LIMIT="$(ulimit -Hn)"
TARGET_OPEN_FILE_LIMIT="${OPEN_FILE_LIMIT}"
if [[ "${HARD_OPEN_FILE_LIMIT}" != "unlimited" ]] && (( TARGET_OPEN_FILE_LIMIT > HARD_OPEN_FILE_LIMIT )); then
  TARGET_OPEN_FILE_LIMIT="${HARD_OPEN_FILE_LIMIT}"
fi
if ! ulimit -S -n "${TARGET_OPEN_FILE_LIMIT}"; then
  echo "[WARN] unable to raise open-file soft limit to ${TARGET_OPEN_FILE_LIMIT}; current=$(ulimit -Sn)" >&2
fi
# QWENPOSE_MP_SHARING_STRATEGY：file_system 避免每个 DataLoader tensor storage 长期占用一个 FD。
export QWENPOSE_MP_SHARING_STRATEGY="${QWENPOSE_MP_SHARING_STRATEGY:-file_system}"

###############################################################################
# 数据集与 DataLoader 参数
###############################################################################

# DATASET_ROOT：数据集根目录，内部应包含 coco/mpii/crowdpose/refhuman 等子目录。
DATASET_ROOT="${DATASET_ROOT:-datasets}"
# SPLIT：训练 split。COCO 会映射到 train2017 等内部 split。
SPLIT="${SPLIT:-train}"
# MIXING_STRATEGY：多数据集混合方式。interleave 会按数据集交错采样。
MIXING_STRATEGY="${MIXING_STRATEGY:-interleave}"
# DATASET_MIX_WEIGHTS：Stage2 的每-epoch 数据集遍历倍率；默认四个数据集各完整遍历一次。
# 小数倍率会逐 epoch 续接上次切片，例如 0.5 依次训练前半与后半。该变量保留为 Stage2 的便捷全局别名。
DATASET_MIX_WEIGHTS="${DATASET_MIX_WEIGHTS:-coco:1,mpii:1,crowdpose:1,refhuman:1}"
# STAGE1_DATASET_MIX_WEIGHTS：Stage1 默认四个数据集各完整遍历一次。
STAGE1_DATASET_MIX_WEIGHTS="${STAGE1_DATASET_MIX_WEIGHTS:-coco:1,mpii:1,crowdpose:1,refhuman:1}"
# STAGE2_DATASET_MIX_WEIGHTS：可独立覆盖合并 Stage2 比例。
STAGE2_DATASET_MIX_WEIGHTS="${STAGE2_DATASET_MIX_WEIGHTS:-${DATASET_MIX_WEIGHTS}}"
# MAX_INSTANCES：每张图最多保留/训练的人体实例数。
MAX_INSTANCES="${MAX_INSTANCES:-80}"
# MAX_SAMPLES_PER_DATASET：每个数据集最多样本数；空值表示不截断。
MAX_SAMPLES_PER_DATASET="${MAX_SAMPLES_PER_DATASET:-}"
# REFHUMAN_MAX_CAPTIONS_PER_INSTANCE：RefHuman 每个人每个 epoch 使用的描述数；默认 1，描述会跨 epoch 随机轮换。
REFHUMAN_MAX_CAPTIONS_PER_INSTANCE="${REFHUMAN_MAX_CAPTIONS_PER_INSTANCE:-1}"
# RECORD_CACHE_DIR：样本 record 缓存目录，加速重复启动。
RECORD_CACHE_DIR="${RECORD_CACHE_DIR:-.cache/qwenpose_records}"
# DISABLE_RECORD_CACHE：是否禁用 record 缓存。1 表示每次重新解析数据集。
DISABLE_RECORD_CACHE="${DISABLE_RECORD_CACHE:-0}"
# NUM_WORKERS：每个训练 rank 的 DataLoader worker 数。增强在原图上执行，默认 2 以便与 GPU 计算重叠。
NUM_WORKERS="${NUM_WORKERS:-2}"
# PREFETCH_FACTOR：DataLoader prefetch factor，仅 NUM_WORKERS>0 时生效。
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
# DISABLE_HOMOGENEOUS_BATCHES：是否关闭同数据集 batch 采样。0 表示启用同源 batch。
DISABLE_HOMOGENEOUS_BATCHES="${DISABLE_HOMOGENEOUS_BATCHES:-0}"
# DISABLE_VISION_TOKEN_BALANCING：是否关闭跨 rank 视觉 token 成本均衡。默认启用。
DISABLE_VISION_TOKEN_BALANCING="${DISABLE_VISION_TOKEN_BALANCING:-0}"

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
# LOCATE_IMAGE_TOKEN_LIMIT：每张图 raw MoonViT patch token 上限。
LOCATE_IMAGE_TOKEN_LIMIT="${LOCATE_IMAGE_TOKEN_LIMIT:-4096}"
# LOCATE_BATCH_TOKEN_LIMIT：可选的两阶段全局覆盖值；留空时按各阶段 batch size 自动计算。
LOCATE_BATCH_TOKEN_LIMIT="${LOCATE_BATCH_TOKEN_LIMIT:-}"
# LOCATE_FEATURE_REFINER_LAYERS：Locate 特征 refiner 层数。
LOCATE_FEATURE_REFINER_LAYERS="${LOCATE_FEATURE_REFINER_LAYERS:-2}"
# LOCATE_FEATURE_REFINER_BOTTLENECK_DIM：Locate 特征 refiner bottleneck 隐藏维度。
LOCATE_FEATURE_REFINER_BOTTLENECK_DIM="${LOCATE_FEATURE_REFINER_BOTTLENECK_DIM:-256}"
# LOCATE_FEATURE_REFINER_INIT_SCALE：refiner 残差初始化尺度，越小越稳。
LOCATE_FEATURE_REFINER_INIT_SCALE="${LOCATE_FEATURE_REFINER_INIT_SCALE:-0.05}"
# LOCATE_LORA_R：LocateAnything 语言/主干 LoRA rank。
LOCATE_LORA_R="${LOCATE_LORA_R:-32}"
# LOCATE_LORA_ALPHA：LocateAnything 语言/主干 LoRA alpha。
LOCATE_LORA_ALPHA="${LOCATE_LORA_ALPHA:-64}"
# LOCATE_LORA_DROPOUT：LocateAnything 语言/主干 LoRA dropout。
LOCATE_LORA_DROPOUT="${LOCATE_LORA_DROPOUT:-0.0}"
# LOCATE_VISION_LORA_R：LocateAnything vision 分支 LoRA rank。
LOCATE_VISION_LORA_R="${LOCATE_VISION_LORA_R:-16}"
# LOCATE_VISION_LORA_ALPHA：LocateAnything vision 分支 LoRA alpha。
LOCATE_VISION_LORA_ALPHA="${LOCATE_VISION_LORA_ALPHA:-32}"
# LOCATE_VISION_LORA_DROPOUT：LocateAnything vision 分支 LoRA dropout。
LOCATE_VISION_LORA_DROPOUT="${LOCATE_VISION_LORA_DROPOUT:-0.0}"

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
# POSE_DROPOUT：快速收敛阶段关闭 Transformer dropout。
POSE_DROPOUT="${POSE_DROPOUT:-0.0}"
# BOX_CONDITION_SCALE：refined person box 送入关键点 decoder 前的上下文扩张比例。
BOX_CONDITION_SCALE="${BOX_CONDITION_SCALE:-1.25}"
# POSE_COORDINATE_INIT：learned_spread 使用非人体形状的可学习分散 reference，避免所有关节重合采样。
# box_center 仅用于消融，schema_prior 仅用于复现旧 checkpoint。
POSE_COORDINATE_INIT="${POSE_COORDINATE_INIT:-learned_spread}"
# SCHEMA_JOINT_PRIORS_PATH：仅 legacy schema_prior 模式使用。
SCHEMA_JOINT_PRIORS_PATH="${SCHEMA_JOINT_PRIORS_PATH:-${PROJECT_ROOT}/configs/schema_joint_priors.json}"
# POSE_ROI_SIZE：P2/P3/P4 每层 ROIAlign 的输出边长。
POSE_ROI_SIZE="${POSE_ROI_SIZE:-16}"
# POSE_PYRAMID_CHANNELS：P2/P3/P4 统一通道数；越大容量和显存越高，128 是速度/精度折中。
POSE_PYRAMID_CHANNELS="${POSE_PYRAMID_CHANNELS:-128}"
# POSE_PYRAMID_BLOCKS：P3/P4 每层深度可分离残差块数量；增加可提升纹理建模但显著增加 800 输入计算量。
POSE_PYRAMID_BLOCKS="${POSE_PYRAMID_BLOCKS:-3}"
# HUMAN_DECODER_LAYERS：person-query 人体框 decoder 层数；每层都会更新框和 objectness，推荐 2。
HUMAN_DECODER_LAYERS="${HUMAN_DECODER_LAYERS:-2}"
# DEFORMABLE_POINTS：每个特征尺度、每个 query 的可学习采样点数；总采样数为 3 个尺度乘该值。
DEFORMABLE_POINTS="${DEFORMABLE_POINTS:-4}"
# DEFORMABLE_MIN_RADIUS_CELLS：小目标采样最小覆盖特征格数；2.0 可避免采样点挤在同一亚像素位置。
DEFORMABLE_MIN_RADIUS_CELLS="${DEFORMABLE_MIN_RADIUS_CELLS:-2.0}"
# REF_TEXT_SCALE：RefHuman 文本向量注入 human query、instance query 和 joint query 的缩放系数；推荐 0.1～0.3。
REF_TEXT_SCALE="${REF_TEXT_SCALE:-0.2}"
# ENABLE_BOX_DENOISING：训练期是否启用仅框 DN；1 可加速框 decoder 收敛，推理时自动不创建 DN query。
ENABLE_BOX_DENOISING="${ENABLE_BOX_DENOISING:-1}"
# MAX_DN_QUERIES：每张图 DN query 上限，包含正负样本；人多时会自动减少 group 数，显存敏感可降到 48。
MAX_DN_QUERIES="${MAX_DN_QUERIES:-96}"
# MAX_DN_GROUPS：同一 GT 重复构造的 DN group 上限；更多 group 增强监督但增加 human decoder 计算量。
MAX_DN_GROUPS="${MAX_DN_GROUPS:-4}"
# DN_POSITIVE_NOISE：正 DN 框中心/宽高扰动强度；0.40 表示需要学习中等幅度框修复。
DN_POSITIVE_NOISE="${DN_POSITIVE_NOISE:-0.40}"
# DN_NEGATIVE_NOISE：负 DN 框最大扰动强度；越大负样本越容易，过小会产生与正框混淆的错误监督。
DN_NEGATIVE_NOISE="${DN_NEGATIVE_NOISE:-1.00}"
# ENABLE_KEYPOINT_DENOISING：训练期启用 DETRPose 风格的 OKS/KS 关键点去噪；推理时不创建去噪 query。
ENABLE_KEYPOINT_DENOISING="${ENABLE_KEYPOINT_DENOISING:-1}"
# MAX_KEYPOINT_DN_QUERIES：每图完整人体骨架去噪 query 上限（正负样本合计）；独立 pose 分支，显存敏感时可调低。
MAX_KEYPOINT_DN_QUERIES="${MAX_KEYPOINT_DN_QUERIES:-16}"
# MAX_KEYPOINT_DN_GROUPS：同一批 GT 骨架重复采样的去噪组上限；不同组在 self-attention 中严格隔离。
MAX_KEYPOINT_DN_GROUPS="${MAX_KEYPOINT_DN_GROUPS:-2}"
# 正样本 KS 位于 [0.5,1.0]，监督带噪骨架恢复到 GT 关键点。
KEYPOINT_DN_POSITIVE_KS_MIN="${KEYPOINT_DN_POSITIVE_KS_MIN:-0.5}"
KEYPOINT_DN_POSITIVE_KS_MAX="${KEYPOINT_DN_POSITIVE_KS_MAX:-1.0}"
# 负样本 KS 位于 [0.1,0.5]，只监督关键点/实例质量为低，不回归坐标且不改变 box objectness。
KEYPOINT_DN_NEGATIVE_KS_MIN="${KEYPOINT_DN_NEGATIVE_KS_MIN:-0.1}"
KEYPOINT_DN_NEGATIVE_KS_MAX="${KEYPOINT_DN_NEGATIVE_KS_MAX:-0.5}"
# ENABLE_PERSON_CONFIDENCE_HEAD：姿态质量分数，与框 objectness 相乘形成最终实例分数。
ENABLE_PERSON_CONFIDENCE_HEAD="${ENABLE_PERSON_CONFIDENCE_HEAD:-1}"
# DISABLE_REFINEMENT：是否关闭 keypoint refinement。
DISABLE_REFINEMENT="${DISABLE_REFINEMENT:-0}"

###############################################################################
# 优化器 / 学习率 / 训练控制参数
###############################################################################

# LOCATE_VISION_SCALE：视觉 LoRA 学习率相对当前 stage Pose LR 的倍率；默认 0.05，例如 1e-4→5e-6。
LOCATE_VISION_SCALE="${LOCATE_VISION_SCALE:-0.05}"
# LOCATE_LLM_SCALE：LLM LoRA 学习率倍率；语言模型更敏感，默认 0.01，例如 1e-4→1e-6。
LOCATE_LLM_SCALE="${LOCATE_LLM_SCALE:-0.01}"
# LOCATE_VISION_LAYERS：selective_lora 解冻的 MoonViT block，支持 15-26 或逗号列表；默认最后 12 层。
LOCATE_VISION_LAYERS="${LOCATE_VISION_LAYERS:-15-26}"
# LOCATE_LLM_LAYERS：selective_lora 解冻的 Qwen2.5 decoder 层；默认最后 4 层 32-35。
LOCATE_LLM_LAYERS="${LOCATE_LLM_LAYERS:-32-35}"
# LOCATE_VISION_MODULES：每个选中视觉 block 中训练的 LoRA 投影名；MoonViT 使用 fused wqkv、wo、fc0、fc1。
LOCATE_VISION_MODULES="${LOCATE_VISION_MODULES:-wqkv,wo,fc0,fc1}"
# LOCATE_LLM_MODULES：每个选中 LLM 层中训练的 LoRA 投影；q_proj,v_proj 是低风险默认组合。
LOCATE_LLM_MODULES="${LOCATE_LLM_MODULES:-q_proj,v_proj}"
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
SAVE_EVERY="${SAVE_EVERY:-2000}"
# SAVE_TOTAL_LIMIT：每个 stage 最多保留多少个 checkpoint。
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
# SEED：随机种子。
SEED="${SEED:-42}"
# VISUALIZE_EVERY：训练可视化保存间隔，0 表示关闭。
VISUALIZE_EVERY="${VISUALIZE_EVERY:-100}"
# VISUALIZE_MAX_INSTANCES：每张可视化最多绘制实例数。
VISUALIZE_MAX_INSTANCES="${VISUALIZE_MAX_INSTANCES:-8}"
# VISUALIZE_MIN_GT_AREA_RATIO：只保存至少有一个清晰 pose GT 的图，默认人体框需占整图 0.5%。
VISUALIZE_MIN_GT_AREA_RATIO="${VISUALIZE_MIN_GT_AREA_RATIO:-0.005}"
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
W_OKS="${W_OKS:-0.5}"
# W_COORD：按 dataset-native loss box 归一化的坐标回归权重。
W_COORD="${W_COORD:-3.0}"
# W_IMAGE_COORD：整图归一化坐标回归权重，不依赖各数据集 box 协议。
W_IMAGE_COORD="${W_IMAGE_COORD:-5.0}"
# W_KEYPOINT_CONFIDENCE：逐关键点定位置信度 BCE 权重。
W_KEYPOINT_CONFIDENCE="${W_KEYPOINT_CONFIDENCE:-0.1}"
# W_PERSON_CONFIDENCE：实例级 evaluator-OKS 质量权重；Stage2 未匹配框为零质量负样本。
W_PERSON_CONFIDENCE="${W_PERSON_CONFIDENCE:-0.5}"
# W_REF_MATCH：RefHuman 文本与候选人体的独立分类损失权重；只在 ref_target 成功匹配到候选时生效。
W_REF_MATCH="${W_REF_MATCH:-1.0}"
# W_HARD_JOINT：hard-joint mining loss 权重，默认关闭。
W_HARD_JOINT="${W_HARD_JOINT:-0.0}"
# HARD_JOINT_FRACTION：hard-joint mining 选取最难关节点比例。
HARD_JOINT_FRACTION="${HARD_JOINT_FRACTION:-0.2}"
# W_COARSE_COORD：coarse 的完整 OKS+框归一化坐标+整图坐标目标的外层权重。
W_COARSE_COORD="${W_COARSE_COORD:-0.5}"
# W_DEFORM_COORD：多尺度 deformable attention 后坐标监督权重；通常略高于 coarse、低于最终 refinement。
W_DEFORM_COORD="${W_DEFORM_COORD:-0.75}"
# W_REFINE_COORDS：每个 refinement step 的坐标深监督权重，逗号分隔。
W_REFINE_COORDS="${W_REFINE_COORDS:-0.75,1.0,1.25}"
# W_BOX_OBJECTNESS：人体 query 是否对应真实人的 focal objectness loss 权重。
W_BOX_OBJECTNESS="${W_BOX_OBJECTNESS:-1.0}"
# W_BOX_L1：refined box 对 GT box 的绝对归一化 L1 权重；DETR 系列常用较大权重 5。
W_BOX_L1="${W_BOX_L1:-5.0}"
# W_BOX_GIOU：refined box 的 GIoU 权重，补充 L1 对重叠关系不敏感的问题。
W_BOX_GIOU="${W_BOX_GIOU:-2.0}"
# W_BOX_RELATIVE：按 GT 宽高归一化的中心/尺寸误差权重，专门增强小目标框的训练信号。
W_BOX_RELATIVE="${W_BOX_RELATIVE:-1.0}"
# W_BOX_DN：正负 DN query 整体 loss 的外层权重；0 可保留结构但关闭 DN 监督。
W_BOX_DN="${W_BOX_DN:-0.5}"
# W_KEYPOINT_DN：关键点去噪内部加权 loss 的外层权重；终端 kdn 显示其对总 loss 的实际贡献。
W_KEYPOINT_DN="${W_KEYPOINT_DN:-1.0}"

###############################################################################
# 两阶段开关与 stage-specific 参数
###############################################################################

# RUN_STAGE1：是否执行 Stage1；1 执行、0 跳过。传入 STAGES 或位置阶段参数时会重新计算该值。
RUN_STAGE1="${RUN_STAGE1:-1}"
# RUN_STAGE2：是否执行 Stage2；单独运行时可通过 STAGE2_INIT_CHECKPOINT 指定 Stage1 权重。
RUN_STAGE2="${RUN_STAGE2:-1}"
# STAGES：逗号分隔的阶段选择，如 stage1,stage2 或 all；优先使用命令行位置参数生成的列表。
STAGES="${STAGES:-${CLI_STAGE_SELECTION}}"
if [[ -n "${STAGES}" ]]; then
  RUN_STAGE1=0
  RUN_STAGE2=0
  normalized_stages="${STAGES// /,}"
  IFS=',' read -r -a selected_stages <<< "${normalized_stages}"
  for raw_stage in "${selected_stages[@]}"; do
    stage_name="${raw_stage,,}"
    [[ -z "${stage_name}" ]] && continue
    case "${stage_name}" in
      all) RUN_STAGE1=1; RUN_STAGE2=1 ;;
      stage1) RUN_STAGE1=1 ;;
      stage2) RUN_STAGE2=1 ;;
      stage3) RUN_STAGE2=1 ;;  # legacy alias after merging the old stage 3 into stage 2
      *) echo "Unknown stage selection: ${raw_stage}. Use stage1, stage2, or all." >&2; exit 1 ;;
    esac
  done
fi

# STAGE1_OUTPUT_DIR：Stage1 checkpoint、状态和可视化输出目录；建议保持默认层级便于自动续训解析。
STAGE1_OUTPUT_DIR="${STAGE1_OUTPUT_DIR:-${OUTPUT_DIR}/stage1_freeze_locate_person_queries}"
# STAGE2_OUTPUT_DIR：选择性解冻后的统一检测、grounding 与姿态训练输出目录。
STAGE2_OUTPUT_DIR="${STAGE2_OUTPUT_DIR:-${OUTPUT_DIR}/stage2_unfreeze_locate_person_queries}"
# STAGE2_INIT_WEIGHTS_DIR：把 Stage1 checkpoint 转成仅权重初始化包时的临时目录，不包含 optimizer 状态。
STAGE2_INIT_WEIGHTS_DIR="${STAGE2_INIT_WEIGHTS_DIR:-${OUTPUT_DIR}/stage2_init_weights}"

# STAGE1_TRAIN_DATASETS：冻结 LocateAnything 时也从第一步联合训练 RefHuman。
STAGE1_TRAIN_DATASETS="${STAGE1_TRAIN_DATASETS:-coco,mpii,crowdpose,refhuman}"
# STAGE2_TRAIN_DATASETS：合并后的 Stage2 同时训练通用姿态与 RefHuman 指代姿态。
STAGE2_TRAIN_DATASETS="${STAGE2_TRAIN_DATASETS:-coco,mpii,crowdpose,refhuman}"

# STAGE1_TRAIN_MODE：Stage1 终止方式，epochs 表示按 epoch，steps 表示按 optimizer step。
STAGE1_TRAIN_MODE="${STAGE1_TRAIN_MODE:-epochs}"
# STAGE1_EPOCHS：Stage1 epoch 数；仅 STAGE1_TRAIN_MODE=epochs 时生效。
STAGE1_EPOCHS="${STAGE1_EPOCHS:-30}"
# STAGE1_MAX_STEPS：Stagehf_cache1 最大 optimizer step；steps 模式必填，epochs 模式会自动改为 0。
STAGE1_MAX_STEPS="${STAGE1_MAX_STEPS:-60000}"
# STAGE2_EPOCHS：合并后的 Stage2 epoch 数。
STAGE2_EPOCHS="${STAGE2_EPOCHS:-25}"
# STAGE2_MAX_STEPS：Stage2 最大 step；0 表示不额外限制，由 epoch 决定。
STAGE2_MAX_STEPS="${STAGE2_MAX_STEPS:-0}"

# STAGE1_BATCH_SIZE：Stage1 每 rank micro-batch；800×800 P2=200²，默认 4 以控制显存。
STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-2}"
# STAGE2_BATCH_SIZE：Stage2 每 rank micro-batch；完整 3B 多模态特征主干默认使用 1。
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-2}"
# STAGE1_GRAD_ACCUM_STEPS：Stage1 梯度累积步数；4 卡时 4×8×4=128 有效全局 batch。
STAGE1_GRAD_ACCUM_STEPS="${STAGE1_GRAD_ACCUM_STEPS:-1}"
# STAGE2_GRAD_ACCUM_STEPS：Stage2 梯度累积步数，用于在 micro-batch=1 时稳定优化。
STAGE2_GRAD_ACCUM_STEPS="${STAGE2_GRAD_ACCUM_STEPS:-4}"
# STAGE1_LOCATE_BATCH_TOKEN_LIMIT：Stage1 单 rank 视觉 token 总预算；默认 batch×3072，用于动态成本均衡。
STAGE1_LOCATE_BATCH_TOKEN_LIMIT="${STAGE1_LOCATE_BATCH_TOKEN_LIMIT:-${LOCATE_BATCH_TOKEN_LIMIT:-$((STAGE1_BATCH_SIZE * 3072))}}"
# STAGE2_LOCATE_BATCH_TOKEN_LIMIT：Stage2 单 rank token 总预算；完整模型默认 batch×4096。
STAGE2_LOCATE_BATCH_TOKEN_LIMIT="${STAGE2_LOCATE_BATCH_TOKEN_LIMIT:-${LOCATE_BATCH_TOKEN_LIMIT:-$((STAGE2_BATCH_SIZE * 4096))}}"

# STAGE1_LR：Stage1 Pose pyramid、box decoder 和 pose decoder 基础学习率；从头训练默认 2e-4。
STAGE1_LR="${STAGE1_LR:-2e-4}"
# STAGE2_LR：Stage2 Pose 侧基础学习率；视觉/LLM LoRA 会再乘各自 scale，默认 1e-4。
STAGE2_LR="${STAGE2_LR:-1e-4}"

# STAGE2_FREEZE_LOCATE：Stage2 是否冻结完整 Locate；0 配合 selective_lora 只解冻选中的 LoRA 参数。
STAGE2_FREEZE_LOCATE="${STAGE2_FREEZE_LOCATE:-0}"
# PRUNE_LOCATE_GENERATION：当前 person-query 方案不做 token 坐标生成；跳过重复 lm_head 权重并禁用 KV cache。
PRUNE_LOCATE_GENERATION="${PRUNE_LOCATE_GENERATION:-1}"
# STAGE2_LOCATE_TRAIN_SCOPE：Stage2 Locate 可训练范围；selective_lora 只训练指定视觉/LLM 层和模块。
STAGE2_LOCATE_TRAIN_SCOPE="${STAGE2_LOCATE_TRAIN_SCOPE:-selective_lora}"
# STAGE2_LOCATE_GRADIENT_CHECKPOINTING：Stage2 是否启用完整模型 gradient checkpointing；1 省显存但增加计算。
STAGE2_LOCATE_GRADIENT_CHECKPOINTING="${STAGE2_LOCATE_GRADIENT_CHECKPOINTING:-${LOCATE_GRADIENT_CHECKPOINTING}}"

# STAGE1_RESUME_FROM_CHECKPOINT：Stage1 完整续训路径；none 表示新训练，续训会恢复 optimizer/step/RNG。
STAGE1_RESUME_FROM_CHECKPOINT="${STAGE1_RESUME_FROM_CHECKPOINT:-none}"
# STAGE2_RESUME_FROM_CHECKPOINT：Stage2 完整续训路径；与 INIT_CHECKPOINT 不同，它会恢复 Stage2 训练状态。
STAGE2_RESUME_FROM_CHECKPOINT="${STAGE2_RESUME_FROM_CHECKPOINT:-none}"
# STAGE2_INIT_CHECKPOINT：Stage2 仅权重初始化来源；通常指向 Stage1 输出，不恢复 Stage1 optimizer。
STAGE2_INIT_CHECKPOINT="${STAGE2_INIT_CHECKPOINT:-}"
# STAGE2_INIT_FROM_STAGE1：未显式指定 Stage2 checkpoint 时是否自动从同一 run 的 Stage1 初始化。
STAGE2_INIT_FROM_STAGE1="${STAGE2_INIT_FROM_STAGE1:-1}"
if [[ -n "${CLI_RESUME_PATH}" ]]; then
  if [[ "${STAGE2_RESUME_FROM_CHECKPOINT}" == "none" && -n "${RESUME_RESOLVED_STAGE2_RESUME:-}" ]]; then
    STAGE2_RESUME_FROM_CHECKPOINT="${RESUME_RESOLVED_STAGE2_RESUME}"
  fi
  if [[ "${STAGE2_RESUME_FROM_CHECKPOINT}" == "none" && -n "${RESUME_RESOLVED_STAGE3_RESUME:-}" && "${RESUME_RESOLVED_STAGE3_RESUME}" != "none" ]]; then
    # Legacy Stage3 has the same unified architecture and now resumes as merged Stage2.
    STAGE2_RESUME_FROM_CHECKPOINT="${RESUME_RESOLVED_STAGE3_RESUME}"
    STAGE2_OUTPUT_DIR="${RESUME_RESOLVED_STAGE3_OUTPUT_DIR:-${STAGE2_OUTPUT_DIR}}"
  fi
  if [[ -z "${STAGE2_INIT_CHECKPOINT}" && -n "${RESUME_RESOLVED_STAGE2_INIT_CHECKPOINT:-}" ]]; then
    STAGE2_INIT_CHECKPOINT="${RESUME_RESOLVED_STAGE2_INIT_CHECKPOINT}"
  fi
fi

###############################################################################
# 参数校验
###############################################################################

# 在调用 torchrun 前检查可见 GPU 数，避免进程数与卡数不一致时到
# torch.cuda.set_device(local_rank) 才报难以理解的 invalid device ordinal。
IFS=',' read -r -a locatepose_visible_gpus <<< "${CUDA_VISIBLE_DEVICES}"
if (( ${#locatepose_visible_gpus[@]} != NPROC_PER_NODE )); then
  echo "GPU configuration mismatch: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} exposes ${#locatepose_visible_gpus[@]} GPUs, but NPROC_PER_NODE=${NPROC_PER_NODE}." >&2
  exit 1
fi

require_positive_int() { local name="$1" value="$2"; if ! [[ "${value}" =~ ^[0-9]+$ ]] || (( value <= 0 )); then echo "${name} must be a positive integer, got: ${value}" >&2; exit 1; fi; }
require_nonnegative_int() { local name="$1" value="$2"; if ! [[ "${value}" =~ ^[0-9]+$ ]]; then echo "${name} must be a non-negative integer, got: ${value}" >&2; exit 1; fi; }
require_bool() { local name="$1" value="$2"; if [[ "${value}" != "0" && "${value}" != "1" ]]; then echo "${name} must be 0 or 1, got: ${value}" >&2; exit 1; fi; }

case "${STAGE1_TRAIN_MODE}" in
  epochs)
    require_positive_int STAGE1_EPOCHS "${STAGE1_EPOCHS}"
    # Epoch mode is exclusive: do not let a stale/global MAX_STEPS cap Stage 1
    # or stretch its cosine scheduler beyond the requested epoch count.
    STAGE1_MAX_STEPS=0
    ;;
  steps)
    require_positive_int STAGE1_MAX_STEPS "${STAGE1_MAX_STEPS}"
    # train_pose.py treats epochs=0 with max_steps>0 as step-only training and
    # derives enough epochs from the actual dataloader length.
    STAGE1_EPOCHS=0
    ;;
  *)
    echo "STAGE1_TRAIN_MODE must be epochs or steps, got: ${STAGE1_TRAIN_MODE}" >&2
    exit 1
    ;;
esac

for spec in \
  "NPROC_PER_NODE:${NPROC_PER_NODE}" \
  "MAX_INSTANCES:${MAX_INSTANCES}" \
  "STAGE1_BATCH_SIZE:${STAGE1_BATCH_SIZE}" \
  "STAGE2_BATCH_SIZE:${STAGE2_BATCH_SIZE}" \
  "STAGE1_GRAD_ACCUM_STEPS:${STAGE1_GRAD_ACCUM_STEPS}" \
  "STAGE2_GRAD_ACCUM_STEPS:${STAGE2_GRAD_ACCUM_STEPS}" \
  "STAGE2_EPOCHS:${STAGE2_EPOCHS}" \
  "REFINEMENT_STEPS:${REFINEMENT_STEPS}" \
  "HUMAN_DECODER_LAYERS:${HUMAN_DECODER_LAYERS}" \
  "POSE_DECODER_LAYERS:${POSE_DECODER_LAYERS}" \
  "POSE_ROI_SIZE:${POSE_ROI_SIZE}" \
  "POSE_PYRAMID_CHANNELS:${POSE_PYRAMID_CHANNELS}" \
  "POSE_PYRAMID_BLOCKS:${POSE_PYRAMID_BLOCKS}" \
  "DEFORMABLE_POINTS:${DEFORMABLE_POINTS}" \
  "MAX_DN_GROUPS:${MAX_DN_GROUPS}" \
  "MAX_KEYPOINT_DN_GROUPS:${MAX_KEYPOINT_DN_GROUPS}" \
  "VISUALIZE_MAX_INSTANCES:${VISUALIZE_MAX_INSTANCES}"; do
  require_positive_int "${spec%%:*}" "${spec#*:}"
done

for spec in \
  "STAGE1_MAX_STEPS:${STAGE1_MAX_STEPS}" \
  "STAGE2_MAX_STEPS:${STAGE2_MAX_STEPS}" \
  "NUM_WORKERS:${NUM_WORKERS}" \
  "VISUALIZE_EVERY:${VISUALIZE_EVERY}" \
  "MAX_DN_QUERIES:${MAX_DN_QUERIES}" \
  "MAX_KEYPOINT_DN_QUERIES:${MAX_KEYPOINT_DN_QUERIES}"; do
  require_nonnegative_int "${spec%%:*}" "${spec#*:}"
done
if [[ "${POSE_COORDINATE_INIT}" != "learned_spread" && "${POSE_COORDINATE_INIT}" != "box_center" && "${POSE_COORDINATE_INIT}" != "schema_prior" ]]; then
  echo "POSE_COORDINATE_INIT must be learned_spread, box_center, or schema_prior, got: ${POSE_COORDINATE_INIT}" >&2
  exit 1
fi
for spec in \
  RUN_STAGE1 RUN_STAGE2 STAGE2_FREEZE_LOCATE STAGE2_INIT_FROM_STAGE1 \
  STAGE2_LOCATE_GRADIENT_CHECKPOINTING \
  LOCATE_GRADIENT_CHECKPOINTING AMP DRY_RUN_DATA PROGRESS_BAR SYNC_TIMING \
  DISABLE_BATCH_TRACE DISABLE_HOMOGENEOUS_BATCHES DISABLE_VISION_TOKEN_BALANCING DISABLE_REFINEMENT DISABLE_RECORD_CACHE \
  ENABLE_BOX_DENOISING ENABLE_KEYPOINT_DENOISING ENABLE_PERSON_CONFIDENCE_HEAD \
  PRUNE_LOCATE_GENERATION; do
  require_bool "${spec}" "${!spec}"
done

person_confidence_weight_positive="$(
  "${PYTHON}" -c 'import math, sys; value=float(sys.argv[1]); sys.exit(1) if not math.isfinite(value) or value < 0.0 else print(int(value > 0.0))' "${W_PERSON_CONFIDENCE}"
)" || {
  echo "W_PERSON_CONFIDENCE must be a valid non-negative number, got: ${W_PERSON_CONFIDENCE}" >&2
  exit 1
}
if [[ "${ENABLE_PERSON_CONFIDENCE_HEAD}" != "${person_confidence_weight_positive}" ]]; then
  echo "ENABLE_PERSON_CONFIDENCE_HEAD and positive W_PERSON_CONFIDENCE must be enabled together." >&2
  echo "Got ENABLE_PERSON_CONFIDENCE_HEAD=${ENABLE_PERSON_CONFIDENCE_HEAD}, W_PERSON_CONFIDENCE=${W_PERSON_CONFIDENCE}." >&2
  exit 1
fi

"${PYTHON}" - "${REF_TEXT_SCALE}" "${W_REF_MATCH}" <<'PY'
import math
import sys

for name, raw in zip(("REF_TEXT_SCALE", "W_REF_MATCH"), sys.argv[1:]):
    value = float(raw)
    if not math.isfinite(value) or value < 0.0:
        raise SystemExit(f"{name} must be finite and non-negative, got {raw!r}")
PY

"${PYTHON}" - \
  "${KEYPOINT_DN_POSITIVE_KS_MIN}" "${KEYPOINT_DN_POSITIVE_KS_MAX}" \
  "${KEYPOINT_DN_NEGATIVE_KS_MIN}" "${KEYPOINT_DN_NEGATIVE_KS_MAX}" \
  "${W_KEYPOINT_DN}" <<'PY'
import math
import sys

pos_min, pos_max, neg_min, neg_max, weight = map(float, sys.argv[1:])
for name, value in (
    ("KEYPOINT_DN_POSITIVE_KS_MIN", pos_min),
    ("KEYPOINT_DN_POSITIVE_KS_MAX", pos_max),
    ("KEYPOINT_DN_NEGATIVE_KS_MIN", neg_min),
    ("KEYPOINT_DN_NEGATIVE_KS_MAX", neg_max),
    ("W_KEYPOINT_DN", weight),
):
    if not math.isfinite(value):
        raise SystemExit(f"{name} must be finite")
if not 0.0 < pos_min <= pos_max <= 1.0:
    raise SystemExit("positive keypoint-DN KS range must satisfy 0 < min <= max <= 1")
if not 0.0 < neg_min <= neg_max <= 1.0:
    raise SystemExit("negative keypoint-DN KS range must satisfy 0 < min <= max <= 1")
if weight < 0.0:
    raise SystemExit("W_KEYPOINT_DN must be non-negative")
PY

"${PYTHON}" - "${LOCATE_VISION_SCALE}" "${LOCATE_LLM_SCALE}" "${LOCATE_VISION_LAYERS}" "${LOCATE_LLM_LAYERS}" <<'PY'
import math
import sys

for name, raw in zip(("LOCATE_VISION_SCALE", "LOCATE_LLM_SCALE"), sys.argv[1:3]):
    value = float(raw)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, got {raw!r}")

for name, spec in zip(("LOCATE_VISION_LAYERS", "LOCATE_LLM_LAYERS"), sys.argv[3:5]):
    selected = set()
    for raw_token in spec.split(","):
        token = raw_token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start < 0 or end < start:
                raise ValueError(f"invalid {name} range: {token!r}")
            selected.update(range(start, end + 1))
        else:
            index = int(token)
            if index < 0:
                raise ValueError(f"invalid {name} index: {token!r}")
            selected.add(index)
    if not selected:
        raise ValueError(f"{name} must select at least one layer")
PY
if [[ -z "${LOCATE_VISION_MODULES//,/}" || -z "${LOCATE_LLM_MODULES//,/}" ]]; then
  echo "LOCATE_VISION_MODULES and LOCATE_LLM_MODULES must be non-empty." >&2
  exit 1
fi

if [[ "${STAGE2_LOCATE_TRAIN_SCOPE}" != "frozen" && "${STAGE2_LOCATE_TRAIN_SCOPE}" != "vision_lora" && "${STAGE2_LOCATE_TRAIN_SCOPE}" != "all_lora" && "${STAGE2_LOCATE_TRAIN_SCOPE}" != "selective_lora" ]]; then
  echo "STAGE2_LOCATE_TRAIN_SCOPE must be frozen, vision_lora, all_lora, or selective_lora, got: ${STAGE2_LOCATE_TRAIN_SCOPE}" >&2
  exit 1
fi
if [[ "${DEVICE}" != "cuda" && "${ZERO_STAGE}" != "none" ]]; then
  echo "DEVICE=${DEVICE} cannot use DeepSpeed ${ZERO_STAGE}. Use ZERO_STAGE=none for CPU debugging." >&2
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
if [[ -n "${LOCATE_IMAGE_TOKEN_LIMIT}" ]]; then require_positive_int LOCATE_IMAGE_TOKEN_LIMIT "${LOCATE_IMAGE_TOKEN_LIMIT}"; fi
if [[ -n "${LOCATE_BATCH_TOKEN_LIMIT}" ]]; then require_positive_int LOCATE_BATCH_TOKEN_LIMIT "${LOCATE_BATCH_TOKEN_LIMIT}"; fi
require_positive_int STAGE1_LOCATE_BATCH_TOKEN_LIMIT "${STAGE1_LOCATE_BATCH_TOKEN_LIMIT}"
require_positive_int STAGE2_LOCATE_BATCH_TOKEN_LIMIT "${STAGE2_LOCATE_BATCH_TOKEN_LIMIT}"
if (( NPROC_PER_NODE > 1 )) || [[ "${ZERO_STAGE}" != "none" ]]; then
  [[ -n "${TORCHRUN}" ]] || { echo "torchrun not found. Set TORCHRUN=/path/to/torchrun or use ZERO_STAGE=none with one visible GPU." >&2; exit 1; }
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

# 单独启动 Stage 2 时，默认策略要求从 Stage 1 权重初始化。找不到来源就立即
# 给出可执行的报错，而不是静默使用随机 PoseHead 开始一次错误的正式训练。
if [[ "${RUN_STAGE2}" == "1" && "${RUN_STAGE1}" == "0" \
      && "${STAGE2_RESUME_FROM_CHECKPOINT}" == "none" \
      && -z "${STAGE2_INIT_CHECKPOINT}" \
      && "${STAGE2_INIT_FROM_STAGE1}" == "1" \
      && "${DRY_RUN_DATA}" == "0" ]] \
   && ! resume_target_has_checkpoint "${STAGE1_OUTPUT_DIR}"; then
  echo "Stage 2 requires Stage 1 weights, but no checkpoint was found at: ${STAGE1_OUTPUT_DIR}" >&2
  echo "Run: ${SCRIPT_PATH_REL} stage2 --STAGE2_INIT_CHECKPOINT /path/to/stage1_or_checkpoint" >&2
  echo "To intentionally train Stage 2 from base LocateAnything with a random PoseHead, pass --STAGE2_INIT_FROM_STAGE1 0." >&2
  exit 1
fi

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
    echo "Cannot initialize next stage; no checkpoint found in ${source_path}" >&2
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
payload["weight_only_init_from"] = str(resolved)
payload["stage2_weight_only_init_from"] = str(resolved)  # legacy metadata key

out = dest / "checkpoint-0"
out.mkdir(parents=True, exist_ok=True)
torch.save(payload, out / CHECKPOINT_PAYLOAD_NAME)
with (out / "qwenpose_state.json").open("w", encoding="utf-8") as f:
    json.dump({"step": 0, "checkpoint": str(out), "payload": CHECKPOINT_PAYLOAD_NAME, "deepspeed_tag": None, "training_state": None, "weight_only_init_from": str(resolved), "stage2_weight_only_init_from": str(resolved)}, f, indent=2, ensure_ascii=False)
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
  a+=(--dataset_root "${DATASET_ROOT}" --split "${SPLIT}" --max_instances "${MAX_INSTANCES}" --num_person_queries "${MAX_INSTANCES}")
  a+=(--refhuman_max_captions_per_instance "${REFHUMAN_MAX_CAPTIONS_PER_INSTANCE}")
  a+=(--num_workers "${NUM_WORKERS}" --prefetch_factor "${PREFETCH_FACTOR}")
  a+=(--mixing_strategy "${MIXING_STRATEGY}" --record_cache_dir "${RECORD_CACHE_DIR}")
  a+=(--locate_model_path "${LOCATE_MODEL_PATH}" --locate_dtype "${LOCATE_DTYPE}" --locate_attn_implementation "${LOCATE_ATTN_IMPLEMENTATION}")
  [[ "${PRUNE_LOCATE_GENERATION}" == "1" ]] && a+=(--prune_locate_generation)
  add_opt a --locate_image_token_limit "${LOCATE_IMAGE_TOKEN_LIMIT}"
  a+=(--locate_feature_refiner_layers "${LOCATE_FEATURE_REFINER_LAYERS}")
  a+=(--locate_feature_refiner_bottleneck_dim "${LOCATE_FEATURE_REFINER_BOTTLENECK_DIM}" --locate_feature_refiner_init_scale "${LOCATE_FEATURE_REFINER_INIT_SCALE}")
  a+=(--locate_lora_r "${LOCATE_LORA_R}" --locate_lora_alpha "${LOCATE_LORA_ALPHA}" --locate_lora_dropout "${LOCATE_LORA_DROPOUT}")
  a+=(--locate_vision_lora_r "${LOCATE_VISION_LORA_R}" --locate_vision_lora_alpha "${LOCATE_VISION_LORA_ALPHA}" --locate_vision_lora_dropout "${LOCATE_VISION_LORA_DROPOUT}")
  a+=(--hidden_dim "${HIDDEN_DIM}" --human_decoder_layers "${HUMAN_DECODER_LAYERS}" --pose_decoder_layers "${POSE_DECODER_LAYERS}" --refinement_steps "${REFINEMENT_STEPS}" --decoder_heads "${DECODER_HEADS}" --pose_dropout "${POSE_DROPOUT}")
  a+=(--box_condition_scale "${BOX_CONDITION_SCALE}" --pose_roi_size "${POSE_ROI_SIZE}")
  a+=(--image_size 800 --pose_pyramid_channels "${POSE_PYRAMID_CHANNELS}" --pose_pyramid_blocks "${POSE_PYRAMID_BLOCKS}")
  a+=(--deformable_points "${DEFORMABLE_POINTS}" --deformable_min_radius_cells "${DEFORMABLE_MIN_RADIUS_CELLS}")
  a+=(--pose_coordinate_init "${POSE_COORDINATE_INIT}")
  a+=(--ref_text_scale "${REF_TEXT_SCALE}")
  a+=(--max_dn_queries "${MAX_DN_QUERIES}" --max_dn_groups "${MAX_DN_GROUPS}" --dn_positive_noise "${DN_POSITIVE_NOISE}" --dn_negative_noise "${DN_NEGATIVE_NOISE}")
  [[ "${ENABLE_BOX_DENOISING}" == "0" ]] && a+=(--disable_box_denoising)
  a+=(--max_keypoint_dn_queries "${MAX_KEYPOINT_DN_QUERIES}" --max_keypoint_dn_groups "${MAX_KEYPOINT_DN_GROUPS}")
  a+=(--keypoint_dn_positive_ks_min "${KEYPOINT_DN_POSITIVE_KS_MIN}" --keypoint_dn_positive_ks_max "${KEYPOINT_DN_POSITIVE_KS_MAX}")
  a+=(--keypoint_dn_negative_ks_min "${KEYPOINT_DN_NEGATIVE_KS_MIN}" --keypoint_dn_negative_ks_max "${KEYPOINT_DN_NEGATIVE_KS_MAX}")
  [[ "${ENABLE_KEYPOINT_DENOISING}" == "0" ]] && a+=(--disable_keypoint_denoising)
  [[ "${ENABLE_PERSON_CONFIDENCE_HEAD}" == "1" ]] && a+=(--enable_person_confidence_head)
  a+=(--schema_joint_priors_path "${SCHEMA_JOINT_PRIORS_PATH}")
  a+=(--locate_vision_scale "${LOCATE_VISION_SCALE}" --locate_llm_scale "${LOCATE_LLM_SCALE}")
  a+=(--locate_vision_layers "${LOCATE_VISION_LAYERS}" --locate_llm_layers "${LOCATE_LLM_LAYERS}")
  a+=(--locate_vision_modules "${LOCATE_VISION_MODULES}" --locate_llm_modules "${LOCATE_LLM_MODULES}")
  a+=(--weight_decay "${WEIGHT_DECAY}" --grad_clip "${GRAD_CLIP}" --warmup_steps "${WARMUP_STEPS}" --min_lr_ratio "${MIN_LR_RATIO}")
  a+=(--log_every "${LOG_EVERY}" --save_every "${SAVE_EVERY}" --save_total_limit "${SAVE_TOTAL_LIMIT}" --seed "${SEED}" --device "${DEVICE}")
  a+=(--w_oks "${W_OKS}" --w_coord "${W_COORD}" --w_image_coord "${W_IMAGE_COORD}" --w_keypoint_confidence "${W_KEYPOINT_CONFIDENCE}" --w_person_confidence "${W_PERSON_CONFIDENCE}" --w_ref_match "${W_REF_MATCH}")
  a+=(--w_hard_joint "${W_HARD_JOINT}" --hard_joint_fraction "${HARD_JOINT_FRACTION}")
  a+=(--w_coarse_coord "${W_COARSE_COORD}" --w_deform_coord "${W_DEFORM_COORD}" --w_refine_coords "${W_REFINE_COORDS}")
  a+=(--w_box_objectness "${W_BOX_OBJECTNESS}" --w_box_l1 "${W_BOX_L1}" --w_box_giou "${W_BOX_GIOU}" --w_box_relative "${W_BOX_RELATIVE}" --w_box_dn "${W_BOX_DN}")
  a+=(--w_keypoint_dn "${W_KEYPOINT_DN}")
  a+=(--visualize_every "${VISUALIZE_EVERY}" --visualize_max_instances "${VISUALIZE_MAX_INSTANCES}" --visualize_min_gt_area_ratio "${VISUALIZE_MIN_GT_AREA_RATIO}")
  add_opt a --max_samples_per_dataset "${MAX_SAMPLES_PER_DATASET}"
  add_opt a --deepspeed_config "${DEEPSPEED_CONFIG}"
  [[ "${DISABLE_RECORD_CACHE}" == "1" ]] && a+=(--disable_record_cache)
  [[ "${AMP}" == "1" ]] && a+=(--amp)
  [[ "${SYNC_TIMING}" == "1" ]] && a+=(--sync_timing)
  [[ "${PROGRESS_BAR}" == "0" ]] && a+=(--disable_progress)
  [[ "${DRY_RUN_DATA}" == "1" ]] && a+=(--dry_run_data)
  [[ "${DISABLE_BATCH_TRACE}" == "1" ]] && a+=(--disable_batch_trace)
  [[ "${DISABLE_HOMOGENEOUS_BATCHES}" == "1" ]] && a+=(--disable_homogeneous_batches)
  [[ "${DISABLE_VISION_TOKEN_BALANCING}" == "1" ]] && a+=(--disable_vision_token_balancing)
  [[ "${DISABLE_REFINEMENT}" == "1" ]] && a+=(--disable_refinement)
  return 0
}

run_stage() {
  local stage_label="$1"
  local output_dir="$2"
  local datasets="$3"
  local epochs="$4"
  local batch_size="$5"
  local grad_accum_steps="$6"
  local lr="$7"
  local max_steps="$8"
  local freeze_locate="$9"
  local resume_arg="${10}"
  local locate_train_scope="${11}"
  local locate_gradient_checkpointing="${12}"
  local locate_batch_token_limit="${13}"
  local dataset_mix_weights="${14}"

  local effective_batch=$((NPROC_PER_NODE * batch_size * grad_accum_steps))
  mkdir -p "${output_dir}" "${output_dir}/logs"
  local args=()
  common_args args
  add_opt args --dataset_mix_weights "${dataset_mix_weights}"
  add_opt args --locate_batch_token_limit "${locate_batch_token_limit}"
  args+=(--datasets "${datasets}" --output_dir "${output_dir}")
  args+=(--locate_feature_source raw_visual --locate_train_scope "${locate_train_scope}")
  [[ "${locate_gradient_checkpointing}" == "1" ]] && args+=(--locate_gradient_checkpointing)
  args+=(--epochs "${epochs}" --batch_size "${batch_size}" --grad_accum_steps "${grad_accum_steps}" --lr "${lr}" --max_steps "${max_steps}")
  args+=(--box_source person_queries)
  [[ "${freeze_locate}" == "1" ]] && args+=(--freeze_locate)
  add_opt args --resume_from_checkpoint "${resume_arg}"

  echo "================ LocatePose ${stage_label} 配置 ================"
  echo "OUTPUT_DIR=${output_dir}"
  echo "DATASETS=${datasets}"
  echo "DATASET_MIX_WEIGHTS=${dataset_mix_weights}"
  echo "ZERO_STAGE=${ZERO_STAGE}"
  echo "DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG}"
  echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "OPEN_FILE_SOFT_LIMIT=$(ulimit -Sn)"
  echo "MP_SHARING_STRATEGY=${QWENPOSE_MP_SHARING_STRATEGY}"
  echo "NUM_WORKERS=${NUM_WORKERS}"
  echo "PREFETCH_FACTOR=${PREFETCH_FACTOR}"
  echo "BATCH_SIZE=${batch_size}"
  echo "GRAD_ACCUM_STEPS=${grad_accum_steps}"
  echo "EFFECTIVE_BATCH=${effective_batch}"
  if [[ "${stage_label}" == Stage\ 1* ]]; then
    echo "TRAIN_MODE=${STAGE1_TRAIN_MODE}"
  fi
  echo "EPOCHS=${epochs}"
  echo "MAX_STEPS=${max_steps}"
  echo "FREEZE_LOCATE=${freeze_locate}"
  echo "LOCATE_FEATURE_SOURCE=raw_visual"
  echo "PRUNE_LOCATE_GENERATION=${PRUNE_LOCATE_GENERATION}"
  echo "LOCATE_TRAIN_SCOPE=${locate_train_scope}"
  echo "LOCATE_GRADIENT_CHECKPOINTING=${locate_gradient_checkpointing}"
  echo "LOCATE_ATTN_IMPLEMENTATION=${LOCATE_ATTN_IMPLEMENTATION}"
  echo "LOCATE_IMAGE_TOKEN_LIMIT=${LOCATE_IMAGE_TOKEN_LIMIT}"
  echo "LOCATE_BATCH_TOKEN_LIMIT=${locate_batch_token_limit}"
  echo "BOX_SOURCE=person_queries"
  echo "POSE_DROPOUT=${POSE_DROPOUT}"
  echo "POSE_COORDINATE_INIT=${POSE_COORDINATE_INIT}"
  echo "IMAGE_SIZE=800"
  echo "POSE_PYRAMID_CHANNELS=${POSE_PYRAMID_CHANNELS}"
  echo "POSE_PYRAMID_GRIDS=200x200/100x100/50x50"
  echo "POSE_PYRAMID_BLOCKS=${POSE_PYRAMID_BLOCKS}"
  echo "HUMAN_DECODER_LAYERS=${HUMAN_DECODER_LAYERS}"
  echo "POSE_DECODER_LAYERS=${POSE_DECODER_LAYERS}"
  echo "DEFORMABLE_POINTS=${DEFORMABLE_POINTS}"
  echo "DEFORMABLE_MIN_RADIUS_CELLS=${DEFORMABLE_MIN_RADIUS_CELLS}"
  echo "REF_TEXT_SCALE=${REF_TEXT_SCALE}"
  echo "ENABLE_BOX_DENOISING=${ENABLE_BOX_DENOISING}"
  echo "MAX_DN_QUERIES=${MAX_DN_QUERIES}"
  echo "MAX_DN_GROUPS=${MAX_DN_GROUPS}"
  echo "DN_POSITIVE_NOISE=${DN_POSITIVE_NOISE}"
  echo "DN_NEGATIVE_NOISE=${DN_NEGATIVE_NOISE}"
  echo "ENABLE_KEYPOINT_DENOISING=${ENABLE_KEYPOINT_DENOISING}"
  echo "MAX_KEYPOINT_DN_QUERIES=${MAX_KEYPOINT_DN_QUERIES}"
  echo "MAX_KEYPOINT_DN_GROUPS=${MAX_KEYPOINT_DN_GROUPS}"
  echo "KEYPOINT_DN_POSITIVE_KS=${KEYPOINT_DN_POSITIVE_KS_MIN},${KEYPOINT_DN_POSITIVE_KS_MAX}"
  echo "KEYPOINT_DN_NEGATIVE_KS=${KEYPOINT_DN_NEGATIVE_KS_MIN},${KEYPOINT_DN_NEGATIVE_KS_MAX}"
  echo "ENABLE_PERSON_CONFIDENCE_HEAD=${ENABLE_PERSON_CONFIDENCE_HEAD}"
  echo "W_KEYPOINT_CONFIDENCE=${W_KEYPOINT_CONFIDENCE}"
  echo "W_PERSON_CONFIDENCE=${W_PERSON_CONFIDENCE}"
  echo "W_REF_MATCH=${W_REF_MATCH}"
  echo "W_COARSE_COORD=${W_COARSE_COORD}"
  echo "W_DEFORM_COORD=${W_DEFORM_COORD}"
  echo "W_REFINE_COORDS=${W_REFINE_COORDS}"
  echo "W_BOX_OBJECTNESS=${W_BOX_OBJECTNESS}"
  echo "W_BOX_L1=${W_BOX_L1}"
  echo "W_BOX_GIOU=${W_BOX_GIOU}"
  echo "W_BOX_RELATIVE=${W_BOX_RELATIVE}"
  echo "W_BOX_DN=${W_BOX_DN}"
  echo "W_KEYPOINT_DN=${W_KEYPOINT_DN}"
  echo "LR=${lr}"
  echo "LOCATE_LORA_DROPOUT=${LOCATE_LORA_DROPOUT}"
  echo "LOCATE_VISION_LORA_DROPOUT=${LOCATE_VISION_LORA_DROPOUT}"
  echo "LOCATE_VISION_SCALE=${LOCATE_VISION_SCALE}"
  echo "LOCATE_LLM_SCALE=${LOCATE_LLM_SCALE}"
  echo "LOCATE_VISION_LAYERS=${LOCATE_VISION_LAYERS}"
  echo "LOCATE_LLM_LAYERS=${LOCATE_LLM_LAYERS}"
  echo "LOCATE_VISION_MODULES=${LOCATE_VISION_MODULES}"
  echo "LOCATE_LLM_MODULES=${LOCATE_LLM_MODULES}"
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
echo "STAGE1_TRAIN_MODE=${STAGE1_TRAIN_MODE}"
echo "STAGE1_EPOCHS=${STAGE1_EPOCHS}"
echo "STAGE1_MAX_STEPS=${STAGE1_MAX_STEPS}"
echo "STAGE1_RESUME_FROM_CHECKPOINT=${STAGE1_RESUME_FROM_CHECKPOINT}"
echo "STAGE2_RESUME_FROM_CHECKPOINT=${STAGE2_RESUME_FROM_CHECKPOINT}"
echo "STAGE2_INIT_CHECKPOINT=${STAGE2_INIT_CHECKPOINT}"
if [[ "${STAGE1_RESUME_FROM_CHECKPOINT}" != "none" ]]; then
  echo "RESUME_CURSOR=stage1 global_step=${RESUME_RESOLVED_STAGE1_GLOBAL_STEP:-unknown} epoch=${RESUME_RESOLVED_STAGE1_EPOCH:-unknown} batch=${RESUME_RESOLVED_STAGE1_BATCH_IN_EPOCH:-unknown}/${RESUME_RESOLVED_STAGE1_BATCHES_PER_EPOCH:-unknown}"
elif [[ "${STAGE2_RESUME_FROM_CHECKPOINT}" != "none" ]]; then
  echo "RESUME_CURSOR=stage2 global_step=${RESUME_RESOLVED_STAGE2_GLOBAL_STEP:-unknown} epoch=${RESUME_RESOLVED_STAGE2_EPOCH:-unknown} batch=${RESUME_RESOLVED_STAGE2_BATCH_IN_EPOCH:-unknown}/${RESUME_RESOLVED_STAGE2_BATCHES_PER_EPOCH:-unknown}"
fi
echo "LOCATE_MODEL_PATH=${LOCATE_MODEL_PATH}"
echo "LOCATE_ATTN_IMPLEMENTATION=${LOCATE_ATTN_IMPLEMENTATION}"
echo "LOCATE_IMAGE_TOKEN_LIMIT=${LOCATE_IMAGE_TOKEN_LIMIT}"
echo "TRAIN_LOG_FILE=${TRAIN_LOG_FILE}"
echo "=========================================================="

last_stage_output=""
if [[ "${RUN_STAGE1}" == "1" ]]; then
  stage1_resume_arg=""
  [[ "${STAGE1_RESUME_FROM_CHECKPOINT}" != "none" ]] && stage1_resume_arg="${STAGE1_RESUME_FROM_CHECKPOINT}"
  run_stage \
    "Stage 1 / frozen Locate unified person queries" \
    "${STAGE1_OUTPUT_DIR}" \
    "${STAGE1_TRAIN_DATASETS}" \
    "${STAGE1_EPOCHS}" \
    "${STAGE1_BATCH_SIZE}" \
    "${STAGE1_GRAD_ACCUM_STEPS}" \
    "${STAGE1_LR}" \
    "${STAGE1_MAX_STEPS}" \
    "1" \
    "${stage1_resume_arg}" \
    "frozen" \
    "0" \
    "${STAGE1_LOCATE_BATCH_TOKEN_LIMIT}" \
    "${STAGE1_DATASET_MIX_WEIGHTS}"
  last_stage_output="${STAGE1_OUTPUT_DIR}"
else
  echo "Skipping stage 1 because RUN_STAGE1=0"
fi

if [[ "${RUN_STAGE2}" == "1" ]]; then
  stage2_resume_arg="${STAGE2_RESUME_FROM_CHECKPOINT}"
  if [[ "${stage2_resume_arg}" == "none" || -z "${stage2_resume_arg}" ]]; then
    stage2_init_source="${STAGE2_INIT_CHECKPOINT}"
    if [[ -z "${stage2_init_source}" && "${STAGE2_INIT_FROM_STAGE1}" == "1" ]]; then
      if [[ -n "${last_stage_output}" ]] && resume_target_has_checkpoint "${last_stage_output}"; then
        stage2_init_source="${last_stage_output}"
      elif resume_target_has_checkpoint "${STAGE1_OUTPUT_DIR}"; then
        stage2_init_source="${STAGE1_OUTPUT_DIR}"
      fi
    fi
    if [[ "${DRY_RUN_DATA}" == "1" ]]; then
      stage2_resume_arg=""
    elif [[ -n "${stage2_init_source}" ]]; then
      echo "Preparing stage 2 weight-only init from ${stage2_init_source}"
      stage2_resume_arg="$(prepare_weights_only_checkpoint "${stage2_init_source}" "${STAGE2_INIT_WEIGHTS_DIR}")"
    else
      stage2_resume_arg=""
      echo "Stage 2 will start from base LocateAnything + newly initialized pose modules."
    fi
  else
    echo "Stage 2 will resume checkpoint state from ${stage2_resume_arg}"
  fi

  run_stage \
    "Stage 2 / merged Locate + RefHuman adaptation" \
    "${STAGE2_OUTPUT_DIR}" \
    "${STAGE2_TRAIN_DATASETS}" \
    "${STAGE2_EPOCHS}" \
    "${STAGE2_BATCH_SIZE}" \
    "${STAGE2_GRAD_ACCUM_STEPS}" \
    "${STAGE2_LR}" \
    "${STAGE2_MAX_STEPS}" \
    "${STAGE2_FREEZE_LOCATE}" \
    "${stage2_resume_arg}" \
    "${STAGE2_LOCATE_TRAIN_SCOPE}" \
    "${STAGE2_LOCATE_GRADIENT_CHECKPOINTING}" \
    "${STAGE2_LOCATE_BATCH_TOKEN_LIMIT}" \
    "${STAGE2_DATASET_MIX_WEIGHTS}"
  last_stage_output="${STAGE2_OUTPUT_DIR}"
else
  echo "Skipping stage 2 because RUN_STAGE2=0"
fi

echo "LocatePose finished. final_stage=${last_stage_output:-none}"
