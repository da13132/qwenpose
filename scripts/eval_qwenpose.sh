#!/usr/bin/env bash
set -Eeuo pipefail

###############################################################################
# QwenPose 验证脚本
#
# 功能：
#   1. 自动定位最新一次训练输出目录中的最新 checkpoint-*；
#   2. 加载 pose head + Qwen3-VL LoRA trainable 参数；
#   3. 在指定数据集 split 上计算 loss；
#   4. 导出 summary.json、predictions.jsonl、report.md。
#
# 注意：
#   QwenPose 的姿态结果来自回归式 PoseHead，不是 vLLM 文本生成；
#   因此这里不调用 vLLM，而是直接运行模型 forward。
#   默认走 Qwen 生成框闭环评估：Qwen generate bbox JSON 后再交给 PoseHead。
#   如需查看 GT-box-conditioned 上限，可设置 BOX_SOURCE=gt。
#   默认验证数据集：COCO / MPII / CrowdPose / RefHuman（默认不含 AIC）。
###############################################################################

# PROJECT_ROOT：项目根目录。脚本从任何位置启动都会回到这里。
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

# PYTHON：验证使用的 Python 解释器。优先使用项目内 envs/qwenpose/bin/python，否则回退到当前环境。
DEFAULT_PYTHON="$(resolve_default_python)"
PYTHON="${PYTHON:-${DEFAULT_PYTHON}}"

cd "${PROJECT_ROOT}"

# PYTHONPATH：优先使用本项目 src 下的 qwenpose 代码和本地依赖。
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

###############################################################################
# 路径与日志
###############################################################################

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
  local stage2_closed_loop_dir="${run_dir}/stage2_qwen_box_closed_loop"
  local legacy_stage3_dir="${run_dir}/stage3_qwen_box_closed_loop"
  local legacy_stage2_tf_dir="${run_dir}/stage2_teacher_forcing"
  local legacy_stage2_lm_dir="${run_dir}/stage2_qwen_lora_lm"
  local stage1_dir="${run_dir}/stage1_freeze_qwen"
  if dir_has_checkpoints "${stage2_closed_loop_dir}"; then
    printf '%s\n' "${stage2_closed_loop_dir}"
  elif dir_has_checkpoints "${legacy_stage3_dir}"; then
    printf '%s\n' "${legacy_stage3_dir}"
  elif dir_has_checkpoints "${legacy_stage2_tf_dir}"; then
    printf '%s\n' "${legacy_stage2_tf_dir}"
  elif dir_has_checkpoints "${legacy_stage2_lm_dir}"; then
    printf '%s\n' "${legacy_stage2_lm_dir}"
  elif dir_has_checkpoints "${stage1_dir}"; then
    printf '%s\n' "${stage1_dir}"
  else
    printf '%s\n' "${run_dir}"
  fi
}

# EVAL_TS：本次验证的时间戳，默认到秒，用于区分多次验证输出。
EVAL_TS="${EVAL_TS:-$(date +%Y%m%d-%H%M%S)}"

# TRAIN_OUTPUT_ROOT：训练 run 的根目录。未指定 TRAIN_OUTPUT_DIR 时，会从这里找最近一次 run。
TRAIN_OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT:-outputs/qwenpose_two_stage_qwen}"

# DEFAULT_TRAIN_OUTPUT_DIR：自动解析出的最近训练 run 目录。
DEFAULT_TRAIN_OUTPUT_DIR="$(resolve_latest_train_dir "${TRAIN_OUTPUT_ROOT}")"

# TRAIN_OUTPUT_DIR：要验证的训练输出目录。默认使用最近一次 run。
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-${DEFAULT_TRAIN_OUTPUT_DIR}}"

# DEFAULT_CHECKPOINT_TARGET：默认优先验证 stage2_qwen_box_closed_loop，其次兼容旧 stage3/stage2/stage1 或 run 根目录。
DEFAULT_CHECKPOINT_TARGET="$(resolve_default_checkpoint_target "${TRAIN_OUTPUT_DIR}")"

# CHECKPOINT：checkpoint 文件、checkpoint-* 目录、stage 目录或训练 run 目录。
# 传两阶段 run 目录时，脚本会优先选择其中的 stage2_qwen_box_closed_loop。
CHECKPOINT="${CHECKPOINT:-${DEFAULT_CHECKPOINT_TARGET}}"

# EVAL_OUTPUT_DIR：验证结果输出目录，默认放到训练 run 下的 eval_pose_时间戳/。
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${TRAIN_OUTPUT_DIR}/eval_pose_${EVAL_TS}}"

# LOG_DIR：验证日志目录，默认放在本次验证输出目录下的 logs/。
LOG_DIR="${LOG_DIR:-${EVAL_OUTPUT_DIR}/logs}"

# LOG_TS：日志文件时间戳，默认复用 EVAL_TS。
LOG_TS="${LOG_TS:-${EVAL_TS}}"

# EVAL_LOG_FILE：完整验证日志文件，包含配置、loss 汇总和报错堆栈。
EVAL_LOG_FILE="${EVAL_LOG_FILE:-${LOG_DIR}/eval_${LOG_TS}.log}"

mkdir -p "${LOG_DIR}" "$(dirname "${EVAL_LOG_FILE}")"
touch "${EVAL_LOG_FILE}"
exec > >(tee -a "${EVAL_LOG_FILE}") 2>&1
echo "Logging all stdout/stderr to ${EVAL_LOG_FILE}"
trap 'status=$?; echo "[ERROR] ${BASH_SOURCE[0]}:${LINENO}: ${BASH_COMMAND} exited with status ${status}" >&2' ERR
trap 'status=$?; echo "========== qwenpose eval exit status ${status} at $(date -Is) =========="; exit ${status}' EXIT

###############################################################################
# 等待训练结束
###############################################################################

# WAIT_FOR_TRAINING：是否等待当前 qwenpose 训练进程结束后再验证。
# 设为 true 时适合把验证脚本挂在训练任务后面自动轮询。
WAIT_FOR_TRAINING="${WAIT_FOR_TRAINING:-false}"

# POLL_SECONDS：WAIT_FOR_TRAINING=true 时，每隔多少秒检查一次训练是否结束。
POLL_SECONDS="${POLL_SECONDS:-300}"

if [[ "${WAIT_FOR_TRAINING}" == "true" ]]; then
  echo "Waiting for current qwenpose training process to finish..."
  while pgrep -f "qwenpose.train_pose|src/qwenpose/train_pose.py" >/dev/null; do
    date
    sleep "${POLL_SECONDS}"
  done
fi

###############################################################################
# 数据与模型参数
###############################################################################

# DATASET_ROOT：数据集根目录，默认使用项目下 datasets/。
DATASET_ROOT="${DATASET_ROOT:-datasets}"

# EVAL_DATASETS：本次验证启用的数据集列表，逗号分隔。
# 默认不包含 AIC；需要验证 AIC 时可设 EVAL_DATASETS=coco,aic,mpii,crowdpose,refhuman。
# 兼容旧变量 DATASETS，但推荐新脚本里显式使用 EVAL_DATASETS。
EVAL_DATASETS="${EVAL_DATASETS:-${DATASETS:-coco,mpii,crowdpose,refhuman}}"

# SPLIT：验证读取的数据 split，默认 val。
SPLIT="${SPLIT:-val}"

# MAX_SAMPLES_PER_DATASET：每个数据集最多验证多少样本。空值表示全量，调试时可设 1/10/100。
MAX_SAMPLES_PER_DATASET="${MAX_SAMPLES_PER_DATASET:-}"

# RECORD_CACHE_DIR：解析后的 PoseRecord 缓存目录。和训练脚本保持一致可复用缓存。
RECORD_CACHE_DIR="${RECORD_CACHE_DIR:-.cache/qwenpose_records}"

# DISABLE_RECORD_CACHE：设为 1 时禁用标注缓存，强制重新解析原始 JSON。
DISABLE_RECORD_CACHE="${DISABLE_RECORD_CACHE:-0}"

# PROGRESS_BAR：是否显示 tqdm 进度条和 ETA。
#   1：默认开启，显示验证进度、loss、data/prep/fwd 耗时。
#   0：关闭进度条，适合某些日志系统不喜欢动态刷新时使用。
PROGRESS_BAR="${PROGRESS_BAR:-1}"

# MAX_INSTANCES：单图最多保留的 GT 人体实例数，需要和训练时设置保持兼容。
MAX_INSTANCES="${MAX_INSTANCES:-80}"

# NUM_WORKERS：DataLoader worker 数。调试多进程问题时可设 0。
NUM_WORKERS="${NUM_WORKERS:-4}"

# PREFETCH_FACTOR：每个 DataLoader worker 预取多少个 batch。NUM_WORKERS>0 时生效。
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"

# BATCH_SIZE：验证 batch size。Qwen3-VL 默认 1，显存充足时可增大。
BATCH_SIZE="${BATCH_SIZE:-1}"

# BACKBONE：验证时使用的视觉特征来源。轻量 smoke 分支已删除，当前只支持 qwen3vl。
BACKBONE="${BACKBONE:-qwen3vl}"

# QWEN_MODEL_PATH：本地 Qwen3-VL 权重路径，加载 LoRA checkpoint 时需要原始 base model。
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-weights/Qwen3-VL-4B-Instruct}"

# QWEN_DTYPE：Qwen3-VL 加载精度。默认 bfloat16，需和机器能力匹配。
QWEN_DTYPE="${QWEN_DTYPE:-bfloat16}"

# QWEN_ATTN_IMPLEMENTATION：Qwen attention 后端，默认 flash_attention_2，需和训练启动方式保持一致。
QWEN_ATTN_IMPLEMENTATION="${QWEN_ATTN_IMPLEMENTATION:-flash_attention_2}"

# QWEN_MIN_PIXELS / QWEN_MAX_PIXELS：可选限制验证时 Qwen processor 动态分辨率。
# 默认留空，不覆盖模型 processor 的默认分辨率；如训练手动限制过，这里建议设成相同值。
QWEN_MIN_PIXELS="${QWEN_MIN_PIXELS:-}"
QWEN_MAX_PIXELS="${QWEN_MAX_PIXELS:-}"

# QWEN_FEATURE_SIZE：Qwen 特征汇聚网格边长，必须和训练时模型结构一致。
QWEN_FEATURE_SIZE="${QWEN_FEATURE_SIZE:-64}"

# QWEN_FEATURE_REFINER_*：兼容训练端插值后的 Qwen 特征细化模块。
# 新格式 checkpoint 会优先使用自身保存的 refiner 配置；这里主要用于手动兼容调试。
QWEN_FEATURE_REFINER_LAYERS="${QWEN_FEATURE_REFINER_LAYERS:-2}"
QWEN_FEATURE_REFINER_BOTTLENECK_DIM="${QWEN_FEATURE_REFINER_BOTTLENECK_DIM:-256}"
QWEN_FEATURE_REFINER_INIT_SCALE="${QWEN_FEATURE_REFINER_INIT_SCALE:-0.1}"

# HIDDEN_DIM：Pose decoder 隐藏维度，必须和训练 checkpoint 一致。当前默认 448。
HIDDEN_DIM="${HIDDEN_DIM:-448}"

# POSE_DECODER_LAYERS：关键点 token decoder 层数，必须和训练 checkpoint 一致。当前默认 1。
POSE_DECODER_LAYERS="${POSE_DECODER_LAYERS:-3}"

# REFINEMENT_STEPS：关键点局部残差细化轮数，必须和训练 checkpoint 一致。
REFINEMENT_STEPS="${REFINEMENT_STEPS:-3}"

# BOX_CONDITION_SCALE：传给 PoseHead 的 bbox 扩大倍数，建议和训练保持一致。
BOX_CONDITION_SCALE="${BOX_CONDITION_SCALE:-1.2}"

# POSE_ROI_SIZE：每个扩大后 bbox 从 Qwen grid 上采样出的局部特征边长，必须匹配训练。
POSE_ROI_SIZE="${POSE_ROI_SIZE:-16}"

###############################################################################
# Transformers / Qwen3-VL 闭环推理参数
#
# QwenPose 这里不接 vLLM。qwen_generate 会调用 transformers Qwen3-VL
# generate 得到框，再用同一个 PyTorch 模型图跑 PoseHead。
###############################################################################

# BOX_SOURCE：验证时 PoseHead 条件框来源。默认 qwen_generate，走最终闭环路径；gt 用于看 GT box 上限。
BOX_SOURCE="${BOX_SOURCE:-qwen_generate}"
# QWEN_BOX_MAX_NEW_TOKENS：Qwen 生成 bbox JSON 最大新 token 数。
QWEN_BOX_MAX_NEW_TOKENS="${QWEN_BOX_MAX_NEW_TOKENS:-4096}"
# BOX_MATCH_IOU_THRESH：闭环验证中生成框和 GT 框匹配阈值，仅用于 loss/GT 对齐。
BOX_MATCH_IOU_THRESH="${BOX_MATCH_IOU_THRESH:-0.10}"
# BOX_NMS_IOU_THRESH：闭环验证中生成框 NMS 阈值。
BOX_NMS_IOU_THRESH="${BOX_NMS_IOU_THRESH:-0.70}"

# DECODER_HEADS：Pose decoder 注意力头数，必须和训练 checkpoint 一致。
DECODER_HEADS="${DECODER_HEADS:-8}"

###############################################################################
# vLLM 参数区
#
# 当前 QwenPose 验证脚本没有 vLLM 推理参数。需要 vLLM 只生成框时应使用
# LocatePose 专用的 scripts/infer_locatepose.sh；需要真正 PoseHead 特征复用时
# 继续使用本脚本/transformers 路径。
###############################################################################

# QWEN_LORA_R：LLM LoRA rank，加载 LoRA 结构时需和训练一致。
QWEN_LORA_R="${QWEN_LORA_R:-32}"

# QWEN_LORA_ALPHA：LLM LoRA alpha，需和训练一致。
QWEN_LORA_ALPHA="${QWEN_LORA_ALPHA:-64}"

# QWEN_LORA_DROPOUT：LLM LoRA dropout。eval 模式下不生效，但结构参数需保留。
QWEN_LORA_DROPOUT="${QWEN_LORA_DROPOUT:-0.05}"

# QWEN_VISION_LORA_R：视觉塔 LoRA rank，需和训练一致。
QWEN_VISION_LORA_R="${QWEN_VISION_LORA_R:-16}"

# QWEN_VISION_LORA_ALPHA：视觉塔 LoRA alpha，需和训练一致。
QWEN_VISION_LORA_ALPHA="${QWEN_VISION_LORA_ALPHA:-32}"

# QWEN_VISION_LORA_DROPOUT：视觉塔 LoRA dropout。eval 模式下不生效，但结构参数需保留。
QWEN_VISION_LORA_DROPOUT="${QWEN_VISION_LORA_DROPOUT:-0.05}"

# DEVICE：验证设备。正式验证用 cuda；CPU 上加载 Qwen3-VL 很慢且占内存，不建议。
DEVICE="${DEVICE:-cuda}"

# SCORE_THRESHOLD：官方 COCO AP 前不预删候选，由 evaluator 自行排序并保留 top-20。
SCORE_THRESHOLD="${SCORE_THRESHOLD:-0.0}"

# MAX_PREDICTIONS_PER_IMAGE：单图最多导出的 box-conditioned pose 数量，避免 predictions.jsonl 过大。
MAX_PREDICTIONS_PER_IMAGE="${MAX_PREDICTIONS_PER_IMAGE:-100}"

# VISUALIZE_MAX_SAMPLES：验证时最多保存多少张预测可视化；0 表示关闭。
VISUALIZE_MAX_SAMPLES="${VISUALIZE_MAX_SAMPLES:-100}"

# VISUALIZE_MAX_INSTANCES：每张可视化图最多绘制多少个实例，避免拥挤图过乱。
VISUALIZE_MAX_INSTANCES="${VISUALIZE_MAX_INSTANCES:-8}"

# W_OKS：OKS loss 权重，和训练脚本保持同名同默认值。
W_OKS="${W_OKS:-0.2}"

# W_COORD：关键点坐标回归 loss 权重，和训练脚本保持同名同默认值。
W_COORD="${W_COORD:-5.0}"

# W_VIS：关键点可见性/有效性 BCE loss 权重，和训练脚本保持同名同默认值。
W_VIS="${W_VIS:-0.05}"

# W_HARD_JOINT：hard keypoint mining loss 权重，和训练脚本保持同名同默认值；默认关闭。
W_HARD_JOINT="${W_HARD_JOINT:-0}"

# HARD_JOINT_FRACTION：hard mining 选取的可见关键点比例，和训练脚本保持一致。
HARD_JOINT_FRACTION="${HARD_JOINT_FRACTION:-0.2}"

# DISABLE_REFINEMENT：设为 1 时按关闭关键点细化分支的模型结构加载 checkpoint。
DISABLE_REFINEMENT="${DISABLE_REFINEMENT:-0}"

###############################################################################
# 参数检查
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

require_positive_int BATCH_SIZE "${BATCH_SIZE}"
require_positive_int REFINEMENT_STEPS "${REFINEMENT_STEPS}"
require_positive_int POSE_ROI_SIZE "${POSE_ROI_SIZE}"
require_positive_int QWEN_BOX_MAX_NEW_TOKENS "${QWEN_BOX_MAX_NEW_TOKENS}"
require_nonnegative_int NUM_WORKERS "${NUM_WORKERS}"
require_positive_int MAX_PREDICTIONS_PER_IMAGE "${MAX_PREDICTIONS_PER_IMAGE}"
require_nonnegative_int VISUALIZE_MAX_SAMPLES "${VISUALIZE_MAX_SAMPLES}"
require_positive_int VISUALIZE_MAX_INSTANCES "${VISUALIZE_MAX_INSTANCES}"

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

if [[ "${BACKBONE}" != "qwen3vl" ]]; then
  echo "BACKBONE=${BACKBONE} is no longer supported. The lightweight smoke backbone was removed; use BACKBONE=qwen3vl." >&2
  exit 1
fi
if [[ "${BOX_SOURCE}" != "gt" && "${BOX_SOURCE}" != "qwen_generate" ]]; then
  echo "BOX_SOURCE must be gt or qwen_generate, got: ${BOX_SOURCE}" >&2
  exit 1
fi

if [[ "${BACKBONE}" == "qwen3vl" && ! -e "${QWEN_MODEL_PATH}" ]]; then
  echo "QWEN_MODEL_PATH not found: ${QWEN_MODEL_PATH}" >&2
  exit 1
fi

###############################################################################
# 组装验证参数
###############################################################################

ARGS=(
  # --checkpoint：待加载 checkpoint 文件、checkpoint-* 目录或训练 run 目录。
  --checkpoint "${CHECKPOINT}"
  # --output_dir：验证结果输出目录。
  --output_dir "${EVAL_OUTPUT_DIR}"
  # --dataset_root：数据集根目录。
  --dataset_root "${DATASET_ROOT}"
  # --datasets：本次验证的数据集列表。
  --datasets "${EVAL_DATASETS}"
  # --split：验证 split，通常为 val。
  --split "${SPLIT}"
  # --max_instances：单图最多保留的 GT 人体实例数。
  --max_instances "${MAX_INSTANCES}"
  # --record_cache_dir：PoseRecord 标注缓存目录。
  --record_cache_dir "${RECORD_CACHE_DIR}"
  # --num_workers：DataLoader worker 数。
  --num_workers "${NUM_WORKERS}"
  # --prefetch_factor：每个 DataLoader worker 的预取 batch 数。
  --prefetch_factor "${PREFETCH_FACTOR}"
  # --batch_size：验证 batch size。
  --batch_size "${BATCH_SIZE}"
  # --backbone：当前只支持 qwen3vl，必须匹配 checkpoint。
  --backbone "${BACKBONE}"
  # --qwen_model_path：Qwen3-VL base model 路径。
  --qwen_model_path "${QWEN_MODEL_PATH}"
  # --qwen_dtype：Qwen3-VL 加载精度。
  --qwen_dtype "${QWEN_DTYPE}"
  # --qwen_attn_implementation：Qwen attention 实现。
  --qwen_attn_implementation "${QWEN_ATTN_IMPLEMENTATION}"
  # --qwen_feature_size：Qwen 特征汇聚网格边长，必须匹配训练。
  --qwen_feature_size "${QWEN_FEATURE_SIZE}"
  # --qwen_feature_refiner_layers：插值后的 Qwen 视觉特征细化残差块层数。
  --qwen_feature_refiner_layers "${QWEN_FEATURE_REFINER_LAYERS}"
  # --qwen_feature_refiner_bottleneck_dim：细化块瓶颈通道数。
  --qwen_feature_refiner_bottleneck_dim "${QWEN_FEATURE_REFINER_BOTTLENECK_DIM}"
  # --qwen_feature_refiner_init_scale：细化残差初始强度。
  --qwen_feature_refiner_init_scale "${QWEN_FEATURE_REFINER_INIT_SCALE}"
  # --qwen_lora_r：LLM LoRA rank，必须匹配训练。
  --qwen_lora_r "${QWEN_LORA_R}"
  # --qwen_lora_alpha：LLM LoRA alpha，必须匹配训练。
  --qwen_lora_alpha "${QWEN_LORA_ALPHA}"
  # --qwen_lora_dropout：LLM LoRA dropout，结构参数需保留。
  --qwen_lora_dropout "${QWEN_LORA_DROPOUT}"
  # --qwen_vision_lora_r：视觉塔 LoRA rank，必须匹配训练。
  --qwen_vision_lora_r "${QWEN_VISION_LORA_R}"
  # --qwen_vision_lora_alpha：视觉塔 LoRA alpha，必须匹配训练。
  --qwen_vision_lora_alpha "${QWEN_VISION_LORA_ALPHA}"
  # --qwen_vision_lora_dropout：视觉塔 LoRA dropout，结构参数需保留。
  --qwen_vision_lora_dropout "${QWEN_VISION_LORA_DROPOUT}"
  # --hidden_dim：pose decoder 隐藏维度，必须匹配训练。
  --hidden_dim "${HIDDEN_DIM}"
  # --pose_decoder_layers：关键点 token decoder 层数，必须匹配训练。
  --pose_decoder_layers "${POSE_DECODER_LAYERS}"
  # --refinement_steps：关键点局部残差细化轮数，必须匹配训练。
  --refinement_steps "${REFINEMENT_STEPS}"
  # --box_condition_scale：传给 PoseHead 的 bbox 扩大倍数，必须匹配训练。
  --box_condition_scale "${BOX_CONDITION_SCALE}"
  # --pose_roi_size：每个 bbox 的 box-local ROI feature 边长，必须匹配训练。
  --pose_roi_size "${POSE_ROI_SIZE}"
  # --box_source：验证条件框来源，默认 qwen_generate 走闭环路径。
  --box_source "${BOX_SOURCE}"
  # --qwen_box_max_new_tokens：Qwen 生成 bbox JSON 最大新 token 数。
  --qwen_box_max_new_tokens "${QWEN_BOX_MAX_NEW_TOKENS}"
  # --box_match_iou_thresh：生成框和 GT 框匹配阈值。
  --box_match_iou_thresh "${BOX_MATCH_IOU_THRESH}"
  # --box_nms_iou_thresh：生成框 NMS 阈值。
  --box_nms_iou_thresh "${BOX_NMS_IOU_THRESH}"
  # --decoder_heads：pose decoder 注意力头数，必须匹配训练。
  --decoder_heads "${DECODER_HEADS}"
  # --device：验证设备。
  --device "${DEVICE}"
  # --score_threshold：官方评估默认 0，不在 COCO evaluator 前预删候选。
  --score_threshold "${SCORE_THRESHOLD}"
  # --max_predictions_per_image：单图最多导出的预测人数。
  --max_predictions_per_image "${MAX_PREDICTIONS_PER_IMAGE}"
  # --visualize_max_samples：最多保存多少张验证预测可视化。
  --visualize_max_samples "${VISUALIZE_MAX_SAMPLES}"
  # --visualize_max_instances：单张验证可视化最多绘制的实例数。
  --visualize_max_instances "${VISUALIZE_MAX_INSTANCES}"
  # --w_oks：OKS loss 权重。
  --w_oks "${W_OKS}"
  # --w_coord：关键点坐标回归 loss 权重。
  --w_coord "${W_COORD}"
  # --w_vis：关键点可见性 loss 权重。
  --w_vis "${W_VIS}"
  # --w_hard_joint：hard keypoint mining loss 权重。
  --w_hard_joint "${W_HARD_JOINT}"
  # --hard_joint_fraction：hard mining 选取的可见关键点比例。
  --hard_joint_fraction "${HARD_JOINT_FRACTION}"
)

if [[ -n "${MAX_SAMPLES_PER_DATASET}" ]]; then
  # --max_samples_per_dataset：限制每个数据集验证样本数，用于快速调试。
  ARGS+=(--max_samples_per_dataset "${MAX_SAMPLES_PER_DATASET}")
fi
if [[ -n "${QWEN_MIN_PIXELS}" ]]; then
  # --qwen_min_pixels：手动覆盖 Qwen processor 最小像素数。
  ARGS+=(--qwen_min_pixels "${QWEN_MIN_PIXELS}")
fi
if [[ -n "${QWEN_MAX_PIXELS}" ]]; then
  # --qwen_max_pixels：手动覆盖 Qwen processor 最大像素数。
  ARGS+=(--qwen_max_pixels "${QWEN_MAX_PIXELS}")
fi
if [[ "${DISABLE_RECORD_CACHE}" == "1" ]]; then
  # --disable_record_cache：强制重新解析原始标注 JSON。
  ARGS+=(--disable_record_cache)
fi
if [[ "${PROGRESS_BAR}" == "0" ]]; then
  # --disable_progress：关闭 tqdm 进度条和 ETA。
  ARGS+=(--disable_progress)
fi
if [[ "${DISABLE_REFINEMENT}" == "1" ]]; then
  # --disable_refinement：按关闭关键点细化分支的模型结构执行验证。
  ARGS+=(--disable_refinement)
fi

echo "================ QwenPose 验证配置 ================"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "TRAIN_OUTPUT_ROOT=${TRAIN_OUTPUT_ROOT}"
echo "TRAIN_OUTPUT_DIR=${TRAIN_OUTPUT_DIR}"
echo "CHECKPOINT=${CHECKPOINT}"
echo "EVAL_TS=${EVAL_TS}"
echo "EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR}"
echo "EVAL_DATASETS=${EVAL_DATASETS}"
echo "SPLIT=${SPLIT}"
echo "RECORD_CACHE_DIR=${RECORD_CACHE_DIR}"
echo "DISABLE_RECORD_CACHE=${DISABLE_RECORD_CACHE}"
echo "PROGRESS_BAR=${PROGRESS_BAR}"
echo "BACKBONE=${BACKBONE}"
echo "QWEN_MODEL_PATH=${QWEN_MODEL_PATH}"
echo "QWEN_ATTN_IMPLEMENTATION=${QWEN_ATTN_IMPLEMENTATION}"
echo "QWEN_MIN_PIXELS=${QWEN_MIN_PIXELS}"
echo "QWEN_MAX_PIXELS=${QWEN_MAX_PIXELS}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "NUM_WORKERS=${NUM_WORKERS}"
echo "PREFETCH_FACTOR=${PREFETCH_FACTOR}"
echo "HIDDEN_DIM=${HIDDEN_DIM}"
echo "POSE_DECODER_LAYERS=${POSE_DECODER_LAYERS}"
echo "REFINEMENT_STEPS=${REFINEMENT_STEPS}"
echo "BOX_CONDITION_SCALE=${BOX_CONDITION_SCALE}"
echo "POSE_ROI_SIZE=${POSE_ROI_SIZE}"
echo "BOX_SOURCE=${BOX_SOURCE}"
echo "QWEN_BOX_MAX_NEW_TOKENS=${QWEN_BOX_MAX_NEW_TOKENS}"
echo "BOX_MATCH_IOU_THRESH=${BOX_MATCH_IOU_THRESH}"
echo "BOX_NMS_IOU_THRESH=${BOX_NMS_IOU_THRESH}"
echo "VISUALIZE_MAX_SAMPLES=${VISUALIZE_MAX_SAMPLES}"
echo "VISUALIZE_MAX_INSTANCES=${VISUALIZE_MAX_INSTANCES}"
echo "W_OKS=${W_OKS}"
echo "W_COORD=${W_COORD}"
echo "W_VIS=${W_VIS}"
echo "W_HARD_JOINT=${W_HARD_JOINT}"
echo "HARD_JOINT_FRACTION=${HARD_JOINT_FRACTION}"
echo "DEVICE=${DEVICE}"
echo "===================================================="

"${PYTHON}" -m qwenpose.eval_pose "${ARGS[@]}"
