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
#   5. 默认使用 integrated vLLM：LocateAnything 生成框，PoseHead 复用同一次 vLLM 图像特征。
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

###############################################################################
# 路径、checkpoint 与输出
###############################################################################

# EVAL_TS：本次验证时间戳，默认精确到秒，用于输出目录名。
EVAL_TS="${EVAL_TS:-$(date +%Y%m%d-%H%M%S)}"
# TRAIN_OUTPUT_ROOT：LocatePose 训练 run 的根目录；未指定 TRAIN_OUTPUT_DIR 时从这里找最新 run。
TRAIN_OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT:-outputs/locatepose}"
# DEFAULT_TRAIN_OUTPUT_DIR：自动解析出的最近训练 run 目录。
DEFAULT_TRAIN_OUTPUT_DIR="$(resolve_latest_train_dir "${TRAIN_OUTPUT_ROOT}")"
# TRAIN_OUTPUT_DIR：要验证的训练 run 目录；也决定默认输出目录位置。
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-${DEFAULT_TRAIN_OUTPUT_DIR}}"
# DEFAULT_CHECKPOINT_TARGET：默认优先 stage2_locate_box_closed_loop，其次 stage1_freeze_locate_gt_box。
DEFAULT_CHECKPOINT_TARGET="$(resolve_default_checkpoint_target "${TRAIN_OUTPUT_DIR}")"
# CHECKPOINT：checkpoint 文件、checkpoint-* 目录、stage 目录或 run 目录；也可用第一个位置参数传入。
CHECKPOINT="${CHECKPOINT:-${1:-${DEFAULT_CHECKPOINT_TARGET}}}"
# EVAL_OUTPUT_DIR：验证结果目录，会写 summary.json、predictions.jsonl/json、report.md 和可视化。
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${TRAIN_OUTPUT_DIR}/eval_locatepose_${EVAL_TS}}"

###############################################################################
# 数据集、DataLoader 与运行设备
###############################################################################

# DATASET_ROOT：数据集根目录，默认项目内 datasets/。
DATASET_ROOT="${DATASET_ROOT:-datasets}"
# DATASETS：验证数据集列表，逗号分隔；支持 coco,crowdpose,mpii,refhuman,aic。
DATASETS="${DATASETS:-coco}"
# SPLIT：验证 split；coco 会映射到 val2017/train2017，其余数据集通常用 val/train。
SPLIT="${SPLIT:-val}"
# MAX_SAMPLES_PER_DATASET：每个数据集最多验证多少条记录；空表示全量，调试可设 10/100。
MAX_SAMPLES_PER_DATASET="${MAX_SAMPLES_PER_DATASET:-}"
# RECORD_CACHE_DIR：解析后的 PoseRecord 缓存目录；和训练脚本保持一致可以复用缓存。
RECORD_CACHE_DIR="${RECORD_CACHE_DIR:-.cache/qwenpose_records}"
# DISABLE_RECORD_CACHE：设为 1 时不读写缓存，强制重新解析原始标注。
DISABLE_RECORD_CACHE="${DISABLE_RECORD_CACHE:-0}"
# PROGRESS_BAR：是否显示 tqdm 进度条；0 关闭，1 开启。
PROGRESS_BAR="${PROGRESS_BAR:-1}"
# DEVICE：验证设备，正式验证建议 cuda。
DEVICE="${DEVICE:-cuda}"
# NUM_WORKERS：DataLoader worker 数；调试多进程/文件句柄问题可设 0。
NUM_WORKERS="${NUM_WORKERS:-2}"
# PREFETCH_FACTOR：每个 worker 预取 batch 数，仅 NUM_WORKERS>0 时生效。
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
# BATCH_SIZE：验证 batch size；Locate 生成框闭环默认 1 更稳，GT-box 上限可适当增大。
BATCH_SIZE="${BATCH_SIZE:-1}"
# MAX_INSTANCES：单张图最多保留/评估的人体实例数，需要和训练最大实例数兼容。
MAX_INSTANCES="${MAX_INSTANCES:-80}"

###############################################################################
# LocateAnything / LocatePose backbone 参数
###############################################################################

# LOCATE_MODEL_PATH：LocateAnything-3B base 权重目录，需要和训练时使用的 base 一致。
LOCATE_MODEL_PATH="${LOCATE_MODEL_PATH:-weights/LocateAnything-3B}"
# LOCATE_DTYPE：LocateAnything PyTorch 加载精度；4090/ADA 通常用 bfloat16。
LOCATE_DTYPE="${LOCATE_DTYPE:-bfloat16}"
# LOCATE_ATTN_IMPLEMENTATION：Locate attention 后端；sdpa 稳定，flash_attention_2 更快但依赖环境。
LOCATE_ATTN_IMPLEMENTATION="${LOCATE_ATTN_IMPLEMENTATION:-sdpa}"
# LOCATE_MIN_PIXELS：可选最小像素预算；空表示不覆盖 processor 默认。
LOCATE_MIN_PIXELS="${LOCATE_MIN_PIXELS:-}"
# LOCATE_MAX_PIXELS：可选最大像素预算；空表示不覆盖 processor 默认。
LOCATE_MAX_PIXELS="${LOCATE_MAX_PIXELS:-}"
# LOCATE_IMAGE_TOKEN_LIMIT：LocateAnything 原生 raw MoonViT patch token 上限；默认 4096，和训练脚本保持一致。
LOCATE_IMAGE_TOKEN_LIMIT="${LOCATE_IMAGE_TOKEN_LIMIT:-4096}"
# LOCATE_FEATURE_SIZE：Locate hidden map 汇聚后的空间边长；checkpoint 有保存配置时优先用 checkpoint。
LOCATE_FEATURE_SIZE="${LOCATE_FEATURE_SIZE:-64}"
# LOCATE_FEATURE_REFINER_LAYERS：Locate feature refiner 层数；checkpoint 有 refiner 权重时必须匹配。
LOCATE_FEATURE_REFINER_LAYERS="${LOCATE_FEATURE_REFINER_LAYERS:-2}"
# LOCATE_FEATURE_REFINER_BOTTLENECK_DIM：feature refiner bottleneck 通道数。
LOCATE_FEATURE_REFINER_BOTTLENECK_DIM="${LOCATE_FEATURE_REFINER_BOTTLENECK_DIM:-256}"
# LOCATE_FEATURE_REFINER_INIT_SCALE：feature refiner 残差初始化尺度。
LOCATE_FEATURE_REFINER_INIT_SCALE="${LOCATE_FEATURE_REFINER_INIT_SCALE:-0.1}"
# LOCATE_LORA_R：Locate 语言/主干 LoRA rank，必须与训练 adapter 结构一致。
LOCATE_LORA_R="${LOCATE_LORA_R:-32}"
# LOCATE_LORA_ALPHA：Locate 语言/主干 LoRA alpha，必须与训练一致。
LOCATE_LORA_ALPHA="${LOCATE_LORA_ALPHA:-64}"
# LOCATE_LORA_DROPOUT：Locate 语言/主干 LoRA dropout；eval 模式不随机，但结构参数需保留。
LOCATE_LORA_DROPOUT="${LOCATE_LORA_DROPOUT:-0.05}"
# LOCATE_VISION_LORA_R：Locate vision tower LoRA rank，必须与训练一致。
LOCATE_VISION_LORA_R="${LOCATE_VISION_LORA_R:-16}"
# LOCATE_VISION_LORA_ALPHA：Locate vision tower LoRA alpha，必须与训练一致。
LOCATE_VISION_LORA_ALPHA="${LOCATE_VISION_LORA_ALPHA:-32}"
# LOCATE_VISION_LORA_DROPOUT：Locate vision tower LoRA dropout；eval 模式不随机，但结构参数需保留。
LOCATE_VISION_LORA_DROPOUT="${LOCATE_VISION_LORA_DROPOUT:-0.05}"

###############################################################################
# PoseHead 结构参数
###############################################################################

# HIDDEN_DIM：PoseHead 隐藏维度；新 checkpoint 保存 pose_config 时会以 checkpoint 为准。
HIDDEN_DIM="${HIDDEN_DIM:-448}"
# POSE_DECODER_LAYERS：Pose decoder 层数；旧 checkpoint 无 pose_config 时使用该值。
POSE_DECODER_LAYERS="${POSE_DECODER_LAYERS:-3}"
# REFINEMENT_STEPS：关键点局部细化步数；必须与 checkpoint 结构匹配。
REFINEMENT_STEPS="${REFINEMENT_STEPS:-3}"
# DECODER_HEADS：Pose decoder attention heads 数；必须与训练结构匹配。
DECODER_HEADS="${DECODER_HEADS:-8}"
# BOX_CONDITION_SCALE：PoseHead 条件框扩展比例，和训练保持一致可避免分布偏移。
BOX_CONDITION_SCALE="${BOX_CONDITION_SCALE:-1.2}"
# POSE_ROI_SIZE：每个 bbox 在 hidden map 上采样的 ROI 特征边长。
POSE_ROI_SIZE="${POSE_ROI_SIZE:-16}"

###############################################################################
# 条件框来源、单次复用与 Locate 生成参数
###############################################################################

# BOX_SOURCE：条件框来源。locate_generate 为闭环评估；gt 用于看 stage1/GT-box 上限。
BOX_SOURCE="${BOX_SOURCE:-locate_generate}"
# LOCATE_GENERATION_BACKEND：Locate 生成框后端。默认 vLLM，走 LocateAnything+PoseHead integrated vLLM 路径。
LOCATE_GENERATION_BACKEND="${LOCATE_GENERATION_BACKEND:-vllm}"
# GPU：vLLM 推理使用的可见 GPU；默认单卡 GPU 0。
GPU="${GPU:-0}"
# DISABLE_VLLM_FALLBACK：设为 1 时 vLLM 失败直接报错，不回退 transformers。
DISABLE_VLLM_FALLBACK="${DISABLE_VLLM_FALLBACK:-1}"
# VLLM_TENSOR_PARALLEL_SIZE：vLLM tensor parallel 大小。
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
# VLLM_GPU_MEMORY_UTILIZATION：vLLM 可使用显存比例。
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
# VLLM_CPU_OFFLOAD_GB：vLLM 权重 CPU offload 大小；显存紧张时可设 8/16/20。
VLLM_CPU_OFFLOAD_GB="${VLLM_CPU_OFFLOAD_GB:-0}"
# VLLM_ENFORCE_EAGER：设为 1 时强制 vLLM eager 执行。
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
# VLLM_MAX_MODEL_LEN：vLLM 最大上下文长度；0 表示模型默认。
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-0}"
# VLLM_BATCH_SIZE：vLLM 请求 batch size；默认与 BATCH_SIZE/PoseHead batch 同步。
VLLM_BATCH_SIZE="${VLLM_BATCH_SIZE:-${BATCH_SIZE}}"
# VLLM_MAX_NUM_SEQS：vLLM scheduler 最大并发序列数；默认与 VLLM_BATCH_SIZE 同步。
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-${VLLM_BATCH_SIZE}}"
# VLLM_MAX_NUM_BATCHED_TOKENS：vLLM prefill/profile token 预算；默认 2048。
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-2048}"
# VLLM_MODEL_IMPL：LocateAnything custom vLLM model 默认 auto。
VLLM_MODEL_IMPL="${VLLM_MODEL_IMPL:-auto}"
# VLLM_LORA_ADAPTER：auto 自动查找，none 禁用，或指定 adapter 目录。
VLLM_LORA_ADAPTER="${VLLM_LORA_ADAPTER:-auto}"
# VLLM_MAX_LORA_RANK：vLLM 允许的最大 LoRA rank。
VLLM_MAX_LORA_RANK="${VLLM_MAX_LORA_RANK:-64}"
# VLLM_TRUST_REMOTE_CODE：是否允许 vLLM 加载 LocateAnything 自定义代码。
VLLM_TRUST_REMOTE_CODE="${VLLM_TRUST_REMOTE_CODE:-1}"
# SINGLE_PASS_PROMPT：单次复用使用的 prompt；locate 与训练中生成框 prompt 一致，pose 与 PoseHead 文本 prompt 一致。
SINGLE_PASS_PROMPT="${SINGLE_PASS_PROMPT:-locate}"
# DISABLE_SINGLE_PASS_FEATURES：设为 1 时禁用 transformers 单次复用，回退旧的“先生成框、再提特征”两次前向。
DISABLE_SINGLE_PASS_FEATURES="${DISABLE_SINGLE_PASS_FEATURES:-0}"
# LOCATE_GENERATION_MODE：LocateAnything generate 模式，训练脚本默认 hybrid。
LOCATE_GENERATION_MODE="${LOCATE_GENERATION_MODE:-hybrid}"
# LOCATE_BOX_MAX_NEW_TOKENS：Locate 生成框最大新 token 数；拥挤图需要大，快速 smoke 可调小。
LOCATE_BOX_MAX_NEW_TOKENS="${LOCATE_BOX_MAX_NEW_TOKENS:-8192}"
# BOX_MATCH_IOU_THRESH：生成框和 GT 框匹配阈值，仅影响 loss 对齐，不改变导出预测框。
BOX_MATCH_IOU_THRESH="${BOX_MATCH_IOU_THRESH:-0.10}"
# BOX_NMS_IOU_THRESH：仅在启用 PoseHead 前 NMS 时使用。
BOX_NMS_IOU_THRESH="${BOX_NMS_IOU_THRESH:-0.70}"
# DISABLE_PRE_POSE_NMS：默认保留全部 Locate 框进入 PoseHead。
DISABLE_PRE_POSE_NMS="${DISABLE_PRE_POSE_NMS:-1}"
# POST_POSE_NMS_IOU_THRESH：PoseHead 输出后的高阈值重复框去重。
POST_POSE_NMS_IOU_THRESH="${POST_POSE_NMS_IOU_THRESH:-0.95}"

###############################################################################
# 输出、可视化与筛选
###############################################################################

# VISUALIZE_MAX_SAMPLES：最多保存多少张验证可视化；0 表示关闭。
VISUALIZE_MAX_SAMPLES="${VISUALIZE_MAX_SAMPLES:-100}"
# VISUALIZE_MAX_INSTANCES：单张可视化最多绘制多少个人体实例。
VISUALIZE_MAX_INSTANCES="${VISUALIZE_MAX_INSTANCES:-8}"
# SCORE_THRESHOLD：导出/可视化预测时的 person score 阈值。
SCORE_THRESHOLD="${SCORE_THRESHOLD:-0.05}"
# MAX_PREDICTIONS_PER_IMAGE：单张图最多导出的预测实例数，避免 JSON 过大。
MAX_PREDICTIONS_PER_IMAGE="${MAX_PREDICTIONS_PER_IMAGE:-100}"

###############################################################################
# Loss 权重，仅用于验证 loss 汇总；AP/PCKh 指标不受这些权重影响
###############################################################################

# W_OKS：OKS loss 权重，需要和训练脚本同名以便复现实验配置。
W_OKS="${W_OKS:-0.5}"
# W_COORD：坐标回归 loss 权重。
W_COORD="${W_COORD:-3.0}"
# W_VIS：关键点可见性 BCE loss 权重。
W_VIS="${W_VIS:-0.05}"
# W_HARD_JOINT：hard keypoint mining loss 权重，默认关闭。
W_HARD_JOINT="${W_HARD_JOINT:-0.0}"
# HARD_JOINT_FRACTION：hard mining 选取的可见关键点比例。
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
  --locate_generation_backend "${LOCATE_GENERATION_BACKEND}"
  --gpu "${GPU}"
  --vllm_tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE}"
  --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
  --vllm_cpu_offload_gb "${VLLM_CPU_OFFLOAD_GB}"
  --vllm_max_model_len "${VLLM_MAX_MODEL_LEN}"
  --vllm_batch_size "${VLLM_BATCH_SIZE}"
  --vllm_max_num_seqs "${VLLM_MAX_NUM_SEQS}"
  --vllm_max_num_batched_tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}"
  --vllm_model_impl "${VLLM_MODEL_IMPL}"
  --vllm_lora_adapter "${VLLM_LORA_ADAPTER}"
  --vllm_max_lora_rank "${VLLM_MAX_LORA_RANK}"
  --single_pass_prompt "${SINGLE_PASS_PROMPT}"
  --locate_generation_mode "${LOCATE_GENERATION_MODE}"
  --locate_box_max_new_tokens "${LOCATE_BOX_MAX_NEW_TOKENS}"
  --box_match_iou_thresh "${BOX_MATCH_IOU_THRESH}"
  --box_nms_iou_thresh "${BOX_NMS_IOU_THRESH}"
  --post_pose_nms_iou_thresh "${POST_POSE_NMS_IOU_THRESH}"
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
[[ -n "${LOCATE_IMAGE_TOKEN_LIMIT}" ]] && args+=(--locate_image_token_limit "${LOCATE_IMAGE_TOKEN_LIMIT}")
[[ "${DISABLE_RECORD_CACHE}" == "1" ]] && args+=(--disable_record_cache)
[[ "${DISABLE_SINGLE_PASS_FEATURES}" == "1" ]] && args+=(--disable_single_pass_features)
if [[ "${DISABLE_PRE_POSE_NMS}" == "1" ]]; then
  args+=(--disable_pre_pose_nms)
else
  args+=(--no-disable_pre_pose_nms)
fi
[[ "${PROGRESS_BAR}" == "0" ]] && args+=(--disable_progress)
[[ "${DISABLE_VLLM_FALLBACK}" == "1" ]] && args+=(--disable_vllm_fallback)
[[ "${VLLM_ENFORCE_EAGER}" == "1" ]] && args+=(--vllm_enforce_eager)
[[ "${VLLM_TRUST_REMOTE_CODE}" != "1" ]] && args+=(--no_vllm_trust_remote_code)

echo "CHECKPOINT=${CHECKPOINT}"
echo "EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR}"
echo "DATASETS=${DATASETS}"
echo "BOX_SOURCE=${BOX_SOURCE}"
echo "LOCATE_GENERATION_BACKEND=${LOCATE_GENERATION_BACKEND}"
echo "SINGLE_PASS_PROMPT=${SINGLE_PASS_PROMPT}"
echo "DISABLE_SINGLE_PASS_FEATURES=${DISABLE_SINGLE_PASS_FEATURES}"
echo "DISABLE_PRE_POSE_NMS=${DISABLE_PRE_POSE_NMS}"
echo "POST_POSE_NMS_IOU_THRESH=${POST_POSE_NMS_IOU_THRESH}"
echo "VLLM_BATCH_SIZE=${VLLM_BATCH_SIZE}"
echo "VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS}"
echo "VLLM_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS}"
echo "VLLM_CPU_OFFLOAD_GB=${VLLM_CPU_OFFLOAD_GB}"

"${PYTHON}" -m qwenpose.eval_pose "${args[@]}"
