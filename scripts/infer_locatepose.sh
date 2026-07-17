#!/usr/bin/env bash
set -Eeuo pipefail

###############################################################################
# LocatePose 推理脚本
#
# 支持：
#   1. 指定单张图片 / 图片列表 / 文件夹；
#   2. 文件夹排序后的 start/end/num_images 子集；
#   3. 随机 random_n 张图；
#   4. coco / mpii / crowdpose / refhuman / aic 输出格式；
#   5. RefHuman 任意图片 caption 推理，以及 RefHuman train/val 标注 split 推理；
#   6. 默认由 LocateAnything 生成框，可选择 vLLM/Transformers；
#   7. 可通过 GPU/CUDA_VISIBLE_DEVICES 指定推理 GPU；
#   8. 保存 predictions.jsonl、predictions.json、格式化预测 JSON、manifest 和可视化。
###############################################################################

DEFAULT_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"
SCRIPT_PATH_REL="scripts/$(basename "${BASH_SOURCE[0]}")"

print_usage() {
  cat <<EOF
Usage:
  ${SCRIPT_PATH_REL} --checkpoint PATH --input PATH [--VAR VALUE|--VAR=VALUE]...

Required:
  --checkpoint PATH     checkpoint dir, stage dir, run dir, or qwenpose_checkpoint.pt
  --input PATH          image file or directory; RefHuman root/images dir is also supported

Common examples:
  # COCO17 格式，推理文件夹前 20 张图
  ${SCRIPT_PATH_REL} --checkpoint outputs/locatepose/.../stage3_generated_box_pose_calibration --input demo/images --format coco --end_index 20

  # 随机 10 张 CrowdPose14 格式图片
  ${SCRIPT_PATH_REL} --checkpoint CKPT --input demo/images --format crowdpose --random_n 10 --seed 123

  # 指定一组图片，MPII16 格式
  ${SCRIPT_PATH_REL} --checkpoint CKPT --image_paths 'a.jpg,b.jpg,c.jpg' --format mpii

  # RefHuman 任意图片，所有图片共用一个 caption
  ${SCRIPT_PATH_REL} --checkpoint CKPT --input demo/ref --format refhuman --caption 'the person in red shirt'

  # RefHuman val/train 标注 split；input 可传 refhuman 根目录或 images 目录
  ${SCRIPT_PATH_REL} --checkpoint CKPT --input datasets/refhuman --format refhuman --split val --random_n 5

Options:
  --VAR VALUE           Override one allowed variable. Supports ALL_CAPS, snake_case, and kebab-case.
  --VAR=VALUE           Same as above.
  -h, --help            Show this help message.

Selection variables:
  INPUT                 image file or directory
  IMAGE                 one explicit image path
  IMAGE_PATHS           comma/semicolon/pathsep separated image list
  RECURSIVE             1 to recursively scan input directories
  START_INDEX           sorted image start index, inclusive; default 0
  END_INDEX             sorted image end index, exclusive; empty means no end bound
  NUM_IMAGES            keep at most this many selected images
  RANDOM_N              randomly sample this many images after start/end slicing
  SEED                  random seed
  NUM_PARTS             split selected records into N parts; default 1
  PART_INDEX            0-based part id; alias: WORKER_INDEX

RefHuman variables:
  CAPTION               caption used for arbitrary RefHuman images
  CAPTION_FILE          JSON/JSONL/TSV caption map for arbitrary RefHuman images
  INTERACTIVE_CAPTIONS  1 prompts in TTY when caption is missing; 0 errors out
  REFHUMAN_ROOT         directory containing RefHuman_train.json/RefHuman_val.json and images/
  REFHUMAN_MAX_CAPTIONS_PER_IMAGE  0 keeps all annotation captions for selected images

Backend variables:
  BATCH_SIZE            PyTorch batch size; default 1
  NUM_WORKERS           DataLoader workers; default 0
  BOX_SOURCE            locate_generate|gt；默认 locate_generate
  LOCATE_GENERATION_BACKEND  仅 BOX_SOURCE=locate_generate 生效；vllm|transformers|auto
  SINGLE_PASS_PROMPT     locate|pose；transformers 单次复用时使用纯定位 prompt 或 PoseHead prompt
  DISABLE_SINGLE_PASS_FEATURES 1 表示禁用 transformers 特征复用，回退两次前向
  LOCATE_BOX_MAX_NEW_TOKENS  Locate bbox max new tokens; default 512
  VISUALIZE_MAX_SAMPLES max visualizations; -1 all, 0 off

vLLM integrated variables:
  GPU                        指定 GPU，例如 0、1 或 0,1；默认 0；会导出 CUDA_VISIBLE_DEVICES
  DISABLE_VLLM_FALLBACK      1 表示 vLLM 失败时不回退 transformers；默认 1
  VLLM_BATCH_SIZE            vLLM 请求 batch size；默认跟 BATCH_SIZE/PoseHead batch 同步
  VLLM_MODEL_IMPL            vLLM model_impl；LocateAnything 已注册项目内 custom model，默认 auto
  VLLM_GPU_MEMORY_UTILIZATION vLLM 显存占用比例，默认 0.85
  VLLM_CPU_OFFLOAD_GB        vLLM CPU offload GiB，显存紧张时可设 8/16/20
  VLLM_ENFORCE_EAGER         1 表示强制 vLLM eager 执行，custom model 调试更稳
  VLLM_MAX_NUM_SEQS          vLLM scheduler 最大并发序列数；默认跟 VLLM_BATCH_SIZE 同步
  VLLM_MAX_NUM_BATCHED_TOKENS vLLM prefill/profile token 预算；默认 2048
  VLLM_TENSOR_PARALLEL_SIZE  vLLM tensor parallel size，默认 1
  VLLM_LORA_ADAPTER          auto|none|adapter路径；auto 从 checkpoint 附近查找
  Note: GPU=0,1 only makes two GPUs visible. transformers backend still uses one GPU per process.
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

is_supported_cli_var_name() {
  local normalized_name
  normalized_name="$(normalize_cli_var_name "$1")"
  case "${normalized_name}" in
    LOCATE_BATCH_TOKEN_LIMIT)
      return 0
      ;;
    PROJECT_ROOT|PYTHON|RUN_TS|CHECKPOINT|FORMAT|SPLIT|OUTPUT_ROOT|RUN_NAME|OUTPUT_DIR|LOG_DIR|LOG_FILE|INPUT|IMAGE|IMAGES|IMAGE_PATHS|RECURSIVE|START_INDEX|END_INDEX|NUM_IMAGES|RANDOM_N|SEED|NUM_PARTS|PART_INDEX|WORKER_INDEX|CAPTION|CAPTION_FILE|INTERACTIVE_CAPTIONS|REFHUMAN_ROOT|REFHUMAN_MAX_CAPTIONS_PER_IMAGE|LOCATE_MODEL_PATH|LOCATE_DTYPE|LOCATE_ATTN_IMPLEMENTATION|LOCATE_IMAGE_TOKEN_LIMIT|LOCATE_FEATURE_REFINER_LAYERS|LOCATE_FEATURE_REFINER_BOTTLENECK_DIM|LOCATE_FEATURE_REFINER_INIT_SCALE|LOCATE_LORA_R|LOCATE_LORA_ALPHA|LOCATE_LORA_DROPOUT|LOCATE_VISION_LORA_R|LOCATE_VISION_LORA_ALPHA|LOCATE_VISION_LORA_DROPOUT|HIDDEN_DIM|POSE_DECODER_LAYERS|REFINEMENT_STEPS|DECODER_HEADS|BOX_CONDITION_SCALE|DISABLE_REFINEMENT|GPU|BOX_SOURCE|LOCATE_GENERATION_BACKEND|SINGLE_PASS_PROMPT|DISABLE_SINGLE_PASS_FEATURES|DISABLE_VLLM_FALLBACK|VLLM_TENSOR_PARALLEL_SIZE|VLLM_GPU_MEMORY_UTILIZATION|VLLM_CPU_OFFLOAD_GB|VLLM_ENFORCE_EAGER|VLLM_MAX_NUM_SEQS|VLLM_MAX_NUM_BATCHED_TOKENS|VLLM_MAX_MODEL_LEN|VLLM_BATCH_SIZE|VLLM_MODEL_IMPL|VLLM_LORA_ADAPTER|VLLM_MAX_LORA_RANK|VLLM_TRUST_REMOTE_CODE|DEVICE|BATCH_SIZE|NUM_WORKERS|PREFETCH_FACTOR|MAX_INSTANCES|LOCATE_GENERATION_MODE|LOCATE_BOX_MAX_NEW_TOKENS|BOX_NMS_IOU_THRESH|DISABLE_PRE_POSE_NMS|POST_POSE_NMS_IOU_THRESH|REF_POSE_QUALITY_ALPHA|KEYPOINT_DECODE_MODE|SCORE_THRESHOLD|MAX_PREDICTIONS_PER_IMAGE|VISUALIZE_MAX_SAMPLES|VISUALIZE_MAX_INSTANCES|VISUALIZE_KEYPOINT_VISIBILITY_THRESHOLD|PROGRESS_BAR)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

set_cli_var() {
  local raw_name="$1"
  local value="$2"
  local name
  name="$(normalize_cli_var_name "${raw_name}")"
  if ! is_cli_var_name "${raw_name}" || ! is_supported_cli_var_name "${raw_name}"; then
    echo "Unsupported argument: --${raw_name}" >&2
    print_usage >&2
    exit 1
  fi
  printf -v "${name}" '%s' "${value}"
  export "${name}"
}

while (($# > 0)); do
  case "$1" in
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

DEFAULT_PYTHON="$(resolve_default_python)"
PYTHON="${PYTHON:-${DEFAULT_PYTHON}}"

# RUN_TS：本次推理 run 的时间戳，用于默认 run 名和日志文件名。
RUN_TS="${RUN_TS:-$(date +%Y%m%d-%H%M%S)}"
# CHECKPOINT：LocatePose checkpoint；可传 run/stage/checkpoint 目录或 qwenpose_checkpoint.pt。
CHECKPOINT="${CHECKPOINT:-}"
# FORMAT：推理输出格式，可选 coco/mpii/crowdpose/refhuman/aic。
FORMAT="${FORMAT:-coco}"
# SPLIT：RefHuman 标注 split，仅 FORMAT=refhuman 且使用标注文件时生效。
SPLIT="${SPLIT:-val}"
# OUTPUT_ROOT：推理输出根目录；默认每次 run 建一个子目录。
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/locatepose_infer}"
# RUN_NAME：本次推理 run 名；默认包含格式和时间戳。
RUN_NAME="${RUN_NAME:-locatepose-infer-${FORMAT}-${RUN_TS}}"
# OUTPUT_DIR：本次推理完整输出目录。
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"
# LOG_DIR：日志目录。
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"
# LOG_FILE：完整 stdout/stderr 日志文件。
LOG_FILE="${LOG_FILE:-${LOG_DIR}/infer_${RUN_TS}.log}"

# INPUT：输入图片文件或目录；RefHuman 标注模式也可传数据集根目录或 images 目录。
INPUT="${INPUT:-}"
# IMAGE：显式指定一张图片；适合单图或命令行重复传参。
IMAGE="${IMAGE:-}"
# IMAGE_PATHS：逗号/分号/pathsep 分隔的图片列表；兼容 IMAGES 变量别名。
IMAGE_PATHS="${IMAGE_PATHS:-${IMAGES:-}}"
# RECURSIVE：是否递归扫描 INPUT 目录下的子目录；1 开启，0 关闭。
RECURSIVE="${RECURSIVE:-0}"
# START_INDEX：排序后的图片起始下标，包含该位置。
START_INDEX="${START_INDEX:-0}"
# END_INDEX：排序后的图片结束下标，不包含；空表示直到末尾。
END_INDEX="${END_INDEX:-}"
# NUM_IMAGES：最终最多保留多少张图片；空表示不额外限制。
NUM_IMAGES="${NUM_IMAGES:-}"
# RANDOM_N：在 start/end 切片后随机抽取多少张图；空表示不随机。
RANDOM_N="${RANDOM_N:-}"
# SEED：随机种子，用于 RANDOM_N 的可复现抽样。
SEED="${SEED:-42}"
# NUM_PARTS：最终记录分成几份；多进程多卡时让每个进程跑一份。
NUM_PARTS="${NUM_PARTS:-1}"
# PART_INDEX：当前进程跑第几份；也兼容命令行 --worker_index。
PART_INDEX="${PART_INDEX:-${WORKER_INDEX:-0}}"

# CAPTION：FORMAT=refhuman 任意图片模式下所有图片共用的文本描述。
CAPTION="${CAPTION:-}"
# CAPTION_FILE：FORMAT=refhuman 任意图片模式下的 caption 映射文件，支持 JSON/JSONL/TSV。
CAPTION_FILE="${CAPTION_FILE:-}"
# INTERACTIVE_CAPTIONS：缺少 caption 时是否在 TTY 中询问操作人员；1 开启。
INTERACTIVE_CAPTIONS="${INTERACTIVE_CAPTIONS:-1}"
# REFHUMAN_ROOT：RefHuman 根目录，包含 RefHuman_train.json/RefHuman_val.json 和 images/。
REFHUMAN_ROOT="${REFHUMAN_ROOT:-}"
# REFHUMAN_MAX_CAPTIONS_PER_IMAGE：标注模式下每张图最多使用几条 caption；0 表示全部。
REFHUMAN_MAX_CAPTIONS_PER_IMAGE="${REFHUMAN_MAX_CAPTIONS_PER_IMAGE:-0}"

# LOCATE_MODEL_PATH：LocateAnything-3B 权重目录；应与训练时 locatepose.sh 一致。
LOCATE_MODEL_PATH="${LOCATE_MODEL_PATH:-weights/LocateAnything-3B}"
# LOCATE_DTYPE：PyTorch LocateAnything dtype；4090 推荐 bfloat16。
LOCATE_DTYPE="${LOCATE_DTYPE:-bfloat16}"
# LOCATE_ATTN_IMPLEMENTATION：Locate vision tower attention；默认 flash_attention_2 更快更省显存。
LOCATE_ATTN_IMPLEMENTATION="${LOCATE_ATTN_IMPLEMENTATION:-flash_attention_2}"
# LOCATE_IMAGE_TOKEN_LIMIT：LocateAnything 原生 raw MoonViT patch token 上限；默认 4096，和训练脚本保持一致。
LOCATE_IMAGE_TOKEN_LIMIT="${LOCATE_IMAGE_TOKEN_LIMIT:-4096}"
# 当前训练使用原生可变 P2/P3 网格；feature refiner 默认关闭。
LOCATE_FEATURE_REFINER_LAYERS="${LOCATE_FEATURE_REFINER_LAYERS:-0}"
# LOCATE_FEATURE_REFINER_BOTTLENECK_DIM：Locate feature refiner bottleneck 维度。
LOCATE_FEATURE_REFINER_BOTTLENECK_DIM="${LOCATE_FEATURE_REFINER_BOTTLENECK_DIM:-256}"
# LOCATE_FEATURE_REFINER_INIT_SCALE：Locate feature refiner 残差初始化尺度。
LOCATE_FEATURE_REFINER_INIT_SCALE="${LOCATE_FEATURE_REFINER_INIT_SCALE:-0.1}"
# LOCATE_LORA_R：Locate 主干 LoRA rank；需与训练配置匹配。
LOCATE_LORA_R="${LOCATE_LORA_R:-32}"
# LOCATE_LORA_ALPHA：Locate 主干 LoRA alpha；需与训练配置匹配。
LOCATE_LORA_ALPHA="${LOCATE_LORA_ALPHA:-64}"
# LOCATE_LORA_DROPOUT：Locate 主干 LoRA dropout；推理时用于构建相同 adapter 结构。
LOCATE_LORA_DROPOUT="${LOCATE_LORA_DROPOUT:-0.05}"
# LOCATE_VISION_LORA_R：Locate vision LoRA rank；需与训练配置匹配。
LOCATE_VISION_LORA_R="${LOCATE_VISION_LORA_R:-16}"
# LOCATE_VISION_LORA_ALPHA：Locate vision LoRA alpha；需与训练配置匹配。
LOCATE_VISION_LORA_ALPHA="${LOCATE_VISION_LORA_ALPHA:-32}"
# LOCATE_VISION_LORA_DROPOUT：Locate vision LoRA dropout；推理时用于构建相同 adapter 结构。
LOCATE_VISION_LORA_DROPOUT="${LOCATE_VISION_LORA_DROPOUT:-0.05}"

# HIDDEN_DIM：PoseHead hidden dimension；旧 checkpoint 没有 pose_config 时使用。
HIDDEN_DIM="${HIDDEN_DIM:-448}"
# POSE_DECODER_LAYERS：Pose decoder 层数；旧 checkpoint 没有 pose_config 时使用。
POSE_DECODER_LAYERS="${POSE_DECODER_LAYERS:-3}"
# REFINEMENT_STEPS：关键点 refinement 步数；旧 checkpoint 没有 pose_config 时使用。
REFINEMENT_STEPS="${REFINEMENT_STEPS:-1}"
# DECODER_HEADS：Pose decoder attention head 数；旧 checkpoint 没有 pose_config 时使用。
DECODER_HEADS="${DECODER_HEADS:-8}"
# BOX_CONDITION_SCALE：PoseHead 条件框扩展比例；给关键点预测更多上下文。
BOX_CONDITION_SCALE="${BOX_CONDITION_SCALE:-1.15}"
# DISABLE_REFINEMENT：是否关闭关键点 refinement；1 关闭，0 开启。
DISABLE_REFINEMENT="${DISABLE_REFINEMENT:-0}"

###############################################################################
# Transformers 单次复用推理参数
###############################################################################

# DEVICE：PyTorch PoseHead/Locate 特征推理设备；指定 GPU 后仍使用 cuda 即可。
DEVICE="${DEVICE:-cuda}"
# BATCH_SIZE：PyTorch 姿态推理 batch size；transformers 单次复用会按这个 batch 喂 PoseHead。
BATCH_SIZE="${BATCH_SIZE:-1}"
# NUM_WORKERS：DataLoader worker 数；本地图片/交互 caption 默认 0 更稳。
NUM_WORKERS="${NUM_WORKERS:-0}"
# PREFETCH_FACTOR：DataLoader 预取因子；仅 NUM_WORKERS>0 时生效。
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
# BOX_SOURCE：默认由 LocateAnything 生成人体框；gt 仅用于调试条件框上限。
BOX_SOURCE="${BOX_SOURCE:-locate_generate}"
if [[ "${BOX_SOURCE}" == "gt" ]]; then
  LOCATE_BATCH_TOKEN_LIMIT="${LOCATE_BATCH_TOKEN_LIMIT:-$((BATCH_SIZE * 3072))}"
else
  LOCATE_BATCH_TOKEN_LIMIT="${LOCATE_BATCH_TOKEN_LIMIT:-$((BATCH_SIZE * 4096))}"
fi
# LOCATE_GENERATION_BACKEND：BOX_SOURCE=locate_generate 时使用的生成框后端。
LOCATE_GENERATION_BACKEND="${LOCATE_GENERATION_BACKEND:-vllm}"
# SINGLE_PASS_PROMPT：单次特征复用时使用的 prompt。locate 更贴近纯框生成；pose 更贴近 PoseHead 训练文本。
SINGLE_PASS_PROMPT="${SINGLE_PASS_PROMPT:-locate}"
# DISABLE_SINGLE_PASS_FEATURES：1 表示禁用 transformers 单次特征复用，回退“生成框 + 再提特征”旧路径。
DISABLE_SINGLE_PASS_FEATURES="${DISABLE_SINGLE_PASS_FEATURES:-0}"

###############################################################################
# integrated vLLM 参数
#
# 旧生成框路径的 vLLM custom model 会在同一个 vLLM 模型对象内加载 LocateAnything、
# Locate LoRA、PoseHead 和 feature adapter；Locate 生成 boxes 后，PoseHead
# 复用同一次 vLLM prefill 缓存的图像特征输出 keypoints。
###############################################################################

# GPU：指定推理可见 GPU；例如 0、1 或 0,1。默认单卡 GPU 0。
GPU="${GPU:-0}"
# DISABLE_VLLM_FALLBACK：1 表示 vLLM 初始化/生成失败时直接报错，不回退 transformers。
DISABLE_VLLM_FALLBACK="${DISABLE_VLLM_FALLBACK:-1}"
# VLLM_TENSOR_PARALLEL_SIZE：vLLM tensor parallel 大小；默认 1。多卡 vLLM 可显式设为可见 GPU 数。
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
# VLLM_GPU_MEMORY_UTILIZATION：vLLM 可使用显存比例；4090 24G 默认 0.85 较稳。
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
# VLLM_CPU_OFFLOAD_GB：vLLM 权重 CPU offload 大小；GPU 显存紧张时可设 8/16/20。
VLLM_CPU_OFFLOAD_GB="${VLLM_CPU_OFFLOAD_GB:-0}"
# VLLM_ENFORCE_EAGER：1 表示强制 vLLM eager 执行；自定义模型调试或显存紧张时更稳。
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
# VLLM_MAX_MODEL_LEN：vLLM 最大上下文长度；0 表示使用模型/引擎默认。
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-0}"
# VLLM_BATCH_SIZE：vLLM 请求 batch size；默认与 BATCH_SIZE/PoseHead batch 同步。
VLLM_BATCH_SIZE="${VLLM_BATCH_SIZE:-${BATCH_SIZE}}"
# VLLM_MAX_NUM_SEQS：vLLM scheduler 最大并发序列数；默认与 VLLM_BATCH_SIZE 同步。
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-${VLLM_BATCH_SIZE}}"
# VLLM_MAX_NUM_BATCHED_TOKENS：vLLM prefill/profile token 预算；默认 2048，避免 profile 过大。
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-2048}"
# VLLM_MODEL_IMPL：vLLM 加载模型实现；LocateAnything custom vLLM model 默认用 auto，由项目注册表接管。
VLLM_MODEL_IMPL="${VLLM_MODEL_IMPL:-auto}"
# VLLM_LORA_ADAPTER：vLLM LoRA adapter；auto 自动查找，none 禁用，或指定具体目录。
VLLM_LORA_ADAPTER="${VLLM_LORA_ADAPTER:-auto}"
# VLLM_MAX_LORA_RANK：vLLM 允许的最大 LoRA rank，应覆盖训练时 Locate LoRA rank。
VLLM_MAX_LORA_RANK="${VLLM_MAX_LORA_RANK:-64}"
# VLLM_TRUST_REMOTE_CODE：vLLM 是否允许加载 LocateAnything 自定义模型代码；默认必须开启。
VLLM_TRUST_REMOTE_CODE="${VLLM_TRUST_REMOTE_CODE:-1}"
###############################################################################
# 通用后处理 / 可视化参数
###############################################################################

# MAX_INSTANCES：每张图片最多保留和推理的人体框数量。
MAX_INSTANCES="${MAX_INSTANCES:-80}"
# LOCATE_GENERATION_MODE：LocateAnything generate 模式；hybrid 兼顾速度与稳定性。
LOCATE_GENERATION_MODE="${LOCATE_GENERATION_MODE:-hybrid}"
# LOCATE_BOX_MAX_NEW_TOKENS：Locate 生成框文本最大新 token 数；多人图可适当增大。
LOCATE_BOX_MAX_NEW_TOKENS="${LOCATE_BOX_MAX_NEW_TOKENS:-512}"
# BOX_NMS_IOU_THRESH：仅在启用 PoseHead 前 NMS 时使用。
BOX_NMS_IOU_THRESH="${BOX_NMS_IOU_THRESH:-0.70}"
# DISABLE_PRE_POSE_NMS：默认保留全部 Locate 框进入 PoseHead。
DISABLE_PRE_POSE_NMS="${DISABLE_PRE_POSE_NMS:-1}"
# POST_POSE_NMS_IOU_THRESH：PoseHead 输出后的高阈值重复框去重。
POST_POSE_NMS_IOU_THRESH="${POST_POSE_NMS_IOU_THRESH:-0.95}"
# REF_POSE_QUALITY_ALPHA：RefHuman 排序中 pose quality 的指数；0 只按文本匹配，默认 0.25。
REF_POSE_QUALITY_ALPHA="${REF_POSE_QUALITY_ALPHA:-0.25}"
# KEYPOINT_DECODE_MODE：关键点坐标只使用直接回归头。
KEYPOINT_DECODE_MODE="${KEYPOINT_DECODE_MODE:-regression}"
# SCORE_THRESHOLD：结果筛选和可视化关键点/person score 阈值。
SCORE_THRESHOLD="${SCORE_THRESHOLD:-0.05}"
# MAX_PREDICTIONS_PER_IMAGE：每张图最多写出的预测实例数量。
MAX_PREDICTIONS_PER_IMAGE="${MAX_PREDICTIONS_PER_IMAGE:-100}"
# VISUALIZE_MAX_SAMPLES：最多保存多少张可视化；-1 全部保存，0 关闭。
VISUALIZE_MAX_SAMPLES="${VISUALIZE_MAX_SAMPLES:--1}"
# VISUALIZE_MAX_INSTANCES：每张可视化最多绘制多少个人体实例。
VISUALIZE_MAX_INSTANCES="${VISUALIZE_MAX_INSTANCES:-8}"
# 只绘制可见性回归概率达到阈值的关键点及骨架边。
VISUALIZE_KEYPOINT_VISIBILITY_THRESHOLD="${VISUALIZE_KEYPOINT_VISIBILITY_THRESHOLD:-0.50}"
# PROGRESS_BAR：是否显示 tqdm 进度条；0 关闭，1 开启。
PROGRESS_BAR="${PROGRESS_BAR:-1}"

if [[ -z "${CHECKPOINT}" ]]; then
  echo "CHECKPOINT is required. Example: --checkpoint outputs/locatepose/.../stage3_generated_box_pose_calibration" >&2
  exit 1
fi
if [[ -z "${INPUT}" && -z "${IMAGE}" && -z "${IMAGE_PATHS}" && -z "${REFHUMAN_ROOT}" ]]; then
  echo "At least one of INPUT, IMAGE, IMAGE_PATHS, or REFHUMAN_ROOT is required." >&2
  exit 1
fi
if [[ "${BOX_SOURCE}" != "person_queries" && "${BOX_SOURCE}" != "locate_generate" && "${BOX_SOURCE}" != "gt" ]]; then
  echo "BOX_SOURCE must be person_queries, locate_generate, or gt, got: ${BOX_SOURCE}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}" "$(dirname "${LOG_FILE}")"
touch "${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

add_opt() {
  local -n arr_ref="$1"
  local opt="$2"
  local value="$3"
  if [[ -n "${value}" ]]; then
    arr_ref+=("${opt}" "${value}")
  fi
}

is_enabled() {
  case "${1:-0}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

args=(
  --checkpoint "${CHECKPOINT}"
  --output_dir "${OUTPUT_DIR}"
  --format "${FORMAT}"
  --split "${SPLIT}"
  --start_index "${START_INDEX}"
  --seed "${SEED}"
  --num_parts "${NUM_PARTS}"
  --worker_index "${PART_INDEX}"
  --backbone locatepose
  --locate_model_path "${LOCATE_MODEL_PATH}"
  --locate_dtype "${LOCATE_DTYPE}"
  --locate_attn_implementation "${LOCATE_ATTN_IMPLEMENTATION}"
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
  --device "${DEVICE}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --prefetch_factor "${PREFETCH_FACTOR}"
  --max_instances "${MAX_INSTANCES}"
  --locate_generation_mode "${LOCATE_GENERATION_MODE}"
  --single_pass_prompt "${SINGLE_PASS_PROMPT}"
  --locate_box_max_new_tokens "${LOCATE_BOX_MAX_NEW_TOKENS}"
  --box_nms_iou_thresh "${BOX_NMS_IOU_THRESH}"
  --post_pose_nms_iou_thresh "${POST_POSE_NMS_IOU_THRESH}"
  --ref_pose_quality_alpha "${REF_POSE_QUALITY_ALPHA}"
  --keypoint_decode_mode "${KEYPOINT_DECODE_MODE}"
  --score_threshold "${SCORE_THRESHOLD}"
  --max_predictions_per_image "${MAX_PREDICTIONS_PER_IMAGE}"
  --visualize_max_samples "${VISUALIZE_MAX_SAMPLES}"
  --visualize_max_instances "${VISUALIZE_MAX_INSTANCES}"
  --visualize_keypoint_visibility_threshold "${VISUALIZE_KEYPOINT_VISIBILITY_THRESHOLD}"
  --refhuman_max_captions_per_image "${REFHUMAN_MAX_CAPTIONS_PER_IMAGE}"
)

add_opt args --input "${INPUT}"
add_opt args --image "${IMAGE}"
add_opt args --image_paths "${IMAGE_PATHS}"
add_opt args --end_index "${END_INDEX}"
add_opt args --num_images "${NUM_IMAGES}"
add_opt args --random_n "${RANDOM_N}"
add_opt args --caption "${CAPTION}"
add_opt args --caption_file "${CAPTION_FILE}"
add_opt args --refhuman_root "${REFHUMAN_ROOT}"
add_opt args --locate_image_token_limit "${LOCATE_IMAGE_TOKEN_LIMIT}"
add_opt args --locate_batch_token_limit "${LOCATE_BATCH_TOKEN_LIMIT}"

if is_enabled "${RECURSIVE}"; then
  args+=(--recursive)
fi
if ! is_enabled "${INTERACTIVE_CAPTIONS}"; then
  args+=(--no_interactive_captions)
fi
if is_enabled "${DISABLE_REFINEMENT}"; then
  args+=(--disable_refinement)
fi
if is_enabled "${DISABLE_VLLM_FALLBACK}"; then
  args+=(--disable_vllm_fallback)
fi
if is_enabled "${VLLM_ENFORCE_EAGER}"; then
  args+=(--vllm_enforce_eager)
fi
if is_enabled "${DISABLE_SINGLE_PASS_FEATURES}"; then
  args+=(--disable_single_pass_features)
fi
if is_enabled "${DISABLE_PRE_POSE_NMS}"; then
  args+=(--disable_pre_pose_nms)
else
  args+=(--no-disable_pre_pose_nms)
fi
if ! is_enabled "${VLLM_TRUST_REMOTE_CODE}"; then
  args+=(--no_vllm_trust_remote_code)
fi
if ! is_enabled "${PROGRESS_BAR}"; then
  args+=(--disable_progress)
fi
if [[ -n "${GPU}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU}"
fi

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "CHECKPOINT=${CHECKPOINT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "FORMAT=${FORMAT}"
echo "SPLIT=${SPLIT}"
echo "INPUT=${INPUT}"
echo "IMAGE=${IMAGE}"
echo "IMAGE_PATHS=${IMAGE_PATHS}"
echo "RANDOM_N=${RANDOM_N}"
echo "NUM_PARTS=${NUM_PARTS}"
echo "PART_ID=${PART_INDEX}"
echo "GPU=${GPU}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "LOCATE_ATTN_IMPLEMENTATION=${LOCATE_ATTN_IMPLEMENTATION}"
echo "LOCATE_IMAGE_TOKEN_LIMIT=${LOCATE_IMAGE_TOKEN_LIMIT}"
echo "LOCATE_BATCH_TOKEN_LIMIT=${LOCATE_BATCH_TOKEN_LIMIT}"
echo "BOX_SOURCE=${BOX_SOURCE}"
echo "LOCATE_GENERATION_BACKEND=${LOCATE_GENERATION_BACKEND}"
echo "LOCATE_GENERATION_MODE=${LOCATE_GENERATION_MODE}"
echo "SINGLE_PASS_PROMPT=${SINGLE_PASS_PROMPT}"
echo "DISABLE_SINGLE_PASS_FEATURES=${DISABLE_SINGLE_PASS_FEATURES}"
echo "DISABLE_PRE_POSE_NMS=${DISABLE_PRE_POSE_NMS}"
echo "POST_POSE_NMS_IOU_THRESH=${POST_POSE_NMS_IOU_THRESH}"
echo "REF_POSE_QUALITY_ALPHA=${REF_POSE_QUALITY_ALPHA}"
echo "KEYPOINT_DECODE_MODE=${KEYPOINT_DECODE_MODE}"
echo "VLLM_TENSOR_PARALLEL_SIZE=${VLLM_TENSOR_PARALLEL_SIZE}"
echo "VLLM_BATCH_SIZE=${VLLM_BATCH_SIZE}"
echo "VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS}"
echo "VLLM_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS}"
echo "VLLM_MODEL_IMPL=${VLLM_MODEL_IMPL}"
echo "VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION}"
echo "VLLM_CPU_OFFLOAD_GB=${VLLM_CPU_OFFLOAD_GB}"
echo "VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER}"
echo "LOG_FILE=${LOG_FILE}"

echo "Command: ${PYTHON} -m qwenpose.infer_locatepose ${args[*]}"
"${PYTHON}" -m qwenpose.infer_locatepose "${args[@]}"
