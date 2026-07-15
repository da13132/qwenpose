#!/usr/bin/env bash
set -euo pipefail

# LocatePose4LLM：恢复由 LocateAnything 的 lm_head 自回归生成框的两阶段训练。
#
# 用法：
#   bash scripts/locatepose4llm.sh stage1
#   bash scripts/locatepose4llm.sh stage2
#   bash scripts/locatepose4llm.sh all
#
# Stage 1：完整加载 LocateAnything，但冻结其参数；PoseHead 使用 GT 框训练。
# Stage 2：从 Stage 1 权重初始化，选择性解冻视觉/LLM LoRA；RefHuman 使用
#          lm_head 生成框，所有数据同时用 GT 坐标 token 监督 grounding LM。
# BoxDN、关键点 DN、800×800 高分辨率三层特征金字塔在两个阶段均保留。

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

STAGE="${1:-all}"
case "${STAGE}" in
  stage1|stage2|all) ;;
  *) echo "未知阶段：${STAGE}；仅支持 stage1、stage2、all。" >&2; exit 2 ;;
esac

# 直接指定物理 GPU，不继承调用环境中的 CUDA_VISIBLE_DEVICES。
# 本脚本默认使用物理卡 3；正式多卡训练可显式设置 LOCATEPOSE4LLM_CUDA_VISIBLE_DEVICES。
export CUDA_VISIBLE_DEVICES="${LOCATEPOSE4LLM_CUDA_VISIBLE_DEVICES:-3}"
IFS=',' read -r -a VISIBLE_GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
NPROC_PER_NODE="${#VISIBLE_GPU_LIST[@]}"
if (( NPROC_PER_NODE != 1 )); then
  echo "locatepose4llm.sh 当前采用单进程安全配置；请只指定一张物理 GPU。" >&2
  exit 2
fi

if [[ -x "${ROOT_DIR}/envs/qwenpose/bin/python" ]]; then
  DEFAULT_PYTHON="${ROOT_DIR}/envs/qwenpose/bin/python"
else
  DEFAULT_PYTHON="$(command -v python)"
fi
PYTHON="${PYTHON:-${DEFAULT_PYTHON}}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/locatepose/locatepose4llm-locateanything-3b-${RUN_ID}}"
STAGE1_OUTPUT_DIR="${STAGE1_OUTPUT_DIR:-${OUTPUT_DIR}/stage1_freeze_locate_gt_box}"
STAGE2_OUTPUT_DIR="${STAGE2_OUTPUT_DIR:-${OUTPUT_DIR}/stage2_locate_box_closed_loop}"
STAGE2_INIT_WEIGHTS_DIR="${STAGE2_INIT_WEIGHTS_DIR:-${OUTPUT_DIR}/stage2_init_weights}"

# 数据比例固定为 1:1:1:1；RefHuman 每个人每个 epoch 只取一个、但会轮换字幕。
STAGE1_TRAIN_DATASETS="${STAGE1_TRAIN_DATASETS:-coco,mpii,crowdpose,refhuman}"
STAGE2_TRAIN_DATASETS="${STAGE2_TRAIN_DATASETS:-coco,mpii,crowdpose,refhuman}"
DATASET_MIX_WEIGHTS="${DATASET_MIX_WEIGHTS:-coco:1,mpii:1,crowdpose:1,refhuman:1}"
DATASET_ROOT="${DATASET_ROOT:-datasets}"
MAX_SAMPLES_PER_DATASET="${MAX_SAMPLES_PER_DATASET:-}"

# 模型与显存参数。lm_head 和 KV cache 必须保留，因此显式传 --no-prune_locate_generation。
LOCATE_MODEL_PATH="${LOCATE_MODEL_PATH:-weights/LocateAnything-3B}"
LOCATE_DTYPE="${LOCATE_DTYPE:-bfloat16}"
LOCATE_ATTN_IMPLEMENTATION="${LOCATE_ATTN_IMPLEMENTATION:-flash_attention_2}"
LOCATE_IMAGE_TOKEN_LIMIT="${LOCATE_IMAGE_TOKEN_LIMIT:-1024}"
LOCATE_FEATURE_SIZE="${LOCATE_FEATURE_SIZE:-100}"
LOCATE_VISION_LAYERS="${LOCATE_VISION_LAYERS:-15-26}"
LOCATE_LLM_LAYERS="${LOCATE_LLM_LAYERS:-32-35}"
LOCATE_VISION_MODULES="${LOCATE_VISION_MODULES:-wqkv,wo,fc0,fc1}"
LOCATE_LLM_MODULES="${LOCATE_LLM_MODULES:-q_proj,v_proj}"

# 两阶段训练时长。MAX_STEPS>0 时可把 EPOCHS 设为 0，适合闭环测试。
STAGE1_EPOCHS="${STAGE1_EPOCHS:-30}"
STAGE1_MAX_STEPS="${STAGE1_MAX_STEPS:-0}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-25}"
STAGE2_MAX_STEPS="${STAGE2_MAX_STEPS:-0}"
STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-1}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-1}"
STAGE1_GRAD_ACCUM_STEPS="${STAGE1_GRAD_ACCUM_STEPS:-4}"
STAGE2_GRAD_ACCUM_STEPS="${STAGE2_GRAD_ACCUM_STEPS:-4}"
STAGE1_LR="${STAGE1_LR:-2e-4}"
STAGE2_LR="${STAGE2_LR:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
# AMP 默认关闭；LocateAnything 本身仍按 LOCATE_DTYPE=bfloat16 加载，避免
# GradScaler 在首批次产生无意义的溢出跳步。需要 autocast 时可显式 AMP=1。
AMP="${AMP:-0}"

# Stage 2 的恢复方式：RESUME 恢复完整训练状态；INIT 只取 Stage 1 权重。
STAGE1_RESUME_FROM_CHECKPOINT="${STAGE1_RESUME_FROM_CHECKPOINT:-}"
STAGE2_RESUME_FROM_CHECKPOINT="${STAGE2_RESUME_FROM_CHECKPOINT:-}"
STAGE2_INIT_CHECKPOINT="${STAGE2_INIT_CHECKPOINT:-}"
STAGE2_INIT_FROM_STAGE1="${STAGE2_INIT_FROM_STAGE1:-1}"

# LocateAnything 原生框生成与 teacher-forcing grounding 监督。
LOCATE_BOX_MAX_NEW_TOKENS="${LOCATE_BOX_MAX_NEW_TOKENS:-512}"
LOCATE_GENERATION_MODE="${LOCATE_GENERATION_MODE:-hybrid}"
LOCATE_GENERATE_REFHUMAN_ONLY="${LOCATE_GENERATE_REFHUMAN_ONLY:-1}"
W_LOCATE_BOX_LM="${W_LOCATE_BOX_LM:-0.02}"
LOCATE_LM_LOSS_EVERY="${LOCATE_LM_LOSS_EVERY:-1}"
LOCATE_LM_MAX_INSTANCES="${LOCATE_LM_MAX_INSTANCES:-20}"

mkdir -p "${OUTPUT_DIR}/logs"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/logs/train_${RUN_ID}.log}"

latest_checkpoint() {
  local root="$1"
  "${PYTHON}" - "$root" <<'PY'
import re, sys
from pathlib import Path
root = Path(sys.argv[1])
candidates = []
if (root / "qwenpose_checkpoint.pt").is_file():
    candidates.append((0, root))
for path in root.glob("checkpoint-*"):
    match = re.search(r"checkpoint-(\d+)$", path.name)
    if match and (path / "qwenpose_checkpoint.pt").is_file():
        candidates.append((int(match.group(1)), path))
if not candidates:
    raise SystemExit(1)
print(sorted(candidates)[-1][1])
PY
}

# Stage 1 和 Stage 2 的 optimizer 参数组不同；这里删除 optimizer/RNG/游标，
# 生成 step=0 的仅权重包，避免把冻结阶段的 optimizer 状态错误带入解冻阶段。
prepare_weights_only_checkpoint() {
  local source="$1" destination="$2" resolved
  resolved="$(latest_checkpoint "${source}")" || {
    echo "在 ${source} 下找不到可用 checkpoint。" >&2; exit 1;
  }
  rm -rf "${destination}"
  mkdir -p "${destination}/checkpoint-0"
  "${PYTHON}" - "${resolved}/qwenpose_checkpoint.pt" "${destination}/checkpoint-0/qwenpose_checkpoint.pt" <<'PY'
import sys, torch
src, dst = sys.argv[1:]
try:
    payload = torch.load(src, map_location="cpu", weights_only=False)
except TypeError:
    payload = torch.load(src, map_location="cpu")
for key in ("optimizer", "scaler", "training_state", "rng_state"):
    payload.pop(key, None)
payload["step"] = 0
payload["deepspeed_managed"] = False
payload["weight_only_init_from"] = src
torch.save(payload, dst)
PY
  echo "${destination}/checkpoint-0"
}

common_args() {
  COMMON_ARGS=(
    --dataset_root "${DATASET_ROOT}"
    --dataset_mix_weights "${DATASET_MIX_WEIGHTS}"
    --mixing_strategy interleave
    --refhuman_max_captions_per_instance 1
    --max_instances 80
    --image_size 800
    --num_workers "${NUM_WORKERS}"
    --prefetch_factor "${PREFETCH_FACTOR}"
    --backbone eagle
    --locate_model_path "${LOCATE_MODEL_PATH}"
    --locate_dtype "${LOCATE_DTYPE}"
    --locate_attn_implementation "${LOCATE_ATTN_IMPLEMENTATION}"
    --locate_image_token_limit "${LOCATE_IMAGE_TOKEN_LIMIT}"
    --locate_feature_source raw_visual
    --no-prune_locate_generation
    --locate_feature_size "${LOCATE_FEATURE_SIZE}"
    --locate_vision_layers "${LOCATE_VISION_LAYERS}"
    --locate_llm_layers "${LOCATE_LLM_LAYERS}"
    --locate_vision_modules "${LOCATE_VISION_MODULES}"
    --locate_llm_modules "${LOCATE_LLM_MODULES}"
    --hidden_dim 448
    --human_decoder_layers 2
    --pose_decoder_layers 3
    --refinement_steps 3
    --decoder_heads 8
    --pose_dropout 0
    --box_condition_scale 1.25
    --pose_coordinate_init learned_spread
    --pose_roi_size 16
    --pose_pyramid_channels 128
    --pose_pyramid_blocks 3
    --deformable_points 4
    --deformable_min_radius_cells 2
    --ref_text_scale 0.2
    --max_dn_queries 96
    --max_dn_groups 4
    --dn_positive_noise 0.4
    --dn_negative_noise 1.0
    --max_keypoint_dn_queries 16
    --max_keypoint_dn_groups 2
    --keypoint_dn_positive_ks_min 0.5
    --keypoint_dn_positive_ks_max 1.0
    --keypoint_dn_negative_ks_min 0.1
    --keypoint_dn_negative_ks_max 0.5
    --w_box_dn 0.5
    --w_keypoint_dn 1.0
    --locate_vision_scale 0.10
    --locate_llm_scale 0.01
    --warmup_steps 100
    --log_every 1
    --save_every 500
    --save_total_limit 1
    --device cuda
  )
  [[ "${AMP}" == "1" ]] && COMMON_ARGS+=(--amp)
  if [[ -n "${MAX_SAMPLES_PER_DATASET}" ]]; then
    COMMON_ARGS+=(--max_samples_per_dataset "${MAX_SAMPLES_PER_DATASET}")
  fi
}

run_stage1() {
  common_args
  mkdir -p "${STAGE1_OUTPUT_DIR}"
  local args=("${COMMON_ARGS[@]}"
    --datasets "${STAGE1_TRAIN_DATASETS}"
    --output_dir "${STAGE1_OUTPUT_DIR}"
    --epochs "${STAGE1_EPOCHS}"
    --max_steps "${STAGE1_MAX_STEPS}"
    --batch_size "${STAGE1_BATCH_SIZE}"
    --grad_accum_steps "${STAGE1_GRAD_ACCUM_STEPS}"
    --lr "${STAGE1_LR}"
    --box_source gt
    --locate_train_scope frozen
    --freeze_locate
    --w_locate_box_lm 0
  )
  [[ -n "${STAGE1_RESUME_FROM_CHECKPOINT}" ]] && args+=(--resume_from_checkpoint "${STAGE1_RESUME_FROM_CHECKPOINT}")
  echo "[Stage1] GPU=${CUDA_VISIBLE_DEVICES}，冻结 LocateAnything，GT 框训练 PoseHead。"
  "${PYTHON}" -m qwenpose.train_pose4llm "${args[@]}"
}

run_stage2() {
  common_args
  mkdir -p "${STAGE2_OUTPUT_DIR}"
  local resume_path="${STAGE2_RESUME_FROM_CHECKPOINT}"
  if [[ -z "${resume_path}" ]]; then
    local init_source="${STAGE2_INIT_CHECKPOINT}"
    if [[ -z "${init_source}" && "${STAGE2_INIT_FROM_STAGE1}" == "1" ]]; then
      init_source="${STAGE1_OUTPUT_DIR}"
    fi
    if [[ -n "${init_source}" ]]; then
      resume_path="$(prepare_weights_only_checkpoint "${init_source}" "${STAGE2_INIT_WEIGHTS_DIR}")"
    elif [[ "${STAGE2_INIT_FROM_STAGE1}" == "1" ]]; then
      echo "Stage2 需要 Stage1 权重；请先运行 stage1，或设置 STAGE2_INIT_CHECKPOINT。" >&2
      exit 1
    fi
  fi
  local args=("${COMMON_ARGS[@]}"
    --datasets "${STAGE2_TRAIN_DATASETS}"
    --output_dir "${STAGE2_OUTPUT_DIR}"
    --epochs "${STAGE2_EPOCHS}"
    --max_steps "${STAGE2_MAX_STEPS}"
    --batch_size "${STAGE2_BATCH_SIZE}"
    --grad_accum_steps "${STAGE2_GRAD_ACCUM_STEPS}"
    --lr "${STAGE2_LR}"
    --box_source locate_generate
    --locate_train_scope selective_lora
    --locate_gradient_checkpointing
    --locate_box_max_new_tokens "${LOCATE_BOX_MAX_NEW_TOKENS}"
    --locate_generation_mode "${LOCATE_GENERATION_MODE}"
    --w_locate_box_lm "${W_LOCATE_BOX_LM}"
    --locate_lm_loss_every "${LOCATE_LM_LOSS_EVERY}"
    --locate_lm_max_instances "${LOCATE_LM_MAX_INSTANCES}"
  )
  if [[ "${LOCATE_GENERATE_REFHUMAN_ONLY}" == "1" ]]; then
    args+=(--locate_generate_refhuman_only)
  else
    args+=(--no-locate_generate_refhuman_only)
  fi
  [[ -n "${resume_path}" ]] && args+=(--resume_from_checkpoint "${resume_path}")
  echo "[Stage2] GPU=${CUDA_VISIBLE_DEVICES}，lm_head 生成 RefHuman 框并联合训练 grounding LM + PoseHead。"
  "${PYTHON}" -m qwenpose.train_pose4llm "${args[@]}"
}

echo "LocatePose4LLM 输出目录：${OUTPUT_DIR}"
echo "训练日志：${LOG_FILE}"
{
  [[ "${STAGE}" == "stage1" || "${STAGE}" == "all" ]] && run_stage1
  [[ "${STAGE}" == "stage2" || "${STAGE}" == "all" ]] && run_stage2
} 2>&1 | tee -a "${LOG_FILE}"
