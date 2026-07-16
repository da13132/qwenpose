#!/usr/bin/env bash
set -euo pipefail

# LocatePose 三阶段解耦训练：
#   Stage 1：只加载 MoonViT + mlp1，不加载 LLM；用 GT 框训练 PoseHead，
#            同时训练全部 MoonViT block 的视觉 LoRA 和完整视觉 projector。
#   Stage 2：加载完整 LocateAnything；冻结 PoseHead 和视觉 LoRA，训练指定
#            Qwen2.5 层的 LLM LoRA 与完整视觉 projector，恢复 grounding 能力。
#   Stage 3：冻结完整 LocateAnything；使用其真实生成框训练/校准 PoseHead，
#            解决 GT 框训练与推理生成框之间的分布差异。
#
# 阶段选择示例：
#   bash scripts/locatepose.sh all
#   bash scripts/locatepose.sh stage1
#   bash scripts/locatepose.sh stage2 stage3
#   bash scripts/locatepose.sh stage1,stage3
#   STAGES=stage1+stage2 bash scripts/locatepose.sh
#
# 分开启动后续阶段时，需要复用同一个 OUTPUT_DIR，或者显式指定该阶段的
# INIT_CHECKPOINT。脚本会自动把前一阶段 checkpoint 转成“仅权重初始化包”，
# 不会错误继承不同参数组对应的 optimizer、GradScaler、RNG 或数据游标。

# ==============================================================================
# 0. 项目路径与 Python 模块搜索路径
# ==============================================================================

# ROOT_DIR：项目根目录；根据脚本自身位置自动解析，一般不需要手工修改。
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# PYTHONPATH：优先导入当前工作区 src/ 中的 qwenpose，避免误用旧安装包。
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

# ==============================================================================
# 1. 阶段选择：支持任意一个、任意两个或全部三个阶段
# ==============================================================================

# STAGE_SPEC：阶段选择字符串。优先读取环境变量 STAGES，否则读取全部位置参数；
# 支持逗号、加号或空格分隔，例如 stage1,stage3、stage1+stage2、stage2 stage3。
STAGE_SPEC="${STAGES:-${*:-all}}"
STAGE_SPEC="${STAGE_SPEC//+/,}"
STAGE_SPEC="${STAGE_SPEC// /,}"

# RUN_STAGE1/RUN_STAGE2/RUN_STAGE3：解析后的内部开关；1 表示执行该阶段。
RUN_STAGE1=0
RUN_STAGE2=0
RUN_STAGE3=0
IFS=',' read -r -a REQUESTED_STAGES <<< "${STAGE_SPEC}"
for requested_stage in "${REQUESTED_STAGES[@]}"; do
  [[ -z "${requested_stage}" ]] && continue
  case "${requested_stage}" in
    all)
      RUN_STAGE1=1
      RUN_STAGE2=1
      RUN_STAGE3=1
      ;;
    stage1) RUN_STAGE1=1 ;;
    stage2) RUN_STAGE2=1 ;;
    stage3) RUN_STAGE3=1 ;;
    *)
      echo "未知阶段：${requested_stage}；仅支持 stage1、stage2、stage3、all。" >&2
      exit 2
      ;;
  esac
done
if (( RUN_STAGE1 + RUN_STAGE2 + RUN_STAGE3 == 0 )); then
  echo "没有选择任何训练阶段。" >&2
  exit 2
fi

# ==============================================================================
# 2. GPU、分布式与 Python 环境
# ==============================================================================

# PYTORCH_CUDA_ALLOC_CONF：默认启用 expandable_segments，降低可变视觉 token batch
# 产生的显存碎片；用户显式设置该环境变量时保留用户值。
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# LOCATEPOSE_CUDA_VISIBLE_DEVICES：物理 GPU 编号列表。默认使用后四张卡 4,5,6,7；
# 单卡示例为 7，多卡示例为 4,5,6,7。该值会覆盖终端原 CUDA_VISIBLE_DEVICES。
export CUDA_VISIBLE_DEVICES="${LOCATEPOSE_CUDA_VISIBLE_DEVICES:-4,5,6,7}"
if [[ ! "${CUDA_VISIBLE_DEVICES}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "LOCATEPOSE_CUDA_VISIBLE_DEVICES 格式错误：${CUDA_VISIBLE_DEVICES}；示例：7 或 4,5,6,7。" >&2
  exit 2
fi

# VISIBLE_GPU_LIST：拆分物理卡列表，用于自动计算每机训练进程数。
IFS=',' read -r -a VISIBLE_GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"

# NPROC_PER_NODE：每张可见 GPU 启动一个训练进程，始终由脚本自动计算。
export NPROC_PER_NODE="${#VISIBLE_GPU_LIST[@]}"

# MASTER_ADDR：torch.distributed rendezvous 地址；单机默认使用回环地址。
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

# MASTER_PORT：torch.distributed rendezvous 端口；默认随机选择 20000～29999。
export MASTER_PORT="${MASTER_PORT:-$((20000 + RANDOM % 10000))}"

# DEEPSPEED_CONFIG：多卡训练使用的 DeepSpeed 配置；默认 ZeRO-2，单卡不传入。
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${ROOT_DIR}/scripts/zero2.json}"

# DEFAULT_PYTHON：优先使用项目自带虚拟环境，不存在时回退到 PATH 中的 python。
if [[ -x "${ROOT_DIR}/envs/qwenpose/bin/python" ]]; then
  DEFAULT_PYTHON="${ROOT_DIR}/envs/qwenpose/bin/python"
else
  DEFAULT_PYTHON="$(command -v python)"
fi

# PYTHON：实际训练解释器；需要切换环境时可传入绝对路径覆盖。
PYTHON="${PYTHON:-${DEFAULT_PYTHON}}"

# ==============================================================================
# 3. 输出目录、运行编号与三个阶段目录
# ==============================================================================

# RUN_ID：本次运行标识；默认使用启动时间。分开跑阶段时建议显式复用 OUTPUT_DIR。
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"

# OUTPUT_DIR：三阶段共同根目录；包含 stage1、stage2、stage3、init_weights 和 logs。
OUTPUT_DIR="${OUTPUT_DIR:-outputs/locatepose/locatepose-3stage-${RUN_ID}}"

# STAGE1_OUTPUT_DIR：全范围视觉 LoRA + projector + PoseHead 的 GT 框训练输出目录。
STAGE1_OUTPUT_DIR="${STAGE1_OUTPUT_DIR:-${OUTPUT_DIR}/stage1_vision_gt_pose}"

# STAGE2_OUTPUT_DIR：冻结视觉、只恢复 LocateAnything LLM grounding 的输出目录。
STAGE2_OUTPUT_DIR="${STAGE2_OUTPUT_DIR:-${OUTPUT_DIR}/stage2_restore_locate_grounding}"

# STAGE3_OUTPUT_DIR：冻结 LocateAnything、使用真实生成框校准 PoseHead 的输出目录。
STAGE3_OUTPUT_DIR="${STAGE3_OUTPUT_DIR:-${OUTPUT_DIR}/stage3_generated_box_pose_calibration}"

# STAGE2_INIT_WEIGHTS_DIR：由 Stage1 checkpoint 生成的 Stage2 仅权重初始化包目录。
STAGE2_INIT_WEIGHTS_DIR="${STAGE2_INIT_WEIGHTS_DIR:-${OUTPUT_DIR}/stage2_init_weights}"

# STAGE3_INIT_WEIGHTS_DIR：由 Stage2 或 Stage1 checkpoint 生成的 Stage3 仅权重初始化包目录。
STAGE3_INIT_WEIGHTS_DIR="${STAGE3_INIT_WEIGHTS_DIR:-${OUTPUT_DIR}/stage3_init_weights}"

# ==============================================================================
# 4. 数据集、数据路径与各阶段混合权重
# ==============================================================================

# DATASET_ROOT：所有数据集的共同根目录，内部包含 coco、mpii、crowdpose、refhuman。
DATASET_ROOT="${DATASET_ROOT:-datasets}"

# STAGE1_TRAIN_DATASETS：第一阶段数据集；默认不含 RefHuman，保证可以完全不加载 LLM。
STAGE1_TRAIN_DATASETS="${STAGE1_TRAIN_DATASETS:-coco,mpii,crowdpose}"

# STAGE2_TRAIN_DATASETS：第二阶段 grounding 恢复数据集；默认四个数据集全部参与。
STAGE2_TRAIN_DATASETS="${STAGE2_TRAIN_DATASETS:-coco,mpii,crowdpose,refhuman}"

# STAGE3_TRAIN_DATASETS：第三阶段真实生成框校准数据集；默认四个数据集全部参与。
STAGE3_TRAIN_DATASETS="${STAGE3_TRAIN_DATASETS:-coco,mpii,crowdpose,refhuman}"

# STAGE1_DATASET_MIX_WEIGHTS：第一阶段各数据集每 epoch 的遍历倍率。
STAGE1_DATASET_MIX_WEIGHTS="${STAGE1_DATASET_MIX_WEIGHTS:-coco:1,mpii:1,crowdpose:1}"

# STAGE2_DATASET_MIX_WEIGHTS：第二阶段各数据集每 epoch 的遍历倍率。
STAGE2_DATASET_MIX_WEIGHTS="${STAGE2_DATASET_MIX_WEIGHTS:-coco:1,mpii:1,crowdpose:1,refhuman:1}"

# STAGE3_DATASET_MIX_WEIGHTS：第三阶段各数据集每 epoch 的遍历倍率。
STAGE3_DATASET_MIX_WEIGHTS="${STAGE3_DATASET_MIX_WEIGHTS:-coco:1,mpii:1,crowdpose:1,refhuman:1}"

# MAX_SAMPLES_PER_DATASET：每个数据集最多加载的样本数；空值表示不截断，仅调试时设置。
MAX_SAMPLES_PER_DATASET="${MAX_SAMPLES_PER_DATASET:-}"

# NUM_WORKERS：每个训练进程的 DataLoader worker 数；正式训练默认 8。
NUM_WORKERS="${NUM_WORKERS:-8}"

# PREFETCH_FACTOR：每个 DataLoader worker 的预取 batch 数；仅 NUM_WORKERS>0 生效。
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"

# ==============================================================================
# 5. LocateAnything 权重、精度与视觉 token 预算
# ==============================================================================

# LOCATE_MODEL_PATH：完整 LocateAnything-3B 本地权重目录。Stage1 只从中读取视觉塔
# 和 mlp1；Stage2/3 会读取完整视觉塔、Qwen2.5、tokenizer、processor 和 lm_head。
LOCATE_MODEL_PATH="${LOCATE_MODEL_PATH:-weights/LocateAnything-3B}"

# LOCATE_DTYPE：LocateAnything 权重精度；RTX 4090 默认推荐 bfloat16。
LOCATE_DTYPE="${LOCATE_DTYPE:-bfloat16}"

# LOCATE_ATTN_IMPLEMENTATION：MoonViT 注意力实现；默认 flash_attention_2。
LOCATE_ATTN_IMPLEMENTATION="${LOCATE_ATTN_IMPLEMENTATION:-flash_attention_2}"

# LETTERBOX_SIZE：统一输入画布边长。原图最长边先等比缩放到该值，短边居中 padding；
# 默认固定为 800，因此视觉输入、GT 框和关键点都在同一个 800×800 坐标系中。
LETTERBOX_SIZE="${LETTERBOX_SIZE:-800}"

# LETTERBOX_FILL：短边 padding 的灰度填充值，0～255；默认 127。
LETTERBOX_FILL="${LETTERBOX_FILL:-127}"

# LOCATE_IMAGE_TOKEN_LIMIT：单张图最大原始 MoonViT patch token 数；越大越利于小人体，
# 但显存和计算量也更高。800×800 letterbox 下默认 4096。
LOCATE_IMAGE_TOKEN_LIMIT="${LOCATE_IMAGE_TOKEN_LIMIT:-4096}"

# LOCATE_BATCH_TOKEN_LIMIT：单个本地 micro-batch 的原始视觉 token 总预算；空值表示
# 只使用单图 token 上限。显存波动较大时可设置，例如 12288 或 16384。
LOCATE_BATCH_TOKEN_LIMIT="${LOCATE_BATCH_TOKEN_LIMIT:-}"

# ==============================================================================
# 6. LocatePose/PoseHead 结构参数
# ==============================================================================

# POSE_HIDDEN_DIM：人体 query、关键点 query 和 Transformer 的主隐藏维度。
POSE_HIDDEN_DIM="${POSE_HIDDEN_DIM:-448}"

# POSE_FEATURE_CHANNELS：MoonViT P2/P3 特征经 1×1 投影后的通道数。
POSE_FEATURE_CHANNELS="${POSE_FEATURE_CHANNELS:-256}"

# POSE_HUMAN_DECODER_LAYERS：人体框 query 的迭代细化层数。
POSE_HUMAN_DECODER_LAYERS="${POSE_HUMAN_DECODER_LAYERS:-2}"

# POSE_DECODER_LAYERS：每个人体 ROI 内关键点 grouped decoder 层数。
POSE_DECODER_LAYERS="${POSE_DECODER_LAYERS:-3}"

# POSE_REFINEMENT_STEPS：主 decoder 后局部关键点 refinement 次数。
POSE_REFINEMENT_STEPS="${POSE_REFINEMENT_STEPS:-3}"

# POSE_DECODER_HEADS：人体与关键点 Transformer 的注意力头数。
POSE_DECODER_HEADS="${POSE_DECODER_HEADS:-8}"

# POSE_DROPOUT：PoseHead Transformer dropout；默认 0。
POSE_DROPOUT="${POSE_DROPOUT:-0}"

# POSE_ROI_SIZE：每个人体框采样出的正方形 ROI memory 边长。
POSE_ROI_SIZE="${POSE_ROI_SIZE:-16}"

# POSE_BOX_CONDITION_SCALE：提取姿态 ROI 前对条件框的中心缩放倍率。
POSE_BOX_CONDITION_SCALE="${POSE_BOX_CONDITION_SCALE:-1.25}"

# POSE_DEFORMABLE_POINTS：人体框和关键点可变形注意力每层每尺度采样点数。
POSE_DEFORMABLE_POINTS="${POSE_DEFORMABLE_POINTS:-4}"

# POSE_DEFORMABLE_MIN_RADIUS_CELLS：可变形采样最小搜索半径，单位为原生特征格。
POSE_DEFORMABLE_MIN_RADIUS_CELLS="${POSE_DEFORMABLE_MIN_RADIUS_CELLS:-2}"

# POSE_REF_TEXT_SCALE：Stage3 RefHuman 文本注入 PoseHead query 的缩放系数。
POSE_REF_TEXT_SCALE="${POSE_REF_TEXT_SCALE:-0.2}"

# POSE_COORDINATE_INIT：关键点 reference 初始化方式；默认动态人体先验。
POSE_COORDINATE_INIT="${POSE_COORDINATE_INIT:-anatomical_dynamic}"

# POSE_DYNAMIC_REFERENCE_OFFSET_SCALE：动态人体先验的最大 logit residual 缩放。
POSE_DYNAMIC_REFERENCE_OFFSET_SCALE="${POSE_DYNAMIC_REFERENCE_OFFSET_SCALE:-1.5}"

# ==============================================================================
# 7. 视觉 LoRA、LLM LoRA 与学习率倍率
# ==============================================================================

# LOCATE_VISION_LAYERS：Stage1 允许训练的 MoonViT block；默认覆盖全部 0～26。
LOCATE_VISION_LAYERS="${LOCATE_VISION_LAYERS:-0-26}"

# LOCATE_LLM_LAYERS：Stage2 允许训练的 Qwen2.5 decoder 层；有效编号为 0～35。
LOCATE_LLM_LAYERS="${LOCATE_LLM_LAYERS:-32-35}"

# LOCATE_VISION_MODULES：Stage1 视觉 LoRA 的目标投影；wqkv/wo 为注意力，
# fc0/fc1 为视觉 MLP。
LOCATE_VISION_MODULES="${LOCATE_VISION_MODULES:-wqkv,wo,fc0,fc1}"

# LOCATE_LLM_MODULES：Stage2 LLM LoRA 的目标投影；默认 q_proj,v_proj。
LOCATE_LLM_MODULES="${LOCATE_LLM_MODULES:-q_proj,v_proj}"

# LOCATE_VISION_SCALE：视觉 LoRA 与全量视觉 projector 的学习率倍率。
LOCATE_VISION_SCALE="${LOCATE_VISION_SCALE:-0.10}"

# TRAIN_LOCATE_PROJECTOR：非 frozen 阶段是否全量训练 LocateAnything 视觉 projector。
# 默认开启；设为 0 可显式冻结 projector。
TRAIN_LOCATE_PROJECTOR="${TRAIN_LOCATE_PROJECTOR:-1}"
if [[ ! "${TRAIN_LOCATE_PROJECTOR}" =~ ^[01]$ ]]; then
  echo "TRAIN_LOCATE_PROJECTOR 只能是 0 或 1；当前值：${TRAIN_LOCATE_PROJECTOR}" >&2
  exit 2
fi

# LOCATE_LLM_SCALE：LLM LoRA 学习率相对阶段基础学习率的倍率。
LOCATE_LLM_SCALE="${LOCATE_LLM_SCALE:-0.10}"

# ==============================================================================
# 8. 三个阶段的 epoch、batch、梯度累积与基础学习率
# ==============================================================================

# STAGE1_EPOCHS：Stage1 训练 epoch 数；默认 30。
STAGE1_EPOCHS="${STAGE1_EPOCHS:-30}"

# STAGE1_MAX_STEPS：Stage1 optimizer step 上限；0 表示只由 epoch 控制。
STAGE1_MAX_STEPS="${STAGE1_MAX_STEPS:-0}"

# STAGE1_BATCH_SIZE：Stage1 单卡 micro-batch。800×800 letterbox 下四卡实测：
# batch=12 OOM，batch=10 峰值约 23.6～24.2GB，batch=9 峰值约 21.7～21.9GB。
# 默认 9，在目标 20～24GB 区间内保留长期训练安全余量。
STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-9}"

# STAGE1_GRAD_ACCUM_STEPS：Stage1 梯度累积步数。
STAGE1_GRAD_ACCUM_STEPS="${STAGE1_GRAD_ACCUM_STEPS:-1}"

# STAGE1_LR：Stage1 PoseHead 基础学习率；视觉 LoRA 再乘 LOCATE_VISION_SCALE。
STAGE1_LR="${STAGE1_LR:-2e-4}"

# STAGE2_EPOCHS：Stage2 grounding 恢复 epoch 数；默认 10。
STAGE2_EPOCHS="${STAGE2_EPOCHS:-10}"

# STAGE2_MAX_STEPS：Stage2 optimizer step 上限；0 表示只由 epoch 控制。
STAGE2_MAX_STEPS="${STAGE2_MAX_STEPS:-0}"

# STAGE2_BATCH_SIZE：Stage2 单卡 teacher-forcing micro-batch；800×800 下实测
# batch=4 四卡峰值约 21.6GB，默认使用 4。
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-4}"

# STAGE2_GRAD_ACCUM_STEPS：Stage2 梯度累积步数；batch=4 时默认 1，
# 四卡有效 global batch 仍为 16。
STAGE2_GRAD_ACCUM_STEPS="${STAGE2_GRAD_ACCUM_STEPS:-1}"

# STAGE2_LR：Stage2 基础学习率；实际 LLM LoRA 学习率再乘 LOCATE_LLM_SCALE。
STAGE2_LR="${STAGE2_LR:-1e-4}"

# STAGE3_EPOCHS：Stage3 真实生成框校准 epoch 数；默认 5。
STAGE3_EPOCHS="${STAGE3_EPOCHS:-5}"

# STAGE3_MAX_STEPS：Stage3 optimizer step 上限；0 表示只由 epoch 控制。
STAGE3_MAX_STEPS="${STAGE3_MAX_STEPS:-0}"

# STAGE3_BATCH_SIZE：Stage3 单卡 micro-batch；真实生成框逐样本自回归，显存主要由
# 冻结 LocateAnything 占用。800×800 下 batch=4 实测峰值约 11.3～11.5GB。
STAGE3_BATCH_SIZE="${STAGE3_BATCH_SIZE:-4}"

# STAGE3_GRAD_ACCUM_STEPS：Stage3 默认 1；四卡 effective global batch 仍为 16。
STAGE3_GRAD_ACCUM_STEPS="${STAGE3_GRAD_ACCUM_STEPS:-1}"

# STAGE3_LR：Stage3 只训练 PoseHead 的基础学习率；默认 5e-5。
STAGE3_LR="${STAGE3_LR:-5e-5}"

# AMP：是否启用训练代码 autocast/GradScaler；0 关闭，1 开启。
AMP="${AMP:-0}"

# WARMUP_STEPS：每个阶段前多少 optimizer step 线性升温。
WARMUP_STEPS="${WARMUP_STEPS:-100}"

# ==============================================================================
# 9. checkpoint 恢复、阶段初始化与回退规则
# ==============================================================================

# STAGE1_RESUME_FROM_CHECKPOINT：Stage1 完整断点续训来源；空值表示新训练。
STAGE1_RESUME_FROM_CHECKPOINT="${STAGE1_RESUME_FROM_CHECKPOINT:-}"

# STAGE2_RESUME_FROM_CHECKPOINT：Stage2 完整断点续训来源；优先级最高。
STAGE2_RESUME_FROM_CHECKPOINT="${STAGE2_RESUME_FROM_CHECKPOINT:-}"

# STAGE2_INIT_CHECKPOINT：Stage2 仅权重初始化来源；通常指向 Stage1 输出目录。
STAGE2_INIT_CHECKPOINT="${STAGE2_INIT_CHECKPOINT:-}"

# STAGE2_INIT_FROM_STAGE1：未显式提供 Stage2 初始化来源时，是否自动读取 Stage1。
STAGE2_INIT_FROM_STAGE1="${STAGE2_INIT_FROM_STAGE1:-1}"

# STAGE3_RESUME_FROM_CHECKPOINT：Stage3 完整断点续训来源；优先级最高。
STAGE3_RESUME_FROM_CHECKPOINT="${STAGE3_RESUME_FROM_CHECKPOINT:-}"

# STAGE3_INIT_CHECKPOINT：Stage3 仅权重初始化来源；可指向 Stage2 或 Stage1。
STAGE3_INIT_CHECKPOINT="${STAGE3_INIT_CHECKPOINT:-}"

# STAGE3_INIT_FROM_STAGE2：未显式提供 Stage3 初始化来源时，是否优先读取 Stage2。
STAGE3_INIT_FROM_STAGE2="${STAGE3_INIT_FROM_STAGE2:-1}"

# STAGE3_ALLOW_STAGE1_FALLBACK：Stage2 不存在时，是否允许 Stage3 回退到 Stage1。
STAGE3_ALLOW_STAGE1_FALLBACK="${STAGE3_ALLOW_STAGE1_FALLBACK:-1}"

# ==============================================================================
# 10. Pose、人体框、DN、grounding LM 与 PBD loss 权重
# ==============================================================================

# W_OKS：最终关键点标准 OKS loss 权重。
W_OKS="${W_OKS:-1.0}"

# W_IMAGE_COORD：最终关键点整图归一化 SmoothL1 loss 权重。
W_IMAGE_COORD="${W_IMAGE_COORD:-5.0}"

# W_KEYPOINT_CONFIDENCE：每个关键点定位质量置信度 loss 权重。
W_KEYPOINT_CONFIDENCE="${W_KEYPOINT_CONFIDENCE:-0.1}"

# W_PERSON_CONFIDENCE：人体实例质量置信度 loss 权重；0 表示关闭该头。
W_PERSON_CONFIDENCE="${W_PERSON_CONFIDENCE:-0.0}"

# W_REF_MATCH：RefHuman 文本与人物候选匹配 loss 权重；Stage1 无 RefHuman。
W_REF_MATCH="${W_REF_MATCH:-1.0}"

# W_HARD_JOINT：困难关键点额外 loss 权重；0 表示关闭。
W_HARD_JOINT="${W_HARD_JOINT:-0.0}"

# HARD_JOINT_FRACTION：被视为困难关键点的比例。
HARD_JOINT_FRACTION="${HARD_JOINT_FRACTION:-0.2}"

# W_DECODER_COORDS：逐层 grouped decoder 的框内归一化坐标辅助权重。
W_DECODER_COORDS="${W_DECODER_COORDS:-0.25,0.5,0.75}"

# W_COARSE_COORD：粗关键点框内坐标辅助 loss 权重。
W_COARSE_COORD="${W_COARSE_COORD:-0.5}"

# W_DEFORM_COORD：可变形关键点阶段坐标辅助 loss 权重。
W_DEFORM_COORD="${W_DEFORM_COORD:-0.75}"

# W_REFINE_COORDS：最终输出之前各 refinement 坐标辅助权重。
W_REFINE_COORDS="${W_REFINE_COORDS:-0.75,1.0}"

# W_BOX_OBJECTNESS：人体框前景/背景分类 loss 权重。
W_BOX_OBJECTNESS="${W_BOX_OBJECTNESS:-1.0}"

# W_BOX_L1：人体框 L1 loss 权重。
W_BOX_L1="${W_BOX_L1:-5.0}"

# W_BOX_GIOU：人体框 GIoU loss 权重。
W_BOX_GIOU="${W_BOX_GIOU:-2.0}"

# W_BOX_RELATIVE：人体框相对条件框偏移约束权重。
W_BOX_RELATIVE="${W_BOX_RELATIVE:-1.0}"

# W_BOX_DN：BoxDN 正负去噪总 loss 权重。
W_BOX_DN="${W_BOX_DN:-0.5}"

# W_KEYPOINT_DN：关键点 DN 重建与对比总 loss 权重。
W_KEYPOINT_DN="${W_KEYPOINT_DN:-1.0}"

# W_LOCATE_BOX_LM：Stage2 GT 坐标 token teacher-forcing CE 权重。
W_LOCATE_BOX_LM="${W_LOCATE_BOX_LM:-0.05}"

# W_LOCATE_PBD：Stage2 六位置框结构、坐标和连续几何 PBD loss 外层权重。
W_LOCATE_PBD="${W_LOCATE_PBD:-0.05}"

# PBD_TEMPERATURE：Stage2 soft coordinate expectation 温度。
PBD_TEMPERATURE="${PBD_TEMPERATURE:-1.0}"

# LOCATE_LM_LOSS_EVERY：Stage2 每隔多少 micro-step 加一次 LM CE；PBD 仍每批计算。
LOCATE_LM_LOSS_EVERY="${LOCATE_LM_LOSS_EVERY:-1}"

# LOCATE_LM_MAX_INSTANCES：普通多人样本写入 grounding 答案的最大人体框数。
LOCATE_LM_MAX_INSTANCES="${LOCATE_LM_MAX_INSTANCES:-30}"

# ==============================================================================
# 11. Stage3 真实生成框配置
# ==============================================================================

# LOCATE_BOX_MAX_NEW_TOKENS：LocateAnything 单样本生成响应的最大新 token 数。
LOCATE_BOX_MAX_NEW_TOKENS="${LOCATE_BOX_MAX_NEW_TOKENS:-512}"

# LOCATE_GENERATION_MODE：LocateAnything 原生生成策略，可取 fast、slow、hybrid。
LOCATE_GENERATION_MODE="${LOCATE_GENERATION_MODE:-hybrid}"

# STAGE3_GENERATE_REFHUMAN_ONLY：1 表示只有 RefHuman 使用生成框，普通姿态数据用 GT；
# 默认 0，四个数据集全部使用真实 LocateAnything 生成框进行校准。
STAGE3_GENERATE_REFHUMAN_ONLY="${STAGE3_GENERATE_REFHUMAN_ONLY:-0}"

# ==============================================================================
# 12. 日志、checkpoint、可视化与 smoke test 参数
# ==============================================================================

# LOG_EVERY：每隔多少 optimizer step 打印一次详细 loss 和显存信息。
LOG_EVERY="${LOG_EVERY:-1}"

# SAVE_EVERY：每隔多少 optimizer step 保存一个滚动 checkpoint。
SAVE_EVERY="${SAVE_EVERY:-500}"

# SAVE_TOTAL_LIMIT：最多保留多少个滚动 checkpoint。
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"

# VISUALIZE_EVERY：Stage1/3 每隔多少 optimizer step 保存训练预测图；0 表示关闭。
VISUALIZE_EVERY="${VISUALIZE_EVERY:-10}"

# VISUALIZE_MAX_INSTANCES：单张训练可视化最多绘制的人体实例数。
VISUALIZE_MAX_INSTANCES="${VISUALIZE_MAX_INSTANCES:-30}"

# VISUALIZE_MIN_GT_AREA_RATIO：最大 GT 人体面积低于该比例时跳过可视化。
VISUALIZE_MIN_GT_AREA_RATIO="${VISUALIZE_MIN_GT_AREA_RATIO:-0.005}"

# LOG_FILE：三个阶段共享的终端日志文件。
mkdir -p "${OUTPUT_DIR}/logs"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/logs/train_${RUN_ID}.log}"

# ==============================================================================
# 13. 内部辅助函数：checkpoint、启动器与配置汇总
# ==============================================================================

# latest_checkpoint：从阶段目录或 checkpoint 目录解析最新可用 checkpoint。
latest_checkpoint() {
  local root="$1"
  "${PYTHON}" - "$root" <<'PY'
import re
import sys
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

# prepare_weights_only_checkpoint：删除 optimizer/RNG/游标，生成 step=0 初始化包。
prepare_weights_only_checkpoint() {
  local source="$1" destination="$2" resolved
  resolved="$(latest_checkpoint "${source}")" || {
    echo "在 ${source} 下找不到可用 checkpoint。" >&2
    exit 1
  }
  rm -rf "${destination}"
  mkdir -p "${destination}/checkpoint-0"
  "${PYTHON}" - "${resolved}/qwenpose_checkpoint.pt" "${destination}/checkpoint-0/qwenpose_checkpoint.pt" <<'PY'
import sys
import torch
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

# run_train_pose：单卡直接运行，多卡使用 torch.distributed.run 启动每卡一个进程。
run_train_pose() {
  if (( NPROC_PER_NODE == 1 )); then
    "${PYTHON}" -m qwenpose.train_pose "$@"
  else
    "${PYTHON}" -m torch.distributed.run \
      --nproc_per_node "${NPROC_PER_NODE}" \
      --master_addr "${MASTER_ADDR}" \
      --master_port "${MASTER_PORT}" \
      "${ROOT_DIR}/src/qwenpose/train_pose.py" "$@"
  fi
}

# common_args：构造三个阶段共同参数；第一个参数是当前阶段的数据混合权重。
common_args() {
  local stage_mix_weights="$1"
  COMMON_ARGS=(
    # --dataset_root：数据集共同根目录。
    --dataset_root "${DATASET_ROOT}"
    # --dataset_mix_weights：当前阶段各数据集遍历倍率。
    --dataset_mix_weights "${stage_mix_weights}"
    # --mixing_strategy：按数据源权重轮转，保证同一 batch 来源一致。
    --mixing_strategy interleave
    # --refhuman_max_captions_per_instance：每个人物每 epoch 只使用一条轮换字幕。
    --refhuman_max_captions_per_instance 1
    # --max_instances：单图最多保留的人体实例数。
    --max_instances 80
    # --letterbox_size：最长边缩放到 800，短边居中 padding 到 800×800。
    --letterbox_size "${LETTERBOX_SIZE}"
    # --letterbox_fill：letterbox padding 灰度值。
    --letterbox_fill "${LETTERBOX_FILL}"
    # --num_workers：每个训练进程的 DataLoader worker 数。
    --num_workers "${NUM_WORKERS}"
    # --prefetch_factor：每个 worker 的预取 batch 数。
    --prefetch_factor "${PREFETCH_FACTOR}"
    # --disable_vision_token_balancing：所有输入已统一为 800×800，禁用按原图尺寸分桶，
    # 避免再次把同类超大原图聚成一批；实际视觉 token 数由固定画布决定。
    --disable_vision_token_balancing

    # --backbone：eagle 表示 LocateAnything/LocatePose 骨干。
    --backbone eagle
    # --locate_model_path：LocateAnything-3B 权重路径。
    --locate_model_path "${LOCATE_MODEL_PATH}"
    # --locate_dtype：LocateAnything 权重精度。
    --locate_dtype "${LOCATE_DTYPE}"
    # --locate_attn_implementation：MoonViT 注意力实现。
    --locate_attn_implementation "${LOCATE_ATTN_IMPLEMENTATION}"
    # --locate_image_token_limit：单图视觉 token 上限。
    --locate_image_token_limit "${LOCATE_IMAGE_TOKEN_LIMIT}"

    # --locate_vision_layers：可选择的 MoonViT LoRA 层范围。
    --locate_vision_layers "${LOCATE_VISION_LAYERS}"
    # --locate_llm_layers：可选择的 Qwen2.5 LoRA 层范围。
    --locate_llm_layers "${LOCATE_LLM_LAYERS}"
    # --locate_vision_modules：视觉 LoRA 目标投影。
    --locate_vision_modules "${LOCATE_VISION_MODULES}"
    # --locate_llm_modules：LLM LoRA 目标投影。
    --locate_llm_modules "${LOCATE_LLM_MODULES}"

    # --hidden_dim：PoseHead 主隐藏维度。
    --hidden_dim "${POSE_HIDDEN_DIM}"
    # --human_decoder_layers：人体框 decoder 层数。
    --human_decoder_layers "${POSE_HUMAN_DECODER_LAYERS}"
    # --pose_decoder_layers：关键点 grouped decoder 层数。
    --pose_decoder_layers "${POSE_DECODER_LAYERS}"
    # --refinement_steps：局部关键点 refinement 次数。
    --refinement_steps "${POSE_REFINEMENT_STEPS}"
    # --decoder_heads：Transformer 注意力头数。
    --decoder_heads "${POSE_DECODER_HEADS}"
    # --pose_dropout：PoseHead dropout。
    --pose_dropout "${POSE_DROPOUT}"
    # --box_condition_scale：姿态 ROI 条件框放大倍率。
    --box_condition_scale "${POSE_BOX_CONDITION_SCALE}"
    # --pose_coordinate_init：关键点 reference 初始化方式。
    --pose_coordinate_init "${POSE_COORDINATE_INIT}"
    # --dynamic_reference_offset_scale：动态人体先验 residual 缩放。
    --dynamic_reference_offset_scale "${POSE_DYNAMIC_REFERENCE_OFFSET_SCALE}"
    # --pose_roi_size：人体 ROI memory 边长。
    --pose_roi_size "${POSE_ROI_SIZE}"
    # --pose_feature_channels：P2/P3 投影通道数。
    --pose_feature_channels "${POSE_FEATURE_CHANNELS}"
    # --deformable_points：可变形注意力采样点数。
    --deformable_points "${POSE_DEFORMABLE_POINTS}"
    # --deformable_min_radius_cells：最小可变形搜索半径。
    --deformable_min_radius_cells "${POSE_DEFORMABLE_MIN_RADIUS_CELLS}"
    # --ref_text_scale：RefHuman 文本条件缩放。
    --ref_text_scale "${POSE_REF_TEXT_SCALE}"

    # --w_oks：最终姿态 OKS loss 权重。
    --w_oks "${W_OKS}"
    # --w_image_coord：最终整图坐标 loss 权重。
    --w_image_coord "${W_IMAGE_COORD}"
    # --w_keypoint_confidence：关键点质量置信度 loss 权重。
    --w_keypoint_confidence "${W_KEYPOINT_CONFIDENCE}"
    # --w_person_confidence：人体实例置信度 loss 权重。
    --w_person_confidence "${W_PERSON_CONFIDENCE}"
    # --w_ref_match：RefHuman 文本匹配 loss 权重。
    --w_ref_match "${W_REF_MATCH}"
    # --w_hard_joint：困难关键点额外 loss 权重。
    --w_hard_joint "${W_HARD_JOINT}"
    # --hard_joint_fraction：困难关键点比例。
    --hard_joint_fraction "${HARD_JOINT_FRACTION}"
    # --w_decoder_coords：各 pose decoder 层坐标辅助权重。
    --w_decoder_coords "${W_DECODER_COORDS}"
    # --w_coarse_coord：粗坐标辅助权重。
    --w_coarse_coord "${W_COARSE_COORD}"
    # --w_deform_coord：可变形坐标辅助权重。
    --w_deform_coord "${W_DEFORM_COORD}"
    # --w_refine_coords：prefinal refinement 坐标辅助权重。
    --w_refine_coords "${W_REFINE_COORDS}"
    # --w_box_objectness：人体框前景分类权重。
    --w_box_objectness "${W_BOX_OBJECTNESS}"
    # --w_box_l1：人体框 L1 权重。
    --w_box_l1 "${W_BOX_L1}"
    # --w_box_giou：人体框 GIoU 权重。
    --w_box_giou "${W_BOX_GIOU}"
    # --w_box_relative：人体框相对偏移权重。
    --w_box_relative "${W_BOX_RELATIVE}"

    # --max_dn_queries：BoxDN 最大 query 数。
    --max_dn_queries 96
    # --max_dn_groups：BoxDN 最大组数。
    --max_dn_groups 4
    # --dn_positive_noise：BoxDN 正样本噪声强度。
    --dn_positive_noise 0.4
    # --dn_negative_noise：BoxDN 负样本噪声强度。
    --dn_negative_noise 1.0
    # --max_keypoint_dn_queries：关键点 DN 最大 query 数。
    --max_keypoint_dn_queries 16
    # --max_keypoint_dn_groups：关键点 DN 最大组数。
    --max_keypoint_dn_groups 2
    # --keypoint_dn_positive_ks_min：关键点 DN 正样本最低 KS。
    --keypoint_dn_positive_ks_min 0.5
    # --keypoint_dn_positive_ks_max：关键点 DN 正样本最高 KS。
    --keypoint_dn_positive_ks_max 1.0
    # --keypoint_dn_negative_ks_min：关键点 DN 负样本最低 KS。
    --keypoint_dn_negative_ks_min 0.1
    # --keypoint_dn_negative_ks_max：关键点 DN 负样本最高 KS。
    --keypoint_dn_negative_ks_max 0.5
    # --w_box_dn：BoxDN loss 权重。
    --w_box_dn "${W_BOX_DN}"
    # --w_keypoint_dn：关键点 DN loss 权重。
    --w_keypoint_dn "${W_KEYPOINT_DN}"

    # --locate_vision_scale：视觉 LoRA 学习率倍率。
    --locate_vision_scale "${LOCATE_VISION_SCALE}"
    # --locate_llm_scale：LLM LoRA 学习率倍率。
    --locate_llm_scale "${LOCATE_LLM_SCALE}"
    # --warmup_steps：学习率线性升温步数。
    --warmup_steps "${WARMUP_STEPS}"
    # --log_every：详细日志间隔。
    --log_every "${LOG_EVERY}"
    # --save_every：滚动 checkpoint 保存间隔。
    --save_every "${SAVE_EVERY}"
    # --save_total_limit：滚动 checkpoint 保留数量。
    --save_total_limit "${SAVE_TOTAL_LIMIT}"
    # --visualize_every：训练可视化间隔。
    --visualize_every "${VISUALIZE_EVERY}"
    # --visualize_max_instances：单图可视化最大人体数。
    --visualize_max_instances "${VISUALIZE_MAX_INSTANCES}"
    # --visualize_min_gt_area_ratio：小人体可视化过滤阈值。
    --visualize_min_gt_area_ratio "${VISUALIZE_MIN_GT_AREA_RATIO}"
    # --device：训练设备类型；物理卡由 CUDA_VISIBLE_DEVICES 控制。
    --device cuda
  )

  # --train_locate_projector：默认全量训练 LocateAnything 视觉 projector；
  # 可用 TRAIN_LOCATE_PROJECTOR=0 显式关闭。
  if [[ "${TRAIN_LOCATE_PROJECTOR}" == "1" ]]; then
    COMMON_ARGS+=(--train_locate_projector)
  else
    COMMON_ARGS+=(--no-train_locate_projector)
  fi

  # --locate_batch_token_limit：可选的本地 micro-batch 视觉 token 总预算。
  if [[ -n "${LOCATE_BATCH_TOKEN_LIMIT}" ]]; then
    COMMON_ARGS+=(--locate_batch_token_limit "${LOCATE_BATCH_TOKEN_LIMIT}")
  fi

  # --amp：AMP=1 时启用训练代码 autocast/GradScaler。
  [[ "${AMP}" == "1" ]] && COMMON_ARGS+=(--amp)

  # --max_samples_per_dataset：仅 smoke test/调试时追加样本数上限。
  if [[ -n "${MAX_SAMPLES_PER_DATASET}" ]]; then
    COMMON_ARGS+=(--max_samples_per_dataset "${MAX_SAMPLES_PER_DATASET}")
  fi

  # --deepspeed_config：多卡时启用 ZeRO-2；单卡不传入。
  if (( NPROC_PER_NODE > 1 )); then
    if [[ ! -f "${DEEPSPEED_CONFIG}" ]]; then
      echo "多卡训练所需的 DeepSpeed 配置不存在：${DEEPSPEED_CONFIG}" >&2
      exit 2
    fi
    COMMON_ARGS+=(--deepspeed_config "${DEEPSPEED_CONFIG}")
  fi
}

# print_configuration_summary：启动前把关键有效配置写入统一日志。
print_configuration_summary() {
  local stage1_effective stage2_effective stage3_effective
  stage1_effective=$((NPROC_PER_NODE * STAGE1_BATCH_SIZE * STAGE1_GRAD_ACCUM_STEPS))
  stage2_effective=$((NPROC_PER_NODE * STAGE2_BATCH_SIZE * STAGE2_GRAD_ACCUM_STEPS))
  stage3_effective=$((NPROC_PER_NODE * STAGE3_BATCH_SIZE * STAGE3_GRAD_ACCUM_STEPS))
  cat <<EOF
================ LocatePose 三阶段训练配置 ================
选择阶段：stage1=${RUN_STAGE1} stage2=${RUN_STAGE2} stage3=${RUN_STAGE3}
物理 GPU：${CUDA_VISIBLE_DEVICES}；进程数：${NPROC_PER_NODE}
输出根目录：${OUTPUT_DIR}
Stage1：vision_only + GT box + full-range vision LoRA + projector；datasets=${STAGE1_TRAIN_DATASETS}
        epochs=${STAGE1_EPOCHS} batch/gpu=${STAGE1_BATCH_SIZE} accum=${STAGE1_GRAD_ACCUM_STEPS} effective=${stage1_effective} lr=${STAGE1_LR}
Stage2：raw_visual + grounding_only + freeze_pose + selective_llm_lora + projector；datasets=${STAGE2_TRAIN_DATASETS}
        epochs=${STAGE2_EPOCHS} batch/gpu=${STAGE2_BATCH_SIZE} accum=${STAGE2_GRAD_ACCUM_STEPS} effective=${stage2_effective} lr=${STAGE2_LR}
Stage3：raw_visual + hard Locate boxes + freeze Locate + train PoseHead；datasets=${STAGE3_TRAIN_DATASETS}
        epochs=${STAGE3_EPOCHS} batch/gpu=${STAGE3_BATCH_SIZE} accum=${STAGE3_GRAD_ACCUM_STEPS} effective=${stage3_effective} lr=${STAGE3_LR}
视觉 LoRA：layers=${LOCATE_VISION_LAYERS} modules=${LOCATE_VISION_MODULES} lr_scale=${LOCATE_VISION_SCALE}
视觉 Projector：full_train=${TRAIN_LOCATE_PROJECTOR} lr_scale=${LOCATE_VISION_SCALE}
LLM LoRA：layers=${LOCATE_LLM_LAYERS} modules=${LOCATE_LLM_MODULES} lr_scale=${LOCATE_LLM_SCALE}
统一输入：letterbox=${LETTERBOX_SIZE}x${LETTERBOX_SIZE} fill=${LETTERBOX_FILL}
视觉 token：image_limit=${LOCATE_IMAGE_TOKEN_LIMIT} batch_limit=${LOCATE_BATCH_TOKEN_LIMIT:-unlimited}
Stage2 grounding：lm=${W_LOCATE_BOX_LM} pbd=${W_LOCATE_PBD} temperature=${PBD_TEMPERATURE}
Stage3 generation：mode=${LOCATE_GENERATION_MODE} max_new_tokens=${LOCATE_BOX_MAX_NEW_TOKENS} refhuman_only=${STAGE3_GENERATE_REFHUMAN_ONLY}
日志：${LOG_FILE}
============================================================
EOF
}

# ==============================================================================
# 14. Stage1：纯视觉 GT 框训练 PoseHead + 指定视觉 LoRA
# ==============================================================================

run_stage1() {
  common_args "${STAGE1_DATASET_MIX_WEIGHTS}"
  mkdir -p "${STAGE1_OUTPUT_DIR}"
  local args=("${COMMON_ARGS[@]}"
    # --datasets：Stage1 默认仅 COCO/MPII/CrowdPose，不训练 RefHuman。
    --datasets "${STAGE1_TRAIN_DATASETS}"
    # --output_dir：Stage1 输出目录。
    --output_dir "${STAGE1_OUTPUT_DIR}"
    # --epochs：Stage1 epoch 数。
    --epochs "${STAGE1_EPOCHS}"
    # --max_steps：Stage1 optimizer step 上限。
    --max_steps "${STAGE1_MAX_STEPS}"
    # --batch_size：Stage1 单卡 micro-batch。
    --batch_size "${STAGE1_BATCH_SIZE}"
    # --grad_accum_steps：Stage1 梯度累积步数。
    --grad_accum_steps "${STAGE1_GRAD_ACCUM_STEPS}"
    # --lr：Stage1 PoseHead 基础学习率。
    --lr "${STAGE1_LR}"
    # --locate_feature_source=vision_only：只实例化 MoonViT + mlp1，不加载 3B LLM。
    --locate_feature_source vision_only
    # --box_source=gt：PoseHead 使用数据集 GT 人体框。
    --box_source gt
    # --locate_train_scope=selective_vision_lora：训练全范围视觉 LoRA；projector 默认全量训练。
    --locate_train_scope selective_vision_lora
    # --w_locate_box_lm=0：Stage1 不计算 grounding LM loss。
    --w_locate_box_lm 0
    # --w_locate_pbd=0：Stage1 不计算 PBD loss。
    --w_locate_pbd 0
    # --pose_box_grad_scale=0：Stage1 不存在 Pose→Locate 框梯度。
    --pose_box_grad_scale 0
  )
  # --resume_from_checkpoint：可选的 Stage1 完整断点续训来源。
  [[ -n "${STAGE1_RESUME_FROM_CHECKPOINT}" ]] && args+=(--resume_from_checkpoint "${STAGE1_RESUME_FROM_CHECKPOINT}")
  echo "[Stage1] 纯视觉塔 + GT 框；训练 PoseHead、全范围 MoonViT LoRA 与视觉 projector。"
  run_train_pose "${args[@]}"
}

# ==============================================================================
# 15. Stage2：冻结视觉 LoRA 和 PoseHead，训练 LLM LoRA 与视觉 projector
# ==============================================================================

run_stage2() {
  common_args "${STAGE2_DATASET_MIX_WEIGHTS}"
  mkdir -p "${STAGE2_OUTPUT_DIR}"
  local resume_path="${STAGE2_RESUME_FROM_CHECKPOINT}"
  if [[ -z "${resume_path}" ]]; then
    local init_source="${STAGE2_INIT_CHECKPOINT}"
    if [[ -z "${init_source}" && "${STAGE2_INIT_FROM_STAGE1}" == "1" ]]; then
      init_source="${STAGE1_OUTPUT_DIR}"
    fi
    if [[ -z "${init_source}" ]]; then
      echo "Stage2 需要 Stage1 权重；请复用 OUTPUT_DIR 或设置 STAGE2_INIT_CHECKPOINT。" >&2
      exit 1
    fi
    resume_path="$(prepare_weights_only_checkpoint "${init_source}" "${STAGE2_INIT_WEIGHTS_DIR}")"
  fi

  local args=("${COMMON_ARGS[@]}"
    # --datasets：Stage2 grounding 恢复数据集。
    --datasets "${STAGE2_TRAIN_DATASETS}"
    # --output_dir：Stage2 输出目录。
    --output_dir "${STAGE2_OUTPUT_DIR}"
    # --epochs：Stage2 epoch 数。
    --epochs "${STAGE2_EPOCHS}"
    # --max_steps：Stage2 optimizer step 上限。
    --max_steps "${STAGE2_MAX_STEPS}"
    # --batch_size：Stage2 单卡 teacher-forcing micro-batch。
    --batch_size "${STAGE2_BATCH_SIZE}"
    # --grad_accum_steps：Stage2 梯度累积步数。
    --grad_accum_steps "${STAGE2_GRAD_ACCUM_STEPS}"
    # --lr：Stage2 基础学习率；LLM LoRA 再乘倍率。
    --lr "${STAGE2_LR}"
    # --locate_feature_source=raw_visual：加载完整 LocateAnything 和文本路径。
    --locate_feature_source raw_visual
    # --no-prune_locate_generation：保留 lm_head 和坐标生成组件。
    --no-prune_locate_generation
    # --box_source=gt：grounding-only 不运行 PoseHead，该值只保持输入契约明确。
    --box_source gt
    # --locate_train_scope=selective_llm_lora：冻结视觉 LoRA，训练指定 LLM LoRA；projector 默认全量训练。
    --locate_train_scope selective_llm_lora
    # --freeze_pose：冻结 Stage1 训练好的全部 PoseHead 参数。
    --freeze_pose
    # --locate_grounding_only：跳过 PoseHead、BoxDN 和关键点 DN 前向。
    --locate_grounding_only
    # --locate_gradient_checkpointing：用重算降低完整 LLM 训练显存。
    --locate_gradient_checkpointing
    # --w_locate_box_lm：GT 坐标 token CE 权重。
    --w_locate_box_lm "${W_LOCATE_BOX_LM}"
    # --w_locate_pbd：六位置框 PBD 几何监督权重。
    --w_locate_pbd "${W_LOCATE_PBD}"
    # --pose_box_grad_scale=0：Stage2 不运行 PoseHead，也不回传姿态梯度。
    --pose_box_grad_scale 0
    # --pbd_temperature：soft coordinate expectation 温度。
    --pbd_temperature "${PBD_TEMPERATURE}"
    # --locate_lm_loss_every：LM CE 监督间隔。
    --locate_lm_loss_every "${LOCATE_LM_LOSS_EVERY}"
    # --locate_lm_max_instances：普通多人样本最大 grounding 框数。
    --locate_lm_max_instances "${LOCATE_LM_MAX_INSTANCES}"
    # --visualize_every=0：Stage2 没有 PoseHead 输出，关闭训练可视化。
    --visualize_every 0
    # --resume_from_checkpoint：Stage2 断点或 Stage1 仅权重初始化包。
    --resume_from_checkpoint "${resume_path}"
  )
  echo "[Stage2] 冻结视觉塔、视觉 LoRA 和 PoseHead；只训练指定 LLM LoRA 恢复 grounding。"
  run_train_pose "${args[@]}"
}

# ==============================================================================
# 16. Stage3：冻结 LocateAnything，使用真实生成框校准 PoseHead
# ==============================================================================

run_stage3() {
  common_args "${STAGE3_DATASET_MIX_WEIGHTS}"
  mkdir -p "${STAGE3_OUTPUT_DIR}"
  local resume_path="${STAGE3_RESUME_FROM_CHECKPOINT}"
  if [[ -z "${resume_path}" ]]; then
    local init_source="${STAGE3_INIT_CHECKPOINT}"
    if [[ -z "${init_source}" && "${STAGE3_INIT_FROM_STAGE2}" == "1" ]]; then
      if latest_checkpoint "${STAGE2_OUTPUT_DIR}" >/dev/null 2>&1; then
        init_source="${STAGE2_OUTPUT_DIR}"
      fi
    fi
    if [[ -z "${init_source}" && "${STAGE3_ALLOW_STAGE1_FALLBACK}" == "1" ]]; then
      if latest_checkpoint "${STAGE1_OUTPUT_DIR}" >/dev/null 2>&1; then
        init_source="${STAGE1_OUTPUT_DIR}"
      fi
    fi
    if [[ -z "${init_source}" ]]; then
      echo "Stage3 找不到 Stage2/Stage1 权重；请复用 OUTPUT_DIR 或设置 STAGE3_INIT_CHECKPOINT。" >&2
      exit 1
    fi
    resume_path="$(prepare_weights_only_checkpoint "${init_source}" "${STAGE3_INIT_WEIGHTS_DIR}")"
  fi

  local args=("${COMMON_ARGS[@]}"
    # --datasets：Stage3 真实生成框校准数据集。
    --datasets "${STAGE3_TRAIN_DATASETS}"
    # --output_dir：Stage3 输出目录。
    --output_dir "${STAGE3_OUTPUT_DIR}"
    # --epochs：Stage3 epoch 数。
    --epochs "${STAGE3_EPOCHS}"
    # --max_steps：Stage3 optimizer step 上限。
    --max_steps "${STAGE3_MAX_STEPS}"
    # --batch_size：Stage3 单卡 micro-batch。
    --batch_size "${STAGE3_BATCH_SIZE}"
    # --grad_accum_steps：Stage3 梯度累积步数。
    --grad_accum_steps "${STAGE3_GRAD_ACCUM_STEPS}"
    # --lr：Stage3 PoseHead 学习率。
    --lr "${STAGE3_LR}"
    # --locate_feature_source=raw_visual：完整加载 LocateAnything，供生成框和共享特征。
    --locate_feature_source raw_visual
    # --no-prune_locate_generation：保留自回归生成和 KV cache。
    --no-prune_locate_generation
    # --box_source=locate_generate：PoseHead 使用真实 LocateAnything 生成框。
    --box_source locate_generate
    # --locate_train_scope=frozen：关闭全部 LocateAnything adapter 梯度。
    --locate_train_scope frozen
    # --freeze_locate：显式冻结视觉塔、视觉 LoRA、LLM LoRA 和 lm_head。
    --freeze_locate
    # --locate_box_max_new_tokens：生成响应最大新 token 数。
    --locate_box_max_new_tokens "${LOCATE_BOX_MAX_NEW_TOKENS}"
    # --locate_generation_mode：LocateAnything 原生生成策略。
    --locate_generation_mode "${LOCATE_GENERATION_MODE}"
    # --w_locate_box_lm=0：Stage3 不再训练 LLM。
    --w_locate_box_lm 0
    # --w_locate_pbd=0：Stage3 不再训练 PBD。
    --w_locate_pbd 0
    # --pose_box_grad_scale=0：生成框作为离散输入，不向 LocateAnything 回传梯度。
    --pose_box_grad_scale 0
    # --resume_from_checkpoint：Stage3 断点或前一阶段仅权重初始化包。
    --resume_from_checkpoint "${resume_path}"
  )

  if [[ "${STAGE3_GENERATE_REFHUMAN_ONLY}" == "1" ]]; then
    # --locate_generate_refhuman_only：仅 RefHuman 用生成框，普通姿态数据仍用 GT。
    args+=(--locate_generate_refhuman_only)
  else
    # --no-locate_generate_refhuman_only：全部姿态数据都用真实生成框校准。
    args+=(--no-locate_generate_refhuman_only)
  fi

  echo "[Stage3] 冻结完整 LocateAnything；使用真实生成框训练 PoseHead。"
  run_train_pose "${args[@]}"
}

# ==============================================================================
# 17. 按用户选择顺序执行阶段
# ==============================================================================

{
  print_configuration_summary
  if (( RUN_STAGE1 == 1 )); then
    run_stage1
  fi
  if (( RUN_STAGE2 == 1 )); then
    run_stage2
  fi
  if (( RUN_STAGE3 == 1 )); then
    run_stage3
  fi
} 2>&1 | tee -a "${LOG_FILE}"
