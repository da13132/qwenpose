# QwenPose 中文说明

版本：`v2.3`

[English](README.md) | 中文

这是一个面向公开复现的 box-conditioned 人体姿态估计训练快照。当前仓库维护两条基于同一套 PoseHead、数据管线和验证代码的主线：

- `LocatePose`：基于 `LocateAnything-3B` 的两阶段 person-query 训练方案
- `QwenPose`：基于 `Qwen3-VL-4B-Instruct` 的两阶段闭环训练方案

下面的文档顺序也与此保持一致：先讲共享环境和数据，再讲 `LocatePose`，最后讲 `QwenPose`。

## 仓库公开内容

- `scripts/locatepose.sh`：LocatePose 两阶段训练主入口
- `scripts/eval_locatepose.sh`：LocatePose 验证入口
- `scripts/infer_locatepose.sh`：LocatePose 单图、文件夹与 RefHuman 推理入口
- `scripts/train_qwenpose_two_stage.sh`：QwenPose 两阶段训练主入口
- `scripts/eval_qwenpose.sh`：QwenPose 验证入口
- `scripts/zero2.json`、`scripts/zero3.json`、`scripts/zero3_offload.json`：两条主线共用的 DeepSpeed 预设
- `requirements-vllm.txt`：LocatePose 集成式 vLLM 推理/验证的可选依赖版本
- `src/qwenpose/`：数据集加载、模型、训练、验证、推理、打分、checkpoint、backbone 适配等核心实现

## 仓库结构

```text
qwenpose/
├── CHANGELOG.md
├── README.md
├── README_zh.md
├── VERSION
├── requirements.txt
├── requirements-cu126.txt
├── requirements-vllm.txt
├── scripts/
│   ├── eval_locatepose.sh
│   ├── eval_qwenpose.sh
│   ├── infer_locatepose.sh
│   ├── locatepose.sh
│   ├── wait_for_4gpu_locatepose.sh
│   ├── train_qwenpose_two_stage.sh
│   ├── zero2.json
│   ├── zero3.json
│   └── zero3_offload.json
└── src/
    └── qwenpose/
        ├── data.py
        ├── eagle_lora.py
        ├── eval_pose.py
        ├── infer_locatepose.py
        ├── losses.py
        ├── merge_full_weights.py
        ├── metrics.py
        ├── model.py
        ├── qwen_lora.py
        ├── score_pose_predictions.py
        ├── schemas.py
        ├── train_pose.py
        ├── vllm_locateanything.py
        └── vllm_locateanything_model.py
```

## 运行时目录约定

脚本默认假设仓库根目录旁边还有这些目录。它们既可以是真实目录，也可以是符号链接。

```text
qwenpose/
├── datasets/
├── outputs/
└── weights/
    ├── LocateAnything-3B/
    └── Qwen3-VL-4B-Instruct/
```

## 已验证环境

这个 `v2.3` 快照在以下环境中完成验证：

- Python `3.11.15`
- CUDA `12.6`
- PyTorch `2.8.0`
- TorchVision `0.23.0`
- TorchAudio `2.8.0`
- Transformers `4.57.6`
- vLLM `0.11.0`，用于 LocatePose 集成式推理/验证路径
- FlashAttention `2.8.3`
- DeepSpeed `0.17.1`
- Accelerate `1.7.0`
- PEFT `0.17.1`
- Hugging Face Hub `0.36.2`
- NumPy `2.2.6`
- Pillow `12.2.0`
- pycocotools `2.0.11`
- safetensors `0.7.0`
- SciPy `1.17.1`
- sentencepiece `0.2.1`
- tokenizers `0.22.2`
- tqdm `4.67.3`

依赖文件说明：

- `requirements.txt`：适合自定义 CUDA 环境时使用的运行时依赖版本
- `requirements-cu126.txt`：Linux + Python 3.11 + CUDA 12.6 的精确验证版本
- `requirements-vllm.txt`：LocatePose 集成式 `vllm` 推理/验证的可选依赖版本

## 环境安装

### 方案 A：直接复现当前验证过的 CUDA 12.6 环境

```bash
python -m venv envs/qwenpose
source envs/qwenpose/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-cu126.txt
```

### 方案 B：按自己的 CUDA 环境安装

先安装和自己机器匹配的 PyTorch，再安装仓库依赖：

```bash
python -m venv envs/qwenpose
source envs/qwenpose/bin/activate
python -m pip install --upgrade pip
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

### 方案 C：为 LocatePose 集成式 vLLM 验证/推理补装可选依赖

如果你希望直接使用 `LocatePose` 默认的集成式 `vllm` 路径，请在方案 A 或方案 B 之后执行：

```bash
pip install -r requirements-vllm.txt
```

如果当前平台无法使用 `flash-attn`，可以先安装其余依赖，再在运行时切换 attention 后端：

```bash
LOCATE_ATTN_IMPLEMENTATION=sdpa
QWEN_ATTN_IMPLEMENTATION=sdpa
```

所有 shell 入口会优先使用本地 `envs/qwenpose/bin/python` 和 `envs/qwenpose/bin/torchrun`。

如果没有安装 `vllm`，请把 LocatePose 验证和推理切回纯 Transformers 路径：

```bash
LOCATE_GENERATION_BACKEND=transformers
```

## 基座模型下载

### LocatePose 基座模型

默认路径：

```text
weights/LocateAnything-3B
```

官方来源：

- Hugging Face 模型页：<https://huggingface.co/nvidia/LocateAnything-3B>
- NVIDIA Eagle 项目仓库：<https://github.com/NVlabs/Eagle>

下载示例：

```bash
huggingface-cli download nvidia/LocateAnything-3B \
  --local-dir weights/LocateAnything-3B
```

### QwenPose 基座模型

默认路径：

```text
weights/Qwen3-VL-4B-Instruct
```

官方来源：

- Hugging Face 模型页：<https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct>
- ModelScope：<https://modelscope.cn/models/Qwen/Qwen3-VL-4B-Instruct>

下载示例：

```bash
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct \
  --local-dir weights/Qwen3-VL-4B-Instruct
```

下载完成后，模型目录中应包含常规 Hugging Face 文件，例如 `config.json`、processor 或 tokenizer 文件，以及模型权重分片。

## 数据集准备

共享数据根目录为：

```text
datasets/
```

当前公开代码支持 `coco`、`aic`、`mpii`、`crowdpose`、`refhuman`。

当前默认配方说明：

- `LocatePose` 的 Stage1/2 都使用 `coco,mpii,crowdpose,refhuman` 与 raw multimodal 特征
- RefHuman 从 Stage1 起就通过 person-query 匹配头与共享 PoseHead 参与训练
- `QwenPose` 的 stage 1 和 stage 2 默认都使用 `coco`
- RefHuman 需要 LLM 文本路径；LocatePose Stage1 会加载该路径但冻结 Locate 参数

### COCO

```text
datasets/coco/
├── annotations/
│   ├── person_keypoints_train2017.json
│   └── person_keypoints_val2017.json
├── train2017/
└── val2017/
```

说明：

- 训练使用 `person_keypoints_train2017.json` 和 `train2017/`
- 验证使用 `person_keypoints_val2017.json` 和 `val2017/`

### AIC

代码兼容以下两种本地目录布局。

布局 A：

```text
datasets/aic/
├── ai_challenger_keypoint_train_annotations_20170909/
│   └── keypoint_train_annotations_20170909.json
└── ai_challenger_keypoint_train_20170902/
    └── keypoint_train_images_20170902/
```

布局 B：

```text
datasets/aic/
└── ai_challenger_keypoint_train_20170902/
    ├── keypoint_train_annotations_20170902.json
    └── keypoint_train_images_20170902/
```

说明：

- 公开代码会自动识别以上两种布局
- 如果本地请求验证 split，当前 loader 会回退到已有的训练标注

### MPII

```text
datasets/mpii/
├── annotations/
│   ├── mpii_train.json
│   ├── mpii_val.json
│   └── mpii_trainval.json
└── images/
```

说明：

- 训练默认使用 `mpii_train.json`
- 验证默认使用 `mpii_val.json`
- `mpii_trainval.json` 可用于自定义训练配置

### CrowdPose

```text
datasets/crowdpose/
└── annotations/
    ├── images/
    ├── mmpose_crowdpose_train.json
    └── mmpose_crowdpose_val.json
```

说明：

- 当前 loader 期望图像位于 `datasets/crowdpose/annotations/images/`
- 标注文件名为 `mmpose_crowdpose_train.json` 和 `mmpose_crowdpose_val.json`

### RefHuman

```text
datasets/refhuman/
├── RefHuman_train.json
├── RefHuman_val.json
└── images/
```

说明：

- RefHuman 会以 referring-person pose 任务方式加载
- JSON 中需要包含图像信息、bbox、keypoints 以及对应的人物文本描述
- 默认每个人每个 epoch 只贡献一条字幕；该人物的全部唯一字幕会按 seed 打乱并跨 epoch 轮换

## DeepSpeed 预设

仓库内包含三份共用 DeepSpeed 配置：

- `scripts/zero2.json`：大多数训练任务的默认推荐
- `scripts/zero3.json`：显存更紧张时可使用纯 ZeRO-3
- `scripts/zero3_offload.json`：显存非常紧张时可使用 CPU offload，但速度会更慢

训练脚本通过 `ZERO_STAGE` 选择这些预设：

```bash
ZERO_STAGE=zero2 bash scripts/locatepose.sh
ZERO_STAGE=zero3 bash scripts/train_qwenpose_two_stage.sh
ZERO_STAGE=zero3_offload bash scripts/locatepose.sh
ZERO_STAGE=none bash scripts/train_qwenpose_two_stage.sh
```

当前 LocatePose 两阶段都使用 person queries。旧 QwenPose 生成框训练路径支持 `ZERO_STAGE=zero2` 或 `ZERO_STAGE=none`。

## LocatePose

LocatePose 以 `LocateAnything-3B` 作为 grounding backbone，Stage1/2 使用完全相同的 800×800 Pose 架构和参数形状。

### LocatePose 默认两阶段设置

| 阶段 | 目录名 | Backbone 状态 | 条件框来源 | 默认数据集 | 默认 epoch |
|------|--------|----------------|------------|------------|------------|
| stage 1 | `stage1_freeze_locate_person_queries` | 冻结 Locate backbone | `person_queries` | `coco,mpii,crowdpose,refhuman` | `50` |
| stage 2 | `stage2_unfreeze_locate_person_queries` | 选择性训练视觉/LLM LoRA | `person_queries` | `coco,mpii,crowdpose,refhuman` | `3` |

两个阶段统一使用单一 RGB Pose 金字塔：800×800 输入产生 200×200、100×100、50×50 的 stride-4/8/16 特征。旧 256 与 640 两条 RGB 分支以及全部 SimCC 头均已删除。可学习 person queries 经过两层人体框 decoder 输出 objectness 与人体框，同一批框再条件化共享关键点 decoder；box DN 与关键点 DN 都保留。

两个阶段都会加载多模态 Locate backbone，并向 PoseHead 提供相同的 raw MoonViT 特征（`raw_visual`）。Stage1 冻结 Locate，Stage2 只训练配置的视觉与 LLM LoRA 层。训练不再包含坐标 token 生成、`lm_head`、KV cache、生成框匹配或 Locate teacher-forcing loss。RefHuman 用独立的文本—人体匹配头在 caption-independent 的 person-query 检测候选中监督目标人物；所有人体仍共用同一个 PoseHead，RefHuman 只额外向姿态 query 注入文本条件。旧生成框架构及双 RGB/SimCC checkpoint 与该训练路径不兼容。

其他关键默认值：

- LocatePose 脚本强制使用物理卡 `1,2,3`，并固定启动三个训练进程
- Stage1：`BATCH_SIZE=1`、`GRAD_ACCUM_STEPS=1`、`LR=2e-4`
- Stage2：`BATCH_SIZE=1`、`GRAD_ACCUM_STEPS=4`、`LR=1e-4`
- 图像尺寸固定为 `800`；公开训练方案中的 Locate 特征图固定为 100×100
- `POSE_PYRAMID_CHANNELS=128`，`POSE_PYRAMID_BLOCKS=3`
- `POSE_ROI_SIZE=16`，`HUMAN_DECODER_LAYERS=2`，`POSE_DECODER_LAYERS=3`
- `DEFORMABLE_POINTS=4`，`DEFORMABLE_MIN_RADIUS_CELLS=2.0`
- `ENABLE_BOX_DENOISING=1`，`MAX_DN_QUERIES=96`，`MAX_DN_GROUPS=4`
- `DN_POSITIVE_NOISE=0.40`，`DN_NEGATIVE_NOISE=1.00`
- `W_BOX_OBJECTNESS=1.0`，`W_BOX_L1=5.0`，`W_BOX_GIOU=2.0`，`W_BOX_RELATIVE=1.0`，`W_BOX_DN=1.0`
- RefHuman：`REF_TEXT_SCALE=0.2`、`W_REF_MATCH=1.0`
- 两个阶段都使用 `raw_visual` 与 `person_queries`
- Stage2 默认使用 `selective_lora`，视觉层 15–26，LLM 层 32–35
- 两阶段默认数据倍率都是 `coco:1,mpii:1,crowdpose:1,refhuman:1`；支持 `0` 与小数倍率
- 关键点坐标只使用直接回归；不存在 SimCC 分支或融合解码
- raw-visual RefHuman 训练路径关闭坐标生成和同步图像增强

### 训练 LocatePose

直接启动新训练：

```bash
bash scripts/locatepose.sh
```

指定 run 名启动；脚本始终使用物理卡 1、2、3：

```bash
RUN_NAME=locatepose_v2_2 \
ZERO_STAGE=zero2 \
bash scripts/locatepose.sh
```

只做数据链路快速检查：

```bash
DRY_RUN_DATA=1 ZERO_STAGE=none bash scripts/locatepose.sh
```

从已有 run、stage 目录、checkpoint 目录或 checkpoint 文件继续：

```bash
bash scripts/locatepose.sh --resume outputs/locatepose/<run_name>
```

### LocatePose 常用变量

- `LOCATE_MODEL_PATH`：LocateAnything-3B 本地权重路径
- `DATASET_ROOT`：数据根目录，默认 `datasets`
- `OUTPUT_ROOT`：训练输出根目录，默认 `outputs/locatepose`
- `ZERO_STAGE`：`zero2`、`zero3`、`zero3_offload` 或 `none`
- `STAGE1_TRAIN_DATASETS`、`STAGE2_TRAIN_DATASETS`：逗号分隔的数据集列表；默认都包含 RefHuman 在内的四个数据集
- `STAGE1_DATASET_MIX_WEIGHTS`、`STAGE2_DATASET_MIX_WEIGHTS`：每 epoch 的数据集遍历倍率；允许非负整数、小数和 `0`
- Stage1 固定使用 `raw_visual` 并冻结 Locate；`STAGE2_LOCATE_TRAIN_SCOPE` 默认使用 `selective_lora`
- `POSE_PYRAMID_CHANNELS`、`POSE_PYRAMID_BLOCKS`：统一 P2/P3/P4 RGB 编码器配置
- `HUMAN_DECODER_LAYERS`：人体框迭代 refinement 层数
- `ENABLE_BOX_DENOISING`、`MAX_DN_QUERIES`、`MAX_DN_GROUPS`、`DN_POSITIVE_NOISE`、`DN_NEGATIVE_NOISE`：仅框 denoising 配置
- `W_BOX_OBJECTNESS`、`W_BOX_L1`、`W_BOX_GIOU`、`W_BOX_RELATIVE`、`W_BOX_DN`：人体框 loss 权重
- `REF_TEXT_SCALE`：共享 PoseHead 的 RefHuman 姿态 query 文本条件缩放系数
- `W_REF_MATCH`：独立的 RefHuman 表达式到候选人体分类 loss
- `REF_POSE_QUALITY_ALPHA`：评估/推理中独立指代分数之后使用的姿态质量指数
- `SCHEMA_JOINT_PRIORS_PATH`：各 schema 的 box-relative 关节点几何先验 JSON 文件
- `W_IMAGE_COORD`、`W_COARSE_COORD`、`W_DEFORM_COORD`、`W_REFINE_COORDS`：坐标回归与深监督权重
- `LOCATE_IMAGE_TOKEN_LIMIT`：每张图的 raw MoonViT token 上限
- `STAGE1_LOCATE_BATCH_TOKEN_LIMIT`、`STAGE2_LOCATE_BATCH_TOKEN_LIMIT`：各阶段单卡 micro batch token 总预算

### 验证 LocatePose

验证最近一次 LocatePose 训练：

```bash
bash scripts/eval_locatepose.sh
```

`scripts/eval_locatepose.sh` 默认单次前向验证 person-query 输出。旧生成框 checkpoint 仍可显式设置 `BOX_SOURCE=locate_generate`。

验证指定 checkpoint 或 stage 目录：

```bash
CHECKPOINT=outputs/locatepose/<run_name>/stage2_unfreeze_locate_person_queries \
bash scripts/eval_locatepose.sh
```

在多数据集上验证：

```bash
DATASETS=coco,mpii,crowdpose,refhuman bash scripts/eval_locatepose.sh
```

查看 GT box 条件下的上限结果：

```bash
BOX_SOURCE=gt bash scripts/eval_locatepose.sh
```

旧生成框 checkpoint 需要显式选择该路径；下面的 Transformers backend 不依赖可选的 `vllm`：

```bash
BOX_SOURCE=locate_generate LOCATE_GENERATION_BACKEND=transformers \
bash scripts/eval_locatepose.sh
```

验证结果默认输出到：

```text
outputs/locatepose/<run_name>/eval_locatepose_<timestamp>/
```

目录中包含 `summary.json`、`predictions.jsonl`、`predictions.json`、`report.md`，以及可选的可视化结果。

### 推理 LocatePose

从训练好的 LocatePose checkpoint 对单图或文件夹做推理：

```bash
bash scripts/infer_locatepose.sh \
  --checkpoint outputs/locatepose/<run_name>/stage2_unfreeze_locate_person_queries \
  --input demo/images \
  --format coco
```

旧生成框 checkpoint 仍可显式使用 Transformers 推理：

```bash
BOX_SOURCE=locate_generate LOCATE_GENERATION_BACKEND=transformers \
bash scripts/infer_locatepose.sh \
  --checkpoint outputs/locatepose/<run_name>/stage2_unfreeze_locate_person_queries \
  --input demo/images \
  --format crowdpose
```

最终关键点坐标只来自直接回归头。

RefHuman 文本条件推理示例：

```bash
REF_POSE_QUALITY_ALPHA=0.25 \
bash scripts/infer_locatepose.sh \
  --checkpoint outputs/locatepose/<run_name>/stage2_unfreeze_locate_person_queries \
  --input datasets/refhuman \
  --format refhuman \
  --split val
```

RefHuman 推理在同一次前向中检测全部 person-query 候选，由 `ref_match_head` 计算文本匹配分数，再按 `ref_score × pose_quality^0.25` 选择指定人物。选中的候选与普通全人体推理共用同一个 PoseHead。

推理目录默认会写出 `summary.json`、`predictions.jsonl`、`predictions.json`、可选格式化导出、`manifest.json` 和可视化结果。

### 给导出预测重新打分

对已经导出的 LocatePose 或 QwenPose 预测文件重新计算指标：

```bash
PYTHONPATH=src python -m qwenpose.score_pose_predictions \
  --predictions outputs/locatepose/<run_name>/eval_locatepose_<timestamp>/predictions.jsonl \
  --dataset_root datasets \
  --split val
```

## QwenPose

QwenPose 以 `Qwen3-VL-4B-Instruct` 作为 backbone，在同一个共享 PoseHead 上执行两阶段训练。

### QwenPose 默认两阶段设置

| 阶段 | 目录名 | Backbone 状态 | 条件框来源 | 默认数据集 | 默认 epoch |
|------|--------|----------------|------------|------------|------------|
| stage 1 | `stage1_freeze_qwen` | 冻结 Qwen | `gt` | `coco` | `30` |
| stage 2 | `stage2_qwen_box_closed_loop` | 解冻 Qwen LoRA 和 vision LoRA | `qwen_generate` | `coco` | `12` |

其他关键默认值：

- `STAGE1_BATCH_SIZE=16`
- `STAGE2_BATCH_SIZE=1`
- `STAGE1_GRAD_ACCUM_STEPS=2`
- `STAGE2_GRAD_ACCUM_STEPS=8`
- `QWEN_FEATURE_SIZE=64`
- `QWEN_FEATURE_REFINER_LAYERS=1`
- `QWEN_BOX_MAX_NEW_TOKENS=4096`
- `BOX_MATCH_IOU_THRESH=0.10`
- `BOX_NMS_IOU_THRESH=0.70`

### 训练 QwenPose

直接启动新训练：

```bash
bash scripts/train_qwenpose_two_stage.sh
```

8 卡训练示例：

```bash
RUN_NAME=qwenpose_v1 \
NPROC_PER_NODE=8 \
ZERO_STAGE=zero2 \
bash scripts/train_qwenpose_two_stage.sh
```

只做数据链路快速检查：

```bash
DRY_RUN_DATA=1 ZERO_STAGE=none NPROC_PER_NODE=1 bash scripts/train_qwenpose_two_stage.sh
```

从已有 run、stage 目录、checkpoint 目录或 checkpoint 文件继续：

```bash
bash scripts/train_qwenpose_two_stage.sh --resume outputs/qwenpose_two_stage_qwen/<run_name>
```

### QwenPose 常用变量

- `QWEN_MODEL_PATH`：Qwen3-VL-4B-Instruct 本地权重路径
- `DATASET_ROOT`：数据根目录，默认 `datasets`
- `OUTPUT_ROOT`：训练输出根目录，默认 `outputs/qwenpose_two_stage_qwen`
- `ZERO_STAGE`：`zero2`、`zero3`、`zero3_offload` 或 `none`
- `STAGE1_TRAIN_DATASETS`、`STAGE2_TRAIN_DATASETS`：逗号分隔的数据集列表
- `QWEN_MIN_PIXELS`、`QWEN_MAX_PIXELS`：Qwen processor 的可选像素预算限制
- `QWEN_BOX_MAX_NEW_TOKENS`：Qwen 生成 bbox JSON 的最大新 token 数
- `BOX_MATCH_IOU_THRESH`、`BOX_NMS_IOU_THRESH`：生成框匹配与 NMS 阈值
- `MERGE_FINAL_WEIGHTS`：启用后可在训练结束时导出 merged Qwen 权重

### 验证 QwenPose

验证最近一次 QwenPose 训练：

```bash
bash scripts/eval_qwenpose.sh
```

验证指定 stage 目录：

```bash
CHECKPOINT=outputs/qwenpose_two_stage_qwen/<run_name>/stage2_qwen_box_closed_loop \
bash scripts/eval_qwenpose.sh
```

查看 GT box 条件下的上限结果：

```bash
BOX_SOURCE=gt bash scripts/eval_qwenpose.sh
```

`scripts/eval_qwenpose.sh` 默认会验证 `coco,mpii,crowdpose,refhuman`，如有需要可以通过 `EVAL_DATASETS` 覆盖。

## 输出目录结构

典型 LocatePose 训练目录：

```text
outputs/locatepose/<run_name>/
├── logs/
├── stage1_freeze_locate_person_queries/
└── stage2_unfreeze_locate_person_queries/
```

典型 QwenPose 训练目录：

```text
outputs/qwenpose_two_stage_qwen/<run_name>/
├── logs/
├── stage1_freeze_qwen/
└── stage2_qwen_box_closed_loop/
```

每个 stage 目录下可能包含 `checkpoint-*`、`checkpoint_step_*.pt`、`qwenpose_checkpoint.pt`、可视化结果和阶段日志，具体取决于当前配置。

## 版本管理

这个仓库通过以下位置记录公开快照版本：

- `VERSION`：仓库版本号
- `CHANGELOG.md`：按时间倒序记录版本变更
- `qwenpose.__version__`：Python 包版本
- Git tag，例如 `v2.3`

每次发布新的公开快照时，建议将代码、README、变更记录和 tag 一起更新，这样 Git 历史与文档说明才能保持一致。
