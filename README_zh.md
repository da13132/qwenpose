# QwenPose 中文说明

版本：`v2.1.1`

[English](README.md) | 中文

这是一个面向公开复现的 box-conditioned 人体姿态估计训练快照。当前仓库维护两条基于同一套 PoseHead、数据管线和验证代码的主线：

- `LocatePose`：基于 `LocateAnything-3B` 的两阶段闭环训练方案
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

这个 `v2.1.1` 快照在以下环境中完成验证：

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

- `LocatePose` 的 stage 1 使用 `coco,mpii,crowdpose`，并只走 MoonViT 视觉特征
- `LocatePose` 的 stage 2 使用 `coco,mpii,crowdpose,refhuman`，恢复完整多模态 Locate 路径
- `QwenPose` 的 stage 1 和 stage 2 默认都使用 `coco`
- RefHuman 需要文本条件，因此明确延后到 stage 2

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

对于 `locate_generate` 和 `qwen_generate` 这两种闭环 stage 2 训练路径，当前公开脚本推荐使用 `ZERO_STAGE=zero2` 或 `ZERO_STAGE=none`。

## LocatePose

LocatePose 以 `LocateAnything-3B` 作为 grounding backbone，在共享 PoseHead 上执行两阶段训练。

### LocatePose 默认两阶段设置

| 阶段 | 目录名 | Backbone 状态 | 条件框来源 | 默认数据集 | 默认 epoch |
|------|--------|----------------|------------|------------|------------|
| stage 1 | `stage1_freeze_locate_gt_box` | 仅用 MoonViT，训练 vision LoRA | `gt` | `coco,mpii,crowdpose` | `20`，并限制最多 `60,000` step |
| stage 2 | `stage2_locate_box_closed_loop` | 恢复完整多模态 Locate，训练全部 LoRA | `locate_generate` | `coco,mpii,crowdpose,refhuman` | `5` |

vision-only 的 Stage 1 现在只实例化 MoonViT、冻结的 `mlp1` 视觉 projector 和 vision LoRA；不会实例化 Qwen2.5 语言模型、tokenizer，也不再构造数据集 prompt。Dataset worker 每张图只打开一次，执行一次同步增强，把同一份原分辨率 uint8 图像交给 MoonViT，并由它生成本地 RGB tensor。Stage 2 再加载完整 LocateAnything，并通过完全一致的 `base_model.model.vision_model.*` 参数命名空间注入 Stage 1 的 vision LoRA。

其他关键默认值：

- `CUDA_VISIBLE_DEVICES=0,1,2,3`
- `NPROC_PER_NODE=4`
- `STAGE1_BATCH_SIZE=6`（4 卡全局 batch 为 `24`）
- `STAGE2_BATCH_SIZE=1`
- `STAGE1_GRAD_ACCUM_STEPS=1`
- `STAGE2_GRAD_ACCUM_STEPS=4`
- `STAGE1_LR=3e-4`
- `STAGE1_MAX_STEPS=60000`
- `STAGE2_LR=5e-5`
- `STAGE1_LOCATE_FEATURE_SOURCE=vision_only`
- `STAGE1_LOCATE_TRAIN_SCOPE=vision_lora`
- `STAGE1_LOCATE_GRADIENT_CHECKPOINTING=0`
- `STAGE2_LOCATE_FEATURE_SOURCE=multimodal`
- `STAGE2_LOCATE_TRAIN_SCOPE=all_lora`
- `POSE_DROPOUT=0.0`
- `LOCATE_LORA_DROPOUT=0.0`、`LOCATE_VISION_LORA_DROPOUT=0.0`
- `LOCATE_VISION_SCALE=0.05`
- `STAGE1_BOX_JITTER_SCALE=0.0`、`STAGE1_BOX_JITTER_SHIFT=0.0` 仅作为全局 fallback；每条数据记录携带各数据集自己的默认扰动策略
- `DATASET_MIX_WEIGHTS=auto` 表示按数据量比例 interleave；手动权重仅用于受控消融
- `W_OKS=0.5`
- `W_COORD=3.0`
- `W_IMAGE_COORD=5.0`
- `POSE_ROI_SIZE=32`
- `SIMCC_BINS=256`
- `W_COARSE_COORD=0.5`
- `W_DEFORM_COORD=0.75`
- `W_REFINE_COORDS=0.75,1.0,1.25`
- SimCC 只在最后一次 refinement 后计算一次
- `W_SIMCC_COARSE=0.0`
- `W_SIMCC_DEFORM=0.0`
- `W_SIMCC_REFINE=0.0,0.0,0.5`
- `SIMCC_SIGMA=2.0`
- `LOCATE_IMAGE_TOKEN_LIMIT=4096`
- `STAGE1_LOCATE_BATCH_TOKEN_LIMIT=STAGE1_BATCH_SIZE*3072`（batch 6 时默认 `18432`）
- `STAGE2_LOCATE_BATCH_TOKEN_LIMIT=STAGE2_BATCH_SIZE*4096`（默认 `4096`）
- 默认启用跨 rank 视觉 token 成本均衡
- Stage 1 默认启用同步姿态增强，Stage 2 默认关闭增强
- 默认增强：水平翻转概率 `0.5`，仿射概率 `0.8`，旋转 `±15°`、缩放 `0.85–1.15`、平移 `±8%`，以及中等强度颜色、模糊和随机遮挡
- `LOCATE_GENERATION_MODE=hybrid`
- `LOCATE_BOX_MAX_NEW_TOKENS=8192`
- `STAGE2_W_LOCATE_BOX_LM=0.04`
- `STAGE2_W_LOCATE_POINT_LM=0.01`

### 训练 LocatePose

直接启动新训练：

```bash
bash scripts/locatepose.sh
```

按当前默认 4 卡布局启动的示例：

```bash
RUN_NAME=locatepose_v2_1_1 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
ZERO_STAGE=zero2 \
bash scripts/locatepose.sh
```

只做数据链路快速检查：

```bash
DRY_RUN_DATA=1 ZERO_STAGE=none NPROC_PER_NODE=1 bash scripts/locatepose.sh
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
- `STAGE1_TRAIN_DATASETS`、`STAGE2_TRAIN_DATASETS`：逗号分隔的数据集列表；vision-only 的 stage 1 不能包含 RefHuman
- `STAGE1_LOCATE_FEATURE_SOURCE`、`STAGE2_LOCATE_FEATURE_SOURCE`：快速姿态 warmup 使用 `vision_only`，闭环阶段使用 `multimodal`
- `STAGE1_LOCATE_TRAIN_SCOPE`、`STAGE2_LOCATE_TRAIN_SCOPE`：可选 `frozen`、`vision_lora` 或 `all_lora`
- `POSE_DROPOUT`：PoseHead 内部 Transformer dropout，默认 `0.0`
- `STAGE1_BOX_JITTER_SCALE`、`STAGE1_BOX_JITTER_SHIFT`：stage 1 GT box 扰动强度
- `LOCATE_ATTN_IMPLEMENTATION`：训练时 LocateAnything 的 attention 后端，默认 `flash_attention_2`
- `SIMCC_BINS`：SimCC 辅助头每个坐标轴的 bin 数，默认 `256`；设为 `0` 可完全关闭 SimCC
- `SCHEMA_JOINT_PRIORS_PATH`：各 schema 的 box-relative 关节点几何先验 JSON 文件
- `W_IMAGE_COORD`：整图归一化坐标 loss 权重
- `W_COARSE_COORD`、`W_DEFORM_COORD`、`W_REFINE_COORDS`：coarse、deformable 和 refinement 各阶段的坐标深监督权重
- `W_SIMCC_COARSE`、`W_SIMCC_DEFORM`、`W_SIMCC_REFINE`、`SIMCC_SIGMA`：SimCC 辅助监督权重与高斯 soft-label 宽度
- `LOCATE_IMAGE_TOKEN_LIMIT`：每张图的 raw MoonViT token 上限
- `STAGE1_LOCATE_BATCH_TOKEN_LIMIT`、`STAGE2_LOCATE_BATCH_TOKEN_LIMIT`：单卡 micro batch token 总预算，默认随各阶段 batch size 自动缩放
- `DISABLE_VISION_TOKEN_BALANCING`：仅在需要关闭跨 rank 成本均衡时设为 `1`
- `STAGE1_POSE_AUGMENT`、`STAGE2_POSE_AUGMENT`：同步姿态增强开关，默认分别为 `1` 和 `0`
- `AUGMENT_FLIP_PROB`、`AUGMENT_AFFINE_PROB`、`AUGMENT_ROTATE_DEGREES`、`AUGMENT_SCALE_MIN/MAX`、`AUGMENT_TRANSLATE_FRACTION`：几何增强参数
- `AUGMENT_COLOR_PROB`、`AUGMENT_BRIGHTNESS`、`AUGMENT_CONTRAST`、`AUGMENT_SATURATION`、`AUGMENT_HUE`、`AUGMENT_GRAYSCALE_PROB`、`AUGMENT_BLUR_PROB`、`AUGMENT_ERASE_PROB`：颜色、模糊和遮挡增强参数
- `LOCATE_GENERATION_MODE`：LocateAnything 生成模式，可选 `fast`、`slow`、`hybrid`
- `LOCATE_VISION_SCALE`：Locate vision LoRA 参数的学习率倍率
- `BOX_MATCH_IOU_THRESH`、`BOX_NMS_IOU_THRESH`：生成框匹配与 NMS 阈值
- `MERGE_FINAL_WEIGHTS`：当前公开 LocatePose 脚本不会导出完整 merged LocateAnything 权重

### 验证 LocatePose

验证最近一次 LocatePose 训练：

```bash
bash scripts/eval_locatepose.sh
```

`scripts/eval_locatepose.sh` 默认使用 `LOCATE_GENERATION_BACKEND=vllm`，即通过集成式 custom vLLM 路径完成 LocateAnything 生成框和 PoseHead 特征复用。

验证指定 checkpoint 或 stage 目录：

```bash
CHECKPOINT=outputs/locatepose/<run_name>/stage2_locate_box_closed_loop \
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

如果不安装 `vllm`，可以这样切回纯 Transformers 路径：

```bash
LOCATE_GENERATION_BACKEND=transformers bash scripts/eval_locatepose.sh
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
  --checkpoint outputs/locatepose/<run_name>/stage2_locate_box_closed_loop \
  --input demo/images \
  --format coco
```

如果不使用 `vllm`：

```bash
LOCATE_GENERATION_BACKEND=transformers \
bash scripts/infer_locatepose.sh \
  --checkpoint outputs/locatepose/<run_name>/stage2_locate_box_closed_loop \
  --input demo/images \
  --format crowdpose
```

RefHuman 文本条件推理示例：

```bash
bash scripts/infer_locatepose.sh \
  --checkpoint outputs/locatepose/<run_name>/stage2_locate_box_closed_loop \
  --input datasets/refhuman \
  --format refhuman \
  --split val
```

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
├── stage1_freeze_locate_gt_box/
└── stage2_locate_box_closed_loop/
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
- Git tag，例如 `v2.1.1`

每次发布新的公开快照时，建议将代码、README、变更记录和 tag 一起更新，这样 Git 历史与文档说明才能保持一致。
