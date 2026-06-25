# QwenPose 中文说明

版本：`v0.2.0`

[English](README.md) | 中文

QwenPose 是一个基于 Qwen3-VL 的 box-conditioned 人体姿态估计公开训练快照。这个仓库发布的是当前分支里的 Qwen3-VL 两阶段训练主线：

- `stage1_freeze_qwen`：冻结 Qwen 主体，先把 pose 模块 warm up 起来
- `stage2_qwen_lora_lm`：打开 Qwen LoRA，并可选启用 LM 辅助监督做短程联合微调

这次公开发布只覆盖 Qwen3-VL 工作流，不把 Eagle 相关 shell 入口作为公开主线的一部分。

## 仓库内容

- `scripts/train_qwenpose_two_stage.sh`：两阶段训练主入口
- `scripts/train_qwenpose_one_stage.sh`：旧脚本名的兼容包装器
- `scripts/eval_qwenpose.sh`：验证入口
- `scripts/zero2.json`、`scripts/zero3.json`、`scripts/zero3_offload.json`：DeepSpeed 预设
- `src/qwenpose/`：数据加载、模型、训练、验证、checkpoint、LoRA 合并等核心代码

## 仓库结构

```text
qwenpose/
├── CHANGELOG.md
├── README.md
├── README_zh.md
├── VERSION
├── requirements.txt
├── requirements-cu126.txt
├── scripts/
│   ├── eval_qwenpose.sh
│   ├── train_qwenpose_one_stage.sh
│   ├── train_qwenpose_two_stage.sh
│   ├── zero2.json
│   ├── zero3.json
│   └── zero3_offload.json
└── src/
    └── qwenpose/
```

运行时建议在仓库根目录旁准备以下目录：

```text
qwenpose/
├── datasets/
├── outputs/
└── weights/
    └── Qwen3-VL-4B-Instruct/
```

这些路径既可以是真实目录，也可以是符号链接。

## 复现实验环境

当前版本在以下软件栈上验证通过：

- Python `3.11.15`
- CUDA `12.6`
- PyTorch `2.8.0`
- TorchVision `0.23.0`
- TorchAudio `2.8.0`
- Transformers `4.57.6`
- FlashAttention `2.8.3`
- DeepSpeed `0.17.1`
- Accelerate `1.7.0`
- PEFT `0.17.1`
- NumPy `2.2.6`
- Pillow `12.2.0`
- pycocotools `2.0.11`
- SciPy `1.17.1`
- sentencepiece `0.2.1`
- tokenizers `0.22.2`
- tqdm `4.67.3`

依赖文件说明：

- `requirements.txt`：公开 Qwen3-VL 工作流的运行时依赖版本
- `requirements-cu126.txt`：Linux + Python 3.11 + CUDA 12.6 的精确验证版本

## 环境安装

### 方案 A：直接复现当前验证过的 CUDA 12.6 环境

```bash
python -m venv envs/qwenpose
source envs/qwenpose/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-cu126.txt
```

### 方案 B：按自己的 CUDA 环境安装

如果你的机器不是 CUDA 12.6，可以先安装匹配自己环境的 PyTorch，再安装仓库依赖。下面给的是已验证的 CUDA 12.6 示例；如果使用别的 CUDA 版本，请把对应的 index URL 替换掉：

```bash
python -m venv envs/qwenpose
source envs/qwenpose/bin/activate
python -m pip install --upgrade pip
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

如果当前平台无法使用 `flash-attn`，可以先安装其余依赖，并在运行时使用：

```bash
QWEN_ATTN_IMPLEMENTATION=sdpa
```

脚本会优先使用本地 `envs/qwenpose/bin/python` 和 `envs/qwenpose/bin/torchrun`。如果这两个路径不存在，就回退到当前环境里的 `python` 和 `torchrun`。

## 基座模型下载

默认基座模型路径为：

```text
weights/Qwen3-VL-4B-Instruct
```

官方模型地址：

- Hugging Face：<https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct>
- ModelScope：<https://modelscope.cn/models/Qwen/Qwen3-VL-4B-Instruct>

使用 Hugging Face CLI 的下载示例：

```bash
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct \
  --local-dir weights/Qwen3-VL-4B-Instruct
```

训练代码通过标准 `transformers` 包加载 Qwen3-VL。模型目录中应包含常规 Hugging Face 文件，例如 `config.json`、processor 或 tokenizer 文件，以及模型权重分片。

## 数据集准备

默认数据根目录为：

```text
datasets/
```

当前公开版本支持 `coco`、`aic`、`mpii`、`crowdpose`、`refhuman`。

### COCO

所需结构：

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

代码当前兼容两种常见本地目录布局。

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

- 公开代码会自动识别上述两种布局
- 当前 AIC 本地树只包含训练集，因此在请求 `val` 时会自动回退到训练标注

### MPII

所需结构：

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
- 如果需要合并训练集，也可以手动使用 `mpii_trainval.json`

### CrowdPose

所需结构：

```text
datasets/crowdpose/
└── annotations/
    ├── images/
    ├── mmpose_crowdpose_train.json
    └── mmpose_crowdpose_val.json
```

说明：

- 图片目录固定读取 `annotations/images/`
- 训练使用 `mmpose_crowdpose_train.json`
- 验证使用 `mmpose_crowdpose_val.json`

### RefHuman

所需结构：

```text
datasets/refhuman/
├── RefHuman_train.json
├── RefHuman_val.json
└── images/
```

说明：

- RefHuman 提供的是带文本描述的 `REF_POSE` 样本
- 两阶段默认配置中，RefHuman 会在 stage 2 使用
- `REFHUMAN_MAX_CAPTIONS_PER_INSTANCE` 用于控制每个实例最多保留多少条 caption

## 默认训练配方

当前公开主入口是：

```bash
scripts/train_qwenpose_two_stage.sh
```

默认阶段设置：

- Stage 1 输出目录：`stage1_freeze_qwen`
- Stage 2 输出目录：`stage2_qwen_lora_lm`
- 总输出根目录：`outputs/qwenpose_two_stage_qwen`
- 当 `MERGE_FINAL_WEIGHTS=1` 时自动导出的发布权重：`weights/<run_name>-merged-<timestamp>`
- Stage 1 数据集：`coco`
- Stage 2 数据集：`coco`
- Stage 1 batch size：每张 GPU `4`
- Stage 2 batch size：每张 GPU `1`
- Stage 1 epoch：`2`
- Stage 2 epoch：`1`
- 默认 ZeRO 方案：`zero2`

如果希望在 stage 1 中加入 AIC，可以显式传入：

```bash
STAGE1_TRAIN_DATASETS=coco,aic,mpii,crowdpose
```

## 快速开始

### 1. 先做数据 dry run

只检查数据解析和一个 batch 的构造，不真正进入训练：

```bash
PYTHON=python \
ZERO_STAGE=none \
DEVICE=cpu \
DRY_RUN_DATA=1 \
MAX_SAMPLES_PER_DATASET=2 \
scripts/train_qwenpose_two_stage.sh
```

### 2. 启动两阶段训练

一个最小多卡示例：

```bash
PYTHON=python \
TORCHRUN=torchrun \
ZERO_STAGE=zero2 \
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
QWEN_MODEL_PATH=weights/Qwen3-VL-4B-Instruct \
DATASET_ROOT=datasets \
scripts/train_qwenpose_two_stage.sh
```

### 3. 断点续训

对已有 run 目录继续训练：

```bash
scripts/train_qwenpose_two_stage.sh \
  --resume outputs/qwenpose_two_stage_qwen/<run_name>
```

脚本会优先自动解析 stage 2 的真实 checkpoint 续训；如果 stage 2 还未开始，也会尽量用 stage 1 结果初始化 stage 2。

### 4. 执行验证

```bash
PYTHON=python \
TRAIN_OUTPUT_DIR=outputs/qwenpose_two_stage_qwen/<run_name> \
scripts/eval_qwenpose.sh
```

当传入完整两阶段 run 目录时，验证脚本会优先选择：

```text
<run>/stage2_qwen_lora_lm
```

如果没有 stage 2 checkpoint，则回退到 stage 1。

## 输出目录

一次典型训练的目录结构如下：

```text
outputs/qwenpose_two_stage_qwen/<run_name>/
├── logs/
├── stage1_freeze_qwen/
├── stage2_init_weights/
├── stage2_qwen_lora_lm/
└── eval_pose_<timestamp>/
```

验证输出包括：

- `summary.json`
- `predictions.jsonl`
- `report.md`

如果 `MERGE_FINAL_WEIGHTS=1`，训练脚本还会在 `weights/` 下自动导出合并后的可部署权重。

## 版本管理

这个仓库现在使用显式版本号管理：

- 当前版本：`VERSION`
- 版本历史：`CHANGELOG.md`
- Python 包版本：`qwenpose.__version__`
- 推荐 git tag 格式：`vX.Y.Z`

后续每次发布新版本时，建议同步更新 `VERSION`、把新的版本记录追加到 `CHANGELOG.md` 顶部，并推送对应的 git tag，这样 GitHub 上最新版本会更清晰。

## 兼容说明

`scripts/train_qwenpose_one_stage.sh` 现在只保留为兼容旧习惯的包装器，公开仓库维护的正式训练入口是 `scripts/train_qwenpose_two_stage.sh`。
