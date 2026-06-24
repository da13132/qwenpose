# QwenPose 中文说明

[English](README.md) | 中文

这个仓库发布的是一个面向 Qwen3-VL 的 QwenPose 两阶段训练快照，重点保留以下主线：

- `stage1_freeze_qwen`：冻结 Qwen 主体，先把 pose 模块和 feature refiner warmup 起来
- `stage2_qwen_lora_lm`：打开 Qwen LoRA，并按需要启用 LM 辅助损失做短程联合微调

这份公开仓库只围绕 Qwen3-VL 工作流整理，不把 Eagle 的 shell 入口作为公开使用路径。

## 仓库包含什么

- `scripts/train_qwenpose_two_stage.sh`：正式训练入口
- `scripts/train_qwenpose_one_stage.sh`：兼容旧名字的 wrapper，会转发到新的 two-stage 脚本
- `scripts/eval_qwenpose.sh`：验证入口
- `scripts/zero2.json`、`scripts/zero3.json`、`scripts/zero3_offload.json`：DeepSpeed 配置
- `src/qwenpose/`：训练、验证、数据、模型、loss、checkpoint 相关代码

## 仓库刻意不跟踪什么

- 本地数据集、模型权重、训练输出、缓存、虚拟环境
- 本地拷贝到项目里的 `src/transformers/`
- Eagle 的 shell 脚本入口

这意味着公开仓库不会再依赖你本地那份 `src/transformers/`。使用者需要在自己的环境里安装 `transformers==4.57.6`。

## 期望目录结构

```text
qwenpose/
├── datasets/
│   ├── coco/
│   ├── aic/
│   ├── mpii/
│   ├── crowdpose/
│   └── refhuman/
├── weights/
│   └── Qwen3-VL-4B-Instruct/
├── outputs/
├── scripts/
│   ├── train_qwenpose_two_stage.sh
│   ├── train_qwenpose_one_stage.sh
│   ├── eval_qwenpose.sh
│   ├── zero2.json
│   ├── zero3.json
│   └── zero3_offload.json
└── src/
    └── qwenpose/
```

默认路径约定如下：

- 数据根目录：`datasets`
- Qwen 基座模型：`weights/Qwen3-VL-4B-Instruct`
- 训练输出根目录：`outputs/qwenpose_two_stage_qwen`

## 环境准备

1. 自己创建并激活 Python 环境
2. 先按你的 CUDA / 驱动环境安装 `torch`、`torchvision`、`flash-attn`
3. 再安装仓库里的 Python 依赖

示例：

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

脚本会优先使用本地的 `envs/qwenpose/bin/python`。如果这个路径不存在，就自动回退到当前环境里的 `python3` / `python`。

## 快速开始

### 1. 先做数据 dry run

这个命令只检查数据解析和一个 batch 的构造，不会真的开始训练：

```bash
PYTHON=python \
ZERO_STAGE=none \
DEVICE=cpu \
DRY_RUN_DATA=1 \
MAX_SAMPLES_PER_DATASET=2 \
scripts/train_qwenpose_two_stage.sh
```

### 2. 启动两阶段训练

一个最小的多卡示例：

```bash
PYTHON=python \
TORCHRUN=torchrun \
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
QWEN_MODEL_PATH=weights/Qwen3-VL-4B-Instruct \
DATASET_ROOT=datasets \
scripts/train_qwenpose_two_stage.sh
```

这个 shell 入口的关键默认值：

- 输出根目录：`outputs/qwenpose_two_stage_qwen`
- run 目录名：`qwenpose-two-stage-qwen3vl-lora-时间戳`
- stage1 输出：`<run>/stage1_freeze_qwen`
- stage2 输出：`<run>/stage2_qwen_lora_lm`
- 最终合并权重：`weights/<run>-merged-时间戳`

### 3. 断点续训

直接对一个历史 run 目录续训：

```bash
scripts/train_qwenpose_two_stage.sh \
  --resume outputs/qwenpose_two_stage_qwen/<run_name>
```

脚本会优先把这个两阶段 run 解析到合适的 stage2 恢复点；如果 stage2 还没有真实断点，也会尽量从 stage1 生成 stage2 的 weight-only 初始化。

### 4. 验证最近的 stage2 checkpoint

```bash
PYTHON=python \
TRAIN_OUTPUT_DIR=outputs/qwenpose_two_stage_qwen/<run_name> \
scripts/eval_qwenpose.sh
```

现在的验证脚本在收到一个两阶段 run 目录时，会优先选择其中的 `stage2_qwen_lora_lm` 作为默认 checkpoint 来源。

## 备注

- `scripts/train_qwenpose_one_stage.sh` 现在只保留兼容入口，真正的主脚本是 `scripts/train_qwenpose_two_stage.sh`
- 公开仓库不再提交本地拷贝的 `transformers` 代码树，请直接在环境里安装 `transformers==4.57.6`
- 这份公开快照的文档只覆盖 Qwen3-VL 主线
