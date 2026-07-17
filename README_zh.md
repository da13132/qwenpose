# QwenPose 中文说明

版本：`v3.0`

[English](README.md) | 中文

这是一个面向公开复现的 box-conditioned 人体姿态估计训练快照。当前仓库维护两条基于同一套 PoseHead、数据管线和验证代码的主线：

- `LocatePose`：基于 `LocateAnything-3B` 的三阶段 noisy-box、grounding 恢复与真实生成框校准方案
- `QwenPose`：基于 `Qwen3-VL-4B-Instruct` 的两阶段闭环训练方案

下面的文档顺序也与此保持一致：先讲共享环境和数据，再讲 `LocatePose`，最后讲 `QwenPose`。

## 仓库公开内容

- `scripts/locatepose.sh`：LocatePose 三阶段训练主入口
- `scripts/eval_locatepose.sh`：LocatePose 验证入口
- `scripts/infer_locatepose.sh`：LocatePose 单图、文件夹与 RefHuman 推理入口
- `scripts/initialize_locatepose_checkpoint.py`：v3.0 架构 checkpoint 初始化/迁移工具
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
│   ├── initialize_locatepose_checkpoint.py
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

这个 `v3.0` 快照在以下环境中完成验证：

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

- LocatePose 三个阶段默认都使用 `coco,mpii,crowdpose,refhuman`
- Stage1 使用 vision-only Locate 特征和 noisy GT-derived boxes；Stage2 冻结 PoseHead 恢复 grounding；Stage3 使用真实 LocateAnything 生成框校准 PoseHead
- RefHuman 从 Stage1 起参与训练，并固定 referred person 的外部框来源身份
- `QwenPose` 的 stage 1 和 stage 2 默认都使用 `coco`

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

QwenPose 训练脚本通过 `ZERO_STAGE` 选择这些预设：

```bash
ZERO_STAGE=zero3 bash scripts/train_qwenpose_two_stage.sh
ZERO_STAGE=none bash scripts/train_qwenpose_two_stage.sh
```

LocatePose 默认使用四张物理卡 `4,5,6,7`，并根据可见 GPU 数自动设置进程数。可通过 `LOCATEPOSE_CUDA_VISIBLE_DEVICES` 覆盖；Stage3 也支持单卡生成。

## LocatePose

LocatePose 以 `LocateAnything-3B` 作为 grounding backbone，通过三个明确阶段训练同一套 800×800 GroupPose 风格姿态架构。

### LocatePose 默认三阶段设置

| 阶段 | 目录名 | 可训练组件 | 条件框来源 | 默认数据集 | 默认 epoch |
|------|--------|------------|------------|------------|------------|
| stage 1 | `stage1_vision_gt_pose` | PoseHead、全范围视觉 LoRA、projector | noisy GT-derived external boxes | `coco,mpii,crowdpose,refhuman` | `50` |
| stage 2 | `stage2_restore_locate_grounding` | 指定 Locate LLM LoRA | GT 坐标 token teacher forcing；跳过 PoseHead | `coco,mpii,crowdpose,refhuman` | `10` |
| stage 3 | `stage3_generated_box_pose_calibration` | 仅 PoseHead | LocateAnything 真实生成框 | `coco,mpii,crowdpose,refhuman` | `5` |

Stage1 使用 Locate 原生 P2/P3/P4 特征、外部框多尺度池化、pre-pose 修框、分组迭代关键点解码、post-pose 修框与 keypoint DN。所有保留的 GT-derived box 都会增加中心和 log-scale 噪声。普通多人图像会分别以 `0.50` 概率模拟漏检和重复误检，受影响框数量随人数从 1 个增长到最多 3 个。RefHuman 只保留 referred person，并记录明确来源身份，因此 Hungarian 匹配不能把它重新分配给其他人物。

存在 external boxes 时，只有 external queries 参与匹配、loss 和输出筛选。这样漏掉的人不会被内部 proposal 自动补回，重复框则作为未匹配误检负样本。

Stage2 冻结 PoseHead 和视觉侧，完全跳过姿态前向，只通过坐标 token 语言模型监督恢复 LocateAnything grounding。Stage3 冻结全部 LocateAnything 参数，让 PoseHead 适应推理时真实自回归框分布。

其他关键默认值：

- 默认物理 GPU 为 `4,5,6,7`，进程数根据可见 GPU 自动计算
- Stage1：单卡 batch `24`、累积 `1`、有效 global batch `96`、学习率 `2e-4`
- Stage2：单卡 batch `4`、累积 `1`、学习率 `1e-4`
- Stage3：单卡 batch `4`、累积 `1`、学习率 `5e-5`
- 输入 letterbox 到 `800×800`；Locate image token 上限默认 `4096`
- `60` 个 person queries、`2` 层多尺度 encoder、`3` 层 pose decoder、`4` 个 deformable points
- keypoint DN 默认 `40` 个 query、`20` 个 group
- Stage1 proxy 中心标准差默认 `0.03`，宽高 log-scale 标准差默认 `0.06`
- 漏检与重复误检概率默认均为 `0.50`
- 每次训练默认自动生成带时间戳的输出根目录

### 训练 LocatePose

顺序执行全部三个阶段：

```bash
bash scripts/locatepose.sh all
```

只启动 Stage1：

```bash
bash scripts/locatepose.sh stage1
```

复用同一个输出根目录启动后续阶段：

```bash
OUTPUT_DIR=outputs/locatepose/locatepose-3stage-<timestamp>-gtbox-prepose \
bash scripts/locatepose.sh stage2 stage3
```

覆盖可见物理 GPU：

```bash
LOCATEPOSE_CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/locatepose.sh stage1
```

不同阶段的优化器参数组不同时，脚本会自动把上一阶段 checkpoint 转换成仅权重初始化包，避免错误恢复 optimizer、GradScaler、RNG 和数据游标。

### LocatePose 常用变量

- `LOCATE_MODEL_PATH`：LocateAnything-3B 本地权重路径
- `DATASET_ROOT`：数据根目录，默认 `datasets`
- `OUTPUT_DIR`：所有所选阶段共用的时间戳根目录
- `LOCATEPOSE_CUDA_VISIBLE_DEVICES`：逗号分隔的物理 GPU 列表
- `STAGE1_TRAIN_DATASETS`、`STAGE2_TRAIN_DATASETS`、`STAGE3_TRAIN_DATASETS`：各阶段数据集列表
- `STAGE1_DATASET_MIX_WEIGHTS`、`STAGE2_DATASET_MIX_WEIGHTS`、`STAGE3_DATASET_MIX_WEIGHTS`：各阶段遍历倍率
- `STAGE1_LOCATE_PROXY_CENTER_NOISE`、`STAGE1_LOCATE_PROXY_SCALE_NOISE`：Stage1 框形变
- `STAGE1_LOCATE_PROXY_MISS_PROBABILITY`、`STAGE1_LOCATE_PROXY_DUPLICATE_PROBABILITY`：普通多人图像的自适应漏检/多检
- `STAGE1_BATCH_SIZE`、`STAGE2_BATCH_SIZE`、`STAGE3_BATCH_SIZE`：各阶段单卡 micro-batch
- `STAGE1_INIT_CHECKPOINT`、`STAGE2_INIT_CHECKPOINT`、`STAGE3_INIT_CHECKPOINT`：显式初始化来源
- `LOCATE_VISION_LAYERS`、`LOCATE_VISION_MODULES`、`LOCATE_LLM_LAYERS`、`LOCATE_LLM_MODULES`：LoRA 层与模块范围
- `MAX_KEYPOINT_DN_QUERIES`、`MAX_KEYPOINT_DN_GROUPS`：keypoint DN 容量
- `LOCATE_GENERATION_MODE`、`LOCATE_BOX_MAX_NEW_TOKENS`：Stage3 自回归生成配置
- `STAGE3_GENERATE_REFHUMAN_ONLY`：设为 `1` 时仅 RefHuman 使用真实生成框

### 验证 LocatePose

验证最近一次 LocatePose 训练：

```bash
bash scripts/eval_locatepose.sh
```

`scripts/eval_locatepose.sh` 默认以 `BOX_SOURCE=locate_generate` 验证 LocateAnything 生成框。

验证指定 checkpoint 或 stage 目录：

```bash
CHECKPOINT=outputs/locatepose/<run_name>/stage3_generated_box_pose_calibration \
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

下面的 Transformers backend 不依赖可选的 `vllm`：

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
  --checkpoint outputs/locatepose/<run_name>/stage3_generated_box_pose_calibration \
  --input demo/images \
  --format coco
```

生成框路径可显式使用 Transformers 推理：

```bash
BOX_SOURCE=locate_generate LOCATE_GENERATION_BACKEND=transformers \
bash scripts/infer_locatepose.sh \
  --checkpoint outputs/locatepose/<run_name>/stage3_generated_box_pose_calibration \
  --input demo/images \
  --format crowdpose
```

最终关键点坐标只来自直接回归头。

RefHuman 文本条件推理示例：

```bash
REF_POSE_QUALITY_ALPHA=0.25 \
bash scripts/infer_locatepose.sh \
  --checkpoint outputs/locatepose/<run_name>/stage3_generated_box_pose_calibration \
  --input datasets/refhuman \
  --format refhuman \
  --split val
```

RefHuman 推理先由 LocateAnything 定位字幕描述的人体，再把生成框交给共享 PoseHead。

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
outputs/locatepose/locatepose-3stage-<timestamp>-gtbox-prepose/
├── logs/
├── stage1_vision_gt_pose/
├── stage2_restore_locate_grounding/
├── stage2_init_weights/
├── stage3_generated_box_pose_calibration/
└── stage3_init_weights/
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
- Git tag，例如 `v3.0`

每次发布新的公开快照时，建议将代码、README、变更记录和 tag 一起更新，这样 Git 历史与文档说明才能保持一致。
