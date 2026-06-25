# QwenPose

Release: `v0.2.2`

English | [中文说明](README_zh.md)

QwenPose is a public training snapshot for box-conditioned human pose estimation built on top of Qwen3-VL. This repository publishes the Qwen3-VL two-stage workflow used in this branch:

- `stage1_freeze_qwen`: freeze the Qwen backbone and warm up the pose modules.
- `stage2_qwen_lora_lm`: enable Qwen LoRA and optionally use LM supervision for short joint finetuning.

This public release is intentionally scoped to the Qwen3-VL path. The Eagle shell entrypoints are excluded from the published workflow.

## Included Scope

- `scripts/train_qwenpose_two_stage.sh`: main two-stage training entrypoint
- `scripts/eval_qwenpose.sh`: evaluation entrypoint
- `scripts/zero2.json`, `scripts/zero3.json`, `scripts/zero3_offload.json`: DeepSpeed presets
- `src/qwenpose/`: data loading, model definition, training, evaluation, checkpointing, and LoRA merge utilities

## Repository Layout

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
│   ├── train_qwenpose_two_stage.sh
│   ├── zero2.json
│   ├── zero3.json
│   └── zero3_offload.json
└── src/
    └── qwenpose/
```

The training scripts expect the following runtime directories beside the repository root:

```text
qwenpose/
├── datasets/
├── outputs/
└── weights/
    └── Qwen3-VL-4B-Instruct/
```

These paths can be regular directories or symlinks.

## Reproducibility Snapshot

This release was verified with the following software stack:

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

Pinned dependency files:

- `requirements.txt`: runtime dependency pins for the public Qwen3-VL workflow
- `requirements-cu126.txt`: exact tested installation target for Linux + Python 3.11 + CUDA 12.6

## Installation

### Option A: exact tested CUDA 12.6 stack

```bash
python -m venv envs/qwenpose
source envs/qwenpose/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-cu126.txt
```

### Option B: custom CUDA stack

If your machine uses a different CUDA build, install the matching PyTorch wheels first, then install the repository requirements. The command below is the verified CUDA 12.6 example; replace the index URL when targeting a different build:

```bash
python -m venv envs/qwenpose
source envs/qwenpose/bin/activate
python -m pip install --upgrade pip
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

If `flash-attn` is unavailable on your platform, install the rest of the stack and run with:

```bash
QWEN_ATTN_IMPLEMENTATION=sdpa
```

The shell entrypoints prefer `envs/qwenpose/bin/python` and `envs/qwenpose/bin/torchrun` when those paths exist locally. Otherwise they fall back to the active `python` or `torchrun` on `PATH`.

## Download The Base Model

The default base model path is:

```text
weights/Qwen3-VL-4B-Instruct
```

Official model sources:

- Hugging Face: <https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct>
- ModelScope: <https://modelscope.cn/models/Qwen/Qwen3-VL-4B-Instruct>

Example with the Hugging Face CLI:

```bash
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct \
  --local-dir weights/Qwen3-VL-4B-Instruct
```

The training code loads Qwen3-VL through the standard `transformers` package. After download, the local model directory should contain the usual Hugging Face files such as `config.json`, tokenizer or processor files, and the model weight shards.

## Dataset Preparation

Set the dataset root to:

```text
datasets/
```

The repository currently supports `coco`, `aic`, `mpii`, `crowdpose`, and `refhuman`.

### COCO

Required structure:

```text
datasets/coco/
├── annotations/
│   ├── person_keypoints_train2017.json
│   └── person_keypoints_val2017.json
├── train2017/
└── val2017/
```

Notes:

- Training uses `person_keypoints_train2017.json` and `train2017/`
- Evaluation uses `person_keypoints_val2017.json` and `val2017/`

### AIC

The loader supports two common local layouts.

Layout A:

```text
datasets/aic/
├── ai_challenger_keypoint_train_annotations_20170909/
│   └── keypoint_train_annotations_20170909.json
└── ai_challenger_keypoint_train_20170902/
    └── keypoint_train_images_20170902/
```

Layout B:

```text
datasets/aic/
└── ai_challenger_keypoint_train_20170902/
    ├── keypoint_train_annotations_20170902.json
    └── keypoint_train_images_20170902/
```

Notes:

- The public code accepts either layout automatically.
- The current local tree only provides the training split. When `split=val` is requested, AIC falls back to the train annotations.

### MPII

Required structure:

```text
datasets/mpii/
├── annotations/
│   ├── mpii_train.json
│   ├── mpii_val.json
│   └── mpii_trainval.json
└── images/
```

Notes:

- `mpii_train.json` is used for training
- `mpii_val.json` is used for evaluation
- `mpii_trainval.json` can be used manually if you want a combined training split

### CrowdPose

Required structure:

```text
datasets/crowdpose/
└── annotations/
    ├── images/
    ├── mmpose_crowdpose_train.json
    └── mmpose_crowdpose_val.json
```

Notes:

- The image directory is expected at `annotations/images/`
- Training uses `mmpose_crowdpose_train.json`
- Evaluation uses `mmpose_crowdpose_val.json`

### RefHuman

Required structure:

```text
datasets/refhuman/
├── RefHuman_train.json
├── RefHuman_val.json
└── images/
```

Notes:

- RefHuman contributes `REF_POSE` samples with text descriptions
- The public default recipe does not enable RefHuman automatically; add it explicitly when needed
- `REFHUMAN_MAX_CAPTIONS_PER_INSTANCE` controls how many captions are kept for each person instance

Example:

```bash
STAGE2_TRAIN_DATASETS=coco,refhuman \
STAGE2_REFHUMAN_MAX_CAPTIONS_PER_INSTANCE=1 \
scripts/train_qwenpose_two_stage.sh
```

## Default Training Recipe

The main published entrypoint is:

```bash
scripts/train_qwenpose_two_stage.sh
```

Default stage configuration:

- Stage 1 output: `stage1_freeze_qwen`
- Stage 2 output: `stage2_qwen_lora_lm`
- Output root: `outputs/qwenpose_two_stage_qwen`
- Optional merged release weights when `MERGE_FINAL_WEIGHTS=1`: `weights/<run_name>-merged-<timestamp>`
- Stage 1 datasets: `coco`
- Stage 2 datasets: `coco`
- Stage 1 batch size: `4` per GPU
- Stage 2 batch size: `1` per GPU
- Stage 1 epochs: `2`
- Stage 2 epochs: `1`
- ZeRO preset: `zero2`

DeepSpeed preset selection:

- `ZERO_STAGE=zero2`: uses `scripts/zero2.json`, recommended default for standard multi-GPU training
- `ZERO_STAGE=zero3`: uses `scripts/zero3.json`, lowers GPU memory usage further at the cost of more runtime overhead
- `ZERO_STAGE=zero3_offload`: uses `scripts/zero3_offload.json`, saves the most GPU memory but is usually the slowest option
- `ZERO_STAGE=none`: disables DeepSpeed and is mainly intended for CPU or single-process debugging

If you want AIC in stage 1, pass it explicitly, for example:

```bash
STAGE1_TRAIN_DATASETS=coco,aic,mpii,crowdpose
```

## Quick Start

### 1. Dry-run the data pipeline

This verifies dataset parsing and one batch build without starting real training:

```bash
PYTHON=python \
ZERO_STAGE=none \
DEVICE=cpu \
DRY_RUN_DATA=1 \
MAX_SAMPLES_PER_DATASET=2 \
scripts/train_qwenpose_two_stage.sh
```

### 2. Launch two-stage training

Minimal multi-GPU example:

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

### 3. Resume an existing run

Resume from a previous run directory:

```bash
scripts/train_qwenpose_two_stage.sh \
  --resume outputs/qwenpose_two_stage_qwen/<run_name>
```

The script automatically resolves stage-2 checkpoint resume when possible. If stage 2 has not started yet, it can initialize stage 2 from the stage-1 weights.

### 4. Evaluate a run

```bash
PYTHON=python \
TRAIN_OUTPUT_DIR=outputs/qwenpose_two_stage_qwen/<run_name> \
scripts/eval_qwenpose.sh
```

When given a full two-stage run directory, the evaluation script prefers:

```text
<run>/stage2_qwen_lora_lm
```

If no stage-2 checkpoint is available, it falls back to stage 1.

## Output Structure

A typical run directory looks like:

```text
outputs/qwenpose_two_stage_qwen/<run_name>/
├── logs/
├── stage1_freeze_qwen/
├── stage2_init_weights/
├── stage2_qwen_lora_lm/
└── eval_pose_<timestamp>/
```

Evaluation outputs include:

- `summary.json`
- `predictions.jsonl`
- `report.md`

If `MERGE_FINAL_WEIGHTS=1`, the training script also exports merged deployable weights under `weights/`.

## Versioning

This repository now uses explicit release versioning:

- Current version: `VERSION`
- Release history: `CHANGELOG.md`
- Python package version: `qwenpose.__version__`
- Recommended git tag format: `vX.Y.Z`

New releases should update `VERSION`, prepend a new entry to `CHANGELOG.md`, and push a matching git tag so the latest version is easy to identify on GitHub.

## Official Entrypoint

The maintained public training entrypoint is `scripts/train_qwenpose_two_stage.sh`.
