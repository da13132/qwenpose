# QwenPose

English | [中文说明](README_zh.md)

This repository is a public snapshot of the QwenPose Qwen3-VL training pipeline focused on the two-stage training workflow:

- `stage1_freeze_qwen`: freeze the Qwen backbone and warm up the pose modules.
- `stage2_qwen_lora_lm`: enable Qwen LoRA and LM auxiliary loss for short joint finetuning.

This snapshot is intentionally scoped to the Qwen3-VL path. The Eagle shell entrypoints are not part of the published workflow.

## What is included

- `scripts/train_qwenpose_two_stage.sh`: main training entrypoint
- `scripts/train_qwenpose_one_stage.sh`: compatibility wrapper that forwards to the renamed two-stage script
- `scripts/eval_qwenpose.sh`: evaluation entrypoint
- `scripts/zero2.json`, `scripts/zero3.json`, `scripts/zero3_offload.json`: DeepSpeed configs
- `src/qwenpose/`: training, evaluation, data loading, model, loss, and checkpoint utilities

## What is intentionally not tracked

- Local datasets, weights, outputs, caches, and virtual environments
- The local vendored `src/transformers/` runtime package
- Eagle shell entrypoints

For this public repo, users should install `transformers==4.57.6` into their own environment instead of relying on a copied local package tree.

## Expected layout

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

The default paths are:

- dataset root: `datasets`
- Qwen base model: `weights/Qwen3-VL-4B-Instruct`
- training outputs: `outputs/qwenpose_two_stage_qwen`

## Environment setup

1. Create and activate your own Python environment.
2. Install PyTorch, TorchVision, and flash-attn for your CUDA/runtime stack.
3. Install the Python dependencies from `requirements.txt`.

Example:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The scripts will prefer `envs/qwenpose/bin/python` if it exists locally. Otherwise they fall back to the active `python` / `python3` in your environment.

## Quick start

### 1. Dry-run the data pipeline

This checks dataset parsing and one batch without starting a real training run:

```bash
PYTHON=python \
ZERO_STAGE=none \
DEVICE=cpu \
DRY_RUN_DATA=1 \
MAX_SAMPLES_PER_DATASET=2 \
scripts/train_qwenpose_two_stage.sh
```

### 2. Start two-stage training

Minimal multi-GPU example:

```bash
PYTHON=python \
TORCHRUN=torchrun \
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
QWEN_MODEL_PATH=weights/Qwen3-VL-4B-Instruct \
DATASET_ROOT=datasets \
scripts/train_qwenpose_two_stage.sh
```

Important defaults from the shell entrypoint:

- output root: `outputs/qwenpose_two_stage_qwen`
- run directory: `qwenpose-two-stage-qwen3vl-lora-<timestamp>`
- stage 1 output: `<run>/stage1_freeze_qwen`
- stage 2 output: `<run>/stage2_qwen_lora_lm`
- merged final weights: `weights/<run>-merged-<timestamp>`

### 3. Resume training

Resume from an existing run directory:

```bash
scripts/train_qwenpose_two_stage.sh \
  --resume outputs/qwenpose_two_stage_qwen/<run_name>
```

The script will resolve the correct stage 2 resume checkpoint automatically when possible.

### 4. Evaluate the latest stage 2 checkpoint

```bash
PYTHON=python \
TRAIN_OUTPUT_DIR=outputs/qwenpose_two_stage_qwen/<run_name> \
scripts/eval_qwenpose.sh
```

By default, the evaluation script now prefers `<run>/stage2_qwen_lora_lm` when a two-stage run directory is provided.

## Notes

- `scripts/train_qwenpose_one_stage.sh` is kept only as a compatibility wrapper. The real entrypoint is `scripts/train_qwenpose_two_stage.sh`.
- This repository does not ship the local copied `transformers` tree. Install `transformers==4.57.6` into your environment instead.
- The public snapshot is documented around the Qwen3-VL workflow only.
