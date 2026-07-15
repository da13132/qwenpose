# QwenPose

Release: `v2.3`

English | [中文说明](README_zh.md)

This repository is a public training snapshot for box-conditioned human pose estimation. It contains two maintained workflows built on the same pose head, data pipeline, and evaluation code:

- `LocatePose`: a LocateAnything-3B-based two-stage person-query recipe
- `QwenPose`: a Qwen3-VL-4B-Instruct-based two-stage closed-loop recipe

The public documentation below is ordered in the same way: shared setup first, `LocatePose` first, `QwenPose` second.

## Included Scope

- `scripts/locatepose.sh`: main two-stage LocatePose training entrypoint
- `scripts/eval_locatepose.sh`: LocatePose evaluation entrypoint
- `scripts/infer_locatepose.sh`: LocatePose image, folder, and RefHuman inference entrypoint
- `scripts/train_qwenpose_two_stage.sh`: main two-stage QwenPose training entrypoint
- `scripts/eval_qwenpose.sh`: QwenPose evaluation entrypoint
- `scripts/zero2.json`, `scripts/zero3.json`, `scripts/zero3_offload.json`: DeepSpeed presets used by both workflows
- `requirements-vllm.txt`: optional add-on dependency pin for integrated LocatePose vLLM inference/evaluation
- `src/qwenpose/`: datasets, pose model, training loop, evaluation, inference, scoring, checkpointing, and backbone adapters

## Repository Layout

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

## Runtime Directory Convention

The scripts expect the following directories next to the repository root. They can be real directories or symlinks.

```text
qwenpose/
├── datasets/
├── outputs/
└── weights/
    ├── LocateAnything-3B/
    └── Qwen3-VL-4B-Instruct/
```

## Tested Environment

This `v2.3` snapshot was validated with:

- Python `3.11.15`
- CUDA `12.6`
- PyTorch `2.8.0`
- TorchVision `0.23.0`
- TorchAudio `2.8.0`
- Transformers `4.57.6`
- vLLM `0.11.0` for the integrated LocatePose inference/evaluation path
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

Pinned dependency files:

- `requirements.txt`: runtime dependency pins for custom CUDA setups
- `requirements-cu126.txt`: exact tested target for Linux + Python 3.11 + CUDA 12.6
- `requirements-vllm.txt`: optional add-on pin for integrated LocatePose `vllm` inference/evaluation

## Installation

### Option A: exact tested CUDA 12.6 stack

```bash
python -m venv envs/qwenpose
source envs/qwenpose/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-cu126.txt
```

### Option B: custom CUDA stack

Install the matching PyTorch wheels for your system first, then install the repository requirements:

```bash
python -m venv envs/qwenpose
source envs/qwenpose/bin/activate
python -m pip install --upgrade pip
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

### Option C: optional vLLM add-on for integrated LocatePose eval/inference

Install this after either option A or option B when you want the default integrated LocatePose `vllm` path:

```bash
pip install -r requirements-vllm.txt
```

If `flash-attn` is unavailable on your platform, install the rest of the stack and switch the attention backend at runtime:

```bash
LOCATE_ATTN_IMPLEMENTATION=sdpa
QWEN_ATTN_IMPLEMENTATION=sdpa
```

All shell entrypoints prefer `envs/qwenpose/bin/python` and `envs/qwenpose/bin/torchrun` when those paths exist locally.

If you do not install `vllm`, keep LocatePose evaluation and inference on the pure Transformers path:

```bash
LOCATE_GENERATION_BACKEND=transformers
```

## Base Model Download

### LocatePose base model

Default path:

```text
weights/LocateAnything-3B
```

Official sources:

- Hugging Face model card: <https://huggingface.co/nvidia/LocateAnything-3B>
- NVIDIA Eagle repository: <https://github.com/NVlabs/Eagle>

Example download:

```bash
huggingface-cli download nvidia/LocateAnything-3B \
  --local-dir weights/LocateAnything-3B
```

### QwenPose base model

Default path:

```text
weights/Qwen3-VL-4B-Instruct
```

Official sources:

- Hugging Face model card: <https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct>
- ModelScope: <https://modelscope.cn/models/Qwen/Qwen3-VL-4B-Instruct>

Example download:

```bash
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct \
  --local-dir weights/Qwen3-VL-4B-Instruct
```

After download, each model directory should contain the standard Hugging Face files such as `config.json`, tokenizer or processor files, and model weight shards.

## Dataset Preparation

Set the shared dataset root to:

```text
datasets/
```

The public code supports `coco`, `aic`, `mpii`, `crowdpose`, and `refhuman`.

Important defaults:

- both LocatePose stages use `coco,mpii,crowdpose,refhuman` with raw multimodal features
- RefHuman trains from Stage 1 through the person-query match and shared pose heads
- `QwenPose` stage 1 and stage 2 use `coco` by default
- `RefHuman` requires the LLM text path, which is loaded but frozen in LocatePose Stage 1

### COCO

```text
datasets/coco/
├── annotations/
│   ├── person_keypoints_train2017.json
│   └── person_keypoints_val2017.json
├── train2017/
└── val2017/
```

Notes:

- Training uses `person_keypoints_train2017.json` with `train2017/`
- Evaluation uses `person_keypoints_val2017.json` with `val2017/`

### AIC

The loader accepts either of the following layouts.

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

- The public loader auto-detects either layout
- If a validation split is requested locally, the loader falls back to the available training annotations

### MPII

```text
datasets/mpii/
├── annotations/
│   ├── mpii_train.json
│   ├── mpii_val.json
│   └── mpii_trainval.json
└── images/
```

Notes:

- Training uses `mpii_train.json`
- Evaluation uses `mpii_val.json`
- `mpii_trainval.json` is optional and can be used for custom experiments

### CrowdPose

```text
datasets/crowdpose/
└── annotations/
    ├── images/
    ├── mmpose_crowdpose_train.json
    └── mmpose_crowdpose_val.json
```

Notes:

- The current loader expects images under `datasets/crowdpose/annotations/images/`
- The split files are `mmpose_crowdpose_train.json` and `mmpose_crowdpose_val.json`

### RefHuman

```text
datasets/refhuman/
├── RefHuman_train.json
├── RefHuman_val.json
└── images/
```

Notes:

- RefHuman samples are loaded as referring-person pose tasks
- The JSON rows must contain the image metadata, boxes, keypoints, and the text description used as the referring expression
- By default, each person contributes one caption per epoch; its unique captions follow a seed-randomized rotation across epochs

## DeepSpeed Presets

The repository includes three shared DeepSpeed configs:

- `scripts/zero2.json`: default recommendation for most training runs
- `scripts/zero3.json`: use when GPU memory is tighter and pure ZeRO-3 is preferred
- `scripts/zero3_offload.json`: use when GPU memory is very limited and CPU offload is acceptable

Training scripts expose the preset through `ZERO_STAGE`:

```bash
ZERO_STAGE=zero2 bash scripts/locatepose.sh
ZERO_STAGE=zero3 bash scripts/train_qwenpose_two_stage.sh
ZERO_STAGE=zero3_offload bash scripts/locatepose.sh
ZERO_STAGE=none bash scripts/train_qwenpose_two_stage.sh
```

The current LocatePose recipe uses person queries in both stages. The legacy QwenPose generated-box recipe supports `ZERO_STAGE=zero2` or `ZERO_STAGE=none`.

## LocatePose

LocatePose uses `LocateAnything-3B` as the grounding backbone and trains one identical 800×800 pose architecture across two stages.

### Default LocatePose schedule

| Stage | Directory | Backbone state | Box source | Default datasets | Default epochs |
|-------|-----------|----------------|------------|------------------|----------------|
| stage 1 | `stage1_freeze_locate_person_queries` | frozen Locate backbone | `person_queries` | `coco,mpii,crowdpose,refhuman` | `50` |
| stage 2 | `stage2_unfreeze_locate_person_queries` | selective vision/LLM LoRA | `person_queries` | `coco,mpii,crowdpose,refhuman` | `3` |

Both stages use the same unified RGB pose pyramid and parameter shapes: an 800×800 input produces stride-4/8/16 features at 200×200, 100×100, and 50×50. The old 256 and 640 RGB branches and all SimCC heads are removed. Learned person queries pass through a two-layer human box decoder that predicts objectness and boxes; those same boxes condition the shared keypoint decoder. Box DN and keypoint DN remain enabled.

Both stages load the multimodal Locate backbone and feed the same raw MoonViT feature type (`raw_visual`) to PoseHead. Stage 1 freezes Locate; Stage 2 selectively trains configured vision and LLM LoRA layers. Coordinate generation, `lm_head`, KV-cache generation, generated-box matching, and Locate teacher-forcing loss are not part of training. RefHuman trains an independent expression-to-person matching head against caption-independent person-query detections; the same shared pose decoder is used for every person and receives optional RefHuman text conditioning. Old checkpoints from the generated-box or dual-RGB/SimCC architectures are not compatible with this training path.

Additional default knobs:

- the LocatePose script forcibly uses physical GPUs `1,2,3` and launches three processes
- Stage 1: `BATCH_SIZE=1`, `GRAD_ACCUM_STEPS=1`, `LR=2e-4`
- Stage 2: `BATCH_SIZE=1`, `GRAD_ACCUM_STEPS=4`, `LR=1e-4`
- image size is fixed at `800`; the Locate feature map is fixed at 100×100 in the public recipe
- `POSE_PYRAMID_CHANNELS=128`, `POSE_PYRAMID_BLOCKS=3`
- `POSE_ROI_SIZE=16`, `HUMAN_DECODER_LAYERS=2`, `POSE_DECODER_LAYERS=3`
- `DEFORMABLE_POINTS=4`, `DEFORMABLE_MIN_RADIUS_CELLS=2.0`
- `ENABLE_BOX_DENOISING=1`, `MAX_DN_QUERIES=96`, `MAX_DN_GROUPS=4`
- `DN_POSITIVE_NOISE=0.40`, `DN_NEGATIVE_NOISE=1.00`
- `W_BOX_OBJECTNESS=1.0`, `W_BOX_L1=5.0`, `W_BOX_GIOU=2.0`, `W_BOX_RELATIVE=1.0`, `W_BOX_DN=1.0`
- RefHuman: `REF_TEXT_SCALE=0.2`, `W_REF_MATCH=1.0`
- both stages use `raw_visual` and `person_queries`
- Stage 2 uses `selective_lora` with vision layers 15–26 and LLM layers 32–35
- both stages default to `coco:1,mpii:1,crowdpose:1,refhuman:1`; zero and fractional traversal multipliers are supported
- keypoint coordinates use direct regression only; there is no SimCC branch or fused decoder
- coordinate generation and synchronized image augmentation are disabled in this raw-visual RefHuman training path

### Train LocatePose

Start a new run:

```bash
bash scripts/locatepose.sh
```

Example with an explicit run name; the script always uses physical GPUs 1, 2, and 3:

```bash
RUN_NAME=locatepose_v2_2 \
ZERO_STAGE=zero2 \
bash scripts/locatepose.sh
```

Quick data-path smoke test:

```bash
DRY_RUN_DATA=1 ZERO_STAGE=none bash scripts/locatepose.sh
```

Resume from an existing run, stage directory, checkpoint directory, or checkpoint file:

```bash
bash scripts/locatepose.sh --resume outputs/locatepose/<run_name>
```

### Common LocatePose variables

- `LOCATE_MODEL_PATH`: local LocateAnything-3B weights
- `DATASET_ROOT`: dataset root, default `datasets`
- `OUTPUT_ROOT`: training root, default `outputs/locatepose`
- `ZERO_STAGE`: one of `zero2`, `zero3`, `zero3_offload`, or `none`
- `STAGE1_TRAIN_DATASETS`, `STAGE2_TRAIN_DATASETS`: comma-separated dataset lists; both default to all four datasets including RefHuman
- `STAGE1_DATASET_MIX_WEIGHTS`, `STAGE2_DATASET_MIX_WEIGHTS`: per-epoch traversal multipliers; non-negative integers, decimals, and zero are supported
- Stage 1 always uses `raw_visual` with a frozen Locate backbone; `STAGE2_LOCATE_TRAIN_SCOPE` defaults to `selective_lora`
- `POSE_PYRAMID_CHANNELS`, `POSE_PYRAMID_BLOCKS`: unified P2/P3/P4 RGB encoder settings
- `HUMAN_DECODER_LAYERS`: iterative person-box refinement layers
- `ENABLE_BOX_DENOISING`, `MAX_DN_QUERIES`, `MAX_DN_GROUPS`, `DN_POSITIVE_NOISE`, `DN_NEGATIVE_NOISE`: box-only denoising settings
- `W_BOX_OBJECTNESS`, `W_BOX_L1`, `W_BOX_GIOU`, `W_BOX_RELATIVE`, `W_BOX_DN`: human-box loss weights
- `REF_TEXT_SCALE`: RefHuman text-conditioning scale for the shared pose queries
- `W_REF_MATCH`: independent RefHuman expression-to-candidate classification loss
- `REF_POSE_QUALITY_ALPHA`: pose-quality exponent used after the independent RefHuman match score during evaluation/inference
- `SCHEMA_JOINT_PRIORS_PATH`: JSON file containing schema-specific box-relative joint priors
- `W_IMAGE_COORD`, `W_COARSE_COORD`, `W_DEFORM_COORD`, `W_REFINE_COORDS`: regression and coordinate deep-supervision weights
- `LOCATE_IMAGE_TOKEN_LIMIT`: raw MoonViT token budget per image
- `STAGE1_LOCATE_BATCH_TOKEN_LIMIT`, `STAGE2_LOCATE_BATCH_TOKEN_LIMIT`: local micro-batch token budgets

### Evaluate LocatePose

Evaluate the latest LocatePose run:

```bash
bash scripts/eval_locatepose.sh
```

By default, `scripts/eval_locatepose.sh` evaluates the person-query outputs in one forward pass. Legacy generated-box checkpoints can still opt into `BOX_SOURCE=locate_generate`.

Evaluate a specific checkpoint or stage directory:

```bash
CHECKPOINT=outputs/locatepose/<run_name>/stage2_unfreeze_locate_person_queries \
bash scripts/eval_locatepose.sh
```

Evaluate on multiple datasets:

```bash
DATASETS=coco,mpii,crowdpose,refhuman bash scripts/eval_locatepose.sh
```

Evaluate the GT-box upper bound instead of the person-query path:

```bash
BOX_SOURCE=gt bash scripts/eval_locatepose.sh
```

For a legacy generated-box checkpoint, select that path explicitly; `transformers` avoids the optional `vllm` dependency:

```bash
BOX_SOURCE=locate_generate LOCATE_GENERATION_BACKEND=transformers \
bash scripts/eval_locatepose.sh
```

Outputs are written to:

```text
outputs/locatepose/<run_name>/eval_locatepose_<timestamp>/
```

The evaluation directory contains `summary.json`, `predictions.jsonl`, `predictions.json`, `report.md`, and optional visualizations.

### Infer LocatePose

Run image or folder inference from a trained LocatePose checkpoint:

```bash
bash scripts/infer_locatepose.sh \
  --checkpoint outputs/locatepose/<run_name>/stage2_unfreeze_locate_person_queries \
  --input demo/images \
  --format coco
```

Legacy generated-box inference can still use the Transformers backend explicitly:

```bash
BOX_SOURCE=locate_generate LOCATE_GENERATION_BACKEND=transformers \
bash scripts/infer_locatepose.sh \
  --checkpoint outputs/locatepose/<run_name>/stage2_unfreeze_locate_person_queries \
  --input demo/images \
  --format crowdpose
```

Final keypoint coordinates come exclusively from the direct regression head.

RefHuman caption-conditioned inference example:

```bash
REF_POSE_QUALITY_ALPHA=0.25 \
bash scripts/infer_locatepose.sh \
  --checkpoint outputs/locatepose/<run_name>/stage2_unfreeze_locate_person_queries \
  --input datasets/refhuman \
  --format refhuman \
  --split val
```

RefHuman inference detects all person-query candidates in the same forward pass, scores them with `ref_match_head`, and selects the referred person with `ref_score × pose_quality^0.25`. The selected candidate uses the same shared pose decoder as ordinary all-person inference.

The inference directory writes `summary.json`, `predictions.jsonl`, `predictions.json`, optional format-specific exports, `manifest.json`, and visualizations.

### Score Exported Predictions

Rescore a saved LocatePose or QwenPose prediction file against the dataset annotations:

```bash
PYTHONPATH=src python -m qwenpose.score_pose_predictions \
  --predictions outputs/locatepose/<run_name>/eval_locatepose_<timestamp>/predictions.jsonl \
  --dataset_root datasets \
  --split val
```

## QwenPose

QwenPose uses `Qwen3-VL-4B-Instruct` as the backbone and trains the same shared pose head with a two-stage schedule.

### Default QwenPose schedule

| Stage | Directory | Backbone state | Box source | Default datasets | Default epochs |
|-------|-----------|----------------|------------|------------------|----------------|
| stage 1 | `stage1_freeze_qwen` | freeze Qwen | `gt` | `coco` | `30` |
| stage 2 | `stage2_qwen_box_closed_loop` | unfreeze Qwen LoRA and vision LoRA | `qwen_generate` | `coco` | `12` |

Additional default knobs:

- `STAGE1_BATCH_SIZE=16`
- `STAGE2_BATCH_SIZE=1`
- `STAGE1_GRAD_ACCUM_STEPS=2`
- `STAGE2_GRAD_ACCUM_STEPS=8`
- `QWEN_FEATURE_SIZE=64`
- `QWEN_FEATURE_REFINER_LAYERS=1`
- `QWEN_BOX_MAX_NEW_TOKENS=4096`
- `BOX_MATCH_IOU_THRESH=0.10`
- `BOX_NMS_IOU_THRESH=0.70`

### Train QwenPose

Start a new run:

```bash
bash scripts/train_qwenpose_two_stage.sh
```

Example with explicit run name and 8 GPUs:

```bash
RUN_NAME=qwenpose_v1 \
NPROC_PER_NODE=8 \
ZERO_STAGE=zero2 \
bash scripts/train_qwenpose_two_stage.sh
```

Quick data-path smoke test:

```bash
DRY_RUN_DATA=1 ZERO_STAGE=none NPROC_PER_NODE=1 bash scripts/train_qwenpose_two_stage.sh
```

Resume from an existing run, stage directory, checkpoint directory, or checkpoint file:

```bash
bash scripts/train_qwenpose_two_stage.sh --resume outputs/qwenpose_two_stage_qwen/<run_name>
```

### Common QwenPose variables

- `QWEN_MODEL_PATH`: local Qwen3-VL-4B-Instruct weights
- `DATASET_ROOT`: dataset root, default `datasets`
- `OUTPUT_ROOT`: training root, default `outputs/qwenpose_two_stage_qwen`
- `ZERO_STAGE`: one of `zero2`, `zero3`, `zero3_offload`, or `none`
- `STAGE1_TRAIN_DATASETS`, `STAGE2_TRAIN_DATASETS`: comma-separated dataset lists
- `QWEN_MIN_PIXELS`, `QWEN_MAX_PIXELS`: optional processor pixel budget overrides
- `QWEN_BOX_MAX_NEW_TOKENS`: maximum new tokens for generated bbox JSON
- `BOX_MATCH_IOU_THRESH`, `BOX_NMS_IOU_THRESH`: generated-box matching and NMS thresholds
- `MERGE_FINAL_WEIGHTS`: export a merged Qwen checkpoint after training when enabled

### Evaluate QwenPose

Evaluate the latest QwenPose run:

```bash
bash scripts/eval_qwenpose.sh
```

Evaluate a specific stage directory:

```bash
CHECKPOINT=outputs/qwenpose_two_stage_qwen/<run_name>/stage2_qwen_box_closed_loop \
bash scripts/eval_qwenpose.sh
```

Evaluate the GT-box upper bound:

```bash
BOX_SOURCE=gt bash scripts/eval_qwenpose.sh
```

By default, `scripts/eval_qwenpose.sh` evaluates `coco,mpii,crowdpose,refhuman`. Override `EVAL_DATASETS` when needed.

## Output Structure

Typical LocatePose run:

```text
outputs/locatepose/<run_name>/
├── logs/
├── stage1_freeze_locate_person_queries/
└── stage2_unfreeze_locate_person_queries/
```

Typical QwenPose run:

```text
outputs/qwenpose_two_stage_qwen/<run_name>/
├── logs/
├── stage1_freeze_qwen/
└── stage2_qwen_box_closed_loop/
```

Each stage directory may contain `checkpoint-*`, `checkpoint_step_*.pt`, `qwenpose_checkpoint.pt`, visualizations, and stage-local logs depending on the chosen settings.

## Versioning

This repository tracks public snapshots with:

- `VERSION`: repository version string
- `CHANGELOG.md`: newest release first
- `qwenpose.__version__`: Python package version
- Git tags such as `v2.3`

When publishing a new snapshot, update the code, README, changelog, and tag together so the Git history and the documented workflow stay aligned.
