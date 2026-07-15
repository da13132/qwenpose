# Changelog

All notable changes to this repository are recorded here, with the newest release listed first.

## Unreleased - unified 800 LocatePose

- Restored LocateAnything autoregressive box generation as the only default LocatePose training path: `scripts/locatepose.sh` now runs frozen-GT Stage 1 followed by generated-box/coordinate-token-LM Stage 2; the temporary `locatepose4llm.sh` wrapper was removed.
- Replaced the legacy 256/640 dual-RGB branches with one 800×800 P2/P3/P4 pose pyramid at 200×200, 100×100, and 50×50, shared without architectural changes across Stage 1/2.
- Removed every SimCC training, decoding, evaluation, inference, and shell configuration path. Keypoint coordinates now come only from direct regression with coordinate deep supervision.
- Added a two-layer human box refinement decoder, explicit person objectness, refined-box export, relative/L1/GIoU box losses, and DN-DETR/DINO-style positive/negative box denoising queries that never enter the keypoint decoder.
- Kept the Pose visual input identical across stages by using normalized raw MoonViT features. The merged Stage 2 retains the full LLM for grounding and RefHuman text conditioning without an LLM-image feature-fusion branch.
- Reworked RefHuman to use LocateAnything's native referring-expression grounding directly: the primary prompt returns one target box, target-only box LM/matching supervises it, and an all-person generation plus `ref_match_head` is retained only as the empty-result fallback. The existing regression head locally refines the grounded box; inference restores the Locate box if a RefHuman refinement drifts below 0.30 IoU.
- Added strict checkpoint guards: checkpoints from the dual-RGB/SimCC architecture are incompatible and must be replaced by a newly trained Stage 1 checkpoint before Stage 2.
- Merged the old LocatePose Stage 2/3 runs into one 13-epoch Stage 2 over COCO, MPII, CrowdPose, and RefHuman; added per-dataset traversal multipliers with a `3,3,3,1` default, zero disabling, and deterministic cross-epoch continuation for fractional values.

## v2.3 - 2026-07-12

- Reworked LocatePose Stage 2 into trainable human grounding: LocateAnything LoRA and vision LoRA are unfrozen, bbox grounding LM supervision is enabled at `0.10`, point-token supervision is disabled, and the same native Locate prompt is used for generation, multimodal features, and LM training. RefHuman retains single-person referring-expression grounding while AIC remains available but disabled in the default dataset lists.
- Aligned COCO, CrowdPose, MPII, and AIC with a YOLO-Pose-style box/keypoint contract. All valid non-crowd person boxes are retained for bbox supervision, zero-keypoint people are masked out of pose losses, MPII participates in all-person bbox LM training, and the record cache contract is bumped to v11.
- Preserved every generated Locate proposal through PoseHead training, added Hungarian one-to-one matching for pose supervision, excluded unsupervised queries from pose-loss denominators, disabled pre-PoseHead NMS by default, and added high-threshold post-pose duplicate suppression for crowded scenes.
- Hardened training and evaluation with legacy `keypoint_mask` compatibility, first-batch target-contract validation before model loading, distributed cleanup on failures, raw Locate bbox export separate from PoseHead context boxes, schema-aware post-NMS scoring, configurable GPU/process environment overrides, and expanded CPU regression coverage.

## v2.2 - 2026-07-11

- Added synchronized finite-value checks for model outputs, gradients, and trainable parameters. Loss spikes and non-finite gradients now skip the optimizer step consistently on every distributed rank instead of silently replacing invalid values.
- Changed the default vision-only Stage 1 recipe to freeze the Locate backbone and train the PoseHead with batch 32 per GPU, two-step gradient accumulation, a 100-epoch ceiling, and the existing 60,000-step cap. The optional vision-LoRA learning-rate multiplier is reduced to `0.01`.
- Changed Stage 2 multimodal fusion to start exactly from the normalized Stage 1 visual feature map while retaining a learnable zero-initialized context branch and neutral gate.
- Improved long-running multi-worker training robustness by raising the open-file soft limit when possible, using the `file_system` tensor-sharing strategy by default, and adding a locked four-GPU availability waiter script.

## v2.1.1 - 2026-07-10

- Added synchronized Stage-1 pose augmentation: horizontal flip with left/right joint remapping, affine rotation/scale/translation, color jitter, grayscale, blur, and random erasing. Boxes, loss boxes, loss areas, keypoints, validity masks, and visualization all follow the same transform.
- Unified Stage-1 image loading so Dataset workers open augmented images once and share the same original-resolution uint8 result with MoonViT and the local RGB branch. The default two workers per rank overlap augmentation with GPU compute, while non-augmented runs stay path-backed to avoid large tensor IPC. Vision-only Stage 1 no longer constructs or forwards dataset prompts.
- Kept Stage 2 augmentation disabled by default so RefHuman left/right descriptions remain semantically aligned, while retaining prompts for multimodal Locate generation and grounding losses.
- Corrected the documented Stage 1 default to batch 6 per GPU, global batch 24 on four GPUs, with an 18,432-token local micro-batch budget.

## v2.1 - 2026-07-10

- Changed LocatePose vision-only Stage 1 to instantiate and load only MoonViT, the frozen `mlp1` projector, and vision LoRA. The Qwen2.5 language model and tokenizer are no longer constructed or read from checkpoint shards during Stage 1.
- Added selective safetensors loading, an image-processor-only input path, checkpoint metadata for the backbone load mode, and exact Stage-1-to-Stage-2 vision-LoRA namespace compatibility tests.
- Added cross-rank vision-token cost bucketing and stage-specific local micro-batch token budgets, preventing one rank from receiving several maximum-resolution images and exhausting memory while the other ranks remain underutilized.
- Used the Stage 1 default of batch 6 per GPU (global batch 24 on four GPUs) with an automatically scaled 18,432-token local micro-batch budget.

## v2.0 - 2026-07-10

- Reworked Stage 1 into a fast pose warmup: skip the 3B language model, use MoonViT visual features directly, train only vision LoRA plus pose adapters, keep RefHuman for multimodal Stage 2, and use a 60,000-step schedule with a default four-GPU global batch of 12.
- Added explicit `vision_only`/`multimodal` feature sources and `frozen`/`vision_lora`/`all_lora` backbone scopes, with checkpoint metadata and regression coverage that verifies the vision-only path never calls the language model while preserving visual gradients.
- Switched pose transformers and Locate LoRA adapters to zero dropout, made visual/refinement adapters start from exact identity, opened local/deformable refinement gates faster, and removed all coarse/deform/intermediate SimCC forward passes so only the final refinement output allocates logits and receives SimCC supervision at weight `0.5`.
- Restored the positive-visibility policy for all annotated joints, including occluded COCO/CrowdPose/AIC joints and valid MPII joints, while retaining coordinate-valid training visualization and bumping the record cache contract to v10.
- Updated the default LocatePose schedule to `coco,mpii,crowdpose` for vision-only Stage 1 and `coco,mpii,crowdpose,refhuman` for multimodal Stage 2, synchronized the English/Chinese documentation, and bumped the public snapshot to `v2.0`.

## v1.4 - 2026-07-10

- Expanded the maintained LocatePose recipe to the five-dataset `coco,mpii,crowdpose,aic,refhuman` schedule with size-proportional homogeneous-source batches, shared ALL_POSE semantics, RefHuman-only text conditioning, and schema-specific geometric priors.
- Reworked dataset geometry and supervision metadata: MPII now follows the MMPose center/scale convention, AIC uses `human_annotations` boxes, CrowdPose keeps its head joint separate from MPII/AIC `head_top`, and records carry native loss boxes, areas, visibility masks, context scales, and jitter policies.
- Made multi-dataset pose losses comparable by averaging valid joints per person, adding full-image coordinate supervision, excluding unavailable MPII visibility targets, and normalizing SimCC cross entropy by `log(bins)`; the default SimCC resolution is now 256 bins.
- Added schema-prior generation/configuration, per-dataset training diagnostics, corrected generated-box conditioning, checkpoint-persistent priors, and regression tests for geometry, loss normalization, sampling, text gating, visualization edges, and checkpoint reload behavior.
- Updated LocatePose inference, metrics, visualizations, documentation, and launch defaults to match the new data contracts; existing PoseHead checkpoints from the previous 22-joint union are not compatible with the new 23-joint union.
- Bumped the repository snapshot version to `v1.4` through `VERSION`, `qwenpose.__version__`, dependency headers, README release notes, and the Git tag convention used for public releases.

## v1.3 - 2026-07-09

- Updated the LocatePose public defaults to the current multi-dataset stage-1 recipe, four-GPU launch layout, larger pose ROI, and configurable dataset mixing weights in `scripts/locatepose.sh`.
- Fixed dataset conversion details for stronger cross-dataset training: COCO/CrowdPose now preserve visible-vs-occluded labels, CrowdPose uses an explicit CrowdPose14 prompt, and MPII uses center/scale-derived pseudo boxes instead of tight visible-keypoint boxes.
- Added split-aware AIC train/validation path resolution for both training records and metric loading so AIC validation uses the official validation annotations and images.
- Bumped the repository snapshot version to `v1.3` through `VERSION`, `qwenpose.__version__`, dependency headers, README release notes, and the Git tag convention used for public releases.

## v1.2 - 2026-07-01

- Refreshed the published LocatePose launch defaults in `scripts/locatepose.sh`, moving the default visible devices back to `0,3` with `NPROC_PER_NODE=2` and keeping the current two-stage `crowdpose -> coco,mpii,crowdpose,refhuman` training schedule.
- Added the current auxiliary supervision recipe for LocatePose through `simcc_bins`, coarse/deform/refine coordinate deep supervision, and SimCC distribution losses across `scripts/locatepose.sh`, `qwenpose.model`, `qwenpose.losses`, and `qwenpose.train_pose`.
- Updated the public English and Chinese READMEs so the documented LocatePose defaults, example launch command, and exposed knobs match the live training script for this release.
- Bumped the repository snapshot version to `v1.2` through `VERSION`, `qwenpose.__version__`, dependency headers, and the Git tag convention used for public releases.

## v1.1 - 2026-06-30

- Refreshed the published LocatePose training defaults in `scripts/locatepose.sh`, including the current `crowdpose` stage-1 dataset, the `80 + 5` epoch two-stage schedule, updated stage-1 box jitter, aligned pose loss weights, and the current four-GPU launch defaults.
- Added the public LocatePose inference and scoring utilities centered on `scripts/infer_locatepose.sh`, `qwenpose.infer_locatepose`, `qwenpose.metrics`, and `qwenpose.score_pose_predictions`, so single-image, folder, RefHuman-caption, and exported-prediction workflows are documented and versioned.
- Published the integrated LocatePose vLLM evaluation/inference path through the custom LocateAnything backend wrapper and synchronized the README guidance with the new `LOCATE_GENERATION_BACKEND`, `SINGLE_PASS_PROMPT`, and fallback behavior.
- Bumped the repository snapshot version to `v1.1` through `VERSION`, `qwenpose.__version__`, dependency headers, and the Git tag convention used for public releases.

## v1.0 - 2026-06-26

- Promoted the LocatePose workflow to a first-class public entrypoint with `scripts/locatepose.sh` and `scripts/eval_locatepose.sh`, placing the LocateAnything-based recipe alongside the maintained Qwen3-VL recipe in the published snapshot.
- Published the LocatePose backend updates across data loading, LocateAnything LoRA loading, generated-box closed-loop training, and evaluation so the public code can train and validate `locate_generate` end to end.
- Kept `scripts/train_qwenpose_two_stage.sh` as the maintained QwenPose training entrypoint, synchronized its documentation with the current two-stage defaults, and removed stale one-stage/three-stage references from the public release narrative.
- Rewrote the English and Chinese READMEs for public reproducibility, including shared environment setup, dataset directory layouts, model download locations, DeepSpeed preset guidance, and separate LocatePose/QwenPose usage sections.
- Bumped the repository snapshot version to `v1.0` through `VERSION`, `qwenpose.__version__`, dependency headers, and the Git tag convention used for public releases.

## v0.3.1 - 2026-06-25

- Removed `scripts/train_qwenpose_three_stage.sh` from the public repository after reverting the maintained workflow back to the two-stage closed-loop recipe.
- Kept `scripts/train_qwenpose_two_stage.sh` as the sole published training entrypoint and cleaned the English and Chinese READMEs so they no longer mention the removed three-stage wrapper.
- Preserved compatibility in training and evaluation code for reading legacy three-stage output directories, stage names, and checkpoints.

## v0.3.0 - 2026-06-25

- Collapsed the experimental three-stage recipe back into the maintained `scripts/train_qwenpose_two_stage.sh` entrypoint.
- Removed the standalone teacher-forcing middle stage: stage 1 remains GT-box pose warmup, and stage 2 now directly runs closed-loop Qwen-generated box training.
- Kept `scripts/train_qwenpose_three_stage.sh` only as a deprecated compatibility wrapper that forwards to the two-stage script.
- Added generated-box conditioning support across training and evaluation, including bbox JSON parsing, NMS, GT matching, stage-aware resume resolution, and stage-aware checkpoint discovery.
- Updated evaluation defaults to prefer `stage2_qwen_box_closed_loop`, while keeping `BOX_SOURCE=gt` available for GT-box upper-bound evaluation.

## v0.2.2 - 2026-06-25

- Removed `scripts/train_qwenpose_one_stage.sh` from the public repository so the published training workflow exposes only the maintained two-stage entrypoint.
- Updated the English and Chinese READMEs to remove the old one-stage wrapper references and keep the public usage instructions aligned with the repository contents.

## v0.2.1 - 2026-06-25

- Clarified the public README and Chinese README so the documented defaults match the live `scripts/train_qwenpose_two_stage.sh` behavior.
- Added explicit guidance for `scripts/zero2.json`, `scripts/zero3.json`, and `scripts/zero3_offload.json`, including when to use each ZeRO preset.
- Corrected the RefHuman documentation to show that the public default recipe stays on COCO unless RefHuman is enabled explicitly.

## v0.2.0 - 2026-06-25

- Integrated the latest local QwenPose two-stage training updates centered on `scripts/train_qwenpose_two_stage.sh`.
- Added the new RGB visual branch data path through `PoseRecordDataset`, `QwenPoseModel`, training, and evaluation so train and eval both use the same image tensor input route.
- Simplified the default pose loss recipe to the current clean configuration used by the two-stage shell pipeline.
- Updated the public default two-stage recipe to the current COCO-first setup and aligned README defaults with the live shell script.
- Kept Eagle/LocatePose entry scripts out of the public snapshot.
- Verified the refreshed snapshot with shell syntax checks, Python import and compile checks, a model forward smoke test, and a full two-stage `DRY_RUN_DATA=1` shell run.

## v0.1.0 - 2026-06-25

- First versioned public release of the QwenPose two-stage Qwen3-VL training snapshot.
- Expanded the English and Chinese READMEs with reproducible environment setup, dataset layouts, model download links, training flow, evaluation flow, and release/version guidance.
- Pinned the tested runtime dependency versions for the public Qwen3-VL workflow.
- Added AIC dataset path compatibility for both commonly used annotation layouts:
  `ai_challenger_keypoint_train_annotations_20170909/...` and
  `ai_challenger_keypoint_train_20170902/keypoint_train_annotations_20170902.json`.
- Added repository-level version tracking through `VERSION`, `CHANGELOG.md`, and `qwenpose.__version__`.
