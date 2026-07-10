# Changelog

All notable changes to this repository are recorded here, with the newest release listed first.

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
