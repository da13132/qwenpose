# Changelog

All notable changes to this repository are recorded here, with the newest release listed first.

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
