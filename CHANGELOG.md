# Changelog

All notable changes to this repository are recorded here, with the newest release listed first.

## v0.1.0 - 2026-06-25

- First versioned public release of the QwenPose two-stage Qwen3-VL training snapshot.
- Expanded the English and Chinese READMEs with reproducible environment setup, dataset layouts, model download links, training flow, evaluation flow, and release/version guidance.
- Pinned the tested runtime dependency versions for the public Qwen3-VL workflow.
- Added AIC dataset path compatibility for both commonly used annotation layouts:
  `ai_challenger_keypoint_train_annotations_20170909/...` and
  `ai_challenger_keypoint_train_20170902/keypoint_train_annotations_20170902.json`.
- Added repository-level version tracking through `VERSION`, `CHANGELOG.md`, and `qwenpose.__version__`.
