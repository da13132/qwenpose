#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_SCRIPT="${SCRIPT_DIR}/train_qwenpose_two_stage.sh"

echo "[Deprecated] scripts/train_qwenpose_one_stage.sh has been renamed to scripts/train_qwenpose_two_stage.sh." >&2
exec "${TARGET_SCRIPT}" "$@"
