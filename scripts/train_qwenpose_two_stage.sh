#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[deprecated] scripts/train_qwenpose_two_stage.sh has been renamed to scripts/train_qwenpose_three_stage.sh" >&2
exec "${SCRIPT_DIR}/train_qwenpose_three_stage.sh" "$@"
