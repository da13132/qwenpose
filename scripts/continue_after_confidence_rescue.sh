#!/usr/bin/env bash
set -euo pipefail

echo "continue_after_confidence_rescue.sh is obsolete: legacy rescue checkpoints cannot be continued by the unified 800x800 architecture. Train a new Stage1 checkpoint." >&2
exit 2
