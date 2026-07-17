#!/usr/bin/env bash
set -Eeuo pipefail

# Wait for the current Stage1 job to finish checkpoint-4500, stop only that
# detached training session, and resume the same run with the current code.
# Intended to run inside the tmux session named "qwenpose".

PROJECT_ROOT="/home/bitzh_js11/ZYD/qwenpose"
RUN_ID="20260717-005727-promptproxy"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/locatepose/locatepose-3stage-${RUN_ID}"
STAGE1_DIR="${OUTPUT_DIR}/stage1_vision_gt_pose"
CHECKPOINT_DIR="${STAGE1_DIR}/checkpoint-4500"
STATE_FILE="${CHECKPOINT_DIR}/qwenpose_state.json"
TRAIN_LOG="${OUTPUT_DIR}/logs/train_${RUN_ID}.log"
WATCH_LOG="${OUTPUT_DIR}/logs/auto_restart_checkpoint_4500.log"
PYTHON="${PROJECT_ROOT}/envs/qwenpose/bin/python"
POLL_SECONDS=15

mkdir -p "${OUTPUT_DIR}/logs"
exec > >(tee -a "${WATCH_LOG}") 2>&1

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

find_old_master_pid() {
  local output_needle="locatepose-3stage-${RUN_ID}/stage1_vision_gt_pose"
  ps -eo pid=,args= | awk -v output_dir="${output_needle}" '
    $2 ~ /python$/ \
      && index($0, " -m torch.distributed.run ") \
      && index($0, output_dir) {
      print $1
      exit
    }
  '
}

find_old_training_pids() {
  local output_needle="locatepose-3stage-${RUN_ID}/stage1_vision_gt_pose"
  ps -eo pid=,args= | awk -v output_dir="${output_needle}" '
    $2 ~ /python$/ && index($0, output_dir) {
      print $1
    }
  '
}

checkpoint_is_complete() {
  [[ -s "${CHECKPOINT_DIR}/qwenpose_checkpoint.pt" ]] || return 1
  [[ -s "${STATE_FILE}" ]] || return 1
  [[ -s "${CHECKPOINT_DIR}/deepspeed/mp_rank_00_model_states.pt" ]] || return 1
  [[ -s "${CHECKPOINT_DIR}/backbone_lora_adapter/adapter_model.safetensors" ]] || return 1

  local optimizer_shards
  optimizer_shards="$(find "${CHECKPOINT_DIR}/deepspeed" -maxdepth 1 \
    -type f -name '*optim_states.pt' 2>/dev/null | wc -l)"
  [[ "${optimizer_shards}" -eq 4 ]] || return 1

  "${PYTHON}" - "${STATE_FILE}" <<'PY' >/dev/null
import json
import sys
from pathlib import Path

state = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert int(state["step"]) == 4500
assert int(state["training_state"]["global_step"]) == 4500
assert int(state["training_state"]["world_size"]) == 4
PY
}

old_master_pid="$(find_old_master_pid)"
if [[ -z "${old_master_pid}" ]]; then
  echo "[$(timestamp)] ERROR: cannot find the current Stage1 torchrun master." >&2
  exit 2
fi
old_session_id="$(
  ps -o sid= -p "${old_master_pid}" 2>/dev/null \
    | tr -d '[:space:]' \
    || true
)"
if [[ -z "${old_session_id}" ]]; then
  echo "[$(timestamp)] ERROR: cannot resolve the old training session id." >&2
  exit 2
fi

echo "[$(timestamp)] Waiting for checkpoint-4500."
echo "[$(timestamp)] Old torchrun PID=${old_master_pid}, session=${old_session_id}."

last_report=0
while ! checkpoint_is_complete; do
  if ! kill -0 "${old_master_pid}" 2>/dev/null; then
    echo "[$(timestamp)] ERROR: old training exited before checkpoint-4500 completed." >&2
    exit 3
  fi
  now="$(date +%s)"
  if (( now - last_report >= 60 )); then
    latest_step="$(
      tail -c 2000000 "${TRAIN_LOG}" 2>/dev/null \
        | tr '\r' '\n' \
        | grep '^step=' \
        | tail -n 1 \
        | sed -n 's/^step=\([0-9][0-9]*\).*/\1/p'
    )"
    echo "[$(timestamp)] Current step=${latest_step:-unknown}; still waiting."
    last_report="${now}"
  fi
  sleep "${POLL_SECONDS}"
done

echo "[$(timestamp)] checkpoint-4500 is complete; stopping old session ${old_session_id}."
mapfile -t old_training_pids < <(find_old_training_pids)
if (( ${#old_training_pids[@]} > 0 )); then
  kill -TERM "${old_training_pids[@]}" 2>/dev/null || true
fi
pkill -TERM -s "${old_session_id}" 2>/dev/null || true
for _ in $(seq 1 60); do
  mapfile -t remaining_training_pids < <(find_old_training_pids)
  if (( ${#remaining_training_pids[@]} == 0 )); then
    break
  fi
  sleep 1
done
mapfile -t remaining_training_pids < <(find_old_training_pids)
if (( ${#remaining_training_pids[@]} > 0 )); then
  echo "[$(timestamp)] ${#remaining_training_pids[@]} old training processes remain; sending SIGKILL."
  kill -KILL "${remaining_training_pids[@]}" 2>/dev/null || true
  pkill -KILL -s "${old_session_id}" 2>/dev/null || true
  sleep 2
fi

echo "[$(timestamp)] Resuming Stage1 from ${CHECKPOINT_DIR}."
cd "${PROJECT_ROOT}"
exec env \
  RUN_ID="${RUN_ID}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  LOCATEPOSE_CUDA_VISIBLE_DEVICES="4,5,6,7" \
  STAGE1_EPOCHS="50" \
  STAGE1_RESUME_FROM_CHECKPOINT="${CHECKPOINT_DIR}" \
  PROMPT_EMBEDDING_CACHE=".cache/qwenpose_text/locateanything_prompt_tokens.pt" \
  VISUALIZE_MAX_INSTANCES="5" \
  VISUALIZE_NMS_IOU_THRESH="0.50" \
  VISUALIZE_OBJECTNESS_THRESHOLD="0.05" \
  VISUALIZE_POSE_THRESHOLD="0.05" \
  bash scripts/locatepose.sh stage1
