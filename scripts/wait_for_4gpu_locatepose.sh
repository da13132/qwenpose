#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${PROJECT_ROOT}/scripts/locatepose.sh}"

# Monitor the four physical GPUs used by locatepose.sh unless overridden.
GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
MIN_FREE_MIB="${MIN_FREE_MIB:-22000}"
POLL_SECONDS="${POLL_SECONDS:-3}"
STABLE_CHECKS="${STABLE_CHECKS:-2}"
LOCK_FILE="${LOCK_FILE:-/tmp/qwenpose-locatepose-gpu-wait.lock}"

require_positive_int() {
  local name="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${name} must be a positive integer, got: ${value}" >&2
    exit 2
  fi
}

require_positive_int MIN_FREE_MIB "${MIN_FREE_MIB}"
require_positive_int POLL_SECONDS "${POLL_SECONDS}"
require_positive_int STABLE_CHECKS "${STABLE_CHECKS}"

IFS=',' read -r -a gpu_list <<< "${GPU_IDS}"
if [[ "${#gpu_list[@]}" -lt 1 ]]; then
  echo "GPU_IDS must contain at least one physical GPU index, got: ${GPU_IDS}" >&2
  exit 2
fi

declare -A seen_gpu=()
for gpu in "${gpu_list[@]}"; do
  if [[ ! "${gpu}" =~ ^[0-9]+$ ]]; then
    echo "GPU_IDS accepts numeric physical GPU indices only, got: ${gpu}" >&2
    exit 2
  fi
  if [[ -n "${seen_gpu[${gpu}]:-}" ]]; then
    echo "GPU_IDS contains a duplicate index: ${gpu}" >&2
    exit 2
  fi
  seen_gpu["${gpu}"]=1
done

if [[ ! -x "${TRAIN_SCRIPT}" ]]; then
  echo "LocatePose training script is not executable: ${TRAIN_SCRIPT}" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi was not found." >&2
  exit 2
fi
if ! command -v flock >/dev/null 2>&1; then
  echo "flock was not found." >&2
  exit 2
fi

# Keep the lock across exec so another waiter cannot launch a duplicate run while
# the training process is still alive.
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "Another LocatePose GPU waiter or training launch already holds ${LOCK_FILE}." >&2
  exit 3
fi

check_free_memory() {
  local query_output
  if ! query_output="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)"; then
    echo "[$(date -Is)] nvidia-smi query failed; retrying." >&2
    return 1
  fi

  declare -A free_by_gpu=()
  local index free_mib
  while IFS=',' read -r index free_mib; do
    index="${index//[[:space:]]/}"
    free_mib="${free_mib//[[:space:]]/}"
    [[ "${index}" =~ ^[0-9]+$ && "${free_mib}" =~ ^[0-9]+$ ]] || continue
    free_by_gpu["${index}"]="${free_mib}"
  done <<< "${query_output}"

  local ready=0
  local -a status=()
  for gpu in "${gpu_list[@]}"; do
    free_mib="${free_by_gpu[${gpu}]:-}"
    if [[ -z "${free_mib}" ]]; then
      status+=("gpu${gpu}=missing")
      ready=1
    else
      status+=("gpu${gpu}=${free_mib}MiB")
      if (( free_mib < MIN_FREE_MIB )); then
        ready=1
      fi
    fi
  done
  echo "[$(date -Is)] ${status[*]} required>=${MIN_FREE_MIB}MiB"
  return "${ready}"
}

echo "Waiting for GPUs ${GPU_IDS}: each must have at least ${MIN_FREE_MIB} MiB free."
echo "The condition must pass ${STABLE_CHECKS} consecutive checks; polling every ${POLL_SECONDS}s."
echo "Press Ctrl-C to cancel. Training command: ${TRAIN_SCRIPT} $*"

consecutive=0
while true; do
  if check_free_memory; then
    ((consecutive += 1))
    echo "[$(date -Is)] GPU condition passed (${consecutive}/${STABLE_CHECKS})."
    if (( consecutive >= STABLE_CHECKS )); then
      break
    fi
  else
    consecutive=0
  fi
  sleep "${POLL_SECONDS}"
done

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${#gpu_list[@]}}"
echo "[$(date -Is)] GPU condition is stable; starting LocatePose on ${CUDA_VISIBLE_DEVICES}."
cd "${PROJECT_ROOT}"
exec bash "${TRAIN_SCRIPT}" "$@"
