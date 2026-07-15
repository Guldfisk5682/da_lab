#!/bin/bash

set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "Usage: bash $0 <gpu-index> <A|C> <seed>" >&2
  exit 2
fi

GPU_INDEX="$1"
SOURCE="$2"
SEED="$3"
MIN_FREE_MIB="${MIN_FREE_MIB:-12000}"
POLL_SECONDS="${POLL_SECONDS:-60}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "${REPO_ROOT}"

echo "Waiting for GPU${GPU_INDEX}: source=${SOURCE}, seed=${SEED}, minimum free=${MIN_FREE_MIB} MiB"
while true; do
  free_mib="$(
    nvidia-smi -i "${GPU_INDEX}" --query-gpu=memory.free \
      --format=csv,noheader,nounits | tr -d " "
  )"
  timestamp="$(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "${timestamp} GPU${GPU_INDEX} free=${free_mib} MiB"
  if [ "${free_mib}" -ge "${MIN_FREE_MIB}" ]; then
    if CUDA_VISIBLE_DEVICES="${GPU_INDEX}" python - <<'PY'
import torch

assert torch.cuda.is_available()
probe = torch.ones(1, device="cuda")
assert probe.item() == 1
PY
    then
      break
    fi
  fi
  sleep "${POLL_SECONDS}"
done

echo "GPU${GPU_INDEX} is available; starting ${SOURCE} seed${SEED}"
export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export PYTHONUNBUFFERED=1
exec bash scripts/maple_curriculum_mtda/run_student_soft_pilot.sh \
  "${SOURCE}" "${SEED}"
