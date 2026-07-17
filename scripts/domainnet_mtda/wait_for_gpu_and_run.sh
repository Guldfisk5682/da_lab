#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?usage: wait_for_gpu_and_run.sh <gpu> <R|Q> [seed]}"
SOURCE="${2:?usage: wait_for_gpu_and_run.sh <gpu> <R|Q> [seed]}"
SEED="${3:-100}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MAX_USED_MIB="${MAX_USED_MIB:-1000}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

while true; do
  used="$(nvidia-smi -i "${GPU}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  if [[ "${used}" =~ ^[0-9]+$ ]] && (( used <= MAX_USED_MIB )); then
    if CUDA_VISIBLE_DEVICES="${GPU}" python -c \
      'import torch; assert torch.cuda.is_available(); print(torch.ones(1, device="cuda").item())'
    then
      echo "GPU ${GPU} available (${used} MiB); launching DomainNet ${SOURCE} pipeline"
      exec env CUDA_VISIBLE_DEVICES="${GPU}" \
        bash scripts/domainnet_mtda/run_pipeline.sh "${SOURCE}" "${SEED}"
    fi
  fi
  echo "GPU ${GPU} unavailable (${used} MiB used); retrying in ${POLL_SECONDS}s"
  sleep "${POLL_SECONDS}"
done
