#!/usr/bin/env bash
set -euo pipefail

# Protected serial continuation for the formal Office-Home ablations. Every
# completed run must produce structured metrics before its heavy CLIP checkpoint
# is pruned; an error stops the queue instead of silently continuing.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/output/officehome_mtda}"
SEED="${SEED:-100}"
WAIT_RUN="${WAIT_RUN:-}"
POLL_SECONDS="${POLL_SECONDS:-60}"

if [[ -n "${WAIT_RUN}" ]]; then
  echo "[$(date -Is)] waiting for ${WAIT_RUN}/mtda_metrics.json"
  while [[ ! -s "${WAIT_RUN}/mtda_metrics.json" ]]; do
    sleep "${POLL_SECONDS}"
  done
  find "${WAIT_RUN}" -type f \( -name 'model.pth.tar-*' -o -name 'model-best.pth.tar' \) -delete
fi

run_one() {
  local variant="$1"
  local source="$2"
  local base="${OUTPUT_ROOT}/officehome_ablation_${variant}_seed${SEED}"

  echo "[$(date -Is)] starting ${source}/${variant}"
  DATA_ROOT="${DATA_ROOT}" OUTPUT_ROOT="${OUTPUT_ROOT}" \
    bash scripts/maple_curriculum_mtda/run_officehome_ablation_one.sh "${variant}" "${source}" "${SEED}"

  local run_dir
  # A variant directory may already contain runs for another source domain.
  # Restrict discovery to the source-specific run so a completed P run cannot
  # accidentally prevent checkpoint pruning for a later R run.
  run_dir="$(find "${base}" -type f \
    -path "*/${source}2*/seed${SEED}/mtda_metrics.json" -printf '%h\n' | head -n 1)"
  if [[ -z "${run_dir}" || ! -s "${run_dir}/mtda_metrics.json" ]]; then
    echo "[$(date -Is)] missing metrics for ${source}/${variant}" >&2
    return 1
  fi
  find "${run_dir}" -type f \( -name 'model.pth.tar-*' -o -name 'model-best.pth.tar' \) -delete
  echo "[$(date -Is)] complete ${source}/${variant}: $(tr '\n' ' ' < "${run_dir}/mtda_metrics.json")"
}

# P hard-only is already running when this queue is launched.
for variant in soft_only no_replay topk16 topk32; do
  run_one "${variant}" P
done
for variant in e2h joint hard_only soft_only no_replay topk16 topk32; do
  run_one "${variant}" R
done

echo "[$(date -Is)] Office-Home formal P/R ablation queue completed"
