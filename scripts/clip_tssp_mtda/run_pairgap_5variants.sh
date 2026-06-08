#!/bin/bash

set -euo pipefail

DEFAULT_REPO_ROOT="/workspace/txc/da_lab"
AUTO_REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
if [ -d "${DEFAULT_REPO_ROOT}" ]; then
  REPO_ROOT="${REPO_ROOT:-${DEFAULT_REPO_ROOT}}"
else
  REPO_ROOT="${REPO_ROOT:-${AUTO_REPO_ROOT}}"
fi

cd "${REPO_ROOT}"

DEBUG_FLAG="${1:-}"
SEED="${SEED:-1}"

METHODS=(
  clip_tssp_pair_gap_adamw2e3
  clip_tssp_pair_gap_adamw1e4
  clip_tssp_pair_gap_kl
  clip_tssp_pair_gap_pl
  clip_tssp_pair_gap_pl_kl
)

if [ -n "${DEBUG_FLAG}" ] && [ "${DEBUG_FLAG}" != "--debug" ]; then
  echo "Usage: bash scripts/clip_tssp_mtda/run_pairgap_5variants.sh [--debug]" >&2
  exit 1
fi

echo "==============================================="
echo "PairGap five-variant overnight run"
echo "Seed: ${SEED}"
echo "Methods: ${METHODS[*]}"
echo "Debug: ${DEBUG_FLAG:-off}"
echo "Eval every epoch: ${EVAL_EVERY_EPOCH:-0}"
echo "==============================================="

for method in "${METHODS[@]}"; do
  echo ">>> Running ${method}"
  if [ -n "${DEBUG_FLAG}" ]; then
    bash "${REPO_ROOT}/scripts/clip_tssp_mtda/run_officehome_all.sh" "${method}" "${DEBUG_FLAG}"
  else
    bash "${REPO_ROOT}/scripts/clip_tssp_mtda/run_officehome_all.sh" "${method}"
  fi
done

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "Dry run complete; skipping collect and TensorBoard plotting."
  exit 0
fi

PYTHON="${PYTHON:-python}"
"${PYTHON}" "${REPO_ROOT}/scripts/clip_tssp_mtda/collect_officehome_results.py"
"${PYTHON}" "${REPO_ROOT}/scripts/clip_tssp_mtda/plot_tensorboard_curves.py" \
  --methods "${METHODS[@]}" \
  --seed "${SEED}"
