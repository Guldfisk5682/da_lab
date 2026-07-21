#!/usr/bin/env bash
set -euo pipefail

# Low-memory continuation queue for the remaining DomainNet source-only prompt
# baselines. Original config files remain unchanged; batch overrides are passed
# only to these runs. Completed runs are skipped by the per-task launcher.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

SEED="${SEED:-100}"
DATA_ROOT="${DATA_ROOT:-/workspace/dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/output/domainnet_prompt_baselines}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_WORKERS="${NUM_WORKERS:-4}"
COOP_TRAIN_BATCH_SIZE="${COOP_TRAIN_BATCH_SIZE:-16}"
COCOOP_TRAIN_BATCH_SIZE="${COCOOP_TRAIN_BATCH_SIZE:-2}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-16}"

export DATA_ROOT OUTPUT_ROOT PYTHON_BIN NUM_WORKERS TEST_BATCH_SIZE

echo "Low-memory DomainNet queue (target: <=10 GiB GPU)"
echo "data_root=${DATA_ROOT} output_root=${OUTPUT_ROOT} seed=${SEED}"
echo "CoOp train batch=${COOP_TRAIN_BATCH_SIZE}"
echo "CoCoOp train batch=${COCOOP_TRAIN_BATCH_SIZE}"
echo "Test batch=${TEST_BATCH_SIZE}, workers=${NUM_WORKERS}"

"${PYTHON_BIN}" scripts/datasets/verify_domainnet_layout.py --root "${DATA_ROOT}"

for method in coop cocoop; do
  if [[ "${method}" == "coop" ]]; then
    TRAIN_BATCH_SIZE="${COOP_TRAIN_BATCH_SIZE}"
  else
    TRAIN_BATCH_SIZE="${COCOOP_TRAIN_BATCH_SIZE}"
  fi
  export TRAIN_BATCH_SIZE

  for source in C I P Q R S; do
    echo "[$(date -Is)] START ${method} ${source}2O"
    bash scripts/prompt_baselines_mtda/run_domainnet_source_only.sh \
      "${method}" "${source}" "${SEED}"
    echo "[$(date -Is)] DONE ${method} ${source}2O"
  done
done
