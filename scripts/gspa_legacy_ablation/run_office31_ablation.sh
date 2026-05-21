#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/../.."

if [ "$#" -eq 0 ]; then
  EXPERIMENTS=(L0 L1 L2 L3 L4 L5)
else
  EXPERIMENTS=("$@")
fi

SEED="${SEED:-1}"
TASKS=(
  "A W"
  "A D"
  "D W"
  "D A"
  "W D"
  "W A"
)

python scripts/gspa_legacy_ablation/make_ablation_configs.py

for exp_code in "${EXPERIMENTS[@]}"; do
  for task in "${TASKS[@]}"; do
    set -- ${task}
    src="$1"
    tgt="$2"
    bash scripts/gspa_legacy_ablation/run_one.sh "${exp_code}" "${src}" "${tgt}" "${SEED}"
  done
done
