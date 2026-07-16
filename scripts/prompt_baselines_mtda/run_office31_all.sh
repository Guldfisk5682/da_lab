#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-100}"
for method in maple coop cocoop; do
  for source in A D W; do
    bash scripts/prompt_baselines_mtda/run_office31_one.sh \
      "${method}" "${source}" "${SEED}"
  done
done
