#!/usr/bin/env bash
set -euo pipefail

SEED="${SEED:-100}"
for specification in "$@"; do
  read -r method protocol source <<< "${specification}"
  bash scripts/prompt_baselines_mtda/run_one.sh \
    "${method}" "${protocol}" "${source}" "${SEED}"
done
