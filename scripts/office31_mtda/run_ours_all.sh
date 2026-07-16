#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-100}"
for source in A D W; do
  bash scripts/office31_mtda/run_source_probe.sh "${source}" "${SEED}"
done
for source in A D W; do
  bash scripts/office31_mtda/run_ours_one.sh "${source}" "${SEED}"
done
