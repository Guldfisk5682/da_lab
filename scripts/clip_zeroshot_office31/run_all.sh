#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-100}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/data}"

cd "${ROOT_DIR}"
for source in A D W; do
  python scripts/clip_zeroshot_office31/evaluate_mixture.py \
    --source "${source}" \
    --root "${DATA_ROOT}" \
    --seed "${SEED}" \
    --output-dir "output/office31_clip_zeroshot/${source}2mixture/seed${SEED}"
done
