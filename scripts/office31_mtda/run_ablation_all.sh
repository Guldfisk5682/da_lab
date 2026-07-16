#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-100}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

for variant in e2h joint hard_only soft_only no_replay topk16 topk32; do
  for source in A D W; do
    "${ROOT_DIR}/scripts/office31_mtda/run_ablation_one.sh" \
      "${variant}" "${source}" "${SEED}"
  done
done
