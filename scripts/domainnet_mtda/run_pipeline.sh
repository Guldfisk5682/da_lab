#!/usr/bin/env bash
set -euo pipefail

SOURCE="${1:?usage: run_pipeline.sh <R|Q> [seed]}"
SEED="${2:-100}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

bash scripts/domainnet_mtda/run_source_probe.sh "${SOURCE}" "${SEED}"
bash scripts/domainnet_mtda/run_ours_one.sh "${SOURCE}" "${SEED}"
