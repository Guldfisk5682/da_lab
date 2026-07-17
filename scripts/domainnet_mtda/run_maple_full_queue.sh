#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "usage: run_maple_full_queue.sh <C|I|P|Q|R|S> [...]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
SEED="${SEED:-100}"

for source in "$@"; do
  bash scripts/domainnet_mtda/run_source_probe.sh "${source}" "${SEED}"
  BUDGET_MODE=maple_full \
    bash scripts/domainnet_mtda/run_ours_one.sh "${source}" "${SEED}"
done
