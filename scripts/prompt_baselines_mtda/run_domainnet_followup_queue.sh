#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

WAIT_PID="${WAIT_PID:-}"
REQUIRE_FILE="${REQUIRE_FILE:-}"
SEED="${SEED:-100}"

if [[ -n "${WAIT_PID}" ]]; then
  echo "[$(date -Is)] Waiting for PID ${WAIT_PID}"
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    sleep 60
  done
fi

if [[ -n "${REQUIRE_FILE}" && ! -s "${REQUIRE_FILE}" ]]; then
  echo "Required predecessor artifact is missing: ${REQUIRE_FILE}" >&2
  exit 1
fi

for source in C I P Q R S; do
  echo "[$(date -Is)] START clip_zs ${source}2O"
  bash scripts/prompt_baselines_mtda/run_domainnet_zeroshot_clip.sh "${source}" "${SEED}"
  echo "[$(date -Is)] DONE clip_zs ${source}2O"
done

for method in coop cocoop; do
  for source in C I P Q R S; do
    echo "[$(date -Is)] START ${method} ${source}2O"
    bash scripts/prompt_baselines_mtda/run_domainnet_source_only.sh \
      "${method}" "${source}" "${SEED}"
    echo "[$(date -Is)] DONE ${method} ${source}2O"
  done
done
