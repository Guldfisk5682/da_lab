#!/bin/bash

set -euo pipefail

DEFAULT_REPO_ROOT="/workspace/txc/da_lab"
AUTO_REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
if [ -d "${DEFAULT_REPO_ROOT}" ]; then
  REPO_ROOT="${REPO_ROOT:-${DEFAULT_REPO_ROOT}}"
else
  REPO_ROOT="${REPO_ROOT:-${AUTO_REPO_ROOT}}"
fi

cd "${REPO_ROOT}"

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Usage: bash scripts/clip_tssp_mtda/run_officehome_all.sh <clip_tssp_full|clip_tssp_no_gap> [--debug]" >&2
  exit 1
fi

METHOD="$1"
DEBUG_FLAG="${2:-}"
SEED="${SEED:-1}"

for source_code in A C P R; do
  if [ -n "${DEBUG_FLAG}" ]; then
    bash "${REPO_ROOT}/scripts/clip_tssp_mtda/run_officehome_one.sh" "${source_code}" "${SEED}" "${METHOD}" "${DEBUG_FLAG}"
  else
    bash "${REPO_ROOT}/scripts/clip_tssp_mtda/run_officehome_one.sh" "${source_code}" "${SEED}" "${METHOD}"
  fi
done
