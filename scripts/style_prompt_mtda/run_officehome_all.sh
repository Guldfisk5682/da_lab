#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/../.."

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Usage: bash scripts/style_prompt_mtda/run_officehome_all.sh <cocoop_mt|cocoop|style_prompt|style_prompt_mtda> [--debug]" >&2
  exit 1
fi

METHOD="$1"
DEBUG_FLAG="${2:-}"
SEED="${SEED:-1}"

for source_code in A C P R; do
  if [ -n "${DEBUG_FLAG}" ]; then
    bash scripts/style_prompt_mtda/run_officehome_one.sh "${source_code}" "${SEED}" "${METHOD}" "${DEBUG_FLAG}"
  else
    bash scripts/style_prompt_mtda/run_officehome_one.sh "${source_code}" "${SEED}" "${METHOD}"
  fi
done
