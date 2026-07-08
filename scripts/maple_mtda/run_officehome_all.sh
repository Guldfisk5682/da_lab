#!/bin/bash

set -euo pipefail

SEED="${SEED:-1}"

for SOURCE in A C P R; do
  bash "$(dirname "$0")/run_officehome_one.sh" "${SOURCE}" "${SEED}" "$@"
done
