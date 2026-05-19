#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DASSL_DIR="${DASSL_DIR:-${REPO_ROOT}/../Dassl.pytorch}"

if [ ! -d "${DASSL_DIR}" ]; then
  git clone https://github.com/KaiyangZhou/Dassl.pytorch.git "${DASSL_DIR}"
fi

python -m pip install -e "${DASSL_DIR}"
python -m pip install -r "${REPO_ROOT}/requirements.txt"

echo "Dassl and project dependencies are installed."
