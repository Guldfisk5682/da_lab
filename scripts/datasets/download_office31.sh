#!/bin/bash

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/path/to/datasets}"
OFFICE31_ARCHIVE="${OFFICE31_ARCHIVE:-}"
TARGET_DIR="${DATA_ROOT%/}/office31"

if [ -z "${OFFICE31_ARCHIVE}" ]; then
  cat <<EOF
Office-31 download entrypoint

This script intentionally does not fetch anything by itself.
Provide a manually downloaded archive path and rerun:

  OFFICE31_ARCHIVE=/path/to/office31.(tar.gz|tgz|zip) DATA_ROOT=${DATA_ROOT} $0

Expected extracted layout:
  \$DATA_ROOT/office31/amazon/<class_name>/*.jpg
  \$DATA_ROOT/office31/dslr/<class_name>/*.jpg
  \$DATA_ROOT/office31/webcam/<class_name>/*.jpg

If your archive expands to an extra top-level directory, move the extracted
amazon/dslr/webcam folders under \$DATA_ROOT/office31 before training.
EOF
  exit 1
fi

mkdir -p "${TARGET_DIR}"

case "${OFFICE31_ARCHIVE}" in
  *.tar.gz|*.tgz)
    tar -xzf "${OFFICE31_ARCHIVE}" -C "${TARGET_DIR}"
    ;;
  *.zip)
    unzip -q "${OFFICE31_ARCHIVE}" -d "${TARGET_DIR}"
    ;;
  *)
    echo "Unsupported archive format: ${OFFICE31_ARCHIVE}" >&2
    exit 2
    ;;
esac

echo "Extracted Office-31 archive into ${TARGET_DIR}"
