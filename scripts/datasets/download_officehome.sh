#!/bin/bash

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/path/to/datasets}"
OFFICEHOME_ARCHIVE="${OFFICEHOME_ARCHIVE:-}"
OFFICEHOME_URL="${OFFICEHOME_URL:-}"
OFFICEHOME_ARCHIVE_NAME="${OFFICEHOME_ARCHIVE_NAME:-OfficeHomeDataset_10072016.zip}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-${DATA_ROOT%/}/downloads}"
TARGET_DIR="${DATA_ROOT%/}/office_home"
ARCHIVE_PATH=""

canonical_domain_name() {
  local raw_name
  raw_name="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
  raw_name="${raw_name// /_}"
  raw_name="${raw_name//-/_}"
  case "${raw_name}" in
    art) echo "art" ;;
    clipart) echo "clipart" ;;
    product) echo "product" ;;
    real_world|realworld|real_worlds) echo "real_world" ;;
    *)
      echo ""
      ;;
  esac
}

find_domain_dir() {
  local search_root="$1"
  local wanted="$2"
  find "${search_root}" -type d | while read -r candidate; do
    local base
    base="$(basename "${candidate}")"
    if [ "$(canonical_domain_name "${base}")" = "${wanted}" ]; then
      echo "${candidate}"
      break
    fi
  done
}

if [ "${DATA_ROOT}" = "/path/to/datasets" ]; then
  echo "Please set DATA_ROOT to a real dataset root before downloading Office-Home." >&2
  exit 1
fi

mkdir -p "${TARGET_DIR}" "${DOWNLOAD_DIR}"

if [ -d "${TARGET_DIR}/art" ] && [ -d "${TARGET_DIR}/clipart" ] && [ -d "${TARGET_DIR}/product" ] && [ -d "${TARGET_DIR}/real_world" ]; then
  echo "Office-Home already looks prepared under ${TARGET_DIR}"
  exit 0
fi

if [ -d "${TARGET_DIR}/art" ] || [ -d "${TARGET_DIR}/clipart" ] || [ -d "${TARGET_DIR}/product" ] || [ -d "${TARGET_DIR}/real_world" ]; then
  echo "Found a partial Office-Home directory under ${TARGET_DIR}. Please clean it up before rerunning." >&2
  exit 1
fi

if [ -n "${OFFICEHOME_ARCHIVE}" ]; then
  ARCHIVE_PATH="${OFFICEHOME_ARCHIVE}"
else
  ARCHIVE_PATH="${DOWNLOAD_DIR}/${OFFICEHOME_ARCHIVE_NAME}"
  if [ ! -f "${ARCHIVE_PATH}" ]; then
    if [ -z "${OFFICEHOME_URL}" ]; then
      cat <<EOF >&2
Office-Home download information

Dassl documents the official Office-Home page here:
  http://hemanthdv.org/OfficeHome-Dataset/

This repository does not hardcode a direct archive URL because mirrors can change.
Please either:

1. provide a local archive:
   OFFICEHOME_ARCHIVE=/path/to/${OFFICEHOME_ARCHIVE_NAME} DATA_ROOT=${DATA_ROOT} $0

2. or provide a direct download URL:
   OFFICEHOME_URL=https://.../${OFFICEHOME_ARCHIVE_NAME} DATA_ROOT=${DATA_ROOT} $0
EOF
      exit 2
    fi

    if command -v curl >/dev/null 2>&1; then
      curl -L "${OFFICEHOME_URL}" -o "${ARCHIVE_PATH}"
    elif command -v wget >/dev/null 2>&1; then
      wget -O "${ARCHIVE_PATH}" "${OFFICEHOME_URL}"
    else
      echo "Neither curl nor wget is available to download Office-Home." >&2
      exit 3
    fi
  fi
fi

if [ ! -f "${ARCHIVE_PATH}" ]; then
  echo "Office-Home archive not found: ${ARCHIVE_PATH}" >&2
  exit 4
fi

TMP_DIR="$(mktemp -d /tmp/officehome_extract.XXXXXX)"
cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

case "${ARCHIVE_PATH}" in
  *.zip)
    unzip -q "${ARCHIVE_PATH}" -d "${TMP_DIR}"
    ;;
  *.tar.gz|*.tgz)
    tar -xzf "${ARCHIVE_PATH}" -C "${TMP_DIR}"
    ;;
  *.tar)
    tar -xf "${ARCHIVE_PATH}" -C "${TMP_DIR}"
    ;;
  *)
    echo "Unsupported archive format: ${ARCHIVE_PATH}" >&2
    exit 5
    ;;
esac

for domain_name in art clipart product real_world; do
  found_dir="$(find_domain_dir "${TMP_DIR}" "${domain_name}")"
  if [ -z "${found_dir}" ]; then
    echo "Could not locate domain '${domain_name}' after extraction from ${ARCHIVE_PATH}" >&2
    exit 6
  fi

  source_dir="${found_dir}"
  if [ -d "${found_dir}/images" ]; then
    source_dir="${found_dir}/images"
  fi

  mkdir -p "${TARGET_DIR}/${domain_name}"

  copied_any=0
  find "${source_dir}" -mindepth 1 -maxdepth 1 -type d | while read -r class_dir; do
    cp -a "${class_dir}" "${TARGET_DIR}/${domain_name}/"
    copied_any=1
  done

  if [ "$(find "${TARGET_DIR}/${domain_name}" -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 0 ]; then
    echo "No class folders were copied for domain '${domain_name}' from ${source_dir}" >&2
    exit 7
  fi
done

echo "Office-Home is ready under ${TARGET_DIR}"
