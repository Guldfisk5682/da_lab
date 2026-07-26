#!/usr/bin/env bash
set -euo pipefail

# Download the cleaned DomainNet release and the official train/test lists used
# by DomainNetMTDA. The archives total roughly 18 GB; allow substantially more
# free space for extraction.

DATA_ROOT="${DATA_ROOT:-/path/to/datasets}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-${DATA_ROOT%/}/downloads/domainnet}"
TARGET_DIR="${DATA_ROOT%/}/DomainNet"
KEEP_ARCHIVES="${KEEP_ARCHIVES:-1}"
VERIFY_LAYOUT="${VERIFY_LAYOUT:-1}"
BASE_URL="${DOMAINNET_BASE_URL:-https://csr.bu.edu/ftp/visda/2019/multi-source}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -n "${DOMAINNET_DOMAINS:-}" ]]; then
  # Space-separated subset for resumable parallel staging. The default remains
  # the complete official six-domain release.
  read -r -a domains <<< "${DOMAINNET_DOMAINS}"
else
  domains=(clipart infograph painting quickdraw real sketch)
fi

if [[ "${DATA_ROOT}" == "/path/to/datasets" ]]; then
  echo "Set DATA_ROOT to the dataset parent directory." >&2
  exit 1
fi

mkdir -p "${DOWNLOAD_DIR}" "${TARGET_DIR}/image_list"

download_file() {
  local url="$1"
  local destination="$2"
  if [[ -s "${destination}" ]]; then
    echo "Using existing ${destination}"
    return
  fi
  if command -v curl >/dev/null 2>&1; then
    curl --fail --location --retry 8 --retry-all-errors --continue-at - \
      --output "${destination}" "${url}"
  elif command -v wget >/dev/null 2>&1; then
    wget --continue --tries=8 --output-document="${destination}" "${url}"
  else
    echo "Neither curl nor wget is available." >&2
    exit 2
  fi
}

for domain in "${domains[@]}"; do
  case "${domain}" in
    clipart|infograph|painting|quickdraw|real|sketch) ;;
    *) echo "Unsupported DomainNet domain: ${domain}" >&2; exit 3 ;;
  esac
  archive="${DOWNLOAD_DIR}/${domain}.zip"
  if [[ "${domain}" == "clipart" || "${domain}" == "painting" ]]; then
    archive_url="${BASE_URL}/groundtruth/${domain}.zip"
  else
    archive_url="${BASE_URL}/${domain}.zip"
  fi
  download_file "${archive_url}" "${archive}"

  if [[ ! -d "${TARGET_DIR}/${domain}" ]]; then
    echo "Extracting ${archive}"
    unzip -q "${archive}" -d "${TARGET_DIR}"
  else
    echo "Domain directory already exists; not extracting again: ${TARGET_DIR}/${domain}"
  fi

  for split in train test; do
    download_file \
      "${BASE_URL}/domainnet/txt/${domain}_${split}.txt" \
      "${TARGET_DIR}/image_list/${domain}_${split}.txt"
  done

  if [[ "${KEEP_ARCHIVES}" == "0" ]]; then
    rm -f "${archive}"
  fi
done

if [[ "${VERIFY_LAYOUT}" == "1" ]]; then
  python "${REPO_ROOT}/scripts/datasets/verify_domainnet_layout.py" \
    --root "${DATA_ROOT}"
fi
