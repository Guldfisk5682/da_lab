#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/../.."

DATA="${DATA:-/path/to/datasets}"
TRAINER="${TRAINER:-GSPALegacy}"
DATASET_CONFIG="${DATASET_CONFIG:-configs/datasets/office31.yaml}"
CFG="${CFG:-configs/trainers/GSPA_LEGACY/vit_b16.yaml}"
SEED="${SEED:-1}"
TRAINER_DIR="${TRAINER_DIR:-${TRAINER}}"
RESULTS_DIR="${RESULTS_DIR:-results/office31/${TRAINER_DIR}}"
SUMMARY_PATH="${SUMMARY_PATH:-${RESULTS_DIR}/seed${SEED}_summary.md}"
BACKBONE="${BACKBONE:-}"

if [ "${DATA}" = "/path/to/datasets" ]; then
  echo "Please set DATA to a real dataset root before training." >&2
  exit 1
fi

if [ ! -d "${DATA}" ]; then
  echo "Dataset root does not exist: ${DATA}" >&2
  exit 2
fi

TASKS=(
  "amazon dslr"
  "amazon webcam"
  "dslr amazon"
  "dslr webcam"
  "webcam amazon"
  "webcam dslr"
)

for task in "${TASKS[@]}"; do
  set -- ${task}
  SOURCE_DOMAIN="$1"
  TARGET_DOMAIN="$2"

  echo "==============================================="
  echo "Running Office-31: ${SOURCE_DOMAIN} -> ${TARGET_DOMAIN}"
  echo "Seed: ${SEED} | Trainer: ${TRAINER}"
  echo "==============================================="

  DATA="${DATA}" \
  TRAINER="${TRAINER}" \
  TRAINER_DIR="${TRAINER_DIR}" \
  DATASET_CONFIG="${DATASET_CONFIG}" \
  CFG="${CFG}" \
  BACKBONE="${BACKBONE}" \
  SOURCE_DOMAIN="${SOURCE_DOMAIN}" \
  TARGET_DOMAIN="${TARGET_DOMAIN}" \
  SEED="${SEED}" \
    bash scripts/gspa_legacy/office31_train.sh

  echo "Evaluating Office-31: ${SOURCE_DOMAIN} -> ${TARGET_DOMAIN}"

  DATA="${DATA}" \
  TRAINER="${TRAINER}" \
  TRAINER_DIR="${TRAINER_DIR}" \
  DATASET_CONFIG="${DATASET_CONFIG}" \
  CFG="${CFG}" \
  BACKBONE="${BACKBONE}" \
  SOURCE_DOMAIN="${SOURCE_DOMAIN}" \
  TARGET_DOMAIN="${TARGET_DOMAIN}" \
  SEED="${SEED}" \
    bash scripts/gspa_legacy/office31_eval.sh
done

mkdir -p "${RESULTS_DIR}"

python scripts/parse_office31_results.py \
  --root output/office31 \
  --trainer-dir "${TRAINER_DIR}" \
  --pattern "*/seed${SEED}/log.txt" \
  --markdown \
  --output "${SUMMARY_PATH}"

echo "==============================================="
echo "Office-31 summary saved to: ${SUMMARY_PATH}"
echo "==============================================="
