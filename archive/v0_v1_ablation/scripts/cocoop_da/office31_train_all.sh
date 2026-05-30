#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/../.."

DATA="${DATA:-/path/to/datasets}"
TRAINER="${TRAINER:-CoCoOpDAV1}"
DATASET_CONFIG="${DATASET_CONFIG:-configs/datasets/office31.yaml}"
CFG="${CFG:-configs/trainers/CoCoOpDA/vit_b16_v1.yaml}"
SEED="${SEED:-1}"
STAGE="${STAGE:-1}"
TRAINER_DIR="${TRAINER_DIR:-${TRAINER}}"
FORCE_ALPHA="${FORCE_ALPHA:--1.0}"
RESULTS_DIR="${RESULTS_DIR:-results/office31/${TRAINER_DIR}}"
SUMMARY_PATH="${SUMMARY_PATH:-${RESULTS_DIR}/seed${SEED}_stage${STAGE}_summary.md}"

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
  echo "Stage: ${STAGE} | Seed: ${SEED} | Trainer: ${TRAINER}"
  echo "==============================================="

  DATA="${DATA}" \
  TRAINER="${TRAINER}" \
  TRAINER_DIR="${TRAINER_DIR}" \
  DATASET_CONFIG="${DATASET_CONFIG}" \
  CFG="${CFG}" \
  SOURCE_DOMAIN="${SOURCE_DOMAIN}" \
  TARGET_DOMAIN="${TARGET_DOMAIN}" \
  SEED="${SEED}" \
  STAGE="${STAGE}" \
  FORCE_ALPHA="${FORCE_ALPHA}" \
    bash scripts/cocoop_da/office31_train.sh

  echo "Evaluating Office-31: ${SOURCE_DOMAIN} -> ${TARGET_DOMAIN}"

  DATA="${DATA}" \
  TRAINER="${TRAINER}" \
  TRAINER_DIR="${TRAINER_DIR}" \
  DATASET_CONFIG="${DATASET_CONFIG}" \
  CFG="${CFG}" \
  SOURCE_DOMAIN="${SOURCE_DOMAIN}" \
  TARGET_DOMAIN="${TARGET_DOMAIN}" \
  SEED="${SEED}" \
  STAGE="${STAGE}" \
  FORCE_ALPHA="${FORCE_ALPHA}" \
    bash scripts/cocoop_da/office31_eval.sh

done

mkdir -p "${RESULTS_DIR}"

python scripts/parse_office31_results.py \
  --root output/office31 \
  --trainer-dir "${TRAINER_DIR}" \
  --pattern "*/seed${SEED}/stage${STAGE}/log.txt" \
  --markdown \
  --output "${SUMMARY_PATH}"

echo "==============================================="
echo "Office-31 summary saved to: ${SUMMARY_PATH}"
echo "==============================================="
