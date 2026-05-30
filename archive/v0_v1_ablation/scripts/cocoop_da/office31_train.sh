#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/../.."

DATA="${DATA:-/path/to/datasets}"
TRAINER="${TRAINER:-CoCoOpDAV1}"
DATASET_CONFIG="${DATASET_CONFIG:-configs/datasets/office31.yaml}"
CFG="${CFG:-configs/trainers/CoCoOpDA/vit_b16_v1.yaml}"
SOURCE_DOMAIN="${SOURCE_DOMAIN:-amazon}"
TARGET_DOMAIN="${TARGET_DOMAIN:-webcam}"
SEED="${SEED:-1}"
STAGE="${STAGE:-1}"
TRAINER_DIR="${TRAINER_DIR:-${TRAINER}}"
FORCE_ALPHA="${FORCE_ALPHA:--1.0}"
BACKBONE="${BACKBONE:-}"

if [ "${DATA}" = "/path/to/datasets" ]; then
  echo "Please set DATA to a real dataset root before training." >&2
  exit 1
fi

if [ ! -d "${DATA}" ]; then
  echo "Dataset root does not exist: ${DATA}" >&2
  exit 2
fi

PROMPT_TRAIN="False"
if [ "${STAGE}" = "2" ]; then
  PROMPT_TRAIN="True"
fi

TASK_TAG="$(echo "${SOURCE_DOMAIN}" | cut -c1 | tr '[:lower:]' '[:upper:]')2$(echo "${TARGET_DOMAIN}" | cut -c1 | tr '[:lower:]' '[:upper:]')"
OUTPUT_DIR="${OUTPUT_DIR:-output/office31/${TRAINER_DIR}/${TASK_TAG}/seed${SEED}/stage${STAGE}}"

CMD=(
  python train.py
  --root "${DATA}"
  --seed "${SEED}"
  --trainer "${TRAINER}"
  --dataset-config-file "${DATASET_CONFIG}"
  --config-file "${CFG}"
  --output-dir "${OUTPUT_DIR}"
  --source-domains "${SOURCE_DOMAIN}"
  --target-domains "${TARGET_DOMAIN}"
)

if [ -n "${BACKBONE}" ]; then
  CMD+=(--backbone "${BACKBONE}")
fi

CMD+=(
  --
  TRAINER.COCOOP_DA.TRAIN.STAGE "${STAGE}"
  TRAINER.COCOOP_DA.TRAIN.TRAIN_PROMPT_LEARNER "${PROMPT_TRAIN}"
  TRAINER.COCOOP_DA.GATE.FORCE_ALPHA "${FORCE_ALPHA}"
)

"${CMD[@]}"
