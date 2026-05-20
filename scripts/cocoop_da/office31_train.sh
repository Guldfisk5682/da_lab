#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/../.."

DATA="${DATA:-/path/to/datasets}"
TRAINER="${TRAINER:-CoCoOpDAV0}"
DATASET_CONFIG="${DATASET_CONFIG:-configs/datasets/office31.yaml}"
CFG="${CFG:-configs/trainers/CoCoOpDA/vit_b16_v0.yaml}"
SOURCE_DOMAIN="${SOURCE_DOMAIN:-amazon}"
TARGET_DOMAIN="${TARGET_DOMAIN:-webcam}"
SEED="${SEED:-1}"
STAGE="${STAGE:-1}"

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
OUTPUT_DIR="${OUTPUT_DIR:-output/office31/cocoop_da_v0/${TASK_TAG}/seed${SEED}/stage${STAGE}}"

python train.py \
  --root "${DATA}" \
  --seed "${SEED}" \
  --trainer "${TRAINER}" \
  --dataset-config-file "${DATASET_CONFIG}" \
  --config-file "${CFG}" \
  --output-dir "${OUTPUT_DIR}" \
  --source-domains "${SOURCE_DOMAIN}" \
  --target-domains "${TARGET_DOMAIN}" \
  -- \
  TRAINER.COCOOP_DA.TRAIN.STAGE "${STAGE}" \
  TRAINER.COCOOP_DA.TRAIN.TRAIN_PROMPT_LEARNER "${PROMPT_TRAIN}"
