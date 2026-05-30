#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/../.."

DATA="${DATA:-/path/to/datasets}"
TRAINER="${TRAINER:-CoCoOpDAV1}"
DATASET_CONFIG="${DATASET_CONFIG:-configs/datasets/office_home.yaml}"
CFG="${CFG:-configs/trainers/CoCoOpDA/vit_b16_v1.yaml}"
SOURCE_DOMAIN="${SOURCE_DOMAIN:-art}"
TARGET_DOMAIN="${TARGET_DOMAIN:-clipart}"
SEED="${SEED:-1}"
STAGE="${STAGE:-1}"
TRAINER_DIR="${TRAINER_DIR:-${TRAINER}}"
FORCE_ALPHA="${FORCE_ALPHA:--1.0}"

PROMPT_TRAIN="false"
if [ "${STAGE}" = "2" ]; then
  PROMPT_TRAIN="true"
fi

TASK_TAG="$(echo "${SOURCE_DOMAIN}" | cut -c1 | tr '[:lower:]' '[:upper:]')2$(echo "${TARGET_DOMAIN}" | cut -c1 | tr '[:lower:]' '[:upper:]')"
OUTPUT_DIR="${OUTPUT_DIR:-output/office_home/${TRAINER_DIR}/${TASK_TAG}/seed${SEED}/stage${STAGE}}"

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
  TRAINER.COCOOP_DA.TRAIN.TRAIN_PROMPT_LEARNER "${PROMPT_TRAIN}" \
  TRAINER.COCOOP_DA.GATE.FORCE_ALPHA "${FORCE_ALPHA}"
