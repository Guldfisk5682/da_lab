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
MODEL_DIR="${MODEL_DIR:-output/office31/cocoop_da_v0/A2W/seed1/stage1}"
LOAD_EPOCH="${LOAD_EPOCH:-}"

CMD=(
  python train.py
  --root "${DATA}"
  --seed "${SEED}"
  --trainer "${TRAINER}"
  --dataset-config-file "${DATASET_CONFIG}"
  --config-file "${CFG}"
  --source-domains "${SOURCE_DOMAIN}"
  --target-domains "${TARGET_DOMAIN}"
  --eval-only
  --model-dir "${MODEL_DIR}"
  TRAINER.COCOOP_DA.TRAIN.STAGE "${STAGE}"
)

if [ -n "${LOAD_EPOCH}" ]; then
  CMD+=(--load-epoch "${LOAD_EPOCH}")
fi

"${CMD[@]}"
