#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/../.."

DATA="${DATA:-/path/to/datasets}"
TRAINER="${TRAINER:-GSPALegacy}"
DATASET_CONFIG="${DATASET_CONFIG:-configs/datasets/office31.yaml}"
CFG="${CFG:-configs/trainers/GSPA_LEGACY/vit_b16.yaml}"
SOURCE_DOMAIN="${SOURCE_DOMAIN:-amazon}"
TARGET_DOMAIN="${TARGET_DOMAIN:-webcam}"
SEED="${SEED:-1}"
TRAINER_DIR="${TRAINER_DIR:-${TRAINER}}"
TASK_TAG="$(echo "${SOURCE_DOMAIN}" | cut -c1 | tr '[:lower:]' '[:upper:]')2$(echo "${TARGET_DOMAIN}" | cut -c1 | tr '[:lower:]' '[:upper:]')"
MODEL_DIR="${MODEL_DIR:-output/office31/${TRAINER_DIR}/${TASK_TAG}/seed${SEED}}"
LOAD_EPOCH="${LOAD_EPOCH:-}"
BACKBONE="${BACKBONE:-}"
EXTRA_OPTS="${EXTRA_OPTS:-}"

if [ "${DATA}" = "/path/to/datasets" ]; then
  echo "Please set DATA to a real dataset root before evaluation." >&2
  exit 1
fi

if [ ! -d "${DATA}" ]; then
  echo "Dataset root does not exist: ${DATA}" >&2
  exit 2
fi

if [ ! -d "${MODEL_DIR}" ]; then
  echo "Model directory does not exist: ${MODEL_DIR}" >&2
  exit 3
fi

if [ -z "${LOAD_EPOCH}" ]; then
  BEST_MODEL_COUNT="$(find "${MODEL_DIR}" -type f -name 'model-best.pth.tar' | wc -l | tr -d ' ')"
  if [ "${BEST_MODEL_COUNT}" = "0" ]; then
    LATEST_CHECKPOINT="$(find "${MODEL_DIR}" -type f -name 'model.pth.tar-*' | sort -V | tail -n 1)"
    if [ -n "${LATEST_CHECKPOINT}" ]; then
      LOAD_EPOCH="${LATEST_CHECKPOINT##*-}"
      echo "No model-best checkpoint found under ${MODEL_DIR}"
      echo "Falling back to the latest epoch checkpoint: ${LOAD_EPOCH}"
    fi
  fi
fi

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
)

if [ -n "${LOAD_EPOCH}" ]; then
  CMD+=(--load-epoch "${LOAD_EPOCH}")
fi

if [ -n "${BACKBONE}" ]; then
  CMD+=(--backbone "${BACKBONE}")
fi

if [ -n "${EXTRA_OPTS}" ]; then
  CMD+=(-- )
  # shellcheck disable=SC2206
  EXTRA_ARGS=(${EXTRA_OPTS})
  CMD+=("${EXTRA_ARGS[@]}")
fi

"${CMD[@]}"
