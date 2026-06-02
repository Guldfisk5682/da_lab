#!/bin/bash

set -euo pipefail

DEFAULT_REPO_ROOT="/workspace/txc/da_lab"
AUTO_REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
if [ -d "${DEFAULT_REPO_ROOT}" ]; then
  REPO_ROOT="${REPO_ROOT:-${DEFAULT_REPO_ROOT}}"
else
  REPO_ROOT="${REPO_ROOT:-${AUTO_REPO_ROOT}}"
fi

cd "${REPO_ROOT}"

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
  echo "Usage: bash scripts/style_prompt_mtda/run_officehome_one.sh <A|C|P|R> <SEED> <cocoop_mt|style_prompt> [--debug]" >&2
  exit 1
fi

SOURCE_CODE="$(echo "$1" | tr '[:lower:]' '[:upper:]')"
SEED="$2"
METHOD="$3"
DEBUG_FLAG="${4:-}"

DATA="${DATA:-${REPO_ROOT}/data}"
DATASET_CONFIG="${DATASET_CONFIG:-${REPO_ROOT}/configs/datasets/office_home_mtda.yaml}"
BACKBONE="${BACKBONE:-}"
EXTRA_OPTS="${EXTRA_OPTS:-}"

# Make CUDA logical numbering follow nvidia-smi ordering on shared servers.
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

resolve_source_domain() {
  case "$1" in
    A) echo "art" ;;
    C) echo "clipart" ;;
    P) echo "product" ;;
    R) echo "real_world" ;;
    *)
      echo "Unknown source code: $1" >&2
      exit 3
      ;;
  esac
}

SOURCE_DOMAIN="$(resolve_source_domain "${SOURCE_CODE}")"

case "${SOURCE_CODE}" in
  A)
    TARGET_CODES=(C P R)
    TARGET_DOMAINS=(clipart product real_world)
    ;;
  C)
    TARGET_CODES=(A P R)
    TARGET_DOMAINS=(art product real_world)
    ;;
  P)
    TARGET_CODES=(A C R)
    TARGET_DOMAINS=(art clipart real_world)
    ;;
  R)
    TARGET_CODES=(A C P)
    TARGET_DOMAINS=(art clipart product)
    ;;
esac

case "${METHOD}" in
  cocoop_mt|cocoop)
    TRAINER="CoCoOpMTDA"
    TRAINER_DIR="cocoop_mt"
    CFG="${REPO_ROOT}/configs/trainers/CoCoOpMTDA/vit_b16.yaml"
    DEBUG_OPT_KEY="TRAINER.COCOOP_MTDA.DEBUG.PRINT_ONCE"
    ;;
  style_prompt|style_prompt_mtda)
    TRAINER="StylePromptMTDA"
    TRAINER_DIR="style_prompt"
    CFG="${REPO_ROOT}/configs/trainers/StylePromptMTDA/vit_b16.yaml"
    DEBUG_OPT_KEY="TRAINER.STYLE_PROMPT_MTDA.DEBUG.PRINT_ONCE"
    ;;
  *)
    echo "Unknown method: ${METHOD}" >&2
    exit 4
    ;;
esac

TASK_TAG="${SOURCE_CODE}2${TARGET_CODES[0]}${TARGET_CODES[1]}${TARGET_CODES[2]}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/office_home_mtda/${TRAINER_DIR}/${TASK_TAG}/seed${SEED}}"

CMD=(
  python "${REPO_ROOT}/train.py"
  --root "${DATA}"
  --seed "${SEED}"
  --trainer "${TRAINER}"
  --dataset-config-file "${DATASET_CONFIG}"
  --config-file "${CFG}"
  --output-dir "${OUTPUT_DIR}"
  --source-domains "${SOURCE_DOMAIN}"
  --target-domains "${TARGET_DOMAINS[@]}"
)

if [ -n "${BACKBONE}" ]; then
  CMD+=(--backbone "${BACKBONE}")
fi

CMD+=(--)

if [ "${DEBUG_FLAG}" = "--debug" ]; then
  CMD+=(
    "${DEBUG_OPT_KEY}" "True"
    "DATALOADER.NUM_WORKERS" "0"
    "DATALOADER.TRAIN_X.BATCH_SIZE" "2"
    "DATALOADER.TRAIN_U.BATCH_SIZE" "2"
    "OPTIM.MAX_EPOCH" "1"
    "TRAIN.PRINT_FREQ" "1"
  )
fi

if [ -n "${EXTRA_OPTS}" ]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=(${EXTRA_OPTS})
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "==============================================="
echo "Method: ${METHOD}"
echo "Trainer: ${TRAINER}"
echo "Source domain: ${SOURCE_DOMAIN}"
echo "Target domains: ${TARGET_DOMAINS[*]}"
echo "Seed: ${SEED}"
echo "Output dir: ${OUTPUT_DIR}"
echo "CUDA_DEVICE_ORDER: ${CUDA_DEVICE_ORDER}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
if [ "${DEBUG_FLAG}" = "--debug" ]; then
  echo "Debug mode: enabled"
fi
echo "==============================================="

"${CMD[@]}"
