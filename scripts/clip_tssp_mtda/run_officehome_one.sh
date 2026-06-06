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
  echo "Usage: bash scripts/clip_tssp_mtda/run_officehome_one.sh <A|C|P|R> <SEED> <TSSP_METHOD> [--debug]" >&2
  exit 1
fi

SOURCE_CODE="$(echo "$1" | tr '[:lower:]' '[:upper:]')"
SEED="$2"
METHOD="$3"
DEBUG_FLAG="${4:-}"

DATA="${DATA:-${REPO_ROOT}/data}"
DATASET_CONFIG="${DATASET_CONFIG:-${REPO_ROOT}/configs/datasets/office_home_mtda.yaml}"
CFG="${CFG:-${REPO_ROOT}/configs/trainers/CLIPTSSPMTDA/vit_b16.yaml}"
BACKBONE="${BACKBONE:-}"
EXTRA_OPTS="${EXTRA_OPTS:-}"

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

TRAINER="CLIPTSSPMTDA"
case "${METHOD}" in
  clip_tssp_full|tssp_full|full)
    DEFAULT_TAG="clip_tssp_full"
    METHOD_OPTS=(
      "TRAINER.CLIP_TSSP_MTDA.USE_GAP_TOKEN" "True"
      "TRAINER.CLIP_TSSP_MTDA.STYLE_GROUP_SIZE" "1"
      "TRAINER.CLIP_TSSP_MTDA.GAP_GROUP_SIZE" "1"
      "TRAINER.CLIP_TSSP_MTDA.GAP_POSITION" "after_target"
    )
    ;;
  clip_tssp_no_gap|tssp_no_gap|no_gap)
    DEFAULT_TAG="clip_tssp_no_gap"
    METHOD_OPTS=(
      "TRAINER.CLIP_TSSP_MTDA.USE_GAP_TOKEN" "False"
      "TRAINER.CLIP_TSSP_MTDA.STYLE_GROUP_SIZE" "1"
      "TRAINER.CLIP_TSSP_MTDA.GAP_GROUP_SIZE" "1"
      "TRAINER.CLIP_TSSP_MTDA.GAP_POSITION" "after_target"
    )
    ;;
  clip_tssp_gap|tssp_gap|gap)
    DEFAULT_TAG="clip_tssp_gap"
    METHOD_OPTS=(
      "TRAINER.CLIP_TSSP_MTDA.USE_GAP_TOKEN" "True"
      "TRAINER.CLIP_TSSP_MTDA.STYLE_GROUP_SIZE" "1"
      "TRAINER.CLIP_TSSP_MTDA.GAP_GROUP_SIZE" "1"
      "TRAINER.CLIP_TSSP_MTDA.GAP_POSITION" "middle"
    )
    ;;
  clip_tssp_pair|tssp_pair|pair)
    DEFAULT_TAG="clip_tssp_pair"
    METHOD_OPTS=(
      "TRAINER.CLIP_TSSP_MTDA.USE_GAP_TOKEN" "False"
      "TRAINER.CLIP_TSSP_MTDA.STYLE_GROUP_SIZE" "2"
      "TRAINER.CLIP_TSSP_MTDA.GAP_GROUP_SIZE" "1"
      "TRAINER.CLIP_TSSP_MTDA.GAP_POSITION" "after_target"
    )
    ;;
  clip_tssp_pair_gap|tssp_pair_gap|pair_gap)
    DEFAULT_TAG="clip_tssp_pair_gap"
    METHOD_OPTS=(
      "TRAINER.CLIP_TSSP_MTDA.USE_GAP_TOKEN" "True"
      "TRAINER.CLIP_TSSP_MTDA.STYLE_GROUP_SIZE" "2"
      "TRAINER.CLIP_TSSP_MTDA.GAP_GROUP_SIZE" "2"
      "TRAINER.CLIP_TSSP_MTDA.GAP_POSITION" "middle"
    )
    ;;
  clip_tssp_style3_gap1|style3_gap1)
    DEFAULT_TAG="clip_tssp_style3_gap1"
    METHOD_OPTS=(
      "TRAINER.CLIP_TSSP_MTDA.USE_GAP_TOKEN" "True"
      "TRAINER.CLIP_TSSP_MTDA.STYLE_GROUP_SIZE" "3"
      "TRAINER.CLIP_TSSP_MTDA.GAP_GROUP_SIZE" "1"
      "TRAINER.CLIP_TSSP_MTDA.GAP_POSITION" "middle"
    )
    ;;
  clip_tssp_style4_gap1|style4_gap1)
    DEFAULT_TAG="clip_tssp_style4_gap1"
    METHOD_OPTS=(
      "TRAINER.CLIP_TSSP_MTDA.USE_GAP_TOKEN" "True"
      "TRAINER.CLIP_TSSP_MTDA.STYLE_GROUP_SIZE" "4"
      "TRAINER.CLIP_TSSP_MTDA.GAP_GROUP_SIZE" "1"
      "TRAINER.CLIP_TSSP_MTDA.GAP_POSITION" "middle"
    )
    ;;
  clip_tssp_style3_gap3|style3_gap3)
    DEFAULT_TAG="clip_tssp_style3_gap3"
    METHOD_OPTS=(
      "TRAINER.CLIP_TSSP_MTDA.USE_GAP_TOKEN" "True"
      "TRAINER.CLIP_TSSP_MTDA.STYLE_GROUP_SIZE" "3"
      "TRAINER.CLIP_TSSP_MTDA.GAP_GROUP_SIZE" "3"
      "TRAINER.CLIP_TSSP_MTDA.GAP_POSITION" "middle"
    )
    ;;
  clip_tssp_style4_gap4|style4_gap4)
    DEFAULT_TAG="clip_tssp_style4_gap4"
    METHOD_OPTS=(
      "TRAINER.CLIP_TSSP_MTDA.USE_GAP_TOKEN" "True"
      "TRAINER.CLIP_TSSP_MTDA.STYLE_GROUP_SIZE" "4"
      "TRAINER.CLIP_TSSP_MTDA.GAP_GROUP_SIZE" "4"
      "TRAINER.CLIP_TSSP_MTDA.GAP_POSITION" "middle"
    )
    ;;
  clip_tssp_style1_gap3|style1_gap3)
    DEFAULT_TAG="clip_tssp_style1_gap3"
    METHOD_OPTS=(
      "TRAINER.CLIP_TSSP_MTDA.USE_GAP_TOKEN" "True"
      "TRAINER.CLIP_TSSP_MTDA.STYLE_GROUP_SIZE" "1"
      "TRAINER.CLIP_TSSP_MTDA.GAP_GROUP_SIZE" "3"
      "TRAINER.CLIP_TSSP_MTDA.GAP_POSITION" "middle"
    )
    ;;
  clip_tssp_style1_gap4|style1_gap4)
    DEFAULT_TAG="clip_tssp_style1_gap4"
    METHOD_OPTS=(
      "TRAINER.CLIP_TSSP_MTDA.USE_GAP_TOKEN" "True"
      "TRAINER.CLIP_TSSP_MTDA.STYLE_GROUP_SIZE" "1"
      "TRAINER.CLIP_TSSP_MTDA.GAP_GROUP_SIZE" "4"
      "TRAINER.CLIP_TSSP_MTDA.GAP_POSITION" "middle"
    )
    ;;
  *)
    echo "Unknown method: ${METHOD}" >&2
    exit 4
    ;;
esac

TRAINER_DIR="${METHOD_TAG:-${DEFAULT_TAG}}"
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
CMD+=("${METHOD_OPTS[@]}")

if [ "${DEBUG_FLAG}" = "--debug" ]; then
  CMD+=(
    "TRAINER.CLIP_TSSP_MTDA.DEBUG.PRINT_ONCE" "True"
    "DATALOADER.NUM_WORKERS" "0"
    "DATALOADER.TRAIN_X.BATCH_SIZE" "2"
    "DATALOADER.TRAIN_U.BATCH_SIZE" "2"
    "DATALOADER.TEST.BATCH_SIZE" "16"
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
echo "Method tag: ${TRAINER_DIR}"
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
