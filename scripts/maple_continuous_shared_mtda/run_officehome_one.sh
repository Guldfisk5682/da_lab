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

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "Usage: bash scripts/maple_continuous_shared_mtda/run_officehome_one.sh <A|C|P|R> <SEED> [--debug]" >&2
  exit 1
fi

SOURCE_CODE="$(echo "$1" | tr '[:lower:]' '[:upper:]')"
SEED="$2"
DEBUG_FLAG="${3:-}"

DATA="${DATA:-${REPO_ROOT}/data}"
DATASET_CONFIG="${DATASET_CONFIG:-${REPO_ROOT}/configs/datasets/office_home_mtda.yaml}"
CFG="${CFG:-${REPO_ROOT}/configs/trainers/ContinuousSharedProjMaPLeMTDA/vit_b16.yaml}"
EXTRA_OPTS="${EXTRA_OPTS:-}"
METHOD_TAG="${METHOD_TAG:-maple_continuous_shared_mtda}"
DRY_RUN="${DRY_RUN:-0}"
EVAL_EVERY_EPOCH="${EVAL_EVERY_EPOCH:-False}"
POST_INIT_METHOD_TAG="${POST_INIT_METHOD_TAG:-}"
POST_INIT_LOAD_EPOCH="${POST_INIT_LOAD_EPOCH:-5}"

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

TARGET_TAG="$(IFS=''; echo "${TARGET_CODES[*]}")"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/officehome_mtda/${METHOD_TAG}/${SOURCE_CODE}2${TARGET_TAG}/seed${SEED}}"

OPTS=(
  "--root" "${DATA}"
  "--seed" "${SEED}"
  "--trainer" "ContinuousSharedProjMaPLeMTDA"
  "--dataset-config-file" "${DATASET_CONFIG}"
  "--config-file" "${CFG}"
  "--source-domains" "${SOURCE_DOMAIN}"
  "--target-domains" "${TARGET_DOMAINS[@]}"
  "--output-dir" "${OUTPUT_DIR}"
)

CFG_OPTS=(
  "TEST.EVAL_EVERY_EPOCH" "${EVAL_EVERY_EPOCH}"
)

POST_INIT_DIR=""
if [ -n "${POST_INIT_METHOD_TAG}" ]; then
  POST_INIT_DIR="${REPO_ROOT}/output/officehome_mtda/${POST_INIT_METHOD_TAG}/${SOURCE_CODE}2${TARGET_TAG}/seed${SEED}"
  CFG_OPTS+=(
    "TRAINER.MAPLE_MTDA.POST_INIT.ENABLED" "True"
    "TRAINER.MAPLE_MTDA.POST_INIT.MODEL_DIR" "${POST_INIT_DIR}"
    "TRAINER.MAPLE_MTDA.POST_INIT.LOAD_EPOCH" "${POST_INIT_LOAD_EPOCH}"
  )
fi

if [ "${DEBUG_FLAG}" = "--debug" ]; then
  CFG_OPTS+=(
    "OPTIM.MAX_EPOCH" "1"
    "TRAIN.PRINT_FREQ" "1"
    "TRAINER.MAPLE_MTDA.DEBUG.PRINT_ONCE" "True"
  )
fi

if [ -n "${EXTRA_OPTS}" ]; then
  # shellcheck disable=SC2206
  EXTRA_ARRAY=(${EXTRA_OPTS})
  CFG_OPTS+=("${EXTRA_ARRAY[@]}")
fi

echo "ContinuousSharedProjMaPLeMTDA Office-Home SS-MTDA"
echo "  repo: ${REPO_ROOT}"
echo "  source: ${SOURCE_CODE} (${SOURCE_DOMAIN})"
echo "  targets: ${TARGET_CODES[*]} (${TARGET_DOMAINS[*]})"
echo "  method tag: ${METHOD_TAG}"
echo "  seed: ${SEED}"
echo "  output: ${OUTPUT_DIR}"
echo "  cfg: ${CFG}"
if [ -n "${POST_INIT_DIR}" ]; then
  echo "  post init: ${POST_INIT_DIR} (epoch ${POST_INIT_LOAD_EPOCH})"
fi

if [ "${DRY_RUN}" = "1" ]; then
  printf 'python train.py'
  printf ' %q' "${OPTS[@]}"
  printf ' %q' "${CFG_OPTS[@]}"
  printf '\n'
  exit 0
fi

python train.py "${OPTS[@]}" "${CFG_OPTS[@]}"
