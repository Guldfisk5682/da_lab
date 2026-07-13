#!/bin/bash

# Diagnostic upper bound: one independent prompt-tuned model per target domain.

set -euo pipefail

AUTO_REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
REPO_ROOT="${REPO_ROOT:-${AUTO_REPO_ROOT}}"
cd "${REPO_ROOT}"

if [ "$#" -ne 3 ]; then
  echo "Usage: bash $0 <SOURCE:A|C|P|R> <TARGET:A|C|P|R> <SEED>" >&2
  exit 1
fi

SOURCE_CODE="$(echo "$1" | tr '[:lower:]' '[:upper:]')"
TARGET_CODE="$(echo "$2" | tr '[:lower:]' '[:upper:]')"
SEED="$3"
if [ "${SOURCE_CODE}" = "${TARGET_CODE}" ]; then
  echo "Source and target must differ" >&2
  exit 2
fi

domain_name() {
  case "$1" in
    A) echo art ;;
    C) echo clipart ;;
    P) echo product ;;
    R) echo real_world ;;
    *) echo "Unknown domain code: $1" >&2; exit 3 ;;
  esac
}

SOURCE_DOMAIN="$(domain_name "${SOURCE_CODE}")"
TARGET_DOMAIN="$(domain_name "${TARGET_CODE}")"
DATA="${DATA:-${REPO_ROOT}/data}"
DATASET_CONFIG="${DATASET_CONFIG:-${REPO_ROOT}/configs/datasets/office_home_mtda.yaml}"
CFG="${CFG:-${REPO_ROOT}/configs/trainers/ContinuousSharedProjMaPLeMTDA/vit_b16.yaml}"
METHOD_TAG="${METHOD_TAG:-maple_cshared_pl03_independent_stda}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/officehome_stda/${METHOD_TAG}/${SOURCE_CODE}2${TARGET_CODE}/seed${SEED}}"
EXTRA_OPTS="${EXTRA_OPTS:-}"
DRY_RUN="${DRY_RUN:-0}"

OPTS=(
  --root "${DATA}"
  --seed "${SEED}"
  --trainer ContinuousSharedProjMaPLeMTDA
  --dataset-config-file "${DATASET_CONFIG}"
  --config-file "${CFG}"
  --source-domains "${SOURCE_DOMAIN}"
  --target-domains "${TARGET_DOMAIN}"
  --output-dir "${OUTPUT_DIR}"
)
CFG_OPTS=(
  TRAINER.MAPLE_MTDA.LAMBDA_PL 0.3
  TRAINER.MAPLE_MTDA.LAMBDA_PL_FINAL 0.3
  TRAINER.MAPLE_MTDA.PL_THRESHOLD 0.7
  TRAINER.MAPLE_MTDA.PL_STUDENT_THRESHOLD 0.7
  TRAINER.MAPLE_MTDA.PL_USE_STUDENT_LOW_CONF_MASK True
)
if [ -n "${EXTRA_OPTS}" ]; then
  # shellcheck disable=SC2206
  EXTRA_ARRAY=(${EXTRA_OPTS})
  CFG_OPTS+=("${EXTRA_ARRAY[@]}")
fi

echo "Independent ContinuousSharedProjMaPLe STDA diagnostic"
echo "  task: ${SOURCE_CODE}2${TARGET_CODE} (${SOURCE_DOMAIN} -> ${TARGET_DOMAIN})"
echo "  seed: ${SEED}"
echo "  output: ${OUTPUT_DIR}"

if [ "${DRY_RUN}" = "1" ]; then
  printf 'python train.py'
  printf ' %q' "${OPTS[@]}"
  printf ' %q' "${CFG_OPTS[@]}"
  printf '\n'
  exit 0
fi

python scripts/experiment_guard.py \
  --output-dir "${OUTPUT_DIR}" \
  --method-tag "${METHOD_TAG}" \
  --source "${SOURCE_CODE}" \
  --targets "${TARGET_CODE}" \
  --seed "${SEED}" \
  --trainer ContinuousSharedProjMaPLeMTDA \
  --trainer-config "${CFG}" \
  --dataset-config "${DATASET_CONFIG}" \
  --data "${DATA}" \
  --extra-opts "${EXTRA_OPTS}" \
  --effective-opts "${CFG_OPTS[*]}"
python train.py "${OPTS[@]}" "${CFG_OPTS[@]}"
