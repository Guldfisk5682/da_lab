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

if [ "$#" -ne 2 ]; then
  echo "Usage: DOMAIN_ORDER='clipart product real_world' bash $0 <A|C|P|R> <SEED>" >&2
  exit 1
fi

SOURCE_CODE="$(echo "$1" | tr '[:lower:]' '[:upper:]')"
SEED="$2"
DOMAIN_ORDER="${DOMAIN_ORDER:-}"
if [ -z "${DOMAIN_ORDER}" ]; then
  echo "DOMAIN_ORDER must list the three target domains from easy to hard (or its reverse)." >&2
  exit 2
fi
read -r -a ORDER_ARRAY <<< "${DOMAIN_ORDER}"
if [ "${#ORDER_ARRAY[@]}" -ne 3 ]; then
  echo "DOMAIN_ORDER must contain exactly three domains, got: ${DOMAIN_ORDER}" >&2
  exit 2
fi
ORDER_CFG="['${ORDER_ARRAY[0]}','${ORDER_ARRAY[1]}','${ORDER_ARRAY[2]}']"

DATA="${DATA:-${REPO_ROOT}/data}"
DATASET_CONFIG="${DATASET_CONFIG:-${REPO_ROOT}/configs/datasets/office_home_mtda.yaml}"
CFG="${CFG:-${REPO_ROOT}/configs/trainers/CurriculumContinuousSharedProjMaPLeMTDA/vit_b16.yaml}"
METHOD_TAG="${METHOD_TAG:-maple_curriculum_mtda}"
REPLAY_ENABLED="${REPLAY_ENABLED:-False}"
TOPK_PER_CLASS="${TOPK_PER_CLASS:-8}"
REPLAY_LAMBDA="${REPLAY_LAMBDA:-1.0}"
RESET_OPTIM_PER_STAGE="${RESET_OPTIM_PER_STAGE:-False}"
STAGE_VIRTUAL_EPOCHS="${STAGE_VIRTUAL_EPOCHS:-5}"
EXTRA_OPTS="${EXTRA_OPTS:-}"
DRY_RUN="${DRY_RUN:-0}"

case "${SOURCE_CODE}" in
  A) SOURCE_DOMAIN=art; TARGET_CODES=(C P R); TARGET_DOMAINS=(clipart product real_world) ;;
  C) SOURCE_DOMAIN=clipart; TARGET_CODES=(A P R); TARGET_DOMAINS=(art product real_world) ;;
  P) SOURCE_DOMAIN=product; TARGET_CODES=(A C R); TARGET_DOMAINS=(art clipart real_world) ;;
  R) SOURCE_DOMAIN=real_world; TARGET_CODES=(A C P); TARGET_DOMAINS=(art clipart product) ;;
  *) echo "Unknown source code: ${SOURCE_CODE}" >&2; exit 3 ;;
esac

TARGET_TAG="$(IFS=''; echo "${TARGET_CODES[*]}")"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/officehome_mtda/${METHOD_TAG}/${SOURCE_CODE}2${TARGET_TAG}/seed${SEED}}"

OPTS=(
  --root "${DATA}"
  --seed "${SEED}"
  --trainer CurriculumContinuousSharedProjMaPLeMTDA
  --dataset-config-file "${DATASET_CONFIG}"
  --config-file "${CFG}"
  --source-domains "${SOURCE_DOMAIN}"
  --target-domains "${TARGET_DOMAINS[@]}"
  --output-dir "${OUTPUT_DIR}"
)
CFG_OPTS=(
  TRAINER.MAPLE_MTDA.CURRICULUM.DOMAIN_ORDER "${ORDER_CFG}"
  TRAINER.MAPLE_MTDA.CURRICULUM.REPLAY.ENABLED "${REPLAY_ENABLED}"
  TRAINER.MAPLE_MTDA.CURRICULUM.REPLAY.TOPK_PER_CLASS "${TOPK_PER_CLASS}"
  TRAINER.MAPLE_MTDA.CURRICULUM.REPLAY.LAMBDA "${REPLAY_LAMBDA}"
  TRAINER.MAPLE_MTDA.CURRICULUM.RESET_OPTIM_PER_STAGE "${RESET_OPTIM_PER_STAGE}"
  TRAINER.MAPLE_MTDA.CURRICULUM.STAGE_VIRTUAL_EPOCHS "${STAGE_VIRTUAL_EPOCHS}"
)
if [ -n "${EXTRA_OPTS}" ]; then
  # shellcheck disable=SC2206
  EXTRA_ARRAY=(${EXTRA_OPTS})
  CFG_OPTS+=("${EXTRA_ARRAY[@]}")
fi

echo "Curriculum ContinuousSharedProjMaPLeMTDA"
echo "  source: ${SOURCE_CODE} (${SOURCE_DOMAIN})"
echo "  targets: ${TARGET_DOMAINS[*]}"
echo "  order: ${ORDER_ARRAY[*]}"
echo "  replay: ${REPLAY_ENABLED}, top-k/class: ${TOPK_PER_CLASS}, lambda: ${REPLAY_LAMBDA}"
echo "  reset optimizer/scheduler per stage: ${RESET_OPTIM_PER_STAGE}"
if [ "${RESET_OPTIM_PER_STAGE}" = "True" ] || [ "${RESET_OPTIM_PER_STAGE}" = "true" ]; then
  echo "  stage-local virtual scheduler epochs: ${STAGE_VIRTUAL_EPOCHS}"
fi
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
  --targets "${TARGET_CODES[@]}" \
  --seed "${SEED}" \
  --trainer CurriculumContinuousSharedProjMaPLeMTDA \
  --trainer-config "${CFG}" \
  --dataset-config "${DATASET_CONFIG}" \
  --data "${DATA}" \
  --extra-opts "${EXTRA_OPTS}" \
  --effective-opts "${CFG_OPTS[*]}"
python train.py "${OPTS[@]}" "${CFG_OPTS[@]}"
