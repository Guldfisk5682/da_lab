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
DRY_RUN="${DRY_RUN:-0}"

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
      "TRAINER.CLIP_TSSP_MTDA.USE_IMAGE_TOKENS" "False"
      "TRAINER.CLIP_TSSP_MTDA.LAMBDA_EM" "0.0"
      "TRAINER.CLIP_TSSP_MTDA.GAP_POSITION" "middle"
    )
    ;;
  clip_tssp_pair_gap_adamw2e3|pair_gap_adamw2e3|adamw2e3|o0)
    DEFAULT_TAG="clip_tssp_pair_gap_adamw2e3"
    METHOD_OPTS=(
      "TRAINER.CLIP_TSSP_MTDA.USE_GAP_TOKEN" "True"
      "TRAINER.CLIP_TSSP_MTDA.STYLE_GROUP_SIZE" "2"
      "TRAINER.CLIP_TSSP_MTDA.GAP_GROUP_SIZE" "2"
      "TRAINER.CLIP_TSSP_MTDA.USE_IMAGE_TOKENS" "False"
      "TRAINER.CLIP_TSSP_MTDA.LAMBDA_EM" "0.0"
      "TRAINER.CLIP_TSSP_MTDA.GAP_POSITION" "middle"
      "OPTIM.NAME" "adamw"
      "OPTIM.LR" "0.002"
      "OPTIM.WEIGHT_DECAY" "0.0001"
    )
    ;;
  clip_tssp_pair_gap_adamw1e4|pair_gap_adamw1e4|adamw1e4|o1)
    DEFAULT_TAG="clip_tssp_pair_gap_adamw1e4"
    METHOD_OPTS=(
      "TRAINER.CLIP_TSSP_MTDA.USE_GAP_TOKEN" "True"
      "TRAINER.CLIP_TSSP_MTDA.STYLE_GROUP_SIZE" "2"
      "TRAINER.CLIP_TSSP_MTDA.GAP_GROUP_SIZE" "2"
      "TRAINER.CLIP_TSSP_MTDA.USE_IMAGE_TOKENS" "False"
      "TRAINER.CLIP_TSSP_MTDA.LAMBDA_EM" "0.0"
      "TRAINER.CLIP_TSSP_MTDA.GAP_POSITION" "middle"
      "OPTIM.NAME" "adamw"
      "OPTIM.LR" "0.0001"
      "OPTIM.WEIGHT_DECAY" "0.0001"
    )
    ;;
  clip_tssp_pair_gap_kl|pair_gap_kl|kl|k0)
    DEFAULT_TAG="clip_tssp_pair_gap_kl"
    METHOD_OPTS=(
      "TRAINER.CLIP_TSSP_MTDA.USE_GAP_TOKEN" "True"
      "TRAINER.CLIP_TSSP_MTDA.STYLE_GROUP_SIZE" "2"
      "TRAINER.CLIP_TSSP_MTDA.GAP_GROUP_SIZE" "2"
      "TRAINER.CLIP_TSSP_MTDA.USE_IMAGE_TOKENS" "False"
      "TRAINER.CLIP_TSSP_MTDA.LAMBDA_EM" "0.0"
      "TRAINER.CLIP_TSSP_MTDA.LAMBDA_KL" "0.05"
      "TRAINER.CLIP_TSSP_MTDA.KL_TEMPERATURE" "1.0"
      "TRAINER.CLIP_TSSP_MTDA.LAMBDA_PL" "0.0"
      "TRAINER.CLIP_TSSP_MTDA.GAP_POSITION" "middle"
      "OPTIM.NAME" "adamw"
      "OPTIM.LR" "0.0001"
      "OPTIM.WEIGHT_DECAY" "0.0001"
    )
    ;;
  clip_tssp_pair_gap_pl|pair_gap_pl|pl|p0)
    DEFAULT_TAG="clip_tssp_pair_gap_pl"
    METHOD_OPTS=(
      "TRAINER.CLIP_TSSP_MTDA.USE_GAP_TOKEN" "True"
      "TRAINER.CLIP_TSSP_MTDA.STYLE_GROUP_SIZE" "2"
      "TRAINER.CLIP_TSSP_MTDA.GAP_GROUP_SIZE" "2"
      "TRAINER.CLIP_TSSP_MTDA.USE_IMAGE_TOKENS" "False"
      "TRAINER.CLIP_TSSP_MTDA.LAMBDA_EM" "0.0"
      "TRAINER.CLIP_TSSP_MTDA.LAMBDA_KL" "0.0"
      "TRAINER.CLIP_TSSP_MTDA.LAMBDA_PL" "0.2"
      "TRAINER.CLIP_TSSP_MTDA.PL_THRESHOLD" "0.7"
      "TRAINER.CLIP_TSSP_MTDA.PL_STUDENT_THRESHOLD" "0.7"
      "TRAINER.CLIP_TSSP_MTDA.PL_USE_STUDENT_LOW_CONF_MASK" "True"
      "TRAINER.CLIP_TSSP_MTDA.GAP_POSITION" "middle"
      "OPTIM.NAME" "adamw"
      "OPTIM.LR" "0.0001"
      "OPTIM.WEIGHT_DECAY" "0.0001"
    )
    ;;
  clip_tssp_pair_gap_pl_kl|pair_gap_pl_kl|pl_kl|kp0)
    DEFAULT_TAG="clip_tssp_pair_gap_pl_kl"
    METHOD_OPTS=(
      "TRAINER.CLIP_TSSP_MTDA.USE_GAP_TOKEN" "True"
      "TRAINER.CLIP_TSSP_MTDA.STYLE_GROUP_SIZE" "2"
      "TRAINER.CLIP_TSSP_MTDA.GAP_GROUP_SIZE" "2"
      "TRAINER.CLIP_TSSP_MTDA.USE_IMAGE_TOKENS" "False"
      "TRAINER.CLIP_TSSP_MTDA.LAMBDA_EM" "0.0"
      "TRAINER.CLIP_TSSP_MTDA.LAMBDA_KL" "0.05"
      "TRAINER.CLIP_TSSP_MTDA.KL_TEMPERATURE" "1.0"
      "TRAINER.CLIP_TSSP_MTDA.LAMBDA_PL" "0.2"
      "TRAINER.CLIP_TSSP_MTDA.PL_THRESHOLD" "0.7"
      "TRAINER.CLIP_TSSP_MTDA.PL_STUDENT_THRESHOLD" "0.7"
      "TRAINER.CLIP_TSSP_MTDA.PL_USE_STUDENT_LOW_CONF_MASK" "True"
      "TRAINER.CLIP_TSSP_MTDA.GAP_POSITION" "middle"
      "OPTIM.NAME" "adamw"
      "OPTIM.LR" "0.0001"
      "OPTIM.WEIGHT_DECAY" "0.0001"
    )
    ;;
  clip_tssp_pair_gap_em|pair_gap_em)
    DEFAULT_TAG="clip_tssp_pair_gap_em"
    METHOD_OPTS=(
      "TRAINER.CLIP_TSSP_MTDA.USE_GAP_TOKEN" "True"
      "TRAINER.CLIP_TSSP_MTDA.STYLE_GROUP_SIZE" "2"
      "TRAINER.CLIP_TSSP_MTDA.GAP_GROUP_SIZE" "2"
      "TRAINER.CLIP_TSSP_MTDA.USE_IMAGE_TOKENS" "False"
      "TRAINER.CLIP_TSSP_MTDA.LAMBDA_EM" "0.01"
      "TRAINER.CLIP_TSSP_MTDA.GAP_POSITION" "middle"
    )
    ;;
  clip_tssp_pair_gap_vctx8|pair_gap_vctx8|vctx8|c0)
    DEFAULT_TAG="clip_tssp_pair_gap_vctx8"
    METHOD_OPTS=(
      "TRAINER.CLIP_TSSP_MTDA.USE_GAP_TOKEN" "True"
      "TRAINER.CLIP_TSSP_MTDA.STYLE_GROUP_SIZE" "2"
      "TRAINER.CLIP_TSSP_MTDA.GAP_GROUP_SIZE" "2"
      "TRAINER.CLIP_TSSP_MTDA.USE_IMAGE_TOKENS" "False"
      "TRAINER.CLIP_TSSP_MTDA.ENABLE_VPT" "True"
      "TRAINER.CLIP_TSSP_MTDA.N_VCTX" "8"
      "TRAINER.CLIP_TSSP_MTDA.LAMBDA_EM" "0.0"
      "TRAINER.CLIP_TSSP_MTDA.DETACH_ENTROPY_TEXT" "False"
      "TRAINER.CLIP_TSSP_MTDA.GAP_POSITION" "middle"
    )
    ;;
  clip_tssp_pair_gap_vctx8_em_detach|pair_gap_vctx8_em_detach|vctx8_em|c1)
    DEFAULT_TAG="clip_tssp_pair_gap_vctx8_em_detach"
    METHOD_OPTS=(
      "TRAINER.CLIP_TSSP_MTDA.USE_GAP_TOKEN" "True"
      "TRAINER.CLIP_TSSP_MTDA.STYLE_GROUP_SIZE" "2"
      "TRAINER.CLIP_TSSP_MTDA.GAP_GROUP_SIZE" "2"
      "TRAINER.CLIP_TSSP_MTDA.USE_IMAGE_TOKENS" "False"
      "TRAINER.CLIP_TSSP_MTDA.ENABLE_VPT" "True"
      "TRAINER.CLIP_TSSP_MTDA.N_VCTX" "8"
      "TRAINER.CLIP_TSSP_MTDA.LAMBDA_EM" "0.01"
      "TRAINER.CLIP_TSSP_MTDA.DETACH_ENTROPY_TEXT" "True"
      "TRAINER.CLIP_TSSP_MTDA.GAP_POSITION" "middle"
    )
    ;;
  clip_tssp_pair_gap_img12|pair_gap_img12|img12)
    DEFAULT_TAG="clip_tssp_pair_gap_img12"
    METHOD_OPTS=(
      "TRAINER.CLIP_TSSP_MTDA.USE_GAP_TOKEN" "True"
      "TRAINER.CLIP_TSSP_MTDA.STYLE_GROUP_SIZE" "2"
      "TRAINER.CLIP_TSSP_MTDA.GAP_GROUP_SIZE" "2"
      "TRAINER.CLIP_TSSP_MTDA.USE_IMAGE_TOKENS" "True"
      "TRAINER.CLIP_TSSP_MTDA.IMAGE_GROUP_SIZE" "1"
      "TRAINER.CLIP_TSSP_MTDA.GAP_POSITION" "middle"
    )
    ;;
  clip_tssp_pair_gap_img6|pair_gap_img6|img6)
    DEFAULT_TAG="clip_tssp_pair_gap_img6"
    METHOD_OPTS=(
      "TRAINER.CLIP_TSSP_MTDA.USE_GAP_TOKEN" "True"
      "TRAINER.CLIP_TSSP_MTDA.STYLE_GROUP_SIZE" "2"
      "TRAINER.CLIP_TSSP_MTDA.GAP_GROUP_SIZE" "2"
      "TRAINER.CLIP_TSSP_MTDA.USE_IMAGE_TOKENS" "True"
      "TRAINER.CLIP_TSSP_MTDA.IMAGE_GROUP_SIZE" "2"
      "TRAINER.CLIP_TSSP_MTDA.LAMBDA_EM" "0.0"
      "TRAINER.CLIP_TSSP_MTDA.GAP_POSITION" "middle"
    )
    ;;
  clip_tssp_pair_gap_img6_em|pair_gap_img6_em|img6_em)
    DEFAULT_TAG="clip_tssp_pair_gap_img6_em"
    METHOD_OPTS=(
      "TRAINER.CLIP_TSSP_MTDA.USE_GAP_TOKEN" "True"
      "TRAINER.CLIP_TSSP_MTDA.STYLE_GROUP_SIZE" "2"
      "TRAINER.CLIP_TSSP_MTDA.GAP_GROUP_SIZE" "2"
      "TRAINER.CLIP_TSSP_MTDA.USE_IMAGE_TOKENS" "True"
      "TRAINER.CLIP_TSSP_MTDA.IMAGE_GROUP_SIZE" "2"
      "TRAINER.CLIP_TSSP_MTDA.LAMBDA_EM" "0.01"
      "TRAINER.CLIP_TSSP_MTDA.GAP_POSITION" "middle"
    )
    ;;
  clip_tssp_pair_gap_img4|pair_gap_img4|img4)
    DEFAULT_TAG="clip_tssp_pair_gap_img4"
    METHOD_OPTS=(
      "TRAINER.CLIP_TSSP_MTDA.USE_GAP_TOKEN" "True"
      "TRAINER.CLIP_TSSP_MTDA.STYLE_GROUP_SIZE" "2"
      "TRAINER.CLIP_TSSP_MTDA.GAP_GROUP_SIZE" "2"
      "TRAINER.CLIP_TSSP_MTDA.USE_IMAGE_TOKENS" "True"
      "TRAINER.CLIP_TSSP_MTDA.IMAGE_GROUP_SIZE" "3"
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
if [ "${DRY_RUN}" = "1" ]; then
  echo "Dry run: enabled"
fi
echo "==============================================="

if [ "${DRY_RUN}" = "1" ]; then
  printf 'Command:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi

"${CMD[@]}"
