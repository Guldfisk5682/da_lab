#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/../.."

if [ "$#" -ne 4 ]; then
  echo "Usage: bash scripts/gspa_legacy_ablation/run_one.sh <EXP_CODE> <SRC> <TGT> <SEED>" >&2
  echo "Example: bash scripts/gspa_legacy_ablation/run_one.sh L0 A W 1" >&2
  exit 1
fi

EXP_CODE="$1"
SRC_CODE="$(echo "$2" | tr '[:lower:]' '[:upper:]')"
TGT_CODE="$(echo "$3" | tr '[:lower:]' '[:upper:]')"
SEED="$4"

DATA="${DATA:-/path/to/datasets}"
DATASET_CONFIG="${DATASET_CONFIG:-configs/datasets/office31.yaml}"
BACKBONE="${BACKBONE:-}"
DEBUG_PRINT_ONCE="${DEBUG_PRINT_ONCE:-False}"
STAGE="${STAGE:-1}"

if [ "${DATA}" = "/path/to/datasets" ]; then
  echo "Please set DATA to a real dataset root before running ablations." >&2
  exit 2
fi

resolve_domain() {
  case "$1" in
    A) echo "amazon" ;;
    D) echo "dslr" ;;
    W) echo "webcam" ;;
    amazon|dslr|webcam) echo "$1" ;;
    *)
      echo "Unknown domain code: $1" >&2
      exit 3
      ;;
  esac
}

case "${EXP_CODE}" in
  B0)
    EXP_NAME="B0_cocoop"
    TRAINER="CoCoOp"
    CFG="configs/trainers/CoCoOp/vit_b16_c4_ep10_batch1_ctxv1.yaml"
    TRAIN_SCRIPT="scripts/cocoop_da/office31_train.sh"
    EVAL_SCRIPT="scripts/cocoop_da/office31_eval.sh"
    ;;
  B1)
    EXP_NAME="B1_last3_tuning"
    TRAINER="GSPALegacy"
    CFG="configs/trainers/GSPA_LEGACY/ablation/office31_B1_last3_tuning.yaml"
    TRAIN_SCRIPT="scripts/gspa_legacy/office31_train.sh"
    EVAL_SCRIPT="scripts/gspa_legacy/office31_eval.sh"
    ;;
  L0)
    EXP_NAME="L0_full"
    TRAINER="GSPALegacy"
    CFG="configs/trainers/GSPA_LEGACY/ablation/office31_L0_full.yaml"
    TRAIN_SCRIPT="scripts/gspa_legacy/office31_train.sh"
    EVAL_SCRIPT="scripts/gspa_legacy/office31_eval.sh"
    ;;
  L1)
    EXP_NAME="L1_fixed_gate"
    TRAINER="GSPALegacy"
    CFG="configs/trainers/GSPA_LEGACY/ablation/office31_L1_fixed_gate.yaml"
    TRAIN_SCRIPT="scripts/gspa_legacy/office31_train.sh"
    EVAL_SCRIPT="scripts/gspa_legacy/office31_eval.sh"
    ;;
  L2)
    EXP_NAME="L2_normal_only"
    TRAINER="GSPALegacy"
    CFG="configs/trainers/GSPA_LEGACY/ablation/office31_L2_normal_only.yaml"
    TRAIN_SCRIPT="scripts/gspa_legacy/office31_train.sh"
    EVAL_SCRIPT="scripts/gspa_legacy/office31_eval.sh"
    ;;
  L3)
    EXP_NAME="L3_last3_frozen"
    TRAINER="GSPALegacy"
    CFG="configs/trainers/GSPA_LEGACY/ablation/office31_L3_last3_frozen.yaml"
    TRAIN_SCRIPT="scripts/gspa_legacy/office31_train.sh"
    EVAL_SCRIPT="scripts/gspa_legacy/office31_eval.sh"
    ;;
  L4)
    EXP_NAME="L4_identity_style"
    TRAINER="GSPALegacy"
    CFG="configs/trainers/GSPA_LEGACY/ablation/office31_L4_identity_style.yaml"
    TRAIN_SCRIPT="scripts/gspa_legacy/office31_train.sh"
    EVAL_SCRIPT="scripts/gspa_legacy/office31_eval.sh"
    ;;
  L5)
    EXP_NAME="L5_patch_only"
    TRAINER="GSPALegacy"
    CFG="configs/trainers/GSPA_LEGACY/ablation/office31_L5_patch_only.yaml"
    TRAIN_SCRIPT="scripts/gspa_legacy/office31_train.sh"
    EVAL_SCRIPT="scripts/gspa_legacy/office31_eval.sh"
    ;;
  *)
    echo "Unsupported experiment code: ${EXP_CODE}" >&2
    exit 4
    ;;
esac

if [ "${EXP_CODE}" != "B0" ] && [ ! -f "${CFG}" ]; then
  python scripts/gspa_legacy_ablation/make_ablation_configs.py
fi

SOURCE_DOMAIN="$(resolve_domain "${SRC_CODE}")"
TARGET_DOMAIN="$(resolve_domain "${TGT_CODE}")"
TASK_TAG="${SRC_CODE}2${TGT_CODE}"
OUTPUT_DIR="output/office31_ablation/${EXP_NAME}/${TASK_TAG}/seed${SEED}"

echo "==============================================="
echo "Experiment name: ${EXP_NAME}"
echo "Source domain: ${SOURCE_DOMAIN}"
echo "Target domain: ${TARGET_DOMAIN}"
echo "Seed: ${SEED}"
echo "Trainer: ${TRAINER}"
echo "Config: ${CFG}"
echo "Output dir: ${OUTPUT_DIR}"
echo "==============================================="

DATA="${DATA}" \
TRAINER="${TRAINER}" \
DATASET_CONFIG="${DATASET_CONFIG}" \
CFG="${CFG}" \
SOURCE_DOMAIN="${SOURCE_DOMAIN}" \
TARGET_DOMAIN="${TARGET_DOMAIN}" \
SEED="${SEED}" \
TRAINER_DIR="${EXP_NAME}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
DEBUG_PRINT_ONCE="${DEBUG_PRINT_ONCE}" \
BACKBONE="${BACKBONE}" \
STAGE="${STAGE}" \
  bash "${TRAIN_SCRIPT}"

DATA="${DATA}" \
TRAINER="${TRAINER}" \
DATASET_CONFIG="${DATASET_CONFIG}" \
CFG="${CFG}" \
SOURCE_DOMAIN="${SOURCE_DOMAIN}" \
TARGET_DOMAIN="${TARGET_DOMAIN}" \
SEED="${SEED}" \
TRAINER_DIR="${EXP_NAME}" \
MODEL_DIR="${OUTPUT_DIR}" \
BACKBONE="${BACKBONE}" \
STAGE="${STAGE}" \
  bash "${EVAL_SCRIPT}"
