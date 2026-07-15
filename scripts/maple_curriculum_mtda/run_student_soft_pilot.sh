#!/bin/bash

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "${REPO_ROOT}"

if [ "$#" -ne 2 ]; then
  echo "Usage: bash $0 <A|C|P|R> <seed>" >&2
  exit 2
fi

SOURCE="$1"
SEED="$2"
case "${SOURCE}" in
  A) ORDER="clipart product real_world"; TARGET_TAG="CPR" ;;
  C) ORDER="art real_world product"; TARGET_TAG="APR" ;;
  P) ORDER="clipart art real_world"; TARGET_TAG="ACR" ;;
  R) ORDER="clipart art product"; TARGET_TAG="ACP" ;;
  *) echo "Student-soft experiment expects A, C, P, or R; got ${SOURCE}" >&2; exit 2 ;;
esac

METHOD_TAG="maple_dualpl_agreement_hard_student_soft_seed${SEED}"
OUTPUT_DIR="${REPO_ROOT}/output/officehome_mtda/${METHOD_TAG}/${SOURCE}2${TARGET_TAG}/seed${SEED}"

DOMAIN_ORDER="${ORDER}" \
METHOD_TAG="${METHOD_TAG}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
REPLAY_ENABLED=True \
TOPK_PER_CLASS=8 \
REPLAY_LAMBDA=0.75 \
REPLAY_SELECTION_MODE=online \
REPLAY_LABEL_SOURCE=pseudo \
REPLAY_TRAVERSAL=cycle \
REPLAY_NORMALIZATION=none \
DIAGNOSTICS_ENABLED=True \
RESET_OPTIM_PER_STAGE=False \
PL_VARIANT=agreement_hard_student_soft \
PL_DUAL_CONF_THRESHOLD=0.7 \
PL_STUDENT_SOFT_LAMBDA=0.5 \
PL_STRONG_AUGMENT=randaugment_fixmatch \
PL_DIAGNOSTIC_WINDOW=50 \
PL_GRAD_AUDIT_INTERVAL=50 \
bash scripts/maple_curriculum_mtda/run_officehome_one.sh "${SOURCE}" "${SEED}"

test -f "${OUTPUT_DIR}/CurriculumContinuousSharedProjMaPLeMTDA/model.pth.tar-5"
test -s "${OUTPUT_DIR}/pl_sample_audit.jsonl"
test -s "${OUTPUT_DIR}/pl_training_window_audit.jsonl"

echo "Student-soft pilot completed: ${SOURCE}2${TARGET_TAG} seed${SEED}"
