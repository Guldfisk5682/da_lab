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

SEED="${SEED:-1}"
DATA="${DATA:-${REPO_ROOT}/data}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
COLLECT_AFTER_EACH="${COLLECT_AFTER_EACH:-1}"
RESET_OUTPUT="${RESET_OUTPUT:-1}"

export DATA
export CUDA_VISIBLE_DEVICES
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export EVAL_EVERY_EPOCH="${EVAL_EVERY_EPOCH:-False}"

COMMON_OPTS=(
  "TRAINER.CLIP_TSSP_MTDA.USE_GAP_TOKEN" "True"
  "TRAINER.CLIP_TSSP_MTDA.STYLE_GROUP_SIZE" "2"
  "TRAINER.CLIP_TSSP_MTDA.GAP_GROUP_SIZE" "2"
  "TRAINER.CLIP_TSSP_MTDA.USE_IMAGE_TOKENS" "False"
  "TRAINER.CLIP_TSSP_MTDA.LAMBDA_EM" "0.0"
  "TRAINER.CLIP_TSSP_MTDA.GAP_POSITION" "middle"
  "TRAINER.CLIP_TSSP_MTDA.PL_THRESHOLD" "0.7"
  "TRAINER.CLIP_TSSP_MTDA.PL_STUDENT_THRESHOLD" "0.7"
  "TRAINER.CLIP_TSSP_MTDA.PL_USE_STUDENT_LOW_CONF_MASK" "True"
  "TRAINER.CLIP_TSSP_MTDA.KL_TEMPERATURE" "1.0"
  "OPTIM.NAME" "sgd"
  "OPTIM.LR" "0.002"
)

collect_results() {
  python "${REPO_ROOT}/scripts/clip_tssp_mtda/collect_officehome_results.py" --seed "${SEED}" || true
}

run_exp() {
  local tag="$1"
  local lambda_pl="$2"
  local lambda_kl="$3"

  echo "============================================================"
  echo "[$(date '+%F %T')] START ${tag}"
  echo "seed=${SEED}, lambda_pl=${lambda_pl}, lambda_kl=${lambda_kl}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "============================================================"

  if [ "${RESET_OUTPUT}" = "1" ]; then
    rm -rf "${REPO_ROOT}/output/office_home_mtda/${tag}"
  fi

  local extra_opts=(
    "${COMMON_OPTS[@]}"
    "TRAINER.CLIP_TSSP_MTDA.LAMBDA_PL" "${lambda_pl}"
    "TRAINER.CLIP_TSSP_MTDA.LAMBDA_KL" "${lambda_kl}"
  )

  METHOD_TAG="${tag}" EXTRA_OPTS="${extra_opts[*]}" SEED="${SEED}" \
    bash "${REPO_ROOT}/scripts/clip_tssp_mtda/run_officehome_all.sh" clip_tssp_pair_gap

  if [ "${COLLECT_AFTER_EACH}" = "1" ]; then
    collect_results
  fi

  echo "[$(date '+%F %T')] END ${tag}"
}

if [ "$#" -gt 0 ]; then
  REQUESTED_TAGS=("$@")
else
  REQUESTED_TAGS=(
    clip_tssp_pair_gap_sgd_pl01
    clip_tssp_pair_gap_sgd_pl02
    clip_tssp_pair_gap_sgd_pl03
    clip_tssp_pair_gap_sgd_pl02_kl001
    clip_tssp_pair_gap_sgd_pl02_kl005
    clip_tssp_pair_gap_sgd_pl02_kl010
  )
fi

for tag in "${REQUESTED_TAGS[@]}"; do
  case "${tag}" in
    clip_tssp_pair_gap_sgd_pl01)
      run_exp "${tag}" "0.1" "0.0"
      ;;
    clip_tssp_pair_gap_sgd_pl02)
      run_exp "${tag}" "0.2" "0.0"
      ;;
    clip_tssp_pair_gap_sgd_pl03)
      run_exp "${tag}" "0.3" "0.0"
      ;;
    clip_tssp_pair_gap_sgd_pl02_kl001)
      run_exp "${tag}" "0.2" "0.01"
      ;;
    clip_tssp_pair_gap_sgd_pl02_kl005)
      run_exp "${tag}" "0.2" "0.05"
      ;;
    clip_tssp_pair_gap_sgd_pl02_kl010)
      run_exp "${tag}" "0.2" "0.10"
      ;;
    *)
      echo "Unknown sweep tag: ${tag}" >&2
      exit 2
      ;;
  esac
done

collect_results
