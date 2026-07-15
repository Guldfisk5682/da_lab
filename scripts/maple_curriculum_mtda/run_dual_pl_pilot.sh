#!/bin/bash

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "${REPO_ROOT}"

SEED="${SEED:-100}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/dual_pl_pilot_20260715}"
mkdir -p "${LOG_ROOT}"

run_one() {
  local source="$1"
  local order="$2"
  local variant="$3"
  local method_tag="maple_dualpl_${variant}_seed${SEED}"
  local target_tag
  case "${source}" in
    A) target_tag="CPR" ;;
    C) target_tag="APR" ;;
    *) echo "Unsupported pilot source: ${source}" >&2; return 2 ;;
  esac
  local output_dir="${REPO_ROOT}/output/officehome_mtda/${method_tag}/${source}2${target_tag}/seed${SEED}"
  local log_file="${LOG_ROOT}/${source}2${target_tag}_${variant}_seed${SEED}.log"

  echo "Starting ${source}2${target_tag} ${variant} seed${SEED}"
  DOMAIN_ORDER="${order}" \
  METHOD_TAG="${method_tag}" \
  OUTPUT_DIR="${output_dir}" \
  REPLAY_ENABLED=True \
  TOPK_PER_CLASS=8 \
  REPLAY_LAMBDA=0.75 \
  REPLAY_SELECTION_MODE=online \
  REPLAY_LABEL_SOURCE=pseudo \
  REPLAY_TRAVERSAL=cycle \
  REPLAY_NORMALIZATION=none \
  DIAGNOSTICS_ENABLED=True \
  RESET_OPTIM_PER_STAGE=False \
  PL_VARIANT="${variant}" \
  PL_DUAL_CONF_THRESHOLD=0.7 \
  PL_SOFT_BETA=1.0 \
  PL_STRONG_AUGMENT=randaugment_fixmatch \
  PL_DIAGNOSTIC_WINDOW=50 \
  PL_GRAD_AUDIT_INTERVAL=50 \
  bash scripts/maple_curriculum_mtda/run_officehome_one.sh \
    "${source}" "${SEED}" 2>&1 | tee "${log_file}"

  test -f "${output_dir}/CurriculumContinuousSharedProjMaPLeMTDA/model.pth.tar-5"
  test -s "${output_dir}/pl_sample_audit.jsonl"
  test -s "${output_dir}/pl_training_window_audit.jsonl"
  if grep -Eq "Traceback|CUDA out of memory|FloatingPointError" "${log_file}"; then
    echo "Failure signature found in ${log_file}" >&2
    return 3
  fi
  echo "Completed ${source}2${target_tag} ${variant} seed${SEED}"
}

PILOT_SOURCES="${PILOT_SOURCES:-A C}"
for source in ${PILOT_SOURCES}; do
  case "${source}" in
    A)
      run_one A "clipart product real_world" agreement_hard
      run_one A "clipart product real_world" agreement_hard_soft
      ;;
    C)
      run_one C "art real_world product" agreement_hard
      run_one C "art real_world product" agreement_hard_soft
      ;;
    *)
      echo "Unsupported PILOT_SOURCES entry: ${source}" >&2
      exit 2
      ;;
  esac
done

echo "Dual-view PL pilot completed successfully"
