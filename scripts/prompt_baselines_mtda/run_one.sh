#!/usr/bin/env bash
set -euo pipefail

METHOD="${1:?usage: run_one.sh <coop|cocoop|maple> <source_only|mt_ent> <A|C|P|R> [seed]}"
PROTOCOL="${2:?usage: run_one.sh <coop|cocoop|maple> <source_only|mt_ent> <A|C|P|R> [seed]}"
SOURCE="${3:?usage: run_one.sh <coop|cocoop|maple> <source_only|mt_ent> <A|C|P|R> [seed]}"
SEED="${4:-100}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/output/officehome_prompt_baselines}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ENTROPY_WEIGHT="${ENTROPY_WEIGHT:-0.1}"

case "${METHOD}" in
  coop) config=configs/trainers/PromptBaselineMTDA/coop_vit_b16.yaml; trainer=CoOpMTDA ;;
  cocoop) config=configs/trainers/PromptBaselineMTDA/cocoop_vit_b16.yaml; trainer=CoCoOpMTDA ;;
  maple) config=configs/trainers/PromptBaselineMTDA/maple_vit_b16.yaml; trainer=MaPLeMTDA ;;
  *) echo "Unknown method: ${METHOD}" >&2; exit 2 ;;
esac

case "${PROTOCOL}" in
  source_only) lambda_ent=0.0 ;;
  mt_ent) lambda_ent="${ENTROPY_WEIGHT}" ;;
  *) echo "Unknown protocol: ${PROTOCOL}" >&2; exit 2 ;;
esac

case "${SOURCE}" in
  A) source_domain=art; targets=(clipart product real_world); target_code=CPR ;;
  C) source_domain=clipart; targets=(art product real_world); target_code=APR ;;
  P) source_domain=product; targets=(art clipart real_world); target_code=ACR ;;
  R) source_domain=real_world; targets=(art clipart product); target_code=ACP ;;
  *) echo "Unknown source: ${SOURCE}" >&2; exit 2 ;;
esac

run_dir="${OUTPUT_ROOT}/${METHOD}/${PROTOCOL}/${SOURCE}2${target_code}/seed${SEED}"
if [[ -s "${run_dir}/mtda_metrics.json" ]]; then
  echo "Completed metrics already exist; skipping ${run_dir}"
  exit 0
fi

"${PYTHON_BIN}" train.py \
  --root "${DATA_ROOT}" \
  --seed "${SEED}" \
  --trainer "${trainer}" \
  --dataset-config-file configs/datasets/office_home_mtda.yaml \
  --config-file "${config}" \
  --source-domains "${source_domain}" \
  --target-domains "${targets[@]}" \
  --output-dir "${run_dir}" \
  TRAINER.PROMPT_BASELINE_MTDA.MIX_TARGETS True \
  TRAINER.PROMPT_BASELINE_MTDA.LAMBDA_ENT "${lambda_ent}"

"${PYTHON_BIN}" scripts/prompt_baselines_mtda/collect_results.py \
  --run-dir "${run_dir}" \
  --method "${METHOD}" \
  --protocol "${PROTOCOL}" \
  --source "${SOURCE}" \
  --seed "${SEED}" \
  --entropy-weight "${lambda_ent}"
