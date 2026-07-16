#!/usr/bin/env bash
set -euo pipefail

METHOD="${1:?usage: run_office31_one.sh <coop|cocoop|maple> <A|D|W> [seed]}"
SOURCE="${2:?usage: run_office31_one.sh <coop|cocoop|maple> <A|D|W> [seed]}"
SEED="${3:-100}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/output/office31_prompt_baselines}"
PYTHON_BIN="${PYTHON_BIN:-python}"

case "${METHOD}" in
  coop) config=configs/trainers/PromptBaselineMTDA/coop_vit_b16.yaml; trainer=CoOpMTDA ;;
  cocoop) config=configs/trainers/PromptBaselineMTDA/cocoop_vit_b16.yaml; trainer=CoCoOpMTDA ;;
  maple) config=configs/trainers/PromptBaselineMTDA/maple_vit_b16.yaml; trainer=MaPLeMTDA ;;
  *) echo "Unknown method: ${METHOD}" >&2; exit 2 ;;
esac

case "${SOURCE}" in
  A) source_domain=amazon; targets=(dslr webcam); target_code=DW ;;
  D) source_domain=dslr; targets=(amazon webcam); target_code=AW ;;
  W) source_domain=webcam; targets=(amazon dslr); target_code=AD ;;
  *) echo "Unknown Office-31 source: ${SOURCE}" >&2; exit 2 ;;
esac

run_dir="${OUTPUT_ROOT}/${METHOD}/source_only/${SOURCE}2${target_code}/seed${SEED}"
if [[ -s "${run_dir}/mtda_metrics.json" ]]; then
  echo "Completed metrics already exist; skipping ${run_dir}"
  exit 0
fi

"${PYTHON_BIN}" train.py \
  --root "${DATA_ROOT}" \
  --seed "${SEED}" \
  --trainer "${trainer}" \
  --dataset-config-file configs/datasets/office31_mtda.yaml \
  --config-file "${config}" \
  --source-domains "${source_domain}" \
  --target-domains "${targets[@]}" \
  --output-dir "${run_dir}" \
  TRAINER.PROMPT_BASELINE_MTDA.MIX_TARGETS True \
  TRAINER.PROMPT_BASELINE_MTDA.LAMBDA_ENT 0.0

"${PYTHON_BIN}" scripts/prompt_baselines_mtda/collect_results.py \
  --run-dir "${run_dir}" \
  --method "${METHOD}" \
  --protocol source_only \
  --source "${SOURCE}" \
  --seed "${SEED}" \
  --entropy-weight 0.0 \
  --expected-targets 2
