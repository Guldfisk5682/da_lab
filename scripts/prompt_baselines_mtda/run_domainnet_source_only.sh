#!/usr/bin/env bash
set -euo pipefail

METHOD="${1:?usage: run_domainnet_source_only.sh <coop|cocoop|maple> <C|I|P|Q|R|S> [seed]}"
SOURCE="${2:?usage: run_domainnet_source_only.sh <coop|cocoop|maple> <C|I|P|Q|R|S> [seed]}"
SEED="${3:-100}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
DATA_ROOT="${DATA_ROOT:-/workspace/dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/output/domainnet_prompt_baselines}"
PYTHON_BIN="${PYTHON_BIN:-python}"

case "${METHOD}" in
  coop) config=configs/trainers/PromptBaselineMTDA/coop_vit_b16.yaml; trainer=CoOpMTDA ;;
  cocoop) config=configs/trainers/PromptBaselineMTDA/cocoop_vit_b16.yaml; trainer=CoCoOpMTDA ;;
  maple) config=configs/trainers/PromptBaselineMTDA/maple_vit_b16.yaml; trainer=MaPLeMTDA ;;
  *) echo "Unknown method: ${METHOD}" >&2; exit 2 ;;
esac

case "${SOURCE}" in
  C) source_domain=clipart; targets=(infograph painting quickdraw real sketch) ;;
  I) source_domain=infograph; targets=(clipart painting quickdraw real sketch) ;;
  P) source_domain=painting; targets=(clipart infograph quickdraw real sketch) ;;
  Q) source_domain=quickdraw; targets=(clipart infograph painting real sketch) ;;
  R) source_domain=real; targets=(clipart infograph painting quickdraw sketch) ;;
  S) source_domain=sketch; targets=(clipart infograph painting quickdraw real) ;;
  *) echo "Unknown DomainNet source: ${SOURCE}" >&2; exit 2 ;;
esac

run_dir="${OUTPUT_ROOT}/${METHOD}/source_only/${SOURCE}2O/seed${SEED}"
if [[ -s "${run_dir}/mtda_metrics.json" ]]; then
  echo "Completed metrics already exist; skipping ${run_dir}"
  exit 0
fi

"${PYTHON_BIN}" train.py \
  --root "${DATA_ROOT}" \
  --seed "${SEED}" \
  --trainer "${trainer}" \
  --dataset-config-file configs/datasets/domainnet_mtda.yaml \
  --config-file "${config}" \
  --source-domains "${source_domain}" \
  --target-domains "${targets[@]}" \
  --output-dir "${run_dir}" \
  TRAIN.MAX_BATCHES_PER_EPOCH -1 \
  TRAIN.SOURCE_ONLY True \
  TRAINER.PROMPT_BASELINE_MTDA.MIX_TARGETS True \
  TRAINER.PROMPT_BASELINE_MTDA.LAMBDA_ENT 0.0

"${PYTHON_BIN}" scripts/prompt_baselines_mtda/collect_results.py \
  --run-dir "${run_dir}" \
  --method "${METHOD}" \
  --protocol source_only \
  --source "${SOURCE}" \
  --seed "${SEED}" \
  --entropy-weight 0.0 \
  --expected-targets 5
