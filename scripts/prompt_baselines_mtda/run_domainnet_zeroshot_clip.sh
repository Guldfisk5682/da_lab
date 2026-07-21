#!/usr/bin/env bash
set -euo pipefail

SOURCE="${1:?usage: run_domainnet_zeroshot_clip.sh <C|I|P|Q|R|S> [seed]}"
SEED="${2:-100}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
DATA_ROOT="${DATA_ROOT:-/workspace/dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/output/domainnet_prompt_baselines}"
PYTHON_BIN="${PYTHON_BIN:-python}"

case "${SOURCE}" in
  C) source_domain=clipart; targets=(infograph painting quickdraw real sketch) ;;
  I) source_domain=infograph; targets=(clipart painting quickdraw real sketch) ;;
  P) source_domain=painting; targets=(clipart infograph quickdraw real sketch) ;;
  Q) source_domain=quickdraw; targets=(clipart infograph painting real sketch) ;;
  R) source_domain=real; targets=(clipart infograph painting quickdraw sketch) ;;
  S) source_domain=sketch; targets=(clipart infograph painting quickdraw real) ;;
  *) echo "Unknown DomainNet source: ${SOURCE}" >&2; exit 2 ;;
esac

run_dir="${OUTPUT_ROOT}/clip_zs/source_independent/${SOURCE}2O/seed${SEED}"
if [[ -s "${run_dir}/mtda_metrics.json" ]]; then
  echo "Completed metrics already exist; skipping ${run_dir}"
  exit 0
fi

"${PYTHON_BIN}" train.py \
  --root "${DATA_ROOT}" \
  --seed "${SEED}" \
  --trainer CLIPVPTMTDA \
  --dataset-config-file configs/datasets/domainnet_mtda.yaml \
  --config-file configs/trainers/CLIPVPTMTDA/vit_b16.yaml \
  --source-domains "${source_domain}" \
  --target-domains "${targets[@]}" \
  --output-dir "${run_dir}" \
  --eval-only \
  TRAINER.CLIP_VPT_MTDA.ENABLE_VPT False

"${PYTHON_BIN}" scripts/prompt_baselines_mtda/collect_results.py \
  --run-dir "${run_dir}" \
  --method clip_zs \
  --protocol zero_shot \
  --source "${SOURCE}" \
  --seed "${SEED}" \
  --entropy-weight 0.0 \
  --expected-targets 5
