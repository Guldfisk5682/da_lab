#!/usr/bin/env bash
set -euo pipefail

SOURCE="${1:?usage: run_source_probe.sh <R|Q> [seed]}"
SEED="${2:-100}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
DATA_ROOT="${DATA_ROOT:-/workspace/dataset}"

case "${SOURCE}" in
  R) source_domain=real; targets=(clipart infograph painting quickdraw sketch) ;;
  Q) source_domain=quickdraw; targets=(clipart infograph painting real sketch) ;;
  *) echo "Unknown DomainNet source: ${SOURCE}" >&2; exit 2 ;;
esac

run_dir="${ROOT_DIR}/output/domainnet_mtda/source_probe/${SOURCE}2O/seed${SEED}"
score_file="${ROOT_DIR}/results/domainnet_mtda/difficulty_${SOURCE}_seed${SEED}.json"
checkpoint="${run_dir}/ContinuousSharedProjMaPLeMTDA/model.pth.tar-10"

if [[ ! -s "${checkpoint}" ]]; then
  python train.py \
    --root "${DATA_ROOT}" \
    --seed "${SEED}" \
    --trainer ContinuousSharedProjMaPLeMTDA \
    --dataset-config-file configs/datasets/domainnet_mtda.yaml \
    --config-file configs/trainers/ContinuousSharedProjMaPLeMTDA/vit_b16.yaml \
    --source-domains "${source_domain}" \
    --target-domains "${targets[@]}" \
    --output-dir "${run_dir}" \
    OPTIM.MAX_EPOCH 10 \
    TRAIN.MAX_BATCHES_PER_EPOCH 500 \
    TRAINER.PROMPT_BASELINE_MTDA.MIX_TARGETS True \
    TRAINER.PROMPT_BASELINE_MTDA.LAMBDA_ENT 0.0 \
    TEST.NO_TEST True
fi

python scripts/domainnet_mtda/score_domain_difficulty.py \
  --source "${SOURCE}" \
  --root "${DATA_ROOT}" \
  --seed "${SEED}" \
  --model-dir "${run_dir}" \
  --output "${score_file}"
