#!/usr/bin/env bash
set -euo pipefail

# Evaluate an archived ES-MaPLe source-only checkpoint on the five official
# DomainNet target-test splits. This is evaluation only; no target image is
# used to update the model.

SOURCE="${1:?usage: run_esmaple_target_test.sh <C|I|P|Q|R|S> <model-root>}"
MODEL_ROOT="${2:?usage: run_esmaple_target_test.sh <C|I|P|Q|R|S> <model-root>}"
SEED="${SEED:-100}"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the parent of DomainNet/}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/domainnet_mtda/esmaple_target_test}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"

case "${SOURCE}" in
  C) source_domain=clipart; targets=(infograph painting quickdraw real sketch) ;;
  I) source_domain=infograph; targets=(clipart painting quickdraw real sketch) ;;
  P) source_domain=painting; targets=(clipart infograph quickdraw real sketch) ;;
  Q) source_domain=quickdraw; targets=(clipart infograph painting real sketch) ;;
  R) source_domain=real; targets=(clipart infograph painting quickdraw sketch) ;;
  S) source_domain=sketch; targets=(clipart infograph painting quickdraw real) ;;
  *) echo "Unknown DomainNet source: ${SOURCE}" >&2; exit 2 ;;
esac

model_dir="${MODEL_ROOT}/${SOURCE}2O/seed${SEED}"
output_dir="${OUTPUT_ROOT}/${SOURCE}2O/seed${SEED}"
test -s "${model_dir}/ContinuousSharedProjMaPLeMTDA/model.pth.tar-10"

cd "${REPO_ROOT}"
python train.py \
  --root "${DATA_ROOT}" \
  --seed "${SEED}" \
  --trainer ContinuousSharedProjMaPLeMTDA \
  --dataset-config-file configs/datasets/domainnet_mtda.yaml \
  --config-file configs/trainers/ContinuousSharedProjMaPLeMTDA/vit_b16.yaml \
  --source-domains "${source_domain}" \
  --target-domains "${targets[@]}" \
  --output-dir "${output_dir}" \
  --eval-only \
  --model-dir "${model_dir}" \
  --load-epoch 10 \
  DATALOADER.TEST.BATCH_SIZE "${TEST_BATCH_SIZE}" \
  DATALOADER.NUM_WORKERS "${NUM_WORKERS}"
