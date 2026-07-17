#!/usr/bin/env bash
set -euo pipefail

SOURCE="${1:?usage: run_warmstart_stage1.sh <R|Q> [seed]}"
SEED="${2:-100}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
DATA_ROOT="${DATA_ROOT:-/workspace/dataset}"

case "${SOURCE}" in
  R) source_domain=real; targets=(clipart infograph painting quickdraw sketch) ;;
  Q) source_domain=quickdraw; targets=(clipart infograph painting real sketch) ;;
  *) echo "Unknown DomainNet source: ${SOURCE}" >&2; exit 2 ;;
esac

difficulty_file="${ROOT_DIR}/results/domainnet_mtda/difficulty_${SOURCE}_seed${SEED}.json"
mapfile -t order < <(
  python -c 'import json,sys; print(*json.load(open(sys.argv[1]))["hard_to_easy"], sep="\n")' \
    "${difficulty_file}"
)
order_cfg="['${order[0]}','${order[1]}','${order[2]}','${order[3]}','${order[4]}']"
source_checkpoint_dir="${ROOT_DIR}/output/domainnet_mtda/source_probe/${SOURCE}2O/seed${SEED}"
run_dir="${ROOT_DIR}/output/domainnet_mtda/warmstart_stage1/${SOURCE}2O/seed${SEED}"

python train.py \
  --root "${DATA_ROOT}" \
  --seed "${SEED}" \
  --trainer CurriculumContinuousSharedProjMaPLeMTDA \
  --dataset-config-file configs/datasets/domainnet_mtda.yaml \
  --config-file configs/trainers/CurriculumContinuousSharedProjMaPLeMTDA/vit_b16.yaml \
  --source-domains "${source_domain}" \
  --target-domains "${targets[@]}" \
  --output-dir "${run_dir}" \
  OPTIM.MAX_EPOCH 2 \
  TRAIN.MAX_BATCHES_PER_EPOCH 1000 \
  TRAINER.MAPLE_MTDA.CURRICULUM.DOMAIN_ORDER "${order_cfg}" \
  TRAINER.MAPLE_MTDA.CURRICULUM.STAGE_LIMIT 1 \
  TRAINER.MAPLE_MTDA.CURRICULUM.REPLAY.ENABLED True \
  TRAINER.MAPLE_MTDA.CURRICULUM.REPLAY.TOPK_PER_CLASS 8 \
  TRAINER.MAPLE_MTDA.CURRICULUM.REPLAY.LAMBDA 0.75 \
  TRAINER.MAPLE_MTDA.CURRICULUM.REPLAY.SELECTION_MODE online \
  TRAINER.MAPLE_MTDA.CURRICULUM.REPLAY.LABEL_SOURCE pseudo \
  TRAINER.MAPLE_MTDA.CURRICULUM.REPLAY.TRAVERSAL cycle \
  TRAINER.MAPLE_MTDA.CURRICULUM.REPLAY.NORMALIZATION none \
  TRAINER.MAPLE_MTDA.CURRICULUM.DIAGNOSTICS.ENABLED True \
  TRAINER.MAPLE_MTDA.CURRICULUM.DIAGNOSTICS.AUDIT_ALL_DOMAINS True \
  TRAINER.MAPLE_MTDA.CURRICULUM.RESET_OPTIM_PER_STAGE False \
  TRAINER.MAPLE_MTDA.PL_VARIANT agreement_hard_student_soft \
  TRAINER.MAPLE_MTDA.LAMBDA_PL 0.3 \
  TRAINER.MAPLE_MTDA.LAMBDA_PL_FINAL 0.3 \
  TRAINER.MAPLE_MTDA.PL_DUAL_CONF_THRESHOLD 0.7 \
  TRAINER.MAPLE_MTDA.PL_STUDENT_SOFT_LAMBDA 0.5 \
  TRAINER.MAPLE_MTDA.PL_STRONG_AUGMENT randaugment_fixmatch \
  TRAINER.MAPLE_MTDA.POST_INIT.ENABLED True \
  TRAINER.MAPLE_MTDA.POST_INIT.MODEL_DIR "${source_checkpoint_dir}" \
  TRAINER.MAPLE_MTDA.POST_INIT.CHECKPOINT_MODEL_NAME ContinuousSharedProjMaPLeMTDA \
  TRAINER.MAPLE_MTDA.POST_INIT.LOAD_EPOCH 10
