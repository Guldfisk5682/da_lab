#!/usr/bin/env bash
set -euo pipefail

SOURCE="${1:?usage: run_ours_one.sh <A|D|W> [seed]}"
SEED="${2:-100}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/data}"
METHOD_TAG="${METHOD_TAG:-ours_full}"

case "${SOURCE}" in
  A) source_domain=amazon; targets=(dslr webcam); target_codes=(D W); target_tag=DW ;;
  D) source_domain=dslr; targets=(amazon webcam); target_codes=(A W); target_tag=AW ;;
  W) source_domain=webcam; targets=(amazon dslr); target_codes=(A D); target_tag=AD ;;
  *) echo "Unknown Office-31 source: ${SOURCE}" >&2; exit 2 ;;
esac

difficulty_file="${ROOT_DIR}/results/office31_mtda/difficulty_${SOURCE}_seed${SEED}.json"
if [[ ! -s "${difficulty_file}" ]]; then
  echo "Missing source-only difficulty probe: ${difficulty_file}" >&2
  exit 3
fi
mapfile -t order < <(
  python -c 'import json,sys; print(*json.load(open(sys.argv[1]))["hard_to_easy"], sep="\n")' \
    "${difficulty_file}"
)
if [[ "${#order[@]}" -ne 2 ]]; then
  echo "Expected two target domains in ${difficulty_file}" >&2
  exit 3
fi
order_cfg="['${order[0]}','${order[1]}']"

run_dir="${ROOT_DIR}/output/office31_mtda/${METHOD_TAG}/${SOURCE}2${target_tag}/seed${SEED}"
if [[ -s "${run_dir}/mtda_metrics.json" ]]; then
  echo "Completed metrics already exist; skipping ${run_dir}"
  exit 0
fi

cfg_opts=(
  TRAINER.MAPLE_MTDA.CURRICULUM.DOMAIN_ORDER "${order_cfg}"
  TRAINER.MAPLE_MTDA.CURRICULUM.REPLAY.ENABLED True
  TRAINER.MAPLE_MTDA.CURRICULUM.REPLAY.TOPK_PER_CLASS 8
  TRAINER.MAPLE_MTDA.CURRICULUM.REPLAY.LAMBDA 0.75
  TRAINER.MAPLE_MTDA.CURRICULUM.REPLAY.SELECTION_MODE online
  TRAINER.MAPLE_MTDA.CURRICULUM.REPLAY.LABEL_SOURCE pseudo
  TRAINER.MAPLE_MTDA.CURRICULUM.REPLAY.TRAVERSAL cycle
  TRAINER.MAPLE_MTDA.CURRICULUM.REPLAY.NORMALIZATION none
  TRAINER.MAPLE_MTDA.CURRICULUM.DIAGNOSTICS.ENABLED False
  TRAINER.MAPLE_MTDA.CURRICULUM.RESET_OPTIM_PER_STAGE False
  TRAINER.MAPLE_MTDA.PL_VARIANT agreement_hard_student_soft
  TRAINER.MAPLE_MTDA.PL_DUAL_CONF_THRESHOLD 0.7
  TRAINER.MAPLE_MTDA.PL_STUDENT_SOFT_LAMBDA 0.5
  TRAINER.MAPLE_MTDA.PL_STRONG_AUGMENT randaugment_fixmatch
)

python scripts/experiment_guard.py \
  --output-dir "${run_dir}" \
  --method-tag "${METHOD_TAG}" \
  --source "${SOURCE}" \
  --targets "${target_codes[@]}" \
  --seed "${SEED}" \
  --trainer CurriculumContinuousSharedProjMaPLeMTDA \
  --trainer-config configs/trainers/CurriculumContinuousSharedProjMaPLeMTDA/vit_b16.yaml \
  --dataset-config configs/datasets/office31_mtda.yaml \
  --data "${DATA_ROOT}" \
  --effective-opts "${cfg_opts[*]}"

python train.py \
  --root "${DATA_ROOT}" \
  --seed "${SEED}" \
  --trainer CurriculumContinuousSharedProjMaPLeMTDA \
  --dataset-config-file configs/datasets/office31_mtda.yaml \
  --config-file configs/trainers/CurriculumContinuousSharedProjMaPLeMTDA/vit_b16.yaml \
  --source-domains "${source_domain}" \
  --target-domains "${targets[@]}" \
  --output-dir "${run_dir}" \
  "${cfg_opts[@]}"

python scripts/office31_mtda/collect_run.py \
  --run-dir "${run_dir}" \
  --source "${SOURCE}" \
  --seed "${SEED}" \
  --order "${order[@]}"
