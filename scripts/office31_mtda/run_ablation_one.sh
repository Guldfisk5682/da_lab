#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:?usage: run_ablation_one.sh <e2h|joint|hard_only|soft_only|no_replay|topk16|topk32> <A|D|W> [seed]}"
SOURCE="${2:?missing source A, D, or W}"
SEED="${3:-100}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/data}"

case "${SOURCE}" in
  A) source_domain=amazon; targets=(dslr webcam); target_codes=(D W); target_tag=DW ;;
  D) source_domain=dslr; targets=(amazon webcam); target_codes=(A W); target_tag=AW ;;
  W) source_domain=webcam; targets=(amazon dslr); target_codes=(A D); target_tag=AD ;;
  *) echo "Unknown Office-31 source: ${SOURCE}" >&2; exit 2 ;;
esac
case "${VARIANT}" in
  e2h|joint|hard_only|soft_only|no_replay|topk16|topk32) ;;
  *) echo "Unknown ablation: ${VARIANT}" >&2; exit 2 ;;
esac

difficulty_file="${ROOT_DIR}/results/office31_mtda/difficulty_${SOURCE}_seed${SEED}.json"
order_key=hard_to_easy
[[ "${VARIANT}" == e2h ]] && order_key=easy_to_hard
mapfile -t order < <(
  python -c 'import json,sys; print(*json.load(open(sys.argv[1]))[sys.argv[2]], sep="\n")' \
    "${difficulty_file}" "${order_key}"
)
order_cfg="['${order[0]}','${order[1]}']"

run_dir="${ROOT_DIR}/output/office31_mtda/ablations/${VARIANT}/${SOURCE}2${target_tag}/seed${SEED}"
if [[ -s "${run_dir}/mtda_metrics.json" ]]; then
  echo "Completed metrics already exist; skipping ${run_dir}"
  exit 0
fi

trainer=CurriculumContinuousSharedProjMaPLeMTDA
trainer_config=configs/trainers/CurriculumContinuousSharedProjMaPLeMTDA/vit_b16.yaml
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

case "${VARIANT}" in
  joint)
    trainer=ContinuousSharedProjMaPLeMTDA
    trainer_config=configs/trainers/ContinuousSharedProjMaPLeMTDA/vit_b16.yaml
    cfg_opts=(
      TRAINER.MAPLE_MTDA.PL_VARIANT agreement_hard_student_soft
      TRAINER.MAPLE_MTDA.PL_DUAL_CONF_THRESHOLD 0.7
      TRAINER.MAPLE_MTDA.PL_STUDENT_SOFT_LAMBDA 0.5
      TRAINER.MAPLE_MTDA.PL_STRONG_AUGMENT randaugment_fixmatch
      TRAINER.MAPLE_MTDA.LAMBDA_PL 0.3
      TRAINER.MAPLE_MTDA.LAMBDA_PL_FINAL 0.3
    )
    ;;
  hard_only)
    cfg_opts+=(TRAINER.MAPLE_MTDA.PL_STUDENT_SOFT_LAMBDA 0.0)
    ;;
  soft_only)
    cfg_opts+=(TRAINER.MAPLE_MTDA.LAMBDA_PL 0.0 TRAINER.MAPLE_MTDA.LAMBDA_PL_FINAL 0.0)
    ;;
  no_replay)
    cfg_opts+=(TRAINER.MAPLE_MTDA.CURRICULUM.REPLAY.ENABLED False)
    ;;
  topk16)
    cfg_opts+=(TRAINER.MAPLE_MTDA.CURRICULUM.REPLAY.TOPK_PER_CLASS 16)
    ;;
  topk32)
    cfg_opts+=(TRAINER.MAPLE_MTDA.CURRICULUM.REPLAY.TOPK_PER_CLASS 32)
    ;;
esac

python scripts/experiment_guard.py \
  --output-dir "${run_dir}" \
  --method-tag "office31_ablation_${VARIANT}" \
  --source "${SOURCE}" \
  --targets "${target_codes[@]}" \
  --seed "${SEED}" \
  --trainer "${trainer}" \
  --trainer-config "${trainer_config}" \
  --dataset-config configs/datasets/office31_mtda.yaml \
  --data "${DATA_ROOT}" \
  --effective-opts "${cfg_opts[*]}"

python train.py \
  --root "${DATA_ROOT}" \
  --seed "${SEED}" \
  --trainer "${trainer}" \
  --dataset-config-file configs/datasets/office31_mtda.yaml \
  --config-file "${trainer_config}" \
  --source-domains "${source_domain}" \
  --target-domains "${targets[@]}" \
  --output-dir "${run_dir}" \
  "${cfg_opts[@]}"

collect_order=("${order[@]}")
[[ "${VARIANT}" == joint ]] && collect_order=()
python scripts/office31_mtda/collect_ablation.py \
  --run-dir "${run_dir}" \
  --variant "${VARIANT}" \
  --source "${SOURCE}" \
  --seed "${SEED}" \
  --order "${collect_order[@]}"
