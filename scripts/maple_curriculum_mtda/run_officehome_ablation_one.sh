#!/usr/bin/env bash
set -euo pipefail

# Formal Office-Home ablation runner. It preserves the final MINT training
# budget and changes one component only. Full is intentionally excluded: the
# already archived final runs supply that row.

VARIANT="${1:?usage: run_officehome_ablation_one.sh <e2h|joint|hard_only|soft_only|no_replay|topk16|topk32> <A|C|P|R> [seed]}"
SOURCE="${2:?missing Office-Home source domain code}"
SEED="${3:-100}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/output/officehome_mtda}"

case "${SOURCE}" in
  A) source_domain=art; targets=(clipart product real_world); target_codes=(C P R); h2e=(real_world product clipart) ;;
  C) source_domain=clipart; targets=(art product real_world); target_codes=(A P R); h2e=(product real_world art) ;;
  P) source_domain=product; targets=(art clipart real_world); target_codes=(A C R); h2e=(clipart art real_world) ;;
  R) source_domain=real_world; targets=(art clipart product); target_codes=(A C P); h2e=(clipart art product) ;;
  *) echo "Unknown Office-Home source: ${SOURCE}" >&2; exit 2 ;;
esac
case "${VARIANT}" in
  e2h|joint|hard_only|soft_only|no_replay|topk16|topk32) ;;
  *) echo "Unknown ablation variant: ${VARIANT}" >&2; exit 2 ;;
esac

order=("${h2e[@]}")
[[ "${VARIANT}" == "e2h" ]] && order=("${h2e[2]}" "${h2e[1]}" "${h2e[0]}")
order_cfg="['${order[0]}','${order[1]}','${order[2]}']"
target_tag="$(IFS=''; echo "${target_codes[*]}")"
run_dir="${OUTPUT_ROOT}/officehome_ablation_${VARIANT}_seed${SEED}/${SOURCE}2${target_tag}/seed${SEED}"

if [[ -s "${run_dir}/mtda_metrics.json" ]]; then
  echo "Completed metrics already exist; skipping ${run_dir}"
  exit 0
fi

trainer="CurriculumContinuousSharedProjMaPLeMTDA"
trainer_config="configs/trainers/CurriculumContinuousSharedProjMaPLeMTDA/vit_b16.yaml"
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
  TRAINER.MAPLE_MTDA.LAMBDA_PL 0.3
  TRAINER.MAPLE_MTDA.LAMBDA_PL_FINAL 0.3
  TRAINER.MAPLE_MTDA.PL_DUAL_CONF_THRESHOLD 0.7
  TRAINER.MAPLE_MTDA.PL_STUDENT_SOFT_LAMBDA 0.5
  TRAINER.MAPLE_MTDA.PL_STRONG_AUGMENT randaugment_fixmatch
  TRAIN.CHECKPOINT_FREQ 0
)

case "${VARIANT}" in
  joint)
    trainer="ContinuousSharedProjMaPLeMTDA"
    trainer_config="configs/trainers/ContinuousSharedProjMaPLeMTDA/vit_b16.yaml"
    cfg_opts=(
      TRAINER.MAPLE_MTDA.PL_VARIANT agreement_hard_student_soft
      TRAINER.MAPLE_MTDA.LAMBDA_PL 0.3
      TRAINER.MAPLE_MTDA.LAMBDA_PL_FINAL 0.3
      TRAINER.MAPLE_MTDA.PL_DUAL_CONF_THRESHOLD 0.7
      TRAINER.MAPLE_MTDA.PL_STUDENT_SOFT_LAMBDA 0.5
      TRAINER.MAPLE_MTDA.PL_STRONG_AUGMENT randaugment_fixmatch
      TRAIN.CHECKPOINT_FREQ 0
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
  --method-tag "officehome_ablation_${VARIANT}_seed${SEED}" \
  --source "${SOURCE}" \
  --targets "${target_codes[@]}" \
  --seed "${SEED}" \
  --trainer "${trainer}" \
  --trainer-config "${trainer_config}" \
  --dataset-config configs/datasets/office_home_mtda.yaml \
  --data "${DATA_ROOT}" \
  --effective-opts "${cfg_opts[*]}"

python train.py \
  --root "${DATA_ROOT}" \
  --seed "${SEED}" \
  --trainer "${trainer}" \
  --dataset-config-file configs/datasets/office_home_mtda.yaml \
  --config-file "${trainer_config}" \
  --source-domains "${source_domain}" \
  --target-domains "${targets[@]}" \
  --output-dir "${run_dir}" \
  "${cfg_opts[@]}" 2>&1 | tee "${run_dir}/console.log"

python scripts/maple_curriculum_mtda/collect_officehome_run.py \
  --run-dir "${run_dir}" --source "${SOURCE}" --variant "${VARIANT}" --seed "${SEED}"
