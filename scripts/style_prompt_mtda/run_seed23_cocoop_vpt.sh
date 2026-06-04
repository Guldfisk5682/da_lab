#!/bin/bash

set -euo pipefail

DEFAULT_REPO_ROOT="/workspace/txc/da_lab"
AUTO_REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
if [ -d "${DEFAULT_REPO_ROOT}" ]; then
  REPO_ROOT="${REPO_ROOT:-${DEFAULT_REPO_ROOT}}"
else
  REPO_ROOT="${REPO_ROOT:-${AUTO_REPO_ROOT}}"
fi

cd "${REPO_ROOT}"

SEEDS="${SEEDS:-2 3}"
RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_VPT="${RUN_VPT:-1}"
VPT_METHOD_TAG="${VPT_METHOD_TAG:-cocoop_vpt_ctx8_d1}"
VPT_EXTRA_OPTS="${VPT_EXTRA_OPTS:-TRAINER.COCOOP_VPT_MTDA.N_VCTX 8 TRAINER.COCOOP_VPT_MTDA.VISION_PROMPT_DEPTH 1 TRAINER.COCOOP_VPT_MTDA.VCTX_POSITION append}"

echo "==============================================="
echo "Office-Home MTDA seed sweep"
echo "Seeds: ${SEEDS}"
echo "Run baseline CoCoOpMTDA: ${RUN_BASELINE}"
echo "Run CoCoOpVPTMTDA: ${RUN_VPT}"
echo "VPT method tag: ${VPT_METHOD_TAG}"
echo "VPT extra opts: ${VPT_EXTRA_OPTS}"
echo "==============================================="

for seed in ${SEEDS}; do
  if [ "${RUN_BASELINE}" = "1" ]; then
    echo "[seed ${seed}] Running CoCoOpMTDA"
    SEED="${seed}" \
      bash "${REPO_ROOT}/scripts/style_prompt_mtda/run_officehome_all.sh" cocoop_mt
  fi

  if [ "${RUN_VPT}" = "1" ]; then
    echo "[seed ${seed}] Running CoCoOpVPTMTDA"
    SEED="${seed}" \
      METHOD_TAG="${VPT_METHOD_TAG}" \
      EXTRA_OPTS="${VPT_EXTRA_OPTS}" \
      bash "${REPO_ROOT}/scripts/style_prompt_mtda/run_officehome_all.sh" cocoop_vpt
  fi
done

python "${REPO_ROOT}/scripts/style_prompt_mtda/collect_officehome_results.py" --seeds ${SEEDS}
