# AGENT.md

Last updated: 2026-07-10.

This file is the handoff note for future agents working on `da_lab`. Keep it
short, current, and operational. Do not turn it back into a long historical
prompt dump.

## Current Focus

Active branch:

```text
clip-vpt-mtda
```

Current research setting:

```text
Dataset: Office-Home
Protocol: source-available closed-set SS-MTDA
Backbone: CLIP ViT-B/16
Current strongest baseline: MaPLeMTDA + frozen-CLIP pseudo-label loss
```

Most important finding so far:

```text
Frozen zero-shot CLIP pseudo-labeling on target batches is the clearest positive
signal. It improves both CLIPTSSPMTDA and MaPLeMTDA. MaPLeMTDA + PL currently
beats CLIPTSSPMTDA + PL, so future model work should probably use a MaPLe-like
multi-modal prompt backbone rather than continuing to over-tune pure TSSP.
```

Key seed1 Office-Home SS-MTDA numbers:

```text
CoCoOpMTDA baseline:             about 83.40
MaPLeMTDA:                       about 83.67
CLIPTSSPMTDA PairGap + SGD PL03: about 84.07
MaPLeMTDA + PL03:                about 84.45
```

Current recommended main control:

```text
MaPLeMTDA + PL(lambda=0.3)
```

Current recommended comparison:

```text
CLIPTSSPMTDA PairGap + SGD PL(lambda=0.3)
```

## Golden Rules

- Do not download datasets or model weights locally.
- Do not run long training locally.
- Real data and real training happen on the remote server.
- Local work should be code edits, docs, scripts, dry-runs, and lightweight syntax checks.
- Preserve original CoOp/CoCoOp trainers unless the user explicitly asks otherwise.
- Keep new experiment outputs separated by method tags to avoid checkpoint/resume contamination.
- Do not revive old failed routes unless the user explicitly asks for a targeted ablation.

## Paths And Backups

Local repo:

```text
/home/txc_king/dldic/da_lab
```

Remote repo:

```text
~/workspace/da_lab
```

Remote prepared Office-Home data:

```text
~/workspace/da_lab/data/office_home
```

Latest local result backups:

```text
/home/txc_king/dldic/da_lab/results/officehome_clip_tssp_seed1_2026-07-10.md
/home/txc_king/dldic/da_lab/results/officehome_clip_tssp_seed1_2026-07-10.csv
/home/txc_king/dldic/da_lab/results/officehome_clip_tssp_seed1_summary_2026-07-10.csv
/home/txc_king/dldic/da_lab/results/officehome_maple_mtda_seed1_2026-07-10.md
/home/txc_king/dldic/da_lab/results/officehome_maple_mtda_seed1_2026-07-10.csv
```

Latest remote generated results:

```text
~/workspace/da_lab/results/officehome_clip_tssp_seed1.md
~/workspace/da_lab/results/officehome_clip_tssp_seed1.csv
~/workspace/da_lab/results/officehome_clip_tssp_seed1_summary.csv
~/workspace/da_lab/results/officehome_maple_mtda_seed1.md
~/workspace/da_lab/results/officehome_maple_mtda_seed1.csv
```

Older recovered backups also exist under local `results/`, especially:

```text
officehome_key_results_recovered_2026-07-08.*
officehome_maple_mtda_seed1_2026-07-08.*
```

## Remote Server Workflow

SSH:

```bash
ssh lab-server
```

Remote environment:

```bash
cd ~/workspace/da_lab
source ~/miniconda3/etc/profile.d/conda.sh
conda activate coop-da
```

GPU/screen checks:

```bash
nvidia-smi
~/miniconda3/bin/screen -list
```

Notes:

```text
The server has two RTX 4090 GPUs.
Prefer explicit CUDA_VISIBLE_DEVICES before training.
Recent stable runs used CUDA_VISIBLE_DEVICES=1.
Use PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True for long jobs.
screen is installed at ~/miniconda3/bin/screen.
```

If a screen-level log is empty, check the per-task Dassl log instead:

```bash
tail -f ~/workspace/da_lab/output/officehome_mtda/maple_mtda_pl03/A2CPR/seed1/log.txt
```

## Git Workflow

Default workflow:

```text
1. Edit locally.
2. Commit locally.
3. Push to temp/Gitee.
4. SSH to server.
5. Pull on server.
6. Run training in screen.
```

Local:

```bash
cd /home/txc_king/dldic/da_lab
git status --short
git add <files>
git commit -m "<message>"
git push temp clip-vpt-mtda
```

Remote:

```bash
ssh lab-server
cd ~/workspace/da_lab
git pull --ff-only origin clip-vpt-mtda
```

Important:

```text
Remote origin currently points to the Gitee/temp mirror.
Do not assume GitHub origin works from the server.
Local .tmp/ contains temporary reference repos/materials and must not be committed.
```

## Active Training Entrypoints

MaPLe baseline:

```bash
bash scripts/maple_mtda/run_officehome_all.sh
python scripts/maple_mtda/collect_officehome_results.py
```

MaPLe + PL(lambda=0.3):

```bash
METHOD_TAG=maple_mtda_pl03 \
EXTRA_OPTS="TRAINER.MAPLE_MTDA.LAMBDA_PL 0.3 TRAINER.MAPLE_MTDA.PL_THRESHOLD 0.7 TRAINER.MAPLE_MTDA.PL_STUDENT_THRESHOLD 0.7 TRAINER.MAPLE_MTDA.PL_USE_STUDENT_LOW_CONF_MASK True" \
bash scripts/maple_mtda/run_officehome_all.sh

python scripts/maple_mtda/collect_officehome_results.py
```

Example remote screen launch for MaPLe + PL:

```bash
cd ~/workspace/da_lab
mkdir -p logs
LOG=logs/maple_mtda_pl03_$(date +%Y%m%d_%H%M%S).log
CUDA_VISIBLE_DEVICES=1 ~/miniconda3/bin/screen -dmS maple_pl bash -lc \
  "cd ~/workspace/da_lab && \
   source ~/miniconda3/etc/profile.d/conda.sh && \
   conda activate coop-da && \
   export DATA=~/workspace/da_lab/data && \
   export CUDA_VISIBLE_DEVICES=1 && \
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
   METHOD_TAG=maple_mtda_pl03 \
   EXTRA_OPTS=\"TRAINER.MAPLE_MTDA.LAMBDA_PL 0.3 TRAINER.MAPLE_MTDA.PL_THRESHOLD 0.7 TRAINER.MAPLE_MTDA.PL_STUDENT_THRESHOLD 0.7 TRAINER.MAPLE_MTDA.PL_USE_STUDENT_LOW_CONF_MASK True\" \
   bash scripts/maple_mtda/run_officehome_all.sh && \
   python scripts/maple_mtda/collect_officehome_results.py" \
  > "${LOG}" 2>&1
```

CLIPTSSPMTDA PairGap + PL/KL sweep:

```bash
bash scripts/clip_tssp_mtda/run_pair_gap_pl_kl_sgd_sweep.sh
python scripts/clip_tssp_mtda/collect_officehome_results.py --seed 1
```

Completed TSSP SGD sweep tags:

```text
clip_tssp_pair_gap_sgd_pl01
clip_tssp_pair_gap_sgd_pl02
clip_tssp_pair_gap_sgd_pl03
clip_tssp_pair_gap_sgd_pl02_kl001
clip_tssp_pair_gap_sgd_pl02_kl005
clip_tssp_pair_gap_sgd_pl02_kl010
```

Output-root gotcha:

```text
MaPLe scripts use:    output/officehome_mtda/...
CLIPTSSP scripts use: output/office_home_mtda/...
```

## What Worked

Pseudo-label target regularization:

```text
Teacher: frozen zero-shot CLIP logits.
Student: current model target logits.
Loss: CE(student logits, teacher argmax), masked by teacher confidence and,
optionally, low student confidence.
Best current weight: lambda_pl=0.3.
```

PairGap TSSP:

```text
Extract all 12 frozen CLIP ViT hidden states.
Per layer: concat token mean/std -> MLP -> text-space style token.
Compress adjacent layer tokens into six source/target/gap groups.
Prompt order around [source, gap, target] was better than no-gap variants.
This helped, but did not beat MaPLe + PL.
```

MaPLeMTDA:

```text
MaPLe-style multi-modal prompt learning is a strong backbone in this SS-MTDA
setting. Once target PL is added, it becomes the current best seed1 control.
```

## Failed Or Low-Value Routes

Treat these as lessons, not active tasks:

- Office-31 V0/V1 shallow hidden-state style fusion, final feature gate, legacy
  GSPA hidden-state cross-style swap, and gate ablations did not produce a
  durable direction. Keep them archived/history-only.
- Pure CoCoOp text-side target style prompt modulation gave tiny and unstable
  gains. Do not keep tuning style queues, beta scalars, or target-style text
  prompt bias as the main story.
- Persistent VCTX on CoCoOp/CLIP gave small gains, but insert-position, deeper
  replacement prompts, large context counts, domain-text residuals, and ViaPT-like
  instance prompts were not reliable enough to be the core method.
- TSSP image tokens (`img12/img6/img4`) underperformed PairGap; do not revive
  image-token injection unless there is a new reason.
- Target entropy and class-balance information maximization did not become a
  stable main objective.
- KL to frozen CLIP logits was weaker than PL. It can stay as an ablation, but
  should not be the default objective.
- AdamW at tested settings did not beat the stronger SGD PL runs. Use optimizer
  changes as controlled ablations only.

## Direction For Future Agents

If asked to improve the method, start from this hierarchy:

```text
1. Keep MaPLeMTDA + PL03 as the strongest current control.
2. Compare any new idea against MaPLeMTDA + PL03 and CLIPTSSPMTDA PairGap + PL03.
3. Prefer MaPLe-like multi-modal prompt backbones plus MTDA-native target/domain
   modules.
4. Keep frozen-CLIP pseudo-labeling as a first-class target-side signal.
5. Avoid adding many tiny prompt variants without a clear hypothesis.
```

Potential next ideas:

```text
MaPLe + MTDA domain/style/gap tokens.
MaPLe + stronger but still conservative pseudo-label scheduling.
Multi-seed validation for MaPLe + PL03 and the best TSSP + PL03.
DomainNet transfer only after Office-Home direction is stable.
```

## Legacy Files And Infrastructure

Archived/legacy routes may still exist in the repo:

```text
archive/v0_v1_ablation/
archive/legacy_gspa/
trainers/style_prompt_mtda.py
trainers/cocoop_vpt_mtda.py
trainers/clip_tssp_mtda.py
scripts/gspa_legacy_ablation/
scripts/style_prompt_mtda/
scripts/clip_tssp_mtda/
```

Do not delete them casually; they preserve reproducibility and old comparisons.
But do not treat them as current research instructions.

Useful infrastructure to preserve:

```text
datasets/office_home_mtda.py
trainers/mtda_base.py
scripts/datasets/download_officehome.sh
scripts/datasets/verify_officehome_layout.sh
scripts/maple_mtda/
scripts/clip_tssp_mtda/collect_officehome_results.py
scripts/clip_tssp_mtda/plot_tensorboard_curves.py
```

## Minimal Sanity Checks

Local, if dependencies exist:

```bash
python train.py --help
```

Remote:

```bash
cd ~/workspace/da_lab
source ~/miniconda3/etc/profile.d/conda.sh
conda activate coop-da
python train.py --help
```

Before launching long jobs:

```bash
git status --short
nvidia-smi
~/miniconda3/bin/screen -list
```

## Maintenance Rule

When new experiments finish, update only these parts:

```text
Current Focus
Paths And Backups
Active Training Entrypoints
What Worked
Failed Or Low-Value Routes
Direction For Future Agents
```

Keep this file concise. If detailed experiment tables are needed, store them in
`results/` or a dedicated `docs/` file and link the path here.
