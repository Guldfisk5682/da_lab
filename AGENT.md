# AGENT.md

Last updated: 2026-07-12.

This is the operational handoff document for future Agents working on `da_lab`.
Keep it current. After each meaningful code change, experiment launch, result
collection, or major research decision, update this file before handing off.
Do not let it become a stale chat transcript.

## Current Role Of This Document

A new Agent should be able to read this file and answer five questions quickly:

1. How do I access the server and run experiments?
2. Which code paths matter for the current research line?
3. What is the current strongest baseline?
4. Which ideas have already failed or are low priority?
5. Which logs/results must be preserved locally for later paper writing?

## Current Branch And Theme

Active branch:

```text
maple-continuous-prompt-mtda
```

Current research setting:

```text
Dataset: Office-Home
Protocol: source-available closed-set SS-MTDA
Backbone: CLIP ViT-B/16
Main route: MaPLe-like multi-modal prompt tuning + frozen-CLIP pseudo labels
Current extension: post-PL old-student self-distillation for weak/mid-confidence regions
```

The earlier AD-CLIP-like/TSSP route is no longer the main line. Preserve it as
context, but do not spend more compute there unless the user explicitly asks.

## Server Access

SSH:

```bash
ssh lab-server
```

Remote repo:

```bash
cd ~/workspace/da_lab
```

Remote conda environment:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate coop-da
```

Local repo:

```bash
cd ~/dldic/da_lab
```

Local validation environment:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dlenv
```

Useful remote checks:

```bash
nvidia-smi
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
~/miniconda3/bin/screen -list
ps -ef | grep train.py | grep -v grep
```

Important GPU note from 2026-07-12:

```text
Both RTX 4090 cards can appear idle in nvidia-smi while PyTorch still reports
"CUDA driver initialization failed". Always test actual torch CUDA allocation
before launching a long run.
```

Torch CUDA smoke test:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate coop-da
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import torch
print(torch.cuda.is_available(), torch.cuda.device_count())
if torch.cuda.is_available():
    x = torch.ones(1, device='cuda')
    print(float(x.item()), torch.cuda.get_device_name(0))
PY
```

Use `CUDA_VISIBLE_DEVICES=1` only if that same test passes for GPU1. If a GPU is
idle in `nvidia-smi` but fails the torch test, do not start training there.

## Git Workflow

Local remote named `temp` points to the Gitee mirror. The server pulls from its
`origin`, also the Gitee mirror.

Local edit/check/push:

```bash
cd ~/dldic/da_lab
git status --short
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dlenv
python -m py_compile train.py trainers/maple_mtda.py trainers/maple_continuous_mtda.py
git add <files>
git commit -m "<message>"
git push temp maple-continuous-prompt-mtda
```

Remote pull/check:

```bash
ssh lab-server
cd ~/workspace/da_lab
git pull --ff-only origin maple-continuous-prompt-mtda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate coop-da
python -m py_compile train.py trainers/maple_mtda.py trainers/maple_continuous_mtda.py
```

Do not commit `.tmp/` or ad-hoc logs. Keep experiment outputs separated by
`METHOD_TAG` to avoid checkpoint contamination.

## Experiment Infrastructure Safeguards

Local infra hardening added on 2026-07-12 (not yet committed at this handoff):

```text
scripts/experiment_guard.py
tests/test_experiment_infra.py
```

The active ContinuousSharedProj launcher now writes
`experiment_manifest.json` before training. Exact same-config restarts are
allowed; changing config while reusing the output directory is rejected.
Historical non-empty output directories do not have manifests. Resume one only
after manually checking its config and setting:

```bash
export ALLOW_LEGACY_OUTPUT_DIR=1
```

Do not use that override for new experiments. Prefer a new `METHOD_TAG`.

`train.py` appends invocation/git/CUDA metadata to `run_metadata.jsonl` and uses
deterministic PyTorch/CuDNN settings for fixed-seed runs. The MaPLe checkpoint
loaders now reject unknown missing/unexpected keys while allowing regenerated
prompt prefix/suffix buffers.

The MaPLe result collector now requires all three target accuracies before it
reports a macro average. Incomplete runs are marked `Macro Avg=NA`,
`Complete=no`. Use exact method filtering for paper tables:

```bash
python scripts/maple_mtda/collect_officehome_results.py \
  --seeds 42 \
  --method-tags maple_continuous_shared_mtda_pl03_seed42
```

`--allow-incomplete` exists only for debugging and must not be used for paper
numbers.

## Training Entrypoints

Current main training script:

```bash
bash scripts/maple_continuous_shared_mtda/run_officehome_all.sh
```

Single-source run:

```bash
bash scripts/maple_continuous_shared_mtda/run_officehome_one.sh A 42
```

Result collection:

```bash
python scripts/maple_mtda/collect_officehome_results.py --seeds 42
```

The script maps sources automatically:

```text
A -> C P R
C -> A P R
P -> A C R
R -> A C P
```

Current strongest baseline command shape:

```bash
cd ~/workspace/da_lab
source ~/miniconda3/etc/profile.d/conda.sh
conda activate coop-da
export CUDA_VISIBLE_DEVICES=<usable_gpu>
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export METHOD_TAG=maple_continuous_shared_mtda_pl03_seed42
export SEED=42
export EXTRA_OPTS="TRAINER.MAPLE_MTDA.LAMBDA_PL 0.3 TRAINER.MAPLE_MTDA.PL_THRESHOLD 0.7 TRAINER.MAPLE_MTDA.PL_STUDENT_THRESHOLD 0.7 TRAINER.MAPLE_MTDA.PL_USE_STUDENT_LOW_CONF_MASK True"
bash scripts/maple_continuous_shared_mtda/run_officehome_all.sh
python scripts/maple_mtda/collect_officehome_results.py --seeds 42
```

Current post-PL self-distillation experiment command shape:

```bash
cd ~/workspace/da_lab
source ~/miniconda3/etc/profile.d/conda.sh
conda activate coop-da
export CUDA_VISIBLE_DEVICES=<usable_gpu>
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export METHOD_TAG=maple_continuous_shared_mtda_pl03_sdpost1_seed42
export POST_INIT_METHOD_TAG=maple_continuous_shared_mtda_pl03_seed42
export POST_INIT_LOAD_EPOCH=5
export SEED=42
export EXTRA_OPTS="OPTIM.MAX_EPOCH 1 OPTIM.LR 0.0005 OPTIM.WARMUP_EPOCH 0 TRAINER.MAPLE_MTDA.LAMBDA_PL 0.3 TRAINER.MAPLE_MTDA.SELF_DISTILL.ENABLED True TRAINER.MAPLE_MTDA.SELF_DISTILL.LAMBDA 0.03 TRAINER.MAPLE_MTDA.SELF_DISTILL.TEMPERATURE 2.0 TRAINER.MAPLE_MTDA.SELF_DISTILL.OLD_CONF_LOW 0.45 TRAINER.MAPLE_MTDA.SELF_DISTILL.OLD_CONF_HIGH 0.8"
bash scripts/maple_continuous_shared_mtda/run_officehome_all.sh
python scripts/maple_mtda/collect_officehome_results.py --seeds 42
```

`POST_INIT_METHOD_TAG` is handled by
`scripts/maple_continuous_shared_mtda/run_officehome_one.sh`; it resolves the
per-source checkpoint path automatically.

## Current Core Code Map

Main configs:

```text
train.py
configs/trainers/ContinuousSharedProjMaPLeMTDA/vit_b16.yaml
configs/datasets/office_home_mtda.yaml
```

Current trainer/model implementation:

```text
trainers/maple_mtda.py
trainers/maple_continuous_mtda.py
trainers/mtda_base.py
trainers/checkpoint_utils.py
```

Important classes/functions:

```text
CustomMaPLeMTDA.forward_train
CustomMaPLeMTDA._pseudo_label_loss
CustomMaPLeMTDA._weak_pseudo_label_loss
CustomMaPLeMTDA._self_distill_loss
CustomMaPLeMTDA.build_self_distill_old_model
MaPLeMTDA._maybe_load_post_init_model
ContinuousSharedProjMaPLeMTDA.build_model
ContinuousSharedProjPromptLearner
```

Experiment scripts:

```text
scripts/maple_continuous_shared_mtda/run_officehome_one.sh
scripts/maple_continuous_shared_mtda/run_officehome_all.sh
scripts/maple_mtda/collect_officehome_results.py
scripts/maple_mtda/analyze_officehome_pl_blindspots.py
```

Reference code cloned for literature inspection:

```text
.tmp/research_code/DIFO-Plus
```

## Model And PL Design Snapshot

MaPLe correction:

```text
Original MaPLe has trainable text prompt tokens for J layers and one projection
per corresponding visual layer. The projected visual prompts are not an extra
independent prompt parameter family; they couple the trainable text prompts to
visual gradients through projection layers.
```

Current main model:

```text
ContinuousSharedProjMaPLeMTDA
- Continuous text prompt tokens are propagated through the text tower.
- Visual branch uses one shared deep visual projection reused across deep layers.
- Trainable params: prompt_learner.ctx, shallow proj, shared deep projection.
```

Current clean PL:

```text
Teacher: frozen zero-shot CLIP with template "a photo of a {}."
Mask: teacher_conf >= 0.7 and student_conf < 0.7
Loss: hard CE to frozen CLIP argmax
Weight: lambda_pl = 0.3
```

Current post-PL self-distillation idea:

```text
Stage 1: train strongest PL baseline normally.
Stage 2: initialize from PL checkpoint, freeze an old-student snapshot, continue
1 epoch with source CE + clean PL + small KL to old student on target samples
whose old-student confidence lies in [0.45, 0.8).
```

This borrows only the old-policy / trust-region intuition from PPO. It is not an
RL method and should be described as policy-style post-training self-distillation
or old-student KL regularization.

## Strongest Baselines And Results

Current strongest baseline:

```text
ContinuousSharedProjMaPLeMTDA + PL03, seed42: 84.69
```

Per-source seed42:

```text
A2CPR: C 71.55 / P 91.19 / R 90.80 => 84.51
C2APR: A 84.47 / P 91.89 / R 90.84 => 89.07
P2ACR: A 84.26 / C 71.75 / R 91.05 => 82.35
R2ACP: A 84.22 / C 72.10 / P 92.21 => 82.84
Overall: 84.69
```

Seed1 same method:

```text
ContinuousSharedProjMaPLeMTDA + PL03, seed1: 84.60
```

Original MaPLe-like PL comparison, seed42:

```text
MaPLeMTDA + PL03 seed42: about 84.37
ContinuousSharedProjMaPLeMTDA + PL03 seed42: 84.69
```

## Tried Failed Or Low-Priority Cases

Continuous prompt without shared projection:

```text
Continuous prompt direction was plausible, but the strongest variant so far is
continuous prompt + shared visual projection + PL, not every continuous variant.
```

Cosine PL weight decay:

```text
Tag: maple_continuous_shared_mtda_pl03_cosdecay_seed42
Schedule: lambda_pl 0.3 -> 0.0 by cosine over 5 epochs
Result: 84.06, worse than constant PL03 seed42 84.69 by -0.63
Conclusion: plain PL weight decay is not suitable; late teacher signal remains useful.
```

Cosine decay per-source:

```text
A2CPR 83.68 vs 84.51 baseline
C2APR 88.32 vs 89.07 baseline
P2ACR 81.50 vs 82.35 baseline
R2ACP 82.73 vs 82.84 baseline
```

Style-gap token / gap-conditioned PL:

```text
Tag: maple_gapctx_mtda_pl03_seed42
Result did not beat the strongest baseline.
Log inspection suggested the gap token mostly acted as noisy perturbation.
Do not continue DIFO-style gap imitation unless the user explicitly reopens it.
```

RFC-style weak hard PL branch:

```text
Tag: maple_continuous_shared_mtda_pl03_weakrfc005_seed42
A2CPR dropped to 84.17 vs 84.51 baseline, and later diagnostics explained why:
the student is already stronger than frozen CLIP on many weak classes, so lower
teacher thresholds can inject noise instead of useful supervision.
```

Full DIFO imitation:

```text
Deferred. DIFO changes too many modules and would move away from the current
minimal MaPLe-like route.
```

Unfreezing CLIP:

```text
Rejected by user because it departs from parameter-efficient tuning.
```

## Weak-Class Diagnostics

Diagnostic script:

```bash
python scripts/maple_mtda/analyze_officehome_pl_blindspots.py --seed 42 --method-tag maple_continuous_shared_mtda_pl03_seed42 --load-epoch 5
```

Remote diagnostic output:

```text
~/workspace/da_lab/results/pl_blindspots_seed42/
~/workspace/da_lab/logs/pl_blindspots_seed42.log
```

Key aggregate findings:

```text
all classes:
teacher_acc 0.8077
student_acc 0.8364
both_low_rate 0.1973
teacher_true_prob 0.7215
student_true_prob 0.7701

rfc_teacher_weak classes:
teacher_acc 0.5974
student_acc 0.7137
both_low_rate 0.3449
teacher_true_prob 0.5004
student_true_prob 0.6267

both_low_top_fraction classes:
teacher_acc 0.5932
student_acc 0.6415
both_low_rate 0.4374
teacher_true_prob 0.4855
student_true_prob 0.5375
```

Recurring weak classes include:

```text
toys, marker, folder, clipboards, mop, sink
```

Meaning of `true_prob`:

```text
Softmax probability assigned to the ground-truth class. It is diagnostic only
because Office-Home labels are used during analysis. It must not be used in training.
```

Interpretation:

```text
Frozen CLIP is weak on these classes, but the trained student is often much
stronger. This is why weak hard pseudo-labeling from CLIP is risky, and why the
current direction shifted toward old-student soft self-distillation.
```

## Current In-Progress Experiment

Code commit:

```text
76ea11d Add post-PL self-distillation branch
```

Current intended run:

```text
METHOD_TAG=maple_continuous_shared_mtda_pl03_sdpost1_seed42
```

Settings:

```text
Post-init from: maple_continuous_shared_mtda_pl03_seed42 epoch 5
Post-training epochs: 1
LR: 0.0005
Clean PL: lambda 0.3 unchanged
Self-distill KL: lambda 0.03, T=2.0, old_conf in [0.45, 0.8)
```

As of the latest read-only check on 2026-07-12, training has not completed. The
previous run and GPU1 watcher have exited; there is no active self-distill
process and no checkpoint/result for this method. Historical paths are:

```text
watcher log: ~/workspace/da_lab/logs/maple_continuous_shared_mtda_pl03_sdpost1_seed42_gpu1_watcher.log
run log:     ~/workspace/da_lab/logs/maple_continuous_shared_mtda_pl03_sdpost1_seed42_run.log
collect log: ~/workspace/da_lab/logs/maple_continuous_shared_mtda_pl03_sdpost1_seed42_collect.log
```

GPU1 passed a fresh real PyTorch allocation test after the watcher exited. Do
not relaunch the self-distill run until the no-KL continuation control is given
a separate method tag and scheduled alongside it.

### Minimal teacher-handoff pilot approved on 2026-07-12

The first causal pilot uses only `A2CPR`, seed42, initialized from the PL03
epoch-5 checkpoint. All groups continue for one epoch with LR `5e-4`, no warmup,
and clean PL unchanged at `0.3`:

```text
maple_cshared_pl03_post1_nokl_seed42
maple_cshared_pl03_post1_sdall003_seed42
maple_cshared_pl03_post1_sdhandoff003_seed42
```

The handoff mask is intentionally minimal and complementary to clean CLIP PL:

```text
old_student_conf >= 0.7 and frozen_CLIP_conf < 0.7
```

The handoff branch uses old-student soft KL with `lambda=0.03`, `T=2`. The
`sdall` control applies the same KL to all target samples. Compare handoff first
against the exact no-KL continuation, then against all-sample KD. Do not tune
thresholds from target labels in this pilot.

Launch status:

```text
Started: 2026-07-12 21:32 CST
Remote commit: f0cabc9
GPU: CUDA_VISIBLE_DEVICES=1 (real allocation smoke test passed)
Screen: sd_handoff_pilot
Runner log: logs/maple_cshared_pl03_post1_handoff_pilot_seed42_runner.log
Order: no-KL -> all-sample KD -> teacher-handoff KD
Initial ETA: 2026-07-12 21:50-22:00 CST
```

Pilot result (completed on 2026-07-12):

```text
Stage-1 PL03 checkpoint: C 71.55 / P 91.19 / R 90.80 => 84.51
No-KL continuation:      C 70.93 / P 90.70 / R 90.75 => 84.13
All-sample KD, 0.03:     C 70.97 / P 90.70 / R 90.77 => 84.15
Teacher-handoff KD:      C 71.00 / P 90.67 / R 90.80 => 84.16
```

Handoff minus no-KL is only `+0.03`; handoff minus all-sample KD is `+0.01`;
handoff remains `-0.36` below the stage-1 checkpoint. Average handoff coverage
reported during training was about `10.5%`, and its weighted KD loss was only
about `1e-4`, versus clean weighted PL around `2.5e-2`. Conclusion: continuing
training at the current settings is harmful, while KD provides only a tiny
anti-drift signal. Do not expand this exact setting to all four sources.

Important metric caveat: selected-confidence/agreement meters average per-batch
zeros when a batch selects no samples, so their printed running averages are
biased downward. Coverage and final accuracies are valid. Fix the selected
statistics to use selected-count-weighted accumulation before relying on them.

Local archived logs:

```text
.tmp/agent_handoff/logs/self_distill_pilot/
```

## Local `./tmp` / `.tmp` Archival Policy

The repo currently has untracked `.tmp/` material. The user specifically wants
important records to be preserved locally under a temporary archive area for
handoff and paper-writing. Use this convention going forward:

```text
.tmp/agent_handoff/
```

Recommended subdirectories:

```text
.tmp/agent_handoff/results/
.tmp/agent_handoff/logs/
.tmp/agent_handoff/notes/
.tmp/agent_handoff/papers/
```

Preserve these records locally when generated remotely:

```text
results/officehome_maple_mtda_seed42.csv
results/officehome_maple_mtda_seed42.md
results/pl_blindspots_seed42/*.csv
logs/pl_blindspots_seed42.log
logs/maple_continuous_shared_mtda_pl03_cosdecay_seed42_run.log
logs/maple_continuous_shared_mtda_pl03_cosdecay_seed42_collect.log
logs/maple_continuous_shared_mtda_pl03_sdpost1_seed42*_watcher.log
logs/maple_continuous_shared_mtda_pl03_sdpost1_seed42_run.log
logs/maple_continuous_shared_mtda_pl03_sdpost1_seed42_collect.log
```

Use `scp` or `rsync` from local WSL when needed, for example:

```bash
cd ~/dldic/da_lab
mkdir -p .tmp/agent_handoff/results .tmp/agent_handoff/logs
scp lab-server:~/workspace/da_lab/results/officehome_maple_mtda_seed42.* .tmp/agent_handoff/results/
scp lab-server:~/workspace/da_lab/logs/pl_blindspots_seed42.log .tmp/agent_handoff/logs/
rsync -av lab-server:~/workspace/da_lab/results/pl_blindspots_seed42/ .tmp/agent_handoff/results/pl_blindspots_seed42/
```

Do not commit `.tmp/agent_handoff/` unless the user explicitly asks. It is a
local evidence archive, not source code.

## Literature Context Already Discussed

Useful reference directions:

```text
SHOT: source-free DA, pseudo-label/self-training issues, weak categories.
S3DA: dynamic pseudo-label thresholding ideas; useful conceptually but not yet implemented.
RFC: explicit weak-class identification; hard weak PL was not successful here.
DIFO/DIFO-Plus: gap-region idea inspected, but direct structural imitation is too heavy for now.
```

Subagents previously helped with literature review. If the context window gets
large again, use persistent subagents for paper reading and dirty analysis, but
keep this file as the canonical compact handoff.

## Update Rules For Future Agents

Update this file when any of the following happens:

```text
- A new branch becomes active.
- A new method tag is launched.
- A run finishes and produces numbers.
- A run fails for a nontrivial reason.
- A result changes the research direction.
- A new diagnostic script or important log is created.
- A remote GPU/server issue affects experiment scheduling.
```

When updating, prefer compact factual notes:

```text
method tag, seed, command/config, result, conclusion, log/result path
```

Do not paste long terminal output. Record paths and the few numbers needed to
reconstruct the decision.
