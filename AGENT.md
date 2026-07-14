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

Literature freeze decision on 2026-07-12:

```text
Pause new training experiments after the negative post-stage KD pilot.
Complete human reading and a route brainstorm before implementing another method.
Generic second-stage KD, teacher replacement, dual-teacher filtering, and micro
prompt variants are crowded; do not resume them through hyperparameter sweeps.
```

The broad review cache is local and untracked:

```text
.tmp/literature_review/README.md
.tmp/literature_review/literature_matrix.md
.tmp/literature_review/opportunity_report.md
.tmp/literature_review/source_manifest.md
.tmp/literature_review/paper_notes/
```

Current literature-derived opportunity, pending human review:

```text
Per-(target,class), directional and conflict-aware cross-target reliability
transfer with abstention/negative-transfer control, built on a compact shared
prompt backbone. Strong collisions to distinguish from: U3CF global pooled
prototypes, D-CGCT domain curriculum/graph co-teaching, CRPL source-prompt
aggregation/OT, DUET agreement filtering, and COSMo image-conditioned bias.
```

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

## Curriculum + Reliable Replay Pilot (2026-07-13)

The user approved formal implementation and launch of the first A2CPR seed42
training-strategy pilot. The controlled groups are:

```text
2. easy-to-hard sequential targets, no replay
3. easy-to-hard sequential targets, Top-K8 replay
4. hard-to-easy sequential targets, Top-K8 replay
```

Implementation uses `CurriculumContinuousSharedProjMaPLeMTDA`. It preserves
the Joint PL03 baseline's total optimizer steps, source batches, target-image
forwards, scheduler and PL coefficient. Each step consumes three micro-batches
from the active target domain. Stages divide the full five-epoch step budget
into three equal parts and may cross epoch boundaries.

Replay protocol:

```text
admission: student/frozen-CLIP argmax agreement
thresholds: student >= 0.7 and CLIP >= 0.7
selection: top 8 per fitted-domain x predicted-class; no threshold backfill
labels: frozen at the stage boundary
training: independent source-augmentation loader and hard CE, lambda 1.0
exposure: replay loader is traversed at most once per subsequent stage
```

Audit files written into each output directory:

```text
curriculum_stage_audit.jsonl
replay_bank_audit.jsonl
```

The trainer evaluates all target domains at each stage boundary and records
bank class coverage, confidence, replay exposure/loss contribution, and prior
bank label stability. The static easy-to-hard order must be supplied explicitly
and must be chosen from an unlabeled source-only difficulty probe, never target
accuracy.

Source-only seed42 entropy ranking completed on 2026-07-13:

```text
real_world: normalized entropy 0.0874107
product:    normalized entropy 0.0874928
clipart:    normalized entropy 0.2388312
E2H: real_world -> product -> clipart
H2E: clipart -> product -> real_world
artifact: results/curriculum_a2cpr/sourceonly_difficulty_seed42.json
```

Experiment 2 was launched on GPU0 at 2026-07-13 16:55 CST:

```text
method: maple_cshared_pl03_e2h_noreplay_seed42
screen: run
log: logs/maple_cshared_pl03_e2h_noreplay_seed42_run.log
output: output/officehome_mtda/maple_cshared_pl03_e2h_noreplay_seed42/A2CPR/seed42
initial measured throughput: about 0.62 sec/step
initial completion estimate: around 17:27 CST
```

Experiment 3 was launched on GPU1 at 2026-07-13 17:06 CST:

```text
method: maple_cshared_pl03_e2h_topk8replay_seed42
screen: e2h_replay
log: logs/maple_cshared_pl03_e2h_topk8replay_seed42_run.log
output: output/officehome_mtda/maple_cshared_pl03_e2h_topk8replay_seed42/A2CPR/seed42
order: real_world -> product -> clipart
replay: Top-K8 per domain x predicted class, lambda 1.0
initial completion estimate including boundary audits: around 17:42-17:48 CST
```

Experiments 2 and 3 completed on 2026-07-13:

```text
Joint PL03 baseline: C 71.55 / P 91.19 / R 90.80 => 84.51
E2H no replay:      C 70.61 / P 90.67 / R 90.36 => 83.88
E2H Top-K8 replay:  C 70.77 / P 91.03 / R 90.77 => 84.19
```

Replay improved E2H by +0.31 macro but remained -0.32 below Joint. Replay
bank sizes were 513 after real_world and 1022 after product; frozen-label
stability was 1.0 at both later audits. The replay loader traversed 512 images
in stage 2 and 1020 in stage 3, once per stage as intended.

Experiment 4 launched on GPU1 at 2026-07-13 17:52 CST:

```text
method: maple_cshared_pl03_h2e_topk8replay_seed42
screen: h2e_replay
log: logs/maple_cshared_pl03_h2e_topk8replay_seed42_run.log
order: clipart -> product -> real_world
```

Experiment 4 completed on 2026-07-13 with a positive result:

```text
H2E Top-K8 replay stage 1: C 70.81 / P 91.08 / R 90.59 => 84.16
H2E Top-K8 replay stage 2: C 71.91 / P 91.01 / R 90.64 => 84.52
H2E Top-K8 replay final:   C 72.35 / P 90.94 / R 91.30 => 84.86
```

Final H2E Top-K8 is +0.35 over Joint PL03, +0.67 over E2H Top-K8, and
+0.98 over E2H no replay. Relative to Joint, C improves +0.80 and R improves
+0.50 while P drops -0.25. The stage macro average rises monotonically.

The hard-first replay bank contained 475 clipart samples and then 984 cumulative
clipart+product samples. Prior-bank frozen-label stability remained 1.0. Mean
weighted replay losses (averaged over every optimizer step, including inactive
steps) were 0.0831 in stage 2 and 0.1252 in stage 3, substantially larger than
E2H's 0.0396 and 0.0612. This suggests hard-domain replay is more challenging
and potentially more informative, not merely larger.

Interpretation limit: the completed controls establish that H2E is better than
E2H under the same replay rule, but there is no H2E-no-replay control yet.
Therefore do not attribute the gain to hard-first ordering alone. The smallest
next causal run is H2E no replay on A2CPR seed42; if the interaction survives,
then repeat the selected pair on additional seeds before expanding datasets.

Local archives:

```text
.tmp/agent_handoff/logs/curriculum_pilot/maple_cshared_pl03_h2e_topk8replay_seed42_run.log
.tmp/agent_handoff/results/curriculum_pilot/h2e_stage_audit.jsonl
.tmp/agent_handoff/results/curriculum_pilot/h2e_bank_audit.jsonl
```

The approved H2E no-replay causal control was launched on GPU1 at 2026-07-13
18:34 CST:

```text
method: maple_cshared_pl03_h2e_noreplay_seed42
screen: h2e_noreplay
log: logs/maple_cshared_pl03_h2e_noreplay_seed42_run.log
order: clipart -> product -> real_world
replay: disabled
expected completion: around 19:08-19:14 CST
```

Stage-local optimizer/scheduler reset controls were approved and implemented
while H2E no replay was running. The reset mode:

```text
TRAINER.MAPLE_MTDA.CURRICULUM.RESET_OPTIM_PER_STAGE=True
TRAINER.MAPLE_MTDA.CURRICULUM.STAGE_VIRTUAL_EPOCHS=5
```

At every stage boundary it rebuilds SGD (clearing momentum/state) and the
warmup+cosine scheduler. Each 1010-step domain stage experiences the same five
LR segments, 202 steps each:

```text
1e-5, 0.0035, 0.0031657797, 0.0022907797, 0.0012092203
```

Across three stages, each LR occurs 606 times, exactly matching the original
five-epoch global schedule's LR histogram and integrated LR. Thus total steps,
target/source exposure, LR values, and integrated LR remain controlled; only
the assignment of LR to domains and cross-stage optimizer state are removed.
The next paired runs are E2H Top-K8 reset and H2E Top-K8 reset, seed42.

All remaining A2CPR seed42 causal controls completed on 2026-07-13:

```text
H2E no replay:       C 71.32 / P 90.27 / R 90.77 => 84.12
E2H Top-K8 reset:    C 72.16 / P 90.65 / R 90.98 => 84.60
H2E Top-K8 reset:    C 71.82 / P 90.67 / R 91.07 => 84.52
```

Stage-local reset removes the earlier large H2E advantage: reset E2H is only
+0.08 above reset H2E, effectively tied at one seed. Both reset variants are
near/slightly above Joint PL03 (84.51), while stateful E2H was 84.19 and
stateful H2E was 84.86. Therefore the stateful H2E-E2H gap mostly reflects the
interaction between domain order and LR/momentum history, not robust evidence
that hard-to-easy is intrinsically superior.

H2E no replay (84.12) versus stateful H2E replay (84.86) gives a +0.74 replay
gain under H2E. Together with E2H's +0.31 replay gain, reliable replay is the
more consistent mechanism; ordering alone is not. Do not claim a universal
H2E advantage from this pilot.

Local reset-run archives:

```text
.tmp/agent_handoff/logs/curriculum_pilot/*resetopt*_run.log
.tmp/agent_handoff/results/curriculum_pilot/e2h_reset_*_audit.jsonl
.tmp/agent_handoff/results/curriculum_pilot/h2e_reset_*_audit.jsonl
```

## Full Office-Home Stateful H2E Expansion (2026-07-13)

Research hypothesis after the A2CPR causal pilot:

```text
When target-domain difficulty is highly imbalanced, joint MTDA cannot allocate
limited prompt plasticity well. Stateful hard-first curriculum assigns early
high-LR updates to the hardest target, while class-balanced reliable replay
preserves that target as later/easier targets are fitted.
```

Do not describe reset-H2E's lower Clipart result as proven representation
overwriting; it is a supported mechanism hypothesis. Test it through stagewise
hardest-domain retention and replay audits.

The next experiment expands stateful H2E Top-K8 replay, seed42, to C2APR,
P2ACR, and R2ACP. Each task first requires its own source-only entropy probe;
target labels must not influence ordering. C2APR and P2ACR source-only probes
were launched in parallel at 2026-07-13 20:06 CST:

```text
method: maple_cshared_sourceonly_probe_seed42
screens: c_probe (GPU0), p_probe (GPU1)
logs: logs/maple_cshared_sourceonly_probe_seed42_{C2APR,P2ACR}_run.log
```

After all tasks complete, record per source task:

```text
source-only normalized entropy for all targets
hardest-minus-second entropy gap
Joint baseline accuracy gap between hardest and second-hardest
H2E per-domain and macro gains over Joint
hardest-domain stage-1 to final retention
whether easy-domain accuracy is traded for hardest-domain gain
replay bank coverage, exposure, stability, and loss contribution
```

With only four source tasks, entropy-gap correlations are descriptive evidence,
not a statistically strong universal claim. Report both raw task points and
Spearman/Pearson values, then seek confirmation on Office31/DomainNet.

## H2E Result Boundary and Diagnostic Infrastructure (2026-07-14)

Stateful H2E Top-K8 replay, seed42, completed on all Office-Home source tasks:

```text
A2CPR: 84.86 vs Joint 84.51 => +0.35; hardest Clipart +0.80
C2APR: 89.02 vs Joint 89.07 => -0.05; hardest Art +0.20
P2ACR: 82.40 vs Joint 82.35 => +0.05; hardest Clipart +0.48
R2ACP: 82.77 vs Joint 82.84 => -0.07; hardest Clipart +0.23
overall: 84.76 vs 84.69 => +0.07
```

Seed100 A/C controls disproved the strong claim that the current method
consistently improves the hardest domain:

```text
A2CPR: H2E 84.17 vs Joint 84.22 => -0.05; Clipart -0.34
C2APR: H2E 88.50 vs Joint 87.94 => +0.56; Art -0.16
```

The source-only entropy orders were identical across seeds, so the sign flip
was not caused by order instability. Treat H2E+Replay as an analysis baseline,
not the main method. Diagnose pseudo-label quality, replay delivery,
cross-domain gradient conflict, and shared-prompt capacity before adding new
modules or reverting to PairGap.

Local diagnostic code now supports, without changing the default replay path:

```text
online/oracle-correct/frozen-manifest selection
pseudo/ground-truth replay labels
one-pass/cycled replay traversal
all-target sample snapshots at every stage boundary
replay manifests that freeze indices across causal oracle runs
per-class PL/replay quality and cross-stage transition aggregation
pairwise target-domain prompt-gradient cosine audits
independent source-to-single-target STDA upper-bound runs
```

The current default remains `online + pseudo + one_pass`, matching the old
algorithm. Target-label modes require diagnostics to be enabled and print an
explicit invalid-for-UDA warning. Extra all-domain audit passes preserve and
restore Python/NumPy/Torch/CUDA RNG states. Usage is documented in
`scripts/maple_curriculum_mtda/DIAGNOSTICS.md`.

Important implementation fact: the original replay loader is exhausted once
per stage and then disabled. Typical exposure is about 118/1010 optimizer
steps in stage 2 and 240/1010 in stage 3. Therefore a fixed-index GT-label
oracle that does not improve performance only rules out label noise under the
current sparse exposure; it does not by itself prove prompt-capacity failure.

The earlier restriction on remote diagnostic runs was lifted by the user. The
full diagnostic matrix completed in the isolated remote worktree
`~/workspace/da_lab_diag_20260714` before the control below was approved.

## Fixed-PL Cycle Control (approved 2026-07-14)

The user approved the missing `fixed pseudo-label + cycle` control for A2CPR
and C2APR at seeds 42/100. It must load the original online baseline manifest,
retain pseudo labels, Top-K8, H2E order, stateful optimizer/scheduler, total
steps, and all other settings; only replay traversal changes from one pass to
cycle.

The stage audit now additionally records:

```text
replay batches and total sample exposures
unique replay samples and per-sample exposure min/mean/max
raw and weighted replay-loss cumulative sums
mean weighted replay loss over all steps and active steps
mean/RMS/sum of replay-only weighted gradient norms
norm of the summed replay-only gradient vector
```

Replay-only gradient auditing is active only with diagnostics enabled. These
statistics are intended to distinguish benefits from persistent sample
presence versus a larger accumulated replay constraint. The decisive accuracy
comparison remains fixed-PL cycle versus fixed-PL one-pass; compare its gap to
the existing fixed-GT cycle versus fixed-GT one-pass gap to measure whether
repeated pseudo-label errors erase the exposure benefit.

Launch status:

```text
Started: 2026-07-14 16:15 CST
Remote worktree: ~/workspace/da_lab_diag_20260714
Remote code: 4fc1ac7
GPU: 1
Screen: fixedpl_cycle
Order: A2CPR seed42 -> A2CPR seed100 -> C2APR seed42 -> C2APR seed100
Method tags: maple_diag_fixed_pl_cycle_gradstats_seed{42,100}
Logs: logs/fixedpl_cycle_20260714/
State/results: results/fixedpl_cycle_20260714/
Initial ETA for all four sequential runs: 2026-07-14 19:30-20:00 CST
```

GPU0 was not used because an unrelated process held about 8.6 GiB. GPU1 passed
a real PyTorch allocation test and the first A2CPR seed42 run was confirmed
active at about 6.0 GiB before handoff. The runner validates final target
metrics, epoch-5 checkpoint, three stage-audit rows, required new fields, and
full replay-gradient coverage in stages 2/3; it retries a failed run once.

Completion status (2026-07-14 19:06 CST): all four runs completed on the first
attempt and passed validation; no OOM, traceback, NaN, or incomplete audit.

```text
                       fixed-PL one-pass       fixed-PL cycle         delta
A2CPR seed42:          72.35/90.94/91.30 84.86 73.24/90.85/90.73 84.94 +0.08
A2CPR seed100:         71.07/91.24/90.20 84.17 73.65/91.21/90.64 85.17 +1.00
C2APR seed42:          84.67/91.30/91.07 89.01 84.84/91.26/90.91 89.00 -0.01
C2APR seed100:         83.77/90.72/91.03 88.51 84.05/91.10/91.05 88.73 +0.23
```

Hardest-domain cycle gains are `+0.89, +2.58, +0.17, +0.28`, positive in all
four cases and `+0.98` on average. Macro gain is `+0.32` on average. Compared
with GT cycle, pseudo-label cycle is lower by about `0.14` macro on average,
so repeated pseudo-label errors impose a modest cost but do not erase the
benefit of persistent replay exposure. This shifts the primary diagnosis
toward insufficient replay duration in one-pass; replay label quality remains
a secondary issue.

Cycle provides 1010 replay batches / 4040 image exposures in each later stage,
versus 117-119 batches in stage 2 and 245-247 in stage 3 for one-pass. Cycle
increases cumulative weighted replay loss by about `6.4-7.8x` in stage 2 and
`3.5-3.9x` in stage 3. Its loss grows sublinearly because mean loss per active
batch falls with repeated fitting. The norm of the summed replay-gradient
vector is only about `5.5-8.3%` of the sum of per-step gradient norms, so the
repeated gradients are not simply identical and perfectly aligned; do not
interpret this ratio as a direct conflict metric because parameters and data
augmentations change across steps.

This experiment does not by itself separate persistent temporal coverage from
larger cumulative constraint strength, since cycle changes both. Use these
audits to design a budget/frequency control before claiming the mechanism.
Local archives:

```text
.tmp/agent_handoff/logs/fixedpl_cycle_20260714/
.tmp/agent_handoff/results/fixedpl_cycle_20260714/
```

## One-Pass-Step Normalized Cycle (approved 2026-07-14)

The next four-run control keeps the fixed pseudo-label manifest and cycle
coverage but normalizes only the Replay loss coefficient. At each stage:

```text
B_s = min(len(replay_loader), stage optimizer steps)
T_s = stage optimizer steps
replay loss scale = B_s / T_s
```

`B_s` is explicitly the number of optimizer steps that would actually receive
a Replay update under one-pass traversal with the same loader and one-Replay-
batch-per-step rule. It is not bank size or an approximate repeat count. The
scale is fixed at stage entry. Source CE, main PL, replay samples, augmentations,
ordering, optimizer/scheduler state, and total steps remain unchanged.

The new audit records the predicted one-pass update steps, actual cycle Replay
steps, normalized and counterfactual unnormalized loss totals, effective
Replay lambda, nominal reference/actual budgets, actual Replay gradient norms,
and LR-weighted Replay gradient-norm sum. The four intended runs are A2CPR and
C2APR at seeds 42/100 using the original online baseline manifests. Do not
modify the main PL branch until this control is complete and analyzed.

Launch status:

```text
Started: 2026-07-14 19:25 CST
Remote code/worktree: 699ddf7 / ~/workspace/da_lab_diag_20260714
GPU/screen: GPU1 / normalized_cycle
Order: A2CPR seed42 -> A2CPR seed100 -> C2APR seed42 -> C2APR seed100
Method tags: maple_diag_fixed_pl_normalized_cycle_gradstats_seed{42,100}
Logs: logs/fixedpl_normalized_cycle_20260714/
State: results/fixedpl_normalized_cycle_20260714/
Initial ETA: 2026-07-14 22:15-22:30 CST
```

GPU1 passed the real allocation check and the first run was confirmed active at
about 6.0 GiB with no startup exception. The runner retries once and validates
that each later stage has full cycle gradient coverage, that the recorded scale
equals actual one-pass reference steps divided by stage steps, and that actual
and one-pass nominal Replay weight budgets match numerically.
