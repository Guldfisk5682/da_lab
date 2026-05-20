# AGENT.md

## Project Mission

This repository is for a fast, reproducible prototype of a CoCoOp-based unsupervised domain adaptation method.

The immediate goal is **not** to build a full paper system. The immediate goal is to:

1. initialize and reproduce the official CoOp/CoCoOp codebase;
2. verify that CLIP ViT-B/16 + CoCoOp can run on Office-31;
3. implement a V0 shallow hidden-state statistical adaptation module;
4. test whether the V0 forward path, training loop, and evaluation protocol are correct.

The target research setting for V0 is:

```text
Source-available single-source single-target UDA
Dataset: Office-31
Backbone: CLIP ViT-B/16
Base method: CoCoOp
Adaptation point: CLIP visual transformer block 3 output
Modified tokens: patch tokens only
Main module: shallow hidden-state normalize-restore + learnable gate
```

---

## Non-Negotiable Rules

1. **Start from the official CoOp repository.**
   - Use: `https://github.com/KaiyangZhou/CoOp`
   - CoCoOp is implemented in the same repository.
   - Do not rewrite the entire training framework unless absolutely necessary.

2. **Keep the original baseline code intact.**
   - Do not destructively modify the original `trainers/cocoop.py`.
   - Create new files for the V0 method.
   - Recommended names:
     - `trainers/cocoop_da_v0.py`
     - `clip/shallow_adapt.py` or `models/shallow_adapt.py`
     - `configs/trainers/CoCoOpDA/vit_b16_v0.yaml`

3. **Prioritize reproducibility over novelty.**
   - First reproduce official CoCoOp/CLIP behavior.
   - Then add Office-31 support.
   - Then add the V0 module.
   - Do not implement multi-source, source-free, uncertainty, DANN, CDAN, MCC, pseudo-labeling, or FiLM/adapter variants in the first pass.

4. **Do not unfreeze the CLIP visual encoder in V0 Stage 1.**
   - Stage 1 trains only the V0 adaptation module and gate.
   - Stage 2 may optionally unfreeze the CoCoOp prompt learner.
   - Do not unfreeze CLIP visual blocks unless explicitly requested later.

5. **Gate is applied at the shallow hidden-state level.**
   - The V0 method modifies the hidden state after visual transformer block 3.
   - The fused hidden state is passed through visual blocks 4-12.
   - Do not implement the two-final-feature-branch version for V0.

---

## Work Log

- 2026-05-19: Found and fixed the Office-31 domain parsing issue caused by `argparse.REMAINDER` swallowing `TRAINER.COCOOP_DA.TRAIN.STAGE`; added `--` before config overrides in the train scripts.
- 2026-05-19: Added an OfficeHome dataset config and a dedicated OfficeHome training script for CoCoOpDAV0.
- 2026-05-19: Diagnosed that CoOp does not provide an Office-31 downloader or a real dataset-root resolver; added a practical Office-31 download-and-normalize script and made the train/eval scripts fail fast when `DATA` is still a placeholder.
- 2026-05-19: Added an Office-31 layout verification script and expanded the README with a full cloud workflow covering environment setup, download, verification, training, and evaluation.
- 2026-05-19: Fixed Office-31 download to work with older `gdown` by falling back when `--fuzzy` is unsupported.
- 2026-05-19: Office-31 download from Google Drive timed out; verified an existing dataset at `/workspace/qw/DAMP-main/dataset/office31` and linked it into `da_lab/data/office31` for training.
- 2026-05-19: Training crashed because `--` was included in `args.opts`; stripped the leading `--` in `train.py` before merging config overrides.
- 2026-05-19: Training crashed on a YACS type mismatch for `TRAINER.COCOOP_DA.TRAIN.TRAIN_PROMPT_LEARNER`; updated the train script to pass `True/False` instead of lowercase strings.
- 2026-05-19: Training crashed with fp16 vs fp32 mismatch in `ShallowGate`; casted reference stats to the patch-token dtype/device before fusion.
- 2026-05-19: Training still hit fp16 vs fp32 in `ShallowAdaptation`; casted scale/bias to the token dtype to keep adapted tokens in fp16.
- 2026-05-19: Training still hit fp16 vs fp32; casted normalized tokens to patch-token dtype before shallow adaptation.
- 2026-05-19: Training still hit fp16 vs fp32; casted adapted tokens to patch-token dtype before gating.
- 2026-05-19: Training still hit fp16 vs fp32; aligned gate inputs to LayerNorm weight dtype inside `ShallowGate`.
- 2026-05-19: Training hit fp16 vs fp32 in transformer attention; replaced scalar `1.0` with `torch.ones_like(alpha)` to keep fused tokens in fp16.
- 2026-05-19: Training still hit fp16 vs fp32 in transformer blocks; cast fused hidden tokens to the original hidden dtype before forwarding.
- 2026-05-20: Added `office31_train_all.sh` to run all six Office-31 SS-STDA tasks sequentially with auto-eval.
- 2026-05-20: Updated the Office-31 scripts so output directories follow the selected trainer name and `office31_train_all.sh` now performs explicit post-train evaluation per task.
- 2026-05-20: Fixed auto-eval for baseline trainers by making `office31_eval.sh` fall back to the latest `model.pth.tar-*` checkpoint when `model-best.pth.tar` is absent.
- 2026-05-20: Implemented V1 `FinalFeatureGate` as the default DA variant. V1 keeps the shallow layer-3 restat step but delays gating to the final image feature level: `feat_normal` and `feat_adapted` are produced separately, then blended by a scalar gate with `LayerNorm -> Linear(dim, dim/4) -> SiLU -> Linear(dim/4, 1) -> Sigmoid`, zero-initialized last weight, and bias `-4.0`.
- 2026-05-20: Remote experiment workflow for V1:
  1. Pull latest `main` and verify the default trainer/config are `CoCoOpDAV1` and `configs/trainers/CoCoOpDA/vit_b16_v1.yaml`.
  2. Run the official CoCoOp baseline by overriding `TRAINER=CoCoOp` and `CFG=configs/trainers/CoCoOp/vit_b16_c4_ep10_batch1_ctxv1.yaml`.
  3. Run V1 Stage 1 with learned alpha using the default `office31_train_all.sh`.
  4. Run V1 ablations by overriding `TRAINER.COCOOP_DA.GATE.FORCE_ALPHA` to `0.0` and `1.0` in direct `train.py` calls or dedicated shell wrappers.
  5. Only if Stage 1 helps, promote to Stage 2 by setting `STAGE=2`; keep the same V1 config and compare against the Stage 1 checkpoint family.

---

## Phase 0: Repository Initialization

### Expected commands

Use these as a starting point and adjust only if the official repository instructions differ.

```bash
git clone https://github.com/KaiyangZhou/CoOp.git
cd CoOp

# Create an isolated environment.
# Prefer the environment style already used on the server.
# Example:
conda create -n coop-da python=3.8 -y
conda activate coop-da

# Install PyTorch according to the server CUDA version.
# Then install dependencies required by CoOp and Dassl.
pip install -r requirements.txt || true

# Install Dassl if required by the official CoOp repo.
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git ../Dassl.pytorch
cd ../Dassl.pytorch
pip install -e .
cd ../CoOp
```

If `requirements.txt` is missing or incomplete, inspect the official README and install the minimal dependencies manually.

Typical dependencies may include:

```bash
pip install ftfy regex tqdm yacs gdown scipy scikit-learn
pip install git+https://github.com/openai/CLIP.git
```

Do not guess silently. Record the final working environment in:

```text
docs/env_setup.md
```

---

## Phase 1: Baseline Smoke Test

Before touching model internals, verify the following:

```bash
python train.py --help
```

Then inspect:

```text
trainers/cocoop.py
trainers/coop.py
trainers/zsclip.py
clip/
configs/
scripts/cocoop/
```

Document:

1. how CoCoOp builds the CLIP model;
2. where the visual encoder is called;
3. where image features are passed into the prompt learner;
4. where trainable parameters are selected;
5. how datasets are registered.

Write the findings to:

```text
docs/codebase_notes.md
```

---

## Phase 2: Office-31 Dataset Support

### Dataset setting

Office-31 domains:

```text
amazon
dslr
webcam
```

Transfer tasks:

```text
A -> W
A -> D
W -> A
W -> D
D -> A
D -> W
```

Use source labels. Treat target labels as unavailable during training, but use target labels for evaluation only.

### Required behavior

The training batch must contain:

```python
batch_x  # labeled source batch
batch_u  # unlabeled target batch
```

Use source labels only for supervised classification loss.

Target labels must not be used in the loss.

### Implementation guidance

First check whether the installed Dassl/CoOp framework already supports Office-31 or generic domain adaptation datasets.

If Office-31 is not supported, add a minimal dataset wrapper.

Recommended files:

```text
datasets/office31.py
configs/datasets/office31.yaml
```

The dataset wrapper should support:

```text
SOURCE_DOMAINS: ["amazon"]
TARGET_DOMAINS: ["webcam"]
```

or equivalent config fields already used by Dassl.

Do not hard-code absolute paths. Use config or environment variables:

```bash
DATA=/path/to/datasets
```

Expected folder layout can be one of the following, but the loader must document which layout it supports:

```text
$DATA/office31/amazon/images/...
$DATA/office31/dslr/images/...
$DATA/office31/webcam/images/...
```

or:

```text
$DATA/office31/amazon/<class_name>/*.jpg
$DATA/office31/dslr/<class_name>/*.jpg
$DATA/office31/webcam/<class_name>/*.jpg
```

If the actual server layout differs, adapt the dataset wrapper and document it.

---

## Phase 3: CoCoOp ViT-B/16 Baseline on Office-31

Run a baseline before implementing V0.

Minimum baseline list:

```text
CLIP zero-shot, ViT-B/16
CoCoOp, ViT-B/16
```

Recommended output directory convention:

```text
output/office31/cocoop_vit_b16/A2W/seed1/
output/office31/cocoop_vit_b16/A2D/seed1/
...
```

The first smoke test can run only one task:

```text
A -> W
```

Acceptance criteria:

1. training starts without dataset or config errors;
2. image batch shape is correct;
3. text prompt/class names are correct;
4. evaluation produces target-domain top-1 accuracy;
5. the run writes logs and checkpoints into `output/`.

After this works, run all six Office-31 transfer tasks for seed 1.

Record results in:

```text
results/office31_baseline_seed1.md
```

---

## Phase 4: V0 Model Specification

### Core idea

At CLIP ViT-B/16 visual transformer block 3 output, extract patch-token hidden states from source and target images.

For each source image:

1. split CLS token and patch tokens;
2. compute source patch-token mean and std;
3. normalize source patch tokens;
4. restore normalized source patch tokens using target-domain statistics;
5. fuse original source patch tokens and restored patch tokens using a learnable gate;
6. concatenate the original CLS token back;
7. pass the fused hidden state through the remaining visual transformer blocks;
8. use the final image feature for CoCoOp logits.

For target images, V0 may implement the symmetric target-to-source path, but keep a config switch:

```text
ADAPT_MODE: "s2t"      # only source -> target
ADAPT_MODE: "bidirect" # source -> target and target -> source
```

Start experiments with `s2t` first if `bidirect` is unstable.

---

## Required V0 Forward Path

The intended V0 flow is:

```text
image
  -> CLIP patch embedding + positional embedding
  -> visual transformer block 1
  -> visual transformer block 2
  -> visual transformer block 3
  -> shallow hidden state h_l
  -> statistical adaptation module
  -> fused hidden state h_l_fused
  -> visual transformer block 4 ... block 12
  -> final CLIP image feature
  -> CoCoOp prompt learner / text features
  -> cosine logits
```

Do **not** implement this V0 as:

```text
normal branch -> full visual encoder -> final feature
adapted branch -> full visual encoder -> final feature
gate(final normal feature, final adapted feature)
```

That is a possible later version, but not V0.

---

## V0 Tensor Contract

For CLIP ViT-B/16 at injection layer `l=3`:

```python
h_s = visual_forward_until(x_s, layer_idx=3)
h_t = visual_forward_until(x_t, layer_idx=3)
```

Expected shapes:

```python
h_s: [B_s, 1 + N, C]
h_t: [B_t, 1 + N, C]
```

For ViT-B/16 with 224x224 inputs:

```text
N = 196 patch tokens
C = 768 hidden dim
```

Split tokens:

```python
cls_s, p_s = h_s[:, :1, :], h_s[:, 1:, :]
cls_t, p_t = h_t[:, :1, :], h_t[:, 1:, :]
```

V0 modifies `p_s` and `p_t` only.

V0 must keep `cls_s` and `cls_t` unchanged.

---

## V0 Source-to-Target Restat

Use patch tokens only.

```python
mu_s = p_s.mean(dim=1, keepdim=True)
std_s = p_s.std(dim=1, keepdim=True, unbiased=False)

a_s = (p_s - mu_s) / (std_s + eps)

mu_t_bank, std_t_bank = target_stats_bank.get()

p_s_adapted = a_s * std_t_bank + mu_t_bank
```

Then gate:

```python
alpha_s = gate_net(p_s, p_s_adapted)

p_s_fused = (1.0 - alpha_s) * p_s + alpha_s * p_s_adapted
h_s_fused = torch.cat([cls_s, p_s_fused], dim=1)
```

Then continue the visual encoder:

```python
img_feat_s = visual_forward_from(h_s_fused, start_layer=4)
```

---

## V0 Target Path

For `ADAPT_MODE="s2t"`:

```text
Target images may be used only to update target statistics and compute optional target entropy loss.
The target visual path can remain normal or use source-restored target tokens only if implemented safely.
```

For `ADAPT_MODE="bidirect"`:

```python
mu_t = p_t.mean(dim=1, keepdim=True)
std_t = p_t.std(dim=1, keepdim=True, unbiased=False)

a_t = (p_t - mu_t) / (std_t + eps)

mu_s_bank, std_s_bank = source_stats_bank.get()

p_t_adapted = a_t * std_s_bank + mu_s_bank

alpha_t = gate_net(p_t, p_t_adapted)

p_t_fused = (1.0 - alpha_t) * p_t + alpha_t * p_t_adapted
h_t_fused = torch.cat([cls_t, p_t_fused], dim=1)

img_feat_t = visual_forward_from(h_t_fused, start_layer=4)
```

---

## Stats Bank Requirements

Implement two EMA statistics banks:

```text
source_stats_bank
target_stats_bank
```

Each bank should track patch-token statistics.

Start with global domain-level statistics, not class-conditional statistics.

Required fields:

```python
running_mu: Tensor shaped [1, 1, C] or broadcast-compatible
running_std: Tensor shaped [1, 1, C] or broadcast-compatible
momentum: float, default 0.99
initialized: bool
```

Update using detached batch statistics:

```python
bank.update(mu_batch.detach(), std_batch.detach())
```

The batch statistics may first be averaged over batch dimension:

```python
mu_batch_domain = mu_batch.mean(dim=0, keepdim=True)
std_batch_domain = std_batch.mean(dim=0, keepdim=True)
```

Use stable eps:

```python
eps = 1e-6
```

Log bank status regularly:

```text
source_mu_norm
source_std_mean
target_mu_norm
target_std_mean
```

---

## Gate Network Requirements

Start with a simple token-wise and channel-wise gate:

```python
alpha: [B, N, C]
```

Recommended implementation:

```python
class ShallowGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm_ori = nn.LayerNorm(dim)
        self.norm_adp = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )
        nn.init.constant_(self.net[-1].bias, -2.0)

    def forward(self, p_ori, p_adapted):
        x = torch.cat(
            [self.norm_ori(p_ori), self.norm_adp(p_adapted)],
            dim=-1,
        )
        alpha = torch.sigmoid(self.net(x))
        return alpha
```

Important:

```text
The final gate bias must be initialized to a negative value, e.g. -2.0.
This makes alpha small at the beginning and keeps the model close to original CLIP.
```

Log:

```text
alpha_mean
alpha_std
alpha_min
alpha_max
```

---

## V0 Loss Function

Keep V0 simple.

Required:

```python
loss_src = CE(logits_s_fused, y_s)
```

Recommended:

```python
loss_cons = KL(logits_s_normal.detach(), logits_s_fused)
loss_ent = entropy(logits_t_fused)
```

Total:

```python
loss = loss_src + lambda_cons * loss_cons + lambda_ent * loss_ent
```

Default weights:

```yaml
LAMBDA_CONS: 0.1
LAMBDA_ENT: 0.01
```

If target entropy destabilizes training, set:

```yaml
LAMBDA_ENT: 0.0
```

Do not add pseudo-labeling, DANN, CDAN, MCC, or uncertainty loss in V0.

---

## Training Stages

### Stage 0: Baseline

Run official CLIP zero-shot and CoCoOp.

No V0 module.

### Stage 1: V0 visual adaptation only

Freeze:

```text
CLIP visual encoder
CLIP text encoder
CoCoOp prompt learner, initially
```

Train only:

```text
shallow adaptation module
gate network
```

Stats banks are updated by EMA but are not optimized by gradients.

### Stage 2: V0 + CoCoOp prompt learner

Freeze:

```text
CLIP visual encoder
CLIP text encoder
```

Train:

```text
shallow adaptation module
gate network
CoCoOp prompt learner
```

Use smaller LR for prompt learner than gate/adaptation module if separate parameter groups are easy.

Suggested ratio:

```text
adaptation module LR = base LR
prompt learner LR = 0.2x to 0.5x base LR
```

Do not unfreeze CLIP visual blocks in V0.

---

## Implementation Checklist

### Codebase inspection

- [ ] Confirm official CoCoOp trainer entrypoint.
- [ ] Confirm where CLIP image features are extracted.
- [ ] Confirm CLIP visual transformer block list name.
- [ ] Confirm whether the CLIP implementation uses `visual.transformer.resblocks`.
- [ ] Confirm tensor layout inside CLIP ViT: likely `[sequence, batch, dim]` internally and `[batch, sequence, dim]` externally.
- [ ] Write `docs/codebase_notes.md`.

### Dataset

- [ ] Add or verify Office-31 dataset support.
- [ ] Support source domain and target domain config.
- [ ] Ensure target labels are not used during training.
- [ ] Evaluate on target test split.
- [ ] Write `docs/office31_setup.md`.

### Baseline

- [ ] Run CLIP zero-shot ViT-B/16 on A->W.
- [ ] Run CoCoOp ViT-B/16 on A->W.
- [ ] Run all six Office-31 tasks for seed 1 if smoke test succeeds.
- [ ] Save results to `results/office31_baseline_seed1.md`.

### V0 model

- [ ] Add stats bank.
- [ ] Add shallow gate.
- [ ] Add forward-until-layer and forward-from-layer path for CLIP ViT.
- [ ] Inject after block 3.
- [ ] Modify patch tokens only.
- [ ] Preserve CLS token.
- [ ] Continue through remaining visual blocks.
- [ ] Return fused logits and useful debug tensors.

### V0 training

- [ ] Implement source CE.
- [ ] Implement optional source consistency loss.
- [ ] Implement optional target entropy loss.
- [ ] Log all loss components.
- [ ] Log alpha statistics.
- [ ] Log stats-bank diagnostics.
- [ ] Verify trainable parameter names.

### V0 evaluation

- [ ] Evaluate target-domain top-1 accuracy.
- [ ] Save per-task results.
- [ ] Compare to baseline.
- [ ] Save config and git commit hash with each run.

---

## Suggested Config Fields

Add a trainer config similar to:

```yaml
TRAINER:
  NAME: "CoCoOpDAV0"

  COCOOP_DA:
    BACKBONE: "ViT-B/16"
    INJECT_LAYER: 3
    MODIFY_CLS: false
    ADAPT_MODE: "s2t"  # ["s2t", "bidirect"]

    STATS:
      TYPE: "ema"
      MOMENTUM: 0.99
      EPS: 1e-6

    GATE:
      TYPE: "token_channel"
      INIT_BIAS: -2.0

    LOSS:
      LAMBDA_CONS: 0.1
      LAMBDA_ENT: 0.01

    TRAIN:
      STAGE: 1
      FREEZE_VISUAL: true
      FREEZE_TEXT: true
      TRAIN_PROMPT_LEARNER: false
```

Stage 2 can use:

```yaml
TRAINER:
  COCOOP_DA:
    TRAIN:
      STAGE: 2
      FREEZE_VISUAL: true
      FREEZE_TEXT: true
      TRAIN_PROMPT_LEARNER: true
```

---

## Minimal V0 Pseudocode

```python
def forward_train(batch_x, batch_u):
    x_s = batch_x["img"].to(device)
    y_s = batch_x["label"].to(device)

    x_t = batch_u["img"].to(device)

    # 1. forward to shallow layer
    h_s = visual_forward_until(x_s, layer_idx=3)
    h_t = visual_forward_until(x_t, layer_idx=3)

    cls_s, p_s = h_s[:, :1, :], h_s[:, 1:, :]
    cls_t, p_t = h_t[:, :1, :], h_t[:, 1:, :]

    # 2. update stats banks
    mu_s, std_s = compute_patch_stats(p_s)
    mu_t, std_t = compute_patch_stats(p_t)

    source_stats_bank.update(mu_s.detach(), std_s.detach())
    target_stats_bank.update(mu_t.detach(), std_t.detach())

    mu_t_bank, std_t_bank = target_stats_bank.get()

    # 3. source -> target restat
    a_s = (p_s - mu_s) / (std_s + eps)
    p_s_adapted = a_s * std_t_bank + mu_t_bank

    alpha_s = gate(p_s, p_s_adapted)
    p_s_fused = (1 - alpha_s) * p_s + alpha_s * p_s_adapted
    h_s_fused = torch.cat([cls_s, p_s_fused], dim=1)

    # 4. continue visual encoder
    feat_s_fused = visual_forward_from(h_s_fused, start_layer=4)

    # 5. CoCoOp logits
    logits_s_fused = cocoop_logits(feat_s_fused)

    # Optional target path
    feat_t_fused = forward_target_path(h_t, p_t, cls_t)
    logits_t_fused = cocoop_logits(feat_t_fused)

    # 6. losses
    loss_src = cross_entropy(logits_s_fused, y_s)
    loss_ent = entropy(logits_t_fused)

    loss = loss_src + lambda_ent * loss_ent

    if use_consistency:
        with torch.no_grad():
            feat_s_normal = visual_forward_from(h_s, start_layer=4)
            logits_s_normal = cocoop_logits(feat_s_normal)
        loss_cons = kl_divergence(logits_s_normal, logits_s_fused)
        loss = loss + lambda_cons * loss_cons

    return loss
```

---

## Debugging Requirements

Add a debug mode that runs one batch and prints:

```text
x_s shape
x_t shape
h_s shape
h_t shape
p_s shape
p_t shape
mu_s shape
std_s shape
mu_t_bank shape
std_t_bank shape
p_s_adapted shape
alpha_s shape
h_s_fused shape
feat_s_fused shape
logits_s shape
loss_src
```

Also verify:

```python
assert h_s_fused.shape == h_s.shape
assert torch.isfinite(h_s_fused).all()
assert torch.isfinite(logits_s).all()
assert 0 <= alpha_s.min() and alpha_s.max() <= 1
```

---

## Acceptance Criteria for V0

V0 is considered implemented only if all of the following pass:

1. `python train.py --help` works.
2. Official CoCoOp trainer still works.
3. Office-31 A->W baseline runs.
4. V0 trainer runs one full epoch on A->W without NaN.
5. V0 logs target-domain evaluation accuracy.
6. V0 saves config, checkpoint, and log.
7. Trainable parameter list contains only expected modules in Stage 1:
   - gate
   - shallow adaptation module
   - optionally prompt learner only in Stage 2
8. The original CoCoOp trainer remains usable.

---

## Files to Produce

Codex should produce or update:

```text
docs/env_setup.md
docs/codebase_notes.md
docs/office31_setup.md
results/office31_baseline_seed1.md

datasets/office31.py                  # if needed
configs/datasets/office31.yaml         # if needed

trainers/cocoop_da_v0.py
models/shallow_adapt.py                # or equivalent
configs/trainers/CoCoOpDA/vit_b16_v0.yaml
scripts/cocoop_da/office31_train.sh
scripts/cocoop_da/office31_eval.sh
```

If any of these files are impossible due to repository structure, document the alternative path.

---

## Suggested First Commands for Codex

```bash
pwd
ls
git status
find . -maxdepth 3 -type f | sort | sed 's#^\./##' | head -200

python train.py --help

sed -n '1,240p' trainers/cocoop.py
sed -n '1,240p' trainers/coop.py
find configs -maxdepth 3 -type f | sort
find scripts -maxdepth 3 -type f | sort
```

Then inspect the CLIP visual implementation:

```bash
find . -maxdepth 4 -type f | grep -i clip
grep -R "class VisionTransformer" -n .
grep -R "resblocks" -n clip trainers | head -50
grep -R "encode_image" -n .
```

---

## What Not To Do Yet

Do not implement these until V0 passes:

```text
multi-source UDA
multi-target UDA
source-free DA
class-conditional statistics
Wasserstein target-stat selection
Dirichlet uncertainty
pseudo-label training
DANN/CDAN/MCC
FiLM modulation
adapter rewriter
final-feature two-branch gate
unfreezing CLIP visual layers
```

These are valid later extensions, but they will slow down the first prototype.

---

## Research Notes for Later

The originally proposed alternative structure was:

```text
normal hidden state -> remaining encoder -> normal final feature
adapted hidden state -> remaining encoder -> adapted final feature
gate(normal final feature, adapted final feature)
```

This is not V0.

Keep it as a possible V1/V2 variant named:

```text
FinalFeatureGate
```

V0 should be named something like:

```text
ShallowStateGate
CrossStatShallowGate
CoCoOpDA-V0
```

The paper story for V0 should be:

```text
Shallow CLIP-ViT patch tokens preserve domain/style statistics.
A lightweight normalize-restore module transfers target-domain statistics into source hidden states.
A learnable gate preserves the original CLIP semantic stream.
The fused shallow state is processed by the remaining frozen CLIP visual blocks and used by CoCoOp.
```

---

## Final Reminder

The first successful milestone is not high accuracy.

The first successful milestone is:

```text
official CoCoOp runs
Office-31 data flow works
V0 hidden-state injection works
training does not crash
target-domain evaluation is logged
```

Only after that should we optimize accuracy.
