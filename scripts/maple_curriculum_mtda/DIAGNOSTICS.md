# Curriculum replay diagnostic workflow

These controls are for causal diagnosis only. Any mode that reads target
ground truth must not be reported as a deployable UDA method.

## Outputs

With `DIAGNOSTICS_ENABLED=True`, a curriculum run writes:

- `replay_selection_manifest.jsonl`: exact selected paths and frozen labels at
  every stage boundary;
- `pl_sample_audit.jsonl`: teacher/student prediction, confidence, ground
  truth, agreement, eligibility and hypothetical Top-K membership for every
  target sample at every stage boundary;
- `replay_bank_audit.jsonl`: aggregate candidate/selected precision, shared
  teacher-student errors, predicted/true class coverage and oracle shortfall;
- `curriculum_stage_audit.jsonl`: replay exposure and weighted loss per stage.

Extra all-domain scoring preserves and restores Python, NumPy, Torch and CUDA
RNG states so diagnostics do not change the subsequent training stream.

## 1. Instrumented pseudo-label baseline

Use a new method tag. The default replay path remains online pseudo-label
selection with a single bank traversal per stage.

```bash
CUDA_VISIBLE_DEVICES=0 \
DOMAIN_ORDER="clipart product real_world" \
METHOD_TAG=maple_diag_baseline_seed42 \
REPLAY_ENABLED=True \
DIAGNOSTICS_ENABLED=True \
bash scripts/maple_curriculum_mtda/run_officehome_one.sh A 42
```

## 2. Fixed-index ground-truth-label oracle

This run loads every stage's selection from the baseline manifest. It does not
reselect samples after the oracle labels change model optimization.

```bash
CUDA_VISIBLE_DEVICES=0 \
DOMAIN_ORDER="clipart product real_world" \
METHOD_TAG=maple_diag_fixed_gt_seed42 \
REPLAY_ENABLED=True \
DIAGNOSTICS_ENABLED=True \
REPLAY_SELECTION_MODE=manifest \
REPLAY_LABEL_SOURCE=ground_truth \
REPLAY_MANIFEST_PATH=/absolute/baseline/run/replay_selection_manifest.jsonl \
bash scripts/maple_curriculum_mtda/run_officehome_one.sh A 42
```

Only replay CE labels change. The manifest, Top-K count, stage order, optimizer
and default one-pass exposure remain fixed.

## 3. Oracle-correct selection

Correct candidates are ranked before incorrect candidates within each
predicted class; confidence and path retain deterministic tie-breaking.

```bash
REPLAY_SELECTION_MODE=oracle_correct \
REPLAY_LABEL_SOURCE=pseudo \
DIAGNOSTICS_ENABLED=True \
... bash scripts/maple_curriculum_mtda/run_officehome_one.sh A 42
```

Because a correct prediction's pseudo-label equals ground truth, this is an
oracle-clean-selection diagnostic. `oracle_correct_shortfall_per_class` reports
classes with fewer than K correct eligible candidates.

## 4. Replay exposure control

`REPLAY_TRAVERSAL=one_pass` reproduces the original behavior. The diagnostic
stress test below cycles the same fixed bank throughout each later stage:

```bash
REPLAY_SELECTION_MODE=manifest \
REPLAY_LABEL_SOURCE=ground_truth \
REPLAY_TRAVERSAL=cycle \
... bash scripts/maple_curriculum_mtda/run_officehome_one.sh A 42
```

## 5. Aggregate sample-level audits

```bash
python scripts/maple_curriculum_mtda/analyze_replay_diagnostics.py \
  --run-dir /absolute/path/to/run
```

The summary includes per-stage/domain and per-predicted-class label quality,
agreement precision, both-wrong-agree rates, candidate/Top-K precision, label
flip rates, correct-to-wrong transitions and hypothetical Top-K Jaccard.

## 6. Shared-prompt gradient conflict

For a curriculum checkpoint:

```bash
python scripts/maple_curriculum_mtda/audit_domain_gradients.py \
  --source A \
  --seed 42 \
  --trainer-kind curriculum \
  --domain-order clipart product real_world \
  --model-dir /absolute/path/to/run \
  --load-epoch 5 \
  --num-batches 32 \
  --output results/gradient_audit_a2cpr_seed42.json
```

It reports pairwise cosine similarity, negative-cosine fraction and gradient
norms for all prompt parameters and separately for context/projection groups.

## 7. Independent STDA upper bound

Train one prompt model for each source-target pair:

```bash
METHOD_TAG=maple_cshared_pl03_independent_stda_seed42 \
bash scripts/maple_curriculum_mtda/run_officehome_stda_one.sh A C 42
```

Repeat for A-to-P and A-to-R. These runs are a diagnostic upper bound with
separate target-adaptation parameters, not a parameter/compute-matched final
MTDA method.
