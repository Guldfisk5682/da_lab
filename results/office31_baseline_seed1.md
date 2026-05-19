# Office-31 Baseline Seed 1

This file is intentionally a placeholder in this rollout.

## Status

- CLIP zero-shot on Office-31: not run
- CoCoOp baseline on Office-31: not run
- Reason: the current task explicitly excluded dataset download, CLIP weight download, and actual training

## Suggested first runs on the remote server

```bash
bash scripts/cocoop_da/office31_train.sh
```

For a baseline-only smoke test, switch the trainer/config to the official CoCoOp setup first and run only `A -> W`.
