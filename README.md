# da_lab

This repository is a project-specific fork of `CoOp/CoCoOp` for Office-31 domain adaptation with a shallow hidden-state adaptation module (`CoCoOpDAV0`).

## What is already in this repo

- Official `CoOp/CoCoOp` training structure
- Office-31 dataset wrapper
- `CoCoOpDAV0` Stage 1 and Stage 2 trainer scaffold
- Shell entrypoints for dataset extraction, dependency setup, training, and evaluation

## Environment requirements

`pip install -r requirements.txt` is not sufficient by itself.

You also need:

1. A fresh Python environment
2. `torch` and `torchvision` matching the server CUDA version
3. `Dassl.pytorch` installed into the same environment

## Minimal setup

```bash
conda create -n coop-da python=3.10 -y
conda activate coop-da

# Install torch/torchvision for your CUDA version first.
# Replace this with the correct command for your server.
# Example only:
# pip install torch torchvision --index-url <your-cuda-wheel-index>

git clone https://github.com/Guldfisk5682/da_lab.git
cd da_lab

bash scripts/setup/install_dassl.sh
```

`scripts/setup/install_dassl.sh` will:

- clone `Dassl.pytorch` into `../Dassl.pytorch` if missing
- run `pip install -e ../Dassl.pytorch`
- run `pip install -r requirements.txt`

## Dataset preparation

This repo does not auto-download Office-31. Use a manually downloaded archive on the server.

```bash
export DATA_ROOT=/path/to/datasets
export OFFICE31_ARCHIVE=/path/to/office31.zip

bash scripts/datasets/download_office31.sh
```

Expected layout after extraction:

```text
$DATA_ROOT/office31/amazon/<class_name>/*.jpg
$DATA_ROOT/office31/dslr/<class_name>/*.jpg
$DATA_ROOT/office31/webcam/<class_name>/*.jpg
```

The loader also accepts:

```text
$DATA_ROOT/office31/amazon/images/<class_name>/*.jpg
$DATA_ROOT/office31/dslr/images/<class_name>/*.jpg
$DATA_ROOT/office31/webcam/images/<class_name>/*.jpg
```

## One-command startup

### Stage 1

Train only the shallow adaptation module and gate:

```bash
export DATA=/path/to/datasets
export SOURCE_DOMAIN=amazon
export TARGET_DOMAIN=webcam
export SEED=1
export STAGE=1

bash scripts/cocoop_da/office31_train.sh
```

### Stage 2

Train shallow adaptation + gate + CoCoOp prompt learner:

```bash
export DATA=/path/to/datasets
export SOURCE_DOMAIN=amazon
export TARGET_DOMAIN=webcam
export SEED=1
export STAGE=2

bash scripts/cocoop_da/office31_train.sh
```

### Evaluation

```bash
export DATA=/path/to/datasets
export SOURCE_DOMAIN=amazon
export TARGET_DOMAIN=webcam
export SEED=1
export STAGE=1
export MODEL_DIR=output/office31/cocoop_da_v0/A2W/seed1/stage1

bash scripts/cocoop_da/office31_eval.sh
```

## Hyperparameters

### `N_CTX`

- Global fallback default in code: `16`
- Current `CoCoOpDAV0` training config: `4`

That means:

- if you use `configs/trainers/CoCoOpDA/vit_b16_v0.yaml`, then `N_CTX=4`
- if you omit the config override and rely only on `train.py` defaults, then `N_CTX=16`

Current project default for actual runs should be treated as `4`, because the provided entry script points to:

```text
configs/trainers/CoCoOpDA/vit_b16_v0.yaml
```

## Important notes

- CLIP weights are still downloaded lazily at runtime by the original CoCoOp loader.
- `Dassl.pytorch` must be importable in the same environment, otherwise `train.py` will fail before startup.
- `requirements.txt` intentionally does not pin `torch`, because that must match the target server CUDA stack.

## Key files

- `docs/env_setup.md`
- `docs/codebase_notes.md`
- `docs/office31_setup.md`
- `datasets/office31.py`
- `trainers/cocoop_da_v0.py`
- `models/shallow_adapt.py`
- `configs/trainers/CoCoOpDA/vit_b16_v0.yaml`
