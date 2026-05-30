# Environment Setup

This project keeps the official `CoOp` codebase structure and adds the minimum scaffolding needed for Office-31 and `CoCoOpDAV0`.

## Scope

- No model weights were downloaded in this rollout.
- No dataset was downloaded in this rollout.
- No training job was launched in this rollout.
- The remote server is expected to perform the actual CLIP weight download, dataset preparation, and training.

## Recommended setup flow

1. Create and activate an isolated environment on the server.
2. Install PyTorch and torchvision that match the server CUDA runtime.
3. Clone and install `Dassl.pytorch`.
4. Install this repository's Python dependencies from `requirements.txt`.

Example:

```bash
conda create -n coop-da python=3.10 -y
conda activate coop-da

# Install torch/torchvision first according to the server CUDA version.
# Example only; replace with your server-specific command.
# pip install torch torchvision --index-url ...

git clone https://github.com/KaiyangZhou/Dassl.pytorch.git ../Dassl.pytorch
pip install -e ../Dassl.pytorch
pip install -r requirements.txt
```

## Local smoke checks

If you want a lightweight local parse check before pushing further changes, you can use the existing `dlenv` environment:

```bash
conda activate dlenv
python -m compileall train.py datasets trainers models
```

This is only for syntax-level verification. It does not validate dataset availability, CLIP weight download, or training behavior.

## Helper scripts

- `scripts/setup/install_dassl.sh`: clone/install `Dassl.pytorch` and then install `requirements.txt`.
- `scripts/datasets/download_office31.sh`: archive extraction entrypoint for a manually downloaded Office-31 package.
- `scripts/cocoop_da/office31_train.sh`: stage-aware train entrypoint for `CoCoOpDAV0`.
- `scripts/cocoop_da/office31_eval.sh`: eval-only entrypoint for target-domain evaluation.
