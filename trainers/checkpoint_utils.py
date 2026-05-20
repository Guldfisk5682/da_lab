import torch


def load_checkpoint_compat(fpath, map_location="cpu"):
    return torch.load(fpath, map_location=map_location, weights_only=False)
