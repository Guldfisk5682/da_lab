#!/usr/bin/env python3
"""Rank DomainNet targets by source-only predictive entropy."""

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from dassl.engine import build_trainer
from dassl.utils import set_random_seed

import train


DOMAINS = {
    "C": "clipart",
    "I": "infograph",
    "P": "painting",
    "Q": "quickdraw",
    "R": "real",
    "S": "sketch",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, choices=list(DOMAINS))
    parser.add_argument("--root", required=True)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--load-epoch", type=int, default=10)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def build_cfg(args):
    targets = [domain for code, domain in DOMAINS.items() if code != args.source]
    cfg_args = SimpleNamespace(
        root=args.root,
        output_dir=str(Path(args.output).parent / f"difficulty_tmp_{args.source}"),
        resume="",
        seed=args.seed,
        source_domains=[DOMAINS[args.source]],
        target_domains=targets,
        transforms=None,
        trainer="ContinuousSharedProjMaPLeMTDA",
        backbone="",
        head="",
        dataset_config_file="configs/datasets/domainnet_mtda.yaml",
        config_file="configs/trainers/ContinuousSharedProjMaPLeMTDA/vit_b16.yaml",
        opts=[
            "DATALOADER.TEST.BATCH_SIZE",
            str(args.batch_size),
            "DATALOADER.NUM_WORKERS",
            str(args.num_workers),
            "TRAINER.PROMPT_BASELINE_MTDA.MIX_TARGETS",
            "True",
        ],
    )
    return train.setup_cfg(cfg_args)


@torch.no_grad()
def score_domain(trainer, loader):
    entropy_sum = confidence_sum = 0.0
    sample_count = correct = 0
    trainer.set_model_mode("eval")
    for batch in loader:
        image = batch["img"].to(trainer.device)
        label = batch["label"].to(trainer.device)
        probabilities = trainer.model(image).float().softmax(dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
        predictions = probabilities.argmax(dim=-1)
        entropy_sum += float(entropy.sum().item())
        confidence_sum += float(probabilities.max(dim=-1).values.sum().item())
        correct += int(predictions.eq(label).sum().item())
        sample_count += int(image.shape[0])
    return {
        "samples": sample_count,
        "accuracy": 100.0 * correct / sample_count,
        "mean_entropy": entropy_sum / sample_count,
        "mean_normalized_entropy": entropy_sum / (sample_count * math.log(345)),
        "mean_confidence": confidence_sum / sample_count,
    }


def main():
    args = parse_args()
    set_random_seed(args.seed)
    trainer = build_trainer(build_cfg(args))
    trainer.load_model(args.model_dir, epoch=args.load_epoch)
    scores = {
        domain: score_domain(trainer, loader)
        for domain, loader in trainer.test_loaders_by_domain.items()
    }
    easy_to_hard = sorted(
        scores,
        key=lambda domain: (scores[domain]["mean_normalized_entropy"], domain),
    )
    payload = {
        "source": DOMAINS[args.source],
        "seed": args.seed,
        "selection_uses_target_labels": False,
        "scores": scores,
        "easy_to_hard": easy_to_hard,
        "hard_to_easy": list(reversed(easy_to_hard)),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
