#!/usr/bin/env python3
"""Evaluate frozen CLIP on the concatenated Office-31 target domains."""

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dassl.engine import build_trainer
from dassl.utils import set_random_seed

import train


DOMAINS = OrderedDict([("A", "amazon"), ("D", "dslr"), ("W", "webcam")])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, choices=list(DOMAINS))
    parser.add_argument("--root", default="data")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def build_cfg(args):
    targets = [domain for code, domain in DOMAINS.items() if code != args.source]
    cfg_args = SimpleNamespace(
        root=args.root,
        output_dir=str(args.output_dir),
        resume="",
        seed=args.seed,
        source_domains=[DOMAINS[args.source]],
        target_domains=targets,
        transforms=None,
        trainer="CLIPVPTMTDA",
        backbone="",
        head="",
        dataset_config_file="configs/datasets/office31_mtda.yaml",
        config_file="configs/trainers/CLIPVPTMTDA/vit_b16.yaml",
        opts=[
            "TRAINER.CLIP_VPT_MTDA.ENABLE_VPT",
            "False",
            "DATALOADER.TEST.BATCH_SIZE",
            str(args.batch_size),
            "DATALOADER.NUM_WORKERS",
            str(args.num_workers),
        ],
    )
    return train.setup_cfg(cfg_args)


@torch.no_grad()
def evaluate(args):
    set_random_seed(args.seed)
    trainer = build_trainer(build_cfg(args))
    trainer.set_model_mode("eval")

    per_domain = OrderedDict()
    total_correct = 0
    total_samples = 0
    for domain_name, loader in trainer.test_loaders_by_domain.items():
        domain_correct = 0
        domain_samples = 0
        for batch in loader:
            images, labels = trainer.parse_batch_test(batch)
            logits = trainer.model_inference(images, domain_name=domain_name)
            domain_correct += int((logits.argmax(dim=-1) == labels).sum().item())
            domain_samples += int(labels.numel())
        per_domain[domain_name] = {
            "accuracy": 100.0 * domain_correct / domain_samples,
            "correct": domain_correct,
            "samples": domain_samples,
        }
        total_correct += domain_correct
        total_samples += domain_samples

    target_names = list(per_domain)
    return {
        "method": "CLIP-zero-shot",
        "protocol": "mixture_target_inference",
        "backbone": "ViT-B/16",
        "prompt_template": "a photo of a {}.",
        "source": args.source,
        "source_domain": DOMAINS[args.source],
        "targets": target_names,
        "seed": args.seed,
        "uses_source_data": False,
        "uses_unlabeled_target_for_training": False,
        "trainable_params": 0,
        "optimizer_steps": 0,
        "per_domain": per_domain,
        "macro_accuracy": sum(row["accuracy"] for row in per_domain.values())
        / len(per_domain),
        "mixture_target_accuracy": 100.0 * total_correct / total_samples,
        "mixture_correct": total_correct,
        "mixture_samples": total_samples,
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "mtda_metrics.json"
    if output.is_file():
        print(output.read_text(), end="")
        return
    metrics = evaluate(args)
    output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
