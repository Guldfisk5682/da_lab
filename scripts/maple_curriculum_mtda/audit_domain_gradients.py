#!/usr/bin/env python3
"""Measure pairwise target-domain PL gradient conflict on shared prompts."""

import argparse
import json
import math
import sys
from collections import defaultdict
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
    "A": "art",
    "C": "clipart",
    "P": "product",
    "R": "real_world",
}


def parameter_group(name):
    if "ctx" in name:
        return "ctx"
    if "proj" in name or "projection" in name:
        return "projection"
    return "other"


def gradient_cosine(left, right, eps=1e-12):
    left_norm = left.float().norm()
    right_norm = right.float().norm()
    if left_norm.item() <= eps or right_norm.item() <= eps:
        return None
    return float(torch.dot(left.float(), right.float()).item() / (left_norm * right_norm).item())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, choices=list(DOMAINS))
    parser.add_argument("--root", default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-epoch", type=int, default=5)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--trainer-kind", choices=["continuous", "curriculum"], default="curriculum"
    )
    parser.add_argument("--domain-order", nargs="*", default=[])
    parser.add_argument("--num-batches", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--dataset-config-file", default="configs/datasets/office_home_mtda.yaml"
    )
    parser.add_argument("--config-file", default="")
    return parser.parse_args()


def build_cfg(args):
    target_domains = [domain for code, domain in DOMAINS.items() if code != args.source]
    if args.trainer_kind == "curriculum":
        trainer_name = "CurriculumContinuousSharedProjMaPLeMTDA"
        default_config = (
            "configs/trainers/CurriculumContinuousSharedProjMaPLeMTDA/vit_b16.yaml"
        )
        order = args.domain_order or target_domains
        if len(order) != len(target_domains) or set(order) != set(target_domains):
            raise ValueError(
                f"--domain-order must contain {target_domains} exactly once, got {order}"
            )
        opts = [
            "TRAINER.MAPLE_MTDA.CURRICULUM.DOMAIN_ORDER",
            str(order).replace(" ", ""),
        ]
    else:
        trainer_name = "ContinuousSharedProjMaPLeMTDA"
        default_config = "configs/trainers/ContinuousSharedProjMaPLeMTDA/vit_b16.yaml"
        opts = []
    opts.extend(
        [
            "DATALOADER.TEST.BATCH_SIZE",
            str(args.batch_size),
            "DATALOADER.NUM_WORKERS",
            str(args.num_workers),
        ]
    )
    cfg_args = SimpleNamespace(
        root=args.root,
        output_dir=str(Path(args.output).parent / "gradient_audit_tmp"),
        resume="",
        seed=args.seed,
        source_domains=[DOMAINS[args.source]],
        target_domains=target_domains,
        transforms=None,
        trainer=trainer_name,
        backbone="",
        head="",
        dataset_config_file=args.dataset_config_file,
        config_file=args.config_file or default_config,
        opts=opts,
    )
    return train.setup_cfg(cfg_args)


def flatten_group(grads, names, group=None):
    pieces = []
    for grad, name in zip(grads, names):
        if group is not None and parameter_group(name) != group:
            continue
        if grad is not None:
            pieces.append(grad.detach().reshape(-1).float().cpu())
    return torch.cat(pieces) if pieces else torch.zeros(0)


def summarize(values):
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return {
        "observations": len(values),
        "nonzero_observations": len(finite),
        "mean_cosine": sum(finite) / len(finite) if finite else None,
        "negative_fraction": (
            sum(value < 0 for value in finite) / len(finite) if finite else None
        ),
        "minimum_cosine": min(finite) if finite else None,
        "maximum_cosine": max(finite) if finite else None,
    }


def main():
    args = parse_args()
    set_random_seed(args.seed)
    trainer = build_trainer(build_cfg(args))
    trainer.load_model(args.model_dir, epoch=args.load_epoch)
    trainer.set_model_mode("eval")
    model = trainer.model
    named_parameters = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    names = [name for name, _ in named_parameters]
    parameters = [parameter for _, parameter in named_parameters]
    groups = ["all", "ctx", "projection", "other"]
    loaders = trainer.test_loaders_by_domain
    iterators = {domain: iter(loader) for domain, loader in loaders.items()}
    pair_values = defaultdict(list)
    coverage = defaultdict(list)
    gradient_norms = defaultdict(list)

    for _ in range(args.num_batches):
        gradients = {}
        for domain, loader in loaders.items():
            try:
                batch = next(iterators[domain])
            except StopIteration:
                iterators[domain] = iter(loader)
                batch = next(iterators[domain])
            image = batch["img"].to(trainer.device)
            student_logits = model(image)
            reference_logits = model._compute_reference_logits(image)
            loss, stats = model._pseudo_label_loss(student_logits, reference_logits)
            grads = torch.autograd.grad(
                loss, parameters, allow_unused=True, retain_graph=False
            )
            gradients[domain] = {
                group: flatten_group(grads, names, None if group == "all" else group)
                for group in groups
            }
            coverage[domain].append(float(stats["coverage"].detach().item()))
            for group in groups:
                gradient_norms[(domain, group)].append(
                    float(gradients[domain][group].norm().item())
                )

        domains = sorted(gradients)
        for left_index, left in enumerate(domains):
            for right in domains[left_index + 1 :]:
                for group in groups:
                    pair_values[(left, right, group)].append(
                        gradient_cosine(
                            gradients[left][group], gradients[right][group]
                        )
                    )

    report = {
        "source": DOMAINS[args.source],
        "trainer_kind": args.trainer_kind,
        "model_dir": str(Path(args.model_dir).resolve()),
        "load_epoch": args.load_epoch,
        "num_batches": args.num_batches,
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "mean_pl_coverage": {
            domain: sum(values) / len(values) for domain, values in coverage.items()
        },
        "mean_gradient_norm": {
            f"{domain}:{group}": sum(values) / len(values)
            for (domain, group), values in gradient_norms.items()
        },
        "pairwise": {
            f"{left}<->{right}:{group}": summarize(values)
            for (left, right, group), values in sorted(pair_values.items())
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
