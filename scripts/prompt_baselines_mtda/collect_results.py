#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


def last_number(pattern, text, cast=float):
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    return cast(matches[-1].replace(",", "")) if matches else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument(
        "--protocol",
        choices=["zero_shot", "source_only", "mt_ent"],
        required=True,
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--entropy-weight", type=float, required=True)
    parser.add_argument("--train-batch-size", type=int)
    parser.add_argument("--test-batch-size", type=int)
    parser.add_argument("--expected-targets", type=int, default=3)
    args = parser.parse_args()

    text = (args.run_dir / "log.txt").read_text(errors="replace")
    pairs = re.findall(
        r"Target domain ([A-Za-z_]+) accuracy: ([0-9]+(?:\.[0-9]+)?)%", text
    )
    per_domain = {}
    for domain, accuracy in pairs:
        per_domain[domain] = float(accuracy)
    if len(per_domain) != args.expected_targets:
        raise RuntimeError(
            f"Expected {args.expected_targets} target results, got {per_domain}"
        )

    optimizer_steps = last_number(
        r"Baseline budget audit:.*optimizer_steps=([0-9]+)", text, int
    )
    metrics = {
        "method": args.method,
        "protocol": args.protocol,
        "source": args.source,
        "seed": args.seed,
        "backbone": "ViT-B/16",
        "source_available": args.protocol != "zero_shot",
        "uses_unlabeled_target": args.protocol == "mt_ent",
        "pseudo_labels": False,
        "mixed_target_loader": args.protocol == "mt_ent",
        "entropy_weight": args.entropy_weight,
        "per_domain_accuracy": per_domain,
        "macro_accuracy": sum(per_domain.values()) / len(per_domain),
        "trainable_params": last_number(
            r"Trainable parameter count: ([0-9,]+)", text, int
        ),
        "epochs": last_number(
            r"Baseline budget audit: epochs=([0-9]+)", text, int
        ),
        "steps_per_epoch": last_number(
            r"Baseline budget audit: epochs=[0-9]+, steps_per_epoch=([0-9]+)",
            text,
            int,
        ),
        "optimizer_steps": optimizer_steps,
        "train_batch_size": args.train_batch_size,
        "test_batch_size": args.test_batch_size,
        "source_sample_exposures": (
            optimizer_steps * args.train_batch_size
            if optimizer_steps is not None and args.train_batch_size is not None
            else None
        ),
    }
    output = args.run_dir / "mtda_metrics.json"
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
