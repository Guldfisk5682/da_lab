#!/usr/bin/env python3
"""Extract final Office-Home MTDA metrics from a completed training log."""

import argparse
import json
import re
from pathlib import Path


DOMAINS = {
    "A": ("art", ("clipart", "product", "real_world")),
    "C": ("clipart", ("art", "product", "real_world")),
    "P": ("product", ("art", "clipart", "real_world")),
    "R": ("real_world", ("art", "clipart", "product")),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--source", required=True, choices=sorted(DOMAINS))
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, default=100)
    return parser.parse_args()


def last_float(pattern, text, description):
    matches = re.findall(pattern, text)
    if not matches:
        raise ValueError(f"Could not find {description} in log")
    return float(matches[-1])


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    text = (run_dir / "log.txt").read_text(encoding="utf-8", errors="replace")
    _, targets = DOMAINS[args.source]
    target_accuracy = {
        domain: last_float(
            rf"Target domain {re.escape(domain)} accuracy: ([0-9.]+)%",
            text,
            f"final {domain} accuracy",
        )
        for domain in targets
    }
    macro = last_float(
        r"Overall average: ([0-9.]+)%", text, "final macro average"
    )
    computed_macro = sum(target_accuracy.values()) / len(target_accuracy)
    if abs(macro - computed_macro) > 0.02:
        raise ValueError(
            f"Logged macro {macro:.4f} disagrees with target mean {computed_macro:.4f}"
        )
    payload = {
        "dataset": "Office-Home",
        "source": args.source,
        "targets": list(targets),
        "seed": args.seed,
        "variant": args.variant,
        "target_accuracy": target_accuracy,
        "macro_average": macro,
        "log": str(run_dir / "log.txt"),
    }
    path = run_dir / "mtda_metrics.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
