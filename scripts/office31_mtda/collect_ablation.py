#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--order", nargs="*", default=[])
    args = parser.parse_args()

    text = (args.run_dir / "log.txt").read_text(errors="replace")
    pairs = re.findall(
        r"Target domain ([A-Za-z_]+) accuracy: ([0-9]+(?:\.[0-9]+)?)%", text
    )
    per_domain = {domain: float(accuracy) for domain, accuracy in pairs}
    if len(per_domain) != 2:
        raise RuntimeError(f"Expected two target results, got {per_domain}")

    audits = []
    audit_path = args.run_dir / "curriculum_stage_audit.jsonl"
    if audit_path.exists():
        audits = [
            json.loads(line)
            for line in audit_path.read_text().splitlines()
            if line.strip()
        ]
    optimizer_steps = sum(int(row["optimizer_steps"]) for row in audits)
    if not optimizer_steps:
        matches = re.findall(r"optimizer_steps=(\d+)", text)
        if matches:
            optimizer_steps = int(matches[-1])

    metadata = {
        "e2h": {"curriculum": "easy_to_hard", "replay_topk": 8},
        "joint": {"curriculum": "joint", "replay_topk": 0},
        "hard_only": {"curriculum": "hard_to_easy", "replay_topk": 8},
        "soft_only": {"curriculum": "hard_to_easy", "replay_topk": 8},
        "no_replay": {"curriculum": "hard_to_easy", "replay_topk": 0},
        "topk16": {"curriculum": "hard_to_easy", "replay_topk": 16},
        "topk32": {"curriculum": "hard_to_easy", "replay_topk": 32},
    }[args.variant]
    metrics = {
        "method": "ours_office31_ablation",
        "variant": args.variant,
        "protocol": "office31_mtda_ablation",
        "source": args.source,
        "seed": args.seed,
        "backbone": "ViT-B/16",
        "trainable_params": 788992,
        "domain_order": args.order,
        "per_domain_accuracy": per_domain,
        "macro_accuracy": sum(per_domain.values()) / len(per_domain),
        "optimizer_steps": optimizer_steps,
        "replay_weight": 0.75 if metadata["replay_topk"] else 0.0,
        **metadata,
    }
    output = args.run_dir / "mtda_metrics.json"
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
