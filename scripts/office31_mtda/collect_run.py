#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--order", nargs="+", required=True)
    args = parser.parse_args()

    text = (args.run_dir / "log.txt").read_text(errors="replace")
    pairs = re.findall(
        r"Target domain ([A-Za-z_]+) accuracy: ([0-9]+(?:\.[0-9]+)?)%", text
    )
    per_domain = {}
    for domain, accuracy in pairs:
        per_domain[domain] = float(accuracy)
    if len(per_domain) != 2:
        raise RuntimeError(f"Expected two Office-31 target results, got {per_domain}")

    audits = []
    audit_path = args.run_dir / "curriculum_stage_audit.jsonl"
    if audit_path.exists():
        audits = [json.loads(line) for line in audit_path.read_text().splitlines() if line]
    metrics = {
        "method": "ours",
        "protocol": "office31_mtda_full",
        "source": args.source,
        "seed": args.seed,
        "backbone": "ViT-B/16",
        "hard_to_easy_order": args.order,
        "per_domain_accuracy": per_domain,
        "macro_accuracy": sum(per_domain.values()) / len(per_domain),
        "pseudo_label_variant": "agreement_hard_student_soft",
        "replay_topk_per_domain_class": 8,
        "replay_weight": 0.75,
        "replay_traversal": "cycle",
        "optimizer_steps": sum(int(row["optimizer_steps"]) for row in audits),
        "stage_audits": len(audits),
    }
    output = args.run_dir / "mtda_metrics.json"
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
