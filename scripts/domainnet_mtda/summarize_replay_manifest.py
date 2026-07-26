#!/usr/bin/env python3
"""Summarize the post-hoc quality and class coverage of a MINT PCTM manifest."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def quality(records):
    count = len(records)
    correct = sum(
        int(record["pseudo_label"]) == int(record["true_label"])
        for record in records
    )
    return {
        "selected_samples": count,
        "diagnostic_pseudo_accuracy": correct / count if count else None,
        "predicted_class_coverage": len(
            {int(record["pseudo_label"]) for record in records}
        ),
        "diagnostic_true_class_coverage": len(
            {int(record["true_label"]) for record in records}
        ),
    }


def main():
    args = parse_args()
    payloads = [
        json.loads(line)
        for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not payloads:
        raise ValueError("The replay-selection manifest is empty")

    last_stage = max(int(payload["stage"]) for payload in payloads)
    selected_for_memory = [
        record
        for payload in payloads
        if int(payload["stage"]) < last_stage
        for record in payload["records"]
    ]
    by_domain = defaultdict(list)
    for record in selected_for_memory:
        by_domain[str(record["domain"])].append(record)

    report = {
        "protocol": {
            "kind": "persistent class-balanced target memory",
            "included_stages": list(range(last_stage)),
            "excluded_final_stage": last_stage,
            "labels": "diagnostic only; not used to alter the archived training run",
        },
        "overall": quality(selected_for_memory),
        "per_fitted_domain": {
            domain: quality(records) for domain, records in sorted(by_domain.items())
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
