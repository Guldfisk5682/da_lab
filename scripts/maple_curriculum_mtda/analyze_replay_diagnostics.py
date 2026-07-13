#!/usr/bin/env python3
"""Aggregate sample-level curriculum PL/replay diagnostics.

The input is ``pl_sample_audit.jsonl`` emitted at every stage boundary when
CURRICULUM.DIAGNOSTICS.ENABLED is true. Ground truth is used for diagnosis
only; none of these metrics participate in model selection or training.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def summarize_records(records):
    count = len(records)
    agreement = [record for record in records if record["agreement"]]
    clean_pl = [record for record in records if record["clean_pl_selected"]]
    eligible = [record for record in records if record["eligible"]]
    standard = [record for record in records if record["standard_topk"]]
    oracle = [record for record in records if record["oracle_correct_topk"]]
    actual = [record for record in records if record["actual_selected"]]

    def precision(items, key="student_correct"):
        return safe_ratio(sum(bool(item[key]) for item in items), len(items))

    return {
        "samples": count,
        "student_predicted_class_coverage": len(
            {int(record["student_label"]) for record in records}
        ),
        "clip_predicted_class_coverage": len(
            {int(record["clip_label"]) for record in records}
        ),
        "true_class_coverage": len({int(record["true_label"]) for record in records}),
        "student_accuracy": precision(records),
        "clip_accuracy": precision(records, "clip_correct"),
        "agreement_coverage": safe_ratio(len(agreement), count),
        "agreement_precision": precision(agreement),
        "clean_pl": len(clean_pl),
        "clean_pl_coverage": safe_ratio(len(clean_pl), count),
        "clean_pl_teacher_precision": precision(clean_pl, "clip_correct"),
        "both_wrong_agree_rate": safe_ratio(
            sum(bool(record["both_wrong_agree"]) for record in records), count
        ),
        "both_wrong_given_agreement": safe_ratio(
            sum(bool(record["both_wrong_agree"]) for record in agreement),
            len(agreement),
        ),
        "eligible": len(eligible),
        "eligible_predicted_class_coverage": len(
            {int(record["student_label"]) for record in eligible}
        ),
        "eligible_coverage": safe_ratio(len(eligible), count),
        "eligible_precision": precision(eligible),
        "standard_topk": len(standard),
        "standard_topk_true_class_coverage": len(
            {int(record["true_label"]) for record in standard}
        ),
        "standard_topk_precision": precision(standard),
        "oracle_correct_topk": len(oracle),
        "oracle_correct_topk_precision": precision(oracle),
        "actual_selected": len(actual),
        "actual_selected_precision": precision(actual),
        "mean_student_conf": safe_ratio(
            sum(float(record["student_conf"]) for record in records), count
        ),
        "mean_clip_conf": safe_ratio(
            sum(float(record["clip_conf"]) for record in records), count
        ),
    }


def summarize_transitions(previous, current):
    previous_by_path = {record["impath"]: record for record in previous}
    pairs = [
        (previous_by_path[record["impath"]], record)
        for record in current
        if record["impath"] in previous_by_path
    ]
    count = len(pairs)
    previous_topk = {
        record["impath"] for record in previous if record["standard_topk"]
    }
    current_topk = {
        record["impath"] for record in current if record["standard_topk"]
    }
    union = previous_topk | current_topk
    return {
        "matched_samples": count,
        "student_label_flip_rate": safe_ratio(
            sum(a["student_label"] != b["student_label"] for a, b in pairs),
            count,
        ),
        "clip_label_flip_rate": safe_ratio(
            sum(a["clip_label"] != b["clip_label"] for a, b in pairs), count
        ),
        "correct_to_wrong_rate": safe_ratio(
            sum(a["student_correct"] and not b["student_correct"] for a, b in pairs),
            count,
        ),
        "wrong_to_correct_rate": safe_ratio(
            sum(not a["student_correct"] and b["student_correct"] for a, b in pairs),
            count,
        ),
        "agreement_flip_rate": safe_ratio(
            sum(a["agreement"] != b["agreement"] for a, b in pairs), count
        ),
        "persistent_both_wrong_agree_rate": safe_ratio(
            sum(a["both_wrong_agree"] and b["both_wrong_agree"] for a, b in pairs),
            count,
        ),
        "mean_student_conf_delta": safe_ratio(
            sum(float(b["student_conf"]) - float(a["student_conf"]) for a, b in pairs),
            count,
        ),
        "standard_topk_jaccard": safe_ratio(
            len(previous_topk & current_topk), len(union)
        ),
    }


def build_report(records):
    snapshots = defaultdict(list)
    for record in records:
        key = (int(record["boundary_stage"]), str(record["domain"]))
        snapshots[key].append(record)

    snapshot_report = {}
    per_predicted_class = {}
    for (stage, domain), items in sorted(snapshots.items()):
        key = f"stage{stage}:{domain}"
        snapshot_report[key] = summarize_records(items)
        grouped = defaultdict(list)
        for item in items:
            grouped[int(item["student_label"])].append(item)
        per_predicted_class[key] = {
            str(predicted_class): summarize_records(class_items)
            for predicted_class, class_items in sorted(grouped.items())
        }

    transitions = {}
    domains = sorted({domain for _, domain in snapshots})
    for domain in domains:
        stages = sorted(stage for stage, item_domain in snapshots if item_domain == domain)
        for previous_stage, current_stage in zip(stages, stages[1:]):
            key = f"{domain}:stage{previous_stage}->stage{current_stage}"
            transitions[key] = summarize_transitions(
                snapshots[(previous_stage, domain)],
                snapshots[(current_stage, domain)],
            )

    return {
        "snapshots": snapshot_report,
        "per_predicted_class": per_predicted_class,
        "transitions": transitions,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    input_path = run_dir / "pl_sample_audit.jsonl"
    records = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = build_report(records)
    output_path = (
        Path(args.output)
        if args.output
        else run_dir / "replay_diagnostic_summary.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
