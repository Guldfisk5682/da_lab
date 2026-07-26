#!/usr/bin/env python3
"""Post-hoc CPD audit for a finished DomainNet MINT checkpoint.

This script is inference-only. It scores the official *target-training* split
with the final student and frozen CLIP teacher, writes one gzip-compressed JSONL
record per image, and emits an aggregate summary. Ground-truth labels are used
only in fields explicitly named ``diagnostic_*`` and never influence model
loading, ordering, or selection.
"""

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
from dassl.engine import build_trainer
from dassl.utils import set_random_seed

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
    parser.add_argument("--source", required=True, choices=sorted(DOMAINS))
    parser.add_argument("--root", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument(
        "--curriculum-stage-audit",
        required=True,
        help="Archived curriculum_stage_audit.jsonl used to recover the actual stage order",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--load-epoch", type=int, default=10)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.7)
    return parser.parse_args()


def build_cfg(args, targets):
    order_cfg = "[{}]".format(
        ",".join(repr(domain) for domain in targets)
    )
    cfg_args = SimpleNamespace(
        root=args.root,
        output_dir=str(Path(args.output_dir) / "runtime"),
        resume="",
        seed=args.seed,
        source_domains=[DOMAINS[args.source]],
        target_domains=targets,
        transforms=None,
        trainer="CurriculumContinuousSharedProjMaPLeMTDA",
        backbone="",
        head="",
        dataset_config_file="configs/datasets/domainnet_mtda.yaml",
        config_file="configs/trainers/CurriculumContinuousSharedProjMaPLeMTDA/vit_b16.yaml",
        opts=[
            "DATALOADER.TEST.BATCH_SIZE",
            str(args.batch_size),
            "DATALOADER.NUM_WORKERS",
            str(args.num_workers),
            "TRAINER.MAPLE_MTDA.CURRICULUM.DOMAIN_ORDER",
            order_cfg,
            "TRAINER.MAPLE_MTDA.CURRICULUM.REPLAY.ENABLED",
            "False",
            "TRAINER.MAPLE_MTDA.CURRICULUM.DIAGNOSTICS.ENABLED",
            "False",
        ],
    )
    return train.setup_cfg(cfg_args)


def load_curriculum_order(path, targets):
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    order = [row["domain"] for row in sorted(rows, key=lambda row: int(row["stage"]))]
    if len(order) != len(targets) or set(order) != set(targets):
        raise ValueError(
            "The stage audit must contain each target domain exactly once; "
            f"expected={targets}, recovered={order}"
        )
    return order


def empty_branch_summary():
    return {
        "eligible_samples": 0,
        "correct_samples": 0,
        "predicted_classes": set(),
        "true_classes": set(),
        "confidence_buckets": defaultdict(lambda: [0, 0]),
    }


def update_branch(summary, *, label, true_label, confidence):
    summary["eligible_samples"] += 1
    summary["correct_samples"] += int(label == true_label)
    summary["predicted_classes"].add(label)
    summary["true_classes"].add(true_label)
    bucket = min(9, max(0, int(confidence * 10)))
    summary["confidence_buckets"][f"{bucket / 10:.1f}-{(bucket + 1) / 10:.1f}"][0] += 1
    summary["confidence_buckets"][f"{bucket / 10:.1f}-{(bucket + 1) / 10:.1f}"][1] += int(
        label == true_label
    )


def finalize_branch(summary):
    samples = summary["eligible_samples"]
    buckets = {}
    for interval, (count, correct) in sorted(summary["confidence_buckets"].items()):
        buckets[interval] = {
            "samples": count,
            "diagnostic_accuracy": correct / count if count else None,
        }
    return {
        "eligible_samples": samples,
        "coverage": samples,
        "diagnostic_accuracy": (
            summary["correct_samples"] / samples if samples else None
        ),
        "predicted_class_coverage": len(summary["predicted_classes"]),
        "diagnostic_true_class_coverage": len(summary["true_classes"]),
        "confidence_quality": buckets,
    }


def main():
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be in [0, 1]")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = [domain for code, domain in DOMAINS.items() if code != args.source]
    curriculum_order = load_curriculum_order(args.curriculum_stage_audit, targets)
    cfg = build_cfg(args, curriculum_order)
    set_random_seed(args.seed)
    trainer = build_trainer(cfg)
    trainer.load_model(args.model_dir, epoch=args.load_epoch)

    stage_by_domain = {
        domain: index + 1 for index, domain in enumerate(curriculum_order)
    }
    per_domain = {}
    overall_hard = empty_branch_summary()
    overall_soft = empty_branch_summary()
    record_path = output_dir / f"{args.source}2O_target_train_cpd_records.jsonl.gz"

    with gzip.open(record_path, "wt", encoding="utf-8") as handle:
        for domain in targets:
            records = trainer._score_domain_for_replay(domain)
            hard = empty_branch_summary()
            soft = empty_branch_summary()
            for record in records:
                student_label = int(record["student_label"])
                clip_label = int(record["clip_label"])
                student_conf = float(record["student_conf"])
                clip_conf = float(record["clip_conf"])
                true_label = int(record.pop("true_label"))
                hard_selected = (
                    student_label == clip_label
                    and student_conf >= args.threshold
                    and clip_conf >= args.threshold
                )
                soft_selected = (
                    student_conf >= args.threshold and clip_conf < args.threshold
                )
                soft_weight = (
                    (student_conf - args.threshold) / (1.0 - args.threshold)
                    if soft_selected and args.threshold < 1.0
                    else 0.0
                )
                payload = {
                    **record,
                    "source_domain": DOMAINS[args.source],
                    "target_domain": domain,
                    "checkpoint_stage": "final",
                    "curriculum_stage_identifier": stage_by_domain[domain],
                    "hard_selected": hard_selected,
                    "hard_pseudo_label": student_label if hard_selected else None,
                    "soft_selected": soft_selected,
                    "soft_target_top1": student_label if soft_selected else None,
                    "soft_weight": soft_weight,
                    # Diagnostic only: not used by the model or any selection rule.
                    "diagnostic_true_label": true_label,
                }
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
                if hard_selected:
                    update_branch(
                        hard,
                        label=student_label,
                        true_label=true_label,
                        confidence=student_conf,
                    )
                    update_branch(
                        overall_hard,
                        label=student_label,
                        true_label=true_label,
                        confidence=student_conf,
                    )
                if soft_selected:
                    update_branch(
                        soft,
                        label=student_label,
                        true_label=true_label,
                        confidence=student_conf,
                    )
                    update_branch(
                        overall_soft,
                        label=student_label,
                        true_label=true_label,
                        confidence=student_conf,
                    )
            per_domain[domain] = {
                "target_train_samples": len(records),
                "hard": finalize_branch(hard),
                "soft": finalize_branch(soft),
            }
            print(json.dumps({domain: per_domain[domain]}, sort_keys=True))

    summary = {
        "protocol": {
            "checkpoint": "finished MINT model, inference only",
            "scored_split": "official target train split",
            "curriculum_order": curriculum_order,
            "threshold": args.threshold,
            "labels": "diagnostic only; never used for training, ordering, or model selection",
        },
        "source_domain": DOMAINS[args.source],
        "per_target_domain": per_domain,
        "overall": {
            "hard": finalize_branch(overall_hard),
            "soft": finalize_branch(overall_soft),
        },
        "records": str(record_path),
    }
    summary_path = output_dir / f"{args.source}2O_target_train_cpd_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
