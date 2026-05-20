#!/usr/bin/env python3

import csv
import re
import statistics
from pathlib import Path


TASKS = ["A2W", "A2D", "D2W", "D2A", "W2D", "W2A"]
EXPECTED_EXPERIMENTS = ["L0_full", "L1_fixed_gate", "L2_normal_only"]
ACCURACY_RE = re.compile(r"\* accuracy: ([\d.+-eE]+)%")


def parse_last_accuracy(log_path):
    text = log_path.read_text(errors="ignore").splitlines()
    in_result = False
    values = []

    for raw in text:
        line = raw.strip()
        if line == "=> result":
            in_result = True
            continue
        if in_result:
            match = ACCURACY_RE.search(line)
            if match:
                values.append(float(match.group(1)))

    if values:
        return values[-1]

    candidate_lines = [line.strip() for line in text if "accuracy" in line.lower()]
    if candidate_lines:
        print(f"[warn] Could not parse final accuracy from {log_path}")
        for line in candidate_lines[-5:]:
            print(f"  candidate: {line}")

    return None


def collect(root, seed):
    records = {}
    root_path = Path(root)
    discovered = []

    if root_path.exists():
        discovered = sorted(
            path.name for path in root_path.iterdir() if path.is_dir()
        )

    experiments = []
    for name in EXPECTED_EXPERIMENTS + discovered:
        if name not in experiments:
            experiments.append(name)

    for exp_name in experiments:
        task_scores = {}
        for task in TASKS:
            log_path = root_path / exp_name / task / f"seed{seed}" / "log.txt"
            if not log_path.is_file():
                continue
            accuracy = parse_last_accuracy(log_path)
            if accuracy is not None:
                task_scores[task] = accuracy
        records[exp_name] = task_scores

    return records


def summarize(scores):
    values = list(scores.values())
    if not values:
        return "", ""
    mean = statistics.mean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return f"{mean:.2f}", f"{std:.2f}"


def write_csv(records, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Exp", *TASKS, "Mean", "Std"])
        for exp_name, scores in records.items():
            mean, std = summarize(scores)
            row = [exp_name]
            for task in TASKS:
                value = scores.get(task)
                row.append("" if value is None else f"{value:.2f}")
            row.extend([mean, std])
            writer.writerow(row)


def write_markdown(records, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| Exp | A2W | A2D | D2W | D2A | W2D | W2A | Mean | Std |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for exp_name, scores in records.items():
        mean, std = summarize(scores)
        row = [exp_name]
        for task in TASKS:
            value = scores.get(task)
            row.append("" if value is None else f"{value:.2f}")
        row.extend([mean, std])
        lines.append("| " + " | ".join(row) + " |")

    output_path.write_text("\n".join(lines) + "\n")


def main():
    repo_root = Path(__file__).resolve().parents[2]
    root = repo_root / "output" / "office31_ablation"
    seed = 1
    records = collect(root, seed)

    csv_path = repo_root / "results" / f"office31_ablation_seed{seed}.csv"
    md_path = repo_root / "results" / f"office31_ablation_seed{seed}.md"

    write_csv(records, csv_path)
    write_markdown(records, md_path)

    print(csv_path.relative_to(repo_root))
    print(md_path.relative_to(repo_root))


if __name__ == "__main__":
    main()
