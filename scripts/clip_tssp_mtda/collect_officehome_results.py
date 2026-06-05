#!/usr/bin/env python3

import argparse
import csv
import re
import statistics
from collections import OrderedDict
from pathlib import Path


SOURCE_TARGETS = OrderedDict(
    [
        ("A", ["C", "P", "R"]),
        ("C", ["A", "P", "R"]),
        ("P", ["A", "C", "R"]),
        ("R", ["A", "C", "P"]),
    ]
)
TARGET_RE = re.compile(r"Target domain ([a-z_]+) accuracy: ([\d.+-eE]+)%")
MACRO_RE = re.compile(r"Per-source macro average: ([\d.+-eE]+)%")
DOMAIN_TO_CODE = {
    "art": "A",
    "clipart": "C",
    "product": "P",
    "real_world": "R",
}


def parse_log(log_path):
    target_scores = OrderedDict()
    macro_avg = None
    for raw_line in log_path.read_text(errors="ignore").splitlines():
        line = raw_line.strip()
        match = TARGET_RE.search(line)
        if match:
            domain_name, value = match.groups()
            target_scores[DOMAIN_TO_CODE.get(domain_name, domain_name)] = float(value)
            continue
        match = MACRO_RE.search(line)
        if match:
            macro_avg = float(match.group(1))
    return target_scores, macro_avg


def collect(root, seed):
    rows = []
    if not root.is_dir():
        return rows

    method_dirs = [
        child for child in sorted(root.iterdir()) if child.name.startswith("clip_tssp")
    ]
    for method_dir in method_dirs:
        for source_code, targets in SOURCE_TARGETS.items():
            task_tag = f"{source_code}2{''.join(targets)}"
            log_path = method_dir / task_tag / f"seed{seed}" / "log.txt"
            if not log_path.is_file():
                continue
            target_scores, macro_avg = parse_log(log_path)
            rows.append(
                {
                    "method": method_dir.name,
                    "seed": seed,
                    "source": source_code,
                    "targets": targets,
                    "scores": target_scores,
                    "macro_avg": macro_avg,
                }
            )
    return rows


def collect_many(root, seeds):
    rows = []
    for seed in seeds:
        rows.extend(collect(root, seed))
    return rows


def write_csv(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Method", "Seed", "Source", "Target1", "Target2", "Target3", "Macro Avg"]
        )
        for row in rows:
            cells = []
            for target_code in row["targets"]:
                value = row["scores"].get(target_code)
                cells.append("" if value is None else f"{target_code}:{value:.2f}")
            macro = "" if row["macro_avg"] is None else f"{row['macro_avg']:.2f}"
            writer.writerow([row["method"], row["seed"], row["source"], *cells, macro])


def write_markdown(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| Method | Seed | Source | Target1 | Target2 | Target3 | Macro Avg |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        cells = []
        for target_code in row["targets"]:
            value = row["scores"].get(target_code)
            cells.append("" if value is None else f"{target_code}:{value:.2f}")
        macro = "" if row["macro_avg"] is None else f"{row['macro_avg']:.2f}"
        lines.append(
            "| "
            + " | ".join([row["method"], str(row["seed"]), row["source"], *cells, macro])
            + " |"
        )

    if rows:
        overall_values = [row["macro_avg"] for row in rows if row["macro_avg"] is not None]
        overall = statistics.mean(overall_values) if overall_values else 0.0
        lines.append("")
        lines.append(f"Overall average across collected rows: {overall:.2f}%")

    output_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    root = repo_root / "output" / "office_home_mtda"
    seeds = args.seeds if args.seeds is not None else [args.seed if args.seed is not None else 1]
    rows = collect_many(root, seeds)

    suffix = f"seed{seeds[0]}" if len(seeds) == 1 else "seeds" + "-".join(str(seed) for seed in seeds)
    csv_path = repo_root / "results" / f"officehome_clip_tssp_{suffix}.csv"
    md_path = repo_root / "results" / f"officehome_clip_tssp_{suffix}.md"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    print(csv_path.relative_to(repo_root))
    print(md_path.relative_to(repo_root))


if __name__ == "__main__":
    main()
