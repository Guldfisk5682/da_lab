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
METHOD_DIRS = OrderedDict(
    [
        ("cocoop_mt", "CoCoOpMTDA"),
        ("cocoop_vpt", "CoCoOpVPTMTDA"),
        ("style_prompt", "StylePromptMTDA"),
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


def discover_method_dirs(root):
    method_dirs = OrderedDict(METHOD_DIRS)
    if root.is_dir():
        for child in sorted(root.iterdir()):
            if child.is_dir() and child.name.startswith("cocoop_vpt"):
                method_dirs.setdefault(child.name, child.name)
    return method_dirs


def collect(root, seed):
    rows = []
    method_dirs = discover_method_dirs(root)

    for method_dir, method_name in method_dirs.items():
        for source_code, targets in SOURCE_TARGETS.items():
            task_tag = f"{source_code}2{''.join(targets)}"
            log_path = root / method_dir / task_tag / f"seed{seed}" / "log.txt"
            if not log_path.is_file():
                continue
            target_scores, macro_avg = parse_log(log_path)
            rows.append(
                {
                    "method_dir": method_dir,
                    "method_name": method_name,
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


def format_mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return ""
    mean = statistics.mean(values)
    if len(values) == 1:
        return f"{mean:.2f}"
    std = statistics.stdev(values)
    return f"{mean:.2f}+/-{std:.2f}"


def aggregate_rows(rows):
    grouped = OrderedDict()
    for row in rows:
        key = (row["method_name"], row["source"], tuple(row["targets"]))
        grouped.setdefault(key, []).append(row)

    summary = []
    for (method_name, source, targets), group in grouped.items():
        target_stats = OrderedDict()
        for target_code in targets:
            target_stats[target_code] = format_mean_std(
                [row["scores"].get(target_code) for row in group]
            )
        macro_stat = format_mean_std([row["macro_avg"] for row in group])
        summary.append(
            {
                "method_name": method_name,
                "source": source,
                "targets": list(targets),
                "target_stats": target_stats,
                "macro_stat": macro_stat,
            }
        )
    return summary


def write_csv(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Seed", "Source", "Target1", "Target2", "Target3", "Macro Avg"])
        for row in rows:
            cells = []
            for target_code in row["targets"]:
                value = row["scores"].get(target_code)
                cells.append("" if value is None else f"{target_code}:{value:.2f}")
            macro = "" if row["macro_avg"] is None else f"{row['macro_avg']:.2f}"
            writer.writerow([row["method_name"], row["seed"], row["source"], *cells, macro])

        if rows:
            overall_values = [row["macro_avg"] for row in rows if row["macro_avg"] is not None]
            overall = statistics.mean(overall_values) if overall_values else 0.0
            writer.writerow([])
            writer.writerow(["Overall Avg", "", "", "", "", "", f"{overall:.2f}"])

            writer.writerow([])
            writer.writerow(["Mean/Std across seeds"])
            writer.writerow(["Method", "Source", "Target1", "Target2", "Target3", "Macro Avg"])
            for row in aggregate_rows(rows):
                cells = [
                    ""
                    if row["target_stats"].get(target_code) is None
                    else f"{target_code}:{row['target_stats'].get(target_code)}"
                    for target_code in row["targets"]
                ]
                writer.writerow([row["method_name"], row["source"], *cells, row["macro_stat"]])


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
            + " | ".join([row["method_name"], str(row["seed"]), row["source"], *cells, macro])
            + " |"
        )

    if rows:
        overall_values = [row["macro_avg"] for row in rows if row["macro_avg"] is not None]
        overall = statistics.mean(overall_values) if overall_values else 0.0
        lines.append("")
        lines.append(f"Overall average across collected rows: {overall:.2f}%")

        lines.append("")
        lines.append("## Mean/Std Across Seeds")
        lines.append("")
        lines.extend(
            [
                "| Method | Source | Target1 | Target2 | Target3 | Macro Avg |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in aggregate_rows(rows):
            cells = [
                ""
                if row["target_stats"].get(target_code) is None
                else f"{target_code}:{row['target_stats'].get(target_code)}"
                for target_code in row["targets"]
            ]
            lines.append(
                "| "
                + " | ".join([row["method_name"], row["source"], *cells, row["macro_stat"]])
                + " |"
            )

    output_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None, help="single seed to collect")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="one or more seeds to collect and summarize",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    root = repo_root / "output" / "office_home_mtda"
    seeds = args.seeds if args.seeds is not None else [args.seed if args.seed is not None else 1]
    rows = collect_many(root, seeds)

    if len(seeds) == 1:
        suffix = f"seed{seeds[0]}"
    else:
        suffix = "seeds" + "-".join(str(seed) for seed in seeds)

    csv_path = repo_root / "results" / f"officehome_mtda_{suffix}.csv"
    md_path = repo_root / "results" / f"officehome_mtda_{suffix}.md"

    write_csv(rows, csv_path)
    write_markdown(rows, md_path)

    print(csv_path.relative_to(repo_root))
    print(md_path.relative_to(repo_root))


if __name__ == "__main__":
    main()
