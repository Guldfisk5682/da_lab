#!/usr/bin/env python3

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


def collect(root, seed):
    rows = []
    for method_dir, method_name in METHOD_DIRS.items():
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
                    "source": source_code,
                    "targets": targets,
                    "scores": target_scores,
                    "macro_avg": macro_avg,
                }
            )
    return rows


def write_csv(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Source", "Target1", "Target2", "Target3", "Macro Avg"])
        for row in rows:
            cells = []
            for target_code in row["targets"]:
                value = row["scores"].get(target_code)
                cells.append("" if value is None else f"{target_code}:{value:.2f}")
            macro = "" if row["macro_avg"] is None else f"{row['macro_avg']:.2f}"
            writer.writerow([row["method_name"], row["source"], *cells, macro])

        if rows:
            overall_values = [row["macro_avg"] for row in rows if row["macro_avg"] is not None]
            overall = statistics.mean(overall_values) if overall_values else 0.0
            writer.writerow([])
            writer.writerow(["Overall Avg", "", "", "", "", f"{overall:.2f}"])


def write_markdown(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| Method | Source | Target1 | Target2 | Target3 | Macro Avg |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]

    for row in rows:
        cells = []
        for target_code in row["targets"]:
            value = row["scores"].get(target_code)
            cells.append("" if value is None else f"{target_code}:{value:.2f}")
        macro = "" if row["macro_avg"] is None else f"{row['macro_avg']:.2f}"
        lines.append(
            "| "
            + " | ".join([row["method_name"], row["source"], *cells, macro])
            + " |"
        )

    if rows:
        overall_values = [row["macro_avg"] for row in rows if row["macro_avg"] is not None]
        overall = statistics.mean(overall_values) if overall_values else 0.0
        lines.append("")
        lines.append(f"Overall average across collected rows: {overall:.2f}%")

    output_path.write_text("\n".join(lines) + "\n")


def main():
    repo_root = Path(__file__).resolve().parents[2]
    root = repo_root / "output" / "office_home_mtda"
    seed = 1
    rows = collect(root, seed)

    csv_path = repo_root / "results" / f"officehome_mtda_seed{seed}.csv"
    md_path = repo_root / "results" / f"officehome_mtda_seed{seed}.md"

    write_csv(rows, csv_path)
    write_markdown(rows, md_path)

    print(csv_path.relative_to(repo_root))
    print(md_path.relative_to(repo_root))


if __name__ == "__main__":
    main()

