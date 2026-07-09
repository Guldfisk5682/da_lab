#!/usr/bin/env python3

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "output" / "officehome_mtda"
RESULTS_DIR = ROOT / "results"
CSV_PATH = RESULTS_DIR / "officehome_maple_mtda_seed1.csv"
MD_PATH = RESULTS_DIR / "officehome_maple_mtda_seed1.md"

SOURCES = {
    "A": ["C", "P", "R"],
    "C": ["A", "P", "R"],
    "P": ["A", "C", "R"],
    "R": ["A", "C", "P"],
}

DOMAIN_CODE = {
    "art": "A",
    "clipart": "C",
    "product": "P",
    "real_world": "R",
}

ACC_RE = re.compile(r"Target domain ([A-Za-z_]+) accuracy:\s*([0-9.]+)%")


def parse_log(path):
    values = {}
    if not path.exists():
        return values
    text = path.read_text(errors="ignore")
    for domain, acc in ACC_RE.findall(text):
        code = DOMAIN_CODE.get(domain.lower())
        if code:
            values[code] = float(acc)
    return values


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    method_dirs = [
        path for path in sorted(OUTPUT_ROOT.iterdir())
        if path.is_dir() and path.name.startswith("maple_mtda")
    ]
    for method_dir in method_dirs:
        for source, targets in SOURCES.items():
            target_tag = "".join(targets)
            log_path = method_dir / f"{source}2{target_tag}" / "seed1" / "log.txt"
            values = parse_log(log_path)
            accs = [values.get(target) for target in targets]
            present = [acc for acc in accs if acc is not None]
            macro = sum(present) / len(present) if present else None
            rows.append(
                {
                    "Method": method_dir.name,
                    "Source": source,
                    "Target1": f"{targets[0]}:{accs[0]:.2f}" if accs[0] is not None else f"{targets[0]}:NA",
                    "Target2": f"{targets[1]}:{accs[1]:.2f}" if accs[1] is not None else f"{targets[1]}:NA",
                    "Target3": f"{targets[2]}:{accs[2]:.2f}" if accs[2] is not None else f"{targets[2]}:NA",
                    "Macro Avg": f"{macro:.2f}" if macro is not None else "NA",
                    "Log": str(log_path),
                }
            )

    with CSV_PATH.open("w", newline="") as f:
        fieldnames = ["Method", "Source", "Target1", "Target2", "Target3", "Macro Avg", "Log"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| Method | Source | Target1 | Target2 | Target3 | Macro Avg |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['Method']} | {row['Source']} | {row['Target1']} | "
            f"{row['Target2']} | {row['Target3']} | {row['Macro Avg']} |"
        )
    MD_PATH.write_text("\n".join(lines) + "\n")
    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {MD_PATH}")


if __name__ == "__main__":
    main()
