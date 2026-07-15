#!/usr/bin/env python3

import argparse
import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "output" / "officehome_mtda"
RESULTS_DIR = ROOT / "results"

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument(
        "--method-tags",
        nargs="+",
        help="Collect only these exact method directory names",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Compute a partial macro average instead of marking incomplete runs as NA",
    )
    return parser.parse_args()


def output_paths(seeds):
    if len(seeds) == 1:
        suffix = f"seed{seeds[0]}"
    else:
        suffix = "seeds" + "-".join(str(seed) for seed in seeds)
    return (
        RESULTS_DIR / f"officehome_maple_mtda_{suffix}.csv",
        RESULTS_DIR / f"officehome_maple_mtda_{suffix}.md",
    )


def summarize_accs(accs, allow_incomplete=False):
    present = [acc for acc in accs if acc is not None]
    complete = len(present) == len(accs)
    if complete or (allow_incomplete and present):
        macro = sum(present) / len(present)
    else:
        macro = None
    return macro, complete, len(present)


def resolve_method_dirs(output_root, method_tags=None):
    if method_tags:
        method_dirs = [output_root / tag for tag in method_tags]
        missing = sorted(path.name for path in method_dirs if not path.is_dir())
        if missing:
            raise FileNotFoundError(
                "Requested method directories not found under "
                f"{output_root}: {', '.join(missing)}"
            )
        return method_dirs

    return [
        path
        for path in sorted(output_root.iterdir())
        if path.is_dir()
        and path.name.startswith(
            (
                "maple_mtda",
                "maple_continuous_mtda",
                "maple_continuous_shared_mtda",
                "maple_gapctx_mtda",
            )
        )
    ]


def main():
    args = parse_args()
    csv_path, md_path = output_paths(args.seeds)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    method_dirs = resolve_method_dirs(OUTPUT_ROOT, args.method_tags)
    for method_dir in method_dirs:
        for seed in args.seeds:
            for source, targets in SOURCES.items():
                target_tag = "".join(targets)
                log_path = (
                    method_dir / f"{source}2{target_tag}" / f"seed{seed}" / "log.txt"
                )
                values = parse_log(log_path)
                accs = [values.get(target) for target in targets]
                macro, complete, num_present = summarize_accs(
                    accs, allow_incomplete=args.allow_incomplete
                )
                rows.append(
                    {
                        "Method": method_dir.name,
                        "Seed": seed,
                        "Source": source,
                        "Target1": f"{targets[0]}:{accs[0]:.2f}" if accs[0] is not None else f"{targets[0]}:NA",
                        "Target2": f"{targets[1]}:{accs[1]:.2f}" if accs[1] is not None else f"{targets[1]}:NA",
                        "Target3": f"{targets[2]}:{accs[2]:.2f}" if accs[2] is not None else f"{targets[2]}:NA",
                        "Macro Avg": f"{macro:.2f}" if macro is not None else "NA",
                        "Targets Found": f"{num_present}/{len(targets)}",
                        "Complete": "yes" if complete else "no",
                        "Log": str(log_path),
                    }
                )

    with csv_path.open("w", newline="") as f:
        fieldnames = [
            "Method",
            "Seed",
            "Source",
            "Target1",
            "Target2",
            "Target3",
            "Macro Avg",
            "Targets Found",
            "Complete",
            "Log",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| Method | Seed | Source | Target1 | Target2 | Target3 | Macro Avg | Targets Found | Complete |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['Method']} | {row['Seed']} | {row['Source']} | {row['Target1']} | "
            f"{row['Target2']} | {row['Target3']} | {row['Macro Avg']} | "
            f"{row['Targets Found']} | {row['Complete']} |"
        )
    md_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
