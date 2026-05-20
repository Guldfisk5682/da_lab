#!/usr/bin/env python3

import argparse
import os
import re
import statistics
from pathlib import Path


def parse_last_metric(lines, regex, end_signal):
    values = []
    in_result = False
    for raw in lines:
        line = raw.strip()
        if line == end_signal:
            in_result = True
            continue
        if in_result:
            match = regex.search(line)
            if match:
                try:
                    values.append(float(match.group(1)))
                except ValueError:
                    continue
    return values[-1] if values else None


def collect_results(root, pattern, keyword, end_signal):
    regex = re.compile(rf"\* {re.escape(keyword)}: ([\d.+-eE]+)%")
    results = []
    for path in sorted(Path(root).glob(pattern)):
        if not path.is_file():
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        value = parse_last_metric(text.splitlines(), regex, end_signal)
        if value is None:
            continue
        results.append((path, value))
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Summarize Office-31 eval accuracy from log files."
    )
    parser.add_argument(
        "--root",
        default="output/office31",
        help="root directory containing Office-31 outputs",
    )
    parser.add_argument(
        "--pattern",
        default="**/log.txt",
        help="glob pattern for log files under root",
    )
    parser.add_argument(
        "--trainer",
        default=None,
        help="trainer name to filter (e.g., CoCoOp, CoCoOpDAV0)",
    )
    parser.add_argument(
        "--trainer-dir",
        default=None,
        help="trainer output dir name (overrides --trainer and env)",
    )
    parser.add_argument(
        "--keyword",
        default="accuracy",
        help="metric name to extract (e.g., accuracy, macro_f1)",
    )
    parser.add_argument(
        "--end-signal",
        default="=> result",
        help="line that marks the start of an eval result block",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="only print the summary",
    )
    args = parser.parse_args()

    trainer_dir = args.trainer_dir
    if not trainer_dir:
        trainer_dir = args.trainer or os.environ.get("TRAINER_DIR")
    if not trainer_dir:
        trainer_dir = os.environ.get("TRAINER")

    root = Path(args.root)
    if trainer_dir:
        root = root / trainer_dir

    results = collect_results(str(root), args.pattern, args.keyword, args.end_signal)
    if not results:
        print(f"No matching results found under {root}")
        raise SystemExit(1)

    values = [value for _, value in results]

    if not args.quiet:
        for path, value in results:
            try:
                rel_path = path.relative_to(root)
                display = str(rel_path)
            except ValueError:
                display = str(path)
            print(f"{display}: {value:.2f}%")

    mean = statistics.mean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0

    print("===")
    print(f"count: {len(values)}")
    print(f"mean {args.keyword}: {mean:.2f}%")
    print(f"std {args.keyword}: {std:.2f}%")
    print(f"min {args.keyword}: {min(values):.2f}%")
    print(f"max {args.keyword}: {max(values):.2f}%")


if __name__ == "__main__":
    main()
