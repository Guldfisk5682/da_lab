#!/usr/bin/env python3

import argparse
import os
import re
import statistics
from pathlib import Path


TASK_NAME_MAP = {
    "A2D": "amazon -> dslr",
    "A2W": "amazon -> webcam",
    "D2A": "dslr -> amazon",
    "D2W": "dslr -> webcam",
    "W2A": "webcam -> amazon",
    "W2D": "webcam -> dslr",
}


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


def format_results(results, root, keyword, markdown=False):
    values = [value for _, value in results]
    mean = statistics.mean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0

    if markdown:
        lines = [
            "| Task | Transfer | Accuracy | Log |",
            "| --- | --- | ---: | --- |",
        ]
        for path, value in results:
            rel_path = path.relative_to(root)
            task_tag = rel_path.parts[0] if rel_path.parts else "-"
            transfer = TASK_NAME_MAP.get(task_tag, task_tag)
            lines.append(
                f"| {task_tag} | {transfer} | {value:.2f}% | `{rel_path}` |"
            )
        lines.extend(
            [
                "",
                "| Summary | Value |",
                "| --- | ---: |",
                f"| Count | {len(values)} |",
                f"| Mean {keyword} | {mean:.2f}% |",
                f"| Std {keyword} | {std:.2f}% |",
                f"| Min {keyword} | {min(values):.2f}% |",
                f"| Max {keyword} | {max(values):.2f}% |",
            ]
        )
        return "\n".join(lines) + "\n"

    lines = []
    for path, value in results:
        try:
            rel_path = path.relative_to(root)
            display = str(rel_path)
        except ValueError:
            display = str(path)
        lines.append(f"{display}: {value:.2f}%")
    lines.extend(
        [
            "===",
            f"count: {len(values)}",
            f"mean {keyword}: {mean:.2f}%",
            f"std {keyword}: {std:.2f}%",
            f"min {keyword}: {min(values):.2f}%",
            f"max {keyword}: {max(values):.2f}%",
        ]
    )
    return "\n".join(lines) + "\n"


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
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="render a markdown table instead of plain text",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="optional path to save the rendered summary",
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

    rendered = format_results(results, root, args.keyword, markdown=args.markdown)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered)

    if args.markdown or not args.quiet:
        print(rendered, end="")
    else:
        values = [value for _, value in results]
        print("===")
        print(f"count: {len(values)}")
        print(f"mean {args.keyword}: {statistics.mean(values):.2f}%")
        std = statistics.pstdev(values) if len(values) > 1 else 0.0
        print(f"std {args.keyword}: {std:.2f}%")
        print(f"min {args.keyword}: {min(values):.2f}%")
        print(f"max {args.keyword}: {max(values):.2f}%")


if __name__ == "__main__":
    main()
