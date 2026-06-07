#!/usr/bin/env python3

import argparse
import csv
import math
import os
import re
from collections import OrderedDict
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib-da-lab"),
)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator


SOURCE_TARGETS = OrderedDict(
    [
        ("A", ["C", "P", "R"]),
        ("C", ["A", "P", "R"]),
        ("P", ["A", "C", "R"]),
        ("R", ["A", "C", "P"]),
    ]
)

DOMAIN_TO_CODE = {
    "art": "A",
    "clipart": "C",
    "product": "P",
    "real_world": "R",
}

IMPORTANT_TRAIN_TAGS = [
    "train/loss",
    "train/loss_ce",
    "train/loss_kl",
    "train/weighted_loss_kl",
    "train/loss_pl",
    "train/weighted_loss_pl",
    "train/pl_coverage",
    "train/pl_clip_coverage",
    "train/pl_student_coverage",
    "train/clip_student_agreement",
    "train/loss_em",
    "train/weighted_loss_em",
    "train/acc_src",
    "train/lr",
    "train/source_style_norm",
    "train/target_style_norm",
    "train/gap_style_norm",
    "train/image_token_norm",
    "train/vctx_norm",
]

LOG_VALUE_PATTERNS = OrderedDict(
    [
        ("trainer", r"^trainer:\s*(.+)$"),
        ("seed", r"^seed:\s*(.+)$"),
        ("source_domains", r"^source_domains:\s*(.+)$"),
        ("target_domains", r"^target_domains:\s*(.+)$"),
        ("output_dir", r"^output_dir:\s*(.+)$"),
    ]
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot TensorBoard curves for Office-Home MTDA runs."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Experiment root, default: <repo>/output/office_home_mtda",
    )
    parser.add_argument(
        "--method",
        default=None,
        help="Single method directory to plot, e.g. clip_tssp_pair_gap",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Multiple method directories. Overrides --method when set.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Output directory for figures, default: <repo>/results",
    )
    parser.add_argument(
        "--prefix",
        default="officehome_tensorboard",
        help="Filename prefix for generated figures and csv.",
    )
    parser.add_argument(
        "--train-tags",
        nargs="+",
        default=None,
        help="Optional exact train scalar tags to plot.",
    )
    parser.add_argument(
        "--max-train-tags",
        type=int,
        default=12,
        help="Maximum train scalar tags plotted per method/seed.",
    )
    return parser.parse_args()


def repo_root():
    return Path(__file__).resolve().parents[2]


def find_methods(output_root, methods):
    if methods:
        return [output_root / method for method in methods]
    if not output_root.is_dir():
        return []
    return [
        child
        for child in sorted(output_root.iterdir())
        if child.is_dir() and child.name.startswith("clip_tssp")
    ]


def find_event_dir(run_dir):
    tensorboard_dir = run_dir / "tensorboard"
    if not tensorboard_dir.is_dir():
        return None
    if not any(tensorboard_dir.glob("events.out.tfevents*")):
        return None
    return tensorboard_dir


def task_to_source(task_name):
    match = re.match(r"^([ACPR])2", task_name)
    return match.group(1) if match else task_name


def load_scalars(event_dir):
    accumulator = event_accumulator.EventAccumulator(
        str(event_dir),
        size_guidance={"scalars": 0},
    )
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])
    rows = []
    for tag in tags:
        for event in accumulator.Scalars(tag):
            rows.append(
                {
                    "tag": tag,
                    "step": int(event.step),
                    "value": float(event.value),
                    "wall_time": float(event.wall_time),
                }
            )
    return rows


def collect_runs(output_root, method_dirs, seeds):
    rows = []
    run_infos = []
    for method_dir in method_dirs:
        if not method_dir.is_dir():
            continue
        for task_dir in sorted(method_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            for seed_dir in sorted(task_dir.glob("seed*")):
                if not seed_dir.is_dir():
                    continue
                seed_match = re.match(r"seed(\d+)$", seed_dir.name)
                if not seed_match:
                    continue
                seed = int(seed_match.group(1))
                if seeds and seed not in seeds:
                    continue
                event_dir = find_event_dir(seed_dir)
                if event_dir is None:
                    continue
                scalar_rows = load_scalars(event_dir)
                run_info = {
                    "method": method_dir.name,
                    "task": task_dir.name,
                    "source": task_to_source(task_dir.name),
                    "seed": seed,
                    "run_dir": seed_dir,
                    "event_dir": event_dir,
                    "num_scalars": len(scalar_rows),
                }
                run_infos.append(run_info)
                for row in scalar_rows:
                    row.update(
                        {
                            "method": run_info["method"],
                            "task": run_info["task"],
                            "source": run_info["source"],
                            "seed": seed,
                        }
                    )
                    rows.append(row)
    return pd.DataFrame(rows), run_infos


def normalize_target_tag(tag):
    if tag == "test/macro_avg":
        return "Macro"
    match = re.match(r"^test/(.+)_accuracy$", tag)
    if not match:
        return None
    domain_name = match.group(1)
    return DOMAIN_TO_CODE.get(domain_name, domain_name)


def eval_dataframe(df):
    if df.empty:
        return df
    records = []
    for row in df.itertuples(index=False):
        label = normalize_target_tag(row.tag)
        if label is None:
            continue
        records.append(
            {
                "method": row.method,
                "task": row.task,
                "source": row.source,
                "seed": row.seed,
                "step": row.step,
                "target": label,
                "value": row.value,
            }
        )
    return pd.DataFrame(records)


def select_train_tags(df, requested_tags, max_tags):
    if df.empty:
        return []
    available = set(df["tag"].unique())
    if requested_tags:
        return [tag for tag in requested_tags if tag in available]

    selected = [tag for tag in IMPORTANT_TRAIN_TAGS if tag in available]
    if len(selected) < max_tags:
        extra = sorted(
            tag
            for tag in available
            if tag.startswith("train/")
            and tag not in selected
            and any(key in tag for key in ("loss", "coverage", "agreement", "acc"))
        )
        selected.extend(extra)
    return selected[:max_tags]


def parse_log_summary(run_dir):
    log_path = run_dir / "log.txt"
    summary = OrderedDict()
    if not log_path.is_file():
        return summary

    text = log_path.read_text(errors="ignore")
    for key, pattern in LOG_VALUE_PATTERNS.items():
        match = re.search(pattern, text, flags=re.MULTILINE)
        if match:
            summary[key] = match.group(1).strip()

    for section, keys in [
        ("OPTIM", ["NAME", "LR", "WEIGHT_DECAY", "MAX_EPOCH", "LR_SCHEDULER"]),
        (
            "CLIP_TSSP_MTDA",
            [
                "STYLE_GROUP_SIZE",
                "GAP_GROUP_SIZE",
                "USE_GAP_TOKEN",
                "LAMBDA_EM",
                "ENABLE_VPT",
                "DETACH_ENTROPY_TEXT",
            ],
        ),
    ]:
        block = extract_config_block(text, section)
        for cfg_key in keys:
            value = find_config_value(block, cfg_key)
            if value is not None:
                summary[f"{section}.{cfg_key}"] = value
    return summary


def extract_config_block(text, section):
    pattern = rf"^{re.escape(section)}:\n((?:^[ \t].*\n?)*)"
    match = re.search(pattern, text, flags=re.MULTILINE)
    return match.group(1) if match else ""


def find_config_value(block, key):
    match = re.search(rf"^\s+{re.escape(key)}:\s*(.+)$", block, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def write_summary_csv(df, run_infos, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    last_values = []
    if not df.empty:
        idx = df.sort_values("step").groupby(
            ["method", "task", "seed", "tag"], sort=False
        ).tail(1)
        for row in idx.itertuples(index=False):
            last_values.append(
                {
                    "method": row.method,
                    "task": row.task,
                    "seed": row.seed,
                    "tag": row.tag,
                    "last_step": row.step,
                    "last_value": row.value,
                }
            )

    hparams_by_run = {
        (info["method"], info["task"], info["seed"]): parse_log_summary(
            info["run_dir"]
        )
        for info in run_infos
    }
    hparam_keys = sorted(
        {key for summary in hparams_by_run.values() for key in summary.keys()}
    )

    with output_path.open("w", newline="") as f:
        fieldnames = [
            "method",
            "task",
            "seed",
            "tag",
            "last_step",
            "last_value",
            *hparam_keys,
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in last_values:
            summary = hparams_by_run.get(
                (record["method"], record["task"], record["seed"]), {}
            )
            writer.writerow({**record, **summary})


def setup_style():
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.dpi"] = 180


def write_empty_figure(path, title, message):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_eval(eval_df, run_infos, output_path, title):
    if eval_df.empty:
        write_empty_figure(
            output_path,
            title,
            "No TensorBoard eval scalars found. Expected tags like test/clipart_accuracy or test/macro_avg.",
        )
        return

    sources = sorted(eval_df["source"].unique())
    ncols = min(2, max(1, len(sources)))
    nrows = math.ceil(len(sources) / ncols)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(8 * ncols, 4.8 * nrows),
        squeeze=False,
    )
    for ax, source in zip(axes.flat, sources):
        sub = eval_df[eval_df["source"] == source]
        sns.lineplot(
            data=sub,
            x="step",
            y="value",
            hue="target",
            style="task",
            marker="o",
            ax=ax,
        )
        ax.set_title(f"Source {source}: target accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.legend(loc="best", fontsize="small")

    for ax in axes.flat[len(sources):]:
        ax.axis("off")

    subtitle = f"Runs: {len(run_infos)}"
    fig.suptitle(f"{title}\n{subtitle}", y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_train(df, run_infos, output_path, title, requested_tags, max_tags):
    train_df = df[df["tag"].str.startswith("train/")] if not df.empty else df
    tags = select_train_tags(train_df, requested_tags, max_tags)
    if train_df.empty or not tags:
        write_empty_figure(
            output_path,
            title,
            "No selected TensorBoard train scalars found.",
        )
        return

    selected_df = train_df[train_df["tag"].isin(tags)].copy()
    selected_df["run"] = (
        selected_df["method"]
        + "/"
        + selected_df["task"]
        + "/seed"
        + selected_df["seed"].astype(str)
    )

    ncols = 2
    nrows = math.ceil(len(tags) / ncols)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(9 * ncols, 3.8 * nrows),
        squeeze=False,
    )
    for ax, tag in zip(axes.flat, tags):
        sub = selected_df[selected_df["tag"] == tag]
        sns.lineplot(
            data=sub,
            x="step",
            y="value",
            hue="task",
            estimator=None,
            alpha=0.9,
            ax=ax,
        )
        ax.set_title(tag)
        ax.set_xlabel("Iter/Epoch step")
        ax.set_ylabel("Value")
        ax.legend(loc="best", fontsize="x-small")

    for ax in axes.flat[len(tags):]:
        ax.axis("off")

    hparam_text = format_hparams(run_infos)
    fig.suptitle(f"{title}\n{hparam_text}", y=1.01, fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def format_hparams(run_infos):
    if not run_infos:
        return "No run metadata"
    summary = parse_log_summary(run_infos[0]["run_dir"])
    keys = [
        "trainer",
        "OPTIM.NAME",
        "OPTIM.LR",
        "OPTIM.WEIGHT_DECAY",
        "OPTIM.MAX_EPOCH",
        "CLIP_TSSP_MTDA.STYLE_GROUP_SIZE",
        "CLIP_TSSP_MTDA.GAP_GROUP_SIZE",
        "CLIP_TSSP_MTDA.LAMBDA_EM",
        "CLIP_TSSP_MTDA.ENABLE_VPT",
    ]
    parts = [f"runs={len(run_infos)}"]
    for key in keys:
        if key in summary:
            parts.append(f"{key}={summary[key]}")
    return " | ".join(parts)


def output_stem(methods, seeds, prefix):
    method_part = "multi" if len(methods) != 1 else methods[0].name
    seed_part = "allseeds" if not seeds else "seeds" + "-".join(map(str, seeds))
    return f"{prefix}_{method_part}_{seed_part}"


def main():
    args = parse_args()
    root = repo_root()
    output_root = args.output_root or root / "output" / "office_home_mtda"
    results_dir = args.results_dir or root / "results"
    seeds = args.seeds if args.seeds is not None else (
        [args.seed] if args.seed is not None else None
    )
    methods = args.methods if args.methods is not None else (
        [args.method] if args.method else None
    )

    setup_style()
    method_dirs = find_methods(output_root, methods)
    df, run_infos = collect_runs(output_root, method_dirs, seeds)
    stem = output_stem(method_dirs, seeds, args.prefix)

    eval_path = results_dir / f"{stem}_eval.png"
    train_path = results_dir / f"{stem}_train.png"
    csv_path = results_dir / f"{stem}_summary.csv"

    title = f"TensorBoard curves: {', '.join(path.name for path in method_dirs) or 'none'}"
    plot_eval(eval_dataframe(df), run_infos, eval_path, title)
    plot_train(df, run_infos, train_path, title, args.train_tags, args.max_train_tags)
    write_summary_csv(df, run_infos, csv_path)

    print(eval_path.relative_to(root) if eval_path.is_relative_to(root) else eval_path)
    print(train_path.relative_to(root) if train_path.is_relative_to(root) else train_path)
    print(csv_path.relative_to(root) if csv_path.is_relative_to(root) else csv_path)


if __name__ == "__main__":
    main()
