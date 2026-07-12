import argparse
import csv
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from dassl.engine import build_trainer
from dassl.utils import set_random_seed
from tqdm import tqdm

import train


DOMAIN_CODES = OrderedDict(
    [
        ("art", "A"),
        ("clipart", "C"),
        ("product", "P"),
        ("real_world", "R"),
    ]
)
CODE_TO_DOMAIN = {v: k for k, v in DOMAIN_CODES.items()}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze frozen-CLIP teacher and trained MaPLe student blind spots "
            "on Office-Home MTDA targets."
        )
    )
    parser.add_argument("--root", default="data", help="Dataset parent directory")
    parser.add_argument(
        "--dataset-config-file",
        default="configs/datasets/office_home_mtda.yaml",
    )
    parser.add_argument(
        "--config-file",
        default="configs/trainers/ContinuousSharedProjMaPLeMTDA/vit_b16.yaml",
    )
    parser.add_argument("--trainer", default="ContinuousSharedProjMaPLeMTDA")
    parser.add_argument("--method-tag", default="maple_continuous_shared_mtda_pl03_seed42")
    parser.add_argument("--output-root", default="output/officehome_mtda")
    parser.add_argument("--analysis-dir", default="results/pl_blindspots")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["A", "C", "P", "R"],
        choices=["A", "C", "P", "R"],
    )
    parser.add_argument("--load-epoch", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--conf-threshold", type=float, default=0.7)
    parser.add_argument("--weak-fraction", type=float, default=0.2)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Use CPU if GPU memory is not available; CPU is slower but works.",
    )
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="Extra config opts appended after analysis defaults.",
    )
    return parser.parse_args()


def target_codes_for_source(source_code):
    return [code for code in CODE_TO_DOMAIN if code != source_code]


def task_name(source_code):
    return f"{source_code}2{''.join(target_codes_for_source(source_code))}"


def setup_cfg(args, source_code, target_domains):
    opts = [
        "DATALOADER.TEST.BATCH_SIZE",
        str(args.batch_size),
        "DATALOADER.NUM_WORKERS",
        str(args.num_workers),
        "TRAINER.MAPLE_MTDA.LAMBDA_PL",
        "0.3",
        "TRAINER.MAPLE_MTDA.PL_THRESHOLD",
        str(args.conf_threshold),
        "TRAINER.MAPLE_MTDA.PL_STUDENT_THRESHOLD",
        str(args.conf_threshold),
        "TRAINER.MAPLE_MTDA.PL_USE_STUDENT_LOW_CONF_MASK",
        "True",
        "TRAINER.MAPLE_MTDA.WEAK_PL.ENABLED",
        "False",
        "TRAINER.MAPLE_MTDA.WEAK_PL.LAMBDA",
        "0.0",
    ]
    if args.device == "cpu":
        opts += ["USE_CUDA", "False"]
    if args.opts:
        extra = args.opts[1:] if args.opts and args.opts[0] == "--" else args.opts
        opts += extra

    cfg_args = SimpleNamespace(
        root=args.root,
        output_dir=str(Path(args.analysis_dir) / "tmp" / task_name(source_code)),
        resume="",
        seed=args.seed,
        source_domains=[CODE_TO_DOMAIN[source_code]],
        target_domains=target_domains,
        transforms=None,
        trainer=args.trainer,
        backbone="",
        head="",
        dataset_config_file=args.dataset_config_file,
        config_file=args.config_file,
        opts=opts,
    )
    return train.setup_cfg(cfg_args)


def model_dir_for(args, source_code):
    return Path(args.output_root) / args.method_tag / task_name(source_code) / f"seed{args.seed}"


def make_class_stats(num_classes):
    return [
        {
            "n": 0,
            "teacher_correct": 0,
            "student_correct": 0,
            "both_low": 0,
            "clean_candidate": 0,
            "clean_help": 0,
            "clean_harm": 0,
            "teacher_low_student_high": 0,
            "teacher_high_student_high": 0,
            "teacher_conf": [],
            "student_conf": [],
            "teacher_true_prob": [],
            "student_true_prob": [],
        }
        for _ in range(num_classes)
    ]


def quantile(sorted_values, q):
    if not sorted_values:
        return 0.0
    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    weight = pos - lo
    return float(sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight)


def mean(values):
    return float(sum(values) / len(values)) if values else 0.0


def bottom_mean(values, k=3):
    if not values:
        return 0.0
    values = sorted(values)
    return mean(values[: min(k, len(values))])


def rfc_weak_mask(pred_labels, probs, num_classes, weak_fraction):
    entropy = -(probs * (probs + 1e-6).log()).sum(dim=1)
    count = torch.bincount(pred_labels, minlength=num_classes).float()
    count_dist = count / count.sum().clamp_min(1.0)
    entropy_sum = torch.zeros(num_classes)
    entropy_sum.scatter_add_(0, pred_labels.cpu(), entropy.cpu())
    observed = count > 0
    entropy_avg = torch.full((num_classes,), float(entropy.mean().item()))
    entropy_avg[observed] = entropy_sum[observed] / count[observed]

    count_order = torch.argsort(count_dist, descending=False)
    entropy_order = torch.argsort(entropy_avg, descending=True)
    rank_values = torch.arange(num_classes, 0, -1, dtype=torch.float)
    count_rank = torch.empty_like(count_dist)
    entropy_rank = torch.empty_like(entropy_avg)
    count_rank[count_order] = rank_values
    entropy_rank[entropy_order] = rank_values
    weak_score = count_rank + entropy_rank
    weak_score = torch.where(observed, weak_score, torch.full_like(weak_score, -float("inf")))
    topk = max(1, int(round(float(weak_fraction) * num_classes)))
    topk = min(int(observed.float().sum().item()), topk)
    mask = torch.zeros(num_classes, dtype=torch.bool)
    if topk > 0:
        mask[torch.topk(weak_score, k=topk, largest=True).indices] = True
    return mask, count_dist, entropy_avg, weak_score


@torch.no_grad()
def analyze_domain(trainer, domain_name, loader, num_classes, threshold, weak_fraction):
    class_stats = make_class_stats(num_classes)
    all_teacher_probs = []
    all_student_probs = []
    all_teacher_pred = []
    all_student_pred = []

    trainer.set_model_mode("eval")
    for batch in tqdm(loader, desc=f"analyze {domain_name}", leave=False):
        image = batch["img"].to(trainer.device)
        label = batch["label"].to(trainer.device)

        student_logits = trainer.model_inference(image, domain_name=domain_name)
        teacher_logits = trainer.model._compute_reference_logits(image)
        student_prob = F.softmax(student_logits.float(), dim=1)
        teacher_prob = F.softmax(teacher_logits.float(), dim=1)

        teacher_conf, teacher_pred = teacher_prob.max(dim=1)
        student_conf, student_pred = student_prob.max(dim=1)
        teacher_true_prob = teacher_prob.gather(1, label.view(-1, 1)).squeeze(1)
        student_true_prob = student_prob.gather(1, label.view(-1, 1)).squeeze(1)

        all_teacher_probs.append(teacher_prob.cpu())
        all_student_probs.append(student_prob.cpu())
        all_teacher_pred.append(teacher_pred.cpu())
        all_student_pred.append(student_pred.cpu())

        for i in range(label.numel()):
            y = int(label[i].item())
            item = class_stats[y]
            t_correct = int(teacher_pred[i].item() == y)
            s_correct = int(student_pred[i].item() == y)
            t_high = bool(teacher_conf[i].item() >= threshold)
            s_high = bool(student_conf[i].item() >= threshold)
            clean_candidate = t_high and not s_high

            item["n"] += 1
            item["teacher_correct"] += t_correct
            item["student_correct"] += s_correct
            item["both_low"] += int((not t_high) and (not s_high))
            item["clean_candidate"] += int(clean_candidate)
            item["clean_help"] += int(clean_candidate and t_correct)
            item["clean_harm"] += int(clean_candidate and not t_correct)
            item["teacher_low_student_high"] += int((not t_high) and s_high)
            item["teacher_high_student_high"] += int(t_high and s_high)
            item["teacher_conf"].append(float(teacher_conf[i].item()))
            item["student_conf"].append(float(student_conf[i].item()))
            item["teacher_true_prob"].append(float(teacher_true_prob[i].item()))
            item["student_true_prob"].append(float(student_true_prob[i].item()))

    teacher_probs = torch.cat(all_teacher_probs, dim=0)
    student_probs = torch.cat(all_student_probs, dim=0)
    teacher_pred = torch.cat(all_teacher_pred, dim=0)
    student_pred = torch.cat(all_student_pred, dim=0)
    teacher_weak, teacher_count, teacher_entropy, teacher_score = rfc_weak_mask(
        teacher_pred, teacher_probs, num_classes, weak_fraction
    )
    student_weak, student_count, student_entropy, student_score = rfc_weak_mask(
        student_pred, student_probs, num_classes, weak_fraction
    )
    return class_stats, {
        "teacher_weak": teacher_weak,
        "student_weak": student_weak,
        "teacher_count": teacher_count,
        "student_count": student_count,
        "teacher_entropy": teacher_entropy,
        "student_entropy": student_entropy,
        "teacher_score": teacher_score,
        "student_score": student_score,
    }


def class_row(source_code, domain_name, class_id, class_name, stats, rfc):
    n = stats["n"]
    denom = max(n, 1)
    row = {
        "source": source_code,
        "target": domain_name,
        "class_id": class_id,
        "class_name": class_name,
        "n": n,
        "teacher_acc": stats["teacher_correct"] / denom,
        "student_acc": stats["student_correct"] / denom,
        "student_minus_teacher_acc": (stats["student_correct"] - stats["teacher_correct"]) / denom,
        "both_low_rate": stats["both_low"] / denom,
        "clean_candidate_rate": stats["clean_candidate"] / denom,
        "clean_help_rate": stats["clean_help"] / denom,
        "clean_harm_rate": stats["clean_harm"] / denom,
        "teacher_low_student_high_rate": stats["teacher_low_student_high"] / denom,
        "teacher_high_student_high_rate": stats["teacher_high_student_high"] / denom,
        "rfc_teacher_weak": int(bool(rfc["teacher_weak"][class_id].item())),
        "rfc_student_weak": int(bool(rfc["student_weak"][class_id].item())),
        "teacher_pred_count_ema": float(rfc["teacher_count"][class_id].item()),
        "student_pred_count_ema": float(rfc["student_count"][class_id].item()),
        "teacher_entropy": float(rfc["teacher_entropy"][class_id].item()),
        "student_entropy": float(rfc["student_entropy"][class_id].item()),
        "teacher_rfc_score": float(rfc["teacher_score"][class_id].item()),
        "student_rfc_score": float(rfc["student_score"][class_id].item()),
    }
    for prefix in ["teacher_conf", "student_conf", "teacher_true_prob", "student_true_prob"]:
        values = sorted(stats[prefix])
        row[f"{prefix}_mean"] = mean(values)
        row[f"{prefix}_median"] = quantile(values, 0.5)
        row[f"{prefix}_bottom3_mean"] = bottom_mean(values, 3)
    return row


def subset_summary(source_code, domain_name, subset_name, rows):
    out = {
        "source": source_code,
        "target": domain_name,
        "subset": subset_name,
        "num_classes": len(rows),
        "num_samples": sum(int(r["n"]) for r in rows),
    }
    numeric_keys = [
        "teacher_acc",
        "student_acc",
        "both_low_rate",
        "clean_candidate_rate",
        "clean_help_rate",
        "clean_harm_rate",
        "teacher_conf_mean",
        "student_conf_mean",
        "teacher_true_prob_mean",
        "student_true_prob_mean",
        "teacher_true_prob_median",
        "student_true_prob_median",
        "teacher_true_prob_bottom3_mean",
        "student_true_prob_bottom3_mean",
        "teacher_entropy",
        "student_entropy",
    ]
    for key in numeric_keys:
        values = sorted(float(r[key]) for r in rows)
        out[f"{key}_class_mean"] = mean(values)
        out[f"{key}_class_median"] = quantile(values, 0.5)
        out[f"{key}_class_bottom3_mean"] = bottom_mean(values, 3)
    return out


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.seed >= 0:
        set_random_seed(args.seed)

    use_cuda = torch.cuda.is_available() and args.device != "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    if use_cuda:
        torch.cuda.set_device(0)

    analysis_dir = Path(args.analysis_dir)
    all_class_rows = []
    all_subset_rows = []

    for source_code in args.sources:
        target_domains = [CODE_TO_DOMAIN[code] for code in target_codes_for_source(source_code)]
        cfg = setup_cfg(args, source_code, target_domains)
        trainer = build_trainer(cfg)
        model_dir = model_dir_for(args, source_code)
        trainer.load_model(str(model_dir), epoch=args.load_epoch)

        classnames = list(trainer.dm.dataset.classnames)
        num_classes = len(classnames)
        for domain_name, loader in trainer.test_loaders_by_domain.items():
            class_stats, rfc = analyze_domain(
                trainer,
                domain_name,
                loader,
                num_classes,
                args.conf_threshold,
                args.weak_fraction,
            )
            rows = [
                class_row(source_code, domain_name, i, classnames[i], class_stats[i], rfc)
                for i in range(num_classes)
            ]
            all_class_rows.extend(rows)

            topk = max(1, int(round(args.weak_fraction * num_classes)))
            teacher_weak_rows = [r for r in rows if int(r["rfc_teacher_weak"]) == 1]
            student_weak_rows = [r for r in rows if int(r["rfc_student_weak"]) == 1]
            low_low_rows = sorted(rows, key=lambda r: r["both_low_rate"], reverse=True)[:topk]
            for subset_name, subset_rows in [
                ("all", rows),
                ("rfc_teacher_weak", teacher_weak_rows),
                ("rfc_student_weak", student_weak_rows),
                ("both_low_top_fraction", low_low_rows),
            ]:
                all_subset_rows.append(subset_summary(source_code, domain_name, subset_name, subset_rows))

            task = task_name(source_code)
            write_csv(analysis_dir / f"{task}_{domain_name}_classwise.csv", rows)
            write_csv(
                analysis_dir / f"{task}_{domain_name}_rfc_teacher_weak.csv",
                sorted(
                    teacher_weak_rows,
                    key=lambda r: (-r["both_low_rate"], r["teacher_true_prob_mean"]),
                ),
            )

        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_csv(analysis_dir / "officehome_pl_blindspots_classwise.csv", all_class_rows)
    write_csv(analysis_dir / "officehome_pl_blindspots_subset_summary.csv", all_subset_rows)

    meta = {
        "method_tag": args.method_tag,
        "seed": args.seed,
        "load_epoch": args.load_epoch,
        "conf_threshold": args.conf_threshold,
        "weak_fraction": args.weak_fraction,
        "sources": args.sources,
    }
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "officehome_pl_blindspots_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote analysis to {analysis_dir}")


if __name__ == "__main__":
    main()
