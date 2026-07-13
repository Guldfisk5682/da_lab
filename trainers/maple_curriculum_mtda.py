"""Sequential-target MTDA with class-balanced reliable replay.

This trainer is deliberately separate from the joint-target baseline. It keeps
the same number of optimizer steps and target images per step, while changing
only target-domain scheduling. Optional replay is a separately weighted CE
term over frozen pseudo labels selected at each stage boundary.
"""

import datetime
import json
import os
import time
from collections import OrderedDict, defaultdict

import torch
from torch.cuda.amp import autocast
from torch.nn import functional as F

from dassl.data.data_manager import build_data_loader
from dassl.data.datasets import Datum
from dassl.data.transforms import build_transform
from dassl.engine import TRAINER_REGISTRY
from dassl.optim import build_lr_scheduler, build_optimizer
from dassl.utils import AverageMeter, MetricMeter

from trainers.maple_continuous_mtda import (
    ContinuousSharedProjMaPLeMTDA,
    CustomContinuousSharedProjMaPLeMTDA,
)


def select_topk_replay_records(
    records,
    *,
    topk_per_class,
    student_threshold,
    clip_threshold,
):
    """Select deterministic top-k records per predicted class.

    A record is eligible only when student and frozen CLIP agree and both meet
    their confidence thresholds. Ground-truth labels are never consulted.
    """
    grouped = defaultdict(list)
    eligible = 0
    for record in records:
        if record["student_label"] != record["clip_label"]:
            continue
        if record["student_conf"] < student_threshold:
            continue
        if record["clip_conf"] < clip_threshold:
            continue
        eligible += 1
        grouped[int(record["student_label"])].append(record)

    selected = []
    for predicted_class in sorted(grouped):
        candidates = sorted(
            grouped[predicted_class],
            key=lambda item: (
                -float(item["student_conf"]),
                -float(item["clip_conf"]),
                str(item["impath"]),
            ),
        )
        selected.extend(candidates[:topk_per_class])

    selected.sort(key=lambda item: (int(item["student_label"]), str(item["impath"])))
    return selected, eligible


def stage_local_schedule_index(local_step, stage_length, virtual_epochs):
    """Map a stage-local step to an equal-width virtual scheduler epoch."""
    if stage_length <= 0:
        raise ValueError("stage_length must be positive")
    if virtual_epochs <= 0:
        raise ValueError("virtual_epochs must be positive")
    if local_step < 0 or local_step >= stage_length:
        raise ValueError(
            f"local_step must be in [0, {stage_length}), got {local_step}"
        )
    return min(virtual_epochs - 1, local_step * virtual_epochs // stage_length)


class CustomCurriculumContinuousSharedProjMaPLeMTDA(
    CustomContinuousSharedProjMaPLeMTDA
):
    log_prefix = "CurriculumContinuousSharedProjMaPLeMTDA"

    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        replay_cfg = cfg.TRAINER.MAPLE_MTDA.CURRICULUM.REPLAY
        self.lambda_replay = float(replay_cfg.LAMBDA)
        print(f"{self.log_prefix} replay weight: {self.lambda_replay}")

    def forward_train(
        self,
        image_s,
        label_s,
        image_u_dict,
        replay_image=None,
        replay_label=None,
    ):
        outputs = super().forward_train(image_s, label_s, image_u_dict)
        loss_replay = outputs["loss"].new_zeros(())
        replay_accuracy = outputs["loss"].new_zeros(())
        if replay_image is not None:
            if replay_label is None:
                raise ValueError("replay_label is required when replay_image is provided")
            replay_logits = self(replay_image)
            loss_replay = F.cross_entropy(replay_logits, replay_label)
            replay_accuracy = (
                replay_logits.detach().argmax(dim=-1).eq(replay_label).float().mean()
            )
            self._ensure_finite("curriculum_replay_ce", loss_replay)

        weighted_replay = self.lambda_replay * loss_replay
        outputs["loss"] = outputs["loss"] + weighted_replay
        self._ensure_finite("curriculum_loss_total", outputs["loss"])
        outputs["loss_replay"] = loss_replay.detach()
        outputs["weighted_loss_replay"] = weighted_replay.detach()
        outputs["replay_accuracy"] = replay_accuracy.detach()
        outputs["replay_active"] = outputs["loss"].new_tensor(
            float(replay_image is not None)
        )
        return outputs


@TRAINER_REGISTRY.register()
class CurriculumContinuousSharedProjMaPLeMTDA(ContinuousSharedProjMaPLeMTDA):
    """Static easy/hard curriculum with optional frozen Top-K replay."""

    model_name = "CurriculumContinuousSharedProjMaPLeMTDA"
    custom_model_cls = CustomCurriculumContinuousSharedProjMaPLeMTDA

    def check_cfg(self, cfg):
        super().check_cfg(cfg)
        curriculum_cfg = cfg.TRAINER.MAPLE_MTDA.CURRICULUM
        replay_cfg = curriculum_cfg.REPLAY
        assert int(curriculum_cfg.MICROBATCHES_PER_STEP) > 0
        assert int(curriculum_cfg.STAGE_VIRTUAL_EPOCHS) > 0
        assert int(replay_cfg.TOPK_PER_CLASS) > 0
        assert 0.0 <= float(replay_cfg.STUDENT_THRESHOLD) <= 1.0
        assert 0.0 <= float(replay_cfg.CLIP_THRESHOLD) <= 1.0
        assert float(replay_cfg.LAMBDA) >= 0.0

    def build_data_loader(self):
        super().build_data_loader()
        curriculum_cfg = self.cfg.TRAINER.MAPLE_MTDA.CURRICULUM
        order = [str(domain) for domain in curriculum_cfg.DOMAIN_ORDER]
        available = list(self.train_loader_u.keys())
        if len(order) != len(available) or set(order) != set(available):
            raise ValueError(
                "CURRICULUM.DOMAIN_ORDER must contain every target domain exactly "
                f"once; expected {available}, got {order}"
            )
        self.curriculum_order = order
        self.microbatches_per_step = int(curriculum_cfg.MICROBATCHES_PER_STEP)
        self.reset_optim_per_stage = bool(curriculum_cfg.RESET_OPTIM_PER_STAGE)
        self.stage_virtual_epochs = int(curriculum_cfg.STAGE_VIRTUAL_EPOCHS)
        self.replay_cfg = curriculum_cfg.REPLAY
        self.replay_records = []
        self.replay_loader = None
        self.replay_iterator = None
        self._active_stage = None
        self._stage_replay_batches_seen = 0
        self._stage_replay_images_seen = 0
        self._stage_optimizer_steps = 0
        self._stage_target_batches_seen = 0
        self._stage_target_images_seen = 0
        self._stage_weighted_replay_sum = 0.0
        self._stage_scheduler_index = 0
        self._tfm_train = build_transform(self.cfg, is_train=True)
        self._tfm_test = build_transform(self.cfg, is_train=False)
        print(f"Curriculum target order: {self.curriculum_order}")
        print(f"Target micro-batches per optimizer step: {self.microbatches_per_step}")
        print(f"Replay enabled: {bool(self.replay_cfg.ENABLED)}")
        print(f"Reset optimizer/scheduler per stage: {self.reset_optim_per_stage}")
        if self.reset_optim_per_stage:
            print(f"Stage-local virtual scheduler epochs: {self.stage_virtual_epochs}")

    def _compute_num_batches(self):
        len_x = len(self.train_loader_x)
        len_u = [len(loader) for loader in self.train_loader_u.values()]
        if self.cfg.TRAIN.COUNT_ITER == "train_x":
            return len_x
        if self.cfg.TRAIN.COUNT_ITER == "train_u":
            return min(len_u)
        if self.cfg.TRAIN.COUNT_ITER == "smaller_one":
            return min([len_x, *len_u])
        raise ValueError(f"Unsupported TRAIN.COUNT_ITER={self.cfg.TRAIN.COUNT_ITER}")

    def _stage_for_step(self, global_step):
        total_steps = self.max_epoch * self.num_batches
        return min(
            len(self.curriculum_order) - 1,
            global_step * len(self.curriculum_order) // max(total_steps, 1),
        )

    def _stage_bounds(self, stage):
        total_steps = self.max_epoch * self.num_batches
        num_stages = len(self.curriculum_order)
        start = (stage * total_steps + num_stages - 1) // num_stages
        end = ((stage + 1) * total_steps + num_stages - 1) // num_stages
        return start, end

    def _reset_stage_optimizer_scheduler(self, stage):
        self.optim = build_optimizer(self.model, self.cfg.OPTIM)
        stage_optim_cfg = self.cfg.OPTIM.clone()
        stage_optim_cfg.defrost()
        stage_optim_cfg.MAX_EPOCH = self.stage_virtual_epochs
        stage_optim_cfg.freeze()
        self.sched = build_lr_scheduler(self.optim, stage_optim_cfg)
        self._optims[self.model_name] = self.optim
        self._scheds[self.model_name] = self.sched
        self._stage_scheduler_index = 0
        start, end = self._stage_bounds(stage)
        print(
            "Reset optimizer and scheduler for curriculum stage "
            f"{stage + 1}: steps={end - start}, initial_lr={self.get_current_lr():.4e}"
        )

    def _update_stage_local_scheduler(self, stage):
        if not self.reset_optim_per_stage:
            return
        start, end = self._stage_bounds(stage)
        stage_length = end - start
        if self._stage_optimizer_steps >= stage_length:
            return
        target_index = stage_local_schedule_index(
            self._stage_optimizer_steps,
            stage_length,
            self.stage_virtual_epochs,
        )
        while self._stage_scheduler_index < target_index:
            self.sched.step()
            self._stage_scheduler_index += 1
            print(
                "Stage-local scheduler step: "
                f"stage={stage + 1}, virtual_epoch={self._stage_scheduler_index + 1}/"
                f"{self.stage_virtual_epochs}, lr={self.get_current_lr():.4e}"
            )

    @staticmethod
    def _next_cycled(iterator, loader):
        try:
            return next(iterator), iterator
        except StopIteration:
            iterator = iter(loader)
            return next(iterator), iterator

    @torch.no_grad()
    def _score_domain_for_replay(self, domain_name):
        was_training = self.model.training
        self.set_model_mode("eval")
        records = []
        data_source = self.dm.dataset.train_u_by_domain[domain_name]
        loader = self.test_loaders_by_domain[domain_name]
        for batch in loader:
            image = batch["img"].to(self.device)
            student_probs = F.softmax(self.model(image).float(), dim=-1)
            clip_probs = F.softmax(
                self.model._compute_reference_logits(image).float(), dim=-1
            )
            student_conf, student_label = student_probs.max(dim=-1)
            clip_conf, clip_label = clip_probs.max(dim=-1)
            for offset, index in enumerate(batch["index"].tolist()):
                item = data_source[index]
                records.append(
                    {
                        "domain": domain_name,
                        "domain_id": int(item.domain),
                        "impath": item.impath,
                        "student_label": int(student_label[offset].item()),
                        "student_conf": float(student_conf[offset].item()),
                        "clip_label": int(clip_label[offset].item()),
                        "clip_conf": float(clip_conf[offset].item()),
                    }
                )
        if was_training:
            self.set_model_mode("train")
        return records

    def _write_jsonl(self, filename, payload):
        path = os.path.join(self.output_dir, filename)
        os.makedirs(self.output_dir, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def _write_stage_audit(self, stage, domain_name, global_step):
        audit = {
            "stage": stage,
            "domain": domain_name,
            "boundary_global_step": global_step,
            "optimizer_steps": self._stage_optimizer_steps,
            "target_batches": self._stage_target_batches_seen,
            "target_images": self._stage_target_images_seen,
            "replay_batches": self._stage_replay_batches_seen,
            "replay_images": self._stage_replay_images_seen,
            "mean_weighted_replay_loss": (
                self._stage_weighted_replay_sum / self._stage_optimizer_steps
                if self._stage_optimizer_steps else 0.0
            ),
            "reset_optim_per_stage": self.reset_optim_per_stage,
            "stage_virtual_epochs": self.stage_virtual_epochs,
        }
        self._write_jsonl("curriculum_stage_audit.jsonl", audit)
        print("Curriculum stage audit: " + json.dumps(audit, sort_keys=True))

    @torch.no_grad()
    def _measure_bank_stability(self):
        if not self.replay_records:
            return None
        items = [
            Datum(
                impath=record["impath"],
                label=int(record["student_label"]),
                domain=int(record["domain_id"]),
                classname=self.lab2cname[int(record["student_label"])],
            )
            for record in self.replay_records
        ]
        loader = build_data_loader(
            self.cfg,
            sampler_type="SequentialSampler",
            data_source=items,
            batch_size=self.cfg.DATALOADER.TEST.BATCH_SIZE,
            tfm=self._tfm_test,
            is_train=False,
        )
        was_training = self.model.training
        self.set_model_mode("eval")
        matched = 0
        total = 0
        for batch in loader:
            image = batch["img"].to(self.device)
            frozen_label = batch["label"].to(self.device)
            prediction = self.model(image).argmax(dim=-1)
            matched += int(prediction.eq(frozen_label).sum().item())
            total += int(frozen_label.numel())
        if was_training:
            self.set_model_mode("train")
        return matched / total if total else None

    def _write_bank_audit(
        self, stage, domain_name, selected, eligible, total, prior_bank_stability
    ):
        class_counts = defaultdict(int)
        for record in selected:
            class_counts[str(record["student_label"])] += 1
        audit = {
            "stage": stage,
            "fitted_domain": domain_name,
            "total_scored": total,
            "eligible": eligible,
            "selected": len(selected),
            "class_coverage": len(class_counts),
            "num_classes": self.num_classes,
            "selected_per_class": dict(sorted(class_counts.items(), key=lambda x: int(x[0]))),
            "mean_student_conf": (
                sum(record["student_conf"] for record in selected) / len(selected)
                if selected else 0.0
            ),
            "mean_clip_conf": (
                sum(record["clip_conf"] for record in selected) / len(selected)
                if selected else 0.0
            ),
            "cumulative_bank_size": len(self.replay_records),
            "prior_bank_label_stability": prior_bank_stability,
            "topk_per_class": int(self.replay_cfg.TOPK_PER_CLASS),
            "student_threshold": float(self.replay_cfg.STUDENT_THRESHOLD),
            "clip_threshold": float(self.replay_cfg.CLIP_THRESHOLD),
        }
        self._write_jsonl("replay_bank_audit.jsonl", audit)
        print("Replay bank audit: " + json.dumps(audit, sort_keys=True))

    def _build_replay_loader(self):
        if not self.replay_records:
            self.replay_loader = None
            self.replay_iterator = None
            return
        replay_items = [
            Datum(
                impath=record["impath"],
                label=int(record["student_label"]),
                domain=int(record["domain_id"]),
                classname=self.lab2cname[int(record["student_label"])],
            )
            for record in self.replay_records
        ]
        self.replay_loader = build_data_loader(
            self.cfg,
            sampler_type="RandomSampler",
            data_source=replay_items,
            batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            tfm=self._tfm_train,
            is_train=True,
        )
        self.replay_iterator = iter(self.replay_loader)

    def _enter_stage(self, stage, global_step):
        if self._active_stage == stage:
            return
        if self._active_stage is not None:
            fitted_domain = self.curriculum_order[self._active_stage]
            self._write_stage_audit(self._active_stage, fitted_domain, global_step)
            prior_bank_stability = self._measure_bank_stability()
            print(f"Scoring fitted domain for frozen replay: {fitted_domain}")
            scored = self._score_domain_for_replay(fitted_domain)
            selected, eligible = select_topk_replay_records(
                scored,
                topk_per_class=int(self.replay_cfg.TOPK_PER_CLASS),
                student_threshold=float(self.replay_cfg.STUDENT_THRESHOLD),
                clip_threshold=float(self.replay_cfg.CLIP_THRESHOLD),
            )
            if bool(self.replay_cfg.ENABLED):
                self.replay_records.extend(selected)
            self._write_bank_audit(
                self._active_stage,
                fitted_domain,
                selected,
                eligible,
                len(scored),
                prior_bank_stability,
            )
            print(f"Evaluate all targets at the end of stage {self._active_stage + 1}")
            self.test()
            self.set_model_mode("train")
        self._active_stage = stage
        self._stage_replay_batches_seen = 0
        self._stage_replay_images_seen = 0
        self._stage_optimizer_steps = 0
        self._stage_target_batches_seen = 0
        self._stage_target_images_seen = 0
        self._stage_weighted_replay_sum = 0.0
        if self.reset_optim_per_stage:
            self._reset_stage_optimizer_scheduler(stage)
        self._build_replay_loader() if bool(self.replay_cfg.ENABLED) else None
        print(
            f"Entering curriculum stage {stage + 1}/{len(self.curriculum_order)}: "
            f"target={self.curriculum_order[stage]}, replay_items={len(self.replay_records)}"
        )

    def _next_replay_batch(self):
        if self.replay_iterator is None:
            return None
        try:
            batch = next(self.replay_iterator)
            self._stage_replay_batches_seen += 1
            self._stage_replay_images_seen += int(batch["label"].numel())
            return batch
        except StopIteration:
            self.replay_iterator = None
            print(
                "Replay traversal exhausted for current stage after "
                f"{self._stage_replay_batches_seen} batches; replay is now disabled "
                "until the next stage."
            )
            return None

    def forward_backward(self, batch_x, batch_u, replay_batch=None):
        image_x, label_x, image_u = self.parse_batch_train(batch_x, batch_u)
        replay_image = None
        replay_label = None
        if replay_batch is not None:
            replay_image = replay_batch["img"].to(self.device)
            replay_label = replay_batch["label"].to(self.device)

        denom = max(self.max_epoch * self.num_batches - 1, 1)
        progress = (self.epoch * self.num_batches + self.batch_idx) / denom
        if hasattr(self.model, "set_training_progress"):
            self.model.set_training_progress(progress)

        prec = self.cfg.TRAINER.MAPLE_MTDA.PREC
        if prec == "amp":
            with autocast():
                outputs = self.model.forward_train(
                    image_x,
                    label_x,
                    image_u,
                    replay_image=replay_image,
                    replay_label=replay_label,
                )
                loss = outputs["loss"]
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            outputs = self.model.forward_train(
                image_x,
                label_x,
                image_u,
                replay_image=replay_image,
                replay_label=replay_label,
            )
            loss = outputs["loss"]
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        summary = {"loss": float(loss.item())}
        for key, value in outputs.items():
            if key != "loss":
                summary[key] = value.item() if torch.is_tensor(value) else float(value)
        if not self.reset_optim_per_stage and (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        return summary

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = self._compute_num_batches()

        source_iter = iter(self.train_loader_x)
        target_iters = {
            domain: iter(loader) for domain, loader in self.train_loader_u.items()
        }
        end = time.time()
        for self.batch_idx in range(self.num_batches):
            global_step = self.epoch * self.num_batches + self.batch_idx
            stage = self._stage_for_step(global_step)
            self._enter_stage(stage, global_step)
            active_domain = self.curriculum_order[stage]

            batch_x, source_iter = self._next_cycled(
                source_iter, self.train_loader_x
            )
            batch_u = OrderedDict()
            for microbatch_index in range(self.microbatches_per_step):
                batch, target_iters[active_domain] = self._next_cycled(
                    target_iters[active_domain], self.train_loader_u[active_domain]
                )
                batch_u[f"{active_domain}__micro{microbatch_index}"] = batch
            replay_batch = self._next_replay_batch()

            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch_x, batch_u, replay_batch)
            self._stage_optimizer_steps += 1
            self._stage_target_batches_seen += self.microbatches_per_step
            self._stage_target_images_seen += sum(
                int(batch["label"].numel()) for batch in batch_u.values()
            )
            self._stage_weighted_replay_sum += loss_summary[
                "weighted_loss_replay"
            ]
            self._update_stage_local_scheduler(stage)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            if meet_freq or self.num_batches < self.cfg.TRAIN.PRINT_FREQ:
                remaining = self.num_batches - self.batch_idx - 1
                remaining += (self.max_epoch - self.epoch - 1) * self.num_batches
                eta = str(datetime.timedelta(seconds=int(batch_time.avg * remaining)))
                info = [
                    f"epoch [{self.epoch + 1}/{self.max_epoch}]",
                    f"batch [{self.batch_idx + 1}/{self.num_batches}]",
                    f"stage [{stage + 1}/{len(self.curriculum_order)}:{active_domain}]",
                    f"time {batch_time.val:.3f} ({batch_time.avg:.3f})",
                    f"data {data_time.val:.3f} ({data_time.avg:.3f})",
                    f"{losses}",
                    f"lr {self.get_current_lr():.4e}",
                    f"eta {eta}",
                ]
                print(" ".join(info))

            n_iter = global_step
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)
            self.write_scalar("train/curriculum_stage", stage, n_iter)
            end = time.time()

    def after_train(self):
        if self._active_stage is not None:
            final_domain = self.curriculum_order[self._active_stage]
            total_steps = self.max_epoch * self.num_batches
            self._write_stage_audit(self._active_stage, final_domain, total_steps)
            prior_bank_stability = self._measure_bank_stability()
            print(f"Scoring final fitted domain for audit only: {final_domain}")
            scored = self._score_domain_for_replay(final_domain)
            selected, eligible = select_topk_replay_records(
                scored,
                topk_per_class=int(self.replay_cfg.TOPK_PER_CLASS),
                student_threshold=float(self.replay_cfg.STUDENT_THRESHOLD),
                clip_threshold=float(self.replay_cfg.CLIP_THRESHOLD),
            )
            self._write_bank_audit(
                self._active_stage,
                final_domain,
                selected,
                eligible,
                len(scored),
                prior_bank_stability,
            )
            self._active_stage = None
        super().after_train()
