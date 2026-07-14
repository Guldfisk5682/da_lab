"""Sequential-target MTDA with class-balanced reliable replay.

This trainer is deliberately separate from the joint-target baseline. It keeps
the same number of optimizer steps and target images per step, while changing
only target-domain scheduling. Optional replay is a separately weighted CE
term over frozen pseudo labels selected at each stage boundary.
"""

import datetime
import json
import os
import random
import time
from collections import Counter, OrderedDict, defaultdict

import numpy as np
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
    prefer_correct=False,
):
    """Select deterministic top-k records per predicted class.

    A record is eligible only when student and frozen CLIP agree and both meet
    their confidence thresholds. Ground-truth labels are consulted only by the
    explicitly diagnostic ``prefer_correct`` oracle mode.
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
        if prefer_correct:
            for item in grouped[predicted_class]:
                if "true_label" not in item:
                    raise ValueError(
                        "oracle-correct replay selection requires true_label"
                    )
        candidates = sorted(
            grouped[predicted_class],
            key=lambda item: (
                int(
                    prefer_correct
                    and int(item["student_label"]) != int(item["true_label"])
                ),
                -float(item["student_conf"]),
                -float(item["clip_conf"]),
                str(item["impath"]),
            ),
        )
        selected.extend(candidates[:topk_per_class])

    selected.sort(key=lambda item: (int(item["student_label"]), str(item["impath"])))
    return selected, eligible


def load_replay_manifest(path):
    """Load stage-indexed replay selections written by this trainer."""
    manifests = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            key = (int(payload["stage"]), str(payload["fitted_domain"]))
            if key in manifests:
                raise ValueError(
                    f"Duplicate replay manifest entry {key} at line {line_number}"
                )
            manifests[key] = payload
    if not manifests:
        raise ValueError(f"Replay manifest is empty: {path}")
    return manifests


def materialize_manifest_records(payload, current_records, label_source):
    """Join a frozen selection manifest with current dataset metadata."""
    current_by_path = {str(record["impath"]): record for record in current_records}
    selected = []
    missing = []
    frozen_paths = [str(record["impath"]) for record in payload.get("records", [])]
    if len(frozen_paths) != len(set(frozen_paths)):
        raise ValueError("Replay manifest contains duplicate sample paths")
    for frozen in payload.get("records", []):
        impath = str(frozen["impath"])
        current = current_by_path.get(impath)
        if current is None:
            missing.append(impath)
            continue
        record = dict(current)
        record["selection_origin"] = "manifest"
        record["pseudo_label"] = int(
            frozen.get("pseudo_label", frozen["student_label"])
        )
        record["selection_student_label"] = int(
            frozen.get("selection_student_label", frozen["student_label"])
        )
        record["selection_clip_label"] = int(
            frozen.get("selection_clip_label", frozen.get("clip_label", record["pseudo_label"]))
        )
        record["selection_student_conf"] = float(
            frozen.get("student_conf", current["student_conf"])
        )
        record["selection_clip_conf"] = float(
            frozen.get("clip_conf", current["clip_conf"])
        )
        record["replay_label"] = (
            int(record["true_label"])
            if label_source == "ground_truth"
            else int(record["pseudo_label"])
        )
        selected.append(record)
    if missing:
        preview = ", ".join(missing[:3])
        raise ValueError(
            f"Replay manifest contains {len(missing)} paths absent from the "
            f"current dataset, e.g. {preview}"
        )
    return selected


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


def replay_step_budget_scale(normalization, one_pass_steps, stage_steps):
    """Return a stage-fixed scale matching one-pass replay update steps."""
    if stage_steps <= 0:
        raise ValueError("stage_steps must be positive")
    if one_pass_steps < 0:
        raise ValueError("one_pass_steps must be non-negative")
    if normalization == "none":
        return 1.0
    if normalization != "one_pass_steps":
        raise ValueError(f"Unknown replay normalization: {normalization}")
    return min(one_pass_steps, stage_steps) / stage_steps


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
        replay_loss_scale=1.0,
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

        unnormalized_weighted_replay = self.lambda_replay * loss_replay
        weighted_replay = float(replay_loss_scale) * unnormalized_weighted_replay
        outputs["loss"] = outputs["loss"] + weighted_replay
        self._ensure_finite("curriculum_loss_total", outputs["loss"])
        outputs["loss_replay"] = loss_replay.detach()
        outputs["unnormalized_weighted_loss_replay"] = (
            unnormalized_weighted_replay.detach()
        )
        outputs["weighted_loss_replay"] = weighted_replay.detach()
        outputs["replay_loss_scale"] = outputs["loss"].new_tensor(
            float(replay_loss_scale)
        )
        outputs["effective_replay_lambda"] = outputs["loss"].new_tensor(
            self.lambda_replay * float(replay_loss_scale)
        )
        # Keep the differentiable replay objective private to the trainer. It is
        # removed before scalar logging and is used only by diagnostics to
        # measure the replay branch's own gradient contribution.
        outputs["_weighted_replay_objective"] = weighted_replay
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
        selection_mode = str(replay_cfg.SELECTION_MODE).lower()
        label_source = str(replay_cfg.LABEL_SOURCE).lower()
        traversal = str(replay_cfg.TRAVERSAL).lower()
        normalization = str(replay_cfg.NORMALIZATION).lower()
        assert selection_mode in {"online", "oracle_correct", "manifest"}
        assert label_source in {"pseudo", "ground_truth"}
        assert traversal in {"one_pass", "cycle"}
        assert normalization in {"none", "one_pass_steps"}
        if normalization == "one_pass_steps":
            assert traversal == "cycle", (
                "one_pass_steps replay normalization requires cycle traversal"
            )
        if selection_mode == "manifest":
            assert str(replay_cfg.MANIFEST_PATH).strip()
        if selection_mode == "oracle_correct" or label_source == "ground_truth":
            assert bool(curriculum_cfg.DIAGNOSTICS.ENABLED), (
                "Target-label oracle modes require CURRICULUM.DIAGNOSTICS.ENABLED"
            )

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
        self.diagnostics_cfg = curriculum_cfg.DIAGNOSTICS
        self.replay_selection_mode = str(self.replay_cfg.SELECTION_MODE).lower()
        self.replay_label_source = str(self.replay_cfg.LABEL_SOURCE).lower()
        self.replay_traversal = str(self.replay_cfg.TRAVERSAL).lower()
        self.replay_normalization = str(self.replay_cfg.NORMALIZATION).lower()
        self.diagnostics_enabled = bool(self.diagnostics_cfg.ENABLED)
        self.audit_all_domains = bool(self.diagnostics_cfg.AUDIT_ALL_DOMAINS)
        self.frozen_replay_manifests = None
        if self.replay_selection_mode == "manifest":
            manifest_path = os.path.expanduser(str(self.replay_cfg.MANIFEST_PATH))
            self.frozen_replay_manifests = load_replay_manifest(manifest_path)
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
        self._stage_unnormalized_weighted_replay_sum = 0.0
        self._stage_raw_replay_sum = 0.0
        self._stage_replay_path_exposures = Counter()
        self._stage_replay_grad_steps = 0
        self._stage_replay_grad_norm_sum = 0.0
        self._stage_replay_grad_norm_sq_sum = 0.0
        self._stage_lr_weighted_replay_grad_norm_sum = 0.0
        self._stage_replay_grad_vector_sum = None
        self._stage_one_pass_replay_steps = 0
        self._stage_expected_optimizer_steps = 0
        self._stage_replay_loss_scale = 0.0
        self._stage_scheduler_index = 0
        self._tfm_train = build_transform(self.cfg, is_train=True)
        self._tfm_test = build_transform(self.cfg, is_train=False)
        print(f"Curriculum target order: {self.curriculum_order}")
        print(f"Target micro-batches per optimizer step: {self.microbatches_per_step}")
        print(f"Replay enabled: {bool(self.replay_cfg.ENABLED)}")
        print(f"Replay selection mode: {self.replay_selection_mode}")
        print(f"Replay label source: {self.replay_label_source}")
        print(f"Replay traversal: {self.replay_traversal}")
        print(f"Replay normalization: {self.replay_normalization}")
        print(f"Full prediction diagnostics: {self.diagnostics_enabled}")
        if (
            self.replay_selection_mode == "oracle_correct"
            or self.replay_label_source == "ground_truth"
        ):
            print(
                "WARNING: target ground truth is active for diagnosis only; "
                "this run is not a valid UDA result"
            )
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
                        "dataset_index": int(index),
                        "impath": item.impath,
                        "true_label": int(item.label),
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

    def _write_jsonl_many(self, filename, payloads):
        path = os.path.join(self.output_dir, filename)
        os.makedirs(self.output_dir, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            for payload in payloads:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def _eligible_records(self, records):
        return [
            record
            for record in records
            if record["student_label"] == record["clip_label"]
            and record["student_conf"] >= float(self.replay_cfg.STUDENT_THRESHOLD)
            and record["clip_conf"] >= float(self.replay_cfg.CLIP_THRESHOLD)
        ]

    def _prepare_online_selection(self, selected):
        prepared = []
        for original in selected:
            record = dict(original)
            record["selection_origin"] = self.replay_selection_mode
            record["pseudo_label"] = int(record["student_label"])
            record["selection_student_label"] = int(record["student_label"])
            record["selection_clip_label"] = int(record["clip_label"])
            record["selection_student_conf"] = float(record["student_conf"])
            record["selection_clip_conf"] = float(record["clip_conf"])
            record["replay_label"] = (
                int(record["true_label"])
                if self.replay_label_source == "ground_truth"
                else int(record["pseudo_label"])
            )
            prepared.append(record)
        return prepared

    def _select_stage_replay(self, stage, domain_name, scored):
        if self.replay_selection_mode == "manifest":
            key = (int(stage), str(domain_name))
            if key not in self.frozen_replay_manifests:
                raise ValueError(f"Replay manifest is missing stage/domain {key}")
            payload = self.frozen_replay_manifests[key]
            if "seed" in payload and int(payload["seed"]) != int(self.cfg.SEED):
                raise ValueError(
                    f"Replay manifest seed {payload['seed']} does not match "
                    f"run seed {self.cfg.SEED}"
                )
            if "curriculum_order" in payload and list(
                payload["curriculum_order"]
            ) != list(self.curriculum_order):
                raise ValueError("Replay manifest curriculum order does not match run")
            if int(payload.get("topk_per_class", self.replay_cfg.TOPK_PER_CLASS)) != int(
                self.replay_cfg.TOPK_PER_CLASS
            ):
                raise ValueError("Replay manifest Top-K does not match run")
            selected = materialize_manifest_records(
                payload,
                scored,
                self.replay_label_source,
            )
            eligible = len(self._eligible_records(scored))
        else:
            selected, eligible = select_topk_replay_records(
                scored,
                topk_per_class=int(self.replay_cfg.TOPK_PER_CLASS),
                student_threshold=float(self.replay_cfg.STUDENT_THRESHOLD),
                clip_threshold=float(self.replay_cfg.CLIP_THRESHOLD),
                prefer_correct=self.replay_selection_mode == "oracle_correct",
            )
            selected = self._prepare_online_selection(selected)
        self._write_selection_manifest(stage, domain_name, selected)
        return selected, eligible

    def _write_selection_manifest(self, stage, domain_name, selected):
        records = []
        for record in selected:
            records.append(
                {
                    "domain": str(record["domain"]),
                    "domain_id": int(record["domain_id"]),
                    "dataset_index": int(record["dataset_index"]),
                    "impath": str(record["impath"]),
                    "true_label": int(record["true_label"]),
                    "student_label": int(record["student_label"]),
                    "student_conf": float(record["student_conf"]),
                    "clip_label": int(record["clip_label"]),
                    "clip_conf": float(record["clip_conf"]),
                    "pseudo_label": int(record["pseudo_label"]),
                    "selection_student_label": int(
                        record["selection_student_label"]
                    ),
                    "selection_clip_label": int(record["selection_clip_label"]),
                    "replay_label": int(record["replay_label"]),
                    "selection_origin": str(record["selection_origin"]),
                }
            )
        payload = {
            "stage": int(stage),
            "fitted_domain": str(domain_name),
            "seed": int(self.cfg.SEED),
            "curriculum_order": list(self.curriculum_order),
            "selection_mode": self.replay_selection_mode,
            "label_source": self.replay_label_source,
            "topk_per_class": int(self.replay_cfg.TOPK_PER_CLASS),
            "records": records,
        }
        self._write_jsonl("replay_selection_manifest.jsonl", payload)

    def _score_domains_for_boundary(self, fitted_domain):
        scored = {fitted_domain: self._score_domain_for_replay(fitted_domain)}
        if not (self.diagnostics_enabled and self.audit_all_domains):
            return scored

        # Extra diagnostics must not perturb the subsequent training stream.
        # The fitted-domain scoring above is part of the original algorithm;
        # only the newly added all-domain passes are wrapped and restored.
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            for domain in self.curriculum_order:
                if domain != fitted_domain:
                    scored[domain] = self._score_domain_for_replay(domain)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
        return scored

    def _write_prediction_audit(
        self,
        stage,
        fitted_domain,
        global_step,
        scored_by_domain,
        actual_selected,
    ):
        if not self.diagnostics_enabled:
            return
        actual_paths = {str(record["impath"]) for record in actual_selected}
        payloads = []
        for domain_name, records in scored_by_domain.items():
            standard, _ = select_topk_replay_records(
                records,
                topk_per_class=int(self.replay_cfg.TOPK_PER_CLASS),
                student_threshold=float(self.replay_cfg.STUDENT_THRESHOLD),
                clip_threshold=float(self.replay_cfg.CLIP_THRESHOLD),
            )
            oracle, _ = select_topk_replay_records(
                records,
                topk_per_class=int(self.replay_cfg.TOPK_PER_CLASS),
                student_threshold=float(self.replay_cfg.STUDENT_THRESHOLD),
                clip_threshold=float(self.replay_cfg.CLIP_THRESHOLD),
                prefer_correct=True,
            )
            standard_paths = {str(record["impath"]) for record in standard}
            oracle_paths = {str(record["impath"]) for record in oracle}
            eligible_paths = {
                str(record["impath"]) for record in self._eligible_records(records)
            }
            for record in records:
                payload = dict(record)
                impath = str(record["impath"])
                student_correct = int(record["student_label"]) == int(
                    record["true_label"]
                )
                clip_correct = int(record["clip_label"]) == int(
                    record["true_label"]
                )
                agreement = int(record["student_label"]) == int(
                    record["clip_label"]
                )
                clean_pl_selected = float(record["clip_conf"]) >= float(
                    self.model.pl_threshold
                )
                if bool(self.model.pl_use_student_low_conf_mask):
                    clean_pl_selected = clean_pl_selected and float(
                        record["student_conf"]
                    ) < float(self.model.pl_student_threshold)
                payload.update(
                    {
                        "boundary_stage": int(stage),
                        "boundary_global_step": int(global_step),
                        "fitted_domain": str(fitted_domain),
                        "student_correct": bool(student_correct),
                        "clip_correct": bool(clip_correct),
                        "agreement": bool(agreement),
                        "both_wrong_agree": bool(
                            agreement and not student_correct and not clip_correct
                        ),
                        "clean_pl_selected": bool(clean_pl_selected),
                        "eligible": impath in eligible_paths,
                        "standard_topk": impath in standard_paths,
                        "oracle_correct_topk": impath in oracle_paths,
                        "actual_selected": bool(
                            domain_name == fitted_domain and impath in actual_paths
                        ),
                    }
                )
                payloads.append(payload)
        self._write_jsonl_many("pl_sample_audit.jsonl", payloads)

    def _write_stage_audit(self, stage, domain_name, global_step):
        exposure_counts = list(self._stage_replay_path_exposures.values())
        accumulated_gradient_norm = 0.0
        if self._stage_replay_grad_vector_sum:
            squared_norm = sum(
                float(vector.float().pow(2).sum().item())
                for vector in self._stage_replay_grad_vector_sum.values()
            )
            accumulated_gradient_norm = squared_norm**0.5
        audit = {
            "stage": stage,
            "domain": domain_name,
            "boundary_global_step": global_step,
            "optimizer_steps": self._stage_optimizer_steps,
            "target_batches": self._stage_target_batches_seen,
            "target_images": self._stage_target_images_seen,
            "replay_batches": self._stage_replay_batches_seen,
            "replay_images": self._stage_replay_images_seen,
            "replay_sample_exposures": self._stage_replay_images_seen,
            "replay_unique_samples_seen": len(self._stage_replay_path_exposures),
            "replay_bank_unique_samples": len(self.replay_records),
            "replay_exposures_per_seen_sample_min": (
                min(exposure_counts) if exposure_counts else 0
            ),
            "replay_exposures_per_seen_sample_max": (
                max(exposure_counts) if exposure_counts else 0
            ),
            "replay_exposures_per_seen_sample_mean": (
                self._stage_replay_images_seen / len(exposure_counts)
                if exposure_counts
                else 0.0
            ),
            "cumulative_raw_replay_loss": self._stage_raw_replay_sum,
            "cumulative_unnormalized_weighted_replay_loss": (
                self._stage_unnormalized_weighted_replay_sum
            ),
            "cumulative_weighted_replay_loss": self._stage_weighted_replay_sum,
            "replay_normalization": self.replay_normalization,
            "replay_loss_scale": self._stage_replay_loss_scale,
            "effective_replay_lambda": (
                float(self.replay_cfg.LAMBDA) * self._stage_replay_loss_scale
            ),
            "one_pass_reference_optimizer_steps": (
                self._stage_one_pass_replay_steps
            ),
            "expected_stage_optimizer_steps": self._stage_expected_optimizer_steps,
            "nominal_one_pass_replay_weight_budget": (
                float(self.replay_cfg.LAMBDA)
                * self._stage_one_pass_replay_steps
            ),
            "nominal_actual_replay_weight_budget": (
                float(self.replay_cfg.LAMBDA)
                * self._stage_replay_loss_scale
                * self._stage_replay_batches_seen
            ),
            "mean_weighted_replay_loss": (
                self._stage_weighted_replay_sum / self._stage_optimizer_steps
                if self._stage_optimizer_steps else 0.0
            ),
            "mean_weighted_replay_loss_when_active": (
                self._stage_weighted_replay_sum / self._stage_replay_batches_seen
                if self._stage_replay_batches_seen
                else 0.0
            ),
            "replay_gradient_steps": self._stage_replay_grad_steps,
            "mean_weighted_replay_gradient_norm": (
                self._stage_replay_grad_norm_sum / self._stage_replay_grad_steps
                if self._stage_replay_grad_steps
                else 0.0
            ),
            "rms_weighted_replay_gradient_norm": (
                (
                    self._stage_replay_grad_norm_sq_sum
                    / self._stage_replay_grad_steps
                )
                ** 0.5
                if self._stage_replay_grad_steps
                else 0.0
            ),
            "sum_weighted_replay_gradient_norm": self._stage_replay_grad_norm_sum,
            "sum_lr_weighted_replay_gradient_norm": (
                self._stage_lr_weighted_replay_grad_norm_sum
            ),
            "norm_of_summed_weighted_replay_gradients": accumulated_gradient_norm,
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
                label=int(record.get("pseudo_label", record["student_label"])),
                domain=int(record["domain_id"]),
                classname=self.lab2cname[
                    int(record.get("pseudo_label", record["student_label"]))
                ],
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
        pseudo_matched = 0
        replay_matched = 0
        true_matched = 0
        total = 0
        replay_labels = [
            int(record.get("replay_label", record["student_label"]))
            for record in self.replay_records
        ]
        true_labels = [int(record["true_label"]) for record in self.replay_records]
        offset = 0
        for batch in loader:
            image = batch["img"].to(self.device)
            pseudo_label = batch["label"].to(self.device)
            prediction = self.model(image).argmax(dim=-1)
            count = int(pseudo_label.numel())
            replay_label = torch.tensor(
                replay_labels[offset : offset + count], device=self.device
            )
            true_label = torch.tensor(
                true_labels[offset : offset + count], device=self.device
            )
            pseudo_matched += int(prediction.eq(pseudo_label).sum().item())
            replay_matched += int(prediction.eq(replay_label).sum().item())
            true_matched += int(prediction.eq(true_label).sum().item())
            total += count
            offset += count
        if was_training:
            self.set_model_mode("train")
        if not total:
            return None
        return {
            "pseudo_label_stability": pseudo_matched / total,
            "replay_label_agreement": replay_matched / total,
            "true_accuracy": true_matched / total,
            "samples": total,
        }

    def _write_bank_audit(
        self,
        stage,
        domain_name,
        selected,
        eligible,
        total,
        prior_bank_stability,
        scored=None,
    ):
        class_counts = defaultdict(int)
        true_class_counts = defaultdict(int)
        for record in selected:
            class_counts[str(record.get("pseudo_label", record["student_label"]))] += 1
            true_class_counts[str(record["true_label"])] += 1
        eligible_records = self._eligible_records(scored or [])
        eligible_correct = sum(
            int(record["student_label"]) == int(record["true_label"])
            for record in eligible_records
        )
        eligible_both_wrong_agree = sum(
            int(record["student_label"]) == int(record["clip_label"])
            and int(record["student_label"]) != int(record["true_label"])
            for record in eligible_records
        )
        correct_eligible_per_class = defaultdict(int)
        for record in eligible_records:
            if int(record["student_label"]) == int(record["true_label"]):
                correct_eligible_per_class[str(record["student_label"])] += 1
        oracle_shortfall_per_class = {
            str(class_index): max(
                0,
                int(self.replay_cfg.TOPK_PER_CLASS)
                - correct_eligible_per_class.get(str(class_index), 0),
            )
            for class_index in range(self.num_classes)
        }
        selected_correct = sum(
            int(record.get("pseudo_label", record["student_label"]))
            == int(record["true_label"])
            for record in selected
        )
        selected_both_wrong_agree = sum(
            int(record["selection_student_label"])
            == int(record["selection_clip_label"])
            and int(record["selection_student_label"])
            != int(record["true_label"])
            for record in selected
        )
        stability = prior_bank_stability or {}
        audit = {
            "stage": stage,
            "fitted_domain": domain_name,
            "total_scored": total,
            "eligible": eligible,
            "eligible_pseudo_accuracy": (
                eligible_correct / len(eligible_records)
                if eligible_records
                else None
            ),
            "eligible_both_wrong_agree_rate": (
                eligible_both_wrong_agree / len(eligible_records)
                if eligible_records
                else None
            ),
            "correct_eligible_per_class": dict(
                sorted(correct_eligible_per_class.items(), key=lambda x: int(x[0]))
            ),
            "oracle_correct_shortfall_per_class": oracle_shortfall_per_class,
            "oracle_correct_weak_class_count": sum(
                shortfall > 0 for shortfall in oracle_shortfall_per_class.values()
            ),
            "selected": len(selected),
            "class_coverage": len(class_counts),
            "num_classes": self.num_classes,
            "selected_per_class": dict(sorted(class_counts.items(), key=lambda x: int(x[0]))),
            "selected_per_true_class": dict(
                sorted(true_class_counts.items(), key=lambda x: int(x[0]))
            ),
            "true_class_coverage": len(true_class_counts),
            "selected_pseudo_accuracy": (
                selected_correct / len(selected) if selected else None
            ),
            "selected_both_wrong_agree_rate": (
                selected_both_wrong_agree / len(selected) if selected else None
            ),
            "mean_student_conf": (
                sum(record["selection_student_conf"] for record in selected)
                / len(selected)
                if selected else 0.0
            ),
            "mean_clip_conf": (
                sum(record["selection_clip_conf"] for record in selected)
                / len(selected)
                if selected else 0.0
            ),
            "cumulative_bank_size": len(self.replay_records),
            "prior_bank_label_stability": stability.get(
                "pseudo_label_stability"
            ),
            "prior_bank_replay_label_agreement": stability.get(
                "replay_label_agreement"
            ),
            "prior_bank_true_accuracy": stability.get("true_accuracy"),
            "prior_bank_samples": stability.get("samples"),
            "topk_per_class": int(self.replay_cfg.TOPK_PER_CLASS),
            "student_threshold": float(self.replay_cfg.STUDENT_THRESHOLD),
            "clip_threshold": float(self.replay_cfg.CLIP_THRESHOLD),
            "selection_mode": self.replay_selection_mode,
            "label_source": self.replay_label_source,
            "traversal": self.replay_traversal,
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
                label=int(record.get("replay_label", record["student_label"])),
                domain=int(record["domain_id"]),
                classname=self.lab2cname[
                    int(record.get("replay_label", record["student_label"]))
                ],
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

    def _finalize_fitted_stage(
        self,
        stage,
        fitted_domain,
        global_step,
        *,
        add_to_bank,
    ):
        prior_bank_stability = self._measure_bank_stability()
        print(f"Scoring fitted domain for frozen replay: {fitted_domain}")
        scored_by_domain = self._score_domains_for_boundary(fitted_domain)
        scored = scored_by_domain[fitted_domain]
        selected, eligible = self._select_stage_replay(
            stage, fitted_domain, scored
        )
        self._write_prediction_audit(
            stage,
            fitted_domain,
            global_step,
            scored_by_domain,
            selected,
        )
        if add_to_bank and bool(self.replay_cfg.ENABLED):
            self.replay_records.extend(selected)
        self._write_bank_audit(
            stage,
            fitted_domain,
            selected,
            eligible,
            len(scored),
            prior_bank_stability,
            scored=scored,
        )

    def _enter_stage(self, stage, global_step):
        if self._active_stage == stage:
            return
        if self._active_stage is not None:
            fitted_domain = self.curriculum_order[self._active_stage]
            self._write_stage_audit(self._active_stage, fitted_domain, global_step)
            self._finalize_fitted_stage(
                self._active_stage,
                fitted_domain,
                global_step,
                add_to_bank=True,
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
        self._stage_unnormalized_weighted_replay_sum = 0.0
        self._stage_raw_replay_sum = 0.0
        self._stage_replay_path_exposures = Counter()
        self._stage_replay_grad_steps = 0
        self._stage_replay_grad_norm_sum = 0.0
        self._stage_replay_grad_norm_sq_sum = 0.0
        self._stage_lr_weighted_replay_grad_norm_sum = 0.0
        self._stage_replay_grad_vector_sum = None
        if self.reset_optim_per_stage:
            self._reset_stage_optimizer_scheduler(stage)
        self._build_replay_loader() if bool(self.replay_cfg.ENABLED) else None
        stage_start, stage_end = self._stage_bounds(stage)
        self._stage_expected_optimizer_steps = stage_end - stage_start
        self._stage_one_pass_replay_steps = (
            min(len(self.replay_loader), self._stage_expected_optimizer_steps)
            if self.replay_loader is not None
            else 0
        )
        self._stage_replay_loss_scale = (
            replay_step_budget_scale(
                self.replay_normalization,
                self._stage_one_pass_replay_steps,
                self._stage_expected_optimizer_steps,
            )
            if self.replay_loader is not None
            else 0.0
        )
        print(
            f"Entering curriculum stage {stage + 1}/{len(self.curriculum_order)}: "
            f"target={self.curriculum_order[stage]}, replay_items={len(self.replay_records)}, "
            f"one_pass_replay_steps={self._stage_one_pass_replay_steps}, "
            f"stage_steps={self._stage_expected_optimizer_steps}, "
            f"replay_loss_scale={self._stage_replay_loss_scale:.8f}"
        )

    def _next_replay_batch(self):
        if self.replay_iterator is None:
            return None
        try:
            batch = next(self.replay_iterator)
            self._stage_replay_batches_seen += 1
            self._stage_replay_images_seen += int(batch["label"].numel())
            self._record_replay_paths(batch)
            return batch
        except StopIteration:
            if self.replay_traversal == "cycle":
                self.replay_iterator = iter(self.replay_loader)
                batch = next(self.replay_iterator)
                self._stage_replay_batches_seen += 1
                self._stage_replay_images_seen += int(batch["label"].numel())
                self._record_replay_paths(batch)
                return batch
            self.replay_iterator = None
            print(
                "Replay traversal exhausted for current stage after "
                f"{self._stage_replay_batches_seen} batches; replay is now disabled "
                "until the next stage."
            )
            return None

    def _record_replay_paths(self, batch):
        paths = batch.get("impath")
        if paths is None:
            return
        if isinstance(paths, str):
            paths = [paths]
        self._stage_replay_path_exposures.update(str(path) for path in paths)

    def _measure_replay_gradient(self, replay_objective):
        """Measure the weighted replay term's gradient without touching .grad."""
        if not self.diagnostics_enabled or not replay_objective.requires_grad:
            return 0.0
        parameters = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        gradients = torch.autograd.grad(
            replay_objective,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        present = {
            index: gradient.detach().float()
            for index, gradient in enumerate(gradients)
            if gradient is not None
        }
        if not present:
            return 0.0
        squared_norm = sum(
            float(gradient.pow(2).sum().item()) for gradient in present.values()
        )
        gradient_norm = squared_norm**0.5
        if self._stage_replay_grad_vector_sum is None:
            self._stage_replay_grad_vector_sum = {
                index: gradient.clone() for index, gradient in present.items()
            }
        else:
            for index, gradient in present.items():
                if index not in self._stage_replay_grad_vector_sum:
                    self._stage_replay_grad_vector_sum[index] = gradient.clone()
                else:
                    self._stage_replay_grad_vector_sum[index].add_(gradient)
        self._stage_replay_grad_steps += 1
        self._stage_replay_grad_norm_sum += gradient_norm
        self._stage_replay_grad_norm_sq_sum += squared_norm
        return gradient_norm

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
                    replay_loss_scale=self._stage_replay_loss_scale,
                )
                loss = outputs["loss"]
            replay_objective = outputs.pop("_weighted_replay_objective")
            replay_gradient_norm = self._measure_replay_gradient(replay_objective)
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
                replay_loss_scale=self._stage_replay_loss_scale,
            )
            loss = outputs["loss"]
            replay_objective = outputs.pop("_weighted_replay_objective")
            replay_gradient_norm = self._measure_replay_gradient(replay_objective)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        summary = {"loss": float(loss.item())}
        summary["weighted_replay_gradient_norm"] = replay_gradient_norm
        summary["unnormalized_weighted_replay_gradient_norm"] = (
            replay_gradient_norm / self._stage_replay_loss_scale
            if self._stage_replay_loss_scale > 0.0
            else 0.0
        )
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
            self._stage_unnormalized_weighted_replay_sum += loss_summary[
                "unnormalized_weighted_loss_replay"
            ]
            self._stage_raw_replay_sum += loss_summary["loss_replay"]
            self._stage_lr_weighted_replay_grad_norm_sum += (
                self.get_current_lr()
                * loss_summary["weighted_replay_gradient_norm"]
            )
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
            print(f"Scoring final fitted domain for audit only: {final_domain}")
            self._finalize_fitted_stage(
                self._active_stage,
                final_domain,
                total_steps,
                add_to_bank=False,
            )
            self._active_stage = None
        super().after_train()
