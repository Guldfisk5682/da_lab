from collections import Counter
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from trainers.maple_curriculum_mtda import (
    CurriculumContinuousSharedProjMaPLeMTDA,
    load_replay_manifest,
    materialize_manifest_records,
    replay_step_budget_scale,
    select_topk_replay_records,
    stage_local_schedule_index,
    weighted_stage_bounds,
)
from trainers.maple_mtda import CustomMaPLeMTDA


def test_one_pass_step_normalization_matches_actual_replay_update_budget():
    scale = replay_step_budget_scale("one_pass_steps", 118, 1010)
    assert scale == pytest.approx(118 / 1010)
    assert scale * 1010 == pytest.approx(118)


def test_weighted_stage_bounds_reproduce_low_budget_officehome_schedule():
    assert weighted_stage_bounds(2424, [5, 4, 3]) == [
        (0, 1010),
        (1010, 1818),
        (1818, 2424),
    ]


def test_weighted_stage_bounds_preserve_historical_equal_split():
    assert weighted_stage_bounds(3030, [1, 1, 1]) == [
        (0, 1010),
        (1010, 2020),
        (2020, 3030),
    ]


def test_weighted_stage_bounds_reject_invalid_or_empty_stages():
    with pytest.raises(ValueError, match="positive integers"):
        weighted_stage_bounds(10, [1, 0, 1])
    with pytest.raises(ValueError, match="too small"):
        weighted_stage_bounds(2, [1, 1, 1])


def test_replay_step_normalization_caps_reference_at_stage_length():
    assert replay_step_budget_scale("one_pass_steps", 12, 10) == 1.0
    assert replay_step_budget_scale("none", 2, 10) == 1.0


def _record(
    name,
    student_label,
    student_conf,
    clip_label=None,
    clip_conf=0.9,
    true_label=None,
):
    return {
        "impath": name,
        "student_label": student_label,
        "student_conf": student_conf,
        "clip_label": student_label if clip_label is None else clip_label,
        "clip_conf": clip_conf,
        "true_label": student_label if true_label is None else true_label,
    }


def test_topk_is_per_class_and_confidence_ranked():
    records = [
        _record("c0-low", 0, 0.71),
        _record("c0-high", 0, 0.95),
        _record("c1-low", 1, 0.72),
        _record("c1-high", 1, 0.94),
    ]
    selected, eligible = select_topk_replay_records(
        records,
        topk_per_class=1,
        student_threshold=0.7,
        clip_threshold=0.7,
    )
    assert eligible == 4
    assert [record["impath"] for record in selected] == ["c0-high", "c1-high"]


def test_selection_requires_agreement_and_both_thresholds_without_backfill():
    records = [
        _record("disagree", 0, 0.99, clip_label=1),
        _record("student-low", 0, 0.69),
        _record("clip-low", 0, 0.99, clip_conf=0.69),
        _record("valid", 1, 0.8),
    ]
    selected, eligible = select_topk_replay_records(
        records,
        topk_per_class=8,
        student_threshold=0.7,
        clip_threshold=0.7,
    )
    assert eligible == 1
    assert [record["impath"] for record in selected] == ["valid"]


def test_oracle_correct_selection_prioritizes_correct_before_confidence():
    records = [
        _record("wrong-high", 0, 0.99, true_label=1),
        _record("correct-low", 0, 0.75, true_label=0),
    ]
    selected, eligible = select_topk_replay_records(
        records,
        topk_per_class=1,
        student_threshold=0.7,
        clip_threshold=0.7,
        prefer_correct=True,
    )
    assert eligible == 2
    assert [record["impath"] for record in selected] == ["correct-low"]


def test_manifest_materialization_freezes_indices_and_can_swap_only_labels(tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        '{"stage": 0, "fitted_domain": "clipart", "records": '
        '[{"impath": "x.jpg", "student_label": 2, "pseudo_label": 2}]}\n'
    )
    payload = load_replay_manifest(manifest_path)[(0, "clipart")]
    current = [
        {
            "impath": "x.jpg",
            "domain": "clipart",
            "domain_id": 1,
            "dataset_index": 7,
            "true_label": 4,
            "student_label": 3,
            "student_conf": 0.8,
            "clip_label": 3,
            "clip_conf": 0.9,
        }
    ]
    pseudo = materialize_manifest_records(payload, current, "pseudo")
    oracle = materialize_manifest_records(payload, current, "ground_truth")
    assert pseudo[0]["dataset_index"] == oracle[0]["dataset_index"] == 7
    assert pseudo[0]["pseudo_label"] == 2
    assert pseudo[0]["replay_label"] == 2
    assert oracle[0]["replay_label"] == 4


def test_cycle_replay_restarts_loader_instead_of_disabling_it():
    trainer = object.__new__(CurriculumContinuousSharedProjMaPLeMTDA)
    batch = {"label": torch.tensor([0, 1]), "impath": ["a.jpg", "b.jpg"]}
    trainer.replay_loader = [batch]
    trainer.replay_iterator = iter(trainer.replay_loader)
    trainer.replay_traversal = "cycle"
    trainer._stage_replay_batches_seen = 0
    trainer._stage_replay_images_seen = 0
    trainer._stage_replay_path_exposures = Counter()
    assert trainer._next_replay_batch() is batch
    assert trainer._next_replay_batch() is batch
    assert trainer._stage_replay_batches_seen == 2
    assert trainer._stage_replay_images_seen == 4
    assert trainer._stage_replay_path_exposures == {"a.jpg": 2, "b.jpg": 2}


def test_replay_scoring_uses_target_train_loader_metadata_not_test_split():
    class FixedModel(torch.nn.Module):
        def forward(self, image):
            return image

        def _compute_reference_logits(self, image):
            return image

    train_items = [
        SimpleNamespace(impath="train-a.jpg", label=0, domain=1),
        SimpleNamespace(impath="train-b.jpg", label=1, domain=1),
    ]
    train_batch = {
        "img": torch.tensor([[8.0, 0.0], [0.0, 8.0]]),
        "index": torch.tensor([0, 1]),
        "impath": ["train-a.jpg", "train-b.jpg"],
    }

    trainer = object.__new__(CurriculumContinuousSharedProjMaPLeMTDA)
    trainer.model = FixedModel()
    trainer.device = torch.device("cpu")
    trainer.dm = SimpleNamespace(
        dataset=SimpleNamespace(train_u_by_domain={"target": train_items})
    )
    trainer.replay_score_loaders_by_domain = {"target": [train_batch]}
    # A divergent test split must never be consulted when building replay.
    trainer.test_loaders_by_domain = {"target": pytest.fail}
    trainer.set_model_mode = lambda mode: trainer.model.train(mode == "train")

    records = trainer._score_domain_for_replay("target")

    assert [record["impath"] for record in records] == [
        "train-a.jpg",
        "train-b.jpg",
    ]
    assert [record["true_label"] for record in records] == [0, 1]
    assert [record["student_label"] for record in records] == [0, 1]


def test_replay_scoring_rejects_index_path_mismatch():
    class FixedModel(torch.nn.Module):
        def forward(self, image):
            return image

        def _compute_reference_logits(self, image):
            return image

    trainer = object.__new__(CurriculumContinuousSharedProjMaPLeMTDA)
    trainer.model = FixedModel()
    trainer.device = torch.device("cpu")
    trainer.dm = SimpleNamespace(
        dataset=SimpleNamespace(
            train_u_by_domain={
                "target": [SimpleNamespace(impath="expected.jpg", label=0, domain=1)]
            }
        )
    )
    trainer.replay_score_loaders_by_domain = {
        "target": [
            {
                "img": torch.tensor([[8.0, 0.0]]),
                "index": torch.tensor([0]),
                "impath": ["wrong.jpg"],
            }
        ]
    }
    trainer.set_model_mode = lambda mode: trainer.model.train(mode == "train")

    with pytest.raises(RuntimeError, match="index/path mismatch"):
        trainer._score_domain_for_replay("target")


def test_replay_backward_runs_after_main_backward_in_same_optimizer_step():
    class OrderedBackwardModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.prompt = torch.nn.Parameter(torch.tensor(0.0))
            self.replay_saw_main_gradient = False

        def forward_train(self, image_x, label_x, image_u):
            main = (self.prompt - 1.0).pow(2)
            zero = main.new_zeros(())
            return {
                "loss": main,
                "source_ce": main.detach(),
                "loss_replay": zero,
                "unnormalized_weighted_loss_replay": zero,
                "weighted_loss_replay": zero,
                "replay_loss_scale": zero,
                "effective_replay_lambda": zero,
                "replay_accuracy": zero,
                "replay_active": zero,
                "_source_ce_objective": main,
                "_pl_hard_objective": None,
                "_pl_soft_objective": None,
                "_weighted_replay_objective": zero,
            }

        def forward_replay(self, image, label, replay_loss_scale):
            self.replay_saw_main_gradient = self.prompt.grad is not None
            raw = (self.prompt + 2.0).pow(2)
            weighted = 0.75 * replay_loss_scale * raw
            one = raw.new_tensor(1.0)
            return {
                "loss_replay": raw.detach(),
                "unnormalized_weighted_loss_replay": (0.75 * raw).detach(),
                "weighted_loss_replay": weighted.detach(),
                "replay_loss_scale": raw.new_tensor(replay_loss_scale),
                "effective_replay_lambda": raw.new_tensor(
                    0.75 * replay_loss_scale
                ),
                "replay_accuracy": one,
                "replay_active": one,
                "_weighted_replay_objective": weighted,
            }

    trainer = object.__new__(CurriculumContinuousSharedProjMaPLeMTDA)
    trainer.model = OrderedBackwardModel()
    trainer.device = torch.device("cpu")
    trainer.optim = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    trainer.cfg = SimpleNamespace(
        TRAINER=SimpleNamespace(MAPLE_MTDA=SimpleNamespace(PREC="fp16"))
    )
    trainer.parse_batch_train = lambda batch_x, batch_u: (None, None, None)
    trainer.max_epoch = 1
    trainer.num_batches = 2
    trainer.epoch = 0
    trainer.batch_idx = 0
    trainer._stage_replay_loss_scale = 1.0
    trainer.reset_optim_per_stage = False
    trainer._measure_replay_gradient = lambda objective: 0.0
    trainer._measure_pl_branch_gradients = lambda *args: {}
    trainer.update_lr = lambda: None

    summary = trainer.forward_backward(
        {}, {}, replay_batch={"img": torch.zeros(1), "label": torch.zeros(1)}
    )

    assert trainer.model.replay_saw_main_gradient
    # At p=0: main gradient=-2 and weighted replay gradient=+3. One SGD
    # update with their sum gives p=-0.1.
    assert trainer.model.prompt.item() == pytest.approx(-0.1)
    assert summary["loss"] == pytest.approx(4.0)
    assert summary["replay_active"] == 1.0


def test_replay_gradient_audit_measures_only_weighted_replay_objective():
    trainer = object.__new__(CurriculumContinuousSharedProjMaPLeMTDA)
    parameter = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    trainer.model = torch.nn.Module()
    trainer.model.register_parameter("prompt", parameter)
    trainer.diagnostics_enabled = True
    trainer._stage_replay_grad_steps = 0
    trainer._stage_replay_grad_norm_sum = 0.0
    trainer._stage_replay_grad_norm_sq_sum = 0.0
    trainer._stage_replay_grad_vector_sum = None

    norm = trainer._measure_replay_gradient((parameter**2).sum())

    assert norm == pytest.approx(10.0)
    assert trainer._stage_replay_grad_steps == 1
    assert trainer._stage_replay_grad_norm_sum == pytest.approx(10.0)
    accumulated = next(iter(trainer._stage_replay_grad_vector_sum.values()))
    assert torch.equal(accumulated, torch.tensor([6.0, 8.0]))


def _dual_pl_model(variant):
    model = object.__new__(CustomMaPLeMTDA)
    torch.nn.Module.__init__(model)
    model.pl_variant = variant
    model.pl_dual_conf_threshold = 0.7
    model.pl_threshold = 0.7
    model.pl_student_threshold = 0.7
    model.pl_use_student_low_conf_mask = True
    model.pl_student_soft_lambda = 0.5
    return model


def test_dual_pl_separates_agreement_hard_and_disagreement_soft_masks():
    model = _dual_pl_model("agreement_hard_soft")
    strong = torch.tensor(
        [[2.0, 0.0, -1.0], [0.0, 1.0, -1.0], [1.0, 0.0, -1.0]],
        requires_grad=True,
    )
    teacher = torch.log(
        torch.tensor([[0.8, 0.1, 0.1], [0.75, 0.2, 0.05], [0.6, 0.3, 0.1]])
    )
    student = torch.log(
        torch.tensor([[0.75, 0.2, 0.05], [0.1, 0.8, 0.1], [0.8, 0.1, 0.1]])
    ).requires_grad_()
    terms = model._dual_view_pseudo_label_terms(
        strong, student, teacher, true_label=torch.tensor([0, 1, 0])
    )
    assert terms["hard_count"].item() == 1
    assert terms["soft_count"].item() == 1
    assert terms["hard_correct_count"].item() == 1
    assert terms["soft_correct_count"].item() == 1
    (terms["hard_sum"] + terms["soft_sum"]).backward()
    assert strong.grad is not None
    assert student.grad is None


def test_hard_only_control_drops_disagreement_without_changing_hard_gate():
    model = _dual_pl_model("agreement_hard")
    strong = torch.zeros(2, 2, requires_grad=True)
    teacher = torch.log(torch.tensor([[0.8, 0.2], [0.8, 0.2]]))
    student = torch.log(torch.tensor([[0.9, 0.1], [0.1, 0.9]]))
    terms = model._dual_view_pseudo_label_terms(strong, student, teacher)
    assert terms["hard_count"].item() == 1
    assert terms["soft_count"].item() == 0


def test_student_soft_uses_student_high_teacher_low_and_linear_confidence_weight():
    model = _dual_pl_model("agreement_hard_student_soft")
    strong = torch.tensor(
        [[0.2, -0.2], [1.0, -1.0], [-0.5, 0.5]], requires_grad=True
    )
    # Only sample 0 qualifies: student confidence 0.8 and teacher confidence 0.6.
    teacher = torch.log(torch.tensor([[0.6, 0.4], [0.8, 0.2], [0.8, 0.2]]))
    student = torch.log(
        torch.tensor([[0.8, 0.2], [0.1, 0.9], [0.6, 0.4]])
    ).requires_grad_()
    terms = model._dual_view_pseudo_label_terms(
        strong, student, teacher, true_label=torch.tensor([0, 1, 0])
    )
    expected_weight = (0.8 - 0.7) / 0.3
    expected_kl = torch.nn.functional.kl_div(
        torch.log_softmax(strong[0], dim=-1),
        torch.tensor([0.8, 0.2]),
        reduction="sum",
    )
    assert terms["hard_count"].item() == 0
    assert terms["soft_count"].item() == 1
    assert terms["soft_weight_sum"].item() == pytest.approx(expected_weight)
    assert terms["soft_sum"].item() == pytest.approx(
        expected_weight * expected_kl.item()
    )
    assert terms["soft_correct_count"].item() == 1
    terms["soft_sum"].backward()
    assert strong.grad is not None
    assert student.grad is None


def test_student_top1_control_changes_only_the_target_distribution():
    soft_model = _dual_pl_model("agreement_hard_student_soft")
    top1_model = _dual_pl_model("agreement_hard_student_top1")
    strong = torch.tensor([[0.2, -0.2], [1.0, -1.0]], requires_grad=True)
    teacher = torch.log(torch.tensor([[0.6, 0.4], [0.8, 0.2]]))
    student = torch.log(torch.tensor([[0.8, 0.2], [0.1, 0.9]])).requires_grad_()

    soft_terms = soft_model._dual_view_pseudo_label_terms(
        strong, student, teacher, true_label=torch.tensor([0, 1])
    )
    top1_terms = top1_model._dual_view_pseudo_label_terms(
        strong, student, teacher, true_label=torch.tensor([0, 1])
    )

    expected_weight = (0.8 - 0.7) / 0.3
    expected_ce = torch.nn.functional.cross_entropy(
        strong[0].unsqueeze(0), torch.tensor([0]), reduction="sum"
    )
    assert top1_terms["hard_count"].item() == soft_terms["hard_count"].item()
    assert top1_terms["soft_count"].item() == soft_terms["soft_count"].item()
    assert top1_terms["soft_weight_sum"].item() == pytest.approx(
        soft_terms["soft_weight_sum"].item()
    )
    assert top1_terms["soft_sum"].item() == pytest.approx(
        expected_weight * expected_ce.item()
    )
    assert top1_terms["soft_sum"].item() != pytest.approx(
        soft_terms["soft_sum"].item()
    )
    top1_terms["soft_sum"].backward()
    assert strong.grad is not None
    assert student.grad is None


def test_stage_local_schedule_gives_equal_lr_budget_to_each_virtual_epoch():
    indices = [stage_local_schedule_index(step, 1010, 5) for step in range(1010)]
    assert [indices.count(index) for index in range(5)] == [202] * 5


@pytest.mark.parametrize("stage_length", [1009, 1010, 1011])
def test_stage_local_schedule_covers_all_virtual_epochs(stage_length):
    indices = [
        stage_local_schedule_index(step, stage_length, 5)
        for step in range(stage_length)
    ]
    assert indices == sorted(indices)
    assert set(indices) == set(range(5))
    assert max(indices.count(index) for index in range(5)) - min(
        indices.count(index) for index in range(5)
    ) <= 1
