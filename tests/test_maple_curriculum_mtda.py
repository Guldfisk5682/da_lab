from collections import Counter

import pytest

torch = pytest.importorskip("torch")

from trainers.maple_curriculum_mtda import (
    CurriculumContinuousSharedProjMaPLeMTDA,
    load_replay_manifest,
    materialize_manifest_records,
    select_topk_replay_records,
    stage_local_schedule_index,
)


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
