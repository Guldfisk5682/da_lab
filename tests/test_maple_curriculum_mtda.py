import pytest

torch = pytest.importorskip("torch")

from trainers.maple_curriculum_mtda import select_topk_replay_records


def _record(name, student_label, student_conf, clip_label=None, clip_conf=0.9):
    return {
        "impath": name,
        "student_label": student_label,
        "student_conf": student_conf,
        "clip_label": student_label if clip_label is None else clip_label,
        "clip_conf": clip_conf,
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
