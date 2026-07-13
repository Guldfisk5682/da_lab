from scripts.maple_curriculum_mtda.analyze_replay_diagnostics import build_report


def _audit(stage, path, student_label, true_label, *, standard_topk=False):
    correct = student_label == true_label
    return {
        "boundary_stage": stage,
        "domain": "clipart",
        "impath": path,
        "true_label": true_label,
        "student_label": student_label,
        "student_conf": 0.8,
        "student_correct": correct,
        "clip_label": student_label,
        "clip_conf": 0.9,
        "clip_correct": correct,
        "agreement": True,
        "clean_pl_selected": True,
        "both_wrong_agree": not correct,
        "eligible": True,
        "standard_topk": standard_topk,
        "oracle_correct_topk": correct,
        "actual_selected": standard_topk,
    }


def test_report_exposes_shared_bias_and_cross_stage_label_flips():
    records = [
        _audit(0, "stable.jpg", 0, 0, standard_topk=True),
        _audit(0, "flip.jpg", 1, 0),
        _audit(1, "stable.jpg", 0, 0, standard_topk=True),
        _audit(1, "flip.jpg", 0, 0, standard_topk=True),
    ]
    report = build_report(records)
    stage0 = report["snapshots"]["stage0:clipart"]
    transition = report["transitions"]["clipart:stage0->stage1"]
    assert stage0["both_wrong_given_agreement"] == 0.5
    assert transition["student_label_flip_rate"] == 0.5
    assert transition["wrong_to_correct_rate"] == 0.5
    assert transition["standard_topk_jaccard"] == 0.5
