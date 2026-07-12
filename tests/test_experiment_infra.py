import argparse
import json

import pytest
import torch

from scripts import experiment_guard
from scripts.maple_mtda.collect_officehome_results import summarize_accs
from trainers.checkpoint_utils import load_state_dict_checked
from trainers.maple_mtda import build_self_distill_mask


def guard_args(tmp_path, **overrides):
    values = {
        "allow_legacy": False,
        "data": str(tmp_path / "data"),
        "dataset_config": str(tmp_path / "dataset.yaml"),
        "effective_opts": "TEST.EVAL_EVERY_EPOCH False OPTIM.MAX_EPOCH 5",
        "extra_opts": "OPTIM.MAX_EPOCH 5",
        "method_tag": "method_a",
        "output_dir": str(tmp_path / "output"),
        "post_init_load_epoch": 5,
        "post_init_method_tag": "",
        "seed": 42,
        "source": "A",
        "targets": ["C", "P", "R"],
        "trainer": "ContinuousSharedProjMaPLeMTDA",
        "trainer_config": str(tmp_path / "trainer.yaml"),
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_manifest_signature_is_stable_and_sensitive(tmp_path):
    args = guard_args(tmp_path)
    payload = experiment_guard.canonical_payload(args)
    signature = experiment_guard.payload_signature(payload)
    assert signature == experiment_guard.payload_signature(payload)

    changed = guard_args(tmp_path, extra_opts="OPTIM.MAX_EPOCH 6")
    assert signature != experiment_guard.payload_signature(
        experiment_guard.canonical_payload(changed)
    )


def test_incomplete_results_are_not_summarized_by_default():
    macro, complete, found = summarize_accs([80.0, None, 90.0])
    assert macro is None
    assert complete is False
    assert found == 2

    macro, complete, found = summarize_accs(
        [80.0, None, 90.0], allow_incomplete=True
    )
    assert macro == pytest.approx(85.0)
    assert complete is False
    assert found == 2


def test_checked_state_dict_rejects_unknown_mismatch():
    module = torch.nn.Linear(2, 1)
    state = module.state_dict()
    state.pop("bias")
    with pytest.raises(RuntimeError, match="missing=.*bias"):
        load_state_dict_checked(module, state, context="test")


def test_checked_state_dict_accepts_declared_missing_key():
    module = torch.nn.Linear(2, 1)
    state = module.state_dict()
    state.pop("bias")
    missing, unexpected = load_state_dict_checked(
        module, state, allowed_missing=("bias",), context="test"
    )
    assert missing == ["bias"]
    assert unexpected == []


def test_teacher_handoff_mask_uses_old_confident_clip_uncertain_region():
    old_conf = torch.tensor([0.8, 0.8, 0.6, 0.9])
    clip_conf = torch.tensor([0.6, 0.8, 0.5, 0.69])
    mask = build_self_distill_mask(
        "teacher_handoff",
        old_conf,
        old_conf_low=0.7,
        old_conf_high=1.0,
        reference_conf=clip_conf,
        clip_conf_high=0.7,
    )
    assert mask.tolist() == [True, False, False, True]
