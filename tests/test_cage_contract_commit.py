from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external_src" / "GAS"))

from cage.config import CAGEConfig
from cage.state_machine import CAGEController, CAGEState


def dist(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def contract_cfg(**kwargs):
    base = dict(
        use_cage=True,
        contract_commit=True,
        min_commit_steps=0,
        contract_min_commit_steps=0,
        max_subgoal_dist=100.0,
        min_subgoal_dist=1.0,
        final_phase_dist=0.1,
        drift_threshold=1.0,
        contract_lcb_threshold=0.35,
        contract_negative_progress_threshold=0.95,
    )
    base.update(kwargs)
    return CAGEConfig(**base).with_env_defaults()


def test_contract_commit_rejects_far_candidate_and_falls_back_to_gas():
    ctrl = CAGEController(contract_cfg(), dist)
    path = np.asarray([[0.0], [20.0], [40.0]])
    original = np.asarray([0.0])
    ctrl.reset_episode(np.asarray([0.0]), np.asarray([50.0]), path)

    selected, idx, state, should_replan, trace = ctrl.select_subgoal(
        np.asarray([0.0]),
        np.asarray([50.0]),
        path,
        original,
        0,
        info={"original_subgoal": original, "original_subgoal_index": 0},
    )

    assert not should_replan
    assert np.allclose(selected, original)
    assert idx == 0
    assert state == CAGEState.FALLBACK_TO_GAS.value
    assert trace["contract_model_loaded"] == 0
    assert trace["contract_gate_pass"] is False
    assert ctrl.contract_gate_reject_count == 1
    assert ctrl.contract_fallback_to_gas_when_uncertain_count == 1


def test_contract_commit_keeps_committed_target_when_new_target_is_uncertain():
    ctrl = CAGEController(contract_cfg(contract_lcb_threshold=0.01), dist)
    first_path = np.asarray([[0.0], [2.0], [4.0]])
    original = np.asarray([0.0])
    ctrl.reset_episode(np.asarray([0.0]), np.asarray([50.0]), first_path)
    first, first_idx, _, _, _ = ctrl.select_subgoal(
        np.asarray([0.0]),
        np.asarray([50.0]),
        first_path,
        original,
        0,
        info={"original_subgoal": original, "original_subgoal_index": 0},
    )

    far_path = np.asarray([[0.0], [30.0], [60.0]])
    selected, idx, state, should_replan, trace = ctrl.select_subgoal(
        np.asarray([0.0]),
        np.asarray([50.0]),
        far_path,
        original,
        1,
        info={"original_subgoal": original, "original_subgoal_index": 0},
    )

    assert not should_replan
    assert np.allclose(selected, first)
    assert idx == first_idx
    assert state == CAGEState.COMMITTING.value
    assert trace["fallback_reason"].startswith("target_contract_rejected")


def test_contract_commit_drift_does_not_request_replan_storm_by_default():
    ctrl = CAGEController(contract_cfg(disable_recovery=True), dist)
    path = np.asarray([[0.0], [1.0]])
    original = np.asarray([0.0])
    ctrl.reset_episode(np.asarray([100.0]), np.asarray([10.0]), path)

    for step in range(3):
        _, _, _, should_replan, trace = ctrl.select_subgoal(
            np.asarray([100.0]),
            np.asarray([10.0]),
            path,
            original,
            step,
            info={"original_subgoal": original, "original_subgoal_index": 0},
        )
        assert not should_replan
        assert trace["fallback_reason"] == "contract_commit_drift_guard"
    assert ctrl.global_replan_request_count == 0
