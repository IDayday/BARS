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


def guard_cfg(**kwargs):
    base = dict(
        use_cage=True,
        enable_churn_guard=True,
        disable_recovery=True,
        final_phase_dist=0.1,
        drift_threshold=1.0,
        min_commit_steps=0,
        replan_cooldown_steps=10,
        max_global_replans_per_episode=50,
        max_replans_per_100_steps=50,
        max_consecutive_replan_requests=10,
        fallback_to_gas_on_churn=True,
        fallback_to_gas_steps=20,
    )
    base.update(kwargs)
    return CAGEConfig(**base)


def request_replan(ctrl, step: int):
    original = np.asarray([0.0])
    return ctrl.select_subgoal(
        np.asarray([100.0]),
        np.asarray([10.0]),
        np.asarray([[0.0], [1.0]]),
        original,
        step,
        info={"original_subgoal": original, "original_subgoal_index": 0},
    )


def test_churn_guard_suppresses_replans_during_cooldown():
    ctrl = CAGEController(guard_cfg(), dist)
    ctrl.reset_episode(np.asarray([100.0]), np.asarray([10.0]), np.asarray([[0.0], [1.0]]))

    _, _, _, should_replan, _ = request_replan(ctrl, 0)
    assert should_replan
    assert ctrl.global_replan_request_count == 1

    _, _, _, should_replan, trace = request_replan(ctrl, 1)
    assert not should_replan
    assert trace["replan_suppressed_reason"] == "replan_cooldown"
    assert ctrl.replan_suppressed_by_cooldown_count == 1
    assert ctrl.global_replan_request_count == 1


def test_churn_guard_enters_fallback_after_consecutive_replan_burst():
    ctrl = CAGEController(guard_cfg(replan_cooldown_steps=0, max_consecutive_replan_requests=1), dist)
    ctrl.reset_episode(np.asarray([100.0]), np.asarray([10.0]), np.asarray([[0.0], [1.0]]))

    assert request_replan(ctrl, 0)[3]
    selected, _, state, should_replan, trace = request_replan(ctrl, 1)

    assert not should_replan
    assert np.allclose(selected, np.asarray([0.0]))
    assert state == CAGEState.FALLBACK_TO_GAS.value
    assert trace["replan_suppressed_reason"] == "consecutive_replan_burst"
    assert ctrl.churn_guard_trigger_count == 1
    assert ctrl.fallback_to_gas_count == 1


def test_episode_replan_budget_disables_additional_cage_replans():
    ctrl = CAGEController(guard_cfg(replan_cooldown_steps=0, max_global_replans_per_episode=1), dist)
    ctrl.reset_episode(np.asarray([100.0]), np.asarray([10.0]), np.asarray([[0.0], [1.0]]))

    assert request_replan(ctrl, 0)[3]
    _, _, _, should_replan, trace = request_replan(ctrl, 1)

    assert not should_replan
    assert trace["replan_suppressed_reason"] == "episode_replan_budget"
    assert ctrl.global_replan_request_count == 1
    assert ctrl.replan_suppressed_by_budget_count == 1


def test_recovery_lockout_suppresses_immediate_repeated_recovery():
    cfg = guard_cfg(disable_recovery=False, recovery_lockout_steps_after_failure=25)
    ctrl = CAGEController(cfg, dist)
    ctrl.reset_episode(np.asarray([100.0]), np.asarray([10.0]), np.asarray([[0.0], [1.0]]))
    ctrl._last_step = 0
    ctrl._start_recovery_lockout()

    _, _, state, should_replan, trace = request_replan(ctrl, 1)

    assert not should_replan
    assert state == CAGEState.RECOVERY_LOCKOUT.value
    assert trace["recovery_suppressed_reason"] == "recovery_lockout"
    assert ctrl.recovery_suppressed_by_lockout_count == 1


def test_safe_mode_trace_contains_churn_fields():
    ctrl = CAGEController(guard_cfg(), dist)
    ctrl.reset_episode(np.asarray([100.0]), np.asarray([10.0]), np.asarray([[0.0], [1.0]]))

    _, _, _, _, trace = request_replan(ctrl, 0)

    assert trace["cage_safe_mode_enabled"] is True
    assert "consecutive_replan_request_count" in trace
    assert "replan_window_count" in trace
    assert "replan_cooldown_remaining" in trace


def test_cage_full_does_not_enable_churn_guard_by_default():
    cfg = CAGEConfig(
        use_cage=True,
        disable_recovery=True,
        final_phase_dist=0.1,
        drift_threshold=1.0,
        min_commit_steps=0,
    )
    ctrl = CAGEController(cfg, dist)
    ctrl.reset_episode(np.asarray([100.0]), np.asarray([10.0]), np.asarray([[0.0], [1.0]]))

    assert request_replan(ctrl, 0)[3]
    assert request_replan(ctrl, 1)[3]
    assert ctrl.replan_suppressed_by_cooldown_count == 0
    assert ctrl.churn_guard_trigger_count == 0
