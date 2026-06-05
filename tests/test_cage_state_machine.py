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


def test_subgoal_is_not_switched_before_min_commitment():
    cfg = CAGEConfig(
        use_cage=True,
        min_commit_steps=3,
        max_subgoal_dist=4.0,
        min_subgoal_dist=1.0,
        final_phase_dist=0.1,
    )
    ctrl = CAGEController(cfg, dist)
    path = np.asarray([[0.0], [2.0], [4.0], [6.0]])
    ctrl.reset_episode(np.asarray([0.0]), np.asarray([10.0]), path)

    first, first_idx, _, _, _ = ctrl.select_subgoal(np.asarray([0.0]), np.asarray([10.0]), path, None, 0)
    assert first_idx == 2

    new_path = np.asarray([[0.0], [3.0], [6.0], [9.0]])
    second, second_idx, state, should_replan, _ = ctrl.select_subgoal(
        np.asarray([0.0]), np.asarray([10.0]), new_path, first, 1
    )
    assert np.allclose(second, first)
    assert second_idx == first_idx
    assert state == CAGEState.COMMITTING.value
    assert not should_replan


def test_stall_triggers_local_stall_then_recovery():
    cfg = CAGEConfig(
        use_cage=True,
        min_commit_steps=0,
        stall_window=2,
        progress_eps=0.1,
        max_subgoal_dist=8.0,
        min_subgoal_dist=1.0,
        recovery_commit_steps=2,
        max_recovery_attempts=1,
        final_phase_dist=0.1,
    )
    ctrl = CAGEController(cfg, dist)
    path = np.asarray([[0.0], [2.0], [4.0], [6.0]])
    ctrl.reset_episode(np.asarray([0.0]), np.asarray([10.0]), path)

    target, _, _, _, _ = ctrl.select_subgoal(np.asarray([0.0]), np.asarray([10.0]), path, None, 0)
    for _ in range(3):
        ctrl.update_after_step(np.asarray([0.0]), np.asarray([0.0]), target)

    assert ctrl.state == CAGEState.LOCAL_STALL
    _, _, state, should_replan, trace = ctrl.select_subgoal(np.asarray([0.0]), np.asarray([10.0]), path, target, 4)
    assert state == CAGEState.RECOVERY.value
    assert not should_replan
    assert trace["recovery_target_idx"] is not None


def test_failed_recovery_requests_replan():
    cfg = CAGEConfig(
        use_cage=True,
        min_commit_steps=0,
        stall_window=2,
        progress_eps=0.1,
        max_subgoal_dist=8.0,
        min_subgoal_dist=1.0,
        max_recovery_attempts=0,
        final_phase_dist=0.1,
    )
    ctrl = CAGEController(cfg, dist)
    path = np.asarray([[0.0], [2.0], [4.0], [6.0]])
    ctrl.reset_episode(np.asarray([0.0]), np.asarray([10.0]), path)

    target, _, _, _, _ = ctrl.select_subgoal(np.asarray([0.0]), np.asarray([10.0]), path, None, 0)
    for _ in range(3):
        ctrl.update_after_step(np.asarray([0.0]), np.asarray([0.0]), target)

    _, _, state, should_replan, _ = ctrl.select_subgoal(np.asarray([0.0]), np.asarray([10.0]), path, target, 4)
    assert state == CAGEState.REPLAN_MISS.value
    assert should_replan
    assert ctrl.global_replan_request_count == 1


def test_recovery_target_is_committed_before_failure_window():
    cfg = CAGEConfig(
        use_cage=True,
        min_commit_steps=0,
        stall_window=2,
        progress_eps=0.1,
        max_subgoal_dist=8.0,
        min_subgoal_dist=1.0,
        recovery_commit_steps=3,
        max_recovery_attempts=2,
        final_phase_dist=0.1,
    )
    ctrl = CAGEController(cfg, dist)
    path = np.asarray([[0.0], [2.0], [4.0], [6.0]])
    ctrl.reset_episode(np.asarray([0.0]), np.asarray([10.0]), path)

    target, _, _, _, _ = ctrl.select_subgoal(np.asarray([0.0]), np.asarray([10.0]), path, None, 0)
    for _ in range(3):
        ctrl.update_after_step(np.asarray([0.0]), np.asarray([0.0]), target)

    recovery_target, recovery_idx, state, should_replan, _ = ctrl.select_subgoal(
        np.asarray([0.0]), np.asarray([10.0]), path, target, 4
    )
    assert state == CAGEState.RECOVERY.value
    assert not should_replan

    repeated_target, repeated_idx, state, should_replan, _ = ctrl.select_subgoal(
        np.asarray([0.0]), np.asarray([10.0]), path, recovery_target, 5
    )
    assert np.allclose(repeated_target, recovery_target)
    assert repeated_idx == recovery_idx
    assert state == CAGEState.RECOVERY.value
    assert not should_replan


def test_final_goal_phase_bypasses_regular_commitment():
    cfg = CAGEConfig(
        use_cage=True,
        min_commit_steps=10,
        max_subgoal_dist=5.0,
        min_subgoal_dist=1.0,
        final_phase_dist=3.0,
    )
    ctrl = CAGEController(cfg, dist)
    path = np.asarray([[0.0], [2.0], [4.0], [6.0]])
    ctrl.reset_episode(np.asarray([0.0]), np.asarray([10.0]), path)

    target, _, _, _, _ = ctrl.select_subgoal(np.asarray([0.0]), np.asarray([10.0]), path, None, 0)
    final_target, final_idx, state, should_replan, _ = ctrl.select_subgoal(
        np.asarray([8.0]),
        np.asarray([10.0]),
        np.asarray([[8.0], [9.0]]),
        target,
        1,
    )
    assert np.allclose(final_target, np.asarray([10.0]))
    assert final_idx == -1
    assert state == CAGEState.FINAL_GOAL.value
    assert not should_replan
