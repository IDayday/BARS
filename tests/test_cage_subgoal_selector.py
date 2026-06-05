from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external_src" / "GAS"))

from cage.config import CAGEConfig
from cage.recovery import RecoverySelector
from cage.subgoal_selector import SubgoalSelector


def dist(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def test_selector_chooses_farther_waypoint_when_progress_is_good():
    cfg = CAGEConfig(use_cage=True, max_subgoal_dist=10.0, min_subgoal_dist=1.0)
    selector = SubgoalSelector(cfg, dist)
    path = np.asarray([[0.0], [2.0], [5.0], [9.0], [20.0]])

    selected = selector.select(np.asarray([0.0]), np.asarray([20.0]), path, recent_stalls=0)
    assert selected.index == 3
    assert np.allclose(selected.subgoal, np.asarray([9.0]))


def test_selector_chooses_shorter_waypoint_after_stall():
    cfg = CAGEConfig(use_cage=True, max_subgoal_dist=10.0, min_subgoal_dist=1.0)
    selector = SubgoalSelector(cfg, dist)
    path = np.asarray([[0.0], [2.0], [5.0], [9.0], [20.0]])

    selected = selector.select(np.asarray([0.0]), np.asarray([20.0]), path, recent_stalls=1)
    assert selected.index == 2
    assert np.allclose(selected.subgoal, np.asarray([5.0]))


def test_selector_chooses_final_goal_in_final_goal_phase():
    cfg = CAGEConfig(use_cage=True, max_subgoal_dist=10.0, min_subgoal_dist=1.0)
    selector = SubgoalSelector(cfg, dist)
    final_goal = np.asarray([20.0])

    selected = selector.select(
        np.asarray([18.0]),
        final_goal,
        np.asarray([[16.0], [18.0], [19.0]]),
        final_goal_phase=True,
    )
    assert selected.index == -1
    assert np.allclose(selected.subgoal, final_goal)


def test_recovery_picks_path_node_before_global_replan():
    cfg = CAGEConfig(
        use_cage=True,
        max_subgoal_dist=10.0,
        drift_threshold=10.0,
        recovery_suffix_weight=0.25,
    )
    recovery = RecoverySelector(cfg, dist)
    path = np.asarray([[0.0], [2.0], [5.0], [9.0]])

    selected = recovery.select(np.asarray([4.5]), path, current_index=3)
    assert selected is not None
    assert selected.index == 2
    assert np.allclose(selected.target, np.asarray([5.0]))
