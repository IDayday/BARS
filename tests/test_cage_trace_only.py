from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external_src" / "GAS"))

from cage.config import CAGEConfig
from cage.state_machine import CAGEController


def dist(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def test_trace_only_returns_original_gas_subgoal():
    cfg = CAGEConfig(use_cage=True, trace_only=True, final_phase_dist=0.1)
    ctrl = CAGEController(cfg, dist)
    path = np.asarray([[0.0], [5.0], [10.0]])
    original = np.asarray([5.0])
    ctrl.reset_episode(np.asarray([0.0]), np.asarray([10.0]), path)

    selected, idx, _, should_replan, trace = ctrl.select_subgoal(
        np.asarray([0.0]),
        np.asarray([10.0]),
        path,
        original,
        0,
        info={"original_subgoal": original, "original_subgoal_index": 1},
    )

    assert np.allclose(selected, original)
    assert idx == 1
    assert not should_replan
    assert trace["cage_trace_only"] is True
    assert trace["original_subgoal_used"] is True


def test_trace_only_never_requests_cage_replan_without_path():
    cfg = CAGEConfig(use_cage=True, trace_only=True, final_phase_dist=0.1)
    ctrl = CAGEController(cfg, dist)
    original = np.asarray([3.0])
    ctrl.reset_episode(np.asarray([0.0]), np.asarray([10.0]), None)

    selected, idx, _, should_replan, trace = ctrl.select_subgoal(
        np.asarray([0.0]),
        np.asarray([10.0]),
        None,
        original,
        0,
        info={"original_subgoal": original, "original_subgoal_index": 0},
    )

    assert np.allclose(selected, original)
    assert idx == 0
    assert not should_replan
    assert ctrl.global_replan_request_count == 0
    assert trace["original_subgoal_used"] is True
