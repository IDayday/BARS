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


def test_committed_watchdog_locks_out_stale_committed_target():
    cfg = CAGEConfig(
        use_cage=True,
        contract_intervene=True,
        min_commit_steps=0,
        contract_min_commit_steps=0,
        stall_window=1,
        progress_eps=0.1,
        contract_committed_min_target_progress=0.01,
        contract_committed_lockout_steps=20,
    ).with_env_defaults()
    ctrl = CAGEController(cfg, dist)
    ctrl.reset_episode(np.asarray([0.0]), np.asarray([5.0]), np.asarray([[0.0], [5.0]]))
    ctrl.current_subgoal = np.asarray([5.0])
    ctrl.current_subgoal_index = 1
    ctrl.state = CAGEState.COMMITTING
    ctrl._last_step = 3
    ctrl._last_final_goal = np.asarray([5.0])

    ctrl.update_after_step(np.asarray([0.0]), np.asarray([0.0]), np.asarray([5.0]))

    assert ctrl.committed_stale_count == 1
    assert ctrl.committed_target_progresses[-1] == 0.0
    assert ctrl._committed_candidate_allowed(np.asarray([0.0])) is False
