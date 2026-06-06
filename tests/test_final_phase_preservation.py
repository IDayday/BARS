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


def test_contract_intervention_preserves_final_phase_gas_target():
    cfg = CAGEConfig(
        use_cage=True,
        contract_intervene=True,
        min_commit_steps=0,
        contract_min_commit_steps=0,
        min_subgoal_dist=1.0,
        max_subgoal_dist=100.0,
        final_phase_dist=10.0,
        contract_intervention_preserve_final_phase=True,
        contract_intervention_cost=0.0,
    ).with_env_defaults()
    ctrl = CAGEController(cfg, dist)
    path = np.asarray([[0.0], [1.0], [2.0]])
    original = np.asarray([2.0])
    ctrl.reset_episode(np.asarray([1.5]), np.asarray([2.0]), path)

    selected, idx, _, should_replan, trace = ctrl.select_subgoal(
        np.asarray([1.5]),
        np.asarray([2.0]),
        path,
        original,
        0,
        info={
            "original_subgoal": original,
            "original_subgoal_index": -1,
            "planner_final_goal_on": True,
        },
    )

    assert not should_replan
    assert np.allclose(selected, original)
    assert idx == -1
    assert trace["final_phase_preserved"] is True
    assert ctrl.final_phase_preserved_count == 1
