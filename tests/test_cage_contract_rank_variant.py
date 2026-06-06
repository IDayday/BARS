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


def rank_cfg(**kwargs):
    base = dict(
        use_cage=True,
        contract_rank=True,
        min_commit_steps=0,
        contract_min_commit_steps=0,
        min_subgoal_dist=1.0,
        max_subgoal_dist=100.0,
        final_phase_dist=0.1,
        drift_threshold=1000.0,
        contract_rank_prefer_gas_margin=0.05,
    )
    base.update(kwargs)
    return CAGEConfig(**base).with_env_defaults()


def test_contract_rank_keeps_gas_when_cage_candidate_does_not_clear_margin():
    ctrl = CAGEController(rank_cfg(), dist)
    path = np.asarray([[0.0], [20.0], [40.0]])
    original = np.asarray([0.0])
    ctrl.reset_episode(np.asarray([0.0]), np.asarray([50.0]), path)

    selected, idx, _, should_replan, trace = ctrl.select_subgoal(
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
    assert trace["contract_rank_enabled"] is True
    assert trace["contract_selected_source"] == "gas"
    assert ctrl.contract_rank_choose_gas_count == 1


def test_contract_rank_can_replace_gas_with_better_contract_candidate():
    ctrl = CAGEController(rank_cfg(), dist)
    path = np.asarray([[0.0], [1.0], [2.0]])
    original = np.asarray([30.0])
    ctrl.reset_episode(np.asarray([0.0]), np.asarray([2.0]), path)

    selected, idx, _, should_replan, trace = ctrl.select_subgoal(
        np.asarray([0.0]),
        np.asarray([2.0]),
        path,
        original,
        0,
        info={"original_subgoal": original, "original_subgoal_index": 0},
    )

    assert not should_replan
    assert idx in {1, 2}
    assert not np.allclose(selected, original)
    assert trace["contract_selected_source"] != "gas"
    assert trace["contract_best_non_gas_score"] > trace["contract_gas_score"]
    assert ctrl.contract_rank_choose_cage_count == 1
