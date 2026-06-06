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


def intervene_cfg(**kwargs):
    base = dict(
        use_cage=True,
        contract_intervene=True,
        min_commit_steps=0,
        contract_min_commit_steps=0,
        min_subgoal_dist=1.0,
        max_subgoal_dist=100.0,
        final_phase_dist=0.1,
        drift_threshold=1000.0,
        contract_rank_prefer_gas_margin=0.05,
        contract_intervention_margin=0.05,
        contract_intervention_gas_risk_threshold=0.60,
        contract_intervention_cost=0.0,
    )
    base.update(kwargs)
    return CAGEConfig(**base).with_env_defaults()


def test_intervention_preserves_gas_when_gas_contract_is_not_risky():
    ctrl = CAGEController(intervene_cfg(), dist)
    path = np.asarray([[0.0], [1.0], [2.0]])
    original = np.asarray([1.0])
    ctrl.reset_episode(np.asarray([0.0]), np.asarray([2.0]), path)

    selected, idx, _, should_replan, trace = ctrl.select_subgoal(
        np.asarray([0.0]),
        np.asarray([2.0]),
        path,
        original,
        0,
        info={"original_subgoal": original, "original_subgoal_index": 1},
    )

    assert not should_replan
    assert np.allclose(selected, original)
    assert idx == 1
    assert trace["intervention_allowed"] is False
    assert trace["intervention_reason"] == "gas_contract_not_risky"


def test_intervention_replaces_gas_only_when_gas_is_risky_and_alternative_clears_margin():
    ctrl = CAGEController(intervene_cfg(contract_intervention_gas_risk_threshold=0.50), dist)
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
    assert not np.allclose(selected, original)
    assert idx in {1, 2}
    assert trace["intervention_allowed"] is True
    assert trace["intervention_selected_source"] != "gas"
    assert ctrl.contract_intervention_count == 1
