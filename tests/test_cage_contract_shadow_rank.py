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


def cfg(**kwargs):
    base = dict(
        use_cage=True,
        contract_shadow_rank=True,
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


def test_shadow_rank_returns_original_gas_target_even_when_ranker_would_override():
    ctrl = CAGEController(cfg(), dist)
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
    assert np.allclose(selected, original)
    assert idx == 0
    assert trace["shadow_would_override_gas"] is True
    assert trace["original_subgoal_used"] is True
    assert trace["selected_target_mode"] == "original_target"


def test_shadow_episode_summary_records_success_conditional_override_rate():
    ctrl = CAGEController(cfg(), dist)
    path = np.asarray([[0.0], [1.0], [2.0]])
    original = np.asarray([30.0])
    ctrl.reset_episode(np.asarray([0.0]), np.asarray([2.0]), path)
    ctrl.select_subgoal(
        np.asarray([0.0]),
        np.asarray([2.0]),
        path,
        original,
        0,
        info={"original_subgoal": original, "original_subgoal_index": 0},
    )

    summary = ctrl.finish_episode(
        "toy", None, 0, 0, env_info={"episode.success": 1.0}, timeout=False
    )

    assert summary["cage_contract_shadow_rank"] is True
    assert summary["shadow_override_rate"] > 0.0
    assert summary["shadow_override_on_success_rate"] == summary["shadow_override_rate"]
