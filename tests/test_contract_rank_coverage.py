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


def test_contract_rank_episode_summary_reports_coverage_and_selection_counts():
    cfg = CAGEConfig(
        use_cage=True,
        contract_rank=True,
        min_commit_steps=0,
        contract_min_commit_steps=0,
        min_subgoal_dist=1.0,
        max_subgoal_dist=100.0,
        final_phase_dist=0.1,
        drift_threshold=1000.0,
    ).with_env_defaults()
    ctrl = CAGEController(cfg, dist)
    path = np.asarray([[0.0], [5.0], [10.0]])
    original = np.asarray([0.0])
    ctrl.reset_episode(np.asarray([0.0]), np.asarray([20.0]), path)

    ctrl.select_subgoal(
        np.asarray([0.0]),
        np.asarray([20.0]),
        path,
        original,
        0,
        info={"original_subgoal": original, "original_subgoal_index": 0},
    )
    summary = ctrl.finish_episode("antmaze-giant-navigate-v0", 1, 42, 0, {"episode.success": 0.0})

    assert summary["cage_contract_rank"] is True
    assert summary["mean_contract_candidate_coverage"] is not None
    assert summary["mean_contract_candidate_coverage"] >= cfg.contract_rank_min_candidate_coverage
    assert (
        summary["contract_rank_choose_gas_count"]
        + summary["contract_rank_choose_cage_count"]
        + summary["contract_rank_choose_committed_count"]
    ) == 1
