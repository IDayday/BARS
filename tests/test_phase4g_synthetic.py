import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase3e.compatibility_aware_planning import CompatibilityPlannerConfig
from phase3e.direct_repair_policy import (
    apply_direct_policy_scores,
    direct_vs_transfer_diagnostics,
    evaluate_direct_repair_policy_planning,
    repair_bank_segment_indices,
)
from phase3e.edge_risk_calibration import EdgeRiskCalibrationConfig


def _transfer_certification():
    return pd.DataFrame(
        [
            {
                "edge_id": 0,
                "src": 1,
                "dst": 2,
                "heldout_support_lcb": 0.8,
                "edge_policy_support_score": 0.8,
                "edge_ood_score": 0.1,
                "outgoing_mean_termination_bridge_coverage": 1.0,
                "incoming_mean_termination_bridge_coverage": 1.0,
                "outgoing_incompatible_fraction": 0.0,
                "incoming_incompatible_fraction": 0.0,
                "edge_proxy_score": 0.8,
                "certified_offline_binary": True,
                "calibrated_certified_binary": True,
                "calibrated_edge_reliability_score": 0.8,
                "is_repair_edge": False,
                "num_unique_starts": 10,
                "num_unique_episodes": 3,
            },
            {
                "edge_id": 1,
                "src": 2,
                "dst": 3,
                "heldout_support_lcb": 0.6,
                "edge_policy_support_score": 0.2,
                "edge_ood_score": 0.2,
                "outgoing_mean_termination_bridge_coverage": 1.0,
                "incoming_mean_termination_bridge_coverage": 1.0,
                "outgoing_incompatible_fraction": 0.0,
                "incoming_incompatible_fraction": 0.0,
                "edge_proxy_score": 0.3,
                "certified_offline_binary": True,
                "calibrated_certified_binary": True,
                "calibrated_edge_reliability_score": 0.3,
                "is_repair_edge": True,
                "num_unique_starts": 8,
                "num_unique_episodes": 3,
            },
            {
                "edge_id": 2,
                "src": 3,
                "dst": 4,
                "heldout_support_lcb": 0.6,
                "edge_policy_support_score": 0.2,
                "edge_ood_score": 0.2,
                "outgoing_mean_termination_bridge_coverage": 1.0,
                "incoming_mean_termination_bridge_coverage": 1.0,
                "outgoing_incompatible_fraction": 0.0,
                "incoming_incompatible_fraction": 0.0,
                "edge_proxy_score": 0.3,
                "certified_offline_binary": True,
                "calibrated_certified_binary": True,
                "calibrated_edge_reliability_score": 0.3,
                "is_repair_edge": True,
                "num_unique_starts": 8,
                "num_unique_episodes": 3,
            },
        ]
    )


def test_repair_bank_segment_indices_selects_requested_bank_edges():
    segments = {"edge_id": np.array([10, 11, 10, 12, 13], dtype=np.int64)}
    idx = repair_bank_segment_indices(segments, [10, 13])
    assert idx.tolist() == [0, 2, 4]


def test_apply_direct_policy_scores_replaces_repair_policy_component():
    transfer = _transfer_certification()
    direct = pd.DataFrame(
        [
            {"edge_id": 1, "edge_action_mse": 0.01, "edge_action_mse_ucb": 0.02, "direct_edge_policy_support_score": 0.9, "num_policy_eval_samples": 50},
            {"edge_id": 2, "edge_action_mse": 1.00, "edge_action_mse_ucb": 1.10, "direct_edge_policy_support_score": 0.01, "num_policy_eval_samples": 50},
        ]
    )
    calibrated, planner = apply_direct_policy_scores(
        transfer,
        direct,
        EdgeRiskCalibrationConfig(certification_threshold=0.1, min_support_lcb=0.01, min_compatibility=0.0),
    )
    by_id = calibrated.set_index("edge_id")
    assert by_id.loc[1, "edge_policy_support_score"] == 0.9
    assert by_id.loc[2, "edge_policy_support_score"] == 0.01
    assert by_id.loc[1, "calibrated_edge_reliability_score"] > by_id.loc[2, "calibrated_edge_reliability_score"]
    assert planner.set_index("edge_id").loc[1, "certification_source"] == "repair_direct_policy_mse"


def test_direct_vs_transfer_diagnostics_reports_score_delta():
    transfer = _transfer_certification()
    direct = pd.DataFrame(
        [
            {"edge_id": 1, "edge_action_mse": 0.01, "direct_edge_policy_support_score": 0.9, "num_policy_eval_samples": 10},
            {"edge_id": 2, "edge_action_mse": 0.10, "direct_edge_policy_support_score": 0.5, "num_policy_eval_samples": 10},
        ]
    )
    calibrated, _ = apply_direct_policy_scores(
        transfer,
        direct,
        EdgeRiskCalibrationConfig(certification_threshold=0.1, min_support_lcb=0.01, min_compatibility=0.0),
    )
    diag = direct_vs_transfer_diagnostics(transfer, calibrated, direct)
    assert diag["num_repair_edges"] == 2
    assert diag["num_direct_scored_edges"] == 2
    assert diag["mean_policy_score_delta_direct_minus_transfer"] > 0


def test_direct_repair_policy_planning_keeps_repair_usage_metrics():
    transfer = _transfer_certification()
    direct = pd.DataFrame(
        [
            {"edge_id": 1, "edge_action_mse": 0.01, "direct_edge_policy_support_score": 0.9, "num_policy_eval_samples": 10},
            {"edge_id": 2, "edge_action_mse": 0.01, "direct_edge_policy_support_score": 0.9, "num_policy_eval_samples": 10},
        ]
    )
    _, planner = apply_direct_policy_scores(
        transfer,
        direct,
        EdgeRiskCalibrationConfig(certification_threshold=0.1, min_support_lcb=0.01, min_compatibility=0.0),
    )
    augmented_edges = pd.DataFrame(
        [
            {"edge_id": 0, "src": 1, "dst": 2, "cost": 1.0, "median_h": 1.0, "is_repair_edge": False},
            {"edge_id": 1, "src": 2, "dst": 3, "cost": 1.0, "median_h": 1.0, "is_repair_edge": True},
            {"edge_id": 2, "src": 3, "dst": 4, "cost": 1.0, "median_h": 1.0, "is_repair_edge": True},
        ]
    )
    pair_df = pd.DataFrame(
        [
            {"edge_id_first": 0, "edge_id_second": 1, "termination_bridge_coverage": 1.0, "strict_compatible": True},
            {"edge_id_first": 1, "edge_id_second": 2, "termination_bridge_coverage": 1.0, "strict_compatible": True},
        ]
    )
    paths, summary = evaluate_direct_repair_policy_planning(
        augmented_edges,
        pair_df,
        pd.DataFrame([{"query_id": 0, "start_cluster": 1, "goal_cluster": 4}]),
        planner,
        CompatibilityPlannerConfig(min_pair_coverage=0.05),
        methods=["compat_threshold"],
    )
    assert paths.iloc[0]["path_edge_ids"] == "0 1 2"
    assert paths.iloc[0]["repair_edge_fraction"] == 2 / 3
    assert summary.iloc[0]["mean_repair_edge_fraction"] == 2 / 3

