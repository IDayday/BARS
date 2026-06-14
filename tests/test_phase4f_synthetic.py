import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase3e.compatibility_aware_planning import CompatibilityPlannerConfig
from phase3e.edge_risk_calibration import EdgeRiskCalibrationConfig
from phase3e.repair_edge_certification import (
    RepairCertificationConfig,
    build_repair_edge_certification,
    evaluate_repair_certified_planning,
)


def _base_cert():
    return pd.DataFrame(
        [
            {
                "edge_id": 0,
                "src": 1,
                "dst": 2,
                "heldout_support_lcb": 0.8,
                "edge_policy_support_score": 0.8,
                "edge_ood_score": 0.1,
                "outgoing_mean_termination_bridge_coverage": 0.5,
                "incoming_mean_termination_bridge_coverage": 0.5,
                "outgoing_incompatible_fraction": 0.0,
                "incoming_incompatible_fraction": 0.0,
                "edge_proxy_score": 0.7,
                "certified_offline_binary": True,
                "certified_offline_binary_original": True,
                "num_unique_starts": 10,
                "num_unique_episodes": 3,
                "median_h": 2.0,
                "num_segments": 20,
                "num_episodes": 3,
            },
            {
                "edge_id": 1,
                "src": 2,
                "dst": 4,
                "heldout_support_lcb": 0.7,
                "edge_policy_support_score": 0.7,
                "edge_ood_score": 0.2,
                "outgoing_mean_termination_bridge_coverage": 0.5,
                "incoming_mean_termination_bridge_coverage": 0.5,
                "outgoing_incompatible_fraction": 0.0,
                "incoming_incompatible_fraction": 0.0,
                "edge_proxy_score": 0.6,
                "certified_offline_binary": True,
                "certified_offline_binary_original": True,
                "num_unique_starts": 10,
                "num_unique_episodes": 3,
                "median_h": 2.0,
                "num_segments": 20,
                "num_episodes": 3,
            },
        ]
    )


def _augmented_edges():
    return pd.DataFrame(
        [
            {"edge_id": 0, "src": 1, "dst": 2, "num_segments": 20, "num_episodes": 3, "num_unique_starts": 10, "num_unique_episodes": 3, "median_h": 2.0, "cost": 2.0, "is_repair_edge": False},
            {"edge_id": 1, "src": 2, "dst": 4, "num_segments": 20, "num_episodes": 3, "num_unique_starts": 10, "num_unique_episodes": 3, "median_h": 2.0, "cost": 2.0, "is_repair_edge": False},
            {"edge_id": 2, "bank_edge_id": 12, "src": 2, "dst": 3, "num_segments": 30, "num_episodes": 4, "num_unique_starts": 14, "num_unique_episodes": 4, "median_h": 2.0, "cost": 2.0, "is_repair_edge": True, "repair_score": 3.0},
            {"edge_id": 3, "bank_edge_id": 13, "src": 3, "dst": 4, "num_segments": 30, "num_episodes": 4, "num_unique_starts": 14, "num_unique_episodes": 4, "median_h": 2.0, "cost": 2.0, "is_repair_edge": True, "repair_score": 3.0},
            {"edge_id": 4, "bank_edge_id": 14, "src": 5, "dst": 6, "num_segments": 3, "num_episodes": 1, "num_unique_starts": 1, "num_unique_episodes": 1, "median_h": 5.0, "cost": 5.0, "is_repair_edge": True, "repair_score": 0.1},
        ]
    )


def _pair_df():
    return pd.DataFrame(
        [
            {"edge_id_first": 0, "edge_id_second": 1, "termination_bridge_coverage": 0.0, "strict_compatible": False},
            {"edge_id_first": 0, "edge_id_second": 2, "termination_bridge_coverage": 1.0, "strict_compatible": True},
            {"edge_id_first": 2, "edge_id_second": 3, "termination_bridge_coverage": 1.0, "strict_compatible": True},
            {"edge_id_first": 4, "edge_id_second": 1, "termination_bridge_coverage": 0.0, "strict_compatible": False},
        ]
    )


def test_repair_certification_preserves_base_and_marks_source():
    calibrated, planner, diagnostics = build_repair_edge_certification(
        _augmented_edges(),
        _base_cert(),
        _pair_df(),
        RepairCertificationConfig(),
        EdgeRiskCalibrationConfig(certification_threshold=0.1, min_support_lcb=0.01, min_compatibility=0.0),
    )
    by_id = planner.set_index("edge_id")
    assert by_id.loc[0, "certification_source"] == "phase4c_base"
    assert by_id.loc[2, "certification_source"] == "repair_transfer_proxy"
    assert bool(by_id.loc[2, "is_repair_edge"])
    assert diagnostics["num_repair_certification_edges"] == 3
    assert "calibrated_edge_reliability_score" in calibrated.columns


def test_compatible_repair_edge_scores_above_weak_repair_edge():
    _, planner, _ = build_repair_edge_certification(
        _augmented_edges(),
        _base_cert(),
        _pair_df(),
        RepairCertificationConfig(),
        EdgeRiskCalibrationConfig(certification_threshold=0.1, min_support_lcb=0.01, min_compatibility=0.0),
    )
    scores = planner.set_index("edge_id")["edge_proxy_score"]
    assert scores.loc[2] > scores.loc[4]
    assert scores.loc[3] > scores.loc[4]


def test_planning_reports_repair_edge_usage():
    _, planner, _ = build_repair_edge_certification(
        _augmented_edges(),
        _base_cert(),
        _pair_df(),
        RepairCertificationConfig(),
        EdgeRiskCalibrationConfig(certification_threshold=0.1, min_support_lcb=0.01, min_compatibility=0.0),
    )
    paths, summary = evaluate_repair_certified_planning(
        _augmented_edges(),
        _pair_df(),
        pd.DataFrame([{"query_id": 0, "start_cluster": 1, "goal_cluster": 4}]),
        planner,
        CompatibilityPlannerConfig(min_pair_coverage=0.05, pair_weight=10.0),
        methods=["compat_threshold"],
    )
    row = paths.iloc[0]
    assert row["path_edge_ids"] == "0 2 3"
    assert row["num_repair_edges"] == 2
    assert row["repair_edge_fraction"] == 2 / 3
    assert summary.iloc[0]["mean_repair_edge_fraction"] == 2 / 3

