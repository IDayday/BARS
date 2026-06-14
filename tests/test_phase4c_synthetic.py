import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase3e.edge_risk_calibration import (
    EdgeRiskCalibrationConfig,
    calibrate_edge_risk,
    calibration_bins,
    make_planner_certification,
    score_diagnostics,
)


def _certification():
    return pd.DataFrame(
        [
            {
                "edge_id": 0,
                "edge_proxy_score": 0.8,
                "certified_offline_binary": True,
                "heldout_support_lcb": 0.8,
                "heldout_support_binary": True,
                "heldout_support_rate": 0.9,
                "edge_policy_support_score": 0.9,
                "edge_ood_score": 0.1,
                "outgoing_mean_termination_bridge_coverage": 0.8,
                "incoming_mean_termination_bridge_coverage": 0.8,
                "outgoing_incompatible_fraction": 0.1,
                "incoming_incompatible_fraction": 0.1,
                "num_unique_starts": 20,
                "num_unique_episodes": 5,
            },
            {
                "edge_id": 1,
                "edge_proxy_score": 0.8,
                "certified_offline_binary": True,
                "heldout_support_lcb": 0.8,
                "heldout_support_binary": True,
                "heldout_support_rate": 0.9,
                "edge_policy_support_score": 0.9,
                "edge_ood_score": 0.1,
                "outgoing_mean_termination_bridge_coverage": 0.0,
                "incoming_mean_termination_bridge_coverage": 0.0,
                "outgoing_incompatible_fraction": 1.0,
                "incoming_incompatible_fraction": 1.0,
                "num_unique_starts": 20,
                "num_unique_episodes": 5,
            },
            {
                "edge_id": 2,
                "edge_proxy_score": 0.1,
                "certified_offline_binary": False,
                "heldout_support_lcb": 0.0,
                "heldout_support_binary": False,
                "heldout_support_rate": 0.0,
                "edge_policy_support_score": 0.2,
                "edge_ood_score": 0.9,
                "outgoing_mean_termination_bridge_coverage": 0.1,
                "incoming_mean_termination_bridge_coverage": 0.1,
                "outgoing_incompatible_fraction": 0.9,
                "incoming_incompatible_fraction": 0.9,
                "num_unique_starts": 1,
                "num_unique_episodes": 1,
            },
        ]
    )


def test_calibrated_score_penalizes_weak_component():
    calibrated = calibrate_edge_risk(
        _certification(),
        EdgeRiskCalibrationConfig(certification_threshold=0.25, min_support_lcb=0.01),
    )
    scores = calibrated.set_index("edge_id")["calibrated_edge_reliability_score"]
    assert scores.loc[0] > scores.loc[1]
    assert scores.loc[1] > scores.loc[2]
    assert calibrated.set_index("edge_id").loc[0, "calibrated_certified_binary"]
    assert not calibrated.set_index("edge_id").loc[2, "calibrated_certified_binary"]


def test_planner_certification_replaces_proxy_inputs():
    calibrated = calibrate_edge_risk(_certification())
    planner = make_planner_certification(calibrated)
    assert "edge_proxy_score_original" in planner.columns
    assert planner["edge_proxy_score"].equals(planner["calibrated_edge_reliability_score"])
    assert planner["edge_ood_score"].equals(planner["calibrated_edge_ood_score"])
    assert planner["certified_offline_binary"].equals(planner["calibrated_certified_binary"])


def test_diagnostics_and_bins_are_defined():
    calibrated = calibrate_edge_risk(_certification())
    diagnostics = score_diagnostics(calibrated)
    bins = calibration_bins(calibrated, bins=3)
    assert diagnostics["num_edges"] == 3
    assert "calibrated_edge_reliability_score_brier_heldout_support_binary" in diagnostics
    assert not bins.empty
    assert set(["num_edges", "mean_score", "mean_label", "abs_gap"]).issubset(bins.columns)
