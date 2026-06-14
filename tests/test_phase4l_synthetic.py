import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase3e.phase4l_repair_group_diagnostics import (
    add_edge_groups,
    parse_path_edge_ids,
    path_usage_counts,
    rank_group_findings,
    summarize_group_deltas,
    summarize_method_deltas,
)


def test_parse_path_edge_ids_handles_blank_and_space_lists():
    assert parse_path_edge_ids("10 11 12") == [10, 11, 12]
    assert parse_path_edge_ids("10, 12 bad") == [10, 12]
    assert parse_path_edge_ids(float("nan")) == []


def test_path_usage_counts_filters_method_and_reachable():
    paths = pd.DataFrame(
        [
            {"method": "chosen", "reachable": True, "path_edge_ids": "1 2 2"},
            {"method": "chosen", "reachable": False, "path_edge_ids": "3"},
            {"method": "other", "reachable": True, "path_edge_ids": "4"},
        ]
    )
    usage = path_usage_counts(paths, planner_method="chosen").set_index("edge_id")
    assert usage.loc[1, "path_usage_count"] == 1
    assert usage.loc[2, "path_usage_count"] == 2
    assert 3 not in usage.index
    assert 4 not in usage.index


def test_add_edge_groups_splits_support_bottleneck_horizon_and_usage():
    edges = pd.DataFrame(
        {
            "edge_id": [1, 2, 3, 4],
            "num_unique_starts": [1, 2, 10, 12],
            "edge_bottleneck_score": [0.1, 0.2, 0.9, 1.0],
            "median_h": [1, 2, 5, 6],
            "outgoing_mean_termination_bridge_coverage": [0.0, 0.2, 0.8, 1.0],
            "incoming_mean_termination_bridge_coverage": [0.0, 0.2, 0.8, 1.0],
            "used_by_planner": [True, False, False, True],
        }
    )
    grouped = add_edge_groups(edges)
    assert grouped.loc[0, "support_group"] == "low_support"
    assert grouped.loc[3, "support_group"] == "high_support"
    assert grouped.loc[3, "bottleneck_group"] == "high_bottleneck"
    assert grouped.loc[0, "horizon_group"] == "short_horizon"
    assert grouped.loc[0, "planner_usage_group"] == "planner_used"


def test_group_summary_identifies_low_support_gain():
    deltas = pd.DataFrame(
        [
            {
                "dataset": "dummy-v0",
                "phase2_run": "core",
                "method": "weighted",
                "edge_id": 1,
                "support_group": "low_support",
                "baseline_edge_action_mse": 1.0,
                "candidate_edge_action_mse": 0.8,
                "policy_support_delta": 0.1,
                "candidate_path_usage_count": 2,
                "num_policy_eval_samples": 10,
                "improved": True,
            },
            {
                "dataset": "dummy-v0",
                "phase2_run": "core",
                "method": "weighted",
                "edge_id": 2,
                "support_group": "high_support",
                "baseline_edge_action_mse": 1.0,
                "candidate_edge_action_mse": 1.1,
                "policy_support_delta": -0.1,
                "candidate_path_usage_count": 0,
                "num_policy_eval_samples": 10,
                "improved": False,
            },
        ]
    )
    method_summary = summarize_method_deltas(deltas)
    group_summary = summarize_group_deltas(deltas)
    low_support = group_summary[
        (group_summary["group_type"] == "support_group") & (group_summary["group_value"] == "low_support")
    ].iloc[0]
    assert method_summary.iloc[0]["mean_edge_action_mse_delta"] < 0
    assert low_support["mean_edge_action_mse_delta"] < 0
    assert low_support["planner_usage_rate"] == 1.0
    ranked = rank_group_findings(group_summary)
    assert ranked.iloc[0]["group_value"] == "low_support"
