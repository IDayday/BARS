import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase3e.risk_aware_planning import (
    RiskPlannerConfig,
    evaluate_planning_methods,
    load_edge_table,
    summarize_planning_results,
)


def _synthetic_edges():
    return pd.DataFrame(
        [
            {"edge_id": 0, "src": 0, "dst": 1, "median_h": 1.0, "cost": 1.0, "edge_bottleneck_score": 0.1},
            {"edge_id": 1, "src": 1, "dst": 3, "median_h": 1.0, "cost": 1.0, "edge_bottleneck_score": 0.1},
            {"edge_id": 2, "src": 0, "dst": 2, "median_h": 2.0, "cost": 2.0, "edge_bottleneck_score": 0.8},
            {"edge_id": 3, "src": 2, "dst": 3, "median_h": 2.0, "cost": 2.0, "edge_bottleneck_score": 0.8},
        ]
    )


def _synthetic_certification():
    return pd.DataFrame(
        [
            {
                "edge_id": 0,
                "heldout_support_lcb": 0.0,
                "edge_policy_support_score": 0.1,
                "edge_ood_score": 1.0,
                "outgoing_mean_termination_bridge_coverage": 0.0,
                "outgoing_incompatible_fraction": 1.0,
                "edge_proxy_score": 0.05,
                "certified_offline_binary": False,
            },
            {
                "edge_id": 1,
                "heldout_support_lcb": 0.0,
                "edge_policy_support_score": 0.1,
                "edge_ood_score": 1.0,
                "outgoing_mean_termination_bridge_coverage": 0.0,
                "outgoing_incompatible_fraction": 1.0,
                "edge_proxy_score": 0.05,
                "certified_offline_binary": False,
            },
            {
                "edge_id": 2,
                "heldout_support_lcb": 0.5,
                "edge_policy_support_score": 0.9,
                "edge_ood_score": 0.0,
                "outgoing_mean_termination_bridge_coverage": 0.8,
                "outgoing_incompatible_fraction": 0.0,
                "edge_proxy_score": 0.9,
                "certified_offline_binary": True,
            },
            {
                "edge_id": 3,
                "heldout_support_lcb": 0.5,
                "edge_policy_support_score": 0.9,
                "edge_ood_score": 0.0,
                "outgoing_mean_termination_bridge_coverage": 0.8,
                "outgoing_incompatible_fraction": 0.0,
                "edge_proxy_score": 0.9,
                "certified_offline_binary": True,
            },
        ]
    )


def test_proxy_penalized_avoids_low_proxy_shortcut():
    edge_table = load_edge_table(_synthetic_edges(), _synthetic_certification())
    queries = pd.DataFrame([{"query_id": 0, "start_cluster": 0, "goal_cluster": 3}])
    paths, graphs = evaluate_planning_methods(
        edge_table,
        queries,
        methods=["support_shortest_path", "proxy_penalized"],
        config=RiskPlannerConfig(risk_weight=20.0, ood_weight=0.0, incompat_weight=0.0, uncertified_weight=0.0),
    )
    support = paths[paths["method"] == "support_shortest_path"].iloc[0]
    penalized = paths[paths["method"] == "proxy_penalized"].iloc[0]
    assert support["path_edge_ids"] == "0 1"
    assert penalized["path_edge_ids"] == "2 3"
    assert penalized["min_edge_proxy_score"] > support["min_edge_proxy_score"]
    assert graphs.set_index("method").loc["proxy_penalized", "num_graph_edges"] == 4


def test_certified_only_can_reduce_coverage_without_adding_edges():
    edge_table = load_edge_table(_synthetic_edges(), _synthetic_certification())
    queries = pd.DataFrame(
        [
            {"query_id": 0, "start_cluster": 0, "goal_cluster": 3},
            {"query_id": 1, "start_cluster": 0, "goal_cluster": 1},
        ]
    )
    paths, graphs = evaluate_planning_methods(
        edge_table,
        queries,
        methods=["support_shortest_path", "certified_only"],
        config=RiskPlannerConfig(),
    )
    summary = summarize_planning_results(paths, graphs).set_index("method")
    assert summary.loc["support_shortest_path", "path_coverage"] == 1.0
    assert summary.loc["certified_only", "path_coverage"] == 0.5
    assert summary.loc["certified_only", "num_graph_edges"] == 2
    assert summary.loc["certified_only", "mean_uncertified_edge_fraction"] == 0.0


def test_path_risk_fractions_are_computed_on_reachable_paths():
    edge_table = load_edge_table(_synthetic_edges(), _synthetic_certification())
    queries = pd.DataFrame([{"query_id": 0, "start_cluster": 0, "goal_cluster": 3}])
    paths, graphs = evaluate_planning_methods(
        edge_table,
        queries,
        methods=["support_shortest_path"],
        config=RiskPlannerConfig(min_proxy_score=0.25, min_heldout_support_lcb=0.01),
    )
    row = paths.iloc[0]
    assert bool(row["reachable"]) is True
    assert row["uncertified_edge_fraction"] == 1.0
    assert row["low_proxy_edge_fraction"] == 1.0
    assert row["low_support_lcb_edge_fraction"] == 1.0
    summary = summarize_planning_results(paths, graphs).iloc[0]
    assert summary["mean_proxy_path_success_over_all"] == row["proxy_path_success"]
