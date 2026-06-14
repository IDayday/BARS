import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase3e.risk_aware_planning import evaluate_planning_methods, load_edge_table, summarize_planning_results
from phase3e.risk_aware_sweep import (
    RiskSweepConfig,
    make_sweep_configs,
    run_planner_sweep,
    select_recommended_config,
)


def _edges():
    return pd.DataFrame(
        [
            {"edge_id": 0, "src": 0, "dst": 1, "median_h": 1.0, "cost": 1.0, "edge_bottleneck_score": 0.1},
            {"edge_id": 1, "src": 1, "dst": 3, "median_h": 1.0, "cost": 1.0, "edge_bottleneck_score": 0.1},
            {"edge_id": 2, "src": 0, "dst": 2, "median_h": 2.0, "cost": 2.0, "edge_bottleneck_score": 0.8},
            {"edge_id": 3, "src": 2, "dst": 3, "median_h": 2.0, "cost": 2.0, "edge_bottleneck_score": 0.8},
        ]
    )


def _cert():
    rows = []
    for edge_id, proxy, lcb, certified in [
        (0, 0.05, 0.0, False),
        (1, 0.05, 0.0, False),
        (2, 0.90, 0.5, True),
        (3, 0.90, 0.5, True),
    ]:
        rows.append(
            {
                "edge_id": edge_id,
                "heldout_support_lcb": lcb,
                "edge_policy_support_score": proxy,
                "edge_ood_score": 1.0 - proxy,
                "outgoing_mean_termination_bridge_coverage": proxy,
                "outgoing_incompatible_fraction": 1.0 - proxy,
                "edge_proxy_score": proxy,
                "certified_offline_binary": certified,
            }
        )
    return pd.DataFrame(rows)


def test_make_sweep_configs_cross_product():
    configs = make_sweep_configs(
        planner_methods=["floor_proxy_penalized"],
        risk_weights=[0, 1],
        ood_weights=[0],
        incompat_weights=[0],
        uncertified_weights=[0, 1],
        proxy_floors=[0, 0.1],
        heldout_support_lcb_floors=[0],
    )
    assert len(configs) == 8
    assert {c.planner_method for c in configs} == {"floor_proxy_penalized"}


def test_sweep_finds_proxy_cost_tradeoff():
    edge_table = load_edge_table(_edges(), _cert())
    queries = pd.DataFrame([{"query_id": 0, "start_cluster": 0, "goal_cluster": 3}])
    configs = [
        RiskSweepConfig("floor_proxy_penalized", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        RiskSweepConfig("floor_proxy_penalized", 20.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ]
    sweep = run_planner_sweep(edge_table, queries, configs)
    low_risk = sweep.sort_values("risk_weight").iloc[0]
    high_risk = sweep.sort_values("risk_weight").iloc[-1]
    assert low_risk["path_coverage"] == 1.0
    assert high_risk["path_coverage"] == 1.0
    assert high_risk["mean_min_edge_proxy_score"] > low_risk["mean_min_edge_proxy_score"]
    assert high_risk["mean_base_path_cost"] > low_risk["mean_base_path_cost"]
    assert bool(sweep["is_pareto"].any())


def test_proxy_floor_can_reduce_coverage_and_recommendation_is_eligible():
    edge_table = load_edge_table(_edges(), _cert())
    queries = pd.DataFrame(
        [
            {"query_id": 0, "start_cluster": 0, "goal_cluster": 3},
            {"query_id": 1, "start_cluster": 0, "goal_cluster": 1},
        ]
    )
    baseline_paths, baseline_graphs = evaluate_planning_methods(
        edge_table,
        queries,
        methods=["support_shortest_path"],
    )
    baseline_summary = summarize_planning_results(baseline_paths, baseline_graphs)
    configs = [
        RiskSweepConfig("floor_proxy_penalized", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        RiskSweepConfig("floor_proxy_penalized", 0.0, 0.0, 0.0, 0.0, 0.25, 0.01),
    ]
    sweep = run_planner_sweep(edge_table, queries, configs)
    filtered = sweep[sweep["min_proxy_score"] == 0.25].iloc[0]
    assert filtered["path_coverage"] == 0.5
    recommended = select_recommended_config(sweep, baseline_summary, min_coverage_ratio=0.5)
    assert recommended
    assert recommended["path_coverage"] >= 0.5
