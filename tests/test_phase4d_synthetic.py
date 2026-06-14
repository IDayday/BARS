import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase3e.compatibility_aware_planning import (
    CompatibilityPlannerConfig,
    compute_pair_compatibility_from_segments,
    evaluate_compatibility_planning_methods,
    make_compatibility_edge_table,
    summarize_compatibility_planning,
)


def _option_edges():
    return pd.DataFrame(
        [
            {
                "edge_id": 0,
                "src": 1,
                "dst": 2,
                "median_h": 1.0,
                "cost": 1.0,
                "edge_bottleneck_score": 0.1,
            },
            {
                "edge_id": 1,
                "src": 2,
                "dst": 4,
                "median_h": 1.0,
                "cost": 1.0,
                "edge_bottleneck_score": 0.1,
            },
            {
                "edge_id": 2,
                "src": 1,
                "dst": 3,
                "median_h": 2.0,
                "cost": 2.0,
                "edge_bottleneck_score": 0.1,
            },
            {
                "edge_id": 3,
                "src": 3,
                "dst": 4,
                "median_h": 2.0,
                "cost": 2.0,
                "edge_bottleneck_score": 0.1,
            },
        ]
    )


def _certification():
    return pd.DataFrame(
        [
            {
                "edge_id": edge_id,
                "edge_proxy_score": 0.8,
                "heldout_support_lcb": 0.5,
                "edge_policy_support_score": 0.8,
                "edge_ood_score": 0.1,
                "certified_offline_binary": True,
                "certified_offline_binary_original": True,
            }
            for edge_id in range(4)
        ]
    )


def _pair_df():
    return pd.DataFrame(
        [
            {
                "edge_id_first": 0,
                "edge_id_second": 1,
                "termination_bridge_coverage": 0.0,
                "strict_compatible": False,
                "bridge_matches_per_first_segment": 0.0,
                "num_bridge_segments": 0,
                "num_bridge_episodes": 0,
            },
            {
                "edge_id_first": 2,
                "edge_id_second": 3,
                "termination_bridge_coverage": 1.0,
                "strict_compatible": True,
                "bridge_matches_per_first_segment": 1.0,
                "num_bridge_segments": 2,
                "num_bridge_episodes": 1,
            },
        ]
    )


def _query():
    return pd.DataFrame([{"query_id": 0, "start_cluster": 1, "goal_cluster": 4}])


def test_pair_penalty_prefers_longer_compatible_path():
    edge_table = make_compatibility_edge_table(_option_edges(), _certification())
    paths, _ = evaluate_compatibility_planning_methods(
        edge_table,
        _pair_df(),
        _query(),
        methods=["support_shortest_path", "compat_penalized"],
        config=CompatibilityPlannerConfig(pair_weight=10.0, min_pair_coverage=0.05),
    )
    by_method = paths.set_index("method")
    assert by_method.loc["support_shortest_path", "path_edge_ids"] == "0 1"
    assert by_method.loc["compat_penalized", "path_edge_ids"] == "2 3"
    assert by_method.loc["compat_penalized", "min_pair_termination_bridge_coverage"] == 1.0


def test_pair_threshold_blocks_incompatible_shortcut():
    edge_table = make_compatibility_edge_table(_option_edges(), _certification())
    paths, graphs = evaluate_compatibility_planning_methods(
        edge_table,
        _pair_df(),
        _query(),
        methods=["compat_threshold"],
        config=CompatibilityPlannerConfig(pair_weight=10.0, min_pair_coverage=0.05),
    )
    summary = summarize_compatibility_planning(paths, graphs)
    assert paths.iloc[0]["path_edge_ids"] == "2 3"
    assert summary.iloc[0]["path_coverage"] == 1.0
    assert summary.iloc[0]["mean_pair_incompatible_fraction"] == 0.0


def test_pair_compatibility_recomputed_from_edge_segments():
    option_edges = pd.DataFrame(
        [
            {"edge_id": 0, "src": 1, "dst": 2, "median_h": 1.0, "cost": 1.0},
            {"edge_id": 1, "src": 2, "dst": 3, "median_h": 1.0, "cost": 1.0},
        ]
    )
    edge_segments = {
        "edge_id": np.array([0, 0, 1], dtype=np.int64),
        "ep_id": np.array([0, 0, 0], dtype=np.int64),
        "global_i": np.array([0, 6, 3], dtype=np.int64),
        "global_j": np.array([2, 8, 5], dtype=np.int64),
        "h": np.array([2, 2, 2], dtype=np.int64),
    }
    summary, pair_df = compute_pair_compatibility_from_segments(option_edges, edge_segments, H_intra=2)
    pair = pair_df.iloc[0]
    assert summary.iloc[0]["num_adjacent_edge_pairs"] == 1
    assert pair["num_first_terminations"] == 2
    assert pair["num_first_terminations_with_bridge"] == 1
    assert pair["termination_bridge_coverage"] == 0.5
    assert 0.0 <= pair["termination_bridge_coverage"] <= 1.0

