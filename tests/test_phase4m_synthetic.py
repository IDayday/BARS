import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase3.train_gcbc import combine_external_loss_weights, edge_loss_weight_values
from phase3e.phase4m_planner_relevant_weighting import (
    build_planner_relevant_loss_weights,
    grouped_direct_repair_metrics,
    planner_edge_usage_counts,
    summarize_direct_repair_scores,
)


def _edges():
    return pd.DataFrame(
        {
            "edge_id": [0, 1, 2, 3],
            "src": [0, 1, 2, 3],
            "dst": [1, 2, 3, 4],
            "is_repair_edge": [False, True, True, True],
            "num_unique_starts": [20, 2, 6, 10],
            "num_segments": [50, 5, 12, 20],
            "num_unique_episodes": [8, 2, 3, 5],
            "edge_bottleneck_score": [0.1, 0.9, 0.4, 0.2],
            "median_h": [2, 8, 3, 4],
        }
    )


def test_planner_usage_counts_filters_repaired_reachable_paths():
    paths = pd.DataFrame(
        [
            {
                "method": "chosen",
                "graph_variant": "repaired",
                "reachable": True,
                "path_edge_ids": "0 1 1 2",
            },
            {"method": "chosen", "graph_variant": "base", "reachable": True, "path_edge_ids": "3"},
            {"method": "other", "graph_variant": "repaired", "reachable": True, "path_edge_ids": "3"},
            {"method": "chosen", "graph_variant": "repaired", "reachable": False, "path_edge_ids": "2"},
        ]
    )
    usage = planner_edge_usage_counts(paths, planner_method="chosen", graph_variant="repaired").set_index("edge_id")
    assert usage.loc[1, "planner_usage_count"] == 2
    assert usage.loc[2, "planner_usage_count"] == 1
    assert 3 not in usage.index


def test_planner_relevant_weights_lift_used_repair_edges_and_clip():
    paths = pd.DataFrame(
        [
            {
                "method": "calibrated_compat_threshold",
                "graph_variant": "repaired",
                "reachable": True,
                "path_edge_ids": "0 1 1",
            }
        ]
    )
    weights = build_planner_relevant_loss_weights(
        _edges(),
        paths,
        planner_relevance_strength=2.0,
        hard_repair_strength=1.0,
        min_weight=0.7,
        max_weight=1.5,
    ).set_index("edge_id")
    assert weights.loc[1, "planner_relevance_score"] > 0.0
    assert weights.loc[1, "loss_weight"] > weights.loc[3, "loss_weight"]
    assert weights["loss_weight"].max() <= 1.5
    assert weights["loss_weight"].min() >= 0.7
    assert weights.loc[0, "planner_relevance_score"] == 0.0


def test_external_loss_weights_can_replace_builtin_weights():
    edges = _edges()
    base = edge_loss_weight_values(edges, mode="support_bottleneck", strength=0.3)
    external = pd.DataFrame({"edge_id": [0, 1, 2, 3], "loss_weight": [1.0, 4.0, 1.0, 1.0]})
    combined = combine_external_loss_weights(
        base,
        edges,
        external_loss_weights=external,
        combine="replace",
        min_weight=0.5,
        max_weight=3.0,
    ).set_index("edge_id")
    assert combined.loc[1, "loss_weight"] > combined.loc[0, "loss_weight"]
    assert combined["loss_weight"].max() <= 3.0


def test_direct_repair_group_metrics_separate_planner_used_edges():
    edges = _edges()
    paths = pd.DataFrame(
        [
            {
                "method": "calibrated_compat_threshold",
                "graph_variant": "repaired",
                "reachable": True,
                "path_edge_ids": "1",
            }
        ]
    )
    direct = pd.DataFrame(
        {
            "edge_id": [1, 2],
            "edge_action_mse": [0.1, 0.3],
            "direct_edge_policy_support_score": [0.9, 0.5],
            "num_policy_eval_samples": [10, 10],
        }
    )
    scored = summarize_direct_repair_scores(direct, edges, paths)
    grouped = grouped_direct_repair_metrics(scored)
    used = grouped[
        (grouped["group_type"] == "planner_usage_group") & (grouped["group_value"] == "planner_used")
    ].iloc[0]
    unused = grouped[
        (grouped["group_type"] == "planner_usage_group") & (grouped["group_value"] == "not_planner_used")
    ].iloc[0]
    assert used["mean_edge_action_mse"] == 0.1
    assert unused["mean_edge_action_mse"] == 0.3
