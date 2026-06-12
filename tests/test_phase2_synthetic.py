import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

from phase2.compatibility import compute_edge_compatibility
from phase2.edge_dataset import build_option_edges, build_self_loop_summary
from phase2.evaluation import bottleneck_removal_ablation, evaluate_task_path_coverage
from phase2.node_selection import select_nodes
from phase2.option_graph import add_edge_costs, build_option_graph


def _pair_records(ep_id, t, h, global_i, global_j):
    return {
        "ep_id": np.asarray(ep_id, dtype=np.int64),
        "t": np.asarray(t, dtype=np.int64),
        "h": np.asarray(h, dtype=np.int64),
        "global_i": np.asarray(global_i, dtype=np.int64),
        "global_j": np.asarray(global_j, dtype=np.int64),
        "obs_i": np.empty((0, 1), dtype=np.float32),
        "obs_j": np.empty((0, 1), dtype=np.float32),
    }


def _selected(clusters, densities=None, bottlenecks=None):
    densities = densities or {c: 1.0 for c in clusters}
    bottlenecks = bottlenecks or {c: 0.0 for c in clusters}
    return pd.DataFrame(
        {
            "cluster": clusters,
            "selected": [True] * len(clusters),
            "selection_method": ["all"] * len(clusters),
            "density": [densities[c] for c in clusters],
            "bottleneck_score": [bottlenecks[c] for c in clusters],
            "rank_density": list(range(1, len(clusters) + 1)),
            "rank_bottleneck": list(range(1, len(clusters) + 1)),
        }
    )


def test_only_true_segments_become_option_edges():
    labels = np.asarray([0, 1, 2], dtype=np.int64)
    pairs = _pair_records([0, 0], [0, 1], [1, 1], [0, 1], [1, 2])
    edges, segments = build_option_edges(
        pairs,
        labels,
        _selected([0, 1, 2]),
        H=1,
        min_support=1,
        min_episodes=1,
    )
    assert set(zip(edges["src"], edges["dst"])) == {(0, 1), (1, 2)}
    assert segments["edge_id"].shape[0] == 2
    row = edges[(edges["src"] == 0) & (edges["dst"] == 1)].iloc[0]
    assert row["num_unique_starts"] == 1
    assert row["num_unique_terminations"] == 1
    assert row["num_unique_start_goal_pairs"] == 1
    assert row["num_unique_episodes"] == row["num_episodes"]


def test_self_loop_not_high_level_option_edge_by_default():
    labels = np.asarray([0, 0, 1], dtype=np.int64)
    pairs = _pair_records([0, 0], [0, 1], [1, 1], [0, 1], [1, 2])
    edges, _ = build_option_edges(
        pairs,
        labels,
        _selected([0, 1]),
        H=1,
        min_support=1,
        min_episodes=1,
    )
    assert (0, 0) not in set(zip(edges["src"], edges["dst"]))
    assert (0, 1) in set(zip(edges["src"], edges["dst"]))


def test_empty_option_edges_and_self_loop_outputs_keep_schema():
    labels = np.asarray([0, 0, 1], dtype=np.int64)
    pairs = _pair_records([0, 0], [0, 1], [1, 1], [0, 1], [1, 2])
    selected = _selected([0, 1])
    edges, segments = build_option_edges(
        pairs,
        labels,
        selected,
        H=1,
        min_support=99,
        min_episodes=1,
    )
    assert edges.empty
    assert {"edge_id", "src", "dst", "median_h", "edge_bottleneck_score"}.issubset(edges.columns)
    assert segments["edge_id"].shape[0] == 0

    loops = build_self_loop_summary(pairs, labels, selected, H=1)
    assert {"cluster", "num_segments", "median_h"}.issubset(loops.columns)


def test_node_selection_budget_for_density_bottleneck_and_core_plus():
    density = pd.DataFrame(
        {"cluster": [0, 1, 2, 3], "count": [10, 9, 8, 7], "density": [0.4, 0.3, 0.2, 0.1]}
    )
    bottleneck = pd.DataFrame(
        {"cluster": [0, 1, 2, 3], "bottleneck_score": [0.0, 0.2, 1.0, 0.9]}
    )
    assert select_nodes(density, bottleneck, "density", 2, 0)["selected"].sum() == 2
    assert select_nodes(density, bottleneck, "bottleneck", 2, 0)["selected"].sum() == 2
    core = select_nodes(density, bottleneck, "core_plus_bottleneck", 3, 0)
    assert core["selected"].sum() == 3
    assert 0 in set(core[core["selected"]]["cluster"])
    assert 2 in set(core[core["selected"]]["cluster"])


def test_strict_query_selection_rate_and_coverage_over_all():
    G = nx.DiGraph()
    G.add_edge(0, 1, cost=1.0, edge_id=0)
    selected = _selected([0, 1])
    queries = pd.DataFrame(
        [
            {"start_cluster": 0, "goal_cluster": 1},
            {"start_cluster": 2, "goal_cluster": 1},
        ]
    )
    summary, _ = evaluate_task_path_coverage(
        G,
        np.arange(3),
        np.arange(3),
        queries,
        selected,
        H_query=1,
        mode="strict_selected",
    )
    row = summary.iloc[0]
    assert row["all_num_queries"] == 2
    assert row["strict_num_queries"] == 1
    assert row["strict_query_selection_rate"] == 0.5
    assert row["strict_coverage_over_all"] == 0.5


def test_virtual_query_connects_start_and_goal_through_selected_graph():
    G = nx.DiGraph()
    G.add_edge(1, 2, cost=1.0, edge_id=0)
    selected = _selected([1, 2])
    support = sparse.csr_matrix(
        np.asarray(
            [
                [0, 3, 0, 0],
                [0, 0, 3, 0],
                [0, 0, 0, 3],
                [0, 0, 0, 0],
            ],
            dtype=np.int64,
        )
    )
    queries = pd.DataFrame([{"start_cluster": 0, "goal_cluster": 3}])
    summary, paths = evaluate_task_path_coverage(
        G,
        np.arange(4),
        np.arange(4),
        queries,
        selected,
        H_query=1,
        mode="virtual_query",
        support_N=support,
        min_support=1,
    )
    assert summary.iloc[0]["path_coverage"] == 1.0
    assert paths.iloc[0]["num_virtual_edges_used"] == 2
    assert paths.iloc[0]["num_real_option_edges_used"] == 1
    assert paths.iloc[0]["virtual_edge_ratio"] == 2 / 3
    assert summary.iloc[0]["mean_num_virtual_edges_used"] == 2.0


def test_virtual_query_handles_empty_query_table():
    G = nx.DiGraph()
    selected = _selected([1, 2])
    support = sparse.csr_matrix(np.zeros((3, 3), dtype=np.int64))
    queries = pd.DataFrame(columns=["start_cluster", "goal_cluster"])
    summary, paths = evaluate_task_path_coverage(
        G,
        np.arange(3),
        np.arange(3),
        queries,
        selected,
        H_query=1,
        mode="virtual_query",
        support_N=support,
        min_support=1,
    )
    assert summary.iloc[0]["num_queries"] == 0
    assert summary.iloc[0]["path_coverage"] == 0.0
    assert {"reachable", "path_cost", "path_edge_ids"}.issubset(paths.columns)


def test_option_edge_cost_is_written_and_used_by_graph():
    option_edges = pd.DataFrame(
        {
            "edge_id": [0],
            "src": [1],
            "dst": [2],
            "num_segments": [4],
            "num_episodes": [2],
            "min_h": [2],
            "median_h": [3.0],
            "mean_h": [3.0],
            "max_h": [4],
            "support_count": [4],
            "reverse_support_count": [0],
            "asymmetry": [1.0],
        }
    )
    with_cost = add_edge_costs(option_edges, support_penalty=2.0)
    assert with_cost.iloc[0]["cost"] == 4.0
    G = build_option_graph(with_cost)
    assert G[1][2]["cost"] == 4.0


def test_option_edge_cost_can_use_unique_starts():
    option_edges = pd.DataFrame(
        {
            "edge_id": [0],
            "src": [1],
            "dst": [2],
            "num_segments": [100],
            "num_unique_starts": [4],
            "num_episodes": [2],
            "min_h": [2],
            "median_h": [3.0],
            "mean_h": [3.0],
            "max_h": [4],
            "support_count": [100],
            "reverse_support_count": [0],
            "asymmetry": [1.0],
        }
    )
    with_cost = add_edge_costs(option_edges, support_penalty=2.0, support_unit="unique_starts")
    assert with_cost.iloc[0]["cost"] == 4.0


def test_reverse_support_raw_and_certified_can_differ():
    labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
    pairs = _pair_records([0, 1, 0], [0, 0, 1], [1, 1, 1], [0, 2, 1], [1, 3, 0])
    edges, _ = build_option_edges(
        pairs,
        labels,
        _selected([0, 1]),
        H=1,
        min_support=2,
        min_episodes=2,
    )
    assert set(zip(edges["src"], edges["dst"])) == {(0, 1)}
    row = edges.iloc[0]
    assert row["num_segments"] == 2
    assert row["num_unique_starts"] == 2
    assert row["num_unique_terminations"] == 2
    assert row["num_unique_start_goal_pairs"] == 2
    assert row["support_density_per_episode"] == 1.0
    assert row["reverse_support_raw"] == 1
    assert row["reverse_support_certified"] == 0


def test_option_compatibility_detects_termination_initiation_mismatch():
    option_edges = pd.DataFrame(
        {
            "edge_id": [0, 1],
            "src": [1, 2],
            "dst": [2, 3],
        }
    )
    edge_segments = {
        "edge_id": np.asarray([0, 1], dtype=np.int64),
        "ep_id": np.asarray([0, 1], dtype=np.int64),
        "t": np.asarray([0, 0], dtype=np.int64),
        "h": np.asarray([1, 1], dtype=np.int64),
        "global_i": np.asarray([0, 10], dtype=np.int64),
        "global_j": np.asarray([1, 11], dtype=np.int64),
    }
    summary, pairs = compute_edge_compatibility(
        option_edges,
        edge_segments,
        labels=np.arange(12),
        pair_records={},
        H_intra=2,
    )
    assert pairs.shape[0] == 1
    assert summary.iloc[0]["cluster_compatible_rate"] == 1.0
    assert summary.iloc[0]["strict_compatible_rate"] == 0.0
    assert pairs.iloc[0]["num_bridge_segments"] == 0


def test_option_compatibility_reports_support_quantities():
    option_edges = pd.DataFrame(
        {
            "edge_id": [0, 1],
            "src": [1, 2],
            "dst": [2, 3],
        }
    )
    edge_segments = {
        "edge_id": np.asarray([0, 0, 1, 1], dtype=np.int64),
        "ep_id": np.asarray([0, 1, 0, 1], dtype=np.int64),
        "t": np.asarray([0, 0, 0, 0], dtype=np.int64),
        "h": np.asarray([1, 1, 1, 1], dtype=np.int64),
        "global_i": np.asarray([0, 10, 7, 50], dtype=np.int64),
        "global_j": np.asarray([5, 20, 8, 51], dtype=np.int64),
    }
    summary, pairs = compute_edge_compatibility(
        option_edges,
        edge_segments,
        labels=np.arange(52),
        pair_records={},
        H_intra=10,
    )
    row = pairs.iloc[0]
    assert bool(row["strict_compatible"])
    assert row["num_first_edge_segments"] == 2
    assert row["num_bridge_segments"] == 1
    assert row["num_bridge_episodes"] == 1
    assert row["min_bridge_h"] == 2.0
    assert row["median_bridge_h"] == 2.0
    assert row["compatibility_support_rate"] == 0.5
    assert summary.iloc[0]["mean_num_bridge_segments"] == 1.0
    assert summary.iloc[0]["mean_compatibility_support_rate"] == 0.5


def test_bottleneck_removal_reduces_bridge_graph_coverage():
    G = nx.DiGraph()
    G.add_edge(0, 1, cost=1.0, edge_id=0)
    G.add_edge(1, 2, cost=1.0, edge_id=1)
    bottleneck = pd.DataFrame(
        {"cluster": [0, 1, 2], "bottleneck_score": [0.0, 1.0, 0.0]}
    )
    queries = pd.DataFrame([{"start_cluster": 0, "goal_cluster": 2}])
    utility = bottleneck_removal_ablation(G, bottleneck, top_q=1 / 3, path_queries=queries)
    before = utility[utility["condition"] == "before"].iloc[0]["path_coverage"]
    after = utility[utility["condition"] == "after"].iloc[0]["path_coverage"]
    assert before == 1.0
    assert after == 0.0
