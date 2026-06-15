from __future__ import annotations

import pickle
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pandas as pd

from bars.gas_bars.support_keygraph import (
    load_keygraph_pickle,
    patch_gas_keygraph_with_support,
    save_keygraph_pickle,
)
from scripts.stage37_prepare_calibrated_support_scores import add_calibrated_support_columns
from scripts.stage41_mix_task_keygraph_paths import mix_task_keygraph_paths
from scripts.stage42_path_local_gated_keygraph import path_local_gated_mix


def _dummy_keygraph():
    kg = SimpleNamespace()
    kg.way_steps = 2.0
    kg.base_node_cnt = 3
    kg.nodes = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 0.0],
        ],
        dtype=np.float32,
    )
    kg.graph = nx.DiGraph()
    for i, pos in enumerate(kg.nodes):
        kg.graph.add_node(i, pos=pos)
    # Original cached path prefers unsupported 0->2->3 because it is cheaper.
    kg.graph.add_edge(0, 1, weight=2.0)
    kg.graph.add_edge(1, 3, weight=1.0)
    kg.graph.add_edge(0, 2, weight=0.1)
    kg.graph.add_edge(2, 3, weight=0.1)
    # Reverse edges make GAS' target-rooted Dijkstra recover node->target paths.
    kg.graph.add_edge(1, 0, weight=2.0)
    kg.graph.add_edge(3, 1, weight=1.0)
    kg.graph.add_edge(2, 0, weight=0.1)
    kg.graph.add_edge(3, 2, weight=0.1)
    kg.task_goal_dict = {1: np.asarray([2.0, 0.0], dtype=np.float32)}
    kg.task_node_dict = {1: kg.nodes[3]}
    kg.task_node_idx_dict = {1: 3}
    kg.task_paths_dict = {1: {0: [0, 2, 3], 1: [1, 3], 2: [2, 3]}}
    kg.task_paths_dist_dict = {1: {0: 0.2, 1: 1.0, 2: 0.1}}
    return kg


def _edge_scores():
    return pd.DataFrame(
        [
            {"u": 0, "v": 1, "local_support": 3, "r_exec": 0.2},
            {"u": 1, "v": 3, "local_support": 3, "r_exec": 0.1},
            {"u": 0, "v": 2, "local_support": 0, "r_exec": 0.9},
            {"u": 2, "v": 3, "local_support": 3, "r_exec": 0.1},
            {"u": 1, "v": 0, "local_support": 3, "r_exec": 0.2},
            {"u": 3, "v": 1, "local_support": 3, "r_exec": 0.1},
            {"u": 2, "v": 0, "local_support": 0, "r_exec": 0.9},
            {"u": 3, "v": 2, "local_support": 3, "r_exec": 0.1},
        ]
    )


def test_support_penalty_recomputes_cached_gas_task_paths():
    kg = _dummy_keygraph()
    result = patch_gas_keygraph_with_support(
        kg,
        _edge_scores(),
        mode="penalize",
        support_column="local_support",
        min_support=1,
        unsupported_penalty=10.0,
        risk_column="r_exec",
        risk_weight=0.0,
    )
    patched = result.key_graph

    assert patched.graph[0][2]["weight"] > 10.0
    assert patched.task_paths_dict[1][0] == [0, 1, 3]
    assert result.summary["recomputed_task_paths"] == 1
    assert result.summary["num_effective_unsupported_edges"] == 2


def test_directional_support_penalty_uses_forward_execution_costs():
    kg = _dummy_keygraph()
    scores = _edge_scores()
    # Make only the executed shortcut 0->2 unsupported.  The reverse 2->0
    # remains supported, which catches target-rooted Dijkstra on the original
    # directed graph because it would optimize the wrong reverse edge cost.
    scores.loc[(scores["u"] == 0) & (scores["v"] == 2), "local_support"] = 0
    scores.loc[(scores["u"] == 2) & (scores["v"] == 0), "local_support"] = 3

    result = patch_gas_keygraph_with_support(
        kg,
        scores,
        mode="penalize",
        support_column="local_support",
        min_support=1,
        unsupported_penalty=10.0,
        risk_column="r_exec",
        risk_weight=0.0,
    )

    assert result.key_graph.task_paths_dict[1][0] == [0, 1, 3]
    assert result.key_graph.task_paths_dist_dict[1][0] == 3.0


def test_prune_mode_protects_task_goal_edges_by_default():
    kg = _dummy_keygraph()
    scores = _edge_scores()
    scores.loc[(scores["u"] == 2) & (scores["v"] == 3), "local_support"] = 0
    result = patch_gas_keygraph_with_support(
        kg,
        scores,
        mode="prune",
        support_column="local_support",
        min_support=1,
        protect_goal_edges=True,
    )
    patched = result.key_graph

    assert not patched.graph.has_edge(0, 2)
    assert patched.graph.has_edge(2, 3), "task-goal connector must not be pruned by support filtering"
    goal_row = result.edge_audit[(result.edge_audit["u"] == 2) & (result.edge_audit["v"] == 3)].iloc[0]
    assert int(goal_row["protected"]) == 1
    assert int(goal_row["pruned"]) == 0


def test_missing_score_policy_can_prune_non_goal_edges():
    kg = _dummy_keygraph()
    scores = _edge_scores()
    scores = scores[~((scores["u"] == 0) & (scores["v"] == 2))]
    result = patch_gas_keygraph_with_support(
        kg,
        scores,
        mode="prune",
        support_column="local_support",
        min_support=1,
        missing_score_policy="prune",
        protect_goal_edges=True,
    )

    assert not result.key_graph.graph.has_edge(0, 2)
    row = result.edge_audit[(result.edge_audit["u"] == 0) & (result.edge_audit["v"] == 2)].iloc[0]
    assert row["missing_score_action"] == "pruned_missing_score"


def test_keygraph_pickle_roundtrip_uses_gas_dict_payload(tmp_path):
    kg = _dummy_keygraph()
    out = tmp_path / "keygraph.pkl"
    save_keygraph_pickle(kg, out)
    with out.open("rb") as fh:
        payload = pickle.load(fh)
    assert isinstance(payload, dict)
    loaded = load_keygraph_pickle(out)
    assert loaded.base_node_cnt == kg.base_node_cnt
    assert np.allclose(loaded.nodes, kg.nodes)


def test_calibrated_support_risk_columns_are_monotone_and_protect_goal_edges():
    scores = pd.DataFrame(
        [
            {"u": 0, "v": 1, "edge_source": "gas_distance", "local_support": 0, "same_traj_support": 0},
            {"u": 1, "v": 2, "edge_source": "gas_distance", "local_support": 1, "same_traj_support": 3},
            {"u": 2, "v": 3, "edge_source": "gas_distance", "local_support": 1, "same_traj_support": 20},
            {"u": 3, "v": 4, "edge_source": "gas_goal_connector", "local_support": 0, "same_traj_support": 0},
        ]
    )

    out = add_calibrated_support_columns(scores, target_support=8.0, protect_goal_edges=True)

    assert out.loc[0, "risk_hybrid_support"] > out.loc[1, "risk_hybrid_support"]
    assert out.loc[1, "risk_hybrid_support"] > out.loc[2, "risk_hybrid_support"]
    assert out.loc[3, "risk_hybrid_support"] == 0.0
    assert out.loc[3, "risk_unsupported_binary"] == 0.0


def test_stage41_task_path_mixer_replaces_only_selected_task_caches():
    base = SimpleNamespace(
        task_paths_dict={1: {0: [0, 1]}, 2: {0: [0, 2]}},
        task_paths_dist_dict={1: {0: 1.0}, 2: {0: 2.0}},
    )
    method_a = SimpleNamespace(
        task_paths_dict={1: {0: [0, 10, 1]}, 2: {0: [0, 20, 2]}},
        task_paths_dist_dict={1: {0: 11.0}, 2: {0: 22.0}},
    )
    method_b = SimpleNamespace(
        task_paths_dict={1: {0: [0, 100, 1]}, 2: {0: [0, 200, 2]}},
        task_paths_dist_dict={1: {0: 101.0}, 2: {0: 202.0}},
    )

    mixed, rows = mix_task_keygraph_paths(base, {"a": method_a, "b": method_b}, {1: "a", 2: "b"})

    assert mixed.task_paths_dict[1][0] == [0, 10, 1]
    assert mixed.task_paths_dist_dict[1][0] == 11.0
    assert mixed.task_paths_dict[2][0] == [0, 200, 2]
    assert mixed.task_paths_dist_dict[2][0] == 202.0
    assert base.task_paths_dict[1][0] == [0, 1]
    assert mixed.bars_task_method_map == {"1": "a", "2": "b"}
    assert {row["method"] for row in rows} == {"a", "b"}


def test_stage42_path_local_gate_selects_only_local_improvements():
    base = SimpleNamespace(
        base_node_cnt=3,
        task_paths_dict={1: {0: [0, 2], 1: [1, 2]}},
        task_paths_dist_dict={1: {0: 1.0, 1: 1.0}},
        graph=nx.DiGraph(),
    )
    for idx in range(3):
        base.graph.add_node(idx)
    base.graph.add_edge(0, 1, weight=0.4)
    base.graph.add_edge(1, 2, weight=0.4)
    base.graph.add_edge(0, 2, weight=1.0)
    base.graph.add_edge(1, 0, weight=0.4)
    base.graph.add_edge(2, 1, weight=0.4)
    base.graph.add_edge(2, 0, weight=1.0)

    candidate = SimpleNamespace(
        task_paths_dict={1: {0: [0, 1, 2], 1: [1, 0, 2]}},
        # Deliberately risk-inflated distances. The gate should write base cost.
        task_paths_dist_dict={1: {0: 99.0, 1: 77.0}},
    )
    edge_scores = pd.DataFrame(
        [
            {"u": 0, "v": 2, "local_support": 0, "same_traj_support": 0},
            {"u": 0, "v": 1, "local_support": 1, "same_traj_support": 5},
            {"u": 1, "v": 2, "local_support": 1, "same_traj_support": 5},
            {"u": 1, "v": 0, "local_support": 1, "same_traj_support": 0},
        ]
    )

    mixed, selected, summary = path_local_gated_mix(
        base,
        {"candidate": candidate},
        edge_scores,
        max_base_cost_ratio=1.0,
        max_edge_delta=1,
        min_support_gain=1.0,
        min_unsupported_gain=0.1,
        improvement_mode="both",
        unsupported_weight=20.0,
        support_weight=1.0,
        cost_penalty=10.0,
        edge_penalty=5.0,
        distance_mode="base_cost",
    )

    assert mixed.task_paths_dict[1][0] == [0, 1, 2]
    assert mixed.task_paths_dist_dict[1][0] == 0.8
    assert mixed.task_paths_dict[1][1] == [1, 2]
    assert len(selected) == 1
    assert summary["num_selected_paths"] == 1
