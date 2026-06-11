import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import networkx as nx
import numpy as np
from scipy import sparse

from phase1.bottleneck import betweenness_score, removal_impact_score
from phase1.support_graph import (
    build_directed_support_graph,
    compute_support_counts,
    one_way_edge_ratio,
    support_asymmetry,
)
from phase1.trajectory import build_h_step_pairs, split_into_episodes


def _dataset_from_labels(labels, terminal_indices):
    labels = np.asarray(labels, dtype=np.int64)
    obs = labels.reshape(-1, 1).astype(np.float32)
    terminals = np.zeros(labels.shape[0], dtype=bool)
    terminals[np.asarray(terminal_indices, dtype=np.int64)] = True
    return {
        "observations": obs,
        "actions": np.zeros((labels.shape[0], 1), dtype=np.float32),
        "next_observations": np.roll(obs, shift=-1, axis=0),
        "terminals": terminals,
    }


def _support_for_labels(labels, terminal_indices, horizons=(1,), n_clusters=3):
    dataset = _dataset_from_labels(labels, terminal_indices)
    episodes = split_into_episodes(dataset)
    pairs = build_h_step_pairs(episodes, list(horizons), seed=0)
    cluster_labels = np.asarray(labels, dtype=np.int64)
    return compute_support_counts(pairs, cluster_labels, list(horizons), n_clusters)


def test_symmetric_chain_has_low_asymmetry_and_no_one_way_edges():
    counts = _support_for_labels(
        labels=[0, 1, 2, 2, 1, 0],
        terminal_indices=[2, 5],
        horizons=(1,),
        n_clusters=3,
    )
    N = counts[1]
    assert support_asymmetry(N) < 1e-9
    assert one_way_edge_ratio(N, min_support=1) < 1e-9


def test_one_way_chain_has_asymmetry_and_one_way_edges():
    counts = _support_for_labels(
        labels=[0, 1, 2],
        terminal_indices=[2],
        horizons=(1,),
        n_clusters=3,
    )
    N = counts[1]
    assert support_asymmetry(N) > 0.9
    assert one_way_edge_ratio(N, min_support=1) > 0.9


def test_bottleneck_node_has_high_betweenness_and_removal_impact():
    G = nx.DiGraph()
    G.add_nodes_from(range(7))
    G.graph["n_clusters"] = 7
    left = [0, 1, 2]
    right = [4, 5, 6]
    for cluster in (left, right):
        for src in cluster:
            for dst in cluster:
                if src != dst:
                    G.add_edge(src, dst, count=3)
    G.add_edges_from([(2, 3), (3, 4), (4, 3), (3, 2)])

    betweenness = betweenness_score(G, sample_k=7, seed=0)
    removal = removal_impact_score(G, candidate_nodes=range(7), sample_pairs=200, seed=0)

    assert betweenness[3] > betweenness[0]
    assert removal[3] > removal[0]


def test_exact_and_dense_upto_pair_modes_have_different_counts():
    dataset = _dataset_from_labels([0, 1, 2, 3, 4], terminal_indices=[4])
    episodes = split_into_episodes(dataset)

    exact_pairs = build_h_step_pairs(episodes, [1, 3], pair_mode="exact", seed=0)
    dense_pairs = build_h_step_pairs(episodes, [1, 3], pair_mode="dense_upto", seed=0)

    assert exact_pairs["h"].shape[0] == 6
    assert dense_pairs["h"].shape[0] == 9
    assert set(exact_pairs["h"].tolist()) == {1, 3}
    assert set(dense_pairs["h"].tolist()) == {1, 2, 3}


def test_dense_upto_support_counts_include_all_h_up_to_H():
    labels = np.asarray([0, 1, 2, 3], dtype=np.int64)
    dataset = _dataset_from_labels(labels, terminal_indices=[3])
    episodes = split_into_episodes(dataset)
    pairs = build_h_step_pairs(episodes, [3], pair_mode="dense_upto", seed=0)
    counts = compute_support_counts(pairs, labels, horizons=[1, 2, 3], n_clusters=4)

    N1 = counts[1]
    N3 = counts[3]
    assert N1[0, 1] == 1
    assert N1[0, 2] == 0
    assert N3[0, 1] == 1
    assert N3[0, 2] == 1
    assert N3[0, 3] == 1
    assert N3[1, 3] == 1


def test_directed_support_graph_excludes_self_loops_by_default():
    N = sparse.csr_matrix(
        np.asarray(
            [
                [5, 3],
                [0, 2],
            ],
            dtype=np.int64,
        )
    )

    graph = build_directed_support_graph(N, min_support=1)
    graph_with_self = build_directed_support_graph(N, min_support=1, include_self_loops=True)

    assert graph.number_of_edges() == 1
    assert not graph.has_edge(0, 0)
    assert not graph.has_edge(1, 1)
    assert graph_with_self.number_of_edges() == 3
