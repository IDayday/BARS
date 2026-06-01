from __future__ import annotations

import numpy as np

from stage27_gas.config import CalibratorConfig, GraphBuildConfig, NodeSelectConfig
from stage27_gas.dataset import OfflineDataset
from stage27_gas.exec_calibrator import ExecutionCalibrator, build_pair_training_set
from stage27_gas.graph_builder import build_stage27_graph
from stage27_gas.node_selection import select_stage27_nodes
from stage27_gas.planner import shortest_path


def make_toy_dataset(n_traj=4, length=40):
    states = []
    traj_ids = []
    time_idxs = []
    te = []
    for tid in range(n_traj):
        offset = np.array([tid * 0.2, tid * 0.1], dtype=np.float32)
        for t in range(length):
            x = np.array([t / length, np.sin(t / 10.0)], dtype=np.float32) + offset
            states.append(x)
            traj_ids.append(tid)
            time_idxs.append(t)
            te.append(1.0 - abs(t - length / 2) / length)
    states = np.asarray(states, dtype=np.float32)
    extras = {
        "tdr_emb": states.copy(),
        "tmd_emb": states.copy() * np.array([1.0, 0.7], dtype=np.float32),
        "xy": states.copy(),
        "te_scores": np.asarray(te, dtype=np.float32),
    }
    return OfflineDataset(states, np.asarray(traj_ids), np.asarray(time_idxs), extras)


def test_node_selection_graph_and_planner():
    ds = make_toy_dataset()
    nodes, pools = select_stage27_nodes(ds, NodeSelectConfig(max_nodes=80, coverage_k=30, bottleneck_k=10))
    assert len(nodes) <= 80
    assert len(nodes) > 10
    graph = build_stage27_graph(ds, nodes, GraphBuildConfig(candidate_knn=8, same_traj_window=15))
    assert graph.num_nodes == len(nodes)
    assert graph.num_edges > 0
    path, cost, eids = shortest_path(graph, 0, graph.num_nodes - 1)
    assert path or np.isinf(cost)
    if path:
        assert len(eids) == len(path) - 1


def test_execution_calibrator_and_gated_graph():
    ds = make_toy_dataset(n_traj=5, length=35)
    nodes, _ = select_stage27_nodes(ds, NodeSelectConfig(max_nodes=90, coverage_k=30, bottleneck_k=10))
    calib_cfg = CalibratorConfig(horizon=8, positives_per_traj=50, random_negatives=100, hard_negatives=100, same_traj_far_negatives=50)
    train_set = build_pair_training_set(ds, calib_cfg)
    assert train_set.y.mean() > 0
    cal = ExecutionCalibrator()
    metrics = cal.fit(train_set, calib_cfg)
    assert metrics.n_train > 0
    cfg = GraphBuildConfig(
        variant="C2_EXEC_GATE",
        candidate_knn=8,
        same_traj_window=15,
        lambda_exec=0.25,
        exec_gate_threshold=0.2,
        lambda_longhop=0.1,
        use_tmd_gated_shortcut=True,
    )
    graph = build_stage27_graph(ds, nodes, cfg, calibrator=cal)
    assert graph.num_edges > 0
    assert "p_exec" in graph.edge_features
