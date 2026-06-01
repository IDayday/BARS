from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .config import GraphBuildConfig
from .dataset import OfflineDataset
from .edge_features import EdgeFeatureExtractor
from .exec_calibrator import ExecutionCalibrator
from .graph import GraphData
from .math_utils import knn_indices


def _candidate_edges(dataset: OfflineDataset, node_indices: np.ndarray, cfg: GraphBuildConfig) -> tuple[np.ndarray, np.ndarray]:
    node_indices = np.asarray(node_indices, dtype=np.int64)
    x_all = dataset.embedding(cfg.embedding_key, "states")
    x_nodes = x_all[node_indices]
    n_nodes = len(node_indices)
    if n_nodes <= 1:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    # kNN candidate edges in representation space.
    k = min(max(2, cfg.candidate_knn + 1), n_nodes)
    neigh, _ = knn_indices(x_nodes, k)
    src_list = []
    dst_list = []
    for i in range(n_nodes):
        for j in neigh[i]:
            j = int(j)
            if j == i:
                continue
            src_list.append(i)
            dst_list.append(j)
            if not cfg.directed:
                src_list.append(j)
                dst_list.append(i)

    # Temporal support edges along trajectories. These are important for stitch
    # and are not guaranteed to appear in representation kNN.
    node_pos_by_dataset_idx = {int(idx): pos for pos, idx in enumerate(node_indices.tolist())}
    selected_by_traj: dict[object, list[int]] = {}
    for pos, ds_idx in enumerate(node_indices):
        tid = dataset.traj_ids[int(ds_idx)]
        selected_by_traj.setdefault(tid.item() if hasattr(tid, "item") else tid, []).append(pos)
    for positions in selected_by_traj.values():
        positions = sorted(positions, key=lambda p: dataset.time_idxs[node_indices[p]])
        for a_idx, a in enumerate(positions):
            ta = dataset.time_idxs[node_indices[a]]
            for b in positions[a_idx + 1 :]:
                tb = dataset.time_idxs[node_indices[b]]
                if tb - ta > cfg.same_traj_window:
                    break
                src_list.append(a)
                dst_list.append(b)
                if not cfg.directed:
                    src_list.append(b)
                    dst_list.append(a)

    # Deduplicate directed edges.
    pairs = np.asarray(list(set(zip(src_list, dst_list))), dtype=np.int64)
    if len(pairs) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return pairs[:, 0], pairs[:, 1]


def _robust_normalize_feature(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return np.zeros_like(arr, dtype=np.float32)
    med = np.median(finite)
    iqr = np.percentile(finite, 75) - np.percentile(finite, 25)
    return np.nan_to_num((arr - med) / max(iqr, eps), nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)


def _positive_scaled(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return np.ones_like(arr, dtype=np.float32)
    scale = max(float(np.median(finite)), eps)
    return np.nan_to_num(arr / scale, nan=1.0, posinf=1e6, neginf=1e6).astype(np.float32)


def _compute_costs(features: Dict[str, np.ndarray], cfg: GraphBuildConfig, p_exec: Optional[np.ndarray]) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
    d_tdr = _positive_scaled(features["d_tdr"]) if cfg.normalize_features else features["d_tdr"].astype(np.float32)
    d_tmd = _positive_scaled(features["d_tmd"]) if cfg.normalize_features else features["d_tmd"].astype(np.float32)
    d_xy = _positive_scaled(features["d_xy"]) if cfg.normalize_features else features["d_xy"].astype(np.float32)
    disagreement = features["metric_disagreement"].astype(np.float32)

    cost = cfg.lambda_tdr * d_tdr + cfg.lambda_tmd_side * d_tmd + cfg.lambda_xy * d_xy

    # Soft long-hop penalty: penalize edges much larger than the local median.
    hop_ratio = d_tdr
    longhop_excess = np.maximum(hop_ratio - cfg.longhop_threshold, 0.0)
    longhop_penalty = cfg.lambda_longhop * (longhop_excess ** cfg.longhop_power)
    cost = cost + longhop_penalty

    uncertainty = np.zeros_like(cost, dtype=np.float32)
    if p_exec is not None:
        p = np.clip(p_exec.astype(np.float32), cfg.eps, 1.0 - cfg.eps)
        exec_penalty = -np.log(p)
        uncertainty = p * (1.0 - p)
        cost = cost + cfg.lambda_exec * exec_penalty + cfg.lambda_uncertainty * uncertainty
    else:
        p = np.full_like(cost, np.nan, dtype=np.float32)
        exec_penalty = np.zeros_like(cost, dtype=np.float32)

    cross_penalty = cfg.lambda_cross_traj * features["is_cross_traj"].astype(np.float32)
    disagreement_penalty = cfg.lambda_disagreement * disagreement
    cost = cost + cross_penalty + disagreement_penalty

    # TMD side-channel: only allow a shortcut when execution confidence is high.
    tmd_shortcut_used = np.zeros_like(cost, dtype=np.float32)
    if cfg.use_tmd_gated_shortcut:
        if p_exec is None:
            eligible = np.zeros_like(cost, dtype=bool)
        else:
            eligible = p_exec >= cfg.tmd_shortcut_min_p_exec
            if cfg.tmd_shortcut_max_disagreement is not None:
                eligible &= disagreement <= cfg.tmd_shortcut_max_disagreement
        shortcut_cost = (1.0 - cfg.tmd_shortcut_w) * cost + cfg.tmd_shortcut_w * d_tmd
        new_cost = np.where(eligible, np.minimum(cost, shortcut_cost), cost)
        tmd_shortcut_used = (new_cost < cost).astype(np.float32)
        cost = new_cost

    cost = np.maximum(cost.astype(np.float32), cfg.min_edge_cost)
    aux = {
        "norm_d_tdr": d_tdr.astype(np.float32),
        "norm_d_tmd": d_tmd.astype(np.float32),
        "norm_d_xy": d_xy.astype(np.float32),
        "longhop_penalty": longhop_penalty.astype(np.float32),
        "exec_penalty": exec_penalty.astype(np.float32),
        "exec_uncertainty": uncertainty.astype(np.float32),
        "cross_penalty": cross_penalty.astype(np.float32),
        "disagreement_penalty": disagreement_penalty.astype(np.float32),
        "p_exec": p.astype(np.float32),
        "tmd_shortcut_used": tmd_shortcut_used.astype(np.float32),
    }
    return cost, aux


def build_stage27_graph(
    dataset: OfflineDataset,
    node_indices: np.ndarray,
    cfg: GraphBuildConfig,
    calibrator: Optional[ExecutionCalibrator] = None,
) -> GraphData:
    """Build an Execution-Calibrated AdaptiveGAS graph variant."""
    node_indices = np.asarray(node_indices, dtype=np.int64)
    if len(node_indices) == 0:
        raise ValueError("node_indices is empty")

    src_node, dst_node = _candidate_edges(dataset, node_indices, cfg)
    if len(src_node) == 0:
        raise ValueError("No candidate edges generated; increase node count/candidate_knn")
    src_ds = node_indices[src_node]
    dst_ds = node_indices[dst_node]

    extractor = EdgeFeatureExtractor(dataset, tdr_key=cfg.embedding_key, tmd_key=cfg.tmd_key, xy_key=cfg.xy_key)
    features = extractor.pair_features(src_ds, dst_ds)
    p_exec = calibrator.predict_proba_features(features) if calibrator is not None else None
    costs, aux_features = _compute_costs(features, cfg, p_exec)

    keep = np.isfinite(costs)
    if cfg.max_edge_cost is not None:
        keep &= costs <= cfg.max_edge_cost
    if cfg.exec_gate_threshold is not None:
        if p_exec is None:
            raise ValueError(f"Variant {cfg.variant} requests exec_gate_threshold but calibrator is None")
        keep &= p_exec >= cfg.exec_gate_threshold

    if not np.any(keep):
        raise ValueError("All candidate edges were pruned; relax thresholds/gates")

    edge_features = {**features, **aux_features}
    edge_features = {k: np.asarray(v)[keep].astype(np.float32) for k, v in edge_features.items()}
    src_node = src_node[keep]
    dst_node = dst_node[keep]
    costs = costs[keep]

    metadata = {
        "variant": cfg.variant,
        "config": cfg.__dict__.copy(),
        "num_candidate_edges_before_prune": int(len(keep)),
        "num_edges_after_prune": int(len(costs)),
        "has_calibrator": calibrator is not None,
    }
    if calibrator is not None and calibrator.metrics_ is not None:
        metadata["calibrator_metrics"] = calibrator.metrics_.to_dict()

    return GraphData(
        node_indices=node_indices,
        states=dataset.states[node_indices],
        traj_ids=dataset.traj_ids[node_indices],
        time_idxs=dataset.time_idxs[node_indices],
        edges_src=src_node,
        edges_dst=dst_node,
        edge_costs=costs,
        edge_features=edge_features,
        metadata=metadata,
    )
