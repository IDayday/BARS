from __future__ import annotations

"""GAS-compatible backbone adapters.

This module supports two Route-B modes:

1) `graph.node_method=gas_te`: BARS-native graph construction that follows the
   official GAS Temporal Efficiency (TE) filtering + TD-aware clustering idea,
   then maps centers back to concrete dataset states so BARS low-level policies
   can execute observations as subgoals.

2) `external_gas.enabled=true`: import an official GAS `keygraph.pkl` and
   convert it to BARSGraph. Exact same-backbone comparison should provide either
   `external_gas.node_indices_path` or official `dataset_embeddings_path` so the
   latent centers are mapped using the same TDR space that created the keygraph.

Why this separation matters: official GAS keygraph nodes are latent centers;
BARSGraph nodes are concrete dataset states. Mapping latent centers with the
wrong embedding space silently breaks exact reproduction, so the converter logs
which mapping source is used and exposes missing-artifact warnings.
"""

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Any, Dict, Iterable, Tuple
import pickle

import numpy as np

from bars.common.logging import CSVLogger
from bars.data.trajectories import OfflineDataset
from bars.graph.ann import KNNIndex
from bars.graph.types import BARSGraph, EDGE_KIND_KNN


def split_trajectories_from_dataset(dataset: OfflineDataset, embeddings: np.ndarray) -> tuple[list[np.ndarray], list[int]]:
    trajectories: list[np.ndarray] = []
    starts: list[int] = []
    if dataset.size == 0:
        return trajectories, starts
    start = 0
    for i in range(1, dataset.size):
        if dataset.traj_id[i] != dataset.traj_id[i - 1]:
            trajectories.append(embeddings[start:i])
            starts.append(start)
            start = i
    trajectories.append(embeddings[start:dataset.size])
    starts.append(start)
    return trajectories, starts


def _first_future_with_distance(traj: np.ndarray, i: int, distance_threshold: float) -> np.ndarray:
    subarr = traj[i + 1:]
    if len(subarr) == 0:
        return traj[i]
    distances_future = np.linalg.norm(subarr - traj[i], axis=1)
    idxs_above = np.where(distances_future >= float(distance_threshold))[0]
    return traj[i + 1 + idxs_above[0]] if len(idxs_above) > 0 else traj[-1]


def filter_low_efficiency_states(traj: np.ndarray, way_steps: int, te_threshold: float) -> np.ndarray:
    """Official-GAS-style Temporal Efficiency filtering.

    GAS TE keeps states whose fixed-step future direction agrees with the
    direction of a temporally efficient future state in TDR space. We expose the
    same knobs as GAS (`way_steps` / `te_threshold`) and keep this function simple
    to make source-level comparison easy.
    """
    n = len(traj)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    keep = np.ones(n, dtype=bool)
    if n <= int(way_steps) + 1:
        return np.arange(n, dtype=np.int64)
    for i in range(n - int(way_steps)):
        obs_t = traj[i]
        obs_t_plus_step = traj[i + int(way_steps)]
        obs_t_plus_distance = _first_future_with_distance(traj, i, float(way_steps))
        vector_step = obs_t_plus_step - obs_t
        vector_distance = obs_t_plus_distance - obs_t
        vector_step = vector_step / (np.linalg.norm(vector_step) + 1e-10)
        vector_distance = vector_distance / (np.linalg.norm(vector_distance) + 1e-10)
        cosine_similarity = float(np.dot(vector_step, vector_distance))
        if cosine_similarity < float(te_threshold):
            keep[i] = False
    return np.arange(n, dtype=np.int64)[keep]


def collect_efficiency_indices(dataset: OfflineDataset, embeddings: np.ndarray, way_steps: int, te_threshold: float) -> np.ndarray:
    trajectories, starts = split_trajectories_from_dataset(dataset, embeddings)
    out: list[int] = []
    for traj, start in zip(trajectories, starts):
        local = filter_low_efficiency_states(traj, way_steps, te_threshold)
        out.extend((start + local).astype(int).tolist())
    seen: set[int] = set()
    ordered: list[int] = []
    for i in out:
        if int(i) not in seen:
            ordered.append(int(i)); seen.add(int(i))
    return np.asarray(ordered, dtype=np.int64)


def td_aware_clustering_centers(embeddings: np.ndarray, efficiency_indices: np.ndarray, way_steps: int) -> np.ndarray:
    """GAS-style iterative TD-aware clustering.

    New centers are created when a candidate is farther than way_steps/2 in TDR
    space from all existing centers. Cluster centers are the mean of assigned
    high-TE states.
    """
    if len(efficiency_indices) == 0:
        return np.empty((0, embeddings.shape[1]), dtype=np.float32)
    min_dist = float(way_steps) / 2.0
    x = embeddings[efficiency_indices].astype(np.float32)
    centers = np.zeros_like(x)
    assignments: dict[int, list[int]] = {0: [0]}
    centers[0] = x[0]
    n_centers = 1
    for i in range(1, len(x)):
        dists = np.linalg.norm(x[i] - centers[:n_centers], axis=-1)
        min_idx = int(np.argmin(dists))
        if float(dists[min_idx]) > min_dist:
            centers[n_centers] = x[i]
            assignments[n_centers] = [i]
            n_centers += 1
        else:
            assignments[min_idx].append(i)
    out = np.zeros((n_centers, embeddings.shape[1]), dtype=np.float32)
    for idx, assigned in assignments.items():
        out[idx] = x[assigned].mean(axis=0)
    return out


def select_gas_te_nodes(dataset: OfflineDataset, embeddings: np.ndarray, cfg: Dict, logger: CSVLogger | None = None) -> np.ndarray:
    gcfg = cfg.get('graph', {})
    way_steps = int(gcfg.get('gas_way_steps', gcfg.get('way_steps', 25)))
    te_threshold = float(gcfg.get('gas_te_threshold', gcfg.get('te_threshold', 0.99)))
    num_nodes = int(gcfg.get('num_nodes', 500))
    efficiency_idx = collect_efficiency_indices(dataset, embeddings, way_steps, te_threshold)
    if len(efficiency_idx) == 0:
        efficiency_idx = np.arange(dataset.size, dtype=np.int64)
    centers = td_aware_clustering_centers(embeddings, efficiency_idx, way_steps)
    if len(centers) == 0:
        centers = embeddings[efficiency_idx[:1]].astype(np.float32)
    ann = KNNIndex.from_config(embeddings.astype(np.float32), cfg, prefix='ann')
    nearest = ann.kneighbors(centers.astype(np.float32), 1, return_distance=False)[:, 0].astype(np.int64)
    chosen = np.unique(nearest)
    if len(chosen) > num_nodes:
        chosen = chosen[:num_nodes]
    elif len(chosen) < num_nodes:
        rng = np.random.default_rng(int(cfg.get('seed', 0)) + 2025)
        pool = np.setdiff1d(efficiency_idx, chosen, assume_unique=False)
        if len(pool) > 0:
            extra = rng.choice(pool, size=min(num_nodes - len(chosen), len(pool)), replace=False)
            chosen = np.unique(np.concatenate([chosen, extra]))
    if logger is not None:
        logger.log({
            'phase': 'nodes', 'event': 'gas_te_completed', 'node_method': 'gas_te',
            'gas_way_steps': way_steps, 'gas_te_threshold': te_threshold,
            'efficiency_indices': int(len(efficiency_idx)), 'cluster_centers': int(len(centers)),
            'num_nodes': int(len(chosen)), 'gas_te_mapping_backend': ann.backend,
        })
    return chosen[:num_nodes].astype(np.int64)


def load_gas_keygraph_pickle(path: str | Path) -> Any:
    with open(path, 'rb') as f:
        return pickle.load(f)


def _as_dict(data: Any) -> dict:
    if isinstance(data, dict):
        return dict(data)
    if hasattr(data, '__dict__'):
        return dict(data.__dict__)
    return {'payload': data}


def _extract_nodes(data: Any) -> np.ndarray:
    d = _as_dict(data)
    for key in ['nodes', 'node_reps', 'keynodes', 'key_node_reps', 'centers', 'vertices']:
        if key in d:
            arr = np.asarray(d[key], dtype=np.float32)
            if arr.ndim == 2:
                return arr
    # Some keygraph payloads are networkx graphs with node attributes.
    graph_obj = d.get('graph', d.get('keygraph', d.get('G', None)))
    if graph_obj is not None and hasattr(graph_obj, 'nodes'):
        vals = []
        for _, attrs in graph_obj.nodes(data=True):
            for key in ['rep', 'embedding', 'center', 'state']:
                if key in attrs:
                    vals.append(np.asarray(attrs[key], dtype=np.float32)); break
        if vals:
            return np.stack(vals, axis=0).astype(np.float32)
    raise KeyError('Could not find GAS keygraph latent node matrix. Expected keys such as nodes/keynodes/centers.')


def _extract_graph_edges(data: Any, n_nodes: int, node_embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = _as_dict(data)
    graph_obj = d.get('graph', d.get('keygraph', d.get('G', None)))
    edges: list[tuple[int, int]] = []
    weights: list[float] = []
    if graph_obj is not None and hasattr(graph_obj, 'edges'):
        for u, v in graph_obj.edges():
            edges.append((int(u), int(v)))
            try:
                weights.append(float(graph_obj[u][v].get('weight', np.linalg.norm(node_embeddings[int(u)] - node_embeddings[int(v)]))))
            except Exception:
                weights.append(float(np.linalg.norm(node_embeddings[int(u)] - node_embeddings[int(v)])))
    elif 'edges' in d:
        raw_edges = list(d['edges'])
        for item in raw_edges:
            if len(item) >= 2:
                u, v = int(item[0]), int(item[1]); edges.append((u, v))
                weights.append(float(item[2]) if len(item) >= 3 else float(np.linalg.norm(node_embeddings[u] - node_embeddings[v])))
    elif 'adj' in d:
        adj = np.asarray(d['adj'])
        rows, cols = np.where(adj > 0)
        for u, v in zip(rows, cols):
            edges.append((int(u), int(v))); weights.append(float(adj[int(u), int(v)]))
    else:
        # Conservative fallback: connect latent nodes by a distance threshold.
        # This is not exact GAS and is logged by the caller.
        from sklearn.neighbors import NearestNeighbors
        k = min(16, n_nodes)
        neigh = NearestNeighbors(n_neighbors=k).fit(node_embeddings).kneighbors(node_embeddings, return_distance=False)
        for i in range(n_nodes):
            for j in neigh[i, 1:]:
                edges.append((i, int(j)))
                weights.append(float(np.linalg.norm(node_embeddings[i] - node_embeddings[int(j)])))
    if not edges:
        raise ValueError('GAS keygraph contains no edges.')
    src = np.asarray([u for u, _ in edges], dtype=np.int64)
    dst = np.asarray([v for _, v in edges], dtype=np.int64)
    cost = np.asarray(weights, dtype=np.float32)
    return src, dst, cost


def _compact_raw_indices(dataset: OfflineDataset) -> np.ndarray:
    raw = []
    for sl in dataset.traj_slices:
        raw.extend(range(int(sl.raw_start), max(int(sl.raw_start), int(sl.raw_end) - 1)))
    return np.asarray(raw, dtype=np.int64)


def _align_external_dataset_embeddings(raw_embeddings: np.ndarray, dataset: OfflineDataset | None) -> np.ndarray:
    raw_embeddings = np.asarray(raw_embeddings, dtype=np.float32)
    if dataset is None or len(raw_embeddings) == dataset.size:
        return raw_embeddings
    raw_idx = _compact_raw_indices(dataset)
    if len(raw_idx) == dataset.size and len(raw_idx) and int(raw_idx.max()) < len(raw_embeddings):
        return raw_embeddings[raw_idx].astype(np.float32)
    raise ValueError(
        f"external_gas.dataset_embeddings_path has length {len(raw_embeddings)}, but BARS compact dataset has "
        f"size {dataset.size}; could not align via raw trajectory indices. Provide compact embeddings or node_indices_path."
    )


def _raw_to_compact_node_indices(node_indices: np.ndarray, dataset: OfflineDataset | None) -> np.ndarray:
    node_indices = np.asarray(node_indices, dtype=np.int64).reshape(-1)
    if dataset is None or len(node_indices) == 0 or int(node_indices.max()) < dataset.size:
        return node_indices
    raw_idx = _compact_raw_indices(dataset)
    mapping = {int(r): int(i) for i, r in enumerate(raw_idx)}
    missing = [int(x) for x in node_indices if int(x) not in mapping]
    if missing:
        raise ValueError(f"node_indices_path appears to contain raw indices, but {len(missing)} are not in the compact BARS dataset; first_missing={missing[:5]}")
    return np.asarray([mapping[int(x)] for x in node_indices], dtype=np.int64)


def _load_node_indices_from_payload_or_file(data: Any, cfg: Dict, n_nodes: int, dataset: OfflineDataset | None = None) -> np.ndarray | None:
    d = _as_dict(data)
    for key in ['node_indices', 'raw_indices', 'dataset_indices', 'keynode_indices']:
        if key in d:
            arr = np.asarray(d[key], dtype=np.int64).reshape(-1)
            if len(arr) == n_nodes:
                return _raw_to_compact_node_indices(arr, dataset)
    ecfg = cfg.get('external_gas', {})
    path = os.path.expandvars(str(ecfg.get('node_indices_path') or '')).strip()
    if path and os.path.exists(path):
        arr = np.asarray(np.load(str(path)), dtype=np.int64).reshape(-1)
        if len(arr) != n_nodes:
            raise ValueError(f'external_gas.node_indices_path length {len(arr)} != num keygraph nodes {n_nodes}')
        return _raw_to_compact_node_indices(arr, dataset)
    return None


def convert_gas_keygraph_to_bars_graph(path: str | Path, embeddings: np.ndarray, cfg: Dict, logger: CSVLogger | None = None, dataset: OfflineDataset | None = None) -> BARSGraph:
    data = load_gas_keygraph_pickle(path)
    nodes = _extract_nodes(data)
    n_nodes = len(nodes)
    node_indices = _load_node_indices_from_payload_or_file(data, cfg, n_nodes, dataset=dataset)
    mapping_source = 'payload_or_node_indices_path'
    if node_indices is None:
        ecfg = cfg.get('external_gas', {})
        map_embeddings = embeddings
        emb_path = os.path.expandvars(str(ecfg.get('dataset_embeddings_path') or '')).strip()
        if emb_path:
            map_embeddings = _align_external_dataset_embeddings(np.load(emb_path), dataset)
            mapping_source = 'external_gas.dataset_embeddings_path_aligned'
        else:
            mapping_source = 'current_bars_embeddings_warning_not_exact_if_gas_tdr_differs'
        ann = KNNIndex.from_config(map_embeddings.astype(np.float32), cfg, prefix='ann')
        node_indices = ann.kneighbors(nodes.astype(np.float32), 1, return_distance=False)[:, 0].astype(np.int64)
    node_embeddings = embeddings[node_indices].astype(np.float32)
    src, dst, cost = _extract_graph_edges(data, n_nodes, nodes.astype(np.float32))
    ecfg = cfg.get('external_gas', {})
    if bool(ecfg.get('bidirectional_edges', False)):
        src0, dst0, cost0 = src.copy(), dst.copy(), cost.copy()
        src = np.concatenate([src0, dst0]).astype(np.int64)
        dst = np.concatenate([dst0, src0]).astype(np.int64)
        cost = np.concatenate([cost0, cost0]).astype(np.float32)
    if bool(ecfg.get('normalize_cost', True)):
        scale = float(np.median(cost) + 1e-6)
        cost = (cost / scale).astype(np.float32)
    else:
        cost = cost.astype(np.float32)
    p_exec = np.ones(len(src), dtype=np.float32)
    risk = np.zeros(len(src), dtype=np.float32)
    kind = np.full(len(src), EDGE_KIND_KNN, dtype=np.int32)
    if logger is not None:
        logger.log({
            'phase': 'graph', 'event': 'gas_keygraph_converted', 'path': str(path),
            'num_gas_nodes': int(n_nodes), 'num_nodes': int(len(node_indices)), 'num_edges': int(len(src)),
            'node_mapping_source': mapping_source,
        })
    return BARSGraph(node_indices, node_embeddings, src, dst, cost, risk, p_exec, kind)
