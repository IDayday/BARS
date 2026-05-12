from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import time

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from bars.common.logging import CSVLogger
from bars.data.trajectories import OfflineDataset

from .ann import KNNIndex
from .support import build_edge_lookup
from .types import BARSGraph


@dataclass
class BoundaryIndex:
    dep_hist: np.ndarray
    arr_hist: np.ndarray
    has_dep: np.ndarray
    has_arr: np.ndarray
    fallback_psi: float = 0.5
    edge_dir: np.ndarray | None = None
    direction_temperature: float = 1.0
    method: str = "unknown"

    def psi(self, prev_edge: int, next_edge: int) -> float:
        if self.has_arr[prev_edge] and self.has_dep[next_edge]:
            val = float(np.minimum(self.arr_hist[prev_edge], self.dep_hist[next_edge]).sum())
            return float(np.clip(val, 1e-4, 1.0))
        if self.edge_dir is not None:
            a = self.edge_dir[prev_edge]
            b = self.edge_dir[next_edge]
            dist2 = float(np.sum((a - b) ** 2))
            return float(np.clip(np.exp(-dist2 / max(self.direction_temperature, 1e-6)), 1e-4, 1.0))
        return float(self.fallback_psi)

    def boundary_cost(self, prev_edge: int, next_edge: int) -> float:
        return float(-np.log(np.clip(self.psi(prev_edge, next_edge), 1e-4, 1.0)))

    def save_npz(self, path: str) -> None:
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(
            path,
            dep_hist=self.dep_hist,
            arr_hist=self.arr_hist,
            has_dep=self.has_dep,
            has_arr=self.has_arr,
            fallback_psi=self.fallback_psi,
            edge_dir=np.asarray([]) if self.edge_dir is None else self.edge_dir,
            direction_temperature=self.direction_temperature,
            method=np.asarray(self.method),
        )

    @classmethod
    def load_npz(cls, path: str) -> "BoundaryIndex":
        d = np.load(path, allow_pickle=True)
        edge_dir = d["edge_dir"] if "edge_dir" in d and d["edge_dir"].size else None
        temp = float(d["direction_temperature"]) if "direction_temperature" in d else 1.0
        method = str(d["method"].item() if hasattr(d["method"], "item") else d["method"]) if "method" in d else "loaded"
        return cls(d["dep_hist"], d["arr_hist"], d["has_dep"], d["has_arr"], float(d["fallback_psi"]), edge_dir, temp, method)


def _normalize(x: np.ndarray) -> np.ndarray:
    return (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)).astype(np.float32)


def _prev_next_dirs(dataset: OfflineDataset, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = dataset.size
    z = np.asarray(embeddings, dtype=np.float32)
    prev_dir = np.zeros_like(z, dtype=np.float32)
    next_dir = np.zeros_like(z, dtype=np.float32)
    idx = np.arange(n)
    prev_idx = np.maximum(idx - 1, 0)
    next_idx = np.minimum(idx + 1, n - 1)
    same_prev = dataset.traj_id[prev_idx] == dataset.traj_id[idx]
    same_next = dataset.traj_id[next_idx] == dataset.traj_id[idx]
    prev_dir[same_prev] = z[idx[same_prev]] - z[prev_idx[same_prev]]
    next_dir[same_next] = z[next_idx[same_next]] - z[idx[same_next]]
    return _normalize(prev_dir), _normalize(next_dir)


def _direction_vectors(graph: BARSGraph) -> np.ndarray:
    return _normalize(graph.node_embeddings[graph.dst] - graph.node_embeddings[graph.src])


def _direction_boundary(graph: BARSGraph, cfg: Dict, logger: CSVLogger) -> BoundaryIndex:
    bcfg = cfg.get("boundary", {})
    edge_dir = _direction_vectors(graph)
    dep = np.zeros((graph.num_edges, 1), dtype=np.float32)
    arr = np.zeros((graph.num_edges, 1), dtype=np.float32)
    has = np.zeros(graph.num_edges, dtype=bool)
    b = BoundaryIndex(
        dep,
        arr,
        has,
        has.copy(),
        float(bcfg.get("fallback_psi", 0.5)),
        edge_dir=edge_dir,
        direction_temperature=float(bcfg.get("direction_temperature", 1.0)),
        method="direction",
    )
    out = graph.outgoing_edges()
    checked = 0
    psi_sum = 0.0
    progress_every = max(1, graph.num_edges // 5)
    logger.log({"phase": "boundary", "event": "start", "method": "direction", "num_edges": graph.num_edges})
    for eid in range(graph.num_edges):
        for ne in out[int(graph.dst[eid])]:
            checked += 1
            psi_sum += b.psi(eid, int(ne))
        if ((eid + 1) % progress_every) == 0 or eid == graph.num_edges - 1:
            logger.log({"phase": "boundary", "event": "progress", "method": "direction", "processed_edges": eid + 1, "num_edges": graph.num_edges, "composable_pairs_checked": checked})
    logger.log({"phase": "boundary", "event": "completed", "method": "direction", "composable_pairs": checked, "psi_mean": psi_sum / max(1, checked), "direction_temperature": b.direction_temperature})
    return b


def _support_binary_boundary(dataset: OfflineDataset, graph: BARSGraph, cfg: Dict, logger: CSVLogger) -> BoundaryIndex:
    bcfg = cfg.get("boundary", {})
    horizon = int(bcfg.get("segment_horizon", cfg.get("reachability", {}).get("horizon", 30)))
    dep = np.zeros((graph.num_edges, 1), dtype=np.float32)
    arr = np.zeros((graph.num_edges, 1), dtype=np.float32)
    has_dep = np.zeros(graph.num_edges, dtype=bool)
    has_arr = np.zeros(graph.num_edges, dtype=bool)
    gi_src = graph.node_indices[graph.src]
    gi_dst = graph.node_indices[graph.dst]
    same = dataset.traj_id[gi_src] == dataset.traj_id[gi_dst]
    dt = dataset.timestep[gi_dst] - dataset.timestep[gi_src]
    ok = same & (dt > 0) & (dt <= horizon)
    dep[ok, 0] = 1.0
    arr[ok, 0] = 1.0
    has_dep[ok] = True
    has_arr[ok] = True
    b = BoundaryIndex(dep, arr, has_dep, has_arr, float(bcfg.get("fallback_psi", 0.5)), method="support_binary")
    logger.log({"phase": "boundary", "event": "completed", "method": "support_binary", "edge_has_support_rate": float(ok.mean()) if len(ok) else 0.0, "supported_edges": int(ok.sum())})
    return b


def _fit_node_mode_models(dataset: OfflineDataset, embeddings: np.ndarray, graph: BARSGraph, cfg: Dict, logger: CSVLogger):
    """Fit local portal modes for each graph node.

    The mode space is local to each graph node. Compatibility is only evaluated
    at a shared middle node, so local labels are sufficient and avoid a brittle
    global mode clustering.  Features combine local latent position and local
    trajectory direction; this makes modes spatially grounded rather than pure
    direction smoothing.
    """
    bcfg = cfg.get("boundary", {})
    num_modes = int(bcfg.get("num_modes", 4))
    support_k = int(bcfg.get("support_k", 64))
    seed = int(cfg.get("seed", 0))
    z = np.asarray(embeddings, dtype=np.float32)
    prev_dir, next_dir = _prev_next_dirs(dataset, z)
    ann = KNNIndex.from_config(z, cfg, prefix="ann")
    local_ids = ann.kneighbors(graph.node_embeddings, min(support_k, dataset.size), return_distance=False)

    centers: List[Optional[np.ndarray]] = [None] * graph.num_nodes
    scales = np.ones(graph.num_nodes, dtype=np.float32)
    node_has = np.zeros(graph.num_nodes, dtype=bool)
    progress_every = max(1, graph.num_nodes // 5)
    t0 = time.time()
    for ni in range(graph.num_nodes):
        ids = local_ids[ni]
        rel = z[ids] - graph.node_embeddings[ni]
        scale = float(np.median(np.linalg.norm(rel, axis=1)) + 1e-6)
        scales[ni] = scale
        feat = np.concatenate([rel / scale, prev_dir[ids], next_dir[ids]], axis=1).astype(np.float32)
        ok = np.linalg.norm(feat, axis=1) > 1e-6
        feat = feat[ok]
        if len(feat) == 0:
            continue
        k = min(num_modes, len(feat))
        if k <= 1:
            c = feat[:1].astype(np.float32)
        else:
            km = MiniBatchKMeans(n_clusters=k, random_state=seed + ni, batch_size=512, n_init=1, max_iter=int(bcfg.get("mode_kmeans_max_iter", 50))).fit(feat)
            c = km.cluster_centers_.astype(np.float32)
        if c.shape[0] < num_modes:
            pad = np.repeat(c[:1], repeats=num_modes - c.shape[0], axis=0) if len(c) else np.zeros((num_modes, feat.shape[1]), dtype=np.float32)
            c = np.concatenate([c, pad], axis=0) if len(c) else pad
        centers[ni] = c[:num_modes]
        node_has[ni] = True
        if ((ni + 1) % progress_every) == 0 or ni == graph.num_nodes - 1:
            logger.log({"phase": "boundary", "event": "mode_fit_progress", "processed_nodes": ni + 1, "num_nodes": graph.num_nodes})
    logger.log({"phase": "boundary", "event": "mode_fit_completed", "node_has_mode_rate": float(node_has.mean()) if len(node_has) else 0.0, "duration_sec": time.time() - t0})
    return centers, scales, node_has, prev_dir, next_dir


def _assign_mode(centers: List[Optional[np.ndarray]], scales: np.ndarray, graph: BARSGraph, z: np.ndarray, prev_dir: np.ndarray, next_dir: np.ndarray, node: int, idx: int) -> Optional[int]:
    c = centers[node]
    if c is None:
        return None
    rel = (z[idx] - graph.node_embeddings[node]) / max(float(scales[node]), 1e-6)
    feat = np.concatenate([rel, prev_dir[idx], next_dir[idx]], axis=0).astype(np.float32)
    d2 = np.sum((c - feat[None, :]) ** 2, axis=1)
    return int(np.argmin(d2))


def _normalize_hist_rows(x: np.ndarray, counts: np.ndarray) -> np.ndarray:
    y = x.astype(np.float32, copy=True)
    nz = counts > 0
    y[nz] /= counts[nz, None].astype(np.float32)
    return y


def _support_mode_boundary(dataset: OfflineDataset, embeddings: np.ndarray, graph: BARSGraph, cfg: Dict, logger: CSVLogger) -> BoundaryIndex:
    """Trajectory-support portal-mode boundary overlap.

    For sampled same-trajectory segments (s_t, s_{t+k}) that map to an existing
    graph edge e=(u,v), we record:
      dep_hist[e] += local portal mode of s_t at node u
      arr_hist[e] += local portal mode of s_{t+k} at node v

    Compatibility psi((u,z),(z,v)) is then histogram overlap between the arrival
    modes observed for the first edge at z and the departure modes observed for
    the second edge at the same z.  This is the intended BARS boundary signal;
    direction smoothing remains only a fallback for unsupported edges.
    """
    bcfg = cfg.get("boundary", {})
    horizon = int(bcfg.get("segment_horizon", cfg.get("reachability", {}).get("horizon", 30)))
    min_dt = int(bcfg.get("segment_min_dt", 1))
    num_modes = int(bcfg.get("num_modes", 4))
    num_segments = int(bcfg.get("support_segments", 200000))
    batch_size = int(bcfg.get("support_batch_size", 65536))
    fallback_psi = float(bcfg.get("fallback_psi", 0.5))
    seed = int(cfg.get("seed", 0))
    rng = np.random.default_rng(seed + 919)
    z = np.asarray(embeddings, dtype=np.float32)
    logger.log({"phase": "boundary", "event": "start", "method": "support_modes", "num_edges": graph.num_edges, "num_modes": num_modes, "support_segments": num_segments})

    centers, scales, node_has, prev_dir, next_dir = _fit_node_mode_models(dataset, z, graph, cfg, logger)
    node_ann = KNNIndex.from_config(graph.node_embeddings, cfg, prefix="ann")
    lookup = build_edge_lookup(graph)
    dep = np.zeros((graph.num_edges, num_modes), dtype=np.float32)
    arr = np.zeros((graph.num_edges, num_modes), dtype=np.float32)
    dep_count = np.zeros(graph.num_edges, dtype=np.int32)
    arr_count = np.zeros(graph.num_edges, dtype=np.int32)

    sampled = 0
    edge_hits = 0
    mode_hits = 0
    progress_every = max(batch_size, max(1, num_segments // 4))
    next_progress = progress_every
    t0 = time.time()
    while sampled < num_segments:
        bs = min(batch_size, num_segments - sampled)
        try:
            src_idx, dst_idx, _ = dataset.sample_future_pairs(bs, horizon, rng, min_dt=min_dt)
        except Exception:
            break
        src_node = node_ann.kneighbors(z[src_idx], 1, return_distance=False)[:, 0].astype(np.int64)
        dst_node = node_ann.kneighbors(z[dst_idx], 1, return_distance=False)[:, 0].astype(np.int64)
        for i, j, u, v in zip(src_idx, dst_idx, src_node, dst_node):
            eid = lookup.get((int(u), int(v)))
            if eid is None:
                continue
            edge_hits += 1
            if not (node_has[int(u)] and node_has[int(v)]):
                continue
            md = _assign_mode(centers, scales, graph, z, prev_dir, next_dir, int(u), int(i))
            ma = _assign_mode(centers, scales, graph, z, prev_dir, next_dir, int(v), int(j))
            if md is not None:
                dep[eid, md] += 1.0
                dep_count[eid] += 1
            if ma is not None:
                arr[eid, ma] += 1.0
                arr_count[eid] += 1
            if md is not None and ma is not None:
                mode_hits += 1
        sampled += bs
        if sampled >= next_progress:
            logger.log({"phase": "boundary", "event": "support_progress", "sampled_segments": sampled, "target_segments": num_segments, "edge_hits": edge_hits, "mode_hits": mode_hits})
            next_progress += progress_every

    dep = _normalize_hist_rows(dep, dep_count)
    arr = _normalize_hist_rows(arr, arr_count)
    has_dep = dep_count >= int(bcfg.get("min_edge_support_count", 1))
    has_arr = arr_count >= int(bcfg.get("min_edge_support_count", 1))
    edge_dir = _direction_vectors(graph) if bool(bcfg.get("direction_fallback", True)) else None
    b = BoundaryIndex(dep, arr, has_dep, has_arr, fallback_psi, edge_dir=edge_dir, direction_temperature=float(bcfg.get("direction_temperature", 1.0)), method="support_modes")
    logger.log({
        "phase": "boundary",
        "event": "completed",
        "method": "support_modes",
        "sampled_segments": sampled,
        "edge_hits": edge_hits,
        "mode_hits": mode_hits,
        "edge_hit_rate": float(edge_hits / max(1, sampled)),
        "mode_hit_rate": float(mode_hits / max(1, sampled)),
        "edge_has_dep_rate": float(has_dep.mean()) if len(has_dep) else 0.0,
        "edge_has_arr_rate": float(has_arr.mean()) if len(has_arr) else 0.0,
        "duration_sec": time.time() - t0,
    })
    return b


def build_boundary_index(dataset: OfflineDataset, embeddings: np.ndarray, graph: BARSGraph, cfg: Dict, logger: CSVLogger) -> BoundaryIndex:
    method = str(cfg.get("boundary", {}).get("method", "direction")).lower()
    if method in {"none", "disabled", "off"}:
        dep = np.zeros((graph.num_edges, 1), dtype=np.float32)
        arr = np.zeros((graph.num_edges, 1), dtype=np.float32)
        has = np.zeros(graph.num_edges, dtype=bool)
        logger.log({"phase": "boundary", "event": "completed", "method": "none", "num_edges": graph.num_edges})
        return BoundaryIndex(dep, arr, has, has.copy(), method="none")
    if method == "direction":
        return _direction_boundary(graph, cfg, logger)
    if method in {"support", "support_binary"}:
        return _support_binary_boundary(dataset, graph, cfg, logger)
    if method in {"portal", "portal_mode", "portal_mode_overlap"}:
        # Backward-compatible local-mode proxy. Prefer support_modes for actual
        # arrival/departure support overlap.
        return _support_mode_boundary(dataset, embeddings, graph, cfg, logger)
    if method in {"support_modes", "support_mode", "portal_support", "portal_support_modes", "support_mode_overlap"}:
        return _support_mode_boundary(dataset, embeddings, graph, cfg, logger)
    raise ValueError(f"Unknown boundary.method={method}")
