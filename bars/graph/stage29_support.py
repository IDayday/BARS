from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from bars.data.trajectories import OfflineDataset
from bars.graph.ann import KNNIndex
from bars.graph.edges import _deduplicate_edges
from bars.graph.planner import PlanResult
from bars.graph.types import BARSGraph, EDGE_KIND_BRIDGE, EDGE_KIND_KNN, EDGE_KIND_TEMPORAL


TEMPORAL_BACKBONE = 0
PROJECTED_TEMPORAL_SUPPORT = 1
SUPPORTED_CROSS_BRIDGE = 2
CROSS_TRAJ_CANDIDATE = 3
UNSUPPORTED_SHORTCUT = 4

EDGE_TYPE_NAMES = {
    TEMPORAL_BACKBONE: "TEMPORAL_BACKBONE",
    PROJECTED_TEMPORAL_SUPPORT: "PROJECTED_TEMPORAL_SUPPORT",
    SUPPORTED_CROSS_BRIDGE: "SUPPORTED_CROSS_BRIDGE",
    CROSS_TRAJ_CANDIDATE: "CROSS_TRAJ_CANDIDATE",
    UNSUPPORTED_SHORTCUT: "UNSUPPORTED_SHORTCUT",
}

_TYPE_PRIORITY = {
    TEMPORAL_BACKBONE: 0,
    PROJECTED_TEMPORAL_SUPPORT: 1,
    SUPPORTED_CROSS_BRIDGE: 2,
    CROSS_TRAJ_CANDIDATE: 3,
    UNSUPPORTED_SHORTCUT: 4,
}


@dataclass
class SupportEvidenceGraph:
    graph: BARSGraph
    edge_type: np.ndarray
    support_count: np.ndarray
    support_density: np.ndarray
    support_score: np.ndarray
    support_risk: np.ndarray
    execution_risk: np.ndarray
    unsupported_shortcut: np.ndarray
    protected_node: np.ndarray
    projected_node: np.ndarray
    support_out_count: np.ndarray
    metadata: Dict[str, float | int | str]


@dataclass
class SupportPlanResult:
    plan: PlanResult
    unsupported_edges: int
    support_risk: float
    cross_support_risk: float
    execution_risk: float
    path_cross_edge_rate: float
    path_temporal_backbone_rate: float
    path_projected_support_rate: float
    path_supported_bridge_rate: float
    path_candidate_rate: float
    path_unsupported_shortcut_rate: float
    path_min_support_score: float
    path_mean_support_score: float
    path_mean_execution_risk: float
    path_max_execution_risk: float
    path_temporal_dt_mean: float
    path_temporal_dt_max: float

    def to_row(self) -> Dict[str, float | int | str]:
        row = self.plan.to_row()
        row.update(
            {
                "unsupported_edges": int(self.unsupported_edges),
                "support_risk": float(self.support_risk),
                "cross_support_risk": float(self.cross_support_risk),
                "execution_risk": float(self.execution_risk),
                "path_cross_edge_rate": float(self.path_cross_edge_rate),
                "path_temporal_backbone_rate": float(self.path_temporal_backbone_rate),
                "path_projected_support_rate": float(self.path_projected_support_rate),
                "path_supported_bridge_rate": float(self.path_supported_bridge_rate),
                "path_candidate_rate": float(self.path_candidate_rate),
                "path_unsupported_shortcut_rate": float(self.path_unsupported_shortcut_rate),
                "path_min_support_score": float(self.path_min_support_score),
                "path_mean_support_score": float(self.path_mean_support_score),
                "path_mean_execution_risk": float(self.path_mean_execution_risk),
                "path_max_execution_risk": float(self.path_max_execution_risk),
                "path_temporal_dt_mean": float(self.path_temporal_dt_mean),
                "path_temporal_dt_max": float(self.path_temporal_dt_max),
            }
        )
        return row


def _cfg(cfg: Dict, key: str, default=None):
    return cfg.get("stage29_support", {}).get(key, default)


def _as_int_list(value, default: Sequence[int]) -> list[int]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [int(x.strip()) for x in value.split(",") if x.strip()]
    return [int(x) for x in value]


def _endpoint_indices(dataset: OfflineDataset, window: int = 1) -> np.ndarray:
    out: list[int] = []
    w = max(1, int(window))
    for sl in dataset.traj_slices:
        if sl.end <= sl.start:
            continue
        for i in range(sl.start, min(sl.end, sl.start + w)):
            out.append(int(i))
        for i in range(max(sl.start, sl.end - w), sl.end):
            out.append(int(i))
    return np.unique(np.asarray(out, dtype=np.int64)) if out else np.empty(0, dtype=np.int64)


def _project_all(embeddings: np.ndarray, node_embeddings: np.ndarray, cfg: Dict) -> np.ndarray:
    ann = KNNIndex.from_config(node_embeddings, cfg, prefix="ann")
    batch_size = int(_cfg(cfg, "projection_batch_size", 131072))
    return ann.kneighbors(np.asarray(embeddings, dtype=np.float32), 1, return_distance=False, batch_size=batch_size).reshape(-1).astype(np.int64)


def _projection_change_nodes(dataset: OfflineDataset, projection: np.ndarray, top_k: int) -> np.ndarray:
    if top_k <= 0:
        return np.empty(0, dtype=np.int64)
    scores = np.zeros(dataset.size, dtype=np.float32)
    for sl in dataset.traj_slices:
        if sl.end - sl.start < 2:
            continue
        ids = np.arange(sl.start, sl.end, dtype=np.int64)
        p = projection[ids]
        change = np.flatnonzero(p[1:] != p[:-1])
        if len(change):
            scores[ids[change]] += 1.0
            scores[ids[change + 1]] += 1.0
    nz = np.flatnonzero(scores > 0)
    if len(nz) == 0:
        return np.empty(0, dtype=np.int64)
    if len(nz) > top_k:
        keep = nz[np.argpartition(-scores[nz], top_k - 1)[:top_k]]
        keep = keep[np.argsort(-scores[keep], kind="stable")]
    else:
        keep = nz[np.argsort(-scores[nz], kind="stable")]
    return keep.astype(np.int64)


def _even_cap(values: np.ndarray, max_count: int) -> np.ndarray:
    values = np.unique(np.asarray(values, dtype=np.int64))
    if max_count <= 0 or len(values) <= max_count:
        return values
    pos = np.linspace(0, len(values) - 1, int(max_count))
    return values[np.unique(np.round(pos).astype(np.int64))]


def _trajectory_anchor_nodes(dataset: OfflineDataset, stride: int, max_count: int) -> np.ndarray:
    stride = max(1, int(stride))
    parts: list[np.ndarray] = []
    for sl in dataset.traj_slices:
        if sl.end <= sl.start:
            continue
        ids = np.arange(sl.start, sl.end, stride, dtype=np.int64)
        if len(ids) == 0 or ids[-1] != sl.end - 1:
            ids = np.r_[ids, sl.end - 1]
        parts.append(ids.astype(np.int64))
    if not parts:
        return np.empty(0, dtype=np.int64)
    return _even_cap(np.concatenate(parts), int(max_count))


def select_stage29_nodes(dataset: OfflineDataset, embeddings: np.ndarray, base_graph: BARSGraph, cfg: Dict) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    base_nodes = np.unique(np.asarray(base_graph.node_indices, dtype=np.int64))
    endpoint_window = int(_cfg(cfg, "endpoint_window", 1))
    endpoints = _even_cap(_endpoint_indices(dataset, endpoint_window), int(_cfg(cfg, "endpoint_max_nodes", 4096)))
    anchors = _trajectory_anchor_nodes(
        dataset,
        stride=int(_cfg(cfg, "trajectory_anchor_stride", 20)),
        max_count=int(_cfg(cfg, "trajectory_anchor_nodes", 12000)),
    )
    base_projection = _project_all(embeddings, base_graph.node_embeddings, cfg)
    change_nodes = _projection_change_nodes(dataset, base_projection, int(_cfg(cfg, "projection_change_nodes", 1024)))
    bottleneck_nodes = _projection_change_nodes(dataset, base_projection, int(_cfg(cfg, "bottleneck_nodes", 512)))

    priority_parts = [endpoints, anchors, change_nodes, bottleneck_nodes]
    protected_all = np.unique(np.concatenate([x for x in priority_parts if len(x)])) if any(len(x) for x in priority_parts) else np.empty(0, dtype=np.int64)
    max_nodes = int(_cfg(cfg, "max_nodes", 8192))
    if max_nodes > 0:
        remaining = max(0, max_nodes - len(base_nodes))
        extra_chunks: list[np.ndarray] = []
        seen = set(int(x) for x in base_nodes.tolist())
        for part in priority_parts:
            if remaining <= 0:
                break
            vals = [int(x) for x in part.tolist() if int(x) not in seen]
            if not vals:
                continue
            take = vals[:remaining]
            extra_chunks.append(np.asarray(take, dtype=np.int64))
            seen.update(take)
            remaining -= len(take)
        if extra_chunks:
            node_indices = np.unique(np.concatenate([base_nodes, *extra_chunks])).astype(np.int64)
        else:
            node_indices = base_nodes
    else:
        node_indices = np.unique(np.concatenate([base_nodes, protected_all])).astype(np.int64)
    protected_set = set(int(x) for x in protected_all.tolist())
    protected_mask = np.asarray([int(x) in protected_set for x in node_indices], dtype=bool)
    meta = {
        "base_nodes": int(len(base_nodes)),
        "endpoint_candidates": int(len(endpoints)),
        "trajectory_anchor_candidates": int(len(anchors)),
        "projection_change_candidates": int(len(change_nodes)),
        "bottleneck_candidates": int(len(bottleneck_nodes)),
        "protected_candidates": int(len(protected_all)),
        "stage29_nodes": int(len(node_indices)),
        "protected_nodes": int(protected_mask.sum()),
    }
    return node_indices, protected_mask, meta


def _support_counts_from_projection(dataset: OfflineDataset, projection: np.ndarray, num_nodes: int, offsets: Sequence[int]) -> Tuple[Dict[int, int], np.ndarray, int]:
    counts: Dict[int, int] = {}
    out_count = np.zeros(num_nodes, dtype=np.int64)
    total = 0
    for raw_off in offsets:
        off = int(raw_off)
        if off <= 0 or off >= dataset.size:
            continue
        src_idx = np.arange(0, dataset.size - off, dtype=np.int64)
        dst_idx = src_idx + off
        same = dataset.traj_id[src_idx] == dataset.traj_id[dst_idx]
        if not same.any():
            continue
        u = projection[src_idx[same]]
        v = projection[dst_idx[same]]
        ok = u != v
        if not ok.any():
            continue
        u = u[ok].astype(np.int64)
        v = v[ok].astype(np.int64)
        total += int(len(u))
        out_count += np.bincount(u, minlength=num_nodes).astype(np.int64)
        keys = u * int(num_nodes) + v
        uniq, cnt = np.unique(keys, return_counts=True)
        for k, c in zip(uniq.tolist(), cnt.tolist()):
            counts[int(k)] = counts.get(int(k), 0) + int(c)
    return counts, out_count, total


def _support_arrays(src: np.ndarray, dst: np.ndarray, support_counts: Dict[int, int], support_out_count: np.ndarray, cfg: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(len(support_out_count))
    count = np.asarray([support_counts.get(int(u) * n + int(v), 0) for u, v in zip(src, dst)], dtype=np.float32)
    out = np.maximum(1.0, support_out_count[src].astype(np.float32))
    density = count / out
    count_scale = float(_cfg(cfg, "support_count_scale", 4.0))
    density_scale = float(_cfg(cfg, "support_density_scale", 0.02))
    count_term = 1.0 - np.exp(-count / max(count_scale, 1e-6))
    density_term = np.clip(density / max(density_scale, 1e-8), 0.0, 1.0)
    count_weight = float(_cfg(cfg, "support_count_weight", 0.65))
    score = np.clip(count_weight * count_term + (1.0 - count_weight) * density_term, 0.0, 1.0)
    return count.astype(np.int32), density.astype(np.float32), score.astype(np.float32)


def _edge_type_for(
    dataset: OfflineDataset,
    node_indices: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    kind_hint: np.ndarray,
    support_count: np.ndarray,
    support_score: np.ndarray,
    protected_node: np.ndarray,
    cfg: Dict,
) -> np.ndarray:
    same_traj = dataset.traj_id[node_indices[src]] == dataset.traj_id[node_indices[dst]]
    dt = dataset.timestep[node_indices[dst]] - dataset.timestep[node_indices[src]]
    horizon = int(_cfg(cfg, "temporal_support_horizon", cfg.get("reachability", {}).get("horizon", 50)))
    temporal = same_traj & (dt > 0) & (dt <= horizon)
    cross = ~same_traj
    support_threshold = float(_cfg(cfg, "supported_bridge_score", 0.35))
    candidate_threshold = float(_cfg(cfg, "candidate_bridge_score", 0.08))
    protected_edge = protected_node[src] | protected_node[dst]
    edge_type = np.full(len(src), CROSS_TRAJ_CANDIDATE, dtype=np.int32)
    edge_type[temporal | (kind_hint == EDGE_KIND_TEMPORAL)] = TEMPORAL_BACKBONE
    projected = (support_count > 0) & ~cross & (edge_type != TEMPORAL_BACKBONE)
    edge_type[projected] = PROJECTED_TEMPORAL_SUPPORT
    supported_cross = cross & (support_score >= support_threshold)
    edge_type[supported_cross] = SUPPORTED_CROSS_BRIDGE
    unsupported = cross & (support_score < candidate_threshold) & ~protected_edge
    edge_type[unsupported] = UNSUPPORTED_SHORTCUT
    candidate = cross & (edge_type != SUPPORTED_CROSS_BRIDGE) & (edge_type != UNSUPPORTED_SHORTCUT)
    edge_type[candidate] = CROSS_TRAJ_CANDIDATE
    return edge_type


def _add_or_update(
    store: Dict[int, tuple[float, float, float, int, int]],
    num_nodes: int,
    src: np.ndarray,
    dst: np.ndarray,
    cost: np.ndarray,
    risk: np.ndarray,
    p_exec: np.ndarray,
    kind: np.ndarray,
    source_priority: int,
) -> None:
    for u, v, c, r, p, k in zip(src, dst, cost, risk, p_exec, kind):
        u = int(u)
        v = int(v)
        if u == v:
            continue
        key = u * int(num_nodes) + v
        cand = (float(c), float(r), float(p), int(k), int(source_priority))
        old = store.get(key)
        if old is None or source_priority < old[4] or (source_priority == old[4] and float(c) < old[0]):
            store[key] = cand


def _base_edges_in_node_set(base_graph: BARSGraph, node_lookup: Dict[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    src_parts = []
    dst_parts = []
    cost_parts = []
    risk_parts = []
    p_parts = []
    kind_parts = []
    for eid, (u, v) in enumerate(zip(base_graph.src, base_graph.dst)):
        gu = int(base_graph.node_indices[int(u)])
        gv = int(base_graph.node_indices[int(v)])
        if gu not in node_lookup or gv not in node_lookup:
            continue
        src_parts.append(node_lookup[gu])
        dst_parts.append(node_lookup[gv])
        cost_parts.append(float(base_graph.cost[eid]))
        risk_parts.append(float(base_graph.risk[eid]))
        p_parts.append(float(base_graph.p_exec[eid]))
        kind_parts.append(int(base_graph.kind[eid]))
    if not src_parts:
        empty_i = np.empty(0, dtype=np.int64)
        empty_f = np.empty(0, dtype=np.float32)
        return empty_i, empty_i, empty_f, empty_f, empty_f, np.empty(0, dtype=np.int32)
    return (
        np.asarray(src_parts, dtype=np.int64),
        np.asarray(dst_parts, dtype=np.int64),
        np.asarray(cost_parts, dtype=np.float32),
        np.asarray(risk_parts, dtype=np.float32),
        np.asarray(p_parts, dtype=np.float32),
        np.asarray(kind_parts, dtype=np.int32),
    )


def _temporal_backbone_edges(dataset: OfflineDataset, node_indices: np.ndarray, cfg: Dict) -> tuple[np.ndarray, np.ndarray]:
    n = len(node_indices)
    if n < 2:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    connect = int(_cfg(cfg, "temporal_backbone_connect", 8))
    horizon = int(_cfg(cfg, "temporal_support_horizon", cfg.get("reachability", {}).get("horizon", 50)))
    local = np.arange(n, dtype=np.int64)
    traj = dataset.traj_id[node_indices]
    ts = dataset.timestep[node_indices]
    order = np.lexsort((ts, traj))
    local_s = local[order]
    traj_s = traj[order]
    ts_s = ts[order]
    src_parts: list[np.ndarray] = []
    dst_parts: list[np.ndarray] = []
    for off in range(1, connect + 1):
        if off >= n:
            break
        same = traj_s[:-off] == traj_s[off:]
        dt = ts_s[off:] - ts_s[:-off]
        ok = same & (dt > 0) & (dt <= horizon)
        if ok.any():
            src_parts.append(local_s[:-off][ok])
            dst_parts.append(local_s[off:][ok])
    if not src_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(src_parts).astype(np.int64), np.concatenate(dst_parts).astype(np.int64)


def _projected_support_edges(support_counts: Dict[int, int], support_out_count: np.ndarray, cfg: Dict) -> tuple[np.ndarray, np.ndarray]:
    n = int(len(support_out_count))
    if not support_counts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    min_count = int(_cfg(cfg, "projected_support_min_count", 1))
    top_per_src = int(_cfg(cfg, "projected_support_top_per_src", 24))
    by_src: dict[int, list[tuple[int, int]]] = {}
    for key, count in support_counts.items():
        if count < min_count:
            continue
        u = int(key // n)
        v = int(key % n)
        if u == v:
            continue
        by_src.setdefault(u, []).append((v, int(count)))
    src_parts = []
    dst_parts = []
    for u, vals in by_src.items():
        vals = sorted(vals, key=lambda x: (-x[1], x[0]))
        if top_per_src > 0:
            vals = vals[:top_per_src]
        src_parts.extend([u] * len(vals))
        dst_parts.extend([v for v, _ in vals])
    if not src_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.asarray(src_parts, dtype=np.int64), np.asarray(dst_parts, dtype=np.int64)


def build_support_evidence_graph(dataset: OfflineDataset, embeddings: np.ndarray, base_graph: BARSGraph, cfg: Dict) -> SupportEvidenceGraph:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    node_indices, protected_node, node_meta = select_stage29_nodes(dataset, embeddings, base_graph, cfg)
    node_embeddings = embeddings[node_indices].astype(np.float32)
    node_lookup = {int(idx): i for i, idx in enumerate(node_indices.tolist())}
    projection = _project_all(embeddings, node_embeddings, cfg)
    offsets = _as_int_list(_cfg(cfg, "projected_support_offsets", [1, 2, 4, 8, 16, 32, 64]), [1, 2, 4, 8, 16, 32, 64])
    support_counts, support_out_count, projected_segments = _support_counts_from_projection(dataset, projection, len(node_indices), offsets)

    store: Dict[int, tuple[float, float, float, int, int]] = {}
    base_src, base_dst, base_cost, base_risk, base_p, base_kind = _base_edges_in_node_set(base_graph, node_lookup)
    if len(base_src):
        _add_or_update(store, len(node_indices), base_src, base_dst, base_cost, base_risk, base_p, base_kind, source_priority=3)

    t_src, t_dst = _temporal_backbone_edges(dataset, node_indices, cfg)
    if len(t_src):
        dt = dataset.timestep[node_indices[t_dst]] - dataset.timestep[node_indices[t_src]]
        horizon = int(_cfg(cfg, "temporal_support_horizon", cfg.get("reachability", {}).get("horizon", 50)))
        cost = (np.log1p(np.maximum(dt, 1)) / np.log1p(max(horizon, 1))).astype(np.float32)
        p = np.ones(len(t_src), dtype=np.float32)
        risk = np.zeros(len(t_src), dtype=np.float32)
        kind = np.full(len(t_src), EDGE_KIND_TEMPORAL, dtype=np.int32)
        _add_or_update(store, len(node_indices), t_src, t_dst, cost, risk, p, kind, source_priority=0)

    ps_src, ps_dst = _projected_support_edges(support_counts, support_out_count, cfg)
    if len(ps_src):
        dist = np.linalg.norm(node_embeddings[ps_src] - node_embeddings[ps_dst], axis=1).astype(np.float32)
        scale = float(np.median(dist) + 1e-6) if len(dist) else 1.0
        cost = dist / max(scale, 1e-6)
        count, density, score = _support_arrays(ps_src, ps_dst, support_counts, support_out_count, cfg)
        p = np.clip(score, 1e-4, 1.0).astype(np.float32)
        risk = (-np.log(p)).astype(np.float32)
        kind = np.full(len(ps_src), EDGE_KIND_BRIDGE, dtype=np.int32)
        _add_or_update(store, len(node_indices), ps_src, ps_dst, cost, risk, p, kind, source_priority=1)

    if not store:
        raise RuntimeError("Stage29 evidence graph has no edges.")
    keys = np.fromiter(store.keys(), dtype=np.int64)
    src = (keys // int(len(node_indices))).astype(np.int64)
    dst = (keys % int(len(node_indices))).astype(np.int64)
    vals = [store[int(k)] for k in keys.tolist()]
    cost = np.asarray([x[0] for x in vals], dtype=np.float32)
    risk = np.asarray([x[1] for x in vals], dtype=np.float32)
    p_exec = np.asarray([x[2] for x in vals], dtype=np.float32)
    kind = np.asarray([x[3] for x in vals], dtype=np.int32)
    src, dst, kind = _deduplicate_edges(src, dst, kind, len(node_indices))
    # _deduplicate_edges can reorder/drop duplicates, so rebuild scalar arrays by key.
    keys = src * int(len(node_indices)) + dst
    vals = [store[int(k)] for k in keys.tolist()]
    cost = np.asarray([x[0] for x in vals], dtype=np.float32)
    risk = np.asarray([x[1] for x in vals], dtype=np.float32)
    p_exec = np.asarray([x[2] for x in vals], dtype=np.float32)
    support_count, support_density, support_score = _support_arrays(src, dst, support_counts, support_out_count, cfg)
    edge_type = _edge_type_for(dataset, node_indices, src, dst, kind, support_count, support_score, protected_node, cfg)
    unsupported = edge_type == UNSUPPORTED_SHORTCUT
    support_risk = _edge_support_risk(dataset, node_indices, src, dst, edge_type, support_score, support_count, cfg)
    execution_risk = _edge_execution_risk(dataset, node_indices, src, dst, edge_type, support_score, support_count, cfg)
    graph = BARSGraph(node_indices, node_embeddings, src, dst, cost, risk, p_exec, kind)
    metadata: Dict[str, float | int | str] = {
        **node_meta,
        "projected_support_segments": int(projected_segments),
        "projected_support_edges": int(len(support_counts)),
        "num_edges": int(graph.num_edges),
        "unsupported_shortcut_edges": int(unsupported.sum()),
        "unsupported_shortcut_edge_rate": float(unsupported.mean()) if len(unsupported) else 0.0,
    }
    return SupportEvidenceGraph(
        graph=graph,
        edge_type=edge_type.astype(np.int32),
        support_count=support_count.astype(np.int32),
        support_density=support_density.astype(np.float32),
        support_score=support_score.astype(np.float32),
        support_risk=support_risk.astype(np.float32),
        execution_risk=execution_risk.astype(np.float32),
        unsupported_shortcut=unsupported.astype(bool),
        protected_node=protected_node.astype(bool),
        projected_node=projection.astype(np.int64),
        support_out_count=support_out_count.astype(np.int64),
        metadata=metadata,
    )


def _edge_support_risk(
    dataset: OfflineDataset,
    node_indices: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    edge_type: np.ndarray,
    support_score: np.ndarray,
    support_count: np.ndarray,
    cfg: Dict,
) -> np.ndarray:
    same_traj = dataset.traj_id[node_indices[src]] == dataset.traj_id[node_indices[dst]]
    cross = ~same_traj
    bridge_thr = float(_cfg(cfg, "supported_bridge_score", 0.35))
    candidate_thr = float(_cfg(cfg, "candidate_bridge_score", 0.08))
    cross_base_risk = float(_cfg(cfg, "cross_edge_base_risk", 0.25))
    supported_cross_min_risk = float(_cfg(cfg, "supported_cross_min_risk", cross_base_risk))
    risk = np.zeros(len(src), dtype=np.float32)
    same_support_weight = float(_cfg(cfg, "same_traj_support_risk_weight", 0.0))
    same_zero_count_penalty = float(_cfg(cfg, "same_traj_zero_count_penalty", 0.0))
    same_floor = float(_cfg(cfg, "same_traj_support_min_risk", 0.0))
    if same_support_weight > 0.0 or same_zero_count_penalty > 0.0 or same_floor > 0.0:
        same = ~cross
        same_deficit = np.clip(1.0 - support_score, 0.0, 1.0)
        risk[same] = same_floor + same_support_weight * same_deficit[same]
        zero_count = same & (support_count <= 0)
        risk[zero_count] += same_zero_count_penalty
    cross_deficit = np.clip((bridge_thr - support_score) / max(bridge_thr, 1e-6), 0.0, 1.0)
    risk[cross] = cross_base_risk + cross_deficit[cross]
    supported = edge_type == SUPPORTED_CROSS_BRIDGE
    risk[supported] = np.maximum(supported_cross_min_risk, cross_base_risk * np.clip(1.0 - support_score[supported], 0.0, 1.0))
    unsupported = edge_type == UNSUPPORTED_SHORTCUT
    risk[unsupported] = np.maximum(risk[unsupported], 1.0 - np.clip(support_score[unsupported] / max(candidate_thr, 1e-6), 0.0, 1.0))
    return risk.astype(np.float32)


def _edge_execution_risk(
    dataset: OfflineDataset,
    node_indices: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    edge_type: np.ndarray,
    support_score: np.ndarray,
    support_count: np.ndarray,
    cfg: Dict,
) -> np.ndarray:
    same_traj = dataset.traj_id[node_indices[src]] == dataset.traj_id[node_indices[dst]]
    dt = dataset.timestep[node_indices[dst]] - dataset.timestep[node_indices[src]]
    deficit = np.clip(1.0 - support_score, 0.0, 1.0)
    risk = np.zeros(len(src), dtype=np.float32)

    temporal = same_traj & (edge_type == TEMPORAL_BACKBONE)
    projected = same_traj & (edge_type == PROJECTED_TEMPORAL_SUPPORT)
    cross = ~same_traj
    temporal_weight = float(_cfg(cfg, "temporal_execution_support_weight", 0.0))
    projected_weight = float(_cfg(cfg, "projected_execution_support_weight", temporal_weight))
    cross_weight = float(_cfg(cfg, "cross_execution_support_weight", 0.0))
    zero_penalty = float(_cfg(cfg, "execution_zero_support_penalty", 0.0))
    if temporal_weight:
        risk[temporal] += temporal_weight * deficit[temporal]
    if projected_weight:
        risk[projected] += projected_weight * deficit[projected]
    if cross_weight:
        risk[cross] += cross_weight * deficit[cross]
    if zero_penalty:
        risk[(temporal | projected) & (support_count <= 0)] += zero_penalty

    dt_weight = float(_cfg(cfg, "temporal_execution_dt_weight", 0.0))
    if dt_weight:
        horizon = float(_cfg(cfg, "temporal_support_horizon", cfg.get("reachability", {}).get("horizon", 50)))
        dt_ratio = np.clip(np.maximum(dt.astype(np.float32), 0.0) / max(horizon, 1e-6), 0.0, 2.0)
        risk[temporal] += dt_weight * dt_ratio[temporal]
    return risk.astype(np.float32)


def edge_type_name(edge_type: int) -> str:
    return EDGE_TYPE_NAMES.get(int(edge_type), "UNKNOWN")


def edge_type_counts(evidence: SupportEvidenceGraph) -> Dict[str, int]:
    out = {name: 0 for name in EDGE_TYPE_NAMES.values()}
    vals, counts = np.unique(evidence.edge_type, return_counts=True)
    for v, c in zip(vals.tolist(), counts.tolist()):
        out[edge_type_name(v)] = int(c)
    return out


def graph_support_summary(dataset: OfflineDataset, evidence: SupportEvidenceGraph) -> Dict[str, float | int | str]:
    graph = evidence.graph
    src = graph.src
    dst = graph.dst
    node_indices = graph.node_indices
    same = dataset.traj_id[node_indices[src]] == dataset.traj_id[node_indices[dst]]
    dt = dataset.timestep[node_indices[dst]] - dataset.timestep[node_indices[src]]
    temporal_supported = same & (dt > 0)
    endpoints = set(int(x) for x in _endpoint_indices(dataset, 1).tolist())
    node_set = set(int(x) for x in node_indices.tolist())
    endpoint_retention = float(np.mean([int(x) in node_set for x in endpoints])) if endpoints else float("nan")
    row: Dict[str, float | int | str] = {
        "num_nodes": int(graph.num_nodes),
        "num_edges": int(graph.num_edges),
        "protected_nodes": int(evidence.protected_node.sum()),
        "endpoint_exact_retention_rate": endpoint_retention,
        "cross_edge_rate": float((~same).mean()) if len(same) else 0.0,
        "temporal_supported_edge_rate": float(temporal_supported.mean()) if len(temporal_supported) else 0.0,
        "unsupported_shortcut_edge_rate": float(evidence.unsupported_shortcut.mean()) if graph.num_edges else 0.0,
        "support_count_positive_rate": float((evidence.support_count > 0).mean()) if graph.num_edges else 0.0,
        "support_score_mean": float(np.mean(evidence.support_score)) if graph.num_edges else float("nan"),
        "support_score_p50": float(np.quantile(evidence.support_score, 0.50)) if graph.num_edges else float("nan"),
        "support_score_p90": float(np.quantile(evidence.support_score, 0.90)) if graph.num_edges else float("nan"),
        "support_risk_mean": float(np.mean(evidence.support_risk)) if graph.num_edges else float("nan"),
        "execution_risk_mean": float(np.mean(evidence.execution_risk)) if graph.num_edges else float("nan"),
    }
    for name, count in edge_type_counts(evidence).items():
        row[f"edge_type_{name}_count"] = int(count)
        row[f"edge_type_{name}_rate"] = float(count / max(1, graph.num_edges))
    row.update(evidence.metadata)
    return row


def _empty_support_result(variant: str, exec_budget: Optional[float] = None) -> SupportPlanResult:
    return SupportPlanResult(
        plan=PlanResult(False, [], [], float("inf"), float("inf"), 0.0, float("inf"), variant, float("inf"), float(exec_budget) if exec_budget is not None else float("nan")),
        unsupported_edges=0,
        support_risk=float("inf"),
        cross_support_risk=float("inf"),
        execution_risk=float("inf"),
        path_cross_edge_rate=float("nan"),
        path_temporal_backbone_rate=float("nan"),
        path_projected_support_rate=float("nan"),
        path_supported_bridge_rate=float("nan"),
        path_candidate_rate=float("nan"),
        path_unsupported_shortcut_rate=float("nan"),
        path_min_support_score=float("nan"),
        path_mean_support_score=float("nan"),
        path_mean_execution_risk=float("nan"),
        path_max_execution_risk=float("nan"),
        path_temporal_dt_mean=float("nan"),
        path_temporal_dt_max=float("nan"),
    )


def _support_path_stats(dataset: OfflineDataset, evidence: SupportEvidenceGraph, edges: Sequence[int]) -> Dict[str, float | int]:
    if not edges:
        return {
            "unsupported_edges": 0,
            "support_risk": 0.0,
            "cross_support_risk": 0.0,
            "execution_risk": 0.0,
            "path_cross_edge_rate": float("nan"),
            "path_temporal_backbone_rate": float("nan"),
            "path_projected_support_rate": float("nan"),
            "path_supported_bridge_rate": float("nan"),
            "path_candidate_rate": float("nan"),
            "path_unsupported_shortcut_rate": float("nan"),
            "path_min_support_score": float("nan"),
            "path_mean_support_score": float("nan"),
            "path_mean_execution_risk": float("nan"),
            "path_max_execution_risk": float("nan"),
            "path_temporal_dt_mean": float("nan"),
            "path_temporal_dt_max": float("nan"),
        }
    e = np.asarray(edges, dtype=np.int64)
    graph = evidence.graph
    same = dataset.traj_id[graph.node_indices[graph.src[e]]] == dataset.traj_id[graph.node_indices[graph.dst[e]]]
    dt = dataset.timestep[graph.node_indices[graph.dst[e]]] - dataset.timestep[graph.node_indices[graph.src[e]]]
    et = evidence.edge_type[e]
    temporal_dt = dt[(et == TEMPORAL_BACKBONE) & (dt > 0)]
    return {
        "unsupported_edges": int(evidence.unsupported_shortcut[e].sum()),
        "support_risk": float(evidence.support_risk[e].sum()),
        "cross_support_risk": float(evidence.support_risk[e][~same].sum()) if (~same).any() else 0.0,
        "execution_risk": float(evidence.execution_risk[e].sum()),
        "path_cross_edge_rate": float((~same).mean()),
        "path_temporal_backbone_rate": float((et == TEMPORAL_BACKBONE).mean()),
        "path_projected_support_rate": float((et == PROJECTED_TEMPORAL_SUPPORT).mean()),
        "path_supported_bridge_rate": float((et == SUPPORTED_CROSS_BRIDGE).mean()),
        "path_candidate_rate": float((et == CROSS_TRAJ_CANDIDATE).mean()),
        "path_unsupported_shortcut_rate": float((et == UNSUPPORTED_SHORTCUT).mean()),
        "path_min_support_score": float(np.min(evidence.support_score[e])),
        "path_mean_support_score": float(np.mean(evidence.support_score[e])),
        "path_mean_execution_risk": float(np.mean(evidence.execution_risk[e])),
        "path_max_execution_risk": float(np.max(evidence.execution_risk[e])),
        "path_temporal_dt_mean": float(np.mean(temporal_dt)) if len(temporal_dt) else float("nan"),
        "path_temporal_dt_max": float(np.max(temporal_dt)) if len(temporal_dt) else float("nan"),
    }


def _make_support_plan_result(dataset: OfflineDataset, evidence: SupportEvidenceGraph, edges: list[int], nodes: list[int], objective: float, variant: str, exec_budget: Optional[float] = None) -> SupportPlanResult:
    graph = evidence.graph
    stats = _support_path_stats(dataset, evidence, edges)
    total_cost = float(graph.cost[edges].sum()) if edges else 0.0
    total_risk = float(graph.risk[edges].sum()) if edges else 0.0
    plan = PlanResult(True, nodes, edges, total_cost, total_risk, 0.0, float(objective), variant, float(stats["support_risk"]), float(exec_budget) if exec_budget is not None else float("nan"))
    return SupportPlanResult(
        plan=plan,
        unsupported_edges=int(stats["unsupported_edges"]),
        support_risk=float(stats["support_risk"]),
        cross_support_risk=float(stats["cross_support_risk"]),
        execution_risk=float(stats["execution_risk"]),
        path_cross_edge_rate=float(stats["path_cross_edge_rate"]),
        path_temporal_backbone_rate=float(stats["path_temporal_backbone_rate"]),
        path_projected_support_rate=float(stats["path_projected_support_rate"]),
        path_supported_bridge_rate=float(stats["path_supported_bridge_rate"]),
        path_candidate_rate=float(stats["path_candidate_rate"]),
        path_unsupported_shortcut_rate=float(stats["path_unsupported_shortcut_rate"]),
        path_min_support_score=float(stats["path_min_support_score"]),
        path_mean_support_score=float(stats["path_mean_support_score"]),
        path_mean_execution_risk=float(stats["path_mean_execution_risk"]),
        path_max_execution_risk=float(stats["path_max_execution_risk"]),
        path_temporal_dt_mean=float(stats["path_temporal_dt_mean"]),
        path_temporal_dt_max=float(stats["path_temporal_dt_max"]),
    )


def plan_support_lexicographic(
    dataset: OfflineDataset,
    evidence: SupportEvidenceGraph,
    start_node: int,
    goal_node: int,
    *,
    lambda_cost: float = 1.0,
    lambda_support_risk: float = 1.0,
    lambda_execution_risk: float = 1.0,
    max_edges: Optional[int] = None,
    variant: str = "stage29_lexicographic",
) -> SupportPlanResult:
    graph = evidence.graph
    if start_node == goal_node:
        return _make_support_plan_result(dataset, evidence, [], [int(start_node)], 0.0, variant)
    out = graph.outgoing_edges()
    max_hops = int(max_edges) if max_edges is not None and int(max_edges) > 0 else graph.num_nodes + 1
    dist: Dict[tuple[int, int], tuple[int, float, float]] = {(int(start_node), 0): (0, 0.0, 0.0)}
    prev: Dict[tuple[int, int], tuple[tuple[int, int], int]] = {}
    pq: list[tuple[int, float, float, int, int]] = [(0, 0.0, 0.0, int(start_node), 0)]
    best: Optional[tuple[int, int]] = None
    while pq:
        unsup, srisk, cost, u, depth = heapq.heappop(pq)
        state = (u, depth)
        if (unsup, srisk, cost) != dist.get(state, (math.inf, math.inf, math.inf)):
            continue
        if u == goal_node and depth > 0:
            best = state
            break
        if depth >= max_hops:
            continue
        for eid in out[u]:
            eid = int(eid)
            v = int(graph.dst[eid])
            nd = depth + 1
            ns = (
                int(unsup + int(evidence.unsupported_shortcut[eid])),
                float(srisk + lambda_support_risk * float(evidence.support_risk[eid])),
                float(cost + lambda_cost * float(graph.cost[eid]) + lambda_execution_risk * float(evidence.execution_risk[eid])),
            )
            st = (v, nd)
            if ns < dist.get(st, (math.inf, math.inf, math.inf)):
                dist[st] = ns
                prev[st] = (state, eid)
                heapq.heappush(pq, (ns[0], ns[1], ns[2], v, nd))
    if best is None:
        return _empty_support_result(variant)
    edges: list[int] = []
    nodes: list[int] = [goal_node]
    cur = best
    while cur != (int(start_node), 0):
        pstate, eid = prev[cur]
        edges.append(int(eid))
        nodes.append(int(pstate[0]))
        cur = pstate
    edges.reverse()
    nodes.reverse()
    objective = dist[best][2]
    return _make_support_plan_result(dataset, evidence, edges, nodes, objective, variant)


def plan_support_budgeted(
    dataset: OfflineDataset,
    evidence: SupportEvidenceGraph,
    start_node: int,
    goal_node: int,
    *,
    unsupported_budget: int,
    support_risk_budget: float,
    max_edges: Optional[int] = None,
    support_risk_bin: float = 0.05,
    lambda_execution_risk: float = 1.0,
    variant: Optional[str] = None,
) -> SupportPlanResult:
    graph = evidence.graph
    variant = variant or f"stage29_budgeted_k{unsupported_budget}_r{support_risk_budget:g}"
    if start_node == goal_node:
        return _make_support_plan_result(dataset, evidence, [], [int(start_node)], 0.0, variant, support_risk_budget)
    out = graph.outgoing_edges()
    max_hops = int(max_edges) if max_edges is not None and int(max_edges) > 0 else graph.num_nodes + 1
    bin_size = max(float(support_risk_bin), 1e-6)
    max_risk_bin = int(math.floor(float(support_risk_budget) / bin_size + 1e-9))
    start = (int(start_node), 0, 0, 0)
    dist: Dict[tuple[int, int, int, int], float] = {start: 0.0}
    actual_risk: Dict[tuple[int, int, int, int], float] = {start: 0.0}
    prev: Dict[tuple[int, int, int, int], tuple[tuple[int, int, int, int], int]] = {}
    pq: list[tuple[float, float, int, int, int, int]] = [(0.0, 0.0, int(start_node), 0, 0, 0)]
    best: Optional[tuple[int, int, int, int]] = None
    while pq:
        cost, srisk, u, depth, unsup, rbin = heapq.heappop(pq)
        state = (u, depth, unsup, rbin)
        if cost != dist.get(state, math.inf):
            continue
        if u == goal_node and depth > 0:
            best = state
            break
        if depth >= max_hops:
            continue
        for eid in out[u]:
            eid = int(eid)
            n_unsup = unsup + int(evidence.unsupported_shortcut[eid])
            if n_unsup > int(unsupported_budget):
                continue
            n_risk = float(srisk + float(evidence.support_risk[eid]))
            if n_risk > float(support_risk_budget) + 1e-7:
                continue
            n_bin = int(math.floor(n_risk / bin_size + 1e-9))
            if n_bin > max_risk_bin:
                continue
            v = int(graph.dst[eid])
            n_depth = depth + 1
            n_cost = float(cost + float(graph.cost[eid]) + lambda_execution_risk * float(evidence.execution_risk[eid]))
            st = (v, n_depth, n_unsup, n_bin)
            if n_cost < dist.get(st, math.inf):
                dist[st] = n_cost
                actual_risk[st] = n_risk
                prev[st] = (state, eid)
                heapq.heappush(pq, (n_cost, n_risk, v, n_depth, n_unsup, n_bin))
    if best is None:
        return _empty_support_result(variant, support_risk_budget)
    edges: list[int] = []
    nodes: list[int] = [goal_node]
    cur = best
    while cur != start:
        pstate, eid = prev[cur]
        edges.append(int(eid))
        nodes.append(int(pstate[0]))
        cur = pstate
    edges.reverse()
    nodes.reverse()
    return _make_support_plan_result(dataset, evidence, edges, nodes, dist[best], variant, support_risk_budget)


def nearest_nodes(evidence: SupportEvidenceGraph, embeddings: np.ndarray, indices: np.ndarray, cfg: Dict) -> np.ndarray:
    ann = KNNIndex.from_config(evidence.graph.node_embeddings, cfg, prefix="ann")
    batch_size = int(_cfg(cfg, "projection_batch_size", 131072))
    return ann.kneighbors(embeddings[np.asarray(indices, dtype=np.int64)], 1, return_distance=False, batch_size=batch_size).reshape(-1).astype(np.int64)
