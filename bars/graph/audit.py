from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from bars.common.logging import CSVLogger
from bars.data.trajectories import OfflineDataset
from bars.graph.ann import KNNIndex
from bars.graph.edges import _deduplicate_edges, _score_reachability
from bars.graph.planner import PlanResult, plan_path
from bars.graph.types import BARSGraph, EDGE_KIND_BRIDGE, EDGE_KIND_KNN, EDGE_KIND_TEMPORAL


@dataclass(frozen=True)
class AuditPair:
    pair_id: int
    pair_type: str
    start_index: int
    goal_index: int
    true_dt: int
    data_supported: int


def _cfg(cfg: Dict, key: str, default=None):
    return cfg.get("stage28_audit", {}).get(key, default)


def _quantiles(prefix: str, x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x)
    if x.size == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_p10": float("nan"),
            f"{prefix}_p50": float("nan"),
            f"{prefix}_p90": float("nan"),
        }
    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_p10": float(np.quantile(x, 0.10)),
        f"{prefix}_p50": float(np.quantile(x, 0.50)),
        f"{prefix}_p90": float(np.quantile(x, 0.90)),
    }


def _as_int_list(value, default: Sequence[int]) -> List[int]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [int(x.strip()) for x in value.split(",") if x.strip()]
    return [int(x) for x in value]


def _xy(dataset: OfflineDataset, indices: Optional[np.ndarray] = None, dims: Sequence[int] = (0, 1)) -> Optional[np.ndarray]:
    dims = [int(d) for d in dims]
    if not dims or max(dims) >= dataset.obs_dim:
        return None
    obs = dataset.observations if indices is None else dataset.observations[np.asarray(indices, dtype=np.int64)]
    return obs[:, dims].astype(np.float32)


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def _evidence_fields(gate: str, evidence_class: str) -> Dict[str, str]:
    return {"gate": gate, "evidence_class": evidence_class}


def _endpoint_indices(dataset: OfflineDataset) -> np.ndarray:
    out: list[int] = []
    for sl in dataset.traj_slices:
        if sl.end > sl.start:
            out.append(int(sl.start))
            out.append(int(sl.end - 1))
    return np.unique(np.asarray(out, dtype=np.int64)) if out else np.empty(0, dtype=np.int64)


def _project_to_nodes(embeddings: np.ndarray, node_indices: np.ndarray, cfg: Dict, batch_size: int = 131072) -> np.ndarray:
    node_emb = np.asarray(embeddings[node_indices], dtype=np.float32)
    ann = KNNIndex.from_config(node_emb, cfg, prefix="ann")
    return ann.kneighbors(np.asarray(embeddings, dtype=np.float32), 1, return_distance=False, batch_size=batch_size).reshape(-1).astype(np.int64)


def _projection_bridge_indices(dataset: OfflineDataset, embeddings: np.ndarray, base_node_indices: np.ndarray, cfg: Dict) -> np.ndarray:
    k = int(_cfg(cfg, "bottleneck_extra_nodes", 256))
    if k <= 0 or len(base_node_indices) == 0:
        return np.empty(0, dtype=np.int64)
    proj = _project_to_nodes(embeddings, base_node_indices, cfg, batch_size=int(_cfg(cfg, "projection_batch_size", 131072)))
    scores = np.zeros(dataset.size, dtype=np.float32)
    # High-frequency projected-node changes mark abstraction boundaries and bridge-like states.
    for sl in dataset.traj_slices:
        if sl.end - sl.start < 2:
            continue
        ids = np.arange(sl.start, sl.end, dtype=np.int64)
        p = proj[ids]
        change = np.flatnonzero(p[1:] != p[:-1])
        if len(change):
            scores[ids[change]] += 1.0
            scores[ids[change + 1]] += 1.0
    nz = np.flatnonzero(scores > 0)
    if len(nz) == 0:
        return np.empty(0, dtype=np.int64)
    if len(nz) > k:
        keep = nz[np.argpartition(-scores[nz], k - 1)[:k]]
        keep = keep[np.argsort(-scores[keep], kind="stable")]
    else:
        keep = nz[np.argsort(-scores[nz], kind="stable")]
    return keep.astype(np.int64)


def _cap_extra_nodes(base: np.ndarray, extra: np.ndarray, max_nodes: int) -> np.ndarray:
    base = np.unique(np.asarray(base, dtype=np.int64))
    extra = np.unique(np.asarray(extra, dtype=np.int64))
    if max_nodes <= 0:
        return np.unique(np.concatenate([base, extra])).astype(np.int64)
    remaining = max(0, int(max_nodes) - len(base))
    if remaining <= 0:
        return base
    return np.unique(np.concatenate([base, extra[:remaining]])).astype(np.int64)


def _temporal_edges_from_selected_nodes(dataset: OfflineDataset, node_indices: np.ndarray, temporal_connect: int, temporal_horizon: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(node_indices)
    if n == 0 or temporal_connect <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    local = np.arange(n, dtype=np.int64)
    traj = dataset.traj_id[node_indices]
    ts = dataset.timestep[node_indices]
    order = np.lexsort((ts, traj))
    local_s = local[order]
    traj_s = traj[order]
    ts_s = ts[order]
    src_parts: list[np.ndarray] = []
    dst_parts: list[np.ndarray] = []
    for off in range(1, int(temporal_connect) + 1):
        if off >= n:
            break
        same = traj_s[:-off] == traj_s[off:]
        dt = ts_s[off:] - ts_s[:-off]
        ok = same & (dt > 0) & (dt <= temporal_horizon)
        if ok.any():
            src_parts.append(local_s[:-off][ok])
            dst_parts.append(local_s[off:][ok])
    if not src_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(src_parts).astype(np.int64), np.concatenate(dst_parts).astype(np.int64)


def _projection_temporal_edges(dataset: OfflineDataset, embeddings: np.ndarray, node_indices: np.ndarray, cfg: Dict) -> tuple[np.ndarray, np.ndarray]:
    if len(node_indices) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    offsets = _as_int_list(_cfg(cfg, "projection_temporal_offsets", [1, 2, 4, 8, 16]), [1, 2, 4, 8, 16])
    max_offset = int(max(offsets)) if offsets else 1
    proj = _project_to_nodes(embeddings, node_indices, cfg, batch_size=int(_cfg(cfg, "projection_batch_size", 131072)))
    src_parts: list[np.ndarray] = []
    dst_parts: list[np.ndarray] = []
    for sl in dataset.traj_slices:
        length = sl.end - sl.start
        if length < 2:
            continue
        p = proj[sl.start:sl.end]
        for off in offsets:
            off = int(off)
            if off <= 0 or off >= length or off > max_offset:
                continue
            s = p[:-off]
            d = p[off:]
            ok = s != d
            if ok.any():
                src_parts.append(s[ok])
                dst_parts.append(d[ok])
    if not src_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(src_parts).astype(np.int64), np.concatenate(dst_parts).astype(np.int64)


def _knn_edges(points: np.ndarray, k: int, cfg: Dict, bidirectional: bool = True) -> tuple[np.ndarray, np.ndarray, str]:
    points = np.asarray(points, dtype=np.float32)
    n = len(points)
    if n <= 1 or k <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), "none"
    ann = KNNIndex.from_config(points, cfg, prefix="ann")
    ind = ann.kneighbors(points, min(int(k) + 1, n), return_distance=False)
    neigh = ind[:, 1:].astype(np.int64)
    rows = np.repeat(np.arange(n, dtype=np.int64), neigh.shape[1])
    cols = neigh.reshape(-1)
    if bidirectional:
        rows, cols = np.concatenate([rows, cols]), np.concatenate([cols, rows])
    return rows.astype(np.int64), cols.astype(np.int64), ann.backend


def _make_graph(
    dataset: OfflineDataset,
    embeddings: np.ndarray,
    node_indices: np.ndarray,
    cfg: Dict,
    *,
    graph_id: str,
    metric: str,
    include_knn: bool,
    include_selected_temporal: bool,
    include_projection_temporal: bool,
    reach_model=None,
    device=None,
) -> BARSGraph:
    node_indices = np.unique(np.asarray(node_indices, dtype=np.int64))
    if len(node_indices) < 2:
        raise ValueError(f"Graph variant {graph_id} has fewer than two nodes.")
    node_emb = embeddings[node_indices].astype(np.float32)
    src_parts: list[np.ndarray] = []
    dst_parts: list[np.ndarray] = []
    kind_parts: list[np.ndarray] = []
    acfg = cfg.get("stage28_audit", {})
    xy_dims = _as_int_list(acfg.get("xy_dims", [0, 1]), [0, 1])
    knn_backend = "none"

    if include_knn:
        if metric == "xy":
            points = _xy(dataset, node_indices, xy_dims)
            if points is None:
                points = node_emb
        else:
            points = node_emb
        s, d, knn_backend = _knn_edges(points, int(acfg.get("edge_knn", cfg.get("graph", {}).get("edge_knn", 16))), cfg, bool(acfg.get("bidirectional_knn", True)))
        if len(s):
            src_parts.append(s); dst_parts.append(d); kind_parts.append(np.full(len(s), EDGE_KIND_KNN, dtype=np.int32))

    if include_selected_temporal:
        s, d = _temporal_edges_from_selected_nodes(
            dataset,
            node_indices,
            temporal_connect=int(acfg.get("temporal_connect", cfg.get("graph", {}).get("temporal_connect", 4))),
            temporal_horizon=int(acfg.get("temporal_edge_horizon", cfg.get("graph", {}).get("temporal_edge_horizon", 80))),
        )
        if len(s):
            src_parts.append(s); dst_parts.append(d); kind_parts.append(np.full(len(s), EDGE_KIND_TEMPORAL, dtype=np.int32))

    if include_projection_temporal:
        s, d = _projection_temporal_edges(dataset, embeddings, node_indices, cfg)
        if len(s):
            src_parts.append(s); dst_parts.append(d); kind_parts.append(np.full(len(s), EDGE_KIND_BRIDGE, dtype=np.int32))

    if not src_parts:
        raise RuntimeError(f"No edges constructed for audit graph {graph_id}.")
    src = np.concatenate(src_parts)
    dst = np.concatenate(dst_parts)
    kind = np.concatenate(kind_parts)
    src, dst, kind = _deduplicate_edges(src, dst, kind, len(node_indices))

    if metric == "xy":
        points = _xy(dataset, node_indices, xy_dims)
        if points is None:
            points = node_emb
        dist = np.linalg.norm(points[src] - points[dst], axis=1).astype(np.float32)
    else:
        dist = np.linalg.norm(node_emb[src] - node_emb[dst], axis=1).astype(np.float32)
    scale = float(np.median(dist) + 1e-6) if len(dist) else 1.0
    cost = (dist / scale).astype(np.float32)
    p_exec = _score_reachability(
        reach_model,
        node_emb,
        src,
        dst,
        device,
        batch_size=int(acfg.get("score_batch_size", cfg.get("graph", {}).get("score_batch_size", 32768))),
        fallback_scale=scale,
    )
    p_clip = float(acfg.get("p_clip", cfg.get("graph", {}).get("p_clip", 1e-4)))
    risk = (-np.log(np.clip(p_exec, p_clip, 1.0))).astype(np.float32)

    top_out = int(acfg.get("top_outgoing", 0))
    if top_out > 0 and len(src):
        keep: list[np.ndarray] = []
        score = cost + float(acfg.get("prune_lambda_risk", 0.0)) * risk
        for u in range(len(node_indices)):
            ids = np.flatnonzero(src == u)
            if len(ids) <= top_out:
                keep.append(ids)
            elif len(ids):
                sel = ids[np.argpartition(score[ids], top_out - 1)[:top_out]]
                keep.append(sel)
        keep_idx = np.concatenate(keep).astype(np.int64) if keep else np.empty(0, dtype=np.int64)
        src, dst, kind, cost, risk, p_exec = src[keep_idx], dst[keep_idx], kind[keep_idx], cost[keep_idx], risk[keep_idx], p_exec[keep_idx]

    graph = BARSGraph(node_indices, node_emb, src, dst, cost, risk, p_exec, kind)
    graph.audit_graph_id = graph_id  # type: ignore[attr-defined]
    graph.audit_metric = metric  # type: ignore[attr-defined]
    graph.audit_knn_backend = knn_backend  # type: ignore[attr-defined]
    return graph


def build_audit_graph_variants(dataset: OfflineDataset, embeddings: np.ndarray, base_graph: BARSGraph, cfg: Dict, reach_model=None, device=None) -> Dict[str, BARSGraph]:
    acfg = cfg.get("stage28_audit", {})
    requested = acfg.get("graph_variants", ["base_cached", "projection_temporal", "dense_knn", "xy_knn", "endpoint_aug", "bottleneck_aug"])
    if isinstance(requested, str):
        requested = [x.strip() for x in requested.split(",") if x.strip()]
    requested = [str(x) for x in requested]
    variants: Dict[str, BARSGraph] = {}
    if "base_cached" in requested or "base" in requested:
        base_graph.audit_graph_id = "base_cached"  # type: ignore[attr-defined]
        base_graph.audit_metric = "cached"  # type: ignore[attr-defined]
        base_graph.audit_knn_backend = "cached"  # type: ignore[attr-defined]
        variants["base_cached"] = base_graph

    base_nodes = np.asarray(base_graph.node_indices, dtype=np.int64)
    max_nodes = int(acfg.get("max_nodes_per_variant", 0))
    endpoint_nodes = _cap_extra_nodes(base_nodes, _endpoint_indices(dataset), max_nodes)
    bottleneck_extra = np.concatenate([_endpoint_indices(dataset), _projection_bridge_indices(dataset, embeddings, base_nodes, cfg)])
    bottleneck_nodes = _cap_extra_nodes(base_nodes, bottleneck_extra, max_nodes)

    builders = {
        "projection_temporal": dict(node_indices=base_nodes, metric="embedding", include_knn=False, include_selected_temporal=False, include_projection_temporal=True),
        "dense_knn": dict(node_indices=base_nodes, metric="embedding", include_knn=True, include_selected_temporal=True, include_projection_temporal=True),
        "xy_knn": dict(node_indices=base_nodes, metric="xy", include_knn=True, include_selected_temporal=True, include_projection_temporal=True),
        "endpoint_aug": dict(node_indices=endpoint_nodes, metric="embedding", include_knn=True, include_selected_temporal=True, include_projection_temporal=True),
        "bottleneck_aug": dict(node_indices=bottleneck_nodes, metric="embedding", include_knn=True, include_selected_temporal=True, include_projection_temporal=True),
    }
    for graph_id, kwargs in builders.items():
        if graph_id not in requested:
            continue
        variants[graph_id] = _make_graph(dataset, embeddings, cfg=cfg, graph_id=graph_id, reach_model=reach_model, device=device, **kwargs)
    return variants


def _component_stats(graph: BARSGraph) -> Dict[str, float | int]:
    if graph.num_nodes == 0:
        return {}
    mat = csr_matrix((np.ones(graph.num_edges, dtype=np.int8), (graph.src, graph.dst)), shape=(graph.num_nodes, graph.num_nodes))
    n_weak, weak = connected_components(mat, directed=True, connection="weak", return_labels=True)
    n_strong, strong = connected_components(mat, directed=True, connection="strong", return_labels=True)
    weak_counts = np.bincount(weak, minlength=n_weak) if len(weak) else np.asarray([], dtype=np.int64)
    strong_counts = np.bincount(strong, minlength=n_strong) if len(strong) else np.asarray([], dtype=np.int64)
    return {
        "weak_components": int(n_weak),
        "strong_components": int(n_strong),
        "largest_weak_component_rate": float(weak_counts.max() / max(1, graph.num_nodes)) if len(weak_counts) else 0.0,
        "largest_strong_component_rate": float(strong_counts.max() / max(1, graph.num_nodes)) if len(strong_counts) else 0.0,
    }


def _coverage_stats(dataset: OfflineDataset, embeddings: np.ndarray, graph: BARSGraph, cfg: Dict) -> Dict[str, float | int]:
    acfg = cfg.get("stage28_audit", {})
    sample_cap = int(acfg.get("coverage_sample", 200000))
    rng = np.random.default_rng(int(cfg.get("seed", 0)) + 2801)
    if dataset.size > sample_cap > 0:
        ids = rng.choice(dataset.size, size=sample_cap, replace=False)
    else:
        ids = np.arange(dataset.size, dtype=np.int64)
    ann = KNNIndex.from_config(graph.node_embeddings, cfg, prefix="ann")
    dist, _ = ann.kneighbors(embeddings[ids], 1, return_distance=True, batch_size=int(acfg.get("projection_batch_size", 131072)))
    dist = dist.reshape(-1)
    out: Dict[str, float | int] = {"coverage_sample": int(len(ids))}
    out.update(_quantiles("node_cover_emb_dist", dist))

    end_ids = _endpoint_indices(dataset)
    if len(end_ids):
        ed, _ = ann.kneighbors(embeddings[end_ids], 1, return_distance=True, batch_size=int(acfg.get("projection_batch_size", 131072)))
        out.update(_quantiles("endpoint_cover_emb_dist", ed.reshape(-1)))
        node_set = set(int(x) for x in graph.node_indices.tolist())
        out["endpoint_exact_retention_rate"] = float(np.mean([int(i) in node_set for i in end_ids]))
    else:
        out["endpoint_exact_retention_rate"] = float("nan")
    return out


def _edge_proxy_stats(dataset: OfflineDataset, graph: BARSGraph, cfg: Dict) -> Dict[str, float | int]:
    acfg = cfg.get("stage28_audit", {})
    horizon = int(acfg.get("edge_label_horizon", cfg.get("reachability", {}).get("horizon", 30)))
    gi_src = graph.node_indices[graph.src]
    gi_dst = graph.node_indices[graph.dst]
    same = dataset.traj_id[gi_src] == dataset.traj_id[gi_dst]
    dt = dataset.timestep[gi_dst] - dataset.timestep[gi_src]
    temporal_supported = same & (dt > 0) & (dt <= horizon)
    cross = ~same
    selected = graph.p_exec >= float(acfg.get("selected_p_exec_threshold", 0.5))
    low_cost_thr = float(np.quantile(graph.cost, float(acfg.get("low_cost_quantile", 0.20)))) if graph.num_edges else float("nan")
    high_conf_thr = float(acfg.get("high_conf_p_exec", 0.5))
    low_cost_cross = cross & (graph.cost <= low_cost_thr) & (graph.p_exec >= high_conf_thr)
    out: Dict[str, float | int] = {
        "edge_label_horizon": horizon,
        "temporal_supported_edge_rate": float(temporal_supported.mean()) if len(temporal_supported) else 0.0,
        "cross_edge_rate": float(cross.mean()) if len(cross) else 0.0,
        "selected_edge_rate": float(selected.mean()) if len(selected) else 0.0,
        "selected_cross_rate": float((selected & cross).sum() / max(1, selected.sum())),
        "low_cost_high_conf_cross_edge_rate": float(low_cost_cross.mean()) if len(low_cost_cross) else 0.0,
        "temporal_kind_rate": float((graph.kind == EDGE_KIND_TEMPORAL).mean()) if graph.num_edges else 0.0,
        "bridge_kind_rate": float((graph.kind == EDGE_KIND_BRIDGE).mean()) if graph.num_edges else 0.0,
    }
    out.update(_quantiles("edge_cost", graph.cost))
    out.update(_quantiles("edge_risk", graph.risk))
    out.update(_quantiles("edge_p_exec", graph.p_exec))
    return out


def _log_graph_summary(dataset: OfflineDataset, embeddings: np.ndarray, graph_id: str, graph: BARSGraph, cfg: Dict, logger: CSVLogger) -> None:
    out_degrees = np.asarray([len(x) for x in graph.outgoing_edges()], dtype=np.int32)
    row: Dict[str, float | int | str] = {
        "phase": "stage28_graph_summary",
        "event": "completed",
        **_evidence_fields("PASS_STAGE28_GRAPH_COUNTERFACTUALS", "graph_abstraction_counterfactual"),
        "graph_id": graph_id,
        "audit_metric": str(getattr(graph, "audit_metric", "")),
        "ann_backend": str(getattr(graph, "audit_knn_backend", "")),
        "num_nodes": graph.num_nodes,
        "num_edges": graph.num_edges,
        "mean_out_degree": _safe_div(graph.num_edges, graph.num_nodes),
        "zero_out_degree_nodes": int((out_degrees == 0).sum()) if len(out_degrees) else 0,
    }
    row.update(_component_stats(graph))
    row.update(_coverage_stats(dataset, embeddings, graph, cfg))
    row.update(_edge_proxy_stats(dataset, graph, cfg))
    logger.log(row)


def _sample_audit_pairs(dataset: OfflineDataset, cfg: Dict) -> List[AuditPair]:
    acfg = cfg.get("stage28_audit", {})
    rng = np.random.default_rng(int(cfg.get("seed", 0)) + 2803)
    num_future = int(acfg.get("num_future_pairs", acfg.get("num_pairs", 256)))
    num_cross = int(acfg.get("num_cross_pairs", max(0, num_future // 2)))
    min_dt = int(acfg.get("path_min_dt", cfg.get("reachability", {}).get("horizon", 30) * 2))
    max_dt = int(acfg.get("path_max_dt", 250))
    pairs: list[AuditPair] = []
    if num_future > 0:
        try:
            s, g, dt = dataset.sample_future_pairs(num_future, max_dt, rng, min_dt=min_dt)
            for i in range(len(s)):
                pairs.append(AuditPair(len(pairs), "future_same_traj", int(s[i]), int(g[i]), int(dt[i]), 1))
        except Exception:
            pass
    if num_cross > 0 and dataset.num_trajectories > 1:
        attempts = 0
        max_attempts = max(1000, num_cross * 50)
        while len([p for p in pairs if p.pair_type == "cross_traj_random"]) < num_cross and attempts < max_attempts:
            attempts += 1
            i = int(rng.integers(0, dataset.size))
            j = int(rng.integers(0, dataset.size))
            if dataset.traj_id[i] == dataset.traj_id[j] or i == j:
                continue
            pairs.append(AuditPair(len(pairs), "cross_traj_random", i, j, -1, 0))
    return pairs


def _nearest_node_map(graph: BARSGraph, embeddings: np.ndarray, indices: np.ndarray, cfg: Dict) -> np.ndarray:
    ann = KNNIndex.from_config(graph.node_embeddings, cfg, prefix="ann")
    return ann.kneighbors(embeddings[indices], 1, return_distance=False, batch_size=int(_cfg(cfg, "projection_batch_size", 131072))).reshape(-1).astype(np.int64)


def _path_edge_stats(dataset: OfflineDataset, graph: BARSGraph, result: PlanResult) -> Dict[str, float | int]:
    if not result.found or not result.edge_path:
        return {
            "path_cross_edge_rate": float("nan"),
            "path_temporal_kind_rate": float("nan"),
            "path_bridge_kind_rate": float("nan"),
            "path_largest_edge_cost_ratio": float("nan"),
            "path_min_p_exec": float("nan"),
            "path_mean_p_exec": float("nan"),
            "path_mean_risk": float("nan"),
        }
    e = np.asarray(result.edge_path, dtype=np.int64)
    gi_src = graph.node_indices[graph.src[e]]
    gi_dst = graph.node_indices[graph.dst[e]]
    cross = dataset.traj_id[gi_src] != dataset.traj_id[gi_dst]
    total_cost = float(np.sum(graph.cost[e]))
    return {
        "path_cross_edge_rate": float(cross.mean()) if len(cross) else 0.0,
        "path_temporal_kind_rate": float((graph.kind[e] == EDGE_KIND_TEMPORAL).mean()) if len(e) else 0.0,
        "path_bridge_kind_rate": float((graph.kind[e] == EDGE_KIND_BRIDGE).mean()) if len(e) else 0.0,
        "path_largest_edge_cost_ratio": float(np.max(graph.cost[e]) / max(total_cost, 1e-8)) if len(e) else 0.0,
        "path_min_p_exec": float(np.min(graph.p_exec[e])) if len(e) else float("nan"),
        "path_mean_p_exec": float(np.mean(graph.p_exec[e])) if len(e) else float("nan"),
        "path_mean_risk": float(np.mean(graph.risk[e])) if len(e) else float("nan"),
    }


def _graph_without_edges(graph: BARSGraph, drop_edges: Iterable[int]) -> BARSGraph:
    drop = set(int(e) for e in drop_edges)
    if not drop:
        return graph
    keep = np.asarray([i for i in range(graph.num_edges) if i not in drop], dtype=np.int64)
    return BARSGraph(graph.node_indices, graph.node_embeddings, graph.src[keep], graph.dst[keep], graph.cost[keep], graph.risk[keep], graph.p_exec[keep], graph.kind[keep])


def _alternative_path_stats(graph: BARSGraph, start_node: int, goal_node: int, result: PlanResult, variant: str, cfg: Dict, boundary=None) -> Dict[str, float | int]:
    if not result.found or len(result.edge_path) == 0:
        return {"alt_remove_edge_trials": 0, "alt_path_rate": float("nan"), "alt_objective_ratio_min": float("nan")}
    acfg = cfg.get("stage28_audit", {})
    max_trials = int(acfg.get("path_diversity_remove_edges", 8))
    trials = list(result.edge_path[:max_trials])
    found = 0
    ratios: list[float] = []
    for eid in trials:
        g2 = _graph_without_edges(graph, [int(eid)])
        alt = plan_path(
            g2,
            start_node,
            goal_node,
            variant=variant,
            lambda_risk=float(acfg.get("lambda_risk", cfg.get("planner", {}).get("lambda_risk", 1.0))),
            lambda_boundary=float(acfg.get("lambda_boundary", cfg.get("planner", {}).get("lambda_boundary", 1.0))),
            boundary=None,
            max_edges=int(acfg.get("max_path_edges", 0)) or None,
        )
        if alt.found:
            found += 1
            ratios.append(float(alt.objective / max(result.objective, 1e-8)))
    return {
        "alt_remove_edge_trials": int(len(trials)),
        "alt_path_rate": float(found / max(1, len(trials))),
        "alt_objective_ratio_min": float(np.min(ratios)) if ratios else float("nan"),
    }


def _reference_variant(cfg: Dict, boundary_available: bool) -> str:
    requested = str(_cfg(cfg, "base_reference_variant", "full_bars" if boundary_available else "reachability")).lower()
    if requested in {"full_bars", "bars", "boundary"} and not boundary_available:
        return "reachability"
    return requested


def _variants_for_graph(graph_id: str, cfg: Dict, boundary_available: bool) -> List[str]:
    acfg = cfg.get("stage28_audit", {})
    if graph_id == "base_cached":
        default = ["shortest", "reachability", "full_bars"] if boundary_available else ["shortest", "reachability"]
        variants = acfg.get("base_planner_variants", default)
    else:
        variants = acfg.get("planner_variants", ["shortest", "reachability"])
    if isinstance(variants, str):
        variants = [x.strip() for x in variants.split(",") if x.strip()]
    if not boundary_available:
        variants = ["reachability" if str(v).lower() in {"full_bars", "bars", "boundary"} else str(v) for v in variants]
    return [str(v) for v in variants]


def _label_failure(row: Dict[str, float | int | str], cfg: Dict) -> str:
    acfg = cfg.get("stage28_audit", {})
    projection_found = int(row.get("projection_found", 0))
    base_found = int(row.get("base_found", 0))
    data_supported = int(row.get("data_supported", 0))
    cross_rate = float(row.get("base_path_cross_edge_rate", 0.0) or 0.0)
    hop_ratio = float(row.get("base_path_largest_edge_cost_ratio", 0.0) or 0.0)
    alt_rate = float(row.get("base_alt_path_rate", 0.0) or 0.0)
    if data_supported and not projection_found:
        return "NO_DATA_PATH_AFTER_NODE_PROJECTION"
    if projection_found and not base_found:
        return "BASE_LOST_SUPPORTED_PATH_OR_EDGE_PRUNING"
    if base_found and cross_rate >= float(acfg.get("risky_cross_rate_threshold", 0.50)) and data_supported:
        return "BASE_USES_CROSS_TRAJ_SHORTCUT_FOR_SUPPORTED_PAIR"
    if base_found and hop_ratio >= float(acfg.get("largest_hop_ratio_threshold", 0.55)):
        return "BASE_SINGLE_HOP_DOMINATED_PATH"
    if base_found and alt_rate <= float(acfg.get("low_alt_path_rate_threshold", 0.20)):
        return "SINGLE_PATH_FRAGILITY_PROXY"
    if base_found:
        return "GRAPH_PATH_EXISTS_EXECUTION_PROBE_NEEDED"
    return "NO_GRAPH_PATH_UNRESOLVED"


def run_graph_method_audit(
    dataset: OfflineDataset,
    embeddings: np.ndarray,
    base_graph: BARSGraph,
    cfg: Dict,
    logger: CSVLogger,
    *,
    reach_model=None,
    device=None,
    boundary=None,
) -> Dict[str, BARSGraph]:
    """Run Stage28 evidence-driven diagnostics over cached BARS/GAS artifacts.

    The audit is intentionally planner/evaluator agnostic. It compares the cached
    BARS graph against support-preserving counterfactual graphs, then emits CSV
    rows that identify which mechanism is most consistent with each sampled
    start-goal failure proxy. Online edge rollouts can be run separately with the
    existing edge_rollout diagnostics; this function focuses on graph evidence.
    """
    acfg = cfg.get("stage28_audit", {})
    if hasattr(logger, "default_fields"):
        logger.default_fields.setdefault("report_file", getattr(logger, "path", ""))
        logger.default_fields.setdefault("baseline_graph_role", "sota_study_baseline_cached_bars_gas_aligned")
    embeddings = np.asarray(embeddings, dtype=np.float32)
    traj_lens = [sl.end - sl.start for sl in dataset.traj_slices]
    logger.log({
        "phase": "stage28_dataset_support",
        "event": "completed",
        **_evidence_fields("PASS_STAGE28_DATASET_SUPPORT_AUDIT", "dataset_support"),
        "dataset_size": dataset.size,
        "num_trajectories": dataset.num_trajectories,
        "obs_dim": dataset.obs_dim,
        "action_dim": dataset.action_dim,
        "traj_len_mean": float(np.mean(traj_lens)) if traj_lens else 0.0,
        "traj_len_p10": float(np.quantile(traj_lens, 0.10)) if traj_lens else 0.0,
        "traj_len_p90": float(np.quantile(traj_lens, 0.90)) if traj_lens else 0.0,
        "base_graph_nodes": base_graph.num_nodes,
        "base_graph_edges": base_graph.num_edges,
        "seed_for_audit": int(cfg.get("seed", 0)),
    })

    variants = build_audit_graph_variants(dataset, embeddings, base_graph, cfg, reach_model=reach_model, device=device)
    for graph_id, graph in variants.items():
        _log_graph_summary(dataset, embeddings, graph_id, graph, cfg, logger)

    pairs = _sample_audit_pairs(dataset, cfg)
    logger.log({
        "phase": "stage28_pair_sampling",
        "event": "completed",
        **_evidence_fields("PASS_STAGE28_PAIR_SAMPLING", "audit_pair_sampling"),
        "num_pairs": len(pairs),
        "num_future_pairs": sum(p.pair_type == "future_same_traj" for p in pairs),
        "num_cross_pairs": sum(p.pair_type == "cross_traj_random" for p in pairs),
    })
    if not pairs:
        return variants

    graph_pair_nodes: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    start_idx = np.asarray([p.start_index for p in pairs], dtype=np.int64)
    goal_idx = np.asarray([p.goal_index for p in pairs], dtype=np.int64)
    for graph_id, graph in variants.items():
        graph_pair_nodes[graph_id] = (_nearest_node_map(graph, embeddings, start_idx, cfg), _nearest_node_map(graph, embeddings, goal_idx, cfg))

    lambda_risk = float(acfg.get("lambda_risk", cfg.get("planner", {}).get("lambda_risk", 1.0)))
    lambda_boundary = float(acfg.get("lambda_boundary", cfg.get("planner", {}).get("lambda_boundary", 1.0)))
    max_edges = int(acfg.get("max_path_edges", 0)) or None
    base_ref = _reference_variant(cfg, boundary is not None)
    projection_ref = str(acfg.get("projection_reference_variant", "shortest"))

    taxonomy_rows: list[Dict[str, float | int | str]] = []
    for pair in pairs:
        pair_summary: Dict[str, float | int | str] = {
            "phase": "stage28_failure_taxonomy_proxy",
            **_evidence_fields("PASS_STAGE28_DIAGNOSE_FIRST_TAXONOMY", "failure_taxonomy_proxy"),
            "pair_id": pair.pair_id,
            "pair_type": pair.pair_type,
            "start_index": pair.start_index,
            "goal_index": pair.goal_index,
            "true_dt": pair.true_dt,
            "data_supported": pair.data_supported,
        }
        for graph_id, graph in variants.items():
            s_nodes, g_nodes = graph_pair_nodes[graph_id]
            s_node = int(s_nodes[pair.pair_id])
            g_node = int(g_nodes[pair.pair_id])
            planner_variants = _variants_for_graph(graph_id, cfg, boundary is not None and graph_id == "base_cached")
            for planner_variant in planner_variants:
                use_boundary = boundary if graph_id == "base_cached" and str(planner_variant).lower() in {"full_bars", "bars", "boundary", "gas_bars"} else None
                result = plan_path(graph, s_node, g_node, variant=planner_variant, lambda_risk=lambda_risk, lambda_boundary=lambda_boundary, boundary=use_boundary, max_edges=max_edges)
                path_stats = _path_edge_stats(dataset, graph, result)
                diversity_stats = {}
                if bool(acfg.get("enable_path_diversity_probe", True)) and graph_id == "base_cached" and str(planner_variant).lower() == base_ref:
                    diversity_stats = _alternative_path_stats(graph, s_node, g_node, result, planner_variant, cfg, boundary=use_boundary)
                row = {
                    "phase": "stage28_path_probe",
                    **_evidence_fields("PASS_STAGE28_PATH_PROBE", "path_search_counterfactual"),
                    "graph_id": graph_id,
                    "pair_id": pair.pair_id,
                    "pair_type": pair.pair_type,
                    "start_index": pair.start_index,
                    "goal_index": pair.goal_index,
                    "true_dt": pair.true_dt,
                    "data_supported": pair.data_supported,
                    "start_node": s_node,
                    "goal_node": g_node,
                    "planner_variant": planner_variant,
                    **result.to_row(),
                    **path_stats,
                    **diversity_stats,
                }
                logger.log(row)
                if graph_id == "projection_temporal" and str(planner_variant).lower() == projection_ref:
                    pair_summary["projection_found"] = int(result.found)
                if graph_id == "base_cached" and str(planner_variant).lower() == base_ref:
                    pair_summary["base_found"] = int(result.found)
                    pair_summary["base_num_edges"] = len(result.edge_path)
                    pair_summary["base_objective"] = result.objective
                    pair_summary["base_path_cross_edge_rate"] = path_stats["path_cross_edge_rate"]
                    pair_summary["base_path_largest_edge_cost_ratio"] = path_stats["path_largest_edge_cost_ratio"]
                    if diversity_stats:
                        pair_summary["base_alt_path_rate"] = diversity_stats.get("alt_path_rate", float("nan"))

        pair_summary.setdefault("projection_found", 0)
        pair_summary.setdefault("base_found", 0)
        pair_summary.setdefault("base_path_cross_edge_rate", float("nan"))
        pair_summary.setdefault("base_path_largest_edge_cost_ratio", float("nan"))
        pair_summary.setdefault("base_alt_path_rate", float("nan"))
        pair_summary["failure_label"] = _label_failure(pair_summary, cfg)
        taxonomy_rows.append(pair_summary)
        logger.log(pair_summary)

    if taxonomy_rows:
        labels: Dict[str, int] = {}
        for row in taxonomy_rows:
            label = str(row["failure_label"])
            labels[label] = labels.get(label, 0) + 1
        for label, count in sorted(labels.items(), key=lambda kv: (-kv[1], kv[0])):
            logger.log({
                "phase": "stage28_failure_taxonomy_summary",
                "event": "completed",
                **_evidence_fields("PASS_STAGE28_DIAGNOSE_FIRST_TAXONOMY", "failure_taxonomy_summary"),
                "failure_label": label,
                "count": int(count),
                "rate": float(count / max(1, len(taxonomy_rows))),
                "num_pairs": int(len(taxonomy_rows)),
            })
    return variants
