from __future__ import annotations

from typing import Dict, Optional, Tuple
import time

import numpy as np
import torch

from bars.common.logging import CSVLogger
from bars.data.trajectories import OfflineDataset
from bars.models.reachability import ReachabilityModel
from .types import BARSGraph, EDGE_KIND_KNN, EDGE_KIND_TEMPORAL
from .ann import KNNIndex


def _deduplicate_edges(src: np.ndarray, dst: np.ndarray, kind: np.ndarray, num_nodes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    src = np.asarray(src, dtype=np.int64).reshape(-1)
    dst = np.asarray(dst, dtype=np.int64).reshape(-1)
    kind = np.asarray(kind, dtype=np.int32).reshape(-1)
    if len(src) == 0:
        return src, dst, kind
    ok = (src != dst) & (src >= 0) & (src < num_nodes) & (dst >= 0) & (dst < num_nodes)
    src, dst, kind = src[ok], dst[ok], kind[ok]
    if len(src) == 0:
        return src, dst, kind
    keys = src * int(num_nodes) + dst
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    kind = kind[order]
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    max_kind = np.maximum.reduceat(kind, starts).astype(np.int32)
    uniq = keys[starts]
    return (uniq // int(num_nodes)).astype(np.int64), (uniq % int(num_nodes)).astype(np.int64), max_kind


@torch.no_grad()
def _score_reachability(
    reach_model: Optional[ReachabilityModel],
    node_embeddings: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    device,
    batch_size: int = 32768,
    fallback_scale: Optional[float] = None,
) -> np.ndarray:
    if len(src) == 0:
        return np.empty(0, dtype=np.float32)
    if reach_model is None:
        dist = np.linalg.norm(node_embeddings[src] - node_embeddings[dst], axis=1)
        scale = fallback_scale or (np.median(dist) + 1e-6)
        return np.exp(-dist / scale).astype(np.float32)
    reach_model.eval()
    probs = []
    z_all = torch.as_tensor(node_embeddings, dtype=torch.float32, device=device)
    for st in range(0, len(src), batch_size):
        sl = slice(st, st + batch_size)
        src_t = torch.as_tensor(src[sl], dtype=torch.long, device=device)
        dst_t = torch.as_tensor(dst[sl], dtype=torch.long, device=device)
        zu = z_all[src_t]
        zv = z_all[dst_t]
        probs.append(reach_model.prob(zu, zv).float().cpu().numpy())
    return np.concatenate(probs, 0).astype(np.float32)


def _top_outgoing_keep_indices(
    src: np.ndarray,
    score: np.ndarray,
    temporal_support: np.ndarray,
    *,
    num_nodes: int,
    top_k: int,
    min_outgoing: int,
    preserve_temporal: bool,
) -> Tuple[np.ndarray, int]:
    if len(src) == 0:
        return np.empty(0, dtype=np.int64), 0
    limit = max(int(top_k), int(min_outgoing))
    order = np.argsort(src, kind="stable")
    src_sorted = src[order]
    bounds = np.flatnonzero(np.r_[True, src_sorted[1:] != src_sorted[:-1], True])
    keep_chunks: list[np.ndarray] = []
    temporal_preserved = 0
    for a, b in zip(bounds[:-1], bounds[1:]):
        ids = order[a:b]
        if len(ids) == 0:
            continue
        preserve_ids = ids[temporal_support[ids]] if preserve_temporal else np.empty(0, dtype=np.int64)
        temporal_preserved += int(len(preserve_ids))
        if preserve_temporal:
            rem = ids[~temporal_support[ids]]
        else:
            rem = ids
        quota = limit - len(preserve_ids)
        if quota > 0 and len(rem):
            if len(rem) > quota:
                ranked = rem[np.argpartition(score[rem], quota - 1)[:quota]]
                ranked = ranked[np.argsort(score[ranked], kind="stable")]
            else:
                ranked = rem[np.argsort(score[rem], kind="stable")]
            keep = np.concatenate([preserve_ids, ranked]) if len(preserve_ids) else ranked
        elif len(preserve_ids):
            keep = preserve_ids
        elif limit > 0:
            ranked = ids[np.argsort(score[ids], kind="stable")[:limit]]
            keep = ranked
        else:
            keep = np.empty(0, dtype=np.int64)
        if len(keep):
            keep_chunks.append(np.unique(keep).astype(np.int64))
    if not keep_chunks:
        return np.empty(0, dtype=np.int64), temporal_preserved
    return np.unique(np.concatenate(keep_chunks)).astype(np.int64), temporal_preserved


def _temporal_support_mask(dataset: OfflineDataset, node_indices: np.ndarray, src: np.ndarray, dst: np.ndarray, temporal_max_dt: int) -> np.ndarray:
    same = dataset.traj_id[node_indices[src]] == dataset.traj_id[node_indices[dst]]
    dt = dataset.timestep[node_indices[dst]] - dataset.timestep[node_indices[src]]
    return same & (dt > 0) & (dt <= temporal_max_dt)


def build_edges(
    dataset: OfflineDataset,
    embeddings: np.ndarray,
    node_indices: np.ndarray,
    reach_model: Optional[ReachabilityModel],
    cfg: Dict,
    device,
    logger: CSVLogger,
) -> BARSGraph:
    gcfg = cfg.get('graph', {})
    node_emb = embeddings[node_indices].astype(np.float32)
    n = len(node_indices)
    src_parts: list[np.ndarray] = []
    dst_parts: list[np.ndarray] = []
    kind_parts: list[np.ndarray] = []
    t_build = time.time()
    logger.log({'phase': 'edges', 'event': 'start', 'num_nodes': n, 'node_method': gcfg.get('node_method', 'bars')})

    knn = int(gcfg.get('edge_knn', 16))
    if n > 1 and knn > 0:
        t_knn = time.time()
        ann = KNNIndex.from_config(node_emb, cfg, prefix='ann')
        ind = ann.kneighbors(node_emb, min(knn + 1, n), return_distance=False)
        neigh = ind[:, 1:].astype(np.int64)
        if neigh.size:
            rows = np.repeat(np.arange(n, dtype=np.int64), neigh.shape[1])
            cols = neigh.reshape(-1)
            if bool(gcfg.get('bidirectional_knn', True)):
                src_knn = np.concatenate([rows, cols])
                dst_knn = np.concatenate([cols, rows])
            else:
                src_knn, dst_knn = rows, cols
            kind_knn = np.full(len(src_knn), EDGE_KIND_KNN, dtype=np.int32)
            src_parts.append(src_knn); dst_parts.append(dst_knn); kind_parts.append(kind_knn)
        cand_src, cand_dst, _ = _deduplicate_edges(
            np.concatenate(src_parts) if src_parts else np.empty(0, dtype=np.int64),
            np.concatenate(dst_parts) if dst_parts else np.empty(0, dtype=np.int64),
            np.concatenate(kind_parts) if kind_parts else np.empty(0, dtype=np.int32),
            n,
        )
        logger.log({'phase': 'edges', 'event': 'knn_candidates_built', 'candidate_edges': len(cand_src), 'knn': knn, 'ann_backend': ann.backend, 'duration_sec': time.time() - t_knn})

    temporal_connect = int(gcfg.get('temporal_connect', 4))
    temporal_max_dt = int(gcfg.get('temporal_edge_horizon', 80))
    if temporal_connect > 0:
        t_temporal = time.time()
        local = np.arange(n, dtype=np.int64)
        traj = dataset.traj_id[node_indices]
        ts = dataset.timestep[node_indices]
        order = np.lexsort((ts, traj))
        local_s = local[order]
        traj_s = traj[order]
        ts_s = ts[order]
        temporal_src_parts: list[np.ndarray] = []
        temporal_dst_parts: list[np.ndarray] = []
        for offset in range(1, temporal_connect + 1):
            if offset >= n:
                break
            same = traj_s[:-offset] == traj_s[offset:]
            dt = ts_s[offset:] - ts_s[:-offset]
            ok = same & (dt > 0) & (dt <= temporal_max_dt)
            if ok.any():
                temporal_src_parts.append(local_s[:-offset][ok])
                temporal_dst_parts.append(local_s[offset:][ok])
        if temporal_src_parts:
            src_temporal = np.concatenate(temporal_src_parts).astype(np.int64)
            dst_temporal = np.concatenate(temporal_dst_parts).astype(np.int64)
            if bool(gcfg.get('temporal_reverse', False)):
                src_temporal, dst_temporal = np.concatenate([src_temporal, dst_temporal]), np.concatenate([dst_temporal, src_temporal])
            kind_temporal = np.full(len(src_temporal), EDGE_KIND_TEMPORAL, dtype=np.int32)
            src_parts.append(src_temporal); dst_parts.append(dst_temporal); kind_parts.append(kind_temporal)
        cand_src, _, _ = _deduplicate_edges(
            np.concatenate(src_parts) if src_parts else np.empty(0, dtype=np.int64),
            np.concatenate(dst_parts) if dst_parts else np.empty(0, dtype=np.int64),
            np.concatenate(kind_parts) if kind_parts else np.empty(0, dtype=np.int32),
            n,
        )
        logger.log({'phase': 'edges', 'event': 'temporal_candidates_built', 'candidate_edges': len(cand_src), 'temporal_connect': temporal_connect, 'temporal_edge_horizon': temporal_max_dt, 'duration_sec': time.time() - t_temporal})

    src = np.concatenate(src_parts) if src_parts else np.empty(0, dtype=np.int64)
    dst = np.concatenate(dst_parts) if dst_parts else np.empty(0, dtype=np.int64)
    kind = np.concatenate(kind_parts) if kind_parts else np.empty(0, dtype=np.int32)
    src, dst, kind = _deduplicate_edges(src, dst, kind, n)
    if len(src) == 0:
        raise RuntimeError('No candidate edges constructed.')

    dist = np.linalg.norm(node_emb[src] - node_emb[dst], axis=1).astype(np.float32)
    dist_scale = float(np.median(dist) + 1e-6)
    cost = dist / dist_scale
    temporal_support = _temporal_support_mask(dataset, node_indices, src, dst, temporal_max_dt)
    if str(gcfg.get('cost_mode', 'hybrid')) == 'hybrid':
        dt = dataset.timestep[node_indices[dst]] - dataset.timestep[node_indices[src]]
        cost[temporal_support] = np.log1p(dt[temporal_support]).astype(np.float32) / np.log1p(max(temporal_max_dt, 1))

    t_score = time.time()
    p_exec = _score_reachability(
        reach_model,
        node_emb,
        src,
        dst,
        device,
        batch_size=int(gcfg.get('score_batch_size', 32768)),
        fallback_scale=dist_scale,
    )
    risk = (-np.log(np.clip(p_exec, float(gcfg.get('p_clip', 1e-4)), 1.0))).astype(np.float32)
    logger.log({'phase': 'edges', 'event': 'scored', 'candidate_edges': len(src), 'p_exec_mean': float(np.mean(p_exec)) if len(p_exec) else 0.0, 'risk_mean': float(np.mean(risk)) if len(risk) else 0.0, 'duration_sec': time.time() - t_score})

    if bool(gcfg.get('enforce_p_exec_min', False)):
        pmin = float(gcfg.get('p_exec_min', 0.05))
        # Never pmin-drop temporal support edges unless explicitly requested.
        preserve_temporal = bool(gcfg.get('preserve_temporal_edges', True))
        keep_p = (p_exec >= pmin) | (temporal_support & preserve_temporal)
        dropped = int((~keep_p).sum())
        src, dst, cost, risk, p_exec, kind, temporal_support = src[keep_p], dst[keep_p], cost[keep_p], risk[keep_p], p_exec[keep_p], kind[keep_p], temporal_support[keep_p]
        logger.log({'phase': 'edges', 'event': 'p_exec_filtered', 'p_exec_min': pmin, 'dropped_edges': dropped, 'kept_edges': len(src), 'preserve_temporal_edges': int(preserve_temporal)})

    top_k = int(gcfg.get('top_outgoing', 16))
    min_outgoing = int(gcfg.get('min_outgoing', 0))
    prune_lambda = float(gcfg.get('prune_lambda_risk', 0.25))
    preserve_temporal = bool(gcfg.get('preserve_temporal_edges', True))
    if top_k > 0:
        t_prune = time.time()
        score = cost + prune_lambda * risk
        keep_idx, temporal_preserved = _top_outgoing_keep_indices(
            src,
            score,
            temporal_support,
            num_nodes=n,
            top_k=top_k,
            min_outgoing=min_outgoing,
            preserve_temporal=preserve_temporal,
        )
        src, dst, cost, risk, p_exec, kind, temporal_support = src[keep_idx], dst[keep_idx], cost[keep_idx], risk[keep_idx], p_exec[keep_idx], kind[keep_idx], temporal_support[keep_idx]
        logger.log({'phase': 'edges', 'event': 'pruned', 'top_outgoing': top_k, 'min_outgoing': min_outgoing, 'prune_lambda_risk': prune_lambda, 'preserve_temporal_edges': int(preserve_temporal), 'temporal_edges_preserved': temporal_preserved, 'kept_edges': len(src), 'duration_sec': time.time() - t_prune})

    graph = BARSGraph(node_indices, node_emb, src, dst, cost, risk, p_exec, kind)
    out_degrees = np.asarray([len(x) for x in graph.outgoing_edges()], dtype=np.int32)
    logger.log({
        'phase': 'edges',
        'event': 'completed',
        'num_nodes': graph.num_nodes,
        'num_edges': graph.num_edges,
        'mean_out_degree': graph.num_edges / max(1, graph.num_nodes),
        'min_out_degree': int(out_degrees.min()) if len(out_degrees) else 0,
        'zero_out_degree_nodes': int((out_degrees == 0).sum()) if len(out_degrees) else 0,
        'temporal_edge_rate': float((kind == EDGE_KIND_TEMPORAL).mean()) if len(kind) else 0.0,
        'p_exec_mean': float(np.mean(p_exec)) if len(p_exec) else 0.0,
        'p_exec_p10': float(np.quantile(p_exec, 0.10)) if len(p_exec) else 0.0,
        'risk_mean': float(np.mean(risk)) if len(risk) else 0.0,
        'cost_mean': float(np.mean(cost)) if len(cost) else 0.0,
        'duration_sec': time.time() - t_build,
    })
    return graph
