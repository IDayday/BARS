from __future__ import annotations

import time
from typing import Dict, Tuple
import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from sklearn.cluster import MiniBatchKMeans

from bars.common.logging import CSVLogger
from bars.data.trajectories import OfflineDataset
from .ann import KNNIndex


def farthest_point_sampling(x: np.ndarray, k: int, rng: np.random.Generator, start_idx: int | None = None) -> np.ndarray:
    n = len(x)
    k = min(int(k), n)
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    selected = np.empty(k, dtype=np.int64)
    selected[0] = int(start_idx if start_idx is not None else rng.integers(0, n))
    min_dist = np.sum((x - x[selected[0]]) ** 2, axis=1)
    for i in range(1, k):
        selected[i] = int(np.argmax(min_dist))
        min_dist = np.minimum(min_dist, np.sum((x - x[selected[i]]) ** 2, axis=1))
    return selected


def _nearest_to_centers(x: np.ndarray, centers: np.ndarray, cfg: Dict | None = None) -> np.ndarray:
    ann = KNNIndex.from_config(x, cfg or {}, prefix="ann")
    return np.unique(ann.kneighbors(centers, 1, return_distance=False)[:, 0].astype(np.int64))


def _rank01(score: np.ndarray) -> np.ndarray:
    if len(score) == 0:
        return score
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.linspace(0, 1, len(score), dtype=np.float32)
    return ranks


def _build_knn_sparse(x: np.ndarray, k: int, cfg: Dict | None = None) -> Tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    k_eff = min(k + 1, len(x))
    ann = KNNIndex.from_config(x, cfg or {}, prefix="ann")
    dist, ind = ann.kneighbors(x, k_eff, return_distance=True)
    if k_eff <= 1:
        mat = sparse.csr_matrix((len(x), len(x)))
        return mat, np.empty((len(x), 0), dtype=np.int64), np.empty((len(x), 0), dtype=np.float32)
    neigh = ind[:, 1:].astype(np.int64)
    neigh_dist = dist[:, 1:].astype(np.float32)
    sigma = float(np.median(neigh_dist) + 1e-6)
    rows = np.repeat(np.arange(len(x), dtype=np.int64), neigh.shape[1])
    cols = neigh.reshape(-1)
    vals = np.exp(-((neigh_dist.reshape(-1) ** 2) / (sigma ** 2))).astype(np.float32)
    # Symmetrize by adding reverse entries; CSR summation handles duplicates.
    rows2 = np.concatenate([rows, cols])
    cols2 = np.concatenate([cols, rows])
    vals2 = np.concatenate([vals, vals])
    mat = sparse.coo_matrix((vals2, (rows2, cols2)), shape=(len(x), len(x))).tocsr()
    mat.eliminate_zeros()
    return mat, neigh, neigh_dist


def _spectral_labels_full(x: np.ndarray, knn: int, clusters: int, seed: int, gcfg: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    adj, neigh, neigh_dist = _build_knn_sparse(x, knn, {"graph": gcfg})
    lap = csgraph.laplacian(adj, normed=True)
    metrics = {'spectral_solver_used': 'full', 'support_edges': float(adj.nnz // 2)}
    try:
        k_eig = max(1, min(clusters, len(x) - 2))
        vals, vecs = eigsh(
            lap,
            k=k_eig,
            which='SM',
            tol=float(gcfg.get('spectral_eig_tol', 1e-3)),
            maxiter=int(gcfg.get('spectral_eig_maxiter', 2000)),
        )
        feat = vecs[:, 1:] if vecs.shape[1] > 1 else vecs
        labels = MiniBatchKMeans(n_clusters=clusters, random_state=seed, batch_size=4096, n_init=3).fit_predict(feat)
    except Exception as exc:
        metrics['spectral_fallback'] = f'kmeans_embedding:{type(exc).__name__}'
        labels = MiniBatchKMeans(n_clusters=clusters, random_state=seed, batch_size=4096, n_init=3).fit_predict(x)
    return labels.astype(np.int32), neigh, neigh_dist, metrics


def _spectral_labels_landmark(x: np.ndarray, knn: int, clusters: int, seed: int, gcfg: Dict, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    n_landmarks = min(int(gcfg.get('spectral_landmarks', 2048)), len(x))
    method = str(gcfg.get('spectral_landmark_method', 'fps')).lower()
    if n_landmarks >= len(x):
        return _spectral_labels_full(x, knn, clusters, seed, gcfg)
    t0 = time.time()
    if method == 'random':
        landmark_local = rng.choice(len(x), size=n_landmarks, replace=False).astype(np.int64)
    else:
        landmark_local = farthest_point_sampling(x, n_landmarks, rng)
    lx = x[landmark_local]
    adj_l, _, _ = _build_knn_sparse(lx, knn, {"graph": gcfg})
    lap_l = csgraph.laplacian(adj_l, normed=True)
    metrics: Dict[str, float | str] = {
        'spectral_solver_used': 'landmark',
        'spectral_landmarks': float(n_landmarks),
        'spectral_landmark_seconds': float(time.time() - t0),
        'support_edges': float(adj_l.nnz // 2),
    }
    try:
        k_eig = max(1, min(clusters, len(lx) - 2))
        vals, vecs = eigsh(
            lap_l,
            k=k_eig,
            which='SM',
            tol=float(gcfg.get('spectral_eig_tol', 1e-3)),
            maxiter=int(gcfg.get('spectral_eig_maxiter', 1000)),
        )
        feat_l = vecs[:, 1:] if vecs.shape[1] > 1 else vecs
        labels_l = MiniBatchKMeans(n_clusters=clusters, random_state=seed, batch_size=4096, n_init=3).fit_predict(feat_l)
    except Exception as exc:
        metrics['spectral_fallback'] = f'landmark_kmeans_embedding:{type(exc).__name__}'
        labels_l = MiniBatchKMeans(n_clusters=clusters, random_state=seed, batch_size=4096, n_init=3).fit_predict(lx)
    # Assign every support state to its nearest landmark label. This preserves the
    # full support pool for bottleneck scoring without running eigsh on 12k+ nodes.
    ann_l = KNNIndex.from_config(lx, {"graph": gcfg}, prefix="ann")
    labels = labels_l[ann_l.kneighbors(x, 1, return_distance=False)[:, 0]]
    ann_x = KNNIndex.from_config(x, {"graph": gcfg}, prefix="ann")
    neigh_dist_full, neigh_full = ann_x.kneighbors(x, min(knn + 1, len(x)), return_distance=True)
    return labels.astype(np.int32), neigh_full[:, 1:].astype(np.int64), neigh_dist_full[:, 1:].astype(np.float32), metrics  # type: ignore[return-value]


def spectral_bottleneck_scores(dataset: OfflineDataset, embeddings: np.ndarray, pool_idx: np.ndarray, cfg: Dict) -> Tuple[np.ndarray, Dict[str, float | str]]:
    gcfg = cfg.get('graph', {})
    t0 = time.time()
    x = embeddings[pool_idx].astype(np.float32)
    knn = int(gcfg.get('support_knn', 12))
    clusters = max(2, min(int(gcfg.get('spectral_clusters', 8)), len(x) - 1))
    seed = int(cfg.get('seed', 0))
    rng = np.random.default_rng(seed + 991)
    solver = str(gcfg.get('spectral_solver', 'landmark')).lower()
    full_max = int(gcfg.get('spectral_full_max_nodes', 3000))

    if solver == 'full' or len(x) <= full_max:
        labels, neigh, neigh_dist, metrics = _spectral_labels_full(x, knn, clusters, seed, gcfg)
    else:
        labels, neigh, neigh_dist, metrics = _spectral_labels_landmark(x, knn, clusters, seed, gcfg, rng)

    boundary = np.zeros(len(x), dtype=np.float32)
    if neigh.size:
        boundary = (labels[neigh] != labels[:, None]).mean(axis=1).astype(np.float32)

    pos = {int(g): p for p, g in enumerate(pool_idx)}
    transition = np.zeros(len(x), dtype=np.float32)
    for p, g in enumerate(pool_idx):
        q = pos.get(int(g) + 1)
        if q is not None and dataset.traj_id[g] == dataset.traj_id[int(g) + 1] and labels[p] != labels[q]:
            transition[p] += 1.0
            transition[q] += 1.0

    # A sparse graph degree is not available under landmark mode; use local
    # neighbor distance as a rarity/proximity proxy instead of recomputing a full
    # weighted graph just for this score.
    local_dist = neigh_dist.mean(axis=1).astype(np.float32) if neigh_dist.size else np.ones(len(x), dtype=np.float32)
    score = _rank01(boundary) + _rank01(transition) + 0.25 * _rank01(local_dist)
    metrics.update({
        'support_nodes': float(len(x)),
        'boundary_mean': float(boundary.mean()) if len(boundary) else 0.0,
        'transition_cross_mean': float(transition.mean()) if len(transition) else 0.0,
        'spectral_seconds': float(time.time() - t0),
    })
    return score.astype(np.float32), metrics


def select_graph_nodes(dataset: OfflineDataset, embeddings: np.ndarray, cfg: Dict, logger: CSVLogger) -> np.ndarray:
    t0 = time.time()
    gcfg = cfg.get('graph', {})
    method = str(gcfg.get('node_method', 'bars')).lower()
    num_nodes = int(gcfg.get('num_nodes', 500))
    rng = np.random.default_rng(int(cfg.get('seed', 0)) + 101)
    max_support = min(int(gcfg.get('max_support_states', 20000)), dataset.size)
    pool_idx = rng.choice(dataset.size, size=max_support, replace=False) if max_support < dataset.size else np.arange(dataset.size)
    pool_emb = embeddings[pool_idx].astype(np.float32)
    logger.log({'phase': 'nodes', 'event': 'start', 'node_method': method, 'requested_nodes': num_nodes, 'pool_size': len(pool_idx)})

    if method == 'random':
        chosen = rng.choice(dataset.size, size=min(num_nodes, dataset.size), replace=False)
        logger.log({'phase': 'nodes', 'event': 'completed', 'node_method': method, 'num_nodes': len(chosen), 'duration_sec': time.time() - t0})
        return chosen.astype(np.int64)

    if method == 'fps':
        chosen = pool_idx[farthest_point_sampling(pool_emb, num_nodes, rng)]
        logger.log({'phase': 'nodes', 'event': 'completed', 'node_method': method, 'num_nodes': len(chosen), 'duration_sec': time.time() - t0})
        return chosen.astype(np.int64)

    if method == 'kmeans':
        km = MiniBatchKMeans(n_clusters=min(num_nodes, len(pool_idx)), random_state=int(cfg.get('seed', 0)), batch_size=4096, n_init=3).fit(pool_emb)
        chosen = pool_idx[_nearest_to_centers(pool_emb, km.cluster_centers_, cfg)]
        logger.log({'phase': 'nodes', 'event': 'completed', 'node_method': method, 'num_nodes': len(chosen), 'duration_sec': time.time() - t0})
        return chosen.astype(np.int64)

    if method in {'spectral', 'bars'}:
        score, metrics = spectral_bottleneck_scores(dataset, embeddings, pool_idx, cfg)
        nb = min(int(gcfg.get('num_bottleneck_nodes', max(1, int(0.35 * num_nodes)))), num_nodes, len(pool_idx))
        top_local = np.argsort(-score)[:nb]
        bottleneck = pool_idx[top_local]
        if method == 'spectral':
            chosen = bottleneck
        else:
            remaining = max(0, num_nodes - len(bottleneck))
            mask = np.ones(len(pool_idx), dtype=bool)
            mask[top_local] = False
            cand_idx = pool_idx[mask]
            cand_emb = embeddings[cand_idx].astype(np.float32)
            if remaining > 0 and len(cand_idx) > 0:
                # Avoid the large cand x bottleneck broadcast used in the first
                # prototype. NearestNeighbors is faster and memory-stable as node
                # budgets grow.
                dist_to_b, _ = KNNIndex.from_config(embeddings[bottleneck].astype(np.float32), cfg, prefix="ann").kneighbors(cand_emb, 1, return_distance=True)
                dist_to_b = dist_to_b.reshape(-1)
                keep = dist_to_b > np.quantile(dist_to_b, float(gcfg.get('anchor_exclude_quantile', 0.10)))
                if keep.sum() >= remaining:
                    cand_idx = cand_idx[keep]
                    cand_emb = cand_emb[keep]
                anchors = cand_idx[farthest_point_sampling(cand_emb, remaining, rng)]
                chosen = np.unique(np.concatenate([bottleneck, anchors]))
            else:
                chosen = bottleneck
        logger.log({'phase': 'nodes', 'event': 'completed', 'node_method': method, 'num_nodes': len(chosen), 'duration_sec': time.time() - t0, **metrics})
        return chosen[:num_nodes].astype(np.int64)

    raise ValueError(f'Unknown graph.node_method={method}')
