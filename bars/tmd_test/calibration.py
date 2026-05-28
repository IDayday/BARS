from __future__ import annotations

import numpy as np

from .io import split_trajectories


def calibrate_tmd_scales(
    observations,
    terminals,
    provider,
    temporal_horizon_steps: int,
    sample_size: int = 50000,
    seed: int = 0,
    edge_quantile: float = 0.75,
    target_quantile: float = 0.90,
) -> dict:
    observations = np.asarray(observations, dtype=np.float32)
    terminals = np.asarray(terminals).astype(bool)
    H = int(max(1, temporal_horizon_steps))
    rng = np.random.default_rng(seed)
    trajs = [(s, e) for s, e in split_trajectories(terminals) if e - s > H]
    if not trajs:
        raise RuntimeError("No same-trajectory pairs are available for TMD calibration")
    src_idx = np.empty(int(sample_size), dtype=np.int64)
    dst_idx = np.empty(int(sample_size), dtype=np.int64)
    for k in range(int(sample_size)):
        s, e = trajs[int(rng.integers(0, len(trajs)))]
        i = int(rng.integers(s, e - H))
        src_idx[k] = i
        dst_idx[k] = i + H
    src_emb = provider.encode(observations[src_idx])
    dst_emb = provider.encode(observations[dst_idx])
    repr_l2 = np.linalg.norm(dst_emb - src_emb, axis=1)
    batch_size = int(getattr(provider, "batch_size", 512))
    tmd_dist = np.asarray(provider.paired_distance(src_emb, dst_emb, batch_size=batch_size)).reshape(-1)
    def q(x, pct):
        return float(np.quantile(np.asarray(x, dtype=np.float64), pct)) if len(x) else float("nan")
    edge_quantile = float(edge_quantile)
    target_quantile = float(target_quantile)
    return {
        "temporal_horizon_steps": H,
        "sample_pairs": int(len(src_idx)),
        "edge_quantile": edge_quantile,
        "target_quantile": target_quantile,
        "repr_l2_q50": q(repr_l2, 0.50),
        "repr_l2_q75": q(repr_l2, 0.75),
        "repr_l2_q90": q(repr_l2, 0.90),
        "tmd_dist_q25": q(tmd_dist, 0.25),
        "tmd_dist_q50": q(tmd_dist, 0.50),
        "tmd_dist_q75": q(tmd_dist, 0.75),
        "tmd_dist_q90": q(tmd_dist, 0.90),
        "repr_cluster_threshold": q(repr_l2, 0.50),
        "edge_distance_threshold": q(tmd_dist, edge_quantile),
        "target_distance_threshold": q(tmd_dist, target_quantile),
    }
