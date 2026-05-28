from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bars.conditioning import LowCondStats


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def load_stats(path: str | Path) -> LowCondStats:
    data = np.load(path)
    return LowCondStats(
        z_mean=data["z_mean"],
        z_std=data["z_std"],
        tdr_distance_q90=float(data["tdr_distance_q90"]),
        factor_mean=data["factor_mean"],
        factor_std=data["factor_std"],
        factor_dim_max=int(data["factor_dim_max"]),
        distance_clip=float(data.get("distance_clip", 5.0)),
        residual_clip=float(data.get("residual_clip", 5.0)),
    )


def save_stats(path: str | Path, stats: LowCondStats, metadata: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        z_mean=stats.z_mean,
        z_std=stats.z_std,
        tdr_distance_q90=np.asarray(stats.tdr_distance_q90, dtype=np.float32),
        factor_mean=stats.factor_mean,
        factor_std=stats.factor_std,
        factor_dim_max=np.asarray(stats.padded_factor_dim, dtype=np.int32),
        distance_clip=np.asarray(stats.distance_clip, dtype=np.float32),
        residual_clip=np.asarray(stats.residual_clip, dtype=np.float32),
    )
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def valid_future_indices(terminals: np.ndarray, horizon: int) -> np.ndarray:
    terminals = np.asarray(terminals).astype(bool)
    n = len(terminals)
    if horizon <= 0:
        return np.arange(n)
    idx = np.arange(0, max(n - horizon, 0), dtype=np.int64)
    if len(idx) == 0:
        return idx
    if terminals.any():
        csum = np.concatenate([[0], np.cumsum(terminals.astype(np.int32))])
        ok = (csum[idx + horizon] - csum[idx]) == 0
        idx = idx[ok]
    return idx


def sample_future_pairs(terminals: np.ndarray, horizons: list[int], num_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    src_parts = []
    dst_parts = []
    h_parts = []
    per_h = max(1, int(np.ceil(num_samples / max(len(horizons), 1))))
    for horizon in horizons:
        valid = valid_future_indices(terminals, horizon)
        if len(valid) == 0:
            continue
        pick = rng.choice(valid, size=per_h, replace=len(valid) < per_h)
        src_parts.append(pick)
        dst_parts.append(pick + horizon)
        h_parts.append(np.full((len(pick),), horizon, dtype=np.int32))
    src = np.concatenate(src_parts)[:num_samples]
    dst = np.concatenate(dst_parts)[:num_samples]
    hs = np.concatenate(h_parts)[:num_samples]
    order = rng.permutation(len(src))
    return src[order], dst[order], hs[order]


def trajectory_end_indices(terminals: np.ndarray) -> np.ndarray:
    terminals = np.asarray(terminals).astype(bool)
    if len(terminals) == 0:
        return np.asarray([], dtype=np.int64)
    ends = np.flatnonzero(terminals)
    if len(ends) == 0:
        return np.asarray([len(terminals) - 1], dtype=np.int64)
    if ends[-1] != len(terminals) - 1:
        ends = np.concatenate([ends, np.asarray([len(terminals) - 1], dtype=np.int64)])
    return ends.astype(np.int64)


def trajectory_end_for_indices(terminals: np.ndarray, indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    ends = trajectory_end_indices(terminals)
    if len(ends) == 0:
        return np.asarray([], dtype=np.int64)
    pos = np.searchsorted(ends, indices, side="left")
    pos = np.clip(pos, 0, len(ends) - 1)
    return ends[pos].astype(np.int64)


def antmaze_xy_factors(obs: np.ndarray) -> np.ndarray:
    return np.asarray(obs, dtype=np.float32)[..., :2]


def build_lowcond_batch(
    *,
    obs: np.ndarray,
    local_target_obs: np.ndarray,
    task_goal: np.ndarray | None,
    phi_obs: np.ndarray,
    phi_local: np.ndarray,
    stats: LowCondStats,
    variant: str,
) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    local_target_obs = np.asarray(local_target_obs, dtype=np.float32)
    phi_obs = np.asarray(phi_obs, dtype=np.float32)
    phi_local = np.asarray(phi_local, dtype=np.float32)
    z_obs = stats.normalize_z(phi_obs)
    z_local = stats.normalize_z(phi_local)
    diff = z_local - z_obs
    raw_dist = np.linalg.norm(diff, axis=-1, keepdims=True)
    direction = diff / np.maximum(raw_dist, stats.eps)
    direction = np.where(raw_dist <= stats.eps, 0.0, direction).astype(np.float32)
    if variant.endswith("_rawdist"):
        distance = np.clip(raw_dist / max(float(stats.tdr_distance_q90), stats.eps), 0.0, float(stats.distance_clip)).astype(np.float32)
    else:
        distance = np.log1p(raw_dist / max(float(stats.tdr_distance_q90), stats.eps))
        distance = np.clip(distance, 0.0, float(stats.distance_clip)).astype(np.float32)

    target = local_target_obs if (task_goal is None or variant.endswith("_localres")) else np.asarray(task_goal, dtype=np.float32)
    residual = antmaze_xy_factors(target) - antmaze_xy_factors(obs)
    residual = stats.normalize_residual(residual)
    mask = np.ones_like(residual, dtype=np.float32)
    masked = mask * residual

    parts: list[np.ndarray] = []
    if variant in {"tdr_only"}:
        parts.extend([direction, distance])
    elif variant in {"factor_only"}:
        parts.extend([mask, masked])
    elif variant in {"full", "full_localres", "full_rawdist"}:
        parts.extend([direction, distance, mask, masked])
    elif variant == "full_nomask":
        parts.extend([direction, distance, masked])
    else:
        raise ValueError(f"unknown lowcond variant: {variant}")
    cond = np.concatenate(parts, axis=-1).astype(np.float32)
    if not np.all(np.isfinite(cond)):
        raise FloatingPointError("lowcond batch contains NaN/Inf")
    return cond


def lowcond_dim(z_dim: int, factor_dim: int, variant: str) -> int:
    if variant == "tdr_only":
        return z_dim + 1
    if variant == "factor_only":
        return 2 * factor_dim
    if variant in {"full", "full_localres", "full_rawdist"}:
        return z_dim + 1 + 2 * factor_dim
    if variant == "full_nomask":
        return z_dim + 1 + factor_dim
    raise ValueError(variant)
