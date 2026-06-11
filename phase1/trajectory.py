from __future__ import annotations

import warnings
from typing import Any

import numpy as np


def split_into_episodes(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    """Split transition-aligned OGBench arrays into terminal-delimited episodes."""

    observations = np.asarray(dataset["observations"])
    actions = np.asarray(dataset["actions"])
    next_observations = np.asarray(dataset["next_observations"])
    terminals = np.asarray(dataset["terminals"]).reshape(-1).astype(bool)

    n = observations.shape[0]
    for key, arr in {
        "actions": actions,
        "next_observations": next_observations,
        "terminals": terminals,
    }.items():
        if arr.shape[0] != n:
            raise ValueError(
                f"{key} length {arr.shape[0]} does not match observations length {n}"
            )

    episodes: list[dict[str, Any]] = []
    start = 0
    for idx, terminal in enumerate(terminals):
        if not terminal:
            continue
        end = idx
        episodes.append(
            {
                "observations": observations[start : end + 1],
                "actions": actions[start : end + 1],
                "next_observations": next_observations[start : end + 1],
                "terminals": terminals[start : end + 1],
                "start_index": int(start),
                "end_index": int(end),
                "length": int(end - start + 1),
            }
        )
        start = idx + 1

    if start < n:
        warnings.warn(
            "Last dataset segment has no terminal flag; keeping it as an episode.",
            RuntimeWarning,
            stacklevel=2,
        )
        episodes.append(
            {
                "observations": observations[start:n],
                "actions": actions[start:n],
                "next_observations": next_observations[start:n],
                "terminals": terminals[start:n],
                "start_index": int(start),
                "end_index": int(n - 1),
                "length": int(n - start),
            }
        )
    return episodes


def _empty_pair_records(obs_dim: int, obs_dtype: np.dtype) -> dict[str, np.ndarray]:
    return {
        "ep_id": np.empty(0, dtype=np.int64),
        "t": np.empty(0, dtype=np.int64),
        "h": np.empty(0, dtype=np.int64),
        "global_i": np.empty(0, dtype=np.int64),
        "global_j": np.empty(0, dtype=np.int64),
        "obs_i": np.empty((0, obs_dim), dtype=obs_dtype),
        "obs_j": np.empty((0, obs_dim), dtype=obs_dtype),
    }


def _expand_pair_horizons(
    horizons: list[int],
    pair_mode: str,
    horizon_stride: int,
) -> list[int]:
    requested = sorted({int(h) for h in horizons if int(h) > 0})
    if not requested:
        return []

    pair_mode = pair_mode.lower()
    horizon_stride = max(1, int(horizon_stride))
    max_horizon = max(requested)
    if pair_mode == "exact":
        return requested
    if pair_mode == "dense_upto":
        return list(range(1, max_horizon + 1))
    if pair_mode == "strided_upto":
        values = list(range(1, max_horizon + 1, horizon_stride))
        if values[-1] != max_horizon:
            values.append(max_horizon)
        return values
    raise ValueError(
        f"Unsupported pair_mode {pair_mode!r}; expected exact, dense_upto, or strided_upto"
    )


def build_h_step_pairs(
    episodes: list[dict[str, Any]],
    horizons: list[int],
    max_pairs_per_horizon: int | None = None,
    seed: int = 0,
    pair_mode: str = "exact",
    horizon_stride: int = 1,
    store_observations: bool = True,
) -> dict[str, np.ndarray]:
    """Build within-episode future state pairs for support diagnostics."""

    if not episodes:
        return _empty_pair_records(obs_dim=0, obs_dtype=np.float32)

    pair_horizons = _expand_pair_horizons(horizons, pair_mode, horizon_stride)
    if not pair_horizons:
        first_obs = np.asarray(episodes[0]["observations"])
        return _empty_pair_records(int(np.prod(first_obs.shape[1:])), first_obs.dtype)

    rng = np.random.default_rng(seed)
    ep_id_parts: list[np.ndarray] = []
    t_parts: list[np.ndarray] = []
    h_parts: list[np.ndarray] = []
    gi_parts: list[np.ndarray] = []
    gj_parts: list[np.ndarray] = []
    obs_i_parts: list[np.ndarray] = []
    obs_j_parts: list[np.ndarray] = []

    for horizon in pair_horizons:
        h_ep_ids: list[np.ndarray] = []
        h_ts: list[np.ndarray] = []
        h_gi: list[np.ndarray] = []
        h_gj: list[np.ndarray] = []
        h_obs_i: list[np.ndarray] = []
        h_obs_j: list[np.ndarray] = []

        for ep_id, episode in enumerate(episodes):
            length = int(episode["length"])
            if length <= horizon:
                continue
            local_t = np.arange(0, length - horizon, dtype=np.int64)
            global_i = int(episode["start_index"]) + local_t
            global_j = global_i + horizon
            observations = np.asarray(episode["observations"])

            h_ep_ids.append(np.full(local_t.shape, ep_id, dtype=np.int64))
            h_ts.append(local_t)
            h_gi.append(global_i)
            h_gj.append(global_j)
            if store_observations:
                h_obs_i.append(observations[local_t])
                h_obs_j.append(observations[local_t + horizon])

        if not h_ts:
            continue

        ep_ids = np.concatenate(h_ep_ids)
        ts = np.concatenate(h_ts)
        global_i = np.concatenate(h_gi)
        global_j = np.concatenate(h_gj)
        obs_i = np.concatenate(h_obs_i) if store_observations else None
        obs_j = np.concatenate(h_obs_j) if store_observations else None

        if max_pairs_per_horizon is not None and ep_ids.shape[0] > max_pairs_per_horizon:
            keep = rng.choice(ep_ids.shape[0], size=max_pairs_per_horizon, replace=False)
            keep.sort()
            ep_ids = ep_ids[keep]
            ts = ts[keep]
            global_i = global_i[keep]
            global_j = global_j[keep]
            if store_observations and obs_i is not None and obs_j is not None:
                obs_i = obs_i[keep]
                obs_j = obs_j[keep]

        ep_id_parts.append(ep_ids)
        t_parts.append(ts)
        h_parts.append(np.full(ep_ids.shape, horizon, dtype=np.int64))
        gi_parts.append(global_i)
        gj_parts.append(global_j)
        if store_observations and obs_i is not None and obs_j is not None:
            obs_i_parts.append(obs_i)
            obs_j_parts.append(obs_j)

    first_obs = np.asarray(episodes[0]["observations"])
    obs_dim = int(np.prod(first_obs.shape[1:]))
    if not ep_id_parts:
        return _empty_pair_records(obs_dim, first_obs.dtype)

    if store_observations:
        obs_i = np.concatenate(obs_i_parts)
        obs_j = np.concatenate(obs_j_parts)
    else:
        obs_i = np.empty((0, obs_dim), dtype=first_obs.dtype)
        obs_j = np.empty((0, obs_dim), dtype=first_obs.dtype)

    return {
        "ep_id": np.concatenate(ep_id_parts),
        "t": np.concatenate(t_parts),
        "h": np.concatenate(h_parts),
        "global_i": np.concatenate(gi_parts),
        "global_j": np.concatenate(gj_parts),
        "obs_i": obs_i,
        "obs_j": obs_j,
    }


def count_pairs_by_h(pair_records: dict[str, np.ndarray]) -> dict[int, int]:
    h_values = np.asarray(pair_records["h"], dtype=np.int64)
    if h_values.size == 0:
        return {}
    unique, counts = np.unique(h_values, return_counts=True)
    return {int(h): int(count) for h, count in zip(unique, counts)}
