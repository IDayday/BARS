from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _wilson_lcb(successes: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    p = float(successes) / float(total)
    denom = 1.0 + z * z / float(total)
    center = p + z * z / (2.0 * float(total))
    margin = z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * float(total))) / float(total))
    return float(max(0.0, (center - margin) / denom))


def split_episodes(
    ep_ids: np.ndarray,
    heldout_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[set[int], set[int]]:
    unique = np.asarray(sorted(np.unique(np.asarray(ep_ids, dtype=np.int64)).tolist()), dtype=np.int64)
    if unique.size == 0:
        return set(), set()
    rng = np.random.default_rng(int(seed))
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    n_heldout = int(round(float(heldout_fraction) * shuffled.size))
    n_heldout = min(max(n_heldout, 1 if shuffled.size > 1 and heldout_fraction > 0 else 0), shuffled.size)
    heldout = set(int(x) for x in shuffled[:n_heldout])
    train = set(int(x) for x in shuffled[n_heldout:])
    if not train and heldout:
        moved = next(iter(heldout))
        heldout.remove(moved)
        train.add(moved)
    return train, heldout


def segment_indices_for_episode_split(
    edge_segments: dict[str, np.ndarray],
    heldout_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, set[int], set[int]]:
    ep_id = np.asarray(edge_segments["ep_id"], dtype=np.int64)
    train_eps, heldout_eps = split_episodes(ep_id, heldout_fraction=heldout_fraction, seed=seed)
    train_mask = np.asarray([int(x) in train_eps for x in ep_id], dtype=bool)
    heldout_mask = np.asarray([int(x) in heldout_eps for x in ep_id], dtype=bool)
    return np.flatnonzero(train_mask), np.flatnonzero(heldout_mask), train_eps, heldout_eps


def _segment_frame(edge_segments: dict[str, np.ndarray], indices: np.ndarray | None = None) -> pd.DataFrame:
    if indices is None:
        indices = np.arange(np.asarray(edge_segments["edge_id"]).shape[0], dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)
    return pd.DataFrame(
        {
            "edge_id": np.asarray(edge_segments["edge_id"], dtype=np.int64)[indices],
            "ep_id": np.asarray(edge_segments["ep_id"], dtype=np.int64)[indices],
            "global_i": np.asarray(edge_segments["global_i"], dtype=np.int64)[indices],
            "global_j": np.asarray(edge_segments["global_j"], dtype=np.int64)[indices],
            "h": np.asarray(edge_segments["h"], dtype=np.int64)[indices],
        }
    )


def _aggregate_segments(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    columns = [
        "edge_id",
        f"{prefix}_num_segments",
        f"{prefix}_num_unique_starts",
        f"{prefix}_num_unique_terminations",
        f"{prefix}_num_episodes",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)
    out = (
        df.groupby("edge_id", sort=True)
        .agg(
            **{
                f"{prefix}_num_segments": ("edge_id", "size"),
                f"{prefix}_num_unique_starts": ("global_i", pd.Series.nunique),
                f"{prefix}_num_unique_terminations": ("global_j", pd.Series.nunique),
                f"{prefix}_num_episodes": ("ep_id", pd.Series.nunique),
            }
        )
        .reset_index()
    )
    return out


def compute_heldout_support(
    option_edges: pd.DataFrame,
    edge_segments: dict[str, np.ndarray],
    heldout_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    train_idx, heldout_idx, train_eps, heldout_eps = segment_indices_for_episode_split(
        edge_segments,
        heldout_fraction=heldout_fraction,
        seed=seed,
    )
    train_df = _segment_frame(edge_segments, train_idx)
    heldout_df = _segment_frame(edge_segments, heldout_idx)
    out = option_edges[["edge_id", "src", "dst"]].copy()
    out = out.merge(_aggregate_segments(train_df, "train"), on="edge_id", how="left")
    out = out.merge(_aggregate_segments(heldout_df, "heldout"), on="edge_id", how="left")
    count_cols = [col for col in out.columns if col.startswith("train_") or col.startswith("heldout_")]
    for col in count_cols:
        out[col] = out[col].fillna(0).astype(int)
    total = out["train_num_segments"] + out["heldout_num_segments"]
    out["heldout_support_binary"] = out["heldout_num_segments"] > 0
    out["heldout_support_rate"] = out["heldout_num_segments"] / np.maximum(1, total)
    out["heldout_support_lcb"] = [
        _wilson_lcb(int(success), int(n))
        for success, n in zip(out["heldout_num_segments"].to_numpy(), total.to_numpy())
    ]
    meta = {
        "num_train_episodes": int(len(train_eps)),
        "num_heldout_episodes": int(len(heldout_eps)),
        "num_train_segments": int(train_idx.size),
        "num_heldout_segments": int(heldout_idx.size),
        "heldout_fraction": float(heldout_fraction),
        "seed": int(seed),
    }
    return out, meta
