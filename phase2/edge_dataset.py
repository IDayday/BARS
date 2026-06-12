from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


EPS = 1e-12

OPTION_EDGE_COLUMNS = [
    "edge_id",
    "src",
    "dst",
    "num_segments",
    "num_episodes",
    "min_h",
    "median_h",
    "mean_h",
    "max_h",
    "support_count",
    "reverse_support_count",
    "asymmetry",
    "src_density",
    "dst_density",
    "src_bottleneck_score",
    "dst_bottleneck_score",
    "edge_bottleneck_score",
]

SELF_LOOP_COLUMNS = [
    "cluster",
    "num_segments",
    "num_episodes",
    "min_h",
    "median_h",
    "mean_h",
    "max_h",
]


def _arr(records: dict[str, Any], key: str) -> np.ndarray:
    value = records[key]
    return value.to_numpy() if hasattr(value, "to_numpy") else np.asarray(value)


def _selected_info(selected_nodes: pd.DataFrame) -> tuple[set[int], dict[int, dict[str, float]]]:
    selected = selected_nodes[selected_nodes["selected"]].copy()
    selected_clusters = set(selected["cluster"].astype(int).tolist())
    info: dict[int, dict[str, float]] = {}
    for row in selected.itertuples(index=False):
        info[int(row.cluster)] = {
            "density": float(row.density),
            "bottleneck_score": float(row.bottleneck_score),
        }
    return selected_clusters, info


def _empty_segments() -> dict[str, np.ndarray]:
    return {
        "edge_id": np.empty(0, dtype=np.int64),
        "ep_id": np.empty(0, dtype=np.int64),
        "t": np.empty(0, dtype=np.int64),
        "h": np.empty(0, dtype=np.int64),
        "global_i": np.empty(0, dtype=np.int64),
        "global_j": np.empty(0, dtype=np.int64),
    }


def _empty_option_edges() -> pd.DataFrame:
    return pd.DataFrame(columns=OPTION_EDGE_COLUMNS)


def _empty_self_loop_summary() -> pd.DataFrame:
    return pd.DataFrame(columns=SELF_LOOP_COLUMNS)


def build_option_edges(
    pair_records: dict[str, np.ndarray],
    labels: np.ndarray,
    selected_nodes: pd.DataFrame,
    H: int,
    min_support: int,
    min_episodes: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Collect data-supported option edges between selected clusters."""

    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    selected_clusters, info = _selected_info(selected_nodes)
    if not selected_clusters:
        return _empty_option_edges(), _empty_segments()

    h = _arr(pair_records, "h").astype(np.int64)
    global_i = _arr(pair_records, "global_i").astype(np.int64)
    global_j = _arr(pair_records, "global_j").astype(np.int64)
    valid = (h <= int(H)) & (global_i >= 0) & (global_i < labels.size) & (global_j >= 0) & (
        global_j < labels.size
    )
    if not np.any(valid):
        return _empty_option_edges(), _empty_segments()

    src = labels[global_i[valid]]
    dst = labels[global_j[valid]]
    selected_mask = np.isin(src, list(selected_clusters)) & np.isin(dst, list(selected_clusters))
    inter_mask = selected_mask & (src != dst)
    if not np.any(inter_mask):
        return _empty_option_edges(), _empty_segments()

    base = pd.DataFrame(
        {
            "src": src[inter_mask].astype(np.int64),
            "dst": dst[inter_mask].astype(np.int64),
            "ep_id": _arr(pair_records, "ep_id")[valid][inter_mask].astype(np.int64),
            "t": _arr(pair_records, "t")[valid][inter_mask].astype(np.int64),
            "h": h[valid][inter_mask].astype(np.int64),
            "global_i": global_i[valid][inter_mask].astype(np.int64),
            "global_j": global_j[valid][inter_mask].astype(np.int64),
        }
    )

    grouped = base.groupby(["src", "dst"], sort=True)
    edge_df = grouped.agg(
        num_segments=("h", "size"),
        num_episodes=("ep_id", pd.Series.nunique),
        min_h=("h", "min"),
        median_h=("h", "median"),
        mean_h=("h", "mean"),
        max_h=("h", "max"),
    ).reset_index()
    edge_df = edge_df[
        (edge_df["num_segments"] >= int(min_support))
        & (edge_df["num_episodes"] >= int(min_episodes))
    ].copy()
    if edge_df.empty:
        return _empty_option_edges(), _empty_segments()

    support_lookup = {
        (int(row.src), int(row.dst)): int(row.num_segments)
        for row in edge_df.itertuples(index=False)
    }
    edge_df["support_count"] = edge_df["num_segments"].astype(int)
    edge_df["reverse_support_count"] = [
        int(support_lookup.get((int(row.dst), int(row.src)), 0))
        for row in edge_df.itertuples(index=False)
    ]
    edge_df["asymmetry"] = (
        (edge_df["support_count"] - edge_df["reverse_support_count"])
        / (edge_df["support_count"] + edge_df["reverse_support_count"] + EPS)
    )
    edge_df["src_density"] = edge_df["src"].map(lambda x: info[int(x)]["density"])
    edge_df["dst_density"] = edge_df["dst"].map(lambda x: info[int(x)]["density"])
    edge_df["src_bottleneck_score"] = edge_df["src"].map(
        lambda x: info[int(x)]["bottleneck_score"]
    )
    edge_df["dst_bottleneck_score"] = edge_df["dst"].map(
        lambda x: info[int(x)]["bottleneck_score"]
    )
    edge_df["edge_bottleneck_score"] = np.maximum(
        edge_df["src_bottleneck_score"],
        edge_df["dst_bottleneck_score"],
    )
    edge_df = edge_df.sort_values(["src", "dst"], kind="mergesort").reset_index(drop=True)
    edge_df.insert(0, "edge_id", np.arange(edge_df.shape[0], dtype=np.int64))

    edge_ids = edge_df[["edge_id", "src", "dst"]]
    seg_df = base.merge(edge_ids, on=["src", "dst"], how="inner")
    seg_df = seg_df.sort_values(["edge_id", "ep_id", "t", "h"], kind="mergesort")
    edge_segments = {
        "edge_id": seg_df["edge_id"].to_numpy(dtype=np.int64),
        "ep_id": seg_df["ep_id"].to_numpy(dtype=np.int64),
        "t": seg_df["t"].to_numpy(dtype=np.int64),
        "h": seg_df["h"].to_numpy(dtype=np.int64),
        "global_i": seg_df["global_i"].to_numpy(dtype=np.int64),
        "global_j": seg_df["global_j"].to_numpy(dtype=np.int64),
    }
    return edge_df, edge_segments


def build_self_loop_summary(
    pair_records: dict[str, np.ndarray],
    labels: np.ndarray,
    selected_nodes: pd.DataFrame,
    H: int,
) -> pd.DataFrame:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    selected_clusters, _ = _selected_info(selected_nodes)
    if not selected_clusters:
        return _empty_self_loop_summary()
    h = _arr(pair_records, "h").astype(np.int64)
    global_i = _arr(pair_records, "global_i").astype(np.int64)
    global_j = _arr(pair_records, "global_j").astype(np.int64)
    valid = (h <= int(H)) & (global_i >= 0) & (global_i < labels.size) & (global_j >= 0) & (
        global_j < labels.size
    )
    src = labels[global_i[valid]]
    dst = labels[global_j[valid]]
    mask = np.isin(src, list(selected_clusters)) & (src == dst)
    if not np.any(mask):
        return _empty_self_loop_summary()
    df = pd.DataFrame(
        {
            "cluster": src[mask].astype(np.int64),
            "ep_id": _arr(pair_records, "ep_id")[valid][mask].astype(np.int64),
            "h": h[valid][mask].astype(np.int64),
        }
    )
    return df.groupby("cluster", sort=True).agg(
        num_segments=("h", "size"),
        num_episodes=("ep_id", pd.Series.nunique),
        min_h=("h", "min"),
        median_h=("h", "median"),
        mean_h=("h", "mean"),
        max_h=("h", "max"),
    ).reset_index()
