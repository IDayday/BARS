from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

COMPATIBILITY_PAIR_COLUMNS = [
    "edge_id_first",
    "edge_id_second",
    "first_src",
    "junction",
    "second_dst",
    "cluster_compatible",
    "strict_compatible",
]

PATH_COMPATIBILITY_COLUMNS = [
    "query_id",
    "reachable",
    "num_adjacent_edge_pairs",
    "strict_compatible_rate",
]


def _segments_by_edge(edge_segments: dict[str, np.ndarray]) -> dict[int, pd.DataFrame]:
    if edge_segments["edge_id"].size == 0:
        return {}
    df = pd.DataFrame(
        {
            "edge_id": edge_segments["edge_id"].astype(np.int64),
            "ep_id": edge_segments["ep_id"].astype(np.int64),
            "global_i": edge_segments["global_i"].astype(np.int64),
            "global_j": edge_segments["global_j"].astype(np.int64),
        }
    )
    return {int(edge_id): group for edge_id, group in df.groupby("edge_id", sort=False)}


def _strict_pair_compatible(first: pd.DataFrame, second: pd.DataFrame, H_intra: int) -> bool:
    if first.empty or second.empty:
        return False
    second_by_ep = {
        int(ep): np.sort(group["global_i"].to_numpy(dtype=np.int64))
        for ep, group in second.groupby("ep_id", sort=False)
    }
    for row in first.itertuples(index=False):
        starts = second_by_ep.get(int(row.ep_id))
        if starts is None or starts.size == 0:
            continue
        pos = np.searchsorted(starts, int(row.global_j), side="left")
        if pos < starts.size and 0 <= int(starts[pos]) - int(row.global_j) <= int(H_intra):
            return True
    return False


def compute_edge_compatibility(
    option_edges_df: pd.DataFrame,
    edge_segments: dict[str, np.ndarray],
    labels: np.ndarray,
    pair_records: dict[str, Any],
    H_intra: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    del labels, pair_records
    if option_edges_df.empty:
        summary = pd.DataFrame(
            [
                {
                    "num_adjacent_edge_pairs": 0,
                    "cluster_compatible_rate": 0.0,
                    "strict_compatible_rate": 0.0,
                    "incompatible_rate": 0.0,
                }
            ]
        )
        return summary, pd.DataFrame(columns=COMPATIBILITY_PAIR_COLUMNS)

    segments = _segments_by_edge(edge_segments)
    rows = []
    by_src = {
        int(src): group.copy()
        for src, group in option_edges_df.groupby("src", sort=False)
    }
    for first in option_edges_df.itertuples(index=False):
        next_edges = by_src.get(int(first.dst))
        if next_edges is None:
            continue
        for second in next_edges.itertuples(index=False):
            strict = _strict_pair_compatible(
                segments.get(int(first.edge_id), pd.DataFrame()),
                segments.get(int(second.edge_id), pd.DataFrame()),
                H_intra,
            )
            rows.append(
                {
                    "edge_id_first": int(first.edge_id),
                    "edge_id_second": int(second.edge_id),
                    "first_src": int(first.src),
                    "junction": int(first.dst),
                    "second_dst": int(second.dst),
                    "cluster_compatible": True,
                    "strict_compatible": bool(strict),
                }
            )

    pair_df = pd.DataFrame(rows)
    if pair_df.empty:
        summary = pd.DataFrame(
            [
                {
                    "num_adjacent_edge_pairs": 0,
                    "cluster_compatible_rate": 0.0,
                    "strict_compatible_rate": 0.0,
                    "incompatible_rate": 0.0,
                }
            ]
        )
        return summary, pd.DataFrame(columns=COMPATIBILITY_PAIR_COLUMNS)
    strict_rate = float(pair_df["strict_compatible"].mean())
    summary = pd.DataFrame(
        [
            {
                "num_adjacent_edge_pairs": int(pair_df.shape[0]),
                "cluster_compatible_rate": 1.0,
                "strict_compatible_rate": strict_rate,
                "incompatible_rate": float(1.0 - strict_rate),
            }
        ]
    )
    return summary, pair_df


def evaluate_path_compatibility(
    path_df: pd.DataFrame,
    pair_compatibility_df: pd.DataFrame,
) -> pd.DataFrame:
    if path_df.empty:
        return pd.DataFrame(columns=PATH_COMPATIBILITY_COLUMNS)
    strict_pairs = set()
    if not pair_compatibility_df.empty:
        strict_pairs = {
            (int(row.edge_id_first), int(row.edge_id_second))
            for row in pair_compatibility_df.itertuples(index=False)
            if bool(row.strict_compatible)
        }
    rows = []
    for row in path_df.itertuples(index=False):
        edge_ids = [int(x) for x in str(row.path_edge_ids).split() if x.strip()]
        adjacent = list(zip(edge_ids[:-1], edge_ids[1:]))
        if not adjacent:
            strict_rate = 1.0 if bool(row.reachable) else 0.0
        else:
            strict_rate = float(np.mean([(a, b) in strict_pairs for a, b in adjacent]))
        rows.append(
            {
                "query_id": int(row.query_id),
                "reachable": bool(row.reachable),
                "num_adjacent_edge_pairs": int(len(adjacent)),
                "strict_compatible_rate": strict_rate,
            }
        )
    return pd.DataFrame(rows)
