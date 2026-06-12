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
    "num_first_edge_segments",
    "num_bridge_segments",
    "num_bridge_episodes",
    "min_bridge_h",
    "median_bridge_h",
    "compatibility_support_rate",
]

PATH_COMPATIBILITY_COLUMNS = [
    "query_id",
    "reachable",
    "num_adjacent_edge_pairs",
    "strict_compatible_rate",
]


def _segments_by_edge(edge_segments: dict[str, np.ndarray]) -> dict[int, dict[str, Any]]:
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
    indexed: dict[int, dict[str, Any]] = {}
    for edge_id, group in df.groupby("edge_id", sort=False):
        init_by_ep = {}
        term_by_ep = {}
        for ep_id, ep_group in group.groupby("ep_id", sort=False):
            init_by_ep[int(ep_id)] = np.sort(ep_group["global_i"].to_numpy(dtype=np.int64))
            term_by_ep[int(ep_id)] = np.sort(ep_group["global_j"].to_numpy(dtype=np.int64))
        indexed[int(edge_id)] = {
            "num_segments": int(group.shape[0]),
            "init_by_ep": init_by_ep,
            "term_by_ep": term_by_ep,
        }
    return indexed


def _empty_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "num_adjacent_edge_pairs": 0,
                "cluster_compatible_rate": 0.0,
                "strict_compatible_rate": 0.0,
                "incompatible_rate": 0.0,
                "mean_num_bridge_segments": 0.0,
                "median_num_bridge_segments": 0.0,
                "mean_compatibility_support_rate": 0.0,
                "lcb_compatible_rate": 0.0,
            }
        ]
    )


def _empty_segment_index() -> dict[str, Any]:
    return {"num_segments": 0, "init_by_ep": {}, "term_by_ep": {}}


def _bridge_deltas(terminations: np.ndarray, initiations: np.ndarray, H_intra: int) -> np.ndarray:
    if terminations.size == 0 or initiations.size == 0:
        return np.empty(0, dtype=np.int64)
    left = np.searchsorted(initiations, terminations, side="left")
    right = np.searchsorted(initiations, terminations + int(H_intra), side="right")
    if not np.any(right > left):
        return np.empty(0, dtype=np.int64)
    parts = [
        initiations[int(l) : int(r)] - int(term)
        for term, l, r in zip(terminations, left, right)
        if int(r) > int(l)
    ]
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)


def _strict_pair_support(first: dict[str, Any], second: dict[str, Any], H_intra: int) -> dict[str, float | int | bool]:
    num_first = int(first["num_segments"])
    first_terms = first["term_by_ep"]
    second_inits = second["init_by_ep"]
    if num_first == 0 or not first_terms or not second_inits:
        return {
            "strict_compatible": False,
            "num_first_edge_segments": num_first,
            "num_bridge_segments": 0,
            "num_bridge_episodes": 0,
            "min_bridge_h": 0.0,
            "median_bridge_h": 0.0,
            "compatibility_support_rate": 0.0,
        }
    bridge_parts = []
    bridge_episodes = 0
    for ep_id in first_terms.keys() & second_inits.keys():
        deltas = _bridge_deltas(first_terms[ep_id], second_inits[ep_id], H_intra)
        if deltas.size == 0:
            continue
        bridge_parts.append(deltas)
        bridge_episodes += 1
    bridge_h = np.concatenate(bridge_parts) if bridge_parts else np.empty(0, dtype=np.int64)
    num_bridge = int(bridge_h.size)
    return {
        "strict_compatible": bool(num_bridge > 0),
        "num_first_edge_segments": num_first,
        "num_bridge_segments": num_bridge,
        "num_bridge_episodes": int(bridge_episodes),
        "min_bridge_h": float(np.min(bridge_h)) if bridge_h.size else 0.0,
        "median_bridge_h": float(np.median(bridge_h)) if bridge_h.size else 0.0,
        "compatibility_support_rate": float(num_bridge / max(1, num_first)),
    }


def compute_edge_compatibility(
    option_edges_df: pd.DataFrame,
    edge_segments: dict[str, np.ndarray],
    labels: np.ndarray,
    pair_records: dict[str, Any],
    H_intra: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    del labels, pair_records
    if option_edges_df.empty:
        return _empty_summary(), pd.DataFrame(columns=COMPATIBILITY_PAIR_COLUMNS)

    segments = _segments_by_edge(edge_segments)
    empty_segments = _empty_segment_index()
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
            support = _strict_pair_support(
                segments.get(int(first.edge_id), empty_segments),
                segments.get(int(second.edge_id), empty_segments),
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
                    **support,
                }
            )

    pair_df = pd.DataFrame(rows)
    if pair_df.empty:
        return _empty_summary(), pd.DataFrame(columns=COMPATIBILITY_PAIR_COLUMNS)
    strict_rate = float(pair_df["strict_compatible"].mean())
    n = int(pair_df.shape[0])
    lcb = strict_rate - 1.96 * float(np.sqrt(strict_rate * (1.0 - strict_rate) / max(1, n)))
    summary = pd.DataFrame(
        [
            {
                "num_adjacent_edge_pairs": n,
                "cluster_compatible_rate": 1.0,
                "strict_compatible_rate": strict_rate,
                "incompatible_rate": float(1.0 - strict_rate),
                "mean_num_bridge_segments": float(pair_df["num_bridge_segments"].mean()),
                "median_num_bridge_segments": float(pair_df["num_bridge_segments"].median()),
                "mean_compatibility_support_rate": float(pair_df["compatibility_support_rate"].mean()),
                "lcb_compatible_rate": float(max(0.0, lcb)),
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
