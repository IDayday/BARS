from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


STATE_RISK_COLUMNS = [
    "segment_source",
    "segment_edge_id",
    "planner_edge_id",
    "src",
    "dst",
    "num_initiation_samples",
    "min_initiation_distance",
    "median_initiation_distance",
    "distance_scale",
    "state_conditioned_risk_score",
    "state_conditioned_risk_penalty",
]


def _segments_for_edge(segments: dict[str, np.ndarray], edge_id: int) -> np.ndarray:
    edge_ids = np.asarray(segments.get("edge_id", []), dtype=np.int64)
    if edge_ids.size == 0:
        return np.empty(0, dtype=np.int64)
    return np.flatnonzero(edge_ids == int(edge_id))


def _edge_identity(row: Any, source: str) -> tuple[str, int, int]:
    edge_id = int(getattr(row, "edge_id"))
    bank_edge_id = getattr(row, "bank_edge_id", np.nan)
    if source == "graph" and bool(getattr(row, "is_repair_edge", False)) and pd.notna(bank_edge_id):
        return "bank", int(bank_edge_id), edge_id
    return source, edge_id, edge_id


def _select_dims(values: np.ndarray, dims: list[int] | None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    else:
        arr = arr.reshape(arr.shape[0], -1)
    if dims is not None:
        return arr[:, [int(d) for d in dims]]
    return arr


def _candidate_edge_rows(edges: pd.DataFrame, *, source: str, current_cluster: int) -> list[Any]:
    if edges is None or edges.empty:
        return []
    return [row for row in edges.itertuples(index=False) if int(getattr(row, "src")) == int(current_cluster)]


def compute_state_conditioned_initiation_scores(
    graph_edges: pd.DataFrame,
    *,
    bank_edges: pd.DataFrame | None,
    graph_segments: dict[str, np.ndarray],
    bank_segments: dict[str, np.ndarray],
    observations: np.ndarray,
    current_obs: np.ndarray,
    current_cluster: int,
    distance_dims: list[int] | None = None,
    max_candidates_per_edge: int = 128,
    distance_scale: float | str | None = "auto",
    penalty_weight: float = 1.0,
) -> pd.DataFrame:
    """Score outgoing support edges by current-state distance to their initiations."""

    rows: list[dict[str, float | int | str]] = []
    observations = np.asarray(observations, dtype=np.float32)
    current = _select_dims(np.asarray(current_obs, dtype=np.float32).reshape(1, -1), distance_dims)
    candidates = [
        *[(row, "graph") for row in _candidate_edge_rows(graph_edges, source="graph", current_cluster=current_cluster)],
        *[(row, "bank") for row in _candidate_edge_rows(bank_edges, source="bank", current_cluster=current_cluster)],
    ]

    for row, source in candidates:
        segment_source, segment_edge_id, planner_edge_id = _edge_identity(row, source)
        store = bank_segments if segment_source == "bank" else graph_segments
        seg_idx = _segments_for_edge(store, segment_edge_id)
        if seg_idx.size == 0 or "global_i" not in store:
            continue
        if seg_idx.size > int(max_candidates_per_edge):
            rng = np.random.default_rng(int(segment_edge_id))
            seg_idx = rng.choice(seg_idx, size=int(max_candidates_per_edge), replace=False)
        init_idx = np.asarray(store["global_i"], dtype=np.int64)[seg_idx]
        init_idx = init_idx[(init_idx >= 0) & (init_idx < observations.shape[0])]
        if init_idx.size == 0:
            continue
        init_obs = _select_dims(observations[init_idx], distance_dims)
        distances = np.linalg.norm(init_obs - current, axis=1)
        rows.append(
            {
                "segment_source": segment_source,
                "segment_edge_id": int(segment_edge_id),
                "planner_edge_id": int(planner_edge_id),
                "src": int(getattr(row, "src")),
                "dst": int(getattr(row, "dst")),
                "num_initiation_samples": int(init_idx.size),
                "min_initiation_distance": float(np.min(distances)),
                "median_initiation_distance": float(np.median(distances)),
            }
        )
    if not rows:
        return pd.DataFrame(columns=STATE_RISK_COLUMNS)

    out = pd.DataFrame(rows)
    min_dist = pd.to_numeric(out["min_initiation_distance"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    finite = min_dist[np.isfinite(min_dist)]
    if finite.empty:
        return pd.DataFrame(columns=STATE_RISK_COLUMNS)
    if distance_scale is None or distance_scale == "auto":
        scale = max(float(np.nanmedian(finite.to_numpy(dtype=np.float64))), 1e-6)
    else:
        scale = max(float(distance_scale), 1e-6)
    risk = 1.0 - np.exp(-min_dist.fillna(float(np.nanmax(finite))) / scale)
    out["distance_scale"] = float(scale)
    out["state_conditioned_risk_score"] = risk.clip(lower=0.0, upper=1.0)
    out["state_conditioned_risk_penalty"] = float(penalty_weight) * out["state_conditioned_risk_score"]
    return out[STATE_RISK_COLUMNS]


def state_conditioned_penalty_map(scores: pd.DataFrame) -> dict[tuple[str, int], float]:
    penalties: dict[tuple[str, int], float] = {}
    if scores.empty:
        return penalties
    for row in scores.itertuples(index=False):
        key = (str(getattr(row, "segment_source")), int(getattr(row, "segment_edge_id")))
        value = float(getattr(row, "state_conditioned_risk_penalty", 0.0))
        if np.isfinite(value) and value > 0.0:
            penalties[key] = max(penalties.get(key, 0.0), value)
    return penalties
