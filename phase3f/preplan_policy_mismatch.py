from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from phase3.edge_rollout import policy_action


PREPLAN_POLICY_COLUMNS = [
    "segment_source",
    "segment_edge_id",
    "planner_edge_id",
    "src",
    "dst",
    "num_policy_mse_samples",
    "min_preplan_policy_action_mse",
    "median_preplan_policy_action_mse",
    "mean_preplan_policy_action_mse",
]


def _segments_for_edge(segments: dict[str, np.ndarray], edge_id: int) -> np.ndarray:
    edge_ids = np.asarray(segments.get("edge_id", []), dtype=np.int64)
    if edge_ids.size == 0:
        return np.empty(0, dtype=np.int64)
    return np.flatnonzero(edge_ids == int(edge_id))


def _segment_store(source: str, graph_segments: dict[str, np.ndarray], bank_segments: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return bank_segments if source == "bank" else graph_segments


def compute_preplan_policy_mismatch_scores(
    candidate_edges: pd.DataFrame,
    *,
    graph_segments: dict[str, np.ndarray],
    bank_segments: dict[str, np.ndarray],
    observations: np.ndarray,
    actions: np.ndarray,
    policy: Any,
    device: str | None = None,
    max_candidates_per_edge: int = 64,
) -> pd.DataFrame:
    """Estimate edge-local BC mismatch before graph search for candidate edges."""

    if candidate_edges.empty or policy is None:
        return pd.DataFrame(columns=PREPLAN_POLICY_COLUMNS)
    observations = np.asarray(observations, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    rows: list[dict[str, Any]] = []
    for row in candidate_edges.itertuples(index=False):
        source = str(getattr(row, "segment_source"))
        edge_id = int(getattr(row, "segment_edge_id"))
        planner_edge_id = int(getattr(row, "planner_edge_id", edge_id))
        store = _segment_store(source, graph_segments, bank_segments)
        seg_idx = _segments_for_edge(store, edge_id)
        if seg_idx.size == 0 or "global_i" not in store or "global_j" not in store:
            continue
        if seg_idx.size > int(max_candidates_per_edge):
            rng = np.random.default_rng(edge_id)
            seg_idx = rng.choice(seg_idx, size=int(max_candidates_per_edge), replace=False)
        global_i = np.asarray(store["global_i"], dtype=np.int64)[seg_idx]
        global_j = np.asarray(store["global_j"], dtype=np.int64)[seg_idx]
        valid = (
            (global_i >= 0)
            & (global_i < observations.shape[0])
            & (global_j >= 0)
            & (global_j < observations.shape[0])
            & (global_i < actions.shape[0])
        )
        global_i = global_i[valid]
        global_j = global_j[valid]
        if global_i.size == 0:
            continue

        mse_values: list[float] = []
        for init_idx, term_idx in zip(global_i, global_j, strict=False):
            pred = policy_action(
                policy,
                observations[int(init_idx)],
                observations[int(term_idx)],
                remaining_h=1.0,
                edge_id=planner_edge_id,
                device=device,
            )
            target = actions[int(init_idx)].reshape(-1)
            mse_values.append(float(np.mean((np.asarray(pred, dtype=np.float32).reshape(-1) - target) ** 2)))
        if not mse_values:
            continue
        values = np.asarray(mse_values, dtype=np.float64)
        rows.append(
            {
                "segment_source": source,
                "segment_edge_id": edge_id,
                "planner_edge_id": planner_edge_id,
                "src": int(getattr(row, "src")),
                "dst": int(getattr(row, "dst")),
                "num_policy_mse_samples": int(values.size),
                "min_preplan_policy_action_mse": float(np.min(values)),
                "median_preplan_policy_action_mse": float(np.median(values)),
                "mean_preplan_policy_action_mse": float(np.mean(values)),
            }
        )
    if not rows:
        return pd.DataFrame(columns=PREPLAN_POLICY_COLUMNS)
    return pd.DataFrame(rows)[PREPLAN_POLICY_COLUMNS]


def preplan_policy_mse_map(scores: pd.DataFrame, *, value_col: str = "mean_preplan_policy_action_mse") -> dict[tuple[str, int], float]:
    if scores.empty:
        return {}
    out: dict[tuple[str, int], float] = {}
    for row in scores.itertuples(index=False):
        key = (str(getattr(row, "segment_source")), int(getattr(row, "segment_edge_id")))
        value = float(getattr(row, value_col, np.nan))
        if np.isfinite(value):
            out[key] = value
    return out
