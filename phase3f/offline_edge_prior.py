from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


OFFLINE_PRIOR_COLUMNS = [
    "segment_source",
    "segment_edge_id",
    "planner_edge_id",
    "src",
    "dst",
    "is_bank_connector",
    "is_repair_edge",
    "certification_reliability",
    "support_reliability",
    "diversity_reliability",
    "horizon_reliability",
    "policy_reliability",
    "compatibility_reliability",
    "offline_edge_prior_reliability",
    "offline_edge_prior_risk_score",
    "offline_edge_prior_penalty",
]


def _read_csv(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def _numeric(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.full(len(df), default, dtype=np.float64), index=df.index)
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def _normalize_positive(values: pd.Series, *, default: float = 0.0) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    finite = vals[np.isfinite(vals)]
    if finite.empty:
        return pd.Series(np.full(len(values), default, dtype=np.float64), index=values.index)
    lo = float(np.nanmin(finite))
    hi = float(np.nanmax(finite))
    if hi <= lo:
        return pd.Series(np.full(len(values), 1.0, dtype=np.float64), index=values.index)
    return ((vals.fillna(lo) - lo) / (hi - lo)).clip(lower=0.0, upper=1.0)


def _support_features(edges: pd.DataFrame) -> pd.DataFrame:
    out = edges.copy()
    starts = _numeric(out, "num_unique_starts", 0.0).clip(lower=0.0)
    episodes = _numeric(out, "num_unique_episodes", np.nan)
    if "num_unique_episodes" not in out.columns:
        episodes = _numeric(out, "num_episodes", 0.0).clip(lower=0.0)
    segments = _numeric(out, "num_segments", 0.0).clip(lower=0.0)
    horizon = _numeric(out, "median_h", np.nan)
    if horizon.isna().all():
        horizon = _numeric(out, "mean_h", 1.0)

    out["support_reliability"] = _normalize_positive(np.log1p(segments), default=0.0)
    out["diversity_reliability"] = (
        0.5 * _normalize_positive(np.log1p(starts), default=0.0)
        + 0.5 * _normalize_positive(np.log1p(episodes), default=0.0)
    ).clip(lower=0.0, upper=1.0)
    horizon_norm = _normalize_positive(horizon, default=0.0)
    out["horizon_reliability"] = (1.0 - horizon_norm).clip(lower=0.0, upper=1.0)
    return out


def _edge_key_for_row(row: Any, source: str) -> tuple[str, int]:
    bank_edge_id = getattr(row, "bank_edge_id", np.nan)
    is_repair = bool(getattr(row, "is_repair_edge", False))
    if source == "graph" and is_repair and pd.notna(bank_edge_id):
        return "bank", int(bank_edge_id)
    return source, int(getattr(row, "edge_id"))


def _cert_lookup(certification: pd.DataFrame) -> dict[tuple[str, int], dict[str, float]]:
    lookup: dict[tuple[str, int], dict[str, float]] = {}
    if certification.empty or "edge_id" not in certification.columns:
        return lookup

    for row in certification.itertuples(index=False):
        is_repair = bool(getattr(row, "is_repair_edge", False))
        bank_edge_id = getattr(row, "bank_edge_id", np.nan)
        if is_repair and pd.notna(bank_edge_id):
            key = ("bank", int(bank_edge_id))
        else:
            key = ("graph", int(getattr(row, "edge_id")))
        reliability = getattr(row, "calibrated_edge_reliability_score", np.nan)
        if not np.isfinite(float(reliability)) and hasattr(row, "edge_proxy_score"):
            reliability = getattr(row, "edge_proxy_score")
        policy = getattr(row, "direct_edge_policy_support_score", np.nan)
        if not np.isfinite(float(policy)) and hasattr(row, "edge_policy_support_score"):
            policy = getattr(row, "edge_policy_support_score")
        compatibility = getattr(row, "outgoing_mean_termination_bridge_coverage", np.nan)
        if not np.isfinite(float(compatibility)) and hasattr(row, "compatibility_reliability"):
            compatibility = getattr(row, "compatibility_reliability")
        lookup[key] = {
            "certification_reliability": float(reliability) if np.isfinite(float(reliability)) else np.nan,
            "policy_reliability": float(policy) if np.isfinite(float(policy)) else np.nan,
            "compatibility_reliability": float(compatibility) if np.isfinite(float(compatibility)) else np.nan,
        }
    return lookup


def _edge_rows(
    edges: pd.DataFrame,
    *,
    source: str,
    is_bank_connector: bool,
    cert_lookup: dict[tuple[str, int], dict[str, float]],
) -> list[dict[str, float | int | str | bool]]:
    if edges.empty:
        return []
    enriched = _support_features(edges)
    rows: list[dict[str, float | int | str | bool]] = []
    for row in enriched.itertuples(index=False):
        segment_source, segment_edge_id = _edge_key_for_row(row, source)
        cert = cert_lookup.get((segment_source, segment_edge_id), {})
        planner_edge_id = int(getattr(row, "edge_id"))
        rows.append(
            {
                "segment_source": segment_source,
                "segment_edge_id": int(segment_edge_id),
                "planner_edge_id": planner_edge_id,
                "src": int(getattr(row, "src")),
                "dst": int(getattr(row, "dst")),
                "is_bank_connector": bool(is_bank_connector),
                "is_repair_edge": bool(getattr(row, "is_repair_edge", False)),
                "certification_reliability": cert.get("certification_reliability", np.nan),
                "support_reliability": float(getattr(row, "support_reliability")),
                "diversity_reliability": float(getattr(row, "diversity_reliability")),
                "horizon_reliability": float(getattr(row, "horizon_reliability")),
                "policy_reliability": cert.get("policy_reliability", np.nan),
                "compatibility_reliability": cert.get("compatibility_reliability", np.nan),
            }
        )
    return rows


def build_offline_edge_prior_scores(
    graph_edges: pd.DataFrame,
    *,
    bank_edges: pd.DataFrame | None = None,
    certification_csv: str | Path | None = None,
    penalty_weight: float = 1.0,
    certification_weight: float = 0.45,
    support_weight: float = 0.20,
    diversity_weight: float = 0.15,
    horizon_weight: float = 0.10,
    policy_weight: float = 0.05,
    compatibility_weight: float = 0.05,
) -> pd.DataFrame:
    certification = _read_csv(certification_csv)
    lookup = _cert_lookup(certification)
    rows = _edge_rows(graph_edges, source="graph", is_bank_connector=False, cert_lookup=lookup)
    if bank_edges is not None and not bank_edges.empty:
        rows.extend(_edge_rows(bank_edges, source="bank", is_bank_connector=True, cert_lookup=lookup))
    if not rows:
        return pd.DataFrame(columns=OFFLINE_PRIOR_COLUMNS)

    df = pd.DataFrame(rows).drop_duplicates(["segment_source", "segment_edge_id"], keep="first")
    component_weights = {
        "certification_reliability": float(certification_weight),
        "support_reliability": float(support_weight),
        "diversity_reliability": float(diversity_weight),
        "horizon_reliability": float(horizon_weight),
        "policy_reliability": float(policy_weight),
        "compatibility_reliability": float(compatibility_weight),
    }
    score = np.zeros(len(df), dtype=np.float64)
    weight_sum = np.zeros(len(df), dtype=np.float64)
    for col, weight in component_weights.items():
        vals = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        mask = np.isfinite(vals.to_numpy(dtype=np.float64))
        score[mask] += float(weight) * vals.to_numpy(dtype=np.float64)[mask]
        weight_sum[mask] += float(weight)
    fallback = (
        0.45 * pd.to_numeric(df["support_reliability"], errors="coerce").fillna(0.0)
        + 0.35 * pd.to_numeric(df["diversity_reliability"], errors="coerce").fillna(0.0)
        + 0.20 * pd.to_numeric(df["horizon_reliability"], errors="coerce").fillna(0.0)
    )
    reliability = np.where(weight_sum > 0, score / np.clip(weight_sum, 1e-6, None), fallback)
    df["offline_edge_prior_reliability"] = np.clip(reliability, 0.0, 1.0)
    df["offline_edge_prior_risk_score"] = 1.0 - df["offline_edge_prior_reliability"]
    df["offline_edge_prior_penalty"] = float(penalty_weight) * df["offline_edge_prior_risk_score"]
    return df[OFFLINE_PRIOR_COLUMNS]


def offline_edge_prior_penalty_map(scores: pd.DataFrame) -> dict[tuple[str, int], float]:
    penalties: dict[tuple[str, int], float] = {}
    if scores.empty:
        return penalties
    for row in scores.itertuples(index=False):
        key = (str(getattr(row, "segment_source")), int(getattr(row, "segment_edge_id")))
        value = float(getattr(row, "offline_edge_prior_penalty", 0.0))
        if np.isfinite(value) and value > 0.0:
            penalties[key] = max(penalties.get(key, 0.0), value)
    return penalties
