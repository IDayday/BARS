from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


EPS = 1e-9


@dataclass(frozen=True)
class EdgeRiskCalibrationConfig:
    support_weight: float = 0.30
    policy_weight: float = 0.20
    behavior_weight: float = 0.20
    compatibility_weight: float = 0.20
    diversity_weight: float = 0.10
    min_component_score: float = 0.02
    certification_threshold: float = 0.25
    min_support_lcb: float = 0.01
    min_compatibility: float = 0.0


def _clip01(series: Any, default: float = 0.0) -> pd.Series:
    out = pd.to_numeric(pd.Series(series), errors="coerce").fillna(float(default)).astype(float)
    return out.clip(0.0, 1.0)


def _rank01(series: Any) -> pd.Series:
    values = pd.to_numeric(pd.Series(series), errors="coerce")
    if values.notna().sum() <= 1:
        return pd.Series(np.ones(values.shape[0], dtype=np.float64), index=values.index)
    return values.rank(method="average", pct=True).fillna(0.0).clip(0.0, 1.0).astype(float)


def _weighted_geomean(components: pd.DataFrame, weights: dict[str, float], floor: float) -> pd.Series:
    total = sum(max(0.0, float(w)) for w in weights.values())
    if total <= 0:
        raise ValueError("At least one calibration component weight must be positive")
    log_score = np.zeros(components.shape[0], dtype=np.float64)
    for col, weight in weights.items():
        w = max(0.0, float(weight)) / total
        if w <= 0:
            continue
        score = _clip01(components[col], default=0.0).clip(lower=max(EPS, float(floor)))
        log_score += w * np.log(score.to_numpy(dtype=np.float64))
    return pd.Series(np.exp(log_score), index=components.index).clip(0.0, 1.0)


def compute_component_scores(
    edge_certification: pd.DataFrame,
    config: EdgeRiskCalibrationConfig | None = None,
) -> pd.DataFrame:
    """Compute comparable reliability components from Phase 3E edge evidence."""

    config = config or EdgeRiskCalibrationConfig()
    df = edge_certification.copy()
    out = pd.DataFrame(index=df.index)
    out["edge_id"] = df["edge_id"].astype(int)

    out["support_reliability"] = _clip01(df.get("heldout_support_lcb", 0.0))
    out["policy_reliability"] = _clip01(df.get("edge_policy_support_score", 0.0))
    out["behavior_reliability"] = (1.0 - _clip01(df.get("edge_ood_score", 1.0), default=1.0)).clip(0.0, 1.0)

    bridge = (
        _clip01(df.get("outgoing_mean_termination_bridge_coverage", 0.0))
        + _clip01(df.get("incoming_mean_termination_bridge_coverage", 0.0))
    ) / 2.0
    compatible = 1.0 - (
        _clip01(df.get("outgoing_incompatible_fraction", 1.0), default=1.0)
        + _clip01(df.get("incoming_incompatible_fraction", 1.0), default=1.0)
    ) / 2.0
    out["compatibility_reliability"] = np.sqrt(
        bridge.clip(lower=EPS).to_numpy(dtype=np.float64)
        * compatible.clip(lower=EPS).to_numpy(dtype=np.float64)
    )
    out["compatibility_reliability"] = out["compatibility_reliability"].clip(0.0, 1.0)

    starts = _rank01(np.log1p(pd.to_numeric(df.get("num_unique_starts", 0), errors="coerce").fillna(0.0)))
    episodes = _rank01(np.log1p(pd.to_numeric(df.get("num_unique_episodes", 0), errors="coerce").fillna(0.0)))
    out["diversity_reliability"] = ((starts + episodes) / 2.0).clip(0.0, 1.0)
    return out


def calibrate_edge_risk(
    edge_certification: pd.DataFrame,
    config: EdgeRiskCalibrationConfig | None = None,
) -> pd.DataFrame:
    """Return certification table with conservative calibrated reliability columns."""

    config = config or EdgeRiskCalibrationConfig()
    out = edge_certification.copy()
    if "edge_proxy_score_original" not in out.columns and "edge_proxy_score" in out.columns:
        out["edge_proxy_score_original"] = out["edge_proxy_score"]
    if "certified_offline_binary_original" not in out.columns and "certified_offline_binary" in out.columns:
        out["certified_offline_binary_original"] = out["certified_offline_binary"]

    components = compute_component_scores(out, config)
    for col in components.columns:
        if col != "edge_id":
            out[col] = components[col].to_numpy()

    weights = {
        "support_reliability": config.support_weight,
        "policy_reliability": config.policy_weight,
        "behavior_reliability": config.behavior_weight,
        "compatibility_reliability": config.compatibility_weight,
        "diversity_reliability": config.diversity_weight,
    }
    out["calibrated_edge_reliability_score"] = _weighted_geomean(
        out[list(weights)],
        weights,
        floor=float(config.min_component_score),
    )
    out["calibrated_edge_risk_score"] = 1.0 - out["calibrated_edge_reliability_score"]
    out["calibrated_edge_ood_score"] = 1.0 - out["behavior_reliability"]
    out["calibrated_certified_binary"] = (
        (out["calibrated_edge_reliability_score"] >= float(config.certification_threshold))
        & (_clip01(out.get("heldout_support_lcb", 0.0)) >= float(config.min_support_lcb))
        & (out["compatibility_reliability"] >= float(config.min_compatibility))
    )

    # Planner-facing compatibility: reuse existing column names while preserving originals.
    out["edge_proxy_score_calibrated_input"] = out["calibrated_edge_reliability_score"]
    out["edge_ood_score_calibrated_input"] = out["calibrated_edge_ood_score"]
    return out


def make_planner_certification(calibrated: pd.DataFrame) -> pd.DataFrame:
    """Return a Phase 4 planner-compatible table using calibrated reliability."""

    out = calibrated.copy()
    out["edge_proxy_score"] = out["calibrated_edge_reliability_score"]
    out["edge_ood_score"] = out["calibrated_edge_ood_score"]
    out["certified_offline_binary"] = out["calibrated_certified_binary"]
    return out


def score_diagnostics(edge_scores: pd.DataFrame) -> dict[str, Any]:
    def _corr(x: str, y: str, method: str) -> float | None:
        if x not in edge_scores.columns or y not in edge_scores.columns:
            return None
        corr = pd.to_numeric(edge_scores[x], errors="coerce").corr(
            pd.to_numeric(edge_scores[y], errors="coerce"),
            method=method,
        )
        return None if pd.isna(corr) else float(corr)

    label = "heldout_support_binary"
    out: dict[str, Any] = {
        "num_edges": int(edge_scores.shape[0]),
        "note": "Pseudo-label diagnostics only; calibrated reliability is not rollout success probability.",
    }
    for score_col in ["edge_proxy_score_original", "calibrated_edge_reliability_score"]:
        if score_col not in edge_scores.columns:
            continue
        out[f"{score_col}_mean"] = float(pd.to_numeric(edge_scores[score_col], errors="coerce").mean())
        out[f"{score_col}_median"] = float(pd.to_numeric(edge_scores[score_col], errors="coerce").median())
        out[f"{score_col}_spearman_heldout_support_rate"] = _corr(
            score_col, "heldout_support_rate", "spearman"
        )
        out[f"{score_col}_pearson_heldout_support_rate"] = _corr(
            score_col, "heldout_support_rate", "pearson"
        )
        if label in edge_scores.columns:
            labels = _clip01(edge_scores[label])
            scores = _clip01(edge_scores[score_col])
            out[f"{score_col}_brier_heldout_support_binary"] = float(np.mean((scores - labels) ** 2))
    if "calibrated_certified_binary" in edge_scores.columns:
        out["calibrated_certified_edges"] = int(edge_scores["calibrated_certified_binary"].sum())
        out["calibrated_certified_rate"] = float(edge_scores["calibrated_certified_binary"].mean())
    if "certified_offline_binary_original" in edge_scores.columns:
        original = edge_scores["certified_offline_binary_original"].astype(bool)
        out["original_certified_edges"] = int(original.sum())
        out["original_certified_rate"] = float(original.mean())
    return out


def calibration_bins(
    edge_scores: pd.DataFrame,
    score_col: str = "calibrated_edge_reliability_score",
    label_col: str = "heldout_support_binary",
    bins: int = 10,
) -> pd.DataFrame:
    if edge_scores.empty:
        return pd.DataFrame()
    scores = _clip01(edge_scores.get(score_col, 0.0))
    labels = _clip01(edge_scores.get(label_col, 0.0))
    bucket = np.minimum((scores * int(bins)).astype(int), int(bins) - 1)
    tmp = pd.DataFrame({"score": scores, "label": labels, "bin": bucket})
    out = (
        tmp.groupby("bin", sort=True)
        .agg(num_edges=("score", "size"), mean_score=("score", "mean"), mean_label=("label", "mean"))
        .reset_index()
    )
    out["abs_gap"] = (out["mean_score"] - out["mean_label"]).abs()
    out["weighted_abs_gap"] = out["abs_gap"] * out["num_edges"] / max(1, int(tmp.shape[0]))
    return out
