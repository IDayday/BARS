from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from phase3f.edge_memory import MEMORY_COLUMNS, load_edge_memory


OUTCOME_SCORE_COLUMNS = [
    *MEMORY_COLUMNS,
    "posterior_success_prob",
    "posterior_failure_prob",
    "outcome_uncertainty",
    "policy_mse_risk",
    "subgoal_l2_risk",
    "edge_outcome_risk_score",
    "edge_outcome_penalty",
]


def _safe_numeric(series: pd.Series, fill: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(fill)


def _scaled_risk(values: pd.Series, scale: float | str | None, default_scale: float) -> pd.Series:
    numeric = _safe_numeric(values, fill=np.nan)
    finite = numeric[np.isfinite(numeric)]
    if finite.empty:
        return pd.Series(np.zeros(len(values), dtype=np.float64), index=values.index)
    if scale is None or scale == "auto":
        denom = float(np.nanpercentile(finite.to_numpy(dtype=np.float64), 75))
        denom = max(denom, default_scale, 1e-6)
    else:
        denom = max(float(scale), 1e-6)
    return (numeric.fillna(float(np.nanmedian(finite))) / denom).clip(lower=0.0, upper=1.0)


def fit_edge_outcome_scores(
    memory: pd.DataFrame | str | Path | None,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    min_attempts: int = 1,
    penalty_weight: float = 1.0,
    uncertainty_weight: float = 0.25,
    policy_mse_weight: float = 0.0,
    policy_mse_scale: float = 0.05,
    subgoal_l2_weight: float = 0.0,
    subgoal_l2_scale: float | str | None = "auto",
) -> pd.DataFrame:
    """Fit a conservative edge outcome score from persisted online attempts.

    The score is intentionally simple and data-limited: a Beta-smoothed
    empirical failure probability plus optional diagnostics from the traces.
    It is not an execution-success probability model.
    """

    if isinstance(memory, (str, Path)) or memory is None:
        memory_df = load_edge_memory(memory)
    else:
        memory_df = memory.copy()
    if memory_df.empty:
        return pd.DataFrame(columns=OUTCOME_SCORE_COLUMNS)

    for col in MEMORY_COLUMNS:
        if col not in memory_df.columns:
            memory_df[col] = 0 if col != "segment_source" else ""
    out = memory_df[MEMORY_COLUMNS].copy()
    attempts = _safe_numeric(out["attempts"], fill=0.0).clip(lower=0.0)
    completed = _safe_numeric(out["completed"], fill=0.0).clip(lower=0.0)
    completed = np.minimum(completed, attempts)
    denom = attempts + float(alpha) + float(beta)
    posterior_success = (completed + float(alpha)) / denom.clip(lower=1e-6)
    posterior_failure = 1.0 - posterior_success
    uncertainty = np.sqrt((posterior_success * posterior_failure) / denom.clip(lower=1.0))
    policy_mse_risk = _scaled_risk(out["mean_selected_policy_action_mse"], policy_mse_scale, default_scale=0.05)
    subgoal_l2_risk = _scaled_risk(out["mean_final_subgoal_l2"], subgoal_l2_scale, default_scale=1.0)
    attempted_mask = attempts >= int(min_attempts)

    risk_score = (
        posterior_failure
        + float(uncertainty_weight) * uncertainty
        + float(policy_mse_weight) * policy_mse_risk
        + float(subgoal_l2_weight) * subgoal_l2_risk
    )
    risk_score = pd.Series(risk_score, index=out.index).where(attempted_mask, 0.0).clip(lower=0.0)
    out["posterior_success_prob"] = posterior_success.astype(float)
    out["posterior_failure_prob"] = posterior_failure.astype(float)
    out["outcome_uncertainty"] = uncertainty.astype(float)
    out["policy_mse_risk"] = policy_mse_risk.astype(float)
    out["subgoal_l2_risk"] = subgoal_l2_risk.astype(float)
    out["edge_outcome_risk_score"] = risk_score.astype(float)
    out["edge_outcome_penalty"] = (float(penalty_weight) * risk_score).astype(float)
    return out[OUTCOME_SCORE_COLUMNS]


def edge_outcome_penalty_map(scores: pd.DataFrame) -> dict[tuple[str, int], float]:
    if scores.empty:
        return {}
    penalties: dict[tuple[str, int], float] = {}
    for row in scores.itertuples(index=False):
        source = str(getattr(row, "segment_source"))
        edge_id = int(getattr(row, "segment_edge_id"))
        penalty = float(getattr(row, "edge_outcome_penalty", 0.0))
        if np.isfinite(penalty) and penalty > 0.0:
            penalties[(source, edge_id)] = max(penalties.get((source, edge_id), 0.0), penalty)
    return penalties
