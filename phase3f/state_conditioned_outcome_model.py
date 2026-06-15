from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_FEATURE_COLUMNS = [
    "edge_static_risk_penalty",
    "edge_state_risk_penalty",
    "edge_failure_count",
    "base_planning_cost",
    "selected_init_distance",
]

ATTEMPT_EXAMPLE_COLUMNS = [
    "episode_id",
    "segment_source",
    "segment_edge_id",
    "edge_src",
    "edge_dst",
    "attempt_steps",
    "completed",
    "timeout",
    "final_cluster",
    "final_subgoal_l2",
    "mean_selected_policy_action_mse",
    "edge_static_risk_penalty",
    "edge_state_risk_penalty",
    "edge_failure_count",
    "edge_failure_cost",
    "edge_risk_penalty",
    "edge_planning_cost",
    "base_planning_cost",
    "selected_init_distance",
]

STATE_OUTCOME_SCORE_COLUMNS = [
    "segment_source",
    "segment_edge_id",
    "planner_edge_id",
    "src",
    "dst",
    "edge_static_risk_penalty",
    "edge_state_risk_penalty",
    "edge_failure_count",
    "edge_failure_cost",
    "base_planning_cost",
    "edge_planning_cost_before_state_outcome",
    "selected_init_distance",
    "predicted_failure_prob",
    "state_conditioned_outcome_penalty",
    "state_conditioned_outcome_model_fitted",
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return out


def _attempt_key(step: dict[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(step.get("segment_source", "graph")),
        int(step.get("segment_edge_id", step.get("edge_id", -1))),
        int(step.get("edge_src", -1)),
        int(step.get("edge_dst", -1)),
    )


def _empty_attempt_examples() -> pd.DataFrame:
    return pd.DataFrame(columns=ATTEMPT_EXAMPLE_COLUMNS)


def extract_state_conditioned_attempt_examples_from_traces(traces: list[dict[str, Any]]) -> pd.DataFrame:
    """Extract attempt-level supervised labels with planner-side risk features."""

    rows: list[dict[str, Any]] = []
    for trace in traces:
        episode_id = int(trace.get("episode_id", -1))
        steps = trace.get("steps", []) or []
        current_key: tuple[str, int, int, int] | None = None
        current_steps: list[dict[str, Any]] = []

        def flush() -> None:
            if current_key is None or not current_steps:
                return
            segment_source, segment_edge_id, edge_src, edge_dst = current_key
            entered_dst = any(int(step.get("cluster", -999999)) == int(edge_dst) for step in current_steps)
            first_step = current_steps[0]
            final_step = current_steps[-1]
            policy_mse_values = [
                _safe_float(step.get("selected_policy_action_mse"), np.nan)
                for step in current_steps
                if np.isfinite(_safe_float(step.get("selected_policy_action_mse"), np.nan))
            ]
            planning_cost = _safe_float(first_step.get("edge_planning_cost"), 0.0)
            risk_penalty = _safe_float(first_step.get("edge_risk_penalty"), 0.0)
            failure_cost = _safe_float(first_step.get("edge_failure_cost"), 0.0)
            base_cost = max(0.0, planning_cost - risk_penalty - failure_cost)
            rows.append(
                {
                    "episode_id": episode_id,
                    "segment_source": segment_source,
                    "segment_edge_id": int(segment_edge_id),
                    "edge_src": int(edge_src),
                    "edge_dst": int(edge_dst),
                    "attempt_steps": int(len(current_steps)),
                    "completed": int(entered_dst),
                    "timeout": int(not entered_dst),
                    "final_cluster": int(final_step.get("cluster", -1)),
                    "final_subgoal_l2": _safe_float(final_step.get("subgoal_l2"), np.nan),
                    "mean_selected_policy_action_mse": (
                        float(np.mean(policy_mse_values)) if policy_mse_values else float("nan")
                    ),
                    "edge_static_risk_penalty": _safe_float(first_step.get("edge_static_risk_penalty"), 0.0),
                    "edge_state_risk_penalty": _safe_float(first_step.get("edge_state_risk_penalty"), 0.0),
                    "edge_failure_count": int(_safe_float(first_step.get("edge_failure_count"), 0.0)),
                    "edge_failure_cost": _safe_float(first_step.get("edge_failure_cost"), 0.0),
                    "edge_risk_penalty": _safe_float(first_step.get("edge_risk_penalty"), 0.0),
                    "edge_planning_cost": planning_cost,
                    "base_planning_cost": base_cost,
                    "selected_init_distance": _safe_float(first_step.get("selected_init_distance"), 0.0),
                }
            )

        for step in steps:
            key = _attempt_key(step)
            has_edge_step = "edge_step" in step
            prev_has_edge_step = bool(current_steps and "edge_step" in current_steps[-1])
            edge_step = int(step.get("edge_step", 0))
            prev_edge_step = int(current_steps[-1].get("edge_step", 0)) if current_steps else 0
            restarted_same_edge = bool(
                current_key == key
                and current_steps
                and has_edge_step
                and prev_has_edge_step
                and edge_step <= prev_edge_step
            )
            if current_key is not None and (key != current_key or restarted_same_edge):
                flush()
                current_steps = []
            current_key = key
            current_steps.append(step)
        flush()

    if not rows:
        return _empty_attempt_examples()
    return pd.DataFrame(rows)[ATTEMPT_EXAMPLE_COLUMNS]


def load_state_conditioned_attempt_examples(paths: list[str | Path] | str | Path | None) -> pd.DataFrame:
    if paths is None or paths == "":
        return _empty_attempt_examples()
    if isinstance(paths, (str, Path)):
        paths = [paths]
    traces: list[dict[str, Any]] = []
    for item in paths:
        p = Path(item)
        if p.is_dir():
            candidates = sorted(p.glob("episode_traces.jsonl"))
        else:
            candidates = [p]
        for candidate in candidates:
            if not candidate.exists():
                continue
            with candidate.open("r", encoding="utf-8") as f:
                for line in f:
                    text = line.strip()
                    if text:
                        traces.append(json.loads(text))
    return extract_state_conditioned_attempt_examples_from_traces(traces)


@dataclass
class StateConditionedOutcomeModel:
    feature_columns: list[str]
    feature_means: list[float]
    feature_scales: list[float]
    weights: list[float]
    bias: float
    fallback_failure_prob: float
    is_fitted: bool
    num_examples: int
    num_failures: int
    l2: float

    def predict_failure_proba(self, features: pd.DataFrame) -> np.ndarray:
        if features.empty:
            return np.empty(0, dtype=np.float64)
        if not self.is_fitted:
            return np.full(len(features), float(self.fallback_failure_prob), dtype=np.float64)
        x = _feature_matrix(
            features,
            self.feature_columns,
            np.asarray(self.feature_means, dtype=np.float64),
            np.asarray(self.feature_scales, dtype=np.float64),
        )
        logits = x @ np.asarray(self.weights, dtype=np.float64) + float(self.bias)
        return _sigmoid(logits)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def _feature_matrix(
    df: pd.DataFrame,
    columns: list[str],
    means: np.ndarray | None = None,
    scales: np.ndarray | None = None,
) -> np.ndarray:
    cols: list[np.ndarray] = []
    for col in columns:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        else:
            values = pd.Series(np.zeros(len(df), dtype=np.float64), index=df.index)
        cols.append(values.to_numpy(dtype=np.float64))
    x = np.column_stack(cols) if cols else np.zeros((len(df), 0), dtype=np.float64)
    if means is None:
        means = np.nanmean(x, axis=0) if x.size else np.zeros(x.shape[1], dtype=np.float64)
    if scales is None:
        scales = np.nanstd(x, axis=0) if x.size else np.ones(x.shape[1], dtype=np.float64)
    means = np.where(np.isfinite(means), means, 0.0)
    scales = np.where(np.isfinite(scales) & (scales > 1e-6), scales, 1.0)
    return (x - means.reshape(1, -1)) / scales.reshape(1, -1)


def fit_state_conditioned_outcome_model(
    examples: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    min_examples: int = 6,
    l2: float = 1.0,
    learning_rate: float = 0.1,
    num_steps: int = 1000,
) -> StateConditionedOutcomeModel:
    feature_columns = list(feature_columns or DEFAULT_FEATURE_COLUMNS)
    if examples.empty:
        return StateConditionedOutcomeModel(
            feature_columns=feature_columns,
            feature_means=[0.0 for _ in feature_columns],
            feature_scales=[1.0 for _ in feature_columns],
            weights=[0.0 for _ in feature_columns],
            bias=0.0,
            fallback_failure_prob=0.5,
            is_fitted=False,
            num_examples=0,
            num_failures=0,
            l2=float(l2),
        )

    y = pd.to_numeric(examples["timeout"], errors="coerce").fillna(1.0).clip(lower=0.0, upper=1.0).to_numpy(
        dtype=np.float64
    )
    n = int(len(y))
    num_failures = int(np.sum(y >= 0.5))
    fallback = float((num_failures + 1.0) / (n + 2.0))
    raw = _feature_matrix(examples, feature_columns)
    means = np.nanmean(
        np.column_stack(
            [
                pd.to_numeric(examples[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
                if col in examples.columns
                else np.zeros(n, dtype=np.float64)
                for col in feature_columns
            ]
        ),
        axis=0,
    )
    scales = np.nanstd(
        np.column_stack(
            [
                pd.to_numeric(examples[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
                if col in examples.columns
                else np.zeros(n, dtype=np.float64)
                for col in feature_columns
            ]
        ),
        axis=0,
    )
    means = np.where(np.isfinite(means), means, 0.0)
    scales = np.where(np.isfinite(scales) & (scales > 1e-6), scales, 1.0)
    x = _feature_matrix(examples, feature_columns, means, scales)
    bias = float(np.log(fallback / max(1e-6, 1.0 - fallback)))
    weights = np.zeros(x.shape[1], dtype=np.float64)
    fitted = bool(n >= int(min_examples) and 0 < num_failures < n and x.shape[1] > 0)
    if fitted:
        for _ in range(int(num_steps)):
            pred = _sigmoid(x @ weights + bias)
            err = pred - y
            grad_w = (x.T @ err) / max(1, n) + float(l2) * weights / max(1, n)
            grad_b = float(np.mean(err))
            weights -= float(learning_rate) * grad_w
            bias -= float(learning_rate) * grad_b

    return StateConditionedOutcomeModel(
        feature_columns=feature_columns,
        feature_means=[float(v) for v in means],
        feature_scales=[float(v) for v in scales],
        weights=[float(v) for v in weights],
        bias=float(bias),
        fallback_failure_prob=fallback,
        is_fitted=fitted,
        num_examples=n,
        num_failures=num_failures,
        l2=float(l2),
    )


def _row_cost(row: Any) -> float:
    for name in ("cost", "median_h", "mean_h", "max_h"):
        value = getattr(row, name, None)
        if value is not None and np.isfinite(float(value)):
            return float(value)
    return 1.0


def _edge_identity(row: Any, source: str) -> tuple[str, int, int]:
    edge_id = int(getattr(row, "edge_id"))
    bank_edge_id = getattr(row, "bank_edge_id", np.nan)
    if source == "graph" and bool(getattr(row, "is_repair_edge", False)) and pd.notna(bank_edge_id):
        return "bank", int(bank_edge_id), edge_id
    return source, edge_id, edge_id


def _edge_lookup(edges: pd.DataFrame | None, source: str) -> dict[tuple[str, int], Any]:
    if edges is None or edges.empty:
        return {}
    out: dict[tuple[str, int], Any] = {}
    for row in edges.itertuples(index=False):
        segment_source, segment_edge_id, _ = _edge_identity(row, source)
        out[(segment_source, int(segment_edge_id))] = row
    return out


def build_state_conditioned_outcome_candidate_features(
    state_scores: pd.DataFrame,
    *,
    graph_edges: pd.DataFrame,
    bank_edges: pd.DataFrame | None,
    edge_static_risk_penalties: dict[tuple[str, int], float] | None = None,
    failed_edge_counts: dict[tuple[str, int], int] | None = None,
    failure_penalty: float = 0.0,
) -> pd.DataFrame:
    if state_scores.empty:
        return pd.DataFrame(columns=STATE_OUTCOME_SCORE_COLUMNS)
    edge_static_risk_penalties = edge_static_risk_penalties or {}
    failed_edge_counts = failed_edge_counts or {}
    lookup = _edge_lookup(graph_edges, "graph")
    lookup.update(_edge_lookup(bank_edges, "bank"))
    rows: list[dict[str, Any]] = []
    for row in state_scores.itertuples(index=False):
        key = (str(getattr(row, "segment_source")), int(getattr(row, "segment_edge_id")))
        edge_row = lookup.get(key)
        base_cost = _row_cost(edge_row) if edge_row is not None else 1.0
        static_penalty = float(edge_static_risk_penalties.get(key, 0.0))
        state_penalty = _safe_float(getattr(row, "state_conditioned_risk_penalty", 0.0), 0.0)
        failure_count = int(failed_edge_counts.get(key, 0))
        failure_cost = float(failure_penalty) * failure_count
        rows.append(
            {
                "segment_source": key[0],
                "segment_edge_id": int(key[1]),
                "planner_edge_id": int(getattr(row, "planner_edge_id", key[1])),
                "src": int(getattr(row, "src")),
                "dst": int(getattr(row, "dst")),
                "edge_static_risk_penalty": static_penalty,
                "edge_state_risk_penalty": state_penalty,
                "edge_failure_count": failure_count,
                "edge_failure_cost": failure_cost,
                "base_planning_cost": float(base_cost),
                "edge_planning_cost_before_state_outcome": float(base_cost + static_penalty + state_penalty + failure_cost),
                "selected_init_distance": _safe_float(getattr(row, "min_initiation_distance", 0.0), 0.0),
            }
        )
    return pd.DataFrame(rows)


def score_state_conditioned_outcome_candidates(
    model: StateConditionedOutcomeModel,
    candidate_features: pd.DataFrame,
    *,
    penalty_weight: float = 1.0,
) -> pd.DataFrame:
    if candidate_features.empty:
        return pd.DataFrame(columns=STATE_OUTCOME_SCORE_COLUMNS)
    out = candidate_features.copy()
    probs = model.predict_failure_proba(out)
    out["predicted_failure_prob"] = np.clip(probs, 0.0, 1.0)
    out["state_conditioned_outcome_penalty"] = float(penalty_weight) * out["predicted_failure_prob"]
    out["state_conditioned_outcome_model_fitted"] = bool(model.is_fitted)
    return out[STATE_OUTCOME_SCORE_COLUMNS]


def state_conditioned_outcome_penalty_map(scores: pd.DataFrame) -> dict[tuple[str, int], float]:
    penalties: dict[tuple[str, int], float] = {}
    if scores.empty:
        return penalties
    for row in scores.itertuples(index=False):
        key = (str(getattr(row, "segment_source")), int(getattr(row, "segment_edge_id")))
        value = _safe_float(getattr(row, "state_conditioned_outcome_penalty", 0.0), 0.0)
        if value > 0.0:
            penalties[key] = max(penalties.get(key, 0.0), value)
    return penalties
