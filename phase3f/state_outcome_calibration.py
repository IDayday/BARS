from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from phase3f.state_conditioned_outcome_model import (
    StateConditionedOutcomeModel,
    extract_state_conditioned_attempt_examples_from_traces,
    fit_state_conditioned_outcome_model,
)


def _read_trace_file(path: Path) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    if not path.exists():
        return traces
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                traces.append(json.loads(text))
    return traces


def load_attempt_examples_with_sources(paths: list[str | Path] | str | Path) -> pd.DataFrame:
    """Load attempt examples and attach source/run groups for leakage-safe splits."""

    if isinstance(paths, (str, Path)):
        paths = [paths]
    frames: list[pd.DataFrame] = []
    for item in paths:
        p = Path(item)
        candidates = sorted(p.glob("episode_traces.jsonl")) if p.is_dir() else [p]
        for candidate in candidates:
            traces = _read_trace_file(candidate)
            for trace in traces:
                examples = extract_state_conditioned_attempt_examples_from_traces([trace])
                if examples.empty:
                    continue
                source_run = candidate.parent.name
                source_path = str(candidate)
                episode_id = int(trace.get("episode_id", -1))
                seed = int(trace.get("seed", -1))
                examples = examples.copy()
                examples["trace_source_run"] = source_run
                examples["trace_source_path"] = source_path
                examples["trace_episode_id"] = episode_id
                examples["trace_seed"] = seed
                examples["trace_group"] = f"{source_run}:seed{seed}:episode{episode_id}"
                frames.append(examples)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def split_by_trace_group(
    examples: pd.DataFrame,
    *,
    val_fraction: float = 0.3,
    seed: int = 0,
    group_col: str = "trace_group",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if examples.empty:
        return examples.copy(), examples.copy()
    if group_col not in examples.columns:
        rng = np.random.default_rng(int(seed))
        order = rng.permutation(len(examples))
        val_size = max(1, min(len(examples) - 1, int(round(float(val_fraction) * len(examples)))))
        val_idx = set(order[:val_size].tolist())
        mask = examples.index.to_series().isin(val_idx)
        return examples.loc[~mask].reset_index(drop=True), examples.loc[mask].reset_index(drop=True)

    groups = np.asarray(sorted(str(x) for x in examples[group_col].dropna().unique()))
    if groups.size <= 1:
        return split_by_trace_group(examples.drop(columns=[group_col]), val_fraction=val_fraction, seed=seed)
    rng = np.random.default_rng(int(seed))
    shuffled = groups[rng.permutation(groups.size)]
    val_group_count = max(1, min(groups.size - 1, int(round(float(val_fraction) * groups.size))))
    val_groups = set(shuffled[:val_group_count].tolist())
    mask = examples[group_col].astype(str).isin(val_groups)
    return examples.loc[~mask].reset_index(drop=True), examples.loc[mask].reset_index(drop=True)


def _auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    s = np.asarray(y_score, dtype=np.float64)
    pos = y >= 0.5
    neg = ~pos
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    pos_rank_sum = float(ranks[pos].sum())
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / max(1.0, n_pos * n_neg))


def evaluate_failure_predictions(
    examples: pd.DataFrame,
    predicted_failure_prob: np.ndarray,
    *,
    split: str,
) -> dict[str, float | int | str]:
    y = pd.to_numeric(examples["timeout"], errors="coerce").fillna(1.0).clip(0.0, 1.0).to_numpy(dtype=np.float64)
    p = np.clip(np.asarray(predicted_failure_prob, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    fail_mask = y >= 0.5
    complete_mask = ~fail_mask
    brier = float(np.mean((p - y) ** 2)) if len(y) else float("nan")
    log_loss = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))) if len(y) else float("nan")
    mean_failed = float(np.mean(p[fail_mask])) if fail_mask.any() else float("nan")
    mean_completed = float(np.mean(p[complete_mask])) if complete_mask.any() else float("nan")
    return {
        "split": split,
        "num_examples": int(len(y)),
        "num_failures": int(fail_mask.sum()),
        "num_completions": int(complete_mask.sum()),
        "failure_rate": float(np.mean(y)) if len(y) else float("nan"),
        "brier": brier,
        "log_loss": log_loss,
        "auc": _auc_score(y, p),
        "mean_pred_failed": mean_failed,
        "mean_pred_completed": mean_completed,
        "risk_separation": float(mean_failed - mean_completed)
        if np.isfinite(mean_failed) and np.isfinite(mean_completed)
        else float("nan"),
    }


def prediction_frame(model: StateConditionedOutcomeModel, examples: pd.DataFrame, *, split: str) -> pd.DataFrame:
    out = examples.copy()
    out["split"] = split
    out["predicted_failure_prob"] = model.predict_failure_proba(out)
    return out


def select_penalty_weight(
    validation_predictions: pd.DataFrame,
    *,
    penalty_weights: list[float],
    max_mean_penalty: float = 0.5,
    max_completed_mean_penalty: float = 0.5,
) -> pd.DataFrame:
    if validation_predictions.empty:
        return pd.DataFrame()
    y = pd.to_numeric(validation_predictions["timeout"], errors="coerce").fillna(1.0).clip(0.0, 1.0)
    p = pd.to_numeric(validation_predictions["predicted_failure_prob"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    failed = y >= 0.5
    completed = ~failed
    rows: list[dict[str, Any]] = []
    for weight in penalty_weights:
        penalty = float(weight) * p
        mean_all = float(penalty.mean())
        mean_failed = float(penalty[failed].mean()) if failed.any() else float("nan")
        mean_completed = float(penalty[completed].mean()) if completed.any() else float("nan")
        separation = (
            float(mean_failed - mean_completed)
            if np.isfinite(mean_failed) and np.isfinite(mean_completed)
            else float("nan")
        )
        valid = bool(
            mean_all <= float(max_mean_penalty)
            and (not np.isfinite(mean_completed) or mean_completed <= float(max_completed_mean_penalty))
        )
        score = separation if valid and np.isfinite(separation) else float("-inf")
        rows.append(
            {
                "penalty_weight": float(weight),
                "mean_penalty_all": mean_all,
                "mean_penalty_failed": mean_failed,
                "mean_penalty_completed": mean_completed,
                "penalty_separation": separation,
                "max_mean_penalty": float(max_mean_penalty),
                "max_completed_mean_penalty": float(max_completed_mean_penalty),
                "valid_under_budget": valid,
                "selection_score": score,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    valid_scores = out["selection_score"].replace([np.inf, -np.inf], np.nan)
    if valid_scores.notna().any():
        selected_idx = int(valid_scores.idxmax())
    else:
        selected_idx = int(out["mean_penalty_all"].idxmin())
    out["selected"] = False
    out.loc[selected_idx, "selected"] = True
    return out


def calibrate_state_outcome_model(
    examples: pd.DataFrame,
    *,
    val_fraction: float = 0.3,
    seed: int = 0,
    min_examples: int = 8,
    feature_columns: list[str] | None = None,
    l2: float = 1.0,
    learning_rate: float = 0.05,
    num_steps: int = 800,
    penalty_weights: list[float] | None = None,
    max_mean_penalty: float = 0.5,
    max_completed_mean_penalty: float = 0.5,
) -> dict[str, Any]:
    train, val = split_by_trace_group(examples, val_fraction=val_fraction, seed=seed)
    model = fit_state_conditioned_outcome_model(
        train,
        feature_columns=feature_columns,
        min_examples=min_examples,
        l2=l2,
        learning_rate=learning_rate,
        num_steps=num_steps,
    )
    train_pred = prediction_frame(model, train, split="train")
    val_pred = prediction_frame(model, val, split="val")
    metrics = pd.DataFrame(
        [
            evaluate_failure_predictions(train, train_pred["predicted_failure_prob"].to_numpy(), split="train"),
            evaluate_failure_predictions(val, val_pred["predicted_failure_prob"].to_numpy(), split="val"),
        ]
    )
    weights = penalty_weights or [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
    selection = select_penalty_weight(
        val_pred,
        penalty_weights=[float(w) for w in weights],
        max_mean_penalty=max_mean_penalty,
        max_completed_mean_penalty=max_completed_mean_penalty,
    )
    selected_weight = float(selection.loc[selection["selected"], "penalty_weight"].iloc[0]) if not selection.empty else 0.0
    return {
        "model": model,
        "train_examples": train,
        "val_examples": val,
        "predictions": pd.concat([train_pred, val_pred], ignore_index=True),
        "metrics": metrics,
        "selection": selection,
        "selected_weight": selected_weight,
    }
