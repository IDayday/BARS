from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MEMORY_COLUMNS = [
    "segment_source",
    "segment_edge_id",
    "edge_src",
    "edge_dst",
    "attempts",
    "completed",
    "timeouts",
    "success_rate",
    "failure_excess",
    "mean_attempt_steps",
    "mean_final_subgoal_l2",
    "mean_selected_policy_action_mse",
]

ATTEMPT_COLUMNS = [
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
]


def _attempt_key(step: dict[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(step.get("segment_source", "graph")),
        int(step.get("segment_edge_id", step.get("edge_id", -1))),
        int(step.get("edge_src", -1)),
        int(step.get("edge_dst", -1)),
    )


def extract_edge_attempts_from_traces(traces: list[dict[str, Any]]) -> pd.DataFrame:
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
            final_step = current_steps[-1]
            policy_mse_values = [
                float(step["selected_policy_action_mse"])
                for step in current_steps
                if "selected_policy_action_mse" in step and np.isfinite(float(step["selected_policy_action_mse"]))
            ]
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
                    "final_subgoal_l2": float(final_step.get("subgoal_l2", np.nan)),
                    "mean_selected_policy_action_mse": float(np.mean(policy_mse_values)) if policy_mse_values else np.nan,
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
        return pd.DataFrame(columns=ATTEMPT_COLUMNS)
    return pd.DataFrame(rows)[ATTEMPT_COLUMNS]


def summarize_edge_attempts(attempts: pd.DataFrame) -> pd.DataFrame:
    if attempts.empty:
        return pd.DataFrame(columns=MEMORY_COLUMNS)
    group_cols = ["segment_source", "segment_edge_id", "edge_src", "edge_dst"]
    grouped = attempts.groupby(group_cols, dropna=False)
    out = grouped.agg(
        attempts=("completed", "size"),
        completed=("completed", "sum"),
        timeouts=("timeout", "sum"),
        mean_attempt_steps=("attempt_steps", "mean"),
        mean_final_subgoal_l2=("final_subgoal_l2", "mean"),
        mean_selected_policy_action_mse=("mean_selected_policy_action_mse", "mean"),
    ).reset_index()
    out["success_rate"] = out["completed"] / out["attempts"].clip(lower=1)
    out["failure_excess"] = (out["timeouts"] - out["completed"]).clip(lower=0)
    return out[MEMORY_COLUMNS]


def load_edge_memory(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=MEMORY_COLUMNS)
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=MEMORY_COLUMNS)
    memory = pd.read_csv(p)
    for col in MEMORY_COLUMNS:
        if col not in memory.columns:
            memory[col] = 0 if col not in {"segment_source"} else ""
    return memory[MEMORY_COLUMNS]


def merge_edge_memory(existing: pd.DataFrame, new_summary: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        merged = new_summary.copy()
    elif new_summary.empty:
        merged = existing.copy()
    else:
        merged = pd.concat([existing, new_summary], ignore_index=True)
    if merged.empty:
        return pd.DataFrame(columns=MEMORY_COLUMNS)

    group_cols = ["segment_source", "segment_edge_id", "edge_src", "edge_dst"]
    weighted_cols = ["mean_attempt_steps", "mean_final_subgoal_l2", "mean_selected_policy_action_mse"]
    for col in weighted_cols:
        merged[f"weighted_{col}"] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0) * pd.to_numeric(
            merged["attempts"], errors="coerce"
        ).fillna(0.0)
    out = merged.groupby(group_cols, dropna=False).agg(
        attempts=("attempts", "sum"),
        completed=("completed", "sum"),
        timeouts=("timeouts", "sum"),
        weighted_mean_attempt_steps=("weighted_mean_attempt_steps", "sum"),
        weighted_mean_final_subgoal_l2=("weighted_mean_final_subgoal_l2", "sum"),
        weighted_mean_selected_policy_action_mse=("weighted_mean_selected_policy_action_mse", "sum"),
    ).reset_index()
    denom = out["attempts"].clip(lower=1)
    out["success_rate"] = out["completed"] / denom
    out["failure_excess"] = (out["timeouts"] - out["completed"]).clip(lower=0)
    out["mean_attempt_steps"] = out["weighted_mean_attempt_steps"] / denom
    out["mean_final_subgoal_l2"] = out["weighted_mean_final_subgoal_l2"] / denom
    out["mean_selected_policy_action_mse"] = out["weighted_mean_selected_policy_action_mse"] / denom
    return out[MEMORY_COLUMNS]


def memory_failed_edge_counts(memory: pd.DataFrame, mode: str = "failure_excess") -> dict[tuple[str, int], int]:
    if memory.empty:
        return {}
    if mode not in {"failure_excess", "timeouts", "attempts"}:
        raise ValueError("mode must be failure_excess, timeouts, or attempts")
    counts: dict[tuple[str, int], int] = {}
    for row in memory.itertuples(index=False):
        source = str(getattr(row, "segment_source"))
        edge_id = int(getattr(row, "segment_edge_id"))
        value = int(max(0, round(float(getattr(row, mode)))))
        if value > 0:
            counts[(source, edge_id)] = counts.get((source, edge_id), 0) + value
    return counts


def write_edge_memory_outputs(
    output_dir: str | Path,
    traces: list[dict[str, Any]],
    *,
    edge_memory_csv: str | Path | None = None,
    update_memory: bool = False,
    memory_penalty_mode: str = "failure_excess",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    attempts = extract_edge_attempts_from_traces(traces)
    attempts.to_csv(out / "edge_attempts.csv", index=False)
    summary = summarize_edge_attempts(attempts)
    summary.to_csv(out / "edge_attempt_summary.csv", index=False)
    if update_memory and edge_memory_csv is not None:
        memory_path = Path(edge_memory_csv)
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        existing = load_edge_memory(memory_path)
        merged = merge_edge_memory(existing, summary)
        merged.to_csv(memory_path, index=False)
        failed_counts = memory_failed_edge_counts(merged, mode=memory_penalty_mode)
        pd.DataFrame(
            [
                {
                    "edge_memory_csv": str(memory_path),
                    "num_memory_edges": int(merged.shape[0]),
                    "num_penalized_edges": int(len(failed_counts)),
                    "memory_penalty_mode": memory_penalty_mode,
                }
            ]
        ).to_csv(out / "edge_memory_update_summary.csv", index=False)
    return attempts, summary
