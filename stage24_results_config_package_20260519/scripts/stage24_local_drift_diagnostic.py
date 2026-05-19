#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bars.gas_bars.failure_atlas import enrich_failure_atlas


OUT_COLUMNS = [
    "env",
    "seed",
    "variant",
    "fallback_mode",
    "task_id",
    "episode_id",
    "success",
    "primary_failure_type",
    "first_failure_step",
    "first_failed_subgoal_idx",
    "subgoal_reach_rate",
    "edge_fail_position",
    "initial_goal_dist_phi",
    "best_goal_dist_phi",
    "final_goal_dist_phi",
    "mean_plan_edges",
    "last_plan_edges",
    "replans",
    "local_drift_score",
    "progress_stall_count",
    "oscillation_score",
]


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _read_evals(roots: list[Path]) -> pd.DataFrame:
    frames = []
    for root in roots:
        if root.is_file():
            df = _read_csv(root)
            if len(df):
                df["eval_path"] = str(root)
                frames.append(df)
            continue
        if not root.exists():
            continue
        for path in root.rglob("eval.csv"):
            df = _read_csv(path)
            if len(df) == 0 or not {"env", "seed", "success"}.issubset(df.columns):
                continue
            df["eval_path"] = str(path)
            if "fallback_mode" not in df:
                for part in path.parts:
                    if part.startswith("fallback_"):
                        df["fallback_mode"] = part.replace("fallback_", "")
                        break
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _num(row: pd.Series, key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value == "" or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _trace_metrics(path: Any, window: int, stall_frac: float, drift_threshold: float) -> dict[str, float | int | str]:
    if path is None or (isinstance(path, float) and pd.isna(path)):
        return {}
    trace_path = Path(str(path))
    if not trace_path.exists():
        return {}
    rows = []
    try:
        for line in trace_path.read_text().splitlines():
            if not line.strip():
                continue
            import json

            rows.append(json.loads(line))
    except Exception:
        return {}
    target = np.asarray([float(r.get("target_dist_phi", np.nan)) for r in rows], dtype=np.float64)
    target = target[np.isfinite(target)]
    if target.size < 2:
        return {}
    w = int(max(2, min(window, target.size)))
    drift_scores = []
    stall_count = 0
    first_failure_step = ""
    for start in range(0, target.size - w + 1):
        seg = target[start : start + w]
        denom = max(float(seg[0]), 1e-6)
        drift = float((seg[-1] - seg[0]) / denom)
        progress = float((seg[0] - np.min(seg)) / denom)
        drift_scores.append(drift)
        is_stall = progress < stall_frac
        if is_stall:
            stall_count += 1
        if first_failure_step == "" and (drift > drift_threshold or is_stall):
            first_failure_step = int(start + w - 1)
    deltas = np.diff(target)
    path_motion = float(np.sum(np.abs(deltas)))
    net_improvement = float(max(target[0] - target[-1], 0.0))
    oscillation = float(max(path_motion - net_improvement, 0.0) / max(path_motion, 1e-6))
    return {
        "local_drift_score": float(max(0.0, max(drift_scores) if drift_scores else 0.0)),
        "progress_stall_count": int(stall_count),
        "oscillation_score": oscillation,
        "first_failure_step": first_failure_step,
    }


def _merge_atlas(eval_df: pd.DataFrame, atlas: pd.DataFrame) -> pd.DataFrame:
    if len(eval_df) == 0:
        return atlas.copy()
    df = eval_df.copy()
    if "primary_failure_type" in df:
        return df
    if len(atlas) and "primary_failure_type" in atlas:
        keys = [c for c in ["env", "seed", "variant", "fallback_mode", "task_id", "episode_id"] if c in df.columns and c in atlas.columns]
        if keys:
            cols = keys + [c for c in ["primary_failure_type", "failure_labels"] if c in atlas.columns]
            return df.merge(atlas[cols].drop_duplicates(keys), on=keys, how="left")
    try:
        return enrich_failure_atlas(df)
    except Exception:
        df["primary_failure_type"] = np.where(pd.to_numeric(df.get("success", 0), errors="coerce").fillna(0) >= 0.5, "SUCCESS", "UNCLASSIFIED_FAILURE")
        return df


def build_drift_table(
    eval_df: pd.DataFrame,
    atlas: pd.DataFrame,
    *,
    window: int = 16,
    stall_frac: float = 0.03,
    drift_threshold: float = 0.10,
) -> pd.DataFrame:
    df = _merge_atlas(eval_df, atlas)
    if len(df) == 0:
        return pd.DataFrame(columns=OUT_COLUMNS)
    rows = []
    for _, row in df.iterrows():
        success = int(_num(row, "success", 0.0) >= 0.5)
        trace = _trace_metrics(row.get("debug_trace_path", ""), window, stall_frac, drift_threshold)
        local_drift = max(_num(row, "local_drift_score", 0.0), float(trace.get("local_drift_score", 0.0) or 0.0))
        stall_count = max(int(_num(row, "progress_stall_count", 0.0)), int(trace.get("progress_stall_count", 0) or 0))
        oscillation = max(_num(row, "oscillation_score", 0.0), float(trace.get("oscillation_score", 0.0) or 0.0))
        first_failure_step = trace.get("first_failure_step", "")
        if first_failure_step == "" and not success:
            first_failure_step = int(_num(row, "steps", 0.0))
        first_failed_subgoal_idx = ""
        if not success:
            first_failed_subgoal_idx = int(_num(row, "subgoals_reached", 0.0)) + 1
        rows.append(
            {
                "env": row.get("env", ""),
                "seed": row.get("seed", ""),
                "variant": row.get("variant", ""),
                "fallback_mode": row.get("fallback_mode", ""),
                "task_id": row.get("task_id", ""),
                "episode_id": row.get("episode_id", ""),
                "success": success,
                "primary_failure_type": row.get("primary_failure_type", "SUCCESS" if success else "UNCLASSIFIED_FAILURE"),
                "first_failure_step": first_failure_step,
                "first_failed_subgoal_idx": first_failed_subgoal_idx,
                "subgoal_reach_rate": _num(row, "subgoal_reach_rate", 0.0),
                "edge_fail_position": row.get("edge_fail_position", ""),
                "initial_goal_dist_phi": _num(row, "initial_goal_dist_phi", np.nan),
                "best_goal_dist_phi": _num(row, "best_goal_dist_phi", np.nan),
                "final_goal_dist_phi": _num(row, "final_goal_dist_phi", np.nan),
                "mean_plan_edges": _num(row, "mean_plan_edges", 0.0),
                "last_plan_edges": _num(row, "last_plan_edges", 0.0),
                "replans": _num(row, "replans", 0.0),
                "local_drift_score": local_drift,
                "progress_stall_count": stall_count,
                "oscillation_score": oscillation,
            }
        )
    return pd.DataFrame(rows, columns=OUT_COLUMNS)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-roots", default="runs_stage24_reachability_confirm,runs_stage24_local_drift,runs_stage23_repro")
    p.add_argument("--failure-atlas", default="reports/stage23_failure_atlas.csv")
    p.add_argument("--out", default="reports/stage24_local_drift.csv")
    p.add_argument("--window", type=int, default=16)
    p.add_argument("--stall-frac", type=float, default=0.03)
    p.add_argument("--drift-threshold", type=float, default=0.10)
    args = p.parse_args()

    eval_df = _read_evals([Path(x) for x in args.eval_roots.split(",") if x])
    atlas = _read_csv(Path(args.failure_atlas))
    out = build_drift_table(eval_df, atlas, window=args.window, stall_frac=args.stall_frac, drift_threshold=args.drift_threshold)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


if __name__ == "__main__":
    main()
