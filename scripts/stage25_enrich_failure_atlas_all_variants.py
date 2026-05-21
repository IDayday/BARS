#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


REQUIRED_COLUMNS = [
    "env",
    "seed",
    "variant",
    "fallback_mode",
    "budget",
    "task_id",
    "episode_id",
    "eval_path",
    "success",
    "steps",
    "primary_failure_type",
    "failure_labels",
    "first_failure_step",
    "first_failed_subgoal_idx",
    "subgoal_reach_rate",
    "subgoals_reached",
    "subgoals_total",
    "first_plan_edges",
    "last_plan_edges",
    "mean_plan_edges",
    "replans",
    "no_path_count",
    "budget_reject_count",
    "initial_goal_dist_phi",
    "best_goal_dist_phi",
    "final_goal_dist_phi",
    "local_drift_score",
    "progress_stall_count",
    "oscillation_score",
    "drift_event_count",
    "replan_reason_counts",
    "refresh_count",
    "source_labeler",
    "label_confidence",
]

FAILURE_TYPES = [
    "SUCCESS",
    "F0_NO_PATH_OR_INITIAL_PLAN_FAILURE",
    "F1_SUBGOAL_NOT_REACHED",
    "F2_BUDGET_REJECTED",
    "F3_BOUNDARY_OR_BRIDGE_MISMATCH",
    "F4_LOCAL_EXECUTION_DRIFT",
    "F5_OSCILLATION_OR_REPLAN_LOOP",
    "F6_LONG_PATH_ACCUMULATION",
    "F7_TIMEOUT_WITH_PROGRESS",
    "F8_TIMEOUT_NO_PROGRESS",
    "UNCLASSIFIED_FAILURE",
]

PRIMARY_PRIORITY = {
    "F0_NO_PATH_OR_INITIAL_PLAN_FAILURE": 0,
    "F2_BUDGET_REJECTED": 1,
    "F5_OSCILLATION_OR_REPLAN_LOOP": 2,
    "F4_LOCAL_EXECUTION_DRIFT": 3,
    "F6_LONG_PATH_ACCUMULATION": 4,
    "F7_TIMEOUT_WITH_PROGRESS": 5,
    "F8_TIMEOUT_NO_PROGRESS": 6,
    "F1_SUBGOAL_NOT_REACHED": 7,
    "F3_BOUNDARY_OR_BRIDGE_MISMATCH": 8,
}


def _split_csv(value: str) -> list[str]:
    return [x.strip() for x in str(value or "").split(",") if x.strip()]


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip() == "":
            return default
        out = float(value)
        if math.isnan(out):
            return default
        return out
    except Exception:
        return default


def _str(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def _infer_from_eval_path(path: Path) -> dict[str, Any]:
    parts = list(path.parts)
    out: dict[str, Any] = {}
    try:
        seed_idx = next(i for i, part in enumerate(parts) if re.fullmatch(r"seed\d+", part))
        out["env"] = parts[seed_idx - 1]
        out["seed"] = int(parts[seed_idx].replace("seed", ""))
        out["variant"] = parts[seed_idx + 1]
        if len(parts) > seed_idx + 2 and parts[seed_idx + 2].startswith("budget"):
            out["budget"] = float(parts[seed_idx + 2].replace("budget", ""))
        if len(parts) > seed_idx + 3 and parts[seed_idx + 3].startswith("fallback_"):
            out["fallback_mode"] = parts[seed_idx + 3].replace("fallback_", "")
    except Exception:
        pass
    return out


def _collect_eval_roots(roots: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("eval.csv")):
            try:
                df = pd.read_csv(path)
            except Exception as exc:
                print(f"[stage25_atlas] skip unreadable {path}: {exc}", file=sys.stderr)
                continue
            inferred = _infer_from_eval_path(path)
            for key, value in inferred.items():
                if key not in df.columns or df[key].isna().all() or (df[key].astype(str).str.len() == 0).all():
                    df[key] = value
            df["eval_path"] = str(path)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _trace_path(row: pd.Series) -> Path | None:
    raw = _str(row.get("debug_trace_path", ""))
    if raw:
        p = Path(raw)
        if p.exists():
            return p
    eval_path = Path(_str(row.get("eval_path", "")))
    if not eval_path.exists():
        return None
    task = int(_num(row.get("task_id", 0), 0))
    ep = int(_num(row.get("episode_id", 0), 0))
    cand = eval_path.parent / "debug_traces" / f"task{task}_episode{ep}.jsonl"
    return cand if cand.exists() else None


def _drift_metrics_from_trace(rows: list[dict[str, Any]], window: int, stall_frac: float) -> dict[str, Any]:
    if not rows:
        return {
            "local_drift_score": 0.0,
            "progress_stall_count": 0,
            "oscillation_score": 0.0,
            "drift_event_count": 0,
            "refresh_count": 0,
            "replan_reason_counts": "{}",
            "first_failure_step": "",
        }
    target = np.asarray([_num(r.get("target_dist_phi", np.nan), np.nan) for r in rows], dtype=np.float64)
    target = target[np.isfinite(target)]
    local_drift = 0.0
    stall_count = 0
    if target.size >= 2:
        w = int(max(2, min(window, target.size)))
        for start in range(0, target.size - w + 1):
            seg = target[start : start + w]
            denom = max(float(seg[0]), 1e-6)
            local_drift = max(local_drift, float((seg[-1] - seg[0]) / denom))
            progress = float((seg[0] - np.min(seg)) / denom)
            if progress < stall_frac:
                stall_count += 1
        deltas = np.diff(target)
        path_motion = float(np.sum(np.abs(deltas)))
        net_improvement = float(max(target[0] - target[-1], 0.0))
        oscillation = float(max(path_motion - net_improvement, 0.0) / max(path_motion, 1e-6))
    else:
        oscillation = 0.0
    drift_events = 0
    refresh_count = 0
    reasons: Counter[str] = Counter()
    first_failure_step = ""
    for r in rows:
        if _num(r.get("drift_event_count", 0), 0) > 0 or _num(r.get("drift_event", 0), 0) > 0:
            drift_events += 1
            if first_failure_step == "":
                first_failure_step = int(_num(r.get("step", 0), 0))
        if _num(r.get("refresh_triggered", 0), 0) > 0:
            refresh_count += 1
            if first_failure_step == "":
                first_failure_step = int(_num(r.get("step", 0), 0))
        if _num(r.get("replan_triggered", 0), 0) > 0:
            reason = _str(r.get("replan_reason", "")) or "unknown"
            reasons[reason] += 1
            if first_failure_step == "":
                first_failure_step = int(_num(r.get("step", 0), 0))
        if _num(r.get("refresh_count", 0), 0) > refresh_count:
            refresh_count = int(_num(r.get("refresh_count", 0), 0))
        if _num(r.get("drift_event_count", 0), 0) > drift_events:
            drift_events = int(_num(r.get("drift_event_count", 0), 0))
    return {
        "local_drift_score": float(max(0.0, local_drift)),
        "progress_stall_count": int(stall_count),
        "oscillation_score": float(oscillation),
        "drift_event_count": int(drift_events),
        "refresh_count": int(refresh_count),
        "replan_reason_counts": json.dumps(dict(reasons), sort_keys=True),
        "first_failure_step": first_failure_step,
    }


def _normalize_old_label(label: str) -> str:
    old = _str(label)
    mapping = {
        "SUCCESS": "SUCCESS",
        "F1_NO_GRAPH_PATH": "F0_NO_PATH_OR_INITIAL_PLAN_FAILURE",
        "F2_START_GOAL_PROJECTION_ERROR": "F0_NO_PATH_OR_INITIAL_PLAN_FAILURE",
        "F3_FIRST_SUBGOAL_LOCAL_FAILURE": "F1_SUBGOAL_NOT_REACHED",
        "F4_LOCAL_EXECUTION_DRIFT": "F4_LOCAL_EXECUTION_DRIFT",
        "F5_BRIDGE_FALSE_POSITIVE": "F3_BOUNDARY_OR_BRIDGE_MISMATCH",
        "F6_LONG_PATH_ACCUMULATION": "F6_LONG_PATH_ACCUMULATION",
        "F8_TIMEOUT_AFTER_PROGRESS": "F7_TIMEOUT_WITH_PROGRESS",
        "UNCLASSIFIED_FAILURE": "UNCLASSIFIED_FAILURE",
    }
    return mapping.get(old, old if old in FAILURE_TYPES else "")


def _classify(row: pd.Series, args: argparse.Namespace, trace_metrics: dict[str, Any]) -> tuple[str, list[str], str, float]:
    success = _num(row.get("success", 0), 0) >= 0.5
    if success:
        return "SUCCESS", [], "success", 1.0

    labels: list[str] = []
    source = "stage25_fallback"
    confidence = 0.55
    old_primary = _normalize_old_label(_str(row.get("primary_failure_type", "")))
    if old_primary and old_primary != "SUCCESS":
        labels.append(old_primary)
        source = "existing_primary_plus_stage25"
        confidence = 0.65

    no_path = (
        _num(row.get("no_path_count", 0), 0) > 0
        or _str(row.get("first_plan_reject_reason", "")).startswith(("no_", "disconnected", "planner_error"))
        or (_num(row.get("first_plan_edges", 0), 0) <= 0 and _num(row.get("steps", 0), 0) <= 1)
    )
    budget_reject = _num(row.get("budget_reject_count", 0), 0) > 0 or _str(row.get("first_plan_reject_reason", "")) == "budget_infeasible"
    if no_path:
        labels.append("F0_NO_PATH_OR_INITIAL_PLAN_FAILURE")
    if budget_reject:
        labels.append("F2_BUDGET_REJECTED")

    sub_attempted = _num(row.get("subgoals_attempted", row.get("subgoals_total", 0)), 0)
    sub_reached = _num(row.get("subgoals_reached", 0), 0)
    sub_rate = _num(row.get("subgoal_reach_rate", 0), 0)
    first_edges = _num(row.get("first_plan_edges", 0), 0)
    last_edges = _num(row.get("last_plan_edges", first_edges), first_edges)
    replans = _num(row.get("replans", 0), 0)
    local_drift = max(_num(row.get("local_drift_score", 0), 0), _num(trace_metrics.get("local_drift_score", 0), 0))
    progress_stall = max(_num(row.get("progress_stall_count", 0), 0), _num(trace_metrics.get("progress_stall_count", 0), 0))
    oscillation = max(_num(row.get("oscillation_score", 0), 0), _num(trace_metrics.get("oscillation_score", 0), 0))
    drift_events = max(_num(row.get("drift_event_count", 0), 0), _num(trace_metrics.get("drift_event_count", 0), 0))
    if sub_attempted > 0 and sub_reached < sub_attempted and sub_rate < 0.5:
        labels.append("F1_SUBGOAL_NOT_REACHED")
    if (local_drift >= args.drift_threshold or drift_events > 0 or progress_stall >= 8) and first_edges > 0:
        labels.append("F4_LOCAL_EXECUTION_DRIFT")
        confidence = max(confidence, 0.75)
    if oscillation >= 0.95 and replans >= 4:
        labels.append("F5_OSCILLATION_OR_REPLAN_LOOP")
        confidence = max(confidence, 0.70)
    if max(first_edges, last_edges) >= 6 and sub_reached >= 1:
        labels.append("F6_LONG_PATH_ACCUMULATION")

    steps = _num(row.get("steps", 0), 0)
    max_steps = _num(row.get("max_steps", 1000), 1000)
    initial = _num(row.get("initial_goal_dist_phi", np.nan), np.nan)
    best = _num(row.get("best_goal_dist_phi", np.nan), np.nan)
    improvement = initial - best if np.isfinite(initial) and np.isfinite(best) else _num(row.get("goal_improvement_phi", 0), 0)
    if steps >= 0.95 * max(max_steps, 1):
        if improvement > max(2.0, 0.15 * max(initial if np.isfinite(initial) else 1.0, 1.0)):
            labels.append("F7_TIMEOUT_WITH_PROGRESS")
        else:
            labels.append("F8_TIMEOUT_NO_PROGRESS")

    labels = [x for x in dict.fromkeys(labels) if x in FAILURE_TYPES and x != "SUCCESS"]
    if not labels:
        return "UNCLASSIFIED_FAILURE", ["UNCLASSIFIED_FAILURE"], source, 0.25
    primary = sorted(labels, key=lambda x: PRIMARY_PRIORITY.get(x, 999))[0]
    return primary, labels, source, confidence


def enrich(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        trace_rows = _read_jsonl(p) if (p := _trace_path(row)) is not None else []
        trace_metrics = _drift_metrics_from_trace(trace_rows, args.trace_window, args.stall_frac)
        primary, labels, source, confidence = _classify(row, args, trace_metrics)
        out = row.to_dict()
        sub_attempted = _num(out.get("subgoals_attempted", out.get("subgoals_total", 0)), 0)
        out["subgoals_total"] = int(sub_attempted)
        out["subgoals_reached"] = int(_num(out.get("subgoals_reached", 0), 0))
        out["primary_failure_type"] = primary
        out["failure_labels"] = "|".join(labels)
        out["first_failure_step"] = out.get("first_failure_step", "") or trace_metrics.get("first_failure_step", "")
        if _str(out.get("first_failed_subgoal_idx", "")) == "" and primary not in {"SUCCESS", "F0_NO_PATH_OR_INITIAL_PLAN_FAILURE"}:
            out["first_failed_subgoal_idx"] = int(out["subgoals_reached"]) + 1 if sub_attempted else ""
        for key in ["local_drift_score", "progress_stall_count", "oscillation_score", "drift_event_count", "refresh_count"]:
            out[key] = max(_num(out.get(key, 0), 0), _num(trace_metrics.get(key, 0), 0))
        out["replan_reason_counts"] = _str(out.get("replan_reason_counts", "")) or trace_metrics.get("replan_reason_counts", "{}")
        out["source_labeler"] = source
        out["label_confidence"] = confidence
        for col in REQUIRED_COLUMNS:
            out.setdefault(col, "")
        rows.append(out)
    out_df = pd.DataFrame(rows)
    if len(out_df):
        for col in REQUIRED_COLUMNS:
            if col not in out_df.columns:
                out_df[col] = ""
        front = REQUIRED_COLUMNS
        rest = [c for c in out_df.columns if c not in front]
        out_df = out_df[front + rest]
    return out_df


def _complete_cell_mask(df: pd.DataFrame, min_episodes: int) -> pd.Series:
    if len(df) == 0:
        return pd.Series(dtype=bool)
    keys = [c for c in ["env", "seed", "variant", "fallback_mode"] if c in df.columns]
    if not keys:
        return pd.Series([True] * len(df), index=df.index)
    counts = df.groupby(keys, dropna=False)["success"].transform("count")
    return counts >= int(min_episodes)


def summarize(atlas: pd.DataFrame, min_episodes: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    if len(atlas) == 0:
        summary = pd.DataFrame(columns=["env", "seed", "variant", "fallback_mode", "primary_failure_type", "episodes", "success"])
        integrity = {
            "total_rows": 0,
            "failed_rows": 0,
            "missing_primary_failure_type_failed_rows": 0,
            "missing_label_rate_failed_rows": 0.0,
            "unclassified_failure_rate": 0.0,
            "complete_cells": 0,
            "incomplete_cells": 0,
            "variants": [],
            "variant_missing_label_rates": {},
            "gate": "FAIL_LABEL_INTEGRITY",
        }
        return summary, integrity

    failed = atlas[pd.to_numeric(atlas["success"], errors="coerce").fillna(0) < 0.5].copy()
    missing = failed["primary_failure_type"].astype(str).str.len().eq(0)
    unclassified = failed["primary_failure_type"].astype(str).eq("UNCLASSIFIED_FAILURE")
    keys = [c for c in ["env", "seed", "variant", "fallback_mode", "primary_failure_type"] if c in atlas.columns]
    summary = (
        atlas.groupby(keys, dropna=False)
        .agg(
            episodes=("success", "count"),
            success=("success", "mean"),
            steps=("steps", "mean"),
            local_drift_score=("local_drift_score", "mean"),
            oscillation_score=("oscillation_score", "mean"),
            refresh_count=("refresh_count", "mean"),
            drift_event_count=("drift_event_count", "mean"),
        )
        .reset_index()
    )
    cell_keys = [c for c in ["env", "seed", "variant", "fallback_mode"] if c in atlas.columns]
    cell_counts = atlas.groupby(cell_keys, dropna=False)["success"].count() if cell_keys else pd.Series(dtype=int)
    complete_cells = int((cell_counts >= min_episodes).sum()) if len(cell_counts) else 0
    incomplete_cells = int((cell_counts < min_episodes).sum()) if len(cell_counts) else 0
    variant_missing_rates: dict[str, float] = {}
    for variant, sub in failed.groupby("variant", dropna=False):
        miss = sub["primary_failure_type"].astype(str).str.len().eq(0)
        variant_missing_rates[str(variant)] = float(miss.mean()) if len(sub) else 0.0
    failed_rows = int(len(failed))
    missing_rows = int(missing.sum())
    unclassified_rate = float(unclassified.mean()) if failed_rows else 0.0
    gate = "PASS_LABEL_INTEGRITY"
    if missing_rows > 0:
        gate = "FAIL_LABEL_INTEGRITY"
    elif unclassified_rate > 0.05:
        gate = "HOLD_LABEL_INTEGRITY"
    integrity = {
        "total_rows": int(len(atlas)),
        "failed_rows": failed_rows,
        "missing_primary_failure_type_failed_rows": missing_rows,
        "missing_label_rate_failed_rows": float(missing.mean()) if failed_rows else 0.0,
        "unclassified_failure_rate": unclassified_rate,
        "complete_cells": complete_cells,
        "incomplete_cells": incomplete_cells,
        "variants": sorted(str(x) for x in atlas["variant"].dropna().unique()),
        "variant_missing_label_rates": variant_missing_rates,
        "gate": gate,
    }
    return summary, integrity


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-roots", required=True)
    p.add_argument("--out", default="reports/stage25_failure_atlas_all_variants.csv")
    p.add_argument("--summary-out", default="reports/stage25_failure_atlas_summary.csv")
    p.add_argument("--integrity-out", default="reports/stage25_label_integrity.json")
    p.add_argument("--min-episodes", type=int, default=100)
    p.add_argument("--trace-window", type=int, default=16)
    p.add_argument("--stall-frac", type=float, default=0.03)
    p.add_argument("--drift-threshold", type=float, default=0.10)
    p.add_argument("--include-incomplete-cells", type=int, default=1)
    p.add_argument("--failure-atlas-prior", default="")
    p.add_argument("--strict", type=int, default=1)
    p.add_argument("--max-unclassified-rate", type=float, default=0.05)
    args = p.parse_args()

    roots = [Path(x) for x in _split_csv(args.eval_roots)]
    eval_df = _collect_eval_roots(roots)
    if len(eval_df) and not args.include_incomplete_cells:
        eval_df = eval_df[_complete_cell_mask(eval_df, args.min_episodes)].copy()
    atlas = enrich(eval_df, args) if len(eval_df) else pd.DataFrame(columns=REQUIRED_COLUMNS)
    summary, integrity = summarize(atlas, args.min_episodes)
    if integrity["missing_primary_failure_type_failed_rows"] == 0 and integrity["unclassified_failure_rate"] > args.max_unclassified_rate:
        integrity["gate"] = "HOLD_LABEL_INTEGRITY"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    atlas.to_csv(out, index=False)
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_out, index=False)
    Path(args.integrity_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.integrity_out).write_text(json.dumps(integrity, indent=2, sort_keys=True) + "\n")
    if args.strict and integrity["gate"] == "FAIL_LABEL_INTEGRITY":
        raise SystemExit(2)
    print(json.dumps({"rows": len(atlas), "gate": integrity["gate"]}, sort_keys=True))


if __name__ == "__main__":
    main()
