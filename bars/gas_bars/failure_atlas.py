from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .bridge_graph import BRIDGE_EDGE_TYPES, RISKY_EDGE_TYPES


FAILURE_TYPES = [
    "F0_PROTOCOL_OR_GOAL_MISMATCH",
    "F1_NO_GRAPH_PATH",
    "F2_START_GOAL_PROJECTION_ERROR",
    "F3_FIRST_SUBGOAL_LOCAL_FAILURE",
    "F4_LOCAL_EXECUTION_DRIFT",
    "F5_BRIDGE_FALSE_POSITIVE",
    "F6_LONG_PATH_ACCUMULATION",
    "F7_FINAL_CONNECTOR_FAILURE",
    "F8_TIMEOUT_AFTER_PROGRESS",
]


def compact_hash(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    s = str(value)
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:12]


def parse_ids(value: Any) -> list[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    out = []
    for part in str(value).split("|"):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(float(part)))
        except Exception:
            continue
    return out


def _num(row: pd.Series, key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value == "":
            return default
        return float(value)
    except Exception:
        return default


def edge_type_counts(path_edge_ids: list[int], edge_types: dict[int, str]) -> dict[str, int]:
    counts = {
        "safe_local_count": 0,
        "same_traj_count": 0,
        "gas_cross_count": 0,
        "aggressive_bridge_count": 0,
        "bottleneck_bridge_count": 0,
        "virtual_connector_count": 0,
        "risky_bridge_count": 0,
    }
    for eid in path_edge_ids:
        et = edge_types.get(int(eid), "")
        if et == "safe_local":
            counts["safe_local_count"] += 1
        elif et == "same_traj_temporal":
            counts["same_traj_count"] += 1
        elif et == "gas_cross":
            counts["gas_cross_count"] += 1
        elif et == "aggressive_tdr_bridge":
            counts["aggressive_bridge_count"] += 1
        elif et == "bottleneck_bridge":
            counts["bottleneck_bridge_count"] += 1
        elif et == "virtual_connector":
            counts["virtual_connector_count"] += 1
        if et in RISKY_EDGE_TYPES:
            counts["risky_bridge_count"] += 1
    return counts


def classify_episode(row: pd.Series, edge_types: Optional[dict[int, str]] = None) -> tuple[str, list[str], dict[str, Any]]:
    edge_types = edge_types or {}
    labels: list[str] = []
    path_edge_ids = parse_ids(row.get("path_edge_ids", ""))
    counts = edge_type_counts(path_edge_ids, edge_types)
    success = bool(_num(row, "success", 0.0) >= 0.5)
    steps = _num(row, "steps", 0.0)
    max_steps = _num(row, "max_steps", 1000.0) or 1000.0
    initial = _num(row, "initial_goal_dist_phi", np.nan)
    final = _num(row, "final_goal_dist_phi", np.nan)
    best = _num(row, "best_goal_dist_phi", np.nan)
    improvement = _num(row, "goal_improvement_phi", initial - best if np.isfinite(initial) and np.isfinite(best) else 0.0)
    sub_attempted = _num(row, "subgoals_attempted", 0.0)
    sub_reached = _num(row, "subgoals_reached", 0.0)
    sub_rate = _num(row, "subgoal_reach_rate", 0.0)
    first_edges = _num(row, "first_plan_edges", 0.0)
    no_path = _num(row, "no_path_count", 0.0) > 0 or str(row.get("first_plan_reject_reason", "")).startswith(("no_", "disconnected"))
    budget_reject = _num(row, "budget_reject_count", 0.0) > 0 or str(row.get("first_plan_reject_reason", "")) == "budget_infeasible"
    goal_missing = not str(row.get("actual_goal_raw", "")).strip() or not str(row.get("actual_goal_phi", "")).strip()
    planner_goal_missing = "planner_goal_phi" in row.index and not str(row.get("planner_goal_phi", "")).strip()
    if goal_missing or planner_goal_missing or steps == 0 and not no_path and not budget_reject:
        labels.append("F0_PROTOCOL_OR_GOAL_MISMATCH")
    if no_path or budget_reject or first_edges <= 0:
        labels.append("F1_NO_GRAPH_PATH")
    if first_edges > 0 and sub_attempted == 0 and not success:
        labels.append("F2_START_GOAL_PROJECTION_ERROR")
    if sub_attempted > 0 and sub_reached == 0 and not success:
        labels.append("F3_FIRST_SUBGOAL_LOCAL_FAILURE")
    if sub_attempted >= 3 and sub_rate < 0.35 and not success and counts["risky_bridge_count"] == 0:
        labels.append("F4_LOCAL_EXECUTION_DRIFT")
    if counts["risky_bridge_count"] > 0 and not success:
        labels.append("F5_BRIDGE_FALSE_POSITIVE")
    if sub_reached >= 2 and sub_rate >= 0.35 and not success and (first_edges >= 8 or len(path_edge_ids) >= 8):
        labels.append("F6_LONG_PATH_ACCUMULATION")
    if np.isfinite(final) and final <= 2.5 and not success:
        labels.append("F7_FINAL_CONNECTOR_FAILURE")
    if not success and steps >= 0.95 * max_steps and improvement > max(2.0, 0.15 * max(initial, 1.0)):
        labels.append("F8_TIMEOUT_AFTER_PROGRESS")
    if success:
        labels = []
        primary = "SUCCESS"
    elif labels:
        priority = {name: i for i, name in enumerate(FAILURE_TYPES)}
        primary = sorted(labels, key=lambda x: priority.get(x, 999))[0]
    else:
        primary = "UNCLASSIFIED_FAILURE"
    return primary, labels, counts


def enrich_failure_atlas(eval_df: pd.DataFrame, edge_table: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    edge_types = {}
    if edge_table is not None and len(edge_table) and {"edge_id", "edge_type"}.issubset(edge_table.columns):
        edge_types = {int(r.edge_id): str(r.edge_type) for r in edge_table[["edge_id", "edge_type"]].itertuples(index=False)}
    rows = []
    for _, row in eval_df.iterrows():
        primary, labels, counts = classify_episode(row, edge_types=edge_types)
        out = row.to_dict()
        out["actual_goal_phi_hash"] = compact_hash(row.get("actual_goal_phi", ""))
        out["planner_goal_phi_hash"] = compact_hash(row.get("planner_goal_phi", ""))
        out["primary_failure_type"] = primary
        out["failure_labels"] = "|".join(labels)
        out["failure_at_final_connector"] = int("F7_FINAL_CONNECTOR_FAILURE" in labels)
        out.update(counts)
        rows.append(out)
    return pd.DataFrame(rows)


def grouped_failure_atlas(atlas: pd.DataFrame) -> pd.DataFrame:
    if len(atlas) == 0:
        return pd.DataFrame()
    keys = [k for k in ["env", "seed", "variant", "primary_failure_type"] if k in atlas.columns]
    grouped = atlas.groupby(keys, dropna=False).agg(
        episodes=("success", "count"),
        success=("success", "mean"),
        steps=("steps", "mean"),
        subgoal_reach_rate=("subgoal_reach_rate", "mean"),
        no_path_rate=("no_path_count", lambda x: float((pd.to_numeric(x, errors="coerce").fillna(0) > 0).mean())),
        risky_bridge_count=("risky_bridge_count", "mean"),
    )
    return grouped.reset_index()


def write_failure_report(atlas: pd.DataFrame, grouped: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    lines = ["# Stage23 Failure Atlas", ""]
    if len(atlas) == 0:
        lines.append("No evaluation rows were available.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n")
        return
    failures = atlas[atlas["primary_failure_type"] != "SUCCESS"]
    lines.append(f"- Episodes: {len(atlas)}")
    lines.append(f"- Success rate: {float(pd.to_numeric(atlas['success'], errors='coerce').fillna(0).mean()):.3f}")
    lines.append(f"- Failures classified: {len(failures)}")
    lines.append("")
    if len(grouped):
        try:
            lines.append(grouped.to_markdown(index=False))
        except Exception:
            lines.append("```csv\n" + grouped.to_csv(index=False).strip() + "\n```")
    lines.append("")
    if len(failures):
        top = failures["primary_failure_type"].value_counts(normalize=True)
        lines.append("## Readout")
        lines.append(f"- Dominant failure type: {top.index[0]} ({top.iloc[0]:.1%} of failures).")
        bridge_fail = float((failures.get("risky_bridge_count", pd.Series([0] * len(failures))) > 0).mean())
        lines.append(f"- Failures with risky bridge/path evidence: {bridge_fail:.1%}.")
        local_fail = float(failures["primary_failure_type"].isin(["F3_FIRST_SUBGOAL_LOCAL_FAILURE", "F4_LOCAL_EXECUTION_DRIFT"]).mean())
        lines.append(f"- Failures looking like local low-level execution: {local_fail:.1%}.")
        protocol_fail = float((failures["primary_failure_type"] == "F0_PROTOCOL_OR_GOAL_MISMATCH").mean())
        lines.append(f"- Protocol/goal mismatch share: {protocol_fail:.1%}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def load_edge_table(path: str | Path) -> Optional[pd.DataFrame]:
    p = Path(path)
    if p.is_file():
        return pd.read_csv(p)
    if p.is_dir():
        candidates = list(p.rglob("bridge_table.csv")) + list(p.rglob("gas_graph_edges.csv"))
        if candidates:
            return pd.read_csv(candidates[0])
    return None
