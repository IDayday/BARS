#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REACH_VARIANTS = ["gas_shortest", "gas_reachability_budget_calibrated"]
LOCAL_PRIMARY = [
    "gas_shortest_subgoal_refresh_on_target_distance_increase",
    "gas_shortest_nearest_reachable_subgoal_on_path",
]
LOCAL_DIAGNOSTIC = ["gas_shortest_drift_replan_with_cooldown"]
BASELINE = "gas_shortest"

REACH_COLUMNS = [
    "env",
    "seed",
    "fallback_mode",
    "variant",
    "episodes",
    "success",
    "steps",
    "success_delta_vs_shortest",
    "paired_episode_n",
    "paired_episode_success_delta",
    "paired_wins",
    "paired_losses",
    "paired_ties",
    "steps_inflation_vs_shortest",
    "no_path_rate",
    "budget_reject_rate",
    "no_path_delta_vs_shortest",
    "budget_reject_delta_vs_shortest",
    "subgoal_reach_rate",
    "source",
]

LOCAL_COLUMNS = [
    "env",
    "seed",
    "variant",
    "baseline_variant",
    "episodes",
    "success",
    "baseline_success",
    "success_delta",
    "paired_episode_success_delta",
    "cell_result",
    "steps",
    "steps_inflation",
    "F4_rate",
    "baseline_F4_rate",
    "F4_relative_reduction",
    "subgoal_reach_rate",
    "subgoal_reach_delta",
    "local_drift_score_mean",
    "local_drift_score_delta",
    "progress_stall_count_mean",
    "progress_stall_count_delta",
    "oscillation_score_mean",
    "oscillation_score_delta",
    "refresh_count_mean",
    "drift_event_count_mean",
    "helpful_replan_rate",
    "harmful_replan_rate",
]

ORACLE_COLUMNS = [
    "env",
    "seed",
    "graph_id",
    "node_count",
    "edge_count",
    "bridge_count",
    "shorter_path_rate",
    "bridge_usage_rate",
    "mean_path_cost_reduction",
    "oracle_bridge_count",
    "oracle_bridge_fraction",
    "oracle_shorter_path_rate",
    "oracle_bridge_usage_rate",
    "oracle_mean_path_cost_reduction",
    "useful_bridge_score",
    "safe_local_success_rate",
    "risky_bridge_success_rate",
    "gas_cross_success_rate",
    "set_state_rate",
    "gate",
    "failure_reason",
]

BOUNDARY_COLUMNS = [
    "env",
    "seed",
    "graph_id",
    "junction_class",
    "junction_count",
    "supported_count",
    "unsupported_count",
    "coverage",
    "supported_success_rate",
    "unsupported_success_rate",
    "supported_gap",
    "psi_AUROC",
    "psi_AUPRC",
]


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except Exception:
        return {}


def _split(value: str) -> list[str]:
    return [x.strip() for x in str(value or "").split(",") if x.strip()]


def _num(value: Any, default: float = np.nan) -> float:
    try:
        if value is None or (isinstance(value, str) and value.strip() == ""):
            return default
        out = float(value)
        return default if math.isnan(out) else out
    except Exception:
        return default


def _infer_eval_path(path: Path) -> dict[str, Any]:
    parts = list(path.parts)
    out: dict[str, Any] = {}
    for i, part in enumerate(parts):
        if part.startswith("seed") and part[4:].isdigit() and i >= 1:
            out["env"] = parts[i - 1]
            out["seed"] = int(part[4:])
            if i + 1 < len(parts):
                out["variant"] = parts[i + 1]
            if i + 2 < len(parts) and parts[i + 2].startswith("budget"):
                out["budget"] = _num(parts[i + 2].replace("budget", ""), np.nan)
            if i + 3 < len(parts) and parts[i + 3].startswith("fallback_"):
                out["fallback_mode"] = parts[i + 3].replace("fallback_", "")
            break
    return out


def collect_eval_roots(roots: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("eval.csv")):
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            inferred = _infer_eval_path(path)
            for key, value in inferred.items():
                if key not in df.columns or df[key].isna().all():
                    df[key] = value
            if "fallback_mode" not in df.columns:
                df["fallback_mode"] = inferred.get("fallback_mode", "")
            df["eval_path"] = str(path)
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _complete_cells(df: pd.DataFrame, min_episodes: int) -> pd.DataFrame:
    if len(df) == 0:
        return df
    keys = [c for c in ["env", "seed", "variant", "fallback_mode"] if c in df.columns]
    if not keys:
        return df
    counts = df.groupby(keys, dropna=False)["success"].transform("count")
    return df[counts >= int(min_episodes)].copy()


def _agg_eval(df: pd.DataFrame, source: str, min_episodes: int = 1) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame()
    d = _complete_cells(df, min_episodes)
    if len(d) == 0:
        return pd.DataFrame()
    for col in ["no_path_count", "budget_reject_count", "subgoal_reach_rate", "steps"]:
        if col not in d.columns:
            d[col] = 0.0
    keys = [c for c in ["env", "seed", "fallback_mode", "variant"] if c in d.columns]
    out = (
        d.groupby(keys, dropna=False)
        .agg(
            episodes=("success", "count"),
            success=("success", "mean"),
            steps=("steps", "mean"),
            no_path_rate=("no_path_count", lambda x: float((pd.to_numeric(x, errors="coerce").fillna(0) > 0).mean())),
            budget_reject_rate=("budget_reject_count", lambda x: float((pd.to_numeric(x, errors="coerce").fillna(0) > 0).mean())),
            subgoal_reach_rate=("subgoal_reach_rate", "mean"),
        )
        .reset_index()
    )
    out["source"] = source
    return out


def _paired(df: pd.DataFrame, variant: str, min_episodes: int) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame()
    d = _complete_cells(df, min_episodes)
    d = d[d.get("fallback_mode", "").astype(str).eq("none") & d["variant"].astype(str).isin([BASELINE, variant])].copy()
    keys = [c for c in ["env", "seed", "fallback_mode", "task_id", "episode_id"] if c in d.columns]
    if len(d) == 0 or not keys:
        return pd.DataFrame()
    piv = d.pivot_table(index=keys, columns="variant", values="success", aggfunc="first").reset_index()
    if BASELINE not in piv.columns or variant not in piv.columns:
        return pd.DataFrame()
    rows = []
    for (env, seed, fallback), sub in piv.groupby(["env", "seed", "fallback_mode"], dropna=False):
        valid = sub[[BASELINE, variant]].dropna()
        if len(valid) == 0:
            continue
        diff = pd.to_numeric(valid[variant], errors="coerce") - pd.to_numeric(valid[BASELINE], errors="coerce")
        rows.append(
            {
                "env": env,
                "seed": seed,
                "fallback_mode": fallback,
                "variant": variant,
                "paired_episode_n": int(len(diff)),
                "paired_episode_success_delta": float(diff.mean()),
                "paired_wins": int((diff > 0).sum()),
                "paired_losses": int((diff < 0).sum()),
                "paired_ties": int((diff == 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def analyze_reachability(eval_df: pd.DataFrame, reports: Path, min_episodes: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    grouped = _agg_eval(eval_df, "stage25_run", min_episodes)
    out = pd.DataFrame(columns=REACH_COLUMNS)
    details: dict[str, Any] = {
        "gate": "HOLD_REACHABILITY_INCOMPLETE",
        "completed_cells": 0,
        "mean_delta": np.nan,
        "wins": 0,
        "paired_episode_n": 0,
        "mean_steps_inflation": np.nan,
        "no_path_delta": np.nan,
        "budget_reject_delta": np.nan,
    }
    if len(grouped):
        grouped = grouped[grouped["variant"].astype(str).isin(REACH_VARIANTS) & grouped["fallback_mode"].astype(str).eq("none")].copy()
    if len(grouped):
        rows = []
        for (env, seed, fallback), sub in grouped.groupby(["env", "seed", "fallback_mode"], dropna=False):
            base_rows = sub[sub["variant"].astype(str).eq(BASELINE)]
            base = base_rows.iloc[0] if len(base_rows) else None
            for _, row in sub.iterrows():
                r = row.to_dict()
                if base is not None and row["variant"] != BASELINE:
                    r["success_delta_vs_shortest"] = float(row["success"] - base["success"])
                    r["steps_inflation_vs_shortest"] = float(row["steps"] / max(float(base["steps"]), 1e-9) - 1.0)
                    r["no_path_delta_vs_shortest"] = float(row["no_path_rate"] - base["no_path_rate"])
                    r["budget_reject_delta_vs_shortest"] = float(row["budget_reject_rate"] - base["budget_reject_rate"])
                rows.append(r)
        out = pd.DataFrame(rows)
        pair = _paired(eval_df, "gas_reachability_budget_calibrated", min_episodes)
        if len(pair):
            out = out.merge(pair, on=["env", "seed", "fallback_mode", "variant"], how="left")
        for col in REACH_COLUMNS:
            if col not in out.columns:
                out[col] = np.nan
        out = out[REACH_COLUMNS]
        primary = out[out["variant"].astype(str).eq("gas_reachability_budget_calibrated")].copy()
        complete = primary[pd.to_numeric(primary["episodes"], errors="coerce").fillna(0) >= min_episodes]
        details["completed_cells"] = int(len(complete))
        if len(complete):
            details["mean_delta"] = float(pd.to_numeric(complete["success_delta_vs_shortest"], errors="coerce").mean())
            details["wins"] = int((pd.to_numeric(complete["success_delta_vs_shortest"], errors="coerce") > 0).sum())
            details["paired_episode_n"] = int(pd.to_numeric(complete["paired_episode_n"], errors="coerce").fillna(0).sum())
            details["mean_steps_inflation"] = float(pd.to_numeric(complete["steps_inflation_vs_shortest"], errors="coerce").fillna(0).mean())
            details["no_path_delta"] = float(pd.to_numeric(complete["no_path_delta_vs_shortest"], errors="coerce").fillna(0).mean())
            details["budget_reject_delta"] = float(pd.to_numeric(complete["budget_reject_delta_vs_shortest"], errors="coerce").fillna(0).mean())
        if details["completed_cells"] < 6:
            details["gate"] = "HOLD_REACHABILITY_INCOMPLETE"
        elif (
            details["mean_delta"] >= 0.02
            and details["wins"] >= 4
            and details["paired_episode_n"] >= 6 * min_episodes
            and details["mean_steps_inflation"] <= 0.10
            and details["no_path_delta"] <= 0.02
            and details["budget_reject_delta"] <= 0.02
        ):
            details["gate"] = "PASS_REACHABILITY_CLOSING"
        else:
            details["gate"] = "CLOSE_REACHABILITY_MAINLINE"
    out.to_csv(reports / "stage25_reachability_closing.csv", index=False)
    return out, details


def analyze_local_drift(atlas: pd.DataFrame, reports: Path, min_episodes: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = pd.DataFrame(columns=LOCAL_COLUMNS)
    integrity = _read_json(reports / "stage25_label_integrity.json")
    details = {
        "gate": "HOLD_LOCAL_DRIFT_V2_INCOMPLETE",
        "completed_primary_cells": 0,
        "best_variant": "",
        "best_success_delta": np.nan,
        "best_wins": 0,
        "best_F4_relative_reduction": np.nan,
        "label_integrity_gate": integrity.get("gate", "MISSING_LABEL_INTEGRITY"),
        "missing_label_failed_rows": int(integrity.get("missing_primary_failure_type_failed_rows", 0) or 0),
    }
    if len(atlas) == 0:
        out.to_csv(reports / "stage25_local_drift_v2.csv", index=False)
        return out, details
    d = _complete_cells(atlas, min_episodes)
    if len(d) == 0:
        out.to_csv(reports / "stage25_local_drift_v2.csv", index=False)
        return out, details
    for col in ["local_drift_score", "progress_stall_count", "oscillation_score", "refresh_count", "drift_event_count", "subgoal_reach_rate", "steps"]:
        if col not in d.columns:
            d[col] = 0.0
    rows = []
    variants = LOCAL_PRIMARY + LOCAL_DIAGNOSTIC
    for (env, seed), sub in d.groupby(["env", "seed"], dropna=False):
        base = sub[sub["variant"].astype(str).eq(BASELINE)]
        if len(base) == 0:
            continue
        base_success = float(pd.to_numeric(base["success"], errors="coerce").fillna(0).mean())
        base_steps = float(pd.to_numeric(base["steps"], errors="coerce").fillna(0).mean())
        base_f4 = float(base["primary_failure_type"].astype(str).eq("F4_LOCAL_EXECUTION_DRIFT").mean())
        base_subgoal = float(pd.to_numeric(base["subgoal_reach_rate"], errors="coerce").fillna(0).mean())
        base_drift = float(pd.to_numeric(base["local_drift_score"], errors="coerce").fillna(0).mean())
        base_stall = float(pd.to_numeric(base["progress_stall_count"], errors="coerce").fillna(0).mean())
        base_osc = float(pd.to_numeric(base["oscillation_score"], errors="coerce").fillna(0).mean())
        for variant in variants:
            rep = sub[sub["variant"].astype(str).eq(variant)]
            if len(rep) == 0:
                continue
            success = float(pd.to_numeric(rep["success"], errors="coerce").fillna(0).mean())
            steps = float(pd.to_numeric(rep["steps"], errors="coerce").fillna(0).mean())
            f4 = float(rep["primary_failure_type"].astype(str).eq("F4_LOCAL_EXECUTION_DRIFT").mean())
            subgoal = float(pd.to_numeric(rep["subgoal_reach_rate"], errors="coerce").fillna(0).mean())
            drift = float(pd.to_numeric(rep["local_drift_score"], errors="coerce").fillna(0).mean())
            stall = float(pd.to_numeric(rep["progress_stall_count"], errors="coerce").fillna(0).mean())
            osc = float(pd.to_numeric(rep["oscillation_score"], errors="coerce").fillna(0).mean())
            delta = success - base_success
            rows.append(
                {
                    "env": env,
                    "seed": seed,
                    "variant": variant,
                    "baseline_variant": BASELINE,
                    "episodes": int(len(rep)),
                    "success": success,
                    "baseline_success": base_success,
                    "success_delta": delta,
                    "paired_episode_success_delta": delta,
                    "cell_result": "win" if delta > 0 else "loss" if delta < 0 else "tie",
                    "steps": steps,
                    "steps_inflation": steps / max(base_steps, 1e-9) - 1.0,
                    "F4_rate": f4,
                    "baseline_F4_rate": base_f4,
                    "F4_relative_reduction": (base_f4 - f4) / max(base_f4, 1e-9) if base_f4 > 0 else 0.0,
                    "subgoal_reach_rate": subgoal,
                    "subgoal_reach_delta": subgoal - base_subgoal,
                    "local_drift_score_mean": drift,
                    "local_drift_score_delta": drift - base_drift,
                    "progress_stall_count_mean": stall,
                    "progress_stall_count_delta": stall - base_stall,
                    "oscillation_score_mean": osc,
                    "oscillation_score_delta": osc - base_osc,
                    "refresh_count_mean": float(pd.to_numeric(rep.get("refresh_count", 0), errors="coerce").fillna(0).mean()),
                    "drift_event_count_mean": float(pd.to_numeric(rep.get("drift_event_count", 0), errors="coerce").fillna(0).mean()),
                    "helpful_replan_rate": 0.0,
                    "harmful_replan_rate": 0.0,
                }
            )
    out = pd.DataFrame(rows, columns=LOCAL_COLUMNS)
    if len(out):
        primary = out[out["variant"].isin(LOCAL_PRIMARY) & (pd.to_numeric(out["episodes"], errors="coerce").fillna(0) >= min_episodes)]
        details["completed_primary_cells"] = int(len(primary))
        best = None
        for variant, sub in primary.groupby("variant", dropna=False):
            score = float(pd.to_numeric(sub["success_delta"], errors="coerce").mean()) if len(sub) else -999
            if best is None or score > best[1]:
                best = (variant, score, sub)
        if best is not None:
            variant, score, sub = best
            details["best_variant"] = str(variant)
            details["best_success_delta"] = float(score)
            details["best_wins"] = int((pd.to_numeric(sub["success_delta"], errors="coerce") > 0).sum())
            details["best_F4_relative_reduction"] = float(pd.to_numeric(sub["F4_relative_reduction"], errors="coerce").fillna(0).mean())
            steps_ok = float(pd.to_numeric(sub["steps_inflation"], errors="coerce").fillna(0).mean()) <= 0.10
            osc_ok = float(pd.to_numeric(sub["oscillation_score_delta"], errors="coerce").fillna(0).mean()) <= 0.05
            subgoal_ok = float(pd.to_numeric(sub["subgoal_reach_delta"], errors="coerce").fillna(0).mean()) >= 0.0
        else:
            steps_ok = osc_ok = subgoal_ok = False
        no_missing = details["missing_label_failed_rows"] == 0 and details["label_integrity_gate"] in {"PASS_LABEL_INTEGRITY", "HOLD_LABEL_INTEGRITY"}
        if details["completed_primary_cells"] < 6:
            details["gate"] = "HOLD_LOCAL_DRIFT_V2_INCOMPLETE"
        elif (
            no_missing
            and details["best_success_delta"] >= 0.02
            and details["best_wins"] >= 4
            and details["best_F4_relative_reduction"] >= 0.30
            and steps_ok
            and osc_ok
            and subgoal_ok
        ):
            details["gate"] = "PASS_LOCAL_DRIFT_V2"
        else:
            details["gate"] = "HOLD_LOCAL_DRIFT_V2_WEAK_OR_MIXED"
    out.to_csv(reports / "stage25_local_drift_v2.csv", index=False)
    return out, details


def _edge_success(edge: pd.DataFrame, env: str, seed: int, edge_types: list[str]) -> tuple[float, float]:
    sub = edge[(edge.get("env", "").astype(str) == str(env)) & (pd.to_numeric(edge.get("seed", -1), errors="coerce").fillna(-1).astype(int) == int(seed))]
    sub = sub[sub.get("edge_type", "").astype(str).isin(edge_types)]
    if len(sub) == 0:
        return np.nan, np.nan
    weights = pd.to_numeric(sub.get("edges", 1), errors="coerce").fillna(1).to_numpy(float)
    success = pd.to_numeric(sub.get("success_rate", np.nan), errors="coerce").to_numpy(float)
    set_state = pd.to_numeric(sub.get("set_state_rate", np.nan), errors="coerce").to_numpy(float)
    return float(np.average(success, weights=weights)), float(np.average(set_state, weights=weights))


def _oracle_rows(oracle_root: Path) -> pd.DataFrame:
    ranking = oracle_root / "stage25_oracle_env_ranking.csv"
    if ranking.exists():
        return _read_csv(ranking)
    bridge = _read_csv(oracle_root / "stage23_bridge_graph_summary.csv")
    edge = _read_csv(oracle_root / "stage23_edge_execution_summary.csv")
    oracle = _read_csv(oracle_root / "stage23_oracle_bridge_summary.csv")
    if len(bridge) == 0 and len(oracle) == 0:
        return pd.DataFrame(columns=ORACLE_COLUMNS)
    rows = []
    for _, row in bridge[bridge.get("graph_id", "").astype(str).ne("G0")].iterrows():
        env = row.get("env")
        seed = int(_num(row.get("seed", 0), 0))
        graph_id = row.get("graph_id", "")
        oracle_row = oracle[
            (oracle.get("env", "").astype(str) == str(env))
            & (pd.to_numeric(oracle.get("seed", -1), errors="coerce").fillna(-1).astype(int) == seed)
            & (oracle.get("graph_id", "").astype(str).eq("G_oracle"))
        ]
        if len(oracle_row) == 0:
            oracle_row = oracle[
                (oracle.get("env", "").astype(str) == str(env))
                & (pd.to_numeric(oracle.get("seed", -1), errors="coerce").fillna(-1).astype(int) == seed)
                & (oracle.get("graph_id", "").astype(str).eq(str(graph_id)))
            ]
        orow = oracle_row.iloc[0] if len(oracle_row) else pd.Series(dtype=object)
        safe, safe_set = _edge_success(edge, str(env), seed, ["safe_local"])
        risky, risky_set = _edge_success(edge, str(env), seed, ["aggressive_tdr_bridge", "bottleneck_bridge"])
        gas_cross, cross_set = _edge_success(edge, str(env), seed, ["gas_cross"])
        set_state = np.nanmax([safe_set, risky_set, cross_set]) if any(np.isfinite(x) for x in [safe_set, risky_set, cross_set]) else np.nan
        oracle_usage = _num(orow.get("bridge_usage_rate", np.nan), np.nan)
        oracle_reduction = _num(orow.get("mean_path_cost_reduction", np.nan), np.nan)
        rows.append(
            {
                "env": env,
                "seed": seed,
                "graph_id": graph_id,
                "node_count": row.get("node_count", np.nan),
                "edge_count": row.get("edge_count", np.nan),
                "bridge_count": row.get("bridge_count", np.nan),
                "shorter_path_rate": row.get("shorter_path_rate", np.nan),
                "bridge_usage_rate": row.get("bridge_usage_rate", np.nan),
                "mean_path_cost_reduction": row.get("mean_path_cost_reduction", np.nan),
                "oracle_bridge_count": orow.get("bridge_count", np.nan),
                "oracle_bridge_fraction": _num(orow.get("bridge_count", np.nan), 0) / max(_num(row.get("bridge_count", 0), 0), 1e-9),
                "oracle_shorter_path_rate": orow.get("shorter_path_rate", np.nan),
                "oracle_bridge_usage_rate": oracle_usage,
                "oracle_mean_path_cost_reduction": oracle_reduction,
                "useful_bridge_score": oracle_usage * oracle_reduction if np.isfinite(oracle_usage) and np.isfinite(oracle_reduction) else np.nan,
                "safe_local_success_rate": safe,
                "risky_bridge_success_rate": risky,
                "gas_cross_success_rate": gas_cross,
                "set_state_rate": set_state,
            }
        )
    return pd.DataFrame(rows)


def analyze_oracle(oracle_root: Path, reports: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = _oracle_rows(oracle_root)
    if len(df) == 0:
        out = pd.DataFrame(columns=ORACLE_COLUMNS)
        out.to_csv(reports / "stage25_oracle_headroom.csv", index=False)
        return out, {"gate": "HOLD_ORACLE_INCOMPLETE", "pass_pairs": []}
    rows = []
    pass_pairs = []
    for _, row in df.iterrows():
        reasons = []
        if _num(row.get("set_state_rate", np.nan), np.nan) < 0.95:
            reasons.append("set_state_rate")
        if _num(row.get("safe_local_success_rate", np.nan), np.nan) < 0.85:
            reasons.append("safe_local_success_rate")
        if _num(row.get("oracle_bridge_count", np.nan), np.nan) < 50:
            reasons.append("oracle_bridge_count")
        if _num(row.get("oracle_bridge_usage_rate", np.nan), np.nan) < 0.20:
            reasons.append("oracle_bridge_usage_rate")
        if not (
            _num(row.get("oracle_shorter_path_rate", np.nan), np.nan) >= 0.20
            or _num(row.get("oracle_mean_path_cost_reduction", np.nan), np.nan) >= 1.0
        ):
            reasons.append("oracle_path_reduction")
        if _num(row.get("useful_bridge_score", np.nan), np.nan) < 0.20:
            reasons.append("useful_bridge_score")
        gate = "PASS_ORACLE_HEADROOM" if not reasons else "NO_ORACLE_UPPER_BOUND"
        out_row = row.to_dict()
        out_row["gate"] = gate
        out_row["failure_reason"] = "|".join(reasons)
        rows.append(out_row)
        if gate == "PASS_ORACLE_HEADROOM":
            pass_pairs.append({"env": row.get("env"), "seed": int(_num(row.get("seed", 0), 0))})
    out = pd.DataFrame(rows)
    for col in ORACLE_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out = out[ORACLE_COLUMNS]
    out.to_csv(reports / "stage25_oracle_headroom.csv", index=False)
    return out, {"gate": "PASS_ORACLE_HEADROOM" if pass_pairs else "NO_ORACLE_UPPER_BOUND", "pass_pairs": pass_pairs}


def analyze_boundary(reports: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = _read_csv(reports / "stage25_boundary_coverage.csv")
    if len(df) == 0:
        df = pd.DataFrame(columns=BOUNDARY_COLUMNS)
    for col in BOUNDARY_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[BOUNDARY_COLUMNS]
    df.to_csv(reports / "stage25_boundary_coverage.csv", index=False)
    coverage = float(pd.to_numeric(df.get("coverage", pd.Series(dtype=float)), errors="coerce").fillna(0).max()) if len(df) else 0.0
    gap = float(pd.to_numeric(df.get("supported_gap", pd.Series(dtype=float)), errors="coerce").fillna(-999).max()) if len(df) else np.nan
    auroc = float(pd.to_numeric(df.get("psi_AUROC", pd.Series(dtype=float)), errors="coerce").fillna(0).max()) if len(df) else 0.0
    gate = "PASS_BOUNDARY_DIAGNOSTIC" if coverage >= 0.05 and gap >= 0.10 and auroc >= 0.65 else "HOLD_BOUNDARY_DIAGNOSTIC_ONLY"
    return df, {"gate": gate, "coverage": coverage, "supported_gap": gap, "psi_AUROC": auroc}


def ensure_d4rl_report(reports: Path) -> dict[str, Any]:
    json_path = reports / "stage25_d4rl_protocol_audit.json"
    md_path = reports / "stage25_d4rl_protocol_audit.md"
    status = _read_json(json_path)
    if not status:
        status = {"gate": "HOLD_D4RL_PROTOCOL_REPAIR", "checks": {}, "reason": "audit_not_run"}
        json_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    if not md_path.exists():
        md_path.write_text("# Stage25 D4RL Protocol Audit\n\nGate: HOLD_D4RL_PROTOCOL_REPAIR\n\nThe audit has not been run yet.\n")
    return status


def write_decisions(reports: Path, gates: dict[str, Any], summaries: dict[str, Any]) -> None:
    lines = ["# Stage25 Decisions", "", "## Gates"]
    for key in [
        "reachability_closing",
        "label_integrity",
        "local_drift_v2",
        "oracle_headroom",
        "p_bridge",
        "boundary",
        "d4rl_protocol",
        "integrated",
    ]:
        lines.append(f"- {key}: {gates.get(key, 'unknown')}")
    lines.extend(["", "## Evidence", ""])
    for name, detail in summaries.items():
        lines.append(f"- {name}: `{json.dumps(detail, sort_keys=True, default=str)}`")
    lines.extend(["", "## Decision"])
    if gates.get("integrated") == "READY_INTEGRATED_NO_FALLBACK":
        lines.append("- GO: integrated no-fallback BARS variants may be scheduled for oracle+p_bridge passing env-seed pairs.")
    else:
        lines.append("- HOLD: integrated BARS-v3 remains skipped until oracle headroom and p_bridge gates pass.")
    if gates.get("reachability_closing") == "CLOSE_REACHABILITY_MAINLINE":
        lines.append("- CLOSE: move reachability from mainline claim to appendix/diagnostic track.")
    elif gates.get("reachability_closing") == "PASS_REACHABILITY_CLOSING":
        lines.append("- GO: keep reachability-supported GAS as a narrow no-fallback claim.")
    else:
        lines.append("- HOLD: reachability closing is incomplete.")
    lines.extend(
        [
            "",
            "## Next Commands",
            "```bash",
            "python scripts/stage25_enrich_failure_atlas_all_variants.py --eval-roots runs_stage24_reachability_confirm,runs_stage24_local_drift --out reports/stage25_failure_atlas_all_variants.csv --summary-out reports/stage25_failure_atlas_summary.csv --integrity-out reports/stage25_label_integrity.json --min-episodes 100",
            "bash scripts/stage25_reachability_closing.sh ENVS=antmaze-medium-navigate-v0,antmaze-medium-stitch-v0 SEEDS=0,1,2 EPISODES=300 GPUS=${GPUS:-0,1,2,3} MAX_PARALLEL_EVAL=4 WAIT=1",
            "bash scripts/stage25_run_local_drift_v2.sh ENVS=antmaze-medium-navigate-v0,antmaze-medium-stitch-v0 SEEDS=0,1,2 EPISODES=100 GPUS=${GPUS:-0,1,2,3} MAX_PARALLEL_EVAL=4 WAIT=1",
            "bash scripts/stage25_oracle_headroom_scan_v2.sh CONFIG=configs/stage25_oracle_scan_matrix.json ENVS=scene-play-v0 SEEDS=0 GRAPH_IDS=G3 GPUS=${GPUS:-0} WAIT=1",
            "python scripts/stage25_analyze.py --reports-root reports --min-episodes 100",
            "```",
        ]
    )
    (reports / "stage25_decisions.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reports-root", default="reports")
    p.add_argument("--reachability-roots", default="runs_stage25_reachability_closing")
    p.add_argument("--local-drift-roots", default="runs_stage25_local_drift_v2")
    p.add_argument("--oracle-reports-root", default="reports/stage25_oracle_scan_tmp")
    p.add_argument("--oracle-artifact-root", default="artifacts/stage25")
    p.add_argument("--failure-atlas", default="reports/stage25_failure_atlas_all_variants.csv")
    p.add_argument("--min-episodes", type=int, default=100)
    args = p.parse_args()

    reports = Path(args.reports_root)
    reports.mkdir(parents=True, exist_ok=True)
    reach_eval = collect_eval_roots([Path(x) for x in _split(args.reachability_roots)])
    local_eval = collect_eval_roots([Path(x) for x in _split(args.local_drift_roots)])
    atlas = _read_csv(Path(args.failure_atlas))
    if len(atlas) == 0 and len(local_eval):
        atlas = local_eval
    reach, reach_summary = analyze_reachability(reach_eval, reports, args.min_episodes)
    local, local_summary = analyze_local_drift(atlas, reports, args.min_episodes)
    oracle, oracle_summary = analyze_oracle(Path(args.oracle_reports_root), reports)
    boundary, boundary_summary = analyze_boundary(reports)
    d4rl = ensure_d4rl_report(reports)
    label = _read_json(reports / "stage25_label_integrity.json")
    label_gate = label.get("gate", "HOLD_LABEL_INTEGRITY")
    p_bridge_gate = "SKIP_P_BRIDGE_UNTIL_ORACLE_HEADROOM" if not oracle_summary.get("pass_pairs") else "HOLD_P_BRIDGE"
    integrated_gate = "READY_INTEGRATED_NO_FALLBACK" if oracle_summary.get("pass_pairs") and p_bridge_gate == "PASS_P_BRIDGE" else "SKIP_INTEGRATED_UNTIL_ORACLE_AND_P_BRIDGE"
    gates = {
        "reachability_closing": reach_summary["gate"],
        "label_integrity": label_gate,
        "local_drift_v2": local_summary["gate"],
        "oracle_headroom": oracle_summary["gate"],
        "p_bridge": p_bridge_gate,
        "boundary": boundary_summary["gate"],
        "d4rl_protocol": d4rl.get("gate", "HOLD_D4RL_PROTOCOL_REPAIR"),
        "integrated": integrated_gate,
    }
    status = {
        **gates,
        "details": {
            "reachability": reach_summary,
            "label_integrity": label,
            "local_drift_v2": local_summary,
            "oracle": oracle_summary,
            "boundary": boundary_summary,
            "d4rl_protocol": d4rl,
        },
    }
    (reports / "stage25_gate_status.json").write_text(json.dumps(status, indent=2, sort_keys=True, default=str) + "\n")
    write_decisions(
        reports,
        gates,
        {
            "reachability": reach_summary,
            "label_integrity": label,
            "local_drift_v2": local_summary,
            "oracle": oracle_summary,
            "boundary": boundary_summary,
            "d4rl_protocol": d4rl,
        },
    )
    print(json.dumps(gates, sort_keys=True))


if __name__ == "__main__":
    main()
