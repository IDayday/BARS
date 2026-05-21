#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stage24_local_drift_diagnostic import OUT_COLUMNS as DRIFT_COLUMNS
from stage24_local_drift_diagnostic import build_drift_table


MEDIUM_ENVS = ["antmaze-medium-navigate-v0", "antmaze-medium-stitch-v0"]
REACHABILITY_VARIANTS = [
    "gas_shortest",
    "gas_reachability_budget_calibrated",
    "gas_reachability_soft_calibrated",
]
REPAIR_VARIANTS = [
    "gas_shortest_replan_on_local_drift",
    "gas_shortest_adaptive_subgoal_horizon",
    "gas_reachability_budget_replan_on_local_drift",
]

REACH_COLUMNS = [
    "env",
    "seed",
    "fallback_mode",
    "variant",
    "episodes",
    "success",
    "steps",
    "success_delta_vs_shortest",
    "steps_inflation_vs_shortest",
    "paired_episode_n",
    "paired_episode_success_delta",
    "no_path_rate",
    "budget_reject_rate",
    "no_path_delta_vs_shortest",
    "budget_reject_delta_vs_shortest",
    "first_plan_edges_mean",
    "first_plan_edge_distribution",
    "subgoal_reach_rate",
    "env_mean_success",
    "env_std_success",
    "primary_variant",
    "paired_cell_result",
    "source",
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
    "safe_local_success_rate",
    "risky_bridge_success_rate",
    "gas_cross_success_rate",
    "set_state_rate",
    "gate",
]

BOUNDARY_COLUMNS = [
    "env",
    "seed",
    "junction_class",
    "junction_count",
    "coverage",
    "supported_success_rate",
    "unsupported_success_rate",
    "supported_gap",
    "psi_AUROC",
]


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _write_csv(df: pd.DataFrame, path: Path, columns: list[str]) -> None:
    if len(df) == 0:
        df = pd.DataFrame(columns=columns)
    for col in columns:
        if col not in df:
            df[col] = np.nan
    path.parent.mkdir(parents=True, exist_ok=True)
    df[columns].to_csv(path, index=False)


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if len(df) == 0 or col not in df:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def _scalar(row: pd.Series | None, col: str, default: float = 0.0) -> float:
    if row is None:
        return default
    try:
        val = row.get(col, default)
        if val == "" or pd.isna(val):
            return default
        return float(val)
    except Exception:
        return default


def _collect_eval_roots(roots: list[Path]) -> pd.DataFrame:
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
            if len(df) == 0 or not {"env", "seed", "variant", "success"}.issubset(df.columns):
                continue
            df["eval_path"] = str(path)
            if "fallback_mode" not in df:
                for part in path.parts:
                    if part.startswith("fallback_"):
                        df["fallback_mode"] = part.replace("fallback_", "")
                        break
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _edge_distribution(values: pd.Series) -> str:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if len(vals) == 0:
        return ""
    bins = [
        ("0", vals.eq(0)),
        ("1", vals.eq(1)),
        ("2-3", vals.between(2, 3, inclusive="both")),
        ("4-5", vals.between(4, 5, inclusive="both")),
        ("6-10", vals.between(6, 10, inclusive="both")),
        ("11-20", vals.between(11, 20, inclusive="both")),
        (">20", vals.gt(20)),
    ]
    total = max(len(vals), 1)
    return ";".join(f"{name}:{float(mask.sum() / total):.3f}" for name, mask in bins)


def _group_eval(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame()
    out = df.copy()
    if "fallback_mode" not in out:
        out["fallback_mode"] = ""
    if "first_plan_edges" not in out:
        out["first_plan_edges"] = np.nan
    if "subgoal_reach_rate" not in out:
        out["subgoal_reach_rate"] = np.nan
    if "no_path_count" not in out:
        out["no_path_count"] = 0
    if "budget_reject_count" not in out:
        out["budget_reject_count"] = 0
    if "steps" not in out:
        out["steps"] = np.nan
    keys = [c for c in ["env", "seed", "variant", "fallback_mode"] if c in out.columns]
    grouped = out.groupby(keys, dropna=False).agg(
        episodes=("success", "count"),
        success=("success", "mean"),
        steps=("steps", "mean"),
        no_path_rate=("no_path_count", lambda x: float((pd.to_numeric(x, errors="coerce").fillna(0) > 0).mean())),
        budget_reject_rate=("budget_reject_count", lambda x: float((pd.to_numeric(x, errors="coerce").fillna(0) > 0).mean())),
        first_plan_edges_mean=("first_plan_edges", "mean"),
        first_plan_edge_distribution=("first_plan_edges", _edge_distribution),
        subgoal_reach_rate=("subgoal_reach_rate", "mean"),
    ).reset_index()
    grouped["source"] = source
    return grouped


def _stage23_reachability_prior(stage23_reports: Path) -> pd.DataFrame:
    prior = _read_csv(stage23_reports / "stage23_grouped.csv")
    if len(prior) == 0:
        return pd.DataFrame()
    prior = prior[
        prior.get("env", "").astype(str).isin(MEDIUM_ENVS)
        & prior.get("variant", "").astype(str).isin(REACHABILITY_VARIANTS)
        & prior.get("fallback_mode", "").astype(str).eq("none")
    ].copy()
    if len(prior) == 0:
        return pd.DataFrame()
    prior["source"] = "stage23_seed0_prior"
    if "first_plan_edges_mean" not in prior:
        prior["first_plan_edges_mean"] = np.nan
    if "first_plan_edge_distribution" not in prior:
        prior["first_plan_edge_distribution"] = ""
    if "subgoal_reach_rate" not in prior:
        prior["subgoal_reach_rate"] = np.nan
    return prior


def _complete_eval_cells(eval_df: pd.DataFrame, min_episodes: int) -> pd.DataFrame:
    if len(eval_df) == 0 or min_episodes <= 1:
        return eval_df
    df = eval_df.copy()
    if "fallback_mode" not in df:
        df["fallback_mode"] = ""
    keys = [c for c in ["env", "seed", "variant", "fallback_mode"] if c in df.columns]
    if not keys or "success" not in df:
        return df
    counts = df.groupby(keys, dropna=False)["success"].transform("count")
    return df[counts >= int(min_episodes)].copy()


def _paired_episode_deltas(eval_df: pd.DataFrame, min_episodes: int = 100) -> pd.DataFrame:
    if len(eval_df) == 0 or "variant" not in eval_df:
        return pd.DataFrame()
    df = _complete_eval_cells(eval_df, min_episodes)
    if "fallback_mode" not in df:
        df["fallback_mode"] = ""
    df = df[
        df.get("env", "").astype(str).isin(MEDIUM_ENVS)
        & df.get("variant", "").astype(str).isin(REACHABILITY_VARIANTS + REPAIR_VARIANTS)
        & df.get("fallback_mode", "").astype(str).eq("none")
    ].copy()
    keys = [c for c in ["env", "seed", "fallback_mode", "task_id", "episode_id"] if c in df.columns]
    if not keys or "gas_shortest" not in set(df["variant"].astype(str)):
        return pd.DataFrame()
    piv = df.pivot_table(index=keys, columns="variant", values="success", aggfunc="mean").reset_index()
    rows = []
    for (env, seed, fallback), sub in piv.groupby(["env", "seed", "fallback_mode"], dropna=False):
        if "gas_shortest" not in sub:
            continue
        for variant in [c for c in sub.columns if c not in keys and c != "gas_shortest"]:
            diff = pd.to_numeric(sub[variant], errors="coerce") - pd.to_numeric(sub["gas_shortest"], errors="coerce")
            rows.append(
                {
                    "env": env,
                    "seed": seed,
                    "fallback_mode": fallback,
                    "variant": variant,
                    "paired_episode_n": int(diff.notna().sum()),
                    "paired_episode_success_delta": float(diff.mean()) if diff.notna().any() else np.nan,
                }
            )
    return pd.DataFrame(rows)


def analyze_reachability(eval_df: pd.DataFrame, stage23_reports: Path, min_episodes: int = 100) -> tuple[pd.DataFrame, dict[str, Any]]:
    stage24 = _group_eval(eval_df, "stage24_run")
    if len(stage24) and min_episodes > 1:
        stage24 = stage24[pd.to_numeric(stage24.get("episodes", 0), errors="coerce").fillna(0) >= int(min_episodes)].copy()
    prior = _stage23_reachability_prior(stage23_reports)
    grouped = pd.concat([prior, stage24], ignore_index=True) if len(stage24) or len(prior) else pd.DataFrame()
    if len(grouped) == 0:
        return pd.DataFrame(columns=REACH_COLUMNS), {
            "gate": "HOLD_REACHABILITY_WEAK_EFFECT",
            "primary_variant": "",
            "completed_cells": 0,
        }
    grouped = grouped[
        grouped["env"].astype(str).isin(MEDIUM_ENVS)
        & grouped["variant"].astype(str).isin(REACHABILITY_VARIANTS + REPAIR_VARIANTS)
        & grouped["fallback_mode"].astype(str).eq("none")
    ].copy()
    key_cols = ["env", "seed", "variant", "fallback_mode"]
    if len(grouped):
        grouped["_source_rank"] = np.where(grouped["source"].astype(str).eq("stage24_run"), 1, 0)
        grouped = grouped.sort_values("_source_rank").drop_duplicates(key_cols, keep="last").drop(columns=["_source_rank"])
    rows = []
    reach_rows = grouped[grouped["variant"].astype(str).isin(REACHABILITY_VARIANTS)].copy()
    for (env, seed, fallback), sub in reach_rows.groupby(["env", "seed", "fallback_mode"], dropna=False):
        base_rows = sub[sub["variant"].astype(str).eq("gas_shortest")]
        base = base_rows.iloc[0] if len(base_rows) else None
        for _, row in sub.iterrows():
            delta = np.nan
            steps_infl = np.nan
            no_path_delta = np.nan
            budget_delta = np.nan
            if base is not None and row["variant"] != "gas_shortest":
                delta = _scalar(row, "success", np.nan) - _scalar(base, "success", np.nan)
                base_steps = max(_scalar(base, "steps", 0.0), 1e-6)
                steps_infl = (_scalar(row, "steps", np.nan) - _scalar(base, "steps", np.nan)) / base_steps
                no_path_delta = _scalar(row, "no_path_rate", 0.0) - _scalar(base, "no_path_rate", 0.0)
                budget_delta = _scalar(row, "budget_reject_rate", 0.0) - _scalar(base, "budget_reject_rate", 0.0)
            out = row.to_dict()
            out.update(
                {
                    "success_delta_vs_shortest": delta,
                    "steps_inflation_vs_shortest": steps_infl,
                    "no_path_delta_vs_shortest": no_path_delta,
                    "budget_reject_delta_vs_shortest": budget_delta,
                }
            )
            rows.append(out)
    out = pd.DataFrame(rows)
    if len(out) == 0:
        return pd.DataFrame(columns=REACH_COLUMNS), {
            "gate": "HOLD_REACHABILITY_WEAK_EFFECT",
            "primary_variant": "",
            "completed_cells": 0,
        }
    paired = _paired_episode_deltas(eval_df, min_episodes=min_episodes)
    if len(paired):
        out = out.merge(paired, on=["env", "seed", "fallback_mode", "variant"], how="left")
        out["success_delta_vs_shortest"] = out["paired_episode_success_delta"].where(out["paired_episode_success_delta"].notna(), out["success_delta_vs_shortest"])
    if "paired_episode_n" not in out:
        out["paired_episode_n"] = np.nan
    if "paired_episode_success_delta" not in out:
        out["paired_episode_success_delta"] = np.nan
    env_stats = (
        out.groupby(["env", "variant"], dropna=False)["success"]
        .agg(env_mean_success="mean", env_std_success="std")
        .reset_index()
    )
    out = out.merge(env_stats, on=["env", "variant"], how="left")
    candidate = out[out["variant"].astype(str).isin(REACHABILITY_VARIANTS[1:])].copy()
    primary = ""
    if len(candidate):
        means = candidate.groupby("variant")["success_delta_vs_shortest"].mean(numeric_only=True).sort_values(ascending=False)
        primary = str(means.index[0]) if len(means) else ""
    out["primary_variant"] = primary
    out["paired_cell_result"] = ""
    primary_rows = out[out["variant"].astype(str).eq(primary)].copy() if primary else pd.DataFrame()
    if len(primary_rows):
        deltas = pd.to_numeric(primary_rows["success_delta_vs_shortest"], errors="coerce")
        out.loc[deltas.index[deltas > 1e-12], "paired_cell_result"] = "win"
        out.loc[deltas.index[deltas.abs() <= 1e-12], "paired_cell_result"] = "tie"
        out.loc[deltas.index[deltas < -1e-12], "paired_cell_result"] = "loss"
    paired_primary = primary_rows[pd.to_numeric(primary_rows.get("success_delta_vs_shortest", pd.Series(dtype=float)), errors="coerce").notna()] if len(primary_rows) else pd.DataFrame()
    completed_cells = int(paired_primary[["env", "seed"]].drop_duplicates().shape[0]) if len(paired_primary) else 0
    primary_delta = float(pd.to_numeric(primary_rows.get("success_delta_vs_shortest", pd.Series(dtype=float)), errors="coerce").mean()) if len(primary_rows) else float("nan")
    primary_steps_infl = float(pd.to_numeric(primary_rows.get("steps_inflation_vs_shortest", pd.Series(dtype=float)), errors="coerce").mean()) if len(primary_rows) else float("nan")
    no_path_delta = float(pd.to_numeric(primary_rows.get("no_path_delta_vs_shortest", pd.Series(dtype=float)), errors="coerce").fillna(0).mean()) if len(primary_rows) else 0.0
    budget_delta = float(pd.to_numeric(primary_rows.get("budget_reject_delta_vs_shortest", pd.Series(dtype=float)), errors="coerce").fillna(0).mean()) if len(primary_rows) else 0.0
    wins = int((pd.to_numeric(primary_rows.get("success_delta_vs_shortest", pd.Series(dtype=float)), errors="coerce") > 1e-12).sum()) if len(primary_rows) else 0
    ties = int((pd.to_numeric(primary_rows.get("success_delta_vs_shortest", pd.Series(dtype=float)), errors="coerce").abs() <= 1e-12).sum()) if len(primary_rows) else 0
    losses = int((pd.to_numeric(primary_rows.get("success_delta_vs_shortest", pd.Series(dtype=float)), errors="coerce") < -1e-12).sum()) if len(primary_rows) else 0
    regression = (
        (np.isfinite(primary_delta) and primary_delta < -0.005)
        or (np.isfinite(primary_steps_infl) and primary_steps_infl > 0.10)
        or no_path_delta > 0.02
        or budget_delta > 0.02
    )
    passes = (
        completed_cells >= 6
        and np.isfinite(primary_delta)
        and primary_delta >= 0.02
        and wins >= 4
        and (not np.isfinite(primary_steps_infl) or primary_steps_infl <= 0.10)
        and no_path_delta <= 0.02
        and budget_delta <= 0.02
    )
    if passes:
        gate = "PASS_REACHABILITY_CONFIRM"
    elif regression and completed_cells >= 6:
        gate = "FAIL_REACHABILITY_REGRESSION"
    else:
        gate = "HOLD_REACHABILITY_WEAK_EFFECT"
    summary = {
        "gate": gate,
        "primary_variant": primary,
        "completed_cells": completed_cells,
        "mean_delta": primary_delta,
        "mean_steps_inflation": primary_steps_infl,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "no_path_delta": no_path_delta,
        "budget_reject_delta": budget_delta,
    }
    return out, summary


def analyze_local_drift(
    eval_df: pd.DataFrame,
    stage23_reports: Path,
    reports_root: Path,
    local_drift_roots: list[Path],
    min_episodes: int = 100,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    evals = eval_df
    if len(evals) == 0:
        evals = _collect_eval_roots(local_drift_roots)
    evals_for_gate = _complete_eval_cells(evals, min_episodes)
    atlas = _read_csv(stage23_reports / "stage23_failure_atlas.csv")
    drift = build_drift_table(evals, atlas)
    if len(drift) == 0:
        drift = _read_csv(reports_root / "stage24_local_drift.csv")
    drift_for_gate = build_drift_table(evals_for_gate, atlas) if len(evals_for_gate) else pd.DataFrame(columns=DRIFT_COLUMNS)
    repair_rows = drift_for_gate[drift_for_gate.get("variant", pd.Series(dtype=str)).astype(str).isin(REPAIR_VARIANTS)] if len(drift_for_gate) else pd.DataFrame()
    if len(drift_for_gate) == 0 or len(repair_rows) == 0:
        return drift, {"gate": "HOLD_LOCAL_DRIFT_REPAIR", "best_variant": "", "best_reduction": np.nan}
    baseline_map = {
        "gas_shortest_replan_on_local_drift": "gas_shortest",
        "gas_shortest_adaptive_subgoal_horizon": "gas_shortest",
        "gas_reachability_budget_replan_on_local_drift": "gas_reachability_budget_calibrated",
    }
    best = {"variant": "", "reduction": -np.inf, "success_delta": -np.inf}
    for repair_variant, baseline_variant in baseline_map.items():
        rep = drift_for_gate[drift_for_gate["variant"].astype(str).eq(repair_variant)]
        base = drift_for_gate[drift_for_gate["variant"].astype(str).eq(baseline_variant)]
        if len(rep) == 0 or len(base) == 0:
            continue
        keys = [c for c in ["env", "seed"] if c in drift_for_gate.columns]
        for key_vals, rep_sub in rep.groupby(keys, dropna=False):
            if not isinstance(key_vals, tuple):
                key_vals = (key_vals,)
            mask = pd.Series(True, index=base.index)
            for c, v in zip(keys, key_vals):
                mask &= base[c].astype(str).eq(str(v))
            base_sub = base[mask]
            if len(base_sub) == 0:
                continue
            base_f4 = float(base_sub["primary_failure_type"].astype(str).eq("F4_LOCAL_EXECUTION_DRIFT").mean())
            rep_f4 = float(rep_sub["primary_failure_type"].astype(str).eq("F4_LOCAL_EXECUTION_DRIFT").mean())
            if base_f4 <= 0:
                continue
            reduction = (base_f4 - rep_f4) / base_f4
            success_delta = float(rep_sub["success"].mean() - base_sub["success"].mean())
            if reduction > best["reduction"]:
                best = {"variant": repair_variant, "reduction": reduction, "success_delta": success_delta}
    gate = "PASS_LOCAL_DRIFT_REPAIR" if best["reduction"] >= 0.30 and best["success_delta"] >= -1e-12 else "HOLD_LOCAL_DRIFT_REPAIR"
    return drift, {"gate": gate, "best_variant": best["variant"], "best_reduction": best["reduction"], "success_delta": best["success_delta"]}


def _weighted_rate(df: pd.DataFrame, edge_types: set[str]) -> float:
    if len(df) == 0 or "edge_type" not in df:
        return float("nan")
    sub = df[df["edge_type"].astype(str).isin(edge_types)]
    if len(sub) == 0:
        return float("nan")
    weights = pd.to_numeric(sub.get("edges", 1), errors="coerce").fillna(1.0)
    rates = pd.to_numeric(sub.get("success_rate", 0), errors="coerce").fillna(0.0)
    return float(np.average(rates, weights=weights))


def _weighted_set_state(df: pd.DataFrame) -> float:
    if len(df) == 0 or "set_state_rate" not in df:
        return float("nan")
    weights = pd.to_numeric(df.get("edges", 1), errors="coerce").fillna(1.0)
    rates = pd.to_numeric(df["set_state_rate"], errors="coerce").fillna(0.0)
    return float(np.average(rates, weights=weights))


def analyze_oracle(oracle_reports_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    bridge = _read_csv(oracle_reports_root / "stage23_bridge_graph_summary.csv")
    oracle = _read_csv(oracle_reports_root / "stage23_oracle_bridge_summary.csv")
    edge = _read_csv(oracle_reports_root / "stage23_edge_execution_summary.csv")
    source = oracle if len(oracle) else bridge
    if len(source) == 0:
        return pd.DataFrame(columns=ORACLE_COLUMNS), {"gate": "PENDING_ORACLE_HEADROOM_SCAN", "pass_pairs": []}
    rows = []
    pairs = source[["env", "seed"]].drop_duplicates() if {"env", "seed"}.issubset(source.columns) else pd.DataFrame()
    for _, pair in pairs.iterrows():
        env = pair["env"]
        seed = pair["seed"]
        g3_sub = source[(source["env"].astype(str).eq(str(env))) & (source["seed"].astype(str).eq(str(seed))) & (source["graph_id"].astype(str).eq("G3"))]
        if len(g3_sub) == 0 and len(bridge):
            g3_sub = bridge[(bridge["env"].astype(str).eq(str(env))) & (bridge["seed"].astype(str).eq(str(seed))) & (bridge["graph_id"].astype(str).eq("G3"))]
        oracle_sub = oracle[(oracle["env"].astype(str).eq(str(env))) & (oracle["seed"].astype(str).eq(str(seed))) & (oracle["graph_id"].astype(str).eq("G_oracle"))] if len(oracle) else pd.DataFrame()
        g3 = g3_sub.iloc[0] if len(g3_sub) else None
        go = oracle_sub.iloc[0] if len(oracle_sub) else None
        edge_sub = edge[(edge.get("env", "").astype(str).eq(str(env))) & (edge.get("seed", "").astype(str).eq(str(seed)))] if len(edge) else pd.DataFrame()
        safe_rate = _weighted_rate(edge_sub, {"safe_local", "same_traj_temporal"})
        risky_rate = _weighted_rate(edge_sub, {"aggressive_tdr_bridge", "bottleneck_bridge"})
        gas_cross_rate = _weighted_rate(edge_sub, {"gas_cross"})
        set_state_rate = _weighted_set_state(edge_sub)
        bridge_count = int(_scalar(g3, "bridge_count", 0.0))
        oracle_bridge_count = int(_scalar(go, "bridge_count", 0.0))
        oracle_shorter = _scalar(go, "shorter_path_rate", 0.0)
        oracle_usage = _scalar(go, "bridge_usage_rate", 0.0)
        oracle_reduction = _scalar(go, "mean_path_cost_reduction", 0.0)
        pass_gate = (
            set_state_rate >= 0.95
            and safe_rate >= 0.85
            and oracle_bridge_count >= 50
            and (oracle_shorter >= 0.20 or oracle_reduction >= 1.0)
            and oracle_usage >= 0.20
        )
        if pass_gate:
            gate = "PASS_ORACLE_HEADROOM"
        elif g3 is not None or go is not None:
            gate = "NO_ORACLE_UPPER_BOUND"
        else:
            gate = "PENDING_ORACLE_HEADROOM_SCAN"
        rows.append(
            {
                "env": env,
                "seed": seed,
                "graph_id": "G3",
                "node_count": int(_scalar(g3, "node_count", 0.0)),
                "edge_count": int(_scalar(g3, "edge_count", 0.0)),
                "bridge_count": bridge_count,
                "shorter_path_rate": _scalar(g3, "shorter_path_rate", 0.0),
                "bridge_usage_rate": _scalar(g3, "bridge_usage_rate", 0.0),
                "mean_path_cost_reduction": _scalar(g3, "mean_path_cost_reduction", 0.0),
                "oracle_bridge_count": oracle_bridge_count,
                "oracle_bridge_fraction": oracle_bridge_count / max(bridge_count, 1),
                "oracle_shorter_path_rate": oracle_shorter,
                "oracle_bridge_usage_rate": oracle_usage,
                "oracle_mean_path_cost_reduction": oracle_reduction,
                "safe_local_success_rate": safe_rate,
                "risky_bridge_success_rate": risky_rate,
                "gas_cross_success_rate": gas_cross_rate,
                "set_state_rate": set_state_rate,
                "gate": gate,
            }
        )
    out = pd.DataFrame(rows)
    pass_pairs = out.loc[out["gate"].astype(str).eq("PASS_ORACLE_HEADROOM"), ["env", "seed"]].to_dict("records") if len(out) else []
    gate = "PASS_ORACLE_HEADROOM" if pass_pairs else ("NO_ORACLE_UPPER_BOUND" if len(out) and (out["gate"] == "NO_ORACLE_UPPER_BOUND").all() else "PENDING_ORACLE_HEADROOM_SCAN")
    return out, {"gate": gate, "pass_pairs": pass_pairs}


def analyze_p_bridge(reports_root: Path, oracle_summary: dict[str, Any]) -> dict[str, Any]:
    if not oracle_summary.get("pass_pairs"):
        return {"gate": "SKIP_P_BRIDGE_UNTIL_ORACLE_HEADROOM"}
    pbridge = _read_csv(reports_root / "stage24_p_bridge_metrics.csv")
    if len(pbridge) == 0:
        return {"gate": "HOLD_P_BRIDGE_WEAK_FP_REDUCTION"}
    best_gate = "HOLD_P_BRIDGE_WEAK_FP_REDUCTION"
    for _, row in pbridge.iterrows():
        auroc = _scalar(row, "selected_bridge_AUROC", 0.0)
        auprc = _scalar(row, "selected_bridge_AUPRC", 0.0)
        base = _scalar(row, "selected_bridge_base_success_rate", 0.0)
        fp = _scalar(row, "false_positive_bridge_relative_reduction@0.6", 0.0)
        coverage = _scalar(row, "accepted_bridge_coverage@0.6", 0.0)
        accepted = _scalar(row, "accepted_bridge_success_rate@0.6", 0.0)
        if auroc >= 0.70 and auprc >= base + 0.05 and fp >= 0.20 and coverage >= 0.30 and accepted >= base + 0.10:
            best_gate = "PASS_P_BRIDGE"
            break
    return {"gate": best_gate}


def analyze_boundary(reports_root: Path, stage23_reports: Path, artifact_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    for root in [artifact_root, Path("artifacts/stage23")]:
        if not root.exists():
            continue
        for path in root.rglob("boundary_junction/bridge_boundary_junctions.csv"):
            junctions = _read_csv(path)
            metrics = _read_csv(path.parent / "bridge_boundary_metrics.csv")
            if len(junctions) == 0:
                continue
            parts = path.parts
            try:
                seed = [p for p in parts if p.startswith("seed")][-1].replace("seed", "")
                env = parts[parts.index(f"seed{seed}") - 1]
            except Exception:
                env, seed = "", ""
            for klass, sub in _split_junction_classes(junctions).items():
                base = metrics.iloc[0] if len(metrics) else None
                rows.append(
                    {
                        "env": env,
                        "seed": seed,
                        "junction_class": klass,
                        "junction_count": int(len(sub)),
                        "coverage": _scalar(base, "coverage", 0.0),
                        "supported_success_rate": _scalar(base, "supported_success_rate", np.nan),
                        "unsupported_success_rate": _scalar(base, "unsupported_success_rate", np.nan),
                        "supported_gap": _scalar(base, "supported_gap", np.nan),
                        "psi_AUROC": _scalar(base, "psi_AUROC_for_conditional_success", np.nan),
                    }
                )
    if not rows:
        prior = _read_csv(stage23_reports / "stage23_boundary_junction_metrics.csv")
        for _, row in prior.iterrows():
            rows.append(
                {
                    "env": row.get("env", ""),
                    "seed": row.get("seed", ""),
                    "junction_class": "bridge_junction_prior_unseparated",
                    "junction_count": row.get("junction_count", 0),
                    "coverage": row.get("coverage", 0),
                    "supported_success_rate": row.get("supported_success_rate", np.nan),
                    "unsupported_success_rate": row.get("unsupported_success_rate", np.nan),
                    "supported_gap": row.get("supported_gap", np.nan),
                    "psi_AUROC": row.get("psi_AUROC_for_conditional_success", np.nan),
                }
            )
    out = pd.DataFrame(rows)
    if len(out) == 0:
        return pd.DataFrame(columns=BOUNDARY_COLUMNS), {"gate": "HOLD_BOUNDARY_DIAGNOSTIC_ONLY"}
    coverage = float(pd.to_numeric(out.get("coverage", pd.Series(dtype=float)), errors="coerce").fillna(0).max())
    gap = float(pd.to_numeric(out.get("supported_gap", pd.Series(dtype=float)), errors="coerce").fillna(-np.inf).max())
    auroc = float(pd.to_numeric(out.get("psi_AUROC", pd.Series(dtype=float)), errors="coerce").fillna(0).max())
    gate = "PASS_BOUNDARY_REENTRY" if coverage >= 0.05 and gap >= 0.10 and auroc >= 0.65 else "HOLD_BOUNDARY_DIAGNOSTIC_ONLY"
    return out, {"gate": gate, "coverage": coverage, "supported_gap": gap, "psi_AUROC": auroc}


def _split_junction_classes(junctions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if len(junctions) == 0:
        return {}
    prev_type = junctions.get("prev_edge_type", pd.Series([""] * len(junctions))).astype(str)
    next_type = junctions.get("next_edge_type", pd.Series([""] * len(junctions))).astype(str)
    virtual = prev_type.eq("virtual_connector") | next_type.eq("virtual_connector")
    risky_prev = prev_type.isin(["aggressive_tdr_bridge", "bottleneck_bridge", "gas_cross"])
    risky_next = next_type.isin(["aggressive_tdr_bridge", "bottleneck_bridge", "gas_cross"])
    local_prev = prev_type.isin(["safe_local", "same_traj_temporal"])
    local_next = next_type.isin(["safe_local", "same_traj_temporal"])
    return {
        "virtual_start_goal_connector_pairs": junctions[virtual].copy(),
        "local_local_edge_junctions": junctions[local_prev & local_next].copy(),
        "bridge_entry_exit_junctions": junctions[(risky_prev ^ risky_next)].copy(),
        "bridge_bridge_junctions": junctions[(risky_prev & risky_next)].copy(),
    }


def write_d4rl_report(path: Path) -> str:
    lines = [
        "# Stage24 D4RL Protocol Report",
        "",
        "| Question | Stage24 status |",
        "| --- | --- |",
        "| Are start/goal definitions aligned with official D4RL AntMaze evaluation? | PENDING: no Stage24 D4RL protocol audit evidence yet. |",
        "| Is success read from the same source/threshold as official baselines? | PENDING: success source and threshold still require an explicit adapter audit. |",
        "| Are max episode steps, reset behavior, and goal sampling correct? | PENDING: do not compare D4RL scores until this is verified. |",
        "| Is the low-level policy trained/evaluated with the expected D4RL observation and goal format? | PENDING: observation/goal adapter compatibility remains unproven. |",
        "",
        "Decision: HOLD_D4RL_PROTOCOL_REPAIR. Low D4RL scores must remain diagnostic until these checks are answered.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return "HOLD_D4RL_PROTOCOL_REPAIR"


def _table(df: pd.DataFrame, cols: list[str], max_rows: int = 8) -> str:
    if len(df) == 0:
        return "_No rows yet._"
    keep = [c for c in cols if c in df.columns]
    view = df[keep].head(max_rows)
    try:
        return view.to_markdown(index=False)
    except Exception:
        return "```csv\n" + view.to_csv(index=False).strip() + "\n```"


def write_decisions(
    path: Path,
    gates: dict[str, Any],
    reach: pd.DataFrame,
    oracle: pd.DataFrame,
    drift: pd.DataFrame,
) -> None:
    lines = [
        "# Stage24 Decisions",
        "",
        "## Gates",
        f"- reachability_confirm: {gates['reachability_confirm']}",
        f"- local_drift_repair: {gates['local_drift_repair']}",
        f"- oracle_headroom: {gates['oracle_headroom']}",
        f"- p_bridge: {gates['p_bridge']}",
        f"- boundary: {gates['boundary']}",
        f"- integrated: {gates['integrated']}",
        f"- d4rl_protocol: {gates['d4rl_protocol']}",
        "",
        "## Evidence",
        "",
        "Reachability confirmation:",
        _table(
            reach[reach.get("variant", pd.Series(dtype=str)).astype(str).isin(REACHABILITY_VARIANTS)] if len(reach) else reach,
            ["env", "seed", "variant", "success", "steps", "success_delta_vs_shortest", "steps_inflation_vs_shortest", "source"],
        ),
        "",
        "Oracle headroom:",
        _table(
            oracle,
            ["env", "seed", "bridge_count", "shorter_path_rate", "oracle_bridge_count", "oracle_shorter_path_rate", "oracle_mean_path_cost_reduction", "safe_local_success_rate", "set_state_rate", "gate"],
        ),
        "",
        "Local drift:",
        _table(
            drift,
            ["env", "seed", "variant", "success", "primary_failure_type", "local_drift_score", "progress_stall_count", "oscillation_score"],
        ),
        "",
        "## Decision",
    ]
    if gates["oracle_headroom"] != "PASS_ORACLE_HEADROOM":
        lines.append("- HOLD integrated BARS-v3: oracle headroom has not passed for any Stage24 env/seed.")
    if gates["reachability_confirm"] == "PASS_REACHABILITY_CONFIRM":
        lines.append("- GO narrow reachability-confirmed route for no-fallback medium OGBench.")
    else:
        lines.append("- HOLD reachability claim until seeds 1/2 complete and paired gates pass.")
    if gates["local_drift_repair"] == "PASS_LOCAL_DRIFT_REPAIR":
        lines.append("- GO local-drift repair route for medium execution failures.")
    else:
        lines.append("- HOLD local-drift repair claim until a repair cuts F4 failures by at least 30% without success loss.")
    lines.extend(
        [
            "- STOP using progress_stall_v3/direct-goal fallback as planner evidence.",
            "- KEEP boundary diagnostic-only unless coverage, supported-gap, and psi gates pass.",
            "",
            "## Next commands",
            "```bash",
            "bash scripts/stage24_run_reachability_confirm.sh CONFIG=configs/stage24_reachability_confirm.json ENVS=antmaze-medium-navigate-v0,antmaze-medium-stitch-v0 SEEDS=0,1,2 EPISODES=100 GPUS=${GPUS:-0,1,2,3} MAX_PARALLEL_EVAL=4 WAIT=1",
            "bash scripts/stage24_run_reachability_confirm.sh CONFIG=configs/stage24_reachability_confirm.json ENVS=antmaze-medium-navigate-v0,antmaze-medium-stitch-v0 SEEDS=0,1,2 EPISODES=100 VARIANTS=gas_shortest_replan_on_local_drift,gas_shortest_adaptive_subgoal_horizon,gas_reachability_budget_replan_on_local_drift STAGE24_ROOT=runs_stage24_local_drift LOG_ROOT=runs_stage24_local_drift_logs GPUS=${GPUS:-0,1,2,3} MAX_PARALLEL_EVAL=4 WAIT=1",
            "bash scripts/stage24_oracle_headroom_scan.sh ENVS=scene-play-v0 SEEDS=0 GPUS=${GPUS:-0} EDGE_EXEC_PILOT=1 TOP_K_BRIDGE=4 MAX_SOURCES=200 WAIT=1",
            "python scripts/stage24_local_drift_diagnostic.py --eval-roots runs_stage24_reachability_confirm,runs_stage24_local_drift --out reports/stage24_local_drift.csv",
            "python scripts/stage24_analyze.py",
            "```",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if np.isfinite(v) else None
    return value


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reports-root", default="reports")
    p.add_argument("--stage23-reports-root", default="reports")
    p.add_argument("--reachability-roots", default="runs_stage24_reachability_confirm")
    p.add_argument("--local-drift-roots", default="runs_stage24_reachability_confirm,runs_stage24_local_drift")
    p.add_argument("--oracle-reports-root", default="reports/stage24_oracle_scan_tmp")
    p.add_argument("--oracle-artifact-root", default="artifacts/stage24")
    p.add_argument("--min-episodes", type=int, default=100)
    args = p.parse_args()

    reports = Path(args.reports_root)
    reports.mkdir(parents=True, exist_ok=True)
    stage23_reports = Path(args.stage23_reports_root)
    reach_eval = _collect_eval_roots([Path(x) for x in args.reachability_roots.split(",") if x])
    local_eval = _collect_eval_roots([Path(x) for x in args.local_drift_roots.split(",") if x])

    reach, reach_summary = analyze_reachability(reach_eval, stage23_reports, min_episodes=args.min_episodes)
    drift, drift_summary = analyze_local_drift(
        local_eval,
        stage23_reports,
        reports,
        [Path(x) for x in args.local_drift_roots.split(",") if x],
        min_episodes=args.min_episodes,
    )
    oracle, oracle_summary = analyze_oracle(Path(args.oracle_reports_root))
    pbridge_summary = analyze_p_bridge(reports, oracle_summary)
    boundary, boundary_summary = analyze_boundary(reports, stage23_reports, Path(args.oracle_artifact_root))
    d4rl_gate = write_d4rl_report(reports / "stage24_d4rl_protocol_report.md")

    _write_csv(reach, reports / "stage24_reachability_confirm.csv", REACH_COLUMNS)
    _write_csv(oracle, reports / "stage24_oracle_headroom.csv", ORACLE_COLUMNS)
    _write_csv(drift, reports / "stage24_local_drift.csv", DRIFT_COLUMNS)
    _write_csv(boundary, reports / "stage24_boundary_junction_metrics.csv", BOUNDARY_COLUMNS)

    integrated_gate = (
        "READY_INTEGRATED_NO_FALLBACK"
        if oracle_summary["gate"] == "PASS_ORACLE_HEADROOM" and pbridge_summary["gate"] == "PASS_P_BRIDGE"
        else "SKIP_INTEGRATED_UNTIL_ORACLE_AND_P_BRIDGE"
    )
    gates = {
        "reachability_confirm": reach_summary["gate"],
        "local_drift_repair": drift_summary["gate"],
        "oracle_headroom": oracle_summary["gate"],
        "p_bridge": pbridge_summary["gate"],
        "boundary": boundary_summary["gate"],
        "integrated": integrated_gate,
        "d4rl_protocol": d4rl_gate,
        "details": {
            "reachability": reach_summary,
            "local_drift": drift_summary,
            "oracle": oracle_summary,
            "boundary": boundary_summary,
        },
    }
    (reports / "stage24_gate_status.json").write_text(json.dumps(_json_safe(gates), indent=2, sort_keys=True, allow_nan=False) + "\n")
    write_decisions(reports / "stage24_decisions.md", gates, reach, oracle, drift)


if __name__ == "__main__":
    main()
