from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=str)


def edge_score_summary(edge_scores: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {"num_edges": int(len(edge_scores))}
    for col in ["p_exec", "r_exec", "temporal_cost", "phi_dist"]:
        if col in edge_scores and len(edge_scores):
            x = edge_scores[col].to_numpy(np.float32)
            out[f"{col}_mean"] = float(np.mean(x))
            for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
                out[f"{col}_q{int(q * 100):02d}"] = float(np.quantile(x, q))
    return out


def path_bucket_summary(eval_df: pd.DataFrame) -> pd.DataFrame:
    if len(eval_df) == 0:
        return pd.DataFrame()
    df = eval_df.copy()
    df["first_plan_edges_bucket"] = pd.cut(
        pd.to_numeric(df.get("first_plan_edges", 0), errors="coerce").fillna(0),
        bins=[-1, 0, 1, 3, 5, 10, 20, 999],
        labels=["0", "1", "2-3", "4-5", "6-10", "11-20", ">20"],
    )
    return (
        df.groupby(["variant", "budget", "fallback_mode" if "fallback_mode" in df else "fallback_used", "first_plan_edges_bucket"], dropna=False)
        .agg(episodes=("success", "count"), success_mean=("success", "mean"), steps_mean=("steps", "mean"))
        .reset_index()
    )


def success_by_first_plan_edges(eval_df: pd.DataFrame) -> pd.DataFrame:
    return path_bucket_summary(eval_df)


def success_by_budget(eval_df: pd.DataFrame) -> pd.DataFrame:
    if len(eval_df) == 0:
        return pd.DataFrame()
    keys = [k for k in ["env", "seed", "variant", "budget", "fallback_mode"] if k in eval_df.columns]
    return eval_df.groupby(keys, dropna=False).agg(episodes=("success", "count"), success=("success", "mean"), steps=("steps", "mean")).reset_index()


def fallback_success_vs_no_fallback(eval_df: pd.DataFrame) -> pd.DataFrame:
    if len(eval_df) == 0 or "fallback_used" not in eval_df:
        return pd.DataFrame()
    keys = [k for k in ["env", "seed", "variant", "budget"] if k in eval_df.columns] + ["fallback_used"]
    return eval_df.groupby(keys, dropna=False).agg(episodes=("success", "count"), success=("success", "mean"), steps=("steps", "mean")).reset_index()


def compare_variants(eval_df: pd.DataFrame) -> pd.DataFrame:
    if len(eval_df) == 0:
        return pd.DataFrame()
    keys = [k for k in ["env", "seed", "task_id", "episode_id", "budget", "fallback_mode"] if k in eval_df.columns]
    if not keys:
        return pd.DataFrame()
    piv = eval_df.pivot_table(index=keys, columns="variant", values="success", aggfunc="mean").reset_index()
    variants = [c for c in piv.columns if c not in keys]
    rows = []
    if "gas_shortest" in variants:
        for v in variants:
            if v == "gas_shortest":
                continue
            diff = piv[v] - piv["gas_shortest"]
            rows.append(
                {
                    "variant": v,
                    "baseline": "gas_shortest",
                    "paired_n": int(diff.notna().sum()),
                    "success_delta": float(diff.mean()) if diff.notna().any() else float("nan"),
                    "baseline_success": float(piv["gas_shortest"].mean()),
                    "variant_success": float(piv[v].mean()),
                }
            )
    return pd.DataFrame(rows)


def summarize_eval(run_dir: str | os.PathLike[str]) -> dict[str, Any]:
    run = Path(run_dir)
    csv_path = run / "eval.csv"
    if not csv_path.exists():
        summary = {"completed": False, "reason": "missing eval.csv"}
        _write_json(run / "summary.json", summary)
        return summary
    df = pd.read_csv(csv_path)
    if "fallback_mode" not in df:
        # Infer it from the run directory when possible.
        part = run.name
        df["fallback_mode"] = part.replace("fallback_", "") if part.startswith("fallback_") else ""
    summary: dict[str, Any] = {
        "completed": True,
        "episodes": int(len(df)),
        "success_mean": float(df["success"].mean()) if len(df) else 0.0,
        "steps_mean": float(df["steps"].mean()) if len(df) else 0.0,
        "fallback_used_rate": float(df.get("fallback_used", pd.Series([0] * len(df))).mean()) if len(df) else 0.0,
        "no_path_rate": float((df.get("no_path_count", pd.Series([0] * len(df))) > 0).mean()) if len(df) else 0.0,
        "budget_reject_rate": float((df.get("budget_reject_count", pd.Series([0] * len(df))) > 0).mean()) if len(df) else 0.0,
    }
    _write_json(run / "summary.json", summary)
    grouped = success_by_budget(df)
    if len(grouped):
        grouped.to_csv(run / "grouped.csv", index=False)
    paths = path_bucket_summary(df)
    if len(paths):
        paths.to_csv(run / "path_buckets.csv", index=False)
    report = [
        "# Stage22 Run Summary",
        "",
        f"- Episodes: {summary['episodes']}",
        f"- Success: {summary['success_mean']:.3f}",
        f"- Mean steps: {summary['steps_mean']:.1f}",
        f"- Fallback used: {summary['fallback_used_rate']:.3f}",
        f"- No-path rate: {summary['no_path_rate']:.3f}",
        f"- Budget reject rate: {summary['budget_reject_rate']:.3f}",
        "",
    ]
    (run / "summary.md").write_text("\n".join(report))
    return summary
