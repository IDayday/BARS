#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if len(df) == 0:
        return ""
    view = df.head(max_rows) if max_rows else df
    try:
        return view.to_markdown(index=False)
    except ImportError:
        return "```csv\n" + view.to_csv(index=False).strip() + "\n```"


def read_all_eval(root: Path) -> pd.DataFrame:
    frames = []
    for path in root.rglob("eval.csv"):
        try:
            df = pd.read_csv(path)
            if len(df):
                parts = path.parts
                df["eval_path"] = str(path)
                if "fallback_mode" not in df:
                    df["fallback_mode"] = path.parent.name.replace("fallback_", "")
                frames.append(df)
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def read_jsons(root: Path, name: str) -> list[dict[str, Any]]:
    rows = []
    for path in root.rglob(name):
        try:
            row = json.loads(path.read_text())
            row["path"] = str(path)
            rows.append(row)
        except Exception:
            pass
    return rows


def bucketize(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame()
    out = df.copy()
    out["pred_bucket"] = pd.cut(pd.to_numeric(out.get("first_plan_pred_success", 0), errors="coerce").fillna(0), bins=[-0.01, 0.1, 0.25, 0.5, 0.75, 1.0], labels=["0-0.1", "0.1-0.25", "0.25-0.5", "0.5-0.75", "0.75-1"])
    out["exec_risk_bucket"] = pd.cut(pd.to_numeric(out.get("first_plan_exec_risk", 0), errors="coerce").fillna(0), bins=[-0.01, 0.5, 1, 2, 3, 5, 999], labels=["0-0.5", "0.5-1", "1-2", "2-3", "3-5", ">5"])
    out["first_plan_edges_bucket"] = pd.cut(pd.to_numeric(out.get("first_plan_edges", 0), errors="coerce").fillna(0), bins=[-1, 0, 1, 3, 5, 10, 20, 999], labels=["0", "1", "2-3", "4-5", "6-10", "11-20", ">20"])
    keys = [k for k in ["env", "variant", "budget", "fallback_mode", "pred_bucket", "exec_risk_bucket", "first_plan_edges_bucket"] if k in out.columns]
    grouped = out.groupby(keys, dropna=False, observed=True).agg(
        episodes=("success", "count"),
        success=("success", "mean"),
        no_path_rate=("no_path_count", lambda x: float((x > 0).mean())),
    ).reset_index()
    return grouped[grouped["episodes"] > 0].reset_index(drop=True)


def compare_variants(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame()
    keys = [k for k in ["env", "seed", "task_id", "episode_id", "budget", "fallback_mode"] if k in df.columns]
    piv = df.pivot_table(index=keys, columns="variant", values="success", aggfunc="mean").reset_index()
    if "gas_shortest" not in piv:
        return pd.DataFrame()
    rows = []
    for col in [c for c in piv.columns if c not in keys and c != "gas_shortest"]:
        diff = piv[col] - piv["gas_shortest"]
        mask = diff.notna()
        rows.append({
            "variant": col,
            "baseline": "gas_shortest",
            "paired_n": int(mask.sum()),
            "success_delta": float(diff[mask].mean()) if mask.any() else np.nan,
            "baseline_success": float(piv.loc[mask, "gas_shortest"].mean()) if mask.any() else np.nan,
            "variant_success": float(piv.loc[mask, col].mean()) if mask.any() else np.nan,
        })
    return pd.DataFrame(rows)


def decision_summary(grouped: pd.DataFrame, comparison: pd.DataFrame, edge_diag: pd.DataFrame, boundary_diag: pd.DataFrame, df: pd.DataFrame) -> list[str]:
    decisions = []
    if len(comparison):
        medium = comparison[comparison["success_delta"] >= 0.05]
        if len(medium) >= 2:
            decisions.append("GO_SAME_BACKBONE")
    if "gas_boundary_budget" in set(comparison.get("variant", [])) and "gas_reachability_budget" in set(comparison.get("variant", [])):
        b = comparison.loc[comparison["variant"] == "gas_boundary_budget", "success_delta"].mean()
        r = comparison.loc[comparison["variant"] == "gas_reachability_budget", "success_delta"].mean()
        if pd.notna(b) and pd.notna(r) and b <= r:
            decisions.append("HOLD_BOUNDARY")
    if len(grouped) and "gas_shortest" in set(grouped.get("variant", [])):
        base = grouped[grouped["variant"] == "gas_shortest"]["success"].mean()
        if pd.notna(base) and base < 0.05:
            decisions.append("REPAIR_BACKBONE")
    if len(df) and "fallback_used" in df:
        fb = df["fallback_used"].mean()
        if fb == 0:
            decisions.append("REPAIR_FALLBACK")
    if len(boundary_diag) and "supported_pair_rate" in boundary_diag:
        cov = boundary_diag["supported_pair_rate"].mean()
        if pd.notna(cov) and cov < 0.2:
            decisions.append("BOUNDARY_DIAGNOSTIC_ONLY")
    return decisions or ["INSUFFICIENT_EVIDENCE"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-root", default="runs_stage22_eval")
    p.add_argument("--artifact-root", default="artifacts/stage22")
    p.add_argument("--out", default="reports")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = read_all_eval(Path(args.eval_root))
    if len(df) == 0:
        (out / "stage22_summary.md").write_text("# Stage22 Summary\n\nNo eval rows found.\n")
        pd.DataFrame().to_csv(out / "stage22_results.csv", index=False)
        pd.DataFrame().to_csv(out / "stage22_grouped.csv", index=False)
        pd.DataFrame().to_csv(out / "stage22_variant_comparison.csv", index=False)
        pd.DataFrame().to_csv(out / "stage22_edge_path_diagnostics.csv", index=False)
        return
    df.to_csv(out / "stage22_results.csv", index=False)
    keys = [k for k in ["env", "seed", "variant", "budget", "fallback_mode"] if k in df.columns]
    grouped = df.groupby(keys, dropna=False).agg(
        episodes=("success", "count"),
        success=("success", "mean"),
        steps=("steps", "mean"),
        fallback_used=("fallback_used", "mean"),
        no_path_rate=("no_path_count", lambda x: float((x > 0).mean())),
        budget_reject_rate=("budget_reject_count", lambda x: float((x > 0).mean())),
    ).reset_index()
    grouped.to_csv(out / "stage22_grouped.csv", index=False)
    comp = compare_variants(df)
    comp.to_csv(out / "stage22_variant_comparison.csv", index=False)
    path_diag = bucketize(df)
    path_diag.to_csv(out / "stage22_edge_path_diagnostics.csv", index=False)
    reach = pd.DataFrame(read_jsons(Path(args.artifact_root), "reachability_metrics.json"))
    boundary = pd.DataFrame(read_jsons(Path(args.artifact_root), "boundary_summary.json"))
    if len(reach):
        reach.to_csv(out / "stage22_reachability_diagnostics.csv", index=False)
    if len(boundary):
        boundary.to_csv(out / "stage22_boundary_diagnostics.csv", index=False)
    decisions = decision_summary(grouped, comp, path_diag, boundary, df)
    lines = ["# Stage22 Summary", "", f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]
    lines.append("## Completion")
    lines.append(md_table(grouped))
    lines.append("")
    lines.append("## Same-Backbone Comparison")
    lines.append(md_table(comp) if len(comp) else "No paired comparison available.")
    lines.append("")
    lines.append("## Fallback Attribution")
    if "fallback_mode" in df:
        fb = df.groupby(["fallback_mode", "fallback_used"], dropna=False).agg(episodes=("success", "count"), success=("success", "mean")).reset_index()
        lines.append(md_table(fb))
    else:
        lines.append("No fallback columns available.")
    lines.append("")
    lines.append("## Path Diagnostics")
    lines.append(md_table(path_diag, max_rows=80) if len(path_diag) else "No path diagnostics.")
    lines.append("")
    lines.append("## p_exec Diagnostics")
    if len(reach):
        cols = [c for c in ["path", "val_auroc", "val_auprc", "p_exec_mean", "p_exec_q10", "p_exec_q50", "p_exec_q90"] if c in reach.columns]
        lines.append(md_table(reach[cols]))
    else:
        lines.append("No reachability metrics found.")
    lines.append("")
    lines.append("## Boundary Diagnostics")
    if len(boundary):
        cols = [c for c in ["path", "coverage", "supported_pair_rate", "psi_q10", "psi_q50", "psi_q90"] if c in boundary.columns]
        lines.append(md_table(boundary[cols]))
    else:
        lines.append("No boundary metrics found.")
    lines.append("")
    lines.append("## Decision Summary")
    for d in decisions:
        lines.append(f"- {d}")
    lines.append("")
    (out / "stage22_summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
