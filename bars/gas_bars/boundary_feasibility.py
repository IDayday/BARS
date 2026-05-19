from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .stage22r_common import add_path_metrics, filter_eval, md_table, parse_csv_list, quantile_dict, read_all_eval


def _classify_reject(row: pd.Series) -> str:
    reason = str(row.get("first_plan_reject_reason", "") or row.get("last_plan_reject_reason", ""))
    if "no_start_connection" in reason or "no_goal_connection" in reason or "disconnected" in reason:
        return "disconnected"
    if "max_edges" in reason:
        return "max_edges"
    if int(row.get("budget_reject_count", 0) or 0) > 0 or "budget_infeasible" in reason:
        if float(row.get("first_plan_exec_risk", 0.0) or 0.0) > float(row.get("budget", 0.0) or 0.0):
            return "exec risk too high"
        if float(row.get("first_plan_boundary_risk", 0.0) or 0.0) > 0:
            return "boundary risk too high"
        return "line-graph pruning"
    if float(row.get("missing_boundary_pair_rate", 0.0) or 0.0) > 0:
        return "missing virtual edge-pair"
    return reason or "unknown"


def build_report(df: pd.DataFrame, artifact_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = add_path_metrics(df, artifact_root)
    if len(df) == 0:
        return pd.DataFrame(), pd.DataFrame()
    df["computed_total_risk"] = pd.to_numeric(df.get("first_plan_exec_risk", 0), errors="coerce").fillna(0) + df["boundary_cost_static"].fillna(0)
    df["reject_class"] = df.apply(_classify_reject, axis=1)
    rows = []
    for (env, seed), part in df.groupby(["env", "seed"], dropna=False):
        shortest = part[(part["variant"].astype(str) == "gas_shortest") & (part["fallback_mode"].astype(str) == "none")]
        boundary = part[part["variant"].astype(str).str.contains("boundary", na=False)]
        base = {"env": env, "seed": int(seed)}
        for name, subset in (("shortest", shortest), ("boundary_eval", boundary)):
            row = {**base, "section": name, "episodes": int(len(subset))}
            if len(subset):
                row.update(quantile_dict(subset["first_plan_exec_risk"], "exec_risk"))
                row.update(quantile_dict(subset["boundary_cost_static"], "boundary_risk"))
                row.update(quantile_dict(subset["computed_total_risk"], "total_risk"))
                row["missing_boundary_pair_rate"] = float(subset["missing_boundary_pair_rate"].mean())
                row["virtual_boundary_pair_count"] = float(subset["virtual_boundary_pairs"].sum())
                row["unsupported_pair_rate"] = float(subset["unsupported_boundary_pair_rate"].mean())
                row["budget_reject_rate"] = float((subset.get("budget_reject_count", 0) > 0).mean())
                row["no_path_rate"] = float((subset.get("no_path_count", 0) > 0).mean())
            rows.append(row)
        if len(shortest):
            total = shortest["computed_total_risk"]
            rec = {**base, "section": "recommended_budget", "episodes": int(len(shortest))}
            rec.update(quantile_dict(total, "B"))
            rows.append(rec)
    summary = pd.DataFrame(rows)
    return df, summary


def write_markdown(path: Path, summary: pd.DataFrame, detailed: pd.DataFrame) -> None:
    lines = ["# Stage22R Boundary Feasibility", ""]
    lines.append("## Summary")
    lines.append(md_table(summary))
    lines.append("")
    lines.append("## Reject Reasons")
    if len(detailed):
        reject = (
            detailed[detailed["variant"].astype(str).str.contains("boundary", na=False)]
            .groupby(["env", "seed", "variant", "budget", "fallback_mode", "reject_class"], dropna=False)
            .agg(episodes=("success", "count"), success=("success", "mean"))
            .reset_index()
        )
        lines.append(md_table(reject))
    else:
        lines.append("_No detailed rows._")
    lines.append("")
    lines.append("## Interpretation")
    if len(summary) and (summary.get("budget_reject_rate", pd.Series(dtype=float)).fillna(0) >= 0.5).any():
        lines.append("- HOLD_BOUNDARY: at least one boundary setting rejects >=50% of paths.")
    else:
        lines.append("- Boundary feasibility does not show a high reject-rate gate in the available rows.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--envs", required=True)
    p.add_argument("--seeds", default="0")
    p.add_argument("--artifact-root", default="artifacts/stage22")
    p.add_argument("--eval-root", default="runs_stage22_eval")
    p.add_argument("--out", default="reports")
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    envs = parse_csv_list(args.envs)
    seeds = [int(x) for x in parse_csv_list(args.seeds)]
    df = filter_eval(read_all_eval(args.eval_root), envs, seeds)
    detailed, summary = build_report(df, Path(args.artifact_root))
    detailed.to_csv(out / "stage22r_boundary_feasibility.csv", index=False)
    summary.to_csv(out / "stage22r_boundary_feasibility_summary.csv", index=False)
    write_markdown(out / "stage22r_boundary_feasibility.md", summary, detailed)
    print(json.dumps({"rows": int(len(detailed)), "summary_rows": int(len(summary)), "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
