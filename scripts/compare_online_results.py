#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


def read_eval(root: str, label: str) -> pd.DataFrame:
    path = Path(root) / "_analysis" / "eval_all.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    df.insert(0, "condition", label)
    return df


def summarize(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    if df.empty:
        return df
    metrics = [
        "success", "return", "steps", "replans", "no_path_count",
        "initial_plan_failed_count", "fallback_used", "fallback_count", "direct_goal_attempts",
        "last_plan_edges", "first_plan_edges", "max_plan_edges", "mean_plan_edges",
        "num_plan_calls", "num_subgoal_attempts", "num_subgoal_reached", "subgoal_reach_rate",
        "goal_distance_final", "lambda_risk", "lambda_boundary",
    ]
    metrics = [c for c in metrics if c in df.columns]
    keys = [c for c in keys if c in df.columns]
    if not metrics or not keys:
        return pd.DataFrame()
    out = df.groupby(keys, dropna=False)[metrics].agg(["count", "mean", "std"]).reset_index()
    out.columns = ["_".join([str(x) for x in col if str(x) != ""]).rstrip("_") for col in out.columns.to_flat_index()]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", required=True, help="LABEL=LOG_ROOT")
    ap.add_argument("--out", default="reports/online_comparison.md")
    ap.add_argument("--csv-out", default="reports/online_comparison.csv")
    args = ap.parse_args()

    dfs = []
    for item in args.root:
        if "=" not in item:
            raise SystemExit(f"--root must be LABEL=LOG_ROOT, got {item}")
        label, root = item.split("=", 1)
        df = read_eval(root, label)
        if df.empty:
            print(f"[WARN] no eval data for {label}: {root}")
        else:
            dfs.append(df)
    if not dfs:
        raise SystemExit("No eval data found.")
    all_df = pd.concat(dfs, ignore_index=True, sort=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.csv_out).parent.mkdir(parents=True, exist_ok=True)

    by_env_variant = summarize(all_df, ["condition", "env", "variant"])
    by_variant = summarize(all_df, ["condition", "variant"])
    per_seed = summarize(all_df, ["condition", "env", "variant", "seed"])
    by_env_variant.to_csv(args.csv_out, index=False)

    lines = ["# Online Results Comparison", ""]
    lines.append("## Mean by condition/env/variant")
    lines.append(by_env_variant.round(4).to_markdown(index=False) if not by_env_variant.empty else "_No data._")
    lines.append("")
    lines.append("## Overall by condition/variant")
    lines.append(by_variant.round(4).to_markdown(index=False) if not by_variant.empty else "_No data._")
    lines.append("")
    lines.append("## Per-seed success")
    if not per_seed.empty:
        cols = [c for c in per_seed.columns if c in {"condition", "env", "variant", "seed", "success_mean", "success_count"}]
        lines.append(per_seed[cols].round(4).to_markdown(index=False))
    else:
        lines.append("_No data._")
    lines.append("")

    if "success" in all_df.columns:
        pivot = all_df.groupby(["condition", "variant"])["success"].mean().unstack("variant")
        lines.append("## Decision helper")
        lines.append(pivot.round(4).to_markdown())
        lines.append("")
        for cond, row in pivot.iterrows():
            if "reachability" in row and "shortest" in row:
                lines.append(f"- {cond}: reachability - shortest = {row['reachability'] - row['shortest']:.4f}")
            if "full_bars" in row and "reachability" in row:
                lines.append(f"- {cond}: full_bars - reachability = {row['full_bars'] - row['reachability']:.4f}")
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out)
    print(args.csv_out)


if __name__ == "__main__":
    main()
