#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import pandas as pd


def read_eval(root: str, label: str) -> pd.DataFrame:
    path = Path(root) / "_analysis" / "eval_all.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    if "condition" not in df.columns:
        df["condition"] = label
    df["condition"] = df["condition"].fillna("").replace("", label)
    df["source_root"] = label
    return df


def to_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def summarize(df: pd.DataFrame, keys: List[str], metrics: List[str]) -> pd.DataFrame:
    if df.empty:
        return df
    keys = [k for k in keys if k in df.columns]
    metrics = [m for m in metrics if m in df.columns]
    if not keys or not metrics:
        return pd.DataFrame()
    out = df.groupby(keys, dropna=False)[metrics].agg(["count", "mean", "std"]).reset_index()
    out.columns = ["_".join([str(x) for x in col if str(x) != ""]).rstrip("_") for col in out.columns.to_flat_index()]
    return out.sort_values(keys).reset_index(drop=True)


def edge_bucket(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce").fillna(-1)
    buckets = []
    for v in vals:
        iv = int(v)
        if iv <= 0:
            buckets.append("0")
        elif iv == 1:
            buckets.append("1")
        elif iv == 2:
            buckets.append("2")
        elif iv <= 4:
            buckets.append("3-4")
        elif iv <= 8:
            buckets.append("5-8")
        else:
            buckets.append("9+")
    return pd.Series(buckets, index=series.index)


def markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data._"
    return df.round(4).to_markdown(index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", required=True, help="LABEL=LOG_ROOT")
    ap.add_argument("--out", default="reports/protocol_ablation.md")
    ap.add_argument("--csv-out", default="reports/protocol_ablation_summary.csv")
    ap.add_argument("--edge-bucket-csv", default="reports/protocol_ablation_edge_buckets.csv")
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
    all_df = to_numeric(
        all_df,
        [
            "success",
            "steps",
            "fallback_used",
            "fallback_count",
            "direct_goal_attempts",
            "subgoal_reach_rate",
            "first_plan_edges",
            "mean_plan_edges",
            "last_plan_edges",
            "goal_distance_final",
        ],
    )
    all_df["success_by700"] = ((all_df.get("success", 0) > 0) & (all_df.get("steps", 10 ** 9) <= 700)).astype(float)
    all_df["used_direct_goal"] = (all_df.get("direct_goal_attempts", 0) > 0).astype(float)
    all_df["plan_edge_bucket"] = edge_bucket(all_df.get("first_plan_edges", pd.Series(dtype=float)))

    summary_metrics = [
        "success",
        "success_by700",
        "fallback_used",
        "fallback_count",
        "direct_goal_attempts",
        "used_direct_goal",
        "subgoal_reach_rate",
        "goal_distance_final",
        "first_plan_edges",
        "mean_plan_edges",
    ]
    success_table = summarize(all_df, ["condition", "env", "variant"], summary_metrics)
    fallback_table = summarize(
        all_df,
        ["condition", "env", "variant"],
        ["fallback_used", "fallback_count", "direct_goal_attempts", "used_direct_goal"],
    )
    no_fallback_df = all_df[all_df.get("fallback_used", 0) <= 0].copy()
    no_fallback_table = summarize(no_fallback_df, ["condition", "env", "variant"], ["success", "success_by700", "subgoal_reach_rate"])
    direct_goal_only_df = all_df[all_df.get("variant", "") == "direct_goal"].copy()
    direct_goal_only_table = summarize(
        direct_goal_only_df,
        ["condition", "env", "variant"],
        ["success", "success_by700", "subgoal_reach_rate", "goal_distance_final"],
    )
    edge_bucket_table = summarize(all_df, ["condition", "env", "variant", "plan_edge_bucket"], ["success", "success_by700", "fallback_used"])

    pivot_source = all_df.copy()
    pivot_source["cond_variant"] = pivot_source["condition"].astype(str) + ":" + pivot_source["variant"].astype(str)
    dg_compare = (
        pivot_source.groupby(["env", "cond_variant"], dropna=False)["success"]
        .mean()
        .unstack("cond_variant")
        .reset_index()
        .sort_values("env")
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.csv_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.edge_bucket_csv).parent.mkdir(parents=True, exist_ok=True)
    success_table.to_csv(args.csv_out, index=False)
    edge_bucket_table.to_csv(args.edge_bucket_csv, index=False)

    lines = ["# Protocol Ablation Comparison", ""]
    lines.append("## Success By Condition/Env/Variant")
    lines.append(markdown(success_table))
    lines.append("")
    lines.append("## Fallback Usage")
    lines.append(markdown(fallback_table))
    lines.append("")
    lines.append("## No-Fallback Success")
    lines.append(markdown(no_fallback_table))
    lines.append("")
    lines.append("## Direct-Goal-Only Comparison")
    lines.append(markdown(direct_goal_only_table))
    lines.append("")
    lines.append("## Direct-Goal Context Pivot")
    lines.append(markdown(dg_compare))
    lines.append("")
    lines.append("## Plan Edge Bucket Success")
    lines.append(markdown(edge_bucket_table))
    lines.append("")

    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out)
    print(args.csv_out)
    print(args.edge_bucket_csv)


if __name__ == "__main__":
    main()
