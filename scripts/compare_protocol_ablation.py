#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

import pandas as pd


def _collect(log_root: Path, force: bool = False) -> None:
    if force or not (log_root / "_analysis" / "eval_all.csv").exists():
        subprocess.run([sys.executable, "scripts/collect_csv.py", "--log-root", str(log_root)], check=False)


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _label_from_root(root: str) -> str:
    return Path(root).name.replace("runs_", "")


def _ensure_condition(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "condition" not in out.columns:
        out["condition"] = "unknown"
    # Backfill from run_id when eval.condition was not available.
    if "run_id" in out.columns:
        missing = out["condition"].isna() | (out["condition"].astype(str).isin(["", "nan", "default", "unknown"]))
        if missing.any():
            known = [
                "horizon_threshold_only",
                "planner_fallback_only",
                "direct_goal_fallback",
                "direct_goal_after3",
                "tuned_no_fallback",
                "direct_goal_only",
                "original",
                "time_only",
            ]
            rid = out.loc[missing, "run_id"].astype(str)
            for cond in known:
                hit = missing.copy()
                hit.loc[missing] = rid.str.contains(cond, regex=False).values
                out.loc[hit, "condition"] = cond
    return out


def _load_roots(items: List[str], force_collect: bool) -> pd.DataFrame:
    frames = []
    for item in items:
        if "=" in item:
            label, root = item.split("=", 1)
        else:
            root = item
            label = _label_from_root(root)
        root_path = Path(root)
        _collect(root_path, force_collect)
        df = _read(root_path / "_analysis" / "eval_all.csv")
        if df.empty:
            print(f"[WARN] missing eval_all.csv for {root}", file=sys.stderr)
            continue
        df = _ensure_condition(df)
        df.insert(0, "result_set", label)
        df.insert(1, "log_root", str(root_path))
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _agg(df: pd.DataFrame, keys: List[str], cols: List[str]) -> pd.DataFrame:
    keys = [c for c in keys if c in df.columns]
    cols = [c for c in cols if c in df.columns]
    if not keys or not cols:
        return pd.DataFrame()
    return df.groupby(keys, dropna=False)[cols].agg(["count", "mean", "std"]).reset_index()


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out.columns = [
        "_".join(str(x) for x in col if str(x) != "").rstrip("_") if isinstance(col, tuple) else str(col)
        for col in out.columns
    ]
    return out


def _md_table(df: pd.DataFrame, max_rows: int = 400) -> str:
    if df.empty:
        return "_No data._"
    show = df.head(max_rows).copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].round(4)
    return show.to_markdown(index=False)


def _success_by_bucket(df: pd.DataFrame, col: str, out_col: str) -> pd.DataFrame:
    if df.empty or col not in df.columns or "success" not in df.columns:
        return pd.DataFrame()
    x = df.copy()
    if col in ["last_plan_edges", "first_plan_edges", "max_plan_edges"]:
        x[out_col] = pd.cut(
            pd.to_numeric(x[col], errors="coerce").fillna(-1),
            bins=[-2, 0, 1, 2, 1000],
            labels=["0", "1", "2", "3+"],
        )
    elif col == "fallback_used":
        x[out_col] = pd.to_numeric(x[col], errors="coerce").fillna(0).astype(int).astype(str)
    else:
        x[out_col] = x[col].astype(str)
    keys = [c for c in ["result_set", "condition", "env", "variant", out_col] if c in x.columns]
    return x.groupby(keys, dropna=False)["success"].agg(["count", "mean", "std"]).reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", required=True, help="LABEL=log_root or log_root. Can be repeated.")
    ap.add_argument("--out", default="reports/stage19_protocol_ablation.md")
    ap.add_argument("--csv-out", default="reports/stage19_protocol_ablation_summary.csv")
    ap.add_argument("--force-collect", action="store_true")
    args = ap.parse_args()

    df = _load_roots(args.root, args.force_collect)
    out_path = Path(args.out)
    csv_path = Path(args.csv_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        out_path.write_text("# Protocol Ablation\n\n_No eval data found._\n", encoding="utf-8")
        return

    metric_cols = [
        "success",
        "return",
        "steps",
        "replans",
        "no_path_count",
        "goal_distance_final",
        "fallback_used",
        "fallback_count",
        "direct_goal_attempts",
        "initial_plan_failed_count",
        "plan_failed_initial",
        "last_plan_edges",
        "first_plan_edges",
        "max_plan_edges",
        "mean_plan_edges",
        "num_plan_calls",
        "num_subgoal_attempts",
        "num_subgoal_reached",
        "subgoal_reach_rate",
        "subgoal_horizon",
        "subgoal_threshold",
        "success_threshold",
        "lambda_risk",
        "lambda_boundary",
    ]
    df = _numeric(df, metric_cols)
    if "success" in df.columns and "steps" in df.columns:
        df["success_if_capped_700"] = ((df["success"].fillna(0) > 0) & (df["steps"].fillna(10**9) <= 700)).astype(int)
        metric_cols.append("success_if_capped_700")

    group_keys = ["result_set", "condition", "env", "variant"]
    summary = _flatten(_agg(df, group_keys, metric_cols))
    overall = _flatten(_agg(df, ["result_set", "condition", "variant"], metric_cols))
    by_fallback = _success_by_bucket(df, "fallback_used", "fallback_used_bucket")
    by_last_edges = _success_by_bucket(df, "last_plan_edges", "last_plan_edges_bucket")
    by_first_edges = _success_by_bucket(df, "first_plan_edges", "first_plan_edges_bucket")

    summary.to_csv(csv_path, index=False)
    overall.to_csv(csv_path.with_name(csv_path.stem + "_overall.csv"), index=False)
    by_fallback.to_csv(csv_path.with_name(csv_path.stem + "_fallback.csv"), index=False)
    by_last_edges.to_csv(csv_path.with_name(csv_path.stem + "_last_edges.csv"), index=False)
    by_first_edges.to_csv(csv_path.with_name(csv_path.stem + "_first_edges.csv"), index=False)

    dg = (
        overall[
            (overall.get("variant", pd.Series(dtype=str)).astype(str) == "direct_goal")
            | (overall.get("condition", pd.Series(dtype=str)).astype(str) == "direct_goal_only")
        ].copy()
        if not overall.empty
        else pd.DataFrame()
    )

    lines = [
        "# Stage19 Protocol Ablation Report",
        "",
        "## Inputs",
        "",
        *[f"- `{x}`" for x in args.root],
        "",
        "## Mean by condition / env / variant",
        _md_table(summary),
        "",
        "## Overall by condition / variant",
        _md_table(overall),
        "",
        "## Direct-goal rows",
        _md_table(dg),
        "",
        "## Success by fallback_used",
        _md_table(by_fallback),
        "",
        "## Success by last_plan_edges bucket",
        _md_table(by_last_edges),
        "",
        "## Success by first_plan_edges bucket",
        _md_table(by_first_edges),
        "",
        "## Interpretation checklist",
        "",
        "- If `direct_goal_only` is close to `direct_goal_fallback/full_bars`, Stage18 gains are mostly execution-protocol/direct-goal effects.",
        "- If `tuned_no_fallback/full_bars` beats `tuned_no_fallback/reachability`, boundary has a cleaner contribution.",
        "- If `planner_fallback_only` improves over `tuned_no_fallback`, graph fallback helps independently of direct-goal fallback.",
        "- If `success_if_capped_700` collapses relative to `success`, `max_steps=1000` is a major confound.",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_path)
    print(csv_path)


if __name__ == "__main__":
    main()
