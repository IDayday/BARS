from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .stage22r_common import add_path_metrics, filter_eval, md_table, parse_csv_list, read_all_eval


SLICE_NAMES = [
    "S1_gas_shortest_failed",
    "S2_long_path_p75",
    "S3_high_exec_risk_p75",
    "S4_low_local_support_p25",
    "S5_high_steps_p75",
    "S6_low_pred_success_p25",
]


def build_slices(df: pd.DataFrame, artifact_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = add_path_metrics(df, artifact_root)
    if len(df) == 0:
        return pd.DataFrame(), pd.DataFrame()
    keys = ["env", "seed", "budget", "fallback_mode", "task_id", "episode_id"]
    baseline = df[df["variant"].astype(str) == "gas_shortest"].copy()
    slice_rows = []
    for (env, seed, budget, fallback), base in baseline.groupby(["env", "seed", "budget", "fallback_mode"], dropna=False):
        edges_p75 = pd.to_numeric(base["first_plan_edges"], errors="coerce").quantile(0.75)
        risk_p75 = pd.to_numeric(base["first_plan_exec_risk"], errors="coerce").quantile(0.75)
        support_p25 = pd.to_numeric(base["local_support_rate"], errors="coerce").quantile(0.25)
        steps_p75 = pd.to_numeric(base["steps"], errors="coerce").quantile(0.75)
        pred_p25 = pd.to_numeric(base["first_plan_pred_success"], errors="coerce").quantile(0.25)
        for r in base.itertuples(index=False):
            flags = {
                "S1_gas_shortest_failed": int(getattr(r, "success") == 0),
                "S2_long_path_p75": int(float(getattr(r, "first_plan_edges", 0)) >= edges_p75),
                "S3_high_exec_risk_p75": int(float(getattr(r, "first_plan_exec_risk", 0)) >= risk_p75),
                "S4_low_local_support_p25": int(float(getattr(r, "local_support_rate", np.nan)) <= support_p25) if np.isfinite(support_p25) else 0,
                "S5_high_steps_p75": int(float(getattr(r, "steps", 0)) >= steps_p75),
                "S6_low_pred_success_p25": int(float(getattr(r, "first_plan_pred_success", 0)) <= pred_p25),
            }
            row = {k: getattr(r, k) for k in keys}
            row.update(flags)
            slice_rows.append(row)
    slice_df = pd.DataFrame(slice_rows)
    if len(slice_df) == 0:
        return df, pd.DataFrame()
    merged = df.merge(slice_df, on=keys, how="inner")
    rows = []
    for slice_name in SLICE_NAMES:
        part = merged[merged[slice_name] == 1]
        if len(part) == 0:
            continue
        grouped = (
            part.groupby(["env", "seed", "budget", "fallback_mode", "variant"], dropna=False)
            .agg(
                episodes=("success", "count"),
                success=("success", "mean"),
                steps=("steps", "mean"),
                fallback_used_rate=("fallback_used", "mean"),
                path_len_mean=("first_plan_edges", "mean"),
                exec_risk_mean=("first_plan_exec_risk", "mean"),
                local_support_rate=("local_support_rate", "mean"),
            )
            .reset_index()
        )
        for (env, seed, budget, fallback), sub in grouped.groupby(["env", "seed", "budget", "fallback_mode"], dropna=False):
            base_success = sub.loc[sub["variant"] == "gas_shortest", "success"]
            baseline_val = float(base_success.iloc[0]) if len(base_success) else np.nan
            for row in sub.to_dict("records"):
                row["slice"] = slice_name
                row["success_delta_vs_shortest"] = float(row["success"] - baseline_val) if np.isfinite(baseline_val) else np.nan
                rows.append(row)
    summary = pd.DataFrame(rows)
    return merged, summary


def write_report(path: Path, summary: pd.DataFrame) -> None:
    lines = ["# Stage22R Failure Slice Diagnostics", ""]
    lines.append(md_table(summary, max_rows=120))
    if len(summary):
        hard = summary[(summary["variant"].astype(str).str.contains("reachability", na=False)) & (summary["success_delta_vs_shortest"] >= 0.05)]
        lines.append("")
        if len(hard):
            lines.append(f"- Reachability has >=5pp hard-slice gains in {len(hard)} grouped slice rows.")
        else:
            lines.append("- No grouped hard-slice gain >=5pp found yet.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--envs", required=True)
    p.add_argument("--seeds", default="0")
    p.add_argument("--eval-root", default="runs_stage22_eval")
    p.add_argument("--artifact-root", default="artifacts/stage22")
    p.add_argument("--out", default="reports")
    args = p.parse_args(argv)
    envs = parse_csv_list(args.envs)
    seeds = [int(x) for x in parse_csv_list(args.seeds)]
    df = filter_eval(read_all_eval(args.eval_root), envs, seeds)
    merged, summary = build_slices(df, Path(args.artifact_root))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out / "stage22r_failure_slice_rows.csv", index=False)
    summary.to_csv(out / "stage22r_failure_slices.csv", index=False)
    write_report(out / "stage22r_failure_slices.md", summary)
    print(json.dumps({"slice_rows": int(len(merged)), "summary_rows": int(len(summary)), "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
