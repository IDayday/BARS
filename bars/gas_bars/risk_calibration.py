from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .stage22r_common import add_path_metrics, filter_eval, md_table, parse_csv_list, quantile_dict, read_all_eval


def _collapse_budgets(vals: dict[str, float], eps: float = 0.05) -> dict[str, float]:
    kept: dict[str, float] = {}
    for k, v in vals.items():
        if not np.isfinite(v):
            continue
        if all(abs(v - old) > eps for old in kept.values()):
            kept[k] = float(v)
    return kept


def calibrate(df: pd.DataFrame, artifact_root: Path) -> tuple[dict[str, object], pd.DataFrame]:
    df = add_path_metrics(df, artifact_root)
    if len(df) == 0:
        return {"envs": {}}, pd.DataFrame()
    df["computed_total_risk"] = pd.to_numeric(df.get("first_plan_exec_risk", 0), errors="coerce").fillna(0) + df["boundary_cost_static"].fillna(0)
    report_rows = []
    envs: dict[str, object] = {}
    for (env, seed), part in df.groupby(["env", "seed"], dropna=False):
        shortest = part[(part["variant"].astype(str) == "gas_shortest") & (part["fallback_mode"].astype(str) == "none")]
        reach = part[part["variant"].astype(str).str.contains("reachability", na=False)]
        key = f"{env}/seed{int(seed)}"
        exec_budgets = _collapse_budgets(quantile_dict(shortest["first_plan_exec_risk"], "exec"))
        reach_budgets = _collapse_budgets(quantile_dict(reach["first_plan_exec_risk"], "reach_exec"))
        boundary_median = float(shortest["boundary_cost_static"].replace([np.inf, -np.inf], np.nan).dropna().median()) if len(shortest) else np.nan
        exec_median = float(shortest["first_plan_exec_risk"].replace([np.inf, -np.inf], np.nan).dropna().median()) if len(shortest) else np.nan
        if not np.isfinite(boundary_median) or boundary_median <= 0:
            alpha = None
            boundary_status = "HOLD_BOUNDARY"
        else:
            alpha = float(np.clip(exec_median / boundary_median, 0.05, 1.0))
            boundary_status = "CALIBRATED"
        total_budgets = _collapse_budgets(quantile_dict(shortest["computed_total_risk"], "total"))
        envs[key] = {
            "env": env,
            "seed": int(seed),
            "exec_budgets_from_shortest": exec_budgets,
            "reachability_budgets_from_selected": reach_budgets,
            "total_budgets_for_boundary": total_budgets,
            "recommended_reachability_budget": exec_budgets.get("exec_q60") or exec_budgets.get("exec_q70") or exec_budgets.get("exec_q50"),
            "recommended_boundary_budget": total_budgets.get("total_q60") or total_budgets.get("total_q70"),
            "alpha_boundary": alpha,
            "boundary_status": boundary_status,
        }
        row = {
            "env": env,
            "seed": int(seed),
            "shortest_episodes": int(len(shortest)),
            "reachability_episodes": int(len(reach)),
            "exec_median": exec_median,
            "boundary_median": boundary_median,
            "alpha_boundary": alpha if alpha is not None else np.nan,
            "boundary_status": boundary_status,
        }
        row.update(exec_budgets)
        row.update(reach_budgets)
        row.update(total_budgets)
        report_rows.append(row)
    return {"envs": envs}, pd.DataFrame(report_rows)


def write_report(path: Path, table: pd.DataFrame, recommendations: dict[str, object]) -> None:
    lines = ["# Stage22R Risk Calibration", ""]
    lines.append("## Recommended Budgets")
    lines.append(md_table(table))
    lines.append("")
    lines.append("## JSON")
    lines.append("```json")
    lines.append(json.dumps(recommendations, indent=2, sort_keys=True))
    lines.append("```")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--envs", required=True)
    p.add_argument("--seeds", default="0")
    p.add_argument("--artifact-root", default="artifacts/stage22")
    p.add_argument("--eval-root", default="runs_stage22_eval")
    p.add_argument("--out", default="reports/stage22r_recommended_budgets.json")
    p.add_argument("--report", default="reports/stage22r_risk_calibration.md")
    args = p.parse_args(argv)
    envs = parse_csv_list(args.envs)
    seeds = [int(x) for x in parse_csv_list(args.seeds)]
    df = filter_eval(read_all_eval(args.eval_root), envs, seeds)
    recommendations, table = calibrate(df, Path(args.artifact_root))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(recommendations, indent=2, sort_keys=True))
    table.to_csv(Path(args.report).with_suffix(".csv"), index=False)
    write_report(Path(args.report), table, recommendations)
    print(json.dumps({"envs": len(recommendations.get("envs", {})), "out": str(out), "report": args.report}, indent=2))


if __name__ == "__main__":
    main()
