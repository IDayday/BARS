#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def add_calibrated_support_columns(
    edge_scores: pd.DataFrame,
    *,
    support_count_column: str = "same_traj_support",
    support_binary_column: str = "local_support",
    edge_source_column: str = "edge_source",
    target_support: float = 8.0,
    protect_goal_edges: bool = True,
) -> pd.DataFrame:
    missing = {support_count_column, support_binary_column} - set(edge_scores.columns)
    if missing:
        raise ValueError(f"edge score table is missing columns: {sorted(missing)}")

    out = edge_scores.copy()
    counts = pd.to_numeric(out[support_count_column], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(float)
    binary = pd.to_numeric(out[support_binary_column], errors="coerce").fillna(0.0).to_numpy(float)
    target = max(float(target_support), 1e-6)

    protected = np.zeros(len(out), dtype=bool)
    if protect_goal_edges and edge_source_column in out.columns:
        protected = out[edge_source_column].astype(str).eq("gas_goal_connector").to_numpy()

    inv_sqrt = 1.0 / np.sqrt(1.0 + counts)
    low_support = np.maximum(0.0, (target - counts) / target)
    unsupported = (binary <= 0.0).astype(float)
    hybrid = 0.5 * inv_sqrt + 0.5 * low_support

    for arr in (inv_sqrt, low_support, unsupported, hybrid):
        arr[protected] = 0.0

    out["calibrated_support_count"] = counts
    out["risk_inv_sqrt_support"] = inv_sqrt
    out["risk_low_support_target"] = low_support
    out["risk_unsupported_binary"] = unsupported
    out["risk_hybrid_support"] = hybrid
    out["calibrated_support_target"] = target
    out["calibrated_goal_protected"] = protected.astype(np.int32)
    return out


def summarize_calibrated_scores(df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {"num_edges": int(len(df))}
    if "edge_source" in df.columns:
        summary["num_goal_connector_edges"] = int(df["edge_source"].astype(str).eq("gas_goal_connector").sum())
    for col in [
        "calibrated_support_count",
        "risk_inv_sqrt_support",
        "risk_low_support_target",
        "risk_unsupported_binary",
        "risk_hybrid_support",
    ]:
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0) if col in df.columns else pd.Series(dtype=float)
        summary[f"{col}_mean"] = float(vals.mean()) if len(vals) else 0.0
        summary[f"{col}_p50"] = float(vals.quantile(0.50)) if len(vals) else 0.0
        summary[f"{col}_p90"] = float(vals.quantile(0.90)) if len(vals) else 0.0
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Add calibrated BARS support-risk columns to GAS edge scores.")
    parser.add_argument("--edge-scores-csv", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--support-count-column", default="same_traj_support")
    parser.add_argument("--support-binary-column", default="local_support")
    parser.add_argument("--edge-source-column", default="edge_source")
    parser.add_argument("--target-support", type=float, default=8.0)
    parser.add_argument("--protect-goal-edges", type=int, default=1)
    args = parser.parse_args()

    edge_scores = pd.read_csv(args.edge_scores_csv)
    out = add_calibrated_support_columns(
        edge_scores,
        support_count_column=args.support_count_column,
        support_binary_column=args.support_binary_column,
        edge_source_column=args.edge_source_column,
        target_support=args.target_support,
        protect_goal_edges=bool(args.protect_goal_edges),
    )
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    summary = summarize_calibrated_scores(out)
    summary.update(
        {
            "edge_scores_csv": str(args.edge_scores_csv),
            "out_csv": str(out_csv),
            "support_count_column": args.support_count_column,
            "support_binary_column": args.support_binary_column,
            "target_support": float(args.target_support),
            "protect_goal_edges": int(bool(args.protect_goal_edges)),
        }
    )
    summary_json = Path(args.summary_json) if args.summary_json else out_csv.with_name(out_csv.stem + "_summary.json")
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out_csv": str(out_csv), "summary_json": str(summary_json), **summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
