#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _read_table(path: Path) -> pd.DataFrame:
    if path.is_dir():
        path = path / "contract_scored_rows.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _quantile(values: pd.Series, q: float) -> float:
    if len(values) == 0:
        return float("nan")
    return float(values.quantile(q))


def build_edge_scores(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_table(Path(args.scored_rows))
    if args.env:
        rows = rows[rows["env"].astype(str) == args.env].copy()
    if rows.empty:
        raise RuntimeError("No scored contract rows after filtering")
    rows = rows[pd.to_numeric(rows["edge_forward"], errors="coerce").fillna(0).astype(int) == 1].copy()
    if rows.empty:
        raise RuntimeError("No forward contract rows after filtering edge_forward == 1")
    rows["contract_prob"] = pd.to_numeric(rows["contract_prob"], errors="coerce").clip(0.0, 1.0)
    rows = rows[rows["contract_prob"].notna()].copy()

    group_cols = ["env", "seed", "edge_id", "edge_u", "edge_v", "edge_source"]
    grouped = rows.groupby(group_cols, dropna=False)
    agg = grouped.agg(
        contract_samples=("contract_prob", "size"),
        contract_prob_mean=("contract_prob", "mean"),
        contract_prob_median=("contract_prob", "median"),
        contract_prob_max=("contract_prob", "max"),
        contract_prob_p75=("contract_prob", lambda s: _quantile(s, 0.75)),
        contract_prob_p90=("contract_prob", lambda s: _quantile(s, 0.90)),
        local_support=("local_support", "max"),
        same_traj_support=("same_traj_support", "max"),
        mean_phi_dist_pair=("phi_dist_pair", "mean"),
        mean_node_u_dist=("node_u_dist", "mean"),
        mean_node_v_dist=("node_v_dist", "mean"),
        edge_phi_dist=("edge_phi_dist", "mean"),
    ).reset_index()

    score_col = str(args.score_column)
    if score_col not in agg.columns:
        raise ValueError(f"Unknown score column {score_col!r}; available columns: {sorted(agg.columns)}")
    agg["u"] = pd.to_numeric(agg["edge_u"], errors="raise").astype(int)
    agg["v"] = pd.to_numeric(agg["edge_v"], errors="raise").astype(int)
    agg["contract_prob_edge"] = pd.to_numeric(agg[score_col], errors="raise").clip(0.0, 1.0)
    agg["contract_risk"] = (1.0 - agg["contract_prob_edge"]).clip(0.0, 1.0)
    agg["contract_support"] = (agg["contract_prob_edge"] >= float(args.contract_threshold)).astype(np.int32)
    agg["contract_available"] = 1
    agg["r_exec"] = agg["contract_risk"]
    agg["p_exec"] = agg["contract_prob_edge"]
    agg["support_penalty"] = np.where(agg["contract_support"] > 0, 0.0, agg["contract_risk"])

    cols = [
        "edge_id",
        "u",
        "v",
        "edge_source",
        "edge_phi_dist",
        "local_support",
        "same_traj_support",
        "contract_available",
        "contract_support",
        "contract_samples",
        "contract_prob_edge",
        "contract_prob_mean",
        "contract_prob_median",
        "contract_prob_max",
        "contract_prob_p75",
        "contract_prob_p90",
        "contract_risk",
        "p_exec",
        "r_exec",
        "support_penalty",
        "mean_phi_dist_pair",
        "mean_node_u_dist",
        "mean_node_v_dist",
    ]
    scored = agg[cols].sort_values(["u", "v", "edge_id"]).reset_index(drop=True)
    csv_path = out_dir / "caplite_edge_scores.csv"
    scored.to_csv(csv_path, index=False)
    try:
        scored.to_parquet(out_dir / "caplite_edge_scores.parquet", index=False)
    except Exception:
        pass

    summary: dict[str, Any] = {
        "offline_only": True,
        "uses_environment_rollout": False,
        "scored_rows": str(args.scored_rows),
        "env": args.env or "all",
        "num_forward_rows": int(len(rows)),
        "num_scored_edges": int(len(scored)),
        "score_column": score_col,
        "contract_threshold": float(args.contract_threshold),
        "mean_contract_prob_edge": float(scored["contract_prob_edge"].mean()) if len(scored) else 0.0,
        "median_contract_prob_edge": float(scored["contract_prob_edge"].median()) if len(scored) else 0.0,
        "contract_supported_edge_rate": float(scored["contract_support"].mean()) if len(scored) else 0.0,
        "mean_contract_samples": float(scored["contract_samples"].mean()) if len(scored) else 0.0,
        "edge_scores_csv": str(csv_path),
        "edge_source_counts": {str(k): int(v) for k, v in scored["edge_source"].value_counts().sort_index().items()},
    }
    summary_path = out_dir / "caplite_edge_scores_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Stage45 contract row probabilities into GAS edge scores.")
    parser.add_argument("--scored-rows", required=True)
    parser.add_argument("--env", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--score-column", default="contract_prob_mean")
    parser.add_argument("--contract-threshold", type=float, default=0.5)
    args = parser.parse_args()
    build_edge_scores(args)


if __name__ == "__main__":
    main()
