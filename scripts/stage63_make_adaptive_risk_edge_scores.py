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
        candidate = path / "caplite_edge_scores.csv"
        if candidate.exists():
            path = candidate
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df:
        return pd.Series(default, index=df.index, dtype=np.float32)
    vals = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return vals.fillna(default)


def _source_risk(df: pd.DataFrame, preferred: str) -> pd.Series:
    if preferred in df:
        return _num(df, preferred, 0.0).clip(0.0, 1.0)
    if preferred != "contract_risk" and "contract_risk" in df:
        return _num(df, "contract_risk", 0.0).clip(0.0, 1.0)
    if "r_exec" in df:
        return _num(df, "r_exec", 0.0).clip(0.0, 1.0)
    raise ValueError(
        f"Could not find risk source column {preferred!r}; "
        f"available columns: {sorted(df.columns)}"
    )


def make_adaptive_scores(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    edge_scores = _read_table(Path(args.edge_scores)).copy()
    if args.env and "env" in edge_scores:
        edge_scores = edge_scores[edge_scores["env"].astype(str) == args.env].copy()
    if edge_scores.empty:
        raise RuntimeError("No edge scores after filtering")

    base_risk = _source_risk(edge_scores, str(args.base_risk_column))
    sample_counts = _num(edge_scores, str(args.sample_count_column), 1.0).clip(lower=0.0)
    seq_gate = _num(edge_scores, str(args.sequence_gate_column), 0.0).clip(0.0, 1.0)

    denom = max(float(args.sample_count_target) - float(args.sample_count_offset), 1e-6)
    sample_gate = ((sample_counts - float(args.sample_count_offset)) / denom).clip(0.0, 1.0)
    sample_gate = sample_gate.pow(float(args.sample_gate_power)).clip(0.0, 1.0)

    evidence_gate = np.maximum(sample_gate.to_numpy(np.float32), seq_gate.to_numpy(np.float32))
    evidence_gate = np.maximum(evidence_gate, float(args.evidence_floor))
    evidence_gate = np.clip(evidence_gate, 0.0, 1.0)

    local_support = _num(edge_scores, str(args.local_support_column), 0.0).clip(0.0, 1.0)
    same_traj_support = _num(edge_scores, str(args.same_traj_support_column), 0.0).clip(0.0, 1.0)
    positive_support = np.maximum(local_support.to_numpy(np.float32), same_traj_support.to_numpy(np.float32))
    support_relief = 1.0 - float(args.positive_support_discount) * positive_support
    support_relief = np.clip(support_relief, float(args.min_support_relief), 1.0)

    adaptive_risk = base_risk.to_numpy(np.float32) * evidence_gate * support_relief
    adaptive_risk = np.clip(adaptive_risk, 0.0, 1.0)

    contract_support = _num(edge_scores, "contract_support", 0.0).clip(0.0, 1.0).to_numpy(np.float32)
    supported_cap = float(args.supported_edge_risk_cap)
    if supported_cap < 1.0:
        adaptive_risk = np.where(
            contract_support > 0,
            np.minimum(adaptive_risk, supported_cap * base_risk.to_numpy(np.float32)),
            adaptive_risk,
        )

    out = edge_scores.copy()
    out["r_exec_original"] = _num(edge_scores, "r_exec", 0.0).clip(0.0, 1.0).astype(np.float32)
    out["adaptive_risk_source"] = base_risk.astype(np.float32)
    out["adaptive_sample_gate"] = sample_gate.astype(np.float32)
    out["adaptive_sequence_gate"] = seq_gate.astype(np.float32)
    out["adaptive_evidence_gate"] = evidence_gate.astype(np.float32)
    out["adaptive_positive_support"] = positive_support.astype(np.float32)
    out["adaptive_support_relief"] = support_relief.astype(np.float32)
    out["r_exec_adaptive"] = adaptive_risk.astype(np.float32)
    out["r_exec"] = out["r_exec_adaptive"]
    if "support_penalty" in out:
        out["support_penalty"] = np.where(contract_support > 0, 0.0, adaptive_risk).astype(np.float32)

    out = out.sort_values([c for c in ["u", "v", "edge_id"] if c in out.columns]).reset_index(drop=True)
    csv_path = out_dir / "caplite_edge_scores.csv"
    out.to_csv(csv_path, index=False)
    try:
        out.to_parquet(out_dir / "caplite_edge_scores.parquet", index=False)
    except Exception:
        pass

    delta = base_risk.to_numpy(np.float32) - adaptive_risk
    summary: dict[str, Any] = {
        "offline_only": True,
        "uses_environment_rollout": False,
        "edge_scores": str(args.edge_scores),
        "env": args.env or "all",
        "num_edges": int(len(out)),
        "base_risk_column": str(args.base_risk_column),
        "sample_count_column": str(args.sample_count_column),
        "sequence_gate_column": str(args.sequence_gate_column),
        "sample_count_offset": float(args.sample_count_offset),
        "sample_count_target": float(args.sample_count_target),
        "sample_gate_power": float(args.sample_gate_power),
        "evidence_floor": float(args.evidence_floor),
        "positive_support_discount": float(args.positive_support_discount),
        "min_support_relief": float(args.min_support_relief),
        "supported_edge_risk_cap": float(args.supported_edge_risk_cap),
        "mean_base_risk": float(base_risk.mean()) if len(out) else 0.0,
        "mean_adaptive_risk": float(np.mean(adaptive_risk)) if len(out) else 0.0,
        "median_adaptive_risk": float(np.median(adaptive_risk)) if len(out) else 0.0,
        "mean_adaptive_evidence_gate": float(np.mean(evidence_gate)) if len(out) else 0.0,
        "mean_sample_gate": float(sample_gate.mean()) if len(out) else 0.0,
        "mean_sequence_gate": float(seq_gate.mean()) if len(out) else 0.0,
        "mean_support_relief": float(np.mean(support_relief)) if len(out) else 0.0,
        "fraction_zero_adaptive_risk": float(np.mean(adaptive_risk <= 1e-9)) if len(out) else 0.0,
        "mean_risk_reduction": float(np.mean(delta)) if len(out) else 0.0,
        "risk_reduction_rate": float(np.mean(delta > 1e-9)) if len(out) else 0.0,
        "edge_scores_csv": str(csv_path),
    }
    summary_path = out_dir / "adaptive_risk_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite GAS edge risk into an offline confidence-weighted planner penalty. "
            "This keeps contract probabilities fixed and only adapts r_exec."
        )
    )
    parser.add_argument("--edge-scores", required=True)
    parser.add_argument("--env", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--base-risk-column", default="contract_risk")
    parser.add_argument("--sample-count-column", default="contract_samples")
    parser.add_argument("--sample-count-offset", type=float, default=1.0)
    parser.add_argument("--sample-count-target", type=float, default=4.0)
    parser.add_argument("--sample-gate-power", type=float, default=1.0)
    parser.add_argument("--sequence-gate-column", default="sequence_evidence_gate")
    parser.add_argument("--evidence-floor", type=float, default=0.0)
    parser.add_argument("--local-support-column", default="local_support")
    parser.add_argument("--same-traj-support-column", default="same_traj_support")
    parser.add_argument("--positive-support-discount", type=float, default=0.5)
    parser.add_argument("--min-support-relief", type=float, default=0.25)
    parser.add_argument("--supported-edge-risk-cap", type=float, default=1.0)
    args = parser.parse_args()
    make_adaptive_scores(args)


if __name__ == "__main__":
    main()
