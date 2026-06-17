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
        if (path / "caplite_edge_scores.csv").exists():
            path = path / "caplite_edge_scores.csv"
        elif (path / "contract_scored_rows.csv").exists():
            path = path / "contract_scored_rows.csv"
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


def _aggregate_sequence_rows(rows: pd.DataFrame, env: str) -> pd.DataFrame:
    if env:
        rows = rows[rows["env"].astype(str) == env].copy()
    rows = rows[_num(rows, "edge_forward", 0).astype(int) == 1].copy()
    if rows.empty:
        raise RuntimeError("No forward sequence rows after filtering")
    rows["u"] = _num(rows, "edge_u", 0).astype(int)
    rows["v"] = _num(rows, "edge_v", 0).astype(int)
    rows["seq_contract_prob_mean"] = _num(rows, "contract_prob", 0.0).clip(0.0, 1.0)
    rows["seq_has_segment"] = _num(rows, "seq_has_segment", 0.0).clip(0.0, 1.0)
    grouped = rows.groupby(["env", "seed", "edge_id", "u", "v"], dropna=False).agg(
        seq_rows=("seq_contract_prob_mean", "size"),
        seq_contract_prob_mean=("seq_contract_prob_mean", "mean"),
        seq_contract_prob_max=("seq_contract_prob_mean", "max"),
        seq_segment_rows=("seq_has_segment", "sum"),
        seq_segment_fraction=("seq_has_segment", "mean"),
        seq_actor_mse_mean=("seq_actor_action_mse_mean", "mean") if "seq_actor_action_mse_mean" in rows else ("seq_contract_prob_mean", "size"),
        seq_progress_delta_mean=("seq_mean_progress_delta", "mean") if "seq_mean_progress_delta" in rows else ("seq_contract_prob_mean", "size"),
    )
    return grouped.reset_index()


def make_hybrid(args: argparse.Namespace) -> dict[str, Any]:
    base = _read_table(Path(args.base_edge_scores)).copy()
    seq_rows = _read_table(Path(args.sequence_scored_rows)).copy()
    if args.env:
        base = base[base.get("env", args.env).astype(str) == args.env].copy() if "env" in base else base
    if base.empty:
        raise RuntimeError("No base edge scores after filtering")
    seq = _aggregate_sequence_rows(seq_rows, str(args.env))

    for col in ["edge_id", "u", "v"]:
        base[col] = _num(base, col, 0).astype(int)
    base["base_contract_prob_edge"] = _num(base, "contract_prob_edge", 0.0).clip(0.0, 1.0)
    merged = base.merge(seq, on=["edge_id", "u", "v"], how="left", suffixes=("", "_seq"))
    merged["seq_rows"] = _num(merged, "seq_rows", 0)
    merged["seq_segment_rows"] = _num(merged, "seq_segment_rows", 0)
    merged["seq_segment_fraction"] = _num(merged, "seq_segment_fraction", 0.0).clip(0.0, 1.0)
    merged["seq_contract_prob_mean"] = _num(merged, "seq_contract_prob_mean", 0.0).clip(0.0, 1.0)
    merged["seq_contract_prob_max"] = _num(merged, "seq_contract_prob_max", 0.0).clip(0.0, 1.0)

    min_rows = max(float(args.min_sequence_segment_rows), 1e-6)
    evidence_gate = (merged["seq_segment_rows"] / min_rows).clip(0.0, 1.0) * merged["seq_segment_fraction"]
    evidence_gate = evidence_gate.clip(0.0, float(args.max_sequence_gate))
    seq_prob = (1.0 - float(args.sequence_max_mix)) * merged["seq_contract_prob_mean"] + float(args.sequence_max_mix) * merged["seq_contract_prob_max"]
    seq_prob = seq_prob.clip(0.0, 1.0)

    penalty = (1.0 - seq_prob).clip(0.0, 1.0)
    strength = float(args.sequence_penalty_strength)
    hybrid = merged["base_contract_prob_edge"] * (1.0 - strength * evidence_gate * penalty)
    if float(args.sequence_boost_strength) > 0:
        boost = float(args.sequence_boost_strength) * evidence_gate * np.maximum(seq_prob - merged["base_contract_prob_edge"], 0.0)
        hybrid = hybrid + boost
    max_drop = float(args.max_prob_drop)
    if max_drop < 1.0:
        hybrid = np.maximum(hybrid, merged["base_contract_prob_edge"] - max(0.0, max_drop))
    if int(args.preserve_base_supported_prob):
        base_supported = merged["base_contract_prob_edge"] >= float(args.contract_threshold)
        hybrid = np.where(base_supported, np.maximum(hybrid, merged["base_contract_prob_edge"]), hybrid)
    merged["sequence_evidence_gate"] = evidence_gate.astype(np.float32)
    merged["sequence_contract_prob_edge"] = seq_prob.astype(np.float32)
    merged["contract_prob_edge"] = hybrid.clip(0.0, 1.0).astype(np.float32)
    merged["contract_risk"] = (1.0 - merged["contract_prob_edge"]).clip(0.0, 1.0)
    merged["p_exec"] = merged["contract_prob_edge"]
    merged["r_exec"] = merged["contract_risk"]
    merged["contract_support"] = (merged["contract_prob_edge"] >= float(args.contract_threshold)).astype(np.int32)
    merged["contract_available"] = 1
    merged["support_penalty"] = np.where(merged["contract_support"] > 0, 0.0, merged["contract_risk"])

    out_cols = list(dict.fromkeys(list(base.columns) + [
        "base_contract_prob_edge",
        "sequence_contract_prob_edge",
        "sequence_evidence_gate",
        "seq_rows",
        "seq_segment_rows",
        "seq_segment_fraction",
        "seq_actor_mse_mean",
        "seq_progress_delta_mean",
        "contract_prob_edge",
        "contract_risk",
        "p_exec",
        "r_exec",
        "contract_support",
        "contract_available",
        "support_penalty",
    ]))
    out = merged[out_cols].sort_values(["u", "v", "edge_id"]).reset_index(drop=True)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "caplite_edge_scores.csv"
    out.to_csv(csv_path, index=False)
    try:
        out.to_parquet(out_dir / "caplite_edge_scores.parquet", index=False)
    except Exception:
        pass

    changed = np.abs(out["contract_prob_edge"] - out["base_contract_prob_edge"])
    summary: dict[str, Any] = {
        "offline_only": True,
        "uses_environment_rollout": False,
        "env": args.env or "all",
        "base_edge_scores": str(args.base_edge_scores),
        "sequence_scored_rows": str(args.sequence_scored_rows),
        "edge_scores_csv": str(csv_path),
        "num_edges": int(len(out)),
        "mean_base_prob": float(out["base_contract_prob_edge"].mean()) if len(out) else 0.0,
        "mean_sequence_prob": float(out["sequence_contract_prob_edge"].mean()) if len(out) else 0.0,
        "mean_hybrid_prob": float(out["contract_prob_edge"].mean()) if len(out) else 0.0,
        "median_hybrid_prob": float(out["contract_prob_edge"].median()) if len(out) else 0.0,
        "supported_edge_rate": float(out["contract_support"].mean()) if len(out) else 0.0,
        "mean_sequence_gate": float(out["sequence_evidence_gate"].mean()) if len(out) else 0.0,
        "edge_changed_rate": float((changed > 1e-9).mean()) if len(out) else 0.0,
        "mean_abs_prob_delta": float(changed.mean()) if len(out) else 0.0,
        "sequence_penalty_strength": float(args.sequence_penalty_strength),
        "sequence_boost_strength": float(args.sequence_boost_strength),
        "max_prob_drop": float(args.max_prob_drop),
        "preserve_base_supported_prob": int(args.preserve_base_supported_prob),
        "min_sequence_segment_rows": float(args.min_sequence_segment_rows),
    }
    summary_path = out_dir / "caplite_edge_scores_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compose broad Stage45 contract scores with Stage49 sequence verifier scores.")
    parser.add_argument("--base-edge-scores", required=True)
    parser.add_argument("--sequence-scored-rows", required=True)
    parser.add_argument("--env", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--sequence-penalty-strength", type=float, default=0.5)
    parser.add_argument("--sequence-boost-strength", type=float, default=0.0)
    parser.add_argument("--sequence-max-mix", type=float, default=0.25)
    parser.add_argument("--min-sequence-segment-rows", type=float, default=1.0)
    parser.add_argument("--max-sequence-gate", type=float, default=1.0)
    parser.add_argument("--max-prob-drop", type=float, default=1.0)
    parser.add_argument("--preserve-base-supported-prob", type=int, default=0)
    parser.add_argument("--contract-threshold", type=float, default=0.5)
    args = parser.parse_args()
    make_hybrid(args)


if __name__ == "__main__":
    main()
