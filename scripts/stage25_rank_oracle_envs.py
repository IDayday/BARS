#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ORACLE_COLUMNS = [
    "env",
    "seed",
    "graph_id",
    "node_count",
    "edge_count",
    "bridge_count",
    "shorter_path_rate",
    "bridge_usage_rate",
    "mean_path_cost_reduction",
    "oracle_bridge_count",
    "oracle_bridge_fraction",
    "oracle_shorter_path_rate",
    "oracle_bridge_usage_rate",
    "oracle_mean_path_cost_reduction",
    "useful_bridge_score",
    "safe_local_success_rate",
    "risky_bridge_success_rate",
    "gas_cross_success_rate",
    "set_state_rate",
    "gate",
    "failure_reason",
]


def _read(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _num(value, default=np.nan) -> float:
    try:
        out = float(value)
        return default if np.isnan(out) else out
    except Exception:
        return default


def _weighted(edge: pd.DataFrame, env: str, seed: int, edge_types: list[str]) -> tuple[float, float]:
    if len(edge) == 0:
        return np.nan, np.nan
    sub = edge[
        edge.get("env", "").astype(str).eq(str(env))
        & pd.to_numeric(edge.get("seed", -1), errors="coerce").fillna(-1).astype(int).eq(int(seed))
        & edge.get("edge_type", "").astype(str).isin(edge_types)
    ]
    if len(sub) == 0:
        return np.nan, np.nan
    weights = pd.to_numeric(sub.get("edges", 1), errors="coerce").fillna(1).to_numpy(float)
    success = pd.to_numeric(sub.get("success_rate", np.nan), errors="coerce").to_numpy(float)
    set_state = pd.to_numeric(sub.get("set_state_rate", np.nan), errors="coerce").to_numpy(float)
    return float(np.average(success, weights=weights)), float(np.average(set_state, weights=weights))


def build_ranking(root: Path) -> pd.DataFrame:
    bridge = _read(root / "stage23_bridge_graph_summary.csv")
    edge = _read(root / "stage23_edge_execution_summary.csv")
    oracle = _read(root / "stage23_oracle_bridge_summary.csv")
    if len(bridge) == 0:
        return pd.DataFrame(columns=ORACLE_COLUMNS)
    rows = []
    for _, row in bridge[bridge.get("graph_id", "").astype(str).ne("G0")].iterrows():
        env = str(row.get("env", ""))
        seed = int(_num(row.get("seed", 0), 0))
        graph_id = str(row.get("graph_id", ""))
        orows = oracle[
            oracle.get("env", "").astype(str).eq(env)
            & pd.to_numeric(oracle.get("seed", -1), errors="coerce").fillna(-1).astype(int).eq(seed)
            & oracle.get("graph_id", "").astype(str).eq("G_oracle")
        ]
        if len(orows) == 0:
            orows = oracle[
                oracle.get("env", "").astype(str).eq(env)
                & pd.to_numeric(oracle.get("seed", -1), errors="coerce").fillna(-1).astype(int).eq(seed)
                & oracle.get("graph_id", "").astype(str).eq(graph_id)
            ]
        orow = orows.iloc[0] if len(orows) else pd.Series(dtype=object)
        safe, safe_set = _weighted(edge, env, seed, ["safe_local"])
        risky, risky_set = _weighted(edge, env, seed, ["aggressive_tdr_bridge", "bottleneck_bridge"])
        gas_cross, gas_set = _weighted(edge, env, seed, ["gas_cross"])
        set_state = np.nanmax([safe_set, risky_set, gas_set]) if any(np.isfinite(x) for x in [safe_set, risky_set, gas_set]) else np.nan
        oracle_usage = _num(orow.get("bridge_usage_rate", np.nan), np.nan)
        oracle_reduction = _num(orow.get("mean_path_cost_reduction", np.nan), np.nan)
        useful = oracle_usage * oracle_reduction if np.isfinite(oracle_usage) and np.isfinite(oracle_reduction) else np.nan
        failure = []
        if _num(set_state, np.nan) < 0.95:
            failure.append("set_state_rate")
        if _num(safe, np.nan) < 0.85:
            failure.append("safe_local_success_rate")
        if _num(orow.get("bridge_count", np.nan), np.nan) < 50:
            failure.append("oracle_bridge_count")
        if _num(orow.get("bridge_usage_rate", np.nan), np.nan) < 0.20:
            failure.append("oracle_bridge_usage_rate")
        if not (_num(orow.get("shorter_path_rate", np.nan), np.nan) >= 0.20 or _num(orow.get("mean_path_cost_reduction", np.nan), np.nan) >= 1.0):
            failure.append("oracle_path_reduction")
        if _num(useful, np.nan) < 0.20:
            failure.append("useful_bridge_score")
        rows.append(
            {
                "env": env,
                "seed": seed,
                "graph_id": graph_id,
                "node_count": row.get("node_count", np.nan),
                "edge_count": row.get("edge_count", np.nan),
                "bridge_count": row.get("bridge_count", np.nan),
                "shorter_path_rate": row.get("shorter_path_rate", np.nan),
                "bridge_usage_rate": row.get("bridge_usage_rate", np.nan),
                "mean_path_cost_reduction": row.get("mean_path_cost_reduction", np.nan),
                "oracle_bridge_count": orow.get("bridge_count", np.nan),
                "oracle_bridge_fraction": _num(orow.get("bridge_count", np.nan), 0.0) / max(_num(row.get("bridge_count", 0), 0.0), 1e-9),
                "oracle_shorter_path_rate": orow.get("shorter_path_rate", np.nan),
                "oracle_bridge_usage_rate": oracle_usage,
                "oracle_mean_path_cost_reduction": oracle_reduction,
                "useful_bridge_score": useful,
                "safe_local_success_rate": safe,
                "risky_bridge_success_rate": risky,
                "gas_cross_success_rate": gas_cross,
                "set_state_rate": set_state,
                "gate": "PASS_ORACLE_HEADROOM" if not failure else "NO_ORACLE_UPPER_BOUND",
                "failure_reason": "|".join(failure),
            }
        )
    out = pd.DataFrame(rows)
    for col in ORACLE_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out = out[ORACLE_COLUMNS]
    pass_rank = out["gate"].astype(str).eq("PASS_ORACLE_HEADROOM").astype(int)
    out = out.assign(_pass_rank=pass_rank).sort_values(
        ["_pass_rank", "useful_bridge_score", "oracle_shorter_path_rate", "oracle_bridge_count"],
        ascending=[False, False, False, False],
        na_position="last",
    )
    return out.drop(columns=["_pass_rank"]).reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reports-root", default="reports/stage25_oracle_scan_tmp")
    p.add_argument("--out", default="")
    p.add_argument("--out-md", default="")
    args = p.parse_args()
    root = Path(args.reports_root)
    out_path = Path(args.out) if args.out else root / "stage25_oracle_env_ranking.csv"
    md_path = Path(args.out_md) if args.out_md else root / "stage25_oracle_env_ranking.md"
    df = build_ranking(root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    lines = ["# Stage25 Oracle Env Ranking", ""]
    if len(df):
        try:
            lines.append(df.to_markdown(index=False))
        except Exception:
            lines.append("```csv\n" + df.to_csv(index=False).strip() + "\n```")
    else:
        lines.append("No completed oracle rows were available.")
    md_path.write_text("\n".join(lines) + "\n")
    print(f"[stage25_oracle_rank] rows={len(df)} out={out_path}")


if __name__ == "__main__":
    main()
