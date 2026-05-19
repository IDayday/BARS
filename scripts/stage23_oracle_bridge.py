#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bars.gas_bars.bridge_graph import load_bridge_graph
from bars.gas_bars.oracle_bridge import build_oracle_bridge_graph, oracle_summary, save_oracle_graph


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--envs", required=True)
    p.add_argument("--seeds", default="0")
    p.add_argument("--graph-id", default="G3")
    p.add_argument("--artifact-root", default="artifacts/stage23")
    p.add_argument("--reports-root", default="reports")
    args = p.parse_args()
    rows = []
    for env in [x for x in args.envs.split(",") if x]:
        for seed in [int(x) for x in args.seeds.split(",") if x]:
            graph_dir = Path(args.artifact_root) / env / f"seed{seed}" / "bridge_graphs"
            edge_exec = Path(args.artifact_root) / env / f"seed{seed}" / "edge_exec" / "edge_exec_results.csv"
            if not edge_exec.exists():
                continue
            g0 = load_bridge_graph(graph_dir / "G0.pkl")
            gx = load_bridge_graph(graph_dir / f"{args.graph_id}.pkl")
            oracle = build_oracle_bridge_graph(gx, pd.read_csv(edge_exec))
            save_oracle_graph(oracle, graph_dir / "G_oracle.pkl")
            df = oracle_summary(g0, gx, oracle)
            df.insert(0, "seed", seed)
            df.insert(0, "env", env)
            rows.append(df)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    reports = Path(args.reports_root)
    reports.mkdir(parents=True, exist_ok=True)
    out.to_csv(reports / "stage23_oracle_bridge_summary.csv", index=False)


if __name__ == "__main__":
    main()
