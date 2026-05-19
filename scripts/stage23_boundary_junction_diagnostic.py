#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bars.gas_bars.bridge_boundary import boundary_junction_metrics, filter_bridge_junctions, save_bridge_junctions, synthesize_bridge_junctions
from bars.gas_bars.bridge_graph import load_bridge_graph


def _merge_report(existing_path: Path, df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    if existing_path.exists() and len(df):
        try:
            old = pd.read_csv(existing_path)
            if set(key_cols).issubset(old.columns) and set(key_cols).issubset(df.columns):
                old_key = old[key_cols].astype(str).agg("\t".join, axis=1)
                new_key = df[key_cols].astype(str).agg("\t".join, axis=1)
                old = old.loc[~old_key.isin(set(new_key))].copy()
                df = pd.concat([old, df], ignore_index=True)
                df = df.sort_values(key_cols).reset_index(drop=True)
        except Exception:
            pass
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--envs", required=True)
    p.add_argument("--seeds", default="0")
    p.add_argument("--artifact-root", default="artifacts/stage23")
    p.add_argument("--stage22-artifact-root", default="artifacts/stage22")
    p.add_argument("--graph-id", default="G3")
    p.add_argument("--reports-root", default="reports")
    args = p.parse_args()
    rows = []
    for env in [x for x in args.envs.split(",") if x]:
        for seed in [int(x) for x in args.seeds.split(",") if x]:
            root = Path(args.artifact_root) / env / f"seed{seed}"
            graph_path = root / "bridge_graphs" / f"{args.graph_id}.pkl"
            if not graph_path.exists():
                continue
            graph = load_bridge_graph(graph_path)
            boundary_path = Path(args.stage22_artifact_root) / env / f"seed{seed}" / "boundary_scores.csv"
            if not boundary_path.exists():
                boundary_path = root / "boundary_scores.csv"
            boundary = pd.read_csv(boundary_path) if boundary_path.exists() else pd.DataFrame()
            edge_exec_path = root / "edge_exec" / "edge_exec_results.csv"
            edge_exec = pd.read_csv(edge_exec_path) if edge_exec_path.exists() else None
            junctions = filter_bridge_junctions(boundary, graph.edges) if len(boundary) else synthesize_bridge_junctions(graph.edges)
            metrics = boundary_junction_metrics(junctions, edge_exec=edge_exec)
            metrics["env"] = env
            metrics["seed"] = seed
            rows.append(metrics)
            save_bridge_junctions(junctions, metrics, root / "boundary_junction")
    reports = Path(args.reports_root)
    reports.mkdir(parents=True, exist_ok=True)
    df = _merge_report(reports / "stage23_boundary_junction_metrics.csv", pd.DataFrame(rows), ["env", "seed"])
    df.to_csv(reports / "stage23_boundary_junction_metrics.csv", index=False)


if __name__ == "__main__":
    main()
