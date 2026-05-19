#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bars.external.gas_artifacts import resolve_gas_artifacts
from bars.gas_bars.bridge_graph import analyze_bridge_graphs, build_bridge_graphs, save_bridge_graphs
from bars.gas_bars.graph_table import export_edges, export_nodes, load_gas_keygraph, save_edge_table


def build_one(env: str, seed: int, args: argparse.Namespace) -> pd.DataFrame:
    artifacts = resolve_gas_artifacts(env, seed, args.gas_artifact_root)
    if artifacts.keygraph is None:
        raise FileNotFoundError(f"missing keygraph for {env} seed{seed}")
    kg = load_gas_keygraph(artifacts.keygraph)
    nodes = export_nodes(kg)
    edges = export_edges(kg)
    out_dir = Path(args.artifact_root) / env / f"seed{seed}" / "bridge_graphs"
    save_edge_table(nodes, edges, out_dir, stem="gas_graph")
    graphs = build_bridge_graphs(nodes, edges, way_steps=getattr(kg, "way_steps", None), top_k_bridge=args.top_k_bridge, random_state=seed)
    save_bridge_graphs(graphs, out_dir)
    summary = analyze_bridge_graphs(graphs, max_sources=args.max_sources, random_state=seed)
    summary.insert(0, "seed", seed)
    summary.insert(0, "env", env)
    summary.to_csv(out_dir / "bridge_graph_summary.csv", index=False)
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--envs", required=True)
    p.add_argument("--seeds", default="0")
    p.add_argument("--gas-artifact-root", default="artifacts/gas")
    p.add_argument("--artifact-root", default="artifacts/stage23")
    p.add_argument("--reports-root", default="reports")
    p.add_argument("--top-k-bridge", type=int, default=4)
    p.add_argument("--max-sources", type=int, default=200)
    args = p.parse_args()
    rows = []
    for env in [x for x in args.envs.split(",") if x]:
        for seed in [int(x) for x in args.seeds.split(",") if x]:
            rows.append(build_one(env, seed, args))
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    reports = Path(args.reports_root)
    reports.mkdir(parents=True, exist_ok=True)
    df.to_csv(reports / "stage23_bridge_graph_summary.csv", index=False)
    lines = ["# Stage23 Bridge Graph Summary", ""]
    if len(df):
        try:
            lines.append(df.to_markdown(index=False))
        except Exception:
            lines.append("```csv\n" + df.to_csv(index=False).strip() + "\n```")
        headroom = df[(df["graph_id"] != "G0") & ((df["shorter_path_rate"] > 0) | (df["bridge_usage_rate"] > 0))]
        lines.append("")
        lines.append(f"- Bridge existence gate: {'PASS_BRIDGE_EXISTENCE' if len(headroom) else 'NO_BRIDGE_HEADROOM'}")
    else:
        lines.append("No bridge graphs were built.")
    (reports / "stage23_bridge_graph_summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
