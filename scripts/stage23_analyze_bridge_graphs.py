#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bars.gas_bars.bridge_graph import analyze_bridge_graphs, load_bridge_graph


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--artifact-root", default="artifacts/stage23")
    p.add_argument("--reports-root", default="reports")
    args = p.parse_args()
    rows = []
    root = Path(args.artifact_root)
    for graph_dir in root.glob("*/seed*/bridge_graphs"):
        env = graph_dir.parent.parent.name
        seed = graph_dir.parent.name.replace("seed", "")
        graphs = {}
        for gid in ["G0", "G1", "G2", "G3", "G_oracle"]:
            path = graph_dir / f"{gid}.pkl"
            if path.exists():
                graphs[gid] = load_bridge_graph(path)
        if "G0" in graphs and len(graphs) > 1:
            df = analyze_bridge_graphs(graphs)
            df.insert(0, "seed", int(seed))
            df.insert(0, "env", env)
            rows.append(df)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    reports = Path(args.reports_root)
    reports.mkdir(parents=True, exist_ok=True)
    out.to_csv(reports / "stage23_bridge_graph_summary.csv", index=False)
    lines = ["# Stage23 Bridge Graph Summary", ""]
    if len(out):
        try:
            lines.append(out.to_markdown(index=False))
        except Exception:
            lines.append("```csv\n" + out.to_csv(index=False).strip() + "\n```")
    else:
        lines.append("No bridge graph summaries found.")
    (reports / "stage23_bridge_graph_summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
