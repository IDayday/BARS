#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bars.external.gas_artifacts import resolve_gas_artifacts
from bars.external.gas_backbone import GASBackbone
from bars.gas_bars.bridge_graph import load_bridge_graph
from bars.gas_bars.edge_execution import run_edge_execution


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--envs", required=True)
    p.add_argument("--seeds", default="0")
    p.add_argument("--graph-id", default="G3")
    p.add_argument("--gas-artifact-root", default="artifacts/gas")
    p.add_argument("--artifact-root", default="artifacts/stage23")
    p.add_argument("--gas-repo-path", default="external_src/GAS")
    p.add_argument("--reports-root", default="reports")
    p.add_argument("--gpu", default="cpu")
    p.add_argument("--local-n", type=int, default=200)
    p.add_argument("--bridge-n", type=int, default=300)
    p.add_argument("--pilot", type=int, default=1)
    p.add_argument("--prefer-pretrained", type=int, default=1)
    p.add_argument("--train-if-missing", type=int, default=0)
    args = p.parse_args()
    if args.pilot:
        args.local_n = min(args.local_n, 200)
        args.bridge_n = min(args.bridge_n, 300)
    summaries = []
    for env in [x for x in args.envs.split(",") if x]:
        for seed in [int(x) for x in args.seeds.split(",") if x]:
            artifacts = resolve_gas_artifacts(env, seed, args.gas_artifact_root)
            bb = GASBackbone.load_or_train(
                env,
                seed,
                args.gas_artifact_root,
                args.gas_repo_path,
                args.gpu,
                prefer_pretrained=bool(args.prefer_pretrained),
                train_if_missing=bool(args.train_if_missing),
                quick=True,
            )
            graph_path = Path(args.artifact_root) / env / f"seed{seed}" / "bridge_graphs" / f"{args.graph_id}.pkl"
            graph = load_bridge_graph(graph_path).to_dict()
            out_csv = Path(args.artifact_root) / env / f"seed{seed}" / "edge_exec" / "edge_exec_results.csv"
            df = run_edge_execution(bb, graph, out_csv, local_n=args.local_n, bridge_n=args.bridge_n, random_state=seed)
            by_type = df.groupby("edge_type", dropna=False).agg(edges=("success", "count"), success_rate=("success", "mean"), set_state_rate=("reset_mode", lambda x: float((x.astype(str) == "set_state").mean()))).reset_index()
            by_type.insert(0, "seed", seed)
            by_type.insert(0, "env", env)
            summaries.append(by_type)
    report_df = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    reports = Path(args.reports_root)
    reports.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(reports / "stage23_edge_execution_summary.csv", index=False)
    lines = ["# Stage23 Edge Execution", ""]
    if len(report_df):
        try:
            lines.append(report_df.to_markdown(index=False))
        except Exception:
            lines.append("```csv\n" + report_df.to_csv(index=False).strip() + "\n```")
        weak = report_df["set_state_rate"].max() < 0.5 if "set_state_rate" in report_df else True
        lines.append("")
        lines.append(f"- Evidence strength: {'weak proxy only' if weak else 'rollout labels available'}.")
    else:
        lines.append("No edge execution rows generated.")
    (reports / "stage23_edge_execution.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
