#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "external_src" / "GAS"))

from cage.contract_graph import ContractEdge, ContractFunnelNode, ContractGraph  # noqa: E402
from cage.contract_model import ContractScorer  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an action-anchored ECG graph from offline contract positives.")
    parser.add_argument("--action_contract_dataset", required=True)
    parser.add_argument("--contract_model", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_edges", type=int, default=100000)
    parser.add_argument("--min_contract_lcb", type=float, default=0.0)
    parser.add_argument("--clear", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    if args.clear and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scorer = ContractScorer.from_path(args.contract_model)
    graph, counters = build_graph(Path(args.action_contract_dataset), scorer, max_edges=int(args.max_edges), min_contract_lcb=float(args.min_contract_lcb))
    graph.metadata.update(
        {
            "stage": "stage38_action_anchored_graph_v2",
            "contract_model": str(args.contract_model),
            "action_contract_dataset": str(args.action_contract_dataset),
            **counters,
        }
    )
    graph.save_json(out_dir / "contract_graph.json")
    graph.export_nodes_csv(out_dir / "nodes.csv")
    graph.export_edges_csv(out_dir / "edges.csv")
    graph.export_boundary_csv(out_dir / "boundary_contracts.csv")
    summary = graph.summarize()
    report_path = REPO_ROOT / "reports" / "stage38_action_anchored_graph_v2.md"
    write_report(report_path, summary, counters, out_dir)
    print(json.dumps({"status": counters["status"], "graph": str(out_dir / "contract_graph.json"), "edge_count": len(graph.edges)}, sort_keys=True))
    return 0 if counters["status"] == "ACTION_ANCHORED_GRAPH_READY" else 2


def build_graph(dataset_path: Path, scorer: ContractScorer, *, max_edges: int, min_contract_lcb: float) -> tuple[ContractGraph, dict[str, Any]]:
    nodes: dict[str, ContractFunnelNode] = {}
    edges: dict[str, ContractEdge] = {}
    counters = {
        "source_rows_seen": 0,
        "positive_rows_seen": 0,
        "negative_rows_skipped": 0,
        "final_goal_edge_count": 0,
        "action_anchored_edge_count": 0,
        "filtered_low_contract_count": 0,
        "unverified_knn_main_edge_count": 0,
    }
    with dataset_path.open(encoding="utf-8") as fh:
        for line in fh:
            if len(edges) >= max_edges:
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            counters["source_rows_seen"] += 1
            if not rec.get("label_positive_contract") or not rec.get("action_available"):
                counters["negative_rows_skipped"] += 1
                continue
            counters["positive_rows_seen"] += 1
            phi_s = np.asarray(rec["phi_s"], dtype=np.float32)
            phi_g = np.asarray(rec["phi_g"], dtype=np.float32)
            pred = scorer.predict(
                {
                    "phi_s": phi_s,
                    "phi_g": phi_g,
                    "d_phi": rec.get("d_phi"),
                    "target_mode": rec.get("target_mode"),
                    "env_name": rec.get("env_name"),
                    "final_phase": bool(rec.get("label_final_goal")),
                }
            )
            if pred.lower_confidence_bound < min_contract_lcb:
                counters["filtered_low_contract_count"] += 1
                continue
            src = node_id(rec.get("env_name"), phi_s)
            dst = node_id(rec.get("env_name"), phi_g)
            nodes.setdefault(src, make_node(src, phi_s, rec.get("env_name")))
            nodes.setdefault(dst, make_node(dst, phi_g, rec.get("env_name")))
            nodes[src].support_count += 1
            nodes[dst].support_count += 1
            edge_id = f"aa_edge_{len(edges):08d}"
            edge_type = "final_goal_positive" if rec.get("label_final_goal") else "offline_temporal_future_positive"
            if rec.get("label_final_goal"):
                counters["final_goal_edge_count"] += 1
            counters["action_anchored_edge_count"] += 1
            edges[edge_id] = ContractEdge(
                edge_id=edge_id,
                src=src,
                dst=dst,
                d_phi=float(np.linalg.norm(phi_g - phi_s)),
                gas_edge_exists=False,
                contract_score=float(pred.predicted_contract_positive),
                contract_lcb=float(pred.lower_confidence_bound),
                predicted_hit=float(pred.predicted_hit),
                predicted_contract_positive=float(pred.predicted_contract_positive),
                predicted_negative_progress=float(pred.predicted_negative_progress),
                uncertainty=float(pred.uncertainty),
                edge_type=edge_type,
                bottleneck_score=float(min(pred.lower_confidence_bound, 1.0 - pred.predicted_negative_progress)),
                action_anchored=True,
                horizon=int(rec.get("horizon", -1) or -1),
                action_source=str(rec.get("action_source", "offline_dataset")),
                trust_level="action_anchored_observed",
            )
    graph = ContractGraph(nodes=nodes, edges=edges, boundary_contracts={}, metadata={})
    counters["node_count"] = len(nodes)
    counters["edge_count"] = len(edges)
    counters["action_anchored_edge_rate"] = safe_rate(counters["action_anchored_edge_count"], len(edges))
    counters["final_goal_edge_rate"] = safe_rate(counters["final_goal_edge_count"], len(edges))
    counters["knn_main_edge_rate"] = safe_rate(counters["unverified_knn_main_edge_count"], len(edges))
    counters["status"] = (
        "ACTION_ANCHORED_GRAPH_READY"
        if len(edges) > 0 and counters["action_anchored_edge_count"] > 0 and counters["final_goal_edge_count"] > 0
        else "ACTION_ANCHORED_GRAPH_UNDERPOWERED"
    )
    return graph, counters


def node_id(env_name: str | None, phi: np.ndarray) -> str:
    rounded = np.round(phi.astype(float), 2)
    key = "_".join(f"{x:.2f}" for x in rounded[:8])
    return f"{str(env_name or 'env')}::{key}::{len(rounded)}"


def make_node(node_id_: str, phi: np.ndarray, env_name: str | None) -> ContractFunnelNode:
    return ContractFunnelNode(node_id=node_id_, center_phi=phi.astype(float).round(6).tolist(), radius=0.0, support_count=0, env_name=env_name)


def safe_rate(num: int, den: int) -> float | None:
    return float(num / den) if den else None


def write_report(path: Path, summary: dict[str, Any], counters: dict[str, Any], out_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage38 Action-Anchored ECG Graph v2",
        "",
        f"- status: `{counters.get('status')}`",
        f"- graph: `{out_dir / 'contract_graph.json'}`",
        f"- node_count: {summary.get('node_count')}",
        f"- edge_count: {summary.get('edge_count')}",
        f"- action_anchored_edge_rate: {counters.get('action_anchored_edge_rate')}",
        f"- final_goal_edge_count: {counters.get('final_goal_edge_count')}",
        f"- final_goal_edge_rate: {counters.get('final_goal_edge_rate')}",
        f"- unverified_knn_main_edge_count: {counters.get('unverified_knn_main_edge_count')}",
        f"- knn_main_edge_rate: {counters.get('knn_main_edge_rate')}",
        "",
        "主执行边只来自 offline temporal/final positive action-anchored samples；未验证 KNN bridge 没有进入主 planner edge。",
        "",
        f"- edge_type_counts: `{summary.get('edge_type_counts')}`",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
