#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external_src" / "GAS"))

from cage.contract_graph import ContractEdge, ContractFunnelNode, ContractGraph
from cage.contract_model import ContractScorer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Augment a CAGE ECG graph with final/recovery contract candidates.")
    parser.add_argument("--contract_graph_path", default="results/cage_ecg/transition_contract_graph/contract_graph.json")
    parser.add_argument("--contract_dataset_path", default="results/cage_clp1/datasets/closed_loop_contracts.jsonl")
    parser.add_argument("--oracle_summary", default="results/cage_clp1/oracle_candidate/contract_oracle_summary.json")
    parser.add_argument("--segment_capture_roots", nargs="*", default=["results/cage_clp1/segment_capture", "results/cage_clp1/segment_capture_candidate"])
    parser.add_argument("--contract_model_path", default="results/cage_v02_contract/models/contract_model.json")
    parser.add_argument("--out_graph_path", default="results/cage_ecg/transition_contract_graph/contract_graph_augmented.json")
    parser.add_argument("--out_report", default="reports/stage36_final_recovery_contract_augmentation.md")
    parser.add_argument("--recovery_underpowered_threshold", type=int, default=20)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    graph = ContractGraph.load_json(args.contract_graph_path)
    scorer = ContractScorer.from_path(args.contract_model_path, uncertainty_penalty=0.25)
    key_to_id = {endpoint_key(node.center_phi, node.env_name or "unknown"): node.node_id for node in graph.nodes.values()}
    edge_keys = {(edge.src, edge.dst, edge.edge_type or "") for edge in graph.edges.values()}
    added_counts: Counter[str] = Counter()
    skipped = Counter()

    for row in load_jsonl(Path(args.contract_dataset_path)):
        mode = str(row.get("target_mode") or row.get("pair_source") or "").lower()
        final = bool(row.get("final_phase")) or "final" in mode
        recovery = bool(row.get("recovery_candidate")) or "recovery" in mode
        if not final and not recovery:
            continue
        edge_type = "final_goal_candidate" if final else "recovery_candidate"
        if add_edge(graph, key_to_id, edge_keys, scorer, row.get("phi_s", row.get("phi_start")), row.get("phi_g", row.get("phi_target")), str(row.get("env_name") or "unknown"), edge_type):
            added_counts[edge_type] += 1
        else:
            skipped[edge_type] += 1

    for path in segment_files(args.segment_capture_roots or []):
        for row in load_jsonl(path):
            env = str(row.get("env_name") or "unknown")
            if row.get("start_phi") is not None and row.get("final_goal_phi") is not None:
                if add_edge(graph, key_to_id, edge_keys, scorer, row.get("start_phi"), row.get("final_goal_phi"), env, "final_goal_candidate"):
                    added_counts["final_goal_candidate"] += 1
                else:
                    skipped["final_goal_candidate"] += 1
            if row.get("recovery_active") or "recovery" in str(row.get("target_source") or "").lower():
                if add_edge(graph, key_to_id, edge_keys, scorer, row.get("start_phi"), row.get("target_phi"), env, "recovery_candidate"):
                    added_counts["recovery_candidate"] += 1
                else:
                    skipped["recovery_candidate"] += 1

    graph.metadata = {
        **dict(graph.metadata),
        "final_recovery_augmentation": {
            "script": "augment_final_recovery_contracts.py",
            "contract_model_loaded": bool(scorer.loaded),
            "added_counts": dict(added_counts),
            "skipped_counts": dict(skipped),
            "oracle_summary_present": Path(args.oracle_summary).exists(),
            "oracle_notes": oracle_notes(Path(args.oracle_summary)),
        },
    }
    out_graph = Path(args.out_graph_path)
    graph.save_json(out_graph)
    graph.export_nodes_csv(out_graph.parent / "nodes_augmented.csv")
    graph.export_edges_csv(out_graph.parent / "edges_augmented.csv")
    graph.export_boundary_csv(out_graph.parent / "boundary_contracts_augmented.csv")
    summary = summarize_final_recovery(graph, int(args.recovery_underpowered_threshold))
    write_report(Path(args.out_report), summary, out_graph)
    print(json.dumps({"out_graph_path": str(out_graph), **summary}, sort_keys=True))
    return 0


def add_edge(
    graph: ContractGraph,
    key_to_id: dict[str, str],
    edge_keys: set[tuple[str, str, str]],
    scorer: ContractScorer,
    phi_s: Any,
    phi_g: Any,
    env_name: str,
    edge_type: str,
) -> bool:
    if phi_s is None or phi_g is None:
        return False
    src = node_for(graph, key_to_id, phi_s, env_name)
    dst = node_for(graph, key_to_id, phi_g, env_name)
    if src == dst:
        return False
    key = (src, dst, edge_type)
    if key in edge_keys:
        return False
    d_phi = float(np.linalg.norm(np.asarray(phi_g, dtype=np.float32).reshape(-1) - np.asarray(phi_s, dtype=np.float32).reshape(-1)))
    pred = scorer.predict(
        {
            "phi_s": phi_s,
            "phi_g": phi_g,
            "d_phi": d_phi,
            "target_mode": edge_type,
            "final_phase": edge_type == "final_goal_candidate",
            "recovery_candidate": edge_type == "recovery_candidate",
            "env_name": env_name,
            "fallback_tau": 10.0,
        }
    )
    edge_id = f"fr{len(graph.edges):08d}"
    negative = float(pred.predicted_negative_progress)
    lcb = float(pred.lower_confidence_bound)
    graph.edges[edge_id] = ContractEdge(
        edge_id=edge_id,
        src=src,
        dst=dst,
        d_phi=d_phi,
        gas_edge_exists=False,
        contract_score=float(pred.predicted_contract_positive),
        contract_lcb=lcb,
        predicted_hit=float(pred.predicted_hit),
        predicted_contract_positive=float(pred.predicted_contract_positive),
        predicted_negative_progress=negative,
        uncertainty=float(pred.uncertainty),
        q_train_support=None,
        edge_type=edge_type,
        bottleneck_score=min(lcb, 1.0 - negative),
    )
    edge_keys.add(key)
    return True


def node_for(graph: ContractGraph, key_to_id: dict[str, str], phi: Any, env_name: str) -> str:
    key = endpoint_key(phi, env_name)
    if key not in key_to_id:
        node_id = f"frn{len(key_to_id):07d}"
        key_to_id[key] = node_id
        graph.nodes[node_id] = ContractFunnelNode(
            node_id=node_id,
            center_phi=np.asarray(phi, dtype=np.float32).reshape(-1).astype(float).tolist(),
            radius=0.0,
            support_count=0,
            env_name=env_name,
            source_node_ids=[],
        )
    graph.nodes[key_to_id[key]].support_count += 1
    return key_to_id[key]


def summarize_final_recovery(graph: ContractGraph, recovery_threshold: int) -> dict[str, Any]:
    edges = list(graph.edges.values())
    final_edges = [edge for edge in edges if "final" in str(edge.edge_type or "").lower()]
    recovery_edges = [edge for edge in edges if "recovery" in str(edge.edge_type or "").lower()]
    recovery_rate = rate(len(recovery_edges), len(edges))
    summary = {
        "edge_count": len(edges),
        "final_goal_edge_count": len(final_edges),
        "final_goal_edge_rate": rate(len(final_edges), len(edges)),
        "recovery_edge_count": len(recovery_edges),
        "recovery_edge_rate": recovery_rate,
        "final_goal_positive_rate": positive_rate(final_edges),
        "recovery_positive_rate": positive_rate(recovery_edges),
        "final_goal_negative_rate": negative_rate(final_edges),
        "recovery_negative_rate": negative_rate(recovery_edges),
        "final_goal_status": "ok" if final_edges else "FINAL_GOAL_CONTRACT_MISSING",
        "recovery_status": "ok" if len(recovery_edges) >= recovery_threshold and (recovery_rate or 0.0) >= 0.01 else "RECOVERY_CONTRACT_UNDERPOWERED",
    }
    return summary


def positive_rate(edges: list[ContractEdge]) -> float | None:
    positives = [edge for edge in edges if _float(edge.contract_lcb, 0.0) >= 0.5 and _float(edge.predicted_negative_progress, 1.0) <= 0.25]
    return rate(len(positives), len(edges))


def negative_rate(edges: list[ContractEdge]) -> float | None:
    negatives = [edge for edge in edges if _float(edge.contract_lcb, 0.0) < 0.25 or _float(edge.predicted_negative_progress, 1.0) >= 0.5]
    return rate(len(negatives), len(edges))


def write_report(path: Path, summary: dict[str, Any], graph_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage36 Final/Recovery Contract Augmentation",
        "",
        f"Graph: `{graph_path}`",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key in [
        "edge_count",
        "final_goal_edge_count",
        "final_goal_edge_rate",
        "recovery_edge_count",
        "recovery_edge_rate",
        "final_goal_positive_rate",
        "recovery_positive_rate",
        "final_goal_negative_rate",
        "recovery_negative_rate",
    ]:
        lines.append(f"| {key} | {fmt(summary.get(key))} |")
    lines.extend(
        [
            "",
            "## Status",
            "",
            f"- final_goal_status: `{summary['final_goal_status']}`",
            f"- recovery_status: `{summary['recovery_status']}`",
            "",
            "这些是离线合同覆盖统计，不是 final/recovery 在线成功率；缺失或 underpowered 时不得虚构结果。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def oracle_notes(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing"}
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        return {
            "status": "present",
            "num_probe_rows": record.get("num_probe_rows"),
            "failure_modes": record.get("failure_modes", {}),
        }
    except Exception as exc:
        return {"status": "unreadable", "error": str(exc)}


def endpoint_key(phi: Any, env_name: str, decimals: int = 3) -> str:
    arr = np.asarray(phi, dtype=np.float32).reshape(-1)
    return f"{env_name}::" + ",".join(f"{x:.{decimals}f}" for x in np.round(arr, decimals=decimals))


def segment_files(roots: list[str]) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        path = Path(root)
        if path.is_file() and path.name.endswith("_segments.jsonl"):
            out.append(path)
        elif path.exists():
            out.extend(sorted(path.glob("*_segments.jsonl")))
    return out


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def rate(num: int, den: int) -> float | None:
    return num / den if den else None


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
