#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external_src" / "GAS"))

from cage.contract_graph import BoundaryContract, ContractEdge, ContractFunnelNode, ContractGraph
from cage.contract_model import ContractScorer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an offline CAGE execution-contract graph.")
    parser.add_argument("--contract_model_path", default="results/cage_v02_contract/models/contract_model.json")
    parser.add_argument("--contract_dataset_path", default="results/cage_clp1/datasets/closed_loop_contracts.jsonl")
    parser.add_argument("--out_dir", default="results/cage_ecg/contract_graph")
    parser.add_argument("--env_name", default="")
    parser.add_argument("--max_edges", type=int, default=0)
    parser.add_argument("--min_contract_lcb", type=float, default=0.0)
    parser.add_argument("--clear", action="store_true")
    return parser


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def endpoint_key(phi: Any, env_name: str, *, decimals: int = 3) -> str:
    arr = np.asarray(phi, dtype=np.float32).reshape(-1)
    rounded = np.round(arr, decimals=decimals)
    return f"{env_name}::" + ",".join(f"{x:.{decimals}f}" for x in rounded)


def add_node(nodes: dict[str, ContractFunnelNode], key_to_id: dict[str, str], phi: Any, env_name: str, source_id: str | None) -> str:
    key = endpoint_key(phi, env_name)
    if key not in key_to_id:
        node_id = f"n{len(key_to_id):06d}"
        key_to_id[key] = node_id
        nodes[node_id] = ContractFunnelNode(
            node_id=node_id,
            center_phi=np.asarray(phi, dtype=float).reshape(-1).tolist(),
            radius=0.0,
            support_count=0,
            env_name=env_name,
            source_node_ids=[],
        )
    node = nodes[key_to_id[key]]
    node.support_count += 1
    if source_id and source_id not in node.source_node_ids:
        node.source_node_ids.append(str(source_id))
    return node.node_id


def features_from_record(record: dict[str, Any]) -> dict[str, Any]:
    phi_s = record.get("phi_s", record.get("phi_start"))
    phi_g = record.get("phi_g", record.get("phi_target"))
    return {
        "phi_s": phi_s,
        "phi_g": phi_g,
        "d_phi": record.get("d_phi_start", record.get("d_phi")),
        "target_mode": record.get("target_mode", record.get("pair_source", "unknown")),
        "path_position": record.get("path_position", -1),
        "final_phase": bool(record.get("final_phase", False)),
        "recovery_candidate": bool(record.get("recovery_candidate", False)),
        "recent_stall_count": 0,
        "recent_drift_count": 0,
        "commitment_steps": 0,
        "previous_target_distance": -1.0,
        "current_target_distance": record.get("d_phi_start", record.get("d_phi", -1.0)),
        "q_train_support": record.get("q_train_support", -1.0),
        "env_name": record.get("env_name"),
        "fallback_tau": 10.0,
    }


def build_graph(rows: list[dict[str, Any]], scorer: ContractScorer, *, env_name: str = "", max_edges: int = 0, min_contract_lcb: float = 0.0) -> tuple[ContractGraph, dict[str, Any]]:
    nodes: dict[str, ContractFunnelNode] = {}
    key_to_id: dict[str, str] = {}
    edges: dict[str, ContractEdge] = {}
    by_segment: dict[str, list[str]] = defaultdict(list)

    filtered = [row for row in rows if not env_name or row.get("env_name") == env_name]
    if max_edges > 0:
        filtered = filtered[: int(max_edges)]

    skipped = 0
    for row in filtered:
        phi_s = row.get("phi_s", row.get("phi_start"))
        phi_g = row.get("phi_g", row.get("phi_target"))
        if phi_s is None or phi_g is None:
            skipped += 1
            continue
        env = str(row.get("env_name") or env_name or "unknown")
        features = features_from_record(row)
        pred = scorer.predict(features)
        if pred.lower_confidence_bound < float(min_contract_lcb):
            skipped += 1
            continue
        source_id = row.get("source_segment_id") or row.get("source_probe_id")
        src = add_node(nodes, key_to_id, phi_s, env, source_id)
        dst = add_node(nodes, key_to_id, phi_g, env, source_id)
        edge_id = f"e{len(edges):06d}"
        negative = float(pred.predicted_negative_progress)
        lcb = float(pred.lower_confidence_bound)
        edge_type = str(row.get("target_mode") or row.get("pair_source") or "unknown")
        edge = ContractEdge(
            edge_id=edge_id,
            src=src,
            dst=dst,
            d_phi=_float(row.get("d_phi_start", row.get("d_phi"))),
            gas_edge_exists=edge_type in {"gas_path", "original_target"},
            contract_score=float(pred.predicted_contract_positive),
            contract_lcb=lcb,
            predicted_hit=float(pred.predicted_hit),
            predicted_contract_positive=float(pred.predicted_contract_positive),
            predicted_negative_progress=negative,
            uncertainty=float(pred.uncertainty),
            q_train_support=_float(row.get("q_train_support")),
            edge_type=edge_type,
            bottleneck_score=min(lcb, 1.0 - negative),
        )
        edges[edge_id] = edge
        if source_id is not None:
            by_segment[str(source_id)].append(edge_id)

    boundaries: dict[str, BoundaryContract] = {}
    for edge_ids in by_segment.values():
        ordered = sorted(edge_ids, key=lambda eid: (_float(rows_index_hint(edges[eid]), 0.0), eid))
        for prev, nxt in zip(ordered[:-1], ordered[1:]):
            prev_edge, next_edge = edges[prev], edges[nxt]
            compatibility = min(_float(prev_edge.contract_lcb, 0.0), _float(next_edge.contract_lcb, 0.0))
            risk = max(_float(prev_edge.predicted_negative_progress, 0.0), _float(next_edge.predicted_negative_progress, 0.0))
            bid = f"{prev}__to__{nxt}"
            boundaries[bid] = BoundaryContract(
                prev_edge_id=prev,
                next_edge_id=nxt,
                compatibility_score=compatibility,
                boundary_risk=risk,
                observed_transition_count=1,
            )

    graph = ContractGraph(
        nodes=nodes,
        edges=edges,
        boundary_contracts=boundaries,
        metadata={
            "builder": "build_cage_contract_graph.py",
            "contract_model_loaded": bool(scorer.loaded),
            "input_rows": len(rows),
            "filtered_rows": len(filtered),
            "skipped_rows": skipped,
            "boundary_status": "ok" if boundaries else "BOUNDARY_CONTRACT_INCOMPLETE",
        },
    )
    return graph, graph.summarize()


def rows_index_hint(edge: ContractEdge) -> float:
    return _float(edge.d_phi, 0.0)


def write_report(path: Path, summary: dict[str, Any], graph_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage35 Contract Graph Build",
        "",
        f"Graph: `{graph_path}`",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key in [
        "node_count",
        "edge_count",
        "boundary_contract_count",
        "low_contract_edge_rate",
        "high_negative_edge_rate",
        "uncertain_edge_rate",
        "final_goal_edge_rate",
        "recovery_edge_rate",
    ]:
        lines.append(f"| {key} | {_fmt(summary.get(key))} |")
    lines.extend(["", "## Env Counts", "", json.dumps(summary.get("env_counts", {}), indent=2, sort_keys=True)])
    lines.extend(["", "## Target Mode Counts", "", json.dumps(summary.get("edge_type_counts", {}), indent=2, sort_keys=True)])
    boundary = summary.get("metadata", {}).get("boundary_status")
    if boundary == "BOUNDARY_CONTRACT_INCOMPLETE":
        lines.extend(["", "## Boundary Contract Status", "", "BOUNDARY_CONTRACT_INCOMPLETE: 输入数据缺少足够可靠的相邻 segment 元数据，边界兼容表为空或不完整。"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    if args.clear and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(Path(args.contract_dataset_path))
    scorer = ContractScorer.from_path(args.contract_model_path, uncertainty_penalty=0.25)
    graph, summary = build_graph(
        rows,
        scorer,
        env_name=args.env_name,
        max_edges=int(args.max_edges),
        min_contract_lcb=float(args.min_contract_lcb),
    )
    graph_path = out_dir / "contract_graph.json"
    graph.save_json(graph_path)
    graph.export_nodes_csv(out_dir / "nodes.csv")
    graph.export_edges_csv(out_dir / "edges.csv")
    graph.export_boundary_csv(out_dir / "boundary_contracts.csv")
    write_report(ROOT / "reports" / "stage35_contract_graph_build.md", summary, graph_path)
    print(json.dumps({"graph_path": str(graph_path), **summary}, sort_keys=True))
    return 0


def _float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
