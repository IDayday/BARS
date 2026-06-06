#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict, deque
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external_src" / "GAS"))

from cage.contract_graph import BoundaryContract, ContractEdge, ContractFunnelNode, ContractGraph

OBSERVED_TYPES = {"original_contract", "temporal_transition", "path_adjacency", "qtrain_supported"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build trusted ECG contract graph variants.")
    parser.add_argument("--contract_graph_path", default="results/cage_ecg/transition_contract_graph/contract_graph_augmented.json")
    parser.add_argument("--out_dir", default="results/cage_ecg/trusted_graph")
    parser.add_argument("--min_contract_lcb", type=float, default=-1.0)
    parser.add_argument("--max_knn_negative_risk", type=float, default=0.50)
    parser.add_argument("--max_knn_uncertainty", type=float, default=0.50)
    parser.add_argument("--clear", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    if args.clear and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    graph = ContractGraph.load_json(args.contract_graph_path)

    variants = {
        "observed_only": lambda edge: edge.edge_type in OBSERVED_TYPES,
        "observed_plus_final": lambda edge: edge.edge_type in OBSERVED_TYPES or edge.edge_type == "final_goal_candidate",
        "trusted_conservative": lambda edge: (
            edge.edge_type in OBSERVED_TYPES
            or edge.edge_type == "final_goal_candidate"
            or (
                edge.edge_type == "knn_bridge_candidate"
                and as_float(edge.contract_lcb, -1.0) >= float(args.min_contract_lcb)
                and as_float(edge.predicted_negative_progress, 1.0) <= float(args.max_knn_negative_risk)
                and as_float(edge.uncertainty, 1.0) <= float(args.max_knn_uncertainty)
            )
        ),
        "full": lambda edge: as_float(edge.contract_lcb, -1.0) >= float(args.min_contract_lcb),
    }

    summaries: dict[str, dict[str, Any]] = {}
    for name, predicate in variants.items():
        subgraph = filter_graph(graph, predicate, name)
        variant_dir = out_dir / name
        variant_dir.mkdir(parents=True, exist_ok=True)
        graph_path = variant_dir / "contract_graph.json"
        subgraph.save_json(graph_path)
        subgraph.export_nodes_csv(variant_dir / "nodes.csv")
        subgraph.export_edges_csv(variant_dir / "edges.csv")
        subgraph.export_boundary_csv(variant_dir / "boundary_contracts.csv")
        summaries[name] = summarize_variant(subgraph)
    write_report(ROOT / "reports" / "stage37_trusted_graph_build.md", summaries, args.contract_graph_path, out_dir)
    print(json.dumps({"out_dir": str(out_dir), "variants": summaries}, sort_keys=True))
    return 0


def filter_graph(graph: ContractGraph, predicate: Callable[[ContractEdge], bool], variant_name: str) -> ContractGraph:
    edges = {edge_id: ContractEdge(**asdict(edge)) for edge_id, edge in graph.edges.items() if predicate(edge)}
    node_ids = {edge.src for edge in edges.values()} | {edge.dst for edge in edges.values()}
    nodes = {node_id: ContractFunnelNode(**asdict(graph.nodes[node_id])) for node_id in node_ids if node_id in graph.nodes}
    boundaries = {
        key: BoundaryContract(**asdict(boundary))
        for key, boundary in graph.boundary_contracts.items()
        if boundary.prev_edge_id in edges and boundary.next_edge_id in edges
    }
    return ContractGraph(
        nodes=nodes,
        edges=edges,
        boundary_contracts=boundaries,
        metadata={**dict(graph.metadata), "trusted_graph_variant": variant_name, "source_edge_count": len(graph.edges)},
    )


def summarize_variant(graph: ContractGraph) -> dict[str, Any]:
    base = graph.summarize()
    base.update(connectivity_summary(graph))
    edge_count = max(len(graph.edges), 1)
    type_counts = Counter(str(edge.edge_type or "NA") for edge in graph.edges.values())
    base["edge_type_counts"] = dict(type_counts)
    base["knn_edge_rate"] = type_counts.get("knn_bridge_candidate", 0) / edge_count
    return base


def connectivity_summary(graph: ContractGraph) -> dict[str, Any]:
    outgoing: dict[str, set[str]] = defaultdict(set)
    incoming: dict[str, set[str]] = defaultdict(set)
    undirected: dict[str, set[str]] = defaultdict(set)
    for edge in graph.edges.values():
        outgoing[edge.src].add(edge.dst)
        incoming[edge.dst].add(edge.src)
        undirected[edge.src].add(edge.dst)
        undirected[edge.dst].add(edge.src)
    weak = component_sizes(graph.nodes.keys(), undirected)
    strong = strongly_connected_sizes(graph.nodes.keys(), outgoing, incoming)
    return {
        "avg_out_degree": sum(len(outgoing.get(node_id, ())) for node_id in graph.nodes) / max(len(graph.nodes), 1),
        "weak_component_count": len(weak),
        "largest_weak_component": max(weak) if weak else 0,
        "strong_component_count": len(strong),
        "largest_strong_component": max(strong) if strong else 0,
    }


def component_sizes(nodes: Any, adjacency: dict[str, set[str]]) -> list[int]:
    unseen = set(nodes)
    sizes = []
    while unseen:
        start = unseen.pop()
        queue = deque([start])
        size = 1
        while queue:
            node = queue.popleft()
            for nxt in adjacency.get(node, set()):
                if nxt in unseen:
                    unseen.remove(nxt)
                    size += 1
                    queue.append(nxt)
        sizes.append(size)
    return sizes


def strongly_connected_sizes(nodes: Any, outgoing: dict[str, set[str]], incoming: dict[str, set[str]]) -> list[int]:
    seen: set[str] = set()
    order: list[str] = []
    for node in nodes:
        if node in seen:
            continue
        stack = [(node, False)]
        while stack:
            current, expanded = stack.pop()
            if expanded:
                order.append(current)
                continue
            if current in seen:
                continue
            seen.add(current)
            stack.append((current, True))
            for nxt in outgoing.get(current, set()):
                if nxt not in seen:
                    stack.append((nxt, False))
    seen.clear()
    sizes = []
    for node in reversed(order):
        if node in seen:
            continue
        total = 0
        stack = [node]
        seen.add(node)
        while stack:
            current = stack.pop()
            total += 1
            for nxt in incoming.get(current, set()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        sizes.append(total)
    return sizes


def write_report(path: Path, summaries: dict[str, dict[str, Any]], source_graph: str, out_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage37 Trusted ECG Contract Graph Build",
        "",
        f"Source graph: `{source_graph}`",
        f"Output dir: `{out_dir}`",
        "",
        "| variant | nodes | edges | avg_out | weak | largest_weak | strong | largest_strong | final_rate | recovery_rate | knn_rate | low_contract | high_negative |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, summary in summaries.items():
        lines.append(
            f"| {name} | {summary.get('node_count')} | {summary.get('edge_count')} | {fmt(summary.get('avg_out_degree'))} | {summary.get('weak_component_count')} | {summary.get('largest_weak_component')} | {summary.get('strong_component_count')} | {summary.get('largest_strong_component')} | {fmt(summary.get('final_goal_edge_rate'))} | {fmt(summary.get('recovery_edge_rate'))} | {fmt(summary.get('knn_edge_rate'))} | {fmt(summary.get('low_contract_edge_rate'))} | {fmt(summary.get('high_negative_edge_rate'))} |"
        )
    lines.extend(["", "## Edge Type Distributions"])
    for name, summary in summaries.items():
        lines.extend(["", f"### {name}", "", json.dumps(summary.get("edge_type_counts", {}), indent=2, sort_keys=True)])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
