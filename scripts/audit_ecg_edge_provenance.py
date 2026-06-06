#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external_src" / "GAS"))

from cage.contract_graph import ContractGraph

OBSERVED_TYPES = {"original_contract", "temporal_transition", "path_adjacency", "qtrain_supported"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit ECG edge provenance and planner dependence on candidate bridge edges.")
    parser.add_argument("--contract_graph_path", default="results/cage_ecg/transition_contract_graph/contract_graph_augmented.json")
    parser.add_argument("--planner_audit_path", default="results/cage_ecg/transition_contract_planner/offline_plan_audit.csv")
    parser.add_argument("--out_csv", default="results/cage_ecg/trusted_graph/edge_provenance_audit.csv")
    parser.add_argument("--out_md", default="reports/stage37_edge_provenance_audit.md")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    graph = ContractGraph.load_json(args.contract_graph_path)
    planner_rows = load_csv(Path(args.planner_audit_path))
    rows, summary = audit(graph, planner_rows)
    write_csv(Path(args.out_csv), rows)
    write_report(Path(args.out_md), summary, args.contract_graph_path, args.planner_audit_path)
    print(json.dumps({"out_csv": args.out_csv, "out_md": args.out_md, **summary}, sort_keys=True))
    return 0


def audit(graph: ContractGraph, planner_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    edge_rows = edge_type_rows(graph)
    planner_usage = planner_edge_usage(graph, planner_rows)
    bottleneck_counts = Counter(str(row.get("bottleneck_edge_type") or "NA") for row in planner_rows if row.get("bottleneck_edge_id"))
    edge_count = max(len(graph.edges), 1)
    knn_edge_rate = sum(1 for edge in graph.edges.values() if edge.edge_type == "knn_bridge_candidate") / edge_count
    observed_edge_rate = sum(1 for edge in graph.edges.values() if edge.edge_type in OBSERVED_TYPES) / edge_count
    improved_rows = [
        row
        for row in planner_rows
        if row.get("planner_name") != "shortest_by_dphi"
        and truthy(row.get("path_found"))
        and (truthy(row.get("different_from_shortest")) or truthy(row.get("improves_min_contract_vs_shortest")) or truthy(row.get("reduces_negative_risk_vs_shortest")))
    ]
    improved_edge_types = Counter()
    improved_paths_with_knn = 0
    improved_paths_observed_only = 0
    for row in improved_rows:
        edge_ids = parse_json_list(row.get("edge_ids"))
        types = [edge_type(graph, eid) for eid in edge_ids]
        improved_edge_types.update(types)
        if any(t == "knn_bridge_candidate" for t in types):
            improved_paths_with_knn += 1
        if types and all(t in OBSERVED_TYPES for t in types):
            improved_paths_observed_only += 1
    improved_count = max(len(improved_rows), 1)
    knn_dependent = improved_paths_with_knn / improved_count > 0.5
    trusted_possible = improved_paths_observed_only > 0
    summary = {
        "edge_count": len(graph.edges),
        "knn_edge_rate": knn_edge_rate,
        "observed_edge_rate": observed_edge_rate,
        "planner_rows": len(planner_rows),
        "improved_planner_rows": len(improved_rows),
        "improved_paths_with_knn_rate": improved_paths_with_knn / improved_count if improved_rows else None,
        "improved_paths_observed_only_count": improved_paths_observed_only,
        "provenance_status": "KNN_DEPENDENT_PLANNER_SIGNAL" if knn_dependent else "KNN_NOT_DOMINANT",
        "trusted_signal_status": "TRUSTED_PLANNER_SIGNAL_POSSIBLE" if trusted_possible else "TRUSTED_SIGNAL_NOT_OBSERVED_IN_FULL_AUDIT",
        "planner_edge_type_usage": dict(planner_usage),
        "bottleneck_edge_type_counts": dict(bottleneck_counts),
        "improved_edge_type_counts": dict(improved_edge_types),
    }
    return edge_rows, summary


def edge_type_rows(graph: ContractGraph) -> list[dict[str, Any]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for edge in graph.edges.values():
        grouped[str(edge.edge_type or "NA")].append(edge)
    total = max(len(graph.edges), 1)
    rows = []
    for edge_type_name, edges in sorted(grouped.items()):
        rows.append(
            {
                "edge_type": edge_type_name,
                "count": len(edges),
                "rate": len(edges) / total,
                "mean_contract_lcb": numeric_mean(edge.contract_lcb for edge in edges),
                "mean_negative_risk": numeric_mean(edge.predicted_negative_progress for edge in edges),
                "mean_uncertainty": numeric_mean(edge.uncertainty for edge in edges),
                "observed_or_trusted": edge_type_name in OBSERVED_TYPES,
            }
        )
    return rows


def planner_edge_usage(graph: ContractGraph, rows: list[dict[str, Any]]) -> Counter:
    counts: Counter[str] = Counter()
    for row in rows:
        for edge_id in parse_json_list(row.get("edge_ids")):
            counts[edge_type(graph, edge_id)] += 1
    return counts


def write_report(path: Path, summary: dict[str, Any], graph_path: str, planner_path: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage37 ECG Edge Provenance Audit",
        "",
        f"Graph: `{graph_path}`",
        f"Planner audit: `{planner_path}`",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| edge_count | {summary['edge_count']} |",
        f"| knn_edge_rate | {fmt(summary['knn_edge_rate'])} |",
        f"| observed_edge_rate | {fmt(summary['observed_edge_rate'])} |",
        f"| improved_planner_rows | {summary['improved_planner_rows']} |",
        f"| improved_paths_with_knn_rate | {fmt(summary['improved_paths_with_knn_rate'])} |",
        f"| improved_paths_observed_only_count | {summary['improved_paths_observed_only_count']} |",
        "",
        "## Status",
        "",
        f"- provenance_status: `{summary['provenance_status']}`",
        f"- trusted_signal_status: `{summary['trusted_signal_status']}`",
        "",
        "## Planner Edge Type Usage",
        "",
        json.dumps(summary["planner_edge_type_usage"], indent=2, sort_keys=True),
        "",
        "## Bottleneck Edge Type Counts",
        "",
        json.dumps(summary["bottleneck_edge_type_counts"], indent=2, sort_keys=True),
        "",
        "KNN bridge candidate 不是观测 transition。若 planner improvement 主要依赖 KNN，则只能算离线候选信号，不能写成已验证可执行路径。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def parse_json_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    try:
        parsed = json.loads(str(value))
        return [str(v) for v in parsed] if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


def edge_type(graph: ContractGraph, edge_id: str) -> str:
    edge = graph.edges.get(str(edge_id))
    return str(edge.edge_type or "NA") if edge is not None else "missing_edge"


def truthy(value: Any) -> bool:
    return str(value).lower() in {"1", "true", "yes"}


def numeric_mean(values: Any) -> float | None:
    nums = []
    for value in values:
        try:
            if value is not None:
                nums.append(float(value))
        except (TypeError, ValueError):
            pass
    return mean(nums) if nums else None


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
