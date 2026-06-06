#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external_src" / "GAS"))
sys.path.insert(0, str(ROOT / "scripts"))

from cage.contract_graph import ContractGraph
from cage.contract_planner import ContractPlanner
from evaluate_cage_contract_planner_offline import run_planners, sample_pairs

OBSERVED_TYPES = {"original_contract", "temporal_transition", "path_adjacency", "qtrain_supported"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate trusted ECG planner variants.")
    parser.add_argument("--graph_roots", nargs="+", required=True)
    parser.add_argument("--out_dir", default="results/cage_ecg/trusted_graph/planner_audit")
    parser.add_argument("--num_pairs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--require_multihop_pairs", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows: list[dict[str, Any]] = []
    summaries: dict[str, list[dict[str, Any]]] = {}
    for root in args.graph_roots:
        graph_path = Path(root) / "contract_graph.json"
        if not graph_path.exists():
            summaries[Path(root).name] = [{"status": "missing_graph", "graph_path": str(graph_path)}]
            continue
        graph = ContractGraph.load_json(graph_path)
        variant = Path(root).name
        variant_rows = audit_graph(graph, variant, args.num_pairs, args.seed, args.require_multihop_pairs)
        rows.extend(variant_rows)
        summaries[variant] = summarize_variant(variant_rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "trusted_planner_audit.csv"
    write_csv(out_csv, rows)
    out_md = ROOT / "reports" / "stage37_trusted_planner_audit.md"
    write_report(out_md, summaries)
    print(json.dumps({"out_csv": str(out_csv), "out_md": str(out_md), "num_rows": len(rows), "variants": {k: len(v) for k, v in summaries.items()}}, sort_keys=True))
    return 0


def audit_graph(graph: ContractGraph, variant: str, num_pairs: int, seed: int, require_multihop: bool) -> list[dict[str, Any]]:
    planner = ContractPlanner(graph)
    pairs = sample_pairs(graph, planner, num_pairs, seed, require_multihop_pairs=require_multihop, start_goal_source="same_env_pairs")
    rows: list[dict[str, Any]] = []
    for pair_idx, (src, dst) in enumerate(pairs):
        results = run_planners(planner, src, dst)
        shortest = results[0]
        shortest_edges = shortest.edge_ids
        for result in results:
            edge_types = [edge_type(graph, eid) for eid in result.edge_ids]
            row = asdict(result)
            row.update(
                {
                    "graph_variant": variant,
                    "pair_idx": pair_idx,
                    "src": src,
                    "dst": dst,
                    "path_found": result.reject_reason is None,
                    "path_length": len(result.edge_ids),
                    "multihop_pair": bool(shortest.edge_ids and len(shortest.edge_ids) >= 2),
                    "different_from_shortest": bool(result.edge_ids and result.edge_ids != shortest_edges),
                    "improves_min_contract_vs_shortest": improves(result.path_min_contract, shortest.path_min_contract),
                    "reduces_negative_risk_vs_shortest": reduces(result.path_negative_risk, shortest.path_negative_risk),
                    "knn_edge_usage_rate": rate(sum(1 for t in edge_types if t == "knn_bridge_candidate"), len(edge_types)),
                    "observed_edge_usage_rate": rate(sum(1 for t in edge_types if t in OBSERVED_TYPES), len(edge_types)),
                    "bottleneck_edge_type": edge_type(graph, result.bottleneck_edge_id) if result.bottleneck_edge_id else None,
                }
            )
            rows.append(row)
    if not rows:
        for planner_name in ["shortest_by_dphi", "max_contract_path", "risk_constrained_path", "bottleneck_robust_path", "progress_contract_path"]:
            rows.append({"graph_variant": variant, "planner_name": planner_name, "path_found": False, "reject_reason": "no_multihop_pairs"})
    return rows


def summarize_variant(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("planner_name"))].append(row)
    out = []
    for planner_name, items in sorted(grouped.items()):
        found = [row for row in items if truthy(row.get("path_found"))]
        bottlenecks = Counter(str(row.get("bottleneck_edge_type") or "NA") for row in found if row.get("bottleneck_edge_type"))
        out.append(
            {
                "planner_name": planner_name,
                "num_rows": len(items),
                "found_rate": len(found) / len(items) if items else None,
                "no_path_rate": 1.0 - (len(found) / len(items) if items else 0.0),
                "mean_path_length": mean_key(found, "path_length"),
                "multihop_rate": mean_key(items, "multihop_pair"),
                "min_contract": mean_key(found, "path_min_contract"),
                "success_lcb": mean_key(found, "predicted_success_lower_bound"),
                "negative_risk": mean_key(found, "path_negative_risk"),
                "diff_from_shortest": mean_key(found, "different_from_shortest"),
                "improve_contract_rate": mean_key(found, "improves_min_contract_vs_shortest"),
                "reduce_risk_rate": mean_key(found, "reduces_negative_risk_vs_shortest"),
                "knn_edge_usage_rate": mean_key(found, "knn_edge_usage_rate"),
                "observed_edge_usage_rate": mean_key(found, "observed_edge_usage_rate"),
                "bottleneck_edge_type_distribution": dict(bottlenecks),
            }
        )
    return out


def write_report(path: Path, summaries: dict[str, list[dict[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage37 Trusted ECG Planner Audit",
        "",
        "| variant | planner | found | no_path | length | multihop | min_contract | success_lcb | negative_risk | diff | improve_contract | reduce_risk | knn_usage | observed_usage |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    variant_signal: dict[str, bool] = {}
    for variant, items in summaries.items():
        signal = False
        for row in items:
            if row.get("planner_name") != "shortest_by_dphi" and (as_float(row.get("diff_from_shortest"), 0.0) > 0 or as_float(row.get("improve_contract_rate"), 0.0) > 0 or as_float(row.get("reduce_risk_rate"), 0.0) > 0):
                signal = True
            lines.append(
                f"| {variant} | {row.get('planner_name')} | {fmt(row.get('found_rate'))} | {fmt(row.get('no_path_rate'))} | {fmt(row.get('mean_path_length'))} | {fmt(row.get('multihop_rate'))} | {fmt(row.get('min_contract'))} | {fmt(row.get('success_lcb'))} | {fmt(row.get('negative_risk'))} | {fmt(row.get('diff_from_shortest'))} | {fmt(row.get('improve_contract_rate'))} | {fmt(row.get('reduce_risk_rate'))} | {fmt(row.get('knn_edge_usage_rate'))} | {fmt(row.get('observed_edge_usage_rate'))} |"
            )
        variant_signal[variant] = signal
    full_signal = variant_signal.get("full", False)
    trusted_signal = variant_signal.get("trusted_conservative", False) or variant_signal.get("observed_plus_final", False)
    lines.extend(
        [
            "",
            "## Gate",
            "",
            f"- full_signal: `{full_signal}`",
            f"- trusted_signal: `{trusted_signal}`",
            f"- trusted_planner_status: `{'TRUSTED_OFFLINE_SIGNAL' if trusted_signal else ('FULL_ONLY_SIGNAL' if full_signal else 'NO_PLANNER_SIGNAL')}`",
            "",
            "若只有 full graph 有 signal，则不能进入 online。只有 trusted_conservative 或 observed_plus_final 也有 planner signal，才可考虑下一轮 limited AntMaze smoke。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fields})


def edge_type(graph: ContractGraph, edge_id: str | None) -> str:
    edge = graph.edges.get(str(edge_id))
    return str(edge.edge_type or "NA") if edge is not None else "missing_edge"


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"true", "1", "yes"}


def mean_key(rows: list[dict[str, Any]], key: str) -> float | None:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, bool):
            values.append(1.0 if value else 0.0)
        else:
            try:
                if value is not None:
                    values.append(float(value))
            except (TypeError, ValueError):
                pass
    return mean(values) if values else None


def improves(value: Any, baseline: Any) -> bool:
    try:
        return value is not None and baseline is not None and float(value) > float(baseline) + 1e-9
    except (TypeError, ValueError):
        return False


def reduces(value: Any, baseline: Any) -> bool:
    try:
        return value is not None and baseline is not None and float(value) < float(baseline) - 1e-9
    except (TypeError, ValueError):
        return False


def rate(num: int, den: int) -> float | None:
    return num / den if den else None


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is not None:
            return float(value)
    except (TypeError, ValueError):
        pass
    return default


def csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True)
    return value


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
