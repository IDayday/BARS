#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external_src" / "GAS"))

from cage.contract_graph import ContractGraph
from cage.contract_planner import ContractPlanner, PlanResult


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline audit for CAGE ECG contract planners.")
    parser.add_argument("--contract_graph_path", default="results/cage_ecg/contract_graph/contract_graph.json")
    parser.add_argument("--out_dir", default="results/cage_ecg/contract_planner")
    parser.add_argument("--num_pairs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def sample_pairs(graph: ContractGraph, num_pairs: int, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    direct = [(edge.src, edge.dst) for edge in graph.edges.values()]
    candidates = list(dict.fromkeys(direct))
    nodes = list(graph.nodes)
    for _ in range(max(0, int(num_pairs) - len(candidates))):
        if len(nodes) < 2:
            break
        src, dst = rng.sample(nodes, 2)
        candidates.append((src, dst))
    rng.shuffle(candidates)
    return candidates[: int(num_pairs)]


def run_planners(planner: ContractPlanner, src: str, dst: str) -> list[PlanResult]:
    return [
        planner.shortest_by_dphi(src, dst),
        planner.max_contract_path(src, dst),
        planner.risk_constrained_path(src, dst),
        planner.bottleneck_robust_path(src, dst),
        planner.progress_contract_path(src, dst),
    ]


def main() -> int:
    args = build_parser().parse_args()
    graph = ContractGraph.load_json(args.contract_graph_path)
    planner = ContractPlanner(graph)
    rows: list[dict[str, Any]] = []
    pairs = sample_pairs(graph, args.num_pairs, args.seed)
    for pair_idx, (src, dst) in enumerate(pairs):
        results = run_planners(planner, src, dst)
        shortest_edges = results[0].edge_ids
        for result in results:
            row = asdict(result)
            row.update(
                {
                    "pair_idx": pair_idx,
                    "src": src,
                    "dst": dst,
                    "path_found": result.reject_reason is None,
                    "path_length": len(result.edge_ids),
                    "different_from_shortest": bool(result.edge_ids and result.edge_ids != shortest_edges),
                }
            )
            rows.append(row)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "offline_plan_audit.csv"
    write_csv(out_csv, rows)
    grouped = summarize(rows)
    out_md = ROOT / "reports" / "stage35_contract_planner_offline.md"
    write_report(out_md, grouped, rows, args.contract_graph_path)
    print(json.dumps({"out_csv": str(out_csv), "out_md": str(out_md), "num_rows": len(rows)}, sort_keys=True))
    return 0


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["planner_name"]), []).append(row)
    out: list[dict[str, Any]] = []
    for planner_name, items in sorted(grouped.items()):
        found = [row for row in items if row.get("path_found")]
        out.append(
            {
                "planner_name": planner_name,
                "num_pairs": len(items),
                "path_found_rate": len(found) / len(items) if items else None,
                "mean_path_length": mean([len(row.get("edge_ids") or []) for row in found]) if found else None,
                "mean_path_min_contract": _mean_key(found, "path_min_contract"),
                "mean_predicted_success_lower_bound": _mean_key(found, "predicted_success_lower_bound"),
                "mean_negative_risk": _mean_key(found, "path_negative_risk"),
                "bottleneck_edge_count": len({row.get("bottleneck_edge_id") for row in found if row.get("bottleneck_edge_id")}),
                "paths_different_from_shortest_rate": _mean_key(found, "different_from_shortest"),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def write_report(path: Path, grouped: list[dict[str, Any]], rows: list[dict[str, Any]], graph_path: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage35 Contract Planner Offline Audit",
        "",
        f"Graph: `{graph_path}`",
        "",
        "| planner | found | length | min_contract | success_lcb | negative_risk | bottlenecks | diff_from_shortest |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in grouped:
        lines.append(
            "| {planner_name} | {path_found_rate} | {mean_path_length} | {mean_path_min_contract} | {mean_predicted_success_lower_bound} | {mean_negative_risk} | {bottleneck_edge_count} | {paths_different_from_shortest_rate} |".format(
                **{key: _fmt(value) for key, value in row.items()}
            )
        )
    diff = any(bool(row.get("different_from_shortest")) for row in rows)
    lines.extend(
        [
            "",
            f"Planner difference gate: {'PASS' if diff else 'INCONCLUSIVE'}",
            "",
            "若图中多数 pair 只有 direct edge，则 planner 可能都退化为同一路径；这不是在线结果，只是合同图连通性和风险约束的离线审计。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mean_key(rows: list[dict[str, Any]], key: str) -> float | None:
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


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True)
    return value


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
