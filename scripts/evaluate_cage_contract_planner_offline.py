#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import asdict
from collections import Counter
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
    parser.add_argument("--require_multihop_pairs", action="store_true")
    parser.add_argument("--min_path_length", type=int, default=0)
    parser.add_argument("--start_goal_source", choices=["random_pairs", "same_env_pairs", "hard_pairs"], default="random_pairs")
    parser.add_argument("--compare_graphs", default="")
    return parser


def sample_pairs(
    graph: ContractGraph,
    planner: ContractPlanner,
    num_pairs: int,
    seed: int,
    *,
    require_multihop_pairs: bool = False,
    min_path_length: int = 0,
    start_goal_source: str = "random_pairs",
) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    direct = list(dict.fromkeys((edge.src, edge.dst) for edge in graph.edges.values()))
    nodes = list(graph.nodes)
    env_nodes: dict[str, list[str]] = {}
    for node_id, node in graph.nodes.items():
        env_nodes.setdefault(str(node.env_name or "unknown"), []).append(node_id)
    candidates: list[tuple[str, str]] = []
    if start_goal_source == "hard_pairs":
        risky_edges = sorted(
            graph.edges.values(),
            key=lambda edge: (_float(edge.contract_lcb, 0.0), -_float(edge.predicted_negative_progress, 0.0)),
        )
        candidates.extend((edge.src, edge.dst) for edge in risky_edges[: max(num_pairs * 4, 64)])
    elif start_goal_source == "same_env_pairs":
        for group in env_nodes.values():
            for _ in range(min(max(num_pairs * 4, 64), max(len(group) * 2, 1))):
                if len(group) >= 2:
                    candidates.append(tuple(rng.sample(group, 2)))
    else:
        candidates.extend(direct)
        for _ in range(max(num_pairs * 8, 256)):
            if len(nodes) >= 2:
                candidates.append(tuple(rng.sample(nodes, 2)))
    rng.shuffle(candidates)
    min_len = int(min_path_length or (2 if require_multihop_pairs else 1))
    selected: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for src, dst in candidates:
        if (src, dst) in seen or src == dst:
            continue
        seen.add((src, dst))
        if require_multihop_pairs:
            shortest = planner.shortest_by_dphi(src, dst, max_path_length=32)
            if shortest.reject_reason is not None or len(shortest.edge_ids) < min_len:
                continue
        selected.append((src, dst))
        if len(selected) >= int(num_pairs):
            break
    if not selected and require_multihop_pairs:
        # Do not fabricate multihop paths. Return an empty set and let the report mark the gate.
        return []
    return selected[: int(num_pairs)]


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
    pairs = sample_pairs(
        graph,
        planner,
        args.num_pairs,
        args.seed,
        require_multihop_pairs=bool(args.require_multihop_pairs),
        min_path_length=int(args.min_path_length),
        start_goal_source=args.start_goal_source,
    )
    for pair_idx, (src, dst) in enumerate(pairs):
        results = run_planners(planner, src, dst)
        shortest_edges = results[0].edge_ids
        shortest = results[0]
        for result in results:
            row = asdict(result)
            bottleneck_type = graph.edges[result.bottleneck_edge_id].edge_type if result.bottleneck_edge_id in graph.edges else None
            row.update(
                {
                    "pair_idx": pair_idx,
                    "src": src,
                    "dst": dst,
                    "path_found": result.reject_reason is None,
                    "path_length": len(result.edge_ids),
                    "multihop_pair": bool(shortest.edge_ids and len(shortest.edge_ids) >= 2),
                    "different_from_shortest": bool(result.edge_ids and result.edge_ids != shortest_edges),
                    "improves_min_contract_vs_shortest": _improves(result.path_min_contract, shortest.path_min_contract),
                    "reduces_negative_risk_vs_shortest": _reduces(result.path_negative_risk, shortest.path_negative_risk),
                    "bottleneck_edge_type": bottleneck_type,
                }
            )
            rows.append(row)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "offline_plan_audit.csv"
    write_csv(out_csv, rows)
    grouped = summarize(rows)
    report_name = "stage36_transition_contract_planner_offline.md" if "transition" in str(args.out_dir) or args.require_multihop_pairs else "stage35_contract_planner_offline.md"
    out_md = ROOT / "reports" / report_name
    write_report(out_md, grouped, rows, args.contract_graph_path, stage36=report_name.startswith("stage36"))
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
                "multihop_pair_rate": _mean_key(items, "multihop_pair"),
                "contract_planner_improves_min_contract_rate": _mean_key(found, "improves_min_contract_vs_shortest"),
                "contract_planner_reduces_negative_risk_rate": _mean_key(found, "reduces_negative_risk_vs_shortest"),
                "boundary_bottleneck_rate": _boundary_bottleneck_rate(found),
                "no_path_rate": 1.0 - (len(found) / len(items) if items else 0.0),
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


def write_report(path: Path, grouped: list[dict[str, Any]], rows: list[dict[str, Any]], graph_path: str, *, stage36: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage36 Transition Contract Planner Offline Audit" if stage36 else "# Stage35 Contract Planner Offline Audit",
        "",
        f"Graph: `{graph_path}`",
        "",
        "| planner | found | length | multihop | min_contract | success_lcb | negative_risk | diff_from_shortest | improve_contract | reduce_risk | no_path |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in grouped:
        lines.append(
            "| {planner_name} | {path_found_rate} | {mean_path_length} | {multihop_pair_rate} | {mean_path_min_contract} | {mean_predicted_success_lower_bound} | {mean_negative_risk} | {paths_different_from_shortest_rate} | {contract_planner_improves_min_contract_rate} | {contract_planner_reduces_negative_risk_rate} | {no_path_rate} |".format(
                **{key: _fmt(value) for key, value in row.items()}
            )
        )
    diff = any(bool(row.get("different_from_shortest")) for row in rows)
    multihop = any(bool(row.get("multihop_pair")) for row in rows)
    improved = any(bool(row.get("improves_min_contract_vs_shortest") or row.get("reduces_negative_risk_vs_shortest")) for row in rows if row.get("planner_name") != "shortest_by_dphi")
    bottlenecks = Counter(str(row.get("bottleneck_edge_type") or "NA") for row in rows if row.get("bottleneck_edge_id"))
    lines.extend(
        [
            "",
            f"Planner difference gate: {'PLANNER_OFFLINE_SIGNAL' if diff and improved else ('PASS' if diff else 'INCONCLUSIVE')}",
            f"Multihop pair gate: {'PASS' if multihop else 'FAIL'}",
            "",
            "## Bottleneck Edge Type Distribution",
            "",
            json.dumps(dict(bottlenecks), indent=2, sort_keys=True),
            "",
            "若图中多数 pair 只有 direct edge，则 planner 可能都退化为同一路径；这不是在线结果，只是合同图连通性和风险约束的离线审计，不得写成 online SOTA。",
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


def _boundary_bottleneck_rate(rows: list[dict[str, Any]]) -> float | None:
    values = [1.0 if str(row.get("bottleneck_edge_type") or "") == "boundary_transition" else 0.0 for row in rows if row.get("bottleneck_edge_id")]
    return mean(values) if values else None


def _improves(value: Any, baseline: Any) -> bool:
    try:
        return value is not None and baseline is not None and float(value) > float(baseline) + 1e-9
    except (TypeError, ValueError):
        return False


def _reduces(value: Any, baseline: Any) -> bool:
    try:
        return value is not None and baseline is not None and float(value) < float(baseline) - 1e-9
    except (TypeError, ValueError):
        return False


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
