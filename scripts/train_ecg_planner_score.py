#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
import json
import math
import random
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "external_src" / "GAS"))

from cage.contract_graph import ContractEdge, ContractGraph  # noqa: E402
from cage.contract_planner import ContractPlanner  # noqa: E402


WEIGHT_GRID = [
    {"task_progress": 0.5, "contract": 1.0, "negative": 1.0, "uncertainty": 0.5, "boundary": 0.25, "path_length": 0.05},
    {"task_progress": 0.25, "contract": 1.5, "negative": 1.0, "uncertainty": 0.5, "boundary": 0.25, "path_length": 0.05},
    {"task_progress": 0.25, "contract": 1.0, "negative": 1.5, "uncertainty": 0.75, "boundary": 0.25, "path_length": 0.05},
    {"task_progress": 1.0, "contract": 1.0, "negative": 0.5, "uncertainty": 0.25, "boundary": 0.10, "path_length": 0.02},
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate a lightweight ECG planner edge score on held-out offline graph pairs.")
    parser.add_argument("--graph", required=True)
    parser.add_argument("--contract_dataset", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_pairs", type=int, default=512)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    graph = ContractGraph.load_json(args.graph)
    pairs = sample_pairs(graph, seed=int(args.seed), num_pairs=int(args.num_pairs))
    metrics: dict[str, Any] = {
        "graph": str(args.graph),
        "contract_dataset": str(args.contract_dataset),
        "num_pairs": len(pairs),
        "edge_count": len(graph.edges),
        "node_count": len(graph.nodes),
    }
    if not pairs:
        metrics["status"] = "PLANNER_SCORE_BLOCKED_NO_PAIRS"
        write_outputs(out_dir, None, metrics)
        return 2
    baseline = eval_shortest(graph, pairs)
    candidates = []
    for weights in WEIGHT_GRID:
        row = eval_weighted(graph, pairs, weights)
        row["weights"] = weights
        candidates.append(row)
    best = max(candidates, key=lambda row: row["objective"])
    metrics["shortest_by_dphi"] = baseline
    metrics["candidate_results"] = candidates
    metrics["selected_weights"] = best["weights"]
    metrics["learned_ecg_score_path"] = {k: v for k, v in best.items() if k != "weights"}
    metrics["gate"] = {
        "path_negative_risk_below_shortest": best.get("mean_negative_risk", math.inf) < baseline.get("mean_negative_risk", -math.inf),
        "path_min_contract_above_shortest": best.get("mean_min_contract", -math.inf) > baseline.get("mean_min_contract", math.inf),
        "future_trajectory_agreement_above_shortest": best.get("future_agreement", -math.inf) > baseline.get("future_agreement", math.inf),
    }
    metrics["status"] = "PLANNER_SCORE_READY" if any(metrics["gate"].values()) else "PLANNER_SCORE_NO_OFFLINE_GAIN"
    write_outputs(out_dir, best["weights"], metrics)
    print(json.dumps({"status": metrics["status"], "out_dir": str(out_dir)}, sort_keys=True))
    return 0 if metrics["status"] == "PLANNER_SCORE_READY" else 2


def sample_pairs(graph: ContractGraph, *, seed: int, num_pairs: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    edges = list(graph.edges.values())
    pairs: list[tuple[str, str]] = []
    for edge in edges:
        if edge.src != edge.dst:
            pairs.append((edge.src, edge.dst))
    rng.shuffle(pairs)
    return pairs[:num_pairs]


def eval_shortest(graph: ContractGraph, pairs: list[tuple[str, str]]) -> dict[str, Any]:
    planner = ContractPlanner(graph)
    plans = [planner.shortest_by_dphi(src, dst, max_path_length=16) for src, dst in pairs]
    return summarize_plans(graph, plans)


def eval_weighted(graph: ContractGraph, pairs: list[tuple[str, str]], weights: dict[str, float]) -> dict[str, Any]:
    plans = [weighted_plan(graph, src, dst, weights, max_path_length=16) for src, dst in pairs]
    row = summarize_plans(graph, plans)
    row["objective"] = (
        float(row.get("mean_min_contract") or 0.0)
        - float(row.get("mean_negative_risk") or 0.0)
        + 0.25 * float(row.get("future_agreement") or 0.0)
        - 0.01 * float(row.get("mean_path_length") or 0.0)
    )
    return row


def weighted_plan(graph: ContractGraph, src: str, dst: str, weights: dict[str, float], *, max_path_length: int) -> dict[str, Any]:
    if src not in graph.nodes or dst not in graph.nodes:
        return {"found": False, "edge_ids": []}
    adjacency: dict[str, list[ContractEdge]] = {}
    for edge in graph.edges.values():
        adjacency.setdefault(edge.src, []).append(edge)
    heap: list[tuple[float, str, list[str]]] = [(0.0, src, [])]
    best: dict[str, float] = {src: 0.0}
    while heap:
        cost, node, edge_ids = heapq.heappop(heap)
        if node == dst:
            return {"found": True, "edge_ids": edge_ids}
        if len(edge_ids) >= max_path_length:
            continue
        for edge in adjacency.get(node, []):
            next_cost = cost + edge_cost(edge, weights)
            if next_cost >= best.get(edge.dst, math.inf):
                continue
            best[edge.dst] = next_cost
            heapq.heappush(heap, (next_cost, edge.dst, [*edge_ids, edge.edge_id]))
    return {"found": False, "edge_ids": []}


def edge_cost(edge: ContractEdge, weights: dict[str, float]) -> float:
    d_phi = float(edge.d_phi or 0.0)
    contract = float(edge.contract_lcb or 0.0)
    negative = float(edge.predicted_negative_progress or 0.0)
    uncertainty = float(edge.uncertainty or 0.0)
    progress = 1.0 / (1.0 + max(d_phi, 0.0))
    return (
        weights["path_length"]
        + weights["task_progress"] * (1.0 - progress)
        + weights["negative"] * negative
        + weights["uncertainty"] * uncertainty
        - weights["contract"] * contract
    )


def summarize_plans(graph: ContractGraph, plans: list[Any]) -> dict[str, Any]:
    found = 0
    lengths: list[int] = []
    min_contracts: list[float] = []
    risks: list[float] = []
    agreement = 0
    for plan in plans:
        edge_ids = getattr(plan, "edge_ids", None) if not isinstance(plan, dict) else plan.get("edge_ids")
        reject = getattr(plan, "reject_reason", None) if not isinstance(plan, dict) else (None if plan.get("found") else "no_path")
        if reject or not edge_ids:
            continue
        found += 1
        edges = [graph.edges[eid] for eid in edge_ids]
        lengths.append(len(edges))
        min_contracts.append(min(float(edge.contract_lcb or 0.0) for edge in edges))
        risks.append(sum(float(edge.predicted_negative_progress or 0.0) for edge in edges))
        if all(str(edge.edge_type or "").startswith(("offline_temporal", "final_goal")) for edge in edges):
            agreement += 1
    return {
        "found_rate": safe_rate(found, len(plans)),
        "mean_path_length": mean(lengths),
        "mean_min_contract": mean(min_contracts),
        "mean_negative_risk": mean(risks),
        "future_agreement": safe_rate(agreement, found),
    }


def mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def safe_rate(num: int, den: int) -> float | None:
    return float(num / den) if den else None


def write_outputs(out_dir: Path, weights: dict[str, float] | None, metrics: dict[str, Any]) -> None:
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    if weights is not None:
        (out_dir / "weights.json").write_text(json.dumps({"model_type": "linear_ecg_planner_score", "weights": weights}, indent=2, sort_keys=True), encoding="utf-8")
    report = REPO_ROOT / "reports" / "stage38_ecg_planner_score_train.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage38 ECG Planner Score Calibration",
        "",
        f"- status: `{metrics.get('status')}`",
        f"- pairs: {metrics.get('num_pairs')}",
        f"- weights: `{out_dir / 'weights.json'}`" if weights is not None else "- weights: `NA`",
        f"- metrics: `{metrics_path}`",
        "",
        "## Gate",
        "",
        f"- path_negative_risk_below_shortest: `{metrics.get('gate', {}).get('path_negative_risk_below_shortest')}`",
        f"- path_min_contract_above_shortest: `{metrics.get('gate', {}).get('path_min_contract_above_shortest')}`",
        f"- future_trajectory_agreement_above_shortest: `{metrics.get('gate', {}).get('future_trajectory_agreement_above_shortest')}`",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
