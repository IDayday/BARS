#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

from bars.gas_bars.support_keygraph import load_edge_scores_csv, load_keygraph_pickle, save_keygraph_pickle


@dataclass(frozen=True)
class PathMetrics:
    path_edges: int
    base_graph_cost: float
    num_unsupported_edges: int
    unsupported_edge_fraction: float
    mean_same_traj_support: float


def _parse_name_path(values: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected NAME=PATH, got {value!r}")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"empty candidate name in {value!r}")
        if name in out:
            raise ValueError(f"duplicate candidate name: {name}")
        out[name] = Path(raw_path)
    return out


def _lookup_mapping_key(mapping: dict[Any, Any], task_id: int) -> Any:
    if task_id in mapping:
        return task_id
    task_str = str(task_id)
    if task_str in mapping:
        return task_str
    for key in mapping:
        try:
            if int(key) == int(task_id):
                return key
        except Exception:
            continue
    raise KeyError(task_id)


def _edge_rows(edge_scores: pd.DataFrame) -> dict[tuple[int, int], dict[str, Any]]:
    return {(int(row["u"]), int(row["v"])): row for row in edge_scores.to_dict("records")}


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else float(default)
    except Exception:
        return float(default)


def _path_edges(path: list[int]) -> list[tuple[int, int]]:
    return [(int(u), int(v)) for u, v in zip(path[:-1], path[1:])]


def _copy_keygraph_for_cache_edit(keygraph: Any) -> Any:
    """Copy a GAS keygraph while avoiding JAX array/device deep copies."""
    payload = dict(getattr(keygraph, "__dict__", {}))
    out = SimpleNamespace(**payload)
    out.task_paths_dict = copy.deepcopy(getattr(keygraph, "task_paths_dict", {}) or {})
    out.task_paths_dist_dict = copy.deepcopy(getattr(keygraph, "task_paths_dist_dict", {}) or {})
    return out


def compute_path_metrics(
    base_graph: Any,
    edge_lookup: dict[tuple[int, int], dict[str, Any]],
    path: list[int],
) -> PathMetrics:
    edges = _path_edges(path)
    cost = 0.0
    unsupported = 0
    support_values: list[float] = []
    for u, v in edges:
        if not base_graph.has_edge(u, v):
            cost = float("inf")
        elif math.isfinite(cost):
            cost += _finite_float(base_graph[u][v].get("weight"), 0.0)
        row = edge_lookup.get((u, v), {})
        if _finite_float(row.get("local_support"), 1.0) <= 0:
            unsupported += 1
        support_values.append(_finite_float(row.get("same_traj_support"), 0.0))
    n_edges = len(edges)
    return PathMetrics(
        path_edges=int(n_edges),
        base_graph_cost=float(cost),
        num_unsupported_edges=int(unsupported),
        unsupported_edge_fraction=float(unsupported / n_edges) if n_edges else 0.0,
        mean_same_traj_support=float(sum(support_values) / len(support_values)) if support_values else 0.0,
    )


def _passes_improvement(
    mode: str,
    support_gain: float,
    unsupported_gain: float,
    min_support_gain: float,
    min_unsupported_gain: float,
) -> bool:
    support_ok = support_gain >= min_support_gain
    unsupported_ok = unsupported_gain >= min_unsupported_gain
    if mode == "any":
        return support_ok or unsupported_ok
    if mode == "both":
        return support_ok and unsupported_ok
    if mode == "support":
        return support_ok
    if mode == "unsupported":
        return unsupported_ok
    raise ValueError(f"unknown improvement mode: {mode}")


def path_local_gated_mix(
    base_keygraph: Any,
    candidate_keygraphs: dict[str, Any],
    edge_scores: pd.DataFrame,
    *,
    max_base_cost_ratio: float,
    max_edge_delta: int,
    min_support_gain: float,
    min_unsupported_gain: float,
    improvement_mode: str,
    unsupported_weight: float,
    support_weight: float,
    cost_penalty: float,
    edge_penalty: float,
    distance_mode: str,
) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    if distance_mode not in {"base_cost", "candidate_dist"}:
        raise ValueError("distance_mode must be base_cost or candidate_dist")
    out = _copy_keygraph_for_cache_edit(base_keygraph)
    base_paths_by_task = getattr(base_keygraph, "task_paths_dict", {}) or {}
    base_dists_by_task = getattr(base_keygraph, "task_paths_dist_dict", {}) or {}
    base_node_count = int(getattr(base_keygraph, "base_node_cnt", len(getattr(base_keygraph, "nodes", []))) or 0)
    edge_lookup = _edge_rows(edge_scores)

    decision_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for raw_task_id, base_paths in base_paths_by_task.items():
        task_id = int(raw_task_id)
        out_task_key = _lookup_mapping_key(out.task_paths_dict, task_id)
        out_dist_key = _lookup_mapping_key(out.task_paths_dist_dict, task_id)
        base_dist_key = _lookup_mapping_key(base_dists_by_task, task_id)
        for raw_source, raw_base_path in base_paths.items():
            source = int(raw_source)
            if source >= base_node_count:
                continue
            base_path = [int(x) for x in raw_base_path]
            base_metrics = compute_path_metrics(base_keygraph.graph, edge_lookup, base_path)
            best: dict[str, Any] | None = None
            for method, candidate in candidate_keygraphs.items():
                try:
                    cand_task_key = _lookup_mapping_key(candidate.task_paths_dict, task_id)
                    cand_dist_task_key = _lookup_mapping_key(candidate.task_paths_dist_dict, task_id)
                except KeyError:
                    continue
                cand_paths = candidate.task_paths_dict[cand_task_key]
                if raw_source not in cand_paths and source not in cand_paths and str(source) not in cand_paths:
                    continue
                cand_source_key = raw_source if raw_source in cand_paths else (source if source in cand_paths else str(source))
                cand_path = [int(x) for x in cand_paths[cand_source_key]]
                if tuple(cand_path) == tuple(base_path):
                    continue
                cand_metrics = compute_path_metrics(base_keygraph.graph, edge_lookup, cand_path)
                if not math.isfinite(cand_metrics.base_graph_cost):
                    continue
                base_cost = max(base_metrics.base_graph_cost, 1e-9)
                cost_ratio = cand_metrics.base_graph_cost / base_cost
                edge_delta = cand_metrics.path_edges - base_metrics.path_edges
                support_gain = cand_metrics.mean_same_traj_support - base_metrics.mean_same_traj_support
                unsupported_gain = base_metrics.unsupported_edge_fraction - cand_metrics.unsupported_edge_fraction
                gate_ok = (
                    cost_ratio <= max_base_cost_ratio
                    and edge_delta <= max_edge_delta
                    and _passes_improvement(
                        improvement_mode,
                        support_gain,
                        unsupported_gain,
                        min_support_gain,
                        min_unsupported_gain,
                    )
                )
                score = (
                    unsupported_weight * unsupported_gain
                    + support_weight * support_gain
                    - cost_penalty * max(0.0, cost_ratio - 1.0)
                    - edge_penalty * max(0, edge_delta)
                )
                cand_dist_map = candidate.task_paths_dist_dict[cand_dist_task_key]
                cand_dist = cand_dist_map.get(cand_source_key, cand_dist_map.get(source, cand_dist_map.get(str(source))))
                row = {
                    "task_id": task_id,
                    "source": source,
                    "candidate_method": method,
                    "gate_ok": bool(gate_ok),
                    "score": float(score),
                    "cost_ratio": float(cost_ratio),
                    "edge_delta": int(edge_delta),
                    "support_gain": float(support_gain),
                    "unsupported_gain": float(unsupported_gain),
                    "base_path_edges": int(base_metrics.path_edges),
                    "candidate_path_edges": int(cand_metrics.path_edges),
                    "base_graph_cost": float(base_metrics.base_graph_cost),
                    "candidate_base_graph_cost": float(cand_metrics.base_graph_cost),
                    "base_unsupported_edge_fraction": float(base_metrics.unsupported_edge_fraction),
                    "candidate_unsupported_edge_fraction": float(cand_metrics.unsupported_edge_fraction),
                    "base_mean_same_traj_support": float(base_metrics.mean_same_traj_support),
                    "candidate_mean_same_traj_support": float(cand_metrics.mean_same_traj_support),
                    "candidate_dist": float(cand_dist) if cand_dist is not None else float("nan"),
                    "base_path_nodes": " ".join(str(x) for x in base_path),
                    "candidate_path_nodes": " ".join(str(x) for x in cand_path),
                }
                decision_rows.append(row)
                if gate_ok and (best is None or score > float(best["score"])):
                    best = {**row, "candidate_path": cand_path}
            if best is None:
                continue
            selected_path = list(best.pop("candidate_path"))
            out.task_paths_dict[out_task_key][raw_source] = selected_path
            if distance_mode == "base_cost":
                out.task_paths_dist_dict[out_dist_key][raw_source] = float(best["candidate_base_graph_cost"])
            else:
                out.task_paths_dist_dict[out_dist_key][raw_source] = float(best["candidate_dist"])
            selected_rows.append(best)

    decisions = pd.DataFrame(decision_rows)
    selected = pd.DataFrame(selected_rows)
    setattr(out, "bars_path_local_gate", {
        "max_base_cost_ratio": max_base_cost_ratio,
        "max_edge_delta": max_edge_delta,
        "min_support_gain": min_support_gain,
        "min_unsupported_gain": min_unsupported_gain,
        "improvement_mode": improvement_mode,
        "distance_mode": distance_mode,
    })
    summary = {
        "base_node_count": int(base_node_count),
        "num_candidate_decisions": int(len(decisions)),
        "num_selected_paths": int(len(selected)),
        "selected_path_rate": float(len(selected) / max(1, sum(len(v) for v in base_paths_by_task.values()))),
        "candidate_methods": sorted(candidate_keygraphs),
        "gate": getattr(out, "bars_path_local_gate"),
    }
    if not selected.empty:
        summary.update({
            "selected_by_method": selected["candidate_method"].value_counts().sort_index().to_dict(),
            "mean_selected_cost_ratio": float(selected["cost_ratio"].mean()),
            "mean_selected_support_gain": float(selected["support_gain"].mean()),
            "mean_selected_unsupported_gain": float(selected["unsupported_gain"].mean()),
            "mean_selected_edge_delta": float(selected["edge_delta"].mean()),
        })
    return out, selected, summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-keygraph", required=True, type=Path)
    parser.add_argument("--edge-scores-csv", required=True, type=Path)
    parser.add_argument("--candidate-keygraph", action="append", default=[], help="Repeat as NAME=PATH.")
    parser.add_argument("--output-keygraph", required=True, type=Path)
    parser.add_argument("--selection-csv", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--max-base-cost-ratio", type=float, default=1.02)
    parser.add_argument("--max-edge-delta", type=int, default=0)
    parser.add_argument("--min-support-gain", type=float, default=1e-9)
    parser.add_argument("--min-unsupported-gain", type=float, default=1e-9)
    parser.add_argument("--improvement-mode", choices=["any", "both", "support", "unsupported"], default="any")
    parser.add_argument("--unsupported-weight", type=float, default=20.0)
    parser.add_argument("--support-weight", type=float, default=1.0)
    parser.add_argument("--cost-penalty", type=float, default=10.0)
    parser.add_argument("--edge-penalty", type=float, default=5.0)
    parser.add_argument("--distance-mode", choices=["base_cost", "candidate_dist"], default="base_cost")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    candidate_paths = _parse_name_path(args.candidate_keygraph)
    if not candidate_paths:
        raise ValueError("at least one --candidate-keygraph is required")
    base = load_keygraph_pickle(args.base_keygraph)
    candidates = {name: load_keygraph_pickle(path) for name, path in candidate_paths.items()}
    edge_scores = load_edge_scores_csv(args.edge_scores_csv)
    mixed, selection, summary = path_local_gated_mix(
        base,
        candidates,
        edge_scores,
        max_base_cost_ratio=args.max_base_cost_ratio,
        max_edge_delta=args.max_edge_delta,
        min_support_gain=args.min_support_gain,
        min_unsupported_gain=args.min_unsupported_gain,
        improvement_mode=args.improvement_mode,
        unsupported_weight=args.unsupported_weight,
        support_weight=args.support_weight,
        cost_penalty=args.cost_penalty,
        edge_penalty=args.edge_penalty,
        distance_mode=args.distance_mode,
    )

    save_keygraph_pickle(mixed, args.output_keygraph)
    args.selection_csv.parent.mkdir(parents=True, exist_ok=True)
    selection.to_csv(args.selection_csv, index=False)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary.update({
        "base_keygraph": str(args.base_keygraph),
        "edge_scores_csv": str(args.edge_scores_csv),
        "candidate_keygraphs": {name: str(path) for name, path in candidate_paths.items()},
        "output_keygraph": str(args.output_keygraph),
        "selection_csv": str(args.selection_csv),
    })
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
