#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _load_keygraph(path: str | Path) -> Any:
    with Path(path).open("rb") as fh:
        data = pickle.load(fh)
    if isinstance(data, dict):
        obj = type("LoadedGASKeyGraph", (), {})()
        for key, value in data.items():
            setattr(obj, key, value)
        return obj
    return data


def _edge_lookup(edge_scores: pd.DataFrame) -> dict[tuple[int, int], dict[str, Any]]:
    rows: dict[tuple[int, int], dict[str, Any]] = {}
    for row in edge_scores.to_dict("records"):
        rows[(int(row["u"]), int(row["v"]))] = row
    return rows


def _path_edges(path: list[int]) -> list[tuple[int, int]]:
    return [(int(a), int(b)) for a, b in zip(path[:-1], path[1:])]


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
        return x if np.isfinite(x) else float(default)
    except Exception:
        return float(default)


def _path_row(method: str, key_graph: Any, edge_scores: pd.DataFrame, edge_rows: dict[tuple[int, int], dict[str, Any]], task_id: int, source: int, path: list[int]) -> dict[str, Any]:
    graph = key_graph.graph
    edges = _path_edges(path)
    edge_sources: list[str] = []
    unsupported = 0
    unsupported_scc = 0
    goal_edges = 0
    scc_edges = 0
    same_traj_support = []
    cost = 0.0
    for u, v in edges:
        attrs = graph[u][v] if graph.has_edge(u, v) else {}
        cost += _finite_float(attrs.get("weight"), 0.0)
        row = edge_rows.get((u, v), {})
        edge_source = str(row.get("edge_source", "missing_score"))
        edge_sources.append(edge_source)
        local_support = _finite_float(row.get("local_support"), 1.0)
        if local_support <= 0:
            unsupported += 1
        if edge_source == "gas_scc_connector":
            scc_edges += 1
            if local_support <= 0:
                unsupported_scc += 1
        if edge_source == "gas_goal_connector":
            goal_edges += 1
        if "same_traj_support" in row:
            same_traj_support.append(_finite_float(row.get("same_traj_support"), 0.0))
    n_edges = len(edges)
    return {
        "method": method,
        "task_id": int(task_id),
        "source": int(source),
        "path_nodes": " ".join(str(int(x)) for x in path),
        "path_edges": n_edges,
        "path_cost": float(cost),
        "num_unsupported_edges": int(unsupported),
        "unsupported_edge_fraction": float(unsupported / n_edges) if n_edges else 0.0,
        "num_scc_edges": int(scc_edges),
        "num_unsupported_scc_edges": int(unsupported_scc),
        "unsupported_scc_edge_fraction": float(unsupported_scc / n_edges) if n_edges else 0.0,
        "num_goal_connector_edges": int(goal_edges),
        "mean_same_traj_support": float(np.mean(same_traj_support)) if same_traj_support else 0.0,
        "edge_source_sequence": " ".join(edge_sources),
    }


def audit_paths(method_paths: list[tuple[str, Path]], edge_scores_path: Path, out_dir: Path) -> dict[str, str]:
    edge_scores = pd.read_csv(edge_scores_path)
    edge_scores["u"] = pd.to_numeric(edge_scores["u"], errors="raise").astype(int)
    edge_scores["v"] = pd.to_numeric(edge_scores["v"], errors="raise").astype(int)
    edge_rows = _edge_lookup(edge_scores)
    keygraphs = [(name, _load_keygraph(path)) for name, path in method_paths]
    if not keygraphs:
        raise ValueError("At least one keygraph is required")
    base_count = int(getattr(keygraphs[0][1], "base_node_cnt", len(getattr(keygraphs[0][1], "nodes", []))) or 0)

    rows: list[dict[str, Any]] = []
    path_by_method: dict[str, dict[tuple[int, int], tuple[int, ...]]] = {}
    for method, kg in keygraphs:
        method_paths_map: dict[tuple[int, int], tuple[int, ...]] = {}
        for raw_task_id, paths_for_task in (getattr(kg, "task_paths_dict", {}) or {}).items():
            task_id = int(raw_task_id)
            for raw_source, raw_path in paths_for_task.items():
                source = int(raw_source)
                if source >= base_count:
                    continue
                path = [int(x) for x in raw_path]
                method_paths_map[(task_id, source)] = tuple(path)
                rows.append(_path_row(method, kg, edge_scores, edge_rows, task_id, source, path))
        path_by_method[method] = method_paths_map

    path_audit = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    path_csv = out_dir / "path_audit.csv"
    path_audit.to_csv(path_csv, index=False)
    summary = (
        path_audit.groupby("method")
        .agg(
            num_paths=("source", "count"),
            mean_path_edges=("path_edges", "mean"),
            mean_path_cost=("path_cost", "mean"),
            mean_unsupported_edge_fraction=("unsupported_edge_fraction", "mean"),
            mean_unsupported_scc_edge_fraction=("unsupported_scc_edge_fraction", "mean"),
            mean_num_scc_edges=("num_scc_edges", "mean"),
            mean_num_unsupported_scc_edges=("num_unsupported_scc_edges", "mean"),
            mean_same_traj_support=("mean_same_traj_support", "mean"),
        )
        .reset_index()
    )
    summary_csv = out_dir / "path_summary.csv"
    summary.to_csv(summary_csv, index=False)

    ref_method = keygraphs[0][0]
    ref_paths = path_by_method[ref_method]
    diff_rows: list[dict[str, Any]] = []
    for method, paths in path_by_method.items():
        if method == ref_method:
            continue
        common = sorted(set(ref_paths) & set(paths))
        changed = [key for key in common if ref_paths[key] != paths[key]]
        diff_rows.append(
            {
                "reference_method": ref_method,
                "method": method,
                "num_common_paths": int(len(common)),
                "num_changed_paths": int(len(changed)),
                "path_change_rate": float(len(changed) / len(common)) if common else 0.0,
            }
        )
    diff = pd.DataFrame(diff_rows)
    diff_csv = out_dir / "path_diff_summary.csv"
    diff.to_csv(diff_csv, index=False)

    top_changed_rows: list[dict[str, Any]] = []
    audit_index = path_audit.set_index(["method", "task_id", "source"])
    for method, paths in path_by_method.items():
        if method == ref_method:
            continue
        common = sorted(set(ref_paths) & set(paths))
        for task_id, source in common:
            if ref_paths[(task_id, source)] == paths[(task_id, source)]:
                continue
            ref_row = audit_index.loc[(ref_method, task_id, source)]
            cur_row = audit_index.loc[(method, task_id, source)]
            top_changed_rows.append(
                {
                    "method": method,
                    "task_id": int(task_id),
                    "source": int(source),
                    "delta_path_cost": float(cur_row["path_cost"] - ref_row["path_cost"]),
                    "delta_path_edges": int(cur_row["path_edges"] - ref_row["path_edges"]),
                    "delta_unsupported_edges": int(cur_row["num_unsupported_edges"] - ref_row["num_unsupported_edges"]),
                    "delta_unsupported_scc_edges": int(cur_row["num_unsupported_scc_edges"] - ref_row["num_unsupported_scc_edges"]),
                    "reference_path_nodes": ref_row["path_nodes"],
                    "method_path_nodes": cur_row["path_nodes"],
                }
            )
    top_changed = pd.DataFrame(top_changed_rows)
    if not top_changed.empty:
        top_changed = top_changed.reindex(top_changed["delta_path_cost"].abs().sort_values(ascending=False).index).head(200)
    top_changed_csv = out_dir / "top_changed_paths.csv"
    top_changed.to_csv(top_changed_csv, index=False)

    summary_json = {
        "edge_scores_path": str(edge_scores_path),
        "reference_method": ref_method,
        "base_node_count": int(base_count),
        "methods": [name for name, _ in keygraphs],
        "summary": summary.to_dict("records"),
        "diff": diff.to_dict("records"),
    }
    summary_json_path = out_dir / "path_audit_summary.json"
    summary_json_path.write_text(json.dumps(summary_json, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "path_audit_csv": str(path_csv),
        "path_summary_csv": str(summary_csv),
        "path_diff_summary_csv": str(diff_csv),
        "top_changed_paths_csv": str(top_changed_csv),
        "summary_json": str(summary_json_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit GAS cached task paths against edge support scores.")
    parser.add_argument("--edge-scores-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--keygraph",
        action="append",
        nargs=2,
        metavar=("METHOD", "PATH"),
        required=True,
        help="Method name and keygraph.pkl path. The first keygraph is the reference.",
    )
    args = parser.parse_args()
    paths = [(str(name), Path(path)) for name, path in args.keygraph]
    result = audit_paths(paths, Path(args.edge_scores_csv), Path(args.out_dir))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
