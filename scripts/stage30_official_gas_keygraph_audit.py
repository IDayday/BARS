#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shlex
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from stage30_official_gas_common import (
    ARCHIVED_PRE_STAGE30_STATUS,
    configure_official_env,
    gas_source_identity,
    parse_csv_list,
    parse_seed_list,
    protocol_lock_row,
    recover_node_dataset_indices,
    scan_official_artifacts,
    trajectory_ids_from_dataset,
    write_csv,
)
from stage30_official_gas_instrument import _edge_metadata, _import_official_gas


def _read_csv(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _edge_key(row: dict[str, Any]) -> tuple[int, int] | None:
    try:
        return int(float(row.get("u", ""))), int(float(row.get("v", "")))
    except Exception:
        edge_id = str(row.get("edge_id") or row.get("first_failed_edge") or "")
        if "->" not in edge_id:
            return None
        try:
            u, v = edge_id.split("->", 1)
            return int(float(u)), int(float(v))
        except Exception:
            return None


def _load_keygraph(gas_repo: Path, keygraph_path: Path) -> Any:
    mods = _import_official_gas(gas_repo)
    key_graph = mods["KeyGraph"]()
    key_graph.load_keygraph(os.path.dirname(str(keygraph_path)), os.path.basename(str(keygraph_path)).split("_")[-1].split(".")[0])
    return key_graph


def _usage_index(path_edge_rows: list[dict[str, str]], env_name: str, seed: int) -> dict[tuple[int, int], dict[str, int]]:
    counts: dict[tuple[int, int], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in path_edge_rows:
        if row.get("env_name") != env_name or str(row.get("seed")) not in {str(seed), f"{float(seed):.1f}"}:
            continue
        edge = _edge_key(row)
        if edge is None:
            continue
        try:
            count = int(float(row.get("path_usage_count", 1) or 1))
        except Exception:
            count = 1
        counts[edge]["edge_usage_count"] += count
        if str(row.get("success", "")) in {"1", "1.0"} or str(row.get("path_usage", "")) == "success_path":
            counts[edge]["success_episode_usage"] += count
        else:
            counts[edge]["failure_episode_usage"] += count
        if str(row.get("failure_association", "")) in {"1", "1.0"}:
            counts[edge]["first_failed_frequency"] += 1
    return counts


def _degree(graph: Any, node: int, name: str) -> int | str:
    fn = getattr(graph, name, None)
    if fn is None:
        return ""
    try:
        return int(fn(node))
    except Exception:
        return ""


def _node_rows(art: Any, key_graph: Any, node_map: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    graph_nodes = getattr(key_graph.graph, "nodes", {})
    for node_id in range(len(key_graph.nodes)):
        data = graph_nodes[node_id] if node_id in graph_nodes else {}
        mapped = node_map.get(node_id, {})
        rows.append(
            {
                "stage": "stage30_official_gas_keygraph_audit",
                "evidence_class": "OFFICIAL_GAS_KEYGRAPH_NODE_DIAGNOSTIC",
                "pre_stage30_results_status": ARCHIVED_PRE_STAGE30_STATUS,
                "env_name": art.env_name,
                "seed": art.seed,
                "node_id": node_id,
                "is_task_goal_node": int(node_id >= key_graph.base_node_cnt),
                "degree": _degree(key_graph.graph, node_id, "degree"),
                "in_degree": _degree(key_graph.graph, node_id, "in_degree"),
                "out_degree": _degree(key_graph.graph, node_id, "out_degree"),
                "node_metadata_keys": ",".join(sorted(str(k) for k in getattr(data, "keys", lambda: [])())),
                "dataset_idx": mapped.get("dataset_idx", ""),
                "dataset_idx_available": int(bool(mapped)),
                "embedding_match_dist": mapped.get("embedding_match_dist", ""),
                "embedding_match_tolerance": mapped.get("embedding_match_tolerance", ""),
                "exact_embedding_match": mapped.get("exact_embedding_match", ""),
            }
        )
    return rows


def _edge_rows(
    art: Any,
    key_graph: Any,
    node_map: dict[int, dict[str, Any]],
    traj_ids: dict[int, int],
    usage: dict[tuple[int, int], dict[str, int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for u, v, _ in key_graph.graph.edges(data=True):
        u = int(u)
        v = int(v)
        meta = _edge_metadata(key_graph, u, v, node_map, traj_ids)
        trajectory_semantics_valid = int(
            str(meta.get("edge_dataset_mapping_exact", "")) in {"1", "1.0"}
            and str(meta.get("same_trajectory_available", "")) in {"1", "1.0"}
        )
        counts = usage.get((u, v), {})
        rows.append(
            {
                "stage": "stage30_official_gas_keygraph_audit",
                "evidence_class": "OFFICIAL_GAS_KEYGRAPH_EDGE_DIAGNOSTIC",
                "pre_stage30_results_status": ARCHIVED_PRE_STAGE30_STATUS,
                "env_name": art.env_name,
                "seed": art.seed,
                **meta,
                "trajectory_semantics_valid": trajectory_semantics_valid,
                "src_degree": _degree(key_graph.graph, u, "degree"),
                "dst_degree": _degree(key_graph.graph, v, "degree"),
                "src_out_degree": _degree(key_graph.graph, u, "out_degree"),
                "dst_in_degree": _degree(key_graph.graph, v, "in_degree"),
                "edge_usage_count": counts.get("edge_usage_count", 0),
                "success_episode_usage": counts.get("success_episode_usage", 0),
                "failure_episode_usage": counts.get("failure_episode_usage", 0),
                "first_failed_frequency": counts.get("first_failed_frequency", 0),
            }
        )
    return rows


def _write_report(out_dir: Path, edge_rows: list[dict[str, Any]], node_rows: list[dict[str, Any]]) -> None:
    by_env: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in edge_rows:
        by_env[str(row.get("env_name", ""))].append(row)
    lines = [
        "# Stage30 Official GAS Keygraph Audit Report",
        "",
        "Status: OFFICIAL_GAS_KEYGRAPH_DIAGNOSTIC_ONLY.",
        f"Pre-Stage30 BARS/Stage28/Stage29 evidence: {ARCHIVED_PRE_STAGE30_STATUS}.",
        "The official keygraph is parsed for diagnostics only; missing same/cross/dt metadata is recorded as unavailable, not inferred.",
        "",
        "| env_name | edges | nodes | used_edges | first_failed_edges | recoverable_same_cross_edges |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    node_counts = Counter(str(r.get("env_name", "")) for r in node_rows)
    for env_name, part in sorted(by_env.items()):
        used = sum(1 for r in part if int(float(r.get("edge_usage_count", 0) or 0)) > 0)
        failed = sum(1 for r in part if int(float(r.get("first_failed_frequency", 0) or 0)) > 0)
        recoverable = sum(1 for r in part if str(r.get("same_trajectory_available", "")) in {"1", "1.0"})
        lines.append(f"| {env_name} | {len(part)} | {node_counts.get(env_name, 0)} | {used} | {failed} | {recoverable} |")
    lines.extend(["", "## Files", ""])
    lines.append(f"- keygraph edges: `{out_dir / 'official_gas_keygraph_edges.csv'}`")
    lines.append(f"- keygraph nodes: `{out_dir / 'official_gas_keygraph_nodes.csv'}`")
    lines.append(f"- protocol lock: `{out_dir / 'protocol_lock.csv'}`")
    (out_dir / "keygraph_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage30 official GAS keygraph diagnostic-only audit.")
    parser.add_argument("--artifact-root", default="artifacts/gas_ogbench_offline_full_20260522_165138")
    parser.add_argument("--gas-repo-path", default="external_src/GAS")
    parser.add_argument("--out-root", default="runs_stage30_official_gas/keygraph_audit")
    parser.add_argument("--envs", default="antmaze-medium-navigate-v0")
    parser.add_argument("--seeds", default="44")
    parser.add_argument("--path-edge-csv", default="")
    parser.add_argument("--recover-dataset-indices", type=int, default=0)
    parser.add_argument("--node-map-batch-size", type=int, default=4096)
    parser.add_argument("--node-map-tolerance", type=float, default=1e-5)
    parser.add_argument("--gpu", default="cpu")
    args = parser.parse_args()

    os.environ.update(configure_official_env(args.gpu))
    out_dir = Path(args.out_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    gas_repo = Path(args.gas_repo_path)
    source_identity = gas_source_identity(gas_repo)
    command_line = " ".join(shlex.quote(x) for x in [sys.executable, *sys.argv])
    artifacts = scan_official_artifacts(Path(args.artifact_root), parse_csv_list(args.envs), parse_seed_list(args.seeds))
    path_edge_rows = _read_csv(Path(args.path_edge_csv) if args.path_edge_csv else None)
    all_edge_rows: list[dict[str, Any]] = []
    all_node_rows: list[dict[str, Any]] = []
    protocol_rows: list[dict[str, Any]] = []
    for art in artifacts:
        key_graph = _load_keygraph(gas_repo, art.keygraph_path)
        if args.recover_dataset_indices:
            node_map = recover_node_dataset_indices(
                key_graph.nodes,
                art.dataset_embeddings_path,
                base_node_count=int(key_graph.base_node_cnt),
                tolerance=float(args.node_map_tolerance),
                batch_size=args.node_map_batch_size,
            )
            traj_ids = trajectory_ids_from_dataset(art.dataset_npz_path)
        else:
            node_map = {}
            traj_ids = {}
        usage = _usage_index(path_edge_rows, art.env_name, art.seed)
        protocol_rows.append(
            protocol_lock_row(
                art,
                gas_repo,
                stage="stage30_official_gas_keygraph_audit",
                evidence_class="OFFICIAL_GAS_PROTOCOL_LOCK",
                wrapper_status="OFFICIAL_GAS_KEYGRAPH_DIAGNOSTIC_ONLY",
                command_line=command_line,
                task_id="keygraph_audit",
                subgoal_horizon=int(key_graph.way_steps),
                gpu=args.gpu,
                source_identity=source_identity,
                extra={
                    "recover_dataset_indices": args.recover_dataset_indices,
                    "node_map_tolerance": args.node_map_tolerance,
                    "path_edge_csv": args.path_edge_csv,
                },
            )
        )
        all_node_rows.extend(_node_rows(art, key_graph, node_map))
        all_edge_rows.extend(_edge_rows(art, key_graph, node_map, traj_ids, usage))
    write_csv(out_dir / "official_gas_keygraph_edges.csv", all_edge_rows)
    write_csv(out_dir / "official_gas_keygraph_nodes.csv", all_node_rows)
    write_csv(out_dir / "protocol_lock.csv", protocol_rows)
    _write_report(out_dir, all_edge_rows, all_node_rows)
    print(out_dir / "keygraph_audit_report.md")


if __name__ == "__main__":
    main()
