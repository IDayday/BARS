#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import Counter, defaultdict, deque
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external_src" / "GAS"))

from cage.contract_graph import BoundaryContract, ContractEdge, ContractFunnelNode, ContractGraph
from cage.contract_model import ContractScorer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a transition-augmented CAGE execution-contract graph.")
    parser.add_argument("--base_contract_graph", default="results/cage_ecg/contract_graph/contract_graph.json")
    parser.add_argument("--contract_dataset_path", default="results/cage_clp1/datasets/closed_loop_contracts.jsonl")
    parser.add_argument("--segment_capture_roots", nargs="*", default=["results/cage_clp1/segment_capture", "results/cage_clp1/segment_capture_candidate"])
    parser.add_argument("--qtrain_path", default="")
    parser.add_argument("--contract_model_path", default="results/cage_v02_contract/models/contract_model.json")
    parser.add_argument("--out_dir", default="results/cage_ecg/transition_contract_graph")
    parser.add_argument("--max_transition_edges", type=int, default=50000)
    parser.add_argument("--max_knn_edges_per_node", type=int, default=8)
    parser.add_argument("--min_contract_lcb", type=float, default=-1.0)
    parser.add_argument("--clear", action="store_true")
    return parser


class GraphBuilder:
    def __init__(self, scorer: ContractScorer, min_contract_lcb: float, max_added_edges: int):
        self.scorer = scorer
        self.min_contract_lcb = float(min_contract_lcb)
        self.max_added_edges = int(max_added_edges)
        self.nodes: dict[str, ContractFunnelNode] = {}
        self.edges: dict[str, ContractEdge] = {}
        self.boundaries: dict[str, BoundaryContract] = {}
        self.key_to_id: dict[str, str] = {}
        self.edge_keys: set[tuple[str, str, str]] = set()
        self.added_edges = 0
        self.skipped_low_lcb = 0
        self.skipped_missing_phi = 0
        self.observed_edge_sequences: list[list[str]] = []

    def load_base(self, graph: ContractGraph) -> None:
        for node in graph.nodes.values():
            copied = ContractFunnelNode(**asdict(node))
            self.nodes[copied.node_id] = copied
            self.key_to_id[endpoint_key(copied.center_phi, copied.env_name or "unknown")] = copied.node_id
        for edge in graph.edges.values():
            data = asdict(edge)
            data["edge_type"] = "original_contract"
            copied = ContractEdge(**data)
            self.edges[copied.edge_id] = copied
            self.edge_keys.add((copied.src, copied.dst, copied.edge_type or ""))
        for key, boundary in graph.boundary_contracts.items():
            self.boundaries[key] = BoundaryContract(**asdict(boundary))

    def node_for(self, phi: Any, env_name: str, source_id: str | None = None) -> str | None:
        if phi is None:
            self.skipped_missing_phi += 1
            return None
        arr = np.asarray(phi, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            self.skipped_missing_phi += 1
            return None
        key = endpoint_key(arr.tolist(), env_name)
        if key not in self.key_to_id:
            node_id = f"tn{len(self.key_to_id):07d}"
            self.key_to_id[key] = node_id
            self.nodes[node_id] = ContractFunnelNode(
                node_id=node_id,
                center_phi=arr.astype(float).tolist(),
                radius=0.0,
                support_count=0,
                env_name=env_name,
                source_node_ids=[],
            )
        node = self.nodes[self.key_to_id[key]]
        node.support_count += 1
        if source_id and source_id not in node.source_node_ids:
            node.source_node_ids.append(str(source_id))
        return node.node_id

    def add_scored_edge(
        self,
        phi_s: Any,
        phi_g: Any,
        *,
        env_name: str,
        edge_type: str,
        source_id: str | None = None,
        observed_sequence: list[str] | None = None,
        q_train_support: float | None = None,
        gas_edge_exists: bool = False,
        allow_over_budget: bool = False,
        extra_features: dict[str, Any] | None = None,
    ) -> str | None:
        if not allow_over_budget and self.added_edges >= self.max_added_edges:
            return None
        src = self.node_for(phi_s, env_name, source_id)
        dst = self.node_for(phi_g, env_name, source_id)
        if src is None or dst is None or src == dst:
            return None
        edge_key = (src, dst, edge_type)
        if edge_key in self.edge_keys:
            return None
        d_phi = float(np.linalg.norm(np.asarray(phi_g, dtype=np.float32).reshape(-1) - np.asarray(phi_s, dtype=np.float32).reshape(-1)))
        features = {
            "phi_s": phi_s,
            "phi_g": phi_g,
            "d_phi": d_phi,
            "target_mode": edge_type,
            "path_position": -1,
            "final_phase": edge_type == "final_goal_candidate",
            "recovery_candidate": edge_type == "recovery_candidate",
            "q_train_support": q_train_support if q_train_support is not None else -1.0,
            "env_name": env_name,
            "fallback_tau": 10.0,
        }
        if extra_features:
            features.update(extra_features)
        pred = self.scorer.predict(features)
        if float(pred.lower_confidence_bound) < self.min_contract_lcb:
            self.skipped_low_lcb += 1
            return None
        edge_id = f"te{len(self.edges):08d}"
        negative = float(pred.predicted_negative_progress)
        lcb = float(pred.lower_confidence_bound)
        self.edges[edge_id] = ContractEdge(
            edge_id=edge_id,
            src=src,
            dst=dst,
            d_phi=d_phi,
            gas_edge_exists=bool(gas_edge_exists),
            contract_score=float(pred.predicted_contract_positive),
            contract_lcb=lcb,
            predicted_hit=float(pred.predicted_hit),
            predicted_contract_positive=float(pred.predicted_contract_positive),
            predicted_negative_progress=negative,
            uncertainty=float(pred.uncertainty),
            q_train_support=q_train_support,
            edge_type=edge_type,
            bottleneck_score=min(lcb, 1.0 - negative),
        )
        self.edge_keys.add(edge_key)
        self.added_edges += 1
        if observed_sequence is not None:
            observed_sequence.append(edge_id)
        return edge_id

    def add_boundaries_from_sequences(self) -> None:
        for sequence in self.observed_edge_sequences:
            for prev, nxt in zip(sequence[:-1], sequence[1:]):
                if prev not in self.edges or nxt not in self.edges:
                    continue
                prev_edge = self.edges[prev]
                next_edge = self.edges[nxt]
                key = f"{prev}__to__{nxt}"
                compatibility = min(_float(prev_edge.contract_lcb, 0.0), _float(next_edge.contract_lcb, 0.0))
                risk = max(_float(prev_edge.predicted_negative_progress, 0.0), _float(next_edge.predicted_negative_progress, 0.0))
                if key in self.boundaries:
                    self.boundaries[key].observed_transition_count += 1
                    continue
                self.boundaries[key] = BoundaryContract(
                    prev_edge_id=prev,
                    next_edge_id=nxt,
                    compatibility_score=compatibility,
                    boundary_risk=risk,
                    observed_transition_count=1,
                )

    def add_conservative_boundaries(self, limit: int = 20000) -> int:
        incoming: dict[str, list[str]] = defaultdict(list)
        outgoing: dict[str, list[str]] = defaultdict(list)
        for edge in self.edges.values():
            incoming[edge.dst].append(edge.edge_id)
            outgoing[edge.src].append(edge.edge_id)
        added = 0
        for node_id, prev_ids in incoming.items():
            for prev in prev_ids[:8]:
                for nxt in outgoing.get(node_id, [])[:8]:
                    key = f"{prev}__to__{nxt}"
                    if key in self.boundaries:
                        continue
                    prev_edge = self.edges[prev]
                    next_edge = self.edges[nxt]
                    compatibility = min(_float(prev_edge.contract_lcb, 0.0), _float(next_edge.contract_lcb, 0.0)) * 0.75
                    risk = max(_float(prev_edge.predicted_negative_progress, 0.0), _float(next_edge.predicted_negative_progress, 0.0))
                    self.boundaries[key] = BoundaryContract(
                        prev_edge_id=prev,
                        next_edge_id=nxt,
                        compatibility_score=compatibility,
                        boundary_risk=risk,
                        observed_transition_count=0,
                    )
                    added += 1
                    if added >= limit:
                        return added
        return added

    def build(self, metadata: dict[str, Any]) -> ContractGraph:
        graph = ContractGraph(
            nodes=self.nodes,
            edges=self.edges,
            boundary_contracts=self.boundaries,
            metadata=metadata,
        )
        return graph


def add_contract_dataset_edges(builder: GraphBuilder, rows: list[dict[str, Any]]) -> Counter:
    counts: Counter[str] = Counter()
    by_segment: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        env = str(row.get("env_name") or "unknown")
        phi_s = row.get("phi_s", row.get("phi_start"))
        phi_g = row.get("phi_g", row.get("phi_target"))
        mode = str(row.get("target_mode") or row.get("pair_source") or "original_contract")
        if "recovery" in mode or row.get("recovery_candidate"):
            edge_type = "recovery_candidate"
        elif "final" in mode or row.get("final_phase"):
            edge_type = "final_goal_candidate"
        elif mode in {"nearest_path_target", "farther_path_target"}:
            edge_type = "path_adjacency"
        else:
            edge_type = "original_contract"
        source_id = row.get("source_segment_id") or row.get("source_probe_id")
        sequence = by_segment[str(source_id)] if source_id else None
        edge_id = builder.add_scored_edge(
            phi_s,
            phi_g,
            env_name=env,
            edge_type=edge_type,
            source_id=str(source_id) if source_id else None,
            observed_sequence=sequence,
            q_train_support=_float(row.get("q_train_support"), None),
            gas_edge_exists=edge_type == "original_contract",
            extra_features={
                "path_position": row.get("path_position", -1),
                "final_phase": bool(row.get("final_phase", False)),
                "recovery_candidate": bool(row.get("recovery_candidate", False)),
            },
        )
        if edge_id:
            counts[edge_type] += 1
    builder.observed_edge_sequences.extend([seq for seq in by_segment.values() if len(seq) >= 2])
    return counts


def add_segment_capture_edges(builder: GraphBuilder, roots: list[str]) -> Counter:
    counts: Counter[str] = Counter()
    for path in segment_files(roots):
        records = load_jsonl(path)
        records.sort(key=lambda r: (str(r.get("env_name")), int(_float(r.get("seed"), 0)), str(r.get("variant")), int(_float(r.get("episode_idx"), 0)), int(_float(r.get("segment_idx"), 0))))
        by_episode: dict[tuple[str, int, str, int], list[dict[str, Any]]] = defaultdict(list)
        for rec in records:
            key = (str(rec.get("env_name") or "unknown"), int(_float(rec.get("seed"), 0)), str(rec.get("variant") or "unknown"), int(_float(rec.get("episode_idx"), 0)))
            by_episode[key].append(rec)
        for episode_records in by_episode.values():
            sequence: list[str] = []
            for rec in episode_records:
                env = str(rec.get("env_name") or "unknown")
                sid = rec.get("segment_id")
                start_phi = rec.get("start_phi")
                end_phi = rec.get("end_phi")
                target_phi = rec.get("target_phi")
                final_goal_phi = rec.get("final_goal_phi")
                if start_phi is not None and end_phi is not None:
                    edge_id = builder.add_scored_edge(start_phi, end_phi, env_name=env, edge_type="temporal_transition", source_id=sid, observed_sequence=sequence)
                    counts["temporal_transition"] += int(edge_id is not None)
                if final_goal_phi is not None and start_phi is not None:
                    edge_id = builder.add_scored_edge(start_phi, final_goal_phi, env_name=env, edge_type="final_goal_candidate", source_id=sid, observed_sequence=None, extra_features={"final_phase": True})
                    counts["final_goal_candidate"] += int(edge_id is not None)
                if rec.get("recovery_active") or "recovery" in str(rec.get("target_source") or "").lower():
                    edge_id = builder.add_scored_edge(start_phi, target_phi, env_name=env, edge_type="recovery_candidate", source_id=sid, observed_sequence=None, extra_features={"recovery_candidate": True})
                    counts["recovery_candidate"] += int(edge_id is not None)
                path_phi = rec.get("path_phi")
                if isinstance(path_phi, list) and path_phi and isinstance(path_phi[0], list):
                    for left, right in zip(path_phi[:-1], path_phi[1:]):
                        edge_id = builder.add_scored_edge(left, right, env_name=env, edge_type="path_adjacency", source_id=sid, observed_sequence=sequence, gas_edge_exists=True)
                        counts["path_adjacency"] += int(edge_id is not None)
                        if builder.added_edges >= builder.max_added_edges:
                            break
                if builder.added_edges >= builder.max_added_edges:
                    break
            if len(sequence) >= 2:
                builder.observed_edge_sequences.append(sequence)
            if builder.added_edges >= builder.max_added_edges:
                break
    return counts


def add_qtrain_edges(builder: GraphBuilder, qtrain_path: str) -> Counter:
    counts: Counter[str] = Counter()
    if not qtrain_path:
        return counts
    for row in load_jsonl(Path(qtrain_path)):
        env = str(row.get("env_name") or "unknown")
        edge_id = builder.add_scored_edge(
            row.get("phi_s", row.get("phi_start")),
            row.get("phi_g", row.get("phi_target")),
            env_name=env,
            edge_type="qtrain_supported",
            q_train_support=_float(row.get("q_train_support"), 1.0),
        )
        counts["qtrain_supported"] += int(edge_id is not None)
    return counts


def add_knn_edges(builder: GraphBuilder, max_per_node: int) -> int:
    if max_per_node <= 0:
        return 0
    added = 0
    by_env: dict[str, list[ContractFunnelNode]] = defaultdict(list)
    for node in builder.nodes.values():
        by_env[str(node.env_name or "unknown")].append(node)
    for env, nodes in by_env.items():
        if len(nodes) < 2:
            continue
        centers = np.asarray([node.center_phi for node in nodes], dtype=np.float32)
        for idx, node in enumerate(nodes):
            if builder.added_edges >= builder.max_added_edges:
                return added
            dists = np.linalg.norm(centers - centers[idx], axis=1)
            nearest = np.argsort(dists)[1 : max_per_node + 1]
            for nbr_idx in nearest:
                nbr = nodes[int(nbr_idx)]
                edge_id = builder.add_scored_edge(node.center_phi, nbr.center_phi, env_name=env, edge_type="knn_bridge_candidate")
                if edge_id:
                    added += 1
                if builder.added_edges >= builder.max_added_edges:
                    return added
    return added


def graph_connectivity_summary(graph: ContractGraph) -> dict[str, Any]:
    outgoing: dict[str, set[str]] = defaultdict(set)
    incoming: dict[str, set[str]] = defaultdict(set)
    undirected: dict[str, set[str]] = defaultdict(set)
    for edge in graph.edges.values():
        outgoing[edge.src].add(edge.dst)
        incoming[edge.dst].add(edge.src)
        undirected[edge.src].add(edge.dst)
        undirected[edge.dst].add(edge.src)
    weak_sizes = component_sizes(graph.nodes.keys(), undirected)
    strong_sizes = strongly_connected_sizes(graph.nodes.keys(), outgoing, incoming)
    avg_out = sum(len(outgoing.get(node_id, ())) for node_id in graph.nodes) / max(len(graph.nodes), 1)
    reachability = reachability_proxy(list(graph.nodes), outgoing)
    return {
        "weak_component_count": len(weak_sizes),
        "largest_weak_component": max(weak_sizes) if weak_sizes else 0,
        "strong_component_count": len(strong_sizes),
        "largest_strong_component": max(strong_sizes) if strong_sizes else 0,
        "avg_out_degree": avg_out,
        "path_pair_reachability_proxy": reachability,
    }


def component_sizes(nodes: Any, adjacency: dict[str, set[str]]) -> list[int]:
    unseen = set(nodes)
    sizes = []
    while unseen:
        start = unseen.pop()
        size = 1
        queue = deque([start])
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
        stack: list[tuple[str, bool]] = [(node, False)]
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
    sizes: list[int] = []

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


def reachability_proxy(nodes: list[str], outgoing: dict[str, set[str]], sample_count: int = 256) -> float | None:
    if len(nodes) < 2:
        return None
    checks = 0
    found = 0
    step = max(1, len(nodes) // max(1, int(math.sqrt(sample_count))))
    sampled = nodes[::step][: int(math.sqrt(sample_count)) + 1]
    for src in sampled:
        reachable = bfs_reachable(src, outgoing, max_depth=4)
        for dst in sampled:
            if src == dst:
                continue
            checks += 1
            found += int(dst in reachable)
    return found / checks if checks else None


def bfs_reachable(src: str, outgoing: dict[str, set[str]], max_depth: int) -> set[str]:
    reached = {src}
    queue = deque([(src, 0)])
    while queue:
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for nxt in outgoing.get(node, set()):
            if nxt not in reached:
                reached.add(nxt)
                queue.append((nxt, depth + 1))
    return reached


def write_report(path: Path, summary: dict[str, Any], graph_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage36 Transition Contract Graph Build",
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
        "weak_component_count",
        "largest_weak_component",
        "strong_component_count",
        "largest_strong_component",
        "avg_out_degree",
        "path_pair_reachability_proxy",
        "low_contract_edge_rate",
        "high_negative_edge_rate",
        "final_goal_edge_rate",
        "recovery_edge_rate",
        "qtrain_supported_edge_rate",
        "transition_edge_rate",
    ]:
        lines.append(f"| {key} | {_fmt(summary.get(key))} |")
    lines.extend(["", "## Edge Type Counts", "", json.dumps(summary.get("edge_type_counts", {}), indent=2, sort_keys=True)])
    lines.extend(["", "## Notes", "", summary.get("notes", "")])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    if args.clear and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_graph = ContractGraph.load_json(args.base_contract_graph)
    scorer = ContractScorer.from_path(args.contract_model_path, uncertainty_penalty=0.25)
    builder = GraphBuilder(scorer, args.min_contract_lcb, args.max_transition_edges)
    builder.load_base(base_graph)

    dataset_rows = load_jsonl(Path(args.contract_dataset_path))
    dataset_counts = add_contract_dataset_edges(builder, dataset_rows)
    segment_counts = add_segment_capture_edges(builder, list(args.segment_capture_roots or []))
    qtrain_counts = add_qtrain_edges(builder, args.qtrain_path)
    knn_count = add_knn_edges(builder, int(args.max_knn_edges_per_node))
    builder.add_boundaries_from_sequences()
    conservative_boundary_count = builder.add_conservative_boundaries()

    graph = builder.build(
        {
            "builder": "build_cage_transition_contract_graph.py",
            "base_contract_graph": str(args.base_contract_graph),
            "contract_dataset_path": str(args.contract_dataset_path),
            "segment_capture_roots": list(args.segment_capture_roots or []),
            "contract_model_loaded": bool(scorer.loaded),
            "min_contract_lcb": float(args.min_contract_lcb),
            "max_transition_edges": int(args.max_transition_edges),
            "dataset_edge_counts": dict(dataset_counts),
            "segment_edge_counts": dict(segment_counts),
            "qtrain_edge_counts": dict(qtrain_counts),
            "knn_bridge_edges": knn_count,
            "skipped_low_lcb": builder.skipped_low_lcb,
            "skipped_missing_phi": builder.skipped_missing_phi,
            "conservative_boundary_count": conservative_boundary_count,
        }
    )
    summary = graph.summarize()
    summary.update(graph_connectivity_summary(graph))
    edge_count = max(int(summary.get("edge_count") or 0), 1)
    type_counts = summary.get("edge_type_counts", {})
    summary["qtrain_supported_edge_rate"] = type_counts.get("qtrain_supported", 0) / edge_count
    summary["transition_edge_rate"] = (
        type_counts.get("temporal_transition", 0)
        + type_counts.get("path_adjacency", 0)
        + type_counts.get("knn_bridge_candidate", 0)
    ) / edge_count
    summary["notes"] = "No online benchmark was run. KNN bridge edges are candidate connectivity edges, not observed transitions."

    graph_path = out_dir / "contract_graph.json"
    graph.save_json(graph_path)
    graph.export_nodes_csv(out_dir / "nodes.csv")
    graph.export_edges_csv(out_dir / "edges.csv")
    graph.export_boundary_csv(out_dir / "boundary_contracts.csv")
    write_report(ROOT / "reports" / "stage36_transition_contract_graph_build.md", summary, graph_path)
    print(json.dumps({"graph_path": str(graph_path), **summary}, sort_keys=True))
    return 0


def endpoint_key(phi: Any, env_name: str, decimals: int = 3) -> str:
    arr = np.asarray(phi, dtype=np.float32).reshape(-1)
    return f"{env_name}::" + ",".join(f"{x:.{decimals}f}" for x in np.round(arr, decimals=decimals))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def segment_files(roots: list[str]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        path = Path(root)
        if path.is_file() and path.name.endswith("_segments.jsonl"):
            files.append(path)
        elif path.exists():
            files.extend(sorted(path.glob("*_segments.jsonl")))
    return files


def _float(value: Any, default: float | None = 0.0) -> float | None:
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
