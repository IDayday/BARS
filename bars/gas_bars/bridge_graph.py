from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import networkx as nx
import numpy as np
import pandas as pd


BRIDGE_EDGE_TYPES = {"aggressive_tdr_bridge", "bottleneck_bridge"}
RISKY_EDGE_TYPES = BRIDGE_EDGE_TYPES | {"gas_cross"}
SAFE_EDGE_TYPES = {"safe_local", "same_traj_temporal", "virtual_connector"}


@dataclass
class BridgeGraphBundle:
    graph_id: str
    nodes: pd.DataFrame
    edges: pd.DataFrame
    way_steps: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "nodes": self.nodes,
            "edges": self.edges,
            "way_steps": self.way_steps,
            "metadata": self.metadata,
        }


def phi_columns(nodes: pd.DataFrame) -> list[str]:
    cols = [c for c in nodes.columns if c.startswith("phi_")]
    return sorted(cols, key=lambda c: int(c.split("_")[1]))


def node_phis(nodes: pd.DataFrame) -> np.ndarray:
    return nodes[phi_columns(nodes)].to_numpy(np.float32)


def _node_index(nodes: pd.DataFrame) -> dict[int, int]:
    return {int(n): i for i, n in enumerate(nodes["node_id"].to_numpy())}


def _edge_phi_dist(nodes: pd.DataFrame, u: int, v: int) -> float:
    idx = _node_index(nodes)
    phis = node_phis(nodes)
    return float(np.linalg.norm(phis[idx[int(u)]] - phis[idx[int(v)]]))


def infer_way_steps(edges: pd.DataFrame, default: float = 8.0) -> float:
    if "edge_type" in edges:
        local = edges[edges["edge_type"].isin(["safe_local", "same_traj_temporal"])]
    else:
        local = edges[edges.get("edge_source", "").astype(str).eq("gas_distance")] if "edge_source" in edges else edges
    vals = pd.to_numeric(local.get("phi_dist", pd.Series(dtype=float)), errors="coerce").dropna()
    if len(vals):
        return float(max(vals.quantile(0.90), 1e-6))
    vals = pd.to_numeric(edges.get("phi_dist", pd.Series(dtype=float)), errors="coerce").dropna()
    return float(max(vals.median(), default)) if len(vals) else default


def annotate_official_edges(nodes: pd.DataFrame, edges: pd.DataFrame, way_steps: Optional[float] = None) -> pd.DataFrame:
    """Add Stage23 edge taxonomy to an exported GAS edge table.

    Official GAS does not explicitly store trajectory ids for key nodes in the
    saved graph, so this function keeps same-trajectory labeling conservative:
    distance edges are marked safe local, SCC connectors are marked gas_cross,
    and start/goal connectors are virtual connectors.
    """
    out = edges.copy()
    if "phi_dist" not in out:
        idx = _node_index(nodes)
        phis = node_phis(nodes)
        out["phi_dist"] = [
            float(np.linalg.norm(phis[idx[int(r.u)]] - phis[idx[int(r.v)]]))
            for r in out[["u", "v"]].itertuples(index=False)
        ]
    if "temporal_cost" not in out:
        out["temporal_cost"] = out.get("gas_weight", out["phi_dist"])
    way = float(way_steps if way_steps is not None else infer_way_steps(out))
    edge_types = []
    for r in out.itertuples(index=False):
        source = str(getattr(r, "edge_source", ""))
        phi_dist = float(getattr(r, "phi_dist", 0.0))
        if source == "gas_goal_connector":
            et = "virtual_connector"
        elif source == "gas_scc_connector":
            et = "gas_cross"
        elif phi_dist <= way * 1.05:
            et = "safe_local"
        else:
            et = "gas_cross"
        edge_types.append(et)
    out["edge_type"] = edge_types
    out["graph_id"] = out.get("graph_id", "G0")
    out["is_bridge_candidate"] = out["edge_type"].isin(BRIDGE_EDGE_TYPES).astype(int)
    out["structural_reason"] = out.get("structural_reason", out["edge_type"])
    return out.reset_index(drop=True)


def _existing_pairs(edges: pd.DataFrame) -> set[tuple[int, int]]:
    return set((int(r.u), int(r.v)) for r in edges[["u", "v"]].itertuples(index=False))


def _new_edge_id(edges: pd.DataFrame) -> int:
    return int(edges["edge_id"].max()) + 1 if len(edges) else 0


def _make_edge_row(
    edge_id: int,
    u: int,
    v: int,
    phi_dist: float,
    edge_type: str,
    graph_id: str,
    reason: str,
    score: float = 0.0,
) -> dict[str, Any]:
    return {
        "edge_id": int(edge_id),
        "u": int(u),
        "v": int(v),
        "gas_weight": float(phi_dist),
        "temporal_cost": float(phi_dist),
        "phi_dist": float(phi_dist),
        "is_bidirectional_partner": 0,
        "edge_source": edge_type,
        "edge_type": edge_type,
        "graph_id": graph_id,
        "is_bridge_candidate": 1,
        "structural_reason": reason,
        "bridge_score": float(score),
    }


def aggressive_tdr_bridges(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    top_k_bridge: int = 4,
    min_dist_mult: float = 1.10,
    max_dist_mult: float = 3.50,
) -> pd.DataFrame:
    """Generate nonlocal TDR-kNN shortcut candidates with bounded fanout."""
    from sklearn.neighbors import NearestNeighbors

    base = nodes[~nodes["node_type"].astype(str).str.startswith("virtual")].copy()
    phis = node_phis(base)
    node_ids = base["node_id"].to_numpy(np.int64)
    way = infer_way_steps(edges)
    k = min(len(base), max(2, top_k_bridge * 8 + 1))
    nn = NearestNeighbors(n_neighbors=k).fit(phis)
    dists, idxs = nn.kneighbors(phis)
    existing = _existing_pairs(edges)
    rows = []
    next_id = _new_edge_id(edges)
    for i, u in enumerate(node_ids):
        added = 0
        for dist, j in zip(dists[i][1:], idxs[i][1:]):
            v = int(node_ids[j])
            if (int(u), v) in existing:
                continue
            if dist < way * min_dist_mult or dist > way * max_dist_mult:
                continue
            score = 1.0 / max(float(dist), 1e-6)
            rows.append(_make_edge_row(next_id, int(u), v, float(dist), "aggressive_tdr_bridge", "G1", "tdr_knn_nonlocal", score))
            next_id += 1
            added += 1
            if added >= top_k_bridge:
                break
    return pd.DataFrame(rows)


def bottleneck_bridges(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    top_k_bridge: int = 4,
    n_clusters: Optional[int] = None,
    random_state: int = 0,
) -> pd.DataFrame:
    """Lightweight cut-crossing bridge pool using clustered latent regions."""
    from sklearn.cluster import KMeans
    from sklearn.neighbors import NearestNeighbors

    base = nodes[~nodes["node_type"].astype(str).str.startswith("virtual")].copy()
    if len(base) < 4:
        return pd.DataFrame()
    phis = node_phis(base)
    node_ids = base["node_id"].to_numpy(np.int64)
    k_clusters = int(n_clusters or np.clip(np.sqrt(len(base) / 8.0), 2, 12))
    labels = KMeans(n_clusters=k_clusters, n_init=5, random_state=random_state).fit_predict(phis)
    nn = NearestNeighbors(n_neighbors=min(len(base), top_k_bridge * 16 + 1)).fit(phis)
    dists, idxs = nn.kneighbors(phis)
    existing = _existing_pairs(edges)
    way = infer_way_steps(edges)
    rows = []
    next_id = _new_edge_id(edges) + 1_000_000
    for i, u in enumerate(node_ids):
        added = 0
        for dist, j in zip(dists[i][1:], idxs[i][1:]):
            v = int(node_ids[j])
            if labels[i] == labels[j] or (int(u), v) in existing:
                continue
            if dist < way * 0.75 or dist > way * 4.0:
                continue
            score = float(abs(labels[i] - labels[j]) + 1) / max(float(dist), 1e-6)
            rows.append(_make_edge_row(next_id, int(u), v, float(dist), "bottleneck_bridge", "G2", "cluster_boundary_crossing", score))
            next_id += 1
            added += 1
            if added >= top_k_bridge:
                break
    out = pd.DataFrame(rows)
    if len(out):
        region = dict(zip(node_ids.astype(int), labels.astype(int)))
        out["u_region"] = out["u"].map(region)
        out["v_region"] = out["v"].map(region)
        out["bottleneck_score"] = out["bridge_score"]
    return out


def build_bridge_graphs(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    way_steps: Optional[float] = None,
    top_k_bridge: int = 4,
    random_state: int = 0,
) -> dict[str, BridgeGraphBundle]:
    g0_edges = annotate_official_edges(nodes, edges, way_steps=way_steps)
    way = float(way_steps if way_steps is not None else infer_way_steps(g0_edges))
    aggressive = aggressive_tdr_bridges(nodes, g0_edges, top_k_bridge=top_k_bridge)
    bottleneck = bottleneck_bridges(nodes, g0_edges, top_k_bridge=top_k_bridge, random_state=random_state)
    g0 = BridgeGraphBundle("G0", nodes.copy(), g0_edges.assign(graph_id="G0"), way, {"description": "official_gas_graph"})
    g1_edges = pd.concat([g0_edges, aggressive.assign(graph_id="G1")], ignore_index=True)
    g2_edges = pd.concat([g0_edges, bottleneck.assign(graph_id="G2")], ignore_index=True)
    g3_edges = pd.concat([g0_edges, aggressive.assign(graph_id="G3"), bottleneck.assign(graph_id="G3")], ignore_index=True)
    return {
        "G0": g0,
        "G1": BridgeGraphBundle("G1", nodes.copy(), g1_edges, way, {"top_k_bridge": top_k_bridge, "bridge_family": "aggressive_tdr"}),
        "G2": BridgeGraphBundle("G2", nodes.copy(), g2_edges, way, {"top_k_bridge": top_k_bridge, "bridge_family": "bottleneck"}),
        "G3": BridgeGraphBundle("G3", nodes.copy(), g3_edges, way, {"top_k_bridge": top_k_bridge, "bridge_family": "aggressive_plus_bottleneck"}),
    }


def save_bridge_graphs(graphs: dict[str, BridgeGraphBundle], out_dir: str | Path) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    bridge_rows = []
    for gid, bundle in graphs.items():
        path = out / f"{gid}.pkl"
        with open(path, "wb") as f:
            pickle.dump(bundle.to_dict(), f)
        paths[gid] = str(path)
        bridges = bundle.edges[bundle.edges["edge_type"].isin(BRIDGE_EDGE_TYPES)].copy()
        if len(bridges):
            bridges["graph_id"] = gid
            bridge_rows.append(bridges)
    if bridge_rows:
        pd.concat(bridge_rows, ignore_index=True).drop_duplicates(["graph_id", "edge_id"]).to_csv(out / "bridge_table.csv", index=False)
    else:
        pd.DataFrame().to_csv(out / "bridge_table.csv", index=False)
    meta = {gid: bundle.metadata | {"edge_count": int(len(bundle.edges)), "node_count": int(len(bundle.nodes))} for gid, bundle in graphs.items()}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    return paths


def load_bridge_graph(path: str | Path) -> BridgeGraphBundle:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, BridgeGraphBundle):
        return data
    return BridgeGraphBundle(
        graph_id=str(data.get("graph_id", Path(path).stem)),
        nodes=data["nodes"],
        edges=data["edges"],
        way_steps=float(data.get("way_steps", infer_way_steps(data["edges"]))),
        metadata=dict(data.get("metadata", {})),
    )


def _nx_graph(edges: pd.DataFrame) -> tuple[nx.DiGraph, dict[tuple[int, int], int]]:
    g = nx.DiGraph()
    pair_to_eid: dict[tuple[int, int], int] = {}
    for r in edges.itertuples(index=False):
        u, v = int(r.u), int(r.v)
        eid = int(r.edge_id)
        g.add_edge(u, v, weight=float(getattr(r, "temporal_cost", getattr(r, "phi_dist", 1.0))), edge_id=eid)
        pair_to_eid[(u, v)] = eid
    return g, pair_to_eid


def _shortest(g: nx.DiGraph, source: int, target: int) -> tuple[Optional[list[int]], float]:
    try:
        path = nx.shortest_path(g, source=source, target=target, weight="weight")
        dist = nx.shortest_path_length(g, source=source, target=target, weight="weight")
        return [int(x) for x in path], float(dist)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None, float("inf")


def analyze_bridge_graphs(graphs: dict[str, BridgeGraphBundle], max_sources: int = 200, random_state: int = 0) -> pd.DataFrame:
    """Compare path-cost headroom and bridge usage against G0."""
    if "G0" not in graphs:
        raise ValueError("G0 graph is required for bridge graph analysis")
    rng = np.random.default_rng(random_state)
    g0 = graphs["G0"]
    goal_nodes = g0.nodes.loc[g0.nodes["node_type"].astype(str).eq("gas_task_goal"), "node_id"].astype(int).tolist()
    base_nodes = g0.nodes.loc[g0.nodes["node_type"].astype(str).eq("gas_keynode"), "node_id"].astype(int).tolist()
    if not goal_nodes:
        goal_nodes = g0.nodes["node_id"].astype(int).tail(min(5, len(g0.nodes))).tolist()
    if len(base_nodes) > max_sources:
        base_nodes = rng.choice(base_nodes, size=max_sources, replace=False).astype(int).tolist()
    g0_nx, _ = _nx_graph(g0.edges)
    rows = []
    for gid, bundle in graphs.items():
        gx, pair_to_eid = _nx_graph(bundle.edges)
        edge_type = {int(r.edge_id): str(r.edge_type) for r in bundle.edges.itertuples(index=False)}
        reductions = []
        bridge_used = 0
        compared = 0
        shorter = 0
        no_path_g0 = 0
        no_path_gx = 0
        for s in base_nodes:
            for t in goal_nodes:
                p0, d0 = _shortest(g0_nx, s, t)
                px, dx = _shortest(gx, s, t)
                if p0 is None:
                    no_path_g0 += 1
                    continue
                if px is None:
                    no_path_gx += 1
                    continue
                compared += 1
                reductions.append(d0 - dx)
                if dx + 1e-6 < d0:
                    shorter += 1
                eids = [pair_to_eid.get((a, b), -1) for a, b in zip(px[:-1], px[1:])]
                if any(edge_type.get(eid) in BRIDGE_EDGE_TYPES for eid in eids):
                    bridge_used += 1
        bridge_count = int(bundle.edges["edge_type"].isin(BRIDGE_EDGE_TYPES).sum()) if len(bundle.edges) else 0
        rows.append(
            {
                "graph_id": gid,
                "node_count": int(len(bundle.nodes)),
                "edge_count": int(len(bundle.edges)),
                "bridge_count": bridge_count,
                "compared_paths": int(compared),
                "no_path_g0": int(no_path_g0),
                "no_path_graph": int(no_path_gx),
                "mean_path_cost_reduction": float(np.mean(reductions)) if reductions else 0.0,
                "median_path_cost_reduction": float(np.median(reductions)) if reductions else 0.0,
                "shorter_path_rate": float(shorter / max(compared, 1)),
                "bridge_usage_rate": float(bridge_used / max(compared, 1)),
            }
        )
    return pd.DataFrame(rows)
