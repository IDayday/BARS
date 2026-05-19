from __future__ import annotations

import copy
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


@dataclass
class GASGraphTables:
    key_graph: Any
    nodes: pd.DataFrame
    edges: pd.DataFrame


def load_gas_keygraph(path: str | os.PathLike[str]) -> Any:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        kg = type("LoadedGASKeyGraph", (), {})()
        for k, v in data.items():
            setattr(kg, k, v)
        return kg
    return data


def _nodes_array(key_graph: Any) -> np.ndarray:
    nodes = np.asarray(getattr(key_graph, "nodes"))
    if nodes.ndim != 2:
        raise ValueError("GAS keygraph nodes must be a 2D latent array")
    return nodes.astype(np.float32)


def export_nodes(key_graph: Any) -> pd.DataFrame:
    nodes = _nodes_array(key_graph)
    base_count = int(getattr(key_graph, "base_node_cnt", len(nodes)) or len(nodes))
    rows = []
    task_goal_ids = set(int(x) for x in getattr(key_graph, "task_node_idx_dict", {}).values())
    for i, phi in enumerate(nodes):
        if i in task_goal_ids or i >= base_count:
            node_type = "gas_task_goal"
        else:
            node_type = "gas_keynode"
        row = {"node_id": int(i), "node_type": node_type}
        row.update({f"phi_{j}": float(v) for j, v in enumerate(phi)})
        rows.append(row)
    return pd.DataFrame(rows)


def export_edges(key_graph: Any) -> pd.DataFrame:
    nodes = _nodes_array(key_graph)
    graph = getattr(key_graph, "graph", None)
    if graph is None:
        raise ValueError("GAS keygraph has no networkx graph")
    base_count = int(getattr(key_graph, "base_node_cnt", len(nodes)) or len(nodes))
    pairs = set((int(u), int(v)) for u, v in graph.edges())
    rows = []
    for edge_id, (u, v) in enumerate(graph.edges()):
        u = int(u)
        v = int(v)
        attrs = graph[u][v]
        phi_dist = float(np.linalg.norm(nodes[u] - nodes[v]))
        weight = float(attrs.get("weight", phi_dist))
        if u >= base_count or v >= base_count:
            edge_source = "gas_goal_connector"
        elif phi_dist <= float(getattr(key_graph, "way_steps", np.inf)) + 1e-6:
            edge_source = "gas_distance"
        else:
            edge_source = "gas_scc_connector"
        rows.append(
            {
                "edge_id": int(edge_id),
                "u": u,
                "v": v,
                "gas_weight": weight,
                "temporal_cost": weight,
                "phi_dist": phi_dist,
                "is_bidirectional_partner": int((v, u) in pairs),
                "edge_source": edge_source,
            }
        )
    return pd.DataFrame(rows)


def _append_virtual_node(node_df: pd.DataFrame, phi: np.ndarray, node_id: int, node_type: str) -> pd.DataFrame:
    row = {"node_id": int(node_id), "node_type": node_type}
    row.update({f"phi_{j}": float(v) for j, v in enumerate(np.asarray(phi).reshape(-1))})
    return pd.concat([node_df, pd.DataFrame([row])], ignore_index=True)


def add_virtual_start_goal_edges(
    key_graph: Any,
    phi_start: np.ndarray,
    phi_goal: np.ndarray,
    way_steps: Optional[float],
    k: int = 16,
    force_closest: bool = True,
    copy_graph: bool = True,
) -> GASGraphTables:
    kg = copy.deepcopy(key_graph) if copy_graph else key_graph
    node_df = export_nodes(kg)
    edge_df = export_edges(kg)
    base_nodes = _nodes_array(kg)
    start_id = int(node_df["node_id"].max()) + 1
    goal_id = start_id + 1
    phi_start = np.asarray(phi_start, dtype=np.float32).reshape(-1)
    phi_goal = np.asarray(phi_goal, dtype=np.float32).reshape(-1)
    node_df = _append_virtual_node(node_df, phi_start, start_id, "virtual_start")
    node_df = _append_virtual_node(node_df, phi_goal, goal_id, "virtual_goal")
    way = float(way_steps if way_steps is not None else getattr(kg, "way_steps", np.inf))
    k = max(1, int(k))

    start_dist = np.linalg.norm(base_nodes - phi_start[None, :], axis=1)
    goal_dist = np.linalg.norm(base_nodes - phi_goal[None, :], axis=1)
    start_idx = np.where(start_dist <= way)[0]
    goal_idx = np.where(goal_dist <= way)[0]
    if force_closest or len(start_idx) == 0:
        start_idx = np.unique(np.concatenate([start_idx, np.argsort(start_dist)[:k]]))
    else:
        start_idx = start_idx[np.argsort(start_dist[start_idx])[:k]]
    if force_closest or len(goal_idx) == 0:
        goal_idx = np.unique(np.concatenate([goal_idx, np.argsort(goal_dist)[:k]]))
    else:
        goal_idx = goal_idx[np.argsort(goal_dist[goal_idx])[:k]]

    rows = []
    next_edge_id = int(edge_df["edge_id"].max()) + 1 if len(edge_df) else 0
    for idx in start_idx[:k]:
        dist = float(start_dist[idx])
        rows.append(
            {
                "edge_id": next_edge_id,
                "u": start_id,
                "v": int(idx),
                "gas_weight": dist,
                "temporal_cost": dist,
                "phi_dist": dist,
                "is_bidirectional_partner": 0,
                "edge_source": "gas_goal_connector",
            }
        )
        next_edge_id += 1
    for idx in goal_idx[:k]:
        dist = float(goal_dist[idx])
        rows.append(
            {
                "edge_id": next_edge_id,
                "u": int(idx),
                "v": goal_id,
                "gas_weight": dist,
                "temporal_cost": dist,
                "phi_dist": dist,
                "is_bidirectional_partner": 0,
                "edge_source": "gas_goal_connector",
            }
        )
        next_edge_id += 1
    direct_dist = float(np.linalg.norm(phi_goal - phi_start))
    if direct_dist <= max(way, 1e-6):
        rows.append(
            {
                "edge_id": next_edge_id,
                "u": start_id,
                "v": goal_id,
                "gas_weight": direct_dist,
                "temporal_cost": direct_dist,
                "phi_dist": direct_dist,
                "is_bidirectional_partner": 0,
                "edge_source": "gas_goal_connector",
            }
        )
    if rows:
        edge_df = pd.concat([edge_df, pd.DataFrame(rows)], ignore_index=True)
    return GASGraphTables(kg, node_df, edge_df)


def save_edge_table(nodes: pd.DataFrame, edges: pd.DataFrame, out_dir: str | os.PathLike[str], stem: str = "gas_graph") -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, df in (("nodes", nodes), ("edges", edges)):
        csv_path = out / f"{stem}_{name}.csv"
        df.to_csv(csv_path, index=False)
        paths[f"{name}_csv"] = str(csv_path)
        pq_path = out / f"{stem}_{name}.parquet"
        try:
            df.to_parquet(pq_path, index=False)
            paths[f"{name}_parquet"] = str(pq_path)
        except Exception:
            pass
    return paths
