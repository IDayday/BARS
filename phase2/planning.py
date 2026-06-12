from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def shortest_path_record(
    G: nx.DiGraph,
    source: int,
    target: int,
    weight: str = "cost",
) -> dict[str, object]:
    source = int(source)
    target = int(target)
    if source not in G or target not in G:
        return {"reachable": False, "path": [], "path_edges": [], "path_cost": np.inf}
    try:
        path = nx.shortest_path(G, source, target, weight=weight)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return {"reachable": False, "path": [], "path_edges": [], "path_cost": np.inf}
    edge_ids: list[int] = []
    cost = 0.0
    for src, dst in zip(path[:-1], path[1:]):
        data = G[src][dst]
        cost += float(data.get(weight, 1.0))
        edge_ids.append(int(data.get("edge_id", -1)))
    return {"reachable": True, "path": path, "path_edges": edge_ids, "path_cost": float(cost)}


def evaluate_query_paths(
    G: nx.DiGraph,
    queries: pd.DataFrame,
    source_col: str = "start_cluster",
    target_col: str = "goal_cluster",
    weight: str = "cost",
) -> tuple[pd.DataFrame, dict[str, float]]:
    if queries.empty:
        paths = pd.DataFrame(
            columns=[
                "query_id",
                "src",
                "dst",
                "reachable",
                "path_edges_count",
                "path_cost",
                "path_nodes",
                "path_edge_ids",
            ]
        )
        return paths, {
            "num_queries": 0,
            "num_reachable": 0,
            "path_coverage": 0.0,
            "mean_path_edges": 0.0,
            "mean_path_cost": 0.0,
            "median_path_cost": 0.0,
        }
    rows = []
    for idx, row in queries.reset_index(drop=True).iterrows():
        rec = shortest_path_record(G, int(row[source_col]), int(row[target_col]), weight=weight)
        rows.append(
            {
                "query_id": int(idx),
                "src": int(row[source_col]),
                "dst": int(row[target_col]),
                "reachable": bool(rec["reachable"]),
                "path_edges_count": int(max(0, len(rec["path"]) - 1)),
                "path_cost": float(rec["path_cost"]),
                "path_nodes": " ".join(str(x) for x in rec["path"]),
                "path_edge_ids": " ".join(str(x) for x in rec["path_edges"]),
            }
        )
    paths = pd.DataFrame(rows)
    reachable = paths[paths["reachable"]]
    metrics = {
        "num_queries": int(paths.shape[0]),
        "num_reachable": int(reachable.shape[0]),
        "path_coverage": float(reachable.shape[0] / max(1, paths.shape[0])),
        "mean_path_edges": float(reachable["path_edges_count"].mean()) if not reachable.empty else 0.0,
        "mean_path_cost": float(reachable["path_cost"].mean()) if not reachable.empty else 0.0,
        "median_path_cost": float(reachable["path_cost"].median()) if not reachable.empty else 0.0,
    }
    return paths, metrics
