from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd


def load_support_graph(phase2_run_dir: str | Path) -> nx.DiGraph:
    edges = pd.read_csv(Path(phase2_run_dir) / "option_edges.csv")
    graph = nx.DiGraph()
    for row in edges.itertuples(index=False):
        graph.add_edge(
            int(row.src),
            int(row.dst),
            edge_id=int(row.edge_id),
            cost=float(getattr(row, "cost", row.median_h)),
        )
    return graph


def plan_option_path(graph: nx.DiGraph, src_cluster: int, goal_cluster: int, risk_aware: bool = False) -> list[int]:
    del risk_aware
    return [int(x) for x in nx.shortest_path(graph, int(src_cluster), int(goal_cluster), weight="cost")]
