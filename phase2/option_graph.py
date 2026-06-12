from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


EPS = 1e-12


def _edge_cost(row: object, cost_type: str, support_penalty: float) -> float:
    median_h = float(getattr(row, "median_h"))
    num_segments = max(1.0, float(getattr(row, "num_segments")))
    if cost_type == "median_h":
        return median_h
    if cost_type == "support_weighted":
        return median_h * (1.0 + float(support_penalty) / np.sqrt(num_segments))
    if cost_type == "default":
        return median_h + float(support_penalty) / np.sqrt(num_segments)
    raise ValueError(f"Unsupported cost_type {cost_type!r}")


def add_edge_costs(
    option_edges_df: pd.DataFrame,
    cost_type: str = "default",
    support_penalty: float = 1.0,
    cost_column: str = "cost",
) -> pd.DataFrame:
    """Return option edges annotated with the planner cost used in the graph."""

    if option_edges_df is None or option_edges_df.empty:
        df = pd.DataFrame() if option_edges_df is None else option_edges_df.copy()
        if cost_column not in df.columns:
            df[cost_column] = pd.Series(dtype=np.float64)
        return df
    df = option_edges_df.copy()
    df[cost_column] = [
        _edge_cost(row, cost_type=cost_type, support_penalty=support_penalty)
        for row in df.itertuples(index=False)
    ]
    return df


def build_option_graph(
    option_edges_df: pd.DataFrame,
    cost_type: str = "default",
    support_penalty: float = 1.0,
    selected_nodes: pd.DataFrame | None = None,
) -> nx.DiGraph:
    G = nx.DiGraph()
    if selected_nodes is not None:
        selected = selected_nodes[selected_nodes["selected"]]
        G.add_nodes_from(selected["cluster"].astype(int).tolist())
    if option_edges_df is None or option_edges_df.empty:
        G.graph["cost_type"] = cost_type
        G.graph["support_penalty"] = float(support_penalty)
        return G
    for row in option_edges_df.itertuples(index=False):
        cost = (
            float(getattr(row, "cost"))
            if hasattr(row, "cost")
            else _edge_cost(row, cost_type, support_penalty)
        )
        G.add_edge(
            int(row.src),
            int(row.dst),
            edge_id=int(row.edge_id),
            cost=float(cost),
            median_h=float(row.median_h),
            mean_h=float(row.mean_h),
            num_segments=int(row.num_segments),
            num_episodes=int(row.num_episodes),
            support_count=int(row.support_count),
            reverse_support_count=int(row.reverse_support_count),
            asymmetry=float(row.asymmetry),
        )
    G.graph["cost_type"] = cost_type
    G.graph["support_penalty"] = float(support_penalty)
    return G


def sampled_reachable_pair_ratio(G: nx.DiGraph, sample_pairs: int = 5000, seed: int = 0) -> float:
    nodes = np.asarray(list(G.nodes()), dtype=np.int64)
    if nodes.size < 2 or sample_pairs <= 0:
        return 0.0
    rng = np.random.default_rng(seed)
    n_samples = min(int(sample_pairs), int(nodes.size * (nodes.size - 1)))
    src_idx = rng.integers(0, nodes.size, size=n_samples)
    dst_idx = rng.integers(0, nodes.size - 1, size=n_samples)
    dst_idx = dst_idx + (dst_idx >= src_idx)
    srcs = nodes[src_idx]
    dsts = nodes[dst_idx]
    reachable = 0
    for source in np.unique(srcs):
        lengths = nx.single_source_shortest_path_length(G, int(source))
        mask = srcs == source
        reachable += sum(int(dst) in lengths for dst in dsts[mask])
    return float(reachable / max(1, n_samples))


def graph_summary(G: nx.DiGraph, sample_pairs: int = 5000, seed: int = 0) -> pd.DataFrame:
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    weak = list(nx.weakly_connected_components(G)) if n_nodes else []
    strong = list(nx.strongly_connected_components(G)) if n_nodes else []
    row = {
        "num_selected_nodes": int(n_nodes),
        "num_option_edges": int(n_edges),
        "average_out_degree": float(n_edges / max(1, n_nodes)),
        "average_in_degree": float(n_edges / max(1, n_nodes)),
        "num_weakly_connected_components": int(len(weak)),
        "largest_wcc_size": int(max((len(c) for c in weak), default=0)),
        "num_strongly_connected_components": int(len(strong)),
        "largest_scc_size": int(max((len(c) for c in strong), default=0)),
        "reachable_pair_ratio_sampled": sampled_reachable_pair_ratio(G, sample_pairs, seed),
    }
    return pd.DataFrame([row])
