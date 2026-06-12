from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

from phase2.planning import evaluate_query_paths


def make_episode_queries(
    labels_obs: np.ndarray,
    goal_labels: np.ndarray,
    episodes: list[dict],
    max_queries: int | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    labels_obs = np.asarray(labels_obs, dtype=np.int64)
    goal_labels = np.asarray(goal_labels, dtype=np.int64)
    rows = []
    for ep_id, episode in enumerate(episodes):
        start = int(episode["start_index"])
        if start >= labels_obs.size or ep_id >= goal_labels.size:
            continue
        rows.append(
            {
                "query_id": len(rows),
                "ep_id": int(ep_id),
                "start_cluster": int(labels_obs[start]),
                "goal_cluster": int(goal_labels[ep_id]),
            }
        )
    queries = pd.DataFrame(rows)
    if max_queries is not None and queries.shape[0] > int(max_queries):
        queries = queries.sample(n=int(max_queries), random_state=int(seed)).reset_index(drop=True)
        queries["query_id"] = np.arange(queries.shape[0], dtype=np.int64)
    return queries


def _support_value(N: sparse.spmatrix | np.ndarray, src: int, dst: int) -> float:
    if sparse.issparse(N):
        return float(N.tocsr()[int(src), int(dst)])
    return float(np.asarray(N)[int(src), int(dst)])


def _virtual_graph(
    G: nx.DiGraph,
    start: int,
    goal: int,
    support_N: sparse.spmatrix | np.ndarray,
    min_support: int,
    H_query: int,
) -> nx.DiGraph:
    graph = G.copy()
    graph.add_node(int(start))
    graph.add_node(int(goal))
    nodes = list(G.nodes())
    for node in nodes + [int(goal)]:
        if int(start) != int(node) and _support_value(support_N, int(start), int(node)) >= min_support:
            graph.add_edge(int(start), int(node), cost=float(H_query), edge_id=-1)
    for node in nodes + [int(start)]:
        if int(node) != int(goal) and _support_value(support_N, int(node), int(goal)) >= min_support:
            graph.add_edge(int(node), int(goal), cost=float(H_query), edge_id=-1)
    if int(start) != int(goal) and _support_value(support_N, int(start), int(goal)) >= min_support:
        graph.add_edge(int(start), int(goal), cost=float(H_query), edge_id=-1)
    return graph


def evaluate_task_path_coverage(
    G: nx.DiGraph,
    labels_train: np.ndarray,
    labels_val: np.ndarray,
    val_episodes_or_train_episodes: pd.DataFrame | list[dict],
    selected_nodes: pd.DataFrame,
    H_query: int,
    mode: str,
    dataset_name: str = "",
    node_selection: str = "",
    node_budget: int = 0,
    support_N: sparse.spmatrix | np.ndarray | None = None,
    min_support: int = 1,
    goal_labels: np.ndarray | None = None,
    max_queries: int | None = None,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    del labels_train
    mode = mode.lower()
    selected = set(selected_nodes[selected_nodes["selected"]]["cluster"].astype(int).tolist())
    if isinstance(val_episodes_or_train_episodes, pd.DataFrame):
        queries = val_episodes_or_train_episodes.copy()
    else:
        if goal_labels is None:
            raise ValueError("goal_labels are required when episodes are passed")
        queries = make_episode_queries(labels_val, goal_labels, val_episodes_or_train_episodes)
    if max_queries is not None and queries.shape[0] > int(max_queries):
        queries = queries.sample(n=int(max_queries), random_state=int(seed)).reset_index(drop=True)
        queries["query_id"] = np.arange(queries.shape[0], dtype=np.int64)

    if mode == "strict_selected":
        eval_queries = queries[
            queries["start_cluster"].isin(selected) & queries["goal_cluster"].isin(selected)
        ].reset_index(drop=True)
        paths, metrics = evaluate_query_paths(G, eval_queries)
    elif mode == "virtual_query":
        if support_N is None:
            raise ValueError("support_N is required for virtual_query mode")
        if queries.empty:
            paths, metrics = evaluate_query_paths(G, queries)
            summary = pd.DataFrame(
                [
                    {
                        "dataset_name": dataset_name,
                        "H": int(H_query),
                        "node_selection": node_selection,
                        "node_budget": int(node_budget),
                        "mode": mode,
                        **metrics,
                    }
                ]
            )
            return summary, paths
        path_rows = []
        for idx, row in queries.reset_index(drop=True).iterrows():
            graph = _virtual_graph(
                G,
                int(row.start_cluster),
                int(row.goal_cluster),
                support_N,
                int(min_support),
                int(H_query),
            )
            path_df, _ = evaluate_query_paths(
                graph,
                pd.DataFrame(
                    [
                        {
                            "start_cluster": int(row.start_cluster),
                            "goal_cluster": int(row.goal_cluster),
                        }
                    ]
                ),
            )
            out = path_df.iloc[0].to_dict()
            out["query_id"] = int(idx)
            out["ep_id"] = int(row.get("ep_id", -1))
            path_rows.append(out)
        paths = pd.DataFrame(path_rows)
        reachable = paths[paths["reachable"]]
        metrics = {
            "num_queries": int(paths.shape[0]),
            "num_reachable": int(reachable.shape[0]),
            "path_coverage": float(reachable.shape[0] / max(1, paths.shape[0])),
            "mean_path_edges": float(reachable["path_edges_count"].mean()) if not reachable.empty else 0.0,
            "mean_path_cost": float(reachable["path_cost"].mean()) if not reachable.empty else 0.0,
            "median_path_cost": float(reachable["path_cost"].median()) if not reachable.empty else 0.0,
        }
    else:
        raise ValueError("mode must be strict_selected or virtual_query")

    summary = pd.DataFrame(
        [
            {
                "dataset_name": dataset_name,
                "H": int(H_query),
                "node_selection": node_selection,
                "node_budget": int(node_budget),
                "mode": mode,
                **metrics,
            }
        ]
    )
    return summary, paths


def reachable_pair_metrics(G: nx.DiGraph, queries: pd.DataFrame | None = None) -> dict[str, float]:
    if queries is None:
        nodes = list(G.nodes())
        queries = pd.DataFrame(
            [{"start_cluster": src, "goal_cluster": dst} for src in nodes for dst in nodes if src != dst]
        )
    _, metrics = evaluate_query_paths(G, queries)
    try:
        lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight="cost"))
        finite = [
            float(length)
            for src, targets in lengths.items()
            for dst, length in targets.items()
            if src != dst
        ]
    except Exception:
        finite = []
    metrics["average_shortest_path_length"] = float(np.mean(finite)) if finite else 0.0
    return metrics


def bottleneck_removal_ablation(
    G: nx.DiGraph,
    bottleneck_df: pd.DataFrame,
    top_q: float,
    path_queries: pd.DataFrame,
) -> pd.DataFrame:
    if G.number_of_nodes() == 0:
        return pd.DataFrame()
    n_remove = max(1, int(np.ceil(float(top_q) * G.number_of_nodes())))
    ranked = bottleneck_df[bottleneck_df["cluster"].isin(G.nodes())].sort_values(
        ["bottleneck_score", "cluster"],
        ascending=[False, True],
        kind="mergesort",
    )
    remove_nodes = ranked["cluster"].head(n_remove).astype(int).tolist()
    before = reachable_pair_metrics(G, path_queries)
    G_removed = G.copy()
    G_removed.remove_nodes_from(remove_nodes)
    after = reachable_pair_metrics(G_removed, path_queries)
    rows = []
    for label, metrics in [("before", before), ("after", after)]:
        rows.append(
            {
                "condition": label,
                "top_q": float(top_q),
                "removed_nodes": " ".join(str(x) for x in remove_nodes) if label == "after" else "",
                "path_coverage": float(metrics["path_coverage"]),
                "mean_path_cost": float(metrics["mean_path_cost"]),
                "reachable_pair_ratio": float(metrics["path_coverage"]),
                "average_shortest_path_length": float(metrics["average_shortest_path_length"]),
            }
        )
    return pd.DataFrame(rows)
