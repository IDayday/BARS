#!/usr/bin/env python
"""Run Phase 2 support-only compressed option graph diagnostics.

Phase 2 does not train policies, implement TDR/TMD/MQE, or run environment
rollouts. It constructs data-supported directed option graphs from offline
trajectories and Phase 1-style cluster/support/bottleneck diagnostics.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase1.bottleneck import (  # noqa: E402
    betweenness_score,
    bottleneck_score,
    crossing_score,
    removal_impact_score,
)
from phase1.clustering import assign_clusters, compute_cluster_density, fit_state_clusters  # noqa: E402
from phase1.data import load_ogbench_dataset, save_dataset_summary  # noqa: E402
from phase1.diagnostics import build_grid_adjacent_edges, build_knn_edges, unsupported_edge_rate  # noqa: E402
from phase1.support_graph import build_directed_support_graph, compute_support_counts  # noqa: E402
from phase1.trajectory import build_h_step_pairs, split_into_episodes  # noqa: E402
from phase2.compatibility import compute_edge_compatibility, evaluate_path_compatibility  # noqa: E402
from phase2.edge_dataset import build_option_edges, build_self_loop_summary  # noqa: E402
from phase2.evaluation import (  # noqa: E402
    bottleneck_removal_ablation,
    evaluate_task_path_coverage,
    make_episode_queries,
)
from phase2.node_selection import select_nodes  # noqa: E402
from phase2.option_graph import add_edge_costs, build_option_graph, graph_summary  # noqa: E402
from phase2.planning import evaluate_query_paths  # noqa: E402
from phase2.plotting import plot_all  # noqa: E402

SUPPORT_BASELINE_LABELS = {
    "density": "support_density",
    "bottleneck": "support_bottleneck",
    "core_plus_bottleneck": "support_core_plus_bottleneck",
    "all": "support_all_upper_bound",
}


def _parse_int_list(value: str | list[int] | None) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [int(x) for x in value]
    if value == "":
        return None
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def _load_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must contain a mapping")
    return data


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, sort_keys=True)


def _merge_args(args: argparse.Namespace) -> argparse.Namespace:
    config = _load_config(args.config)
    merged = vars(args).copy()
    for key, value in config.items():
        if key not in merged or merged[key] is None:
            merged[key] = value
    defaults = {
        "dataset_dir": "/mnt/project/offlinerl_datasets/ogbench",
        "output_dir": "results/phase2",
        "max_transitions": None,
        "cluster_method": "kmeans",
        "state_dims": None,
        "geometry_dims": None,
        "n_clusters": 512,
        "pair_mode": "dense_upto",
        "horizon_stride": 1,
        "H": 10,
        "H_values": None,
        "min_support": 3,
        "min_episodes": 2,
        "node_budget": 128,
        "node_budgets": None,
        "node_selection": "core_plus_bottleneck",
        "seed": 0,
        "path_eval_pairs": 5000,
        "compat_H_intra": 10,
        "support_penalty": 1.0,
        "cost_type": "default",
        "sweep": False,
        "knn_k": 10,
        "task_type": "generic",
        "bottleneck_top_q": 0.1,
    }
    for key, value in defaults.items():
        if merged.get(key) is None:
            merged[key] = value
    merged["state_dims"] = _parse_int_list(merged.get("state_dims"))
    merged["geometry_dims"] = _parse_int_list(merged.get("geometry_dims"))
    h_values = merged.get("H_values", None)
    if h_values is None and merged.get("Hs", None) is not None:
        h_values = merged.get("Hs")
    merged["H_values"] = _parse_int_list(h_values)
    merged["node_budgets"] = _parse_int_list(merged.get("node_budgets"))
    if merged.get("dataset_dir") is not None:
        merged["dataset_dir"] = str(Path(str(merged["dataset_dir"])).expanduser())
    merged["output_dir"] = str(Path(str(merged["output_dir"])).expanduser())
    if not merged.get("dataset_name"):
        raise ValueError("--dataset_name is required")
    return argparse.Namespace(**merged)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--dataset_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_transitions", type=int, default=None)
    parser.add_argument("--cluster_method", default=None, choices=["kmeans", "grid_xy"])
    parser.add_argument("--state_dims", default=None)
    parser.add_argument("--geometry_dims", default=None)
    parser.add_argument("--n_clusters", type=int, default=None)
    parser.add_argument("--pair_mode", default=None, choices=["exact", "dense_upto", "strided_upto"])
    parser.add_argument("--horizon_stride", type=int, default=None)
    parser.add_argument("--H", type=int, default=None)
    parser.add_argument("--H_values", default=None)
    parser.add_argument("--min_support", type=int, default=None)
    parser.add_argument("--min_episodes", type=int, default=None)
    parser.add_argument("--node_budget", type=int, default=None)
    parser.add_argument("--node_budgets", default=None)
    parser.add_argument(
        "--node_selection",
        default=None,
        choices=["density", "bottleneck", "core_plus_bottleneck", "all"],
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--path_eval_pairs", type=int, default=None)
    parser.add_argument("--compat_H_intra", type=int, default=None)
    parser.add_argument("--support_penalty", type=float, default=None)
    parser.add_argument("--cost_type", default=None, choices=["default", "median_h", "support_weighted"])
    parser.add_argument("--sweep", action="store_true", default=None)
    parser.add_argument("--knn_k", type=int, default=None)
    parser.add_argument("--task_type", default=None, choices=["maze", "manipulation", "generic"])
    parser.add_argument("--bottleneck_top_q", type=float, default=None)
    return _merge_args(parser.parse_args())


def _dataset_key(dataset_name: str) -> str:
    return dataset_name.replace("-v0", "").replace("-", "_")


def _final_goal_labels(episodes: list[dict[str, Any]], cluster_model: dict[str, Any]) -> np.ndarray:
    final_obs = np.asarray([episode["next_observations"][-1] for episode in episodes])
    return assign_clusters(final_obs, cluster_model)


def _load_val_queries(args: argparse.Namespace, cluster_model: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray]:
    try:
        val_dataset = load_ogbench_dataset(
            args.dataset_name,
            args.dataset_dir,
            split="val",
            max_transitions=args.max_transitions,
        )
        val_episodes = split_into_episodes(val_dataset)
        val_obs = np.asarray(val_dataset["observations"])
        labels_val = assign_clusters(val_obs.reshape(val_obs.shape[0], -1), cluster_model)
        goal_labels = _final_goal_labels(val_episodes, cluster_model)
        queries = make_episode_queries(
            labels_val,
            goal_labels,
            val_episodes,
            max_queries=args.path_eval_pairs,
            seed=args.seed,
        )
        return queries, labels_val
    except Exception as exc:
        warnings.warn(f"Val split unavailable for Phase 2 path queries; using train episodes: {exc}")
        return pd.DataFrame(), np.empty(0, dtype=np.int64)


def _graph_from_edges(nodes: list[int], edges: set[tuple[int, int]]) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    for src, dst in edges:
        if src != dst:
            graph.add_edge(int(src), int(dst), cost=1.0, edge_id=-1)
    return graph


def _support_baseline_label(method: str) -> str:
    return SUPPORT_BASELINE_LABELS.get(method, f"support_{method}")


def _filter_edges_to_budget(
    edge_source: set[tuple[int, int]] | list[tuple[int, int]],
    selected_set: set[int],
    edge_budget: int,
) -> set[tuple[int, int]]:
    edges = sorted(
        (int(src), int(dst))
        for src, dst in edge_source
        if int(src) in selected_set and int(dst) in selected_set and int(src) != int(dst)
    )
    if edge_budget <= 0:
        return set()
    return set(edges[: int(edge_budget)])


def _select_feature_dims(features: np.ndarray, dims: list[int] | None) -> np.ndarray:
    if dims is None:
        return features
    dim_idx = np.asarray(dims, dtype=np.int64)
    if np.any(dim_idx < 0) or np.any(dim_idx >= features.shape[1]):
        raise ValueError(f"geometry_dims {dims} are invalid for feature dim {features.shape[1]}")
    return features[:, dim_idx]


def _candidate_baseline_row(
    label: str,
    edges: set[tuple[int, int]],
    nodes: list[int],
    support_N: Any,
    min_support: int,
    filtered_queries: pd.DataFrame,
) -> dict[str, Any]:
    graph = _graph_from_edges(nodes, edges)
    _, metrics = evaluate_query_paths(graph, filtered_queries)
    unsupported = unsupported_edge_rate(edges, support_N, min_support)
    return {
        "baseline": label,
        "num_edges": int(len(edges)),
        "unsupported_edge_rate": float(unsupported["unsupported_edge_rate"]),
        "path_coverage_strict": float(metrics["path_coverage"]),
        "mean_path_cost_strict": float(metrics["mean_path_cost"]),
        "num_queries_strict": int(metrics["num_queries"]),
        "num_reachable_strict": int(metrics["num_reachable"]),
    }


def _baseline_summary(
    observations: np.ndarray,
    labels: np.ndarray,
    cluster_model: dict[str, Any],
    selected_nodes: pd.DataFrame,
    option_edges: pd.DataFrame,
    support_N: Any,
    min_support: int,
    queries: pd.DataFrame,
    seed: int,
    knn_k: int,
    geometry_dims: list[int] | None,
) -> pd.DataFrame:
    selected = selected_nodes[selected_nodes["selected"]]["cluster"].astype(int).tolist()
    selected_set = set(selected)
    edge_budget = int(option_edges.shape[0])
    if not selected or queries.empty:
        return pd.DataFrame()
    filtered_queries = queries[
        queries["start_cluster"].isin(selected_set) & queries["goal_cluster"].isin(selected_set)
    ].reset_index(drop=True)

    rows = []
    n_clusters = int(max(labels) + 1) if len(labels) else 0
    k = max(knn_k, int(np.ceil(max(1, edge_budget) / max(1, len(selected)))))
    raw_edges = build_knn_edges(
        observations,
        labels,
        n_clusters,
        k,
        "cluster_center_knn",
        seed=seed,
    )
    rows.append(
        _candidate_baseline_row(
            "kNN_same_nodes",
            _filter_edges_to_budget(raw_edges, selected_set, edge_budget),
            selected,
            support_N,
            min_support,
            filtered_queries,
        )
    )

    if geometry_dims is not None:
        geometry_edges = build_knn_edges(
            _select_feature_dims(observations, geometry_dims),
            labels,
            n_clusters,
            k,
            "cluster_center_knn",
            seed=seed,
        )
        rows.append(
            _candidate_baseline_row(
                "kNN_geometry_same_nodes",
                _filter_edges_to_budget(geometry_edges, selected_set, edge_budget),
                selected,
                support_N,
                min_support,
                filtered_queries,
            )
        )

    grid_edges = build_grid_adjacent_edges(cluster_model, labels=labels, occupied_only=True)
    if grid_edges:
        rows.append(
            _candidate_baseline_row(
                "grid_adjacent_same_nodes",
                _filter_edges_to_budget(grid_edges, selected_set, edge_budget),
                selected,
                support_N,
                min_support,
                filtered_queries,
            )
        )

    rng = np.random.default_rng(seed)
    possible = [(s, d) for s in selected for d in selected if s != d]
    if possible and edge_budget > 0:
        take = min(edge_budget, len(possible))
        idx = rng.choice(len(possible), size=take, replace=False)
        random_edges = {possible[int(i)] for i in idx}
    else:
        random_edges = set()
    rows.append(
        _candidate_baseline_row(
            "random_same_edge_budget",
            random_edges,
            selected,
            support_N,
            min_support,
            filtered_queries,
        )
    )
    return pd.DataFrame(rows)


def _support_comparison_row(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "baseline": _support_baseline_label(str(metrics["node_selection"])),
        "node_selection": metrics["node_selection"],
        "H": int(metrics["H"]),
        "node_budget": int(metrics["node_budget"]),
        "num_edges": int(metrics["num_option_edges"]),
        "unsupported_edge_rate": 0.0,
        "path_coverage_strict": float(metrics["strict_path_coverage"]),
        "path_coverage_virtual": float(metrics["virtual_path_coverage"]),
        "mean_path_cost_strict": float(metrics["strict_mean_path_cost"]),
        "mean_path_cost_virtual": float(metrics["virtual_mean_path_cost"]),
        "num_queries_strict": int(metrics["strict_num_queries"]),
        "num_reachable_strict": int(metrics["strict_num_reachable"]),
        "num_queries_virtual": int(metrics["virtual_num_queries"]),
        "num_reachable_virtual": int(metrics["virtual_num_reachable"]),
    }


def _candidate_comparison_rows(
    method: str,
    H: int,
    node_budget: int,
    baselines: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in baselines.to_dict("records"):
        rows.append(
            {
                "baseline": row["baseline"],
                "node_selection": method,
                "H": int(H),
                "node_budget": int(node_budget),
                "num_edges": int(row["num_edges"]),
                "unsupported_edge_rate": float(row["unsupported_edge_rate"]),
                "path_coverage_strict": float(row["path_coverage_strict"]),
                "path_coverage_virtual": np.nan,
                "mean_path_cost_strict": float(row["mean_path_cost_strict"]),
                "mean_path_cost_virtual": np.nan,
                "num_queries_strict": int(row["num_queries_strict"]),
                "num_reachable_strict": int(row["num_reachable_strict"]),
                "num_queries_virtual": np.nan,
                "num_reachable_virtual": np.nan,
            }
        )
    return rows


def _run_method(
    args: argparse.Namespace,
    method: str,
    base_output_dir: Path,
    observations: np.ndarray,
    labels: np.ndarray,
    episodes: list[dict[str, Any]],
    pair_records: dict[str, np.ndarray],
    support_N: Any,
    density_df: pd.DataFrame,
    bottleneck_df: pd.DataFrame,
    queries: pd.DataFrame,
    cluster_model: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    dataset_key = _dataset_key(args.dataset_name)
    run_dir = base_output_dir / dataset_key / f"{method}_budget{args.node_budget}_H{args.H}"
    run_dir.mkdir(parents=True, exist_ok=True)

    selected_nodes = select_nodes(density_df, bottleneck_df, method, args.node_budget, args.seed)
    selected_nodes.to_csv(run_dir / "selected_nodes.csv", index=False)
    option_edges, edge_segments = build_option_edges(
        pair_records,
        labels,
        selected_nodes,
        H=args.H,
        min_support=args.min_support,
        min_episodes=args.min_episodes,
    )
    option_edges = add_edge_costs(
        option_edges,
        cost_type=args.cost_type,
        support_penalty=args.support_penalty,
    )
    option_edges.to_csv(run_dir / "option_edges.csv", index=False)
    np.savez_compressed(run_dir / "edge_segments.npz", **edge_segments)
    self_loop = build_self_loop_summary(pair_records, labels, selected_nodes, args.H)
    self_loop.to_csv(run_dir / "self_loop_summary.csv", index=False)

    G = build_option_graph(
        option_edges,
        cost_type=args.cost_type,
        support_penalty=args.support_penalty,
        selected_nodes=selected_nodes,
    )
    graph_df = graph_summary(G, sample_pairs=args.path_eval_pairs, seed=args.seed)
    graph_df.to_csv(run_dir / "graph_summary.csv", index=False)

    strict_summary, strict_paths = evaluate_task_path_coverage(
        G,
        labels,
        labels,
        queries,
        selected_nodes,
        args.H,
        "strict_selected",
        dataset_name=args.dataset_name,
        node_selection=method,
        node_budget=args.node_budget,
        max_queries=args.path_eval_pairs,
        seed=args.seed,
    )
    virtual_summary, virtual_paths = evaluate_task_path_coverage(
        G,
        labels,
        labels,
        queries,
        selected_nodes,
        args.H,
        "virtual_query",
        dataset_name=args.dataset_name,
        node_selection=method,
        node_budget=args.node_budget,
        support_N=support_N,
        min_support=args.min_support,
        max_queries=args.path_eval_pairs,
        seed=args.seed,
    )
    path_coverage = pd.concat([strict_summary, virtual_summary], ignore_index=True)
    path_coverage.to_csv(run_dir / "path_coverage.csv", index=False)
    strict_paths.to_csv(run_dir / "strict_paths.csv", index=False)
    virtual_paths.to_csv(run_dir / "virtual_paths.csv", index=False)

    compatibility_summary, pair_compat = compute_edge_compatibility(
        option_edges,
        edge_segments,
        labels,
        pair_records,
        args.compat_H_intra,
    )
    compatibility_summary.to_csv(run_dir / "compatibility_summary.csv", index=False)
    path_compat = evaluate_path_compatibility(strict_paths, pair_compat)
    path_compat.to_csv(run_dir / "path_compatibility.csv", index=False)

    utility = bottleneck_removal_ablation(
        G,
        bottleneck_df,
        top_q=args.bottleneck_top_q,
        path_queries=queries[
            queries["start_cluster"].isin(G.nodes()) & queries["goal_cluster"].isin(G.nodes())
        ].reset_index(drop=True),
    )
    utility.to_csv(run_dir / "bottleneck_utility.csv", index=False)

    baselines = _baseline_summary(
        observations,
        labels,
        cluster_model,
        selected_nodes,
        option_edges,
        support_N,
        args.min_support,
        queries,
        args.seed,
        args.knn_k,
        args.geometry_dims,
    )
    baselines.to_csv(run_dir / "baseline_summary.csv", index=False)

    plot_all(selected_nodes, option_edges, path_coverage, utility, run_dir)

    strict_row = strict_summary.iloc[0].to_dict() if not strict_summary.empty else {}
    virtual_row = virtual_summary.iloc[0].to_dict() if not virtual_summary.empty else {}
    metrics = {
        "dataset_name": args.dataset_name,
        "H": int(args.H),
        "node_selection": method,
        "support_baseline": _support_baseline_label(method),
        "node_budget": int(args.node_budget),
        "num_selected_nodes": int(selected_nodes["selected"].sum()),
        "num_option_edges": int(option_edges.shape[0]),
        "num_edge_segments": int(edge_segments["edge_id"].size),
        "mean_option_median_h": float(option_edges["median_h"].mean()) if not option_edges.empty else 0.0,
        "mean_option_cost": float(option_edges["cost"].mean()) if not option_edges.empty else 0.0,
        "median_option_cost": float(option_edges["cost"].median()) if not option_edges.empty else 0.0,
        "mean_num_segments_per_edge": float(option_edges["num_segments"].mean())
        if not option_edges.empty
        else 0.0,
        "strict_num_queries": int(strict_row.get("num_queries", 0)),
        "strict_num_reachable": int(strict_row.get("num_reachable", 0)),
        "strict_path_coverage": float(strict_row.get("path_coverage", 0.0)),
        "strict_mean_path_cost": float(strict_row.get("mean_path_cost", 0.0)),
        "virtual_num_queries": int(virtual_row.get("num_queries", 0)),
        "virtual_num_reachable": int(virtual_row.get("num_reachable", 0)),
        "virtual_path_coverage": float(virtual_row.get("path_coverage", 0.0)),
        "virtual_mean_path_cost": float(virtual_row.get("mean_path_cost", 0.0)),
        "cluster_compatible_rate": float(compatibility_summary.iloc[0]["cluster_compatible_rate"]),
        "strict_compatible_rate": float(compatibility_summary.iloc[0]["strict_compatible_rate"]),
        "graph_reachable_pair_ratio_sampled": float(
            graph_df.iloc[0]["reachable_pair_ratio_sampled"]
        ),
    }
    _write_json(run_dir / "metrics_summary.json", metrics)
    return metrics, baselines


def main() -> None:
    args = parse_args()
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_ogbench_dataset(
        args.dataset_name,
        args.dataset_dir,
        split="train",
        max_transitions=args.max_transitions,
    )
    observations = np.asarray(dataset["observations"])
    flat_observations = observations.reshape(observations.shape[0], -1)
    episodes = split_into_episodes(dataset)
    dataset_key = _dataset_key(args.dataset_name)
    dataset_dir = base_output_dir / dataset_key
    dataset_dir.mkdir(parents=True, exist_ok=True)
    save_dataset_summary(dataset, dataset_dir)

    cluster_model = fit_state_clusters(
        flat_observations,
        method=args.cluster_method,
        n_clusters=args.n_clusters,
        seed=args.seed,
        state_dims=args.state_dims,
    )
    labels = np.asarray(cluster_model["labels"], dtype=np.int64)
    n_clusters = int(cluster_model["metadata"]["n_clusters"])
    density_df = compute_cluster_density(labels, n_clusters=n_clusters)
    density_df.to_csv(dataset_dir / "cluster_density.csv", index=False)

    queries, _ = _load_val_queries(args, cluster_model)
    if queries.empty:
        goal_labels = _final_goal_labels(episodes, cluster_model)
        queries = make_episode_queries(
            labels,
            goal_labels,
            episodes,
            max_queries=args.path_eval_pairs,
            seed=args.seed,
        )
    queries.to_csv(dataset_dir / "path_queries.csv", index=False)

    methods = (
        ["density", "bottleneck", "core_plus_bottleneck", "all"]
        if args.sweep
        else [args.node_selection]
    )
    h_values = args.H_values or [args.H]
    budget_values = args.node_budgets or [args.node_budget]
    rows = []
    baseline_rows = []

    for H in h_values:
        run_args_h = argparse.Namespace(**vars(args))
        run_args_h.H = int(H)
        pair_records = build_h_step_pairs(
            episodes,
            [run_args_h.H],
            seed=run_args_h.seed,
            pair_mode=run_args_h.pair_mode,
            horizon_stride=run_args_h.horizon_stride,
            store_observations=False,
        )
        support_N = compute_support_counts(pair_records, labels, [run_args_h.H], n_clusters)[int(run_args_h.H)]
        support_G = build_directed_support_graph(
            support_N,
            run_args_h.min_support,
            include_self_loops=False,
        )
        crossing = crossing_score(episodes, labels, n_clusters, lag=min(5, max(1, run_args_h.H // 2)))
        betweenness = betweenness_score(support_G, sample_k=min(256, n_clusters), seed=run_args_h.seed)
        removal = removal_impact_score(
            support_G,
            candidate_nodes=range(n_clusters),
            sample_pairs=run_args_h.path_eval_pairs,
            seed=run_args_h.seed,
        )
        bottleneck_df = bottleneck_score(crossing, betweenness, removal, density_df)
        bottleneck_df.to_csv(dataset_dir / f"bottleneck_scores_H{run_args_h.H}.csv", index=False)

        for node_budget in budget_values:
            run_args = argparse.Namespace(**vars(run_args_h))
            run_args.node_budget = int(node_budget)
            for method in methods:
                print(f"[phase2] running method={method} budget={run_args.node_budget} H={run_args.H}")
                metrics, baselines = _run_method(
                    run_args,
                    method,
                    base_output_dir,
                    flat_observations,
                    labels,
                    episodes,
                    pair_records,
                    support_N,
                    density_df,
                    bottleneck_df,
                    queries,
                    cluster_model,
                )
                rows.append(metrics)
                baseline_rows.append(_support_comparison_row(metrics))
                baseline_rows.extend(
                    _candidate_comparison_rows(method, run_args.H, run_args.node_budget, baselines)
                )

    aggregate = pd.DataFrame(rows)
    aggregate.to_csv(dataset_dir / "aggregate_summary.csv", index=False)
    pd.DataFrame(baseline_rows).to_csv(dataset_dir / "baseline_comparison.csv", index=False)
    print(f"[phase2] wrote outputs under {dataset_dir}")


if __name__ == "__main__":
    main()
