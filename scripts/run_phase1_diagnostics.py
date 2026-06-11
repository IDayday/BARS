#!/usr/bin/env python
"""Run Phase 1 offline trajectory diagnostics.

Phase 1 does not train policies, implement TDR/TMD/MQE, or run environment
rollouts. It only analyzes offline OGBench trajectory arrays.
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
    filtering_risk_analysis,
    removal_impact_score,
)
from phase1.clustering import compute_cluster_density, fit_state_clusters  # noqa: E402
from phase1.clustering import assign_clusters  # noqa: E402
from phase1.data import load_ogbench_dataset, save_dataset_summary  # noqa: E402
from phase1.diagnostics import compare_candidate_graphs  # noqa: E402
from phase1.plotting import plot_all  # noqa: E402
from phase1.support_graph import (  # noqa: E402
    build_directed_support_graph,
    compute_support_counts,
    directed_shortest_path_asymmetry,
    one_way_edge_ratio,
    supported_edge_stats,
    support_asymmetry,
)
from phase1.trajectory import build_h_step_pairs, count_pairs_by_h, split_into_episodes  # noqa: E402


def _parse_int_list(value: str | list[int] | None) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [int(x) for x in value]
    if value == "":
        return None
    return [int(x.strip()) for x in value.split(",") if x.strip()]


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


def _load_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a mapping")
    return data


def _merge_args(args: argparse.Namespace) -> argparse.Namespace:
    config = _load_config(args.config)
    merged = vars(args).copy()
    for key, value in config.items():
        if key not in merged or merged[key] is None:
            merged[key] = value

    defaults = {
        "split": "train",
        "dataset_dir": None,
        "max_transitions": None,
        "cluster_method": "kmeans",
        "state_dims": None,
        "n_clusters": 512,
        "horizons": [1, 5, 10, 25, 50],
        "min_support": 3,
        "knn_k": 10,
        "seed": 0,
        "max_pairs_per_horizon": None,
        "pair_mode": "exact",
        "horizon_stride": 1,
        "task_type": "generic",
        "geometry_dims": None,
        "pca_dim": 16,
        "crossing_lag": 5,
        "betweenness_sample_k": 256,
        "path_sample_pairs": 5000,
        "removal_sample_pairs": 2000,
        "filter_top_p": 0.2,
        "high_bottleneck_q": 0.1,
    }
    for key, value in defaults.items():
        if merged.get(key) is None:
            merged[key] = value

    merged["state_dims"] = _parse_int_list(merged.get("state_dims"))
    geometry_dims_explicit = args.geometry_dims is not None or (
        "geometry_dims" in config and config.get("geometry_dims") not in (None, "")
    )
    merged["geometry_dims"] = _parse_int_list(merged.get("geometry_dims"))
    merged["horizons"] = _parse_int_list(merged.get("horizons")) or defaults["horizons"]
    merged["geometry_dims_explicit"] = bool(geometry_dims_explicit)
    merged["task_type"] = str(merged["task_type"]).lower()
    if merged["task_type"] not in {"maze", "manipulation", "generic"}:
        raise ValueError(
            f"task_type must be maze, manipulation, or generic; got {merged['task_type']!r}"
        )
    if merged.get("dataset_dir") is not None:
        merged["dataset_dir"] = str(Path(str(merged["dataset_dir"])).expanduser())
    if merged.get("output_dir") is not None:
        merged["output_dir"] = str(Path(str(merged["output_dir"])).expanduser())

    missing = [key for key in ("dataset_name", "output_dir") if not merged.get(key)]
    if missing:
        raise ValueError(f"Missing required argument(s): {missing}")
    return argparse.Namespace(**merged)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Optional YAML config path")
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--dataset_dir", default=None)
    parser.add_argument("--split", default=None, choices=["train", "val"])
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_transitions", type=int, default=None)
    parser.add_argument("--cluster_method", default=None, choices=["kmeans", "grid_xy"])
    parser.add_argument("--state_dims", default=None, help="Comma-separated state dimensions")
    parser.add_argument("--n_clusters", type=int, default=None)
    parser.add_argument("--horizons", default=None, help="Comma-separated horizons")
    parser.add_argument("--min_support", type=int, default=None)
    parser.add_argument("--knn_k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_pairs_per_horizon", type=int, default=None)
    parser.add_argument("--pair_mode", default=None, choices=["exact", "dense_upto", "strided_upto"])
    parser.add_argument("--horizon_stride", type=int, default=None)
    parser.add_argument("--task_type", default=None, choices=["maze", "manipulation", "generic"])
    parser.add_argument("--geometry_dims", default=None, help="Comma-separated geometry dims for optional geometry kNN")
    parser.add_argument("--pca_dim", type=int, default=None)
    parser.add_argument("--crossing_lag", type=int, default=None)
    parser.add_argument("--betweenness_sample_k", type=int, default=None)
    parser.add_argument("--path_sample_pairs", type=int, default=None)
    parser.add_argument("--removal_sample_pairs", type=int, default=None)
    parser.add_argument("--filter_top_p", type=float, default=None)
    parser.add_argument("--high_bottleneck_q", type=float, default=None)
    return _merge_args(parser.parse_args())


def _task_path_coverage(
    dataset_name: str,
    dataset_dir: str | None,
    max_transitions: int | None,
    cluster_model: dict[str, Any],
    graphs: dict[int, nx.DiGraph],
) -> pd.DataFrame:
    try:
        val_dataset = load_ogbench_dataset(
            dataset_name=dataset_name,
            dataset_dir=dataset_dir,
            split="val",
            max_transitions=max_transitions,
        )
    except Exception as exc:
        warnings.warn(f"Skipping task path coverage because val split could not be loaded: {exc}")
        return pd.DataFrame()

    val_episodes = split_into_episodes(val_dataset)
    if not val_episodes:
        warnings.warn("Skipping task path coverage because val split has no episodes.")
        return pd.DataFrame()

    start_obs = np.asarray([episode["observations"][0] for episode in val_episodes])
    final_obs = np.asarray([episode["next_observations"][-1] for episode in val_episodes])
    start_clusters = assign_clusters(start_obs, cluster_model)
    goal_clusters = assign_clusters(final_obs, cluster_model)

    rows: list[dict[str, Any]] = []
    for horizon, graph in sorted(graphs.items()):
        reachable = 0
        total = int(start_clusters.shape[0])
        for source in np.unique(start_clusters):
            source_i = int(source)
            mask = start_clusters == source_i
            if source_i not in graph:
                continue
            lengths = nx.single_source_shortest_path_length(graph, source_i)
            reachable += sum(int(goal) in lengths for goal in goal_clusters[mask])
        rows.append(
            {
                "H": int(horizon),
                "num_task_pairs": total,
                "num_reachable": int(reachable),
                "path_coverage": float(reachable / max(1, total)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_ogbench_dataset(
        dataset_name=args.dataset_name,
        dataset_dir=args.dataset_dir,
        split=args.split,
        max_transitions=args.max_transitions,
    )
    dataset_summary = save_dataset_summary(dataset, output_dir)

    observations = np.asarray(dataset["observations"])
    flat_observations = observations.reshape(observations.shape[0], -1)
    episodes = split_into_episodes(dataset)
    episode_lengths = np.asarray([ep["length"] for ep in episodes], dtype=np.float64)
    print(f"[phase1] split episodes={len(episodes)} mean_length={episode_lengths.mean():.2f}")

    pair_records = build_h_step_pairs(
        episodes,
        horizons=args.horizons,
        max_pairs_per_horizon=args.max_pairs_per_horizon,
        seed=args.seed,
        pair_mode=args.pair_mode,
        horizon_stride=args.horizon_stride,
        store_observations=False,
    )
    num_pairs_by_h = count_pairs_by_h(pair_records)
    print(
        f"[phase1] built h-step pairs={pair_records['h'].shape[0]} "
        f"mode={args.pair_mode}"
    )

    cluster_model = fit_state_clusters(
        flat_observations,
        method=args.cluster_method,
        n_clusters=args.n_clusters,
        seed=args.seed,
        state_dims=args.state_dims,
    )
    labels = np.asarray(cluster_model["labels"], dtype=np.int64)
    n_clusters = int(cluster_model["metadata"]["n_clusters"])
    _write_json(output_dir / "cluster_metadata.json", cluster_model["metadata"])

    density_df = compute_cluster_density(labels, n_clusters=n_clusters)
    density_df.to_csv(output_dir / "cluster_density.csv", index=False)
    print(
        f"[phase1] fitted clusters method={args.cluster_method} "
        f"n_clusters={n_clusters} occupied={(density_df['count'] > 0).sum()}"
    )

    support_counts = compute_support_counts(pair_records, labels, args.horizons, n_clusters)
    graph_rows: list[dict[str, Any]] = []
    graphs = {}
    for horizon in args.horizons:
        N = support_counts[int(horizon)]
        G = build_directed_support_graph(N, args.min_support, include_self_loops=False)
        graphs[int(horizon)] = G
        path_stats = directed_shortest_path_asymmetry(
            G,
            sample_pairs=args.path_sample_pairs,
            seed=args.seed + int(horizon),
        )
        edge_stats = supported_edge_stats(N, args.min_support)
        graph_rows.append(
            {
                "H": int(horizon),
                "support_asymmetry": support_asymmetry(N),
                "one_way_edge_ratio": one_way_edge_ratio(
                    N,
                    args.min_support,
                    include_self_loops=False,
                ),
                "num_support_edges": int(G.number_of_edges()),
                "num_support_nodes": int(G.number_of_nodes()),
                **edge_stats,
                **path_stats,
            }
        )
    graph_summary = pd.DataFrame(graph_rows)
    graph_summary.to_csv(output_dir / "graph_summary.csv", index=False)

    final_horizon = int(max(args.horizons))
    final_N = support_counts[final_horizon]
    final_G = graphs[final_horizon]
    geometry_baseline_semantic = args.task_type == "maze"
    effective_geometry_dims = None
    geometry_label = "first2_or_geometry_dims_kNN"
    include_grid_adjacent = False
    if args.task_type == "maze":
        geometry_label = "xy_state_kNN"
        include_grid_adjacent = cluster_model.get("method") == "grid_xy"
        if args.geometry_dims is not None:
            effective_geometry_dims = args.geometry_dims
        elif args.state_dims is not None and len(args.state_dims) == 2:
            effective_geometry_dims = list(args.state_dims)
        elif flat_observations.shape[1] >= 2:
            effective_geometry_dims = [0, 1]
    elif args.geometry_dims_explicit and args.geometry_dims is not None:
        effective_geometry_dims = args.geometry_dims

    candidate_summary, _ = compare_candidate_graphs(
        flat_observations,
        labels,
        n_clusters,
        final_N,
        min_support=args.min_support,
        k=args.knn_k,
        cluster_model=cluster_model,
        geometry_dims=effective_geometry_dims,
        geometry_label=geometry_label,
        include_grid_adjacent=include_grid_adjacent,
        include_self_loops=False,
        seed=args.seed,
        pca_dim=args.pca_dim,
    )
    candidate_summary.to_csv(output_dir / "candidate_edge_summary.csv", index=False)

    crossing = crossing_score(episodes, labels, n_clusters, lag=args.crossing_lag)
    bottleneck_by_h: dict[int, pd.DataFrame] = {}
    filtering_by_h: dict[int, pd.DataFrame] = {}
    retention_rows: list[dict[str, Any]] = []
    for horizon in args.horizons:
        horizon_i = int(horizon)
        graph = graphs[horizon_i]
        betweenness = betweenness_score(
            graph,
            sample_k=args.betweenness_sample_k,
            seed=args.seed + horizon_i,
        )
        removal_impact = removal_impact_score(
            graph,
            candidate_nodes=range(n_clusters),
            sample_pairs=args.removal_sample_pairs,
            seed=args.seed + horizon_i,
        )
        bottleneck_df_h = bottleneck_score(crossing, betweenness, removal_impact, density_df)
        filtering_df_h = filtering_risk_analysis(
            bottleneck_df_h,
            top_p=args.filter_top_p,
            high_bottleneck_q=args.high_bottleneck_q,
            seed=args.seed + horizon_i,
        )
        bottleneck_df_h.to_csv(output_dir / f"bottleneck_scores_H{horizon_i}.csv", index=False)
        filtering_df_h.to_csv(
            output_dir / f"filtering_risk_summary_H{horizon_i}.csv",
            index=False,
        )
        bottleneck_by_h[horizon_i] = bottleneck_df_h
        filtering_by_h[horizon_i] = filtering_df_h
        row = {"H": horizon_i}
        for filter_row in filtering_df_h.itertuples(index=False):
            row[str(filter_row.filter)] = float(filter_row.retention_rate)
        retention_rows.append(row)

    bottleneck_df = bottleneck_by_h[final_horizon]
    filtering_df = filtering_by_h[final_horizon]
    bottleneck_df.to_csv(output_dir / "bottleneck_scores.csv", index=False)
    filtering_df.to_csv(output_dir / "filtering_risk_summary.csv", index=False)
    bottleneck_retention_by_h = pd.DataFrame(retention_rows).sort_values("H")
    bottleneck_retention_by_h.to_csv(output_dir / "bottleneck_retention_by_H.csv", index=False)

    task_path_coverage = _task_path_coverage(
        dataset_name=args.dataset_name,
        dataset_dir=args.dataset_dir,
        max_transitions=args.max_transitions,
        cluster_model=cluster_model,
        graphs=graphs,
    )
    if not task_path_coverage.empty:
        task_path_coverage.to_csv(output_dir / "task_path_coverage_by_H.csv", index=False)

    plot_all(
        graph_summary,
        candidate_summary,
        bottleneck_df,
        filtering_df,
        final_N,
        output_dir,
        bottleneck_retention_by_h=bottleneck_retention_by_h,
    )

    metrics_summary = {
        "dataset_name": args.dataset_name,
        "split": args.split,
        "num_transitions_used": int(dataset_summary["num_transitions"]),
        "num_episodes": int(len(episodes)),
        "mean_episode_length": float(episode_lengths.mean()) if episode_lengths.size else 0.0,
        "n_clusters": int(n_clusters),
        "horizons": [int(h) for h in args.horizons],
        "pair_mode": args.pair_mode,
        "horizon_stride": int(args.horizon_stride),
        "task_type": args.task_type,
        "num_pairs_by_h": {str(h): int(count) for h, count in num_pairs_by_h.items()},
        "geometry_dims": effective_geometry_dims,
        "geometry_dims_explicit": bool(args.geometry_dims_explicit),
        "geometry_baseline_semantic": bool(geometry_baseline_semantic),
        "support_asymmetry_by_H": {
            str(int(row.H)): float(row.support_asymmetry)
            for row in graph_summary.itertuples(index=False)
        },
        "one_way_edge_ratio_by_H": {
            str(int(row.H)): float(row.one_way_edge_ratio)
            for row in graph_summary.itertuples(index=False)
        },
        "directed_reachable_pair_ratio_by_H": {
            str(int(row.H)): float(row.directed_reachable_pair_ratio)
            for row in graph_summary.itertuples(index=False)
        },
        "directed_oneway_reachable_ratio_by_H": {
            str(int(row.H)): float(row.one_way_reachable_ratio)
            for row in graph_summary.itertuples(index=False)
        },
        "unsupported_edge_rate_by_candidate_type": {
            str(row.candidate_type): float(row.unsupported_edge_rate)
            for row in candidate_summary.itertuples(index=False)
        },
        "high_bottleneck_retention_by_filter": {
            str(row.filter): float(row.retention_rate)
            for row in filtering_df.itertuples(index=False)
        },
        "high_bottleneck_retention_by_H": {
            str(int(row.H)): {
                key: float(getattr(row, key))
                for key in bottleneck_retention_by_h.columns
                if key != "H"
            }
            for row in bottleneck_retention_by_h.itertuples(index=False)
        },
    }
    if not task_path_coverage.empty:
        metrics_summary["task_path_coverage_by_H"] = {
            str(int(row.H)): float(row.path_coverage)
            for row in task_path_coverage.itertuples(index=False)
        }
    _write_json(output_dir / "metrics_summary.json", metrics_summary)
    print(f"[phase1] wrote diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
