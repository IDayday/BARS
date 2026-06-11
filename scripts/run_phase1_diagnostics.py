#!/usr/bin/env python
"""Run Phase 1 offline trajectory diagnostics.

Phase 1 does not train policies, implement TDR/TMD/MQE, or run environment
rollouts. It only analyzes offline OGBench trajectory arrays.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

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
from phase1.data import load_ogbench_dataset, save_dataset_summary  # noqa: E402
from phase1.diagnostics import compare_candidate_graphs  # noqa: E402
from phase1.plotting import plot_all  # noqa: E402
from phase1.support_graph import (  # noqa: E402
    build_directed_support_graph,
    compute_support_counts,
    directed_shortest_path_asymmetry,
    one_way_edge_ratio,
    support_asymmetry,
)
from phase1.trajectory import build_h_step_pairs, split_into_episodes  # noqa: E402


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
    merged["horizons"] = _parse_int_list(merged.get("horizons")) or defaults["horizons"]
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
    parser.add_argument("--pca_dim", type=int, default=None)
    parser.add_argument("--crossing_lag", type=int, default=None)
    parser.add_argument("--betweenness_sample_k", type=int, default=None)
    parser.add_argument("--path_sample_pairs", type=int, default=None)
    parser.add_argument("--removal_sample_pairs", type=int, default=None)
    parser.add_argument("--filter_top_p", type=float, default=None)
    parser.add_argument("--high_bottleneck_q", type=float, default=None)
    return _merge_args(parser.parse_args())


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
    )
    print(f"[phase1] built h-step pairs={pair_records['h'].shape[0]}")

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
        G = build_directed_support_graph(N, args.min_support)
        graphs[int(horizon)] = G
        path_stats = directed_shortest_path_asymmetry(
            G,
            sample_pairs=args.path_sample_pairs,
            seed=args.seed + int(horizon),
        )
        graph_rows.append(
            {
                "H": int(horizon),
                "support_asymmetry": support_asymmetry(N),
                "one_way_edge_ratio": one_way_edge_ratio(N, args.min_support),
                "num_support_edges": int(G.number_of_edges()),
                "num_support_nodes": int(G.number_of_nodes()),
                **path_stats,
            }
        )
    graph_summary = pd.DataFrame(graph_rows)
    graph_summary.to_csv(output_dir / "graph_summary.csv", index=False)

    final_horizon = int(max(args.horizons))
    final_N = support_counts[final_horizon]
    final_G = graphs[final_horizon]

    candidate_summary, _ = compare_candidate_graphs(
        flat_observations,
        labels,
        n_clusters,
        final_N,
        min_support=args.min_support,
        k=args.knn_k,
        seed=args.seed,
        pca_dim=args.pca_dim,
    )
    candidate_summary.to_csv(output_dir / "candidate_edge_summary.csv", index=False)

    crossing = crossing_score(episodes, labels, n_clusters, lag=args.crossing_lag)
    betweenness = betweenness_score(
        final_G,
        sample_k=args.betweenness_sample_k,
        seed=args.seed,
    )
    removal_impact = removal_impact_score(
        final_G,
        candidate_nodes=range(n_clusters),
        sample_pairs=args.removal_sample_pairs,
        seed=args.seed,
    )
    bottleneck_df = bottleneck_score(crossing, betweenness, removal_impact, density_df)
    bottleneck_df.to_csv(output_dir / "bottleneck_scores.csv", index=False)

    filtering_df = filtering_risk_analysis(
        bottleneck_df,
        top_p=args.filter_top_p,
        high_bottleneck_q=args.high_bottleneck_q,
        seed=args.seed,
    )
    filtering_df.to_csv(output_dir / "filtering_risk_summary.csv", index=False)

    plot_all(
        graph_summary,
        candidate_summary,
        bottleneck_df,
        filtering_df,
        final_N,
        output_dir,
    )

    metrics_summary = {
        "dataset_name": args.dataset_name,
        "split": args.split,
        "num_transitions_used": int(dataset_summary["num_transitions"]),
        "num_episodes": int(len(episodes)),
        "mean_episode_length": float(episode_lengths.mean()) if episode_lengths.size else 0.0,
        "n_clusters": int(n_clusters),
        "horizons": [int(h) for h in args.horizons],
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
    }
    _write_json(output_dir / "metrics_summary.json", metrics_summary)
    print(f"[phase1] wrote diagnostics to {output_dir}")


if __name__ == "__main__":
    main()

