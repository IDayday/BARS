from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def _plots_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir) / "plots"
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_selected_nodes(selected_nodes: pd.DataFrame, output_dir: str | Path) -> None:
    plots = _plots_dir(output_dir)
    df = selected_nodes.copy()
    plt.figure(figsize=(6, 4))
    plt.scatter(df["density"], df["bottleneck_score"], s=12, alpha=0.5, label="all occupied")
    chosen = df[df["selected"]]
    if not chosen.empty:
        plt.scatter(chosen["density"], chosen["bottleneck_score"], s=18, alpha=0.9, label="selected")
    plt.xlabel("Cluster density")
    plt.ylabel("Bottleneck score")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plots / "selected_nodes_density_bottleneck.png", dpi=180)
    plt.close()


def plot_path_coverage(path_coverage: pd.DataFrame, output_dir: str | Path) -> None:
    plots = _plots_dir(output_dir)
    if path_coverage.empty:
        return
    plt.figure(figsize=(6, 4))
    plt.bar(path_coverage["mode"], path_coverage["path_coverage"], color="#4c78a8")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Path coverage")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(plots / "path_coverage.png", dpi=180)
    plt.close()


def plot_option_edges(option_edges: pd.DataFrame, output_dir: str | Path) -> None:
    plots = _plots_dir(output_dir)
    if option_edges.empty:
        return
    plt.figure(figsize=(6, 4))
    plt.hist(option_edges["median_h"], bins=20, color="#2f7f6f", alpha=0.85)
    plt.xlabel("Median segment horizon")
    plt.ylabel("Option edge count")
    plt.tight_layout()
    plt.savefig(plots / "option_edge_median_h.png", dpi=180)
    plt.close()


def plot_bottleneck_utility(utility: pd.DataFrame, output_dir: str | Path) -> None:
    plots = _plots_dir(output_dir)
    if utility.empty:
        return
    plt.figure(figsize=(6, 4))
    plt.bar(utility["condition"], utility["path_coverage"], color="#e68633")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Path coverage")
    plt.tight_layout()
    plt.savefig(plots / "bottleneck_removal_coverage.png", dpi=180)
    plt.close()


def plot_all(
    selected_nodes: pd.DataFrame,
    option_edges: pd.DataFrame,
    path_coverage: pd.DataFrame,
    bottleneck_utility: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    plot_selected_nodes(selected_nodes, output_dir)
    plot_option_edges(option_edges, output_dir)
    plot_path_coverage(path_coverage, output_dir)
    plot_bottleneck_utility(bottleneck_utility, output_dir)


def _series_label(row: tuple[object, object]) -> str:
    method, H = row
    return f"{method}, H={H}"


def _plot_metric_by_budget(
    aggregate: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_dir: str | Path,
    filename: str,
) -> None:
    if aggregate.empty or metric not in aggregate.columns:
        return
    plots = _plots_dir(output_dir)
    df = aggregate.dropna(subset=["node_budget", metric]).copy()
    if df.empty:
        return
    plt.figure(figsize=(7, 4.5))
    for key, group in df.groupby(["node_selection", "H"], sort=True):
        group = group.sort_values("node_budget", kind="mergesort")
        plt.plot(
            group["node_budget"],
            group[metric],
            marker="o",
            linewidth=1.5,
            markersize=4,
            label=_series_label(key),
        )
    plt.xlabel("Node budget")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(plots / filename, dpi=180)
    plt.close()


def plot_aggregate_summary(aggregate: pd.DataFrame, output_dir: str | Path) -> None:
    _plot_metric_by_budget(
        aggregate,
        "strict_coverage_over_all",
        "Strict coverage over all queries",
        output_dir,
        "coverage_vs_budget_by_method.png",
    )
    _plot_metric_by_budget(
        aggregate,
        "virtual_path_coverage",
        "Virtual path coverage",
        output_dir,
        "virtual_coverage_vs_budget_by_method.png",
    )
    _plot_metric_by_budget(
        aggregate,
        "strict_compatible_rate",
        "Strict compatible rate",
        output_dir,
        "compatibility_vs_budget_by_method.png",
    )
    _plot_metric_by_budget(
        aggregate,
        "num_option_edges",
        "Option edges",
        output_dir,
        "num_edges_vs_budget_by_method.png",
    )
    _plot_metric_by_budget(
        aggregate,
        "bottleneck_removal_delta_coverage",
        "Coverage drop after bottleneck removal",
        output_dir,
        "bottleneck_removal_delta_by_method.png",
    )
