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

