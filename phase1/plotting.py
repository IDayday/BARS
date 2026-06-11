from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse


def _ensure_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_line_plot(
    df: pd.DataFrame,
    output_dir: Path,
    x: str,
    y: str,
    filename: str,
    ylabel: str,
) -> None:
    csv_path = output_dir / filename.replace(".png", ".csv")
    df[[x, y]].to_csv(csv_path, index=False)
    plt.figure(figsize=(6, 4))
    plt.plot(df[x], df[y], marker="o", linewidth=2)
    plt.xlabel(x)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=180)
    plt.close()


def plot_support_metrics(graph_summary: pd.DataFrame, output_dir: str | Path) -> None:
    output_path = _ensure_output_dir(output_dir)
    _save_line_plot(
        graph_summary,
        output_path,
        "H",
        "support_asymmetry",
        "support_asymmetry_by_H.png",
        "A_support",
    )
    _save_line_plot(
        graph_summary,
        output_path,
        "H",
        "one_way_edge_ratio",
        "one_way_edge_ratio_by_H.png",
        "R_oneway",
    )


def plot_unsupported_edge_rate(candidate_summary: pd.DataFrame, output_dir: str | Path) -> None:
    output_path = _ensure_output_dir(output_dir)
    plot_df = candidate_summary[
        candidate_summary["candidate_type"].isin(
            ["raw_state_kNN", "PCA_state_kNN", "random_edges"]
        )
    ].copy()
    plot_df[["candidate_type", "unsupported_edge_rate"]].to_csv(
        output_path / "unsupported_edge_rate_bar.csv",
        index=False,
    )
    plt.figure(figsize=(7, 4))
    plt.bar(plot_df["candidate_type"], plot_df["unsupported_edge_rate"], color="#4c78a8")
    plt.ylabel("Unsupported edge rate")
    plt.ylim(0.0, 1.0)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path / "unsupported_edge_rate_bar.png", dpi=180)
    plt.close()


def plot_density_vs_bottleneck(bottleneck_scores: pd.DataFrame, output_dir: str | Path) -> None:
    output_path = _ensure_output_dir(output_dir)
    bottleneck_scores[["cluster", "density", "bottleneck_score"]].to_csv(
        output_path / "cluster_density_vs_bottleneck_score.csv",
        index=False,
    )
    plt.figure(figsize=(6, 4))
    plt.scatter(
        bottleneck_scores["density"],
        bottleneck_scores["bottleneck_score"],
        s=14,
        alpha=0.75,
        color="#2f7f6f",
    )
    plt.xlabel("Cluster density")
    plt.ylabel("Bottleneck score")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path / "cluster_density_vs_bottleneck_score.png", dpi=180)
    plt.close()


def plot_bottleneck_retention(filtering_summary: pd.DataFrame, output_dir: str | Path) -> None:
    output_path = _ensure_output_dir(output_dir)
    filtering_summary[["filter", "retention_rate"]].to_csv(
        output_path / "bottleneck_retention_bar.csv",
        index=False,
    )
    plt.figure(figsize=(7, 4))
    plt.bar(filtering_summary["filter"], filtering_summary["retention_rate"], color="#e68633")
    plt.ylabel("High-bottleneck retention")
    plt.ylim(0.0, 1.0)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path / "bottleneck_retention_bar.png", dpi=180)
    plt.close()


def _matrix_to_dense(matrix: sparse.spmatrix | np.ndarray, order: np.ndarray) -> np.ndarray:
    if sparse.issparse(matrix):
        return matrix.tocsr()[order][:, order].toarray()
    arr = np.asarray(matrix)
    return arr[np.ix_(order, order)]


def plot_support_matrix_heatmaps(
    N: sparse.spmatrix | np.ndarray,
    bottleneck_scores: pd.DataFrame,
    output_dir: str | Path,
    max_clusters: int = 128,
) -> None:
    output_path = _ensure_output_dir(output_dir)
    n_clusters = N.shape[0]
    ranked = bottleneck_scores.sort_values("density", ascending=False)["cluster"].to_numpy(
        dtype=np.int64
    )
    ranked = ranked[(ranked >= 0) & (ranked < n_clusters)]
    if ranked.size == 0:
        return
    order = ranked[: min(int(max_clusters), ranked.size)]
    support = _matrix_to_dense(N, order)
    asym = support - support.T

    support_df = pd.DataFrame(support, index=order, columns=order)
    support_df.to_csv(output_path / "support_matrix_heatmap.csv")
    plt.figure(figsize=(7, 6))
    plt.imshow(support, aspect="auto", interpolation="nearest", cmap="viridis")
    plt.colorbar(label="N_ij")
    plt.xlabel("Destination cluster")
    plt.ylabel("Source cluster")
    plt.tight_layout()
    plt.savefig(output_path / "support_matrix_heatmap.png", dpi=180)
    plt.close()

    asym_df = pd.DataFrame(asym, index=order, columns=order)
    asym_df.to_csv(output_path / "asymmetry_matrix_heatmap.csv")
    vmax = float(np.max(np.abs(asym))) if asym.size else 1.0
    vmax = max(vmax, 1.0)
    plt.figure(figsize=(7, 6))
    plt.imshow(
        asym,
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )
    plt.colorbar(label="N_ij - N_ji")
    plt.xlabel("Destination cluster")
    plt.ylabel("Source cluster")
    plt.tight_layout()
    plt.savefig(output_path / "asymmetry_matrix_heatmap.png", dpi=180)
    plt.close()


def plot_all(
    graph_summary: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    bottleneck_scores: pd.DataFrame,
    filtering_summary: pd.DataFrame,
    support_matrix: sparse.spmatrix | np.ndarray,
    output_dir: str | Path,
) -> None:
    plot_support_metrics(graph_summary, output_dir)
    plot_unsupported_edge_rate(candidate_summary, output_dir)
    plot_density_vs_bottleneck(bottleneck_scores, output_dir)
    plot_bottleneck_retention(filtering_summary, output_dir)
    plot_support_matrix_heatmaps(support_matrix, bottleneck_scores, output_dir)

