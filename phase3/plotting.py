from __future__ import annotations

from pathlib import Path

import pandas as pd


def _maybe_import_pyplot():
    try:
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def plot_training_curves(train_metrics: pd.DataFrame, val_metrics: pd.DataFrame, output_dir: str | Path) -> None:
    plt = _maybe_import_pyplot()
    if plt is None:
        return
    out = Path(output_dir) / "plots"
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    if not train_metrics.empty:
        ax.plot(train_metrics["step"], train_metrics["train_action_mse"], label="train")
    if not val_metrics.empty:
        ax.plot(val_metrics["step"], val_metrics["val_action_mse"], label="val")
    ax.set_xlabel("step")
    ax.set_ylabel("action MSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "gcbc_action_mse.png", dpi=160)
    plt.close(fig)


def plot_edge_execution(baseline_summary: pd.DataFrame, output_dir: str | Path) -> None:
    plt = _maybe_import_pyplot()
    if plt is None or baseline_summary.empty:
        return
    out = Path(output_dir) / "plots"
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(baseline_summary["edge_source"], baseline_summary["mean_success_rate"])
    ax.set_ylabel("mean success rate")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(out / "baseline_edge_execution.png", dpi=160)
    plt.close(fig)
