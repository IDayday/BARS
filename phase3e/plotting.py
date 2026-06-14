from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_edge_certification(cert: pd.DataFrame, output_dir: str | Path) -> None:
    if cert.empty:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(cert["edge_proxy_score"], bins=30, color="#3b6ea8", alpha=0.85)
    ax.set_xlabel("offline edge proxy score")
    ax.set_ylabel("num edges")
    fig.tight_layout()
    fig.savefig(out / "edge_proxy_score_hist.png", dpi=160)
    plt.close(fig)

    if "heldout_support_lcb" in cert.columns:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(cert["heldout_support_lcb"], cert["edge_proxy_score"], s=12, alpha=0.65)
        ax.set_xlabel("heldout support LCB")
        ax.set_ylabel("offline edge proxy score")
        fig.tight_layout()
        fig.savefig(out / "edge_proxy_vs_support.png", dpi=160)
        plt.close(fig)

    if "edge_bottleneck_score" in cert.columns:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(cert["edge_bottleneck_score"], cert["edge_proxy_score"], s=12, alpha=0.65)
        ax.set_xlabel("edge bottleneck score")
        ax.set_ylabel("offline edge proxy score")
        fig.tight_layout()
        fig.savefig(out / "edge_proxy_vs_bottleneck.png", dpi=160)
        plt.close(fig)
