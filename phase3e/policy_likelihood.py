from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from phase3.edge_bc_dataset import build_edge_bc_examples
from phase3.models import GCBCMLP
from phase3.train_gcbc import evaluate_policy_mse


def load_cached_edge_mse(edge_metrics_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(Path(edge_metrics_csv).expanduser())
    if "num_val_samples" not in df.columns and "num_examples" in df.columns:
        df = df.rename(columns={"num_examples": "num_policy_eval_samples"})
    if "num_policy_eval_samples" not in df.columns:
        sample_col = "num_val_samples" if "num_val_samples" in df.columns else None
        df["num_policy_eval_samples"] = df[sample_col] if sample_col else 0
    return df[["edge_id", "val_action_mse", "num_policy_eval_samples"]].rename(
        columns={"val_action_mse": "edge_action_mse"}
    )


def compute_model_edge_mse(
    dataset: dict[str, Any],
    option_edges: pd.DataFrame,
    edge_segments: dict[str, np.ndarray],
    heldout_segment_indices: np.ndarray,
    model_path: str | Path,
    batch_size: int = 2048,
    max_examples: int = 50000,
    seed: int = 0,
    device: str | None = None,
) -> pd.DataFrame:
    import torch

    heldout = build_edge_bc_examples(
        dataset,
        option_edges,
        edge_segments,
        max_examples=max_examples,
        sampling_mode="uniform_transition",
        seed=seed,
    ).with_segment_indices(
        np.asarray(heldout_segment_indices, dtype=np.int64),
        max_examples=max_examples,
        sampling_mode="uniform_transition",
        seed=seed + 19,
    )
    ckpt = torch.load(Path(model_path).expanduser(), map_location="cpu", weights_only=False)
    model = GCBCMLP(**ckpt["model_config"])
    model.load_state_dict(ckpt["model_state_dict"])
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev)
    _, edge_metrics = evaluate_policy_mse(
        model,
        heldout,
        batch_size=int(batch_size),
        max_examples=int(max_examples),
        device=dev,
    )
    if edge_metrics.empty:
        return pd.DataFrame(columns=["edge_id", "edge_action_mse", "num_policy_eval_samples"])
    edge_metrics = edge_metrics.rename(
        columns={"val_action_mse": "edge_action_mse", "num_examples": "num_policy_eval_samples"}
    )
    return edge_metrics[["edge_id", "edge_action_mse", "num_policy_eval_samples"]]


def edge_policy_support_scores(
    edge_mse: pd.DataFrame,
    temperature: float = 0.05,
) -> pd.DataFrame:
    out = edge_mse.copy()
    if out.empty:
        out["edge_action_mse_ucb"] = pd.Series(dtype=np.float64)
        out["edge_policy_support_score"] = pd.Series(dtype=np.float64)
        return out
    counts = pd.to_numeric(out.get("num_policy_eval_samples", 0), errors="coerce").fillna(0).clip(lower=1)
    mse = pd.to_numeric(out["edge_action_mse"], errors="coerce")
    out["edge_action_mse_ucb"] = mse + 1.96 * np.sqrt(np.maximum(mse, 0.0) / counts)
    out["edge_policy_support_score"] = np.exp(-mse.fillna(float("inf")) / max(1e-12, float(temperature)))
    out.loc[~np.isfinite(out["edge_policy_support_score"]), "edge_policy_support_score"] = 0.0
    return out
