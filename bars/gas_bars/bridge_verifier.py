from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

from .bridge_dataset import BridgeDataset, FEATURE_COLUMNS, load_bridge_dataset, make_bridge_table, split_bridge_dataset


class BridgeMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def _safe_auprc(y: np.ndarray, p: np.ndarray) -> float:
    return float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def train_bridge_verifier(
    dataset: BridgeDataset,
    out_dir: str | Path,
    epochs: int = 25,
    lr: float = 1e-3,
    batch_size: int = 256,
    seed: int = 0,
    device: str = "cpu",
) -> dict[str, Any]:
    torch.manual_seed(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device_t = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    mean = dataset.x_train.mean(axis=0, keepdims=True)
    std = dataset.x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    x_train_np = (dataset.x_train - mean) / std
    x_val_np = (dataset.x_val - mean) / std
    model = BridgeMLP(dataset.x_train.shape[1]).to(device_t)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    x = torch.as_tensor(x_train_np, dtype=torch.float32, device=device_t)
    y = torch.as_tensor(dataset.y_train, dtype=torch.float32, device=device_t)
    n = x.shape[0]
    history = []
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n, device=device_t)
        losses = []
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            logits = model(x[idx])
            loss = loss_fn(logits, y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "train_bce": float(np.mean(losses)) if losses else 0.0})
    scores_val = score_array(model, x_val_np, device_t)
    scores_train = score_array(model, x_train_np, device_t)
    metrics = bridge_metrics(dataset, scores_train, scores_val)
    metrics["history"] = history
    metrics["feature_columns"] = dataset.feature_columns
    torch.save({"state_dict": model.state_dict(), "feature_columns": dataset.feature_columns, "mean": mean.astype(np.float32), "std": std.astype(np.float32)}, out / "p_bridge.pt")
    (out / "p_bridge_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True, default=str))
    pd.DataFrame([metrics]).to_csv(out / "p_bridge_metrics.csv", index=False)
    return metrics


def score_array(model: BridgeMLP, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(x, dtype=torch.float32, device=device))
        return torch.sigmoid(logits).detach().cpu().numpy()


def bridge_metrics(dataset: BridgeDataset, train_scores: np.ndarray, val_scores: np.ndarray) -> dict[str, Any]:
    yv = dataset.y_val.astype(int)
    yt = dataset.y_train.astype(int)
    metrics: dict[str, Any] = {
        "train_auroc": _safe_auc(yt, train_scores),
        "train_auprc": _safe_auprc(yt, train_scores),
        "val_auroc": _safe_auc(yv, val_scores),
        "val_auprc": _safe_auprc(yv, val_scores),
        "val_size": int(len(yv)),
        "train_size": int(len(yt)),
    }
    val = dataset.val_df.copy()
    val["p_bridge"] = val_scores
    selected = val[val.get("is_selected_bridge", 0).astype(int) == 1] if len(val) else val
    if len(selected):
        ys = selected["label"].to_numpy(np.int32)
        ps = selected["p_bridge"].to_numpy(np.float32)
        metrics["selected_bridge_AUROC"] = _safe_auc(ys, ps)
        metrics["selected_bridge_AUPRC"] = _safe_auprc(ys, ps)
        base = float(ys.mean())
        metrics["selected_bridge_base_success_rate"] = base
        for thr in [0.5, 0.6, 0.7, 0.8]:
            acc = selected[selected["p_bridge"] >= thr]
            metrics[f"accepted_bridge_success_rate@{thr}"] = float(acc["label"].mean()) if len(acc) else float("nan")
            metrics[f"accepted_bridge_coverage@{thr}"] = float(len(acc) / len(selected))
        acc06 = selected[selected["p_bridge"] >= 0.6]
        accepted_rate = float(acc06["label"].mean()) if len(acc06) else 0.0
        fp_base = 1.0 - base
        fp_acc = 1.0 - accepted_rate if len(acc06) else 1.0
        metrics["false_positive_bridge_relative_reduction@0.6"] = float((fp_base - fp_acc) / max(fp_base, 1e-6))
    else:
        metrics["selected_bridge_AUROC"] = float("nan")
        metrics["selected_bridge_AUPRC"] = float("nan")
    return metrics


def load_model(path: str | Path, in_dim: int | None = None, device: str = "cpu") -> tuple[BridgeMLP, list[str], np.ndarray, np.ndarray]:
    ckpt = torch.load(path, map_location=device)
    cols = ckpt.get("feature_columns", FEATURE_COLUMNS)
    model = BridgeMLP(in_dim or len(cols))
    model.load_state_dict(ckpt["state_dict"])
    model.to(torch.device(device))
    model.eval()
    mean = np.asarray(ckpt.get("mean", np.zeros((1, len(cols)), dtype=np.float32)), dtype=np.float32)
    std = np.asarray(ckpt.get("std", np.ones((1, len(cols)), dtype=np.float32)), dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return model, cols, mean, std


def score_bridge_table(model_path: str | Path, edge_table: pd.DataFrame, out_csv: str | Path, device: str = "cpu") -> pd.DataFrame:
    df = make_bridge_table(edge_table.rename(columns={"edge_exec_success": "success"}) if "success" not in edge_table else edge_table)
    model, cols, mean, std = load_model(model_path, in_dim=len(FEATURE_COLUMNS), device=device)
    x = df[cols].to_numpy(np.float32)
    x = (x - mean) / std
    scores = score_array(model, x, torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu"))
    df["p_bridge"] = scores
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df
