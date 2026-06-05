from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def auroc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    y = np.asarray(y_true, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    pos = y == 1
    neg = y == 0
    if not np.any(pos) or not np.any(neg):
        return None
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    pos_ranks = np.sum(ranks[pos])
    n_pos = float(np.sum(pos))
    n_neg = float(np.sum(neg))
    return float((pos_ranks - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg))


def auprc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    y = np.asarray(y_true, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    if not np.any(y == 1) or not np.any(y == 0):
        return None
    order = np.argsort(-s)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(float(np.sum(y == 1)), 1.0)
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


@dataclass
class LinearContractModel:
    mean: list[float]
    std: list[float]
    weights: dict[str, list[float]]

    def transform(self, x: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.std, dtype=np.float32)
        return (x - mean) / np.maximum(std, 1e-6)

    def predict_proba(self, x: np.ndarray, target: str) -> np.ndarray:
        w = np.asarray(self.weights[target], dtype=np.float32)
        xz = self.transform(x)
        xb = np.concatenate([xz, np.ones((len(xz), 1), dtype=np.float32)], axis=1)
        return sigmoid(xb @ w)

    def to_dict(self) -> dict[str, Any]:
        return {"model_type": "linear_logistic_contract", "mean": self.mean, "std": self.std, "weights": self.weights}

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> "LinearContractModel":
        return cls(mean=list(record["mean"]), std=list(record["std"]), weights={k: list(v) for k, v in record["weights"].items()})


def train_logistic(x: np.ndarray, y: np.ndarray, *, epochs: int = 400, lr: float = 0.05, l2: float = 1e-4) -> np.ndarray | None:
    if len(x) == 0 or len(np.unique(y)) < 2:
        return None
    xb = np.concatenate([x, np.ones((len(x), 1), dtype=np.float32)], axis=1)
    w = np.zeros((xb.shape[1],), dtype=np.float32)
    y = y.astype(np.float32)
    for _ in range(int(epochs)):
        pred = sigmoid(xb @ w)
        grad = xb.T @ (pred - y) / max(len(y), 1)
        grad[:-1] += l2 * w[:-1]
        w -= lr * grad.astype(np.float32)
    return w


def train_contract_model(x: np.ndarray, labels: dict[str, np.ndarray], *, epochs: int = 400, lr: float = 0.05) -> tuple[LinearContractModel | None, dict[str, Any]]:
    if len(x) == 0:
        return None, {"status": "empty_dataset"}
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    xz = (x - mean) / np.maximum(std, 1e-6)
    weights: dict[str, list[float]] = {}
    metrics: dict[str, Any] = {"status": "trained", "num_examples": int(len(x))}
    for target in ["hit", "contract_positive", "negative_progress"]:
        y = labels.get(target)
        if y is None or len(y) != len(x) or len(np.unique(y)) < 2:
            metrics[f"{target}_status"] = "single_class_or_missing"
            continue
        w = train_logistic(xz, y, epochs=epochs, lr=lr)
        if w is None:
            metrics[f"{target}_status"] = "single_class_or_missing"
            continue
        pred = sigmoid(np.concatenate([xz, np.ones((len(xz), 1), dtype=np.float32)], axis=1) @ w)
        weights[target] = w.tolist()
        metrics[f"{target}_auroc_train"] = auroc(y, pred)
        metrics[f"{target}_auprc_train"] = auprc(y, pred)
    if not weights:
        return None, {**metrics, "status": "no_trainable_binary_targets"}
    return LinearContractModel(mean=mean.tolist(), std=std.tolist(), weights=weights), metrics


def save_model(path: str | Path, model: LinearContractModel) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(model.to_dict(), fh, indent=2, sort_keys=True)


def load_model(path: str | Path) -> LinearContractModel:
    with Path(path).open("r", encoding="utf-8") as fh:
        return LinearContractModel.from_dict(json.load(fh))
