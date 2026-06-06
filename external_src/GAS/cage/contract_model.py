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


TARGET_MODE_IDS = {
    "gas_path": 0.0,
    "original_target": 0.0,
    "cage_selected": 1.0,
    "nearest_path_target": 2.0,
    "farther_path_target": 3.0,
    "final_goal": 4.0,
    "recovery": 5.0,
    "recovery_candidate": 5.0,
    "fallback_to_gas": 6.0,
}


@dataclass(frozen=True)
class ContractPrediction:
    predicted_hit: float
    predicted_contract_positive: float
    predicted_negative_progress: float
    uncertainty: float
    lower_confidence_bound: float
    model_loaded: bool
    model_path: str | None = None


def _float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool(value: Any) -> float:
    return 1.0 if bool(value) else 0.0


def _env_bucket(env_name: str | None) -> float:
    text = (env_name or "").lower()
    if "humanoid" in text:
        return 2.0
    if "teleport" in text:
        return 1.0
    if "antmaze" in text:
        return 0.0
    if "kitchen" in text:
        return 3.0
    return -1.0


def build_contract_feature_vector(features: dict[str, Any], expected_dim: int | None = None) -> np.ndarray:
    phi_s = np.asarray(features.get("phi_s", []), dtype=np.float32).reshape(-1)
    phi_g = np.asarray(features.get("phi_g", []), dtype=np.float32).reshape(-1)
    if phi_s.size == 0 or phi_g.size == 0:
        raise ValueError("contract features require nonempty phi_s and phi_g")
    if phi_s.shape != phi_g.shape:
        raise ValueError(f"phi_s/phi_g shape mismatch: {phi_s.shape} vs {phi_g.shape}")
    delta = phi_g - phi_s
    d_phi = _float(features.get("d_phi"), float(np.linalg.norm(delta)))
    base = np.concatenate([phi_s, phi_g, delta, np.abs(delta), np.asarray([d_phi], dtype=np.float32)])
    extras = np.asarray(
        [
            TARGET_MODE_IDS.get(str(features.get("target_mode", "original_target")), -1.0),
            _float(features.get("path_position"), -1.0),
            _bool(features.get("final_phase", False)),
            _bool(features.get("recovery_candidate", False)),
            _float(features.get("recent_stall_count"), 0.0),
            _float(features.get("recent_drift_count"), 0.0),
            _float(features.get("commitment_steps"), 0.0),
            _float(features.get("previous_target_distance"), -1.0),
            _float(features.get("current_target_distance"), d_phi),
            _float(features.get("q_train_support"), -1.0),
            _env_bucket(features.get("env_name")),
        ],
        dtype=np.float32,
    )
    vector = np.concatenate([base, extras])
    if expected_dim is None:
        return vector
    if expected_dim <= len(base):
        vector = base
    if len(vector) < expected_dim:
        vector = np.pad(vector, (0, expected_dim - len(vector)), mode="constant")
    elif len(vector) > expected_dim:
        vector = vector[:expected_dim]
    return vector.astype(np.float32)


class ContractScorer:
    """Small online scorer for CAGE contract-gated execution."""

    def __init__(self, model: LinearContractModel | None = None, model_path: str | None = None, uncertainty_penalty: float = 0.25):
        self.model = model
        self.model_path = model_path
        self.uncertainty_penalty = float(uncertainty_penalty)

    @classmethod
    def from_path(cls, path: str | Path | None, uncertainty_penalty: float = 0.25) -> "ContractScorer":
        if not path:
            return cls(None, None, uncertainty_penalty)
        path = Path(path)
        if not path.exists():
            return cls(None, str(path), uncertainty_penalty)
        return cls(load_model(path), str(path), uncertainty_penalty)

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def predict(self, features: dict[str, Any]) -> ContractPrediction:
        d_phi = _float(features.get("d_phi"), 0.0)
        if self.model is None:
            # Conservative fallback: short targets are plausible, long targets are uncertain.
            tau = max(1.0, _float(features.get("fallback_tau"), 10.0))
            base_score = float(np.exp(-max(0.0, d_phi) / tau))
            uncertainty = min(1.0, 0.35 + max(0.0, d_phi) / (4.0 * tau))
            predicted_negative = 1.0 - base_score
            lcb = max(0.0, base_score - self.uncertainty_penalty * uncertainty)
            return ContractPrediction(
                predicted_hit=base_score,
                predicted_contract_positive=base_score,
                predicted_negative_progress=predicted_negative,
                uncertainty=uncertainty,
                lower_confidence_bound=lcb,
                model_loaded=False,
                model_path=self.model_path,
            )
        x = build_contract_feature_vector(features, expected_dim=len(self.model.mean))[None, :]
        predicted_hit = self._predict_target(x, "hit")
        predicted_positive = self._predict_target(x, "contract_positive", default=predicted_hit)
        predicted_negative = self._predict_target(x, "negative_progress", default=1.0 - predicted_positive)
        confidence = max(predicted_positive, 1.0 - predicted_positive)
        uncertainty = float(1.0 - confidence)
        lcb = max(0.0, predicted_positive - self.uncertainty_penalty * uncertainty - 0.5 * predicted_negative)
        return ContractPrediction(
            predicted_hit=predicted_hit,
            predicted_contract_positive=predicted_positive,
            predicted_negative_progress=predicted_negative,
            uncertainty=uncertainty,
            lower_confidence_bound=lcb,
            model_loaded=True,
            model_path=self.model_path,
        )

    def _predict_target(self, x: np.ndarray, target: str, default: float | None = None) -> float:
        if self.model is None or target not in self.model.weights:
            return float(default if default is not None else 0.0)
        return float(self.model.predict_proba(x, target)[0])
