from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class ECGPolicyAdapter:
    def __init__(self, payload: dict[str, Any]):
        if payload.get("model_type") != "linear_ecg_policy_adapter":
            raise ValueError(f"unsupported ECG policy adapter type: {payload.get('model_type')}")
        self.mean = np.asarray(payload["mean"], dtype=np.float32)
        self.std = np.asarray(payload["std"], dtype=np.float32)
        self.weights = np.asarray(payload["weights"], dtype=np.float32)
        self.bias = np.asarray(payload["bias"], dtype=np.float32)

    @classmethod
    def from_path(cls, path: str | Path) -> "ECGPolicyAdapter":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"ECG policy adapter not found: {path}")
        with path.open(encoding="utf-8") as fh:
            return cls(json.load(fh))

    def predict(self, observation: np.ndarray, phi_s: np.ndarray, phi_g: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float32).reshape(-1)
        phi_s = np.asarray(phi_s, dtype=np.float32).reshape(-1)
        phi_g = np.asarray(phi_g, dtype=np.float32).reshape(-1)
        x = np.concatenate([observation, phi_s, phi_g, phi_g - phi_s, np.abs(phi_g - phi_s)], axis=0)
        if x.shape[0] != self.mean.shape[0]:
            raise ValueError(f"ECG adapter feature dim mismatch: got {x.shape[0]}, expected {self.mean.shape[0]}")
        xz = (x - self.mean) / np.maximum(self.std, 1e-6)
        return xz @ self.weights + self.bias
