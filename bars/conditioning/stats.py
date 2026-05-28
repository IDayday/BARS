from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LowCondStats:
    z_mean: np.ndarray
    z_std: np.ndarray
    tdr_distance_q90: float
    factor_mean: np.ndarray
    factor_std: np.ndarray
    factor_dim_max: int | None = None
    distance_clip: float = 5.0
    residual_clip: float = 5.0
    eps: float = 1e-6

    def __post_init__(self):
        object.__setattr__(self, "z_mean", np.asarray(self.z_mean, dtype=np.float32))
        object.__setattr__(self, "z_std", np.asarray(self.z_std, dtype=np.float32))
        object.__setattr__(self, "factor_mean", np.asarray(self.factor_mean, dtype=np.float32))
        object.__setattr__(self, "factor_std", np.asarray(self.factor_std, dtype=np.float32))
        if self.z_mean.shape != self.z_std.shape:
            raise ValueError(f"z_mean/z_std shape mismatch: {self.z_mean.shape} vs {self.z_std.shape}")
        if self.factor_mean.shape != self.factor_std.shape:
            raise ValueError(f"factor_mean/factor_std shape mismatch: {self.factor_mean.shape} vs {self.factor_std.shape}")

    @classmethod
    def identity(cls, z_dim: int, factor_dim: int, factor_dim_max: int | None = None) -> "LowCondStats":
        return cls(
            z_mean=np.zeros((z_dim,), dtype=np.float32),
            z_std=np.ones((z_dim,), dtype=np.float32),
            tdr_distance_q90=1.0,
            factor_mean=np.zeros((factor_dim,), dtype=np.float32),
            factor_std=np.ones((factor_dim,), dtype=np.float32),
            factor_dim_max=factor_dim if factor_dim_max is None else factor_dim_max,
        )

    @property
    def z_dim(self) -> int:
        return int(self.z_mean.shape[0])

    @property
    def factor_dim(self) -> int:
        return int(self.factor_mean.shape[0])

    @property
    def padded_factor_dim(self) -> int:
        return int(self.factor_dim if self.factor_dim_max is None else self.factor_dim_max)

    def normalize_z(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float32)
        return (z - self.z_mean) / np.maximum(self.z_std, self.eps)

    def distance_scale(self, raw_distance: float) -> float:
        denom = max(float(self.tdr_distance_q90), self.eps)
        value = np.log1p(max(float(raw_distance), 0.0) / denom)
        return float(np.clip(value, 0.0, float(self.distance_clip)))

    def normalize_residual(self, residual: np.ndarray) -> np.ndarray:
        residual = np.asarray(residual, dtype=np.float32)
        value = (residual - self.factor_mean) / np.maximum(self.factor_std, self.eps)
        return np.clip(value, -float(self.residual_clip), float(self.residual_clip)).astype(np.float32)

    def pad_factor(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32)
        target = self.padded_factor_dim
        if value.shape[0] > target:
            raise ValueError(f"factor length {value.shape[0]} exceeds factor_dim_max {target}")
        if value.shape[0] == target:
            return value.astype(np.float32)
        out = np.zeros((target,), dtype=np.float32)
        out[: value.shape[0]] = value
        return out
