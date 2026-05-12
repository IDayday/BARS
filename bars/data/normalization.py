from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np
@dataclass
class Normalizer:
    mean: np.ndarray; std: np.ndarray; eps: float = 1e-6
    @classmethod
    def fit(cls, x: np.ndarray, eps: float = 1e-6) -> 'Normalizer':
        mean = x.mean(axis=0).astype(np.float32); std = x.std(axis=0).astype(np.float32); std = np.where(std < eps, 1.0, std)
        return cls(mean, std, eps)
    def encode(self, x: np.ndarray) -> np.ndarray: return ((x - self.mean) / (self.std + self.eps)).astype(np.float32)
    def decode(self, x: np.ndarray) -> np.ndarray: return (x * (self.std + self.eps) + self.mean).astype(np.float32)
    def state_dict(self) -> Dict[str, np.ndarray]: return {'mean': self.mean, 'std': self.std, 'eps': np.array(self.eps)}
    @classmethod
    def from_state_dict(cls, d: Dict[str, np.ndarray]) -> 'Normalizer': return cls(d['mean'], d['std'], float(d.get('eps', 1e-6)))
