from __future__ import annotations

"""Dataset-level embedding helpers for same-backbone Route-B evaluation.

When an official backbone such as GAS exports dataset embeddings, BARS should not
silently retrain or use a random local TDR model.  This lightweight object lets
the pipeline carry those embeddings through graph construction and provides a
best-effort online encoder for current observations.  If the external policy
exposes its own encoder, rollout should prefer that; this class is a fallback
that maps an online observation to the nearest dataset observation and returns
its stored embedding.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass
class DatasetEmbeddingLookupModel:
    observations: np.ndarray
    embeddings: np.ndarray
    raw_obs_dim: Optional[int] = None
    n_neighbors: int = 1

    def __post_init__(self) -> None:
        obs = np.asarray(self.observations, dtype=np.float32)
        emb = np.asarray(self.embeddings, dtype=np.float32)
        if len(obs) != len(emb):
            raise ValueError(f"observations length {len(obs)} != embeddings length {len(emb)}")
        dim = int(self.raw_obs_dim or obs.shape[1])
        dim = max(1, min(dim, obs.shape[1]))
        self.raw_obs_dim = dim
        self.observations = obs
        self.embeddings = emb
        self._nn = NearestNeighbors(n_neighbors=max(1, int(self.n_neighbors))).fit(obs[:, :dim])
        self.external_dataset_embeddings = True

    def eval(self):
        return self

    def encode_obs(self, obs: np.ndarray, dataset=None, device=None) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        dim = int(self.raw_obs_dim or x.shape[1])
        idx = int(self._nn.kneighbors(x[:, :dim], return_distance=False)[0, 0])
        return np.asarray(self.embeddings[idx], dtype=np.float32)
