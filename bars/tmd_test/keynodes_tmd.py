from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np


@dataclass
class TMDNode:
    node_id: int
    embed: np.ndarray
    dataset_idx: int
    raw_obs: np.ndarray
    cluster_members: list[int]
    metadata: dict


class TMDKeyNodes:
    def __init__(self, max_nodes: int = 256):
        self.max_nodes = int(max_nodes)
        self.nodes: list[TMDNode] = []

    def construct(self, embeddings: np.ndarray, observations: np.ndarray, terminals: np.ndarray, repr_cluster_threshold: float, temporal_horizon_steps: int, te_threshold: float = 0.99, method: str = "td_aware_medoid") -> "TMDKeyNodes":
        embeddings = np.asarray(embeddings, dtype=np.float32)
        observations = np.asarray(observations, dtype=np.float32)
        terminals = np.asarray(terminals).astype(bool)
        if len(embeddings) != len(observations):
            raise ValueError("embeddings and observations must have the same length")
        candidates = self._te_candidates(embeddings, terminals, te_threshold)
        if len(candidates) == 0:
            candidates = np.arange(len(embeddings), dtype=np.int64)
        if len(candidates) > self.max_nodes * 20:
            candidates = candidates[:: max(1, len(candidates) // (self.max_nodes * 20))]
        threshold = float(repr_cluster_threshold)
        if not np.isfinite(threshold) or threshold <= 0:
            diff = np.diff(embeddings[: min(len(embeddings), 1024)], axis=0)
            threshold = max(float(np.median(np.linalg.norm(diff, axis=1))), 1e-6)
        clusters: list[list[int]] = []
        centers: list[np.ndarray] = []
        for idx in candidates:
            emb = embeddings[int(idx)]
            if not centers:
                centers.append(emb.copy()); clusters.append([int(idx)]); continue
            d = np.linalg.norm(np.stack(centers) - emb[None, :], axis=1)
            j = int(np.argmin(d))
            if float(d[j]) <= threshold or len(clusters) >= self.max_nodes:
                clusters[j].append(int(idx)); centers[j] = embeddings[clusters[j]].mean(axis=0)
            else:
                centers.append(emb.copy()); clusters.append([int(idx)])
        self.nodes = []
        for node_id, members in enumerate(clusters):
            mem = np.asarray(members, dtype=np.int64)
            embs = embeddings[mem]
            mean = embs.mean(axis=0)
            dataset_idx = int(mem[int(np.argmin(np.linalg.norm(embs - mean[None, :], axis=1)))])
            self.nodes.append(TMDNode(node_id, embeddings[dataset_idx].copy(), dataset_idx, observations[dataset_idx].copy(), [int(x) for x in mem.tolist()], {"method": method, "cluster_size": int(len(mem)), "te_threshold": float(te_threshold), "temporal_horizon_steps": int(temporal_horizon_steps)}))
        return self

    def _te_candidates(self, embeddings: np.ndarray, terminals: np.ndarray, te_threshold: float) -> np.ndarray:
        if len(embeddings) < 3:
            return np.arange(len(embeddings), dtype=np.int64)
        d = np.diff(embeddings, axis=0)
        dirs = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-8)
        cos = np.sum(dirs[:-1] * dirs[1:], axis=1)
        valid = (~terminals[:-2]) & (~terminals[1:-1])
        idx = np.flatnonzero((cos >= float(te_threshold)) & valid) + 1
        if len(idx) < 8:
            idx = np.flatnonzero(~terminals)
        return idx.astype(np.int64)

    def embeds(self) -> np.ndarray:
        return np.stack([n.embed for n in self.nodes]).astype(np.float32) if self.nodes else np.empty((0, 0), dtype=np.float32)

    def raw_observations(self) -> np.ndarray:
        return np.stack([n.raw_obs for n in self.nodes]).astype(np.float32) if self.nodes else np.empty((0, 0), dtype=np.float32)

    def save(self, path: str | Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f: pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "TMDKeyNodes":
        with open(path, "rb") as f: return pickle.load(f)
