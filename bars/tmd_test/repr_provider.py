from __future__ import annotations

import numpy as np

from .tmd_agent_adapter import TMDAgentAdapter


class RepresentationProvider:
    def encode(self, observations) -> np.ndarray:
        raise NotImplementedError

    def distance_embeddings(self, src_embeds, dst_embeds) -> np.ndarray:
        raise NotImplementedError

    def pairwise_distance(self, src_embeds, dst_embeds, batch_size: int = 512) -> np.ndarray:
        src = np.asarray(src_embeds, dtype=np.float32)
        dst = np.asarray(dst_embeds, dtype=np.float32)
        out = np.empty((len(src), len(dst)), dtype=np.float32)
        for i in range(0, len(src), batch_size):
            out[i : i + batch_size] = self.distance_embeddings(src[i : i + batch_size], dst)
        return out

    def paired_distance(self, src_embeds, dst_embeds, batch_size: int = 512) -> np.ndarray:
        src = np.asarray(src_embeds, dtype=np.float32)
        dst = np.asarray(dst_embeds, dtype=np.float32)
        if len(src) != len(dst):
            raise ValueError(f"paired_distance requires equal lengths, got {len(src)} and {len(dst)}")
        out = np.empty((len(src),), dtype=np.float32)
        for i in range(0, len(src), batch_size):
            j = min(i + batch_size, len(src))
            block = np.asarray(self.distance_embeddings(src[i:j], dst[i:j]), dtype=np.float32)
            out[i:j] = np.diag(block)
        return out

    def direction(self, src_embed, dst_embed, eps: float = 1e-8) -> np.ndarray:
        v = np.asarray(dst_embed, dtype=np.float32) - np.asarray(src_embed, dtype=np.float32)
        return v / max(float(np.linalg.norm(v)), eps)


class TMDRepresentationProvider(RepresentationProvider):
    def __init__(self, tmd_agent: TMDAgentAdapter, batch_size: int = 512):
        self.tmd_agent = tmd_agent
        self.batch_size = int(batch_size)

    def encode(self, observations) -> np.ndarray:
        obs = np.asarray(observations, dtype=np.float32)
        squeeze = obs.ndim == 1
        if squeeze:
            obs = obs[None, :]
        chunks = []
        for i in range(0, len(obs), self.batch_size):
            chunks.append(np.asarray(self.tmd_agent.get_psi(obs[i : i + self.batch_size]), dtype=np.float32))
        out = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 0), dtype=np.float32)
        return out[0] if squeeze else out

    def distance_embeddings(self, src_embeds, dst_embeds) -> np.ndarray:
        return np.asarray(self.tmd_agent.distance_from_embeddings(src_embeds, dst_embeds), dtype=np.float32)
