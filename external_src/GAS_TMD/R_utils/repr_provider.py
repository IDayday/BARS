import numpy as np

from tqdm import tqdm


class TMDRepresentationProvider:
    """Batched TMD representation and distance API for graph construction."""

    def __init__(self, agent, batch_size: int, show_progress: bool = True):
        self.agent = agent
        self.batch_size = int(batch_size)
        self.show_progress = bool(show_progress)

    def encode(self, observations) -> np.ndarray:
        """Return psi(s), with the TMD ensemble averaged by the agent."""
        observations = np.asarray(observations)
        single = observations.ndim == 1
        if single:
            observations = observations[None]

        chunks = []
        for start in tqdm(
            range(0, len(observations), self.batch_size),
            desc="Encoding TMD psi",
            leave=False,
            disable=not self.show_progress,
        ):
            end = min(start + self.batch_size, len(observations))
            chunks.append(np.asarray(self.agent.get_psi(observations[start:end])))
        embeds = np.concatenate(chunks, axis=0)
        return embeds[0] if single else embeds

    def distance_embeddings(self, src_embeds, dst_embeds, batch_size=None) -> np.ndarray:
        """Return directed pairwise distances D[i, j] = d_TMD(src_i, dst_j)."""
        src_embeds = np.asarray(src_embeds, dtype=np.float32)
        dst_embeds = np.asarray(dst_embeds, dtype=np.float32)
        src_single = src_embeds.ndim == 1
        dst_single = dst_embeds.ndim == 1
        if src_single:
            src_embeds = src_embeds[None]
        if dst_single:
            dst_embeds = dst_embeds[None]

        distances = pairwise_tmd_distance_matrix(
            self.agent,
            src_embeds,
            dst_embeds,
            batch_size or self.batch_size,
            show_progress=self.show_progress,
        )
        if src_single and dst_single:
            return distances[0, 0]
        if src_single:
            return distances[0]
        if dst_single:
            return distances[:, 0]
        return distances

    def distance_obs(self, src_obs, dst_obs) -> np.ndarray:
        """Encode observations and return directed TMD distances."""
        return self.distance_embeddings(self.encode(src_obs), self.encode(dst_obs))

    def make_direction_skill(self, obs_embed, goal_embed, eps=1e-10):
        """Return normalized Euclidean direction in TMD psi space."""
        direction = np.asarray(goal_embed) - np.asarray(obs_embed)
        return direction / (np.linalg.norm(direction, axis=-1, keepdims=True) + eps)


def pairwise_tmd_distance_matrix(agent, src_embeds, dst_embeds, batch_size, show_progress=True):
    """Compute D[i, j] = d_TMD(src_i, dst_j) in source chunks."""
    src_embeds = np.asarray(src_embeds, dtype=np.float32)
    dst_embeds = np.asarray(dst_embeds, dtype=np.float32)
    num_src = src_embeds.shape[0]
    num_dst = dst_embeds.shape[0]
    distance_matrix = np.zeros((num_src, num_dst), dtype=np.float32)
    for start in tqdm(
        range(0, num_src, int(batch_size)),
        desc="TMD pairwise distances",
        leave=False,
        disable=not show_progress,
    ):
        end = min(start + int(batch_size), num_src)
        chunk = src_embeds[start:end]
        dists = agent.get_tmd_distance_from_embeddings(chunk[:, None, :], dst_embeds[None, :, :])
        distance_matrix[start:end] = np.asarray(dists, dtype=np.float32)
    return distance_matrix
