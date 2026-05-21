import dataclasses
from typing import Any

import jax
import numpy as np
from tqdm import tqdm

from O_utils.datasets import Dataset, batched_random_crop


@dataclasses.dataclass
class TMDGASDataset:
    """Low-level GAS dataset with precomputed TMD psi features."""

    dataset: Dataset
    config: Any

    def __post_init__(self):
        self.size = self.dataset.size
        (self.terminal_locs,) = np.nonzero(self.dataset["terminals"] > 0)
        assert self.terminal_locs[-1] == self.size - 1
        self.provider = None
        self.edge_distance_threshold = None
        self.psi_obs = None
        self.psi_next_obs = None
        self.waysteps_idx = None

    def process_features(self, provider, edge_distance_threshold):
        self.provider = provider
        self.edge_distance_threshold = float(edge_distance_threshold)
        self.psi_obs = provider.encode(self.dataset["observations"])
        self.psi_next_obs = provider.encode(self.dataset["next_observations"])
        self.waysteps_idx = self.build_waysteps_idx_by_tmd_distance()

    def build_waysteps_idx_by_tmd_distance(self):
        all_idxs = np.arange(self.dataset["observations"].shape[0])
        all_final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, all_idxs)]
        waysteps_idx = np.zeros(len(self.psi_obs), dtype=np.int32)

        traj_starts = np.concatenate([[0], self.terminal_locs[:-1] + 1])
        for traj_start, traj_end in tqdm(
            zip(traj_starts, self.terminal_locs),
            total=len(self.terminal_locs),
            desc="Computing TMD waypoints",
        ):
            traj_embeds = self.psi_obs[traj_start : traj_end + 1]
            dist_matrix = self.provider.distance_embeddings(traj_embeds, traj_embeds)
            for local_i in range(len(traj_embeds)):
                row = dist_matrix[local_i, local_i:]
                idxs = np.where(row >= self.edge_distance_threshold)[0]
                local_j = int(local_i + idxs[0]) if len(idxs) > 0 else len(traj_embeds) - 1
                waysteps_idx[traj_start + local_i] = traj_start + local_j
        return waysteps_idx

    def augment(self, batch, keys):
        padding = 3
        batch_size = len(batch[keys[0]])
        crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate([crop_froms, np.zeros((batch_size, 1), dtype=np.int64)], axis=1)
        for key in keys:
            batch[key] = jax.tree_util.tree_map(
                lambda arr: np.array(batched_random_crop(arr, crop_froms, padding)) if len(arr.shape) == 4 else arr,
                batch[key],
            )

    def sample(self, batch_size: int, idxs=None, evaluation=False):
        if self.provider is None:
            raise RuntimeError("Call process_features(provider, edge_distance_threshold) before sampling.")
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)
        batch = self.dataset.sample(batch_size, idxs)

        actor_goal_idxs = self.waysteps_idx[idxs]
        offsets = np.random.geometric(p=1 - self.config["discount"], size=batch_size)
        actor_goal_idxs = np.minimum(idxs + offsets, actor_goal_idxs)

        if self.config["p_aug"] is not None and not evaluation and np.random.rand() < self.config["p_aug"]:
            batch["actor_goals"] = jax.tree_util.tree_map(lambda arr: arr[actor_goal_idxs], self.dataset["observations"])
            self.augment(batch, ["observations", "next_observations", "actor_goals"])
            batch["psi_obs"] = self.provider.encode(batch["observations"])
            batch["psi_next_obs"] = self.provider.encode(batch["next_observations"])
            batch["psi_actor_goals"] = self.provider.encode(batch["actor_goals"])
        else:
            batch["psi_obs"] = self.psi_obs[idxs]
            batch["psi_next_obs"] = self.psi_next_obs[idxs]
            batch["psi_actor_goals"] = self.psi_obs[actor_goal_idxs]

        batch["tmd_actor_dist"] = np.asarray(
            self.provider.agent.get_tmd_distance_from_embeddings(batch["psi_obs"], batch["psi_actor_goals"]),
            dtype=np.float32,
        )
        return batch
