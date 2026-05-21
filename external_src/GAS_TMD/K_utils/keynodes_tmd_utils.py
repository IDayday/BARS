import os
import pickle
from collections import defaultdict

import numpy as np
from tqdm import tqdm


class TMDKeyNodes:
    """TE-filtered, TMD-representation key nodes stored as dataset medoids."""

    def __init__(self):
        self.temporal_horizon_steps = None
        self.repr_cluster_threshold = None
        self.te_threshold = None
        self.efficiency_indices = None
        self.nodes = None
        self.node_indices = None
        self.node_observations = None
        self.cluster_members = None

    def construct_nodes(
        self,
        embeddings,
        observations,
        terminals,
        temporal_horizon_steps,
        repr_cluster_threshold,
        te_threshold,
    ):
        self.temporal_horizon_steps = int(temporal_horizon_steps)
        self.repr_cluster_threshold = float(repr_cluster_threshold)
        self.te_threshold = float(te_threshold)

        trajectories, start_indices = split_trajectories(np.asarray(embeddings), np.asarray(terminals))
        self.efficiency_indices = collect_efficiency_states(
            trajectories,
            start_indices,
            self.temporal_horizon_steps,
            self.repr_cluster_threshold,
            self.te_threshold,
        )
        if len(self.efficiency_indices) == 0:
            raise ValueError("TMD keynode construction produced no TE-efficient states.")

        self.nodes, self.node_indices, self.node_observations, self.cluster_members = tmd_medoid_clustering(
            np.asarray(embeddings),
            np.asarray(observations),
            np.asarray(self.efficiency_indices, dtype=np.int64),
            self.repr_cluster_threshold,
        )
        return self

    def save_keynodes(self, save_dir, filename="keynodes_tmd"):
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{filename}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(self.__dict__, f)
        print(f"[TMDKeyNodes] Saved to {save_path}")

    def load_keynodes(self, save_dir, filename="keynodes_tmd"):
        load_path = os.path.join(save_dir, f"{filename}.pkl")
        with open(load_path, "rb") as f:
            data = pickle.load(f)
        for k, v in data.items():
            setattr(self, k, v)
        print(f"[TMDKeyNodes] Loaded from {load_path}")
        return self


def split_trajectories(embeddings, terminals):
    trajectories = []
    start_indices = []
    start_idx = 0
    done_indices = np.where(terminals > 0)[0]
    for end_idx in tqdm(done_indices, desc="Splitting TMD trajectories"):
        trajectories.append(embeddings[start_idx : end_idx + 1])
        start_indices.append(start_idx)
        start_idx = end_idx + 1
    if start_idx < len(embeddings):
        trajectories.append(embeddings[start_idx:])
        start_indices.append(start_idx)
    return trajectories, start_indices


def collect_efficiency_states(trajectories, start_indices, temporal_horizon_steps, repr_cluster_threshold, te_threshold):
    global_efficiency_indices = []
    for traj_idx, traj in tqdm(
        enumerate(trajectories),
        desc="Collecting TMD TE states",
        total=len(trajectories),
    ):
        local = filter_low_efficiency_states(
            traj,
            temporal_horizon_steps,
            repr_cluster_threshold,
            te_threshold,
        )
        global_efficiency_indices.extend([start_indices[traj_idx] + idx for idx in local])
    if len(set(global_efficiency_indices)) != len(global_efficiency_indices):
        raise AssertionError("Duplicate TMD efficiency indices found.")
    return global_efficiency_indices


def filter_low_efficiency_states(traj, temporal_horizon_steps, repr_cluster_threshold, te_threshold):
    num_points = len(traj)
    indices = np.arange(num_points)
    keep = np.ones(num_points, dtype=bool)
    horizon = int(temporal_horizon_steps)
    for i in range(max(0, num_points - horizon)):
        obs_t = traj[i]
        obs_t_plus_step = traj[i + horizon]
        future = traj[i + 1 :]
        distances_future = np.linalg.norm(future - obs_t, axis=1)
        idxs_above = np.where(distances_future >= repr_cluster_threshold)[0]
        obs_t_plus_distance = future[idxs_above[0]] if len(idxs_above) > 0 else traj[-1]

        vector_step = obs_t_plus_step - obs_t
        vector_distance = obs_t_plus_distance - obs_t
        vector_step = vector_step / (np.linalg.norm(vector_step) + 1e-10)
        vector_distance = vector_distance / (np.linalg.norm(vector_distance) + 1e-10)

        cosine_similarity = float(np.dot(vector_step, vector_distance))
        if cosine_similarity < te_threshold:
            keep[i] = False
    return indices[keep]


def tmd_medoid_clustering(embeddings, observations, efficiency_indices, repr_cluster_threshold):
    min_dist = float(repr_cluster_threshold)
    f_s_sub = embeddings[efficiency_indices]
    stickers = np.zeros_like(f_s_sub)
    sticker_assignments = defaultdict(list)

    stickers[0] = f_s_sub[0]
    sticker_assignments[0].append(0)
    num_stickers = 1
    for i in tqdm(range(1, len(f_s_sub)), desc="TMD medoid clustering"):
        dists = np.linalg.norm(f_s_sub[i] - stickers[:num_stickers], axis=-1)
        min_idx = int(np.argmin(dists))
        if dists[min_idx] > min_dist:
            stickers[num_stickers] = f_s_sub[i]
            sticker_assignments[num_stickers].append(i)
            num_stickers += 1
        else:
            sticker_assignments[min_idx].append(i)

    node_embeds = []
    node_indices = []
    cluster_members = {}
    for s_idx, assigned_list in sticker_assignments.items():
        assigned_points = f_s_sub[assigned_list]
        cluster_mean = assigned_points.mean(axis=0)
        medoid_local = int(np.argmin(np.linalg.norm(assigned_points - cluster_mean, axis=-1)))
        medoid_global_idx = int(efficiency_indices[assigned_list[medoid_local]])
        node_embeds.append(embeddings[medoid_global_idx])
        node_indices.append(medoid_global_idx)
        cluster_members[int(s_idx)] = [int(efficiency_indices[i]) for i in assigned_list]

    node_embeds = np.asarray(node_embeds, dtype=np.float32)
    node_indices = np.asarray(node_indices, dtype=np.int64)
    node_observations = observations[node_indices]
    return node_embeds, node_indices, node_observations, cluster_members
