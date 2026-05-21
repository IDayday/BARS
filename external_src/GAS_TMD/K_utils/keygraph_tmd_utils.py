import csv
import os
import pickle

import networkx as nx
import numpy as np
from tqdm import tqdm


class TMDKeyGraph:
    """Directed TMD quasimetric graph over medoid keynodes."""

    def __init__(self):
        self.nodes = None
        self.node_indices = None
        self.node_observations = None
        self.graph = None
        self.base_node_cnt = 0
        self.edge_distance_threshold = None
        self.target_distance_threshold = None
        self.task_goal_dict = {}
        self.task_node_dict = {}
        self.task_node_idx_dict = {}
        self.task_paths_dict = {}
        self.task_paths_dist_dict = {}
        self.graph_stats = {}

    def construct_graph(self, key_nodes, provider, edge_distance_threshold, batch_size, topk=None):
        self.nodes = np.asarray(key_nodes.nodes, dtype=np.float32)
        self.node_indices = np.asarray(key_nodes.node_indices, dtype=np.int64)
        self.node_observations = np.asarray(key_nodes.node_observations)
        self.base_node_cnt = len(self.nodes)
        self.edge_distance_threshold = float(edge_distance_threshold)

        graph = nx.DiGraph()
        for node_idx, node_embed in enumerate(self.nodes):
            graph.add_node(node_idx, pos=node_embed, dataset_idx=int(self.node_indices[node_idx]))

        if topk is not None:
            dist_matrix = score_topk_candidates(self.nodes, provider, int(topk), int(batch_size))
        else:
            dist_matrix = provider.distance_embeddings(self.nodes, self.nodes, batch_size=int(batch_size))
        np.fill_diagonal(dist_matrix, np.inf)

        for i, row in tqdm(enumerate(dist_matrix), total=len(dist_matrix), desc="Adding TMD directed edges"):
            neighbors = np.where(row <= self.edge_distance_threshold)[0]
            for j in neighbors:
                graph.add_edge(i, int(j), weight=float(row[j]), tmd_dist=float(row[j]))

        self.graph = graph
        self.graph_stats = compute_graph_stats(
            self.graph,
            self.nodes,
            dist_matrix,
            self.edge_distance_threshold,
            self.target_distance_threshold,
        )
        return self

    def precompute_shortest_paths_to_all_tasks(self, task_goal_dict, task_node_dict, task_obs_dict, provider, target_distance_threshold):
        self.clear_task_goals()
        self.target_distance_threshold = float(target_distance_threshold)
        self.task_goal_dict = dict(task_goal_dict)
        self.task_node_dict = dict(task_node_dict)

        for task_id, target_embed in tqdm(self.task_node_dict.items(), desc="Adding TMD task goals"):
            target_idx = self.graph.number_of_nodes()
            target_obs = task_obs_dict[task_id]
            self.graph, self.nodes, self.node_indices, self.node_observations = add_target_node_directed(
                self.graph,
                self.nodes,
                self.node_indices,
                self.node_observations,
                np.asarray(target_embed, dtype=np.float32),
                np.asarray(target_obs),
                target_idx,
                provider,
                self.target_distance_threshold,
                self.base_node_cnt,
            )
            self.task_node_idx_dict[task_id] = target_idx
            paths, distances = compute_shortest_paths_to_target_directed(self.graph, target_idx)
            self.task_paths_dict[task_id] = paths
            self.task_paths_dist_dict[task_id] = distances

        self.graph_stats = compute_graph_stats(
            self.graph,
            self.nodes[: self.base_node_cnt],
            None,
            self.edge_distance_threshold,
            self.target_distance_threshold,
            self.task_paths_dict,
        )
        return self

    def clear_task_goals(self):
        for idx in sorted(self.task_node_idx_dict.values(), reverse=True):
            if self.graph is not None and self.graph.has_node(idx):
                self.graph.remove_node(idx)
        if self.nodes is not None:
            self.nodes = self.nodes[: self.base_node_cnt]
        if self.node_indices is not None:
            self.node_indices = self.node_indices[: self.base_node_cnt]
        if self.node_observations is not None:
            self.node_observations = self.node_observations[: self.base_node_cnt]
        self.task_goal_dict.clear()
        self.task_node_dict.clear()
        self.task_node_idx_dict.clear()
        self.task_paths_dict.clear()
        self.task_paths_dist_dict.clear()

    def get_shortest_path(
        self,
        task_id,
        source_embed,
        provider=None,
        force_closest=False,
        edge_distance_threshold=None,
    ):
        if task_id not in self.task_paths_dict:
            return None
        shortest_paths = self.task_paths_dict[task_id]
        shortest_paths_dist = self.task_paths_dist_dict[task_id]
        if len(shortest_paths) == 0:
            return None

        sp_keys = np.asarray(list(shortest_paths.keys()), dtype=np.int64)
        candidate_embeds = self.nodes[sp_keys]
        if provider is None:
            start_distances = np.linalg.norm(candidate_embeds - np.asarray(source_embed), axis=-1)
        else:
            start_distances = provider.distance_embeddings(np.asarray(source_embed)[None], candidate_embeds)[0]

        threshold = float(edge_distance_threshold or self.edge_distance_threshold)
        valid_indices = np.where(start_distances <= threshold)[0]
        if len(valid_indices) == 0:
            if not force_closest:
                return None
            valid_indices = np.asarray([int(np.argmin(start_distances))])

        best_total_distance = float("inf")
        best_path = None
        for idx in valid_indices:
            path_key = int(sp_keys[idx])
            total_distance = float(start_distances[idx]) + float(shortest_paths_dist[path_key])
            if total_distance < best_total_distance:
                best_total_distance = total_distance
                best_path = shortest_paths[path_key]
        if best_path is None:
            return None
        path_indices = np.asarray(best_path, dtype=np.int64)
        return {
            "path_indices": path_indices,
            "path_embeds": self.nodes[path_indices],
            "path_observations": self.node_observations[path_indices],
            "path_weight": float(best_total_distance),
        }

    def save_keygraph(self, save_dir, filename="keygraph_tmd"):
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{filename}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(self.__dict__, f)
        print(f"[TMDKeyGraph] Saved to {save_path}")

    def load_keygraph(self, save_dir, filename="keygraph_tmd"):
        load_path = os.path.join(save_dir, f"{filename}.pkl")
        with open(load_path, "rb") as f:
            data = pickle.load(f)
        for k, v in data.items():
            setattr(self, k, v)
        print(f"[TMDKeyGraph] Loaded from {load_path}")
        return self

    def save_graph_stats(self, save_dir, filename="graph_stats.csv"):
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(self.graph_stats.keys()))
            writer.writeheader()
            writer.writerow(self.graph_stats)
        return path


def score_topk_candidates(nodes, provider, topk, batch_size):
    l2 = np.linalg.norm(nodes[:, None, :] - nodes[None, :, :], axis=-1)
    np.fill_diagonal(l2, np.inf)
    candidates = np.argsort(l2, axis=1)[:, :topk]
    dist_matrix = np.full((len(nodes), len(nodes)), np.inf, dtype=np.float32)
    for i in tqdm(range(len(nodes)), desc="Scoring top-k TMD candidates"):
        dists = provider.distance_embeddings(nodes[i : i + 1], nodes[candidates[i]], batch_size=batch_size)[0]
        dist_matrix[i, candidates[i]] = dists
    return dist_matrix


def add_target_node_directed(
    graph,
    nodes,
    node_indices,
    node_observations,
    target_embed,
    target_obs,
    target_idx,
    provider,
    target_distance_threshold,
    base_node_cnt,
):
    graph.add_node(target_idx, pos=target_embed, dataset_idx=-1)
    base_nodes = nodes[:base_node_cnt]
    to_target = provider.distance_embeddings(base_nodes, target_embed[None])[:, 0]
    from_target = provider.distance_embeddings(target_embed[None], base_nodes)[0]

    for i, dist in enumerate(to_target):
        if dist <= target_distance_threshold:
            graph.add_edge(i, target_idx, weight=float(dist), tmd_dist=float(dist))
    for i, dist in enumerate(from_target):
        if dist <= target_distance_threshold:
            graph.add_edge(target_idx, i, weight=float(dist), tmd_dist=float(dist))

    nodes = np.vstack([nodes, target_embed[None]])
    node_indices = np.concatenate([node_indices, np.asarray([-1], dtype=node_indices.dtype)])
    node_observations = np.concatenate([node_observations, target_obs[None]], axis=0)
    return graph, nodes, node_indices, node_observations


def compute_shortest_paths_to_target_directed(graph, target_idx):
    rev = graph.reverse(copy=False)
    lengths, paths = nx.single_source_dijkstra(rev, source=target_idx, weight="weight")
    shortest_paths = {}
    shortest_paths_dist = {}
    for node_idx, rev_path in paths.items():
        if node_idx == target_idx:
            continue
        shortest_paths[int(node_idx)] = [int(x) for x in rev_path[::-1]]
        shortest_paths_dist[int(node_idx)] = float(lengths[node_idx])
    return shortest_paths, shortest_paths_dist


def compute_graph_stats(graph, base_nodes, dist_matrix, edge_threshold, target_threshold, task_paths_dict=None):
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    out_degrees = np.asarray([deg for _, deg in graph.out_degree()], dtype=np.float32)
    weights = np.asarray([data.get("weight", np.nan) for _, _, data in graph.edges(data=True)], dtype=np.float32)
    weights = weights[np.isfinite(weights)]

    stats = {
        "num_nodes": int(num_nodes),
        "num_edges": int(num_edges),
        "edge_threshold": float(edge_threshold) if edge_threshold is not None else np.nan,
        "target_threshold": float(target_threshold) if target_threshold is not None else np.nan,
        "scc_count": int(nx.number_strongly_connected_components(graph)) if num_nodes else 0,
        "mean_out_degree": float(out_degrees.mean()) if len(out_degrees) else 0.0,
        "mean_edge_weight": float(weights.mean()) if len(weights) else np.nan,
        "edge_dist_q50": float(np.quantile(weights, 0.5)) if len(weights) else np.nan,
        "edge_dist_q90": float(np.quantile(weights, 0.9)) if len(weights) else np.nan,
    }
    if dist_matrix is not None and dist_matrix.size:
        finite = np.isfinite(dist_matrix) & np.isfinite(dist_matrix.T)
        finite &= ~np.eye(dist_matrix.shape[0], dtype=bool)
        asym = np.abs(dist_matrix[finite] - dist_matrix.T[finite])
        stats["cross_direction_asymmetry_mean"] = float(asym.mean()) if len(asym) else np.nan
        stats["tmd_asymmetry_mean"] = stats["cross_direction_asymmetry_mean"]
    else:
        stats["cross_direction_asymmetry_mean"] = np.nan
        stats["tmd_asymmetry_mean"] = np.nan

    if task_paths_dict:
        total = 0
        no_path = 0
        path_lens = []
        for paths in task_paths_dict.values():
            total += len(base_nodes)
            no_path += len(base_nodes) - len(paths)
            path_lens.extend([len(path) for path in paths.values()])
        stats["no_path_rate"] = float(no_path / total) if total else np.nan
        stats["mean_path_len"] = float(np.mean(path_lens)) if path_lens else np.nan
    else:
        stats["no_path_rate"] = np.nan
        stats["mean_path_len"] = np.nan
    return stats
