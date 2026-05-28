from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any

import networkx as nx
import numpy as np

from .keynodes_tmd import TMDKeyNodes


class TMDKeyGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.keynodes: TMDKeyNodes | None = None
        self.edge_distance_threshold = 0.0
        self.target_distance_threshold = 0.0
        self.target_nodes: dict[int, str] = {}
        self.paths_to_goal: dict[int, dict[Any, list[Any]]] = {}
        self.lengths_to_goal: dict[int, dict[Any, float]] = {}

    def construct_graph(self, keynodes: TMDKeyNodes, provider, edge_distance_threshold: float, pairwise_batch_size: int = 512, topk_l2_candidates: int | None = None) -> "TMDKeyGraph":
        self.keynodes = keynodes
        self.edge_distance_threshold = float(edge_distance_threshold)
        embeds = keynodes.embeds()
        for node in keynodes.nodes:
            self.graph.add_node(node.node_id, node=node, kind="key")
        if len(embeds) == 0:
            return self
        if topk_l2_candidates is None:
            D = provider.pairwise_distance(embeds, embeds, batch_size=pairwise_batch_size)
            for i in range(D.shape[0]):
                for j in range(D.shape[1]):
                    if i != j and float(D[i, j]) <= self.edge_distance_threshold:
                        self.graph.add_edge(i, j, weight=float(D[i, j]), edge_type="tmd")
        else:
            l2 = np.linalg.norm(embeds[:, None, :] - embeds[None, :, :], axis=-1)
            k = int(max(1, min(topk_l2_candidates, len(embeds) - 1)))
            for i in range(len(embeds)):
                cand = np.argsort(l2[i])[: k + 1]
                cand = cand[cand != i]
                d = provider.distance_embeddings(embeds[i : i + 1], embeds[cand]).reshape(-1)
                for j, dist in zip(cand, d):
                    if float(dist) <= self.edge_distance_threshold:
                        self.graph.add_edge(int(i), int(j), weight=float(dist), edge_type="tmd")
        return self

    def add_task_goal(self, task_id: int, goal_obs: np.ndarray, provider, target_distance_threshold: float):
        if self.keynodes is None:
            raise RuntimeError("construct_graph must be called first")
        self.target_distance_threshold = float(target_distance_threshold)
        target = f"target:{int(task_id)}"
        goal_embed = provider.encode(np.asarray(goal_obs, dtype=np.float32))
        self.target_nodes[int(task_id)] = target
        self.graph.add_node(target, kind="target", task_id=int(task_id), embed=goal_embed, raw_obs=np.asarray(goal_obs, dtype=np.float32))
        embeds = self.keynodes.embeds()
        if len(embeds) == 0:
            return
        forward = provider.distance_embeddings(embeds, goal_embed[None, :]).reshape(-1)
        backward = provider.distance_embeddings(goal_embed[None, :], embeds).reshape(-1)
        for i, dist in enumerate(forward):
            if float(dist) <= self.target_distance_threshold:
                self.graph.add_edge(int(i), target, weight=float(dist), edge_type="target_forward")
        for i, dist in enumerate(backward):
            if float(dist) <= self.target_distance_threshold:
                self.graph.add_edge(target, int(i), weight=float(dist), edge_type="target_backward")

    def precompute_shortest_paths_to_goal(self, task_id: int):
        target = self.target_nodes[int(task_id)]
        lengths, paths = nx.single_source_dijkstra(self.graph.reverse(copy=False), source=target, weight="weight")
        self.paths_to_goal[int(task_id)] = {node: list(reversed(path)) for node, path in paths.items()}
        self.lengths_to_goal[int(task_id)] = {node: float(v) for node, v in lengths.items()}

    def get_shortest_path(self, task_id: int, source_embed: np.ndarray, provider, source_threshold: float):
        if self.keynodes is None:
            raise RuntimeError("construct_graph must be called first")
        task_id = int(task_id)
        if task_id not in self.paths_to_goal:
            self.precompute_shortest_paths_to_goal(task_id)
        embeds = self.keynodes.embeds()
        if len(embeds) == 0:
            return {"no_path": True, "node_ids": [], "embeds": np.empty((0, 0)), "raw_obs": np.empty((0, 0))}
        d = provider.distance_embeddings(np.asarray(source_embed, dtype=np.float32)[None, :], embeds).reshape(-1)
        candidates = np.flatnonzero(d <= float(source_threshold))
        if len(candidates) == 0:
            candidates = np.asarray([int(np.argmin(d))], dtype=np.int64)
        best = None
        for cand in candidates:
            cand = int(cand)
            if cand not in self.paths_to_goal[task_id]:
                continue
            score = float(d[cand]) + float(self.lengths_to_goal[task_id].get(cand, np.inf))
            if best is None or score < best[0]:
                best = (score, cand)
        if best is None:
            return {"no_path": True, "node_ids": [], "embeds": np.empty((0, embeds.shape[1])), "raw_obs": np.empty((0, self.keynodes.raw_observations().shape[1]))}
        path = [x for x in self.paths_to_goal[task_id][best[1]] if isinstance(x, int)]
        nodes = [self.keynodes.nodes[int(i)] for i in path]
        return {"no_path": False, "node_ids": [n.node_id for n in nodes], "embeds": np.stack([n.embed for n in nodes]).astype(np.float32) if nodes else np.empty((0, embeds.shape[1])), "raw_obs": np.stack([n.raw_obs for n in nodes]).astype(np.float32) if nodes else np.empty((0, self.keynodes.raw_observations().shape[1]))}

    def graph_stats(self) -> dict[str, float | int]:
        key_nodes = [node for node, data in self.graph.nodes(data=True) if data.get("kind") == "key"]
        out_degrees = np.asarray([self.graph.out_degree(node) for node in key_nodes], dtype=np.float64)
        edge_weights = np.asarray([data.get("weight", 0.0) for _, _, data in self.graph.edges(data=True)], dtype=np.float64)
        sccs = list(nx.strongly_connected_components(self.graph.subgraph(key_nodes).copy())) if key_nodes else []
        asym = []
        for u, v, data in self.graph.edges(data=True):
            if self.graph.has_edge(v, u):
                asym.append(abs(float(data.get("weight", 0.0)) - float(self.graph[v][u].get("weight", 0.0))))
        return {"num_nodes": int(len(key_nodes)), "num_edges": int(self.graph.number_of_edges()), "mean_out_degree": float(out_degrees.mean()) if out_degrees.size else 0.0, "median_out_degree": float(np.median(out_degrees)) if out_degrees.size else 0.0, "scc_count": int(len(sccs)), "largest_scc_ratio": float(max((len(s) for s in sccs), default=0) / max(len(key_nodes), 1)), "edge_distance_threshold": float(self.edge_distance_threshold), "target_distance_threshold": float(self.target_distance_threshold), "mean_edge_weight": float(edge_weights.mean()) if edge_weights.size else 0.0, "edge_weight_q50": float(np.quantile(edge_weights, 0.5)) if edge_weights.size else 0.0, "edge_weight_q90": float(np.quantile(edge_weights, 0.9)) if edge_weights.size else 0.0, "directed_asymmetry_mean": float(np.mean(asym)) if asym else 0.0}

    def save(self, path: str | Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f: pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "TMDKeyGraph":
        with open(path, "rb") as f: return pickle.load(f)
