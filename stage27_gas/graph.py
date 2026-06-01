from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


@dataclass
class GraphData:
    """Directed weighted graph over selected offline dataset nodes."""

    node_indices: np.ndarray  # graph node -> dataset row index
    states: np.ndarray
    traj_ids: np.ndarray
    time_idxs: np.ndarray
    edges_src: np.ndarray
    edges_dst: np.ndarray
    edge_costs: np.ndarray
    edge_features: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.node_indices = np.asarray(self.node_indices, dtype=np.int64)
        self.states = np.asarray(self.states, dtype=np.float32)
        self.traj_ids = np.asarray(self.traj_ids)
        self.time_idxs = np.asarray(self.time_idxs)
        self.edges_src = np.asarray(self.edges_src, dtype=np.int64)
        self.edges_dst = np.asarray(self.edges_dst, dtype=np.int64)
        self.edge_costs = np.asarray(self.edge_costs, dtype=np.float32)
        if len(self.edges_src) != len(self.edges_dst) or len(self.edges_src) != len(self.edge_costs):
            raise ValueError("edge arrays must have the same length")
        n = len(self.node_indices)
        if self.states.shape[0] != n or len(self.traj_ids) != n or len(self.time_idxs) != n:
            raise ValueError("node arrays must have the same leading dimension")
        for k, v in self.edge_features.items():
            arr = np.asarray(v)
            if arr.shape[0] != len(self.edges_src):
                raise ValueError(f"edge_features[{k}] has leading {arr.shape[0]}, expected {len(self.edges_src)}")
            self.edge_features[k] = arr

    @property
    def num_nodes(self) -> int:
        return int(len(self.node_indices))

    @property
    def num_edges(self) -> int:
        return int(len(self.edge_costs))

    def adjacency(self) -> list[list[tuple[int, float, int]]]:
        adj: list[list[tuple[int, float, int]]] = [[] for _ in range(self.num_nodes)]
        for eid, (s, d, c) in enumerate(zip(self.edges_src, self.edges_dst, self.edge_costs)):
            adj[int(s)].append((int(d), float(c), int(eid)))
        return adj

    def edge_lookup(self) -> dict[tuple[int, int], int]:
        return {(int(s), int(d)): eid for eid, (s, d) in enumerate(zip(self.edges_src, self.edges_dst))}

    def nearest_node(self, state: np.ndarray, embedding: Optional[np.ndarray] = None) -> int:
        x = self.states if embedding is None else np.asarray(embedding, dtype=np.float32)
        state = np.asarray(state, dtype=np.float32)
        d2 = np.sum((x - state) ** 2, axis=1)
        return int(np.argmin(d2))

    def path_edge_ids(self, path: Iterable[int]) -> list[int]:
        lookup = self.edge_lookup()
        nodes = list(path)
        out = []
        for a, b in zip(nodes[:-1], nodes[1:]):
            eid = lookup.get((int(a), int(b)))
            if eid is None:
                raise KeyError(f"Path contains missing edge {a}->{b}")
            out.append(eid)
        return out

    def to_npz(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta_items = np.array([self.metadata], dtype=object)
        np.savez_compressed(
            path,
            node_indices=self.node_indices,
            states=self.states,
            traj_ids=self.traj_ids,
            time_idxs=self.time_idxs,
            edges_src=self.edges_src,
            edges_dst=self.edges_dst,
            edge_costs=self.edge_costs,
            metadata=meta_items,
            **{f"edgefeat__{k}": v for k, v in self.edge_features.items()},
        )

    @staticmethod
    def from_npz(path: str | Path) -> "GraphData":
        data = np.load(path, allow_pickle=True)
        edge_features = {k.replace("edgefeat__", "", 1): data[k] for k in data.files if k.startswith("edgefeat__")}
        metadata = {}
        if "metadata" in data.files:
            raw = data["metadata"]
            if len(raw) > 0:
                metadata = dict(raw[0])
        return GraphData(
            node_indices=data["node_indices"],
            states=data["states"],
            traj_ids=data["traj_ids"],
            time_idxs=data["time_idxs"],
            edges_src=data["edges_src"],
            edges_dst=data["edges_dst"],
            edge_costs=data["edge_costs"],
            edge_features=edge_features,
            metadata=metadata,
        )
