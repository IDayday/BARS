from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import numpy as np

from .config import AdaptiveConfig
from .edge_features import EdgeFeatureExtractor
from .exec_calibrator import ExecutionCalibrator
from .graph import GraphData


def shortest_path(graph: GraphData, start: int, goal: int) -> tuple[list[int], float, list[int]]:
    """Dijkstra shortest path over GraphData.

    Returns:
        path_nodes, total_cost, path_edge_ids
    """
    n = graph.num_nodes
    start = int(start)
    goal = int(goal)
    if not (0 <= start < n and 0 <= goal < n):
        raise IndexError(f"start/goal out of range: start={start}, goal={goal}, n={n}")
    if start == goal:
        return [start], 0.0, []

    adj = graph.adjacency()
    dist = np.full(n, np.inf, dtype=np.float64)
    prev = np.full(n, -1, dtype=np.int64)
    prev_eid = np.full(n, -1, dtype=np.int64)
    dist[start] = 0.0
    heap = [(0.0, start)]
    seen = np.zeros(n, dtype=bool)
    while heap:
        d, u = heapq.heappop(heap)
        if seen[u]:
            continue
        seen[u] = True
        if u == goal:
            break
        for v, c, eid in adj[u]:
            nd = d + c
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                prev_eid[v] = eid
                heapq.heappush(heap, (nd, v))

    if not np.isfinite(dist[goal]):
        return [], float("inf"), []

    nodes = []
    eids = []
    cur = goal
    while cur != -1:
        nodes.append(int(cur))
        eid = int(prev_eid[cur])
        if eid >= 0:
            eids.append(eid)
        if cur == start:
            break
        cur = int(prev[cur])
    nodes.reverse()
    eids.reverse()
    return nodes, float(dist[goal]), eids


def shortest_path_to_any(graph: GraphData, start: int, goals: Iterable[int]) -> tuple[list[int], float, list[int], Optional[int]]:
    best_path: list[int] = []
    best_cost = float("inf")
    best_eids: list[int] = []
    best_goal: Optional[int] = None
    for g in goals:
        path, cost, eids = shortest_path(graph, start, int(g))
        if path and cost < best_cost:
            best_path, best_cost, best_eids, best_goal = path, cost, eids, int(g)
    return best_path, best_cost, best_eids, best_goal


@dataclass
class AdaptiveSelection:
    subgoal_node: int
    path_index: int
    reason: str
    predicted_distance: float
    p_exec: float


class AdaptiveWaypointSelector:
    """TTGS-style adaptive subgoal selection for GAS paths.

    At each execution step, choose the farthest future node on the current path
    that is predicted to be reachable. The progress index is monotone by default.
    """

    def __init__(
        self,
        graph: GraphData,
        cfg: AdaptiveConfig,
        dataset_extractor: Optional[EdgeFeatureExtractor] = None,
        calibrator: Optional[ExecutionCalibrator] = None,
        state_to_feature: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        self.graph = graph
        self.cfg = cfg
        self.extractor = dataset_extractor
        self.calibrator = calibrator
        self.state_to_feature = state_to_feature
        self.progress_index = 0

    def reset(self) -> None:
        self.progress_index = 0

    def _fallback_distance(self, current_state: np.ndarray, node_id: int) -> float:
        current_state = np.asarray(current_state, dtype=np.float32)
        target = self.graph.states[int(node_id)]
        dim = min(len(current_state), len(target))
        return float(np.sqrt(np.sum((current_state[:dim] - target[:dim]) ** 2)))

    def _predict_pair(self, current_state: np.ndarray, node_id: int) -> tuple[float, float]:
        # If no online feature adapter is supplied, use conservative state-space distance.
        if self.calibrator is None or self.extractor is None or self.state_to_feature is None:
            d = self._fallback_distance(current_state, node_id)
            p = float(np.exp(-d / max(self.cfg.reach_budget, 1e-6)))
            return d, p

        # Create a temporary source feature is project-specific. The hook maps raw
        # current obs/state to the same representation used by the dataset. Most
        # projects will override this with the existing GAS encoder.
        # For a generic reference implementation, we estimate reachability by
        # comparing the hook output with graph states and use a distance-derived p.
        feat = np.asarray(self.state_to_feature(current_state), dtype=np.float32)
        target = self.graph.states[int(node_id)]
        dim = min(len(feat), len(target))
        d = float(np.sqrt(np.sum((feat[:dim] - target[:dim]) ** 2)))
        p = float(np.exp(-d / max(self.cfg.reach_budget, 1e-6)))
        return d, p

    def select(self, current_state: np.ndarray, path: list[int]) -> AdaptiveSelection:
        if not path:
            raise ValueError("path is empty")
        if self.progress_index >= len(path):
            self.progress_index = len(path) - 1

        start_idx = self.progress_index if self.cfg.monotonic_progress else 0
        end_idx = len(path) - 1
        if self.cfg.max_skip is not None:
            end_idx = min(end_idx, start_idx + self.cfg.max_skip)

        best: Optional[AdaptiveSelection] = None
        # Farthest-first selection.
        for k in range(end_idx, start_idx - 1, -1):
            node_id = int(path[k])
            d, p = self._predict_pair(current_state, node_id)
            if d <= self.cfg.reach_budget and p >= self.cfg.min_p_exec:
                best = AdaptiveSelection(node_id, k, "farthest_reachable", d, p)
                break

        if best is None:
            if self.cfg.fallback_to_next:
                k = min(start_idx + 1, len(path) - 1)
                node_id = int(path[k])
                d, p = self._predict_pair(current_state, node_id)
                best = AdaptiveSelection(node_id, k, "fallback_next", d, p)
            else:
                node_id = int(path[start_idx])
                d, p = self._predict_pair(current_state, node_id)
                best = AdaptiveSelection(node_id, start_idx, "no_reachable_keep_current", d, p)

        if self.cfg.monotonic_progress:
            self.progress_index = max(self.progress_index, best.path_index)
        return best


def nearest_graph_node_by_state(graph: GraphData, state: np.ndarray) -> int:
    state = np.asarray(state, dtype=np.float32)
    dim = min(state.shape[-1], graph.states.shape[-1])
    d2 = np.sum((graph.states[:, :dim] - state[:dim]) ** 2, axis=1)
    return int(np.argmin(d2))


def plan_from_states(graph: GraphData, start_state: np.ndarray, goal_state: np.ndarray) -> tuple[list[int], float, list[int]]:
    s = nearest_graph_node_by_state(graph, start_state)
    g = nearest_graph_node_by_state(graph, goal_state)
    return shortest_path(graph, s, g)
