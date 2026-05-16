from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .boundary import BoundaryIndex
from .types import BARSGraph


@dataclass
class PlanResult:
    found: bool
    node_path: List[int]
    edge_path: List[int]
    total_cost: float
    total_risk: float
    total_boundary: float
    objective: float
    variant: str
    exec_cost: float = 0.0
    exec_budget: float = float("nan")

    def to_row(self) -> Dict[str, float | int | str]:
        return {
            "found": int(self.found),
            "num_subgoals": max(0, len(self.node_path) - 2),
            "num_edges": len(self.edge_path),
            "total_cost": self.total_cost,
            "total_risk": self.total_risk,
            "total_boundary": self.total_boundary,
            "exec_cost": self.exec_cost,
            "exec_budget": self.exec_budget,
            "objective": self.objective,
            "variant": self.variant,
        }


def _empty(variant: str, exec_budget: Optional[float] = None) -> PlanResult:
    return PlanResult(
        False,
        [],
        [],
        float("inf"),
        float("inf"),
        float("inf"),
        float("inf"),
        variant,
        float("inf"),
        float(exec_budget) if exec_budget is not None else float("nan"),
    )


def plan_path(
    graph: BARSGraph,
    start_node: int,
    goal_node: int,
    variant: str = "full_bars",
    lambda_risk: float = 1.0,
    lambda_boundary: float = 1.0,
    boundary: Optional[BoundaryIndex] = None,
    exec_budget: Optional[float] = None,
    max_edges: Optional[int] = None,
    max_labels_per_edge: int = 32,
) -> PlanResult:
    """Plan a path on a BARS graph.

    Variants:
      shortest/gas/tdr_shortest:
        Pure temporal/progress cost.
      reachability/bars_lite/risk:
        Lagrangian node-level cost: c(e) + lambda_risk r(e).
      full_bars/bars/boundary/gas_bars:
        Lagrangian line-graph cost with edge-to-edge boundary penalty.
      constrained_bars/budget_bars/bars_budget/full_bars_budget:
        Constrained line-graph search: minimize temporal cost subject to
        sum r(e) + lambda_boundary * sum b(e_i,e_j) <= exec_budget.

    The constrained variant is the closest implementation of the BARS design
    objective. It is intentionally exposed as a separate variant so Stage19
    results and previous Lagrangian baselines remain reproducible.
    """
    variant = str(variant).lower()
    start_node = int(start_node)
    goal_node = int(goal_node)
    if start_node == goal_node:
        return PlanResult(True, [start_node], [], 0.0, 0.0, 0.0, 0.0, variant, 0.0, float(exec_budget) if exec_budget is not None else float("nan"))

    if variant in {"shortest", "gas", "tdr_shortest"}:
        return _node_dijkstra(graph, start_node, goal_node, 0.0, variant, max_edges=max_edges)
    if variant in {"reachability", "bars_lite", "risk"}:
        return _node_dijkstra(graph, start_node, goal_node, lambda_risk, variant, max_edges=max_edges)
    if variant in {"boundary", "full_bars", "bars", "gas_bars", "gas_bars_lagrangian"}:
        if boundary is None:
            return _node_dijkstra(graph, start_node, goal_node, lambda_risk, "reachability_no_boundary", max_edges=max_edges)
        return _line_graph_dijkstra(graph, start_node, goal_node, lambda_risk, lambda_boundary, boundary, variant, max_edges=max_edges)
    if variant in {"constrained_bars", "budget_bars", "bars_budget", "full_bars_budget"}:
        budget = float(8.0 if exec_budget is None else exec_budget)
        return _constrained_line_graph_search(
            graph,
            start_node,
            goal_node,
            exec_budget=budget,
            lambda_boundary=lambda_boundary,
            boundary=boundary,
            variant=variant,
            max_edges=max_edges,
            max_labels_per_edge=max_labels_per_edge,
        )
    raise ValueError(f"Unknown planner variant: {variant}")


def _edge_weight(graph: BARSGraph, eid: int, lambda_risk: float) -> float:
    return float(graph.cost[eid] + lambda_risk * graph.risk[eid])


def _edge_weights(graph: BARSGraph, lambda_risk: float) -> np.ndarray:
    return (graph.cost + float(lambda_risk) * graph.risk).astype(np.float64, copy=False)


def _boundary_costs(boundary: BoundaryIndex, prev_edge: int, next_edges: np.ndarray) -> np.ndarray:
    if len(next_edges) == 0:
        return np.empty(0, dtype=np.float32)
    if hasattr(boundary, "transition_costs"):
        return boundary.transition_costs(int(prev_edge), next_edges)
    if hasattr(boundary, "boundary_cost_batch"):
        return boundary.boundary_cost_batch(np.full(len(next_edges), int(prev_edge), dtype=np.int64), next_edges)
    return np.asarray([boundary.boundary_cost(int(prev_edge), int(ne)) for ne in next_edges], dtype=np.float32)


def _node_dijkstra(
    graph: BARSGraph,
    start_node: int,
    goal_node: int,
    lambda_risk: float,
    variant: str,
    max_edges: Optional[int] = None,
) -> PlanResult:
    out = graph.outgoing_edges()
    n = graph.num_nodes
    max_hops = int(max_edges) if max_edges is not None and int(max_edges) > 0 else None
    edge_w = _edge_weights(graph, lambda_risk)

    # If max_edges is set, use expanded-state Dijkstra over (node, depth).
    if max_hops is not None:
        dist: Dict[tuple[int, int], float] = {(start_node, 0): 0.0}
        prev: Dict[tuple[int, int], tuple[tuple[int, int], int]] = {}
        pq: list[tuple[float, int, int]] = [(0.0, start_node, 0)]
        best_state: Optional[tuple[int, int]] = None
        while pq:
            d, u, depth = heapq.heappop(pq)
            if d != dist.get((u, depth), float("inf")):
                continue
            if u == goal_node and depth > 0:
                best_state = (u, depth)
                break
            if depth >= max_hops:
                continue
            for eid in out[u]:
                eid = int(eid)
                v = int(graph.dst[eid])
                nd = d + float(edge_w[eid])
                st = (v, depth + 1)
                if nd < dist.get(st, float("inf")):
                    dist[st] = nd
                    prev[st] = ((u, depth), eid)
                    heapq.heappush(pq, (nd, v, depth + 1))
        if best_state is None:
            return _empty(variant)
        edges: list[int] = []
        nodes: list[int] = [goal_node]
        cur = best_state
        while cur != (start_node, 0):
            pstate, eid = prev[cur]
            edges.append(int(eid))
            nodes.append(int(pstate[0]))
            cur = pstate
        nodes.reverse()
        edges.reverse()
        total_cost = float(graph.cost[edges].sum()) if edges else 0.0
        total_risk = float(graph.risk[edges].sum()) if edges else 0.0
        return PlanResult(True, nodes, edges, total_cost, total_risk, 0.0, float(dist[best_state]), variant, total_risk, float("nan"))

    dist_arr = np.full(n, np.inf)
    prev_node = np.full(n, -1, dtype=np.int64)
    prev_edge = np.full(n, -1, dtype=np.int64)
    dist_arr[start_node] = 0.0
    pq = [(0.0, start_node)]
    while pq:
        d, u = heapq.heappop(pq)
        if d != dist_arr[u]:
            continue
        if u == goal_node:
            break
        for eid in out[u]:
            eid = int(eid)
            v = int(graph.dst[eid])
            nd = d + float(edge_w[eid])
            if nd < dist_arr[v]:
                dist_arr[v] = nd
                prev_node[v] = u
                prev_edge[v] = eid
                heapq.heappush(pq, (nd, v))
    if not np.isfinite(dist_arr[goal_node]):
        return _empty(variant)
    edges: list[int] = []
    nodes: list[int] = []
    cur = goal_node
    nodes.append(cur)
    while cur != start_node:
        eid = int(prev_edge[cur])
        edges.append(eid)
        cur = int(prev_node[cur])
        nodes.append(cur)
    nodes.reverse()
    edges.reverse()
    total_cost = float(graph.cost[edges].sum()) if edges else 0.0
    total_risk = float(graph.risk[edges].sum()) if edges else 0.0
    return PlanResult(True, nodes, edges, total_cost, total_risk, 0.0, float(dist_arr[goal_node]), variant, total_risk, float("nan"))


def _line_graph_dijkstra(
    graph: BARSGraph,
    start_node: int,
    goal_node: int,
    lambda_risk: float,
    lambda_boundary: float,
    boundary: BoundaryIndex,
    variant: str,
    max_edges: Optional[int] = None,
) -> PlanResult:
    out = graph.outgoing_edges()
    start_edges = out[start_node]
    if len(start_edges) == 0:
        return _empty(variant)
    max_hops = int(max_edges) if max_edges is not None and int(max_edges) > 0 else None
    m = graph.num_edges
    edge_w = _edge_weights(graph, lambda_risk)
    # Expanded depth when max_edges is set; otherwise original compact version.
    if max_hops is not None:
        dist: Dict[tuple[int, int], float] = {}
        prev: Dict[tuple[int, int], tuple[tuple[int, int] | None, int]] = {}
        pq: list[tuple[float, int, int]] = []
        for eid in start_edges:
            eid = int(eid)
            st = (eid, 1)
            dist[st] = float(edge_w[eid])
            prev[st] = (None, eid)
            heapq.heappush(pq, (dist[st], eid, 1))
        best: Optional[tuple[int, int]] = None
        while pq:
            d, eid, depth = heapq.heappop(pq)
            if d != dist.get((eid, depth), float("inf")):
                continue
            if int(graph.dst[eid]) == goal_node:
                best = (eid, depth)
                break
            if depth >= max_hops:
                continue
            next_edges = out[int(graph.dst[eid])]
            bcost = _boundary_costs(boundary, eid, next_edges)
            for ne, bc in zip(next_edges, bcost):
                ne = int(ne)
                nd = d + float(edge_w[ne]) + float(lambda_boundary) * float(bc)
                st = (ne, depth + 1)
                if nd < dist.get(st, float("inf")):
                    dist[st] = nd
                    prev[st] = ((eid, depth), ne)
                    heapq.heappush(pq, (nd, ne, depth + 1))
        if best is None:
            return _empty(variant)
        edges: list[int] = []
        cur: Optional[tuple[int, int]] = best
        while cur is not None:
            p, eid = prev[cur]
            edges.append(int(eid))
            cur = p
        edges.reverse()
        nodes = [int(graph.src[edges[0]])] + [int(graph.dst[e]) for e in edges]
        total_boundary = sum(boundary.boundary_cost(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]))
        total_cost = float(graph.cost[edges].sum())
        total_risk = float(graph.risk[edges].sum())
        exec_cost = total_risk + float(lambda_boundary) * float(total_boundary)
        return PlanResult(True, nodes, edges, total_cost, total_risk, float(total_boundary), float(dist[best]), variant, exec_cost, float("nan"))

    dist_arr = np.full(m, np.inf)
    prev_edge = np.full(m, -1, dtype=np.int64)
    pq: list[tuple[float, int]] = []
    for eid in start_edges:
        eid = int(eid)
        dist_arr[eid] = float(edge_w[eid])
        heapq.heappush(pq, (dist_arr[eid], eid))
    best = -1
    while pq:
        d, eid = heapq.heappop(pq)
        if d != dist_arr[eid]:
            continue
        if int(graph.dst[eid]) == goal_node:
            best = eid
            break
        next_edges = out[int(graph.dst[eid])]
        bcost = _boundary_costs(boundary, eid, next_edges)
        for ne, bc in zip(next_edges, bcost):
            ne = int(ne)
            nd = d + float(edge_w[ne]) + float(lambda_boundary) * float(bc)
            if nd < dist_arr[ne]:
                dist_arr[ne] = nd
                prev_edge[ne] = eid
                heapq.heappush(pq, (nd, ne))
    if best < 0:
        return _empty(variant)
    edges: list[int] = []
    cur = best
    while cur >= 0:
        edges.append(int(cur))
        cur = int(prev_edge[cur])
    edges.reverse()
    nodes = [int(graph.src[edges[0]])] + [int(graph.dst[e]) for e in edges]
    total_boundary = sum(boundary.boundary_cost(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]))
    total_cost = float(graph.cost[edges].sum())
    total_risk = float(graph.risk[edges].sum())
    exec_cost = total_risk + float(lambda_boundary) * float(total_boundary)
    return PlanResult(True, nodes, edges, total_cost, total_risk, float(total_boundary), float(dist_arr[best]), variant, exec_cost, float("nan"))


@dataclass
class _Label:
    edge: int
    cost: float
    exec_cost: float
    risk: float
    boundary: float
    prev: int
    depth: int


def _dominates(a: _Label, b: _Label, eps: float = 1e-7) -> bool:
    return (a.cost <= b.cost + eps and a.exec_cost <= b.exec_cost + eps) and (a.cost < b.cost - eps or a.exec_cost < b.exec_cost - eps)


def _constrained_line_graph_search(
    graph: BARSGraph,
    start_node: int,
    goal_node: int,
    exec_budget: float,
    lambda_boundary: float,
    boundary: Optional[BoundaryIndex],
    variant: str,
    max_edges: Optional[int] = None,
    max_labels_per_edge: int = 32,
) -> PlanResult:
    out = graph.outgoing_edges()
    start_edges = out[start_node]
    if len(start_edges) == 0:
        return _empty(variant, exec_budget)
    max_hops = int(max_edges) if max_edges is not None and int(max_edges) > 0 else graph.num_nodes + 1
    max_labels = max(1, int(max_labels_per_edge))
    edge_w = graph.cost.astype(np.float64, copy=False)

    labels: list[_Label] = []
    labels_by_edge: dict[int, list[int]] = {}
    pq: list[tuple[float, float, int]] = []  # temporal cost first, then exec cost.

    def add_label(label: _Label) -> Optional[int]:
        if label.exec_cost > exec_budget + 1e-7:
            return None
        existing_ids = labels_by_edge.setdefault(label.edge, [])
        existing = [labels[i] for i in existing_ids]
        if any(_dominates(old, label) or (old.cost <= label.cost + 1e-7 and old.exec_cost <= label.exec_cost + 1e-7) for old in existing):
            return None
        kept_ids = [i for i in existing_ids if not _dominates(label, labels[i])]
        idx = len(labels)
        labels.append(label)
        kept_ids.append(idx)
        if len(kept_ids) > max_labels:
            # Keep a small Pareto frontier biased toward both short paths and low execution cost.
            kept_ids = sorted(kept_ids, key=lambda i: (labels[i].cost, labels[i].exec_cost))[:max_labels]
        labels_by_edge[label.edge] = kept_ids
        heapq.heappush(pq, (label.cost, label.exec_cost, idx))
        return idx

    for eid in start_edges:
        eid = int(eid)
        risk = float(graph.risk[eid])
        add_label(_Label(edge=eid, cost=float(edge_w[eid]), exec_cost=risk, risk=risk, boundary=0.0, prev=-1, depth=1))

    best_idx: Optional[int] = None
    while pq:
        _, _, idx = heapq.heappop(pq)
        if idx >= len(labels):
            continue
        cur = labels[idx]
        if idx not in labels_by_edge.get(cur.edge, []):
            continue
        if int(graph.dst[cur.edge]) == goal_node:
            best_idx = idx
            break
        if cur.depth >= max_hops:
            continue
        u = int(graph.dst[cur.edge])
        next_edges = out[u]
        if boundary is not None:
            bcosts = _boundary_costs(boundary, cur.edge, next_edges)
        else:
            bcosts = np.zeros(len(next_edges), dtype=np.float32)
        for ne, bcost in zip(next_edges, bcosts):
            ne = int(ne)
            bcost = float(bcost)
            nrisk = cur.risk + float(graph.risk[ne])
            nb = cur.boundary + bcost
            nexec = nrisk + float(lambda_boundary) * nb
            ncost = cur.cost + float(edge_w[ne])
            add_label(_Label(edge=ne, cost=ncost, exec_cost=nexec, risk=nrisk, boundary=nb, prev=idx, depth=cur.depth + 1))

    if best_idx is None:
        return _empty(variant, exec_budget)
    edges: list[int] = []
    cur = best_idx
    while cur >= 0:
        lab = labels[cur]
        edges.append(int(lab.edge))
        cur = int(lab.prev)
    edges.reverse()
    nodes = [int(graph.src[edges[0]])] + [int(graph.dst[e]) for e in edges]
    best = labels[best_idx]
    return PlanResult(
        True,
        nodes,
        edges,
        float(best.cost),
        float(best.risk),
        float(best.boundary),
        float(best.cost),
        variant,
        float(best.exec_cost),
        float(exec_budget),
    )


def nearest_graph_node(graph: BARSGraph, embedding: np.ndarray) -> int:
    return int(np.argmin(np.sum((graph.node_embeddings - embedding[None, :]) ** 2, axis=1)))
