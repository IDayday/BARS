from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Callable

from cage.contract_graph import ContractEdge, ContractGraph


@dataclass
class PlanResult:
    planner_name: str
    node_ids: list[str]
    edge_ids: list[str]
    path_cost: float | None = None
    path_contract_product: float | None = None
    path_min_contract: float | None = None
    path_negative_risk: float | None = None
    path_uncertainty: float | None = None
    path_boundary_risk: float | None = None
    predicted_success_lower_bound: float | None = None
    bottleneck_edge_id: str | None = None
    reject_reason: str | None = None


class ContractPlanner:
    def __init__(self, graph: ContractGraph):
        self.graph = graph
        self.adjacency: dict[str, list[ContractEdge]] = {}
        for edge in graph.edges.values():
            self.adjacency.setdefault(edge.src, []).append(edge)

    def shortest_by_dphi(self, src: str, dst: str, *, max_path_length: int = 64) -> PlanResult:
        return self._dijkstra(src, dst, "shortest_by_dphi", lambda edge: max(_float(edge.d_phi, 1.0), 1e-6), max_path_length=max_path_length)

    def max_contract_path(self, src: str, dst: str, *, max_path_length: int = 64) -> PlanResult:
        return self._dijkstra(
            src,
            dst,
            "max_contract_path",
            lambda edge: -math.log(max(_float(edge.contract_lcb, 0.0), 1e-6)),
            max_path_length=max_path_length,
        )

    def risk_constrained_path(
        self,
        src: str,
        dst: str,
        *,
        min_edge_contract_lcb: float = 0.20,
        max_cumulative_negative_risk: float = 5.0,
        max_path_length: int = 64,
    ) -> PlanResult:
        def allowed(edge: ContractEdge, path_edges: list[str]) -> bool:
            if _float(edge.contract_lcb, 0.0) < float(min_edge_contract_lcb):
                return False
            risk = sum(_float(self.graph.edges[eid].predicted_negative_progress, 0.0) for eid in path_edges)
            return risk + _float(edge.predicted_negative_progress, 0.0) <= float(max_cumulative_negative_risk)

        return self._dijkstra(
            src,
            dst,
            "risk_constrained_path",
            lambda edge: max(_float(edge.d_phi, 1.0), 1e-6) + _float(edge.predicted_negative_progress, 0.0),
            edge_allowed=allowed,
            max_path_length=max_path_length,
        )

    def bottleneck_robust_path(self, src: str, dst: str, *, max_path_length: int = 64) -> PlanResult:
        if src not in self.graph.nodes or dst not in self.graph.nodes:
            return _reject("bottleneck_robust_path", "missing_src_or_dst")
        heap: list[tuple[float, str, list[str], list[str]]] = [(-1.0, src, [src], [])]
        best: dict[str, float] = {src: 1.0}
        while heap:
            neg_width, node, nodes, edge_ids = heapq.heappop(heap)
            width = -neg_width
            if node == dst:
                return self._plan_from_edges("bottleneck_robust_path", nodes, edge_ids, path_cost=-width)
            if len(edge_ids) >= int(max_path_length):
                continue
            for edge in self.adjacency.get(node, []):
                edge_width = max(_float(edge.contract_lcb, 0.0), 1e-6)
                next_width = min(width, edge_width)
                if next_width <= best.get(edge.dst, -1.0):
                    continue
                best[edge.dst] = next_width
                heapq.heappush(heap, (-next_width, edge.dst, [*nodes, edge.dst], [*edge_ids, edge.edge_id]))
        return _reject("bottleneck_robust_path", "graph_disconnected")

    def progress_contract_path(self, src: str, dst: str, *, max_path_length: int = 64) -> PlanResult:
        def cost(edge: ContractEdge) -> float:
            contract = max(_float(edge.contract_lcb, 0.0), 1e-6)
            mode_penalty = 0.0
            text = str(edge.edge_type or "")
            if "final" in text:
                mode_penalty -= 0.05
            if "recovery" in text:
                mode_penalty += 0.25
            return max(_float(edge.d_phi, 1.0), 1e-6) / contract + mode_penalty

        return self._dijkstra(src, dst, "progress_contract_path", cost, max_path_length=max_path_length)

    def _dijkstra(
        self,
        src: str,
        dst: str,
        planner_name: str,
        edge_cost: Callable[[ContractEdge], float],
        *,
        edge_allowed: Callable[[ContractEdge, list[str]], bool] | None = None,
        max_path_length: int = 64,
    ) -> PlanResult:
        if src not in self.graph.nodes or dst not in self.graph.nodes:
            return _reject(planner_name, "missing_src_or_dst")
        heap: list[tuple[float, str, list[str], list[str]]] = [(0.0, src, [src], [])]
        best: dict[str, float] = {src: 0.0}
        while heap:
            cost, node, nodes, edge_ids = heapq.heappop(heap)
            if node == dst:
                return self._plan_from_edges(planner_name, nodes, edge_ids, path_cost=cost)
            if len(edge_ids) >= int(max_path_length):
                continue
            for edge in self.adjacency.get(node, []):
                if edge_allowed is not None and not edge_allowed(edge, edge_ids):
                    continue
                next_cost = cost + float(edge_cost(edge))
                if next_cost >= best.get(edge.dst, math.inf):
                    continue
                best[edge.dst] = next_cost
                heapq.heappush(heap, (next_cost, edge.dst, [*nodes, edge.dst], [*edge_ids, edge.edge_id]))
        return _reject(planner_name, "graph_disconnected")

    def _plan_from_edges(self, planner_name: str, nodes: list[str], edge_ids: list[str], *, path_cost: float | None) -> PlanResult:
        edges = [self.graph.edges[eid] for eid in edge_ids]
        if not edges:
            return PlanResult(planner_name=planner_name, node_ids=nodes, edge_ids=edge_ids, path_cost=path_cost, reject_reason=None)
        contracts = [max(_float(edge.contract_lcb, 0.0), 0.0) for edge in edges]
        negatives = [_float(edge.predicted_negative_progress, 0.0) for edge in edges]
        uncertainties = [_float(edge.uncertainty, 0.0) for edge in edges]
        product = 1.0
        for value in contracts:
            product *= value
        min_contract = min(contracts)
        bottleneck = min(edges, key=lambda edge: _float(edge.contract_lcb, 0.0))
        boundary_risk = self._path_boundary_risk(edge_ids)
        return PlanResult(
            planner_name=planner_name,
            node_ids=nodes,
            edge_ids=edge_ids,
            path_cost=path_cost,
            path_contract_product=product,
            path_min_contract=min_contract,
            path_negative_risk=sum(negatives),
            path_uncertainty=sum(uncertainties),
            path_boundary_risk=boundary_risk,
            predicted_success_lower_bound=max(0.0, min_contract - 0.1 * sum(negatives) - 0.05 * sum(uncertainties) - 0.1 * (boundary_risk or 0.0)),
            bottleneck_edge_id=bottleneck.edge_id,
            reject_reason=None,
        )

    def _path_boundary_risk(self, edge_ids: list[str]) -> float | None:
        risks: list[float] = []
        for prev, nxt in zip(edge_ids[:-1], edge_ids[1:]):
            boundary = self.graph.boundary_contracts.get(f"{prev}__to__{nxt}")
            if boundary is not None and boundary.boundary_risk is not None:
                risks.append(float(boundary.boundary_risk))
        return sum(risks) if risks else None


def _reject(planner_name: str, reason: str) -> PlanResult:
    return PlanResult(planner_name=planner_name, node_ids=[], edge_ids=[], reject_reason=reason)


def _float(value, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
