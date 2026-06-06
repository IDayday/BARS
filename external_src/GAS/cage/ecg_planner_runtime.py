from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cage.contract_graph import ContractEdge, ContractGraph
from cage.contract_planner import ContractPlanner


@dataclass(frozen=True)
class ECGSelection:
    target_phi: np.ndarray
    target_index: int | None
    trace: dict[str, Any]


class ECGPlannerRuntime:
    """Small runtime adapter for action-anchored ECG graph plans.

    This class is intentionally conservative: missing graph/model files fail at
    construction, and no-path decisions fall back to the original GAS target with
    an explicit trace reason.
    """

    def __init__(self, graph: ContractGraph, score_weights: dict[str, float] | None = None):
        self.graph = graph
        self.score_weights = score_weights or {}
        self.planner = ContractPlanner(graph)
        self._node_ids = list(graph.nodes)
        self._node_phi = np.asarray([graph.nodes[node_id].center_phi for node_id in self._node_ids], dtype=np.float32)

    @classmethod
    def from_paths(cls, graph_path: str | Path, planner_score_path: str | Path | None = None) -> "ECGPlannerRuntime":
        graph_path = Path(graph_path)
        if not graph_path.exists():
            raise FileNotFoundError(f"ECG graph not found: {graph_path}")
        weights: dict[str, float] | None = None
        if planner_score_path:
            planner_score_path = Path(planner_score_path)
            if not planner_score_path.exists():
                raise FileNotFoundError(f"ECG planner score not found: {planner_score_path}")
            with planner_score_path.open(encoding="utf-8") as fh:
                payload = json.load(fh)
            weights = dict(payload.get("weights") or {})
        return cls(ContractGraph.load_json(graph_path), weights)

    def select_target(
        self,
        current_phi: np.ndarray,
        final_goal_phi: np.ndarray,
        gas_target_phi: np.ndarray,
        gas_target_idx: int | None,
        *,
        trace_only: bool = False,
    ) -> ECGSelection:
        trace = {
            "ecg_runtime_enabled": True,
            "ecg_trace_only": bool(trace_only),
            "ecg_policy_adapter_used": False,
            "ecg_plan_recompute_count": 1,
            "ecg_fallback_reason": None,
        }
        if len(self._node_ids) == 0:
            trace["ecg_fallback_reason"] = "empty_graph"
            return ECGSelection(np.asarray(gas_target_phi), gas_target_idx, trace)
        src = self._nearest_node(current_phi)
        dst = self._nearest_node(gas_target_phi if gas_target_phi is not None else final_goal_phi)
        plan = self._weighted_plan(src, dst) if self.score_weights else self.planner.progress_contract_path(src, dst, max_path_length=16)
        edge_ids = plan.get("edge_ids") if isinstance(plan, dict) else plan.edge_ids
        reject_reason = plan.get("reject_reason") if isinstance(plan, dict) else plan.reject_reason
        if reject_reason or not edge_ids:
            trace["ecg_fallback_reason"] = reject_reason or "no_ecg_edge"
            return ECGSelection(np.asarray(gas_target_phi), gas_target_idx, trace)
        first_edge = self.graph.edges[edge_ids[0]]
        selected = np.asarray(self.graph.nodes[first_edge.dst].center_phi, dtype=np.float32)
        trace.update(
            {
                "ecg_plan_length": len(edge_ids),
                "ecg_selected_edge_id": first_edge.edge_id,
                "ecg_selected_edge_type": first_edge.edge_type,
                "ecg_contract_lcb": first_edge.contract_lcb,
                "ecg_predicted_negative": first_edge.predicted_negative_progress,
                "ecg_action_anchored": first_edge.action_anchored,
            }
        )
        if trace_only:
            return ECGSelection(np.asarray(gas_target_phi), gas_target_idx, trace)
        return ECGSelection(selected, gas_target_idx, trace)

    def _nearest_node(self, phi: np.ndarray) -> str:
        phi = np.asarray(phi, dtype=np.float32).reshape(1, -1)
        dim = min(phi.shape[1], self._node_phi.shape[1])
        distances = np.linalg.norm(self._node_phi[:, :dim] - phi[:, :dim], axis=1)
        return self._node_ids[int(np.argmin(distances))]

    def _weighted_plan(self, src: str, dst: str) -> dict[str, Any]:
        import heapq
        import math

        adjacency: dict[str, list[ContractEdge]] = {}
        for edge in self.graph.edges.values():
            adjacency.setdefault(edge.src, []).append(edge)
        heap: list[tuple[float, str, list[str]]] = [(0.0, src, [])]
        best: dict[str, float] = {src: 0.0}
        while heap:
            cost, node, edge_ids = heapq.heappop(heap)
            if node == dst:
                return {"edge_ids": edge_ids, "reject_reason": None}
            if len(edge_ids) >= 16:
                continue
            for edge in adjacency.get(node, []):
                next_cost = cost + self._edge_cost(edge)
                if next_cost >= best.get(edge.dst, math.inf):
                    continue
                best[edge.dst] = next_cost
                heapq.heappush(heap, (next_cost, edge.dst, [*edge_ids, edge.edge_id]))
        return {"edge_ids": [], "reject_reason": "graph_disconnected"}

    def _edge_cost(self, edge: ContractEdge) -> float:
        w = self.score_weights
        d_phi = float(edge.d_phi or 0.0)
        contract = float(edge.contract_lcb or 0.0)
        negative = float(edge.predicted_negative_progress or 0.0)
        uncertainty = float(edge.uncertainty or 0.0)
        progress = 1.0 / (1.0 + max(d_phi, 0.0))
        return (
            float(w.get("path_length", 0.05))
            + float(w.get("task_progress", 0.5)) * (1.0 - progress)
            + float(w.get("negative", 1.0)) * negative
            + float(w.get("uncertainty", 0.5)) * uncertainty
            - float(w.get("contract", 1.0)) * contract
        )
