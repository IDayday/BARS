from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external_src" / "GAS"))

from cage.contract_graph import ContractEdge, ContractFunnelNode, ContractGraph
from cage.contract_planner import ContractPlanner


def toy_graph():
    nodes = {
        "a": ContractFunnelNode("a", [0.0]),
        "b": ContractFunnelNode("b", [1.0]),
        "c": ContractFunnelNode("c", [2.0]),
    }
    edges = {
        "ab": ContractEdge("ab", "a", "b", d_phi=1.0, contract_lcb=0.9, predicted_negative_progress=0.05, uncertainty=0.1, edge_type="original_target"),
        "bc": ContractEdge("bc", "b", "c", d_phi=1.0, contract_lcb=0.9, predicted_negative_progress=0.05, uncertainty=0.1, edge_type="original_target"),
        "ac": ContractEdge("ac", "a", "c", d_phi=1.1, contract_lcb=0.1, predicted_negative_progress=0.8, uncertainty=0.4, edge_type="farther_path_target"),
    }
    return ContractGraph(nodes=nodes, edges=edges)


def test_contract_planners_find_paths_and_bottleneck_differs_from_shortest():
    planner = ContractPlanner(toy_graph())
    shortest = planner.shortest_by_dphi("a", "c")
    robust = planner.bottleneck_robust_path("a", "c")
    constrained = planner.risk_constrained_path("a", "c", min_edge_contract_lcb=0.5)

    assert shortest.edge_ids == ["ac"]
    assert robust.edge_ids == ["ab", "bc"]
    assert constrained.edge_ids == ["ab", "bc"]
    assert robust.bottleneck_edge_id in {"ab", "bc"}
    assert robust.predicted_success_lower_bound is not None


def test_contract_planner_reports_disconnected_graph():
    planner = ContractPlanner(toy_graph())
    result = planner.shortest_by_dphi("c", "a")

    assert result.reject_reason == "graph_disconnected"
    assert result.edge_ids == []
