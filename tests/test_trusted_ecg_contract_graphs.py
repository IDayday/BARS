from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external_src" / "GAS"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_trusted_ecg_contract_graphs import filter_graph, summarize_variant
from cage.contract_graph import ContractEdge, ContractFunnelNode, ContractGraph


def test_trusted_graph_filters_knn_by_predicate():
    graph = ContractGraph(
        nodes={
            "n0": ContractFunnelNode("n0", [0.0], env_name="toy"),
            "n1": ContractFunnelNode("n1", [1.0], env_name="toy"),
            "n2": ContractFunnelNode("n2", [2.0], env_name="toy"),
        },
        edges={
            "e0": ContractEdge("e0", "n0", "n1", contract_lcb=0.2, predicted_negative_progress=0.2, uncertainty=0.1, edge_type="temporal_transition"),
            "e1": ContractEdge("e1", "n1", "n2", contract_lcb=0.8, predicted_negative_progress=0.1, uncertainty=0.1, edge_type="knn_bridge_candidate"),
            "e2": ContractEdge("e2", "n2", "n0", contract_lcb=0.8, predicted_negative_progress=0.9, uncertainty=0.1, edge_type="knn_bridge_candidate"),
        },
    )
    trusted = filter_graph(
        graph,
        lambda edge: edge.edge_type == "temporal_transition"
        or (edge.edge_type == "knn_bridge_candidate" and edge.predicted_negative_progress <= 0.5),
        "trusted_conservative",
    )
    summary = summarize_variant(trusted)

    assert set(trusted.edges) == {"e0", "e1"}
    assert summary["knn_edge_rate"] == 0.5
