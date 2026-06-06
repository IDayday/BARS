from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external_src" / "GAS"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_ecg_edge_provenance import audit
from cage.contract_graph import ContractEdge, ContractFunnelNode, ContractGraph


def test_edge_provenance_marks_knn_dependent_signal():
    graph = ContractGraph(
        nodes={
            "n0": ContractFunnelNode("n0", [0.0], env_name="toy"),
            "n1": ContractFunnelNode("n1", [1.0], env_name="toy"),
            "n2": ContractFunnelNode("n2", [2.0], env_name="toy"),
        },
        edges={
            "e0": ContractEdge("e0", "n0", "n1", contract_lcb=0.5, predicted_negative_progress=0.2, uncertainty=0.1, edge_type="original_contract"),
            "e1": ContractEdge("e1", "n1", "n2", contract_lcb=0.8, predicted_negative_progress=0.1, uncertainty=0.1, edge_type="knn_bridge_candidate"),
        },
    )
    rows = [
        {
            "planner_name": "max_contract_path",
            "path_found": "True",
            "different_from_shortest": "True",
            "improves_min_contract_vs_shortest": "True",
            "reduces_negative_risk_vs_shortest": "False",
            "edge_ids": '["e0", "e1"]',
            "bottleneck_edge_id": "e1",
            "bottleneck_edge_type": "knn_bridge_candidate",
        }
    ]

    _, summary = audit(graph, rows)

    assert summary["provenance_status"] == "KNN_DEPENDENT_PLANNER_SIGNAL"
    assert summary["knn_edge_rate"] == 0.5
