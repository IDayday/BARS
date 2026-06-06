from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external_src" / "GAS"))
sys.path.insert(0, str(ROOT / "scripts"))

from augment_final_recovery_contracts import add_edge, summarize_final_recovery
from cage.contract_graph import ContractFunnelNode, ContractGraph
from cage.contract_model import ContractScorer


def test_final_recovery_augmentation_reports_underpowered_recovery():
    graph = ContractGraph(nodes={"n0": ContractFunnelNode("n0", [0.0], env_name="toy")})
    key_to_id = {"toy::0.000": "n0"}
    edge_keys: set[tuple[str, str, str]] = set()
    scorer = ContractScorer()

    assert add_edge(graph, key_to_id, edge_keys, scorer, [0.0], [1.0], "toy", "final_goal_candidate")
    assert add_edge(graph, key_to_id, edge_keys, scorer, [0.0], [0.5], "toy", "recovery_candidate")

    summary = summarize_final_recovery(graph, recovery_threshold=3)

    assert summary["final_goal_edge_count"] == 1
    assert summary["recovery_edge_count"] == 1
    assert summary["final_goal_status"] == "ok"
    assert summary["recovery_status"] == "RECOVERY_CONTRACT_UNDERPOWERED"
