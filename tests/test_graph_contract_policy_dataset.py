from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external_src" / "GAS"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_graph_contract_policy_dataset import build_examples
from cage.contract_graph import ContractEdge, ContractFunnelNode, ContractGraph


def test_graph_contract_policy_dataset_marks_unlabeled_examples_as_ranking_not_bc():
    graph = ContractGraph(
        nodes={
            "n0": ContractFunnelNode("n0", [0.0], env_name="toy"),
            "n1": ContractFunnelNode("n1", [1.0], env_name="toy"),
        },
        edges={
            "e0": ContractEdge(
                "e0",
                "n0",
                "n1",
                contract_lcb=0.8,
                predicted_hit=0.7,
                predicted_negative_progress=0.1,
                uncertainty=0.2,
                edge_type="original_target",
            )
        },
    )
    rows = [{"env_name": "toy", "phi_s": [0.0], "phi_g": [1.0], "target_mode": "original_target"}]

    examples = build_examples(graph, rows)

    assert len(examples) == 1
    assert examples[0]["label_positive_contract"] is True
    assert examples[0]["action_supervision_available"] is False
    assert examples[0]["recommended_objective"] == "contract_ranking_or_curriculum"
