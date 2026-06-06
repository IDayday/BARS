from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external_src" / "GAS"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_ecg_policy_alignment_dataset_v3 import build_examples
from cage.contract_graph import ContractEdge, ContractFunnelNode, ContractGraph


def test_ecg_policy_alignment_v3_blocks_knn_bc_even_with_action():
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
                predicted_negative_progress=0.1,
                predicted_hit=0.9,
                edge_type="knn_bridge_candidate",
            )
        },
        metadata={"trusted_graph_variant": "trusted_conservative"},
    )
    actions = [{"env_name": "toy", "phi_s": [0.0], "phi_g": [1.0], "action_available": True, "action": [0.1], "action_source": "toy"}]

    examples = build_examples(graph, [], actions)

    assert examples[0]["edge_trust_level"] == "candidate_knn"
    assert examples[0]["trainable_for_bc"] is False
    assert examples[0]["recommended_training_stage"] == "trusted_bridge_conservative_filtering"
