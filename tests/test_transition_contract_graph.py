from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external_src" / "GAS"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_cage_transition_contract_graph import GraphBuilder, add_contract_dataset_edges, add_segment_capture_edges, graph_connectivity_summary
from cage.contract_graph import ContractEdge, ContractFunnelNode, ContractGraph
from cage.contract_model import ContractScorer


def test_transition_graph_adds_multitype_edges(tmp_path: Path):
    base = ContractGraph(
        nodes={
            "n0": ContractFunnelNode("n0", [0.0], env_name="toy"),
            "n1": ContractFunnelNode("n1", [1.0], env_name="toy"),
        },
        edges={"e0": ContractEdge("e0", "n0", "n1", contract_lcb=0.6, predicted_negative_progress=0.1, edge_type="original_target")},
    )
    segment_path = tmp_path / "toy_segments.jsonl"
    segment_path.write_text(
        json.dumps(
            {
                "env_name": "toy",
                "seed": 0,
                "variant": "gas",
                "episode_idx": 0,
                "segment_idx": 0,
                "segment_id": "toy__seed0__task0__ep0__seg0",
                "start_phi": [0.0],
                "end_phi": [0.5],
                "target_phi": [1.0],
                "final_goal_phi": [2.0],
                "path_phi": [[0.0], [0.5], [1.0]],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    builder = GraphBuilder(ContractScorer(), min_contract_lcb=-1.0, max_added_edges=100)
    builder.load_base(base)
    add_contract_dataset_edges(builder, [{"env_name": "toy", "phi_s": [0.0], "phi_g": [1.5], "target_mode": "original_target"}])
    counts = add_segment_capture_edges(builder, [str(tmp_path)])
    builder.add_boundaries_from_sequences()
    graph = builder.build({"test": True})
    summary = graph.summarize()
    conn = graph_connectivity_summary(graph)

    assert summary["edge_count"] > 1
    assert counts["temporal_transition"] >= 1
    assert counts["final_goal_candidate"] >= 1
    assert conn["avg_out_degree"] > 0
