from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external_src" / "GAS"))

from cage.contract_graph import BoundaryContract, ContractEdge, ContractFunnelNode, ContractGraph


def test_contract_graph_roundtrip_and_exports(tmp_path: Path):
    graph = ContractGraph(
        nodes={
            "n0": ContractFunnelNode("n0", [0.0, 0.0], support_count=2, env_name="toy"),
            "n1": ContractFunnelNode("n1", [1.0, 0.0], support_count=1, env_name="toy"),
        },
        edges={
            "e0": ContractEdge(
                "e0",
                "n0",
                "n1",
                d_phi=1.0,
                contract_lcb=0.8,
                predicted_negative_progress=0.1,
                uncertainty=0.2,
                edge_type="original_target",
                bottleneck_score=0.8,
            )
        },
        boundary_contracts={"e0__to__e0": BoundaryContract("e0", "e0", compatibility_score=0.8, boundary_risk=0.1)},
    )

    path = tmp_path / "graph.json"
    graph.save_json(path)
    loaded = ContractGraph.load_json(path)

    assert loaded.summarize()["node_count"] == 2
    assert loaded.summarize()["edge_count"] == 1
    assert loaded.edges["e0"].contract_lcb == 0.8

    loaded.export_nodes_csv(tmp_path / "nodes.csv")
    loaded.export_edges_csv(tmp_path / "edges.csv")
    loaded.export_boundary_csv(tmp_path / "boundary.csv")

    assert (tmp_path / "nodes.csv").exists()
    assert (tmp_path / "edges.csv").exists()
    assert (tmp_path / "boundary.csv").exists()
