#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bars.gas_bars.bars_v3_planner import plan_bars_v3


def _nodes():
    rows = []
    for i, x in enumerate([0.0, 1.0, 2.0, 1.0]):
        rows.append({"node_id": i, "node_type": "gas_keynode", "phi_0": x, "phi_1": 0.0 if i != 3 else 1.0})
    return pd.DataFrame(rows)


def test_short_risky_vs_long_safe() -> None:
    nodes = _nodes()
    edges = pd.DataFrame(
        [
            {"edge_id": 0, "u": 0, "v": 1, "temporal_cost": 1.0, "phi_dist": 1.0, "edge_type": "safe_local"},
            {"edge_id": 1, "u": 1, "v": 2, "temporal_cost": 1.0, "phi_dist": 1.0, "edge_type": "safe_local"},
            {"edge_id": 2, "u": 0, "v": 2, "temporal_cost": 0.2, "phi_dist": 2.0, "edge_type": "aggressive_tdr_bridge", "p_bridge": 0.1},
        ]
    )
    graph = {"nodes": nodes, "edges": edges, "way_steps": 1.0}
    low_budget = plan_bars_v3(graph, nodes.loc[0, ["phi_0", "phi_1"]].to_numpy(dtype=float), nodes.loc[2, ["phi_0", "phi_1"]].to_numpy(dtype=float), bridge_risk_budget=1.0, max_risky_bridges=1, virtual_nodes=False)
    high_budget = plan_bars_v3(graph, nodes.loc[0, ["phi_0", "phi_1"]].to_numpy(dtype=float), nodes.loc[2, ["phi_0", "phi_1"]].to_numpy(dtype=float), p_bridge_min=0.0, bridge_risk_budget=3.0, max_risky_bridges=1, virtual_nodes=False)
    assert low_budget.edge_ids == [0, 1], low_budget
    assert high_budget.edge_ids == [2], high_budget


def test_local_edges_do_not_consume_risk() -> None:
    nodes = _nodes()
    edges = pd.DataFrame(
        [
            {"edge_id": 0, "u": 0, "v": 1, "temporal_cost": 1.0, "phi_dist": 1.0, "edge_type": "safe_local", "p_bridge": 0.01},
            {"edge_id": 1, "u": 1, "v": 2, "temporal_cost": 1.0, "phi_dist": 1.0, "edge_type": "safe_local", "p_bridge": 0.01},
        ]
    )
    graph = {"nodes": nodes, "edges": edges, "way_steps": 1.0}
    res = plan_bars_v3(graph, nodes.loc[0, ["phi_0", "phi_1"]].to_numpy(dtype=float), nodes.loc[2, ["phi_0", "phi_1"]].to_numpy(dtype=float), bridge_risk_budget=0.0, max_risky_bridges=0, virtual_nodes=False)
    assert not res.no_path
    assert res.exec_risk == 0.0


def test_boundary_only_bridge_junctions() -> None:
    nodes = _nodes()
    edges = pd.DataFrame(
        [
            {"edge_id": 0, "u": 0, "v": 3, "temporal_cost": 1.0, "phi_dist": 1.0, "edge_type": "safe_local"},
            {"edge_id": 1, "u": 3, "v": 2, "temporal_cost": 1.0, "phi_dist": 1.0, "edge_type": "safe_local"},
            {"edge_id": 2, "u": 0, "v": 1, "temporal_cost": 0.1, "phi_dist": 1.0, "edge_type": "safe_local"},
            {"edge_id": 3, "u": 1, "v": 2, "temporal_cost": 0.1, "phi_dist": 1.0, "edge_type": "aggressive_tdr_bridge", "p_bridge": 0.9},
        ]
    )
    boundary = pd.DataFrame(
        [
            {"prev_edge_id": 0, "next_edge_id": 1, "psi": 0.001},
            {"prev_edge_id": 2, "next_edge_id": 3, "psi": 0.001},
        ]
    )
    graph = {"nodes": nodes, "edges": edges, "way_steps": 1.0}
    res = plan_bars_v3(graph, nodes.loc[0, ["phi_0", "phi_1"]].to_numpy(dtype=float), nodes.loc[2, ["phi_0", "phi_1"]].to_numpy(dtype=float), variant="p_bridge_boundary", boundary_scores=boundary, bridge_risk_budget=0.5, max_risky_bridges=1, virtual_nodes=False)
    assert res.edge_ids == [0, 1], res
    assert res.boundary_risk == 0.0


def main() -> None:
    test_short_risky_vs_long_safe()
    test_local_edges_do_not_consume_risk()
    test_boundary_only_bridge_junctions()
    print("stage23 smoke planner tests passed")


if __name__ == "__main__":
    main()
