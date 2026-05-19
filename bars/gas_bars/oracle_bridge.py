from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from .bridge_graph import BRIDGE_EDGE_TYPES, BridgeGraphBundle, analyze_bridge_graphs, load_bridge_graph


def build_oracle_bridge_graph(graph: BridgeGraphBundle, edge_exec: pd.DataFrame) -> BridgeGraphBundle:
    labels = edge_exec.copy()
    labels["success"] = pd.to_numeric(labels.get("success", 0), errors="coerce").fillna(0).astype(int)
    success_bridge_ids = set(labels.loc[labels["success"] == 1, "edge_id"].astype(int).tolist()) if "edge_id" in labels else set()
    edges = graph.edges.copy()
    is_bridge = edges["edge_type"].astype(str).isin(BRIDGE_EDGE_TYPES)
    keep = (~is_bridge) | edges["edge_id"].astype(int).isin(success_bridge_ids)
    oracle_edges = edges.loc[keep].copy()
    oracle_edges["graph_id"] = "G_oracle"
    return BridgeGraphBundle(
        graph_id="G_oracle",
        nodes=graph.nodes.copy(),
        edges=oracle_edges.reset_index(drop=True),
        way_steps=graph.way_steps,
        metadata={"source_graph": graph.graph_id, "oracle_success_bridge_count": int(is_bridge.sum() - ((is_bridge) & (~keep)).sum())},
    )


def save_oracle_graph(bundle: BridgeGraphBundle, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bundle.to_dict(), f)


def oracle_summary(g0: BridgeGraphBundle, aggressive: BridgeGraphBundle, oracle: BridgeGraphBundle) -> pd.DataFrame:
    return analyze_bridge_graphs({"G0": g0, aggressive.graph_id: aggressive, "G_oracle": oracle})


def build_oracle_from_paths(graph_path: str | Path, edge_exec_csv: str | Path, out_path: str | Path) -> BridgeGraphBundle:
    graph = load_bridge_graph(graph_path)
    edge_exec = pd.read_csv(edge_exec_csv)
    oracle = build_oracle_bridge_graph(graph, edge_exec)
    save_oracle_graph(oracle, out_path)
    return oracle
