from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ContractFunnelNode:
    node_id: str
    center_phi: list[float]
    radius: float | None = None
    support_count: int = 0
    env_name: str | None = None
    source_node_ids: list[str] = field(default_factory=list)
    entry_contract_score: float | None = None
    exit_contract_score: float | None = None


@dataclass
class ContractEdge:
    edge_id: str
    src: str
    dst: str
    d_phi: float | None = None
    gas_edge_exists: bool = False
    contract_score: float | None = None
    contract_lcb: float | None = None
    predicted_hit: float | None = None
    predicted_contract_positive: float | None = None
    predicted_negative_progress: float | None = None
    uncertainty: float | None = None
    q_train_support: float | None = None
    edge_type: str | None = None
    bottleneck_score: float | None = None
    action_anchored: bool | None = None
    horizon: int | None = None
    action_source: str | None = None
    trust_level: str | None = None


@dataclass
class BoundaryContract:
    prev_edge_id: str
    next_edge_id: str
    compatibility_score: float | None = None
    boundary_risk: float | None = None
    observed_transition_count: int = 0


class ContractGraph:
    def __init__(
        self,
        nodes: dict[str, ContractFunnelNode] | None = None,
        edges: dict[str, ContractEdge] | None = None,
        boundary_contracts: dict[str, BoundaryContract] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.nodes = nodes or {}
        self.edges = edges or {}
        self.boundary_contracts = boundary_contracts or {}
        self.metadata = metadata or {}
        self._validate()

    def _validate(self) -> None:
        for node_id, node in self.nodes.items():
            if node_id != node.node_id:
                raise ValueError(f"node key/id mismatch: {node_id} != {node.node_id}")
            if not node.center_phi:
                raise ValueError(f"node {node_id} has empty center_phi")
        for edge_id, edge in self.edges.items():
            if edge_id != edge.edge_id:
                raise ValueError(f"edge key/id mismatch: {edge_id} != {edge.edge_id}")
            if edge.src not in self.nodes or edge.dst not in self.nodes:
                raise ValueError(f"edge {edge_id} references missing nodes: {edge.src}->{edge.dst}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": _json_ready(self.metadata),
            "nodes": [asdict(node) for node in self.nodes.values()],
            "edges": [asdict(edge) for edge in self.edges.values()],
            "boundary_contracts": [asdict(boundary) for boundary in self.boundary_contracts.values()],
        }

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> "ContractGraph":
        nodes = {
            str(item["node_id"]): ContractFunnelNode(**item)
            for item in record.get("nodes", [])
        }
        edges = {
            str(item["edge_id"]): ContractEdge(**item)
            for item in record.get("edges", [])
        }
        boundary_contracts = {
            _boundary_id(item["prev_edge_id"], item["next_edge_id"]): BoundaryContract(**item)
            for item in record.get("boundary_contracts", [])
        }
        return cls(nodes=nodes, edges=edges, boundary_contracts=boundary_contracts, metadata=dict(record.get("metadata") or {}))

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True)

    @classmethod
    def load_json(cls, path: str | Path) -> "ContractGraph":
        with Path(path).open(encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def export_nodes_csv(self, path: str | Path) -> None:
        self._export_csv(path, [asdict(node) for node in self.nodes.values()])

    def export_edges_csv(self, path: str | Path) -> None:
        self._export_csv(path, [asdict(edge) for edge in self.edges.values()])

    def export_boundary_csv(self, path: str | Path) -> None:
        self._export_csv(path, [asdict(boundary) for boundary in self.boundary_contracts.values()])

    def summarize(self) -> dict[str, Any]:
        edges = list(self.edges.values())
        nodes = list(self.nodes.values())
        low_contract = [edge for edge in edges if _float(edge.contract_lcb) is not None and _float(edge.contract_lcb) < 0.35]
        high_negative = [edge for edge in edges if _float(edge.predicted_negative_progress) is not None and _float(edge.predicted_negative_progress) > 0.45]
        uncertain = [edge for edge in edges if _float(edge.uncertainty) is not None and _float(edge.uncertainty) > 0.35]
        final_edges = [edge for edge in edges if "final" in str(edge.edge_type or "").lower()]
        recovery_edges = [edge for edge in edges if "recovery" in str(edge.edge_type or "").lower()]
        env_counts: dict[str, int] = {}
        for node in nodes:
            env = str(node.env_name or "NA")
            env_counts[env] = env_counts.get(env, 0) + 1
        type_counts: dict[str, int] = {}
        for edge in edges:
            key = str(edge.edge_type or "NA")
            type_counts[key] = type_counts.get(key, 0) + 1
        return {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "boundary_contract_count": len(self.boundary_contracts),
            "low_contract_edge_rate": _rate(len(low_contract), len(edges)),
            "high_negative_edge_rate": _rate(len(high_negative), len(edges)),
            "uncertain_edge_rate": _rate(len(uncertain), len(edges)),
            "final_goal_edge_rate": _rate(len(final_edges), len(edges)),
            "recovery_edge_rate": _rate(len(recovery_edges), len(edges)),
            "env_counts": env_counts,
            "edge_type_counts": type_counts,
            "metadata": dict(self.metadata),
        }

    @staticmethod
    def _export_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = sorted({key for row in rows for key in row})
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def _boundary_id(prev_edge_id: str, next_edge_id: str) -> str:
    return f"{prev_edge_id}__to__{next_edge_id}"


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(_json_ready(value), sort_keys=True)
    return value


def _json_ready(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rate(num: int, den: int) -> float | None:
    return float(num / den) if den else None
