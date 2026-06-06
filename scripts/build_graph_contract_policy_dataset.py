#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external_src" / "GAS"))

from cage.contract_graph import ContractGraph


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build graph-induced policy-alignment examples from a CAGE contract graph.")
    parser.add_argument("--contract_graph_path", default="results/cage_ecg/contract_graph/contract_graph.json")
    parser.add_argument("--contract_dataset_path", default="results/cage_clp1/datasets/closed_loop_contracts.jsonl")
    parser.add_argument("--out_jsonl", default="results/cage_ecg/policy_alignment/graph_contract_policy_dataset.jsonl")
    parser.add_argument("--out_report", default="reports/stage35_graph_contract_policy_dataset.md")
    return parser


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def row_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(record.get("env_name") or ""),
        _phi_key(record.get("phi_s", record.get("phi_start"))),
        _phi_key(record.get("phi_g", record.get("phi_target"))),
    )


def build_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    lookup: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        lookup[row_key(row)].append(row)
    return lookup


def build_examples(graph: ContractGraph, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = build_lookup(rows)
    out: list[dict[str, Any]] = []
    for edge in graph.edges.values():
        src = graph.nodes[edge.src]
        dst = graph.nodes[edge.dst]
        matches = lookup.get((str(src.env_name or ""), _phi_key(src.center_phi), _phi_key(dst.center_phi)), [])
        ref = matches[0] if matches else {}
        lcb = _float(edge.contract_lcb, 0.0)
        negative = _float(edge.predicted_negative_progress, 1.0)
        positive = bool(lcb >= 0.50 and negative <= 0.25)
        negative_label = bool(lcb < 0.25 or negative >= 0.50)
        action = ref.get("action")
        action_available = action is not None
        if action_available and positive:
            objective = "bc_hard_positive"
        elif negative_label:
            objective = "ranking_contrastive_conservative_filtering"
        else:
            objective = "contract_ranking_or_curriculum"
        out.append(
            {
                "env_name": src.env_name,
                "seed": ref.get("seed"),
                "source_edge_id": edge.edge_id,
                "phi_s": src.center_phi,
                "phi_g": dst.center_phi,
                "target_mode": edge.edge_type,
                "edge_type": edge.edge_type,
                "contract_lcb": edge.contract_lcb,
                "predicted_hit": edge.predicted_hit,
                "predicted_negative_progress": edge.predicted_negative_progress,
                "uncertainty": edge.uncertainty,
                "q_train_support": edge.q_train_support,
                "label_positive_contract": positive,
                "label_negative_contract": negative_label,
                "final_phase": bool(ref.get("final_phase", edge.edge_type == "final_goal")),
                "recovery_candidate": bool(ref.get("recovery_candidate", "recovery" in str(edge.edge_type or ""))),
                "boundary_transition": False,
                "action_supervision_available": bool(action_available),
                "action": action,
                "curriculum_weight": _curriculum_weight(edge.contract_lcb, edge.predicted_negative_progress),
                "recommended_objective": objective,
            }
        )
    for boundary in graph.boundary_contracts.values():
        prev_edge = graph.edges.get(boundary.prev_edge_id)
        next_edge = graph.edges.get(boundary.next_edge_id)
        if prev_edge is None or next_edge is None:
            continue
        src = graph.nodes[prev_edge.src]
        dst = graph.nodes[next_edge.dst]
        out.append(
            {
                "env_name": src.env_name,
                "seed": None,
                "source_edge_id": f"{boundary.prev_edge_id}->{boundary.next_edge_id}",
                "phi_s": src.center_phi,
                "phi_g": dst.center_phi,
                "target_mode": "boundary_transition",
                "edge_type": "boundary_transition",
                "contract_lcb": boundary.compatibility_score,
                "predicted_hit": None,
                "predicted_negative_progress": boundary.boundary_risk,
                "uncertainty": None,
                "q_train_support": None,
                "label_positive_contract": bool(_float(boundary.compatibility_score, 0.0) >= 0.50),
                "label_negative_contract": bool(_float(boundary.boundary_risk, 1.0) >= 0.50),
                "final_phase": False,
                "recovery_candidate": False,
                "boundary_transition": True,
                "action_supervision_available": False,
                "action": None,
                "curriculum_weight": _curriculum_weight(boundary.compatibility_score, boundary.boundary_risk),
                "recommended_objective": "boundary_compatibility_ranking",
            }
        )
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def write_report(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = len(rows)
    positives = sum(1 for row in rows if row.get("label_positive_contract"))
    negatives = sum(1 for row in rows if row.get("label_negative_contract"))
    actions = sum(1 for row in rows if row.get("action_supervision_available"))
    finals = sum(1 for row in rows if row.get("final_phase"))
    recoveries = sum(1 for row in rows if row.get("recovery_candidate"))
    env_counts = Counter(str(row.get("env_name") or "NA") for row in rows)
    objective_counts = Counter(str(row.get("recommended_objective") or "NA") for row in rows)
    summary = {
        "total_examples": total,
        "positive_contract_rate": _rate(positives, total),
        "negative_contract_rate": _rate(negatives, total),
        "action_supervision_rate": _rate(actions, total),
        "final_phase_rate": _rate(finals, total),
        "recovery_rate": _rate(recoveries, total),
        "env_counts": dict(env_counts),
        "objective_counts": dict(objective_counts),
        "policy_training_feasible": bool(actions > 0 and positives > 0),
    }
    lines = [
        "# Stage35 Graph-Contract Policy Dataset",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| total_examples | {total} |",
        f"| positive_contract_rate | {_fmt(summary['positive_contract_rate'])} |",
        f"| negative_contract_rate | {_fmt(summary['negative_contract_rate'])} |",
        f"| action_supervision_rate | {_fmt(summary['action_supervision_rate'])} |",
        f"| final_phase_rate | {_fmt(summary['final_phase_rate'])} |",
        f"| recovery_rate | {_fmt(summary['recovery_rate'])} |",
        "",
        "## Objective Counts",
        "",
        json.dumps(summary["objective_counts"], indent=2, sort_keys=True),
        "",
        "## Feasibility",
        "",
        "Policy finetuning is feasible only if positive contract examples also have action supervision. If action supervision rate is zero, this dataset should be used for ranking, contrastive objectives, conservative filtering, or future data collection, not naive BC.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    args = build_parser().parse_args()
    graph = ContractGraph.load_json(args.contract_graph_path)
    rows = load_jsonl(args.contract_dataset_path)
    examples = build_examples(graph, rows)
    out_jsonl = Path(args.out_jsonl)
    write_jsonl(out_jsonl, examples)
    summary = write_report(Path(args.out_report), examples)
    print(json.dumps({"out_jsonl": str(out_jsonl), **summary}, sort_keys=True))
    return 0


def _phi_key(phi: Any, decimals: int = 3) -> str:
    if phi is None:
        return ""
    arr = np.asarray(phi, dtype=np.float32).reshape(-1)
    return ",".join(f"{x:.{decimals}f}" for x in np.round(arr, decimals=decimals))


def _float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _rate(num: int, den: int) -> float | None:
    return float(num / den) if den else None


def _curriculum_weight(contract_lcb: Any, negative: Any) -> float:
    return max(0.0, min(1.0, _float(contract_lcb, 0.0) * (1.0 - _float(negative, 1.0))))


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
