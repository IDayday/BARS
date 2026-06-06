#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external_src" / "GAS"))

from cage.contract_graph import ContractGraph


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Stage36 ECG policy-alignment dataset v2.")
    parser.add_argument("--contract_graph_path", default="results/cage_ecg/transition_contract_graph/contract_graph_augmented.json")
    parser.add_argument("--contract_dataset_path", default="results/cage_clp1/datasets/closed_loop_contracts.jsonl")
    parser.add_argument("--action_supervised_path", default="results/cage_ecg/policy_alignment/action_supervised_contract_examples.jsonl")
    parser.add_argument("--out_jsonl", default="results/cage_ecg/policy_alignment/ecg_policy_alignment_v2.jsonl")
    parser.add_argument("--out_report", default="reports/stage36_ecg_policy_alignment_v2.md")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    graph = ContractGraph.load_json(args.contract_graph_path)
    contract_rows = load_jsonl(Path(args.contract_dataset_path))
    action_rows = load_jsonl(Path(args.action_supervised_path))
    examples = build_examples(graph, contract_rows, action_rows)
    write_jsonl(Path(args.out_jsonl), examples)
    summary = write_report(Path(args.out_report), examples)
    print(json.dumps({"out_jsonl": args.out_jsonl, **summary}, sort_keys=True))
    return 0


def build_examples(graph: ContractGraph, contract_rows: list[dict[str, Any]], action_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    contract_lookup = build_contract_lookup(contract_rows)
    action_lookup = build_action_lookup(action_rows)
    examples: list[dict[str, Any]] = []
    for edge in graph.edges.values():
        src = graph.nodes[edge.src]
        dst = graph.nodes[edge.dst]
        key = (str(src.env_name or ""), phi_key(src.center_phi), phi_key(dst.center_phi))
        ref = (contract_lookup.get(key) or [{}])[0]
        action = action_lookup.get(key, {})
        lcb = as_float(edge.contract_lcb, 0.0)
        negative = as_float(edge.predicted_negative_progress, 1.0)
        positive = bool(lcb >= 0.50 and negative <= 0.25)
        negative_label = bool(lcb < 0.25 or negative >= 0.50)
        action_available = bool(action.get("action_available"))
        objective_type = choose_objective(positive, negative_label, action_available, edge.edge_type)
        examples.append(
            {
                "env_name": src.env_name,
                "seed": ref.get("seed", action.get("seed")),
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
                "final_phase": bool(ref.get("final_phase") or "final" in str(edge.edge_type or "")),
                "recovery_candidate": bool(ref.get("recovery_candidate") or "recovery" in str(edge.edge_type or "")),
                "boundary_transition": False,
                "action_supervision_available": action_available,
                "action_available": action_available,
                "action_source": action.get("action_source"),
                "supervision_quality": action.get("supervision_quality", "missing"),
                "action": action.get("first_action"),
                "action_sequence": action.get("action_sequence"),
                "objective_type": objective_type,
                "recommended_objective": objective_type,
                "trainable_for_bc": bool(action_available and positive),
                "trainable_for_ranking": True,
                "trainable_for_contrastive": True,
                "trainable_for_conservative_filtering": bool(negative_label or not action_available),
                "curriculum_weight": curriculum_weight(lcb, negative),
            }
        )
    for boundary in graph.boundary_contracts.values():
        prev_edge = graph.edges.get(boundary.prev_edge_id)
        next_edge = graph.edges.get(boundary.next_edge_id)
        if prev_edge is None or next_edge is None:
            continue
        src = graph.nodes[prev_edge.src]
        dst = graph.nodes[next_edge.dst]
        negative_label = as_float(boundary.boundary_risk, 1.0) >= 0.50
        positive = as_float(boundary.compatibility_score, 0.0) >= 0.50 and not negative_label
        examples.append(
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
                "label_positive_contract": positive,
                "label_negative_contract": negative_label,
                "final_phase": False,
                "recovery_candidate": False,
                "boundary_transition": True,
                "action_supervision_available": False,
                "action_available": False,
                "action_source": None,
                "supervision_quality": "missing",
                "action": None,
                "action_sequence": None,
                "objective_type": "boundary_compatibility_ranking",
                "recommended_objective": "boundary_compatibility_ranking",
                "trainable_for_bc": False,
                "trainable_for_ranking": True,
                "trainable_for_contrastive": True,
                "trainable_for_conservative_filtering": True,
                "curriculum_weight": curriculum_weight(boundary.compatibility_score, boundary.boundary_risk),
            }
        )
    return examples


def choose_objective(positive: bool, negative: bool, action_available: bool, edge_type: str | None) -> str:
    if action_available and positive:
        return "bc_hard_positive"
    if "final" in str(edge_type or ""):
        return "final_goal_contract_ranking"
    if "recovery" in str(edge_type or ""):
        return "recovery_contract_filtering"
    if negative:
        return "ranking_contrastive_conservative_filtering"
    return "contract_ranking_or_curriculum"


def build_contract_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    lookup: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row.get("env_name") or ""), phi_key(row.get("phi_s", row.get("phi_start"))), phi_key(row.get("phi_g", row.get("phi_target"))))
        lookup[key].append(row)
    return lookup


def build_action_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("env_name") or ""), phi_key(row.get("phi_s")), phi_key(row.get("phi_g")))
        if row.get("action_available") or key not in lookup:
            lookup[key] = row
    return lookup


def write_report(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = len(rows)
    actions = sum(1 for row in rows if row.get("action_available"))
    positives = sum(1 for row in rows if row.get("label_positive_contract"))
    negatives = sum(1 for row in rows if row.get("label_negative_contract"))
    bc = sum(1 for row in rows if row.get("trainable_for_bc"))
    final = sum(1 for row in rows if row.get("final_phase"))
    recovery = sum(1 for row in rows if row.get("recovery_candidate"))
    objective_counts = Counter(str(row.get("objective_type") or "NA") for row in rows)
    summary = {
        "total_examples": total,
        "positive_contract_rate": rate(positives, total),
        "negative_contract_rate": rate(negatives, total),
        "action_supervision_rate": rate(actions, total),
        "bc_trainable_count": bc,
        "final_phase_rate": rate(final, total),
        "recovery_rate": rate(recovery, total),
        "objective_counts": dict(objective_counts),
        "policy_training_feasible": bool(bc > 0),
    }
    lines = [
        "# Stage36 ECG Policy Alignment Dataset v2",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| total_examples | {total} |",
        f"| positive_contract_rate | {fmt(summary['positive_contract_rate'])} |",
        f"| negative_contract_rate | {fmt(summary['negative_contract_rate'])} |",
        f"| action_supervision_rate | {fmt(summary['action_supervision_rate'])} |",
        f"| bc_trainable_count | {bc} |",
        f"| final_phase_rate | {fmt(summary['final_phase_rate'])} |",
        f"| recovery_rate | {fmt(summary['recovery_rate'])} |",
        "",
        "## Objective Counts",
        "",
        json.dumps(summary["objective_counts"], indent=2, sort_keys=True),
        "",
        "## Interpretation",
        "",
        "该数据集可用于 ranking / contrastive / conservative filtering。只有 `bc_trainable_count > 0` 时才可以规划 BC policy alignment。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def phi_key(phi: Any, decimals: int = 3) -> str:
    if phi is None:
        return ""
    arr = np.asarray(phi, dtype=np.float32).reshape(-1)
    return ",".join(f"{x:.{decimals}f}" for x in np.round(arr, decimals=decimals))


def as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def curriculum_weight(contract_lcb: Any, negative: Any) -> float:
    return max(0.0, min(1.0, as_float(contract_lcb, 0.0) * (1.0 - as_float(negative, 1.0))))


def rate(num: int, den: int) -> float | None:
    return num / den if den else None


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
