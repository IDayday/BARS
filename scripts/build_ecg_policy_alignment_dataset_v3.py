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
    parser = argparse.ArgumentParser(description="Build trusted ECG policy-alignment dataset v3.")
    parser.add_argument("--contract_graph_path", default="results/cage_ecg/trusted_graph/trusted_conservative/contract_graph.json")
    parser.add_argument("--contract_dataset_path", default="results/cage_clp1/datasets/closed_loop_contracts.jsonl")
    parser.add_argument("--action_supervised_path", default="results/cage_ecg/policy_alignment/ogbench_action_supervised_contract_examples.jsonl")
    parser.add_argument("--out_jsonl", default="results/cage_ecg/policy_alignment/ecg_policy_alignment_v3.jsonl")
    parser.add_argument("--out_report", default="reports/stage37_ecg_policy_alignment_v3.md")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    graph = ContractGraph.load_json(args.contract_graph_path)
    contract_rows = load_jsonl(Path(args.contract_dataset_path))
    action_rows = load_jsonl(Path(args.action_supervised_path))
    examples = build_examples(graph, contract_rows, action_rows)
    write_jsonl(Path(args.out_jsonl), examples)
    summary = write_report(Path(args.out_report), examples, Path(args.out_jsonl))
    maybe_write_bc_split(Path(args.out_jsonl).parent, examples)
    print(json.dumps({"out_jsonl": args.out_jsonl, **summary}, sort_keys=True))
    return 0


def build_examples(graph: ContractGraph, contract_rows: list[dict[str, Any]], action_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    contract_lookup = build_contract_lookup(contract_rows)
    action_lookup = build_action_lookup(action_rows)
    graph_variant = str(graph.metadata.get("trusted_graph_variant") or Path("trusted_conservative").name)
    out = []
    for edge in graph.edges.values():
        src = graph.nodes[edge.src]
        dst = graph.nodes[edge.dst]
        key = (str(src.env_name or ""), phi_key(src.center_phi), phi_key(dst.center_phi))
        ref = (contract_lookup.get(key) or [{}])[0]
        action = action_lookup.get(key, {})
        trust = edge_trust_level(edge.edge_type)
        lcb = as_float(edge.contract_lcb, 0.0)
        negative = as_float(edge.predicted_negative_progress, 1.0)
        positive = bool(lcb >= 0.50 and negative <= 0.25)
        negative_label = bool(lcb < 0.25 or negative >= 0.50)
        action_available = bool(action.get("action_available"))
        trainable_for_bc = bool(action_available and positive and trust != "candidate_knn")
        stage = recommended_stage(edge.edge_type, trust, positive, negative_label, trainable_for_bc)
        out.append(
            {
                "env_name": src.env_name,
                "seed": ref.get("seed", action.get("seed")),
                "graph_variant": graph_variant,
                "source_edge_id": edge.edge_id,
                "edge_trust_level": trust,
                "edge_type": edge.edge_type,
                "phi_s": src.center_phi,
                "phi_g": dst.center_phi,
                "target_mode": edge.edge_type,
                "contract_lcb": edge.contract_lcb,
                "predicted_hit": edge.predicted_hit,
                "predicted_negative_progress": edge.predicted_negative_progress,
                "uncertainty": edge.uncertainty,
                "q_train_support": edge.q_train_support,
                "label_positive_contract": positive,
                "label_negative_contract": negative_label,
                "final_phase": bool(ref.get("final_phase") or "final" in str(edge.edge_type or "")),
                "recovery_candidate": bool(ref.get("recovery_candidate") or "recovery" in str(edge.edge_type or "")),
                "action_available": action_available,
                "action": action.get("action"),
                "action_source": action.get("action_source"),
                "match_quality": action.get("match_quality"),
                "trainable_for_bc": trainable_for_bc,
                "trainable_for_q_filtering": True,
                "trainable_for_contrastive": True,
                "trainable_for_conservative_filtering": bool(negative_label or trust == "candidate_knn" or not trainable_for_bc),
                "recommended_training_stage": stage,
            }
        )
    return out


def recommended_stage(edge_type: str | None, trust: str, positive: bool, negative: bool, trainable_for_bc: bool) -> str:
    if trainable_for_bc:
        return "bc_candidate"
    if "final" in str(edge_type or ""):
        return "final_goal_contract_ranking"
    if "recovery" in str(edge_type or ""):
        return "recovery_underpowered_filtering"
    if trust == "candidate_knn":
        return "trusted_bridge_conservative_filtering"
    if negative:
        return "negative_contract_conservative_filtering"
    if positive:
        return "positive_contract_ranking"
    return "contract_ranking_contrastive"


def edge_trust_level(edge_type: str | None) -> str:
    text = str(edge_type or "")
    if text in {"original_contract", "temporal_transition", "path_adjacency", "qtrain_supported"}:
        return "observed"
    if text == "final_goal_candidate":
        return "final_candidate"
    if text == "knn_bridge_candidate":
        return "candidate_knn"
    if "recovery" in text:
        return "recovery_candidate"
    return "unknown"


def write_report(path: Path, rows: list[dict[str, Any]], out_jsonl: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = len(rows)
    actions = sum(1 for row in rows if row.get("action_available"))
    positives_with_action = sum(1 for row in rows if row.get("label_positive_contract") and row.get("action_available"))
    final_with_action = sum(1 for row in rows if row.get("final_phase") and row.get("action_available"))
    recovery_with_action = sum(1 for row in rows if row.get("recovery_candidate") and row.get("action_available"))
    bc = sum(1 for row in rows if row.get("trainable_for_bc"))
    summary = {
        "total_examples": total,
        "action_supervision_rate": rate(actions, total),
        "positive_with_action_count": positives_with_action,
        "final_goal_with_action_count": final_with_action,
        "recovery_with_action_count": recovery_with_action,
        "bc_trainable_count": bc,
        "trust_counts": dict(Counter(str(row.get("edge_trust_level")) for row in rows)),
        "training_stage_counts": dict(Counter(str(row.get("recommended_training_stage")) for row in rows)),
        "bc_split_generated": bool(bc > 0),
        "out_jsonl": str(out_jsonl),
    }
    lines = [
        "# Stage37 ECG Policy Alignment Dataset v3",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| total_examples | {total} |",
        f"| action_supervision_rate | {fmt(summary['action_supervision_rate'])} |",
        f"| positive_with_action_count | {positives_with_action} |",
        f"| final_goal_with_action_count | {final_with_action} |",
        f"| recovery_with_action_count | {recovery_with_action} |",
        f"| bc_trainable_count | {bc} |",
        f"| bc_split_generated | {summary['bc_split_generated']} |",
        "",
        "## Trust Counts",
        "",
        json.dumps(summary["trust_counts"], indent=2, sort_keys=True),
        "",
        "## Training Stage Counts",
        "",
        json.dumps(summary["training_stage_counts"], indent=2, sort_keys=True),
        "",
        "KNN bridge candidate 只能进入 conservative filtering 或 ranking，不能直接 BC。BC split 只有在 positive_with_action_count > 0 时生成。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def maybe_write_bc_split(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    bc_rows = [row for row in rows if row.get("trainable_for_bc")]
    if not bc_rows:
        return
    train, val, test = [], [], []
    for idx, row in enumerate(bc_rows):
        bucket = idx % 10
        if bucket < 8:
            train.append(row)
        elif bucket == 8:
            val.append(row)
        else:
            test.append(row)
    write_jsonl(out_dir / "ecg_policy_alignment_v3_bc_train.jsonl", train)
    write_jsonl(out_dir / "ecg_policy_alignment_v3_bc_val.jsonl", val)
    write_jsonl(out_dir / "ecg_policy_alignment_v3_bc_test.jsonl", test)
    summary = {"train": len(train), "val": len(val), "test": len(test)}
    (out_dir / "ecg_policy_alignment_v3_split_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_contract_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    lookup: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        lookup[(str(row.get("env_name") or ""), phi_key(row.get("phi_s", row.get("phi_start"))), phi_key(row.get("phi_g", row.get("phi_target"))))].append(row)
    return lookup


def build_action_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    lookup = {}
    for row in rows:
        key = (str(row.get("env_name") or ""), phi_key(row.get("phi_s")), phi_key(row.get("phi_g")))
        if row.get("action_available") or key not in lookup:
            lookup[key] = row
    return lookup


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
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
    try:
        if value is not None:
            return float(value)
    except (TypeError, ValueError):
        pass
    return default


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
