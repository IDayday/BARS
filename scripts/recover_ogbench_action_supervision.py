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
    parser = argparse.ArgumentParser(description="Recover action supervision for ECG contracts from offline OGBench-style datasets.")
    parser.add_argument("--contract_dataset_path", default="results/cage_clp1/datasets/closed_loop_contracts.jsonl")
    parser.add_argument("--contract_graph_path", default="results/cage_ecg/transition_contract_graph/contract_graph_augmented.json")
    parser.add_argument("--envs", nargs="+", default=["antmaze-giant-navigate-v0", "antmaze-giant-stitch-v0", "humanoidmaze-large-navigate-v0"])
    parser.add_argument("--dataset_cache_root", default="")
    parser.add_argument("--max_examples", type=int, default=50000)
    parser.add_argument("--phi_match_threshold", type=float, default=1e-4)
    parser.add_argument("--loose_phi_match_threshold", type=float, default=1e-2)
    parser.add_argument("--out_jsonl", default="results/cage_ecg/policy_alignment/ogbench_action_supervised_contract_examples.jsonl")
    parser.add_argument("--out_report", default="reports/stage37_ogbench_action_supervision_recovery.md")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    envs = list(args.envs or [])
    dataset_roots = dataset_roots_from_arg(args.dataset_cache_root)
    dataset_index = {env: load_env_datasets(env, dataset_roots) for env in envs}
    graph = ContractGraph.load_json(args.contract_graph_path) if Path(args.contract_graph_path).exists() else None
    contract_rows = load_jsonl(Path(args.contract_dataset_path))
    examples = recover_examples(contract_rows, graph, dataset_index, envs, args.max_examples, args.phi_match_threshold, args.loose_phi_match_threshold)
    write_jsonl(Path(args.out_jsonl), examples)
    summary = write_report(Path(args.out_report), examples, dataset_index, dataset_roots)
    print(json.dumps({"out_jsonl": args.out_jsonl, **summary}, sort_keys=True))
    return 0


def recover_examples(
    contract_rows: list[dict[str, Any]],
    graph: ContractGraph | None,
    dataset_index: dict[str, list[dict[str, Any]]],
    envs: list[str],
    max_examples: int,
    exact_threshold: float,
    loose_threshold: float,
) -> list[dict[str, Any]]:
    rows = candidate_rows(contract_rows, graph, envs, max_examples)
    examples: list[dict[str, Any]] = []
    for row in rows:
        env = str(row.get("env_name") or "")
        phi = row.get("phi_s", row.get("phi_start"))
        datasets = dataset_index.get(env) or []
        if not datasets:
            examples.append(output_row(row, action=None, source=None, distance=None, quality="env_or_dataset_unavailable", reason="no_dataset_npz_with_tdr_emb_actions"))
            continue
        match = match_action(phi, datasets, exact_threshold, loose_threshold)
        if match is None:
            examples.append(output_row(row, action=None, source=None, distance=None, quality="no_action_match", reason="no_phi_match_in_dataset"))
        else:
            action, distance, source, quality = match
            examples.append(output_row(row, action=action, source=source, distance=distance, quality=quality, reason=None))
    return examples


def candidate_rows(contract_rows: list[dict[str, Any]], graph: ContractGraph | None, envs: list[str], max_examples: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    allowed = set(envs)
    for row in contract_rows:
        if allowed and row.get("env_name") not in allowed:
            continue
        if not (row.get("label_contract_positive") or row.get("label_good_contract") or row.get("final_phase") or row.get("recovery_candidate")):
            continue
        rows.append(row)
        if len(rows) >= max_examples:
            return rows
    if graph is not None and len(rows) < max_examples:
        for edge in graph.edges.values():
            src = graph.nodes.get(edge.src)
            dst = graph.nodes.get(edge.dst)
            if src is None or dst is None:
                continue
            if allowed and src.env_name not in allowed:
                continue
            positive = as_float(edge.contract_lcb, 0.0) >= 0.5 and as_float(edge.predicted_negative_progress, 1.0) <= 0.25
            final = "final" in str(edge.edge_type or "")
            recovery = "recovery" in str(edge.edge_type or "")
            if not (positive or final or recovery):
                continue
            rows.append(
                {
                    "env_name": src.env_name,
                    "seed": None,
                    "phi_s": src.center_phi,
                    "phi_g": dst.center_phi,
                    "target_mode": edge.edge_type,
                    "contract_lcb": edge.contract_lcb,
                    "predicted_hit": edge.predicted_hit,
                    "predicted_negative_progress": edge.predicted_negative_progress,
                    "label_contract_positive": positive,
                    "label_contract_negative": bool(as_float(edge.contract_lcb, 0.0) < 0.25 or as_float(edge.predicted_negative_progress, 1.0) >= 0.5),
                    "final_phase": final,
                    "recovery_candidate": recovery,
                }
            )
            if len(rows) >= max_examples:
                return rows
    return rows


def match_action(phi: Any, datasets: list[dict[str, Any]], exact_threshold: float, loose_threshold: float) -> tuple[list[float], float, str, str] | None:
    if phi is None:
        return None
    target = np.asarray(phi, dtype=np.float32).reshape(1, -1)
    best: tuple[float, np.ndarray, str] | None = None
    for dataset in datasets:
        emb = dataset["tdr_emb"]
        if emb.shape[1] != target.shape[1]:
            continue
        dists = np.linalg.norm(emb - target, axis=1)
        idx = int(np.argmin(dists))
        dist = float(dists[idx])
        if best is None or dist < best[0]:
            best = (dist, dataset["actions"][idx], dataset["path"])
    if best is None:
        return None
    dist, action, source = best
    if dist <= exact_threshold:
        quality = "exact_dataset_action"
    elif dist <= loose_threshold:
        quality = "loose_dataset_action"
    else:
        return None
    return np.asarray(action, dtype=float).reshape(-1).tolist(), dist, source, quality


def output_row(row: dict[str, Any], *, action: Any, source: str | None, distance: float | None, quality: str, reason: str | None) -> dict[str, Any]:
    available = action is not None
    positive = bool(row.get("label_contract_positive") or row.get("label_good_contract") or row.get("hit"))
    negative = bool(row.get("label_contract_negative") or row.get("negative_progress"))
    final = bool(row.get("final_phase") or "final" in str(row.get("target_mode") or ""))
    recovery = bool(row.get("recovery_candidate") or "recovery" in str(row.get("target_mode") or ""))
    return {
        "env_name": row.get("env_name"),
        "seed": row.get("seed"),
        "phi_s": row.get("phi_s", row.get("phi_start")),
        "phi_g": row.get("phi_g", row.get("phi_target")),
        "target_mode": row.get("target_mode", row.get("pair_source")),
        "contract_lcb": row.get("contract_lcb"),
        "predicted_hit": row.get("predicted_hit"),
        "predicted_negative_progress": row.get("predicted_negative_progress"),
        "label_positive_contract": positive,
        "label_negative_contract": negative,
        "final_phase": final,
        "recovery_candidate": recovery,
        "action_available": available,
        "action": action,
        "action_source": source,
        "match_distance": distance,
        "match_quality": quality,
        "trainable_for_bc": bool(available and positive and "knn" not in str(row.get("target_mode") or "")),
        "trainable_for_q_filtering": True,
        "trainable_for_contrastive": True,
        "missing_reason": reason,
    }


def dataset_roots_from_arg(root_arg: str) -> list[Path]:
    roots = []
    if root_arg:
        roots.append(Path(root_arg))
    roots.extend([Path("artifacts/stage27_gas/datasets"), Path("_data/ogbench")])
    return list(dict.fromkeys(roots))


def load_env_datasets(env: str, roots: list[Path]) -> list[dict[str, Any]]:
    datasets: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        paths = list(root.glob(f"{env}/**/dataset.npz")) + list(root.glob(f"**/{env}/**/dataset.npz"))
        for path in sorted(set(paths)):
            try:
                data = np.load(path, allow_pickle=True)
                if "tdr_emb" not in data.files or "actions" not in data.files:
                    continue
                emb = np.asarray(data["tdr_emb"], dtype=np.float32)
                actions = np.asarray(data["actions"], dtype=np.float32)
                if emb.ndim != 2 or len(emb) != len(actions):
                    continue
                datasets.append({"path": str(path), "tdr_emb": emb, "actions": actions})
            except Exception:
                continue
    return datasets


def write_report(path: Path, rows: list[dict[str, Any]], dataset_index: dict[str, list[dict[str, Any]]], roots: list[Path]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = len(rows)
    exact = sum(1 for row in rows if row.get("match_quality") == "exact_dataset_action")
    loose = sum(1 for row in rows if row.get("match_quality") == "loose_dataset_action")
    actions = sum(1 for row in rows if row.get("action_available"))
    positives_with_action = sum(1 for row in rows if row.get("label_positive_contract") and row.get("action_available"))
    final_with_action = sum(1 for row in rows if row.get("final_phase") and row.get("action_available"))
    recovery_with_action = sum(1 for row in rows if row.get("recovery_candidate") and row.get("action_available"))
    missing = Counter(str(row.get("missing_reason") or row.get("match_quality")) for row in rows)
    per_env = {}
    for env in sorted({str(row.get("env_name")) for row in rows} | set(dataset_index)):
        env_rows = [row for row in rows if str(row.get("env_name")) == env]
        per_env[env] = {
            "dataset_files": len(dataset_index.get(env, [])),
            "examples": len(env_rows),
            "action_available": sum(1 for row in env_rows if row.get("action_available")),
            "action_supervision_rate": rate(sum(1 for row in env_rows if row.get("action_available")), len(env_rows)),
        }
    status = "ok" if actions else "BLOCKED_NO_ACTION_MATCH"
    summary = {
        "status": status,
        "dataset_roots": [str(root) for root in roots],
        "total_examples": total,
        "exact_action_count": exact,
        "loose_action_count": loose,
        "action_supervision_rate": rate(actions, total),
        "positive_with_action_count": positives_with_action,
        "final_goal_with_action_count": final_with_action,
        "recovery_with_action_count": recovery_with_action,
        "missing_reason_counts": dict(missing),
        "per_env": per_env,
    }
    lines = [
        "# Stage37 OGBench Action Supervision Recovery",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| status | {status} |",
        f"| total_examples | {total} |",
        f"| exact_action_count | {exact} |",
        f"| loose_action_count | {loose} |",
        f"| action_supervision_rate | {fmt(summary['action_supervision_rate'])} |",
        f"| positive_with_action_count | {positives_with_action} |",
        f"| final_goal_with_action_count | {final_with_action} |",
        f"| recovery_with_action_count | {recovery_with_action} |",
        "",
        "## Per-Env",
        "",
        json.dumps(per_env, indent=2, sort_keys=True),
        "",
        "## Missing Reason Counts",
        "",
        json.dumps(summary["missing_reason_counts"], indent=2, sort_keys=True),
        "",
        "如果 action_supervision_rate 为 0，则 BC policy alignment 继续 BLOCKED；不能从 phi-only 样本虚构 action。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


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
