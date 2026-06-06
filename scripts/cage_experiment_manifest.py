#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from build_cage_eval_command import (
    DEFAULT_ENVS,
    UnsupportedVariant,
    build_eval_command,
    command_to_string,
    infer_checkpoint_paths,
    safe_id,
)


DEFAULT_VARIANTS = ["gas", "cage_fixed_commit", "cage_drift_only", "cage_recovery_only", "cage_full"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a JSONL manifest for focused GAS/CAGE experiments.")
    parser.add_argument("--checkpoint_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--envs", nargs="+", default=DEFAULT_ENVS)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--episodes_per_goal", type=int, required=True)
    parser.add_argument("--goals_per_env", type=int, required=True)
    parser.add_argument("--eval_horizon", default="default")
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--strict_paths", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cage_debug", action="store_true", help="Pass --cage_debug to CAGE variants.")
    parser.add_argument("--cage_debug_light", action="store_true", help="Pass --cage_debug_light to CAGE variants.")
    parser.add_argument("--cage_disable_exact_state_ref_trace", action="store_true", help="Disable exact StateRef payloads in CAGE debug traces.")
    parser.add_argument("--cage_max_debug_steps_per_episode", type=int, default=0)
    parser.add_argument("--cage_trace_phi_vectors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cage_contract_rank_debug_candidates", action="store_true")
    parser.add_argument("--cage_contract_shadow_debug_candidates", action="store_true")
    parser.add_argument("--cage_contract_model_path", default="", help="Optional contract model JSON for cage_contract_commit.")
    parser.add_argument("--ecg_graph_path", default="", help="Action-anchored ECG graph JSON for ECG variants.")
    parser.add_argument("--ecg_contract_model_path", default="", help="Action-anchored ECG contract model path.")
    parser.add_argument("--ecg_policy_adapter_path", default="", help="ECG policy adapter path.")
    parser.add_argument("--ecg_planner_score_path", default="", help="ECG planner score weights path.")
    return parser


def missing_checkpoint(paths: dict[str, str | None]) -> list[str]:
    missing: list[str] = []
    for key in ["keygraph_path", "policy_path"]:
        value = paths.get(key)
        if not value or not Path(value).exists():
            missing.append(key)
    return missing


def make_row(args: argparse.Namespace, env_name: str, seed: int, variant: str) -> dict[str, Any]:
    output_root = Path(args.output_root).resolve()
    checkpoint_root = Path(args.checkpoint_root).resolve()
    safe_env = safe_id(env_name)
    safe_variant = safe_id(variant)
    job_id = f"{safe_env}__seed{seed}__{safe_variant}"
    job_root = output_root / "jobs" / safe_env / f"seed{seed}" / safe_variant
    trace_path = job_root / "cage_trace.jsonl"
    result_path = job_root / "eval.csv"
    paths = infer_checkpoint_paths(checkpoint_root, env_name, seed)
    row: dict[str, Any] = {
        "job_id": job_id,
        "env_name": env_name,
        "seed": seed,
        "variant": variant,
        "checkpoint_root": str(checkpoint_root),
        "output_root": str(output_root),
        "keygraph_path": paths["keygraph_path"],
        "policy_path": paths["policy_path"],
        "tdr_path": paths["tdr_path"],
        "cage_trace_path": str(trace_path),
        "result_path": str(result_path),
        "episodes_per_goal": int(args.episodes_per_goal),
        "goals_per_env": int(args.goals_per_env),
        "eval_horizon": str(args.eval_horizon),
        "gpu": int(args.gpu),
        "cage_debug": bool(args.cage_debug),
        "cage_debug_light": bool(args.cage_debug_light),
        "cage_disable_exact_state_ref_trace": bool(args.cage_disable_exact_state_ref_trace or args.cage_debug_light),
        "cage_max_debug_steps_per_episode": int(args.cage_max_debug_steps_per_episode),
        "cage_trace_phi_vectors": bool(args.cage_trace_phi_vectors),
        "cage_contract_rank_debug_candidates": bool(args.cage_contract_rank_debug_candidates),
        "cage_contract_shadow_debug_candidates": bool(args.cage_contract_shadow_debug_candidates),
        "status": "initialized",
    }
    if args.cage_contract_model_path:
        row["cage_contract_model_path"] = str(Path(args.cage_contract_model_path).resolve())
    for attr in ["ecg_graph_path", "ecg_contract_model_path", "ecg_policy_adapter_path", "ecg_planner_score_path"]:
        value = getattr(args, attr, "")
        if value:
            row[attr] = str(Path(value).resolve())
    if args.eval_horizon != "default":
        row["status"] = "unsupported_horizon"
        row["error"] = "evaluate_gas.py currently supports the environment default horizon only"
    elif args.strict_paths:
        missing = missing_checkpoint(paths)
        if variant.startswith("cage_ecg_"):
            for attr in ["ecg_graph_path", "ecg_planner_score_path"]:
                value = row.get(attr)
                if not value or not Path(value).exists():
                    missing.append(attr)
            if variant == "cage_ecg_adapter":
                value = row.get("ecg_policy_adapter_path")
                if not value or not Path(value).exists():
                    missing.append("ecg_policy_adapter_path")
        if missing:
            row["status"] = "missing_checkpoint"
            row["missing_checkpoint_fields"] = missing

    try:
        command = build_eval_command(row)
    except UnsupportedVariant as exc:
        row["status"] = "unsupported_variant"
        row["error"] = str(exc)
        row["command"] = None
        row["command_string"] = None
    except ValueError as exc:
        if row["status"] == "initialized":
            row["status"] = "missing_checkpoint"
        row["error"] = str(exc)
        row["command"] = None
        row["command_string"] = None
    else:
        row["command"] = command
        row["command_string"] = command_to_string(command)
    return row


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> int:
    args = build_parser().parse_args()
    rows = [
        make_row(args, env_name, seed, variant)
        for env_name in args.envs
        for seed in args.seeds
        for variant in args.variants
    ]
    write_jsonl(Path(args.manifest_path), rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(json.dumps({"manifest_path": args.manifest_path, "num_jobs": len(rows), "status_counts": counts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
