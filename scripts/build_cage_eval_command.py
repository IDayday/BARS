#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import Any


DEFAULT_ENVS = [
    "antmaze-giant-navigate-v0",
    "antmaze-giant-stitch-v0",
    "humanoidmaze-medium-navigate-v0",
    "humanoidmaze-large-navigate-v0",
    "humanoidmaze-large-stitch-v0",
    "scene-play-v0",
    "kitchen-partial-v0",
    "visual-antmaze-giant-stitch-v0",
]

SUPPORTED_VARIANTS = {
    "gas",
    "cage_trace_only",
    "cage_fixed_commit",
    "cage_drift_only",
    "cage_recovery_only",
    "cage_full",
    "cage_safe_full",
    "cage_contract_commit",
    "cage_contract_rank",
}

UNSUPPORTED_VARIANTS = {
    "cage_reachability": "current evaluator has no learned reachability model loader",
    "cage_risk_path": "current evaluator has no risk-aware path executor",
}


class UnsupportedVariant(ValueError):
    pass


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def artifact_env_name(env_name: str) -> str:
    return env_name[:-3] if env_name.endswith("-v0") else env_name


def default_checkpoint_epoch(env_name: str) -> int:
    low = env_name.lower()
    if low.startswith("visual-") or "kitchen" in low:
        return 500000
    return 1000000


def _latest_params(root: Path) -> Path | None:
    matches = sorted(root.glob("params_*.pkl"))
    return matches[-1] if matches else None


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def infer_checkpoint_paths(checkpoint_root: str | Path, env_name: str, seed: int) -> dict[str, str | None]:
    root = Path(checkpoint_root)
    env_alias = artifact_env_name(env_name)
    epoch = default_checkpoint_epoch(env_name)
    seed_names = [f"seed{seed}", f"seed_{seed}", str(seed)]
    candidates: list[Path] = []
    for env_dir in [env_name, env_alias]:
        for seed_dir in seed_names:
            candidates.append(root / env_dir / seed_dir)
        candidates.append(root / env_dir)
    for seed_dir in seed_names:
        candidates.append(root / f"{env_alias}_{seed_dir}")
        candidates.append(root / f"{env_name}_{seed_dir}")

    existing_root = _first_existing(candidates)
    base = existing_root if existing_root is not None else candidates[0]
    policy = _first_existing([base / "policy" / f"params_{epoch}.pkl", base / f"params_{epoch}.pkl"])
    if policy is None and (base / "policy").exists():
        policy = _latest_params(base / "policy")
    if policy is None:
        policy = base / "policy" / f"params_{epoch}.pkl"

    tdr = _first_existing([base / "tdr" / f"params_{epoch}.pkl", base / f"tdr_params_{epoch}.pkl"])
    if tdr is None and (base / "tdr").exists():
        tdr = _latest_params(base / "tdr")
    if tdr is None:
        tdr = base / "tdr" / f"params_{epoch}.pkl"

    keygraph = _first_existing([base / "graph" / "keygraph.pkl", base / "keygraph.pkl"])
    if keygraph is None:
        keygraph = base / "graph" / "keygraph.pkl"

    return {
        "keygraph_path": str(keygraph) if keygraph is not None else None,
        "policy_path": str(policy) if policy is not None else None,
        "tdr_path": str(tdr) if tdr is not None else None,
    }


def validate_variant(variant: str, reachability_path: str | None = None) -> None:
    if variant in SUPPORTED_VARIANTS:
        return
    reason = UNSUPPORTED_VARIANTS.get(variant)
    if reason is None:
        reason = f"unknown CAGE experiment variant: {variant}"
    raise UnsupportedVariant(reason)


def variant_args(variant: str, reachability_path: str | None = None) -> list[str]:
    validate_variant(variant, reachability_path=reachability_path)
    if variant == "gas":
        return []
    args = ["--use_cage"]
    if variant == "cage_trace_only":
        args.extend(["--cage_trace_only"])
    elif variant == "cage_fixed_commit":
        args.extend(
            [
                "--cage_disable_drift_monitor",
                "--cage_disable_recovery",
                "--cage_disable_adaptive_horizon",
                "--cage_disable_final_phase_controller",
            ]
        )
    elif variant == "cage_drift_only":
        args.extend(
            [
                "--cage_disable_commitment=false",
                "--cage_disable_recovery",
                "--cage_disable_adaptive_horizon",
                "--cage_disable_final_phase_controller",
            ]
        )
    elif variant == "cage_recovery_only":
        args.extend(["--cage_disable_adaptive_horizon", "--cage_disable_final_phase_controller"])
    elif variant == "cage_full":
        pass
    elif variant == "cage_safe_full":
        args.extend(
            [
                "--cage_enable_churn_guard",
                "--cage_replan_cooldown_steps",
                "10",
                "--cage_max_global_replans_per_episode",
                "50",
                "--cage_max_replans_per_100_steps",
                "10",
                "--cage_max_consecutive_replan_requests",
                "5",
                "--cage_fallback_to_gas_on_churn",
                "--cage_fallback_to_gas_steps",
                "50",
                "--cage_recovery_lockout_steps_after_failure",
                "25",
                "--cage_min_steps_between_recovery_attempts",
                "20",
                "--cage_min_progress_for_recovery_success",
                "0.0001",
                "--cage_disable_recovery_after_churn",
                "--cage_log_churn_events",
            ]
        )
    elif variant == "cage_contract_commit":
        args.extend(
            [
                "--cage_contract_commit",
                "--cage_enable_churn_guard",
                "--cage_replan_cooldown_steps",
                "10",
                "--cage_max_global_replans_per_episode",
                "50",
                "--cage_max_replans_per_100_steps",
                "10",
                "--cage_max_consecutive_replan_requests",
                "5",
                "--cage_fallback_to_gas_on_churn",
                "--cage_fallback_to_gas_steps",
                "50",
                "--cage_disable_recovery_after_churn",
                "--cage_log_churn_events",
            ]
        )
    elif variant == "cage_contract_rank":
        args.extend(
            [
                "--cage_contract_rank",
                "--cage_enable_churn_guard",
                "--cage_replan_cooldown_steps",
                "10",
                "--cage_max_global_replans_per_episode",
                "50",
                "--cage_max_replans_per_100_steps",
                "10",
                "--cage_max_consecutive_replan_requests",
                "5",
                "--cage_fallback_to_gas_on_churn",
                "--cage_fallback_to_gas_steps",
                "50",
                "--cage_disable_recovery_after_churn",
                "--cage_log_churn_events",
            ]
        )
    return args


def build_eval_command(row: dict[str, Any]) -> list[str]:
    variant = str(row["variant"])
    validate_variant(variant, reachability_path=row.get("cage_reachability_path"))
    keygraph_path = row.get("keygraph_path")
    policy_path = row.get("policy_path")
    if not keygraph_path or not policy_path:
        raise ValueError(f"job {row.get('job_id')} is missing keygraph_path or policy_path")

    command = [
        "python",
        "evaluate_gas.py",
        "--run_eval_project",
        str(row.get("run_eval_project", "CAGEFocused")),
        "--run_group",
        str(row.get("run_group", f"cage_{safe_id(row['env_name'])}_{variant}_seed{row['seed']}")),
        "--env_name",
        str(row["env_name"]),
        "--seed",
        str(row["seed"]),
        "--gpu",
        str(row.get("gpu", 0)),
        "--save_eval_dir",
        str(row.get("save_eval_dir", Path(row["output_root"]) / "eval_logs")),
        "--eval_on_cpu",
        str(row.get("eval_on_cpu", 1)),
        "--eval_episodes",
        str(row["episodes_per_goal"]),
        "--eval_video_episodes",
        str(row.get("eval_video_episodes", 0)),
        "--eval_final_goal_threshold",
        str(row.get("eval_final_goal_threshold", 2)),
        "--eval_max_tasks",
        str(row["goals_per_env"]),
        "--eval_result_path",
        str(row["result_path"]),
        "--keygraph_path",
        str(keygraph_path),
        "--policy_path",
        str(policy_path),
    ]
    if variant != "gas":
        command.extend(["--cage_trace_path", str(row["cage_trace_path"])])
    command.extend(variant_args(variant, reachability_path=row.get("cage_reachability_path")))
    if variant != "gas" and row.get("cage_debug", False):
        command.append("--cage_debug")
    if variant != "gas" and row.get("cage_debug_light", False):
        command.append("--cage_debug_light")
    if variant != "gas" and row.get("cage_disable_exact_state_ref_trace", False):
        command.append("--cage_disable_exact_state_ref_trace")
    if variant != "gas" and int(row.get("cage_max_debug_steps_per_episode", 0) or 0) > 0:
        command.extend(["--cage_max_debug_steps_per_episode", str(int(row["cage_max_debug_steps_per_episode"]))])
    if variant != "gas" and row.get("cage_trace_phi_vectors") is False:
        command.append("--cage_trace_phi_vectors=false")
    if variant != "gas" and row.get("cage_contract_rank_debug_candidates", False):
        command.append("--cage_contract_rank_debug_candidates")
    if variant != "gas" and row.get("cage_contract_model_path"):
        command.extend(["--cage_use_contract_model", "--cage_contract_model_path", str(row["cage_contract_model_path"])])
    if row.get("contract_trace_path"):
        command.extend(["--contract_trace_path", str(row["contract_trace_path"])])
        if row.get("store_contract_state_refs", False):
            command.append("--store_contract_state_refs")
        if row.get("contract_state_ref_mode"):
            command.extend(["--contract_state_ref_mode", str(row["contract_state_ref_mode"])])
        if row.get("contract_capture_variants"):
            capture_variants = row["contract_capture_variants"]
            if isinstance(capture_variants, (list, tuple)):
                capture_variants = ",".join(str(x) for x in capture_variants)
            command.extend(["--contract_capture_variants", str(capture_variants)])
        if row.get("contract_trace_debug", False):
            command.append("--contract_trace_debug")
    return command


def command_to_string(command: list[str] | None) -> str | None:
    if command is None:
        return None
    return shlex.join(command)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build one evaluate_gas.py command from a manifest row JSON.")
    parser.add_argument("--row_json", required=True, help="Manifest row as JSON.")
    args = parser.parse_args()
    row = json.loads(args.row_json)
    print(command_to_string(build_eval_command(row)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
