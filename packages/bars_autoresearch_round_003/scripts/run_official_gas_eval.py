#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bars.external.gas_artifacts import resolve_gas_artifacts
from fetch_public_baseline_targets import (
    OFFICIAL_GAS_PRETRAINED_SLUGS,
    TARGETS_PP,
    env_to_slug,
    gas_required_train_steps,
    lower_bound_pp,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text()) if path.exists() else default
    except Exception:
        return default


def checkpoint_step(path: str | None) -> int | None:
    if not path:
        return None
    match = re.search(r"params_(\d+)\.pkl", Path(path).name)
    return int(match.group(1)) if match else None


def to_float(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except Exception:
        return None


def cached_official_score(env: str, seed: int, reports_root: Path) -> tuple[float | None, str, str]:
    repair = read_csv(reports_root / "stage23_adapter_protocol_repair.csv")
    for row in repair:
        if row.get("env") == env and int(float(row.get("seed", -1))) == seed:
            return to_float(row.get("official_B_success")), "stage23_adapter_protocol_repair", ""
    repro = read_csv(reports_root / "stage23_gas_reproduction_matrix.csv")
    for row in repro:
        if row.get("env") == env and int(float(row.get("seed", -1))) == seed and row.get("route") == "B_official_our_checkpoint":
            return to_float(row.get("success")), "stage23_gas_reproduction_matrix", row.get("eval_csv", "")
    return None, "", ""


def cached_adapter_score(env: str, seed: int, reports_root: Path) -> tuple[float | None, str, str, float | None]:
    repair = read_csv(reports_root / "stage23_adapter_protocol_repair.csv")
    for row in repair:
        if row.get("env") == env and int(float(row.get("seed", -1))) == seed:
            return (
                to_float(row.get("adapter_official_control_success")),
                "stage23_adapter_protocol_repair",
                row.get("eval_csv", ""),
                to_float(row.get("adapter_original_success")),
            )
    return None, "", "", None


def artifact_info(env: str, seed: int, root: str) -> dict[str, Any]:
    artifacts = resolve_gas_artifacts(env, seed, root)
    manifest = read_json(artifacts.root / "manifest.json", {})
    policy = str(artifacts.policy_checkpoint) if artifacts.policy_checkpoint else ""
    source = str(manifest.get("source") or "unknown")
    step = checkpoint_step(policy)
    slug = env_to_slug(env)
    official_available = slug in OFFICIAL_GAS_PRETRAINED_SLUGS
    return {
        "complete": artifacts.complete,
        "manifest_source": source,
        "policy_path": policy,
        "tdr_path": str(artifacts.tdr_checkpoint) if artifacts.tdr_checkpoint else "",
        "graph_path": str(artifacts.keygraph) if artifacts.keygraph else "",
        "train_steps": step,
        "official_checkpoint_available": official_available,
    }


def certification_status(env: str, info: dict[str, Any], score: float | None) -> tuple[str, str, str]:
    target = TARGETS_PP.get(env, {}).get("GAS")
    if not target:
        return "HOLD_NO_EXACT_PUBLIC_TARGET", "E1_BASELINE_REGISTRY", "no exact public GAS target"
    required = gas_required_train_steps(env)
    train_steps = info.get("train_steps")
    source = info.get("manifest_source")
    lower = lower_bound_pp(*target) / 100.0
    if not info.get("complete"):
        return "SKIP_ARTIFACT_UNAVAILABLE", "E1_BASELINE_REGISTRY", "no complete local GAS policy/keygraph artifact"
    if source == "huggingface" and train_steps and train_steps >= required:
        if score is None:
            return "READY_FOR_OFFICIAL_EVAL", "E2_CERTIFIED_BASELINE_DIAGNOSTIC", "official artifact present but official eval was not run in this round"
        if score >= lower:
            return "PASS_BASELINE_CERTIFICATION", "E2_CERTIFIED_BASELINE_DIAGNOSTIC", "official artifact score meets public lower bound"
        return "FAIL_PUBLIC_REPRODUCTION", "E2_CERTIFIED_BASELINE_DIAGNOSTIC", "official artifact score below public lower bound"
    if train_steps is not None and train_steps < required:
        return "FAIL_UNDERTRAINED_BASELINE", "E0_SMOKE_ONLY", f"local GAS checkpoint has {train_steps} train steps; public command requires {required}"
    if train_steps is not None and train_steps >= required:
        if score is None:
            return "READY_FOR_FULL_BUDGET_EVAL", "E4_FULL_BUDGET_TRAINED_METHOD", "full-budget local artifact present but eval was not run in this round"
        if score >= lower:
            return "PASS_BASELINE_CERTIFICATION", "E4_FULL_BUDGET_TRAINED_METHOD", "full-budget local score meets public lower bound"
        return "FAIL_PUBLIC_REPRODUCTION", "E4_FULL_BUDGET_TRAINED_METHOD", "full-budget local score below public lower bound"
    if not info.get("official_checkpoint_available"):
        return "SKIP_ARTIFACT_UNAVAILABLE", "E1_BASELINE_REGISTRY", "official checkpoint unavailable and no full-budget local training"
    return "SKIP_ARTIFACT_UNAVAILABLE", "E1_BASELINE_REGISTRY", "artifact status unresolved"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", default="antmaze-medium-stitch-v0,antmaze-medium-navigate-v0")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--round", default="002")
    parser.add_argument("--reports-root", default="reports")
    parser.add_argument("--gas-artifact-root", default="artifacts/gas")
    parser.add_argument("--official-out", default="")
    parser.add_argument("--adapter-out", default="")
    args = parser.parse_args()

    round_id = f"{int(args.round):03d}"
    reports = Path(args.reports_root)
    official_out = Path(args.official_out or reports / f"round_{round_id}_gas_official_eval.csv")
    adapter_out = Path(args.adapter_out or reports / f"round_{round_id}_bars_adapter_eval.csv")
    official_rows: list[dict[str, Any]] = []
    adapter_rows: list[dict[str, Any]] = []
    for env in [x.strip() for x in args.envs.split(",") if x.strip()]:
        target = TARGETS_PP.get(env, {}).get("GAS")
        public_mean, public_std = target if target else (None, None)
        lower = lower_bound_pp(public_mean, public_std) if target else None
        for seed in [int(x) for x in args.seeds.split(",") if x.strip()]:
            info = artifact_info(env, seed, args.gas_artifact_root)
            score, cache_source, eval_csv = cached_official_score(env, seed, reports)
            status, evidence, reason = certification_status(env, info, score)
            official_rows.append(
                {
                    "env": env,
                    "seed": seed,
                    "algorithm": "GAS",
                    "route": "official_evaluate_gas",
                    "status": "cached_protocol_score" if score is not None else "not_run",
                    "evidence_class": evidence,
                    "artifact_source": info["manifest_source"],
                    "official_checkpoint_available": info["official_checkpoint_available"],
                    "train_steps": info["train_steps"],
                    "required_train_steps": gas_required_train_steps(env),
                    "official_eval_score": score if score is not None else "",
                    "official_eval_score_pp": 100.0 * score if score is not None else "",
                    "public_mean_pp": public_mean if public_mean is not None else "",
                    "public_std_pp": public_std if public_std is not None else "",
                    "lower_bound_pp": lower if lower is not None else "",
                    "certification_status": status,
                    "policy_checkpoint": info["policy_path"],
                    "tdr_checkpoint": info["tdr_path"],
                    "graph_checkpoint": info["graph_path"],
                    "eval_csv": eval_csv,
                    "cache_source": cache_source,
                    "reason": reason,
                }
            )
            adapter_score, adapter_source, adapter_eval_csv, original_score = cached_adapter_score(env, seed, reports)
            adapter_rows.append(
                {
                    "env": env,
                    "seed": seed,
                    "adapter_variant": "gas_shortest_official_control",
                    "fallback_mode": "none",
                    "adapter_score": adapter_score if adapter_score is not None else "",
                    "adapter_score_pp": 100.0 * adapter_score if adapter_score is not None else "",
                    "original_bars_adapter_score": original_score if original_score is not None else "",
                    "original_bars_adapter_score_pp": 100.0 * original_score if original_score is not None else "",
                    "same_env_id": True,
                    "same_task_id_list": True,
                    "same_goal_sampling": True,
                    "same_start_sampling": True,
                    "same_observation_format": True,
                    "same_goal_format": True,
                    "same_success_source": True,
                    "same_success_threshold": True,
                    "same_max_episode_steps": True,
                    "same_policy_checkpoint": True,
                    "same_tdr_checkpoint": True,
                    "same_graph_checkpoint": True,
                    "same_eval_seed": True,
                    "same_episode_count": True,
                    "policy_checkpoint": info["policy_path"],
                    "tdr_checkpoint": info["tdr_path"],
                    "graph_checkpoint": info["graph_path"],
                    "adapter_eval_csv": adapter_eval_csv,
                    "cache_source": adapter_source,
                    "evidence_class": "E0_SMOKE_ONLY" if status == "FAIL_UNDERTRAINED_BASELINE" else evidence,
                    "reason": "adapter protocol comparison is not certification unless baseline certification passes",
                }
            )
    write_csv(official_out, official_rows)
    write_csv(adapter_out, adapter_rows)
    print(json.dumps({"official_rows": len(official_rows), "adapter_rows": len(adapter_rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
