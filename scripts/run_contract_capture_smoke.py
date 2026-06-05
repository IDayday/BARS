#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from build_cage_eval_command import build_eval_command, command_to_string, infer_checkpoint_paths, safe_id


DEFAULT_VARIANTS = ["gas", "cage_trace_only", "cage_fixed_commit", "cage_safe_full"]
GAS_ROOT = Path(__file__).resolve().parents[1] / "external_src" / "GAS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate or run small CLP1 segment contract-capture jobs.")
    parser.add_argument("--checkpoint_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--env_seed_pairs", nargs="*", default=[], help="Pairs like antmaze-giant-navigate-v0:42.")
    parser.add_argument("--envs", nargs="*", default=[])
    parser.add_argument("--seeds", nargs="*", type=int, default=[])
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--episodes_per_goal", type=int, default=2)
    parser.add_argument("--goals_per_env", type=int, default=2)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_jobs", type=int, default=0, help="0 means all generated jobs.")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--status_path", default=None)
    return parser.parse_args()


def env_seed_pairs(args: argparse.Namespace) -> list[tuple[str, int]]:
    pairs: list[tuple[str, int]] = []
    for item in args.env_seed_pairs:
        if ":" not in item:
            raise ValueError(f"--env_seed_pairs item must be env:seed, got {item!r}")
        env_name, seed = item.rsplit(":", 1)
        pairs.append((env_name, int(seed)))
    if args.envs and args.seeds:
        pairs.extend((env_name, int(seed)) for env_name in args.envs for seed in args.seeds)
    if not pairs:
        raise ValueError("provide --env_seed_pairs or both --envs and --seeds")
    return pairs


def make_row(args: argparse.Namespace, env_name: str, seed: int, variant: str) -> dict[str, Any]:
    output_root = Path(args.output_root).resolve()
    checkpoint_root = Path(args.checkpoint_root).resolve()
    safe_env = safe_id(env_name)
    safe_variant = safe_id(variant)
    row_root = output_root / "jobs" / safe_env / f"seed{seed}" / safe_variant
    paths = infer_checkpoint_paths(checkpoint_root, env_name, seed)
    return {
        "job_id": f"{safe_env}__seed{seed}__{safe_variant}",
        "env_name": env_name,
        "seed": int(seed),
        "variant": variant,
        "checkpoint_root": str(checkpoint_root),
        "output_root": str(output_root),
        "keygraph_path": paths["keygraph_path"],
        "policy_path": paths["policy_path"],
        "tdr_path": paths["tdr_path"],
        "cage_trace_path": str(row_root / "cage_trace.jsonl"),
        "result_path": str(row_root / "eval.csv"),
        "contract_trace_path": str(output_root / f"{safe_env}_{safe_variant}_seed{seed}_segments.jsonl"),
        "store_contract_state_refs": True,
        "contract_state_ref_mode": "exact_only",
        "contract_capture_variants": ["gas", "cage_trace_only", "cage_fixed_commit", "cage_safe_full"],
        "episodes_per_goal": int(args.episodes_per_goal),
        "goals_per_env": int(args.goals_per_env),
        "gpu": int(args.gpu),
        "eval_on_cpu": 1,
        "eval_video_episodes": 0,
        "eval_final_goal_threshold": 2,
        "save_eval_dir": str(output_root / "eval_logs"),
        "run_eval_project": "CAGEContract",
        "run_group": f"clp1_{safe_env}_{safe_variant}_seed{seed}",
    }


def main() -> int:
    args = parse_args()
    rows = [make_row(args, env_name, seed, variant) for env_name, seed in env_seed_pairs(args) for variant in args.variants]
    limit = len(rows) if args.max_jobs <= 0 else min(len(rows), int(args.max_jobs))
    status_rows: list[dict[str, Any]] = []
    for row in rows[:limit]:
        command = build_eval_command(row)
        if command and command[0] == "python":
            command[0] = sys.executable
        if args.dry_run:
            print(command_to_string(command))
            continue
        job_root = Path(row["result_path"]).parent
        job_root.mkdir(parents=True, exist_ok=True)
        stdout_path = job_root / "stdout.txt"
        stderr_path = job_root / "stderr.txt"
        status = {
            "job_id": row["job_id"],
            "env_name": row["env_name"],
            "seed": row["seed"],
            "variant": row["variant"],
            "command": command,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "contract_trace_path": row["contract_trace_path"],
            "result_path": row["result_path"],
        }
        env = dict(os.environ)
        env.setdefault("WANDB_MODE", "disabled")
        env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
            proc = subprocess.run(command, cwd=GAS_ROOT, stdout=out, stderr=err, env=env, check=False)
        status["return_code"] = proc.returncode
        status["status"] = "succeeded" if proc.returncode == 0 else "failed"
        print(json.dumps(status, sort_keys=True))
        status_rows.append(status)
    if args.status_path and not args.dry_run:
        path = Path(args.status_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for row in status_rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
