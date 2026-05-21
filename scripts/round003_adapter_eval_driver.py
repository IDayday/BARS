#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from round003_lib import (
    adapter_score_from_csv,
    boolish,
    local_artifact_record,
    max_episode_steps,
    normalize_task_id_list,
    read_json,
    split_csv,
    split_int_csv,
    way_steps,
    write_csv,
)


FIELDNAMES = [
    "env",
    "seed",
    "adapter_variant",
    "fallback_mode",
    "job_status",
    "returncode",
    "adapter_score",
    "adapter_score_pp",
    "task_id_list",
    "num_task_ids",
    "rollouts_per_task",
    "episode_count",
    "goal_sampling",
    "start_sampling",
    "observation_format",
    "goal_format",
    "success_source",
    "success_threshold",
    "max_episode_steps",
    "eval_seed",
    "same_checkpoint_source",
    "policy_checkpoint",
    "tdr_checkpoint",
    "graph_checkpoint",
    "adapter_eval_csv",
    "log",
    "command",
    "task_counts",
    "steps_mean",
    "reason",
]


def certified_envs(path: Path) -> set[str]:
    data = read_json(path, {})
    return set(str(x) for x in data.get("certified_envs", []))


def run_command(cmd: list[str], cwd: Path, env: dict[str, str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", buffering=1) as f:
        f.write("$ " + " ".join(cmd) + "\n")
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=f, stderr=subprocess.STDOUT)
        return int(proc.returncode)


def row_from_eval(
    env_name: str,
    seed: int,
    variant: str,
    fallback_mode: str,
    rec: dict[str, Any],
    task_ids: list[int],
    episodes_per_task: int,
    eval_csv: Path,
    log: Path,
    cmd: list[str],
    returncode: int,
    job_status: str,
    reason: str,
) -> dict[str, Any]:
    score = adapter_score_from_csv(eval_csv, task_ids, episodes_per_task)
    return {
        "env": env_name,
        "seed": seed,
        "adapter_variant": variant,
        "fallback_mode": fallback_mode,
        "job_status": job_status,
        "returncode": returncode,
        "adapter_score": "" if score["score"] is None else score["score"],
        "adapter_score_pp": "" if score["score_pp"] is None else score["score_pp"],
        "task_id_list": normalize_task_id_list(task_ids),
        "num_task_ids": len(task_ids),
        "rollouts_per_task": episodes_per_task,
        "episode_count": len(task_ids) * episodes_per_task,
        "goal_sampling": "OGBench task reset options task_id with per-episode reset seed=eval_seed+i",
        "start_sampling": "OGBench environment reset seed=eval_seed+i",
        "observation_format": "state",
        "goal_format": "env reset info['goal'] encoded by GAS TDR",
        "success_source": "env info['episode']['success'] / normalized_return==100",
        "success_threshold": 2,
        "max_episode_steps": max_episode_steps(env_name),
        "eval_seed": seed,
        "same_checkpoint_source": rec.get("source", ""),
        "policy_checkpoint": rec.get("policy_checkpoint", ""),
        "tdr_checkpoint": rec.get("tdr_checkpoint", ""),
        "graph_checkpoint": rec.get("graph_checkpoint", ""),
        "adapter_eval_csv": str(eval_csv) if eval_csv.exists() else "",
        "log": str(log),
        "command": " ".join(cmd),
        "task_counts": json.dumps(score.get("task_counts", {}), sort_keys=True),
        "steps_mean": score.get("steps_mean", ""),
        "reason": reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", default="")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--task-ids", default="1,2,3,4,5")
    parser.add_argument("--eval-episodes-per-task", type=int, default=50)
    parser.add_argument("--variant", default="gas_shortest_official_control")
    parser.add_argument("--fallback-mode", default="none")
    parser.add_argument("--use-official-artifacts", default="1")
    parser.add_argument("--round", default="003")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--gas-artifact-root", default="artifacts/gas")
    parser.add_argument("--artifact-root", default="artifacts/round003_adapter")
    parser.add_argument("--gas-repo-path", default="external_src/GAS")
    parser.add_argument("--run-root", default="runs_round003_bars_adapter_eval")
    parser.add_argument("--reports-root", default="reports")
    parser.add_argument("--baseline-certification", default="reports/round_003_baseline_certification.json")
    parser.add_argument("--force", default="0")
    parser.add_argument("--eval-on-cpu", default="1")
    args = parser.parse_args()

    cert_envs = certified_envs(Path(args.baseline_certification))
    envs = split_csv(args.envs) if args.envs else sorted(cert_envs)
    seeds = split_int_csv(args.seeds)
    task_ids = split_int_csv(args.task_ids)
    gpus = split_csv(args.gpus) or ["0"]
    round_id = f"{int(args.round):03d}"
    out = Path(args.reports_root) / f"round_{round_id}_bars_adapter_eval.csv"
    rows: list[dict[str, Any]] = []
    job_index = 0
    for env_name in envs:
        for seed in seeds:
            rec = local_artifact_record(env_name, seed, args.gas_artifact_root)
            run_dir = (
                Path(args.run_root)
                / env_name
                / f"seed{seed}"
                / args.variant
                / "budget999"
                / f"fallback_{args.fallback_mode}"
            )
            eval_csv = run_dir / "eval.csv"
            log = run_dir / "adapter_eval.log"
            if env_name not in cert_envs:
                rows.append(
                    row_from_eval(
                        env_name,
                        seed,
                        args.variant,
                        args.fallback_mode,
                        rec,
                        task_ids,
                        args.eval_episodes_per_task,
                        eval_csv,
                        log,
                        [],
                        -1,
                        "skipped",
                        "skipped because baseline certification did not pass for this env",
                    )
                )
                continue
            score = adapter_score_from_csv(eval_csv, task_ids, args.eval_episodes_per_task)
            if eval_csv.exists() and score["complete"] and not boolish(args.force):
                rows.append(
                    row_from_eval(
                        env_name,
                        seed,
                        args.variant,
                        args.fallback_mode,
                        rec,
                        task_ids,
                        args.eval_episodes_per_task,
                        eval_csv,
                        log,
                        [],
                        0,
                        "cached",
                        "cached complete adapter eval.csv",
                    )
                )
                continue
            gpu = gpus[job_index % len(gpus)]
            job_index += 1
            if boolish(args.force) and run_dir.exists():
                shutil.rmtree(run_dir)
            cmd = [
                sys.executable,
                "-m",
                "bars.gas_bars.evaluate",
                "--env",
                env_name,
                "--seed",
                str(seed),
                "--task-ids",
                normalize_task_id_list(task_ids),
                "--episodes",
                str(args.eval_episodes_per_task),
                "--episodes-per-task",
                "1",
                "--variant",
                args.variant,
                "--budget",
                "999",
                "--fallback-mode",
                args.fallback_mode,
                "--gas-artifact-root",
                args.gas_artifact_root,
                "--artifact-root",
                args.artifact_root,
                "--stage22-root",
                args.run_root,
                "--gas-repo-path",
                args.gas_repo_path,
                "--gpu",
                str(gpu),
                "--eval-on-cpu",
                "1" if boolish(args.eval_on_cpu) else "0",
                "--prefer-pretrained",
                "1" if boolish(args.use_official_artifacts) else "0",
                "--train-if-missing",
                "0",
                "--quick",
                "1",
                "--max-steps",
                str(max_episode_steps(env_name)),
                "--subgoal-threshold",
                str(way_steps(env_name)),
                "--final-goal-threshold",
                "2",
                "--control-mode",
                "official",
                "--debug-jsonl",
                "0",
            ]
            env = os.environ.copy()
            env.setdefault("WANDB_MODE", "disabled")
            env.setdefault("WANDB_DISABLED", "true")
            env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
            env.setdefault("MUJOCO_GL", "osmesa")
            rc = run_command(cmd, Path("."), env, log)
            score = adapter_score_from_csv(eval_csv, task_ids, args.eval_episodes_per_task)
            job_status = "completed" if rc == 0 and score["complete"] else "failed"
            reason = "BARS adapter official-control eval completed" if job_status == "completed" else "BARS adapter eval failed or incomplete"
            rows.append(
                row_from_eval(
                    env_name,
                    seed,
                    args.variant,
                    args.fallback_mode,
                    rec,
                    task_ids,
                    args.eval_episodes_per_task,
                    eval_csv,
                    log,
                    cmd,
                    rc,
                    job_status,
                    reason,
                )
            )
    write_csv(out, rows, FIELDNAMES)
    print(json.dumps({"rows": len(rows), "out": str(out)}, sort_keys=True))


if __name__ == "__main__":
    main()
