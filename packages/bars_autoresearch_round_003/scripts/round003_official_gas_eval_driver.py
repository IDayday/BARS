#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bars.external.gas_artifacts import download_official_gas_if_available, gas_agent_flag_args
from round003_lib import (
    boolish,
    graph_file_url,
    local_artifact_record,
    max_episode_steps,
    normalize_task_id_list,
    official_artifact_url,
    official_eval_score_from_csv,
    policy_file_url,
    public_target,
    required_train_steps,
    split_csv,
    split_int_csv,
    way_steps,
    write_csv,
)


FIELDNAMES = [
    "env",
    "seed",
    "algorithm",
    "route",
    "job_status",
    "returncode",
    "artifact_status",
    "artifact_source",
    "train_steps",
    "required_train_steps",
    "official_eval_score",
    "official_eval_score_pp",
    "public_mean_pp",
    "public_std_pp",
    "lower_bound_pp",
    "protocol_match",
    "raw_official_protocol",
    "normalized_protocol",
    "official_eval_episodes",
    "official_eval_video_episodes",
    "rollouts_per_task",
    "num_task_ids",
    "task_id_list",
    "goal_sampling",
    "start_sampling",
    "observation_format",
    "goal_format",
    "success_source",
    "success_threshold",
    "max_episode_steps",
    "eval_seed",
    "episode_count",
    "policy_checkpoint",
    "tdr_checkpoint",
    "graph_checkpoint",
    "official_artifact_path_or_url",
    "official_graph_url",
    "official_policy_url",
    "eval_csv",
    "log",
    "command",
    "reason",
]


def run_command(cmd: list[str], cwd: Path, env: dict[str, str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", buffering=1) as f:
        f.write("$ " + " ".join(cmd) + "\n")
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=f, stderr=subprocess.STDOUT)
        return int(proc.returncode)


def row_from_eval(
    env_name: str,
    seed: int,
    rec: dict[str, Any],
    run_dir: Path,
    eval_csv: Path,
    log: Path,
    cmd: list[str],
    returncode: int,
    job_status: str,
    task_ids: list[int],
    eval_episodes: int,
    eval_video_episodes: int,
    reason: str,
) -> dict[str, Any]:
    score = official_eval_score_from_csv(eval_csv)
    mean, std, lower = public_target(env_name)
    rollouts_per_task = eval_episodes + eval_video_episodes
    protocol_match = (
        rec.get("artifact_status") == "OFFICIAL_FULL_BUDGET"
        and score["num_task_ids"] == len(task_ids) == 5
        and rollouts_per_task >= 49
        and seed is not None
        and job_status in {"completed", "cached"}
    )
    return {
        "env": env_name,
        "seed": seed,
        "algorithm": "GAS",
        "route": "official_evaluate_gas",
        "job_status": job_status,
        "returncode": returncode,
        "artifact_status": rec.get("artifact_status", ""),
        "artifact_source": rec.get("source", ""),
        "train_steps": "" if rec.get("local_train_steps") is None else rec.get("local_train_steps"),
        "required_train_steps": required_train_steps(env_name),
        "official_eval_score": "" if score["score"] is None else score["score"],
        "official_eval_score_pp": "" if score["score_pp"] is None else score["score_pp"],
        "public_mean_pp": "" if mean is None else mean,
        "public_std_pp": "" if std is None else std,
        "lower_bound_pp": "" if lower is None else lower,
        "protocol_match": protocol_match,
        "raw_official_protocol": f"evaluate_gas.py eval_episodes={eval_episodes} eval_video_episodes={eval_video_episodes}",
        "normalized_protocol": f"{rollouts_per_task} rollouts/task over task_ids={normalize_task_id_list(task_ids)}",
        "official_eval_episodes": eval_episodes,
        "official_eval_video_episodes": eval_video_episodes,
        "rollouts_per_task": rollouts_per_task,
        "num_task_ids": score["num_task_ids"],
        "task_id_list": normalize_task_id_list(task_ids),
        "goal_sampling": "OGBench task reset options task_id with per-episode reset seed=eval_seed+i",
        "start_sampling": "OGBench environment reset seed=eval_seed+i",
        "observation_format": "state",
        "goal_format": "env reset info['goal'] encoded by GAS TDR",
        "success_source": "env info['episode']['success'] / normalized_return==100",
        "success_threshold": 2,
        "max_episode_steps": max_episode_steps(env_name),
        "eval_seed": seed,
        "episode_count": rollouts_per_task * score["num_task_ids"] if score["num_task_ids"] else "",
        "policy_checkpoint": rec.get("policy_checkpoint", ""),
        "tdr_checkpoint": rec.get("tdr_checkpoint", ""),
        "graph_checkpoint": rec.get("graph_checkpoint", ""),
        "official_artifact_path_or_url": official_artifact_url(env_name),
        "official_graph_url": graph_file_url(env_name),
        "official_policy_url": policy_file_url(env_name),
        "eval_csv": str(eval_csv) if eval_csv.exists() else "",
        "log": str(log),
        "command": " ".join(cmd),
        "reason": reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", required=True)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--task-ids", default="1,2,3,4,5")
    parser.add_argument("--eval-episodes-per-task", type=int, default=50)
    parser.add_argument("--use-official-artifacts", default="1")
    parser.add_argument("--full-budget-only", default="1")
    parser.add_argument("--round", default="003")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--gas-artifact-root", default="artifacts/gas")
    parser.add_argument("--gas-repo-path", default="external_src/GAS")
    parser.add_argument("--run-root", default="runs_round003_official_gas_eval")
    parser.add_argument("--reports-root", default="reports")
    parser.add_argument("--force", default="0")
    parser.add_argument("--official-protocol", default="1")
    args = parser.parse_args()

    envs = split_csv(args.envs)
    seeds = split_int_csv(args.seeds)
    task_ids = split_int_csv(args.task_ids)
    gpus = split_csv(args.gpus) or ["0"]
    round_id = f"{int(args.round):03d}"
    out = Path(args.reports_root) / f"round_{round_id}_gas_official_eval.csv"
    rows: list[dict[str, Any]] = []
    job_index = 0
    for env_name in envs:
        for seed in seeds:
            gpu = gpus[job_index % len(gpus)]
            job_index += 1
            if boolish(args.use_official_artifacts):
                download_official_gas_if_available(env_name, args.gas_artifact_root)
            rec = local_artifact_record(env_name, seed, args.gas_artifact_root)
            run_dir = Path(args.run_root) / env_name / f"seed{seed}" / "official_evaluate_gas"
            eval_csv = run_dir / "eval.csv"
            log = run_dir / "official_eval.log"
            eval_video_episodes = 1 if boolish(args.official_protocol) and args.eval_episodes_per_task >= 1 else 0
            eval_episodes = args.eval_episodes_per_task - eval_video_episodes
            cmd: list[str] = []
            reason = ""
            if boolish(args.full_budget_only) and rec.get("artifact_status") != "OFFICIAL_FULL_BUDGET":
                reason = f"skipped because artifact_status={rec.get('artifact_status')} under FULL_BUDGET_ONLY"
                rows.append(
                    row_from_eval(
                        env_name,
                        seed,
                        rec,
                        run_dir,
                        eval_csv,
                        log,
                        cmd,
                        -1,
                        "skipped",
                        task_ids,
                        eval_episodes,
                        eval_video_episodes,
                        reason,
                    )
                )
                continue
            if eval_csv.exists() and not boolish(args.force):
                rows.append(
                    row_from_eval(
                        env_name,
                        seed,
                        rec,
                        run_dir,
                        eval_csv,
                        log,
                        cmd,
                        0,
                        "cached",
                        task_ids,
                        eval_episodes,
                        eval_video_episodes,
                        "cached run_dir eval.csv",
                    )
                )
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            policy_checkpoint = str(Path(rec.get("policy_checkpoint", "")).resolve())
            graph_checkpoint = str(Path(rec.get("graph_checkpoint", "")).resolve())
            policy_eval = Path(policy_checkpoint).parent / "eval.csv"
            cmd = [
                sys.executable,
                "evaluate_gas.py",
                "--run_eval_project",
                "Round003_GAS_Official_Eval",
                "--run_group",
                f"round003_{env_name}_seed{seed}",
                "--env_name",
                env_name,
                "--seed",
                str(seed),
                "--gpu",
                str(gpu),
                "--save_eval_dir",
                str((run_dir / "_raw_eval").resolve()),
                "--eval_on_cpu",
                "1",
                "--eval_episodes",
                str(eval_episodes),
                "--eval_video_episodes",
                str(eval_video_episodes),
                "--eval_final_goal_threshold",
                "2",
                "--keygraph_path",
                graph_checkpoint,
                "--policy_path",
                policy_checkpoint,
            ] + gas_agent_flag_args(env_name)
            env = os.environ.copy()
            env.setdefault("WANDB_MODE", "disabled")
            env.setdefault("WANDB_DISABLED", "true")
            env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
            env.setdefault("MUJOCO_GL", "osmesa")
            returncode = run_command(cmd, Path(args.gas_repo_path), env, log)
            if returncode == 0 and policy_eval.exists():
                shutil.copy2(policy_eval, eval_csv)
            job_status = "completed" if returncode == 0 and eval_csv.exists() else "failed"
            reason = "official evaluate_gas.py completed" if job_status == "completed" else "official evaluate_gas.py failed or produced no eval.csv"
            rows.append(
                row_from_eval(
                    env_name,
                    seed,
                    rec,
                    run_dir,
                    eval_csv,
                    log,
                    cmd,
                    returncode,
                    job_status,
                    task_ids,
                    eval_episodes,
                    eval_video_episodes,
                    reason,
                )
            )
    write_csv(out, rows, FIELDNAMES)
    print(json.dumps({"rows": len(rows), "out": str(out)}, sort_keys=True))


if __name__ == "__main__":
    main()
