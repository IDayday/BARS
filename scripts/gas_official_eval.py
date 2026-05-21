#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bars.external.gas_artifacts import env_to_hf_slug, gas_agent_flag_args, official_eval_on_cpu, resolve_gas_artifacts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--gpu", default="0")
    p.add_argument("--artifact-root", required=True)
    p.add_argument("--gas-repo-path", default="external_src/GAS")
    p.add_argument("--eval-episodes", type=int, default=49)
    p.add_argument("--eval-video-episodes", type=int, default=1)
    p.add_argument("--eval-on-cpu", type=int, default=None)
    args = p.parse_args()

    artifacts = resolve_gas_artifacts(args.env, args.seed, args.artifact_root)
    if not artifacts.complete:
        raise RuntimeError(f"Missing GAS artifacts for {args.env} seed {args.seed}: {artifacts.to_dict()}")

    eval_on_cpu = official_eval_on_cpu(args.env) if args.eval_on_cpu is None else args.eval_on_cpu
    final_goal_threshold = 1 if args.env == "kitchen-partial-v0" else 2
    run_group = f"gas_official_full_{env_to_hf_slug(args.env)}_seed{args.seed}"
    cmd = [
        sys.executable,
        "evaluate_gas.py",
        "--run_eval_project",
        "GAS_Official_Full_Eval",
        "--run_group",
        run_group,
        "--env_name",
        args.env,
        "--seed",
        str(args.seed),
        "--gpu",
        str(args.gpu),
        "--save_eval_dir",
        str((artifacts.root / "_raw_gas" / "eval").resolve()),
        "--eval_on_cpu",
        str(eval_on_cpu),
        "--eval_episodes",
        str(args.eval_episodes),
        "--eval_video_episodes",
        str(args.eval_video_episodes),
        "--eval_final_goal_threshold",
        str(final_goal_threshold),
        "--keygraph_path",
        str(artifacts.keygraph.resolve()),
        "--policy_path",
        str(artifacts.policy_checkpoint.resolve()),
    ] + gas_agent_flag_args(args.env)

    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("WANDB_DISABLED", "true")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("MUJOCO_GL", "egl")
    subprocess.run(cmd, cwd=str(Path(args.gas_repo_path).resolve()), env=env, check=True)


if __name__ == "__main__":
    main()
