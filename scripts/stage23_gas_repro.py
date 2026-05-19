#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bars.external.gas_artifacts import gas_agent_flag_args, resolve_gas_artifacts


def _csv_success(path: Path) -> tuple[float | None, int]:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None, 0
    if len(df) == 0:
        return None, 0
    if "success" in df.columns:
        return float(df["success"].mean()), len(df)
    success_cols = [c for c in df.columns if c.endswith("episode.success") or "overall_episode.success" in c]
    if success_cols:
        return float(df[success_cols[-1]].iloc[-1]), len(df)
    return None, len(df)


def _cached_row(route: str, env_name: str, seed: int, path: Path, log: Path, episodes: int, *, official: bool) -> dict | None:
    if not path.exists():
        return None
    success, n_rows = _csv_success(path)
    if official or n_rows >= episodes:
        return {
            "route": route,
            "env": env_name,
            "seed": seed,
            "status": "completed",
            "returncode": 0,
            "success": success,
            "eval_csv": str(path),
            "log": str(log),
            "reason": "cached",
        }
    return None


def _run(cmd: list[str], cwd: Path, log: Path, env: dict[str, str]) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", buffering=1) as f:
        f.write("$ " + " ".join(cmd) + "\n")
        return subprocess.run(cmd, cwd=str(cwd), env=env, stdout=f, stderr=subprocess.STDOUT).returncode


def _official_eval(env_name: str, seed: int, gas_repo: Path, artifacts, gpu: str, episodes: int, run_dir: Path, route: str) -> dict:
    log = run_dir / "official_eval.log"
    cached = _cached_row(route, env_name, seed, run_dir / "eval.csv", log, episodes, official=True)
    if cached:
        return cached
    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("WANDB_DISABLED", "true")
    env.setdefault("BARS_USE_TENSORBOARD", "1")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("MUJOCO_GL", "egl")
    if str(gpu).lower() in {"cpu", "-1", ""}:
        env["JAX_PLATFORMS"] = "cpu"
        env["JAX_PLATFORM_NAME"] = "cpu"
    cmd = [
        sys.executable,
        "evaluate_gas.py",
        "--run_eval_project",
        "Stage23_GAS_Repro",
        "--run_group",
        f"stage23_repro_{env_name}_seed{seed}_{route}",
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
        str(episodes),
        "--eval_video_episodes",
        "0",
        "--eval_final_goal_threshold",
        "2",
        "--keygraph_path",
        str(Path(artifacts.keygraph).resolve()),
        "--policy_path",
        str(Path(artifacts.policy_checkpoint).resolve()),
    ] + gas_agent_flag_args(env_name)
    rc = _run(cmd, gas_repo, log, env)
    copied = None
    candidates = sorted(Path(artifacts.policy_dir).glob("eval.csv")) + sorted((run_dir / "_raw_eval").rglob("eval.csv"))
    if candidates:
        copied = run_dir / "eval.csv"
        shutil.copy2(candidates[-1], copied)
    success = None
    if copied and copied.exists():
        success, _ = _csv_success(copied)
    return {"route": route, "env": env_name, "seed": seed, "status": "completed" if rc == 0 else "failed", "returncode": rc, "success": success, "eval_csv": str(copied) if copied else "", "log": str(log)}


def _adapter_eval(env_name: str, seed: int, gpu: str, episodes: int, run_dir: Path, args: argparse.Namespace) -> dict:
    log = run_dir / "adapter_eval.log"
    control_mode = str(getattr(args, "adapter_control_mode", "official"))
    variant = "gas_shortest_official_control" if control_mode == "official" else "gas_shortest"
    expected = run_dir / env_name / f"seed{seed}" / variant / "budget999" / "fallback_none" / "eval.csv"
    cached = _cached_row("C_adapter_same_checkpoint", env_name, seed, expected, log, episodes, official=False)
    if cached:
        cached["adapter_control_mode"] = control_mode
        return cached
    env = os.environ.copy()
    use_cpu = str(gpu).lower() in {"cpu", "-1", ""}
    if use_cpu:
        env["JAX_PLATFORMS"] = "cpu"
        env["JAX_PLATFORM_NAME"] = "cpu"
    cmd = [
        sys.executable,
        "-m",
        "bars.gas_bars.evaluate",
        "--env",
        env_name,
        "--seed",
        str(seed),
        "--episodes",
        str(episodes),
        "--variant",
        variant,
        "--budget",
        "999",
        "--fallback-mode",
        "none",
        "--gas-artifact-root",
        args.gas_artifact_root,
        "--artifact-root",
        args.stage22_artifact_root,
        "--stage22-root",
        str(run_dir),
        "--gas-repo-path",
        args.gas_repo_path,
        "--gpu",
        str(gpu),
        "--eval-on-cpu",
        "1" if use_cpu else "0",
        "--prefer-pretrained",
        str(args.prefer_pretrained),
        "--train-if-missing",
        str(args.train_if_missing),
        "--quick",
        "1",
        "--debug-jsonl",
        "0",
        "--control-mode",
        control_mode,
    ]
    rc = _run(cmd, Path("."), log, env)
    evals = list(run_dir.rglob("eval.csv"))
    success = None
    copied = evals[-1] if evals else None
    if copied:
        success, _ = _csv_success(copied)
    return {"route": "C_adapter_same_checkpoint", "env": env_name, "seed": seed, "status": "completed" if rc == 0 else "failed", "returncode": rc, "success": success, "eval_csv": str(copied) if copied else "", "log": str(log), "adapter_control_mode": control_mode}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--envs", default="antmaze-medium-navigate-v0,antmaze-medium-stitch-v0")
    p.add_argument("--seeds", default="0")
    p.add_argument("--gpus", default="cpu")
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--gas-artifact-root", default="artifacts/gas")
    p.add_argument("--stage22-artifact-root", default="artifacts/stage22")
    p.add_argument("--gas-repo-path", default="external_src/GAS")
    p.add_argument("--run-root", default="runs_stage23_repro")
    p.add_argument("--reports-root", default="reports")
    p.add_argument("--prefer-pretrained", type=int, default=1)
    p.add_argument("--train-if-missing", type=int, default=0)
    p.add_argument("--adapter-control-mode", default="official", choices=["official", "bars"])
    args = p.parse_args()
    rows = []
    gpus = [x for x in args.gpus.split(",") if x] or ["cpu"]
    i = 0
    for env_name in [x for x in args.envs.split(",") if x]:
        for seed in [int(x) for x in args.seeds.split(",") if x]:
            gpu = gpus[i % len(gpus)]
            i += 1
            artifacts = resolve_gas_artifacts(env_name, seed, args.gas_artifact_root)
            base = Path(args.run_root) / env_name / f"seed{seed}"
            if artifacts.complete:
                manifest = artifacts.root / "manifest.json"
                source = ""
                if manifest.exists():
                    try:
                        source = str(json.loads(manifest.read_text()).get("source", ""))
                    except Exception:
                        source = ""
                if source == "huggingface":
                    rows.append(_official_eval(env_name, seed, Path(args.gas_repo_path), artifacts, gpu, args.episodes, base / "A_official_pretrained", "A_official_pretrained"))
                else:
                    rows.append({"route": "A_official_pretrained", "env": env_name, "seed": seed, "status": "skipped", "returncode": "", "success": "", "eval_csv": "", "log": "", "reason": "official pretrained checkpoint not available in local artifacts"})
                rows.append(_official_eval(env_name, seed, Path(args.gas_repo_path), artifacts, gpu, args.episodes, base / "B_official_our_checkpoint", "B_official_our_checkpoint"))
                rows.append(_adapter_eval(env_name, seed, gpu, args.episodes, base / "C_adapter_same_checkpoint", args))
            else:
                rows.append({"route": "all", "env": env_name, "seed": seed, "status": "skipped", "returncode": "", "success": "", "eval_csv": "", "log": "", "reason": "missing artifacts"})
    reports = Path(args.reports_root)
    reports.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(reports / "stage23_gas_reproduction_matrix.csv", index=False)
    lines = ["# Stage23 GAS Reproduction Summary", ""]
    if len(df):
        try:
            lines.append(df.to_markdown(index=False))
        except Exception:
            lines.append("```csv\n" + df.to_csv(index=False).strip() + "\n```")
    (reports / "stage23_repro_summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
