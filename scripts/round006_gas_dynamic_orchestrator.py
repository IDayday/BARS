#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = Path("/root/remote/datasets/ogbench")
DEFAULT_ROUND = "006"
DEFAULT_SEEDS = (42, 43, 44, 45, 46)


@dataclass(frozen=True)
class GasConfig:
    env: str
    encoder: str
    train_steps: int
    batch_size: int
    discount: str
    tdr_expectile: str
    alpha: str
    p_aug: str
    way_steps: int
    te_threshold: str
    eval_on_cpu: int
    priority: int
    slot_cost: int = 1


def _antmaze_config(env: str, priority: int) -> GasConfig:
    slug = env.removesuffix("-v0")
    discount = "0.995" if "giant" in slug else "0.99"
    alpha = "0.01" if "explore" in slug else "1.0"
    return GasConfig(
        env=env,
        encoder="not_used",
        train_steps=1_000_000,
        batch_size=1024,
        discount=discount,
        tdr_expectile="0.999",
        alpha=alpha,
        p_aug="0.0",
        way_steps=8,
        te_threshold="0.99",
        eval_on_cpu=1,
        priority=priority,
    )


def _scene_config(env: str, priority: int) -> GasConfig:
    return GasConfig(
        env=env,
        encoder="not_used",
        train_steps=1_000_000,
        batch_size=1024,
        discount="0.99",
        tdr_expectile="0.999",
        alpha="1.0",
        p_aug="0.0",
        way_steps=48,
        te_threshold="0.99",
        eval_on_cpu=1,
        priority=priority,
    )


def _humanoid_config(env: str, priority: int) -> GasConfig:
    slug = env.removesuffix("-v0")
    discount = "0.995" if "giant" in slug else "0.99"
    return GasConfig(
        env=env,
        encoder="not_used",
        train_steps=1_000_000,
        batch_size=1024,
        discount=discount,
        tdr_expectile="0.95",
        alpha="0.1",
        p_aug="0.0",
        way_steps=32,
        te_threshold="0.99",
        eval_on_cpu=1,
        priority=priority,
    )


def _visual_config(env: str, priority: int) -> GasConfig:
    slug = env.removesuffix("-v0")
    discount = "0.995" if "giant" in slug else "0.99"
    alpha = "0.01" if "explore" in slug else "1.0"
    way_steps = 24 if "scene" in slug else 8
    return GasConfig(
        env=env,
        encoder="impala_small",
        train_steps=500_000,
        batch_size=256,
        discount=discount,
        tdr_expectile="0.95",
        alpha=alpha,
        p_aug="0.5",
        way_steps=way_steps,
        te_threshold="0.9",
        eval_on_cpu=0,
        priority=priority,
        slot_cost=2,
    )


def gas_configs() -> dict[str, GasConfig]:
    common = [
        "antmaze-giant-navigate-v0",
        "antmaze-giant-stitch-v0",
        "antmaze-large-explore-v0",
        "scene-play-v0",
    ]
    state_extra = [
        "antmaze-large-navigate-v0",
        "antmaze-medium-navigate-v0",
        "antmaze-large-stitch-v0",
        "antmaze-medium-stitch-v0",
        "antmaze-medium-explore-v0",
    ]
    humanoid = [
        "humanoidmaze-giant-navigate-v0",
        "humanoidmaze-large-navigate-v0",
        "humanoidmaze-medium-navigate-v0",
        "humanoidmaze-giant-stitch-v0",
        "humanoidmaze-large-stitch-v0",
        "humanoidmaze-medium-stitch-v0",
    ]
    visual = [
        "visual-antmaze-giant-navigate-v0",
        "visual-antmaze-large-navigate-v0",
        "visual-antmaze-medium-navigate-v0",
        "visual-antmaze-giant-stitch-v0",
        "visual-antmaze-large-stitch-v0",
        "visual-antmaze-medium-stitch-v0",
        "visual-antmaze-large-explore-v0",
        "visual-antmaze-medium-explore-v0",
        "visual-scene-play-v0",
    ]
    out: dict[str, GasConfig] = {}
    pr = 0
    for env in common:
        out[env] = _scene_config(env, pr) if env.startswith("scene") else _antmaze_config(env, pr)
        pr += 1
    for env in state_extra:
        out[env] = _antmaze_config(env, pr)
        pr += 1
    for env in humanoid:
        out[env] = _humanoid_config(env, pr)
        pr += 1
    for env in visual:
        out[env] = _visual_config(env, pr)
        pr += 1
    return out


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def abspath(path: Path) -> str:
    return str(path.resolve())


def split_csv(raw: str | None) -> list[str]:
    return [x.strip() for x in (raw or "").split(",") if x.strip()]


def parse_seeds(raw: str) -> list[int]:
    vals: list[int] = []
    for part in split_csv(raw):
        if "-" in part:
            a, b = part.split("-", 1)
            vals.extend(range(int(a), int(b) + 1))
        else:
            vals.append(int(part))
    return vals


def ensure_dirs(round_id: str, run_root: Path, out_root: Path) -> None:
    for path in [
        run_root,
        out_root,
        run_root / "_orchestrator",
        run_root / "_workers",
        REPO_ROOT / "reports",
        REPO_ROOT / "commands",
        REPO_ROOT / "rounds" / f"round_{round_id}",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def source_stage24_env() -> None:
    env_file = REPO_ROOT / "scripts" / "stage24_env_mirrors.sh"
    if not env_file.exists():
        return
    cmd = f"set -a; source {env_file}; env -0"
    proc = subprocess.run(["bash", "-lc", cmd], capture_output=True, check=True)
    for item in proc.stdout.split(b"\0"):
        if not item or b"=" not in item:
            continue
        key, val = item.split(b"=", 1)
        key_s = key.decode("utf-8", errors="ignore")
        if key_s in {
            "BARS_HF_ENDPOINTS",
            "BARS_SHARED_DATASET_ROOT",
            "OGBENCH_DATASET_DIR",
            "D4RL_DATASET_DIR",
            "OGBENCH_DATASET_SHARED_DIRS",
            "BARS_OGBENCH_CN_ENDPOINTS",
            "OGBENCH_DATASET_ENDPOINTS",
            "BARS_DOWNLOAD_WITH_ARIA2",
            "BARS_ARIA2_SPLIT",
            "BARS_ARIA2_CONNECTIONS",
            "BARS_DOWNLOAD_PARALLEL_CHUNKS",
            "HTTP_PROXY",
            "HTTPS_PROXY",
        }:
            os.environ.setdefault(key_s, val.decode("utf-8", errors="ignore"))


def python_bin() -> str:
    return os.environ.get("PYTHON") or (
        "/root/anaconda3/envs/gcrlo/bin/python"
        if Path("/root/anaconda3/envs/gcrlo/bin/python").exists()
        else sys.executable
    )


def base_env(dataset_dir: Path, offline: bool) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}".rstrip(":")
    env["WANDB_MODE"] = "disabled"
    env["WANDB_DISABLED"] = "true"
    env["BARS_USE_TENSORBOARD"] = "0"
    env["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
    env["BARS_OGBENCH_OFFLINE"] = "1" if offline else "0"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env.setdefault("MUJOCO_GL", "osmesa")
    env["OGBENCH_DATASET_DIR"] = str(dataset_dir)
    env.setdefault("BARS_DOWNLOAD_PARALLEL_CHUNKS", "1")
    return env


def dataset_names(env_name: str) -> tuple[str, str]:
    parts = env_name.split("-")
    dataset_name = env_name
    if "singletask" in parts:
        pos = parts.index("singletask")
        dataset_name = "-".join(parts[:pos] + parts[-1:])
    elif "oraclerep" in parts:
        dataset_name = "-".join(parts[:-2] + parts[-1:])
    return f"{dataset_name}.npz", f"{dataset_name}-val.npz"


def npz_ready(path: Path) -> bool:
    try:
        from bars.data.ogbench_dataset import _npz_is_ready  # type: ignore

        return bool(_npz_is_ready(path))
    except Exception:
        return path.exists() and path.stat().st_size > 0


def dataset_ready(env_name: str, dataset_dir: Path) -> bool:
    train, val = dataset_names(env_name)
    return npz_ready(dataset_dir / train) and npz_ready(dataset_dir / val)


def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")


def read_status(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"status": "corrupt_status"}


def write_status(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def latest_checkpoint(root: Path) -> Path | None:
    paths = list(root.glob("**/params_*.pkl"))
    if not paths:
        return None

    def step(path: Path) -> int:
        try:
            return int(path.stem.split("_")[-1])
        except Exception:
            return -1

    return sorted(paths, key=lambda p: (step(p), str(p)))[-1]


def final_checkpoint(root: Path, train_steps: int) -> Path | None:
    paths = list(root.glob(f"**/params_{train_steps}.pkl"))
    return sorted(paths)[-1] if paths else None


def keygraph_path(root: Path) -> Path | None:
    paths = list(root.glob("**/keygraph.pkl"))
    return sorted(paths)[-1] if paths else None


def latest_eval_csv(root: Path) -> Path | None:
    paths = list(root.glob("**/eval.csv"))
    return sorted(paths, key=lambda p: (p.stat().st_mtime, str(p)))[-1] if paths else None


def gas_flags(config: GasConfig, seed: int, gpu: str) -> list[str]:
    return [
        "--env_name",
        config.env,
        "--seed",
        str(seed),
        "--gpu",
        str(gpu),
        "--agent_config.encoder",
        config.encoder,
        "--agent_config.discount",
        config.discount,
        "--agent_config.tdr_expectile",
        config.tdr_expectile,
        "--agent_config.alpha",
        config.alpha,
        "--agent_config.batch_size",
        str(config.batch_size),
        "--agent_config.p_aug",
        config.p_aug,
        "--agent_config.way_steps",
        str(config.way_steps),
    ]


def run_phase(name: str, cmd: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"{now_iso()} START {name}\n")
        log.write(" ".join(cmd) + "\n")
        log.flush()
        subprocess.run(cmd, cwd=REPO_ROOT / "external_src" / "GAS", env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        log.write(f"{now_iso()} DONE {name}\n")


def run_worker(args: argparse.Namespace) -> int:
    source_stage24_env()
    configs = gas_configs()
    config = configs[args.env]
    dataset_dir = Path(args.dataset_dir)
    round_id = args.round
    run_root = Path(args.run_root)
    out_root = Path(args.out_root)
    seed = int(args.seed)
    gpu = str(args.gpu)
    env = base_env(dataset_dir, offline=True)
    py = python_bin()

    env_out = out_root / config.env / f"seed{seed}"
    run_dir = run_root / config.env / f"seed{seed}"
    tdr_root = env_out / "tdr"
    graph_root = env_out / "graph"
    policy_root = env_out / "policy"
    eval_root = run_dir / "evaluate_gas"
    status_path = run_dir / "status.json"
    run_group = f"round{round_id}_selftrain_{config.env.removesuffix('-v0')}_seed{seed}"

    for p in [tdr_root, graph_root, policy_root, eval_root, run_dir]:
        p.mkdir(parents=True, exist_ok=True)

    if latest_eval_csv(policy_root) is not None:
        status = read_status(status_path)
        status.update(
            {
                "env": config.env,
                "seed": seed,
                "gpu": gpu,
                "status": "completed",
                "already_completed": True,
                "completed_at": status.get("completed_at") or now_iso(),
                "eval_csv": rel(latest_eval_csv(policy_root)),
            }
        )
        write_status(status_path, status)
        return 0

    write_status(
        status_path,
        {
            "env": config.env,
            "seed": seed,
            "gpu": gpu,
            "round": round_id,
            "train_steps": config.train_steps,
            "artifact_source": "full_budget_train_from_scratch",
            "evidence_class": "E4_FULL_BUDGET_TRAINED_METHOD",
            "official_weights_used": False,
            "checkpoint_policy": "full_stage_checkpoints_only_no_intermediate_resume",
            "status": "launched",
            "started_at": now_iso(),
            "config": config.__dict__,
        },
    )

    flags = gas_flags(config, seed, gpu)
    try:
        tdr_path = final_checkpoint(tdr_root, config.train_steps)
        if tdr_path is None:
            cmd = [
                py,
                "pretrain_tdr.py",
                "--run_tdr_project",
                f"Round{round_id}_GAS_SelfTrain",
                "--run_group",
                run_group,
                "--save_tdr_dir",
                abspath(tdr_root),
                "--train_steps",
                str(config.train_steps),
                "--log_interval",
                str(args.log_interval),
                "--save_interval",
                str(args.save_interval),
            ]
            cmd.extend(flags)
            run_phase("pretrain_tdr", cmd, run_dir / "pretrain_tdr.log", env)
            tdr_path = final_checkpoint(tdr_root, config.train_steps)
        if tdr_path is None:
            raise FileNotFoundError(f"Missing final TDR checkpoint for {config.env} seed {seed}")

        graph_path = keygraph_path(graph_root)
        if graph_path is None:
            cmd = [
                py,
                "construct_graph.py",
                "--run_group",
                run_group,
                "--save_graph_dir",
                abspath(graph_root),
                "--te_threshold",
                config.te_threshold,
                "--tdr_path",
                abspath(tdr_path),
            ]
            cmd.extend(flags)
            run_phase("construct_graph", cmd, run_dir / "construct_graph.log", env)
            graph_path = keygraph_path(graph_root)
        if graph_path is None:
            raise FileNotFoundError(f"Missing keygraph for {config.env} seed {seed}")

        policy_path = final_checkpoint(policy_root, config.train_steps)
        if policy_path is None:
            cmd = [
                py,
                "train_policy.py",
                "--run_policy_project",
                f"Round{round_id}_GAS_SelfTrain",
                "--run_group",
                run_group,
                "--save_policy_dir",
                abspath(policy_root),
                "--train_steps",
                str(config.train_steps),
                "--log_interval",
                str(args.log_interval),
                "--save_interval",
                str(args.save_interval),
                "--tdr_path",
                abspath(tdr_path),
            ]
            cmd.extend(flags)
            run_phase("train_policy", cmd, run_dir / "train_policy.log", env)
            policy_path = final_checkpoint(policy_root, config.train_steps)
        if policy_path is None:
            raise FileNotFoundError(f"Missing final policy checkpoint for {config.env} seed {seed}")

        cmd = [
            py,
            "evaluate_gas.py",
            "--run_eval_project",
            f"Round{round_id}_GAS_SelfTrain_Eval",
            "--run_group",
            run_group,
            "--save_eval_dir",
            abspath(eval_root),
            "--eval_on_cpu",
            str(config.eval_on_cpu),
            "--eval_episodes",
            str(args.eval_episodes),
            "--eval_video_episodes",
            str(args.eval_video_episodes),
            "--eval_final_goal_threshold",
            "2",
            "--keygraph_path",
            abspath(graph_path),
            "--policy_path",
            abspath(policy_path),
        ]
        cmd.extend(flags)
        run_phase("evaluate_gas", cmd, run_dir / "evaluate_gas.log", env)
        eval_csv = latest_eval_csv(policy_root) or latest_eval_csv(eval_root)

        status = read_status(status_path)
        status.update(
            {
                "tdr_checkpoint": rel(tdr_path),
                "graph_checkpoint": rel(graph_path),
                "policy_checkpoint": rel(policy_path),
                "eval_csv": rel(eval_csv) if eval_csv is not None else None,
                "status": "completed",
                "completed_at": now_iso(),
            }
        )
        write_status(status_path, status)
        return 0
    except Exception as exc:
        status = read_status(status_path)
        status.update({"status": "failed", "failed_at": now_iso(), "error": repr(exc)})
        write_status(status_path, status)
        raise


def write_launch_report(args: argparse.Namespace, envs: list[str], seeds: list[int], configs: dict[str, GasConfig]) -> None:
    report = REPO_ROOT / "reports" / f"round_{args.round}_gas_dynamic_launch.md"
    skipped = ["kitchen-partial-v0 (D4RL, not OGBench; skipped by this OGBench queue)"]
    lines = [
        f"# Round {args.round} GAS Dynamic Download/Training Launch",
        "",
        f"Generated: {now_iso()}.",
        "",
        "- Evidence class while running: `E4_FULL_BUDGET_TRAINED_METHOD` pending completion.",
        "- Baseline-only run: no p_bridge, integrated BARS, oracle-headroom, boundary, or failure-taxonomy interpretation.",
        f"- Seeds: {','.join(map(str, seeds))}.",
        f"- Target OGBench envs: {','.join(envs)}.",
        f"- Dataset root: `{args.dataset_dir}`.",
        f"- Artifact root: `{args.out_root}`.",
        f"- Run root: `{args.run_root}`.",
        f"- GPUs: `{args.gpus}` with slot capacity `{args.gpu_slots_per_gpu}` per GPU.",
        "- Checkpoint policy: full completed stage checkpoints may feed the next stage; interrupted intermediate checkpoints are never resumed.",
        f"- Download uses proxy-aware HTTP(S) environment variables inherited by curl/urllib/aria2.",
        f"- Common datasets are prioritized before additional antmaze, humanoidmaze, and visual datasets.",
        f"- Skipped: {', '.join(skipped)}.",
        "",
        "## Config Summary",
        "",
        "| env | steps | encoder | batch | discount | expectile | alpha | p_aug | way_steps | te | eval_on_cpu | priority | slots |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for env in envs:
        c = configs[env]
        lines.append(
            f"| {env} | {c.train_steps} | {c.encoder} | {c.batch_size} | {c.discount} | {c.tdr_expectile} | {c.alpha} | {c.p_aug} | {c.way_steps} | {c.te_threshold} | {c.eval_on_cpu} | {c.priority} | {c.slot_cost} |"
        )
    lines.extend(
        [
            "",
            "## Live Files",
            "",
            f"- Jobs: `reports/round_{args.round}_gas_dynamic_jobs.tsv`",
            f"- Dataset status: `reports/round_{args.round}_ogbench_download_status.tsv`",
            f"- Events: `{args.run_root}/_orchestrator/events.jsonl`",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_command(args: argparse.Namespace) -> None:
    path = REPO_ROOT / "commands" / f"round_{args.round}_gas_dynamic_launch.sh"
    cmd = [
        "python",
        "scripts/round006_gas_dynamic_orchestrator.py",
        "--round",
        args.round,
        "--dataset-dir",
        args.dataset_dir,
        "--run-root",
        args.run_root,
        "--out-root",
        args.out_root,
        "--seeds",
        args.seeds,
        "--gpus",
        args.gpus,
        "--gpu-slots-per-gpu",
        str(args.gpu_slots_per_gpu),
        "--poll-seconds",
        str(args.poll_seconds),
        "--download-poll-seconds",
        str(args.download_poll_seconds),
    ]
    if args.envs:
        cmd.extend(["--envs", args.envs])
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + " ".join(cmd) + "\n", encoding="utf-8")
    path.chmod(0o755)


def download_one(env_name: str, dataset_dir: Path, log: Path, events: Path) -> bool:
    if dataset_ready(env_name, dataset_dir):
        return True
    from bars.data.ogbench_dataset import ensure_ogbench_dataset_files

    with log.open("a", encoding="utf-8") as f:
        f.write(f"{now_iso()} START download {env_name}\n")
        f.flush()
        try:
            train, val = ensure_ogbench_dataset_files(env_name, str(dataset_dir))
            f.write(f"{now_iso()} DONE download {env_name}: {train} | {val}\n")
            append_jsonl(events, {"time": now_iso(), "event": "download_completed", "env": env_name, "train": train, "val": val})
            return True
        except Exception as exc:
            f.write(f"{now_iso()} FAIL download {env_name}: {exc!r}\n")
            append_jsonl(events, {"time": now_iso(), "event": "download_failed", "env": env_name, "error": repr(exc)})
            return False


def process_alive(pid: int) -> bool:
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        stat = stat_path.read_text(encoding="utf-8", errors="ignore")
        fields = stat.split()
        if len(fields) >= 3 and fields[2] == "Z":
            return False
    except FileNotFoundError:
        return False
    except Exception:
        pass
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def worker_matches(pid: int, env: str, seed: int) -> bool:
    cmdline_path = Path(f"/proc/{pid}/cmdline")
    try:
        raw = cmdline_path.read_bytes()
    except FileNotFoundError:
        return False
    except Exception:
        return process_alive(pid)
    parts = [p.decode("utf-8", errors="ignore") for p in raw.split(b"\0") if p]
    joined = " ".join(parts)
    return (
        "--worker" in parts
        and "round006_gas_dynamic_orchestrator.py" in joined
        and env in parts
        and str(seed) in parts
    )


def job_phase_for_slots(args: argparse.Namespace, config: GasConfig, seed: int) -> str:
    out_seed = Path(args.out_root) / config.env / f"seed{seed}"
    run_seed = Path(args.run_root) / config.env / f"seed{seed}"
    if latest_eval_csv(out_seed / "policy") is not None or latest_eval_csv(run_seed / "evaluate_gas") is not None:
        return "evaluated"
    if final_checkpoint(out_seed / "policy", config.train_steps) is not None:
        return "await_eval"
    if keygraph_path(out_seed / "graph") is not None:
        return "policy"
    if final_checkpoint(out_seed / "tdr", config.train_steps) is not None:
        return "graph"
    return "train"


def job_slot_cost(args: argparse.Namespace, config: GasConfig, seed: int, meta: dict | None = None) -> int:
    phase = job_phase_for_slots(args, config, seed)
    if phase in {"await_eval", "evaluated"}:
        return 0
    if meta is not None:
        return int(meta.get("slot_cost", config.slot_cost))
    return int(config.slot_cost)


def gpu_slots_used(
    args: argparse.Namespace,
    configs: dict[str, GasConfig],
    children: dict[tuple[str, int], dict],
    gpu: str,
) -> int:
    used = 0
    for key, meta in children.items():
        if str(meta["gpu"]) == str(gpu) and process_alive(int(meta["pid"])):
            used += job_slot_cost(args, configs[key[0]], key[1], meta)
    return used


def parse_launch_events(events: Path) -> list[dict]:
    if not events.exists():
        return []
    out: list[dict] = []
    try:
        with events.open(encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("event") == "job_launched":
                    out.append(obj)
    except Exception:
        return []
    return out


def adopt_live_children(
    args: argparse.Namespace,
    envs: list[str],
    seeds: list[int],
    configs: dict[str, GasConfig],
    events: Path,
) -> dict[tuple[str, int], dict]:
    children: dict[tuple[str, int], dict] = {}
    valid_envs = set(envs)
    valid_seeds = set(seeds)
    for obj in parse_launch_events(events):
        env = obj.get("env")
        try:
            seed = int(obj.get("seed"))
            pid = int(obj.get("pid"))
        except Exception:
            continue
        if env not in valid_envs or seed not in valid_seeds:
            continue
        if not process_alive(pid) or not worker_matches(pid, str(env), seed):
            continue
        status = read_status(Path(args.run_root) / str(env) / f"seed{seed}" / "status.json").get("status")
        if status in {"completed", "failed"}:
            continue
        config = configs[str(env)]
        children[(str(env), seed)] = {
            "pid": pid,
            "gpu": str(obj.get("gpu", "")),
            "slot_cost": int(obj.get("slot_cost", config.slot_cost)),
            "started_at": obj.get("started_at", obj.get("time", "")),
            "log": obj.get("log", rel(Path(args.run_root) / str(env) / f"seed{seed}" / "worker_supervisor.log")),
            "adopted": True,
        }
    if children:
        append_jsonl(
            events,
            {
                "time": now_iso(),
                "event": "live_workers_adopted",
                "count": len(children),
                "workers": [
                    {"env": env, "seed": seed, "pid": meta["pid"], "gpu": meta["gpu"]}
                    for (env, seed), meta in sorted(children.items())
                ],
            },
        )
    return children


def collect_score(eval_csv: Path | None) -> float | None:
    if eval_csv is None or not eval_csv.exists():
        return None
    try:
        with eval_csv.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return None
        return float(rows[-1]["eval/overall_episode.success"]) * 100.0
    except Exception:
        return None


def write_jobs_table(
    args: argparse.Namespace,
    envs: list[str],
    seeds: list[int],
    configs: dict[str, GasConfig],
    children: dict[tuple[str, int], dict],
) -> None:
    rows = []
    for env in envs:
        c = configs[env]
        for seed in seeds:
            run_dir = Path(args.run_root) / env / f"seed{seed}"
            status = read_status(run_dir / "status.json")
            eval_csv = latest_eval_csv(Path(args.out_root) / env / f"seed{seed}" / "policy")
            key = (env, seed)
            rows.append(
                {
                    "env": env,
                    "seed": seed,
                    "priority": c.priority,
                    "slot_cost": c.slot_cost,
                    "gpu": status.get("gpu", children.get(key, {}).get("gpu", "")),
                    "pid": children.get(key, {}).get("pid", ""),
                    "status": status.get("status", "pending_dataset" if not dataset_ready(env, Path(args.dataset_dir)) else "queued"),
                    "score_pp": "" if collect_score(eval_csv) is None else f"{collect_score(eval_csv):.1f}",
                    "eval_csv": "" if eval_csv is None else rel(eval_csv),
                    "status_file": rel(run_dir / "status.json"),
                }
            )
    for dest in [
        REPO_ROOT / "reports" / f"round_{args.round}_gas_dynamic_jobs.tsv",
        REPO_ROOT / "rounds" / f"round_{args.round}" / "gas_dynamic_jobs.tsv",
    ]:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)


def write_download_table(args: argparse.Namespace, envs: list[str]) -> None:
    rows = []
    dataset_dir = Path(args.dataset_dir)
    for env in envs:
        train, val = dataset_names(env)
        rows.append(
            {
                "env": env,
                "ready": dataset_ready(env, dataset_dir),
                "train_file": str(dataset_dir / train),
                "train_size": (dataset_dir / train).stat().st_size if (dataset_dir / train).exists() else 0,
                "val_file": str(dataset_dir / val),
                "val_size": (dataset_dir / val).stat().st_size if (dataset_dir / val).exists() else 0,
            }
        )
    for dest in [
        REPO_ROOT / "reports" / f"round_{args.round}_ogbench_download_status.tsv",
        REPO_ROOT / "rounds" / f"round_{args.round}" / "ogbench_download_status.tsv",
    ]:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)


def launch_child(args: argparse.Namespace, config: GasConfig, seed: int, gpu: str, events: Path) -> dict:
    run_dir = Path(args.run_root) / config.env / f"seed{seed}"
    worker_log = run_dir / "worker_supervisor.log"
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--round",
        args.round,
        "--env",
        config.env,
        "--seed",
        str(seed),
        "--gpu",
        str(gpu),
        "--dataset-dir",
        args.dataset_dir,
        "--run-root",
        args.run_root,
        "--out-root",
        args.out_root,
        "--log-interval",
        str(args.log_interval),
        "--save-interval",
        str(args.save_interval),
        "--eval-episodes",
        str(args.eval_episodes),
        "--eval-video-episodes",
        str(args.eval_video_episodes),
    ]
    with worker_log.open("a", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=base_env(Path(args.dataset_dir), offline=True),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    meta = {"pid": proc.pid, "gpu": gpu, "slot_cost": config.slot_cost, "started_at": now_iso(), "log": rel(worker_log)}
    append_jsonl(events, {"time": now_iso(), "event": "job_launched", "env": config.env, "seed": seed, **meta})
    return meta


def orchestrate(args: argparse.Namespace) -> int:
    source_stage24_env()
    dataset_dir = Path(args.dataset_dir)
    run_root = Path(args.run_root)
    out_root = Path(args.out_root)
    ensure_dirs(args.round, run_root, out_root)
    configs = gas_configs()
    envs = split_csv(args.envs) if args.envs else [env for env, _ in sorted(configs.items(), key=lambda kv: kv[1].priority)]
    unknown = [env for env in envs if env not in configs]
    if unknown:
        raise ValueError(f"Unsupported GAS/OGBench envs: {unknown}")
    seeds = parse_seeds(args.seeds)
    events = run_root / "_orchestrator" / "events.jsonl"
    downloader_log = run_root / "_orchestrator" / "download.log"
    append_jsonl(events, {"time": now_iso(), "event": "orchestrator_started", "envs": envs, "seeds": seeds})
    write_launch_report(args, envs, seeds, configs)
    write_command(args)

    next_download_at = 0.0
    download_index = 0
    gpus = split_csv(args.gpus)
    if not gpus:
        raise ValueError("No GPUs specified")
    children = adopt_live_children(args, envs, seeds, configs, events)

    while True:
        # Reap stale child metadata.
        for key, meta in list(children.items()):
            if not process_alive(int(meta["pid"])):
                append_jsonl(events, {"time": now_iso(), "event": "job_process_exited", "env": key[0], "seed": key[1], **meta})
                children.pop(key, None)

        t = time.time()
        if t >= next_download_at and download_index < len(envs):
            for offset in range(len(envs)):
                idx = (download_index + offset) % len(envs)
                env = envs[idx]
                if not dataset_ready(env, dataset_dir):
                    download_one(env, dataset_dir, downloader_log, events)
                    download_index = idx + 1
                    break
            else:
                download_index = len(envs)
            next_download_at = t + args.download_poll_seconds

        # Launch ready jobs in priority order.
        for env in envs:
            config = configs[env]
            if not dataset_ready(env, dataset_dir):
                continue
            for seed in seeds:
                key = (env, seed)
                status_path = run_root / env / f"seed{seed}" / "status.json"
                status = read_status(status_path).get("status")
                if status == "completed" or key in children:
                    continue
                if status == "launched" and latest_eval_csv(out_root / env / f"seed{seed}" / "policy") is not None:
                    continue
                if status == "failed" and not args.retry_failed:
                    continue
                required_slots = job_slot_cost(args, config, seed)
                chosen_gpu = None
                if required_slots == 0:
                    status_gpu = str(read_status(status_path).get("gpu", ""))
                    chosen_gpu = status_gpu if status_gpu in gpus else gpus[0]
                else:
                    for gpu in gpus:
                        if gpu_slots_used(args, configs, children, gpu) + required_slots <= args.gpu_slots_per_gpu:
                            chosen_gpu = gpu
                            break
                if chosen_gpu is None:
                    break
                children[key] = launch_child(args, config, seed, chosen_gpu, events)

        write_download_table(args, envs)
        write_jobs_table(args, envs, seeds, configs, children)

        all_datasets_done = all(dataset_ready(env, dataset_dir) for env in envs) or download_index >= len(envs)
        all_jobs_terminal = True
        for env in envs:
            for seed in seeds:
                status = read_status(run_root / env / f"seed{seed}" / "status.json").get("status")
                if status != "completed" and not (status == "failed" and not args.retry_failed):
                    all_jobs_terminal = False
                    break
            if not all_jobs_terminal:
                break
        if all_datasets_done and all_jobs_terminal and not children:
            append_jsonl(events, {"time": now_iso(), "event": "orchestrator_completed"})
            return 0

        time.sleep(args.poll_seconds)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true")
    p.add_argument("--round", default=DEFAULT_ROUND)
    p.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    p.add_argument("--run-root", default="runs_round006_gas_dynamic")
    p.add_argument("--out-root", default="artifacts/gas_selftrain_round006")
    p.add_argument("--envs", default="")
    p.add_argument("--seeds", default="42,43,44,45,46")
    p.add_argument("--gpus", default="0,1,2,3,4,5")
    p.add_argument("--gpu-slots-per-gpu", type=int, default=int(os.environ.get("ROUND006_GPU_SLOTS_PER_GPU", "2")))
    p.add_argument("--poll-seconds", type=int, default=60)
    p.add_argument("--download-poll-seconds", type=int, default=30)
    p.add_argument("--log-interval", type=int, default=5000)
    p.add_argument("--save-interval", type=int, default=100000)
    p.add_argument("--eval-episodes", type=int, default=49)
    p.add_argument("--eval-video-episodes", type=int, default=1)
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--env", default="")
    p.add_argument("--seed", default="")
    p.add_argument("--gpu", default="")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.worker:
        return run_worker(args)
    return orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
