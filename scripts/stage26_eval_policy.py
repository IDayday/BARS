from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from bars.conditioning.lowcond_policy import load_lowcond_actor
from bars.external.gas_artifacts import resolve_gas_artifacts
from bars.external.gas_backbone import GASBackbone
from stage26_lowcond_common import build_lowcond_batch, load_stats


def _load_gas(env_name: str, gas_seed: int, gpu: str, artifact_root: str) -> GASBackbone:
    artifacts = resolve_gas_artifacts(env_name, gas_seed, artifact_root)
    gas = GASBackbone(
        env_name=env_name,
        seed=gas_seed,
        artifact_root=Path(artifact_root),
        gas_repo_path=Path("external_src/GAS"),
        gpu=gpu,
        artifacts=artifacts,
    )
    gas.load_policy(artifacts.policy_checkpoint)
    gas.load_keygraph(artifacts.keygraph)
    return gas


def _parse_tasks(spec: str, available: int) -> list[int]:
    if spec == "all":
        return list(range(1, available + 1))
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return out


def _success(info: dict) -> float:
    ep = info.get("episode", {}) if isinstance(info, dict) else {}
    for src in (info, ep):
        for key in ("success", "goal_achieved", "is_success"):
            if key in src:
                return float(src[key])
    return 0.0


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _lookup_obs_for_phi(gas: GASBackbone, limit: int, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    _, train_dataset, _ = gas.load_env_and_dataset()
    obs = np.asarray(train_dataset["observations"], dtype=np.float32)
    if limit > 0:
        obs = obs[:limit]
    phi_parts = []
    for st in range(0, len(obs), batch_size):
        phi_parts.append(np.asarray(gas.get_phi(obs[st : st + batch_size]), dtype=np.float32))
    return obs, np.concatenate(phi_parts, axis=0)


def _nearest_obs(target_phi: np.ndarray, lookup_obs: np.ndarray, lookup_phi: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(lookup_phi - np.asarray(target_phi, dtype=np.float32)[None, :], axis=-1)
    return lookup_obs[int(np.argmin(d))]


def _lowcond_action(model, device, stats, variant: str, gas: GASBackbone, obs: np.ndarray, local_obs: np.ndarray, final_goal: np.ndarray) -> np.ndarray:
    phi_obs = np.asarray(gas.get_phi(obs), dtype=np.float32)[None, :]
    phi_local = np.asarray(gas.get_phi(local_obs), dtype=np.float32)[None, :]
    cond = build_lowcond_batch(
        obs=np.asarray(obs, dtype=np.float32)[None, :],
        local_target_obs=np.asarray(local_obs, dtype=np.float32)[None, :],
        task_goal=np.asarray(final_goal, dtype=np.float32)[None, :],
        phi_obs=phi_obs,
        phi_local=phi_local,
        stats=stats,
        variant=variant,
    )
    with torch.no_grad():
        pred = model(
            torch.as_tensor(np.asarray(obs, dtype=np.float32)[None, :], device=device),
            torch.as_tensor(cond, dtype=torch.float32, device=device),
        )
    return np.clip(pred.cpu().numpy()[0], -1.0, 1.0)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate Stage-26 policies with fixed GAS graph planner.")
    parser.add_argument("--mode", choices=["gas_graph_policy", "gas_graph_lowcond_policy"], required=True)
    parser.add_argument("--env", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gas-seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--artifact-root", default="artifacts/gas_ogbench_offline_full_20260522_165138")
    parser.add_argument("--stats")
    parser.add_argument("--actor")
    parser.add_argument("--variant", choices=["tdr_only", "factor_only", "full", "full_localres", "full_nomask", "full_rawdist"], default="full")
    parser.add_argument("--lookup-observations", type=int, default=50000)
    parser.add_argument("--lookup-batch-size", type=int, default=4096)
    parser.add_argument("--gas-final-goal-threshold", type=int, default=2)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    os.environ["CUDA_VISIBLE_DEVICES"] = "" if str(args.gpu).lower() in {"cpu", "-1", ""} else str(args.gpu)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    gas = _load_gas(args.env, args.gas_seed, args.gpu, args.artifact_root)
    env, _, _ = gas.load_env_and_dataset()
    model = stats = None
    device = torch.device("cuda" if torch.cuda.is_available() and args.mode == "gas_graph_lowcond_policy" and str(args.gpu).lower() not in {"cpu", "-1", ""} else "cpu")
    lookup_obs = lookup_phi = None
    if args.mode == "gas_graph_lowcond_policy":
        if not args.stats or not args.actor:
            raise ValueError("--stats and --actor are required for gas_graph_lowcond_policy")
        stats = load_stats(args.stats)
        model, actor_extra = load_lowcond_actor(args.actor, map_location=device)
        model.to(device).eval()
        lookup_obs, lookup_phi = _lookup_obs_for_phi(gas, args.lookup_observations, args.lookup_batch_size)
    else:
        actor_extra = {}
    eval_args = vars(args).copy()
    if actor_extra:
        eval_args["actor_extra"] = actor_extra
    (out / "eval_args.json").write_text(json.dumps(eval_args, indent=2, sort_keys=True), encoding="utf-8")
    task_count = len(getattr(getattr(env, "unwrapped", env), "task_infos", []))
    rows = []
    debug_rows = []
    for task_id in _parse_tasks(args.tasks, task_count):
        for ep in range(args.episodes):
            reset_seed = int(args.seed * 1000 + ep)
            env, obs, goal, _, _, _ = gas.setup_task_env(env, args.env, task_id, reset_seed, render_goal=False)
            gas_phi_goal = gas.get_phi(goal)
            phi_obs = gas.get_phi(obs)
            gas.key_graph.precompute_shortest_paths_to_all_tasks({task_id: goal}, {task_id: gas_phi_goal})
            gas_shortest_path = gas.key_graph.get_shortest_path(task_id=task_id, source=phi_obs, force_closest=True)
            done = False
            steps = 0
            final_goal_on = False
            final_goal_mode_steps = 0
            subgoal_switch_count = 0
            last_node = None
            info = {}
            while not done:
                phi_obs = gas.get_phi(obs)
                selected_node = -1
                path_len = 0
                if not final_goal_on:
                    cached_path = gas.key_graph.get_shortest_path(task_id=task_id, source=phi_obs)
                    if cached_path is not None:
                        gas_shortest_path = cached_path
                    path_len = len(gas_shortest_path)
                    distances = np.linalg.norm(np.asarray(gas_shortest_path) - phi_obs, axis=1)
                    valid = np.flatnonzero(distances <= float(gas.config["way_steps"]))
                    selected_node = int(valid[-1]) if len(valid) else 0
                    if len(gas_shortest_path) <= int(args.gas_final_goal_threshold):
                        final_goal_on = True
                if final_goal_on:
                    final_goal_mode_steps += 1
                    local_obs = goal
                    action = gas.sample_action(obs, gas_phi_goal) if args.mode == "gas_graph_policy" else _lowcond_action(model, device, stats, args.variant, gas, obs, local_obs, goal)
                else:
                    target_phi = np.asarray(gas_shortest_path)[selected_node]
                    if args.mode == "gas_graph_policy":
                        action = gas.sample_action(obs, target_phi)
                    else:
                        local_obs = _nearest_obs(target_phi, lookup_obs, lookup_phi)
                        action = _lowcond_action(model, device, stats, args.variant, gas, obs, local_obs, goal)
                if last_node is not None and selected_node != last_node:
                    subgoal_switch_count += 1
                last_node = selected_node
                next_obs, reward, done, info = gas.step_env(env, args.env, np.asarray(action))
                steps += 1
                debug_rows.append({"task_id": task_id, "episode": ep, "step": steps, "selected_node": selected_node, "path_len": path_len, "final_goal_mode": int(final_goal_on), "mode": args.mode, "variant": args.variant})
                obs = next_obs
            rows.append(
                {
                    "env": args.env,
                    "seed": args.seed,
                    "gas_seed": args.gas_seed,
                    "mode": args.mode,
                    "variant": args.variant if args.mode == "gas_graph_lowcond_policy" else "gas",
                    "task_id": task_id,
                    "episode": ep,
                    "episodes": 1,
                    "success": _success(info),
                    "steps": steps,
                    "final_goal_mode_steps": final_goal_mode_steps,
                    "subgoal_switch_count": subgoal_switch_count,
                    "no_path_rate": 0.0,
                }
            )
    _write_csv(out / "eval.csv", rows)
    _write_csv(out / "step_debug.csv", debug_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
