from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from bars.conditioning.lowcond_policy import LowCondActor, save_lowcond_actor
from bars.external.gas_artifacts import resolve_gas_artifacts
from bars.external.gas_backbone import GASBackbone
from stage26_lowcond_common import build_lowcond_batch, load_stats, parse_ints, sample_future_pairs, trajectory_end_for_indices


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
    return gas


def _batched_phi(gas: GASBackbone, obs: np.ndarray, batch_size: int) -> np.ndarray:
    parts = []
    for st in range(0, len(obs), batch_size):
        parts.append(np.asarray(gas.get_phi(obs[st : st + batch_size]), dtype=np.float32))
    return np.concatenate(parts, axis=0)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _task_goal_xy_from_env(env) -> np.ndarray:
    task_infos = getattr(getattr(env, "unwrapped", env), "task_infos", None)
    if task_infos is None:
        task_infos = getattr(env, "task_infos", None)
    goals = []
    for info in task_infos or []:
        if isinstance(info, dict) and "goal_xy" in info:
            goals.append(info["goal_xy"])
    return np.asarray(goals, dtype=np.float32)


def _make_task_goals(
    source: str,
    *,
    env,
    observations: np.ndarray,
    local: np.ndarray,
    src: np.ndarray,
    terminals: np.ndarray,
) -> np.ndarray:
    if source == "local":
        return local
    end_idx = trajectory_end_for_indices(terminals, src)
    if source == "trajectory_end":
        return observations[end_idx]
    if source == "nearest_task_goal":
        goals_xy = _task_goal_xy_from_env(env)
        if len(goals_xy) == 0:
            return observations[end_idx]
        end_xy = observations[end_idx, :2]
        nearest = np.linalg.norm(end_xy[:, None, :] - goals_xy[None, :, :], axis=-1).argmin(axis=1)
        return goals_xy[nearest].astype(np.float32)
    raise ValueError(f"unknown task goal source: {source}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train Stage-26 lowcond BC actor.")
    parser.add_argument("--env", required=True)
    parser.add_argument("--gas-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--artifact-root", default="artifacts/gas_ogbench_offline_full_20260522_165138")
    parser.add_argument("--stats", required=True)
    parser.add_argument("--variant", choices=["tdr_only", "factor_only", "full", "full_localres", "full_nomask", "full_rawdist"], default="full")
    parser.add_argument("--task-goal-source", choices=["local", "trajectory_end", "nearest_task_goal"], default="local")
    parser.add_argument("--num-samples", type=int, default=200000)
    parser.add_argument("--horizons", default="1,2,4,8,16,32,64,128")
    parser.add_argument("--phi-batch-size", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dims", default="256,256,256")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    os.environ["CUDA_VISIBLE_DEVICES"] = "" if str(args.gpu).lower() in {"cpu", "-1", ""} else str(args.gpu)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and str(args.gpu).lower() not in {"cpu", "-1", ""} else "cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")

    stats = load_stats(args.stats)
    gas = _load_gas(args.env, args.gas_seed, args.gpu, args.artifact_root)
    env, train_dataset, _ = gas.load_env_and_dataset()
    observations = np.asarray(train_dataset["observations"], dtype=np.float32)
    actions = np.asarray(train_dataset["actions"], dtype=np.float32)
    terminals = np.asarray(train_dataset.get("terminals", np.zeros((len(observations),), dtype=np.float32)))
    src, dst, _ = sample_future_pairs(terminals, parse_ints(args.horizons), args.num_samples, args.seed)
    obs = observations[src]
    local = observations[dst]
    task_goals = _make_task_goals(
        args.task_goal_source,
        env=env,
        observations=observations,
        local=local,
        src=src,
        terminals=terminals,
    )
    target_actions = actions[src]
    phi_obs = _batched_phi(gas, obs, args.phi_batch_size)
    phi_local = _batched_phi(gas, local, args.phi_batch_size)
    cond = build_lowcond_batch(
        obs=obs,
        local_target_obs=local,
        task_goal=task_goals,
        phi_obs=phi_obs,
        phi_local=phi_local,
        stats=stats,
        variant=args.variant,
    )
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    cond_t = torch.as_tensor(cond, dtype=torch.float32, device=device)
    act_t = torch.as_tensor(target_actions, dtype=torch.float32, device=device)
    hidden = tuple(parse_ints(args.hidden_dims))
    model = LowCondActor(obs_dim=obs.shape[1], cond_dim=cond.shape[1], action_dim=target_actions.shape[1], hidden_dims=hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    rng = np.random.default_rng(args.seed)
    logs: list[dict] = []
    for step in range(1, args.steps + 1):
        idx = rng.integers(0, len(obs), size=args.batch_size)
        pred = model(obs_t[idx], cond_t[idx])
        loss = loss_fn(pred, act_t[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        if step == 1 or step % max(args.steps // 20, 1) == 0 or step == args.steps:
            with torch.no_grad():
                eval_idx = rng.integers(0, len(obs), size=min(8192, len(obs)))
                eval_loss = loss_fn(model(obs_t[eval_idx], cond_t[eval_idx]), act_t[eval_idx]).item()
            logs.append({"step": step, "train_loss": float(loss.item()), "sample_mse": float(eval_loss)})
            print(json.dumps(logs[-1]), flush=True)
    _write_csv(out / "train_log.csv", logs)
    save_lowcond_actor(
        out / "actor.pt",
        model.cpu(),
        {
            "env": args.env,
            "gas_seed": args.gas_seed,
            "variant": args.variant,
            "stats": args.stats,
            "num_samples": int(len(obs)),
            "task_goal_source": args.task_goal_source,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
