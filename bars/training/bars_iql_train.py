from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

from bars.common.checkpoint import load_checkpoint, save_checkpoint
from bars.common.logging import CSVLogger
from bars.common.progress import tqdm_kwargs
from bars.common.stopper import Stopper
from bars.data.trajectories import OfflineDataset
from bars.models.bars_iql import BARSIQLPolicy
from bars.training.goal_sampler import GraphAwareGoalSampler


def expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    weight = torch.where(diff > 0, torch.full_like(diff, float(expectile)), torch.full_like(diff, 1.0 - float(expectile)))
    return weight * diff.pow(2)


def _success_reward(next_obs: np.ndarray, goal_obs: np.ndarray, goal_dim: int, threshold: float, mode: str) -> np.ndarray:
    dist = np.linalg.norm(next_obs[:, :goal_dim] - goal_obs[:, :goal_dim], axis=1).astype(np.float32)
    success = dist <= float(threshold)
    mode = str(mode).lower()
    if mode in {"sparse", "sparse_minus_one", "minus_one"}:
        reward = np.where(success, 0.0, -1.0).astype(np.float32)
    elif mode in {"binary", "success_one"}:
        reward = success.astype(np.float32)
    elif mode in {"dense", "dense_distance"}:
        scale = float(max(1e-6, threshold))
        reward = (-dist / scale).astype(np.float32)
        reward[success] = 0.0
    else:
        raise ValueError(f"Unknown policy.reward_mode={mode}")
    return reward.astype(np.float32)


def train_bars_iql_policy(
    dataset: OfflineDataset,
    cfg: Dict,
    run_dir: str,
    device,
    logger: CSVLogger,
    stopper: Optional[Stopper] = None,
    embeddings: Optional[np.ndarray] = None,
) -> BARSIQLPolicy:
    pcfg = cfg.get("policy", {})
    ckpt_path = os.path.join(run_dir, "checkpoints", "policy.pt")
    model = BARSIQLPolicy(
        dataset.obs_dim,
        dataset.action_dim,
        actor_hidden_dims=tuple(pcfg.get("actor_hidden_dims", pcfg.get("hidden_dims", [256, 256, 256]))),
        value_hidden_dims=tuple(pcfg.get("value_hidden_dims", [256, 256])),
        goal_delta=bool(pcfg.get("goal_delta", True)),
        value_temperature=float(pcfg.get("value_temperature", 5.0)),
    ).to(device)
    if pcfg.get("load_if_exists", True) and os.path.exists(ckpt_path):
        load_checkpoint(ckpt_path, model, map_location=str(device))
        logger.log({"phase": "bars_iql", "event": "loaded", "path": ckpt_path})
        return model

    steps = int(pcfg.get("steps", 100000))
    batch_size = int(pcfg.get("batch_size", 1024))
    log_every = int(pcfg.get("log_every", 1000))
    discount = float(pcfg.get("discount", 0.99))
    expectile = float(pcfg.get("expectile", 0.7))
    temperature = float(pcfg.get("temperature", 3.0))
    awr_weight_clip = float(pcfg.get("awr_weight_clip", 100.0))
    actor_loss_weight = float(pcfg.get("actor_loss_weight", 1.0))
    value_loss_weight = float(pcfg.get("value_loss_weight", 1.0))
    q_loss_weight = float(pcfg.get("q_loss_weight", 1.0))
    bc_loss_weight = float(pcfg.get("bc_loss_weight", 0.05))
    goal_dim = int(pcfg.get("goal_dim", cfg.get("eval", {}).get("goal_dim", 2)))
    threshold = float(pcfg.get("success_threshold", cfg.get("eval", {}).get("subgoal_threshold", 0.5)))
    reward_mode = str(pcfg.get("reward_mode", "sparse_minus_one"))

    sampler = GraphAwareGoalSampler(dataset, cfg, embeddings=embeddings, run_dir=run_dir, logger=logger)
    obs_norm = dataset.obs_normalizer.encode(dataset.observations).astype(np.float32)
    next_norm = dataset.obs_normalizer.encode(dataset.next_observations).astype(np.float32)
    actions = dataset.actions.astype(np.float32, copy=False)
    raw_next = dataset.next_observations.astype(np.float32, copy=False)
    raw_goal = dataset.observations.astype(np.float32, copy=False)
    raw_next_goal = raw_next[:, :goal_dim].astype(np.float32, copy=False)
    raw_goal_goal = raw_goal[:, :goal_dim].astype(np.float32, copy=False)
    cache_default = bool(getattr(device, "type", str(device)) == "cuda")
    cache_on_device = bool(pcfg.get("cache_dataset_on_device", cache_default))
    cache_max_mb = float(pcfg.get("cache_dataset_max_mb", 768))
    cache_mb = float((obs_norm.nbytes + next_norm.nbytes + actions.nbytes + raw_next_goal.nbytes + raw_goal_goal.nbytes) / (1024 * 1024))
    obs_all = next_all = actions_all = raw_next_goal_all = raw_goal_goal_all = None
    if cache_on_device and cache_mb <= cache_max_mb:
        obs_all = torch.as_tensor(obs_norm, dtype=torch.float32, device=device)
        next_all = torch.as_tensor(next_norm, dtype=torch.float32, device=device)
        actions_all = torch.as_tensor(actions, dtype=torch.float32, device=device)
        raw_next_goal_all = torch.as_tensor(raw_next_goal, dtype=torch.float32, device=device)
        raw_goal_goal_all = torch.as_tensor(raw_goal_goal, dtype=torch.float32, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(pcfg.get("lr", 3e-4)), weight_decay=float(pcfg.get("weight_decay", 1e-5)))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(pcfg.get("amp", True)) and getattr(device, "type", str(device)) == "cuda")
    logger.log({
        "phase": "bars_iql",
        "event": "train_start",
        "steps": steps,
        "batch_size": batch_size,
        "discount": discount,
        "expectile": expectile,
        "temperature": temperature,
        "reward_mode": reward_mode,
        "goal_dim": goal_dim,
        "success_threshold": threshold,
        "cache_dataset_on_device": int(obs_all is not None),
        "cache_dataset_mb": cache_mb,
    })

    reward_mode_l = str(reward_mode).lower()
    if reward_mode_l not in {"sparse", "sparse_minus_one", "minus_one", "binary", "success_one", "dense", "dense_distance"}:
        raise ValueError(f"Unknown policy.reward_mode={reward_mode}")

    for step in trange(steps, desc="train_bars_iql", **tqdm_kwargs(cfg)):
        if stopper is not None and stopper.stop_requested:
            break
        batch = sampler.sample(batch_size)
        if obs_all is not None:
            obs_idx_t = torch.as_tensor(batch.obs_idx, dtype=torch.long, device=device)
            goal_idx_t = torch.as_tensor(batch.goal_idx, dtype=torch.long, device=device)
            obs = obs_all[obs_idx_t]
            true_next = next_all[obs_idx_t]
            goal = obs_all[goal_idx_t]
            act = actions_all[obs_idx_t]
            dist = torch.linalg.vector_norm(raw_next_goal_all[obs_idx_t] - raw_goal_goal_all[goal_idx_t], dim=1)
            success = dist <= threshold
            if reward_mode_l in {"sparse", "sparse_minus_one", "minus_one"}:
                rew = torch.where(success, torch.zeros_like(dist), -torch.ones_like(dist))
            elif reward_mode_l in {"binary", "success_one"}:
                rew = success.to(torch.float32)
            else:
                scale = float(max(1e-6, threshold))
                rew = -dist / scale
                rew = torch.where(success, torch.zeros_like(rew), rew)
            done = success.to(torch.float32)
        else:
            obs = torch.as_tensor(obs_norm[batch.obs_idx], dtype=torch.float32, device=device)
            true_next = torch.as_tensor(next_norm[batch.obs_idx], dtype=torch.float32, device=device)
            goal = torch.as_tensor(obs_norm[batch.goal_idx], dtype=torch.float32, device=device)
            act = torch.as_tensor(actions[batch.obs_idx], dtype=torch.float32, device=device)
            rewards_np = _success_reward(raw_next[batch.obs_idx], raw_goal[batch.goal_idx], goal_dim, threshold, reward_mode)
            success_np = (np.linalg.norm(raw_next_goal[batch.obs_idx] - raw_goal_goal[batch.goal_idx], axis=1) <= threshold).astype(np.float32)
            rew = torch.as_tensor(rewards_np, dtype=torch.float32, device=device)
            done = torch.as_tensor(success_np, dtype=torch.float32, device=device)

        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            with torch.no_grad():
                target_v = model.v(true_next, goal)
                q_target = rew + discount * (1.0 - done) * target_v
            q1 = model.q1(obs, goal, act)
            q2 = model.q2(obs, goal, act)
            q_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

            q_min = torch.minimum(q1.detach(), q2.detach())
            v = model.v(obs, goal)
            v_loss = expectile_loss(q_min - v, expectile).mean()

            adv = q_min - v.detach()
            weights = torch.exp(adv / max(temperature, 1e-6)).clamp(max=awr_weight_clip).detach()
            pred_act = model.actor(obs, goal)
            actor_mse = ((pred_act - act) ** 2).mean(dim=-1)
            actor_loss = (weights * actor_mse).mean()
            bc_loss = F.mse_loss(pred_act, act)
            loss = q_loss_weight * q_loss + value_loss_weight * v_loss + actor_loss_weight * actor_loss + bc_loss_weight * bc_loss
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(pcfg.get("grad_clip", 10.0)))
        scaler.step(opt)
        scaler.update()

        if step % log_every == 0 or step == steps - 1:
            mix = sampler.format_mix(batch.kind)
            logger.log({
                "phase": "bars_iql",
                "step": step,
                "loss": float(loss.detach().cpu()),
                "q_loss": float(q_loss.detach().cpu()),
                "v_loss": float(v_loss.detach().cpu()),
                "actor_loss": float(actor_loss.detach().cpu()),
                "bc_loss": float(bc_loss.detach().cpu()),
                "adv_mean": float(adv.detach().mean().cpu()),
                "awr_weight_mean": float(weights.detach().mean().cpu()),
                "reward_mean": float(rew.detach().mean().cpu()),
                "success_frac": float(done.detach().mean().cpu()),
                "goal_mix_batch": mix,
            })

    save_checkpoint(ckpt_path, model, optimizer=opt, policy_type="bars_iql", obs_mean=dataset.obs_normalizer.mean, obs_std=dataset.obs_normalizer.std)
    logger.log({"phase": "bars_iql", "event": "saved", "path": ckpt_path})
    return model
