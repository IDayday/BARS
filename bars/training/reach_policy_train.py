from __future__ import annotations

"""Policy-conditioned reachability training for Full-BARS.

This trainer keeps the final scorer in the same BARSGraph-compatible form as the
existing ReachabilityModel, but biases training toward the low-level policy that
will execute the graph.  If the policy exposes a value/reachability estimate, the
MLP is also regularized to agree with that estimate on sampled pairs.
"""

import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

from bars.common.checkpoint import load_checkpoint, save_checkpoint
from bars.common.logging import CSVLogger
from bars.common.progress import tqdm_kwargs
from bars.common.stopper import Stopper
from bars.data.trajectories import OfflineDataset
from bars.models.reachability import ReachabilityModel
from bars.graph.ann import KNNIndex


def _sample_cross_random_pairs(dataset: OfflineDataset, n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    src = dataset.sample_indices(n, rng)
    dst = dataset.sample_indices(n, rng)
    same = dataset.traj_id[src] == dataset.traj_id[dst]
    tries = 0
    while same.any() and tries < 20:
        dst[same] = dataset.sample_indices(int(same.sum()), rng)
        same = dataset.traj_id[src] == dataset.traj_id[dst]
        tries += 1
    return src, dst


def _sample_far_same_traj_negatives(dataset: OfflineDataset, n: int, rng: np.random.Generator, min_dt: int, max_dt: int) -> Tuple[np.ndarray, np.ndarray]:
    try:
        src, dst, _ = dataset.sample_future_pairs(n, max_dt, rng, min_dt=min_dt)
        return src, dst
    except Exception:
        return _sample_cross_random_pairs(dataset, n, rng)


def _build_latent_hard_negative_pool(dataset: OfflineDataset, embeddings: np.ndarray, cfg: Dict, rng: np.random.Generator) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    rcfg = cfg.get("reachability", {})
    if not bool(rcfg.get("use_hard_neg", True)):
        return None
    n_pool = min(int(rcfg.get("hard_neg_pool", 20000)), dataset.size)
    k = int(rcfg.get("hard_neg_knn", 16))
    if n_pool <= 1 or k <= 0:
        return None
    idx = rng.choice(dataset.size, size=n_pool, replace=False)
    ann = KNNIndex.from_config(embeddings[idx].astype(np.float32), cfg, prefix="ann")
    neigh = ann.kneighbors(embeddings[idx].astype(np.float32), min(k + 1, n_pool), return_distance=False)
    src, dst = [], []
    horizon = int(rcfg.get("horizon", 30))
    for row, ns in enumerate(neigh):
        i = int(idx[row])
        for n in ns[1:]:
            j = int(idx[int(n)])
            same = dataset.traj_id[i] == dataset.traj_id[j]
            dt = int(dataset.timestep[j] - dataset.timestep[i]) if same else 10**9
            if (not same) or dt <= 0 or dt > horizon:
                src.append(i); dst.append(j); break
    if not src:
        return None
    return np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)


def _policy_pair_prob_tensor(
    policy,
    dataset: OfflineDataset,
    src: np.ndarray,
    dst: np.ndarray,
    device,
    default: float = 0.5,
    batch_size: int = 65536,
) -> torch.Tensor:
    n = int(len(src))
    if n == 0:
        return torch.empty(0, dtype=torch.float32, device=device)
    if policy is None:
        return torch.full((n,), float(default), dtype=torch.float32, device=device)
    if hasattr(policy, "reachability_prob_batch"):
        try:
            out = policy.reachability_prob_batch(
                dataset.observations[src],
                dataset.observations[dst],
                dataset.obs_normalizer,
                device=str(device),
                batch_size=batch_size,
                return_tensor=True,
            )
            return torch.as_tensor(out, dtype=torch.float32, device=device).view(-1)
        except Exception:
            return torch.full((n,), float(default), dtype=torch.float32, device=device)
    if not hasattr(policy, "reachability_prob"):
        return torch.full((n,), float(default), dtype=torch.float32, device=device)
    vals = []
    for i, j in zip(src, dst):
        try:
            vals.append(float(policy.reachability_prob(dataset.observations[int(i)], dataset.observations[int(j)], dataset.obs_normalizer, device=str(device))))
        except Exception:
            vals.append(float(default))
    return torch.as_tensor(vals, dtype=torch.float32, device=device)


def _soft_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if weights is not None:
        loss = loss * weights
    return loss.mean()


def train_policy_conditioned_reachability(
    dataset: OfflineDataset,
    embeddings: np.ndarray,
    cfg: Dict,
    run_dir: str,
    device,
    logger: CSVLogger,
    stopper: Optional[Stopper] = None,
    policy=None,
) -> ReachabilityModel:
    rcfg = cfg.get("reachability", {})
    ckpt_path = os.path.join(run_dir, "checkpoints", "reachability.pt")
    model = ReachabilityModel(embeddings.shape[1], tuple(rcfg.get("hidden_dims", [256, 256]))).to(device)
    if rcfg.get("load_if_exists", True) and os.path.exists(ckpt_path):
        load_checkpoint(ckpt_path, model, map_location=str(device))
        logger.log({"phase": "reachability", "event": "loaded", "path": ckpt_path, "type": "policy_conditioned"})
        return model

    batch_size = int(rcfg.get("batch_size", 2048))
    steps = int(rcfg.get("steps", 20000))
    horizon = int(rcfg.get("horizon", 30))
    log_every = int(rcfg.get("log_every", 500))
    pos_frac = float(rcfg.get("pos_frac", 0.35))
    unlabeled_frac = float(rcfg.get("unlabeled_frac", 0.25))
    hard_weight = float(rcfg.get("hard_neg_weight", 0.75))
    same_neg_weight = float(rcfg.get("same_neg_weight", 1.0))
    cross_random_weight = float(rcfg.get("cross_random_weight", 0.10))
    policy_consistency_weight = float(rcfg.get("policy_consistency_weight", 0.25))
    policy_soft_target_weight = float(rcfg.get("policy_soft_target_weight", 0.15))
    policy_prob_batch_size = int(rcfg.get("policy_prob_batch_size", 65536))
    far_min_dt = int(rcfg.get("far_neg_min_dt", horizon + 1))
    far_max_dt = int(rcfg.get("far_neg_max_dt", max(horizon * 4, far_min_dt + 1)))
    rng = np.random.default_rng(int(cfg.get("seed", 0)) + 143)
    hard_pool = _build_latent_hard_negative_pool(dataset, embeddings, cfg, rng)
    z_all = torch.as_tensor(embeddings, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(rcfg.get("lr", 3e-4)), weight_decay=float(rcfg.get("weight_decay", 1e-5)))

    n_pos = max(1, int(batch_size * pos_frac))
    n_unl = max(0, int(batch_size * unlabeled_frac))
    n_base_neg = max(1, batch_size - n_pos - n_unl)
    logger.log({
        "phase": "reachability",
        "event": "train_start",
        "type": "policy_conditioned",
        "batch_size": batch_size,
        "n_pos": n_pos,
        "n_unlabeled": n_unl,
        "n_base_neg": n_base_neg,
        "policy_consistency_weight": policy_consistency_weight,
        "policy_soft_target_weight": policy_soft_target_weight,
        "policy_prob_batch_size": policy_prob_batch_size,
        "hard_pool_size": 0 if hard_pool is None else len(hard_pool[0]),
    })

    for step in trange(steps, desc="train_policy_reachability", **tqdm_kwargs(cfg)):
        if stopper is not None and stopper.stop_requested:
            break
        pos_src, pos_dst, _ = dataset.sample_future_pairs(n_pos, horizon, rng)
        n_far = n_base_neg // 2
        far_src, far_dst = _sample_far_same_traj_negatives(dataset, n_far, rng, min_dt=far_min_dt, max_dt=far_max_dt) if n_far > 0 else (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
        n_cross = max(0, n_base_neg - n_far)
        cross_src, cross_dst = _sample_cross_random_pairs(dataset, n_cross, rng) if n_cross > 0 else (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
        unl_src, unl_dst = _sample_cross_random_pairs(dataset, n_unl, rng) if n_unl > 0 else (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))

        hard_src_parts = [far_src]
        hard_dst_parts = [far_dst]
        hard_w_parts = [np.full(len(far_src), same_neg_weight, dtype=np.float32)]
        if len(cross_src):
            hard_src_parts.append(cross_src); hard_dst_parts.append(cross_dst)
            hard_w_parts.append(np.full(len(cross_src), cross_random_weight, dtype=np.float32))
        if hard_pool is not None and hard_weight > 0:
            hs, hd = hard_pool
            hn = min(n_base_neg, len(hs))
            hi = rng.choice(len(hs), size=hn, replace=len(hs) < hn)
            hard_src_parts.append(hs[hi]); hard_dst_parts.append(hd[hi])
            hard_w_parts.append(np.full(hn, hard_weight, dtype=np.float32))
        hard_src = np.concatenate(hard_src_parts) if hard_src_parts else np.empty(0, dtype=np.int64)
        hard_dst = np.concatenate(hard_dst_parts) if hard_dst_parts else np.empty(0, dtype=np.int64)
        hard_w = np.concatenate(hard_w_parts) if hard_w_parts else np.empty(0, dtype=np.float32)

        src = np.concatenate([pos_src, hard_src, unl_src])
        dst = np.concatenate([pos_dst, hard_dst, unl_dst])
        labels = np.concatenate([
            np.ones(len(pos_src), dtype=np.float32),
            np.zeros(len(hard_src), dtype=np.float32),
            np.full(len(unl_src), 0.5, dtype=np.float32),
        ])
        weights = np.concatenate([
            np.ones(len(pos_src), dtype=np.float32),
            hard_w if len(hard_w) else np.empty(0, dtype=np.float32),
            np.full(len(unl_src), policy_soft_target_weight, dtype=np.float32),
        ])
        src_t = torch.as_tensor(src, dtype=torch.long, device=device)
        dst_t = torch.as_tensor(dst, dtype=torch.long, device=device)
        logits = model(z_all[src_t], z_all[dst_t])
        label_t = torch.as_tensor(labels, dtype=torch.float32, device=device)
        if policy is not None and policy_consistency_weight > 0:
            p_soft = _policy_pair_prob_tensor(policy, dataset, src, dst, device=device, default=0.5, batch_size=policy_prob_batch_size)
            # Keep strong labels strong; use policy value mainly for unlabeled and as a mild regularizer.
            target_t = (1.0 - policy_soft_target_weight) * label_t + policy_soft_target_weight * p_soft
        else:
            p_soft = torch.full((len(src),), 0.5, dtype=torch.float32, device=device)
            target_t = label_t
        weight_t = torch.as_tensor(weights, dtype=torch.float32, device=device)
        loss = _soft_bce_with_logits(logits, target_t, weight_t)
        if policy is not None and policy_consistency_weight > 0:
            p_t = torch.as_tensor(p_soft, dtype=torch.float32, device=device)
            loss = loss + policy_consistency_weight * _soft_bce_with_logits(logits, p_t, None)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(rcfg.get("grad_clip", 10.0)))
        opt.step()

        if step % log_every == 0 or step == steps - 1:
            with torch.no_grad():
                prob = torch.sigmoid(logits)
                pos_prob = prob[:len(pos_src)].mean().item() if len(pos_src) else float("nan")
                hard_prob = prob[len(pos_src):len(pos_src) + len(hard_src)].mean().item() if len(hard_src) else float("nan")
                unl_prob = prob[len(pos_src) + len(hard_src):].mean().item() if len(unl_src) else float("nan")
            logger.log({
                "phase": "reachability",
                "step": step,
                "loss": float(loss.detach().cpu()),
                "pos_prob": pos_prob,
                "hard_neg_prob": hard_prob,
                "unlabeled_prob": unl_prob,
                "policy_soft_mean": float(p_soft.detach().mean().cpu()) if len(p_soft) else float("nan"),
                "batch_pos": int(len(pos_src)),
                "batch_hard_neg": int(len(hard_src)),
                "batch_unlabeled": int(len(unl_src)),
            })

    save_checkpoint(ckpt_path, model, optimizer=opt, latent_dim=embeddings.shape[1], reachability_type="policy_conditioned")
    logger.log({"phase": "reachability", "event": "saved", "path": ckpt_path, "type": "policy_conditioned"})
    return model
