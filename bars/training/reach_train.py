from __future__ import annotations

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
from bars.graph.ann import KNNIndex
from bars.models.reachability import ReachabilityModel
from bars.training.reach_policy_train import train_policy_conditioned_reachability


def _build_latent_hard_negative_pool(dataset: OfflineDataset, embeddings: np.ndarray, cfg: Dict, rng: np.random.Generator) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    rcfg = cfg.get('reachability', {})
    if not bool(rcfg.get('use_hard_neg', True)):
        return None
    n_pool = min(int(rcfg.get('hard_neg_pool', 20000)), dataset.size)
    k = int(rcfg.get('hard_neg_knn', 16))
    idx = rng.choice(dataset.size, size=n_pool, replace=False)
    ann = KNNIndex.from_config(embeddings[idx].astype(np.float32), cfg, prefix='ann')
    neigh = ann.kneighbors(embeddings[idx].astype(np.float32), min(k + 1, n_pool), return_distance=False)
    src, dst = [], []
    horizon = int(rcfg.get('horizon', 30))
    for row, ns in enumerate(neigh):
        i = int(idx[row])
        for n in ns[1:]:
            j = int(idx[int(n)])
            same = dataset.traj_id[i] == dataset.traj_id[j]
            dt = int(dataset.timestep[j] - dataset.timestep[i]) if same else 10**9
            # Prefer latent-near but unsupported pairs: cross-trajectory, or far
            # same-trajectory beyond the low-level horizon.
            if (not same) or dt <= 0 or dt > horizon:
                src.append(i)
                dst.append(j)
                break
    if not src:
        return None
    return np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)


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


def _bce(logits: torch.Tensor, target: float) -> torch.Tensor:
    labels = torch.full_like(logits, float(target))
    return F.binary_cross_entropy_with_logits(logits, labels, reduction='mean')


def _nnpu_loss(pos_logits: torch.Tensor, unl_logits: torch.Tensor, prior: float, beta: float = 0.0, gamma: float = 1.0) -> torch.Tensor:
    """Non-negative PU-style risk for reachability.

    Cross-trajectory pairs are not assumed negative.  The unlabeled term is only
    used through the negative risk correction, while hard negatives are handled
    separately with BCE(0).  This implements the core behavior we need without
    making cross-trajectory bridges disappear.
    """
    if len(pos_logits) == 0 or len(unl_logits) == 0:
        return torch.tensor(0.0, device=pos_logits.device if len(pos_logits) else unl_logits.device)
    pi = float(np.clip(prior, 1e-4, 0.999))
    loss_pos_pos = _bce(pos_logits, 1.0)
    loss_pos_neg = _bce(pos_logits, 0.0)
    loss_unl_neg = _bce(unl_logits, 0.0)
    neg_risk = loss_unl_neg - pi * loss_pos_neg
    if neg_risk.item() < -beta:
        # nnPU correction: avoid driving unlabeled examples to negatives when
        # the corrected negative risk is already too small.
        return pi * loss_pos_pos - gamma * neg_risk
    return pi * loss_pos_pos + torch.clamp(neg_risk, min=0.0)


def train_reachability(dataset: OfflineDataset, embeddings: np.ndarray, cfg: Dict, run_dir: str, device, logger: CSVLogger, stopper: Optional[Stopper] = None, policy=None) -> ReachabilityModel:
    rcfg = cfg.get('reachability', {})
    if str(rcfg.get('type', 'standard')).lower() in {'policy_conditioned', 'bars_policy', 'bars_v2'}:
        return train_policy_conditioned_reachability(dataset, embeddings, cfg, run_dir, device, logger, stopper=stopper, policy=policy)
    ckpt_path = os.path.join(run_dir, 'checkpoints', 'reachability.pt')
    model = ReachabilityModel(embeddings.shape[1], tuple(rcfg.get('hidden_dims', [256, 256]))).to(device)
    if rcfg.get('load_if_exists', True) and os.path.exists(ckpt_path):
        load_checkpoint(ckpt_path, model, map_location=str(device))
        logger.log({'phase': 'reachability', 'event': 'loaded', 'path': ckpt_path})
        return model

    opt = torch.optim.AdamW(model.parameters(), lr=float(rcfg.get('lr', 3e-4)), weight_decay=float(rcfg.get('weight_decay', 1e-5)))
    batch_size = int(rcfg.get('batch_size', 2048))
    steps = int(rcfg.get('steps', 20000))
    horizon = int(rcfg.get('horizon', 30))
    log_every = int(rcfg.get('log_every', 500))
    loss_mode = str(rcfg.get('loss_mode', 'pu')).lower()
    pos_frac = float(rcfg.get('pos_frac', 0.35 if loss_mode == 'pu' else 0.5))
    unlabeled_frac = float(rcfg.get('unlabeled_frac', 0.35 if loss_mode == 'pu' else 0.0))
    hard_weight = float(rcfg.get('hard_neg_weight', 0.5))
    same_neg_weight = float(rcfg.get('same_neg_weight', 1.0))
    cross_random_weight = float(rcfg.get('cross_random_weight', 0.25))
    pu_weight = float(rcfg.get('pu_weight', 1.0))
    pu_prior = float(rcfg.get('pu_prior', 0.15))
    pu_beta = float(rcfg.get('pu_beta', 0.0))
    pu_gamma = float(rcfg.get('pu_gamma', 1.0))
    far_min_dt = int(rcfg.get('far_neg_min_dt', horizon + 1))
    far_max_dt = int(rcfg.get('far_neg_max_dt', max(horizon * 4, far_min_dt + 1)))
    rng = np.random.default_rng(int(cfg.get('seed', 0)) + 43)
    hard_pool = _build_latent_hard_negative_pool(dataset, embeddings, cfg, rng)
    z_all = torch.as_tensor(embeddings, dtype=torch.float32, device=device)

    n_pos = max(1, int(batch_size * pos_frac))
    n_unl = max(0, int(batch_size * unlabeled_frac)) if loss_mode in {'pu', 'nnpu'} else 0
    n_base_neg = max(1, batch_size - n_pos - n_unl)
    logger.log({
        'phase': 'reachability',
        'event': 'train_start',
        'loss_mode': loss_mode,
        'batch_size': batch_size,
        'n_pos': n_pos,
        'n_unlabeled': n_unl,
        'n_base_neg': n_base_neg,
        'pos_frac': pos_frac,
        'unlabeled_frac': unlabeled_frac,
        'same_neg_weight': same_neg_weight,
        'cross_random_weight': cross_random_weight,
        'hard_neg_weight': hard_weight,
        'pu_weight': pu_weight,
        'pu_prior': pu_prior,
        'hard_pool_size': 0 if hard_pool is None else len(hard_pool[0]),
    })

    for step in trange(steps, desc='train_reachability', **tqdm_kwargs(cfg)):
        if stopper is not None and stopper.stop_requested:
            break
        ip, jp, _ = dataset.sample_future_pairs(n_pos, horizon, rng)
        pos_src, pos_dst = ip, jp

        n_far = n_base_neg // 2
        far_src, far_dst = _sample_far_same_traj_negatives(dataset, n_far, rng, min_dt=far_min_dt, max_dt=far_max_dt) if n_far > 0 else (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))

        n_cross_neg = max(0, n_base_neg - n_far)
        # In weighted-BCE mode these are weak negatives. In PU mode they are
        # added to unlabeled instead of hard negatives.
        cross_src, cross_dst = _sample_cross_random_pairs(dataset, n_cross_neg, rng) if n_cross_neg > 0 else (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))

        unl_src_parts, unl_dst_parts = [], []
        if n_unl > 0:
            us, ud = _sample_cross_random_pairs(dataset, n_unl, rng)
            unl_src_parts.append(us); unl_dst_parts.append(ud)
        if loss_mode in {'pu', 'nnpu'} and len(cross_src):
            unl_src_parts.append(cross_src); unl_dst_parts.append(cross_dst)
            cross_src = np.empty(0, dtype=np.int64); cross_dst = np.empty(0, dtype=np.int64)
        unl_src = np.concatenate(unl_src_parts) if unl_src_parts else np.empty(0, dtype=np.int64)
        unl_dst = np.concatenate(unl_dst_parts) if unl_dst_parts else np.empty(0, dtype=np.int64)

        hard_src_parts, hard_dst_parts = [far_src], [far_dst]
        hard_weights = [np.full(len(far_src), same_neg_weight, dtype=np.float32)]
        if len(cross_src):
            hard_src_parts.append(cross_src); hard_dst_parts.append(cross_dst)
            hard_weights.append(np.full(len(cross_src), cross_random_weight, dtype=np.float32))
        if hard_pool is not None and hard_weight > 0:
            hs, hd = hard_pool
            hn = min(n_base_neg, len(hs))
            hi = rng.choice(len(hs), size=hn, replace=len(hs) < hn)
            hard_src_parts.append(hs[hi]); hard_dst_parts.append(hd[hi])
            hard_weights.append(np.full(hn, hard_weight, dtype=np.float32))
        hard_src = np.concatenate(hard_src_parts) if hard_src_parts else np.empty(0, dtype=np.int64)
        hard_dst = np.concatenate(hard_dst_parts) if hard_dst_parts else np.empty(0, dtype=np.int64)
        hard_w = np.concatenate(hard_weights) if hard_weights else np.empty(0, dtype=np.float32)

        src_all = np.concatenate([pos_src, hard_src, unl_src])
        dst_all = np.concatenate([pos_dst, hard_dst, unl_dst])
        src_t = torch.as_tensor(src_all, dtype=torch.long, device=device)
        dst_t = torch.as_tensor(dst_all, dtype=torch.long, device=device)
        logits_all = model(z_all[src_t], z_all[dst_t])
        pos_logits = logits_all[:len(pos_src)]
        hard_logits = logits_all[len(pos_src):len(pos_src) + len(hard_src)]
        unl_logits = logits_all[len(pos_src) + len(hard_src):]

        if loss_mode in {'pu', 'nnpu'}:
            loss_pos = _bce(pos_logits, 1.0)
            if len(hard_logits):
                hw = torch.as_tensor(hard_w, dtype=torch.float32, device=device)
                loss_hard = (F.binary_cross_entropy_with_logits(hard_logits, torch.zeros_like(hard_logits), reduction='none') * hw).mean()
            else:
                loss_hard = torch.tensor(0.0, device=device)
            loss_pu = _nnpu_loss(pos_logits, unl_logits, prior=pu_prior, beta=pu_beta, gamma=pu_gamma) if len(unl_logits) else torch.tensor(0.0, device=device)
            loss = loss_pos + loss_hard + pu_weight * loss_pu
        else:
            labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(hard_logits), torch.zeros_like(unl_logits)])
            weights = torch.cat([
                torch.ones_like(pos_logits),
                torch.as_tensor(hard_w, dtype=torch.float32, device=device) if len(hard_logits) else torch.empty(0, device=device),
                torch.full_like(unl_logits, cross_random_weight) if len(unl_logits) else torch.empty(0, device=device),
            ])
            loss = (F.binary_cross_entropy_with_logits(logits_all, labels, reduction='none') * weights).mean()
            loss_pos = _bce(pos_logits, 1.0)
            loss_hard = loss - loss_pos
            loss_pu = torch.tensor(0.0, device=device)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(rcfg.get('grad_clip', 10.0)))
        opt.step()
        if step % log_every == 0 or step == steps - 1:
            with torch.no_grad():
                pos_prob = torch.sigmoid(pos_logits).mean().item() if len(pos_logits) else float('nan')
                hard_prob = torch.sigmoid(hard_logits).mean().item() if len(hard_logits) else float('nan')
                unl_prob = torch.sigmoid(unl_logits).mean().item() if len(unl_logits) else float('nan')
            logger.log({
                'phase': 'reachability',
                'step': step,
                'loss': float(loss.detach().cpu()),
                'loss_pos': float(loss_pos.detach().cpu()),
                'loss_hard': float(loss_hard.detach().cpu()),
                'loss_pu': float(loss_pu.detach().cpu()),
                'pos_prob': pos_prob,
                'hard_neg_prob': hard_prob,
                'unlabeled_prob': unl_prob,
                'hard_pool_size': 0 if hard_pool is None else len(hard_pool[0]),
                'batch_pos': int(len(pos_src)),
                'batch_hard_neg': int(len(hard_src)),
                'batch_unlabeled': int(len(unl_src)),
            })
    save_checkpoint(ckpt_path, model, optimizer=opt, latent_dim=embeddings.shape[1], loss_mode=loss_mode)
    logger.log({'phase': 'reachability', 'event': 'saved', 'path': ckpt_path, 'loss_mode': loss_mode})
    return model
