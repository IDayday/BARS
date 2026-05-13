from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from bars.common.logging import CSVLogger
from bars.common.stopper import Stopper
from bars.data.trajectories import OfflineDataset
from bars.graph.types import BARSGraph
from bars.graph.support import sample_edge_support_counts
from bars.models.policy import GoalConditionedPolicy


def _unwrap_env(env):
    cur = env
    seen = set()
    for _ in range(10):
        if id(cur) in seen:
            break
        seen.add(id(cur))
        if hasattr(cur, 'unwrapped'):
            cur = cur.unwrapped
        elif hasattr(cur, 'env'):
            cur = cur.env
        else:
            break
    return cur


def _reset_env(env):
    out = env.reset()
    return out[0] if isinstance(out, tuple) else out


def _step_env(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, rew, terminated, truncated, info = out
        return obs, rew, bool(terminated or truncated), info
    return out


def _try_reset_to_obs(env, obs: np.ndarray) -> Tuple[bool, str]:
    """Best-effort reset-to-state for MuJoCo-style D4RL envs.

    AntMaze wrappers differ across D4RL/Gym versions.  We only emit rollout
    labels when a reset-to-state method is available; otherwise diagnostics log
    reset_unavailable rather than fabricating labels.
    """
    _reset_env(env)
    target = _unwrap_env(env)
    obs = np.asarray(obs, dtype=np.float64).reshape(-1)
    try:
        sim = getattr(target, 'sim', None)
        model = getattr(sim, 'model', None) if sim is not None else getattr(target, 'model', None)
        nq = int(getattr(model, 'nq')) if model is not None and hasattr(model, 'nq') else None
        nv = int(getattr(model, 'nv')) if model is not None and hasattr(model, 'nv') else None
        if nq is not None and nv is not None and len(obs) >= nq + nv and hasattr(target, 'set_state'):
            qpos = obs[:nq].copy()
            qvel = obs[nq:nq + nv].copy()
            target.set_state(qpos, qvel)
            if sim is not None and hasattr(sim, 'forward'):
                sim.forward()
            return True, 'mujoco_set_state'
    except Exception as exc:
        return False, f'mujoco_set_state_failed:{type(exc).__name__}'

    try:
        if hasattr(target, 'set_xy') and len(obs) >= 2:
            target.set_xy(obs[:2].copy())
            return True, 'set_xy'
    except Exception as exc:
        return False, f'set_xy_failed:{type(exc).__name__}'

    return False, 'unsupported_reset_to_obs'


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return float('nan'), float('nan')
    return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))


def _group_masks(dataset: OfflineDataset, graph: BARSGraph, supported: np.ndarray, selected: np.ndarray) -> Dict[str, np.ndarray]:
    gi_src = graph.node_indices[graph.src]
    gi_dst = graph.node_indices[graph.dst]
    cross = dataset.traj_id[gi_src] != dataset.traj_id[gi_dst]
    hard_neg_proxy = (~supported) & cross
    unlabeled_bridge = supported & cross
    same_supported = supported & (~cross)
    return {
        'selected_supported': selected & supported,
        'selected_unlabeled_bridge': selected & unlabeled_bridge,
        'selected_hard_neg_proxy': selected & hard_neg_proxy,
        'unselected_supported': (~selected) & supported,
        'unselected_unlabeled_bridge': (~selected) & unlabeled_bridge,
        'unselected_hard_neg_proxy': (~selected) & hard_neg_proxy,
        'selected_same_traj_supported': selected & same_supported,
        'all_selected': selected,
        'all_unselected': ~selected,
    }


def _edge_group_name(eid: int, masks: Dict[str, np.ndarray], preferred: list[str]) -> str:
    for name in preferred:
        mask = masks.get(name)
        if mask is not None and eid < len(mask) and bool(mask[eid]):
            return name
    return 'other'


def _choose_edges_by_group(graph: BARSGraph, masks: Dict[str, np.ndarray], rng: np.random.Generator, rcfg: Dict, total: int) -> list[int]:
    group_names = rcfg.get('groups', [
        'selected_supported',
        'selected_unlabeled_bridge',
        'selected_hard_neg_proxy',
        'unselected_supported',
        'unselected_hard_neg_proxy',
    ])
    if isinstance(group_names, str):
        group_names = [x.strip() for x in group_names.split(',') if x.strip()]
    group_names = [str(x) for x in group_names if str(x) in masks]
    active = [(g, np.where(masks[g])[0]) for g in group_names if np.where(masks[g])[0].size > 0]
    if not active:
        return rng.choice(graph.num_edges, size=total, replace=graph.num_edges < total).astype(int).tolist()

    per_group = int(rcfg.get('per_group', max(1, int(np.ceil(total / max(1, len(active)))))))
    chosen: list[int] = []
    for _, ids in active:
        n = min(per_group, len(ids), max(0, total - len(chosen)))
        if n > 0:
            chosen.extend(rng.choice(ids, size=n, replace=False).astype(int).tolist())
        if len(chosen) >= total:
            break
    if len(chosen) < total:
        already = set(chosen)
        pool = np.asarray([i for i in range(graph.num_edges) if i not in already], dtype=np.int64)
        if len(pool) == 0:
            pool = np.arange(graph.num_edges, dtype=np.int64)
        fill = rng.choice(pool, size=total - len(chosen), replace=len(pool) < (total - len(chosen)))
        chosen.extend(fill.astype(int).tolist())
    return chosen[:total]


@torch.no_grad()
def run_edge_rollout_diagnostics(
    env,
    dataset: OfflineDataset,
    policy: GoalConditionedPolicy,
    graph: BARSGraph,
    cfg: Dict,
    device,
    logger: CSVLogger,
    stopper: Optional[Stopper] = None,
    embeddings: Optional[np.ndarray] = None,
) -> None:
    dcfg = cfg.get('diagnostics', {})
    rcfg = dcfg.get('edge_rollout', {}) if isinstance(dcfg.get('edge_rollout', {}), dict) else {}
    enabled = bool(dcfg.get('edge_rollout_enabled', rcfg.get('enabled', False)))
    logger.log({'phase': 'edge_rollout_diag', 'event': 'start', 'enabled': int(enabled)})
    if not enabled or env is None:
        logger.log({'phase': 'edge_rollout_diag', 'event': 'completed', 'enabled': int(enabled), 'available': int(env is not None)})
        return

    episodes = int(rcfg.get('num_edges', dcfg.get('edge_rollout_num_edges', 64)))
    horizon = int(rcfg.get('horizon', cfg.get('policy', {}).get('horizon', 30)))
    goal_dim = int(rcfg.get('goal_dim', cfg.get('eval', {}).get('goal_dim', 2)))
    threshold = float(rcfg.get('success_threshold', rcfg.get('goal_tolerance', cfg.get('eval', {}).get('subgoal_threshold', 0.5))))
    selected_threshold = float(rcfg.get('selected_threshold', dcfg.get('edge_selected_threshold', 0.5)))
    stratified = bool(rcfg.get('stratified', True))
    rng = np.random.default_rng(int(cfg.get('seed', 0)) + 517)
    action_low = getattr(env.action_space, 'low', None)
    action_high = getattr(env.action_space, 'high', None)

    if graph.num_edges == 0:
        logger.log({'phase': 'edge_rollout_diag', 'event': 'completed', 'enabled': 1, 'num_edges_eval': 0})
        return

    selected = graph.p_exec >= selected_threshold
    support_counts = np.zeros(graph.num_edges, dtype=np.int32)
    support_available = False
    if stratified and embeddings is not None:
        support = sample_edge_support_counts(
            dataset,
            embeddings,
            graph,
            horizon=int(rcfg.get('support_horizon', dcfg.get('edge_label_horizon', cfg.get('reachability', {}).get('horizon', 30)))),
            num_segments=int(rcfg.get('support_segments', dcfg.get('edge_support_segments', 100000))),
            min_dt=int(rcfg.get('support_min_dt', dcfg.get('edge_support_min_dt', 1))),
            rng=rng,
            batch_size=int(rcfg.get('support_batch_size', dcfg.get('edge_support_batch_size', 65536))),
            logger=logger,
            phase='edge_rollout_support',
            cfg=cfg,
        )
        support_counts = support.counts
        support_available = True
        logger.log({
            'phase': 'edge_rollout_diag',
            'event': 'support_completed',
            'support_sampled_segments': support.sampled_segments,
            'support_edge_hits': support.edge_hits,
            'support_hit_rate': support.hit_rate,
            'supported_edge_rate': support.edge_support_rate,
            'num_supported_edges': int((support_counts > 0).sum()),
        })

    supported = support_counts > 0
    masks = _group_masks(dataset, graph, supported, selected) if support_available else {
        'all_selected': selected,
        'all_unselected': ~selected,
    }
    preferred_groups = rcfg.get('groups', [
        'selected_supported',
        'selected_unlabeled_bridge',
        'selected_hard_neg_proxy',
        'unselected_supported',
        'unselected_hard_neg_proxy',
        'all_selected',
        'all_unselected',
    ])
    if isinstance(preferred_groups, str):
        preferred_groups = [x.strip() for x in preferred_groups.split(',') if x.strip()]
    edge_ids = _choose_edges_by_group(graph, masks, rng, rcfg, episodes)

    labels = []
    scores = []
    selected_flags = []
    groups = []
    final_dists = []
    reset_ok_count = 0
    reset_unavailable_count = 0
    for k, eid in enumerate(edge_ids):
        if stopper is not None and stopper.stop_requested:
            break
        eid = int(eid)
        src_node = int(graph.src[eid])
        dst_node = int(graph.dst[eid])
        src_obs = dataset.observations[int(graph.node_indices[src_node])]
        dst_obs = dataset.observations[int(graph.node_indices[dst_node])]
        edge_group = _edge_group_name(eid, masks, preferred_groups)
        ok, reset_method = _try_reset_to_obs(env, src_obs)
        if not ok:
            reset_unavailable_count += 1
            logger.log({'phase': 'edge_rollout_diag', 'event': 'reset_unavailable', 'edge_id': eid, 'edge_group': edge_group, 'reset_method': reset_method, 'reset_unavailable_count': reset_unavailable_count})
            break
        reset_ok_count += 1
        obs = np.asarray(src_obs, dtype=np.float32).copy()
        success = False
        final_dist = float('nan')
        total_reward = 0.0
        steps_taken = 0
        for t in range(horizon):
            action = policy.act(obs, dst_obs, dataset.obs_normalizer, action_low, action_high, device=str(device))
            obs, rew, done, info = _step_env(env, action)
            obs = np.asarray(obs, dtype=np.float32)
            total_reward += float(rew)
            steps_taken = t + 1
            final_dist = float(np.linalg.norm(obs[:goal_dim] - dst_obs[:goal_dim]))
            if final_dist <= threshold or bool(info.get('success', False) or info.get('goal_achieved', False) or info.get('is_success', False)):
                success = True
                break
            if done:
                break
        labels.append(int(success))
        scores.append(float(graph.p_exec[eid]))
        selected_flags.append(int(selected[eid]))
        groups.append(edge_group)
        final_dists.append(final_dist)
        gi_src = int(graph.node_indices[src_node])
        gi_dst = int(graph.node_indices[dst_node])
        is_cross = int(dataset.traj_id[gi_src] != dataset.traj_id[gi_dst])
        is_supported = int(support_counts[eid] > 0) if support_available else -1
        logger.log({
            'phase': 'edge_rollout_diag',
            'event': 'edge',
            'edge_id': eid,
            'edge_rank': k,
            'edge_group': edge_group,
            'src_node': src_node,
            'dst_node': dst_node,
            'p_exec': float(graph.p_exec[eid]),
            'selected_edge': int(selected[eid]),
            'support_count': int(support_counts[eid]) if support_available else -1,
            'is_supported_edge': is_supported,
            'is_cross_traj_edge': is_cross,
            'risk': float(graph.risk[eid]),
            'cost': float(graph.cost[eid]),
            'success': int(success),
            'steps': steps_taken,
            'final_dist': final_dist,
            'return': total_reward,
            'reset_method': reset_method,
        })

    labels_np = np.asarray(labels, dtype=np.int32)
    scores_np = np.asarray(scores, dtype=np.float32)
    selected_np = np.asarray(selected_flags, dtype=bool)
    groups_np = np.asarray(groups, dtype=object)
    final_np = np.asarray(final_dists, dtype=np.float32)
    auc, auprc = _safe_auc(labels_np, scores_np)
    selected_success_rate = float(labels_np[selected_np].mean()) if selected_np.any() else float('nan')
    unselected_success_rate = float(labels_np[~selected_np].mean()) if (~selected_np).any() else float('nan')

    summary = {
        'phase': 'edge_rollout_diag',
        'event': 'completed',
        'enabled': 1,
        'support_available': int(support_available),
        'reset_ok_count': reset_ok_count,
        'reset_unavailable_count': reset_unavailable_count,
        'reset_available': int(reset_ok_count > 0),
        'num_edges_eval': int(len(labels_np)),
        'num_selected_edges_eval': int(selected_np.sum()),
        'num_unselected_edges_eval': int((~selected_np).sum()),
        'success_rate': float(labels_np.mean()) if len(labels_np) else float('nan'),
        'selected_edge_success_rate': selected_success_rate,
        'unselected_edge_success_rate': unselected_success_rate,
        'p_exec_mean': float(scores_np.mean()) if len(scores_np) else float('nan'),
        'selected_p_exec_mean': float(scores_np[selected_np].mean()) if selected_np.any() else float('nan'),
        'unselected_p_exec_mean': float(scores_np[~selected_np].mean()) if (~selected_np).any() else float('nan'),
        'edge_rollout_auc': auc,
        'edge_rollout_auprc': auprc,
        'horizon': horizon,
        'success_threshold': threshold,
        'selected_threshold': selected_threshold,
    }
    for group_name in sorted(set(groups)):
        mask = groups_np == group_name
        key = ''.join(ch if ch.isalnum() else '_' for ch in str(group_name)).strip('_')
        summary[f'n_{key}'] = int(mask.sum())
        summary[f'success_rate_{key}'] = float(labels_np[mask].mean()) if mask.any() else float('nan')
        summary[f'p_exec_mean_{key}'] = float(scores_np[mask].mean()) if mask.any() else float('nan')
        summary[f'final_dist_mean_{key}'] = float(np.nanmean(final_np[mask])) if mask.any() else float('nan')
    logger.log(summary)
