from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from bars.common.logging import CSVLogger
from bars.common.stopper import Stopper
from bars.data.trajectories import OfflineDataset
from bars.graph.types import BARSGraph
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

    For AntMaze/PointMaze wrappers, reset-to-state availability varies by gym /
    D4RL version. This function is deliberately conservative: if we cannot infer
    a valid qpos/qvel split, edge rollout diagnostics are skipped rather than
    corrupting labels.
    """
    _reset_env(env)
    target = _unwrap_env(env)
    obs = np.asarray(obs, dtype=np.float64).reshape(-1)
    try:
        # Common MuJoCo API.
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
    threshold = float(rcfg.get('success_threshold', cfg.get('eval', {}).get('subgoal_threshold', 0.5)))
    selected_threshold = float(rcfg.get('selected_threshold', dcfg.get('edge_selected_threshold', 0.5)))
    rng = np.random.default_rng(int(cfg.get('seed', 0)) + 517)
    action_low = getattr(env.action_space, 'low', None)
    action_high = getattr(env.action_space, 'high', None)

    if graph.num_edges == 0:
        logger.log({'phase': 'edge_rollout_diag', 'event': 'completed', 'enabled': 1, 'num_edges_eval': 0})
        return

    selected = np.where(graph.p_exec >= selected_threshold)[0]
    other = np.where(graph.p_exec < selected_threshold)[0]
    n_sel = min(len(selected), episodes // 2)
    n_other = min(len(other), episodes - n_sel)
    edge_ids = []
    if n_sel > 0:
        edge_ids.extend(rng.choice(selected, size=n_sel, replace=False).tolist())
    if n_other > 0:
        edge_ids.extend(rng.choice(other, size=n_other, replace=False).tolist())
    if len(edge_ids) < episodes:
        fill = rng.choice(graph.num_edges, size=episodes - len(edge_ids), replace=graph.num_edges < episodes)
        edge_ids.extend([int(x) for x in fill])

    labels = []
    scores = []
    reset_ok_count = 0
    for k, eid in enumerate(edge_ids):
        if stopper is not None and stopper.stop_requested:
            break
        eid = int(eid)
        src_node = int(graph.src[eid])
        dst_node = int(graph.dst[eid])
        src_obs = dataset.observations[int(graph.node_indices[src_node])]
        dst_obs = dataset.observations[int(graph.node_indices[dst_node])]
        ok, reset_method = _try_reset_to_obs(env, src_obs)
        if not ok:
            logger.log({'phase': 'edge_rollout_diag', 'event': 'reset_unavailable', 'edge_id': eid, 'reset_method': reset_method})
            break
        reset_ok_count += 1
        obs = np.asarray(src_obs, dtype=np.float32).copy()
        success = False
        final_dist = float('nan')
        total_reward = 0.0
        for t in range(horizon):
            action = policy.act(obs, dst_obs, dataset.obs_normalizer, action_low, action_high, device=str(device))
            obs, rew, done, info = _step_env(env, action)
            obs = np.asarray(obs, dtype=np.float32)
            total_reward += float(rew)
            final_dist = float(np.linalg.norm(obs[:goal_dim] - dst_obs[:goal_dim]))
            if final_dist <= threshold or bool(info.get('success', False) or info.get('goal_achieved', False) or info.get('is_success', False)):
                success = True
                break
            if done:
                break
        labels.append(int(success))
        scores.append(float(graph.p_exec[eid]))
        logger.log({
            'phase': 'edge_rollout_diag',
            'event': 'edge',
            'edge_id': eid,
            'edge_rank': k,
            'src_node': src_node,
            'dst_node': dst_node,
            'p_exec': float(graph.p_exec[eid]),
            'risk': float(graph.risk[eid]),
            'cost': float(graph.cost[eid]),
            'success': int(success),
            'steps': t + 1,
            'final_dist': final_dist,
            'return': total_reward,
            'reset_method': reset_method,
        })

    labels_np = np.asarray(labels, dtype=np.int32)
    scores_np = np.asarray(scores, dtype=np.float32)
    auc, auprc = _safe_auc(labels_np, scores_np)
    logger.log({
        'phase': 'edge_rollout_diag',
        'event': 'completed',
        'enabled': 1,
        'reset_ok_count': reset_ok_count,
        'num_edges_eval': int(len(labels_np)),
        'success_rate': float(labels_np.mean()) if len(labels_np) else float('nan'),
        'p_exec_mean': float(scores_np.mean()) if len(scores_np) else float('nan'),
        'edge_rollout_auc': auc,
        'edge_rollout_auprc': auprc,
        'horizon': horizon,
        'success_threshold': threshold,
    })
