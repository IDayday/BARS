from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from bars.common.logging import CSVLogger
from bars.common.stopper import Stopper
from bars.data.trajectories import OfflineDataset
from bars.graph.boundary import BoundaryIndex
from bars.graph.planner import PlanResult, plan_path
from bars.graph.types import BARSGraph
from bars.models.policy import GoalConditionedPolicy
from bars.models.tdr import TemporalDistanceModel


def _reset_env(env):
    out = env.reset()
    return out[0] if isinstance(out, tuple) else out


def _step_env(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, rew, terminated, truncated, info = out
        return obs, rew, bool(terminated or truncated), info
    return out


def _goal_obs_from_env(env, obs: np.ndarray, dataset: OfflineDataset, goal_dim: int, rng: np.random.Generator) -> np.ndarray:
    """Construct AntMaze-compatible goal observation.

    D4RL AntMaze exposes the target xy in different attribute names depending on
    the wrapper version. Following common AntMaze practice, only the leading xy
    coordinates are overwritten and the remaining proprioceptive coordinates are
    copied from the current observation. If no target is discoverable, fall back
    to a dataset observation rather than crashing; the eval logs still expose the
    final goal distance for diagnosis.
    """
    goal_obs = np.array(obs, copy=True)
    target = None
    for attr in ["target_goal", "goal", "_target_goal"]:
        if hasattr(env, attr):
            try:
                target = np.asarray(getattr(env, attr), dtype=np.float32).reshape(-1)
                break
            except Exception:
                pass
    if target is not None and len(target) >= goal_dim:
        goal_obs[:goal_dim] = target[:goal_dim]
    else:
        goal_obs = np.array(dataset.observations[int(rng.integers(0, dataset.size))], copy=True)
    return goal_obs.astype(np.float32)


@torch.no_grad()
def _embed_one(model: TemporalDistanceModel, obs: np.ndarray, dataset: OfflineDataset, device) -> np.ndarray:
    x = torch.as_tensor(dataset.obs_normalizer.encode(obs[None]), dtype=torch.float32, device=device)
    return model.encode(x).float().cpu().numpy()[0]


def _as_list(value, default: List[str]) -> List[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        # Accept either JSON-like comma-separated strings or a single variant.
        if "," in value:
            return [x.strip() for x in value.split(",") if x.strip()]
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    return list(default)


def _resolve_fallback_mode(ecfg: Dict) -> str:
    mode = ecfg.get("fallback_mode")
    if mode is None:
        if "fallback_enabled" in ecfg:
            return "direct_goal" if bool(ecfg.get("fallback_enabled", False)) else "none"
        if bool(ecfg.get("direct_goal_on_no_path", False)):
            return "direct_goal"
        return "none"
    mode = str(mode).lower()
    valid = {"none", "planner_only", "direct_goal", "direct_goal_after_k"}
    if mode not in valid:
        raise ValueError(f"Unknown eval.fallback_mode={mode!r}; expected one of {sorted(valid)}")
    return mode


def _planner_fallback_chain(current_variant: str, ecfg: Dict) -> List[str]:
    """Return graph-planner attempts in degradation order.

    Fallback never upgrades a weaker baseline to a stronger planner. Direct-goal
    is handled separately by eval.fallback_mode so that planner-only ablations
    remain causal and easy to interpret.
    """
    cur = str(current_variant).lower()
    default = ["reachability", "shortest"]
    configured = _as_list(ecfg.get("fallback_variants", default), default)
    rank = {
        "shortest": 0,
        "gas": 0,
        "tdr_shortest": 0,
        "reachability": 1,
        "bars_lite": 1,
        "risk": 1,
        "full_bars": 2,
        "bars": 2,
        "boundary": 2,
    }
    cur_rank = rank.get(cur, 2)
    out: List[str] = [cur]
    for item in configured:
        v = str(item).lower()
        if v in {cur, "direct_goal"}:
            continue
        if rank.get(v, 99) <= cur_rank and v not in out:
            out.append(v)
    return out


def _try_plan_with_fallback(
    graph: BARSGraph,
    start_node: int,
    goal_node: int,
    variant: str,
    lambda_risk: float,
    lambda_boundary: float,
    boundary: Optional[BoundaryIndex],
    ecfg: Dict,
) -> Tuple[Optional[PlanResult], str, bool, bool, int, List[int], bool]:
    """Try planner fallbacks and optional direct-goal fallback.

    Returns:
      plan: chosen graph plan or None for direct_goal.
      chosen_variant: selected variant or direct_goal.
      fallback_used: whether chosen variant differs from original.
      initial_failed: whether the original planner produced no usable subgoal.
      fallback_count: number of fallback attempts needed before success.
      attempted_edges: usable graph edge counts for attempts; -1 for failed,
        0 for direct_goal.
      planner_succeeded: whether a graph planner produced the chosen subgoal.
    """
    requested = str(variant).lower()
    mode = _resolve_fallback_mode(ecfg)
    allow_planner_fallback = mode in {"planner_only", "direct_goal", "direct_goal_after_k"}
    attempts = _planner_fallback_chain(requested, ecfg) if allow_planner_fallback else [requested]
    direct_goal_after_k = max(1, int(ecfg.get("direct_goal_after_k", 1)))
    planner_failure_streak = int(ecfg.get("_planner_failure_streak", 0))
    attempted_edges: List[int] = []
    initial_failed = False
    for idx, cand in enumerate(attempts):
        plan = plan_path(
            graph,
            int(start_node),
            int(goal_node),
            variant=cand,
            lambda_risk=lambda_risk,
            lambda_boundary=lambda_boundary,
            boundary=boundary,
        )
        usable = bool(plan.found and len(plan.node_path) > 1)
        attempted_edges.append(len(plan.edge_path) if usable else -1)
        if idx == 0 and not usable:
            initial_failed = True
        if usable:
            return plan, cand, idx > 0, initial_failed, idx, attempted_edges, True
    if mode == "direct_goal":
        attempted_edges.append(0)
        return None, "direct_goal", requested != "direct_goal", initial_failed, len(attempts), attempted_edges, False
    if mode == "direct_goal_after_k" and planner_failure_streak + 1 >= direct_goal_after_k:
        attempted_edges.append(0)
        return None, "direct_goal", requested != "direct_goal", len(attempts) > 0, len(attempts), attempted_edges, False
    return None, "none", False, initial_failed, max(0, len(attempts) - 1), attempted_edges, False


def evaluate_planner_policy(
    env,
    dataset: OfflineDataset,
    tdr_model: TemporalDistanceModel,
    policy: GoalConditionedPolicy,
    graph: BARSGraph,
    boundary: Optional[BoundaryIndex],
    cfg: Dict,
    device,
    logger: CSVLogger,
    stopper: Optional[Stopper] = None,
) -> None:
    ecfg = cfg.get("eval", {})
    if not bool(ecfg.get("enabled", False)):
        logger.log({"phase": "eval", "enabled": 0})
        return

    episodes = int(ecfg.get("episodes", 20))
    max_steps = int(ecfg.get("max_steps", 1000))
    subgoal_horizon = int(ecfg.get("subgoal_horizon", cfg.get("policy", {}).get("horizon", 30)))
    success_threshold = float(ecfg.get("success_threshold", 0.5))
    subgoal_threshold = float(ecfg.get("subgoal_threshold", success_threshold))
    goal_dim = int(ecfg.get("goal_dim", 2))
    variant = str(ecfg.get("variant", cfg.get("planner", {}).get("variant", "full_bars"))).lower()
    fallback_mode = _resolve_fallback_mode(ecfg)
    direct_goal_after_k = max(1, int(ecfg.get("direct_goal_after_k", 1)))
    lambda_r = float(cfg.get("planner", {}).get("lambda_risk", ecfg.get("lambda_risk", 1.0)))
    lambda_b = float(cfg.get("planner", {}).get("lambda_boundary", ecfg.get("lambda_boundary", 1.0)))
    direct_goal_variant = variant == "direct_goal"

    rng = np.random.default_rng(int(cfg.get("seed", 0)) + 211)
    nbrs = None if direct_goal_variant else NearestNeighbors(n_neighbors=1).fit(graph.node_embeddings)
    action_low = getattr(env.action_space, "low", None)
    action_high = getattr(env.action_space, "high", None)

    for ep in range(episodes):
        if stopper is not None and stopper.stop_requested:
            break
        obs = np.asarray(_reset_env(env), dtype=np.float32)
        goal_obs = _goal_obs_from_env(env, obs, dataset, goal_dim, rng)
        total_reward = 0.0
        success = False
        no_path_count = 0
        initial_plan_failed_count = 0
        fallback_used_any = False
        fallback_count_total = 0
        fallback_variants_used: List[str] = []
        replans = 0
        steps = 0
        last_plan_edges = 0
        plan_edges_seen: List[int] = []
        first_plan_edges = -1
        num_plan_calls = 0
        subgoal_attempts = 0
        num_subgoal_reached = 0
        direct_goal_attempts = 0
        planner_failure_streak = 0

        while steps < max_steps:
            if np.linalg.norm(obs[:goal_dim] - goal_obs[:goal_dim]) <= success_threshold:
                success = True
                break

            replans += 1
            chosen_variant = variant
            plan = None
            fallback_used = False
            initial_failed = False
            fallback_count = 0
            planner_succeeded = False
            plan_edges = 0

            if direct_goal_variant:
                subgoal_obs = goal_obs
                direct_goal_attempts += 1
            else:
                z_s = _embed_one(tdr_model, obs, dataset, device)
                z_g = _embed_one(tdr_model, goal_obs, dataset, device)
                s_node = int(nbrs.kneighbors(z_s[None], return_distance=False)[0, 0])
                g_node = int(nbrs.kneighbors(z_g[None], return_distance=False)[0, 0])
                plan, chosen_variant, fallback_used, initial_failed, fallback_count, _attempted_edges, planner_succeeded = _try_plan_with_fallback(
                    graph,
                    s_node,
                    g_node,
                    variant,
                    lambda_r,
                    lambda_b,
                    boundary,
                    {**ecfg, "_planner_failure_streak": planner_failure_streak},
                )
                num_plan_calls += 1
                if initial_failed:
                    initial_plan_failed_count += 1
                if fallback_used:
                    fallback_used_any = True
                    fallback_count_total += int(fallback_count)
                    fallback_variants_used.append(chosen_variant)

                if planner_succeeded and plan is not None and plan.found and len(plan.node_path) > 1:
                    planner_failure_streak = 0
                    plan_edges = len(plan.edge_path)
                    next_node = plan.node_path[1]
                    subgoal_obs = dataset.observations[graph.node_indices[next_node]]
                    if next_node == g_node:
                        subgoal_obs = goal_obs
                elif chosen_variant == "direct_goal":
                    planner_failure_streak += 1
                    subgoal_obs = goal_obs
                    direct_goal_attempts += 1
                    plan_edges = 0
                else:
                    planner_failure_streak += 1
                    no_path_count += 1
                    break

            last_plan_edges = int(plan_edges)
            plan_edges_seen.append(int(plan_edges))
            if first_plan_edges < 0:
                first_plan_edges = int(plan_edges)

            reached_subgoal = False
            subgoal_attempts += 1
            for _ in range(subgoal_horizon):
                if stopper is not None and stopper.stop_requested:
                    break
                action = policy.act(obs, subgoal_obs, dataset.obs_normalizer, action_low, action_high, device=str(device))
                obs, rew, done, info = _step_env(env, action)
                obs = np.asarray(obs, dtype=np.float32)
                total_reward += float(rew)
                steps += 1
                info_success = bool(info.get("success", False) or info.get("goal_achieved", False) or info.get("is_success", False))
                if info_success or np.linalg.norm(obs[:goal_dim] - goal_obs[:goal_dim]) <= success_threshold:
                    success = True
                    reached_subgoal = True
                    break
                if np.linalg.norm(obs[:goal_dim] - subgoal_obs[:goal_dim]) <= subgoal_threshold:
                    reached_subgoal = True
                    break
                if done or steps >= max_steps:
                    break
            if reached_subgoal:
                num_subgoal_reached += 1
            if success or steps >= max_steps:
                break

        plan_edges_arr = np.asarray(plan_edges_seen, dtype=np.float32) if plan_edges_seen else np.asarray([], dtype=np.float32)
        fallback_variant = ";".join(fallback_variants_used[:8]) if fallback_variants_used else ""
        logger.log(
            {
                "phase": "eval",
                "enabled": 1,
                "episode": ep,
                "variant": variant,
                "success": int(success),
                "return": total_reward,
                "steps": steps,
                "replans": replans,
                "no_path_count": no_path_count,
                "initial_plan_failed_count": initial_plan_failed_count,
                "plan_failed_initial": int(initial_plan_failed_count > 0),
                "fallback_enabled": int(fallback_mode != "none"),
                "fallback_mode": fallback_mode,
                "fallback_used": int(fallback_used_any),
                "fallback_variant": fallback_variant,
                "fallback_count": fallback_count_total,
                "direct_goal_attempts": direct_goal_attempts,
                "direct_goal_after_k": direct_goal_after_k,
                "last_plan_edges": last_plan_edges,
                "first_plan_edges": int(first_plan_edges),
                "max_plan_edges": int(plan_edges_arr.max()) if len(plan_edges_arr) else 0,
                "mean_plan_edges": float(plan_edges_arr.mean()) if len(plan_edges_arr) else 0.0,
                "num_plan_calls": num_plan_calls,
                "num_subgoal_attempts": subgoal_attempts,
                "num_subgoal_reached": num_subgoal_reached,
                "subgoal_reach_rate": float(num_subgoal_reached / max(1, subgoal_attempts)),
                "goal_distance_final": float(np.linalg.norm(obs[:goal_dim] - goal_obs[:goal_dim])),
                "subgoal_horizon": subgoal_horizon,
                "subgoal_threshold": subgoal_threshold,
                "success_threshold": success_threshold,
                "lambda_risk": lambda_r,
                "lambda_boundary": lambda_b,
            }
        )
