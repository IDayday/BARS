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
    """Construct AntMaze-compatible goal observation."""
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
def _embed_one(
    model: TemporalDistanceModel,
    obs: np.ndarray,
    dataset: OfflineDataset,
    device,
    *,
    encoder=None,
    source: str = "tdr",
) -> np.ndarray:
    """Embed one observation in the same space used by graph.node_embeddings.

    For exact GAS same-backbone mode, graph.node_embeddings may come from the
    official GAS/TDR encoder. In that case eval.embedding_source=policy lets the
    external policy wrapper provide the online embedding. Otherwise the local
    BARS TDR model or DatasetEmbeddingLookupModel is used.
    """
    source = str(source or "tdr").lower()
    prefer_policy = source in {"policy", "external_policy", "gas_policy"}
    auto_policy = source == "auto" and bool(getattr(model, "external_dataset_embeddings", False))
    if encoder is not None and (prefer_policy or auto_policy):
        method = getattr(encoder, "embed", None)
        if method is not None:
            try:
                z = np.asarray(method(obs), dtype=np.float32)
                return z[0] if z.ndim > 1 else z
            except Exception:
                if prefer_policy:
                    # Fall through to model fallback, but keep deterministic behavior.
                    pass
    if hasattr(model, "encode_obs"):
        return np.asarray(model.encode_obs(obs, dataset=dataset, device=device), dtype=np.float32)
    x = torch.as_tensor(dataset.obs_normalizer.encode(obs[None]), dtype=torch.float32, device=device)
    return model.encode(x).float().cpu().numpy()[0]


def _as_list(value, default: List[str]) -> List[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
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
    valid = {
        "none",
        "planner_only",
        "direct_goal",
        "direct_goal_after_k",
        "direct_goal_after_progress",
        "direct_goal_after_k_or_progress",
        "direct_goal_after_k_and_progress",
    }
    if mode not in valid:
        raise ValueError(f"Unknown eval.fallback_mode={mode!r}; expected one of {sorted(valid)}")
    return mode


def _planner_fallback_chain(current_variant: str, ecfg: Dict) -> List[str]:
    """Return graph-planner attempts in causal degradation order."""
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
        "constrained_bars": 3,
        "budget_bars": 3,
        "bars_budget": 3,
        "full_bars_budget": 3,
    }
    cur_rank = rank.get(cur, 99)
    out: List[str] = [cur]
    for item in configured:
        v = str(item).lower()
        if v in {cur, "direct_goal"}:
            continue
        if rank.get(v, 99) <= cur_rank and v not in out:
            out.append(v)
    return out


def _progress_triggered(
    ecfg: Dict,
    *,
    goal_distance_initial: float,
    goal_distance_best: float,
    graph_plan_success_count: int,
    num_subgoal_reached: int,
) -> bool:
    # Any of these criteria can authorize recovery. Defaults are conservative:
    # at least one reached graph subgoal, or a visible reduction in final-goal
    # distance, or an absolute distance gate if configured.
    min_subgoals = int(ecfg.get("direct_goal_min_subgoals_reached", 1))
    min_graph_plans = int(ecfg.get("direct_goal_min_graph_plan_successes", 0))
    min_improve = float(ecfg.get("direct_goal_min_distance_improvement", 1.0))
    max_dist = ecfg.get("direct_goal_progress_distance", None)
    if num_subgoal_reached >= max(0, min_subgoals) and min_subgoals > 0:
        return True
    if graph_plan_success_count >= max(1, min_graph_plans) and min_graph_plans > 0:
        return True
    if goal_distance_initial - goal_distance_best >= min_improve:
        return True
    if max_dist is not None and goal_distance_best <= float(max_dist):
        return True
    return False


def _direct_goal_allowed(mode: str, ecfg: Dict, planner_failure_streak: int, progress_ok: bool) -> bool:
    k = max(1, int(ecfg.get("direct_goal_after_k", 1)))
    k_ok = planner_failure_streak + 1 >= k
    if mode == "direct_goal":
        return True
    if mode == "direct_goal_after_k":
        return k_ok
    if mode == "direct_goal_after_progress":
        return progress_ok
    if mode == "direct_goal_after_k_or_progress":
        return k_ok or progress_ok
    if mode == "direct_goal_after_k_and_progress":
        return k_ok and progress_ok
    return False


def _try_plan_with_fallback(
    graph: BARSGraph,
    start_node: int,
    goal_node: int,
    variant: str,
    lambda_risk: float,
    lambda_boundary: float,
    boundary: Optional[BoundaryIndex],
    ecfg: Dict,
    *,
    exec_budget: Optional[float],
    max_edges: Optional[int],
    max_labels_per_edge: int,
) -> Tuple[Optional[PlanResult], str, bool, bool, int, List[int], bool]:
    """Try graph planner fallbacks and optional direct-goal recovery."""
    requested = str(variant).lower()
    mode = _resolve_fallback_mode(ecfg)
    allow_planner_fallback = mode in {
        "planner_only",
        "direct_goal",
        "direct_goal_after_k",
        "direct_goal_after_progress",
        "direct_goal_after_k_or_progress",
        "direct_goal_after_k_and_progress",
    }
    attempts = _planner_fallback_chain(requested, ecfg) if allow_planner_fallback else [requested]
    planner_failure_streak = int(ecfg.get("_planner_failure_streak", 0))
    progress_ok = bool(ecfg.get("_progress_triggered", False))
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
            exec_budget=exec_budget,
            max_edges=max_edges,
            max_labels_per_edge=max_labels_per_edge,
        )
        usable = bool(plan.found and len(plan.node_path) > 1)
        attempted_edges.append(len(plan.edge_path) if usable else -1)
        if idx == 0 and not usable:
            initial_failed = True
        if usable:
            return plan, cand, idx > 0, initial_failed, idx, attempted_edges, True
    if _direct_goal_allowed(mode, ecfg, planner_failure_streak, progress_ok):
        attempted_edges.append(0)
        return None, "direct_goal", requested != "direct_goal", initial_failed, len(attempts), attempted_edges, False
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
    condition = str(ecfg.get("condition", ecfg.get("protocol_condition", "default")))
    fallback_mode = _resolve_fallback_mode(ecfg)
    direct_goal_after_k = max(1, int(ecfg.get("direct_goal_after_k", 1)))
    lambda_r = float(cfg.get("planner", {}).get("lambda_risk", ecfg.get("lambda_risk", 1.0)))
    lambda_b = float(cfg.get("planner", {}).get("lambda_boundary", ecfg.get("lambda_boundary", 1.0)))
    exec_budget = ecfg.get("exec_budget", cfg.get("planner", {}).get("exec_budget", None))
    exec_budget = None if exec_budget is None else float(exec_budget)
    max_plan_edges_cfg = ecfg.get("max_plan_edges", cfg.get("planner", {}).get("max_edges", None))
    max_plan_edges = None if max_plan_edges_cfg is None else int(max_plan_edges_cfg)
    max_labels_per_edge = int(cfg.get("planner", {}).get("max_labels_per_edge", ecfg.get("max_labels_per_edge", 32)))
    direct_goal_variant = variant == "direct_goal"
    max_deferred_no_path_replans = int(ecfg.get("max_deferred_no_path_replans", max(2, direct_goal_after_k + 1)))
    embedding_source = str(ecfg.get("embedding_source", cfg.get("embedding", {}).get("online_source", "tdr"))).lower()

    rng = np.random.default_rng(int(cfg.get("seed", 0)) + 211)
    nbrs = None if direct_goal_variant else NearestNeighbors(n_neighbors=1).fit(graph.node_embeddings)
    action_low = getattr(env.action_space, "low", None)
    action_high = getattr(env.action_space, "high", None)

    for ep in range(episodes):
        if stopper is not None and stopper.stop_requested:
            break
        obs = np.asarray(_reset_env(env), dtype=np.float32)
        goal_obs = _goal_obs_from_env(env, obs, dataset, goal_dim, rng)
        goal_distance_initial = float(np.linalg.norm(obs[:goal_dim] - goal_obs[:goal_dim]))
        goal_distance_best = goal_distance_initial
        total_reward = 0.0
        success = False
        no_path_count = 0
        initial_plan_failed_count = 0
        fallback_used_any = False
        fallback_count_total = 0
        fallback_variants_used: List[str] = []
        replans = 0
        deferred_no_path_replans = 0
        steps = 0
        last_plan_edges = 0
        plan_edges_seen: List[int] = []
        attempted_edges_seen: List[str] = []
        first_plan_edges = -1
        num_plan_calls = 0
        subgoal_attempts = 0
        num_subgoal_reached = 0
        direct_goal_attempts = 0
        planner_failure_streak = 0
        graph_plan_success_count = 0
        graph_subgoal_reached_count = 0
        progress_gate_triggered_count = 0

        while steps < max_steps:
            cur_goal_dist = float(np.linalg.norm(obs[:goal_dim] - goal_obs[:goal_dim]))
            goal_distance_best = min(goal_distance_best, cur_goal_dist)
            if cur_goal_dist <= success_threshold:
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
                z_s = _embed_one(tdr_model, obs, dataset, device, encoder=policy, source=embedding_source)
                z_g = _embed_one(tdr_model, goal_obs, dataset, device, encoder=policy, source=embedding_source)
                s_node = int(nbrs.kneighbors(z_s[None], return_distance=False)[0, 0])
                g_node = int(nbrs.kneighbors(z_g[None], return_distance=False)[0, 0])
                progress_ok = _progress_triggered(
                    ecfg,
                    goal_distance_initial=goal_distance_initial,
                    goal_distance_best=goal_distance_best,
                    graph_plan_success_count=graph_plan_success_count,
                    num_subgoal_reached=graph_subgoal_reached_count,
                )
                if progress_ok:
                    progress_gate_triggered_count += 1
                plan, chosen_variant, fallback_used, initial_failed, fallback_count, attempted_edges, planner_succeeded = _try_plan_with_fallback(
                    graph,
                    s_node,
                    g_node,
                    variant,
                    lambda_r,
                    lambda_b,
                    boundary,
                    {**ecfg, "_planner_failure_streak": planner_failure_streak, "_progress_triggered": progress_ok},
                    exec_budget=exec_budget,
                    max_edges=max_plan_edges,
                    max_labels_per_edge=max_labels_per_edge,
                )
                num_plan_calls += 1
                attempted_edges_seen.append("/".join(str(x) for x in attempted_edges))
                if initial_failed:
                    initial_plan_failed_count += 1
                if fallback_used:
                    fallback_used_any = True
                    fallback_count_total += int(fallback_count)
                    fallback_variants_used.append(chosen_variant)

                if planner_succeeded and plan is not None and plan.found and len(plan.node_path) > 1:
                    planner_failure_streak = 0
                    graph_plan_success_count += 1
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
                    deferred_no_path_replans += 1
                    # Fix Stage19 direct_goal_after_k: do not terminate before the
                    # failure streak can reach k. This costs no env steps, only a
                    # bounded number of replans.
                    if fallback_mode in {"direct_goal_after_k", "direct_goal_after_k_or_progress", "direct_goal_after_k_and_progress"} and deferred_no_path_replans < max_deferred_no_path_replans:
                        continue
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
                cur_goal_dist = float(np.linalg.norm(obs[:goal_dim] - goal_obs[:goal_dim]))
                goal_distance_best = min(goal_distance_best, cur_goal_dist)
                info_success = bool(info.get("success", False) or info.get("goal_achieved", False) or info.get("is_success", False))
                if info_success or cur_goal_dist <= success_threshold:
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
                if chosen_variant != "direct_goal" and not direct_goal_variant:
                    graph_subgoal_reached_count += 1
            if success or steps >= max_steps:
                break

        plan_edges_arr = np.asarray(plan_edges_seen, dtype=np.float32) if plan_edges_seen else np.asarray([], dtype=np.float32)
        fallback_variant = ";".join(fallback_variants_used[:8]) if fallback_variants_used else ""
        final_goal_dist = float(np.linalg.norm(obs[:goal_dim] - goal_obs[:goal_dim]))
        logger.log(
            {
                "phase": "eval",
                "enabled": 1,
                "episode": ep,
                "condition": condition,
                "variant": variant,
                "success": int(success),
                "return": total_reward,
                "steps": steps,
                "replans": replans,
                "deferred_no_path_replans": deferred_no_path_replans,
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
                "progress_gate_triggered_count": progress_gate_triggered_count,
                "graph_plan_success_count": graph_plan_success_count,
                "graph_subgoal_reached_count": graph_subgoal_reached_count,
                "last_plan_edges": last_plan_edges,
                "first_plan_edges": int(first_plan_edges),
                "max_plan_edges": int(plan_edges_arr.max()) if len(plan_edges_arr) else 0,
                "mean_plan_edges": float(plan_edges_arr.mean()) if len(plan_edges_arr) else 0.0,
                "num_plan_calls": num_plan_calls,
                "num_subgoal_attempts": subgoal_attempts,
                "num_subgoal_reached": num_subgoal_reached,
                "subgoal_reach_rate": float(num_subgoal_reached / max(1, subgoal_attempts)),
                "goal_distance_initial": goal_distance_initial,
                "goal_distance_best": goal_distance_best,
                "goal_distance_final": final_goal_dist,
                "goal_distance_improvement": float(goal_distance_initial - goal_distance_best),
                "subgoal_horizon": subgoal_horizon,
                "subgoal_threshold": subgoal_threshold,
                "success_threshold": success_threshold,
                "lambda_risk": lambda_r,
                "lambda_boundary": lambda_b,
                "exec_budget": float(exec_budget) if exec_budget is not None else float("nan"),
                "max_plan_edges_cfg": int(max_plan_edges) if max_plan_edges is not None else 0,
                "embedding_source": embedding_source,
                "attempted_edges_summary": ";".join(attempted_edges_seen[:16]),
            }
        )
