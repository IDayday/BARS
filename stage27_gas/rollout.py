from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Protocol

import numpy as np

from .config import AdaptiveConfig
from .diagnostics import compute_path_diagnostics
from .graph import GraphData
from .planner import AdaptiveWaypointSelector, nearest_graph_node_by_state, shortest_path


class LowLevelPolicy(Protocol):
    def act(self, obs: np.ndarray, subgoal: np.ndarray, deterministic: bool = True) -> np.ndarray:
        ...


@dataclass
class RolloutResult:
    env: str
    variant: str
    seed: int
    episode: int
    success: float
    ret: float
    length: int
    info: Dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict:
        row = {
            "env": self.env,
            "variant": self.variant,
            "seed": self.seed,
            "episode": self.episode,
            "success": self.success,
            "return": self.ret,
            "length": self.length,
        }
        row.update(self.info)
        return row


def _extract_obs(reset_out: Any) -> tuple[np.ndarray, dict]:
    if isinstance(reset_out, tuple) and len(reset_out) == 2:
        return np.asarray(reset_out[0], dtype=np.float32), dict(reset_out[1] or {})
    return np.asarray(reset_out, dtype=np.float32), {}


def _step_env(env: Any, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return np.asarray(obs, dtype=np.float32), float(reward), bool(terminated), bool(truncated), dict(info or {})
    if len(out) == 4:
        obs, reward, done, info = out
        return np.asarray(obs, dtype=np.float32), float(reward), bool(done), False, dict(info or {})
    raise ValueError("env.step must return Gymnasium 5-tuple or Gym 4-tuple")


def default_goal_from_env(env: Any, obs: np.ndarray, info: dict) -> np.ndarray:
    # Common goal-conditioned env APIs.
    if "goal" in info:
        return np.asarray(info["goal"], dtype=np.float32)
    if "desired_goal" in info:
        return np.asarray(info["desired_goal"], dtype=np.float32)
    if isinstance(obs, dict):
        return np.asarray(obs["desired_goal"], dtype=np.float32)
    if hasattr(env, "goal"):
        return np.asarray(env.goal, dtype=np.float32)
    if hasattr(env, "target_goal"):
        return np.asarray(env.target_goal, dtype=np.float32)
    raise ValueError("Cannot infer goal. Provide goal_fn(env, obs, info).")


def default_success_fn(env: Any, obs: np.ndarray, info: dict) -> bool:
    for key in ["success", "is_success", "goal_achieved"]:
        if key in info:
            return bool(info[key])
    if hasattr(env, "is_success"):
        return bool(env.is_success(obs))
    return False


def rollout_adaptive_gas_episode(
    env: Any,
    graph: GraphData,
    policy: LowLevelPolicy,
    variant: str,
    seed: int,
    episode: int,
    adaptive_cfg: AdaptiveConfig,
    max_steps: int = 1000,
    goal_fn: Optional[Callable[[Any, np.ndarray, dict], np.ndarray]] = None,
    success_fn: Optional[Callable[[Any, np.ndarray, dict], bool]] = None,
    deterministic: bool = True,
) -> RolloutResult:
    """Run one episode with AdaptiveGAS.

    This function is intentionally generic. In the actual GAS codebase, replace
    goal/state extraction and policy.act hooks with the existing interfaces.
    """
    try:
        reset_out = env.reset(seed=seed + episode)
    except TypeError:
        reset_out = env.reset()
    obs, reset_info = _extract_obs(reset_out)
    goal_fn = goal_fn or default_goal_from_env
    success_fn = success_fn or default_success_fn
    goal = goal_fn(env, obs, reset_info)

    start_node = nearest_graph_node_by_state(graph, obs)
    goal_node = nearest_graph_node_by_state(graph, goal)
    path, path_cost, eids = shortest_path(graph, start_node, goal_node)
    path_diag = compute_path_diagnostics(graph, path, eids)

    if not path:
        return RolloutResult(
            env=getattr(env, "spec", None).id if getattr(env, "spec", None) is not None else env.__class__.__name__,
            variant=variant,
            seed=seed,
            episode=episode,
            success=0.0,
            ret=0.0,
            length=0,
            info={**path_diag, "planned_path_cost": path_cost, "failure_reason": "no_path"},
        )

    selector = AdaptiveWaypointSelector(graph, adaptive_cfg)
    ret = 0.0
    subgoal_reached = []
    first_failed_edge_idx = np.nan
    final_info = dict(reset_info)
    for t in range(max_steps):
        sel = selector.select(obs, path)
        subgoal = graph.states[sel.subgoal_node]
        action = policy.act(obs, subgoal, deterministic=deterministic)
        next_obs, reward, terminated, truncated, info = _step_env(env, action)
        ret += reward
        final_info = info

        dist_to_subgoal = float(np.linalg.norm(next_obs[: min(len(next_obs), len(subgoal))] - subgoal[: min(len(next_obs), len(subgoal))]))
        reached = dist_to_subgoal <= adaptive_cfg.subgoal_tolerance
        subgoal_reached.append(float(reached))
        if not reached and not np.isfinite(first_failed_edge_idx):
            first_failed_edge_idx = float(sel.path_index)
        obs = next_obs
        if success_fn(env, obs, info):
            return RolloutResult(
                env=getattr(env, "spec", None).id if getattr(env, "spec", None) is not None else env.__class__.__name__,
                variant=variant,
                seed=seed,
                episode=episode,
                success=1.0,
                ret=ret,
                length=t + 1,
                info={
                    **path_diag,
                    "planned_path_cost": path_cost,
                    "subgoal_reach_rate": float(np.mean(subgoal_reached)) if subgoal_reached else np.nan,
                    "first_failed_edge_idx": first_failed_edge_idx,
                    "final_selector_index": selector.progress_index,
                },
            )
        if terminated or truncated:
            break

    return RolloutResult(
        env=getattr(env, "spec", None).id if getattr(env, "spec", None) is not None else env.__class__.__name__,
        variant=variant,
        seed=seed,
        episode=episode,
        success=0.0,
        ret=ret,
        length=len(subgoal_reached),
        info={
            **path_diag,
            "planned_path_cost": path_cost,
            "subgoal_reach_rate": float(np.mean(subgoal_reached)) if subgoal_reached else np.nan,
            "first_failed_edge_idx": first_failed_edge_idx,
            "final_selector_index": selector.progress_index,
            "failure_reason": final_info.get("failure_reason", "timeout_or_env_done"),
        },
    )
