from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FallbackTrigger:
    episode_id: int
    task_id: int
    trigger_state_hash: str
    reason: str
    best_goal_distance: float
    reached_subgoals: int


CONDITIONS = [
    "trigger_log_only_continue_graph",
    "trigger_switch_direct_goal",
    "direct_goal_from_start",
    "no_fallback",
]
