from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ExecutionStep:
    edge_id: int
    src_cluster: int
    dst_cluster: int
    status: str
    steps: int = 0


class HierarchicalExecutor:
    """Interface scaffold for natural-start option execution.

    This class intentionally does not assume arbitrary reset-to-state support.
    A caller must provide an environment, cluster assignment function, and GCBC
    policy before online execution can be attempted.
    """

    def __init__(self, planner: Any, policy: Any, assign_cluster_fn: Any) -> None:
        self.planner = planner
        self.policy = policy
        self.assign_cluster_fn = assign_cluster_fn

    def run_episode(self, env: Any, goal: Any, max_steps: int) -> dict[str, Any]:
        del env, goal, max_steps
        return {
            "status": "not_implemented_scaffold",
            "success": False,
            "steps": 0,
            "trace": [],
            "failure_reason": "Phase 3F scaffold only; rollout is gated by env availability.",
        }
