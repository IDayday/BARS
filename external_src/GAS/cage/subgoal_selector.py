from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from cage.config import CAGEConfig
from cage.monitor import DistanceFn, as_path_array


ReachabilityFn = Callable[[object, object], float]


@dataclass(frozen=True)
class SubgoalSelection:
    subgoal: np.ndarray
    index: int
    distance: float
    reliability: float
    reason: str


class SubgoalSelector:
    """Distance-based CAGE-MVP waypoint selector.

    `reachability_fn` is intentionally optional. The MVP uses a distance-based
    heuristic so a learned certificate can be plugged in later without changing
    evaluator hooks.
    """

    def __init__(
        self,
        config: CAGEConfig,
        distance_fn: DistanceFn,
        reachability_fn: ReachabilityFn | None = None,
    ):
        self.config = config
        self.distance_fn = distance_fn
        self.reachability_fn = reachability_fn

    def reliability(self, current_state, target) -> float:
        if self.reachability_fn is not None:
            return float(self.reachability_fn(current_state, target))
        distance = float(self.distance_fn(current_state, target))
        tau = max(1e-6, float(self.config.reachability_tau))
        return float(math.exp(-distance / tau))

    def select(
        self,
        current_state,
        final_goal,
        current_path,
        recent_stalls: int = 0,
        final_goal_phase: bool = False,
    ) -> SubgoalSelection | None:
        if final_goal_phase:
            return SubgoalSelection(
                subgoal=np.asarray(final_goal),
                index=-1,
                distance=float(self.distance_fn(current_state, final_goal)),
                reliability=self.reliability(current_state, final_goal),
                reason="final_goal_phase",
            )

        path = as_path_array(current_path)
        if path is None:
            return None

        path_distances = np.asarray([self.distance_fn(current_state, node) for node in path], dtype=float)
        support_idx = int(np.argmin(path_distances))
        max_dist = float(self.config.max_subgoal_dist)
        if recent_stalls > 0:
            max_dist = max(float(self.config.min_subgoal_dist), max_dist * 0.5)
        min_dist = float(self.config.min_subgoal_dist)

        candidates: list[SubgoalSelection] = []
        for idx in range(support_idx, len(path)):
            node = path[idx]
            distance = float(path_distances[idx])
            near_final = idx == len(path) - 1
            if distance > max_dist:
                continue
            if distance < min_dist and not near_final:
                continue
            candidates.append(
                SubgoalSelection(
                    subgoal=np.asarray(node),
                    index=idx,
                    distance=distance,
                    reliability=self.reliability(current_state, node),
                    reason="distance_horizon",
                )
            )

        if not candidates:
            idx = support_idx
            node = path[idx]
            return SubgoalSelection(
                subgoal=np.asarray(node),
                index=idx,
                distance=float(path_distances[idx]),
                reliability=self.reliability(current_state, node),
                reason="nearest_path_support",
            )

        return max(candidates, key=lambda item: (item.index, item.reliability))
