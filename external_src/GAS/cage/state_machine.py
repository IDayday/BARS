from __future__ import annotations

from enum import Enum
from statistics import mean
from typing import Any

import numpy as np

from cage.config import CAGEConfig
from cage.monitor import DistanceFn, ProgressMonitor, as_path_array, distance_to_path
from cage.recovery import RecoverySelector
from cage.subgoal_selector import ReachabilityFn, SubgoalSelector


class CAGEState(str, Enum):
    NORMAL = "NORMAL"
    COMMITTING = "COMMITTING"
    LOCAL_STALL = "LOCAL_STALL"
    PATH_DRIFT = "PATH_DRIFT"
    RECOVERY = "RECOVERY"
    REPLAN_MISS = "REPLAN_MISS"
    FINAL_GOAL = "FINAL_GOAL"
    FINAL_GOAL_STALL = "FINAL_GOAL_STALL"


class CAGEController:
    def __init__(
        self,
        config: CAGEConfig,
        distance_fn: DistanceFn,
        logger=None,
        reachability_fn: ReachabilityFn | None = None,
    ):
        self.config = config
        self.distance_fn = distance_fn
        self.logger = logger
        self.selector = SubgoalSelector(config, distance_fn, reachability_fn=reachability_fn)
        self.recovery = RecoverySelector(config, distance_fn)
        self.monitor = ProgressMonitor(distance_fn, config.stall_window, config.progress_eps)
        self.reset_episode(None, None, None)

    def reset_episode(self, start_state, final_goal, initial_path=None):
        self.state = CAGEState.NORMAL
        self.current_subgoal = None
        self.current_subgoal_index = None
        self.commitment_steps = 0
        self.same_target_duration = 0
        self.target_switch_count = 0
        self.early_switch_count = 0
        self.commitment_lengths: list[int] = []
        self.stall_count = 0
        self.drift_count = 0
        self.recovery_attempt_count = 0
        self.recovery_success_count = 0
        self.recovery_failure_count = 0
        self.global_replan_request_count = 0
        self.final_goal_on_step = None
        self.final_goal_switch_count = 0
        self.final_goal_stall_count = 0
        self.segment_progresses: list[float] = []
        self.distance_to_path_values: list[float] = []
        self.segment_count = 0
        self.segment_target_reach_count = 0
        self.target_reached_this_segment = False
        self.no_path = False
        self.initial_path_length = self._path_len(initial_path)
        self.final_active_path_length = self.initial_path_length
        self._stall_active = False
        self._drift_active = False
        self._in_final_goal_phase = False
        self._step_records: list[dict[str, Any]] = []
        self.monitor.reset()
        return self

    def select_subgoal(
        self,
        current_state,
        final_goal,
        current_path,
        current_subgoal,
        step,
        info=None,
    ):
        info = info or {}
        path = as_path_array(current_path)
        path_length = self._path_len(path)
        self.final_active_path_length = path_length
        dist_to_path = distance_to_path(current_state, path, self.distance_fn)
        if dist_to_path is not None:
            self.distance_to_path_values.append(float(dist_to_path))

        planner_final_goal = bool(info.get("planner_final_goal_on", False))
        final_goal_distance = float(self.distance_fn(current_state, final_goal))
        final_goal_phase = planner_final_goal or final_goal_distance < float(self.config.final_phase_dist)
        if final_goal_phase and self.final_goal_on_step is None:
            self.final_goal_on_step = int(step)
        if final_goal_phase:
            self._in_final_goal_phase = True

        trace_info: dict[str, Any] = {
            "step": int(step),
            "distance_to_path": dist_to_path,
            "progress_window_value": self.monitor.progress_window_value,
            "final_goal_phase": final_goal_phase,
            "recovery_target_idx": None,
        }

        if path is None:
            self.no_path = True
            self.state = CAGEState.REPLAN_MISS
            self.global_replan_request_count += 1
            selected = np.asarray(final_goal)
            trace_info.update(self._trace_target(selected, -1, True, current_state=current_state))
            return selected, -1, self.state.value, True, trace_info

        hard_drift = dist_to_path is not None and dist_to_path > float(self.config.drift_threshold)
        if hard_drift and not self._drift_active:
            self.drift_count += 1
            self._drift_active = True
        elif not hard_drift:
            self._drift_active = False

        target_reached = self._target_reached(current_state)
        min_commit = self._current_min_commit()
        if (
            self.current_subgoal is not None
            and not target_reached
            and not hard_drift
            and (not self.monitor.is_stalled or self.state == CAGEState.RECOVERY)
            and not final_goal_phase
            and self.state not in {CAGEState.LOCAL_STALL, CAGEState.FINAL_GOAL_STALL, CAGEState.REPLAN_MISS}
            and self.commitment_steps < min_commit
        ):
            if self.state != CAGEState.RECOVERY:
                self.state = CAGEState.COMMITTING
            selected = np.asarray(self.current_subgoal)
            trace_info.update(self._trace_target(selected, self.current_subgoal_index, False, current_state=current_state))
            return selected, self.current_subgoal_index, self.state.value, False, trace_info

        if final_goal_phase:
            if self.monitor.is_stalled:
                self.state = CAGEState.FINAL_GOAL_STALL
            else:
                self.state = CAGEState.FINAL_GOAL
            selection = self.selector.select(current_state, final_goal, path, final_goal_phase=True)
            self._set_target(selection.subgoal, selection.index, step)
            trace_info.update(
                self._trace_target(
                    selection.subgoal,
                    selection.index,
                    False,
                    selection.reason,
                    distance=selection.distance,
                )
            )
            return selection.subgoal, selection.index, self.state.value, False, trace_info

        if hard_drift:
            self.state = CAGEState.PATH_DRIFT
            selection = self._try_recovery(current_state, path, trace_info)
            if selection is not None:
                return selection
            selected = np.asarray(final_goal)
            trace_info.update(self._trace_target(selected, -1, True, "hard_drift_replan", current_state=current_state))
            return selected, -1, self.state.value, True, trace_info

        if self.monitor.is_stalled:
            if not self._stall_active:
                self.stall_count += 1
                self._stall_active = True
            self.state = CAGEState.FINAL_GOAL_STALL if self._in_final_goal_phase else CAGEState.LOCAL_STALL
            selection = self._try_recovery(current_state, path, trace_info)
            if selection is not None:
                return selection
            selected = np.asarray(final_goal)
            trace_info.update(self._trace_target(selected, -1, True, "stall_replan", current_state=current_state))
            return selected, -1, self.state.value, True, trace_info
        self._stall_active = False

        recent_stalls = self.stall_count + self.recovery_failure_count
        selection = self.selector.select(current_state, final_goal, path, recent_stalls=recent_stalls)
        if selection is None:
            self.no_path = True
            self.state = CAGEState.REPLAN_MISS
            self.global_replan_request_count += 1
            selected = np.asarray(final_goal)
            trace_info.update(self._trace_target(selected, -1, True, "no_selection", current_state=current_state))
            return selected, -1, self.state.value, True, trace_info

        self.state = CAGEState.NORMAL
        self._set_target(selection.subgoal, selection.index, step)
        trace_info.update(
            self._trace_target(
                selection.subgoal,
                selection.index,
                False,
                selection.reason,
                distance=selection.distance,
            )
        )
        return selection.subgoal, selection.index, self.state.value, False, trace_info

    def update_after_step(
        self,
        prev_state,
        current_state,
        selected_subgoal,
        action=None,
        env_info=None,
    ):
        self.commitment_steps += 1
        self.same_target_duration += 1
        snapshot = self.monitor.update(prev_state, current_state, selected_subgoal)
        self.segment_progresses.append(float(snapshot.step_progress))

        if self.distance_fn(current_state, selected_subgoal) <= self.config.effective_target_reach_dist:
            if not self.target_reached_this_segment:
                self.segment_target_reach_count += 1
                self.target_reached_this_segment = True
            if self.state == CAGEState.RECOVERY:
                self.recovery_success_count += 1
                self.state = CAGEState.NORMAL

        if snapshot.stalled:
            if self._in_final_goal_phase:
                self.final_goal_stall_count += 1
                self.state = CAGEState.FINAL_GOAL_STALL
            elif self.state == CAGEState.RECOVERY:
                if self.commitment_steps >= int(self.config.recovery_commit_steps):
                    self.recovery_failure_count += 1
                    self.state = CAGEState.REPLAN_MISS
            else:
                self.state = CAGEState.LOCAL_STALL

    def finish_episode(
        self,
        env_name: str,
        task_id: int | None,
        seed: int,
        episode_idx: int,
        env_info: dict[str, Any] | None = None,
        timeout: bool | None = None,
    ) -> dict[str, Any]:
        if self.same_target_duration:
            lengths = [*self.commitment_lengths, self.same_target_duration]
        else:
            lengths = list(self.commitment_lengths)
        env_info = env_info or {}
        success = self._get_metric(env_info, "episode.success")
        episode_return = self._get_metric(env_info, "episode.return")
        normalized = self._get_metric(env_info, "episode.normalized_return")
        summary = {
            "record_type": "episode",
            "env_name": env_name,
            "task_id": task_id,
            "seed": seed,
            "episode_idx": episode_idx,
            "success": success,
            "return": episode_return,
            "normalized_score": normalized,
            "no_path": bool(self.no_path),
            "timeout": bool(timeout) if timeout is not None else bool(env_info.get("TimeLimit.truncated", False)),
            "path_length": int(self.final_active_path_length),
            "initial_path_length": int(self.initial_path_length),
            "final_active_path_length": int(self.final_active_path_length),
            "target_switch_count": int(self.target_switch_count),
            "early_switch_count": int(self.early_switch_count),
            "mean_commitment_length": float(mean(lengths)) if lengths else 0.0,
            "stall_count": int(self.stall_count),
            "drift_count": int(self.drift_count),
            "recovery_attempt_count": int(self.recovery_attempt_count),
            "recovery_success_count": int(self.recovery_success_count),
            "recovery_failure_count": int(self.recovery_failure_count),
            "global_replan_request_count": int(self.global_replan_request_count),
            "final_goal_on_step": self.final_goal_on_step,
            "final_goal_switch_count": int(self.final_goal_switch_count),
            "final_goal_stall_count": int(self.final_goal_stall_count),
            "segment_target_reach_rate": self._safe_rate(self.segment_target_reach_count, self.segment_count),
            "mean_segment_progress": float(mean(self.segment_progresses)) if self.segment_progresses else 0.0,
            "mean_distance_to_path": float(mean(self.distance_to_path_values)) if self.distance_to_path_values else None,
        }
        if self.logger is not None:
            self.logger.write_episode(summary)
            for record in self._step_records:
                self.logger.write_step(record)
        return summary

    def record_step_trace(self, env_name: str, task_id: int | None, seed: int, episode_idx: int, trace_info: dict[str, Any]) -> None:
        if not self.config.debug:
            return
        row = {
            "record_type": "step",
            "env_name": env_name,
            "task_id": task_id,
            "seed": seed,
            "episode_idx": episode_idx,
            **trace_info,
        }
        self._step_records.append(row)

    def _try_recovery(self, current_state, path, trace_info):
        if self.recovery_attempt_count >= int(self.config.max_recovery_attempts):
            self.recovery_failure_count += 1
            self.state = CAGEState.REPLAN_MISS
            self.global_replan_request_count += 1
            return None
        recovery = self.recovery.select(current_state, path, self.current_subgoal_index)
        if recovery is None:
            self.recovery_failure_count += 1
            self.state = CAGEState.REPLAN_MISS
            self.global_replan_request_count += 1
            return None
        self.recovery_attempt_count += 1
        self.state = CAGEState.RECOVERY
        self._set_target(recovery.target, recovery.index, trace_info.get("step", 0))
        trace_info["recovery_target_idx"] = recovery.index
        trace_info.update(
            self._trace_target(
                recovery.target,
                recovery.index,
                False,
                "local_recovery",
                distance=recovery.distance,
            )
        )
        return recovery.target, recovery.index, self.state.value, False, trace_info

    def _set_target(self, target, index, step) -> None:
        target = np.asarray(target)
        changed = self.current_subgoal is None or self.distance_fn(self.current_subgoal, target) > 1e-8
        if changed:
            if self.current_subgoal is not None:
                self.commitment_lengths.append(int(self.same_target_duration))
                self.target_switch_count += 1
                if self._in_final_goal_phase:
                    self.final_goal_switch_count += 1
                if self.commitment_steps < self._current_min_commit():
                    self.early_switch_count += 1
            self.current_subgoal = target
            self.current_subgoal_index = int(index) if index is not None else None
            self.commitment_steps = 0
            self.same_target_duration = 0
            self.target_reached_this_segment = False
            self.segment_count += 1
            self.monitor.reset()
        else:
            self.current_subgoal_index = int(index) if index is not None else self.current_subgoal_index

    def _target_reached(self, current_state) -> bool:
        if self.current_subgoal is None:
            return False
        return self.distance_fn(current_state, self.current_subgoal) <= self.config.effective_target_reach_dist

    def _current_min_commit(self) -> int:
        if self.state in {CAGEState.RECOVERY, CAGEState.PATH_DRIFT, CAGEState.LOCAL_STALL}:
            return int(self.config.recovery_commit_steps)
        if self._in_final_goal_phase:
            return int(self.config.final_min_commit_steps)
        return int(self.config.min_commit_steps)

    def _trace_target(
        self,
        target,
        index,
        should_replan,
        reason: str | None = None,
        current_state=None,
        distance: float | None = None,
    ) -> dict[str, Any]:
        if distance is None and current_state is not None and target is not None:
            distance = float(self.distance_fn(current_state, target))
        return {
            "cage_state": self.state.value,
            "selected_subgoal_idx": None if index is None else int(index),
            "selected_subgoal_distance": distance,
            "should_replan": bool(should_replan),
            "selection_reason": reason,
        }

    @staticmethod
    def _path_len(path) -> int:
        path_arr = as_path_array(path)
        return int(len(path_arr)) if path_arr is not None else 0

    @staticmethod
    def _safe_rate(num: int, den: int) -> float:
        return float(num / den) if den else 0.0

    @staticmethod
    def _get_metric(info: dict[str, Any], key: str):
        if key in info:
            value = info[key]
            try:
                return float(value)
            except Exception:
                return value
        return None
