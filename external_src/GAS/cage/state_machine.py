from __future__ import annotations

from collections import deque
from enum import Enum
from statistics import mean
from typing import Any

import numpy as np

from cage.config import CAGEConfig
from cage.closed_loop_contracts import ContractGateResult, evaluate_contract_gate
from cage.contract_model import ContractScorer
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
    CHURN_GUARD = "CHURN_GUARD"
    FALLBACK_TO_GAS = "FALLBACK_TO_GAS"
    RECOVERY_LOCKOUT = "RECOVERY_LOCKOUT"


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
        self.contract_scorer = (
            ContractScorer.from_path(
                config.contract_model_path if config.use_contract_model else None,
                uncertainty_penalty=config.contract_uncertainty_penalty,
            )
            if config.contract_commit
            else None
        )
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
        self.churn_guard_trigger_count = 0
        self.fallback_to_gas_count = 0
        self.fallback_to_gas_step_count = 0
        self.replan_suppressed_by_cooldown_count = 0
        self.replan_suppressed_by_budget_count = 0
        self.recovery_suppressed_by_lockout_count = 0
        self.contract_gate_pass_count = 0
        self.contract_gate_reject_count = 0
        self.contract_fallback_to_gas_when_uncertain_count = 0
        self.contract_recovery_reject_count = 0
        self.contract_final_goal_reject_count = 0
        self.max_consecutive_replan_burst = 0
        self._consecutive_replan_request_count = 0
        self._last_replan_attempt_step = None
        self._last_allowed_replan_step = None
        self._replan_request_steps: deque[int] = deque()
        self._fallback_until_step = -1
        self._recovery_lockout_until_step = -1
        self._last_recovery_attempt_step = None
        self._recovery_disabled_after_churn = False
        self._last_original_subgoal = None
        self._last_original_subgoal_index = None
        self._last_step = 0
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
        self._last_step = int(step)
        self._last_original_subgoal = info.get("original_subgoal", current_subgoal)
        self._last_original_subgoal_index = info.get("original_subgoal_index")
        path = as_path_array(current_path)
        path_length = self._path_len(path)
        self.final_active_path_length = path_length
        dist_to_path = distance_to_path(current_state, path, self.distance_fn)
        if dist_to_path is not None:
            self.distance_to_path_values.append(float(dist_to_path))

        planner_final_goal = bool(info.get("planner_final_goal_on", False))
        final_goal_distance = float(self.distance_fn(current_state, final_goal))
        final_goal_phase = (
            not self.config.disable_final_phase_controller
            and (planner_final_goal or final_goal_distance < float(self.config.final_phase_dist))
        )
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
            "original_gas_target_mode": "final_goal" if planner_final_goal else "gas_path",
        }
        trace_info.update(self._contract_default_trace_fields())
        self._add_state_ref_summary(trace_info, info)

        if self.config.trace_only:
            self._attach_state_ref(trace_info, info, "trace_only")
            return self._select_original_subgoal(
                current_state=current_state,
                final_goal=final_goal,
                step=step,
                trace_info=trace_info,
                reason="trace_only_passthrough",
                trace_only=True,
            )

        if self._fallback_active(step):
            self._attach_state_ref(trace_info, info, "fallback_to_gas")
            return self._select_original_subgoal(
                current_state=current_state,
                final_goal=final_goal,
                step=step,
                trace_info=trace_info,
                reason="fallback_to_gas",
                fallback=True,
            )

        if path is None:
            self.no_path = True
            self.state = CAGEState.REPLAN_MISS
            return self._request_global_replan(
                final_goal=final_goal,
                reason="no_path",
                trace_info=trace_info,
                current_state=current_state,
            )

        hard_drift = (
            not self.config.disable_drift_monitor
            and dist_to_path is not None
            and dist_to_path > float(self.config.drift_threshold)
        )
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
            and (not self.monitor.is_stalled or self.state == CAGEState.RECOVERY or self.config.disable_recovery)
            and not final_goal_phase
            and self.state not in {CAGEState.LOCAL_STALL, CAGEState.FINAL_GOAL_STALL, CAGEState.REPLAN_MISS}
            and self.commitment_steps < min_commit
        ):
            if self.state != CAGEState.RECOVERY:
                self.state = CAGEState.COMMITTING
            selected = np.asarray(self.current_subgoal)
            trace_info.update(
                self._trace_target(
                    selected,
                    self.current_subgoal_index,
                    False,
                    current_state=current_state,
                    target_mode="committed_target",
                )
            )
            return selected, self.current_subgoal_index, self.state.value, False, trace_info

        if final_goal_phase:
            if self.monitor.is_stalled:
                self.state = CAGEState.FINAL_GOAL_STALL
            else:
                self.state = CAGEState.FINAL_GOAL
            selection = self.selector.select(current_state, final_goal, path, final_goal_phase=True)
            gate = self._contract_gate(
                current_state,
                selection.subgoal,
                target_mode="final_goal",
                path_position=selection.index,
                final_phase=True,
                info=info,
            )
            if gate is not None and not gate.passed:
                self.contract_final_goal_reject_count += 1
                trace_info.update(gate.trace_fields())
                self._attach_state_ref(trace_info, info, "contract_final_goal_reject")
                return self._contract_safe_fallback(
                    current_state=current_state,
                    final_goal=final_goal,
                    step=step,
                    trace_info=trace_info,
                    reason=f"final_goal_contract_rejected:{gate.reason}",
                    info=info,
                )
            if gate is not None:
                trace_info.update(gate.trace_fields())
            self._set_target(selection.subgoal, selection.index, step)
            trace_info.update(
                self._trace_target(
                    selection.subgoal,
                    selection.index,
                    False,
                    selection.reason,
                    distance=selection.distance,
                    target_mode="final_goal",
                )
            )
            return selection.subgoal, selection.index, self.state.value, False, trace_info

        if hard_drift:
            self.state = CAGEState.PATH_DRIFT
            selection = self._try_recovery(current_state, path, trace_info)
            if selection is not None:
                return selection
            if self.config.contract_commit:
                self._attach_state_ref(trace_info, info, "contract_drift_fallback")
                return self._contract_safe_fallback(
                    current_state=current_state,
                    final_goal=final_goal,
                    step=step,
                    trace_info=trace_info,
                    reason="contract_commit_drift_guard",
                    info=info,
                )
            return self._request_global_replan(
                final_goal=final_goal,
                reason="hard_drift_replan",
                trace_info=trace_info,
                current_state=current_state,
            )

        if self.monitor.is_stalled:
            if not self._stall_active:
                self.stall_count += 1
                self._stall_active = True
            if not self.config.disable_recovery:
                self.state = CAGEState.FINAL_GOAL_STALL if self._in_final_goal_phase else CAGEState.LOCAL_STALL
                selection = self._try_recovery(current_state, path, trace_info)
                if selection is not None:
                    return selection
                if self.config.contract_commit:
                    self._attach_state_ref(trace_info, info, "contract_stall_fallback")
                    return self._contract_safe_fallback(
                        current_state=current_state,
                        final_goal=final_goal,
                        step=step,
                        trace_info=trace_info,
                        reason="contract_commit_stall_guard",
                        info=info,
                    )
                return self._request_global_replan(
                    final_goal=final_goal,
                    reason="stall_replan",
                    trace_info=trace_info,
                    current_state=current_state,
                )
        else:
            self._stall_active = False

        recent_stalls = self.stall_count + self.recovery_failure_count
        selection = self.selector.select(current_state, final_goal, path, recent_stalls=recent_stalls)
        if selection is None:
            self.no_path = True
            self.state = CAGEState.REPLAN_MISS
            return self._request_global_replan(
                final_goal=final_goal,
                reason="no_selection",
                trace_info=trace_info,
                current_state=current_state,
            )

        self.state = CAGEState.NORMAL
        gate = self._contract_gate(
            current_state,
            selection.subgoal,
            target_mode="cage_selected",
            path_position=selection.index,
            final_phase=False,
            info=info,
        )
        if gate is not None and not gate.passed:
            trace_info.update(gate.trace_fields())
            self._attach_state_ref(trace_info, info, "contract_gate_reject")
            return self._contract_safe_fallback(
                current_state=current_state,
                final_goal=final_goal,
                step=step,
                trace_info=trace_info,
                reason=f"target_contract_rejected:{gate.reason}",
                info=info,
            )
        if gate is not None:
            trace_info.update(gate.trace_fields())
        self._set_target(selection.subgoal, selection.index, step)
        trace_info.update(
            self._trace_target(
                selection.subgoal,
                selection.index,
                False,
                selection.reason,
                distance=selection.distance,
                target_mode="cage_selected",
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
            if self.config.disable_recovery and not self._in_final_goal_phase:
                return
            if self._in_final_goal_phase:
                self.final_goal_stall_count += 1
                self.state = CAGEState.FINAL_GOAL_STALL
            elif self.state == CAGEState.RECOVERY:
                if self.commitment_steps >= int(self.config.recovery_commit_steps):
                    self.recovery_failure_count += 1
                    self._start_recovery_lockout()
                    self.state = CAGEState.REPLAN_MISS
            else:
                self.state = CAGEState.LOCAL_STALL
        elif (
            self.config.enable_churn_guard
            and self.state == CAGEState.RECOVERY
            and self.commitment_steps >= int(self.config.recovery_commit_steps)
            and snapshot.progress_window_value < float(self.config.min_progress_for_recovery_success)
        ):
            self.recovery_failure_count += 1
            self._start_recovery_lockout()
            self.state = CAGEState.RECOVERY_LOCKOUT

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
            "global_replan_request_rate_per_100_steps": self._global_replan_rate_per_100_steps(),
            "max_consecutive_replan_burst": int(self.max_consecutive_replan_burst),
            "churn_guard_trigger_count": int(self.churn_guard_trigger_count),
            "fallback_to_gas_count": int(self.fallback_to_gas_count),
            "fallback_to_gas_step_count": int(self.fallback_to_gas_step_count),
            "replan_suppressed_by_cooldown_count": int(self.replan_suppressed_by_cooldown_count),
            "replan_suppressed_by_budget_count": int(self.replan_suppressed_by_budget_count),
            "recovery_suppressed_by_lockout_count": int(self.recovery_suppressed_by_lockout_count),
            "churn_guard_active_on_timeout": bool(self.config.enable_churn_guard and self._fallback_active(self._last_step)),
            "cage_safe_mode_enabled": bool(self.config.enable_churn_guard),
            "cage_trace_only": bool(self.config.trace_only),
            "cage_contract_commit": bool(self.config.contract_commit),
            "contract_model_loaded": int(bool(self.contract_scorer.loaded) if self.contract_scorer is not None else False),
            "contract_gate_pass_count": int(self.contract_gate_pass_count),
            "contract_gate_reject_count": int(self.contract_gate_reject_count),
            "contract_gate_reject_rate": self._safe_rate(
                int(self.contract_gate_reject_count),
                int(self.contract_gate_pass_count) + int(self.contract_gate_reject_count),
            ),
            "contract_fallback_to_gas_when_uncertain_count": int(self.contract_fallback_to_gas_when_uncertain_count),
            "contract_recovery_reject_count": int(self.contract_recovery_reject_count),
            "contract_final_goal_reject_count": int(self.contract_final_goal_reject_count),
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
        step = int(trace_info.get("step", self._last_step))
        if self.config.enable_churn_guard:
            suppressed_reason = None
            if self._recovery_disabled_after_churn:
                suppressed_reason = "recovery_disabled_after_churn"
            elif self._recovery_lockout_active(step):
                suppressed_reason = "recovery_lockout"
            elif self._recovery_too_soon(step):
                suppressed_reason = "min_steps_between_recovery_attempts"
            if suppressed_reason is not None:
                self.recovery_suppressed_by_lockout_count += 1
                trace_info["recovery_suppressed_reason"] = suppressed_reason
                return self._select_original_subgoal(
                    current_state=current_state,
                    final_goal=self._last_original_subgoal if self._last_original_subgoal is not None else path[-1],
                    step=step,
                    trace_info=trace_info,
                    reason="recovery_suppressed",
                    state=CAGEState.RECOVERY_LOCKOUT,
                )

        if self.config.disable_recovery:
            self.state = CAGEState.REPLAN_MISS
            return None
        if self.recovery_attempt_count >= int(self.config.max_recovery_attempts):
            self.recovery_failure_count += 1
            self.state = CAGEState.REPLAN_MISS
            self._start_recovery_lockout()
            return None
        recovery = self.recovery.select(current_state, path, self.current_subgoal_index)
        if recovery is None:
            self.recovery_failure_count += 1
            self.state = CAGEState.REPLAN_MISS
            self._start_recovery_lockout()
            return None
        gate = self._contract_gate(
            current_state,
            recovery.target,
            target_mode="recovery_candidate",
            path_position=recovery.index,
            final_phase=False,
            recovery_candidate=True,
            info=None,
        )
        if gate is not None and not gate.passed:
            self.contract_recovery_reject_count += 1
            self.recovery_failure_count += 1
            self._start_recovery_lockout()
            trace_info.update(gate.trace_fields())
            trace_info["recovery_suppressed_reason"] = f"recovery_contract_rejected:{gate.reason}"
            return self._contract_safe_fallback(
                current_state=current_state,
                final_goal=self._last_original_subgoal if self._last_original_subgoal is not None else path[-1],
                step=step,
                trace_info=trace_info,
                reason=f"recovery_contract_rejected:{gate.reason}",
                info=None,
            )
        if gate is not None:
            trace_info.update(gate.trace_fields())
        self.recovery_attempt_count += 1
        self._last_recovery_attempt_step = step
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
                target_mode="recovery_candidate",
            )
        )
        return recovery.target, recovery.index, self.state.value, False, trace_info

    def _request_global_replan(self, final_goal, reason: str, trace_info: dict[str, Any], current_state):
        allowed, suppressed_reason = self._guard_replan_request(reason)
        if allowed:
            self.global_replan_request_count += 1
            selected = np.asarray(final_goal)
            trace_info.update(self._trace_target(selected, -1, True, reason, current_state=current_state, target_mode="final_goal"))
            return selected, -1, self.state.value, True, trace_info

        trace_info["replan_suppressed_reason"] = suppressed_reason
        return self._select_original_subgoal(
            current_state=current_state,
            final_goal=final_goal,
            step=self._last_step,
            trace_info=trace_info,
            reason=f"{reason}_suppressed",
            state=CAGEState.FALLBACK_TO_GAS if self._fallback_active(self._last_step) else CAGEState.CHURN_GUARD,
        )

    def _select_original_subgoal(
        self,
        current_state,
        final_goal,
        step,
        trace_info: dict[str, Any],
        reason: str,
        trace_only: bool = False,
        fallback: bool = False,
        state: CAGEState | None = None,
    ):
        selected = self._last_original_subgoal
        index = self._last_original_subgoal_index
        if selected is None:
            selected = self.current_subgoal if self.current_subgoal is not None else final_goal
            index = self.current_subgoal_index if self.current_subgoal is not None else -1
        selected = np.asarray(selected)
        if state is not None:
            self.state = state
        elif fallback:
            self.state = CAGEState.FALLBACK_TO_GAS
        else:
            self.state = CAGEState.NORMAL
        if fallback:
            self.fallback_to_gas_step_count += 1
        self._set_target(selected, index, step)
        trace_info.update(
            self._trace_target(
                selected,
                index,
                False,
                reason,
                current_state=current_state,
                target_mode="gas_path" if trace_only else "fallback_to_gas" if fallback else "original_target",
            )
        )
        trace_info["cage_trace_only"] = bool(trace_only)
        trace_info["original_subgoal_used"] = True
        return selected, index, self.state.value, False, trace_info

    def _contract_safe_fallback(
        self,
        current_state,
        final_goal,
        step,
        trace_info: dict[str, Any],
        reason: str,
        info: dict[str, Any] | None = None,
    ):
        """Keep the current target when possible; otherwise use the GAS target."""
        self.contract_fallback_to_gas_when_uncertain_count += 1
        trace_info["fallback_reason"] = reason
        trace_info["contract_uncertain_fallback"] = True
        if self.current_subgoal is not None and not self._target_reached(current_state):
            selected = np.asarray(self.current_subgoal)
            self.state = CAGEState.COMMITTING
            trace_info.update(
                self._trace_target(
                    selected,
                    self.current_subgoal_index,
                    False,
                    reason,
                    current_state=current_state,
                    target_mode="committed_target",
                )
            )
            return selected, self.current_subgoal_index, self.state.value, False, trace_info
        if info is not None:
            self._attach_state_ref(trace_info, info, "fallback_to_gas")
        return self._select_original_subgoal(
            current_state=current_state,
            final_goal=final_goal,
            step=step,
            trace_info=trace_info,
            reason=reason,
            fallback=bool(self.config.contract_fallback_to_gas_when_uncertain),
            state=CAGEState.FALLBACK_TO_GAS if self.config.contract_fallback_to_gas_when_uncertain else CAGEState.NORMAL,
        )

    def _contract_gate(
        self,
        current_state,
        target,
        *,
        target_mode: str,
        path_position: int | None = None,
        final_phase: bool = False,
        recovery_candidate: bool = False,
        info: dict[str, Any] | None = None,
    ) -> ContractGateResult | None:
        if not self.config.contract_commit or self.contract_scorer is None or target is None:
            return None
        features = self._contract_features(
            current_state,
            target,
            target_mode=target_mode,
            path_position=path_position,
            final_phase=final_phase,
            recovery_candidate=recovery_candidate,
            info=info,
        )
        prediction = self.contract_scorer.predict(features)
        gate = evaluate_contract_gate(
            prediction,
            lcb_threshold=float(self.config.contract_lcb_threshold),
            negative_progress_threshold=float(self.config.contract_negative_progress_threshold),
            target_mode=target_mode,
            final_goal_threshold=float(self.config.contract_final_goal_threshold),
            recovery_threshold=float(self.config.contract_recovery_threshold),
        )
        if gate.passed:
            self.contract_gate_pass_count += 1
        else:
            self.contract_gate_reject_count += 1
        return gate

    def _contract_features(
        self,
        current_state,
        target,
        *,
        target_mode: str,
        path_position: int | None,
        final_phase: bool,
        recovery_candidate: bool,
        info: dict[str, Any] | None,
    ) -> dict[str, Any]:
        current_state = np.asarray(current_state)
        target = np.asarray(target)
        previous_distance = None
        if self.current_subgoal is not None:
            previous_distance = float(self.distance_fn(current_state, self.current_subgoal))
        current_distance = float(self.distance_fn(current_state, target))
        info = info or {}
        return {
            "phi_s": current_state,
            "phi_g": target,
            "d_phi": current_distance,
            "target_mode": target_mode,
            "path_position": path_position if path_position is not None else -1,
            "final_phase": bool(final_phase),
            "recovery_candidate": bool(recovery_candidate),
            "recent_stall_count": int(self.stall_count),
            "recent_drift_count": int(self.drift_count),
            "commitment_steps": int(self.commitment_steps),
            "previous_target_distance": previous_distance if previous_distance is not None else -1.0,
            "current_target_distance": current_distance,
            "q_train_support": info.get("q_train_support", -1.0),
            "env_name": self.config.env_name,
            "fallback_tau": self.config.reachability_tau,
        }

    def _contract_default_trace_fields(self) -> dict[str, Any]:
        loaded = bool(self.contract_scorer.loaded) if self.contract_scorer is not None else False
        return {
            "contract_model_loaded": int(loaded),
            "contract_score": None,
            "contract_lcb": None,
            "contract_uncertainty": None,
            "predicted_hit": None,
            "predicted_negative_progress": None,
            "contract_gate_pass": None,
            "contract_reject_reason": None,
            "selected_target_mode": None,
            "fallback_reason": None,
        }

    def _add_state_ref_summary(self, trace_info: dict[str, Any], info: dict[str, Any]) -> None:
        state_ref = info.get("state_ref") if info else None
        if not isinstance(state_ref, dict):
            trace_info["state_ref_exact"] = None
            trace_info["state_ref_mode"] = None
            return
        trace_info["state_ref_exact"] = bool(state_ref.get("exact_reset", False))
        trace_info["state_ref_mode"] = state_ref.get("reset_mode")

    def _attach_state_ref(self, trace_info: dict[str, Any], info: dict[str, Any] | None, event: str) -> None:
        if not self.config.debug or not info:
            return
        state_ref = info.get("state_ref")
        if isinstance(state_ref, dict):
            trace_info[f"{event}_state_ref"] = state_ref

    def _guard_replan_request(self, reason: str) -> tuple[bool, str | None]:
        if not self.config.enable_churn_guard:
            return True, None

        step = int(self._last_step)
        if self._last_replan_attempt_step is None or step - int(self._last_replan_attempt_step) > 1:
            self._consecutive_replan_request_count = 0
        self._consecutive_replan_request_count += 1
        self._last_replan_attempt_step = step
        self.max_consecutive_replan_burst = max(
            int(self.max_consecutive_replan_burst),
            int(self._consecutive_replan_request_count),
        )
        self._replan_request_steps.append(step)
        while self._replan_request_steps and self._replan_request_steps[0] <= step - 100:
            self._replan_request_steps.popleft()

        if self._fallback_active(step):
            return False, "fallback_to_gas"

        if self._consecutive_replan_request_count > int(self.config.max_consecutive_replan_requests):
            self._enter_churn_guard(step, f"consecutive_replan_burst:{reason}")
            self.replan_suppressed_by_budget_count += 1
            return False, "consecutive_replan_burst"

        if self.global_replan_request_count >= int(self.config.max_global_replans_per_episode):
            self._enter_churn_guard(step, f"episode_replan_budget:{reason}")
            self.replan_suppressed_by_budget_count += 1
            return False, "episode_replan_budget"

        if len(self._replan_request_steps) > int(self.config.max_replans_per_100_steps):
            self._enter_churn_guard(step, f"replan_rate_budget:{reason}")
            self.replan_suppressed_by_budget_count += 1
            return False, "replan_rate_budget"

        if (
            self._last_allowed_replan_step is not None
            and step - int(self._last_allowed_replan_step) < int(self.config.replan_cooldown_steps)
        ):
            self.replan_suppressed_by_cooldown_count += 1
            return False, "replan_cooldown"

        self._last_allowed_replan_step = step
        return True, None

    def _enter_churn_guard(self, step: int, reason: str) -> None:
        self.churn_guard_trigger_count += 1
        self.state = CAGEState.CHURN_GUARD
        if self.config.disable_recovery_after_churn:
            self._recovery_disabled_after_churn = True
        if self.config.fallback_to_gas_on_churn:
            if not self._fallback_active(step):
                self.fallback_to_gas_count += 1
            self._fallback_until_step = max(
                int(self._fallback_until_step),
                int(step) + int(self.config.fallback_to_gas_steps),
            )
        if self.config.log_churn_events and self.logger is not None:
            self.logger.write(
                {
                    "record_type": "churn_event",
                    "step": int(step),
                    "reason": reason,
                    "consecutive_replan_request_count": int(self._consecutive_replan_request_count),
                    "replan_window_count": int(len(self._replan_request_steps)),
                    "global_replan_request_count": int(self.global_replan_request_count),
                }
            )

    def _fallback_active(self, step: int) -> bool:
        return bool(self.config.enable_churn_guard and int(step) < int(self._fallback_until_step))

    def _start_recovery_lockout(self) -> None:
        if not self.config.enable_churn_guard:
            return
        self._recovery_lockout_until_step = max(
            int(self._recovery_lockout_until_step),
            int(self._last_step) + int(self.config.recovery_lockout_steps_after_failure),
        )

    def _recovery_lockout_active(self, step: int) -> bool:
        return bool(self.config.enable_churn_guard and int(step) < int(self._recovery_lockout_until_step))

    def _recovery_too_soon(self, step: int) -> bool:
        return bool(
            self.config.enable_churn_guard
            and self._last_recovery_attempt_step is not None
            and int(step) - int(self._last_recovery_attempt_step) < int(self.config.min_steps_between_recovery_attempts)
        )

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
        if self.config.disable_commitment:
            return 0
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
        target_mode: str | None = None,
    ) -> dict[str, Any]:
        if distance is None and current_state is not None and target is not None:
            distance = float(self.distance_fn(current_state, target))
        target_arr = np.asarray(target).reshape(-1) if target is not None else None
        original_arr = np.asarray(self._last_original_subgoal).reshape(-1) if self._last_original_subgoal is not None else None
        row = {
            "cage_state": self.state.value,
            "selected_subgoal_idx": None if index is None else int(index),
            "selected_subgoal_distance": distance,
            "should_replan": bool(should_replan),
            "selection_reason": reason,
            "selected_target_mode": target_mode or "unknown",
            "selected_subgoal_phi": target_arr.tolist() if target_arr is not None else None,
            "original_gas_subgoal_phi": original_arr.tolist() if original_arr is not None else None,
        }
        if current_state is not None and self._last_original_subgoal is not None:
            row["original_gas_subgoal_distance"] = float(self.distance_fn(current_state, self._last_original_subgoal))
        row.update(self._guard_trace_fields())
        return row

    def _guard_trace_fields(self) -> dict[str, Any]:
        step = int(self._last_step)
        cooldown_remaining = 0
        if self._last_allowed_replan_step is not None:
            cooldown_remaining = max(
                0,
                int(self.config.replan_cooldown_steps) - (step - int(self._last_allowed_replan_step)),
            )
        recovery_lockout_remaining = max(0, int(self._recovery_lockout_until_step) - step)
        while self._replan_request_steps and self._replan_request_steps[0] <= step - 100:
            self._replan_request_steps.popleft()
        return {
            "churn_guard_active": bool(self.config.enable_churn_guard and self.state == CAGEState.CHURN_GUARD),
            "fallback_to_gas_active": bool(self._fallback_active(step)),
            "replan_cooldown_remaining": int(cooldown_remaining),
            "recovery_lockout_remaining": int(recovery_lockout_remaining),
            "consecutive_replan_request_count": int(self._consecutive_replan_request_count),
            "replan_window_count": int(len(self._replan_request_steps)),
            "cage_safe_mode_enabled": bool(self.config.enable_churn_guard),
            "cage_trace_only": bool(self.config.trace_only),
            "cage_contract_commit": bool(self.config.contract_commit),
        }

    def _global_replan_rate_per_100_steps(self) -> float:
        steps = max(1, int(self._last_step) + 1)
        return float(100.0 * float(self.global_replan_request_count) / float(steps))

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
