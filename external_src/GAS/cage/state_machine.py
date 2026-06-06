from __future__ import annotations

from collections import deque
from enum import Enum
from statistics import mean
from typing import Any

import numpy as np

from cage.config import CAGEConfig
from cage.closed_loop_contracts import ContractGateResult, evaluate_contract_gate
from cage.contract_intervention import decide_contract_intervention, decision_trace
from cage.contract_model import ContractScorer
from cage.contract_ranker import ContractCandidate, candidate_trace, rank_contract_candidates
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
            or config.contract_rank
            or config.contract_shadow_rank
            or config.contract_intervene
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
        self.contract_rank_coverages: list[float] = []
        self.contract_rank_selected_scores: list[float] = []
        self.contract_rank_gas_scores: list[float] = []
        self.contract_rank_choose_gas_count = 0
        self.contract_rank_choose_cage_count = 0
        self.contract_rank_choose_committed_count = 0
        self.contract_rank_extreme_reject_count = 0
        self.shadow_total_count = 0
        self.shadow_override_count = 0
        self.shadow_final_phase_count = 0
        self.shadow_override_final_phase_count = 0
        self.shadow_override_margins: list[float] = []
        self.shadow_choose_committed_count = 0
        self.shadow_choose_cage_count = 0
        self.shadow_choose_path_later_count = 0
        self.contract_intervention_total_count = 0
        self.contract_intervention_count = 0
        self.contract_intervention_gains: list[float] = []
        self.final_phase_preserved_count = 0
        self.final_phase_override_count = 0
        self.final_phase_override_reasons: list[str] = []
        self.committed_target_progresses: list[float] = []
        self.committed_goal_progresses: list[float] = []
        self.committed_stale_count = 0
        self.committed_lockout_count = 0
        self.committed_lockout_step_count = 0
        self.committed_lockout_until_step = -1
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
        self._last_final_goal = final_goal
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
        self._last_final_goal = final_goal
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
            and not (self.config.contract_shadow_rank or self.config.contract_intervene)
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
            if self.config.contract_shadow_rank:
                return self._select_contract_shadow_rank_subgoal(
                    current_state=current_state,
                    final_goal=final_goal,
                    path=path,
                    selection=selection,
                    step=step,
                    trace_info=trace_info,
                    info=info,
                    final_goal_phase=True,
                    reason="shadow_rank_final_phase",
                )
            if self.config.contract_intervene:
                return self._select_contract_intervention_subgoal(
                    current_state=current_state,
                    final_goal=final_goal,
                    path=path,
                    selection=selection,
                    step=step,
                    trace_info=trace_info,
                    info=info,
                    final_goal_phase=True,
                    reason="contract_intervene_final_phase",
                )
            if self.config.contract_rank:
                return self._select_contract_ranked_subgoal(
                    current_state=current_state,
                    final_goal=final_goal,
                    path=path,
                    selection=selection,
                    step=step,
                    trace_info=trace_info,
                    info=info,
                    final_goal_phase=True,
                    reason="contract_rank_final_phase",
                )
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
            if self.config.contract_shadow_rank:
                selection = self.selector.select(
                    current_state,
                    final_goal,
                    path,
                    recent_stalls=self.stall_count + self.recovery_failure_count,
                )
                return self._select_contract_shadow_rank_subgoal(
                    current_state=current_state,
                    final_goal=final_goal,
                    path=path,
                    selection=selection,
                    step=step,
                    trace_info=trace_info,
                    info=info,
                    final_goal_phase=False,
                    reason="shadow_rank_drift_guard",
                )
            if self.config.contract_intervene:
                selection = self.selector.select(
                    current_state,
                    final_goal,
                    path,
                    recent_stalls=self.stall_count + self.recovery_failure_count,
                )
                return self._select_contract_intervention_subgoal(
                    current_state=current_state,
                    final_goal=final_goal,
                    path=path,
                    selection=selection,
                    step=step,
                    trace_info=trace_info,
                    info=info,
                    final_goal_phase=False,
                    reason="contract_intervene_drift_guard",
                )
            if self.config.contract_rank:
                selection = self.selector.select(
                    current_state,
                    final_goal,
                    path,
                    recent_stalls=self.stall_count + self.recovery_failure_count,
                )
                return self._select_contract_ranked_subgoal(
                    current_state=current_state,
                    final_goal=final_goal,
                    path=path,
                    selection=selection,
                    step=step,
                    trace_info=trace_info,
                    info=info,
                    final_goal_phase=False,
                    reason="contract_rank_drift_guard",
                )
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
                if self.config.contract_shadow_rank:
                    selection = self.selector.select(
                        current_state,
                        final_goal,
                        path,
                        recent_stalls=self.stall_count + self.recovery_failure_count,
                    )
                    return self._select_contract_shadow_rank_subgoal(
                        current_state=current_state,
                        final_goal=final_goal,
                        path=path,
                        selection=selection,
                        step=step,
                        trace_info=trace_info,
                        info=info,
                        final_goal_phase=False,
                        reason="shadow_rank_stall_guard",
                    )
                if self.config.contract_intervene:
                    selection = self.selector.select(
                        current_state,
                        final_goal,
                        path,
                        recent_stalls=self.stall_count + self.recovery_failure_count,
                    )
                    return self._select_contract_intervention_subgoal(
                        current_state=current_state,
                        final_goal=final_goal,
                        path=path,
                        selection=selection,
                        step=step,
                        trace_info=trace_info,
                        info=info,
                        final_goal_phase=False,
                        reason="contract_intervene_stall_guard",
                    )
                if self.config.contract_rank:
                    selection = self.selector.select(
                        current_state,
                        final_goal,
                        path,
                        recent_stalls=self.stall_count + self.recovery_failure_count,
                    )
                    return self._select_contract_ranked_subgoal(
                        current_state=current_state,
                        final_goal=final_goal,
                        path=path,
                        selection=selection,
                        step=step,
                        trace_info=trace_info,
                        info=info,
                        final_goal_phase=False,
                        reason="contract_rank_stall_guard",
                    )
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
        if self.config.contract_shadow_rank:
            return self._select_contract_shadow_rank_subgoal(
                current_state=current_state,
                final_goal=final_goal,
                path=path,
                selection=selection,
                step=step,
                trace_info=trace_info,
                info=info,
                final_goal_phase=False,
                reason="shadow_rank_path",
            )
        if self.config.contract_intervene:
            return self._select_contract_intervention_subgoal(
                current_state=current_state,
                final_goal=final_goal,
                path=path,
                selection=selection,
                step=step,
                trace_info=trace_info,
                info=info,
                final_goal_phase=False,
                reason="contract_intervene_path",
            )
        if self.config.contract_rank:
            return self._select_contract_ranked_subgoal(
                current_state=current_state,
                final_goal=final_goal,
                path=path,
                selection=selection,
                step=step,
                trace_info=trace_info,
                info=info,
                final_goal_phase=False,
                reason="contract_rank_path",
            )
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
        self._update_committed_watchdog(prev_state, current_state, selected_subgoal, snapshot)

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
            "cage_contract_rank": bool(self.config.contract_rank),
            "cage_contract_shadow_rank": bool(self.config.contract_shadow_rank),
            "cage_contract_intervene": bool(self.config.contract_intervene),
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
            "mean_contract_candidate_coverage": float(mean(self.contract_rank_coverages)) if self.contract_rank_coverages else None,
            "mean_contract_selected_score": float(mean(self.contract_rank_selected_scores)) if self.contract_rank_selected_scores else None,
            "mean_contract_gas_score": float(mean(self.contract_rank_gas_scores)) if self.contract_rank_gas_scores else None,
            "contract_rank_choose_gas_count": int(self.contract_rank_choose_gas_count),
            "contract_rank_choose_cage_count": int(self.contract_rank_choose_cage_count),
            "contract_rank_choose_committed_count": int(self.contract_rank_choose_committed_count),
            "contract_rank_extreme_reject_count": int(self.contract_rank_extreme_reject_count),
            "shadow_override_rate": self._safe_rate(self.shadow_override_count, self.shadow_total_count),
            "shadow_override_on_success_rate": self._safe_rate(self.shadow_override_count, self.shadow_total_count) if bool(success) else None,
            "shadow_override_final_phase_rate": self._safe_rate(self.shadow_override_final_phase_count, self.shadow_final_phase_count),
            "shadow_mean_override_margin": float(mean(self.shadow_override_margins)) if self.shadow_override_margins else None,
            "shadow_choose_committed_rate": self._safe_rate(self.shadow_choose_committed_count, self.shadow_total_count),
            "shadow_choose_cage_rate": self._safe_rate(self.shadow_choose_cage_count, self.shadow_total_count),
            "shadow_choose_path_later_rate": self._safe_rate(self.shadow_choose_path_later_count, self.shadow_total_count),
            "intervention_count": int(self.contract_intervention_count),
            "intervention_rate": self._safe_rate(self.contract_intervention_count, self.contract_intervention_total_count),
            "mean_intervention_gain": float(mean(self.contract_intervention_gains)) if self.contract_intervention_gains else None,
            "final_phase_preserved_count": int(self.final_phase_preserved_count),
            "final_phase_override_count": int(self.final_phase_override_count),
            "final_phase_override_reason": ";".join(self.final_phase_override_reasons[-8:]) if self.final_phase_override_reasons else None,
            "committed_stale_count": int(self.committed_stale_count),
            "committed_lockout_count": int(self.committed_lockout_count),
            "committed_lockout_step_count": int(self.committed_lockout_step_count),
            "committed_target_progress_mean": float(mean(self.committed_target_progresses)) if self.committed_target_progresses else None,
            "committed_goal_progress_mean": float(mean(self.committed_goal_progresses)) if self.committed_goal_progresses else None,
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
        max_steps = int(getattr(self.config, "max_debug_steps_per_episode", 0))
        if max_steps > 0 and len(self._step_records) >= max_steps:
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

    def _select_contract_ranked_subgoal(
        self,
        *,
        current_state,
        final_goal,
        path,
        selection,
        step: int,
        trace_info: dict[str, Any],
        info: dict[str, Any] | None,
        final_goal_phase: bool,
        reason: str,
    ):
        if self.contract_scorer is None:
            self._attach_state_ref(trace_info, info, "contract_rank_no_scorer")
            return self._select_original_subgoal(
                current_state=current_state,
                final_goal=final_goal,
                step=step,
                trace_info=trace_info,
                reason="contract_rank_no_scorer",
                fallback=True,
            )
        candidates = self._build_contract_rank_candidates(
            current_state=current_state,
            final_goal=final_goal,
            path=path,
            selection=selection,
            info=info,
            final_goal_phase=final_goal_phase,
        )
        result = self._rank_contract_candidate_set(candidates)
        selected = result.selected_candidate
        self._record_contract_rank_result(result)

        if selected.source == "committed":
            self.state = CAGEState.COMMITTING
        elif selected.target_mode == "final_goal":
            self.state = CAGEState.FINAL_GOAL
        elif selected.is_recovery:
            self.state = CAGEState.RECOVERY
        else:
            self.state = CAGEState.NORMAL

        self._set_target(selected.target, selected.index, step)
        trace_info.update(self._contract_rank_trace_fields(result, reason))
        if selected.prediction is not None:
            trace_info.update(
                {
                    "contract_model_loaded": int(selected.prediction.model_loaded),
                    "contract_score": selected.score,
                    "contract_lcb": selected.prediction.lower_confidence_bound,
                    "contract_uncertainty": selected.prediction.uncertainty,
                    "predicted_hit": selected.prediction.predicted_hit,
                    "predicted_negative_progress": selected.prediction.predicted_negative_progress,
                    "contract_gate_pass": True,
                    "contract_reject_reason": selected.reject_reason,
                }
            )
        trace_info.update(
            self._trace_target(
                selected.target,
                selected.index,
                False,
                result.ranking_reason,
                current_state=current_state,
                distance=selected.distance,
                target_mode=selected.target_mode,
            )
        )
        return np.asarray(selected.target), selected.index, self.state.value, False, trace_info

    def _select_contract_shadow_rank_subgoal(
        self,
        *,
        current_state,
        final_goal,
        path,
        selection,
        step: int,
        trace_info: dict[str, Any],
        info: dict[str, Any] | None,
        final_goal_phase: bool,
        reason: str,
    ):
        if self.contract_scorer is None:
            trace_info["shadow_rank_reason"] = "no_scorer"
            return self._select_original_subgoal(
                current_state=current_state,
                final_goal=final_goal,
                step=step,
                trace_info=trace_info,
                reason="shadow_rank_no_scorer_passthrough",
            )
        candidates = self._build_contract_rank_candidates(
            current_state=current_state,
            final_goal=final_goal,
            path=path,
            selection=selection,
            info=info,
            final_goal_phase=final_goal_phase,
        )
        result = self._rank_contract_candidate_set(candidates)
        self._record_shadow_rank_result(result, final_goal_phase=final_goal_phase)
        trace_info.update(self._shadow_rank_trace_fields(result, final_goal_phase=final_goal_phase, reason=reason))
        return self._select_original_subgoal(
            current_state=current_state,
            final_goal=final_goal,
            step=step,
            trace_info=trace_info,
            reason="shadow_rank_passthrough",
        )

    def _select_contract_intervention_subgoal(
        self,
        *,
        current_state,
        final_goal,
        path,
        selection,
        step: int,
        trace_info: dict[str, Any],
        info: dict[str, Any] | None,
        final_goal_phase: bool,
        reason: str,
    ):
        if self.contract_scorer is None:
            trace_info["intervention_reason"] = "no_scorer"
            return self._select_original_subgoal(
                current_state=current_state,
                final_goal=final_goal,
                step=step,
                trace_info=trace_info,
                reason="contract_intervene_no_scorer",
            )
        candidates = self._build_contract_rank_candidates(
            current_state=current_state,
            final_goal=final_goal,
            path=path,
            selection=selection,
            info=info,
            final_goal_phase=final_goal_phase,
        )
        rank_result = self._rank_contract_candidate_set(candidates)
        committed_locked_out = self._committed_lockout_active(step)
        decision = decide_contract_intervention(
            rank_result.candidates,
            intervention_margin=float(self.config.contract_intervention_margin),
            gas_risk_threshold=float(self.config.contract_intervention_gas_risk_threshold),
            min_final_progress_gain=float(self.config.contract_intervention_min_final_progress_gain),
            min_path_index_gain=float(self.config.contract_intervention_min_path_index_gain),
            intervention_cost=float(self.config.contract_intervention_cost),
            extreme_negative_threshold=float(self.config.contract_rank_extreme_negative_threshold),
            preserve_final_phase=bool(self.config.contract_intervention_preserve_final_phase),
            allow_final_override_only_extreme=bool(self.config.contract_intervention_allow_final_override_only_extreme),
            gas_progress_stalled=bool(self.monitor.is_stalled),
            committed_locked_out=bool(committed_locked_out),
        )
        self._record_contract_intervention_result(decision, rank_result)
        selected = decision.selected_candidate
        if decision.intervention_allowed and final_goal_phase and not selected.is_original_gas:
            self.final_phase_override_count += 1
            self.final_phase_override_reasons.append(decision.intervention_reason)
        if selected.source == "committed":
            self.state = CAGEState.COMMITTING
        elif selected.target_mode == "final_goal":
            self.state = CAGEState.FINAL_GOAL
        else:
            self.state = CAGEState.NORMAL
        self._set_target(selected.target, selected.index, step)
        trace_info.update(self._contract_rank_trace_fields(rank_result, reason))
        trace_info.update(decision_trace(decision))
        if selected.prediction is not None:
            trace_info.update(
                {
                    "contract_model_loaded": int(selected.prediction.model_loaded),
                    "contract_score": selected.score,
                    "contract_lcb": selected.prediction.lower_confidence_bound,
                    "contract_uncertainty": selected.prediction.uncertainty,
                    "predicted_hit": selected.prediction.predicted_hit,
                    "predicted_negative_progress": selected.prediction.predicted_negative_progress,
                    "contract_gate_pass": True,
                    "contract_reject_reason": selected.reject_reason,
                }
            )
        trace_info.update(
            self._trace_target(
                selected.target,
                selected.index,
                False,
                decision.intervention_reason,
                current_state=current_state,
                distance=selected.distance,
                target_mode=selected.target_mode,
            )
        )
        return np.asarray(selected.target), selected.index, self.state.value, False, trace_info

    def _rank_contract_candidate_set(self, candidates: list[ContractCandidate]):
        return rank_contract_candidates(
            candidates,
            self.contract_scorer,
            contract_weight=float(self.config.contract_rank_contract_weight),
            progress_weight=float(self.config.contract_rank_progress_weight),
            negative_weight=float(self.config.contract_rank_negative_weight),
            uncertainty_weight=float(self.config.contract_rank_uncertainty_weight),
            switch_penalty=float(self.config.contract_rank_switch_penalty),
            extreme_negative_threshold=float(self.config.contract_rank_extreme_negative_threshold),
            prefer_gas_margin=float(self.config.contract_rank_prefer_gas_margin),
            min_candidate_coverage=float(self.config.contract_rank_min_candidate_coverage),
        )

    def _build_contract_rank_candidates(
        self,
        *,
        current_state,
        final_goal,
        path,
        selection,
        info: dict[str, Any] | None,
        final_goal_phase: bool,
    ) -> list[ContractCandidate]:
        candidates: list[ContractCandidate] = []
        original = self._last_original_subgoal
        original_index = self._last_original_subgoal_index
        if original is not None:
            candidates.append(
                self._make_contract_candidate(
                    current_state,
                    original,
                    original_index,
                    target_mode="final_goal" if final_goal_phase else "gas_path",
                    source="gas",
                    is_original_gas=True,
                    path_position=original_index,
                    final_phase=final_goal_phase,
                    info=info,
                )
            )
        elif selection is not None:
            candidates.append(
                self._make_contract_candidate(
                    current_state,
                    selection.subgoal,
                    selection.index,
                    target_mode="final_goal" if final_goal_phase else "gas_path",
                    source="gas",
                    is_original_gas=True,
                    path_position=selection.index,
                    final_phase=final_goal_phase,
                    info=info,
                )
            )

        if selection is not None:
            mode = "final_goal" if final_goal_phase else "cage_selected"
            candidates.append(
                self._make_contract_candidate(
                    current_state,
                    selection.subgoal,
                    selection.index,
                    target_mode=mode,
                    source="cage",
                    path_position=selection.index,
                    final_phase=final_goal_phase,
                    info=info,
                )
            )

        if (
            self.current_subgoal is not None
            and not self._target_reached(current_state)
            and self._committed_candidate_allowed(current_state)
        ):
            candidates.append(
                self._make_contract_candidate(
                    current_state,
                    self.current_subgoal,
                    self.current_subgoal_index,
                    target_mode="committed_target",
                    source="committed",
                    is_current_committed=True,
                    path_position=self.current_subgoal_index,
                    final_phase=final_goal_phase,
                    info=info,
                )
            )

        later = self._nearest_later_path_candidate(current_state, path, info)
        if later is not None:
            idx, target = later
            candidates.append(
                self._make_contract_candidate(
                    current_state,
                    target,
                    idx,
                    target_mode="nearest_path_target",
                    source="path_later",
                    path_position=idx,
                    final_phase=False,
                    info=info,
                )
            )

        if final_goal_phase:
            candidates.append(
                self._make_contract_candidate(
                    current_state,
                    final_goal,
                    -1,
                    target_mode="final_goal",
                    source="final_goal",
                    path_position=-1,
                    final_phase=True,
                    info=info,
                )
            )

        if not candidates:
            candidates.append(
                self._make_contract_candidate(
                    current_state,
                    final_goal,
                    -1,
                    target_mode="gas_path",
                    source="gas",
                    is_original_gas=True,
                    path_position=-1,
                    final_phase=final_goal_phase,
                    info=info,
                )
            )
        return candidates

    def _make_contract_candidate(
        self,
        current_state,
        target,
        index,
        *,
        target_mode: str,
        source: str,
        path_position: int | None,
        final_phase: bool,
        info: dict[str, Any] | None,
        is_original_gas: bool = False,
        is_current_committed: bool = False,
        is_recovery: bool = False,
    ) -> ContractCandidate:
        distance = float(self.distance_fn(current_state, target))
        features = self._contract_features(
            current_state,
            target,
            target_mode=target_mode,
            path_position=path_position,
            final_phase=final_phase,
            recovery_candidate=is_recovery,
            info=info,
        )
        return ContractCandidate(
            target=np.asarray(target),
            index=None if index is None else int(index),
            target_mode=target_mode,
            source=source,
            path_position=path_position,
            final_phase=bool(final_phase),
            is_original_gas=bool(is_original_gas),
            is_current_committed=bool(is_current_committed),
            is_recovery=bool(is_recovery),
            distance=distance,
            path_progress_score=self._candidate_path_progress_score(path_position, final_phase=final_phase),
            switch_cost=self._candidate_switch_cost(target),
            features=features,
        )

    def _nearest_later_path_candidate(self, current_state, path, info: dict[str, Any] | None):
        path = as_path_array(path)
        if path is None or len(path) == 0:
            return None
        info = info or {}
        raw_idx = info.get("original_subgoal_index", self.current_subgoal_index)
        try:
            start_idx = int(raw_idx) + 1 if raw_idx is not None else 0
        except (TypeError, ValueError):
            start_idx = 0
        start_idx = max(0, min(start_idx, len(path) - 1))
        suffix = path[start_idx:]
        if len(suffix) == 0:
            return None
        distances = np.asarray([self.distance_fn(current_state, node) for node in suffix], dtype=float)
        offset = int(np.argmin(distances))
        idx = start_idx + offset
        return idx, np.asarray(path[idx])

    def _candidate_path_progress_score(self, path_position: int | None, *, final_phase: bool) -> float:
        if final_phase:
            return 1.0
        if path_position is None:
            return 0.0
        try:
            idx = float(path_position)
        except (TypeError, ValueError):
            return 0.0
        if idx < 0:
            return 1.0
        denom = max(1.0, float(self.final_active_path_length - 1))
        return float(max(0.0, min(1.0, idx / denom)))

    def _candidate_switch_cost(self, target) -> float:
        if self.current_subgoal is None:
            return 0.0
        return 0.0 if float(self.distance_fn(self.current_subgoal, target)) <= 1e-8 else 1.0

    def _record_contract_rank_result(self, result) -> None:
        self.contract_rank_coverages.append(float(result.coverage))
        self.contract_rank_selected_scores.append(float(result.selected_score))
        if result.gas_score is not None:
            self.contract_rank_gas_scores.append(float(result.gas_score))
        self.contract_rank_extreme_reject_count += int(result.extreme_reject_count)
        selected = result.selected_candidate
        if selected.is_original_gas or selected.source == "gas":
            self.contract_rank_choose_gas_count += 1
        elif selected.is_current_committed or selected.source == "committed":
            self.contract_rank_choose_committed_count += 1
        else:
            self.contract_rank_choose_cage_count += 1

    def _record_shadow_rank_result(self, result, *, final_goal_phase: bool) -> None:
        self.shadow_total_count += 1
        if final_goal_phase:
            self.shadow_final_phase_count += 1
        selected = result.selected_candidate
        override = not bool(selected.is_original_gas or selected.source == "gas")
        if override:
            self.shadow_override_count += 1
            if final_goal_phase:
                self.shadow_override_final_phase_count += 1
        if result.gas_score is not None and result.best_non_gas_score is not None:
            self.shadow_override_margins.append(float(result.best_non_gas_score) - float(result.gas_score))
        if selected.source == "committed":
            self.shadow_choose_committed_count += 1
        elif selected.source == "path_later":
            self.shadow_choose_path_later_count += 1
        elif not selected.is_original_gas and selected.source != "gas":
            self.shadow_choose_cage_count += 1

    def _record_contract_intervention_result(self, decision, rank_result) -> None:
        self._record_contract_rank_result(rank_result)
        self.contract_intervention_total_count += 1
        self.contract_intervention_gains.append(float(decision.intervention_gain))
        if decision.intervention_allowed:
            self.contract_intervention_count += 1
        if decision.final_phase_preserved:
            self.final_phase_preserved_count += 1

    def _contract_rank_trace_fields(self, result, reason: str) -> dict[str, Any]:
        row = {
            "contract_rank_enabled": True,
            "contract_candidate_count": int(result.candidate_count),
            "contract_candidate_coverage": float(result.coverage),
            "contract_selected_source": result.selected_candidate.source,
            "contract_selected_score": float(result.selected_score),
            "contract_gas_score": result.gas_score,
            "contract_best_non_gas_score": result.best_non_gas_score,
            "contract_rank_reason": result.ranking_reason,
            "contract_rank_context": reason,
            "contract_extreme_reject_count": int(result.extreme_reject_count),
            "contract_rejected_count": int(result.rejected_count),
        }
        if self.config.contract_rank_debug_candidates:
            row["contract_rank_candidates"] = [candidate_trace(candidate) for candidate in result.candidates]
        return row

    def _shadow_rank_trace_fields(self, result, *, final_goal_phase: bool, reason: str) -> dict[str, Any]:
        selected = result.selected_candidate
        override = not bool(selected.is_original_gas or selected.source == "gas")
        margin = None
        if result.gas_score is not None and result.best_non_gas_score is not None:
            margin = float(result.best_non_gas_score) - float(result.gas_score)
        row = {
            "contract_shadow_rank_enabled": True,
            "shadow_rank_context": reason,
            "shadow_selected_source": selected.source,
            "shadow_selected_score": float(result.selected_score),
            "shadow_gas_score": result.gas_score,
            "shadow_best_non_gas_score": result.best_non_gas_score,
            "shadow_would_override_gas": bool(override),
            "shadow_override_margin": margin,
            "shadow_selected_target_mode": selected.target_mode,
            "shadow_candidate_count": int(result.candidate_count),
            "shadow_candidate_coverage": float(result.coverage),
            "shadow_final_phase": bool(final_goal_phase),
            "shadow_final_phase_override": bool(final_goal_phase and override),
        }
        if self.config.contract_shadow_debug_candidates:
            row["shadow_rank_candidates"] = [candidate_trace(candidate) for candidate in result.candidates]
        return row

    def _committed_candidate_allowed(self, current_state) -> bool:
        if self.current_subgoal is None:
            return False
        if not self.config.contract_intervene:
            return float(self.monitor.progress_window_value) > float(self.config.progress_eps)
        if self._committed_lockout_active(self._last_step):
            self.committed_lockout_step_count += 1
            return False
        if self.commitment_steps >= int(self.config.contract_max_commit_steps):
            self._start_committed_lockout("max_commit_steps")
            return False
        target_distance = float(self.distance_fn(current_state, self.current_subgoal))
        if target_distance <= self.config.effective_target_reach_dist:
            return False
        target_progress = float(self.monitor.progress_window_value)
        goal_progress = float(self.committed_goal_progresses[-1]) if self.committed_goal_progresses else 0.0
        if target_progress < float(self.config.contract_committed_min_target_progress):
            return False
        if goal_progress < float(self.config.contract_committed_min_goal_progress):
            return False
        return True

    def _update_committed_watchdog(self, prev_state, current_state, selected_subgoal, snapshot) -> None:
        if not self.config.contract_intervene:
            return
        if self.state != CAGEState.COMMITTING:
            return
        if self.current_subgoal is None or selected_subgoal is None:
            return
        if float(self.distance_fn(self.current_subgoal, selected_subgoal)) > 1e-8:
            return
        target_progress = float(self.distance_fn(prev_state, selected_subgoal) - self.distance_fn(current_state, selected_subgoal))
        self.committed_target_progresses.append(target_progress)
        goal_progress = 0.0
        if self._last_final_goal is not None:
            goal_progress = float(self.distance_fn(prev_state, self._last_final_goal) - self.distance_fn(current_state, self._last_final_goal))
            self.committed_goal_progresses.append(goal_progress)
        stale = (
            target_progress < float(self.config.contract_committed_min_target_progress)
            or goal_progress < float(self.config.contract_committed_min_goal_progress)
            or bool(snapshot.stalled and self.config.contract_disable_committed_on_stall)
        )
        if stale:
            self.committed_stale_count += 1
            if snapshot.stalled and self.config.contract_disable_committed_on_stall:
                self._start_committed_lockout("stall")

    def _start_committed_lockout(self, reason: str) -> None:
        if not self.config.contract_intervene:
            return
        self.committed_lockout_count += 1
        self.committed_lockout_until_step = max(
            int(self.committed_lockout_until_step),
            int(self._last_step) + int(self.config.contract_committed_lockout_steps),
        )

    def _committed_lockout_active(self, step: int) -> bool:
        return bool(self.config.contract_intervene and int(step) < int(self.committed_lockout_until_step))

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
            "contract_rank_enabled": bool(self.config.contract_rank),
            "contract_shadow_rank_enabled": bool(self.config.contract_shadow_rank),
            "contract_intervene_enabled": bool(self.config.contract_intervene),
            "contract_candidate_count": None,
            "contract_candidate_coverage": None,
            "contract_selected_source": None,
            "contract_selected_score": None,
            "contract_gas_score": None,
            "contract_best_non_gas_score": None,
            "contract_rank_reason": None,
            "contract_rank_context": None,
            "contract_extreme_reject_count": 0,
            "contract_rejected_count": 0,
            "shadow_selected_source": None,
            "shadow_selected_score": None,
            "shadow_gas_score": None,
            "shadow_best_non_gas_score": None,
            "shadow_would_override_gas": None,
            "shadow_override_margin": None,
            "shadow_selected_target_mode": None,
            "shadow_candidate_count": None,
            "shadow_candidate_coverage": None,
            "intervention_allowed": None,
            "intervention_reason": None,
            "intervention_selected_source": None,
            "intervention_gain": None,
            "intervention_gas_score": None,
            "intervention_best_alternative_score": None,
            "intervention_gas_risk": None,
            "intervention_gas_progress_stalled": None,
            "committed_locked_out": None,
            "final_phase_preserved": None,
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
        if (
            not self.config.debug
            or self.config.debug_light
            or self.config.disable_exact_state_ref_trace
            or not info
        ):
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
        }
        if self.config.trace_phi_vectors:
            row["selected_subgoal_phi"] = target_arr.tolist() if target_arr is not None else None
            row["original_gas_subgoal_phi"] = original_arr.tolist() if original_arr is not None else None
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
        committed_lockout_remaining = max(0, int(self.committed_lockout_until_step) - step)
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
            "cage_contract_rank": bool(self.config.contract_rank),
            "cage_contract_shadow_rank": bool(self.config.contract_shadow_rank),
            "cage_contract_intervene": bool(self.config.contract_intervene),
            "committed_lockout_remaining": int(committed_lockout_remaining),
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
