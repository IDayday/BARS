from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class CAGEConfig:
    """Configuration for the CAGE execution wrapper.

    The defaults are conservative and are only used when CAGE is explicitly
    enabled by the evaluator.
    """

    use_cage: bool = False
    trace_path: str = ""
    min_commit_steps: int = 8
    stall_window: int = 8
    progress_eps: float = 0.01
    drift_threshold: float = 16.0
    max_subgoal_dist: float = 24.0
    min_subgoal_dist: float = 2.0
    recovery_commit_steps: int = 12
    max_recovery_attempts: int = 2
    recovery_suffix_weight: float = 0.25
    final_phase_dist: float = 8.0
    final_min_commit_steps: int = 12
    debug: bool = False
    env_name: str = ""
    reachability_tau: float = 10.0
    target_reach_dist: float | None = None
    disable_commitment: bool = False
    disable_drift_monitor: bool = False
    disable_recovery: bool = False
    disable_adaptive_horizon: bool = False
    disable_final_phase_controller: bool = False
    use_reachability: bool = False
    reachability_path: str = ""
    risk_aware_path: bool = False
    trace_only: bool = False
    enable_churn_guard: bool = False
    replan_cooldown_steps: int = 10
    max_global_replans_per_episode: int = 50
    max_replans_per_100_steps: int = 10
    max_consecutive_replan_requests: int = 5
    fallback_to_gas_on_churn: bool = True
    fallback_to_gas_steps: int = 50
    recovery_lockout_steps_after_failure: int = 25
    min_steps_between_recovery_attempts: int = 20
    min_progress_for_recovery_success: float = 1e-4
    disable_recovery_after_churn: bool = False
    log_churn_events: bool = False
    contract_commit: bool = False
    use_contract_model: bool = False
    contract_model_path: str = ""
    contract_lcb_threshold: float = 0.35
    contract_negative_progress_threshold: float = 0.45
    contract_uncertainty_penalty: float = 0.25
    contract_min_commit_steps: int = 12
    contract_final_goal_threshold: float = 0.50
    contract_recovery_threshold: float = 0.55
    contract_disable_recovery_when_uncertain: bool = True
    contract_fallback_to_gas_when_uncertain: bool = True
    contract_rank: bool = False
    contract_rank_min_candidate_coverage: float = 0.30
    contract_rank_max_reject_rate: float = 0.80
    contract_rank_contract_weight: float = 1.0
    contract_rank_progress_weight: float = 0.5
    contract_rank_negative_weight: float = 0.5
    contract_rank_uncertainty_weight: float = 0.25
    contract_rank_switch_penalty: float = 0.10
    contract_rank_extreme_negative_threshold: float = 0.90
    contract_rank_prefer_gas_margin: float = 0.05
    contract_rank_disable_hard_lcb_gate: bool = True
    contract_rank_debug_candidates: bool = False
    contract_shadow_rank: bool = False
    contract_intervene: bool = False
    contract_intervention_margin: float = 0.10
    contract_intervention_gas_risk_threshold: float = 0.60
    contract_intervention_min_final_progress_gain: float = 0.00
    contract_intervention_min_path_index_gain: float = 0.00
    contract_intervention_cost: float = 0.05
    contract_intervention_preserve_final_phase: bool = True
    contract_intervention_allow_final_override_only_extreme: bool = True
    contract_max_commit_steps: int = 24
    contract_committed_min_target_progress: float = 0.01
    contract_committed_min_goal_progress: float = 0.00
    contract_committed_lockout_steps: int = 20
    contract_disable_committed_on_stall: bool = True
    contract_shadow_debug_candidates: bool = False
    debug_light: bool = False
    disable_exact_state_ref_trace: bool = False
    max_debug_steps_per_episode: int = 0
    trace_phi_vectors: bool = True

    @property
    def effective_target_reach_dist(self) -> float:
        if self.target_reach_dist is not None:
            return float(self.target_reach_dist)
        return max(1e-6, float(self.min_subgoal_dist))

    def with_env_defaults(self) -> "CAGEConfig":
        """Return a conservative environment-adjusted config."""
        cfg = self
        if cfg.contract_commit or cfg.contract_rank or cfg.contract_shadow_rank or cfg.contract_intervene:
            cfg = replace(
                cfg,
                min_commit_steps=max(int(cfg.min_commit_steps), int(cfg.contract_min_commit_steps)),
                enable_churn_guard=True,
                fallback_to_gas_on_churn=True,
                disable_recovery=True if cfg.contract_disable_recovery_when_uncertain else cfg.disable_recovery,
                disable_recovery_after_churn=True,
                max_consecutive_replan_requests=min(int(cfg.max_consecutive_replan_requests), 5),
                max_global_replans_per_episode=min(int(cfg.max_global_replans_per_episode), 50),
                max_replans_per_100_steps=min(int(cfg.max_replans_per_100_steps), 10),
            )
        env_name = (cfg.env_name or "").lower()
        if "humanoid" not in env_name:
            return cfg
        return replace(
            cfg,
            max_subgoal_dist=min(float(cfg.max_subgoal_dist), 16.0),
            min_commit_steps=max(int(cfg.min_commit_steps), 10),
            final_min_commit_steps=max(int(cfg.final_min_commit_steps), 14),
        )

    @classmethod
    def from_flags(cls, flags_obj) -> "CAGEConfig":
        return cls(
            use_cage=bool(flags_obj.use_cage),
            trace_path=str(flags_obj.cage_trace_path or ""),
            min_commit_steps=int(flags_obj.cage_min_commit_steps),
            stall_window=int(flags_obj.cage_stall_window),
            progress_eps=float(flags_obj.cage_progress_eps),
            drift_threshold=float(flags_obj.cage_drift_threshold),
            max_subgoal_dist=float(flags_obj.cage_max_subgoal_dist),
            min_subgoal_dist=float(flags_obj.cage_min_subgoal_dist),
            recovery_commit_steps=int(flags_obj.cage_recovery_commit_steps),
            max_recovery_attempts=int(flags_obj.cage_max_recovery_attempts),
            recovery_suffix_weight=float(flags_obj.cage_recovery_suffix_weight),
            final_phase_dist=float(flags_obj.cage_final_phase_dist),
            final_min_commit_steps=int(flags_obj.cage_final_min_commit_steps),
            debug=bool(flags_obj.cage_debug or flags_obj.cage_debug_light),
            env_name=str(flags_obj.env_name),
            disable_commitment=bool(flags_obj.cage_disable_commitment),
            disable_drift_monitor=bool(flags_obj.cage_disable_drift_monitor),
            disable_recovery=bool(flags_obj.cage_disable_recovery),
            disable_adaptive_horizon=bool(flags_obj.cage_disable_adaptive_horizon),
            disable_final_phase_controller=bool(flags_obj.cage_disable_final_phase_controller),
            use_reachability=bool(flags_obj.cage_use_reachability),
            reachability_path=str(flags_obj.cage_reachability_path or ""),
            risk_aware_path=bool(flags_obj.cage_risk_aware_path),
            trace_only=bool(flags_obj.cage_trace_only),
            enable_churn_guard=bool(flags_obj.cage_enable_churn_guard),
            replan_cooldown_steps=int(flags_obj.cage_replan_cooldown_steps),
            max_global_replans_per_episode=int(flags_obj.cage_max_global_replans_per_episode),
            max_replans_per_100_steps=int(flags_obj.cage_max_replans_per_100_steps),
            max_consecutive_replan_requests=int(flags_obj.cage_max_consecutive_replan_requests),
            fallback_to_gas_on_churn=bool(flags_obj.cage_fallback_to_gas_on_churn),
            fallback_to_gas_steps=int(flags_obj.cage_fallback_to_gas_steps),
            recovery_lockout_steps_after_failure=int(flags_obj.cage_recovery_lockout_steps_after_failure),
            min_steps_between_recovery_attempts=int(flags_obj.cage_min_steps_between_recovery_attempts),
            min_progress_for_recovery_success=float(flags_obj.cage_min_progress_for_recovery_success),
            disable_recovery_after_churn=bool(flags_obj.cage_disable_recovery_after_churn),
            log_churn_events=bool(flags_obj.cage_log_churn_events),
            contract_commit=bool(flags_obj.cage_contract_commit),
            use_contract_model=bool(flags_obj.cage_use_contract_model),
            contract_model_path=str(flags_obj.cage_contract_model_path or ""),
            contract_lcb_threshold=float(flags_obj.cage_contract_lcb_threshold),
            contract_negative_progress_threshold=float(flags_obj.cage_contract_negative_progress_threshold),
            contract_uncertainty_penalty=float(flags_obj.cage_contract_uncertainty_penalty),
            contract_min_commit_steps=int(flags_obj.cage_contract_min_commit_steps),
            contract_final_goal_threshold=float(flags_obj.cage_contract_final_goal_threshold),
            contract_recovery_threshold=float(flags_obj.cage_contract_recovery_threshold),
            contract_disable_recovery_when_uncertain=bool(flags_obj.cage_contract_disable_recovery_when_uncertain),
            contract_fallback_to_gas_when_uncertain=bool(flags_obj.cage_contract_fallback_to_gas_when_uncertain),
            contract_rank=bool(flags_obj.cage_contract_rank),
            contract_rank_min_candidate_coverage=float(flags_obj.cage_contract_rank_min_candidate_coverage),
            contract_rank_max_reject_rate=float(flags_obj.cage_contract_rank_max_reject_rate),
            contract_rank_contract_weight=float(flags_obj.cage_contract_rank_contract_weight),
            contract_rank_progress_weight=float(flags_obj.cage_contract_rank_progress_weight),
            contract_rank_negative_weight=float(flags_obj.cage_contract_rank_negative_weight),
            contract_rank_uncertainty_weight=float(flags_obj.cage_contract_rank_uncertainty_weight),
            contract_rank_switch_penalty=float(flags_obj.cage_contract_rank_switch_penalty),
            contract_rank_extreme_negative_threshold=float(flags_obj.cage_contract_rank_extreme_negative_threshold),
            contract_rank_prefer_gas_margin=float(flags_obj.cage_contract_rank_prefer_gas_margin),
            contract_rank_disable_hard_lcb_gate=bool(flags_obj.cage_contract_rank_disable_hard_lcb_gate),
            contract_rank_debug_candidates=bool(flags_obj.cage_contract_rank_debug_candidates),
            contract_shadow_rank=bool(flags_obj.cage_contract_shadow_rank),
            contract_intervene=bool(flags_obj.cage_contract_intervene),
            contract_intervention_margin=float(flags_obj.cage_contract_intervention_margin),
            contract_intervention_gas_risk_threshold=float(flags_obj.cage_contract_intervention_gas_risk_threshold),
            contract_intervention_min_final_progress_gain=float(flags_obj.cage_contract_intervention_min_final_progress_gain),
            contract_intervention_min_path_index_gain=float(flags_obj.cage_contract_intervention_min_path_index_gain),
            contract_intervention_cost=float(flags_obj.cage_contract_intervention_cost),
            contract_intervention_preserve_final_phase=bool(flags_obj.cage_contract_intervention_preserve_final_phase),
            contract_intervention_allow_final_override_only_extreme=bool(flags_obj.cage_contract_intervention_allow_final_override_only_extreme),
            contract_max_commit_steps=int(flags_obj.cage_contract_max_commit_steps),
            contract_committed_min_target_progress=float(flags_obj.cage_contract_committed_min_target_progress),
            contract_committed_min_goal_progress=float(flags_obj.cage_contract_committed_min_goal_progress),
            contract_committed_lockout_steps=int(flags_obj.cage_contract_committed_lockout_steps),
            contract_disable_committed_on_stall=bool(flags_obj.cage_contract_disable_committed_on_stall),
            contract_shadow_debug_candidates=bool(flags_obj.cage_contract_shadow_debug_candidates),
            debug_light=bool(flags_obj.cage_debug_light),
            disable_exact_state_ref_trace=bool(flags_obj.cage_disable_exact_state_ref_trace),
            max_debug_steps_per_episode=int(flags_obj.cage_max_debug_steps_per_episode),
            trace_phi_vectors=bool(flags_obj.cage_trace_phi_vectors),
        ).with_env_defaults()
