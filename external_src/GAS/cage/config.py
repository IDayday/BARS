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

    @property
    def effective_target_reach_dist(self) -> float:
        if self.target_reach_dist is not None:
            return float(self.target_reach_dist)
        return max(1e-6, float(self.min_subgoal_dist))

    def with_env_defaults(self) -> "CAGEConfig":
        """Return a conservative environment-adjusted config."""
        env_name = (self.env_name or "").lower()
        if "humanoid" not in env_name:
            return self
        return replace(
            self,
            max_subgoal_dist=min(float(self.max_subgoal_dist), 16.0),
            min_commit_steps=max(int(self.min_commit_steps), 10),
            final_min_commit_steps=max(int(self.final_min_commit_steps), 14),
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
            debug=bool(flags_obj.cage_debug),
            env_name=str(flags_obj.env_name),
        ).with_env_defaults()
