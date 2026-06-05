from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from cage.state_ref import capture_state_ref, serialize_state_ref, state_ref_is_exact


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


class ContractTraceWriter:
    """Append-only JSONL writer for branchable segment contract records."""

    def __init__(self, path: str, debug: bool = False):
        self.path = path
        self.debug = bool(debug)
        self._fh = None
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            self._fh = open(path, "a", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        if self._fh is None:
            return
        row = dict(record)
        row.setdefault("record_type", "segment_contract")
        self._fh.write(json.dumps(to_jsonable(row), sort_keys=True) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


class SegmentContractRecorder:
    """Records executed target segments without changing control behavior."""

    def __init__(
        self,
        writer: ContractTraceWriter | None,
        *,
        env_name: str,
        seed: int,
        episode_idx: int,
        variant: str,
        task_id: int | None,
        config: dict[str, Any],
    ):
        self.writer = writer
        self.env_name = env_name
        self.seed = int(seed)
        self.episode_idx = int(episode_idx)
        self.variant = variant
        self.task_id = task_id
        self.config = dict(config)
        self.segment_idx = 0
        self.active: dict[str, Any] | None = None

    def enabled(self) -> bool:
        return self.writer is not None and bool(self.config.get("enabled", False))

    def begin(
        self,
        *,
        env: Any,
        obs: Any,
        phi: np.ndarray,
        target_phi: np.ndarray,
        step: int,
        target_idx: int | None,
        target_source: str,
        path_position: int | None,
        final_phase: bool,
        recovery_active: bool,
        policy_skill_norm: float | None = None,
        path_phi: Any = None,
        final_goal_phi: Any = None,
    ) -> None:
        if not self.enabled():
            return
        state_ref = self._capture(env, obs, phi, step, source=target_source)
        d_phi_start = float(np.linalg.norm(np.asarray(phi) - np.asarray(target_phi)))
        self.active = {
            "segment_idx": self.segment_idx,
            "start_step": int(step),
            "target_phi": np.asarray(target_phi, dtype=np.float32),
            "start_phi": np.asarray(phi, dtype=np.float32),
            "path_phi": None if path_phi is None else np.asarray(path_phi, dtype=np.float32),
            "final_goal_phi": None if final_goal_phi is None else np.asarray(final_goal_phi, dtype=np.float32),
            "start_state_ref": state_ref,
            "selected_target_idx": None if target_idx is None else int(target_idx),
            "target_source": target_source,
            "path_position": None if path_position is None else int(path_position),
            "final_phase": bool(final_phase),
            "recovery_active": bool(recovery_active),
            "d_phi_start": d_phi_start,
            "actions": [],
            "skills": [] if policy_skill_norm is None else [float(policy_skill_norm)],
            "global_replans": 0,
            "target_switches": 0,
            "stalls": 0,
            "drifts": 0,
        }

    def maybe_switch(
        self,
        *,
        env: Any,
        obs: Any,
        phi: np.ndarray,
        target_phi: np.ndarray,
        step: int,
        target_idx: int | None,
        target_source: str,
        path_position: int | None,
        final_phase: bool,
        recovery_active: bool,
        path_phi: Any = None,
        final_goal_phi: Any = None,
    ) -> None:
        if not self.enabled():
            return
        if self.active is None:
            self.begin(
                env=env,
                obs=obs,
                phi=phi,
                target_phi=target_phi,
                step=step,
                target_idx=target_idx,
                target_source=target_source,
                path_position=path_position,
                final_phase=final_phase,
                recovery_active=recovery_active,
                path_phi=path_phi,
                final_goal_phi=final_goal_phi,
            )
            return
        old = np.asarray(self.active["target_phi"])
        if old.shape != np.asarray(target_phi).shape or not np.allclose(old, target_phi, atol=1e-6):
            self.active["target_switches"] += 1
            self.close(env=env, obs=obs, phi=phi, step=step, release_reason="switched", reached_target=False)
            self.begin(
                env=env,
                obs=obs,
                phi=phi,
                target_phi=target_phi,
                step=step,
                target_idx=target_idx,
                target_source=target_source,
                path_position=path_position,
                final_phase=final_phase,
                recovery_active=recovery_active,
                path_phi=path_phi,
                final_goal_phi=final_goal_phi,
            )

    def record_action(self, action: Any, skill_norm: float | None = None) -> None:
        if self.active is None:
            return
        self.active["actions"].append(float(np.linalg.norm(np.asarray(action))))
        if skill_norm is not None:
            self.active["skills"].append(float(skill_norm))

    def record_cage_trace(self, trace_info: dict[str, Any] | None, should_replan: bool = False) -> None:
        if self.active is None:
            return
        if should_replan:
            self.active["global_replans"] += 1
        state = str((trace_info or {}).get("cage_state", ""))
        if "STALL" in state:
            self.active["stalls"] += 1
        if "DRIFT" in state:
            self.active["drifts"] += 1

    def close(self, *, env: Any, obs: Any, phi: np.ndarray, step: int, release_reason: str, reached_target: bool | None = None) -> None:
        if not self.enabled() or self.active is None:
            return
        active = self.active
        end_state_ref = self._capture(env, obs, phi, step, source="segment_end")
        target_phi = np.asarray(active["target_phi"], dtype=np.float32)
        end_phi = np.asarray(phi, dtype=np.float32)
        d_phi_end = float(np.linalg.norm(end_phi - target_phi))
        d_phi_start = float(active["d_phi_start"])
        delta_phi = d_phi_start - d_phi_end
        normalized_progress = delta_phi / (d_phi_start + 1e-10)
        actions = active["actions"]
        skills = active["skills"]
        if reached_target is None:
            threshold = float(self.config.get("hit_threshold", 1.0))
            reached_target = d_phi_end <= threshold
        record = {
            "record_type": "segment_contract",
            "env_name": self.env_name,
            "seed": self.seed,
            "episode_idx": self.episode_idx,
            "task_id": self.task_id,
            "segment_idx": int(active["segment_idx"]),
            "segment_id": f"{self.env_name}__{self.variant}__seed{self.seed}__task{self.task_id}__ep{self.episode_idx}__seg{active['segment_idx']}",
            "variant": self.variant,
            "start_step": int(active["start_step"]),
            "end_step": int(step),
            "release_reason": release_reason,
            "planner_path_id": None,
            "path_position": active["path_position"],
            "original_gas_target_idx": None,
            "selected_target_idx": active["selected_target_idx"],
            "target_source": active["target_source"],
            "start_state_ref": active["start_state_ref"],
            "end_state_ref": end_state_ref,
            "target_state_ref": None,
            "target_phi": target_phi,
            "start_phi": active["start_phi"],
            "end_phi": end_phi,
            "path_phi": active.get("path_phi"),
            "final_goal_phi": active.get("final_goal_phi"),
            "d_phi_start": d_phi_start,
            "d_phi_end": d_phi_end,
            "delta_phi": delta_phi,
            "normalized_progress": normalized_progress,
            "hit_threshold": self.config.get("hit_threshold"),
            "reached_target": bool(reached_target),
            "switched_before_reach": bool(release_reason == "switched" and not reached_target),
            "recovery_active": active["recovery_active"],
            "final_phase": active["final_phase"],
            "global_replan_count_in_segment": int(active["global_replans"]),
            "target_switch_count_in_segment": int(active["target_switches"]),
            "local_stall_count_in_segment": int(active["stalls"]),
            "drift_count_in_segment": int(active["drifts"]),
            "action_norm_mean": float(np.mean(actions)) if actions else None,
            "action_norm_max": float(np.max(actions)) if actions else None,
            "policy_skill_norm_mean": float(np.mean(skills)) if skills else None,
            "policy_skill_norm_max": float(np.max(skills)) if skills else None,
            "exact_start_state_ref": bool(active["start_state_ref"].get("exact_reset")) if isinstance(active["start_state_ref"], dict) else False,
            "exact_end_state_ref": bool(end_state_ref.get("exact_reset")) if isinstance(end_state_ref, dict) else False,
        }
        self.writer.write(record)
        self.active = None
        self.segment_idx += 1

    def _capture(self, env: Any, obs: Any, phi: np.ndarray, step: int, source: str) -> dict[str, Any] | None:
        if not self.config.get("store_state_refs", False):
            return None
        try:
            ref = capture_state_ref(
                env,
                obs=obs,
                phi=phi if self.config.get("capture_phi", True) else None,
                metadata={
                    "env_name": self.env_name,
                    "seed": self.seed,
                    "episode_idx": self.episode_idx,
                    "step_idx": int(step),
                    "source": source,
                    "source_variant": self.variant,
                },
            )
            mode = self.config.get("state_ref_mode", "best_effort")
            if mode == "exact_only" and not state_ref_is_exact(ref):
                return {
                    "env_name": self.env_name,
                    "seed": self.seed,
                    "step_idx": int(step),
                    "source": source,
                    "reset_mode": ref.reset_mode,
                    "exact_reset": False,
                    "failure_reason": "exact StateRef unavailable",
                }
            return serialize_state_ref(ref)
        except Exception as exc:
            return {
                "env_name": self.env_name,
                "seed": self.seed,
                "step_idx": int(step),
                "source": source,
                "reset_mode": "unsupported",
                "exact_reset": False,
                "failure_reason": f"{type(exc).__name__}: {exc}",
            }
