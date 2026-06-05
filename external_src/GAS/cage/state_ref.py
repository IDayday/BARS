from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


EXACT_MUJOCO_STATE = "exact_mujoco_state"
DATASET_STATE_REF = "dataset_state_ref"
OBSERVATION_ONLY_NOT_EXACT = "observation_only_not_exact"
UNSUPPORTED = "unsupported"


@dataclass
class StateRef:
    """Serializable reference to an environment state used for closed-loop probes."""

    env_name: str
    dataset_name: str | None = None
    seed: int | None = None
    episode_idx: int | None = None
    step_idx: int | None = None
    obs: np.ndarray | None = None
    goal_obs: np.ndarray | None = None
    phi: np.ndarray | None = None
    qpos: np.ndarray | None = None
    qvel: np.ndarray | None = None
    raw_state_dict: dict[str, Any] | None = None
    source: str = "unknown"
    reset_mode: str = UNSUPPORTED
    metadata: dict[str, Any] = field(default_factory=dict)


def _array_or_none(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=np.float32)


def _env_data_qpos_qvel(env: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
    unwrapped = getattr(env, "unwrapped", env)
    data = getattr(unwrapped, "data", None)
    if data is None or not hasattr(data, "qpos") or not hasattr(data, "qvel"):
        return None, None
    return np.asarray(data.qpos, dtype=np.float64).copy(), np.asarray(data.qvel, dtype=np.float64).copy()


def make_state_ref_from_env(env: Any, obs: Any = None, phi: Any = None, metadata: dict[str, Any] | None = None) -> StateRef:
    """Capture the current simulator state when available."""
    metadata = dict(metadata or {})
    qpos, qvel = _env_data_qpos_qvel(env)
    reset_mode = EXACT_MUJOCO_STATE if qpos is not None and qvel is not None else OBSERVATION_ONLY_NOT_EXACT if obs is not None else UNSUPPORTED
    return StateRef(
        env_name=str(metadata.get("env_name", getattr(env, "spec", None).id if getattr(env, "spec", None) is not None else "")),
        dataset_name=metadata.get("dataset_name"),
        seed=metadata.get("seed"),
        episode_idx=metadata.get("episode_idx"),
        step_idx=metadata.get("step_idx"),
        obs=_array_or_none(obs),
        goal_obs=_array_or_none(metadata.get("goal_obs")),
        phi=_array_or_none(phi),
        qpos=qpos,
        qvel=qvel,
        raw_state_dict=metadata.get("raw_state_dict"),
        source=str(metadata.get("source", "env")),
        reset_mode=reset_mode,
        metadata={k: v for k, v in metadata.items() if k not in {"goal_obs", "raw_state_dict", "source"}},
    )


def make_state_ref_from_observation(
    env_name: str,
    obs: Any,
    *,
    phi: Any = None,
    seed: int | None = None,
    dataset_name: str | None = None,
    source: str = "dataset",
    qpos: Any = None,
    qvel: Any = None,
    reset_mode: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> StateRef:
    obs_arr = _array_or_none(obs)
    qpos_arr = _array_or_none(qpos)
    qvel_arr = _array_or_none(qvel)
    if reset_mode is None:
        reset_mode = EXACT_MUJOCO_STATE if qpos_arr is not None and qvel_arr is not None else OBSERVATION_ONLY_NOT_EXACT
    return StateRef(
        env_name=env_name,
        dataset_name=dataset_name,
        seed=seed,
        obs=obs_arr,
        phi=_array_or_none(phi),
        qpos=qpos_arr,
        qvel=qvel_arr,
        source=source,
        reset_mode=reset_mode,
        metadata=dict(metadata or {}),
    )


def restore_env_from_state_ref(env: Any, state_ref: StateRef, *, allow_approximate: bool = False) -> Any:
    """Restore an environment from a StateRef.

    Exact restore requires qpos/qvel and an environment set_state method. Observation-only
    records are never silently treated as exact.
    """
    if state_ref.qpos is not None and state_ref.qvel is not None:
        unwrapped = getattr(env, "unwrapped", env)
        if not hasattr(unwrapped, "set_state"):
            raise RuntimeError("StateRef has qpos/qvel, but env.unwrapped.set_state is unavailable")
        unwrapped.set_state(np.asarray(state_ref.qpos, dtype=np.float64), np.asarray(state_ref.qvel, dtype=np.float64))
        return env
    if allow_approximate and state_ref.obs is not None:
        raise RuntimeError("Approximate observation-only restore is not implemented for GAS/OGBench probes")
    raise RuntimeError(f"StateRef is not exactly restorable: reset_mode={state_ref.reset_mode}")


def state_ref_is_exact(state_ref: StateRef) -> bool:
    return state_ref.reset_mode == EXACT_MUJOCO_STATE and state_ref.qpos is not None and state_ref.qvel is not None


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def serialize_state_ref(state_ref: StateRef) -> dict[str, Any]:
    return {
        "env_name": state_ref.env_name,
        "dataset_name": state_ref.dataset_name,
        "seed": state_ref.seed,
        "episode_idx": state_ref.episode_idx,
        "step_idx": state_ref.step_idx,
        "obs": _to_jsonable(state_ref.obs),
        "goal_obs": _to_jsonable(state_ref.goal_obs),
        "phi": _to_jsonable(state_ref.phi),
        "qpos": _to_jsonable(state_ref.qpos),
        "qvel": _to_jsonable(state_ref.qvel),
        "raw_state_dict": _to_jsonable(state_ref.raw_state_dict),
        "source": state_ref.source,
        "reset_mode": state_ref.reset_mode,
        "exact_reset": state_ref_is_exact(state_ref),
        "metadata": _to_jsonable(state_ref.metadata),
    }


def deserialize_state_ref(record: dict[str, Any]) -> StateRef:
    return StateRef(
        env_name=str(record.get("env_name", "")),
        dataset_name=record.get("dataset_name"),
        seed=record.get("seed"),
        episode_idx=record.get("episode_idx"),
        step_idx=record.get("step_idx"),
        obs=_array_or_none(record.get("obs")),
        goal_obs=_array_or_none(record.get("goal_obs")),
        phi=_array_or_none(record.get("phi")),
        qpos=_array_or_none(record.get("qpos")),
        qvel=_array_or_none(record.get("qvel")),
        raw_state_dict=record.get("raw_state_dict"),
        source=str(record.get("source", "unknown")),
        reset_mode=str(record.get("reset_mode", UNSUPPORTED)),
        metadata=dict(record.get("metadata") or {}),
    )
