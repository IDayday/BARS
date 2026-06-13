from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from typing import Any, Callable

import numpy as np
import pandas as pd


DEFAULT_RECONSTRUCTION_TOLERANCE = 1e-4
RESET_STATUS_ENV_UNAVAILABLE = "env_unavailable"
RESET_STATUS_SUPPORTED = "reset_supported"
RESET_STATUS_UNSUPPORTED = "reset_unsupported"
RESET_STATUS_UNCERTAIN = "reset_uncertain"
RESET_PROBE_STATUSES = {
    RESET_STATUS_ENV_UNAVAILABLE,
    RESET_STATUS_SUPPORTED,
    RESET_STATUS_UNSUPPORTED,
    RESET_STATUS_UNCERTAIN,
}


@dataclass(frozen=True)
class ProbeState:
    observation: np.ndarray
    qpos: np.ndarray | None = None
    qvel: np.ndarray | None = None
    state: np.ndarray | None = None
    sim_state: np.ndarray | None = None


@dataclass(frozen=True)
class ResetAttempt:
    method: str
    fn: Callable[[], Any]


def _as_flat_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.dtype == object and arr.ndim == 0:
        return None
    return arr.astype(np.float64, copy=False).reshape(-1)


def _first_present(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def normalize_probe_state(dataset_observation: Any) -> ProbeState:
    """Normalize a dataset state reference without inventing hidden state schema.

    ``dataset_observation`` may be a raw observation array or a mapping with
    explicit ``observation``/``qpos``/``qvel``/``state``/``sim_state`` entries.
    qpos/qvel reset attempts are only generated when qpos and qvel are present.
    """

    if isinstance(dataset_observation, dict):
        obs = _first_present(dataset_observation, ("observation", "observations", "obs"))
        if obs is None:
            raise KeyError("dataset_observation mapping must include observation/observations/obs")
        return ProbeState(
            observation=np.asarray(obs, dtype=np.float64).reshape(-1),
            qpos=_as_flat_array(_first_present(dataset_observation, ("qpos", "infos/qpos"))),
            qvel=_as_flat_array(_first_present(dataset_observation, ("qvel", "infos/qvel"))),
            state=_as_flat_array(_first_present(dataset_observation, ("state", "states"))),
            sim_state=_as_flat_array(_first_present(dataset_observation, ("sim_state", "sim_states"))),
        )
    return ProbeState(observation=np.asarray(dataset_observation, dtype=np.float64).reshape(-1))


def _target_objects(env: Any) -> tuple[Any, Any]:
    unwrapped = getattr(env, "unwrapped", env)
    return env, unwrapped


def _forward(target: Any) -> None:
    if hasattr(target, "forward"):
        target.forward()
        return
    sim = getattr(target, "sim", None)
    if sim is not None and hasattr(sim, "forward"):
        sim.forward()


def _extract_obs_from_return(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, tuple) and value:
        value = value[0]
    arr = _as_flat_array(value)
    return arr


def current_observation(env: Any) -> np.ndarray | None:
    """Best-effort observation readback after reset-to-state."""

    env_obj, unwrapped = _target_objects(env)
    candidates = [env_obj, unwrapped]
    for obj in candidates:
        for name in ("get_ob", "get_obs", "get_observation", "_get_obs", "_get_obs_from_state"):
            if hasattr(obj, name):
                try:
                    obs = getattr(obj, name)()
                    arr = _as_flat_array(obs)
                    if arr is not None:
                        return arr
                except Exception:
                    continue
    for obj in candidates:
        for name in ("state_vector",):
            if hasattr(obj, name):
                try:
                    arr = _as_flat_array(getattr(obj, name)())
                    if arr is not None:
                        return arr
                except Exception:
                    continue
    for obj in candidates:
        if hasattr(obj, "state"):
            arr = _as_flat_array(getattr(obj, "state"))
            if arr is not None:
                return arr
    return None


def _obs_error(env: Any, target_obs: np.ndarray, returned: Any = None) -> tuple[float | None, str | None]:
    obs = current_observation(env)
    if obs is None:
        obs = _extract_obs_from_return(returned)
    if obs is None:
        return None, "could_not_read_observation_after_reset"
    target = np.asarray(target_obs, dtype=np.float64).reshape(-1)
    obs = np.asarray(obs, dtype=np.float64).reshape(-1)
    if obs.shape != target.shape:
        return None, f"observation_shape_mismatch: got={obs.shape} expected={target.shape}"
    return float(np.max(np.abs(obs - target))), None


def _same_object(a: Any, b: Any) -> bool:
    return id(a) == id(b)


def _set_state_attempts(env: Any, state: ProbeState) -> list[ResetAttempt]:
    env_obj, unwrapped = _target_objects(env)
    attempts: list[ResetAttempt] = []

    def add_for(obj: Any, prefix: str) -> None:
        if not hasattr(obj, "set_state"):
            return
        set_state = getattr(obj, "set_state")
        attempts.append(
            ResetAttempt(
                f"{prefix}.set_state(obs)",
                lambda set_state=set_state, obs=state.observation.copy(): set_state(obs),
            )
        )
        if state.state is not None:
            attempts.append(
                ResetAttempt(
                    f"{prefix}.set_state(state)",
                    lambda set_state=set_state, s=state.state.copy(): set_state(s),
                )
            )
        if state.sim_state is not None:
            attempts.append(
                ResetAttempt(
                    f"{prefix}.set_state(sim_state)",
                    lambda set_state=set_state, s=state.sim_state.copy(): set_state(s),
                )
            )
        if state.qpos is not None and state.qvel is not None:
            attempts.append(
                ResetAttempt(
                    f"{prefix}.set_state(qpos,qvel)",
                    lambda set_state=set_state, qpos=state.qpos.copy(), qvel=state.qvel.copy(): set_state(qpos, qvel),
                )
            )

    add_for(unwrapped, "unwrapped")
    if not _same_object(env_obj, unwrapped):
        add_for(env_obj, "env")
    return attempts


def _mujoco_attempts(env: Any, state: ProbeState) -> list[ResetAttempt]:
    if state.qpos is None or state.qvel is None:
        return []
    _, unwrapped = _target_objects(env)
    data = getattr(unwrapped, "data", None)
    if data is None or not hasattr(data, "qpos") or not hasattr(data, "qvel"):
        return []
    qpos_ref = np.asarray(data.qpos).reshape(-1)
    qvel_ref = np.asarray(data.qvel).reshape(-1)
    if qpos_ref.shape != state.qpos.shape or qvel_ref.shape != state.qvel.shape:
        return []

    def apply_mujoco_data() -> None:
        np.asarray(data.qpos).reshape(-1)[:] = state.qpos
        np.asarray(data.qvel).reshape(-1)[:] = state.qvel
        _forward(unwrapped)

    return [ResetAttempt("mujoco_data(qpos,qvel)", apply_mujoco_data)]


def _reset_to_state_attempts(env: Any, state: ProbeState) -> list[ResetAttempt]:
    env_obj, unwrapped = _target_objects(env)
    attempts: list[ResetAttempt] = []

    def add_for(obj: Any, prefix: str) -> None:
        if not hasattr(obj, "reset_to_state"):
            return
        reset_to_state = getattr(obj, "reset_to_state")
        payloads: list[tuple[str, Any]] = [("obs", state.observation.copy())]
        if state.state is not None:
            payloads.append(("state", state.state.copy()))
        if state.sim_state is not None:
            payloads.append(("sim_state", state.sim_state.copy()))
        if state.qpos is not None and state.qvel is not None:
            payloads.append(("qpos_qvel_dict", {"qpos": state.qpos.copy(), "qvel": state.qvel.copy()}))
        for suffix, payload in payloads:
            attempts.append(
                ResetAttempt(
                    f"{prefix}.reset_to_state({suffix})",
                    lambda reset_to_state=reset_to_state, payload=payload: reset_to_state(payload),
                )
            )

    add_for(env_obj, "env")
    if not _same_object(env_obj, unwrapped):
        add_for(unwrapped, "unwrapped")
    return attempts


def _attempts(env: Any, state: ProbeState) -> list[ResetAttempt]:
    # Required order: unwrapped/env set_state, explicit MuJoCo state, wrapper reset_to_state.
    return _set_state_attempts(env, state) + _mujoco_attempts(env, state) + _reset_to_state_attempts(env, state)


def _legacy_method_name(method: str) -> str:
    if method.endswith(".set_state(obs)"):
        return "set_state(obs)"
    if method.endswith(".set_state(qpos,qvel)"):
        return "set_state(qpos,qvel)"
    return method


def missing_reset_env_packages(packages: tuple[str, ...] = ("gymnasium", "gym")) -> list[str]:
    return [name for name in packages if importlib.util.find_spec(name) is None]


def env_unavailable_probe_result(
    env_construction_error: str | Exception,
    missing_packages: list[str] | None = None,
    num_probe_states: int | None = None,
) -> dict[str, Any]:
    if isinstance(env_construction_error, Exception):
        error = f"{type(env_construction_error).__name__}: {env_construction_error}"
    else:
        error = str(env_construction_error)
    return {
        "env_available": False,
        "env_construction_error": error,
        "reset_probe_status": RESET_STATUS_ENV_UNAVAILABLE,
        "reset_supported": None,
        "reset_method": None,
        "method": None,
        "attempted_method": "env_construction",
        "obs_reconstruction_error": np.nan,
        "obs_reconstruction_error_mean": np.nan,
        "obs_reconstruction_error_max": np.nan,
        "num_probe_states": None if num_probe_states is None else int(num_probe_states),
        "missing_packages": list(missing_packages if missing_packages is not None else missing_reset_env_packages()),
        "failure_reason": error,
        "attempted_methods": [],
    }


def _single_probe_result(
    *,
    status: str,
    reset_supported: bool | None,
    reset_method: str | None,
    attempted_method: str,
    obs_error: float | None,
    failure_reason: str,
    attempted_methods: list[str],
) -> dict[str, Any]:
    return {
        "env_available": True,
        "env_construction_error": None,
        "reset_probe_status": status,
        "reset_supported": reset_supported,
        "reset_method": reset_method,
        "method": _legacy_method_name(reset_method) if reset_method else None,
        "attempted_method": attempted_method,
        "obs_reconstruction_error": np.nan if obs_error is None else float(obs_error),
        "obs_reconstruction_error_mean": np.nan if obs_error is None else float(obs_error),
        "obs_reconstruction_error_max": np.nan if obs_error is None else float(obs_error),
        "missing_packages": [],
        "failure_reason": failure_reason,
        "attempted_methods": attempted_methods,
    }


def probe_reset_capability(
    env: Any,
    dataset_observation: Any,
    reconstruction_tolerance: float = DEFAULT_RECONSTRUCTION_TOLERANCE,
) -> dict[str, Any]:
    """Probe whether one dataset state can be restored and observation-verified."""

    state = normalize_probe_state(dataset_observation)
    attempted: list[str] = []
    failures: list[str] = []
    had_successful_call = False
    had_unverified_success = False
    for attempt in _attempts(env, state):
        attempted.append(attempt.method)
        try:
            returned = attempt.fn()
            had_successful_call = True
            _forward(getattr(env, "unwrapped", env))
            error, reason = _obs_error(env, state.observation, returned)
            if reason is not None:
                had_unverified_success = True
                failures.append(f"{attempt.method}: {reason}")
                continue
            assert error is not None
            if error <= float(reconstruction_tolerance):
                return _single_probe_result(
                    status=RESET_STATUS_SUPPORTED,
                    reset_supported=True,
                    reset_method=attempt.method,
                    attempted_method=attempt.method,
                    obs_error=error,
                    failure_reason="",
                    attempted_methods=attempted,
                )
            had_unverified_success = True
            failures.append(
                f"{attempt.method}: reconstruction_error={error:.6g} "
                f"> tolerance={float(reconstruction_tolerance):.6g}"
            )
        except Exception as exc:  # noqa: PERF203 - each method is an independent capability probe.
            failures.append(f"{attempt.method}: {type(exc).__name__}: {exc}")
    if not attempted:
        failures.append("no supported reset method found on env/unwrapped")
    status = RESET_STATUS_UNCERTAIN if (had_successful_call or had_unverified_success) else RESET_STATUS_UNSUPPORTED
    return _single_probe_result(
        status=status,
        reset_supported=False,
        reset_method=None,
        attempted_method=attempted[-1] if attempted else "unsupported",
        obs_error=None,
        failure_reason="; ".join(failures),
        attempted_methods=attempted,
    )


def probe_reset_capability_many(
    env: Any,
    dataset_observations: list[Any],
    reconstruction_tolerance: float = DEFAULT_RECONSTRUCTION_TOLERANCE,
) -> tuple[dict[str, Any], pd.DataFrame]:
    rows = []
    for idx, state in enumerate(dataset_observations):
        result = probe_reset_capability(env, state, reconstruction_tolerance)
        rows.append(
            {
                "probe_index": int(idx),
                "env_available": bool(result["env_available"]),
                "reset_probe_status": result["reset_probe_status"],
                "reset_supported": result["reset_supported"],
                "reset_method": result["reset_method"],
                "obs_reconstruction_error": result["obs_reconstruction_error"],
                "failure_reason": result["failure_reason"],
                "attempted_methods": " | ".join(result.get("attempted_methods", [])),
                "missing_packages": " | ".join(result.get("missing_packages", [])),
            }
        )
    examples = pd.DataFrame(rows)
    supported_rows = examples[examples["reset_supported"] == True]  # noqa: E712 - nullable object column.
    errors = pd.to_numeric(supported_rows["obs_reconstruction_error"], errors="coerce").dropna()
    method_counts = supported_rows["reset_method"].value_counts() if not supported_rows.empty else pd.Series(dtype=int)
    if examples.empty:
        status = RESET_STATUS_UNCERTAIN
        reset_supported: bool | None = None
        reset_method = None
        failure_reason = "no probe states"
    elif bool((examples["reset_probe_status"] == RESET_STATUS_SUPPORTED).all()) and method_counts.shape[0] == 1:
        status = RESET_STATUS_SUPPORTED
        reset_supported = True
        reset_method = str(method_counts.index[0])
        failure_reason = ""
    elif bool((examples["reset_probe_status"] == RESET_STATUS_SUPPORTED).all()):
        status = RESET_STATUS_UNCERTAIN
        reset_supported = False
        reset_method = None
        failure_reason = "inconsistent_reset_methods_across_probe_states"
    else:
        status_values = set(str(x) for x in examples["reset_probe_status"].dropna().tolist())
        status = RESET_STATUS_UNCERTAIN if RESET_STATUS_UNCERTAIN in status_values else RESET_STATUS_UNSUPPORTED
        reset_supported = False
        reset_method = None
        failed = examples[examples["reset_probe_status"] != RESET_STATUS_SUPPORTED]
        failure_reason = str(failed.iloc[0]["failure_reason"]) if not failed.empty else "reset probe failed"
    summary = {
        "env_available": True,
        "env_construction_error": None,
        "reset_probe_status": status,
        "reset_supported": reset_supported,
        "reset_method": reset_method,
        "obs_reconstruction_error_mean": float(errors.mean()) if not errors.empty else np.nan,
        "obs_reconstruction_error_max": float(errors.max()) if not errors.empty else np.nan,
        "num_probe_states": int(examples.shape[0]),
        "missing_packages": [],
        "failure_reason": failure_reason,
    }
    return summary, examples


def reset_env_to_state(
    env: Any,
    dataset_observation: Any,
    reconstruction_tolerance: float = DEFAULT_RECONSTRUCTION_TOLERANCE,
) -> dict[str, Any]:
    result = probe_reset_capability(env, dataset_observation, reconstruction_tolerance)
    if result["reset_probe_status"] != RESET_STATUS_SUPPORTED:
        raise RuntimeError(f"Environment does not support reliable reset-to-state: {result['failure_reason']}")
    return result
