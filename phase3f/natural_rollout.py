from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from phase3.edge_rollout import policy_action
from phase3.models import GCBCMLP


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _load_torch_checkpoint(path: str | Path, device: str | torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_gcbc_policy(model_path: str | Path, device: str | torch.device = "cpu") -> GCBCMLP:
    checkpoint = _load_torch_checkpoint(model_path, device)
    model_config = checkpoint["model_config"]
    model = GCBCMLP(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def resolve_device(device: str | None) -> torch.device:
    if device is None or str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _reset_env(env: Any, seed: int | None = None, task_id: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    options = {"task_id": int(task_id)} if task_id is not None else None
    attempts: list[dict[str, Any]] = []
    if seed is not None and options is not None:
        attempts.append({"seed": int(seed), "options": options})
    if seed is not None:
        attempts.append({"seed": int(seed)})
    if options is not None:
        attempts.append({"options": options})
    attempts.append({})
    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            out = env.reset(**kwargs)
            if isinstance(out, tuple) and len(out) == 2:
                obs, info = out
            else:
                obs, info = out, {}
            return np.asarray(obs, dtype=np.float32), dict(info or {})
        except TypeError as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise RuntimeError("env.reset failed without raising a captured exception")


def _step_env(env: Any, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
    out = env.step(action)
    if isinstance(out, tuple) and len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return (
            np.asarray(obs, dtype=np.float32),
            float(reward),
            bool(terminated),
            bool(truncated),
            dict(info or {}),
        )
    if isinstance(out, tuple) and len(out) == 4:
        obs, reward, done, info = out
        return np.asarray(obs, dtype=np.float32), float(reward), bool(done), False, dict(info or {})
    raise RuntimeError(f"Unsupported env.step return signature: {type(out)}")


def _clip_action(env: Any, action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    space = getattr(env, "action_space", None)
    if space is None or not hasattr(space, "low") or not hasattr(space, "high"):
        return action
    return np.clip(action, np.asarray(space.low), np.asarray(space.high)).astype(np.float32)


def _goal_from_info(info: dict[str, Any]) -> np.ndarray | None:
    for key in ("goal", "desired_goal"):
        if key in info and info[key] is not None:
            goal = np.asarray(info[key], dtype=np.float32)
            if goal.size > 0:
                return goal.reshape(-1)
    return None


def _success_value(info: dict[str, Any]) -> float:
    value = info.get("success", 0.0)
    if isinstance(value, dict):
        if not value:
            return 0.0
        return float(all(bool(v) for v in value.values()))
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value)
        if arr.size == 0:
            return 0.0
        return float(bool(np.all(arr)))
    return float(bool(value))


def _action_for_mode(
    env: Any,
    policy: Any,
    obs: np.ndarray,
    goal: np.ndarray,
    remaining_h: int,
    action_mode: str,
    device: str | torch.device | None,
) -> np.ndarray:
    if action_mode == "direct_gcbc":
        return policy_action(policy, obs, goal, remaining_h=remaining_h, edge_id=None, device=device)
    if action_mode == "random":
        return np.asarray(env.action_space.sample(), dtype=np.float32)
    if action_mode == "zero":
        shape = getattr(getattr(env, "action_space", None), "shape", None)
        if shape is None:
            raise ValueError("zero action mode requires env.action_space.shape")
        return np.zeros(shape, dtype=np.float32)
    raise ValueError("action_mode must be direct_gcbc, random, or zero")


def run_natural_start_episodes(
    env: Any,
    policy: Any | None,
    *,
    dataset_name: str,
    method: str,
    num_episodes: int,
    max_steps: int,
    task_ids: list[int] | None = None,
    seed: int = 0,
    action_mode: str = "direct_gcbc",
    device: str | torch.device | None = None,
    stop_on_success: bool = True,
    trace_every: int = 1,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if action_mode == "direct_gcbc" and policy is None:
        raise ValueError("direct_gcbc action_mode requires a loaded policy")
    rng = np.random.default_rng(seed)
    episode_rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for episode_id in range(int(num_episodes)):
        task_id = None
        if task_ids:
            task_id = int(task_ids[episode_id % len(task_ids)])
        ep_seed = int(seed + episode_id)
        steps: list[dict[str, Any]] = []
        total_reward = 0.0
        final_reward = 0.0
        final_goal_l2 = float("nan")
        success = 0.0
        terminated = False
        truncated = False
        failure_reason = ""
        try:
            obs, reset_info = _reset_env(env, seed=ep_seed, task_id=task_id)
            goal = _goal_from_info(reset_info)
            if goal is None:
                raise RuntimeError("missing_goal_observation")
            initial_goal_l2 = float(np.linalg.norm(obs.reshape(-1) - goal.reshape(-1)))
            num_steps = 0
            for step in range(int(max_steps)):
                num_steps = step + 1
                action = _action_for_mode(
                    env,
                    policy,
                    obs,
                    goal,
                    remaining_h=max(1, int(max_steps) - step),
                    action_mode=action_mode,
                    device=device,
                )
                action = _clip_action(env, action)
                obs, reward, terminated, truncated, info = _step_env(env, action)
                total_reward += float(reward)
                final_reward = float(reward)
                success = max(success, _success_value(info))
                final_goal_l2 = float(np.linalg.norm(obs.reshape(-1) - goal.reshape(-1)))
                if trace_every > 0 and step % int(trace_every) == 0:
                    steps.append(
                        {
                            "t": int(step),
                            "reward": float(reward),
                            "success": float(success),
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                            "goal_l2": final_goal_l2,
                            "action_norm": float(np.linalg.norm(action.reshape(-1))),
                        }
                    )
                if (stop_on_success and success >= 1.0) or terminated or truncated:
                    break
            if success < 1.0 and not (terminated or truncated):
                failure_reason = "max_steps_without_success"
        except Exception as exc:  # noqa: PERF203 - per-episode failures should not abort the run.
            initial_goal_l2 = float("nan")
            num_steps = 0
            failure_reason = f"{type(exc).__name__}: {exc}"
        episode_rows.append(
            {
                "dataset_name": dataset_name,
                "method": method,
                "episode_id": int(episode_id),
                "seed": ep_seed,
                "task_id": task_id if task_id is not None else "",
                "num_steps": int(num_steps),
                "total_reward": float(total_reward),
                "final_reward": float(final_reward),
                "success": float(success),
                "initial_goal_l2": initial_goal_l2,
                "final_goal_l2": final_goal_l2,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "failure_reason": failure_reason,
            }
        )
        traces.append(
            {
                "dataset_name": dataset_name,
                "method": method,
                "episode_id": int(episode_id),
                "seed": ep_seed,
                "task_id": task_id,
                "success": float(success),
                "failure_reason": failure_reason,
                "steps": steps,
                "rng_marker": int(rng.integers(0, np.iinfo(np.int32).max)),
            }
        )
    return pd.DataFrame(episode_rows), traces


def write_natural_rollout_outputs(
    output_dir: str | Path,
    *,
    dataset_name: str,
    method: str,
    episodes: pd.DataFrame,
    traces: list[dict[str, Any]],
    skipped: bool = False,
    skipped_reason: str = "",
) -> pd.DataFrame:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(out / "episode_summary.csv", index=False)
    if episodes.empty:
        summary = pd.DataFrame(
            [
                {
                    "dataset_name": dataset_name,
                    "method": method,
                    "num_episodes": 0,
                    "success_rate": 0.0,
                    "mean_steps": 0.0,
                    "mean_total_reward": 0.0,
                    "mean_final_goal_l2": np.nan,
                    "skipped": bool(skipped),
                    "skipped_reason": skipped_reason,
                }
            ]
        )
    else:
        mean_final_goal_l2 = float(episodes["final_goal_l2"].mean()) if "final_goal_l2" in episodes else np.nan
        summary = pd.DataFrame(
            [
                {
                    "dataset_name": dataset_name,
                    "method": method,
                    "num_episodes": int(episodes.shape[0]),
                    "success_rate": float(episodes["success"].mean()),
                    "mean_steps": float(episodes["num_steps"].mean()),
                    "mean_total_reward": float(episodes["total_reward"].mean()),
                    "mean_final_goal_l2": mean_final_goal_l2,
                    "skipped": bool(skipped),
                    "skipped_reason": skipped_reason,
                }
            ]
        )
    summary.to_csv(out / "task_rollout_summary.csv", index=False)
    if episodes.empty:
        failure = pd.DataFrame([{"failure_reason": skipped_reason or "no_episodes", "count": 1}])
    else:
        reasons = episodes["failure_reason"].replace("", "success_or_done")
        failure = reasons.value_counts(dropna=False).rename_axis("failure_reason").reset_index(name="count")
    failure.to_csv(out / "failure_reason_summary.csv", index=False)
    with (out / "episode_traces.jsonl").open("w", encoding="utf-8") as f:
        for record in traces:
            f.write(json.dumps(_json_safe(record), sort_keys=True) + "\n")
    return summary
