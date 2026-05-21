#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any


def _split(value: str) -> list[str]:
    return [x.strip() for x in str(value or "").split(",") if x.strip()]


def audit_env(env_name: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "env": env_name,
        "make_env": False,
        "reset_ok": False,
        "has_goal": False,
        "success_source_seen": False,
        "max_episode_steps": None,
        "set_state_available": False,
        "observation_goal_compatible": False,
        "raw_error": "",
    }
    try:
        import gym
        try:
            import d4rl  # noqa: F401
        except Exception:
            pass
        env = gym.make(env_name)
        row["make_env"] = True
        row["max_episode_steps"] = getattr(getattr(env, "spec", None), "max_episode_steps", None)
        reset = env.reset()
        obs = reset[0] if isinstance(reset, tuple) else reset
        row["reset_ok"] = True
        if isinstance(obs, dict):
            row["has_goal"] = "desired_goal" in obs or "goal" in obs
            row["observation_goal_compatible"] = any(k in obs for k in ["observation", "achieved_goal"]) and row["has_goal"]
        else:
            row["has_goal"] = hasattr(env, "target_goal") or hasattr(getattr(env, "unwrapped", env), "target_goal")
            row["observation_goal_compatible"] = hasattr(obs, "shape")
        row["set_state_available"] = hasattr(env, "set_state") or hasattr(getattr(env, "unwrapped", env), "set_state")
        try:
            action = env.action_space.sample()
            step = env.step(action)
            info = step[-1] if isinstance(step, tuple) else {}
            if isinstance(info, dict):
                row["success_source_seen"] = any(k in info for k in ["success", "is_success", "goal_achieved"])
        except Exception:
            pass
        try:
            env.close()
        except Exception:
            pass
    except Exception as exc:
        row["raw_error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--envs", default="antmaze-medium-play-v2,antmaze-medium-diverse-v2,antmaze-large-play-v2,antmaze-large-diverse-v2")
    p.add_argument("--gas-artifact-root", default="artifacts/gas")
    p.add_argument("--out-md", default="reports/stage25_d4rl_protocol_audit.md")
    p.add_argument("--out-json", default="reports/stage25_d4rl_protocol_audit.json")
    args = p.parse_args()
    rows = [audit_env(env) for env in _split(args.envs)]
    pass_envs = [
        r
        for r in rows
        if r["make_env"]
        and r["reset_ok"]
        and r["has_goal"]
        and r["success_source_seen"]
        and r["max_episode_steps"]
        and r["observation_goal_compatible"]
    ]
    if len(pass_envs) == len(rows) and rows:
        gate = "PASS_D4RL_PROTOCOL"
    elif any(r["make_env"] for r in rows):
        gate = "HOLD_D4RL_PROTOCOL_REPAIR"
    else:
        gate = "HOLD_D4RL_PROTOCOL_REPAIR"
    status = {
        "gate": gate,
        "envs": rows,
        "questions": {
            "start_goal_definitions": "checked_via_reset_goal_fields",
            "success_source_threshold": "checked_via_step_info_success_keys",
            "max_episode_steps": "checked_via_env_spec",
            "reset_set_state": "checked_via_reset_and_set_state_attribute",
            "observation_goal_adapter": "checked_via_observation_goal_fields",
            "normalized_vs_raw": "audit_only_no_scores_generated",
            "task_episode_stability": "not_run_large_experiment",
        },
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(status, indent=2, sort_keys=True, default=str) + "\n")
    lines = ["# Stage25 D4RL Protocol Audit", "", f"Gate: {gate}", "", "## Environment Checks", ""]
    for r in rows:
        lines.append(f"- {r['env']}: make_env={r['make_env']} reset_ok={r['reset_ok']} has_goal={r['has_goal']} success_source_seen={r['success_source_seen']} max_steps={r['max_episode_steps']} set_state={r['set_state_available']} adapter_compatible={r['observation_goal_compatible']}")
        if r["raw_error"]:
            lines.append(f"  - error: `{r['raw_error'].splitlines()[0]}`")
    lines.extend(
        [
            "",
            "## Audit Notes",
            "- This script does not launch large D4RL experiments.",
            "- Low D4RL scores remain uninterpreted until this audit passes.",
            "- Raw success/return and normalized scores must remain separated in later D4RL runs.",
        ]
    )
    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print(json.dumps({"gate": gate, "envs": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
