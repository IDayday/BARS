#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


PLANNER_VARIANTS = ["shortest", "reachability", "full_bars"]
DIRECT_GOAL_VARIANTS = ["direct_goal"]

CONDITIONS: List[Dict[str, object]] = [
    {
        "name": "original",
        "variants": PLANNER_VARIANTS,
        "set": {
            "eval.max_steps": 700,
            "eval.subgoal_horizon": 30,
            "eval.subgoal_threshold": 0.5,
            "eval.fallback_mode": "none",
        },
    },
    {
        "name": "horizon_threshold_only",
        "variants": PLANNER_VARIANTS,
        "set": {
            "eval.max_steps": 700,
            "eval.subgoal_horizon": 50,
            "eval.subgoal_threshold": 1.0,
            "eval.fallback_mode": "none",
        },
    },
    {
        "name": "time_only",
        "variants": PLANNER_VARIANTS,
        "set": {
            "eval.max_steps": 1000,
            "eval.subgoal_horizon": 30,
            "eval.subgoal_threshold": 0.5,
            "eval.fallback_mode": "none",
        },
    },
    {
        "name": "tuned_no_fallback",
        "variants": PLANNER_VARIANTS,
        "set": {
            "eval.max_steps": 1000,
            "eval.subgoal_horizon": 50,
            "eval.subgoal_threshold": 1.0,
            "eval.fallback_mode": "none",
        },
    },
    {
        "name": "planner_fallback_only",
        "variants": PLANNER_VARIANTS,
        "set": {
            "eval.max_steps": 1000,
            "eval.subgoal_horizon": 50,
            "eval.subgoal_threshold": 1.0,
            "eval.fallback_mode": "planner_only",
        },
    },
    {
        "name": "direct_goal_fallback",
        "variants": PLANNER_VARIANTS,
        "set": {
            "eval.max_steps": 1000,
            "eval.subgoal_horizon": 50,
            "eval.subgoal_threshold": 1.0,
            "eval.fallback_mode": "direct_goal",
        },
    },
    {
        "name": "direct_goal_only",
        "variants": DIRECT_GOAL_VARIANTS,
        "set": {
            "eval.max_steps": 1000,
            "eval.subgoal_horizon": 50,
            "eval.subgoal_threshold": 1.0,
            "eval.fallback_mode": "none",
        },
    },
]


def make_task(env_name: str, seed: int, variant: str, condition: str, cond_set: Dict[str, object]) -> Dict[str, object]:
    shared = {
        "experiment.condition": condition,
        "experiment.warmstart_root": "runs_stage16_full12",
        "experiment.warmstart_source_variant": "full_bars",
        "experiment.warmstart_artifacts": ["tdr", "policy", "reachability", "embeddings", "graph", "boundary"],
        "eval.enabled": True,
        "eval.episodes": 50,
        "eval.success_threshold": 0.5,
        "eval.fallback_variants": ["reachability", "shortest"],
        "diagnostics.enabled": False,
        "diagnostics.edge_rollout_enabled": False,
        "planner.lambda_risk": 1.0,
        "planner.lambda_boundary": 0.1,
        "boundary.enabled": True,
        "boundary.method": "support_modes",
    }
    merged = dict(shared)
    merged.update(cond_set)
    return {
        "env": env_name,
        "seed": seed,
        "variant": variant,
        "node_method": "bars",
        "set": merged,
    }


def build_sweep(base_config: str, envs: List[str], seeds: List[int]) -> Dict[str, object]:
    tasks: List[Dict[str, object]] = []
    for cond in CONDITIONS:
        name = str(cond["name"])
        variants = [str(v) for v in cond["variants"]]
        cond_set = dict(cond["set"])
        for env_name in envs:
            for seed in seeds:
                for variant in variants:
                    tasks.append(make_task(env_name, seed, variant, name, cond_set))
    return {
        "base_config": base_config,
        "resources": {"default_mem_mb": 5000},
        "tasks": tasks,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", default="../d4rl_antmaze_stage16.json")
    ap.add_argument("--out", default="configs/sweeps/d4rl_stage18_protocol_ablation_medium50.json")
    ap.add_argument("--episodes", type=int, default=50, help="Override eval.episodes for all tasks.")
    args = ap.parse_args()

    envs = ["antmaze-medium-play-v2", "antmaze-medium-diverse-v2"]
    seeds = [0, 1, 2]
    sweep = build_sweep(args.base_config, envs, seeds)
    if args.episodes != 50:
        for task in sweep["tasks"]:
            task["set"]["eval.episodes"] = int(args.episodes)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(sweep, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(out_path)
    print(f"tasks={len(sweep['tasks'])}")


if __name__ == "__main__":
    main()
