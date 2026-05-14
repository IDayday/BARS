#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

ENVS = ["antmaze-medium-play-v2", "antmaze-medium-diverse-v2"]
SEEDS = [0, 1, 2]
PLANNER_VARIANTS = ["shortest", "reachability", "full_bars"]


def _task(
    env: str,
    seed: int,
    variant: str,
    condition: str,
    episodes: int,
    max_steps: int,
    horizon: int,
    threshold: float,
    fallback_mode: str,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    set_items: Dict[str, Any] = {
        "experiment.warmstart_root": "runs_stage16_full12",
        "experiment.warmstart_source_variant": "full_bars",
        "experiment.warmstart_artifacts": ["tdr", "policy", "reachability", "embeddings", "graph", "boundary"],
        "eval.enabled": True,
        "eval.episodes": episodes,
        "eval.max_steps": max_steps,
        "eval.subgoal_horizon": horizon,
        "eval.subgoal_threshold": threshold,
        "eval.success_threshold": 0.5,
        "eval.condition": condition,
        "eval.fallback_mode": fallback_mode,
        "eval.fallback_variants": ["reachability", "shortest"],
        "diagnostics.enabled": False,
        "diagnostics.edge_rollout_enabled": False,
        "planner.lambda_risk": 1.0,
        "planner.lambda_boundary": 0.1,
        "boundary.enabled": True,
        "boundary.method": "support_modes",
    }
    if fallback_mode == "direct_goal":
        set_items["eval.fallback_variants"] = ["reachability", "shortest", "direct_goal"]
    if extra:
        set_items.update(extra)
    return {
        "run_id": f"{condition}_{env}_{variant}_seed{seed}",
        "env": env,
        "seed": seed,
        "variant": variant,
        "node_method": "bars",
        "set": set_items,
    }


def build(episodes: int, core: bool) -> Dict[str, Any]:
    if core:
        specs = [
            ("tuned_no_fallback", 1000, 50, 1.0, "none", PLANNER_VARIANTS, {}),
            ("planner_fallback_only", 1000, 50, 1.0, "planner_only", PLANNER_VARIANTS, {}),
            ("direct_goal_fallback", 1000, 50, 1.0, "direct_goal", PLANNER_VARIANTS, {}),
            ("direct_goal_after3", 1000, 50, 1.0, "direct_goal_after_k", PLANNER_VARIANTS, {"eval.direct_goal_after_k": 3}),
            ("direct_goal_only", 1000, 50, 1.0, "none", ["direct_goal"], {}),
        ]
    else:
        specs = [
            ("original", 700, 30, 0.5, "none", PLANNER_VARIANTS, {}),
            ("horizon_threshold_only", 700, 50, 1.0, "none", PLANNER_VARIANTS, {}),
            ("time_only", 1000, 30, 0.5, "none", PLANNER_VARIANTS, {}),
            ("tuned_no_fallback", 1000, 50, 1.0, "none", PLANNER_VARIANTS, {}),
            ("planner_fallback_only", 1000, 50, 1.0, "planner_only", PLANNER_VARIANTS, {}),
            ("direct_goal_fallback", 1000, 50, 1.0, "direct_goal", PLANNER_VARIANTS, {}),
            ("direct_goal_after3", 1000, 50, 1.0, "direct_goal_after_k", PLANNER_VARIANTS, {"eval.direct_goal_after_k": 3}),
            ("direct_goal_only", 1000, 50, 1.0, "none", ["direct_goal"], {}),
        ]
    tasks: List[Dict[str, Any]] = []
    for env in ENVS:
        for seed in SEEDS:
            for condition, max_steps, horizon, threshold, fallback_mode, variants, extra in specs:
                for variant in variants:
                    tasks.append(_task(env, seed, variant, condition, episodes, max_steps, horizon, threshold, fallback_mode, extra))
    return {"base_config": "../d4rl_antmaze_stage16.json", "resources": {"default_mem_mb": 5000}, "tasks": tasks}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="configs/sweeps")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = [
        ("d4rl_stage19_protocol_ablation_medium50.json", build(50, core=False)),
        ("d4rl_stage19_protocol_ablation_medium100_core.json", build(100, core=True)),
    ]
    for name, obj in files:
        path = out / name
        path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        print(path, len(obj["tasks"]))


if __name__ == "__main__":
    main()
