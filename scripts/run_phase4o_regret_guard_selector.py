#!/usr/bin/env python
"""Select Phase 4M planner-relevant candidates with an offline regret guard."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase3e.phase4o_regret_guard import (  # noqa: E402
    RegretGuardConfig,
    run_regret_guard_selection,
    write_regret_guard_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result_root", default="results/phase4m")
    parser.add_argument("--output_dir", default="results/phase4o/regret_guard")
    parser.add_argument("--baseline_method", default="augmented_loss_support_bottleneck_s03")
    parser.add_argument("--max_final_val_ratio", type=float, default=1.01)
    parser.add_argument("--max_direct_repair_ratio", type=float, default=1.0)
    parser.add_argument("--max_planner_used_ratio", type=float, default=0.99)
    parser.add_argument("--min_policy_support_ratio", type=float, default=1.0)
    parser.add_argument("--disable_relaxed_improvement_fallback", action="store_true")
    parser.add_argument("--relaxed_max_direct_repair_ratio", type=float, default=1.0)
    parser.add_argument("--relaxed_max_planner_used_ratio", type=float, default=1.0)
    parser.add_argument("--relaxed_min_policy_support_ratio", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RegretGuardConfig(
        baseline_method=args.baseline_method,
        max_final_val_ratio=args.max_final_val_ratio,
        max_direct_repair_ratio=args.max_direct_repair_ratio,
        max_planner_used_ratio=args.max_planner_used_ratio,
        min_policy_support_ratio=args.min_policy_support_ratio,
        allow_relaxed_improvement_fallback=not args.disable_relaxed_improvement_fallback,
        relaxed_max_direct_repair_ratio=args.relaxed_max_direct_repair_ratio,
        relaxed_max_planner_used_ratio=args.relaxed_max_planner_used_ratio,
        relaxed_min_policy_support_ratio=args.relaxed_min_policy_support_ratio,
    )
    annotated, selection, payload = run_regret_guard_selection(args.result_root, config)
    write_regret_guard_outputs(args.output_dir, annotated, selection, payload)
    print(f"Wrote Phase 4O regret-guard outputs to {args.output_dir}")
    if not selection.empty:
        cols = [
            "dataset_key",
            "run_name",
            "selected_method",
            "selection_status",
            "final_val_action_mse_ratio_vs_baseline",
            "planner_used_repair_edge_mse_ratio_vs_baseline",
        ]
        present = [col for col in cols if col in selection.columns]
        print(selection[present].to_string(index=False))


if __name__ == "__main__":
    main()
