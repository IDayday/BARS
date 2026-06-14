#!/usr/bin/env python
"""Phase 3F natural-start rollout scaffold with safe env-unavailable skip."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase3f.task_eval import load_preflight_status, write_env_unavailable_skip  # noqa: E402


def _dataset_key(dataset_name: str) -> str:
    return dataset_name.replace("-v0", "").replace("-", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--method", default="support_shortest_path")
    parser.add_argument("--output_dir", default="results/phase3f")
    parser.add_argument("--preflight_root", default="results/phase3/env_preflight")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preflight = load_preflight_status(args.dataset_name, args.preflight_root)
    dataset_key = _dataset_key(args.dataset_name)
    out_dir = Path(args.output_dir) / dataset_key / args.method
    if preflight.get("status") == "env_unavailable" or not preflight.get("ogbench_env_constructed", False):
        reason = str(preflight.get("status", "env_unavailable"))
        write_env_unavailable_skip(out_dir, args.dataset_name, args.method, reason)
        print(f"[phase3F] skipped natural-start rollout: {reason} output_dir={out_dir}")
        return
    write_env_unavailable_skip(out_dir, args.dataset_name, args.method, "scaffold_only_not_run")
    print(f"[phase3F] env available, but this scaffold intentionally did not run rollout: {out_dir}")


if __name__ == "__main__":
    main()
