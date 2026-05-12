#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

from bars.common.config import apply_dotlist, load_json, save_json
from bars.experiments.pipeline import rerun_diagnostics


def _find_run_dirs(log_root: Path) -> List[Path]:
    # A run dir is any directory with config.json and cache/graph.npz.
    out = []
    for cfg_path in log_root.glob("**/config.json"):
        rd = cfg_path.parent
        if (rd / "cache" / "graph.npz").exists() and (rd / "cache" / "embeddings.npy").exists():
            out.append(rd)
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Rerun cached BARS diagnostics without retraining or rebuilding graph edges.")
    ap.add_argument("--run-dir", default=None, help="Single run directory.")
    ap.add_argument("--log-root", default=None, help="Rerun diagnostics for all cached runs under this root.")
    ap.add_argument("--set", action="append", default=[], help="Dotlist config override, e.g. diagnostics.path_min_graph_edges=1")
    ap.add_argument("--clear", action="store_true", help="Move existing logs/diagnostics.csv to .bak_<time> before rerun.")
    ap.add_argument("--rebuild-boundary", action="store_true", help="Rebuild boundary.npz using current boundary config.")
    ap.add_argument("--package", action="store_true", help="Package each run after rerun.")
    ap.add_argument("--save-config", action="store_true", help="Write modified config back to each run_dir/config.json.")
    args = ap.parse_args()

    if not args.run_dir and not args.log_root:
        raise SystemExit("Provide --run-dir or --log-root")
    run_dirs = [Path(args.run_dir)] if args.run_dir else _find_run_dirs(Path(args.log_root))
    print(f"Found {len(run_dirs)} run dirs")
    for rd in run_dirs:
        cfg_path = rd / "config.json"
        if not cfg_path.exists():
            print(f"SKIP missing config: {rd}")
            continue
        cfg = load_json(str(cfg_path))
        cfg = apply_dotlist(cfg, args.set or [])
        cfg["run_id"] = rd.name
        if args.save_config:
            save_json(cfg, str(cfg_path))
        print(f"RERUN {rd}")
        rerun_diagnostics(cfg, str(rd), clear=args.clear, rebuild_boundary=args.rebuild_boundary, package=args.package)


if __name__ == "__main__":
    main()
