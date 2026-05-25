#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    p = argparse.ArgumentParser(
        description="Restore lightweight Round006 completion markers from a git-tracked migration snapshot."
    )
    p.add_argument(
        "--snapshot-root",
        default="reports/round_006_migration_snapshot",
        help="Snapshot directory containing completed_eval_manifest.csv and eval files.",
    )
    p.add_argument("--out-root", default="artifacts/gas_selftrain_round006")
    p.add_argument("--run-root", default="runs_round006_gas_dynamic")
    args = p.parse_args()

    snapshot_root = (REPO_ROOT / args.snapshot_root).resolve()
    out_root = (REPO_ROOT / args.out_root).resolve()
    run_root = (REPO_ROOT / args.run_root).resolve()
    manifest_path = snapshot_root / "completed_eval_manifest.csv"
    rows = read_manifest(manifest_path)

    restored = 0
    for row in rows:
        env = row["env"]
        seed = row["seed"]
        seed_name = f"seed{seed}"
        eval_src = snapshot_root / row["snapshot_eval_csv"]
        flags_rel = row.get("snapshot_flags_json", "")
        flags_src = snapshot_root / flags_rel if flags_rel else None

        policy_dir = out_root / env / seed_name / "policy" / "round006_migration_import"
        policy_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(eval_src, policy_dir / "eval.csv")
        if flags_src is not None and flags_src.exists():
            shutil.copy2(flags_src, policy_dir / "flags.json")

        status_dir = run_root / env / seed_name
        status_dir.mkdir(parents=True, exist_ok=True)
        status = {
            "env": env,
            "seed": int(seed),
            "status": "completed",
            "already_completed": True,
            "artifact_source": "round006_migration_snapshot_eval_only_no_weights",
            "eval_csv": rel(policy_dir / "eval.csv"),
            "score_pp": row.get("score_pp", ""),
            "restored_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "snapshot_manifest": str(manifest_path.relative_to(REPO_ROOT)),
            "note": "Lightweight completion marker only; checkpoint weights were intentionally not restored.",
        }
        (status_dir / "status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        restored += 1

    print(f"restored_completed_eval_markers {restored}")
    print(f"out_root {out_root}")
    print(f"run_root {run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
