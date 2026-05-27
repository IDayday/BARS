#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def latest_file(root: Path, pattern: str) -> Path | None:
    paths = [p for p in root.glob(pattern) if p.is_file()]
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def last_csv_row(path: Path) -> dict[str, str] | None:
    try:
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return None
    return rows[-1] if rows else None


def score_from_eval(path: Path) -> str:
    row = last_csv_row(path)
    if not row:
        return ""
    for key in ("eval/overall_episode.success", "score", "success", "return", "eval_score"):
        raw = row.get(key)
        if raw in (None, ""):
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if abs(value) <= 1.000001:
            value *= 100.0
        return f"{value:.1f}"
    return ""


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def copy_if_exists(src: Path | None, dst: Path) -> str:
    if src is None or not src.exists():
        return ""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst.relative_to(REPO_ROOT))


def main() -> int:
    p = argparse.ArgumentParser(description="Build a lightweight git-trackable Round006 migration snapshot.")
    p.add_argument("--round", default="006")
    p.add_argument("--out-root", default="artifacts/gas_selftrain_round006")
    p.add_argument("--run-root", default="runs_round006_gas_dynamic")
    p.add_argument("--snapshot-root", default="reports/round_006_migration_snapshot")
    args = p.parse_args()

    out_root = REPO_ROOT / args.out_root
    run_root = REPO_ROOT / args.run_root
    snapshot_root = REPO_ROOT / args.snapshot_root
    if snapshot_root.exists():
        shutil.rmtree(snapshot_root)
    (snapshot_root / "evals").mkdir(parents=True, exist_ok=True)
    (snapshot_root / "status").mkdir(parents=True, exist_ok=True)
    (snapshot_root / "tables").mkdir(parents=True, exist_ok=True)

    jobs_path = REPO_ROOT / "reports" / f"round_{args.round}_gas_dynamic_jobs.tsv"
    rows = read_tsv(jobs_path)
    completed: list[dict[str, str]] = []
    status_counts: Counter[str] = Counter()
    job_table_status_counts: Counter[str] = Counter(row.get("status", "") or "unknown" for row in rows)
    status_json_counts: Counter[str] = Counter()
    status_json_manifest: list[dict[str, str]] = []
    status_snapshot_rel: dict[tuple[str, str], str] = {}

    for row in rows:
        env = row["env"]
        seed = row["seed"]
        status_json = run_root / env / f"seed{seed}" / "status.json"
        status_rel = copy_if_exists(
            status_json if status_json.exists() else None,
            snapshot_root / "status" / env / f"seed{seed}" / "status.json",
        )
        snapshot_status_json = ""
        run_status = ""
        completed_at = ""
        phase = ""
        error = ""
        if status_rel:
            snapshot_status_json = str(Path(status_rel).relative_to(args.snapshot_root))
            status_snapshot_rel[(env, seed)] = snapshot_status_json
            try:
                status_data = json.loads(status_json.read_text(encoding="utf-8"))
                run_status = status_data.get("status", "")
                completed_at = status_data.get("completed_at", "")
                phase = status_data.get("phase", "")
                error = str(status_data.get("error", ""))
            except Exception:
                run_status = "unreadable_status"
        status_json_counts[run_status or "missing"] += 1
        status_json_manifest.append(
            {
                "env": env,
                "seed": seed,
                "job_table_status": row.get("status", ""),
                "job_table_score_pp": row.get("score_pp", ""),
                "source_eval_csv": row.get("eval_csv", ""),
                "source_status_json": str(status_json.relative_to(REPO_ROOT)) if status_json.exists() else "",
                "snapshot_status_json": snapshot_status_json,
                "status_json_status": run_status,
                "completed_at": completed_at,
                "phase": phase,
                "error": error[:200],
            }
        )

    for row in rows:
        env = row["env"]
        seed = row["seed"]
        out_seed = out_root / env / f"seed{seed}"
        run_seed = run_root / env / f"seed{seed}"
        eval_csv = latest_file(out_seed / "policy", "**/eval.csv")
        if eval_csv is None:
            continue
        flags_json = latest_file(eval_csv.parent, "flags.json")
        status_json = run_seed / "status.json"

        dst_base = snapshot_root / "evals" / env / f"seed{seed}"
        eval_rel = copy_if_exists(eval_csv, dst_base / "eval.csv")
        flags_rel = copy_if_exists(flags_json, dst_base / "flags.json")
        snapshot_status_json = status_snapshot_rel.get((env, seed), "")
        status = ""
        if status_json.exists():
            try:
                status = json.loads(status_json.read_text(encoding="utf-8")).get("status", "")
            except Exception:
                status = "unreadable_status"
        status_counts[status or "unknown"] += 1
        completed.append(
            {
                "env": env,
                "seed": seed,
                "score_pp": score_from_eval(eval_csv),
                "source_eval_csv": str(eval_csv.relative_to(REPO_ROOT)),
                "snapshot_eval_csv": str(Path(eval_rel).relative_to(args.snapshot_root)),
                "source_flags_json": "" if flags_json is None else str(flags_json.relative_to(REPO_ROOT)),
                "snapshot_flags_json": "" if not flags_rel else str(Path(flags_rel).relative_to(args.snapshot_root)),
                "snapshot_status_json": snapshot_status_json,
                "status": status,
            }
        )

    completed.sort(key=lambda r: (r["env"], int(r["seed"])))
    manifest = snapshot_root / "completed_eval_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "env",
            "seed",
            "score_pp",
            "source_eval_csv",
            "snapshot_eval_csv",
            "source_flags_json",
            "snapshot_flags_json",
            "snapshot_status_json",
            "status",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(completed)

    status_manifest = snapshot_root / "job_status_manifest.csv"
    with status_manifest.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "env",
            "seed",
            "job_table_status",
            "job_table_score_pp",
            "source_eval_csv",
            "source_status_json",
            "snapshot_status_json",
            "status_json_status",
            "completed_at",
            "phase",
            "error",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(status_json_manifest)

    for rel_path in [
        f"reports/round_{args.round}_gas_dynamic_jobs.tsv",
        f"reports/round_{args.round}_ogbench_download_status.tsv",
        f"reports/round_{args.round}_gas_dynamic_monitor_latest.md",
        f"reports/round_{args.round}_completed_eval_summary.csv",
        f"reports/round_{args.round}_completed_eval_summary.md",
        f"reports/round_{args.round}_gas_config_alignment_audit.md",
        f"reports/round_{args.round}_no_intermediate_resume_restart.md",
        f"reports/round_{args.round}_training_interruption_diagnosis.md",
        f"commands/round_{args.round}_gas_dynamic_active_command.txt",
        f"commands/round_{args.round}_gas_dynamic_launch.sh",
    ]:
        src = REPO_ROOT / rel_path
        if src.exists():
            copy_if_exists(src, snapshot_root / "tables" / src.name)

    total = len(rows)
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "round": args.round,
        "total_jobs": total,
        "completed_eval": len(completed),
        "remaining_jobs": total - len(completed),
        "source_out_root": args.out_root,
        "source_run_root": args.run_root,
        "weights_included": False,
        "weights_excluded_patterns": [
            "params_*.pkl",
            "*.pkl",
            "*.npz",
            "*.npy",
            "*.pt",
            "*.pth",
            "*.ckpt",
            "checkpoints/",
        ],
        "job_table_status_counts": dict(sorted(job_table_status_counts.items())),
        "status_json_count": sum(1 for row in status_json_manifest if row["snapshot_status_json"]),
        "status_json_status_counts": dict(sorted(status_json_counts.items())),
        "status_counts_among_completed": dict(sorted(status_counts.items())),
    }
    (snapshot_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    readme = f"""# Round {args.round} Migration Snapshot

Generated: `{summary['generated_at']}`

This snapshot is intentionally lightweight and git-trackable. It includes completed `eval.csv`
files, selected `flags.json` files, job tables, and reports. It does not include checkpoint
weights, videos, TensorBoard files, or dataset files.

## Current Snapshot

- Total jobs: `{total}`
- Completed eval rows: `{len(completed)}`
- Remaining jobs to run on the next server: `{total - len(completed)}`
- Status JSON files included: `{summary['status_json_count']}`
- Weights included: `false`

## Included Files

- Completed `eval.csv` files and associated lightweight `flags.json` files.
- `completed_eval_manifest.csv` for completed eval rows.
- `job_status_manifest.csv` plus available `status.json` files for current Round006 job state.
- Round006 job tables, config audit, interruption/restart notes, and monitor snapshot.

## Excluded Files

- Checkpoint weights such as `params_*.pkl`.
- Graph/checkpoint pickle files.
- Dataset, tensorboard, video, and raw training output files.

## Restore On A New Server

After cloning the repository and preparing datasets/dependencies, restore completion markers:

```bash
python scripts/round006_restore_migration_snapshot.py \\
  --snapshot-root reports/round_006_migration_snapshot \\
  --out-root artifacts/gas_selftrain_round006 \\
  --run-root runs_round006_gas_dynamic
```

Then start Round006 training with the normal launcher:

```bash
ROUND=006 SEEDS=42,43,44,45,46 GPUS=0,1,2,3,4,5 \\
ROUND006_GPU_SLOTS_PER_GPU=2 POLL_SECONDS=60 DOWNLOAD_POLL_SECONDS=30 \\
bash scripts/round006_launch_gas_dynamic.sh
```

The restored `eval.csv` markers make the orchestrator skip already completed env/seed pairs.
Interrupted jobs are not marked completed and should be rerun from scratch for comparability.
"""
    (snapshot_root / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
