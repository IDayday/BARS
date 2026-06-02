#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shlex
import sys
from pathlib import Path
from typing import Any

from stage30_official_gas_common import ARCHIVED_PRE_STAGE30_STATUS, gas_source_identity, parse_seed_list, write_csv


TIER1_ENVS = {
    "antmaze-giant-navigate-v0",
    "antmaze-giant-stitch-v0",
    "antmaze-large-explore-v0",
    "scene-play-v0",
    "kitchen-partial-v0",
}

TIER2_ENVS = {
    "antmaze-medium-navigate-v0",
    "antmaze-medium-stitch-v0",
    "antmaze-medium-explore-v0",
    "antmaze-large-navigate-v0",
    "antmaze-large-stitch-v0",
    "antmaze-large-explore-v0",
}

TIER3_ENVS = {
    "visual-antmaze-giant-navigate-v0",
    "visual-antmaze-giant-stitch-v0",
    "visual-antmaze-large-explore-v0",
    "visual-scene-play-v0",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _classify_env(env_name: str) -> dict[str, str]:
    base = env_name.removeprefix("visual-")
    parts = base.removesuffix("-v0").split("-")
    family = parts[0] if parts else ""
    task_type = parts[-1] if parts else ""
    size = ""
    for token in ("medium", "large", "giant", "teleport"):
        if token in parts:
            size = token
            break
    if env_name in TIER1_ENVS:
        tier = "Tier1"
    elif env_name in TIER2_ENVS:
        tier = "Tier2"
    elif env_name in TIER3_ENVS:
        tier = "Tier3"
    else:
        tier = "Extra"
    dataset_type = "d4rl" if family == "kitchen" else "ogbench"
    return {
        "env_family": family,
        "env_size": size,
        "task_type": task_type,
        "dataset_type": dataset_type,
        "observation_type": "visual" if env_name.startswith("visual-") else "state",
        "target_tier": tier,
    }


def _latest_file(paths: list[Path]) -> Path | None:
    return sorted(paths)[-1] if paths else None


def _d4rl_dataset_path(dataset_root: Path, env_name: str) -> Path | None:
    d4rl_dir = dataset_root / "d4rl"
    if env_name == "kitchen-partial-v0":
        matches = sorted(d4rl_dir.glob("kitchen*.hdf5"))
        return matches[0] if matches else None
    matches = sorted(d4rl_dir.glob(f"{env_name}*.hdf5"))
    return matches[0] if matches else None


def _exists(path: Path | None) -> int:
    return int(path is not None and path.exists())


def _size(path: Path | None) -> int | str:
    if path is None or not path.exists():
        return ""
    return path.stat().st_size


def _artifact_status(row: dict[str, Any]) -> str:
    missing: list[str] = []
    if not row["train_dataset_exists"]:
        missing.append("LOCAL_TRAIN_DATASET")
    if not row["keygraph_exists"]:
        missing.append("KEYGRAPH")
    if not row["policy_exists"]:
        missing.append("POLICY")
    if not missing:
        return "READY_OFFICIAL_GAS"
    if "LOCAL_TRAIN_DATASET" in missing:
        return "MISSING_LOCAL_DATASET"
    return "MISSING_" + "_AND_".join(missing)


def _training_status(row: dict[str, Any]) -> str:
    if row["artifact_status"] == "READY_OFFICIAL_GAS":
        return "NOT_NEEDED_READY"
    if not row["train_dataset_exists"]:
        return "NOT_QUEUED_MISSING_LOCAL_DATASET"
    if row["dataset_type"] == "d4rl":
        return "PENDING_D4RL_PROTOCOL_OR_OFFICIAL_GAS_TRAINING"
    return "MISSING_CKPT_TRAINING_PENDING"


def _target_envs() -> set[str]:
    return set(TIER1_ENVS) | set(TIER2_ENVS) | set(TIER3_ENVS)


def _inventory_artifacts(artifact_root: Path, dataset_root: Path, seeds: list[int], include_all_seeds: bool) -> list[dict[str, Any]]:
    ogbench_dir = dataset_root / "ogbench"
    env_names = {p.name for p in artifact_root.iterdir() if p.is_dir()} if artifact_root.exists() else set()
    env_names |= _target_envs()
    rows: list[dict[str, Any]] = []
    for env_name in sorted(env_names):
        env_dir = artifact_root / env_name
        seed_dirs: list[Path] = []
        if env_dir.exists():
            seed_dirs = sorted(p for p in env_dir.glob("seed*") if p.is_dir())
        if include_all_seeds:
            wanted_seeds = sorted(
                {
                    *(int(p.name.removeprefix("seed")) for p in seed_dirs if p.name.removeprefix("seed").isdigit()),
                    *seeds,
                }
            )
        else:
            wanted_seeds = seeds
        for seed in wanted_seeds:
            root = env_dir / f"seed{seed}"
            keygraph = root / "graph" / "keygraph.pkl"
            policy = _latest_file(list((root / "policy").glob("params_*.pkl"))) if root.exists() else None
            tdr = _latest_file(list((root / "tdr").glob("params_*.pkl"))) if root.exists() else None
            dataset_embeddings = root / "features" / "dataset_embeddings.npy"
            eval_csv = root / "policy" / "eval.csv"
            manifest = root / "manifest.json"
            meta = _classify_env(env_name)
            if meta["dataset_type"] == "d4rl":
                train_dataset = _d4rl_dataset_path(dataset_root, env_name)
                val_dataset = None
            else:
                train_dataset = ogbench_dir / f"{env_name}.npz"
                val_dataset = ogbench_dir / f"{env_name}-val.npz"
            row: dict[str, Any] = {
                "stage": "stage31_official_gas_artifact_inventory",
                "evidence_class": "OFFICIAL_GAS_ARTIFACT_AVAILABILITY",
                "pre_stage30_results_status": ARCHIVED_PRE_STAGE30_STATUS,
                "env_name": env_name,
                "seed": seed,
                **meta,
                "artifact_seed_root": str(root),
                "seed_dir_exists": int(root.exists()),
                "keygraph_path": str(keygraph) if keygraph.exists() else "",
                "keygraph_exists": int(keygraph.exists()),
                "keygraph_size_bytes": _size(keygraph),
                "policy_path": str(policy) if policy else "",
                "policy_exists": _exists(policy),
                "policy_size_bytes": _size(policy),
                "tdr_path": str(tdr) if tdr else "",
                "tdr_exists": _exists(tdr),
                "tdr_size_bytes": _size(tdr),
                "dataset_embeddings_path": str(dataset_embeddings) if dataset_embeddings.exists() else "",
                "dataset_embeddings_exists": int(dataset_embeddings.exists()),
                "eval_csv_path": str(eval_csv) if eval_csv.exists() else "",
                "eval_csv_exists": int(eval_csv.exists()),
                "manifest_path": str(manifest) if manifest.exists() else "",
                "manifest_exists": int(manifest.exists()),
                "train_dataset_path": str(train_dataset) if train_dataset and train_dataset.exists() else "",
                "train_dataset_exists": int(train_dataset is not None and train_dataset.exists()),
                "val_dataset_path": str(val_dataset) if val_dataset and val_dataset.exists() else "",
                "val_dataset_exists": int(val_dataset is not None and val_dataset.exists()),
            }
            row["artifact_status"] = _artifact_status(row)
            row["can_evaluate_official_gas"] = int(row["artifact_status"] == "READY_OFFICIAL_GAS")
            row["training_queue_status"] = _training_status(row)
            rows.append(row)
    return rows


def _inventory_d4rl(dataset_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((dataset_root / "d4rl").glob("*.hdf5")):
        name = path.name.removesuffix(".hdf5")
        family = name.split("-")[0].split("_")[0]
        rows.append(
            {
                "stage": "stage31_local_dataset_inventory",
                "evidence_class": "LOCAL_D4RL_DATASET_AVAILABILITY",
                "dataset_name": name,
                "dataset_path": str(path),
                "dataset_size_bytes": path.stat().st_size,
                "dataset_family": family,
                "artifact_status": "LOCAL_DATA_ONLY_PROTOCOL_DEBUG_UNLESS_GATE_PASSES",
            }
        )
    return rows


def _write_report(out_dir: Path, artifact_rows: list[dict[str, Any]], d4rl_rows: list[dict[str, Any]], gas_repo: Path, command_line: str) -> None:
    ready = [r for r in artifact_rows if r["artifact_status"] == "READY_OFFICIAL_GAS"]
    pending = [r for r in artifact_rows if str(r.get("training_queue_status", "")).endswith("_PENDING")]
    by_tier: dict[str, dict[str, int]] = {}
    for row in artifact_rows:
        tier = str(row.get("target_tier", ""))
        by_tier.setdefault(tier, {"rows": 0, "ready": 0, "pending": 0})
        by_tier[tier]["rows"] += 1
        by_tier[tier]["ready"] += int(row["artifact_status"] == "READY_OFFICIAL_GAS")
        by_tier[tier]["pending"] += int(str(row.get("training_queue_status", "")).endswith("_PENDING"))
    source = gas_source_identity(gas_repo)
    lines = [
        "# Stage31 Official GAS Artifact Inventory",
        "",
        "Status: OFFICIAL_GAS_ARTIFACT_INVENTORY.",
        f"Pre-Stage30 BARS/Stage28/Stage29 evidence: {ARCHIVED_PRE_STAGE30_STATUS}.",
        "READY rows can be evaluated directly with official GAS graph/planner/policy/action outputs unchanged.",
        "Missing checkpoint rows are queued as training candidates, not interpreted as diagnosis evidence.",
        "",
        "## Source Lock",
        "",
        f"- official repo SHA: `{source.get('official_repo_sha', '')}`",
        f"- GAS repo path: `{source.get('gas_repo_path', '')}`",
        f"- command: `{command_line}`",
        "",
        "## Counts",
        "",
        f"- artifact rows: {len(artifact_rows)}",
        f"- ready official GAS rows: {len(ready)}",
        f"- missing checkpoint training pending rows: {len(pending)}",
        f"- local D4RL datasets: {len(d4rl_rows)}",
        "",
        "| tier | rows | ready | pending_training |",
        "| --- | --- | --- | --- |",
    ]
    for tier, counts in sorted(by_tier.items()):
        lines.append(f"| {tier} | {counts['rows']} | {counts['ready']} | {counts['pending']} |")
    ready_envs = sorted({str(r["env_name"]) for r in ready})
    pending_envs = sorted({str(r["env_name"]) for r in pending})
    lines.extend(["", "## Ready Envs", ""])
    lines.append(", ".join(ready_envs) if ready_envs else "None")
    lines.extend(["", "## Training Pending Envs", ""])
    lines.append(", ".join(pending_envs) if pending_envs else "None")
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- artifact inventory: `{out_dir / 'official_gas_artifact_inventory.csv'}`",
            f"- D4RL inventory: `{out_dir / 'd4rl_dataset_inventory.csv'}`",
            f"- ready matrix: `{out_dir / 'official_gas_ready_matrix.csv'}`",
            f"- missing checkpoint queue: `{out_dir / 'official_gas_missing_ckpt_training_queue.csv'}`",
        ]
    )
    (out_dir / "stage31_artifact_inventory_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory official GAS artifacts and local datasets for Stage31.")
    parser.add_argument("--artifact-root", default="artifacts/gas_ogbench_offline_full_20260522_165138")
    parser.add_argument("--dataset-root", default="/mnt/project/offlinerl_datasets")
    parser.add_argument("--gas-repo-path", default="external_src/GAS")
    parser.add_argument("--seeds", default="44,45,46")
    parser.add_argument("--include-all-seeds", type=int, default=0)
    parser.add_argument("--out-root", default="runs_stage31_official_gas/wide_artifact_inventory")
    args = parser.parse_args()

    out_dir = Path(args.out_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seed_list(args.seeds)
    artifact_rows = _inventory_artifacts(Path(args.artifact_root), Path(args.dataset_root), seeds, bool(args.include_all_seeds))
    d4rl_rows = _inventory_d4rl(Path(args.dataset_root))
    ready_rows = [r for r in artifact_rows if r["artifact_status"] == "READY_OFFICIAL_GAS"]
    pending_rows = [r for r in artifact_rows if str(r.get("training_queue_status", "")).endswith("_PENDING")]
    command_line = " ".join(shlex.quote(x) for x in [sys.executable, *sys.argv])
    write_csv(out_dir / "official_gas_artifact_inventory.csv", artifact_rows)
    write_csv(out_dir / "d4rl_dataset_inventory.csv", d4rl_rows)
    write_csv(out_dir / "official_gas_ready_matrix.csv", ready_rows)
    write_csv(out_dir / "official_gas_missing_ckpt_training_queue.csv", pending_rows)
    write_csv(
        out_dir / "inventory_command.csv",
        [
            {
                "stage": "stage31_official_gas_artifact_inventory",
                "evidence_class": "OFFICIAL_GAS_ARTIFACT_AVAILABILITY",
                "artifact_root": args.artifact_root,
                "dataset_root": args.dataset_root,
                "gas_repo_path": args.gas_repo_path,
                "command": command_line,
            }
        ],
    )
    _write_report(out_dir, artifact_rows, d4rl_rows, Path(args.gas_repo_path), command_line)
    print(out_dir / "stage31_artifact_inventory_report.md")


if __name__ == "__main__":
    main()
