#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any

import numpy as np

from stage30_official_gas_common import (
    configure_official_env,
    file_sha256,
    gas_config_overrides,
    load_gas_protocol_registry,
    parse_seed_list,
    write_csv,
)
from stage30_official_gas_instrument import _import_official_gas


def _latest_param(root: Path, epoch: int | str | None) -> Path | None:
    if epoch:
        expected = root / f"params_{epoch}.pkl"
        if expected.exists():
            return expected
    matches = sorted(root.glob("params_*.pkl"))
    return matches[-1] if matches else None


def _safe_hash(path: Path | None) -> str:
    return file_sha256(path) if path is not None and path.exists() else ""


def _dataset_embeddings_path(seed_root: Path) -> Path:
    return seed_root / "features" / "dataset_embeddings.npy"


def _generate_dataset_embeddings(
    *,
    env_name: str,
    seed: int,
    seed_root: Path,
    tdr_path: Path,
    gas_repo: Path,
    batch_size: int,
) -> Path:
    mods = _import_official_gas(gas_repo)
    config = mods["get_config"]()
    for key, value in gas_config_overrides(env_name).items():
        config[key] = value
    if env_name == "kitchen-partial-v0":
        _, train_dataset = mods["d4rl_make_env_and_dataset"](env_name, seed)
        train_gc_dataset = mods["GCDataset"](train_dataset, config)
    else:
        dataset_dir = os.environ.get("OGBENCH_DATASET_DIR", "/mnt/project/offlinerl_datasets/ogbench")
        _, train_dataset, _ = mods["ogbench"].make_env_and_datasets(env_name, dataset_dir=dataset_dir, compact_dataset=False)
        train_gc_dataset = mods["GCDataset"](mods["Dataset"].create(**train_dataset), config)
    example_batch = train_gc_dataset.sample(1)
    agent = mods["agents_dict"][config["agent_name"]].create(
        seed,
        example_batch["observations"],
        example_batch["actions"],
        config,
    )
    restore_epoch = tdr_path.name.split("_")[-1].split(".")[0]
    agent = mods["restore_agent"](agent, str(tdr_path.parent), restore_epoch)
    observations = train_gc_dataset.dataset["observations"]
    embeddings = np.concatenate(
        [np.asarray(agent.get_phi(observations[i : i + batch_size])) for i in range(0, observations.shape[0], batch_size)],
        axis=0,
    )
    out = _dataset_embeddings_path(seed_root)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, embeddings)
    return out


def _repair_one(args: argparse.Namespace, env_name: str, seed: int, protocol: dict[str, Any]) -> dict[str, Any]:
    seed_root = Path(args.artifact_root) / env_name / f"seed{seed}"
    tdr = _latest_param(seed_root / "tdr", protocol.get("tdr_checkpoint_epoch"))
    policy = _latest_param(seed_root / "policy", protocol.get("policy_checkpoint_epoch"))
    keygraph = seed_root / "graph" / "keygraph.pkl"
    dataset_embeddings = _dataset_embeddings_path(seed_root)
    generated_embeddings = 0
    if (
        args.generate_dataset_embeddings
        and tdr is not None
        and tdr.exists()
        and (not dataset_embeddings.exists() or not args.skip_existing_embeddings)
    ):
        dataset_embeddings = _generate_dataset_embeddings(
            env_name=env_name,
            seed=seed,
            seed_root=seed_root,
            tdr_path=tdr,
            gas_repo=Path(args.gas_repo_path),
            batch_size=int(protocol.get("batch_size", args.embedding_batch_size) or args.embedding_batch_size),
        )
        generated_embeddings = 1
    manifest = {
        "stage": "stage32_official_gas_metadata_repair",
        "source": "local_trained_gas_artifacts",
        "official_weights_used": False,
        "env_name": env_name,
        "seed": seed,
        "root": str(seed_root.resolve()),
        "complete": bool(tdr and policy and keygraph.exists()),
        "tdr_checkpoint": str(tdr.resolve()) if tdr else "",
        "tdr_checkpoint_sha256": _safe_hash(tdr),
        "policy_checkpoint": str(policy.resolve()) if policy else "",
        "policy_checkpoint_sha256": _safe_hash(policy),
        "keygraph": str(keygraph.resolve()) if keygraph.exists() else "",
        "keygraph_sha256": _safe_hash(keygraph),
        "dataset_embeddings": str(dataset_embeddings.resolve()) if dataset_embeddings.exists() else "",
        "dataset_embeddings_sha256": _safe_hash(dataset_embeddings),
        "official_protocol": protocol,
    }
    manifest_path = seed_root / "manifest.json"
    if args.write_manifest:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "env_name": env_name,
        "seed": seed,
        "seed_root": str(seed_root),
        "tdr_exists": int(tdr is not None and tdr.exists()),
        "policy_exists": int(policy is not None and policy.exists()),
        "keygraph_exists": int(keygraph.exists()),
        "dataset_embeddings_exists": int(dataset_embeddings.exists()),
        "dataset_embeddings_generated": generated_embeddings,
        "manifest_written": int(args.write_manifest),
        "manifest_path": str(manifest_path) if manifest_path.exists() else "",
        "status": "COMPLETE" if manifest["complete"] and dataset_embeddings.exists() else "PARTIAL",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair Stage32 local GAS metadata without using official weights.")
    parser.add_argument("--artifact-root", default="artifacts/gas_ogbench_offline_full_20260522_165138")
    parser.add_argument("--registry", default="configs/stage32_official_gas_protocol_registry.json")
    parser.add_argument("--gas-repo-path", default="external_src/GAS")
    parser.add_argument("--envs", required=True)
    parser.add_argument("--seeds", default="44,45,46")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--generate-dataset-embeddings", type=int, default=1)
    parser.add_argument("--skip-existing-embeddings", type=int, default=1)
    parser.add_argument("--embedding-batch-size", type=int, default=4096)
    parser.add_argument("--write-manifest", type=int, default=1)
    parser.add_argument("--out-csv", default="runs_stage32_official_gas_eval/_metadata_repair/stage32_metadata_repair_status.csv")
    args = parser.parse_args()
    os.environ.update(configure_official_env(args.gpu))
    registry = load_gas_protocol_registry(args.registry)
    envs = [x.strip() for x in args.envs.split(",") if x.strip()]
    seeds = parse_seed_list(args.seeds)
    rows: list[dict[str, Any]] = []
    for env_name in envs:
        protocol = registry.get(env_name)
        if not protocol:
            rows.append({"env_name": env_name, "status": "MISSING_PROTOCOL"})
            continue
        for seed in seeds:
            rows.append(_repair_one(args, env_name, seed, protocol))
    out_csv = Path(args.out_csv)
    write_csv(out_csv, rows)
    command_line = " ".join(shlex.quote(x) for x in [sys.executable, *sys.argv])
    report = out_csv.with_suffix(".md")
    report.write_text(
        "# Stage32 GAS Metadata Repair\n\n"
        "Local trained GAS artifacts only; official weights are not used.\n\n"
        f"Command: `{command_line}`\n\n"
        f"CSV: `{out_csv}`\n",
        encoding="utf-8",
    )
    print(out_csv)


if __name__ == "__main__":
    main()
