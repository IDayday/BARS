#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bars.external.gas_artifacts import resolve_gas_artifacts
from bars.external.gas_backbone import GASBackbone
from bars.tmd_test.repr_provider import TMDRepresentationProvider
from bars.tmd_test.tmd_agent_adapter import TMDAgentAdapter, _add_tmd_paths


def _get_any(dataset: dict, *names: str):
    for name in names:
        if name in dataset:
            return dataset[name]
    return None


def _aligned(dataset: dict, key: str, n: int):
    if key not in dataset:
        return None
    arr = np.asarray(dataset[key])
    if arr.ndim > 0 and arr.shape[0] >= n:
        return arr[:n]
    return None


def _infer_traj_ids(n: int, terminals=None, timeouts=None) -> tuple[np.ndarray, np.ndarray]:
    traj_ids = np.zeros((n,), dtype=np.int64)
    time_idxs = np.zeros((n,), dtype=np.int64)
    boundaries = np.zeros((n,), dtype=bool)
    if terminals is not None:
        boundaries |= np.asarray(terminals[:n]).astype(bool).reshape(-1)
    if timeouts is not None:
        boundaries |= np.asarray(timeouts[:n]).astype(bool).reshape(-1)
    tid = 0
    t = 0
    for i in range(n):
        traj_ids[i] = tid
        time_idxs[i] = t
        if boundaries[i] and i < n - 1:
            tid += 1
            t = 0
        else:
            t += 1
    return traj_ids, time_idxs


def _batched_encode(fn, x: np.ndarray, batch_size: int) -> np.ndarray:
    chunks = []
    for st in range(0, len(x), int(batch_size)):
        chunks.append(np.asarray(fn(x[st : st + int(batch_size)]), dtype=np.float32))
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 0), dtype=np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export BARS/GAS OGBench data to the Stage27 flat .npz format")
    p.add_argument("--env", required=True)
    p.add_argument("--gas-seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0, help="Optional max observations; 0 means all")
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--gas-policy-path")
    p.add_argument("--gas-artifact-root", default="artifacts/gas_ogbench_offline_full_20260522_165138")
    p.add_argument("--gas-repo", default="external_src/GAS")
    p.add_argument("--gas-gpu", default="cpu")
    p.add_argument("--dataset-root", default="/mnt/project/offlinerl_datasets/ogbench")
    p.add_argument("--tmd-checkpoint", default="")
    p.add_argument("--tmd-repo", default="external_src/tmd-release")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    start = time.time()
    if str(args.gas_gpu).lower() in {"", "cpu", "-1"}:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gas_gpu)
    artifacts = resolve_gas_artifacts(args.env, int(args.gas_seed), args.gas_artifact_root)
    policy_path = Path(args.gas_policy_path) if args.gas_policy_path else artifacts.policy_checkpoint
    if policy_path is None:
        raise FileNotFoundError(f"No GAS policy checkpoint found for {args.env} seed {args.gas_seed}")

    gas = GASBackbone(
        env_name=args.env,
        seed=int(args.gas_seed),
        artifact_root=Path(args.gas_artifact_root),
        gas_repo_path=Path(args.gas_repo),
        gpu=args.gas_gpu,
        artifacts=artifacts,
    )
    gas.load_policy(policy_path)
    _, train_dataset, _ = gas.load_env_and_dataset()

    obs = np.asarray(train_dataset["observations"], dtype=np.float32)
    n = len(obs) if int(args.limit) <= 0 else min(len(obs), int(args.limit))
    obs = obs[:n]
    tdr_emb = _batched_encode(gas.get_phi, obs, int(args.batch_size))

    next_obs = _aligned(train_dataset, "next_observations", n)
    if next_obs is not None:
        tdr_next = _batched_encode(gas.get_phi, np.asarray(next_obs, dtype=np.float32), int(args.batch_size))
        te_scores = np.linalg.norm(tdr_next - tdr_emb, axis=1).astype(np.float32)
    else:
        te_scores = np.zeros((n,), dtype=np.float32)

    terminals = _get_any(train_dataset, "terminals", "dones")
    timeouts = _get_any(train_dataset, "timeouts")
    if "traj_ids" in train_dataset and "time_idxs" in train_dataset:
        traj_ids = np.asarray(train_dataset["traj_ids"][:n])
        time_idxs = np.asarray(train_dataset["time_idxs"][:n])
    else:
        traj_ids, time_idxs = _infer_traj_ids(n, terminals=terminals, timeouts=timeouts)

    xy = np.zeros((n, 2), dtype=np.float32)
    dim = min(2, obs.shape[1])
    xy[:, :dim] = obs[:, :dim]

    arrays = {
        "states": obs.astype(np.float32),
        "observations": obs.astype(np.float32),
        "traj_ids": traj_ids,
        "time_idxs": time_idxs,
        "tdr_emb": tdr_emb.astype(np.float32),
        "xy": xy,
        "te_scores": te_scores,
    }
    for key in ["actions", "rewards", "terminals", "dones", "timeouts", "masks", "valids", "next_observations"]:
        value = _aligned(train_dataset, key, n)
        if value is not None:
            arrays[key] = value

    if args.tmd_checkpoint:
        _add_tmd_paths(args.tmd_repo)
        tmd = TMDAgentAdapter.restore(
            args.tmd_checkpoint,
            {"env_name": args.env, "seed": int(args.gas_seed), "dataset_root": args.dataset_root, "tmd_repo": args.tmd_repo},
        )
        provider = TMDRepresentationProvider(tmd, batch_size=int(args.batch_size))
        arrays["tmd_emb"] = provider.encode(obs).astype(np.float32)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    meta = {
        "env": args.env,
        "gas_seed": int(args.gas_seed),
        "policy_path": str(policy_path),
        "gas_artifact_root": args.gas_artifact_root,
        "tmd_checkpoint": args.tmd_checkpoint,
        "num_rows": int(n),
        "duration_sec": time.time() - start,
        "keys": sorted(arrays),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {out} rows={n} keys={sorted(arrays)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
