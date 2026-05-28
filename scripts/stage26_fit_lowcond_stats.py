from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from bars.conditioning import LowCondStats
from bars.external.gas_artifacts import resolve_gas_artifacts
from bars.external.gas_backbone import GASBackbone
from stage26_lowcond_common import antmaze_xy_factors, parse_ints, sample_future_pairs, save_stats


def _load_gas(env_name: str, gas_seed: int, gpu: str, artifact_root: str) -> GASBackbone:
    artifacts = resolve_gas_artifacts(env_name, gas_seed, artifact_root)
    gas = GASBackbone(
        env_name=env_name,
        seed=gas_seed,
        artifact_root=Path(artifact_root),
        gas_repo_path=Path("external_src/GAS"),
        gpu=gpu,
        artifacts=artifacts,
    )
    gas.load_policy(artifacts.policy_checkpoint)
    return gas


def _batched_phi(gas: GASBackbone, obs: np.ndarray, batch_size: int) -> np.ndarray:
    parts = []
    for st in range(0, len(obs), batch_size):
        parts.append(np.asarray(gas.get_phi(obs[st : st + batch_size]), dtype=np.float32))
    return np.concatenate(parts, axis=0)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fit Stage-26 low-level condition statistics.")
    parser.add_argument("--env", required=True)
    parser.add_argument("--gas-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--artifact-root", default="artifacts/gas_ogbench_offline_full_20260522_165138")
    parser.add_argument("--num-samples", type=int, default=200000)
    parser.add_argument("--horizons", default="1,2,4,8,16,32,64,128")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    os.environ["CUDA_VISIBLE_DEVICES"] = "" if str(args.gpu).lower() in {"cpu", "-1", ""} else str(args.gpu)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    gas = _load_gas(args.env, args.gas_seed, args.gpu, args.artifact_root)
    _, train_dataset, _ = gas.load_env_and_dataset()
    observations = np.asarray(train_dataset["observations"], dtype=np.float32)
    terminals = np.asarray(train_dataset.get("terminals", np.zeros((len(observations),), dtype=np.float32)))
    horizons = parse_ints(args.horizons)
    src, dst, hs = sample_future_pairs(terminals, horizons, args.num_samples, args.seed)
    obs = observations[src]
    future = observations[dst]
    phi_obs = _batched_phi(gas, obs, args.batch_size)
    phi_future = _batched_phi(gas, future, args.batch_size)
    z = np.concatenate([phi_obs, phi_future], axis=0)
    z_mean = z.mean(axis=0).astype(np.float32)
    z_std = np.maximum(z.std(axis=0), 1e-6).astype(np.float32)
    norm_diff = (phi_future - z_mean) / z_std - (phi_obs - z_mean) / z_std
    dist = np.linalg.norm(norm_diff, axis=-1)
    q90 = float(np.quantile(dist, 0.90))
    residual = antmaze_xy_factors(future) - antmaze_xy_factors(obs)
    factor_mean = residual.mean(axis=0).astype(np.float32)
    factor_std = np.maximum(residual.std(axis=0), 1e-6).astype(np.float32)
    stats = LowCondStats(
        z_mean=z_mean,
        z_std=z_std,
        tdr_distance_q90=q90,
        factor_mean=factor_mean,
        factor_std=factor_std,
        factor_dim_max=2,
    )
    metadata = {
        "env": args.env,
        "gas_seed": args.gas_seed,
        "seed": args.seed,
        "num_samples": int(len(src)),
        "horizons": horizons,
        "horizon_counts": {str(h): int((hs == h).sum()) for h in horizons},
        "z_dim": int(z.shape[1]),
        "factor_dim": 2,
        "tdr_distance_q90": q90,
    }
    save_stats(args.output, stats, metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
