#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bars.external.gas_artifacts import resolve_gas_artifacts  # noqa: E402


def _dataset_path(env_name: str, dataset_dir: str | os.PathLike[str]) -> Path:
    path = Path(dataset_dir) / f"{env_name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing OGBench dataset: {path}")
    return path


def _load_aligned_dataset_arrays(dataset_path: Path, expected_len: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(dataset_path, allow_pickle=False) as data:
        if "observations" not in data.files or "actions" not in data.files:
            raise KeyError(f"{dataset_path} must contain observations and actions")
        observations = np.asarray(data["observations"])
        actions = np.asarray(data["actions"], dtype=np.float32)
        if "terminals" in data.files:
            terminals = np.asarray(data["terminals"], dtype=bool).copy()
        elif "dones" in data.files:
            terminals = np.asarray(data["dones"], dtype=bool).copy()
        else:
            terminals = np.zeros(len(observations), dtype=bool)
            terminals[-1] = True
    raw_len = int(len(observations))
    if raw_len == expected_len:
        return {"observations": observations, "actions": actions}, {
            "mode": "as_is",
            "raw_length": raw_len,
            "aligned_length": int(expected_len),
        }
    if raw_len - int(terminals.sum()) == expected_len:
        keep = ~terminals
        return {"observations": observations[keep], "actions": actions[keep]}, {
            "mode": "drop_raw_terminal_rows",
            "raw_length": raw_len,
            "raw_terminal_count": int(terminals.sum()),
            "aligned_length": int(expected_len),
        }
    raise RuntimeError(f"Dataset length {raw_len} cannot be aligned to embedding length {expected_len}")


def _load_rows(path: Path) -> pd.DataFrame:
    if path.is_dir():
        path = path / "offline_contract_pairs.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _configure_gas_imports(gpu: int) -> None:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    gas_path = REPO_ROOT / "external_src" / "GAS"
    tmd_path = REPO_ROOT / "external_src" / "tmd-release"
    sys.path.insert(0, str(gas_path))
    sys.path.insert(0, str(tmd_path))


def _load_agent(*, env_name: str, seed: int, policy_path: Path, observations: np.ndarray, actions: np.ndarray, args: argparse.Namespace):
    from M_utils.agents import agents_dict
    from M_utils.agents.gas import get_config
    from M_utils.flax_utils import restore_agent

    config = get_config()
    config.encoder = args.encoder
    config.discount = float(args.discount)
    config.tdr_expectile = float(args.tdr_expectile)
    config.alpha = float(args.alpha)
    config.batch_size = int(args.batch_size)
    config.p_aug = float(args.p_aug)
    config.way_steps = int(args.way_steps)

    agent_class = agents_dict[config["agent_name"]]
    example_obs = observations[:1]
    example_actions = actions[:1]
    agent = agent_class.create(int(seed), example_obs, example_actions, config)
    restore_dir = policy_path.parent
    restore_epoch = policy_path.name.split("_")[-1].split(".")[0]
    agent = restore_agent(agent, str(restore_dir), restore_epoch)
    return agent, config


def add_actor_features(args: argparse.Namespace) -> dict[str, Any]:
    _configure_gas_imports(int(args.gpu))
    import jax

    rows = _load_rows(Path(args.contract_rows))
    if args.env:
        rows = rows[rows["env"].astype(str) == args.env].copy()
    if rows.empty:
        raise RuntimeError("No rows after env filter")
    if args.max_rows and len(rows) > args.max_rows:
        rows = rows.sample(n=int(args.max_rows), random_state=int(args.seed)).sort_index().copy()

    artifacts = resolve_gas_artifacts(args.env, int(args.seed), args.gas_artifact_root)
    if artifacts.dataset_embeddings is None:
        raise RuntimeError(f"Missing dataset embeddings for {args.env} under {args.gas_artifact_root}")
    phis = np.load(artifacts.dataset_embeddings, mmap_mode="r")
    arrays, alignment = _load_aligned_dataset_arrays(_dataset_path(args.env, args.dataset_dir), len(phis))
    observations = arrays["observations"]
    actions = arrays["actions"]
    if len(observations) != len(phis) or len(actions) != len(phis):
        raise RuntimeError("Aligned observation/action arrays do not match embedding length")

    agent, config = _load_agent(
        env_name=args.env,
        seed=int(args.seed),
        policy_path=Path(args.policy_path),
        observations=observations,
        actions=actions,
        args=args,
    )

    u_idx = pd.to_numeric(rows["u_idx"], errors="raise").to_numpy(np.int64)
    v_idx = pd.to_numeric(rows["v_idx"], errors="raise").to_numpy(np.int64)
    if u_idx.max(initial=0) >= len(phis) or v_idx.max(initial=0) >= len(phis):
        raise RuntimeError("Contract row index exceeds aligned dataset length")

    actor_action_mse = np.empty(len(rows), dtype=np.float32)
    actor_action_l2 = np.empty(len(rows), dtype=np.float32)
    actor_pred_action_norm = np.empty(len(rows), dtype=np.float32)
    dataset_action_norm = np.empty(len(rows), dtype=np.float32)
    skill_norm = np.empty(len(rows), dtype=np.float32)
    rng = jax.random.PRNGKey(int(args.seed))

    batch_size = int(args.actor_batch_size)
    for start in range(0, len(rows), batch_size):
        end = min(start + batch_size, len(rows))
        batch_u = u_idx[start:end]
        batch_v = v_idx[start:end]
        obs = observations[batch_u]
        target_actions = actions[batch_u]
        raw_skills = np.asarray(phis[batch_v], dtype=np.float32) - np.asarray(phis[batch_u], dtype=np.float32)
        norms = np.linalg.norm(raw_skills, axis=1, keepdims=True)
        skills = raw_skills / np.maximum(norms, 1e-6)
        rng, subkey = jax.random.split(rng)
        pred = np.asarray(agent.sample_actions(obs, skills, temperature=float(args.temperature), seed=subkey), dtype=np.float32)
        diff = pred - target_actions
        actor_action_mse[start:end] = np.mean(diff * diff, axis=1)
        actor_action_l2[start:end] = np.linalg.norm(diff, axis=1)
        actor_pred_action_norm[start:end] = np.linalg.norm(pred, axis=1)
        dataset_action_norm[start:end] = np.linalg.norm(target_actions, axis=1)
        skill_norm[start:end] = norms[:, 0]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    augmented = rows.copy()
    augmented["actor_action_mse"] = actor_action_mse
    augmented["actor_action_l2"] = actor_action_l2
    augmented["actor_pred_action_norm"] = actor_pred_action_norm
    augmented["dataset_action_norm"] = dataset_action_norm
    augmented["actor_skill_norm"] = skill_norm
    augmented.to_csv(out, index=False)
    try:
        augmented.to_parquet(out.with_suffix(".parquet"), index=False)
    except Exception:
        pass

    summary = {
        "offline_only": True,
        "uses_environment_rollout": False,
        "env": args.env,
        "seed": int(args.seed),
        "rows": int(len(augmented)),
        "policy_path": str(args.policy_path),
        "contract_rows": str(args.contract_rows),
        "output_csv": str(out),
        "dataset_alignment": alignment,
        "encoder": str(args.encoder),
        "discount": float(args.discount),
        "tdr_expectile": float(args.tdr_expectile),
        "alpha": float(args.alpha),
        "way_steps": int(args.way_steps),
        "actor_action_mse_mean": float(np.mean(actor_action_mse)),
        "actor_action_mse_median": float(np.median(actor_action_mse)),
        "actor_action_mse_p90": float(np.quantile(actor_action_mse, 0.90)),
        "actor_action_l2_mean": float(np.mean(actor_action_l2)),
    }
    summary_path = out.with_name(out.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Add frozen-actor action agreement features to Stage45 offline contract rows.")
    parser.add_argument("--env", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--contract-rows", required=True)
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--gas-artifact-root", required=True)
    parser.add_argument("--dataset-dir", default="/mnt/project/offlinerl_datasets/ogbench")
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--actor-batch-size", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--encoder", default="not_used")
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tdr-expectile", type=float, default=0.999)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--p-aug", type=float, default=0.0)
    parser.add_argument("--way-steps", type=int, default=8)
    args = parser.parse_args()
    add_actor_features(args)


if __name__ == "__main__":
    main()
