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


def _read_table(path: Path) -> pd.DataFrame:
    if path.is_dir():
        path = path / "actor_augmented_contract_pairs.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _configure_gas_imports(device: str, gpu: int) -> None:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    gas_path = REPO_ROOT / "external_src" / "GAS"
    tmd_path = REPO_ROOT / "external_src" / "tmd-release"
    sys.path.insert(0, str(gas_path))
    sys.path.insert(0, str(tmd_path))


def _load_agent(*, seed: int, policy_path: Path, observations: np.ndarray, actions: np.ndarray, args: argparse.Namespace):
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
    agent = agent_class.create(int(seed), observations[:1], actions[:1], config)
    restore_dir = policy_path.parent
    restore_epoch = policy_path.name.split("_")[-1].split(".")[0]
    agent = restore_agent(agent, str(restore_dir), restore_epoch)
    return agent, config


def _finite_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return vals.fillna(default)


def _segment_indices(u_idx: int, v_idx: int, max_points: int) -> np.ndarray:
    if v_idx <= u_idx:
        return np.empty(0, dtype=np.int64)
    # Actions correspond to transitions from state t, so exclude v_idx itself.
    last_action_idx = v_idx - 1
    n = min(max_points, last_action_idx - u_idx + 1)
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.linspace(u_idx, last_action_idx, num=n, dtype=np.int64))


def _threshold(env_df: pd.DataFrame, column: str, q: float, fallback: float, higher_is_better: bool) -> float:
    train = env_df[env_df["split"].astype(str) == "train"]
    pos = train[(train["label_reach_base"] == 1) & (train["seq_has_segment"] == 1)]
    source = pos if len(pos) else train[train["seq_has_segment"] == 1]
    if not len(source):
        source = env_df[env_df["seq_has_segment"] == 1]
    if not len(source):
        return float(fallback)
    value = float(source[column].quantile(q))
    if not np.isfinite(value):
        return float(fallback)
    return value


def add_sequence_features(args: argparse.Namespace) -> dict[str, Any]:
    _configure_gas_imports(str(args.device), int(args.gpu))
    import jax

    rows = _read_table(Path(args.contract_rows))
    if args.env:
        rows = rows[rows["env"].astype(str) == args.env].copy()
    if rows.empty:
        raise RuntimeError("No rows after env filter")
    if args.max_rows and len(rows) > args.max_rows:
        rows = rows.sample(n=int(args.max_rows), random_state=int(args.seed)).sort_index().copy()

    if "label_reach_base" not in rows:
        rows["label_reach_base"] = pd.to_numeric(rows["label_reach"], errors="coerce").fillna(-1).astype(int)
    else:
        rows["label_reach_base"] = pd.to_numeric(rows["label_reach_base"], errors="coerce").fillna(-1).astype(int)
    if "label_weight_base" not in rows:
        rows["label_weight_base"] = _finite_numeric(rows.get("label_weight", pd.Series(1.0, index=rows.index)), 1.0)
    else:
        rows["label_weight_base"] = _finite_numeric(rows["label_weight_base"], 1.0)

    artifacts = resolve_gas_artifacts(args.env, int(args.seed), args.gas_artifact_root)
    if artifacts.dataset_embeddings is None:
        raise RuntimeError(f"Missing dataset embeddings for {args.env} under {args.gas_artifact_root}")
    phis = np.load(artifacts.dataset_embeddings, mmap_mode="r")
    arrays, alignment = _load_aligned_dataset_arrays(_dataset_path(args.env, args.dataset_dir), len(phis))
    observations = arrays["observations"]
    actions = arrays["actions"]

    agent, _ = _load_agent(
        seed=int(args.seed),
        policy_path=Path(args.policy_path),
        observations=observations,
        actions=actions,
        args=args,
    )

    n = len(rows)
    seq_has_segment = np.zeros(n, dtype=np.int32)
    seq_num_points = np.zeros(n, dtype=np.int32)
    seq_actor_action_mse_mean = np.full(n, np.nan, dtype=np.float32)
    seq_actor_action_mse_max = np.full(n, np.nan, dtype=np.float32)
    seq_actor_action_l2_mean = np.full(n, np.nan, dtype=np.float32)
    seq_start_target_dist = np.full(n, np.nan, dtype=np.float32)
    seq_final_target_dist = np.full(n, np.nan, dtype=np.float32)
    seq_progress_ratio = np.full(n, np.nan, dtype=np.float32)
    seq_mean_progress_delta = np.full(n, np.nan, dtype=np.float32)
    seq_min_progress_delta = np.full(n, np.nan, dtype=np.float32)

    u_idx = pd.to_numeric(rows["u_idx"], errors="raise").to_numpy(np.int64)
    v_idx = pd.to_numeric(rows["v_idx"], errors="raise").to_numpy(np.int64)
    same_traj = pd.to_numeric(rows["same_traj"], errors="coerce").fillna(0).to_numpy(np.int32)
    dt = pd.to_numeric(rows["dt_if_same_traj"], errors="coerce").fillna(-1).to_numpy(np.int64)
    rng = jax.random.PRNGKey(int(args.seed))

    obs_chunks: list[np.ndarray] = []
    action_chunks: list[np.ndarray] = []
    skill_chunks: list[np.ndarray] = []
    row_slices: list[tuple[int, int, int]] = []
    flat_row_ids: list[int] = []

    for row_pos in range(n):
        if same_traj[row_pos] != 1 or dt[row_pos] <= 0:
            continue
        if u_idx[row_pos] < 0 or v_idx[row_pos] >= len(phis) or v_idx[row_pos] >= len(actions):
            continue
        seg = _segment_indices(int(u_idx[row_pos]), int(v_idx[row_pos]), int(args.segment_points))
        if len(seg) == 0:
            continue
        target_phi = np.asarray(phis[v_idx[row_pos]], dtype=np.float32)
        seg_phi = np.asarray(phis[seg], dtype=np.float32)
        raw_skills = target_phi[None, :] - seg_phi
        norms = np.linalg.norm(raw_skills, axis=1, keepdims=True)
        skills = raw_skills / np.maximum(norms, 1e-6)
        start = len(flat_row_ids)
        flat_row_ids.extend([row_pos] * len(seg))
        row_slices.append((row_pos, start, len(flat_row_ids)))
        obs_chunks.append(observations[seg])
        action_chunks.append(actions[seg])
        skill_chunks.append(skills.astype(np.float32))

        dists = np.linalg.norm(np.asarray(phis[int(u_idx[row_pos]) : int(v_idx[row_pos]) + 1], dtype=np.float32) - target_phi[None, :], axis=1)
        deltas = dists[:-1] - dists[1:]
        seq_has_segment[row_pos] = 1
        seq_num_points[row_pos] = len(seg)
        seq_start_target_dist[row_pos] = float(dists[0])
        seq_final_target_dist[row_pos] = float(dists[-1])
        seq_progress_ratio[row_pos] = float((dists[0] - dists[-1]) / max(dists[0], 1e-6))
        seq_mean_progress_delta[row_pos] = float(np.mean(deltas)) if len(deltas) else 0.0
        seq_min_progress_delta[row_pos] = float(np.min(deltas)) if len(deltas) else 0.0

    if flat_row_ids:
        obs_all = np.concatenate(obs_chunks, axis=0)
        actions_all = np.concatenate(action_chunks, axis=0)
        skills_all = np.concatenate(skill_chunks, axis=0)
        pred_all = np.empty_like(actions_all, dtype=np.float32)
        for start in range(0, len(obs_all), int(args.actor_batch_size)):
            end = min(start + int(args.actor_batch_size), len(obs_all))
            rng, subkey = jax.random.split(rng)
            pred_all[start:end] = np.asarray(
                agent.sample_actions(
                    obs_all[start:end],
                    skills_all[start:end],
                    temperature=float(args.temperature),
                    seed=subkey,
                ),
                dtype=np.float32,
            )
        diff = pred_all - actions_all
        mse = np.mean(diff * diff, axis=1)
        l2 = np.linalg.norm(diff, axis=1)
        for row_pos, start, end in row_slices:
            seq_actor_action_mse_mean[row_pos] = float(np.mean(mse[start:end]))
            seq_actor_action_mse_max[row_pos] = float(np.max(mse[start:end]))
            seq_actor_action_l2_mean[row_pos] = float(np.mean(l2[start:end]))

    out = rows.copy()
    out["seq_has_segment"] = seq_has_segment
    out["seq_num_points"] = seq_num_points
    out["seq_actor_action_mse_mean"] = seq_actor_action_mse_mean
    out["seq_actor_action_mse_max"] = seq_actor_action_mse_max
    out["seq_actor_action_l2_mean"] = seq_actor_action_l2_mean
    out["seq_start_target_dist"] = seq_start_target_dist
    out["seq_final_target_dist"] = seq_final_target_dist
    out["seq_progress_ratio"] = seq_progress_ratio
    out["seq_mean_progress_delta"] = seq_mean_progress_delta
    out["seq_min_progress_delta"] = seq_min_progress_delta

    mse_threshold = _threshold(out, "seq_actor_action_mse_mean", float(args.positive_mse_quantile), fallback=float("inf"), higher_is_better=False)
    progress_threshold = _threshold(
        out,
        "seq_mean_progress_delta",
        float(args.positive_progress_quantile),
        fallback=-float("inf"),
        higher_is_better=True,
    )
    base_label = out["label_reach_base"].to_numpy(np.int64)
    mse_vals = _finite_numeric(out["seq_actor_action_mse_mean"], float("inf")).to_numpy(np.float32)
    progress_vals = _finite_numeric(out["seq_mean_progress_delta"], -float("inf")).to_numpy(np.float32)
    has_segment = out["seq_has_segment"].to_numpy(np.int32) == 1

    demote = (base_label == 1) & ((~has_segment) | (mse_vals > mse_threshold) | (progress_vals < progress_threshold))
    seq_label = base_label.copy()
    if bool(args.demote_disagreeing_positives):
        seq_label[demote] = 0

    temp = float(args.mse_temperature)
    if temp <= 0:
        pos_mse = mse_vals[(base_label == 1) & has_segment & np.isfinite(mse_vals)]
        temp = float(np.median(pos_mse)) if len(pos_mse) else 1.0
        temp = max(temp, 1e-6)
    agree_weight = np.exp(-np.clip(mse_vals, 0.0, 1e6) / temp).astype(np.float32)
    label_weight = out["label_weight_base"].to_numpy(np.float32).copy()
    positive_mask = seq_label == 1
    label_weight[positive_mask] *= np.clip(agree_weight[positive_mask], float(args.min_positive_weight), 1.0)
    if bool(args.upweight_sequence_hard_negatives):
        hard_neg = (base_label == 0) & has_segment & (mse_vals <= mse_threshold)
        label_weight[hard_neg] *= float(args.hard_negative_weight)
    else:
        hard_neg = np.zeros(n, dtype=bool)

    reason = np.full(n, "base_label_kept", dtype=object)
    reason[demote] = "demoted_positive_sequence_contract_fail"
    reason[hard_neg] = "sequence_actor_hard_negative"
    reason[base_label < 0] = "unlabeled_kept"
    out["seq_actor_conditioned_label_reach"] = seq_label
    out["seq_actor_conditioned_label_weight"] = label_weight
    out["seq_actor_agree_weight"] = agree_weight
    out["seq_actor_mse_threshold"] = float(mse_threshold)
    out["seq_progress_threshold"] = float(progress_threshold)
    out["seq_actor_demoted_positive"] = demote.astype(np.int32)
    out["seq_actor_label_reason"] = reason
    out["label_reach"] = out["seq_actor_conditioned_label_reach"]
    out["label_weight"] = out["seq_actor_conditioned_label_weight"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    try:
        out.to_parquet(out_path.with_suffix(".parquet"), index=False)
    except Exception:
        pass

    labeled_base = base_label >= 0
    labeled_seq = seq_label >= 0
    summary = {
        "offline_only": True,
        "uses_environment_rollout": False,
        "device": str(args.device),
        "env": args.env,
        "seed": int(args.seed),
        "rows": int(len(out)),
        "rows_with_sequence": int(seq_has_segment.sum()),
        "policy_path": str(args.policy_path),
        "contract_rows": str(args.contract_rows),
        "output_csv": str(out_path),
        "dataset_alignment": alignment,
        "segment_points": int(args.segment_points),
        "mse_threshold": float(mse_threshold),
        "progress_threshold": float(progress_threshold),
        "mse_temperature": float(temp),
        "base_labeled_rows": int(labeled_base.sum()),
        "sequence_labeled_rows": int(labeled_seq.sum()),
        "base_positive_rate": float((base_label[labeled_base] == 1).mean()) if labeled_base.any() else 0.0,
        "sequence_positive_rate": float((seq_label[labeled_seq] == 1).mean()) if labeled_seq.any() else 0.0,
        "num_demoted_positives": int(demote.sum()),
        "demoted_positive_rate_among_base_positives": float(demote.sum() / max(int((base_label == 1).sum()), 1)),
        "num_sequence_hard_negatives": int(hard_neg.sum()),
        "seq_actor_action_mse_mean": float(np.nanmean(seq_actor_action_mse_mean)) if np.isfinite(seq_actor_action_mse_mean).any() else 0.0,
        "seq_actor_action_mse_median": float(np.nanmedian(seq_actor_action_mse_mean)) if np.isfinite(seq_actor_action_mse_mean).any() else 0.0,
        "seq_mean_progress_delta_mean": float(np.nanmean(seq_mean_progress_delta)) if np.isfinite(seq_mean_progress_delta).any() else 0.0,
        "encoder": str(args.encoder),
        "discount": float(args.discount),
        "tdr_expectile": float(args.tdr_expectile),
        "alpha": float(args.alpha),
        "way_steps": int(args.way_steps),
    }
    summary_path = out_path.with_name(out_path.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Add sequence-level frozen-actor agreement and progress features to "
            "offline contract rows, then build stricter sequence-conditioned labels."
        )
    )
    parser.add_argument("--env", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--contract-rows", required=True)
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--gas-artifact-root", required=True)
    parser.add_argument("--dataset-dir", default="/mnt/project/offlinerl_datasets/ogbench")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--segment-points", type=int, default=8)
    parser.add_argument("--actor-batch-size", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--positive-mse-quantile", type=float, default=0.75)
    parser.add_argument("--positive-progress-quantile", type=float, default=0.25)
    parser.add_argument("--mse-temperature", type=float, default=0.0)
    parser.add_argument("--min-positive-weight", type=float, default=0.20)
    parser.add_argument("--demote-disagreeing-positives", type=int, default=1)
    parser.add_argument("--upweight-sequence-hard-negatives", type=int, default=1)
    parser.add_argument("--hard-negative-weight", type=float, default=1.25)
    parser.add_argument("--encoder", default="not_used")
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tdr-expectile", type=float, default=0.999)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--p-aug", type=float, default=0.0)
    parser.add_argument("--way-steps", type=int, default=8)
    args = parser.parse_args()
    add_sequence_features(args)


if __name__ == "__main__":
    main()

