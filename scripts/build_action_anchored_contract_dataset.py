#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "reports" / "stage38_action_anchored_contract_dataset.md"
EXTRA_DATASET_ROOTS = [
    REPO_ROOT / "artifacts" / "stage27_gas" / "datasets",
    Path("/root/.ogbench/data"),
    REPO_ROOT / "_data" / "ogbench",
]
TARGET_MODE_IDS = {
    "offline_future_positive": 0,
    "hard_negative": 1,
    "final_goal": 2,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build action-anchored ECG contract samples from raw/offline trajectories.")
    parser.add_argument("--envs", nargs="+", required=True)
    parser.add_argument("--dataset_roots", nargs="+", required=True)
    parser.add_argument("--checkpoint_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_rows_per_env", type=int, default=300000)
    parser.add_argument("--positive_horizons", nargs="+", type=int, default=[4, 8, 16, 32, 64])
    parser.add_argument("--negative_samples_per_state", type=int, default=4)
    parser.add_argument("--final_goal_samples_per_traj", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clear", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    if args.clear and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.seed))
    roots = normalize_roots(args.dataset_roots)
    rows_path = out_dir / "action_contracts.jsonl"
    npz_path = out_dir / "action_contracts.npz"
    summary_path = out_dir / "dataset_summary.json"

    all_npz: dict[str, list[np.ndarray]] = {
        "observation": [],
        "next_observation": [],
        "action": [],
        "phi_s": [],
        "phi_next": [],
        "phi_g": [],
        "label_positive_contract": [],
        "label_negative_contract": [],
        "label_final_goal": [],
        "horizon": [],
        "target_mode_id": [],
        "d_phi": [],
    }
    summary: dict[str, Any] = {
        "status": "initialized",
        "dataset_roots": [str(root) for root in roots],
        "checkpoint_root": str(Path(args.checkpoint_root).expanduser()),
        "envs": {},
        "missing_reason_counts": {},
    }
    total_rows = 0
    total_action = 0
    total_positive_action = 0
    total_final_action = 0
    opened = rows_path.open("w", encoding="utf-8")
    try:
        for env_name in args.envs:
            candidates = discover_dataset_files(env_name, roots)
            if not candidates:
                summary["envs"][env_name] = {
                    "status": "missing_dataset",
                    "searched_roots": [str(root) for root in roots],
                    "candidate_paths": [],
                }
                add_missing(summary, "dataset_missing")
                continue
            env_arrays = load_env_arrays(env_name, candidates, max_source_rows=int(args.max_rows_per_env), rng=rng)
            if env_arrays is None:
                summary["envs"][env_name] = {
                    "status": "unreadable_dataset",
                    "candidate_paths": [str(path) for path in candidates],
                }
                add_missing(summary, "dataset_unreadable")
                continue
            env_rows, env_npz = emit_env_samples(
                env_name=env_name,
                arrays=env_arrays,
                output=opened,
                rng=rng,
                max_rows=int(args.max_rows_per_env),
                positive_horizons=sorted(set(int(h) for h in args.positive_horizons if int(h) > 0)),
                negative_samples_per_state=max(0, int(args.negative_samples_per_state)),
                final_goal_samples_per_traj=max(0, int(args.final_goal_samples_per_traj)),
            )
            for key, values in env_npz.items():
                all_npz[key].extend(values)
            total_rows += env_rows["sample_count"]
            total_action += env_rows["action_available_count"]
            total_positive_action += env_rows["positive_with_action_count"]
            total_final_action += env_rows["final_goal_with_action_count"]
            summary["envs"][env_name] = env_rows
    finally:
        opened.close()

    summary["total_examples"] = int(total_rows)
    summary["action_available_count"] = int(total_action)
    summary["action_supervision_rate"] = safe_rate(total_action, total_rows)
    summary["positive_with_action_count"] = int(total_positive_action)
    summary["final_goal_with_action_count"] = int(total_final_action)
    if total_positive_action > 0 and safe_rate(total_action, total_rows) and safe_rate(total_action, total_rows) > 0:
        summary["status"] = "ACTION_ANCHORED_DATASET_READY"
    else:
        summary["status"] = "BLOCKED_NO_ACTION_ANCHORED_POSITIVES"
        add_missing(summary, "positive_action_supervision_missing")

    write_npz(npz_path, all_npz)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_report(REPORT_PATH, summary, rows_path, npz_path)
    print(json.dumps({"status": summary["status"], "examples": total_rows, "summary": str(summary_path)}, sort_keys=True))
    return 0 if summary["status"] == "ACTION_ANCHORED_DATASET_READY" else 2


def normalize_roots(raw_roots: list[str]) -> list[Path]:
    roots: list[Path] = []
    for item in raw_roots:
        root = Path(item).expanduser()
        if not root.is_absolute():
            root = REPO_ROOT / root
        roots.append(root)
    for root in EXTRA_DATASET_ROOTS:
        if root not in roots:
            roots.append(root)
    return roots


def discover_dataset_files(env_name: str, roots: list[Path]) -> list[Path]:
    candidates: list[Path] = []
    names = [env_name, env_name.replace("-v0", "")]
    for root in roots:
        if not root.exists():
            continue
        for name in names:
            direct = [
                root / f"{name}.npz",
                root / "data" / f"{name}.npz",
                root / env_name / "dataset.npz",
                root / name / "dataset.npz",
            ]
            for path in direct:
                if path.exists() and path not in candidates:
                    candidates.append(path)
            for path in sorted((root / env_name).glob("gas_seed*/dataset.npz")) if (root / env_name).exists() else []:
                if path not in candidates:
                    candidates.append(path)
        if root.is_dir():
            for path in sorted(root.glob(f"**/{env_name}*.npz"))[:20]:
                if path.exists() and path not in candidates:
                    candidates.append(path)
    # Prefer trajectory derivatives with TDR embeddings, because they provide the GAS/TDR phi space.
    scored = []
    for path in candidates:
        score = 0
        try:
            with np.load(path, allow_pickle=False) as data:
                files = set(data.files)
            if "tdr_emb" in files:
                score += 100
            if "next_observations" in files:
                score += 10
            if "traj_ids" in files:
                score += 5
        except Exception:
            pass
        scored.append((-score, str(path), path))
    return [path for _, _, path in sorted(scored)]


def load_env_arrays(env_name: str, candidates: list[Path], *, max_source_rows: int, rng: np.random.Generator) -> dict[str, Any] | None:
    parts: list[dict[str, Any]] = []
    remaining = max(1, int(max_source_rows))
    for path in candidates:
        if remaining <= 0:
            break
        try:
            arrays = load_one_npz(path, limit=remaining, rng=rng)
        except Exception:
            continue
        if arrays is None:
            continue
        arrays["dataset_path"] = str(path)
        parts.append(arrays)
        remaining -= len(arrays["observations"])
        # If the first preferred source has TDR embeddings and enough rows, keep the route simple.
        if arrays["phi_source"] == "tdr_emb" and len(arrays["observations"]) >= min(max_source_rows, 50000):
            break
    if not parts:
        return None
    return concatenate_parts(env_name, parts)


def load_one_npz(path: Path, *, limit: int, rng: np.random.Generator) -> dict[str, Any] | None:
    data = np.load(path, allow_pickle=False)
    if "observations" not in data.files or "actions" not in data.files:
        return None
    observations = np.asarray(data["observations"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.float32)
    n = min(len(observations), len(actions))
    if n < 2:
        return None
    if n > limit:
        idx = np.sort(rng.choice(n, size=limit, replace=False))
    else:
        idx = np.arange(n)
    observations = observations[idx]
    actions = actions[idx]
    terminals = np.asarray(data["terminals"], dtype=bool)[idx] if "terminals" in data.files else np.zeros(len(idx), dtype=bool)
    if "traj_ids" in data.files:
        traj_ids = np.asarray(data["traj_ids"])[idx].astype(np.int64)
    else:
        traj_ids = rebuild_traj_ids(terminals)
    if "next_observations" in data.files:
        next_observations = np.asarray(data["next_observations"], dtype=np.float32)[idx]
    else:
        next_observations = shifted_next_observations(observations, terminals, traj_ids)
    if "tdr_emb" in data.files:
        phi = np.asarray(data["tdr_emb"], dtype=np.float32)[idx]
        phi_source = "tdr_emb"
    elif "tmd_emb" in data.files:
        phi = np.asarray(data["tmd_emb"], dtype=np.float32)[idx]
        phi_source = "tmd_emb_fallback"
    else:
        phi = observations.astype(np.float32)
        phi_source = "observation_fallback_no_tdr_embedding"
    phi_next = shifted_next_phi(phi, terminals, traj_ids)
    return {
        "observations": observations,
        "next_observations": next_observations,
        "actions": actions,
        "terminals": terminals,
        "traj_ids": traj_ids,
        "phi": phi,
        "phi_next": phi_next,
        "phi_source": phi_source,
        "action_dim": int(actions.shape[1]) if actions.ndim > 1 else 1,
        "obs_dim": int(observations.shape[1]) if observations.ndim > 1 else 1,
        "phi_dim": int(phi.shape[1]) if phi.ndim > 1 else 1,
    }


def rebuild_traj_ids(terminals: np.ndarray) -> np.ndarray:
    traj = np.zeros(len(terminals), dtype=np.int64)
    cur = 0
    for i in range(len(terminals)):
        traj[i] = cur
        if terminals[i]:
            cur += 1
    return traj


def shifted_next_observations(observations: np.ndarray, terminals: np.ndarray, traj_ids: np.ndarray) -> np.ndarray:
    out = observations.copy()
    if len(out) > 1:
        out[:-1] = observations[1:]
        invalid = terminals[:-1] | (traj_ids[:-1] != traj_ids[1:])
        out[:-1][invalid] = observations[:-1][invalid]
    return out


def shifted_next_phi(phi: np.ndarray, terminals: np.ndarray, traj_ids: np.ndarray) -> np.ndarray:
    out = phi.copy()
    if len(out) > 1:
        out[:-1] = phi[1:]
        invalid = terminals[:-1] | (traj_ids[:-1] != traj_ids[1:])
        out[:-1][invalid] = phi[:-1][invalid]
    return out


def concatenate_parts(env_name: str, parts: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ["observations", "next_observations", "actions", "terminals", "phi", "phi_next"]
    out = {key: np.concatenate([part[key] for part in parts], axis=0) for key in keys}
    trajs: list[np.ndarray] = []
    offset = 0
    for part in parts:
        raw = np.asarray(part["traj_ids"], dtype=np.int64)
        trajs.append(raw + offset)
        offset += int(raw.max()) + 1 if len(raw) else 0
    out["traj_ids"] = np.concatenate(trajs, axis=0)
    out["env_name"] = env_name
    out["dataset_paths"] = [part["dataset_path"] for part in parts]
    out["phi_sources"] = sorted({part["phi_source"] for part in parts})
    out["obs_dim"] = int(out["observations"].shape[1])
    out["action_dim"] = int(out["actions"].shape[1])
    out["phi_dim"] = int(out["phi"].shape[1])
    return out


def emit_env_samples(
    *,
    env_name: str,
    arrays: dict[str, Any],
    output,
    rng: np.random.Generator,
    max_rows: int,
    positive_horizons: list[int],
    negative_samples_per_state: int,
    final_goal_samples_per_traj: int,
) -> tuple[dict[str, Any], dict[str, list[np.ndarray]]]:
    obs = arrays["observations"]
    next_obs = arrays["next_observations"]
    actions = arrays["actions"]
    phi = arrays["phi"]
    phi_next = arrays["phi_next"]
    traj_ids = arrays["traj_ids"]
    n = len(obs)
    by_traj: dict[int, np.ndarray] = {}
    for tid in np.unique(traj_ids):
        by_traj[int(tid)] = np.flatnonzero(traj_ids == tid)
    all_indices = np.arange(n)
    rows_written = 0
    positive_count = 0
    negative_count = 0
    final_count = 0
    npz_rows = empty_npz_rows()
    for tid, traj in by_traj.items():
        if rows_written >= max_rows or final_goal_samples_per_traj <= 0 or len(traj) < 2:
            break
        final_idx = int(traj[-1])
        starts = traj[: max(1, len(traj) - 1)]
        if len(starts) > final_goal_samples_per_traj:
            starts = rng.choice(starts, size=final_goal_samples_per_traj, replace=False)
        for idx in starts:
            if rows_written >= max_rows:
                break
            matches_final = np.flatnonzero(traj == final_idx)
            matches_idx = np.flatnonzero(traj == idx)
            horizon = int(matches_final[0] - matches_idx[0]) if len(matches_final) and len(matches_idx) else -1
            rec = make_record(env_name, int(idx), final_idx, obs, next_obs, actions, phi, phi_next, traj_ids, "final_goal", horizon, True, False)
            write_record(output, rec)
            append_npz(npz_rows, rec)
            rows_written += 1
            final_count += 1
            positive_count += 1
    base_indices = np.array_split(all_indices, max(1, math.ceil(n / max(1, max_rows))))[0]
    rng.shuffle(base_indices)
    for idx in base_indices:
        if rows_written >= max_rows:
            break
        tid = int(traj_ids[idx])
        traj = by_traj.get(tid, np.array([], dtype=int))
        pos_in_traj = int(np.searchsorted(traj, idx))
        for horizon in positive_horizons:
            if rows_written >= max_rows:
                break
            if pos_in_traj + horizon >= len(traj):
                continue
            g_idx = int(traj[pos_in_traj + horizon])
            rec = make_record(env_name, idx, g_idx, obs, next_obs, actions, phi, phi_next, traj_ids, "offline_future_positive", horizon, True, False)
            write_record(output, rec)
            append_npz(npz_rows, rec)
            rows_written += 1
            positive_count += 1
        for _ in range(negative_samples_per_state):
            if rows_written >= max_rows:
                break
            g_idx = sample_negative_index(idx, traj_ids, rng)
            rec = make_record(env_name, idx, g_idx, obs, next_obs, actions, phi, phi_next, traj_ids, "hard_negative", -1, False, True)
            write_record(output, rec)
            append_npz(npz_rows, rec)
            rows_written += 1
            negative_count += 1
    env_summary = {
        "status": "ok" if rows_written else "empty_after_sampling",
        "dataset_paths": arrays["dataset_paths"],
        "phi_sources": arrays["phi_sources"],
        "source_rows": int(n),
        "sample_count": int(rows_written),
        "trajectory_count": int(len(by_traj)),
        "obs_dim": int(arrays["obs_dim"]),
        "action_dim": int(arrays["action_dim"]),
        "phi_dim": int(arrays["phi_dim"]),
        "positive_count": int(positive_count),
        "negative_count": int(negative_count),
        "final_goal_count": int(final_count),
        "action_available_count": int(rows_written),
        "action_supervision_rate": 1.0 if rows_written else 0.0,
        "positive_with_action_count": int(positive_count),
        "final_goal_with_action_count": int(final_count),
    }
    return env_summary, npz_rows


def empty_npz_rows() -> dict[str, list[np.ndarray]]:
    return {
        "observation": [],
        "next_observation": [],
        "action": [],
        "phi_s": [],
        "phi_next": [],
        "phi_g": [],
        "label_positive_contract": [],
        "label_negative_contract": [],
        "label_final_goal": [],
        "horizon": [],
        "target_mode_id": [],
        "d_phi": [],
    }


def sample_negative_index(idx: int, traj_ids: np.ndarray, rng: np.random.Generator) -> int:
    current = traj_ids[idx]
    for _ in range(64):
        cand = int(rng.integers(0, len(traj_ids)))
        if traj_ids[cand] != current:
            return cand
    return int((idx + len(traj_ids) // 2) % len(traj_ids))


def make_record(
    env_name: str,
    idx: int,
    g_idx: int,
    obs: np.ndarray,
    next_obs: np.ndarray,
    actions: np.ndarray,
    phi: np.ndarray,
    phi_next: np.ndarray,
    traj_ids: np.ndarray,
    target_mode: str,
    horizon: int,
    positive: bool,
    negative: bool,
) -> dict[str, Any]:
    d_phi = float(np.linalg.norm(phi[g_idx] - phi[idx]))
    return {
        "record_type": "action_anchored_contract",
        "env_name": env_name,
        "trajectory_id": int(traj_ids[idx]),
        "row_id": int(idx),
        "next_row_id": int(idx + 1) if idx + 1 < len(obs) and traj_ids[idx + 1] == traj_ids[idx] else None,
        "goal_row_id": int(g_idx),
        "observation": as_list(obs[idx]),
        "next_observation": as_list(next_obs[idx]),
        "action": as_list(actions[idx]),
        "phi_s": as_list(phi[idx]),
        "phi_next": as_list(phi_next[idx]),
        "phi_g": as_list(phi[g_idx]),
        "d_phi": d_phi,
        "target_mode": target_mode,
        "horizon": int(horizon),
        "label_positive_contract": bool(positive),
        "label_negative_contract": bool(negative),
        "label_final_goal": bool(target_mode == "final_goal"),
        "label_recovery_candidate": False,
        "action_available": True,
        "action_source": "offline_dataset",
        "trainable_for_bc": bool(positive),
        "trainable_for_ranking": True,
        "trainable_for_contrastive": True,
        "trainable_for_conservative_filtering": True,
    }


def write_record(output, record: dict[str, Any]) -> None:
    output.write(json.dumps(record, sort_keys=True) + "\n")


def append_npz(rows: dict[str, list[np.ndarray]], rec: dict[str, Any]) -> None:
    for key in ["observation", "next_observation", "action", "phi_s", "phi_next", "phi_g"]:
        rows[key].append(np.asarray(rec[key], dtype=np.float32))
    rows["label_positive_contract"].append(np.asarray(rec["label_positive_contract"], dtype=np.int8))
    rows["label_negative_contract"].append(np.asarray(rec["label_negative_contract"], dtype=np.int8))
    rows["label_final_goal"].append(np.asarray(rec["label_final_goal"], dtype=np.int8))
    rows["horizon"].append(np.asarray(rec["horizon"], dtype=np.int32))
    rows["target_mode_id"].append(np.asarray(TARGET_MODE_IDS.get(rec["target_mode"], -1), dtype=np.int16))
    rows["d_phi"].append(np.asarray(rec["d_phi"], dtype=np.float32))


def write_npz(path: Path, rows: dict[str, list[np.ndarray]]) -> None:
    arrays: dict[str, np.ndarray] = {}
    for key, values in rows.items():
        if values:
            arrays[key] = np.asarray(values)
        else:
            arrays[key] = np.asarray([])
    np.savez_compressed(path, **arrays)


def as_list(arr: np.ndarray) -> list[float]:
    return np.asarray(arr, dtype=np.float32).round(6).tolist()


def safe_rate(num: int, den: int) -> float | None:
    return float(num / den) if den else None


def add_missing(summary: dict[str, Any], reason: str) -> None:
    counts = summary.setdefault("missing_reason_counts", {})
    counts[reason] = int(counts.get(reason, 0)) + 1


def write_report(path: Path, summary: dict[str, Any], rows_path: Path, npz_path: Path) -> None:
    lines = [
        "# Stage38 Action-Anchored Contract Dataset",
        "",
        f"- status: `{summary['status']}`",
        f"- jsonl: `{rows_path}`",
        f"- npz: `{npz_path}`",
        f"- total_examples: {summary.get('total_examples', 0)}",
        f"- action_supervision_rate: {summary.get('action_supervision_rate')}",
        f"- positive_with_action_count: {summary.get('positive_with_action_count', 0)}",
        f"- final_goal_with_action_count: {summary.get('final_goal_with_action_count', 0)}",
        "",
        "本数据集从 raw/offline trajectory 的 observation/action/next_observation 重建样本；不使用 Stage37 的 phi-only contract examples 反向匹配 action 作为主路线。",
        "",
        "## Per-Env",
        "",
    ]
    for env_name, row in summary.get("envs", {}).items():
        lines.extend(
            [
                f"### {env_name}",
                f"- status: `{row.get('status')}`",
                f"- dataset_paths: `{row.get('dataset_paths', row.get('candidate_paths'))}`",
                f"- phi_sources: `{row.get('phi_sources')}`",
                f"- source_rows: {row.get('source_rows')}",
                f"- sample_count: {row.get('sample_count')}",
                f"- trajectory_count: {row.get('trajectory_count')}",
                f"- action_dim: {row.get('action_dim')}",
                f"- phi_dim: {row.get('phi_dim')}",
                f"- positive_count: {row.get('positive_count')}",
                f"- negative_count: {row.get('negative_count')}",
                f"- final_goal_count: {row.get('final_goal_count')}",
                "",
            ]
        )
    if summary["status"] != "ACTION_ANCHORED_DATASET_READY":
        lines.extend(["## BLOCKED", "", f"- missing_reason_counts: `{summary.get('missing_reason_counts')}`", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
