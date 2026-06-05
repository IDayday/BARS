#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator

import numpy as np

from cage_gp0_common import (
    DEFAULT_GP0_ENVS,
    embeddings_path,
    keygraph_path,
    load_keygraph,
    manifest_path,
    read_json,
    summarize_numeric,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract or approximate GAS low-level policy training goal distribution q_train.")
    parser.add_argument("--checkpoint_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--envs", nargs="+", default=DEFAULT_GP0_ENVS)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--num_samples", type=int, default=20000)
    parser.add_argument("--audit_seed", type=int, default=0)
    parser.add_argument("--discount", type=float, default=None)
    parser.add_argument("--way_steps", type=float, default=None)
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--max_scan_steps", type=int, default=5000)
    parser.add_argument("--include_vectors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out_jsonl", default=None)
    parser.add_argument("--out_summary_json", default=None)
    return parser.parse_args()


def load_ogbench_terminals(env_name: str, split: str) -> np.ndarray:
    try:
        import ogbench  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("ogbench is required to read GAS OGBench training terminals") from exc
    _, train_dataset, val_dataset = ogbench.make_env_and_datasets(env_name, compact_dataset=False)
    dataset = train_dataset if split == "train" else val_dataset
    if "terminals" in dataset:
        terminals = np.asarray(dataset["terminals"]).astype(bool)
    elif "timeouts" in dataset:
        terminals = np.asarray(dataset["timeouts"]).astype(bool)
    else:
        terminals = np.zeros(len(dataset["observations"]), dtype=bool)
        terminals[-1] = True
    if len(terminals) and not terminals[-1]:
        terminals = terminals.copy()
        terminals[-1] = True
    return terminals


def terminal_locs_from_terminals(terminals: np.ndarray, size: int) -> np.ndarray:
    if len(terminals) != size:
        terminals = np.asarray(terminals[:size], dtype=bool)
        if len(terminals) < size:
            padded = np.zeros(size, dtype=bool)
            padded[: len(terminals)] = terminals
            terminals = padded
    if size and not terminals[-1]:
        terminals = terminals.copy()
        terminals[-1] = True
    locs = np.nonzero(terminals)[0]
    if len(locs) == 0 and size:
        locs = np.asarray([size - 1], dtype=np.int64)
    return locs


def find_waystep_idx(embeddings: np.ndarray, idx: int, terminal_idx: int, way_steps: float, max_scan_steps: int) -> int:
    if idx >= terminal_idx:
        return int(terminal_idx)
    end = min(int(terminal_idx), int(idx + max_scan_steps))
    sub = np.asarray(embeddings[idx : end + 1], dtype=np.float32) - np.asarray(embeddings[idx], dtype=np.float32)
    distances = np.linalg.norm(sub, axis=1)
    hits = np.where(distances >= way_steps)[0]
    return int(idx + hits[0]) if len(hits) else int(terminal_idx)


def rows_for_env_seed(args: argparse.Namespace, env_name: str, seed: int) -> Iterator[dict]:
    emb_path = embeddings_path(args.checkpoint_root, env_name, seed)
    kg_path = keygraph_path(args.checkpoint_root, env_name, seed)
    if not emb_path.exists():
        raise FileNotFoundError(f"Missing dataset embeddings: {emb_path}")
    if not kg_path.exists():
        raise FileNotFoundError(f"Missing keygraph: {kg_path}")
    embeddings = np.load(emb_path, mmap_mode="r")
    keygraph = load_keygraph(kg_path)
    manifest = read_json(manifest_path(args.checkpoint_root, env_name, seed), default={}) or {}
    discount = float(args.discount if args.discount is not None else manifest.get("official_protocol", {}).get("discount", 0.99))
    way_steps = float(args.way_steps if args.way_steps is not None else keygraph.get("way_steps", manifest.get("official_protocol", {}).get("way_steps", 8)))
    terminals = load_ogbench_terminals(env_name, args.split)
    size = min(len(embeddings), len(terminals))
    terminal_locs = terminal_locs_from_terminals(terminals[:size], size)
    rng = np.random.default_rng(args.audit_seed + int(seed))
    sample_count = min(int(args.num_samples), size)
    idxs = rng.choice(size, size=sample_count, replace=False)
    final_for_idx = terminal_locs[np.searchsorted(terminal_locs, idxs)]
    geom_p = max(1e-6, min(1.0, 1.0 - discount))

    for local_i, idx in enumerate(idxs):
        idx = int(idx)
        terminal_idx = int(final_for_idx[local_i])
        waystep_idx = find_waystep_idx(embeddings, idx, terminal_idx, way_steps, args.max_scan_steps)
        offset = int(rng.geometric(p=geom_p))
        goal_idx = int(min(idx + offset, waystep_idx))
        if goal_idx > terminal_idx:
            goal_idx = terminal_idx
        s_ref = np.asarray(embeddings[idx], dtype=np.float32)
        g_ref = np.asarray(embeddings[goal_idx], dtype=np.float32)
        d_phi = float(np.linalg.norm(g_ref - s_ref))
        temporal_gap = int(goal_idx - idx)
        row = {
            "record_type": "policy_train_pair",
            "env_name": env_name,
            "seed": int(seed),
            "state_idx": idx,
            "goal_idx": goal_idx,
            "terminal_idx": terminal_idx,
            "waystep_idx": int(waystep_idx),
            "temporal_gap": temporal_gap,
            "d_phi": d_phi,
            "goal_source": "same_trajectory_geometric_to_waystep",
            "relabeling_mode": "GASDataset.sample_actor_goal",
            "discount": discount,
            "geometric_p": geom_p,
            "way_steps": way_steps,
            "split": args.split,
            "source": "gas_policy_training_distribution_replay",
            "checkpoint_root": str(args.checkpoint_root),
            "embeddings_path": str(emb_path),
            "keygraph_path": str(kg_path),
            "manifest_source": manifest.get("source"),
        }
        if args.include_vectors:
            row["s_ref"] = s_ref
            row["g_ref"] = g_ref
        yield row


def summarize(rows: list[dict]) -> dict:
    by_env: dict[str, list[dict]] = {}
    for row in rows:
        by_env.setdefault(row["env_name"], []).append(row)
    return {
        "num_pairs": len(rows),
        "envs": sorted(by_env),
        "by_env": {
            env: {
                "num_pairs": len(records),
                "d_phi": summarize_numeric(r.get("d_phi") for r in records),
                "temporal_gap": summarize_numeric(r.get("temporal_gap") for r in records),
                "waystep_gap": summarize_numeric((r.get("waystep_idx") or 0) - (r.get("state_idx") or 0) for r in records),
            }
            for env, records in by_env.items()
        },
    }


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    out_jsonl = Path(args.out_jsonl) if args.out_jsonl else output_root / "policy_training_pairs.jsonl"
    out_summary = Path(args.out_summary_json) if args.out_summary_json else output_root / "policy_training_summary.json"

    rows: list[dict] = []
    for env_name in args.envs:
        for seed in args.seeds:
            root = Path(args.checkpoint_root) / env_name / f"seed{seed}"
            if not root.exists():
                continue
            rows.extend(rows_for_env_seed(args, env_name, seed))
    count = write_jsonl(out_jsonl, rows)
    summary = summarize(rows)
    summary.update({"out_jsonl": str(out_jsonl), "count_written": count})
    write_json(out_summary, summary)
    print({"out_jsonl": str(out_jsonl), "out_summary_json": str(out_summary), "rows": count})


if __name__ == "__main__":
    main()
