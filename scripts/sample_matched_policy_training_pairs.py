#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from cage_gp0_common import iter_jsonl, write_jsonl
from extract_graph_planned_goal_distribution import load_dataset_observations, state_ref_for_dataset_idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample q_train controls matched to q_G d_phi bins.")
    parser.add_argument("--graph_pairs_path", required=True)
    parser.add_argument("--train_pairs_path", required=True)
    parser.add_argument("--checkpoint_root", required=True)
    parser.add_argument("--env_name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out_jsonl", required=True)
    parser.add_argument("--num_pairs", type=int, default=128)
    parser.add_argument("--bins", nargs="*", type=float, default=[0, 4, 8, 16, 32, 1e9])
    parser.add_argument("--audit_seed", type=int, default=0)
    return parser.parse_args()


def bin_id(value: float, bins: list[float]) -> int:
    for i in range(len(bins) - 1):
        if bins[i] <= value < bins[i + 1]:
            return i
    return len(bins) - 2


def main() -> None:
    args = parse_args()
    graph_rows = [r for r in iter_jsonl(args.graph_pairs_path) if r.get("env_name") == args.env_name and int(r.get("seed", args.seed)) == args.seed]
    train_rows = [r for r in iter_jsonl(args.train_pairs_path) if r.get("env_name") == args.env_name and int(r.get("seed", args.seed)) == args.seed]
    rng = np.random.default_rng(args.audit_seed + args.seed)
    graph_bins = [bin_id(float(r.get("d_phi", 0)), args.bins) for r in graph_rows]
    counts = {b: graph_bins.count(b) for b in set(graph_bins)}
    total = sum(counts.values()) or 1
    desired = {b: max(1, int(round(args.num_pairs * c / total))) for b, c in counts.items()}
    train_by_bin: dict[int, list[dict[str, Any]]] = {}
    for row in train_rows:
        train_by_bin.setdefault(bin_id(float(row.get("d_phi", 0)), args.bins), []).append(row)

    dataset_obs, state_ref_meta = load_dataset_observations(args.env_name)
    out = []
    for b, n in desired.items():
        candidates = train_by_bin.get(b, [])
        if not candidates:
            continue
        idxs = rng.choice(len(candidates), size=min(n, len(candidates)), replace=False)
        for idx in idxs:
            row = dict(candidates[int(idx)])
            row["pair_role"] = "qtrain_matched_control"
            row["pair_source"] = "qtrain_matched"
            row["path_position_bucket"] = "qtrain_control"
            row["final_phase"] = False
            state_ref, exact, reason = state_ref_for_dataset_idx(
                args.env_name,
                args.seed,
                dataset_obs,
                int(row.get("state_idx", -1)),
                np.asarray(row.get("s_ref"), dtype=np.float32),
                state_ref_meta,
                True,
                "best_effort",
            )
            row["state_ref_s"] = state_ref
            row["s_ref_exact_reset_capable"] = exact
            row["probeable"] = exact
            row["non_probeable_reason"] = None if exact else reason
            row["phi_s"] = row.get("s_ref")
            row["phi_g"] = row.get("g_ref")
            out.append(row)
            if len(out) >= args.num_pairs:
                break
        if len(out) >= args.num_pairs:
            break
    write_jsonl(args.out_jsonl, out)
    print({"out_jsonl": args.out_jsonl, "rows": len(out)})


if __name__ == "__main__":
    main()
