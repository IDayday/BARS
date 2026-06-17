#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bars.external.gas_artifacts import resolve_gas_artifacts  # noqa: E402
from bars.gas_bars.graph_table import export_edges, export_nodes, load_gas_keygraph, save_edge_table  # noqa: E402
from bars.gas_bars.reachability_dataset import trajectory_index_from_terminals  # noqa: E402


TEXT_COLUMNS = {
    "env",
    "edge_source",
    "sample_kind",
    "split",
}


def _dataset_path(env_name: str, dataset_dir: str | os.PathLike[str]) -> Path:
    path = Path(dataset_dir) / f"{env_name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing OGBench dataset: {path}")
    return path


def _load_terminals(dataset_path: Path, expected_len: int) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(dataset_path, allow_pickle=False) as data:
        if "terminals" in data.files:
            terminals = np.asarray(data["terminals"], dtype=bool).copy()
        elif "dones" in data.files:
            terminals = np.asarray(data["dones"], dtype=bool).copy()
        else:
            terminals = np.zeros(expected_len, dtype=bool)
            terminals[-1] = True
    raw_len = int(len(terminals))
    raw_terminal_count = int(terminals.sum())
    if len(terminals):
        terminals[-1] = True
    alignment = {
        "raw_terminal_length": raw_len,
        "raw_terminal_count": raw_terminal_count,
        "embedding_length": int(expected_len),
        "mode": "as_is",
    }
    if len(terminals) == expected_len:
        return terminals, alignment
    if len(terminals) - int(terminals.sum()) == expected_len:
        raw = terminals.copy()
        terminal_locs = np.flatnonzero(raw)
        penult_idx = terminal_locs[terminal_locs > 0] - 1
        raw[penult_idx] = True
        aligned = raw[~terminals]
        if len(aligned):
            aligned[-1] = True
        alignment["mode"] = "drop_raw_terminal_rows_mark_penultimate"
        alignment["aligned_terminal_count"] = int(aligned.sum())
        return aligned, alignment
    raise RuntimeError(f"Embedding/dataset length mismatch: {expected_len} embeddings vs {len(terminals)} terminals")


def _episode_split(traj_id: np.ndarray, train_frac: float, val_frac: float, seed: int) -> tuple[np.ndarray, dict[str, int]]:
    n_traj = int(traj_id.max()) + 1 if len(traj_id) else 0
    ids = np.arange(n_traj, dtype=np.int32)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_train = int(round(n_traj * train_frac))
    n_val = int(round(n_traj * val_frac))
    n_train = min(max(n_train, 0), n_traj)
    n_val = min(max(n_val, 0), max(n_traj - n_train, 0))
    split_by_traj = np.full(n_traj, "test", dtype=object)
    split_by_traj[ids[:n_train]] = "train"
    split_by_traj[ids[n_train : n_train + n_val]] = "val"
    counts = Counter(str(x) for x in split_by_traj)
    return split_by_traj, {"train": int(counts["train"]), "val": int(counts["val"]), "test": int(counts["test"])}


def _fit_neighbors(phis: np.ndarray, node_phis: np.ndarray, k: int, max_index_states: int, seed: int) -> tuple[np.ndarray, np.ndarray, int]:
    n = len(phis)
    if n == 0:
        raise RuntimeError("Empty dataset embeddings")
    rng = np.random.default_rng(seed)
    if max_index_states and max_index_states < n:
        index_ids = np.sort(rng.choice(n, size=int(max_index_states), replace=False))
    else:
        index_ids = np.arange(n, dtype=np.int64)
    k_eff = max(1, min(int(k), len(index_ids)))
    try:
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=k_eff).fit(np.asarray(phis[index_ids], dtype=np.float32))
        dists, local = nn.kneighbors(node_phis, return_distance=True)
        return index_ids[local], dists.astype(np.float32), int(len(index_ids))
    except Exception:
        sub = np.asarray(phis[index_ids], dtype=np.float32)
        d = np.linalg.norm(node_phis[:, None, :] - sub[None, :, :], axis=-1)
        order = np.argsort(d, axis=1)[:, :k_eff]
        return index_ids[order], np.take_along_axis(d, order, axis=1).astype(np.float32), int(len(index_ids))


def _near_lists(indices: np.ndarray, dists: np.ndarray, radius: float, fallback_k: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    near_idx: list[np.ndarray] = []
    near_dist: list[np.ndarray] = []
    fallback_k = max(1, int(fallback_k))
    for row in range(len(indices)):
        mask = dists[row] <= radius
        idx = indices[row][mask]
        dist = dists[row][mask]
        if len(idx) == 0:
            idx = indices[row][:fallback_k]
            dist = dists[row][:fallback_k]
        near_idx.append(np.asarray(idx, dtype=np.int64))
        near_dist.append(np.asarray(dist, dtype=np.float32))
    return near_idx, near_dist


def _filter_endpoint_neighbors(
    indices: np.ndarray,
    dists: np.ndarray,
    *,
    edge_phi_dist: float,
    max_dist_ratio: float,
) -> tuple[np.ndarray, np.ndarray, bool, bool]:
    if max_dist_ratio <= 0 or len(indices) == 0:
        return indices, dists, False, False
    dist_limit = max(float(edge_phi_dist), 1e-6) * float(max_dist_ratio)
    mask = np.asarray(dists, dtype=np.float32) <= dist_limit
    if mask.any():
        trimmed = int(mask.sum()) < int(len(indices))
        return indices[mask], dists[mask], trimmed, False
    best = int(np.argmin(dists))
    return indices[best : best + 1], dists[best : best + 1], True, True


def _sample_positions(mask: np.ndarray, limit: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if limit <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    flat = np.flatnonzero(mask)
    if len(flat) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    if len(flat) > limit:
        flat = rng.choice(flat, size=int(limit), replace=False)
    return np.unravel_index(flat, mask.shape)


def _make_rows(
    *,
    env_name: str,
    seed: int,
    edge: pd.Series,
    query_u_node: int,
    query_v_node: int,
    source_indices: np.ndarray,
    target_indices: np.ndarray,
    source_dists: np.ndarray,
    target_dists: np.ndarray,
    source_pos: np.ndarray,
    target_pos: np.ndarray,
    sample_kind: str,
    label: int,
    edge_forward: int,
    phis: np.ndarray,
    traj_id: np.ndarray,
    timestep: np.ndarray,
    split_by_traj: np.ndarray,
    same_traj_support: int,
    h_exec: int,
    way_steps: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if len(source_pos) == 0:
        return rows
    u_idx = source_indices[source_pos]
    v_idx = target_indices[target_pos]
    u_traj = traj_id[u_idx]
    v_traj = traj_id[v_idx]
    dt = timestep[v_idx] - timestep[u_idx]
    same_traj = (u_traj == v_traj).astype(np.int32)
    pair_dist = np.linalg.norm(np.asarray(phis[v_idx], dtype=np.float32) - np.asarray(phis[u_idx], dtype=np.float32), axis=1)
    weights = np.ones(len(u_idx), dtype=np.float32)
    if label == 1:
        weights = np.exp(-np.maximum(dt, 1) / max(h_exec, 1)).astype(np.float32)
    elif label < 0:
        weights = np.zeros(len(u_idx), dtype=np.float32)

    for i in range(len(u_idx)):
        rows.append(
            {
                "env": env_name,
                "seed": int(seed),
                "edge_id": int(edge.edge_id),
                "edge_u": int(edge.u),
                "edge_v": int(edge.v),
                "query_u_node": int(query_u_node),
                "query_v_node": int(query_v_node),
                "edge_forward": int(edge_forward),
                "edge_source": str(edge.edge_source),
                "sample_kind": sample_kind,
                "u_idx": int(u_idx[i]),
                "v_idx": int(v_idx[i]),
                "u_traj": int(u_traj[i]),
                "v_traj": int(v_traj[i]),
                "u_timestep": int(timestep[u_idx[i]]),
                "v_timestep": int(timestep[v_idx[i]]),
                "same_traj": int(same_traj[i]),
                "dt_if_same_traj": int(dt[i]) if same_traj[i] else -1,
                "label_reach": int(label),
                "label_weight": float(weights[i]),
                "split": str(split_by_traj[u_traj[i]]),
                "phi_dist_pair": float(pair_dist[i]),
                "edge_phi_dist": float(edge.phi_dist),
                "node_u_dist": float(source_dists[source_pos[i]]),
                "node_v_dist": float(target_dists[target_pos[i]]),
                "local_support": int(same_traj_support > 0),
                "same_traj_support": int(same_traj_support),
                "h_exec": int(h_exec),
                "way_steps": float(way_steps),
            }
        )
    return rows


def _save_npz(df: pd.DataFrame, path: Path) -> None:
    arrays: dict[str, np.ndarray] = {}
    for col in df.columns:
        vals = df[col].to_numpy()
        if col in TEXT_COLUMNS:
            arrays[col] = vals.astype(str)
        else:
            arrays[col] = vals
    np.savez_compressed(path, **arrays)


def build_contract_dataset(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.sampling_seed))

    artifacts = resolve_gas_artifacts(args.env, args.seed, args.gas_artifact_root)
    if artifacts.keygraph is None:
        raise RuntimeError(f"Missing keygraph under {artifacts.graph_dir}")
    if artifacts.dataset_embeddings is None:
        raise RuntimeError(f"Missing dataset embeddings under {artifacts.features_dir}")

    phis = np.load(artifacts.dataset_embeddings, mmap_mode="r")
    dataset_path = _dataset_path(args.env, args.dataset_dir)
    terminals, terminal_alignment = _load_terminals(dataset_path, len(phis))
    traj_id, timestep = trajectory_index_from_terminals(terminals)
    split_by_traj, split_episode_counts = _episode_split(
        traj_id,
        train_frac=float(args.train_frac),
        val_frac=float(args.val_frac),
        seed=int(args.sampling_seed),
    )

    key_graph = load_gas_keygraph(artifacts.keygraph)
    nodes = export_nodes(key_graph)
    edges = export_edges(key_graph)
    save_edge_table(nodes, edges, out / "graph")

    phi_cols = [c for c in nodes.columns if c.startswith("phi_")]
    node_phis = nodes[phi_cols].to_numpy(np.float32)
    way_steps = float(getattr(key_graph, "way_steps", 8.0) or 8.0)
    h_exec = int(args.support_horizon or max(1, round(way_steps)))
    radius = float(args.support_radius or max(way_steps, 1e-6))

    nn_indices, nn_dists, indexed_state_count = _fit_neighbors(
        phis,
        node_phis,
        k=int(args.support_k),
        max_index_states=int(args.max_index_states),
        seed=int(args.sampling_seed),
    )
    near_idx, near_dist = _near_lists(nn_indices, nn_dists, radius, fallback_k=min(int(args.support_k), 8))

    rows: list[dict[str, Any]] = []
    edge_positive_counts: list[int] = []
    strict_cross = bool(args.strict_cross_split)
    endpoint_ratio = float(args.max_endpoint_dist_ratio)
    src_trimmed_edges = 0
    dst_trimmed_edges = 0
    src_fallback_edges = 0
    dst_fallback_edges = 0

    for edge in edges.itertuples(index=False):
        u_node = int(edge.u)
        v_node = int(edge.v)
        if u_node >= len(near_idx) or v_node >= len(near_idx):
            continue
        src = near_idx[u_node]
        dst = near_idx[v_node]
        src_dist = near_dist[u_node]
        dst_dist = near_dist[v_node]
        src, src_dist, src_trimmed, src_fallback = _filter_endpoint_neighbors(
            src,
            src_dist,
            edge_phi_dist=float(edge.phi_dist),
            max_dist_ratio=endpoint_ratio,
        )
        dst, dst_dist, dst_trimmed, dst_fallback = _filter_endpoint_neighbors(
            dst,
            dst_dist,
            edge_phi_dist=float(edge.phi_dist),
            max_dist_ratio=endpoint_ratio,
        )
        src_trimmed_edges += int(src_trimmed)
        dst_trimmed_edges += int(dst_trimmed)
        src_fallback_edges += int(src_fallback)
        dst_fallback_edges += int(dst_fallback)
        if len(src) == 0 or len(dst) == 0:
            edge_positive_counts.append(0)
            continue

        same = traj_id[src][:, None] == traj_id[dst][None, :]
        dt = timestep[dst][None, :] - timestep[src][:, None]
        split_match = split_by_traj[traj_id[src]][:, None] == split_by_traj[traj_id[dst]][None, :]
        positive_mask = same & (dt > 0) & (dt <= h_exec)
        cross_mask = ~same
        if strict_cross:
            cross_mask = cross_mask & split_match
        far_mask = same & ((dt <= 0) | (dt > h_exec))

        support_count = int(positive_mask.sum())
        edge_positive_counts.append(support_count)

        edge_series = pd.Series(edge._asdict())
        pos_i, pos_j = _sample_positions(positive_mask, int(args.max_positive_per_edge), rng)
        rows.extend(
            _make_rows(
                env_name=args.env,
                seed=args.seed,
                edge=edge_series,
                query_u_node=u_node,
                query_v_node=v_node,
                source_indices=src,
                target_indices=dst,
                source_dists=src_dist,
                target_dists=dst_dist,
                source_pos=pos_i,
                target_pos=pos_j,
                sample_kind="same_traj_edge_positive",
                label=1,
                edge_forward=1,
                phis=phis,
                traj_id=traj_id,
                timestep=timestep,
                split_by_traj=split_by_traj,
                same_traj_support=support_count,
                h_exec=h_exec,
                way_steps=way_steps,
            )
        )

        rev_i, rev_j = _sample_positions(positive_mask, int(args.max_reverse_per_edge), rng)
        rows.extend(
            _make_rows(
                env_name=args.env,
                seed=args.seed,
                edge=edge_series,
                query_u_node=v_node,
                query_v_node=u_node,
                source_indices=dst,
                target_indices=src,
                source_dists=dst_dist,
                target_dists=src_dist,
                source_pos=rev_j,
                target_pos=rev_i,
                sample_kind="reversed_edge_negative",
                label=0,
                edge_forward=0,
                phis=phis,
                traj_id=traj_id,
                timestep=timestep,
                split_by_traj=split_by_traj,
                same_traj_support=support_count,
                h_exec=h_exec,
                way_steps=way_steps,
            )
        )

        cross_i, cross_j = _sample_positions(cross_mask, int(args.max_cross_negative_per_edge), rng)
        rows.extend(
            _make_rows(
                env_name=args.env,
                seed=args.seed,
                edge=edge_series,
                query_u_node=u_node,
                query_v_node=v_node,
                source_indices=src,
                target_indices=dst,
                source_dists=src_dist,
                target_dists=dst_dist,
                source_pos=cross_i,
                target_pos=cross_j,
                sample_kind="cross_traj_near_negative",
                label=0,
                edge_forward=1,
                phis=phis,
                traj_id=traj_id,
                timestep=timestep,
                split_by_traj=split_by_traj,
                same_traj_support=support_count,
                h_exec=h_exec,
                way_steps=way_steps,
            )
        )

        far_i, far_j = _sample_positions(far_mask, int(args.max_far_negative_per_edge), rng)
        rows.extend(
            _make_rows(
                env_name=args.env,
                seed=args.seed,
                edge=edge_series,
                query_u_node=u_node,
                query_v_node=v_node,
                source_indices=src,
                target_indices=dst,
                source_dists=src_dist,
                target_dists=dst_dist,
                source_pos=far_i,
                target_pos=far_j,
                sample_kind="same_traj_far_negative",
                label=0,
                edge_forward=1,
                phis=phis,
                traj_id=traj_id,
                timestep=timestep,
                split_by_traj=split_by_traj,
                same_traj_support=support_count,
                h_exec=h_exec,
                way_steps=way_steps,
            )
        )

        if args.max_unlabeled_per_edge > 0 and support_count == 0:
            all_mask = np.ones((len(src), len(dst)), dtype=bool)
            unl_i, unl_j = _sample_positions(all_mask, int(args.max_unlabeled_per_edge), rng)
            rows.extend(
                _make_rows(
                    env_name=args.env,
                    seed=args.seed,
                    edge=edge_series,
                    query_u_node=u_node,
                    query_v_node=v_node,
                    source_indices=src,
                    target_indices=dst,
                    source_dists=src_dist,
                    target_dists=dst_dist,
                    source_pos=unl_i,
                    target_pos=unl_j,
                    sample_kind="unsupported_graph_edge_unlabeled",
                    label=-1,
                    edge_forward=1,
                    phis=phis,
                    traj_id=traj_id,
                    timestep=timestep,
                    split_by_traj=split_by_traj,
                    same_traj_support=support_count,
                    h_exec=h_exec,
                    way_steps=way_steps,
                )
            )

    df = pd.DataFrame(rows)
    if len(df) and args.max_rows and len(df) > args.max_rows:
        keep = rng.choice(len(df), size=int(args.max_rows), replace=False)
        df = df.iloc[np.sort(keep)].reset_index(drop=True)

    csv_path = out / "offline_contract_pairs.csv"
    npz_path = out / "offline_contract_pairs.npz"
    df.to_csv(csv_path, index=False)
    _save_npz(df, npz_path)
    try:
        df.to_parquet(out / "offline_contract_pairs.parquet", index=False)
    except Exception:
        pass

    labeled = df[df["label_reach"] >= 0] if len(df) else df
    pos_edges = int(np.sum(np.asarray(edge_positive_counts) > 0))
    summary = {
        "env": args.env,
        "seed": int(args.seed),
        "offline_only": True,
        "uses_environment_rollout": False,
        "gas_artifact_root": str(args.gas_artifact_root),
        "keygraph": str(artifacts.keygraph),
        "dataset_embeddings": str(artifacts.dataset_embeddings),
        "dataset_path": str(dataset_path),
        "terminal_alignment": terminal_alignment,
        "outputs": {
            "csv": str(csv_path),
            "npz": str(npz_path),
            "graph_dir": str(out / "graph"),
        },
        "num_dataset_states": int(len(phis)),
        "num_indexed_states": int(indexed_state_count),
        "num_episodes": int(split_by_traj.shape[0]),
        "episode_split_counts": split_episode_counts,
        "num_nodes": int(len(nodes)),
        "num_edges": int(len(edges)),
        "num_edges_with_positive_support": pos_edges,
        "edge_positive_support_rate": float(pos_edges / len(edges)) if len(edges) else 0.0,
        "mean_positive_pairs_per_edge": float(np.mean(edge_positive_counts)) if edge_positive_counts else 0.0,
        "median_positive_pairs_per_edge": float(np.median(edge_positive_counts)) if edge_positive_counts else 0.0,
        "num_rows": int(len(df)),
        "num_labeled_rows": int(len(labeled)),
        "row_counts_by_kind": {str(k): int(v) for k, v in Counter(df["sample_kind"]).items()} if len(df) else {},
        "row_counts_by_split": {str(k): int(v) for k, v in Counter(df["split"]).items()} if len(df) else {},
        "row_counts_by_label": {str(k): int(v) for k, v in Counter(df["label_reach"]).items()} if len(df) else {},
        "positive_row_rate_labeled": float((labeled["label_reach"] == 1).mean()) if len(labeled) else 0.0,
        "support_k": int(args.support_k),
        "support_radius": float(radius),
        "support_horizon": int(h_exec),
        "max_endpoint_dist_ratio": float(endpoint_ratio),
        "source_endpoint_trimmed_edges": int(src_trimmed_edges),
        "target_endpoint_trimmed_edges": int(dst_trimmed_edges),
        "source_endpoint_fallback_edges": int(src_fallback_edges),
        "target_endpoint_fallback_edges": int(dst_fallback_edges),
        "strict_cross_split": int(strict_cross),
        "max_positive_per_edge": int(args.max_positive_per_edge),
        "max_reverse_per_edge": int(args.max_reverse_per_edge),
        "max_cross_negative_per_edge": int(args.max_cross_negative_per_edge),
        "max_far_negative_per_edge": int(args.max_far_negative_per_edge),
        "max_unlabeled_per_edge": int(args.max_unlabeled_per_edge),
    }
    summary_path = out / "offline_contract_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an offline edge contract dataset from fixed OGBench data and GAS latent graph artifacts."
    )
    parser.add_argument("--env", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gas-artifact-root", required=True)
    parser.add_argument("--dataset-dir", default=os.environ.get("OGBENCH_DATASET_DIR", "/mnt/project/offlinerl_datasets/ogbench"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--support-k", type=int, default=64)
    parser.add_argument("--support-radius", type=float, default=0.0)
    parser.add_argument("--support-horizon", type=int, default=0)
    parser.add_argument(
        "--max-endpoint-dist-ratio",
        type=float,
        default=0.0,
        help=(
            "Optional edge-relative filter: keep only dataset states with "
            "node_dist <= ratio * edge_phi_dist for each endpoint. "
            "Disabled when <= 0."
        ),
    )
    parser.add_argument("--max-index-states", type=int, default=0)
    parser.add_argument("--max-positive-per-edge", type=int, default=8)
    parser.add_argument("--max-reverse-per-edge", type=int, default=4)
    parser.add_argument("--max-cross-negative-per-edge", type=int, default=4)
    parser.add_argument("--max-far-negative-per-edge", type=int, default=4)
    parser.add_argument("--max-unlabeled-per-edge", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--strict-cross-split", type=int, default=1)
    parser.add_argument("--sampling-seed", type=int, default=0)
    args = parser.parse_args()

    if args.train_frac < 0 or args.val_frac < 0 or args.train_frac + args.val_frac >= 1.0:
        raise SystemExit("--train-frac and --val-frac must be nonnegative and leave a positive test split")
    summary = build_contract_dataset(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
