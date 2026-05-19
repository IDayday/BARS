from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


PAIR_COLUMNS = [
    "u_idx",
    "v_idx",
    "phi_dist",
    "same_traj",
    "dt_if_same_traj",
    "local_support",
    "y",
    "weight",
    "split",
]


def trajectory_index_from_terminals(terminals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    terminals = np.asarray(terminals).astype(bool)
    if terminals.size:
        terminals[-1] = True
    traj_id = np.zeros(len(terminals), dtype=np.int32)
    timestep = np.zeros(len(terminals), dtype=np.int32)
    tid = 0
    t = 0
    for i in range(len(terminals)):
        traj_id[i] = tid
        timestep[i] = t
        if terminals[i]:
            tid += 1
            t = 0
        else:
            t += 1
    return traj_id, timestep


def _sample_positive(traj_starts: np.ndarray, traj_ends: np.ndarray, n: int, horizon: int, rng: np.random.Generator):
    u = np.empty(n, dtype=np.int64)
    v = np.empty(n, dtype=np.int64)
    dt = np.empty(n, dtype=np.int32)
    valid = np.where(traj_ends - traj_starts > 1)[0]
    if len(valid) == 0:
        raise ValueError("No trajectory has enough length to sample positive reachability pairs")
    for i in range(n):
        k = int(rng.choice(valid))
        s, e = int(traj_starts[k]), int(traj_ends[k])
        a = int(rng.integers(s, e - 1))
        max_dt = min(horizon, e - a - 1)
        d = int(rng.integers(1, max_dt + 1))
        u[i] = a
        v[i] = a + d
        dt[i] = d
    return u, v, dt


def _sample_cross_near(phis: np.ndarray, traj_id: np.ndarray, n: int, way_steps: float, rng: np.random.Generator):
    try:
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=min(64, len(phis))).fit(phis)
        anchors = rng.integers(0, len(phis), size=n)
        neigh = nn.kneighbors(phis[anchors], return_distance=False)
        u = np.empty(n, dtype=np.int64)
        v = np.empty(n, dtype=np.int64)
        filled = 0
        for row, a in zip(neigh, anchors):
            choices = [int(j) for j in row if traj_id[int(j)] != traj_id[int(a)]]
            if choices:
                u[filled] = int(a)
                v[filled] = int(rng.choice(choices))
                filled += 1
            if filled == n:
                break
        if filled:
            return u[:filled], v[:filled]
    except Exception:
        pass
    u = rng.integers(0, len(phis), size=n, dtype=np.int64)
    v = rng.integers(0, len(phis), size=n, dtype=np.int64)
    mask = traj_id[u] != traj_id[v]
    return u[mask], v[mask]


def build_reachability_pairs(
    phis: np.ndarray,
    terminals: np.ndarray,
    way_steps: float = 8.0,
    h_exec: Optional[int] = None,
    max_pairs: int = 500_000,
    val_frac: float = 0.1,
    seed: int = 0,
    edge_pairs: Optional[np.ndarray] = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    phis = np.asarray(phis, dtype=np.float32)
    traj_id, timestep = trajectory_index_from_terminals(terminals)
    h_exec = int(h_exec or max(1, round(float(way_steps))))
    max_pairs = int(max_pairs)
    n_pos = max_pairs // 2
    n_neg = max_pairs - n_pos

    terminal_locs = np.flatnonzero(np.asarray(terminals).astype(bool))
    if len(terminal_locs) == 0 or terminal_locs[-1] != len(phis) - 1:
        terminal_locs = np.concatenate([terminal_locs, [len(phis) - 1]])
    starts = np.concatenate([[0], terminal_locs[:-1] + 1])
    ends = terminal_locs + 1
    pos_u, pos_v, pos_dt = _sample_positive(starts, ends, n_pos, h_exec, rng)

    neg_u, neg_v = _sample_cross_near(phis, traj_id, n_neg * 2, way_steps, rng)
    if len(neg_u) > n_neg:
        keep = rng.choice(len(neg_u), size=n_neg, replace=False)
        neg_u, neg_v = neg_u[keep], neg_v[keep]
    neg_dt = np.full(len(neg_u), -1, dtype=np.int32)

    rows_u = [pos_u, neg_u]
    rows_v = [pos_v, neg_v]
    rows_y = [np.ones(len(pos_u), dtype=np.float32), np.zeros(len(neg_u), dtype=np.float32)]
    rows_dt = [pos_dt, neg_dt]
    rows_weight = [np.exp(-pos_dt / max(h_exec, 1)).astype(np.float32), np.ones(len(neg_u), dtype=np.float32)]

    if edge_pairs is not None and len(edge_pairs):
        edge_pairs = np.asarray(edge_pairs, dtype=np.int64)
        e_u = edge_pairs[:, 0]
        e_v = edge_pairs[:, 1]
        rows_u.append(e_u)
        rows_v.append(e_v)
        rows_y.append(np.full(len(e_u), -1.0, dtype=np.float32))
        rows_dt.append(np.full(len(e_u), -1, dtype=np.int32))
        rows_weight.append(np.full(len(e_u), 0.25, dtype=np.float32))

    u = np.concatenate(rows_u)
    v = np.concatenate(rows_v)
    y = np.concatenate(rows_y)
    dt = np.concatenate(rows_dt)
    weight = np.concatenate(rows_weight)
    phi_dist = np.linalg.norm(phis[v] - phis[u], axis=1).astype(np.float32)
    same_traj = (traj_id[u] == traj_id[v]).astype(np.int32)
    local_support = ((same_traj == 1) & (dt > 0) & (dt <= h_exec)).astype(np.int32)
    split = np.where(rng.random(len(u)) < val_frac, "val", "train")

    df = pd.DataFrame(
        {
            "u_idx": u,
            "v_idx": v,
            "phi_dist": phi_dist,
            "same_traj": same_traj,
            "dt_if_same_traj": dt,
            "local_support": local_support,
            "y": y,
            "weight": weight,
            "split": split,
        }
    )
    meta = {
        "num_pairs": int(len(df)),
        "num_positive": int((df["y"] == 1).sum()),
        "num_negative": int((df["y"] == 0).sum()),
        "num_unlabeled": int((df["y"] < 0).sum()),
        "h_exec": h_exec,
        "way_steps": float(way_steps),
        "val_frac": float(val_frac),
    }
    return df, meta


def save_pairs(df: pd.DataFrame, phis: np.ndarray, out_dir: str | os.PathLike[str], stem: str = "reachability_pairs") -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    npz_path = out / f"{stem}.npz"
    arrays = {col: df[col].to_numpy() for col in df.columns}
    arrays["phis"] = np.asarray(phis, dtype=np.float32)
    np.savez_compressed(npz_path, **arrays)
    paths["npz"] = str(npz_path)
    csv_path = out / f"{stem}.csv"
    df.to_csv(csv_path, index=False)
    paths["csv"] = str(csv_path)
    try:
        pq_path = out / f"{stem}.parquet"
        df.to_parquet(pq_path, index=False)
        paths["parquet"] = str(pq_path)
    except Exception:
        pass
    return paths


def load_pairs(path: str | os.PathLike[str]) -> tuple[pd.DataFrame, Optional[np.ndarray]]:
    path = Path(path)
    if path.suffix == ".npz":
        d = np.load(path, allow_pickle=True)
        data = {col: d[col] for col in PAIR_COLUMNS if col in d}
        return pd.DataFrame(data), d["phis"] if "phis" in d else None
    if path.suffix == ".parquet":
        return pd.read_parquet(path), None
    return pd.read_csv(path), None


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", required=True)
    p.add_argument("--terminals", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--way-steps", type=float, default=8.0)
    p.add_argument("--h-exec", type=int, default=0)
    p.add_argument("--max-pairs", type=int, default=500000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    phis = np.load(args.embeddings)
    terminals = np.load(args.terminals)
    df, meta = build_reachability_pairs(
        phis,
        terminals,
        way_steps=args.way_steps,
        h_exec=args.h_exec or None,
        max_pairs=args.max_pairs,
        seed=args.seed,
    )
    paths = save_pairs(df, phis, args.out)
    with open(Path(args.out) / "reachability_pairs_meta.json", "w") as f:
        json.dump({"paths": paths, **meta}, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
