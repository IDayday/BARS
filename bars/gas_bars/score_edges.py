from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from bars.external.gas_artifacts import resolve_gas_artifacts
from bars.external.gas_backbone import GASBackbone
from .graph_table import export_edges, export_nodes, load_gas_keygraph, save_edge_table
from .reachability_dataset import build_reachability_pairs, save_pairs, trajectory_index_from_terminals
from .reachability_model import edge_score_diagnostics, load_reachability_model, score_edges, train_reachability


def _read_terminals(backbone: GASBackbone) -> np.ndarray:
    _, train_dataset, _ = backbone.load_env_and_dataset()
    if "terminals" in train_dataset:
        terminals = np.asarray(train_dataset["terminals"]).astype(bool)
    else:
        terminals = np.zeros(len(train_dataset["observations"]), dtype=bool)
        terminals[-1] = True
    return terminals


def _edge_support_counts(
    node_phis: np.ndarray,
    edges: pd.DataFrame,
    dataset_phis: np.ndarray,
    terminals: np.ndarray,
    way_steps: float,
    k: int = 64,
    radius: float = 0.0,
    horizon: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Count same-trajectory offline support for graph edges.

    For every graph edge u->v, look at nearby dataset states around u and v
    and count ordered pairs from the same trajectory within the execution
    horizon.  This is deliberately lightweight and only used as a conservative
    penalty signal when the MLP is overconfident on graph shortcuts.
    """
    n_edges = len(edges)
    if n_edges == 0 or len(dataset_phis) == 0:
        return np.zeros(n_edges, dtype=np.int32), np.zeros(n_edges, dtype=np.int32)
    k = int(max(1, min(k, len(dataset_phis))))
    support_radius = float(radius or max(way_steps, 1e-6))
    h_exec = int(horizon or max(1, round(way_steps * 2)))
    traj_id, timestep = trajectory_index_from_terminals(terminals)
    try:
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=k).fit(dataset_phis)
        neigh = nn.kneighbors(node_phis, return_distance=True)
        dists, indices = neigh
    except Exception:
        sample = np.linspace(0, len(dataset_phis) - 1, num=min(len(dataset_phis), 20000), dtype=np.int64)
        sub = dataset_phis[sample]
        d = np.linalg.norm(node_phis[:, None, :] - sub[None, :, :], axis=-1)
        order = np.argsort(d, axis=1)[:, :k]
        indices = sample[order]
        dists = np.take_along_axis(d, order, axis=1)

    near = []
    for row in range(len(node_phis)):
        mask = dists[row] <= support_radius
        vals = indices[row][mask]
        if len(vals) == 0:
            vals = indices[row][: min(k, 8)]
        near.append(np.asarray(vals, dtype=np.int64))

    local = np.zeros(n_edges, dtype=np.int32)
    same_traj_counts = np.zeros(n_edges, dtype=np.int32)
    for i, r in enumerate(edges[["u", "v"]].itertuples(index=False)):
        u = int(r.u)
        v = int(r.v)
        if u < 0 or v < 0 or u >= len(near) or v >= len(near):
            continue
        a = near[u]
        b = near[v]
        if len(a) == 0 or len(b) == 0:
            continue
        # k is small, so this cross product stays bounded and avoids O(N).
        ta = traj_id[a][:, None]
        tb = traj_id[b][None, :]
        dt = timestep[b][None, :] - timestep[a][:, None]
        supported = (ta == tb) & (dt > 0) & (dt <= h_exec)
        count = int(supported.sum())
        same_traj_counts[i] = count
        local[i] = int(count > 0)
    return local, same_traj_counts


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gas-artifact-root", default="artifacts/gas")
    p.add_argument("--gas-repo-path", default="external_src/GAS")
    p.add_argument("--out", required=True)
    p.add_argument("--quick", type=int, default=1)
    p.add_argument("--gpu", default="0")
    p.add_argument("--device", default="cuda")
    p.add_argument("--prefer-pretrained", type=int, default=1)
    p.add_argument("--train-if-missing", type=int, default=1)
    p.add_argument("--max-pairs", type=int, default=0)
    p.add_argument("--support-k", type=int, default=64)
    p.add_argument("--support-radius", type=float, default=0.0)
    p.add_argument("--support-horizon", type=int, default=0)
    p.add_argument("--unsupported-penalty", type=float, default=0.15)
    args = p.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    artifacts = resolve_gas_artifacts(args.env, args.seed, args.gas_artifact_root)
    if not artifacts.complete or artifacts.dataset_embeddings is None:
        backbone = GASBackbone.load_or_train(
            args.env,
            args.seed,
            args.gas_artifact_root,
            args.gas_repo_path,
            args.gpu,
            prefer_pretrained=bool(args.prefer_pretrained),
            train_if_missing=bool(args.train_if_missing),
            quick=bool(args.quick),
        )
        artifacts = backbone.artifacts or resolve_gas_artifacts(args.env, args.seed, args.gas_artifact_root)
        if artifacts.dataset_embeddings is None:
            emb_path = artifacts.features_dir / "dataset_embeddings.npy"
            backbone.export_dataset_embeddings(emb_path)
            artifacts = resolve_gas_artifacts(args.env, args.seed, args.gas_artifact_root)
    else:
        backbone = GASBackbone(env_name=args.env, seed=args.seed, artifact_root=Path(args.gas_artifact_root), gas_repo_path=Path(args.gas_repo_path), gpu=args.gpu, artifacts=artifacts)

    if artifacts.keygraph is None or artifacts.dataset_embeddings is None:
        raise RuntimeError(f"Missing GAS keygraph or embeddings for {args.env} seed {args.seed}: {artifacts.to_dict()}")
    key_graph = load_gas_keygraph(artifacts.keygraph)
    nodes = export_nodes(key_graph)
    edges = export_edges(key_graph)
    save_edge_table(nodes, edges, out / "graph")

    terminals = _read_terminals(backbone)
    phis = np.load(artifacts.dataset_embeddings)
    max_pairs = args.max_pairs or (500_000 if args.quick else 2_000_000)
    pairs_path = out / "reachability_pairs.npz"
    if not pairs_path.exists():
        df, meta = build_reachability_pairs(
            phis,
            terminals=terminals,
            way_steps=float(getattr(key_graph, "way_steps", 8.0) or 8.0),
            max_pairs=max_pairs,
            seed=args.seed,
        )
        paths = save_pairs(df, phis, out)
        with open(out / "reachability_pairs_meta.json", "w") as f:
            json.dump({"paths": paths, **meta}, f, indent=2, sort_keys=True)
    else:
        d = np.load(pairs_path, allow_pickle=True)
        df = pd.DataFrame({k: d[k] for k in d.files if k != "phis"})

    model_path = out / "model.pt"
    device = args.device
    if model_path.exists():
        model = load_reachability_model(model_path, device=device)
        metrics = {}
        for metrics_path in (out / "calibration.json", out / "reachability_metrics.json"):
            if metrics_path.exists():
                try:
                    loaded = json.load(open(metrics_path))
                    if "metrics" in loaded and isinstance(loaded["metrics"], dict):
                        metrics.update(loaded["metrics"])
                    else:
                        metrics.update({k: v for k, v in loaded.items() if not k.startswith(("p_exec_", "r_exec_", "local_support", "same_traj_support"))})
                except Exception:
                    pass
    else:
        model, metrics = train_reachability(df, phis, out, quick=bool(args.quick), device=device)

    node_phis = nodes[[c for c in nodes.columns if c.startswith("phi_")]].to_numpy(np.float32)
    scored = score_edges(model, node_phis, edges, device=device)
    local_support, same_traj_support = _edge_support_counts(
        node_phis,
        edges,
        phis,
        terminals,
        way_steps=float(getattr(key_graph, "way_steps", 8.0) or 8.0),
        k=args.support_k,
        radius=args.support_radius,
        horizon=args.support_horizon,
    )
    scored["r_exec_model"] = scored["r_exec"]
    scored["local_support"] = local_support
    scored["same_traj_support"] = same_traj_support
    scored["support_penalty"] = np.where(local_support > 0, 0.0, float(args.unsupported_penalty))
    scored["r_exec"] = scored["r_exec_model"] + scored["support_penalty"]
    scored_csv = out / "edge_scores.csv"
    scored.to_csv(scored_csv, index=False)
    try:
        scored.to_parquet(out / "edge_scores.parquet", index=False)
    except Exception:
        pass
    summary = edge_score_diagnostics(scored)
    summary.update(metrics)
    with open(out / "reachability_metrics.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps({"edge_scores": str(scored_csv), "metrics": str(out / "reachability_metrics.json")}, indent=2))


if __name__ == "__main__":
    main()
