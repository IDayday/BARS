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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bars.external.gas_artifacts import resolve_gas_artifacts  # noqa: E402
from bars.gas_bars.graph_table import export_edges, export_nodes, load_gas_keygraph, save_edge_table  # noqa: E402
from bars.gas_bars.reachability_dataset import trajectory_index_from_terminals  # noqa: E402


def _load_terminals(env_name: str, dataset_dir: str | None) -> np.ndarray:
    try:
        import ogbench
    except Exception as exc:
        raise RuntimeError(
            "Failed to import ogbench. Set PYTHONPATH to the local OGBench source, "
            "for example external_src/tmd-release."
        ) from exc
    kwargs: dict[str, Any] = {"compact_dataset": False}
    if dataset_dir:
        kwargs["dataset_dir"] = dataset_dir
    _, train_dataset, _ = ogbench.make_env_and_datasets(env_name, **kwargs)
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
    *,
    way_steps: float,
    k: int,
    radius: float,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
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
        dists, indices = nn.kneighbors(node_phis, return_distance=True)
    except Exception:
        sample = np.linspace(0, len(dataset_phis) - 1, num=min(len(dataset_phis), 20000), dtype=np.int64)
        sub = dataset_phis[sample]
        d = np.linalg.norm(node_phis[:, None, :] - sub[None, :, :], axis=-1)
        order = np.argsort(d, axis=1)[:, :k]
        indices = sample[order]
        dists = np.take_along_axis(d, order, axis=1)

    near: list[np.ndarray] = []
    for row in range(len(node_phis)):
        mask = dists[row] <= support_radius
        vals = indices[row][mask]
        if len(vals) == 0:
            vals = indices[row][: min(k, 8)]
        near.append(np.asarray(vals, dtype=np.int64))

    local = np.zeros(n_edges, dtype=np.int32)
    same_traj_counts = np.zeros(n_edges, dtype=np.int32)
    for i, row in enumerate(edges[["u", "v"]].itertuples(index=False)):
        u = int(row.u)
        v = int(row.v)
        if u < 0 or v < 0 or u >= len(near) or v >= len(near):
            continue
        a = near[u]
        b = near[v]
        if len(a) == 0 or len(b) == 0:
            continue
        same_traj = traj_id[a][:, None] == traj_id[b][None, :]
        dt = timestep[b][None, :] - timestep[a][:, None]
        supported = same_traj & (dt > 0) & (dt <= h_exec)
        count = int(supported.sum())
        same_traj_counts[i] = count
        local[i] = int(count > 0)
    return local, same_traj_counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create GAS edge_scores.csv with temporal support counts only; no policy or reachability training."
    )
    parser.add_argument("--env", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gas-artifact-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset-dir", default=os.environ.get("OGBENCH_DATASET_DIR", ""))
    parser.add_argument("--support-k", type=int, default=64)
    parser.add_argument("--support-radius", type=float, default=0.0)
    parser.add_argument("--support-horizon", type=int, default=0)
    parser.add_argument("--unsupported-penalty", type=float, default=0.15)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    artifacts = resolve_gas_artifacts(args.env, args.seed, args.gas_artifact_root)
    if artifacts.keygraph is None:
        raise RuntimeError(f"Missing keygraph under {artifacts.graph_dir}")
    if artifacts.dataset_embeddings is None:
        raise RuntimeError(f"Missing dataset embeddings under {artifacts.features_dir}")

    key_graph = load_gas_keygraph(artifacts.keygraph)
    nodes = export_nodes(key_graph)
    edges = export_edges(key_graph)
    save_edge_table(nodes, edges, out / "graph")

    terminals = _load_terminals(args.env, args.dataset_dir or None)
    phis = np.load(artifacts.dataset_embeddings)
    if len(phis) != len(terminals):
        raise RuntimeError(f"Embedding/terminal length mismatch: {len(phis)} vs {len(terminals)}")
    node_phis = nodes[[c for c in nodes.columns if c.startswith("phi_")]].to_numpy(np.float32)
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

    scored = edges.copy()
    scored["r_exec_model"] = 0.0
    scored["local_support"] = local_support
    scored["same_traj_support"] = same_traj_support
    scored["unsupported_scc_connector"] = (
        (scored["edge_source"] == "gas_scc_connector") & (scored["local_support"] <= 0)
    ).astype(np.int32)
    scored["scc_only_support"] = np.where(scored["edge_source"] == "gas_scc_connector", scored["local_support"], 1)
    scored["support_penalty"] = np.where(local_support > 0, 0.0, float(args.unsupported_penalty))
    scored["r_exec"] = scored["support_penalty"]
    scored_path = out / "edge_scores.csv"
    scored.to_csv(scored_path, index=False)
    try:
        scored.to_parquet(out / "edge_scores.parquet", index=False)
    except Exception:
        pass

    non_goal = scored[scored["edge_source"] != "gas_goal_connector"]
    summary = {
        "env": args.env,
        "seed": int(args.seed),
        "keygraph": str(artifacts.keygraph),
        "dataset_embeddings": str(artifacts.dataset_embeddings),
        "num_edges": int(len(scored)),
        "num_goal_connector_edges": int((scored["edge_source"] == "gas_goal_connector").sum()),
        "supported_edge_rate": float(scored["local_support"].mean()) if len(scored) else 0.0,
        "non_goal_supported_edge_rate": float(non_goal["local_support"].mean()) if len(non_goal) else 0.0,
        "num_scc_connector_edges": int((scored["edge_source"] == "gas_scc_connector").sum()),
        "unsupported_scc_connector_rate": float(scored["unsupported_scc_connector"].mean()) if len(scored) else 0.0,
        "scc_only_supported_edge_rate": float(scored["scc_only_support"].mean()) if len(scored) else 0.0,
        "mean_same_traj_support": float(scored["same_traj_support"].mean()) if len(scored) else 0.0,
        "support_k": int(args.support_k),
        "support_radius": float(args.support_radius),
        "support_horizon": int(args.support_horizon),
        "unsupported_penalty": float(args.unsupported_penalty),
    }
    summary_path = out / "support_only_metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"edge_scores": str(scored_path), "metrics": str(summary_path), **summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
