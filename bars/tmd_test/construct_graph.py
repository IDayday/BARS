from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .calibration import calibrate_tmd_scales
from .diagnostics import write_failure
from .io import ensure_dir, load_raw_ogbench_npz, terminal_goal_observations, write_json
from .keygraph_tmd import TMDKeyGraph
from .keynodes_tmd import TMDKeyNodes
from .repr_provider import TMDRepresentationProvider
from .tmd_agent_adapter import TMDAgentAdapter


def _load_config(path: str | None) -> dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve(value, cfg, key, default=None):
    return value if value is not None else cfg.get(key, default)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Construct a directed tmd-test key graph.")
    parser.add_argument("--config")
    parser.add_argument("--env")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--tmd-checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--dataset-root")
    parser.add_argument("--tmd-repo", default="external_src/tmd-release")
    parser.add_argument("--temporal-horizon-steps", type=int)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--edge-quantile", type=float, default=0.75)
    parser.add_argument("--target-quantile", type=float, default=0.90)
    parser.add_argument("--repr-cluster-threshold", type=float)
    parser.add_argument("--te-threshold", type=float)
    parser.add_argument("--pairwise-batch-size", type=int)
    parser.add_argument("--max-observations", type=int)
    parser.add_argument("--max-nodes", type=int)
    parser.add_argument("--topk-l2-candidates", type=int)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    cfg = _load_config(args.config)
    env_name = _resolve(args.env, cfg, "env", "antmaze-medium-stitch-v0")
    seed = int(_resolve(args.seed, cfg, "seed", 0))
    output_dir = Path(_resolve(args.output_dir, cfg, "output_dir", f"artifacts/tmd_test/{env_name}/{seed}"))
    ensure_dir(output_dir)
    write_json(output_dir / "construct_args.json", {"args": vars(args), "config_values": cfg})
    try:
        checkpoint = _resolve(args.tmd_checkpoint, cfg, "tmd_checkpoint")
        if not checkpoint:
            raise FileNotFoundError("--tmd-checkpoint is required")
        dataset_root = _resolve(args.dataset_root, cfg, "dataset_root", "/mnt/project/offlinerl_datasets/ogbench")
        max_obs = int(_resolve(args.max_observations, cfg, "max_observations", 2048 if args.quick else 50000))
        sample_size = int(_resolve(args.sample_size, cfg, "sample_size", 512 if args.quick else 50000))
        pairwise_batch = int(_resolve(args.pairwise_batch_size, cfg, "pairwise_batch_size", 256))
        max_nodes = int(_resolve(args.max_nodes, cfg, "max_nodes", 64 if args.quick else 256))
        H = int(_resolve(args.temporal_horizon_steps, cfg, "temporal_horizon_steps", 8))
        te = float(_resolve(args.te_threshold, cfg, "te_threshold", 0.99))
        data = load_raw_ogbench_npz(env_name, dataset_root=dataset_root, max_observations=max_obs)
        observations = np.asarray(data["observations"], dtype=np.float32)
        terminals = np.asarray(data["terminals"]).astype(bool)
        tmd = TMDAgentAdapter.restore(
            checkpoint,
            {"env_name": env_name, "seed": seed, "dataset_root": dataset_root, "tmd_repo": args.tmd_repo},
        )
        provider = TMDRepresentationProvider(tmd, batch_size=pairwise_batch)
        calibration = calibrate_tmd_scales(
            observations,
            terminals,
            provider,
            H,
            sample_size=sample_size,
            seed=seed,
            edge_quantile=float(args.edge_quantile),
            target_quantile=float(args.target_quantile),
        )
        auto_repr_threshold = float(calibration["repr_cluster_threshold"])
        repr_cluster_threshold = float(
            _resolve(args.repr_cluster_threshold, cfg, "repr_cluster_threshold", auto_repr_threshold)
        )
        calibration["repr_cluster_threshold_auto"] = auto_repr_threshold
        calibration["repr_cluster_threshold"] = repr_cluster_threshold
        calibration["repr_cluster_threshold_override"] = bool(
            args.repr_cluster_threshold is not None or "repr_cluster_threshold" in cfg
        )
        write_json(output_dir / "tmd_calibration.json", calibration)
        embeddings = provider.encode(observations)
        keynodes = TMDKeyNodes(max_nodes=max_nodes).construct(
            embeddings,
            observations,
            terminals,
            repr_cluster_threshold,
            H,
            te_threshold=te,
        )
        keynodes.save(output_dir / "keynodes_tmd.pkl")
        graph = TMDKeyGraph().construct_graph(
            keynodes,
            provider,
            float(calibration["edge_distance_threshold"]),
            pairwise_batch_size=pairwise_batch,
            topk_l2_candidates=args.topk_l2_candidates,
        )
        goals = terminal_goal_observations(env_name, dataset_root=dataset_root, max_goals=int(cfg.get("tasks", 5)))
        for task_id, goal_obs in enumerate(goals, start=1):
            graph.add_task_goal(task_id, goal_obs, provider, float(calibration["target_distance_threshold"]))
            graph.precompute_shortest_paths_to_goal(task_id)
        graph.save(output_dir / "keygraph_tmd.pkl")
        stats = graph.graph_stats()
        stats.update(
            {
                "env": env_name,
                "seed": seed,
                "dataset_path": data["dataset_path"],
                "checkpoint": str(checkpoint),
                "num_observations_used": int(len(observations)),
            }
        )
        write_json(output_dir / "graph_stats.json", stats)
        return 0
    except Exception as exc:
        write_failure(output_dir / "construct_error.json", "construct_graph", exc, {"env": env_name, "seed": seed})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
