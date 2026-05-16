from __future__ import annotations

"""Goal samplers for full BARS low-level training.

The key change from vanilla GCBC is that the low-level policy is exposed to the
same subgoal distribution that the graph planner will use at evaluation time:
future states, BARS graph nodes, bottleneck nodes, hard edge endpoints, and a
small amount of long-horizon/final-goal relabeling.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from bars.data.trajectories import OfflineDataset
from bars.graph.nodes import select_graph_nodes


def _normalise_probs(items: list[tuple[str, float]]) -> list[tuple[str, float]]:
    cleaned = [(k, max(0.0, float(v))) for k, v in items]
    s = sum(v for _, v in cleaned)
    if s <= 0:
        return [("future", 1.0)]
    return [(k, v / s) for k, v in cleaned if v > 0]


@dataclass
class SampleBatch:
    obs_idx: np.ndarray
    goal_idx: np.ndarray
    kind: np.ndarray
    kind_names: Tuple[str, ...] = ()


class GraphAwareGoalSampler:
    def __init__(
        self,
        dataset: OfflineDataset,
        cfg: Dict,
        embeddings: Optional[np.ndarray] = None,
        run_dir: Optional[str] = None,
        logger=None,
    ):
        self.dataset = dataset
        self.cfg = cfg
        self.embeddings = embeddings
        self.pcfg = cfg.get("policy", {})
        self.seed = int(cfg.get("seed", 0)) + 733
        self.rng = np.random.default_rng(self.seed)
        self.horizon = int(self.pcfg.get("horizon", cfg.get("reachability", {}).get("horizon", 30)))
        self.final_goal_horizon = int(self.pcfg.get("final_goal_horizon", max(self.horizon * 4, self.horizon + 1)))
        self.graph_nodes = self._load_or_build_graph_nodes(logger)
        self.bottleneck_nodes = self._bottleneck_subset()
        self.hard_edges = self._load_hard_edges(run_dir)
        self.probs = _normalise_probs([
            ("future", self.pcfg.get("future_goal_frac", 0.45)),
            ("graph", self.pcfg.get("graph_goal_frac", 0.25)),
            ("bottleneck", self.pcfg.get("bottleneck_goal_frac", 0.15)),
            ("hard_edge", self.pcfg.get("hard_edge_goal_frac", 0.05)),
            ("final", self.pcfg.get("final_goal_frac", 0.10)),
        ])
        self.kind_names = tuple(k for k, _ in self.probs)
        self.kind_probs = np.asarray([v for _, v in self.probs], dtype=np.float64)
        self.kind_probs = self.kind_probs / max(float(self.kind_probs.sum()), 1e-12)
        if logger is not None:
            logger.log({
                "phase": "bars_goal_sampler",
                "event": "initialized",
                "graph_nodes": int(len(self.graph_nodes)),
                "bottleneck_nodes": int(len(self.bottleneck_nodes)),
                "hard_edges": int(len(self.hard_edges[0])) if self.hard_edges is not None else 0,
                "goal_mix": ";".join(f"{k}:{v:.3f}" for k, v in self.probs),
            })

    def _load_or_build_graph_nodes(self, logger=None) -> np.ndarray:
        pcfg = self.pcfg
        path = str(pcfg.get("graph_node_indices_path", "") or "").strip()
        if path and Path(path).exists():
            return np.asarray(np.load(path), dtype=np.int64).reshape(-1)
        if self.embeddings is not None and bool(pcfg.get("build_graph_goal_nodes", True)):
            try:
                return select_graph_nodes(self.dataset, self.embeddings, self.cfg, logger).astype(np.int64)
            except Exception as exc:
                if logger is not None:
                    logger.log({"phase": "bars_goal_sampler", "event": "graph_node_build_failed", "error": repr(exc)})
        n = min(int(pcfg.get("num_graph_goal_nodes", 512)), self.dataset.size)
        return self.rng.choice(self.dataset.size, size=n, replace=False).astype(np.int64)

    def _bottleneck_subset(self) -> np.ndarray:
        if len(self.graph_nodes) == 0:
            return np.empty(0, dtype=np.int64)
        n = min(int(self.pcfg.get("num_bottleneck_goal_nodes", max(1, int(0.35 * len(self.graph_nodes))))), len(self.graph_nodes))
        # BARS node selection returns bottleneck-scored states first for the bars method.
        return np.asarray(self.graph_nodes[:n], dtype=np.int64)

    def _load_hard_edges(self, run_dir: Optional[str]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        paths = []
        if run_dir:
            paths.append(Path(run_dir) / "cache" / "hard_edges.npz")
            paths.append(Path(run_dir) / "cache" / "failed_edge_pairs.npz")
        extra = str(self.pcfg.get("hard_edge_cache", "") or "").strip()
        if extra:
            paths.append(Path(extra))
        for p in paths:
            if not p.exists():
                continue
            try:
                d = np.load(p)
                for sk, dk in [("src_index", "dst_index"), ("src", "dst"), ("i", "j")]:
                    if sk in d and dk in d:
                        src = np.asarray(d[sk], dtype=np.int64).reshape(-1)
                        dst = np.asarray(d[dk], dtype=np.int64).reshape(-1)
                        ok = (src >= 0) & (src < self.dataset.size) & (dst >= 0) & (dst < self.dataset.size)
                        if ok.any():
                            return src[ok], dst[ok]
            except Exception:
                pass
        return None

    def _sample_future(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        i, j, _ = self.dataset.sample_future_pairs(n, self.horizon, self.rng)
        return i, j

    def _sample_final(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        try:
            i, j, _ = self.dataset.sample_future_pairs(n, self.final_goal_horizon, self.rng, min_dt=max(1, self.horizon))
            return i, j
        except Exception:
            return self._sample_future(n)

    def _sample_random_to_pool(self, n: int, pool: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(pool) == 0:
            return self._sample_future(n)
        i = self.dataset.sample_indices(n, self.rng)
        j = self.rng.choice(pool, size=n, replace=len(pool) < n).astype(np.int64)
        return i, j

    def _sample_hard_edges(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.hard_edges is None or len(self.hard_edges[0]) == 0:
            return self._sample_random_to_pool(n, self.graph_nodes)
        src, dst = self.hard_edges
        ids = self.rng.choice(len(src), size=n, replace=len(src) < n)
        return src[ids].astype(np.int64), dst[ids].astype(np.int64)

    def format_mix(self, kind_codes: np.ndarray) -> str:
        codes = np.asarray(kind_codes, dtype=np.int64)
        counts = np.bincount(codes, minlength=len(self.kind_names))
        return ";".join(f"{name}:{int(counts[i])}" for i, name in enumerate(self.kind_names) if counts[i] > 0)

    def sample(self, batch_size: int) -> SampleBatch:
        batch_size = int(batch_size)
        counts = self.rng.multinomial(batch_size, self.kind_probs)
        obs_parts = []
        goal_parts = []
        kind_parts = []
        for code, n in enumerate(counts):
            n = int(n)
            if n <= 0:
                continue
            kind = self.kind_names[code]
            if kind == "future":
                i, j = self._sample_future(n)
            elif kind == "graph":
                i, j = self._sample_random_to_pool(n, self.graph_nodes)
            elif kind == "bottleneck":
                i, j = self._sample_random_to_pool(n, self.bottleneck_nodes)
            elif kind == "hard_edge":
                i, j = self._sample_hard_edges(n)
            elif kind == "final":
                i, j = self._sample_final(n)
            else:
                i, j = self._sample_future(n)
            obs_parts.append(i)
            goal_parts.append(j)
            kind_parts.append(np.full(n, code, dtype=np.int16))
        obs = np.concatenate(obs_parts).astype(np.int64)
        goal = np.concatenate(goal_parts).astype(np.int64)
        kind_arr = np.concatenate(kind_parts)
        perm = self.rng.permutation(len(obs))
        return SampleBatch(obs[perm], goal[perm], kind_arr[perm], self.kind_names)
