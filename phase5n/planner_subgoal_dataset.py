from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:  # pragma: no cover - minimal analysis envs may omit torch.
    torch = None
    Dataset = object  # type: ignore[misc,assignment]

from phase3.train_gcbc import edge_loss_weight_values


TARGET_SOURCE_TO_ID = {
    "final_goal_hindsight": 0,
    "support_edge_local": 1,
    "planner_first_edge_replay": 2,
}
TARGET_ID_TO_SOURCE = {value: key for key, value in TARGET_SOURCE_TO_ID.items()}


def load_npz_arrays(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path).expanduser()) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def episode_bounds_from_terminals(terminals: np.ndarray, n_observations: int) -> np.ndarray:
    terminals = np.asarray(terminals).reshape(-1).astype(bool)
    n = min(int(n_observations), int(terminals.shape[0]))
    starts: list[int] = []
    ends: list[int] = []
    start = 0
    for idx in range(n):
        if bool(terminals[idx]):
            end = idx + 1
            if end - start >= 2:
                starts.append(start)
                ends.append(end)
            start = end
    if start < n and n - start >= 2:
        starts.append(start)
        ends.append(n)
    if not starts and n >= 2:
        starts.append(0)
        ends.append(n)
    return np.asarray(list(zip(starts, ends)), dtype=np.int64)


def _row_cost(row: Any) -> float:
    for col in ("cost", "median_h", "mean_h", "max_h"):
        value = getattr(row, col, None)
        if value is not None:
            try:
                f = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(f):
                return f
    return 1.0


def build_edge_planning_graph(option_edges: pd.DataFrame) -> nx.DiGraph:
    graph = nx.DiGraph()
    if option_edges.empty:
        return graph
    for row in option_edges.itertuples(index=False):
        src = int(getattr(row, "src"))
        dst = int(getattr(row, "dst"))
        edge_id = int(getattr(row, "edge_id"))
        cost = _row_cost(row)
        existing = graph.get_edge_data(src, dst)
        if existing is None or cost < float(existing.get("cost", np.inf)):
            graph.add_edge(src, dst, edge_id=edge_id, cost=cost)
    return graph


def _edge_sampling_weights(option_edges: pd.DataFrame) -> np.ndarray:
    if option_edges.empty:
        return np.empty(0, dtype=np.float64)
    for col in ("num_unique_starts", "num_segments", "support_count"):
        if col in option_edges.columns:
            weights = pd.to_numeric(option_edges[col], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(float)
            if float(weights.sum()) > 0.0:
                return weights / float(weights.sum())
    return np.ones(option_edges.shape[0], dtype=np.float64) / float(option_edges.shape[0])


def _path_edge_ids(graph: nx.DiGraph, node_path: list[int]) -> list[int]:
    return [int(graph[node_path[i]][node_path[i + 1]]["edge_id"]) for i in range(len(node_path) - 1)]


def sample_support_graph_planner_paths(
    option_edges: pd.DataFrame,
    *,
    num_queries: int = 5000,
    seed: int = 0,
    queries: Iterable[tuple[int, int]] | None = None,
) -> pd.DataFrame:
    """Sample graph queries and record support-only shortest paths.

    This is reset-free planner target generation. It never creates new graph
    edges; every returned edge id is already present in ``option_edges``.
    """

    graph = build_edge_planning_graph(option_edges)
    if option_edges.empty or graph.number_of_edges() == 0:
        return pd.DataFrame(
            columns=[
                "query_id",
                "start_cluster",
                "goal_cluster",
                "reachable",
                "path_num_edges",
                "path_cost",
                "path_edge_ids",
                "first_edge_id",
            ]
        )

    rng = np.random.default_rng(int(seed))
    edges = option_edges.reset_index(drop=True).copy()
    edge_weights = _edge_sampling_weights(edges)
    if queries is None:
        sampled_queries: list[tuple[int, int]] = []
        for _ in range(int(num_queries)):
            start_row = edges.iloc[int(rng.choice(edges.index.to_numpy(), p=edge_weights))]
            goal_row = edges.iloc[int(rng.choice(edges.index.to_numpy(), p=edge_weights))]
            sampled_queries.append((int(start_row["src"]), int(goal_row["dst"])))
    else:
        sampled_queries = [(int(src), int(dst)) for src, dst in queries]

    rows: list[dict[str, Any]] = []
    for query_id, (start, goal) in enumerate(sampled_queries):
        if start == goal:
            rows.append(
                {
                    "query_id": int(query_id),
                    "start_cluster": int(start),
                    "goal_cluster": int(goal),
                    "reachable": True,
                    "path_num_edges": 0,
                    "path_cost": 0.0,
                    "path_edge_ids": "",
                    "first_edge_id": -1,
                }
            )
            continue
        try:
            node_path = [int(x) for x in nx.shortest_path(graph, start, goal, weight="cost")]
            edge_ids = _path_edge_ids(graph, node_path)
            path_cost = float(sum(float(graph[node_path[i]][node_path[i + 1]]["cost"]) for i in range(len(node_path) - 1)))
            rows.append(
                {
                    "query_id": int(query_id),
                    "start_cluster": int(start),
                    "goal_cluster": int(goal),
                    "reachable": True,
                    "path_num_edges": int(len(edge_ids)),
                    "path_cost": path_cost,
                    "path_edge_ids": " ".join(str(int(x)) for x in edge_ids),
                    "first_edge_id": int(edge_ids[0]) if edge_ids else -1,
                }
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            rows.append(
                {
                    "query_id": int(query_id),
                    "start_cluster": int(start),
                    "goal_cluster": int(goal),
                    "reachable": False,
                    "path_num_edges": 0,
                    "path_cost": np.nan,
                    "path_edge_ids": "",
                    "first_edge_id": -1,
                }
            )
    return pd.DataFrame(rows)


def _parse_path_edge_ids(value: Any) -> list[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).replace(",", " ").replace("[", " ").replace("]", " ").strip()
    if not text:
        return []
    out: list[int] = []
    for token in text.split():
        try:
            edge_id = int(float(token))
        except ValueError:
            continue
        if edge_id >= 0:
            out.append(edge_id)
    return out


def planner_edge_usage_from_paths(path_rows: pd.DataFrame) -> pd.DataFrame:
    path_counts: Counter[int] = Counter()
    first_counts: Counter[int] = Counter()
    if not path_rows.empty:
        reachable = path_rows[path_rows.get("reachable", False).astype(bool)] if "reachable" in path_rows else path_rows
        for value in reachable.get("path_edge_ids", pd.Series(dtype=object)).tolist():
            path_counts.update(_parse_path_edge_ids(value))
        for value in reachable.get("first_edge_id", pd.Series(dtype=float)).tolist():
            try:
                edge_id = int(value)
            except (TypeError, ValueError):
                continue
            if edge_id >= 0:
                first_counts[edge_id] += 1
    all_ids = sorted(set(path_counts) | set(first_counts))
    return pd.DataFrame(
        [
            {
                "edge_id": int(edge_id),
                "planner_usage_count": int(path_counts.get(edge_id, 0)),
                "planner_first_edge_count": int(first_counts.get(edge_id, 0)),
            }
            for edge_id in all_ids
        ]
    )


def build_planner_relevant_edge_weights(
    option_edges: pd.DataFrame,
    *,
    path_rows: pd.DataFrame | None = None,
    num_queries: int = 5000,
    seed: int = 0,
    base_loss_weight_mode: str = "support_bottleneck",
    base_loss_weight_strength: float = 0.3,
    base_loss_weight_min: float = 0.7,
    base_loss_weight_max: float = 1.8,
    planner_usage_strength: float = 0.35,
    planner_first_edge_strength: float = 0.65,
    min_weight: float = 0.5,
    max_weight: float = 2.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build edge weights from graph-planner target frequency.

    ``planner_first_edge_count`` is the main alignment term: it lifts edges that
    the runtime planner would issue as immediate subgoals. ``planner_usage`` is
    a weaker path-level term for downstream option edges.
    """

    edges = option_edges.copy().reset_index(drop=True)
    if "edge_id" not in edges.columns:
        raise ValueError("option_edges must contain edge_id")
    edges["edge_id"] = edges["edge_id"].astype(int)
    if path_rows is None:
        path_rows = sample_support_graph_planner_paths(edges, num_queries=num_queries, seed=seed)
    usage = planner_edge_usage_from_paths(path_rows)
    base = edge_loss_weight_values(
        edges,
        mode=base_loss_weight_mode,
        strength=base_loss_weight_strength,
        min_weight=base_loss_weight_min,
        max_weight=base_loss_weight_max,
    ).rename(columns={"loss_weight": "base_loss_weight"})
    out = edges[["edge_id"]].merge(base, on="edge_id", how="left").merge(usage, on="edge_id", how="left")
    out["base_loss_weight"] = pd.to_numeric(out["base_loss_weight"], errors="coerce").fillna(1.0)
    out["planner_usage_count"] = pd.to_numeric(out.get("planner_usage_count", 0), errors="coerce").fillna(0.0)
    out["planner_first_edge_count"] = pd.to_numeric(out.get("planner_first_edge_count", 0), errors="coerce").fillna(0.0)

    def _log_score(values: pd.Series) -> np.ndarray:
        arr = np.log1p(pd.to_numeric(values, errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(float))
        hi = float(arr.max(initial=0.0))
        return arr / hi if hi > 0.0 else np.zeros_like(arr)

    usage_score = _log_score(out["planner_usage_count"])
    first_score = _log_score(out["planner_first_edge_count"])
    out["planner_usage_score"] = usage_score
    out["planner_first_edge_score"] = first_score
    multiplier = 1.0 + float(planner_usage_strength) * usage_score + float(planner_first_edge_strength) * first_score
    raw = out["base_loss_weight"].to_numpy(float) * multiplier
    if raw.size and np.isfinite(raw).all() and float(raw.mean()) > 0.0:
        raw = raw / float(raw.mean())
    lo = min(float(min_weight), float(max_weight))
    hi = max(float(min_weight), float(max_weight))
    out["loss_weight"] = np.clip(raw, lo, hi)
    sampling = np.maximum(1e-6, out["loss_weight"].to_numpy(float) * (1.0 + first_score))
    out["planner_sampling_weight"] = sampling / float(sampling.mean()) if float(sampling.mean()) > 0.0 else 1.0
    return out[
        [
            "edge_id",
            "loss_weight",
            "planner_sampling_weight",
            "base_loss_weight",
            "planner_usage_count",
            "planner_first_edge_count",
            "planner_usage_score",
            "planner_first_edge_score",
        ]
    ], path_rows


@dataclass(frozen=True)
class MixedSampleIndex:
    source: str
    segment_index: int = -1
    offset: int = 0
    obs_index: int = -1
    goal_index: int = -1


class PlannerMixedGCBCDataset(Dataset):
    """Lazy BARS-native GCBC samples from final goals, support edges, and planner replay."""

    def __init__(
        self,
        dataset: dict[str, Any],
        option_edges: pd.DataFrame,
        edge_segments: dict[str, np.ndarray],
        planner_edge_weights: pd.DataFrame | None = None,
        *,
        max_examples: int | None = None,
        source_probabilities: dict[str, float] | None = None,
        source_loss_weights: dict[str, float] | None = None,
        seed: int = 0,
        segment_indices: np.ndarray | None = None,
        episode_indices: np.ndarray | None = None,
    ) -> None:
        if torch is None:  # pragma: no cover
            raise ImportError("PlannerMixedGCBCDataset requires PyTorch")
        self.observations = np.asarray(dataset["observations"], dtype=np.float32)
        self.actions = np.asarray(dataset["actions"], dtype=np.float32)
        if self.observations.ndim == 1:
            self.observations = self.observations.reshape(-1, 1)
        if self.actions.ndim == 1:
            self.actions = self.actions.reshape(-1, 1)
        self.option_edges = option_edges.copy()
        self.edge_segments = {key: np.asarray(value) for key, value in edge_segments.items()}
        self.planner_edge_weights = pd.DataFrame() if planner_edge_weights is None else planner_edge_weights.copy()
        self.max_examples = None if max_examples is None else int(max_examples)
        self.seed = int(seed)
        self.source_loss_weights = {
            "final_goal_hindsight": 1.0,
            "support_edge_local": 1.0,
            "planner_first_edge_replay": 1.0,
            **(source_loss_weights or {}),
        }

        required = {"edge_id", "global_i", "global_j", "h"}
        missing = required - set(self.edge_segments)
        if missing:
            raise KeyError(f"edge_segments is missing required arrays: {sorted(missing)}")
        n_segments = int(np.asarray(self.edge_segments["edge_id"]).shape[0])
        if segment_indices is None:
            segment_indices = np.arange(n_segments, dtype=np.int64)
        else:
            segment_indices = np.asarray(segment_indices, dtype=np.int64)
        if segment_indices.size and (int(segment_indices.min()) < 0 or int(segment_indices.max()) >= n_segments):
            raise IndexError("segment_indices out of range")

        raw_i = np.asarray(self.edge_segments["global_i"], dtype=np.int64)[segment_indices]
        raw_j = np.asarray(self.edge_segments["global_j"], dtype=np.int64)[segment_indices]
        raw_h = np.asarray(self.edge_segments["h"], dtype=np.int64)[segment_indices]
        raw_edge = np.asarray(self.edge_segments["edge_id"], dtype=np.int64)[segment_indices]
        lengths = np.minimum(raw_h, raw_j - raw_i).astype(np.int64)
        valid = (
            (lengths > 0)
            & (raw_i >= 0)
            & (raw_i < self.actions.shape[0])
            & (raw_j >= 0)
            & (raw_j < self.observations.shape[0])
            & (raw_i + lengths <= self.actions.shape[0])
        )
        self.segment_indices = segment_indices[valid]
        self.global_i = raw_i[valid]
        self.global_j = raw_j[valid]
        self.h = raw_h[valid]
        self.edge_id = raw_edge[valid]
        self.segment_lengths = lengths[valid]
        self.prefix_lengths = np.cumsum(self.segment_lengths, dtype=np.int64)
        self.total_transition_examples = int(self.prefix_lengths[-1]) if self.prefix_lengths.size else 0

        edge_meta = self.option_edges.set_index("edge_id", drop=False) if not self.option_edges.empty else pd.DataFrame()
        self.src_cluster = np.asarray(
            [int(edge_meta.loc[eid, "src"]) if eid in edge_meta.index else -1 for eid in self.edge_id],
            dtype=np.int64,
        )
        self.dst_cluster = np.asarray(
            [int(edge_meta.loc[eid, "dst"]) if eid in edge_meta.index else -1 for eid in self.edge_id],
            dtype=np.int64,
        )
        self.unique_edge_ids = np.asarray(sorted(np.unique(self.edge_id).tolist()), dtype=np.int64)
        self._segments_by_edge = {
            int(edge_id): np.flatnonzero(self.edge_id == int(edge_id)).astype(np.int64)
            for edge_id in self.unique_edge_ids
        }

        self.terminals = np.asarray(dataset.get("terminals", np.zeros(self.observations.shape[0], dtype=bool))).reshape(-1)
        all_bounds = episode_bounds_from_terminals(self.terminals, self.observations.shape[0])
        if episode_indices is None:
            episode_indices = np.arange(all_bounds.shape[0], dtype=np.int64)
        else:
            episode_indices = np.asarray(episode_indices, dtype=np.int64)
        if episode_indices.size and (int(episode_indices.min()) < 0 or int(episode_indices.max()) >= all_bounds.shape[0]):
            raise IndexError("episode_indices out of range")
        self.episode_indices = episode_indices
        self.episode_bounds = all_bounds[episode_indices] if episode_indices.size else all_bounds[:0]

        self.edge_loss_weight = self._edge_weight_lookup("loss_weight", default=1.0)
        self.support_edge_probabilities = self._edge_probabilities(np.ones(self.unique_edge_ids.size, dtype=np.float64))
        self.planner_edge_probabilities = self._edge_probabilities(self._planner_sampling_weights())
        self.obs_dim = int(self.observations.shape[1])
        self.action_dim = int(self.actions.shape[1])
        self.max_h = int(max(1, np.max(self.h))) if self.h.size else 1
        self.source_probabilities = self._normalize_source_probabilities(source_probabilities)
        self.source_cdf = np.cumsum([self.source_probabilities[source] for source in TARGET_SOURCE_TO_ID], dtype=np.float64)
        if self.source_cdf.size:
            self.source_cdf[-1] = 1.0

    def _edge_weight_lookup(self, column: str, default: float) -> dict[int, float]:
        if self.planner_edge_weights.empty or column not in self.planner_edge_weights.columns:
            return {int(edge_id): float(default) for edge_id in self.unique_edge_ids}
        indexed = self.planner_edge_weights.set_index("edge_id", drop=False)
        out: dict[int, float] = {}
        for edge_id in self.unique_edge_ids:
            if int(edge_id) in indexed.index:
                value = pd.to_numeric(pd.Series([indexed.loc[int(edge_id), column]]), errors="coerce").fillna(default).iloc[0]
                out[int(edge_id)] = float(value) if np.isfinite(float(value)) and float(value) > 0.0 else float(default)
            else:
                out[int(edge_id)] = float(default)
        return out

    def _planner_sampling_weights(self) -> np.ndarray:
        lookup = self._edge_weight_lookup("planner_sampling_weight", default=1.0)
        return np.asarray([max(1e-6, lookup.get(int(edge_id), 1.0)) for edge_id in self.unique_edge_ids], dtype=np.float64)

    def _edge_probabilities(self, weights: np.ndarray) -> np.ndarray:
        if self.unique_edge_ids.size == 0:
            return np.empty(0, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        if weights.size != self.unique_edge_ids.size or not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
            weights = np.ones(self.unique_edge_ids.size, dtype=np.float64)
        return weights / float(weights.sum())

    def _normalize_source_probabilities(self, source_probabilities: dict[str, float] | None) -> dict[str, float]:
        raw = {
            "final_goal_hindsight": 0.25,
            "support_edge_local": 0.35,
            "planner_first_edge_replay": 0.40,
            **(source_probabilities or {}),
        }
        available = {
            "final_goal_hindsight": self.episode_bounds.shape[0] > 0,
            "support_edge_local": self.unique_edge_ids.size > 0,
            "planner_first_edge_replay": self.unique_edge_ids.size > 0,
        }
        weights = {
            source: max(0.0, float(raw.get(source, 0.0))) if available[source] else 0.0
            for source in TARGET_SOURCE_TO_ID
        }
        total = float(sum(weights.values()))
        if total <= 0.0:
            raise ValueError("PlannerMixedGCBCDataset has no available sample sources")
        return {source: value / total for source, value in weights.items()}

    def __len__(self) -> int:
        if self.max_examples is not None:
            return max(0, int(self.max_examples))
        final_examples = int(sum(max(0, int(end - start - 1)) for start, end in self.episode_bounds))
        return max(1, int(self.total_transition_examples + final_examples))

    def _rand01(self, index: int, salt: int) -> float:
        mask = (1 << 64) - 1
        x = (int(self.seed) + (int(index) + 1) * 0x9E3779B97F4A7C15 + (int(salt) + 1) * 0xBF58476D1CE4E5B9) & mask
        x ^= x >> 30
        x = (x * 0xBF58476D1CE4E5B9) & mask
        x ^= x >> 27
        x = (x * 0x94D049BB133111EB) & mask
        x ^= x >> 31
        return float((x >> 11) & ((1 << 53) - 1)) / float(1 << 53)

    def _sample_source(self, index: int) -> str:
        pos = int(np.searchsorted(self.source_cdf, self._rand01(index, 0), side="right"))
        pos = min(max(pos, 0), len(TARGET_SOURCE_TO_ID) - 1)
        return list(TARGET_SOURCE_TO_ID.keys())[pos]

    def _sample_edge_segment(self, index: int, probabilities: np.ndarray, salt: int) -> tuple[int, int]:
        if self.unique_edge_ids.size == 0:
            raise IndexError("empty edge segment set")
        cdf = np.cumsum(probabilities, dtype=np.float64)
        cdf[-1] = 1.0
        edge_pos = int(np.searchsorted(cdf, self._rand01(index, salt), side="right"))
        edge_pos = min(max(edge_pos, 0), int(self.unique_edge_ids.size) - 1)
        edge_id = int(self.unique_edge_ids[edge_pos])
        candidates = self._segments_by_edge[edge_id]
        segment_index = int(candidates[min(int(self._rand01(index, salt + 1) * candidates.size), candidates.size - 1)])
        length = int(self.segment_lengths[segment_index])
        offset = int(min(int(self._rand01(index, salt + 2) * length), max(0, length - 1)))
        return segment_index, offset

    def sample_index(self, index: int) -> MixedSampleIndex:
        source = self._sample_source(int(index))
        if source == "final_goal_hindsight":
            episode_pos = min(int(self._rand01(index, 11) * self.episode_bounds.shape[0]), self.episode_bounds.shape[0] - 1)
            start, end = [int(x) for x in self.episode_bounds[episode_pos]]
            obs_stop = max(start + 1, end - 1)
            obs_idx = start + min(int(self._rand01(index, 12) * max(1, obs_stop - start)), max(0, obs_stop - start - 1))
            goal_low = obs_idx + 1
            goal_span = max(1, end - goal_low)
            goal_idx = goal_low + min(int(self._rand01(index, 13) * goal_span), goal_span - 1)
            return MixedSampleIndex(source=source, obs_index=int(obs_idx), goal_index=int(goal_idx))
        if source == "planner_first_edge_replay":
            segment_index, offset = self._sample_edge_segment(int(index), self.planner_edge_probabilities, 21)
            return MixedSampleIndex(source=source, segment_index=segment_index, offset=offset)
        segment_index, offset = self._sample_edge_segment(int(index), self.support_edge_probabilities, 31)
        return MixedSampleIndex(source=source, segment_index=segment_index, offset=offset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.sample_index(int(index))
        source_id = TARGET_SOURCE_TO_ID[sample.source]
        if sample.source == "final_goal_hindsight":
            obs_idx = int(sample.obs_index)
            goal_idx = int(sample.goal_index)
            edge_id = -1
            src_cluster = -1
            dst_cluster = -1
            edge_weight = 1.0
        else:
            seg = int(sample.segment_index)
            obs_idx = int(self.global_i[seg] + sample.offset)
            goal_idx = int(self.global_j[seg])
            edge_id = int(self.edge_id[seg])
            src_cluster = int(self.src_cluster[seg])
            dst_cluster = int(self.dst_cluster[seg])
            edge_weight = self.edge_loss_weight.get(edge_id, 1.0) if sample.source == "planner_first_edge_replay" else 1.0
        remaining_h = max(1, int(goal_idx - obs_idx))
        source_weight = float(self.source_loss_weights.get(sample.source, 1.0))
        return {
            "obs": torch.as_tensor(self.observations[obs_idx], dtype=torch.float32),
            "goal": torch.as_tensor(self.observations[goal_idx], dtype=torch.float32),
            "action": torch.as_tensor(self.actions[obs_idx], dtype=torch.float32),
            "edge_id": torch.as_tensor(edge_id, dtype=torch.long),
            "src_cluster": torch.as_tensor(src_cluster, dtype=torch.long),
            "dst_cluster": torch.as_tensor(dst_cluster, dtype=torch.long),
            "remaining_h": torch.as_tensor(float(remaining_h), dtype=torch.float32),
            "target_source_id": torch.as_tensor(source_id, dtype=torch.long),
            "sample_weight": torch.as_tensor(float(source_weight * edge_weight), dtype=torch.float32),
        }

    def with_indices(
        self,
        *,
        segment_indices: np.ndarray | None = None,
        episode_indices: np.ndarray | None = None,
        max_examples: int | None = None,
        seed: int | None = None,
    ) -> "PlannerMixedGCBCDataset":
        return PlannerMixedGCBCDataset(
            {"observations": self.observations, "actions": self.actions, "terminals": self.terminals},
            self.option_edges,
            self.edge_segments,
            self.planner_edge_weights,
            max_examples=self.max_examples if max_examples is None else max_examples,
            source_probabilities=self.source_probabilities,
            source_loss_weights=self.source_loss_weights,
            seed=self.seed if seed is None else int(seed),
            segment_indices=self.segment_indices if segment_indices is None else np.asarray(segment_indices, dtype=np.int64),
            episode_indices=self.episode_indices if episode_indices is None else np.asarray(episode_indices, dtype=np.int64),
        )

    def split(self, val_fraction: float, seed: int = 0) -> tuple["PlannerMixedGCBCDataset", "PlannerMixedGCBCDataset"]:
        rng = np.random.default_rng(int(seed))
        segments = self.segment_indices.copy()
        rng.shuffle(segments)
        n_val_segments = int(round(float(val_fraction) * segments.size))
        n_val_segments = min(max(n_val_segments, 1 if val_fraction > 0 and segments.size > 1 else 0), segments.size)
        val_segments = segments[:n_val_segments]
        val_set = set(int(x) for x in val_segments)
        train_segments = np.asarray([int(x) for x in self.segment_indices if int(x) not in val_set], dtype=np.int64)
        if train_segments.size == 0 and val_segments.size:
            train_segments = val_segments[:1]

        episodes = self.episode_indices.copy()
        rng.shuffle(episodes)
        n_val_episodes = int(round(float(val_fraction) * episodes.size))
        n_val_episodes = min(max(n_val_episodes, 1 if val_fraction > 0 and episodes.size > 1 else 0), episodes.size)
        val_episodes = episodes[:n_val_episodes]
        val_ep_set = set(int(x) for x in val_episodes)
        train_episodes = np.asarray([int(x) for x in self.episode_indices if int(x) not in val_ep_set], dtype=np.int64)
        if train_episodes.size == 0 and val_episodes.size:
            train_episodes = val_episodes[:1]
        return (
            self.with_indices(segment_indices=train_segments, episode_indices=train_episodes, seed=self.seed),
            self.with_indices(segment_indices=val_segments, episode_indices=val_episodes, seed=self.seed + 17),
        )
