from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:  # pragma: no cover - handled by callers in minimal envs.
    torch = None
    Dataset = object  # type: ignore[misc,assignment]


SamplingMode = Literal["uniform_edge", "uniform_transition", "bottleneck_weighted"]


def _load_edges(option_edges_csv: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(option_edges_csv, pd.DataFrame):
        return option_edges_csv.copy()
    return pd.read_csv(Path(option_edges_csv).expanduser())


def _load_segments(edge_segments_npz: str | Path | dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if isinstance(edge_segments_npz, dict):
        return {key: np.asarray(value) for key, value in edge_segments_npz.items()}
    with np.load(Path(edge_segments_npz).expanduser()) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _edge_lookup(option_edges: pd.DataFrame, column: str, default: float = 0.0) -> dict[int, float]:
    if column not in option_edges.columns:
        return {int(row.edge_id): float(default) for row in option_edges.itertuples(index=False)}
    return {int(row.edge_id): float(getattr(row, column)) for row in option_edges.itertuples(index=False)}


@dataclass(frozen=True)
class EdgeBCSampleIndex:
    segment_index: int
    offset: int


class EdgeBCDataset(Dataset):
    """Lazy one-step GCBC samples expanded from Phase 2 option-edge segments."""

    def __init__(
        self,
        dataset: dict[str, Any],
        option_edges: pd.DataFrame,
        edge_segments: dict[str, np.ndarray],
        max_examples: int | None = None,
        sampling_mode: SamplingMode = "uniform_edge",
        seed: int = 0,
        segment_indices: np.ndarray | None = None,
    ) -> None:
        if torch is None:  # pragma: no cover
            raise ImportError("Phase 3 GCBC dataset requires PyTorch")
        sampling_mode = str(sampling_mode)
        if sampling_mode not in {"uniform_edge", "uniform_transition", "bottleneck_weighted"}:
            raise ValueError("sampling_mode must be uniform_edge, uniform_transition, or bottleneck_weighted")

        self.observations = np.asarray(dataset["observations"], dtype=np.float32)
        self.actions = np.asarray(dataset["actions"], dtype=np.float32)
        self.option_edges = option_edges.copy()
        self.edge_segments = {key: np.asarray(value) for key, value in edge_segments.items()}
        self.max_examples = None if max_examples is None else int(max_examples)
        self.sampling_mode: SamplingMode = sampling_mode  # type: ignore[assignment]
        self.seed = int(seed)

        if self.observations.ndim == 1:
            self.observations = self.observations.reshape(-1, 1)
        if self.actions.ndim == 1:
            self.actions = self.actions.reshape(-1, 1)

        required = {"edge_id", "global_i", "global_j", "h"}
        missing = required - set(self.edge_segments)
        if missing:
            raise KeyError(f"edge_segments is missing required arrays: {sorted(missing)}")

        n_segments = int(np.asarray(self.edge_segments["edge_id"]).shape[0])
        if segment_indices is None:
            segment_indices = np.arange(n_segments, dtype=np.int64)
        else:
            segment_indices = np.asarray(segment_indices, dtype=np.int64)
        if segment_indices.size:
            if np.min(segment_indices) < 0 or np.max(segment_indices) >= n_segments:
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
        self.edge_sampling_probabilities = self._make_edge_probabilities()
        self.obs_dim = int(self.observations.shape[1])
        self.action_dim = int(self.actions.shape[1])
        self.max_h = int(max(1, np.max(self.h))) if self.h.size else 1

    def _make_edge_probabilities(self) -> np.ndarray:
        if self.unique_edge_ids.size == 0:
            return np.empty(0, dtype=np.float64)
        if self.sampling_mode == "bottleneck_weighted":
            bottleneck = _edge_lookup(self.option_edges, "edge_bottleneck_score", default=0.0)
            weights = np.asarray(
                [max(0.0, float(bottleneck.get(int(edge_id), 0.0))) for edge_id in self.unique_edge_ids],
                dtype=np.float64,
            )
            if float(weights.sum()) <= 0.0:
                weights = np.ones_like(weights)
        else:
            weights = np.ones(self.unique_edge_ids.size, dtype=np.float64)
        return weights / float(weights.sum())

    def __len__(self) -> int:
        if self.max_examples is not None:
            return max(0, self.max_examples)
        return max(0, self.total_transition_examples)

    def _rng_for_index(self, index: int) -> np.random.Generator:
        return np.random.default_rng(self.seed + int(index) * 1000003)

    def _flat_to_segment_offset(self, flat_index: int) -> EdgeBCSampleIndex:
        if self.total_transition_examples <= 0:
            raise IndexError("empty EdgeBCDataset")
        flat_index = int(flat_index) % self.total_transition_examples
        segment_index = int(np.searchsorted(self.prefix_lengths, flat_index, side="right"))
        prev = int(self.prefix_lengths[segment_index - 1]) if segment_index > 0 else 0
        return EdgeBCSampleIndex(segment_index=segment_index, offset=int(flat_index - prev))

    def _random_segment_offset(self, rng: np.random.Generator) -> EdgeBCSampleIndex:
        if self.unique_edge_ids.size == 0:
            raise IndexError("empty EdgeBCDataset")
        edge_id = int(rng.choice(self.unique_edge_ids, p=self.edge_sampling_probabilities))
        candidates = self._segments_by_edge[edge_id]
        segment_index = int(rng.choice(candidates))
        offset = int(rng.integers(0, int(self.segment_lengths[segment_index])))
        return EdgeBCSampleIndex(segment_index=segment_index, offset=offset)

    def sample_index(self, index: int) -> EdgeBCSampleIndex:
        if self.sampling_mode == "uniform_transition" and self.max_examples is None:
            return self._flat_to_segment_offset(index)
        rng = self._rng_for_index(index)
        if self.sampling_mode == "uniform_transition":
            flat = int(rng.integers(0, max(1, self.total_transition_examples)))
            return self._flat_to_segment_offset(flat)
        return self._random_segment_offset(rng)

    def sample_edge_ids(self, n: int, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(self.seed if seed is None else int(seed))
        if self.unique_edge_ids.size == 0:
            return np.empty(0, dtype=np.int64)
        if self.sampling_mode == "uniform_transition":
            flat = rng.integers(0, max(1, self.total_transition_examples), size=int(n))
            return np.asarray(
                [self.edge_id[self._flat_to_segment_offset(int(x)).segment_index] for x in flat],
                dtype=np.int64,
            )
        return rng.choice(self.unique_edge_ids, size=int(n), p=self.edge_sampling_probabilities).astype(np.int64)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.sample_index(int(index))
        seg = sample.segment_index
        u = int(self.global_i[seg] + sample.offset)
        goal_idx = int(self.global_j[seg])
        remaining_h = int(goal_idx - u)
        return {
            "obs": torch.as_tensor(self.observations[u], dtype=torch.float32),
            "goal": torch.as_tensor(self.observations[goal_idx], dtype=torch.float32),
            "action": torch.as_tensor(self.actions[u], dtype=torch.float32),
            "edge_id": torch.as_tensor(int(self.edge_id[seg]), dtype=torch.long),
            "src_cluster": torch.as_tensor(int(self.src_cluster[seg]), dtype=torch.long),
            "dst_cluster": torch.as_tensor(int(self.dst_cluster[seg]), dtype=torch.long),
            "remaining_h": torch.as_tensor(float(remaining_h), dtype=torch.float32),
        }

    def with_segment_indices(
        self,
        segment_indices: np.ndarray,
        max_examples: int | None = None,
        sampling_mode: SamplingMode | None = None,
        seed: int | None = None,
    ) -> "EdgeBCDataset":
        return EdgeBCDataset(
            {"observations": self.observations, "actions": self.actions},
            self.option_edges,
            self.edge_segments,
            max_examples=self.max_examples if max_examples is None else max_examples,
            sampling_mode=self.sampling_mode if sampling_mode is None else sampling_mode,
            seed=self.seed if seed is None else int(seed),
            segment_indices=np.asarray(segment_indices, dtype=np.int64),
        )

    def split_by_segments(self, val_fraction: float, seed: int = 0) -> tuple["EdgeBCDataset", "EdgeBCDataset"]:
        rng = np.random.default_rng(seed)
        all_indices = np.arange(int(np.asarray(self.edge_segments["edge_id"]).shape[0]), dtype=np.int64)
        valid_original = self.segment_indices.copy()
        if valid_original.size == 0:
            return self.with_segment_indices(valid_original), self.with_segment_indices(valid_original)
        shuffled = valid_original.copy()
        rng.shuffle(shuffled)
        n_val = int(round(float(val_fraction) * shuffled.size))
        n_val = min(max(n_val, 1 if val_fraction > 0 and shuffled.size > 1 else 0), shuffled.size)
        val_idx = shuffled[:n_val]
        val_set = set(int(x) for x in val_idx)
        train_idx = np.asarray([int(x) for x in valid_original if int(x) not in val_set], dtype=np.int64)
        if train_idx.size == 0 and val_idx.size:
            train_idx = val_idx[:1]
        del all_indices
        return (
            self.with_segment_indices(train_idx, seed=self.seed),
            self.with_segment_indices(val_idx, seed=self.seed + 17),
        )


def build_edge_bc_examples(
    dataset: dict[str, Any],
    option_edges_csv: str | Path | pd.DataFrame,
    edge_segments_npz: str | Path | dict[str, np.ndarray],
    max_examples: int | None = None,
    sampling_mode: SamplingMode = "uniform_edge",
    seed: int = 0,
) -> EdgeBCDataset:
    option_edges = _load_edges(option_edges_csv)
    edge_segments = _load_segments(edge_segments_npz)
    return EdgeBCDataset(
        dataset,
        option_edges,
        edge_segments,
        max_examples=max_examples,
        sampling_mode=sampling_mode,
        seed=seed,
    )
