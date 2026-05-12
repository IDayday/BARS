from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np

from bars.common.logging import CSVLogger
from bars.data.trajectories import OfflineDataset
from .types import BARSGraph
from .ann import KNNIndex


@dataclass
class EdgeSupportResult:
    counts: np.ndarray
    sampled_segments: int
    edge_hits: int
    unique_edges_hit: int
    hit_rate: float
    edge_support_rate: float


def build_edge_lookup(graph: BARSGraph) -> Dict[Tuple[int, int], int]:
    return {(int(s), int(d)): int(eid) for eid, (s, d) in enumerate(zip(graph.src, graph.dst))}


def map_indices_to_graph_nodes(
    embeddings: np.ndarray,
    graph: BARSGraph,
    indices: np.ndarray,
    nn: Optional[KNNIndex] = None,
    batch_size: int = 65536,
) -> np.ndarray:
    if nn is None:
        nn = KNNIndex(graph.node_embeddings, backend="auto")
    out = np.empty(len(indices), dtype=np.int64)
    for st in range(0, len(indices), batch_size):
        sl = slice(st, min(st + batch_size, len(indices)))
        out[sl] = nn.kneighbors(embeddings[indices[sl]], 1, return_distance=False)[:, 0].astype(np.int64)
    return out


def sample_edge_support_counts(
    dataset: OfflineDataset,
    embeddings: np.ndarray,
    graph: BARSGraph,
    horizon: int,
    num_segments: int,
    rng: np.random.Generator,
    min_dt: int = 1,
    batch_size: int = 65536,
    logger: Optional[CSVLogger] = None,
    phase: str = 'edge_support',
    cfg: Optional[dict] = None,
) -> EdgeSupportResult:
    """Estimate observed same-trajectory support for graph edges.

    This maps sampled dataset transitions (s_t, s_{t+k}) to nearest graph nodes
    and counts how often the corresponding node-to-node pair exists as a graph
    edge. It is intentionally a support estimate rather than a false-positive
    label: graph edges without support remain unlabeled unless a diagnostic
    explicitly treats them as hard-negative proxies.
    """
    num_segments = int(max(0, num_segments))
    counts = np.zeros(graph.num_edges, dtype=np.int32)
    if graph.num_edges == 0 or num_segments == 0:
        return EdgeSupportResult(counts, 0, 0, 0, 0.0, 0.0)

    lookup = build_edge_lookup(graph)
    nn = KNNIndex.from_config(graph.node_embeddings, cfg or {}, prefix="ann")
    sampled = 0
    hits = 0
    progress_every = max(batch_size, num_segments // 4)
    next_progress = progress_every

    while sampled < num_segments:
        bs = min(batch_size, num_segments - sampled)
        try:
            src_idx, dst_idx, _ = dataset.sample_future_pairs(bs, horizon, rng, min_dt=min_dt)
        except Exception:
            break
        src_node = map_indices_to_graph_nodes(embeddings, graph, src_idx, nn=nn, batch_size=batch_size)
        dst_node = map_indices_to_graph_nodes(embeddings, graph, dst_idx, nn=nn, batch_size=batch_size)
        for u, v in zip(src_node, dst_node):
            eid = lookup.get((int(u), int(v)))
            if eid is not None:
                counts[eid] += 1
                hits += 1
        sampled += bs
        if logger is not None and sampled >= next_progress:
            logger.log({
                'phase': phase,
                'event': 'progress',
                'sampled_segments': sampled,
                'target_segments': num_segments,
                'edge_hits': hits,
            })
            next_progress += progress_every

    unique = int((counts > 0).sum())
    return EdgeSupportResult(
        counts=counts,
        sampled_segments=int(sampled),
        edge_hits=int(hits),
        unique_edges_hit=unique,
        hit_rate=float(hits / max(1, sampled)),
        edge_support_rate=float(unique / max(1, graph.num_edges)),
    )
