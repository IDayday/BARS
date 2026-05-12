from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np
from bars.common.logging import CSVLogger
from bars.data.trajectories import OfflineDataset
from .types import BARSGraph

@dataclass
class BoundaryIndex:
    dep_hist: np.ndarray
    arr_hist: np.ndarray
    has_dep: np.ndarray
    has_arr: np.ndarray
    fallback_psi: float = 0.5
    edge_dir: np.ndarray | None = None
    direction_temperature: float = 1.0

    def psi(self, prev_edge: int, next_edge: int) -> float:
        if self.has_arr[prev_edge] and self.has_dep[next_edge]:
            return float(np.minimum(self.arr_hist[prev_edge], self.dep_hist[next_edge]).sum())
        if self.edge_dir is not None:
            a = self.edge_dir[prev_edge]
            b = self.edge_dir[next_edge]
            # Smooth-turn proxy: high when two consecutive latent directions agree.
            dist2 = float(np.sum((a - b) ** 2))
            return float(np.clip(np.exp(-dist2 / max(self.direction_temperature, 1e-6)), 1e-4, 1.0))
        return float(self.fallback_psi)

    def boundary_cost(self, prev_edge: int, next_edge: int) -> float:
        return float(-np.log(np.clip(self.psi(prev_edge, next_edge), 1e-4, 1.0)))

    def save_npz(self, path: str) -> None:
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, dep_hist=self.dep_hist, arr_hist=self.arr_hist, has_dep=self.has_dep, has_arr=self.has_arr, fallback_psi=self.fallback_psi, edge_dir=np.asarray([]) if self.edge_dir is None else self.edge_dir, direction_temperature=self.direction_temperature)

    @classmethod
    def load_npz(cls, path: str) -> 'BoundaryIndex':
        d = np.load(path)
        edge_dir = d['edge_dir'] if 'edge_dir' in d and d['edge_dir'].size else None
        temp = float(d['direction_temperature']) if 'direction_temperature' in d else 1.0
        return cls(d['dep_hist'], d['arr_hist'], d['has_dep'], d['has_arr'], float(d['fallback_psi']), edge_dir, temp)


def _direction_boundary(graph: BARSGraph, cfg: Dict, logger: CSVLogger) -> BoundaryIndex:
    bcfg = cfg.get('boundary', {})
    vec = graph.node_embeddings[graph.dst] - graph.node_embeddings[graph.src]
    norm = np.linalg.norm(vec, axis=1, keepdims=True) + 1e-8
    edge_dir = (vec / norm).astype(np.float32)
    dep = np.zeros((graph.num_edges, 1), dtype=np.float32)
    arr = np.zeros((graph.num_edges, 1), dtype=np.float32)
    has = np.zeros(graph.num_edges, dtype=bool)
    b = BoundaryIndex(dep, arr, has, has.copy(), float(bcfg.get('fallback_psi', 0.5)), edge_dir=edge_dir, direction_temperature=float(bcfg.get('direction_temperature', 1.0)))
    out = graph.outgoing_edges(); checked = 0; psi_sum = 0.0
    for eid in range(graph.num_edges):
        for ne in out[int(graph.dst[eid])]:
            checked += 1; psi_sum += b.psi(eid, int(ne))
    logger.log({'phase':'boundary','method':'direction','composable_pairs':checked,'psi_mean':psi_sum/max(1,checked),'direction_temperature':b.direction_temperature})
    return b


def build_boundary_index(dataset: OfflineDataset, embeddings: np.ndarray, graph: BARSGraph, cfg: Dict, logger: CSVLogger) -> BoundaryIndex:
    # Default to a fast direction-based boundary proxy. It is cheap enough for
    # multi-run sweeps and can be replaced by support-hist portal modes later by
    # changing only this module.
    method = str(cfg.get('boundary', {}).get('method', 'direction')).lower()
    if method == 'direction':
        return _direction_boundary(graph, cfg, logger)
    # Minimal support-hist fallback: one mode, support if same-trajectory short segment exists.
    bcfg = cfg.get('boundary', {})
    horizon = int(bcfg.get('segment_horizon', cfg.get('reachability', {}).get('horizon', 30)))
    dep = np.zeros((graph.num_edges, 1), dtype=np.float32); arr = np.zeros((graph.num_edges, 1), dtype=np.float32)
    has_dep = np.zeros(graph.num_edges, dtype=bool); has_arr = np.zeros(graph.num_edges, dtype=bool)
    gi_src = graph.node_indices[graph.src]; gi_dst = graph.node_indices[graph.dst]
    same = dataset.traj_id[gi_src] == dataset.traj_id[gi_dst]
    dt = dataset.timestep[gi_dst] - dataset.timestep[gi_src]
    ok = same & (dt > 0) & (dt <= horizon)
    dep[ok, 0] = 1.0; arr[ok, 0] = 1.0; has_dep[ok] = True; has_arr[ok] = True
    b = BoundaryIndex(dep, arr, has_dep, has_arr, float(bcfg.get('fallback_psi', 0.5)))
    logger.log({'phase':'boundary','method':'support_binary','edge_has_support_rate':float(ok.mean()) if len(ok) else 0.0})
    return b
