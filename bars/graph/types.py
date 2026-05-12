from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
@dataclass
class BARSGraph:
    node_indices: np.ndarray; node_embeddings: np.ndarray; src: np.ndarray; dst: np.ndarray; cost: np.ndarray; risk: np.ndarray; p_exec: np.ndarray; kind: np.ndarray
    def __post_init__(self):
        self.node_indices=self.node_indices.astype(np.int64); self.src=self.src.astype(np.int64); self.dst=self.dst.astype(np.int64); self.cost=self.cost.astype(np.float32); self.risk=self.risk.astype(np.float32); self.p_exec=self.p_exec.astype(np.float32); self.kind=self.kind.astype(np.int32); self._out_edges: Optional[List[np.ndarray]]=None
    @property
    def num_nodes(self) -> int: return int(len(self.node_indices))
    @property
    def num_edges(self) -> int: return int(len(self.src))
    def outgoing_edges(self) -> List[np.ndarray]:
        if self._out_edges is None:
            out=[[] for _ in range(self.num_nodes)]
            for eid,s in enumerate(self.src): out[int(s)].append(eid)
            self._out_edges=[np.asarray(x,dtype=np.int64) for x in out]
        return self._out_edges
    def save_npz(self, path: str) -> None:
        import os; os.makedirs(os.path.dirname(path),exist_ok=True); np.savez_compressed(path,node_indices=self.node_indices,node_embeddings=self.node_embeddings,src=self.src,dst=self.dst,cost=self.cost,risk=self.risk,p_exec=self.p_exec,kind=self.kind)
    @classmethod
    def load_npz(cls, path: str) -> 'BARSGraph':
        d=np.load(path); return cls(d['node_indices'],d['node_embeddings'],d['src'],d['dst'],d['cost'],d['risk'],d['p_exec'],d['kind'])
EDGE_KIND_KNN=0; EDGE_KIND_TEMPORAL=1; EDGE_KIND_BRIDGE=2
