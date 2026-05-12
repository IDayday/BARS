from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


def _try_import_faiss():
    try:
        import faiss  # type: ignore
        return faiss
    except Exception:
        return None


@dataclass
class KNNResult:
    distances: np.ndarray
    indices: np.ndarray
    backend: str


class KNNIndex:
    """Small abstraction over FAISS / sklearn kNN.

    The graph code calls kNN repeatedly during graph construction, edge support
    mapping and boundary construction.  FAISS is optional: if unavailable, this
    class falls back to sklearn without changing experiment code.  Distances are
    Euclidean distances, not squared distances.
    """

    def __init__(self, data: np.ndarray, *, backend: str = "auto", use_gpu: bool = False, gpu_id: int = 0, n_threads: Optional[int] = None):
        self.data = np.asarray(data, dtype=np.float32)
        if self.data.ndim != 2:
            raise ValueError(f"KNNIndex data must be 2D, got shape={self.data.shape}")
        self.backend_requested = backend
        self.backend = "sklearn"
        self._nn = None
        self._faiss = None
        self._gpu_res = None
        backend_l = str(backend or "auto").lower()
        if backend_l in {"auto", "faiss", "faiss_cpu", "faiss_gpu"}:
            faiss = _try_import_faiss()
            if faiss is not None:
                try:
                    if n_threads is not None and hasattr(faiss, "omp_set_num_threads"):
                        faiss.omp_set_num_threads(int(n_threads))
                    index = faiss.IndexFlatL2(self.data.shape[1])
                    want_gpu = bool(use_gpu or backend_l == "faiss_gpu")
                    if want_gpu and hasattr(faiss, "StandardGpuResources"):
                        self._gpu_res = faiss.StandardGpuResources()
                        index = faiss.index_cpu_to_gpu(self._gpu_res, int(gpu_id), index)
                        self.backend = "faiss_gpu"
                    else:
                        self.backend = "faiss_cpu"
                    index.add(np.ascontiguousarray(self.data, dtype=np.float32))
                    self._faiss = index
                except Exception:
                    self._faiss = None
                    self.backend = "sklearn"
        if self._faiss is None:
            from sklearn.neighbors import NearestNeighbors
            self._nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(self.data)
            self.backend = "sklearn"

    @classmethod
    def from_config(cls, data: np.ndarray, cfg: dict, prefix: str = "ann") -> "KNNIndex":
        acfg = cfg.get(prefix, {}) if isinstance(cfg, dict) else {}
        # Accept graph.ann_* too, because most users will set it under graph.
        gcfg = cfg.get("graph", {}) if isinstance(cfg, dict) else {}
        backend = acfg.get("backend", gcfg.get("ann_backend", "auto"))
        use_gpu = bool(acfg.get("use_gpu", gcfg.get("ann_use_gpu", False)))
        gpu_id = int(acfg.get("gpu_id", gcfg.get("ann_gpu_id", 0)))
        n_threads = acfg.get("n_threads", gcfg.get("ann_threads", None))
        return cls(data, backend=backend, use_gpu=use_gpu, gpu_id=gpu_id, n_threads=None if n_threads is None else int(n_threads))

    def kneighbors(self, queries: np.ndarray, n_neighbors: int, *, return_distance: bool = True, batch_size: int = 131072) -> Tuple[np.ndarray, np.ndarray] | np.ndarray:
        q = np.asarray(queries, dtype=np.float32)
        if q.ndim == 1:
            q = q[None]
        k = int(max(1, min(n_neighbors, len(self.data))))
        if self._faiss is not None:
            all_d = []
            all_i = []
            for st in range(0, len(q), int(batch_size)):
                sl = slice(st, min(st + int(batch_size), len(q)))
                d2, ind = self._faiss.search(np.ascontiguousarray(q[sl], dtype=np.float32), k)
                all_d.append(np.sqrt(np.maximum(d2, 0.0)).astype(np.float32))
                all_i.append(ind.astype(np.int64))
            dist = np.concatenate(all_d, axis=0) if all_d else np.empty((0, k), dtype=np.float32)
            ind = np.concatenate(all_i, axis=0) if all_i else np.empty((0, k), dtype=np.int64)
        else:
            # Refit with requested k; sklearn stores n_neighbors in estimator.
            from sklearn.neighbors import NearestNeighbors
            if getattr(self._nn, "n_neighbors", None) != k:
                self._nn = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(self.data)
            dist, ind = self._nn.kneighbors(q, n_neighbors=k, return_distance=True)
            dist = dist.astype(np.float32)
            ind = ind.astype(np.int64)
        return (dist, ind) if return_distance else ind

    def query(self, queries: np.ndarray, k: int, batch_size: int = 131072) -> KNNResult:
        d, i = self.kneighbors(queries, k, return_distance=True, batch_size=batch_size)  # type: ignore[misc]
        return KNNResult(distances=d, indices=i, backend=self.backend)
