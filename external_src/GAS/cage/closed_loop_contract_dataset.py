from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import json

import numpy as np


class ClosedLoopContractDataset:
    """Lightweight JSONL-backed dataset for offline contract-model training."""

    def __init__(self, records: Iterable[dict[str, Any]]):
        self.records = list(records)
        self.size = len(self.records)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "ClosedLoopContractDataset":
        rows = []
        with Path(path).open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
        return cls(rows)

    def feature_matrix(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        feats = []
        labels = {
            "hit": [],
            "contract_positive": [],
            "negative_progress": [],
        }
        for row in self.records:
            phi_s = np.asarray(row.get("phi_start") or row.get("phi_s") or [], dtype=np.float32)
            phi_g = np.asarray(row.get("phi_target") or row.get("phi_g") or [], dtype=np.float32)
            if phi_s.size == 0 or phi_g.size == 0 or phi_s.shape != phi_g.shape:
                continue
            delta = phi_g - phi_s
            feature = np.concatenate([
                phi_s,
                phi_g,
                delta,
                np.abs(delta),
                np.asarray([float(row.get("d_phi_start", row.get("d_phi", np.linalg.norm(delta))) or 0.0)], dtype=np.float32),
            ])
            feats.append(feature)
            labels["hit"].append(float(bool(row.get("hit", row.get("label_hit", False)))))
            labels["contract_positive"].append(float(bool(row.get("label_contract_positive", row.get("label_good_contract", False)))))
            labels["negative_progress"].append(float(bool(row.get("negative_progress", row.get("label_negative_progress", False)))))
        if not feats:
            return np.empty((0, 0), dtype=np.float32), {k: np.empty((0,), dtype=np.float32) for k in labels}
        return np.stack(feats).astype(np.float32), {k: np.asarray(v, dtype=np.float32) for k, v in labels.items()}
