from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .config import CalibratorConfig
from .dataset import OfflineDataset
from .edge_features import EdgeFeatureExtractor, FEATURE_COLUMNS
from .math_utils import knn_indices


@dataclass
class PairTrainingSet:
    src_idx: np.ndarray
    dst_idx: np.ndarray
    y: np.ndarray
    features: Dict[str, np.ndarray]
    columns: list[str] = field(default_factory=lambda: FEATURE_COLUMNS.copy())

    def matrix(self) -> np.ndarray:
        return EdgeFeatureExtractor.to_matrix(self.features, self.columns)


def _sample_same_traj_positives(dataset: OfflineDataset, cfg: CalibratorConfig, rng: np.random.Generator) -> tuple[list[int], list[int], list[int]]:
    src, dst, y = [], [], []
    max_delta = cfg.max_positive_delta or cfg.horizon
    for _, idx in dataset.trajectory_slices().items():
        if len(idx) <= cfg.min_positive_delta:
            continue
        n_samples = min(cfg.positives_per_traj, max(1, len(idx) * 2))
        for _ in range(n_samples):
            pos = int(rng.integers(0, max(1, len(idx) - cfg.min_positive_delta)))
            max_d = min(max_delta, len(idx) - 1 - pos)
            if max_d < cfg.min_positive_delta:
                continue
            delta = int(rng.integers(cfg.min_positive_delta, max_d + 1))
            src.append(int(idx[pos]))
            dst.append(int(idx[pos + delta]))
            y.append(1)
    return src, dst, y


def _sample_random_cross_negatives(dataset: OfflineDataset, cfg: CalibratorConfig, rng: np.random.Generator) -> tuple[list[int], list[int], list[int]]:
    src, dst, y = [], [], []
    if dataset.n <= 1:
        return src, dst, y
    attempts = 0
    target = cfg.random_negatives
    while len(src) < target and attempts < target * 20:
        attempts += 1
        a = int(rng.integers(0, dataset.n))
        b = int(rng.integers(0, dataset.n))
        if a == b:
            continue
        if dataset.traj_ids[a] == dataset.traj_ids[b]:
            # keep a small fraction of same-traj negatives for robustness only in far sampler
            continue
        src.append(a)
        dst.append(b)
        y.append(0)
    return src, dst, y


def _sample_same_traj_far_negatives(dataset: OfflineDataset, cfg: CalibratorConfig, rng: np.random.Generator) -> tuple[list[int], list[int], list[int]]:
    src, dst, y = [], [], []
    if cfg.same_traj_far_negatives <= 0:
        return src, dst, y
    trajs = [idx for idx in dataset.trajectory_slices().values() if len(idx) > cfg.horizon + 2]
    if not trajs:
        return src, dst, y
    for _ in range(cfg.same_traj_far_negatives):
        idx = trajs[int(rng.integers(0, len(trajs)))]
        a_pos = int(rng.integers(0, len(idx)))
        # Pick a same-trajectory state outside horizon, either direction.
        candidates = np.flatnonzero(np.abs(np.arange(len(idx)) - a_pos) > cfg.horizon)
        if len(candidates) == 0:
            continue
        b_pos = int(rng.choice(candidates))
        src.append(int(idx[a_pos]))
        dst.append(int(idx[b_pos]))
        y.append(0)
    return src, dst, y


def _sample_hard_negatives(dataset: OfflineDataset, cfg: CalibratorConfig, rng: np.random.Generator) -> tuple[list[int], list[int], list[int]]:
    src, dst, y = [], [], []
    if cfg.hard_negatives <= 0 or dataset.n <= 2:
        return src, dst, y
    x = dataset.embedding("tdr_emb", "states")
    k = min(max(3, cfg.hard_negative_knn), dataset.n)
    ind, _ = knn_indices(x, k)
    target = cfg.hard_negatives
    attempts = 0
    while len(src) < target and attempts < target * 20:
        attempts += 1
        a = int(rng.integers(0, dataset.n))
        b = int(rng.choice(ind[a, 1:] if ind.shape[1] > 1 else ind[a]))
        if a == b:
            continue
        same = dataset.traj_ids[a] == dataset.traj_ids[b]
        if same and 0 < (dataset.time_idxs[b] - dataset.time_idxs[a]) <= cfg.horizon:
            continue  # likely positive support
        # Hard negative: close in representation but not supported as near future.
        src.append(a)
        dst.append(b)
        y.append(0)
    return src, dst, y


def build_pair_training_set(dataset: OfflineDataset, cfg: CalibratorConfig) -> PairTrainingSet:
    rng = np.random.default_rng(cfg.seed)
    src, dst, y = [], [], []
    for sampler in [
        _sample_same_traj_positives,
        _sample_random_cross_negatives,
        _sample_same_traj_far_negatives,
        _sample_hard_negatives,
    ]:
        a, b, labels = sampler(dataset, cfg, rng)
        src.extend(a)
        dst.extend(b)
        y.extend(labels)

    if not src:
        raise ValueError("No pair samples were generated; check dataset trajectory structure")
    src_idx = np.asarray(src, dtype=np.int64)
    dst_idx = np.asarray(dst, dtype=np.int64)
    labels = np.asarray(y, dtype=np.int64)
    extractor = EdgeFeatureExtractor(dataset)
    features = extractor.pair_features(src_idx, dst_idx)
    return PairTrainingSet(src_idx=src_idx, dst_idx=dst_idx, y=labels, features=features)


@dataclass
class CalibratorMetrics:
    train_auc: float
    val_auc: float
    train_brier: float
    val_brier: float
    n_train: int
    n_val: int
    positive_rate: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class ExecutionCalibrator:
    """Small calibrated classifier estimating p_exec(s_i -> s_j).

    This is intentionally simple and inspectable. It is a planner calibration
    model, not a replacement for the low-level actor or value function.
    """

    def __init__(self, columns: Optional[list[str]] = None) -> None:
        self.columns = columns or FEATURE_COLUMNS.copy()
        self.pipeline = None
        self.metrics_: Optional[CalibratorMetrics] = None

    def fit(self, training_set: PairTrainingSet, cfg: CalibratorConfig) -> CalibratorMetrics:
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import roc_auc_score, brier_score_loss
            from sklearn.model_selection import train_test_split
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
        except Exception as exc:
            raise ImportError("ExecutionCalibrator requires scikit-learn. Install scikit-learn or disable calibrator.") from exc

        x = training_set.matrix()
        y = training_set.y.astype(np.int64)
        stratify = y if len(np.unique(y)) == 2 and min(np.bincount(y)) >= 2 else None
        x_train, x_val, y_train, y_val = train_test_split(
            x,
            y,
            test_size=cfg.validation_fraction,
            random_state=cfg.seed,
            stratify=stratify,
        )
        clf = LogisticRegression(
            max_iter=cfg.max_iter,
            class_weight=cfg.class_weight,
            solver="lbfgs",
            n_jobs=None,
        )
        self.pipeline = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        self.pipeline.fit(x_train, y_train)

        def _predict(xx: np.ndarray) -> np.ndarray:
            return self.pipeline.predict_proba(xx)[:, 1]

        p_train = _predict(x_train)
        p_val = _predict(x_val)
        train_auc = float(roc_auc_score(y_train, p_train)) if len(np.unique(y_train)) == 2 else float("nan")
        val_auc = float(roc_auc_score(y_val, p_val)) if len(np.unique(y_val)) == 2 else float("nan")
        metrics = CalibratorMetrics(
            train_auc=train_auc,
            val_auc=val_auc,
            train_brier=float(brier_score_loss(y_train, p_train)),
            val_brier=float(brier_score_loss(y_val, p_val)),
            n_train=int(len(y_train)),
            n_val=int(len(y_val)),
            positive_rate=float(y.mean()),
        )
        self.metrics_ = metrics
        return metrics

    def predict_proba_matrix(self, x: np.ndarray) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("ExecutionCalibrator is not fitted")
        return self.pipeline.predict_proba(np.asarray(x, dtype=np.float32))[:, 1].astype(np.float32)

    def predict_proba_features(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        x = EdgeFeatureExtractor.to_matrix(features, self.columns)
        return self.predict_proba_matrix(x)

    def save(self, path: str | Path) -> None:
        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"columns": self.columns, "pipeline": self.pipeline, "metrics": self.metrics_}, f)

    @staticmethod
    def load(path: str | Path) -> "ExecutionCalibrator":
        import pickle

        with open(path, "rb") as f:
            obj = pickle.load(f)
        cal = ExecutionCalibrator(columns=obj["columns"])
        cal.pipeline = obj["pipeline"]
        cal.metrics_ = obj.get("metrics")
        return cal
