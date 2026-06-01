from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence


@dataclass(frozen=True)
class NodeSelectConfig:
    """Configuration for Stage27 node selection.

    The selector deliberately forms a union of several node families:
    high-TE states, coverage/FPS states, trajectory endpoints, and optional
    bottleneck-like states. This is designed to avoid the Stage26 failure mode
    where a single metric produces cheap but brittle shortcuts.
    """

    max_nodes: int = 2500
    te_quantile: float = 0.85
    te_top_k: Optional[int] = None
    coverage_k: int = 1000
    endpoint_stride: int = 1
    include_segment_endpoints: bool = True
    include_bottlenecks: bool = True
    bottleneck_k: int = 250
    bottleneck_knn: int = 12
    fps_seed: int = 0
    embedding_key: str = "tdr_emb"
    fallback_embedding_key: str = "states"


@dataclass(frozen=True)
class CalibratorConfig:
    """Configuration for execution-calibrated edge classifier."""

    horizon: int = 20
    positives_per_traj: int = 512
    random_negatives: int = 20000
    hard_negatives: int = 20000
    same_traj_far_negatives: int = 5000
    hard_negative_knn: int = 30
    seed: int = 0
    class_weight: str = "balanced"
    max_iter: int = 2000
    validation_fraction: float = 0.2
    min_positive_delta: int = 1
    max_positive_delta: Optional[int] = None


@dataclass(frozen=True)
class GraphBuildConfig:
    """Configuration for graph construction and edge costing."""

    variant: str = "B0_GAS"
    candidate_knn: int = 24
    same_traj_window: int = 30
    directed: bool = True
    embedding_key: str = "tdr_emb"
    tmd_key: str = "tmd_emb"
    xy_key: str = "xy"
    normalize_features: bool = True

    # Baseline edge composition.
    lambda_tdr: float = 1.0
    lambda_tmd_side: float = 0.0
    lambda_xy: float = 0.0

    # Stage27 additions.
    lambda_longhop: float = 0.0
    longhop_threshold: float = 2.5
    longhop_power: float = 2.0
    lambda_exec: float = 0.0
    exec_gate_threshold: Optional[float] = None
    lambda_uncertainty: float = 0.0
    lambda_cross_traj: float = 0.0
    lambda_disagreement: float = 0.0

    # Gated TMD shortcut; does not replace original GAS cost globally.
    use_tmd_gated_shortcut: bool = False
    tmd_shortcut_w: float = 0.25
    tmd_shortcut_min_p_exec: float = 0.65
    tmd_shortcut_max_disagreement: Optional[float] = None

    # Pruning / numerical safety.
    max_edge_cost: Optional[float] = None
    min_edge_cost: float = 1e-6
    eps: float = 1e-8

    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class AdaptiveConfig:
    """Configuration for adaptive waypoint execution."""

    reach_budget: float = 1.5
    min_p_exec: float = 0.55
    max_skip: Optional[int] = None
    monotonic_progress: bool = True
    fallback_to_next: bool = True
    replan_interval: int = 10
    deviation_replan_threshold: Optional[float] = None
    goal_tolerance: float = 0.5
    subgoal_tolerance: float = 0.5
    distance_key: str = "tdr_emb"


STAGE27_VARIANTS = {
    "B0_GAS": GraphBuildConfig(variant="B0_GAS"),
    "B2_LONGHOP": GraphBuildConfig(
        variant="B2_LONGHOP",
        lambda_longhop=0.35,
        longhop_threshold=2.5,
    ),
    "B4_TMD_GATED": GraphBuildConfig(
        variant="B4_TMD_GATED",
        lambda_longhop=0.35,
        use_tmd_gated_shortcut=True,
        tmd_shortcut_w=0.25,
        tmd_shortcut_min_p_exec=0.65,
        lambda_disagreement=0.05,
    ),
    "C1_EXEC_PENALTY": GraphBuildConfig(
        variant="C1_EXEC_PENALTY",
        lambda_longhop=0.35,
        lambda_exec=0.5,
    ),
    "C2_EXEC_GATE": GraphBuildConfig(
        variant="C2_EXEC_GATE",
        lambda_longhop=0.35,
        lambda_exec=0.25,
        exec_gate_threshold=0.45,
    ),
    "C3_EXEC_UNCERT": GraphBuildConfig(
        variant="C3_EXEC_UNCERT",
        lambda_longhop=0.35,
        lambda_exec=0.5,
        exec_gate_threshold=0.45,
        lambda_uncertainty=0.10,
        lambda_disagreement=0.05,
        lambda_cross_traj=0.05,
    ),
    "C4_EXEC_TMD": GraphBuildConfig(
        variant="C4_EXEC_TMD",
        lambda_longhop=0.35,
        lambda_exec=0.5,
        exec_gate_threshold=0.45,
        lambda_uncertainty=0.10,
        lambda_disagreement=0.05,
        lambda_cross_traj=0.05,
        use_tmd_gated_shortcut=True,
        tmd_shortcut_w=0.25,
        tmd_shortcut_min_p_exec=0.65,
    ),
}


def resolve_variants(names: Optional[Sequence[str]]) -> dict[str, GraphBuildConfig]:
    if not names:
        return STAGE27_VARIANTS.copy()
    unknown = [name for name in names if name not in STAGE27_VARIANTS]
    if unknown:
        raise ValueError(f"Unknown Stage27 variants: {unknown}. Known={list(STAGE27_VARIANTS)}")
    return {name: STAGE27_VARIANTS[name] for name in names}
