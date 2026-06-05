from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HardGoalExample:
    env_name: str | None
    target_mode: str | None
    category: str
    available_action_supervision: bool
    d_phi: float | None
    q_train_support: float | None
    normalized_progress: float | None
    source_probe_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": "graph_induced_hard_goal",
            "env_name": self.env_name,
            "target_mode": self.target_mode,
            "category": self.category,
            "available_action_supervision": self.available_action_supervision,
            "d_phi": self.d_phi,
            "q_train_support": self.q_train_support,
            "normalized_progress": self.normalized_progress,
            "source_probe_id": self.source_probe_id,
        }


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def has_action_supervision(row: dict[str, Any]) -> bool:
    if row.get("available_action_supervision") is not None:
        return bool(row.get("available_action_supervision"))
    source = str(row.get("pair_source", "")).lower()
    target_mode = str(row.get("target_mode", "")).lower()
    goal_source = str(row.get("goal_source", "")).lower()
    return "qtrain" in target_mode or "q_train" in source or "hindsight" in goal_source or "future" in goal_source


def is_hard_goal(row: dict[str, Any], *, max_progress: float = 0.05) -> bool:
    if bool(row.get("label_contract_negative")):
        return True
    if bool(row.get("hit", row.get("label_hit", False))):
        return False
    progress = _float_or_none(row.get("normalized_progress"))
    return progress is not None and progress <= max_progress


def classify_hard_goal(row: dict[str, Any], *, max_progress: float = 0.05) -> str | None:
    if not is_hard_goal(row, max_progress=max_progress):
        return None
    supervised = has_action_supervision(row)
    if supervised:
        return "hard_positive"
    if bool(row.get("label_contract_negative")) or bool(row.get("negative_progress")):
        return "hard_negative"
    return "hard_unlabeled"


def hard_goal_example(row: dict[str, Any], *, max_progress: float = 0.05) -> HardGoalExample | None:
    category = classify_hard_goal(row, max_progress=max_progress)
    if category is None:
        return None
    return HardGoalExample(
        env_name=row.get("env_name"),
        target_mode=row.get("target_mode"),
        category=category,
        available_action_supervision=has_action_supervision(row),
        d_phi=_float_or_none(row.get("d_phi_start", row.get("d_phi"))),
        q_train_support=_float_or_none(row.get("q_train_support")),
        normalized_progress=_float_or_none(row.get("normalized_progress")),
        source_probe_id=row.get("source_probe_id"),
    )
