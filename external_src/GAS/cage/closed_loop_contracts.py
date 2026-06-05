from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ContractThresholds:
    contractive_progress: float = 0.2
    action_norm_high: float = 0.95


def label_closed_loop_contract(record: dict[str, Any], thresholds: ContractThresholds | None = None) -> dict[str, Any]:
    thresholds = thresholds or ContractThresholds()
    hit = bool(record.get("hit", False))
    progress = record.get("normalized_progress")
    action_norm_max = record.get("action_norm_max")
    d_start = record.get("d_phi_start")
    d_final = record.get("final_d_phi", record.get("d_phi_end"))
    terminated = bool(record.get("terminated", False))
    truncated = bool(record.get("truncated", False))

    label_contractive = progress is not None and float(progress) >= thresholds.contractive_progress
    label_negative_progress = progress is not None and float(progress) < 0.0
    final_farther = d_start is not None and d_final is not None and float(d_final) > float(d_start)
    action_high = action_norm_max is not None and float(action_norm_max) >= thresholds.action_norm_high
    label_unstable = bool(action_high or final_farther or (terminated and not hit) or (truncated and not hit))
    return {
        "label_hit": hit,
        "label_contractive": bool(label_contractive),
        "label_negative_progress": bool(label_negative_progress),
        "label_unstable": bool(label_unstable),
        "label_good_contract": bool(hit or label_contractive),
    }


def contract_dataset_row(record: dict[str, Any], thresholds: ContractThresholds | None = None) -> dict[str, Any]:
    labels = label_closed_loop_contract(record, thresholds)
    return {
        "phi_s": record.get("phi_s"),
        "phi_g": record.get("phi_g"),
        "d_phi": record.get("d_phi_start", record.get("qg_d_phi")),
        "pair_source": record.get("pair_source"),
        "path_position": record.get("path_position"),
        "path_position_bucket": record.get("path_position_bucket"),
        "final_phase": bool(record.get("final_phase", False)),
        "recovery_candidate": bool(record.get("recovery_candidate", False)),
        "q_train_support": record.get("q_train_support"),
        "hit": bool(record.get("hit", False)),
        "normalized_progress": record.get("normalized_progress"),
        "delta_phi": record.get("delta_phi"),
        "action_norm_max": record.get("action_norm_max"),
        "failure_reason": record.get("failure_reason"),
        **labels,
    }
