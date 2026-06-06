from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ContractThresholds:
    contractive_progress: float = 0.2
    low_progress: float = 0.0
    action_norm_high: float = 0.95


@dataclass(frozen=True)
class ContractGateResult:
    passed: bool
    reason: str
    contract_score: float
    contract_lcb: float
    contract_uncertainty: float
    predicted_hit: float
    predicted_contract_positive: float
    predicted_negative_progress: float
    contract_model_loaded: bool

    def trace_fields(self) -> dict[str, Any]:
        return {
            "contract_model_loaded": int(self.contract_model_loaded),
            "contract_score": self.contract_score,
            "contract_lcb": self.contract_lcb,
            "contract_uncertainty": self.contract_uncertainty,
            "predicted_hit": self.predicted_hit,
            "predicted_negative_progress": self.predicted_negative_progress,
            "contract_gate_pass": bool(self.passed),
            "contract_reject_reason": None if self.passed else self.reason,
        }


def evaluate_contract_gate(
    prediction: Any,
    *,
    lcb_threshold: float,
    negative_progress_threshold: float,
    target_mode: str,
    final_goal_threshold: float | None = None,
    recovery_threshold: float | None = None,
) -> ContractGateResult:
    threshold = float(lcb_threshold)
    if target_mode == "final_goal" and final_goal_threshold is not None:
        threshold = float(final_goal_threshold)
    if target_mode in {"recovery", "recovery_candidate"} and recovery_threshold is not None:
        threshold = float(recovery_threshold)
    lcb = float(prediction.lower_confidence_bound)
    positive = float(prediction.predicted_contract_positive)
    negative = float(prediction.predicted_negative_progress)
    score = min(positive, lcb)
    if negative > float(negative_progress_threshold):
        passed = False
        reason = "negative_progress_risk"
    elif lcb < threshold:
        passed = False
        reason = "low_contract_lcb"
    else:
        passed = True
        reason = "contract_pass"
    return ContractGateResult(
        passed=passed,
        reason=reason,
        contract_score=score,
        contract_lcb=lcb,
        contract_uncertainty=float(prediction.uncertainty),
        predicted_hit=float(prediction.predicted_hit),
        predicted_contract_positive=positive,
        predicted_negative_progress=negative,
        contract_model_loaded=bool(prediction.model_loaded),
    )


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_list(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def label_closed_loop_contract(record: dict[str, Any], thresholds: ContractThresholds | None = None) -> dict[str, Any]:
    thresholds = thresholds or ContractThresholds()
    hit = bool(record.get("hit", False))
    progress = _float_or_none(record.get("normalized_progress"))
    action_norm_max = _float_or_none(record.get("action_norm_max"))
    d_start = _float_or_none(record.get("d_phi_start", record.get("d_phi")))
    d_final = _float_or_none(record.get("final_d_phi", record.get("d_phi_end")))
    terminated = bool(record.get("terminated", False))
    truncated = bool(record.get("truncated", False))

    label_contractive = progress is not None and progress >= thresholds.contractive_progress
    label_negative_progress = progress is not None and progress < 0.0
    label_contract_positive = bool(hit or label_contractive)
    label_contract_negative = bool(label_negative_progress or (progress is not None and progress <= thresholds.low_progress))
    final_farther = d_start is not None and d_final is not None and d_final > d_start
    action_high = action_norm_max is not None and action_norm_max >= thresholds.action_norm_high
    label_unstable = bool(action_high or final_farther or (terminated and not hit) or (truncated and not hit))
    recovery_candidate = bool(record.get("recovery_candidate", False) or record.get("target_mode") == "recovery_candidate")
    return {
        "label_hit": hit,
        "label_contractive": bool(label_contractive),
        "label_negative_progress": bool(label_negative_progress),
        "label_unstable": bool(label_unstable),
        "label_good_contract": label_contract_positive,
        "label_contract_positive": label_contract_positive,
        "label_contract_negative": label_contract_negative,
        "label_recovery_bad": bool(recovery_candidate and label_contract_negative),
        "label_policy_weak": False,
    }


def contract_dataset_row(record: dict[str, Any], thresholds: ContractThresholds | None = None) -> dict[str, Any]:
    labels = label_closed_loop_contract(record, thresholds)
    phi_start = record.get("phi_start", record.get("phi_s"))
    phi_target = record.get("phi_target", record.get("phi_g"))
    phi_delta = None
    abs_phi_delta = None
    if phi_start is not None and phi_target is not None:
        try:
            import numpy as np

            phi_delta_arr = np.asarray(phi_target, dtype=float) - np.asarray(phi_start, dtype=float)
            phi_delta = phi_delta_arr.tolist()
            abs_phi_delta = np.abs(phi_delta_arr).tolist()
        except Exception:
            phi_delta = None
            abs_phi_delta = None
    target_mode = record.get("target_mode", record.get("pair_source"))
    negative_progress = bool(record.get("negative_progress", labels["label_negative_progress"]))
    return {
        "record_type": "closed_loop_contract_example",
        "source_probe_id": record.get("probe_id", record.get("pair_id")),
        "source_segment_id": record.get("source_segment_id"),
        "env_name": record.get("env_name"),
        "seed": record.get("seed"),
        "variant_source": record.get("variant_source", record.get("variant")),
        "phi_s": _as_list(phi_start),
        "phi_g": _as_list(phi_target),
        "phi_start": _as_list(phi_start),
        "phi_target": _as_list(phi_target),
        "phi_delta": phi_delta,
        "abs_phi_delta": abs_phi_delta,
        "d_phi": record.get("d_phi_start", record.get("qg_d_phi", record.get("d_phi"))),
        "d_phi_start": record.get("d_phi_start", record.get("qg_d_phi", record.get("d_phi"))),
        "pair_source": record.get("pair_source", target_mode),
        "target_mode": target_mode,
        "path_position": record.get("path_position"),
        "path_position_bucket": record.get("path_position_bucket"),
        "final_phase": bool(record.get("final_phase", False)),
        "recovery_candidate": bool(record.get("recovery_candidate", False)),
        "q_train_support": record.get("q_train_support"),
        "horizon": record.get("horizon"),
        "hit": bool(record.get("hit", False)),
        "normalized_progress": record.get("normalized_progress"),
        "negative_progress": negative_progress,
        "delta_phi": record.get("delta_phi"),
        "action_norm_mean": record.get("action_norm_mean"),
        "action_norm_max": record.get("action_norm_max"),
        "failure_reason": record.get("failure_reason"),
        **labels,
    }
