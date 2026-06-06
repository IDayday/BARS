from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cage.contract_ranker import ContractCandidate


@dataclass(frozen=True)
class InterventionCandidate:
    candidate: ContractCandidate
    intervention_gain: float
    path_index_gain: float
    final_goal_progress_gain: float
    gas_risk: float
    gas_progress_stalled: bool
    blocked_reason: str | None = None


@dataclass(frozen=True)
class InterventionDecision:
    selected_candidate: ContractCandidate
    selected_source: str
    gas_candidate: ContractCandidate
    gas_score: float
    best_alternative_score: float | None
    intervention_gain: float
    intervention_allowed: bool
    intervention_reason: str
    gas_risk: float
    gas_progress_stalled: bool
    committed_locked_out: bool
    final_phase_preserved: bool
    candidates: list[InterventionCandidate]


def compute_intervention_gain(
    alternative: ContractCandidate,
    gas: ContractCandidate,
    *,
    final_goal_progress_gain: float,
    path_index_gain: float,
    intervention_cost: float,
    switch_cost: float | None = None,
) -> float:
    """Score the marginal value of replacing the GAS target."""
    alt_score = _score(alternative)
    gas_score = _score(gas)
    switch = float(alternative.switch_cost if switch_cost is None else switch_cost)
    return float(
        alt_score
        - gas_score
        + float(final_goal_progress_gain)
        + float(path_index_gain)
        - float(intervention_cost)
        - switch
    )


def decide_contract_intervention(
    candidates: list[ContractCandidate],
    *,
    intervention_margin: float,
    gas_risk_threshold: float,
    min_final_progress_gain: float,
    min_path_index_gain: float,
    intervention_cost: float,
    extreme_negative_threshold: float,
    preserve_final_phase: bool,
    allow_final_override_only_extreme: bool,
    gas_progress_stalled: bool,
    committed_locked_out: bool,
) -> InterventionDecision:
    if not candidates:
        raise ValueError("decide_contract_intervention requires at least one candidate")

    gas = next((candidate for candidate in candidates if candidate.is_original_gas), candidates[0])
    gas_score = _score(gas)
    gas_risk = _risk(gas)
    gas_extreme = gas_risk > float(extreme_negative_threshold)

    final_phase_preserved = False
    if preserve_final_phase and gas.final_phase and not gas_extreme:
        final_phase_preserved = True
        return InterventionDecision(
            selected_candidate=gas,
            selected_source=gas.source,
            gas_candidate=gas,
            gas_score=gas_score,
            best_alternative_score=_best_alt_score(candidates, gas),
            intervention_gain=0.0,
            intervention_allowed=False,
            intervention_reason="final_phase_preserved",
            gas_risk=gas_risk,
            gas_progress_stalled=bool(gas_progress_stalled),
            committed_locked_out=bool(committed_locked_out),
            final_phase_preserved=True,
            candidates=_wrap_candidates(
                candidates,
                gas,
                gas_risk=gas_risk,
                gas_progress_stalled=gas_progress_stalled,
                intervention_cost=intervention_cost,
            ),
        )

    wrapped = _wrap_candidates(
        candidates,
        gas,
        gas_risk=gas_risk,
        gas_progress_stalled=gas_progress_stalled,
        intervention_cost=intervention_cost,
    )
    alternatives = [item for item in wrapped if item.candidate is not gas and item.blocked_reason is None]
    best = max(alternatives, key=lambda item: item.intervention_gain, default=None)
    best_alt_score = _best_alt_score(candidates, gas)

    if best is None:
        reason = "no_valid_alternative"
    elif not (gas_risk > float(gas_risk_threshold) or gas_progress_stalled):
        reason = "gas_contract_not_risky"
    elif best.path_index_gain < float(min_path_index_gain):
        reason = "path_index_gain_too_small"
    elif best.final_goal_progress_gain < float(min_final_progress_gain):
        reason = "final_progress_gain_too_small"
    elif best.intervention_gain <= float(intervention_margin):
        reason = "intervention_gain_below_margin"
    else:
        return InterventionDecision(
            selected_candidate=best.candidate,
            selected_source=best.candidate.source,
            gas_candidate=gas,
            gas_score=gas_score,
            best_alternative_score=best_alt_score,
            intervention_gain=float(best.intervention_gain),
            intervention_allowed=True,
            intervention_reason="intervention_allowed",
            gas_risk=gas_risk,
            gas_progress_stalled=bool(gas_progress_stalled),
            committed_locked_out=bool(committed_locked_out),
            final_phase_preserved=False,
            candidates=wrapped,
        )

    return InterventionDecision(
        selected_candidate=gas,
        selected_source=gas.source,
        gas_candidate=gas,
        gas_score=gas_score,
        best_alternative_score=best_alt_score,
        intervention_gain=0.0 if best is None else float(best.intervention_gain),
        intervention_allowed=False,
        intervention_reason=reason,
        gas_risk=gas_risk,
        gas_progress_stalled=bool(gas_progress_stalled),
        committed_locked_out=bool(committed_locked_out),
        final_phase_preserved=final_phase_preserved,
        candidates=wrapped,
    )


def decision_trace(decision: InterventionDecision) -> dict[str, Any]:
    return {
        "contract_intervene_enabled": True,
        "intervention_allowed": bool(decision.intervention_allowed),
        "intervention_reason": decision.intervention_reason,
        "intervention_selected_source": decision.selected_source,
        "intervention_gain": float(decision.intervention_gain),
        "intervention_gas_score": float(decision.gas_score),
        "intervention_best_alternative_score": decision.best_alternative_score,
        "intervention_gas_risk": float(decision.gas_risk),
        "intervention_gas_progress_stalled": bool(decision.gas_progress_stalled),
        "committed_locked_out": bool(decision.committed_locked_out),
        "final_phase_preserved": bool(decision.final_phase_preserved),
    }


def _wrap_candidates(
    candidates: list[ContractCandidate],
    gas: ContractCandidate,
    *,
    gas_risk: float,
    gas_progress_stalled: bool,
    intervention_cost: float,
) -> list[InterventionCandidate]:
    wrapped: list[InterventionCandidate] = []
    gas_position = _position(gas)
    gas_progress = float(gas.path_progress_score)
    for candidate in candidates:
        path_index_gain = max(0.0, _position(candidate) - gas_position)
        final_goal_progress_gain = max(0.0, float(candidate.path_progress_score) - gas_progress)
        blocked = None
        if candidate is not gas and candidate.rejected:
            blocked = candidate.reject_reason or "candidate_rejected"
        elif candidate is not gas and _position(candidate) < gas_position:
            blocked = "path_position_regression"
        gain = compute_intervention_gain(
            candidate,
            gas,
            final_goal_progress_gain=final_goal_progress_gain,
            path_index_gain=path_index_gain,
            intervention_cost=intervention_cost,
        )
        wrapped.append(
            InterventionCandidate(
                candidate=candidate,
                intervention_gain=float(gain),
                path_index_gain=float(path_index_gain),
                final_goal_progress_gain=float(final_goal_progress_gain),
                gas_risk=float(gas_risk),
                gas_progress_stalled=bool(gas_progress_stalled),
                blocked_reason=blocked,
            )
        )
    return wrapped


def _score(candidate: ContractCandidate | None) -> float:
    if candidate is None or candidate.score is None:
        return 0.0
    return float(candidate.score)


def _risk(candidate: ContractCandidate | None) -> float:
    if candidate is None or candidate.prediction is None:
        return 1.0
    return float(candidate.prediction.predicted_negative_progress)


def _position(candidate: ContractCandidate | None) -> float:
    if candidate is None or candidate.path_position is None:
        return 0.0
    try:
        return float(candidate.path_position)
    except (TypeError, ValueError):
        return 0.0


def _best_alt_score(candidates: list[ContractCandidate], gas: ContractCandidate) -> float | None:
    scores = [_score(candidate) for candidate in candidates if candidate is not gas]
    return max(scores) if scores else None
