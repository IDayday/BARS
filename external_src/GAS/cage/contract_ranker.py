from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cage.contract_model import ContractPrediction, ContractScorer


@dataclass
class ContractCandidate:
    target: Any
    index: int | None
    target_mode: str
    source: str
    path_position: int | None
    final_phase: bool
    is_original_gas: bool
    is_current_committed: bool
    is_recovery: bool
    distance: float
    path_progress_score: float
    switch_cost: float
    features: dict[str, Any] = field(default_factory=dict)
    prediction: ContractPrediction | None = None
    score: float | None = None
    rejected: bool = False
    reject_reason: str | None = None


@dataclass
class ContractRankResult:
    selected_candidate: ContractCandidate
    candidates: list[ContractCandidate]
    selected_score: float
    gas_score: float | None
    best_non_gas_score: float | None
    rejected_count: int
    extreme_reject_count: int
    candidate_count: int
    coverage: float
    ranking_reason: str


def rank_contract_candidates(
    candidates: list[ContractCandidate],
    scorer: ContractScorer,
    *,
    contract_weight: float = 1.0,
    progress_weight: float = 0.5,
    negative_weight: float = 0.5,
    uncertainty_weight: float = 0.25,
    switch_penalty: float = 0.10,
    extreme_negative_threshold: float = 0.90,
    prefer_gas_margin: float = 0.05,
    min_candidate_coverage: float = 0.30,
) -> ContractRankResult:
    if not candidates:
        raise ValueError("rank_contract_candidates requires at least one candidate")

    deduped = _dedupe_candidates(candidates)
    for candidate in deduped:
        prediction = scorer.predict(candidate.features)
        candidate.prediction = prediction
        candidate.rejected = bool(prediction.predicted_negative_progress > float(extreme_negative_threshold))
        candidate.reject_reason = "extreme_negative_progress" if candidate.rejected else None
        candidate.score = contract_rank_score(
            prediction,
            path_progress_score=candidate.path_progress_score,
            switch_cost=candidate.switch_cost,
            contract_weight=contract_weight,
            progress_weight=progress_weight,
            negative_weight=negative_weight,
            uncertainty_weight=uncertainty_weight,
            switch_penalty=switch_penalty,
        )

    extreme_reject_count = sum(1 for candidate in deduped if candidate.rejected)
    _enforce_min_coverage(deduped, float(min_candidate_coverage))

    active = [candidate for candidate in deduped if not candidate.rejected]
    if not active:
        active = list(deduped)
        for candidate in active:
            candidate.rejected = False
            candidate.reject_reason = None

    gas = next((candidate for candidate in active if candidate.is_original_gas), None)
    committed = next((candidate for candidate in active if candidate.is_current_committed), None)
    best = max(active, key=lambda item: float(item.score if item.score is not None else -np.inf))
    best_non_gas = max(
        (candidate for candidate in active if not candidate.is_original_gas),
        key=lambda item: float(item.score if item.score is not None else -np.inf),
        default=None,
    )
    gas_score = _score(gas)
    best_non_gas_score = _score(best_non_gas)

    if gas is not None and (best_non_gas is None or best_non_gas_score is None or best_non_gas_score < gas_score + float(prefer_gas_margin)):
        selected = gas
        reason = "prefer_gas_margin"
    elif committed is not None and _score(committed) is not None and _score(committed) >= _score(best) - float(prefer_gas_margin):
        selected = committed
        reason = "continue_committed_close_to_best"
    else:
        selected = best
        reason = "best_ranked_contract"

    rejected_count = sum(1 for candidate in deduped if candidate.rejected)
    coverage = float((len(deduped) - rejected_count) / max(len(deduped), 1))
    return ContractRankResult(
        selected_candidate=selected,
        candidates=deduped,
        selected_score=float(selected.score if selected.score is not None else 0.0),
        gas_score=gas_score,
        best_non_gas_score=best_non_gas_score,
        rejected_count=int(rejected_count),
        extreme_reject_count=int(extreme_reject_count),
        candidate_count=int(len(deduped)),
        coverage=coverage,
        ranking_reason=reason,
    )


def contract_rank_score(
    prediction: ContractPrediction,
    *,
    path_progress_score: float,
    switch_cost: float,
    contract_weight: float,
    progress_weight: float,
    negative_weight: float,
    uncertainty_weight: float,
    switch_penalty: float,
) -> float:
    return float(
        float(contract_weight) * float(prediction.lower_confidence_bound)
        + float(progress_weight) * float(path_progress_score)
        - float(negative_weight) * float(prediction.predicted_negative_progress)
        - float(uncertainty_weight) * float(prediction.uncertainty)
        - float(switch_penalty) * float(switch_cost)
    )


def candidate_trace(candidate: ContractCandidate) -> dict[str, Any]:
    prediction = candidate.prediction
    return {
        "source": candidate.source,
        "target_mode": candidate.target_mode,
        "index": candidate.index,
        "score": candidate.score,
        "distance": candidate.distance,
        "path_progress_score": candidate.path_progress_score,
        "switch_cost": candidate.switch_cost,
        "rejected": candidate.rejected,
        "reject_reason": candidate.reject_reason,
        "predicted_hit": None if prediction is None else prediction.predicted_hit,
        "predicted_negative_progress": None if prediction is None else prediction.predicted_negative_progress,
        "contract_lcb": None if prediction is None else prediction.lower_confidence_bound,
        "contract_uncertainty": None if prediction is None else prediction.uncertainty,
        "contract_model_loaded": None if prediction is None else int(prediction.model_loaded),
    }


def _score(candidate: ContractCandidate | None) -> float | None:
    if candidate is None or candidate.score is None:
        return None
    return float(candidate.score)


def _dedupe_candidates(candidates: list[ContractCandidate]) -> list[ContractCandidate]:
    out: list[ContractCandidate] = []
    for candidate in candidates:
        target = np.asarray(candidate.target).reshape(-1)
        duplicate = False
        for existing in out:
            existing_target = np.asarray(existing.target).reshape(-1)
            if target.shape == existing_target.shape and float(np.linalg.norm(target - existing_target)) <= 1e-8:
                duplicate = True
                if candidate.is_original_gas and not existing.is_original_gas:
                    existing.is_original_gas = True
                    existing.source = "gas"
                    existing.target_mode = "gas_path"
                    existing.index = candidate.index
                break
        if not duplicate:
            out.append(candidate)
    return out


def _enforce_min_coverage(candidates: list[ContractCandidate], min_candidate_coverage: float) -> None:
    if not candidates:
        return
    min_keep = max(1, int(np.ceil(float(min_candidate_coverage) * len(candidates))))
    active = [candidate for candidate in candidates if not candidate.rejected]
    if len(active) >= min_keep:
        return
    priority = sorted(
        [candidate for candidate in candidates if candidate.rejected],
        key=lambda item: (
            0 if item.is_original_gas else 1 if item.is_current_committed else 2,
            -float(item.score if item.score is not None else -np.inf),
        ),
    )
    for candidate in priority:
        candidate.rejected = False
        candidate.reject_reason = "coverage_floor_unrejected"
        active.append(candidate)
        if len(active) >= min_keep:
            break
