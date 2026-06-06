from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external_src" / "GAS"))

from cage.contract_model import ContractPrediction
from cage.contract_ranker import ContractCandidate, rank_contract_candidates


class DummyScorer:
    loaded = True

    def predict(self, features):
        positive = float(features.get("positive", 0.5))
        negative = float(features.get("negative", 0.1))
        uncertainty = float(features.get("uncertainty", 0.1))
        return ContractPrediction(
            predicted_hit=positive,
            predicted_contract_positive=positive,
            predicted_negative_progress=negative,
            uncertainty=uncertainty,
            lower_confidence_bound=max(0.0, positive - 0.1),
            model_loaded=True,
        )


def candidate(source, *, positive, negative=0.1, gas=False, committed=False, progress=0.0):
    return ContractCandidate(
        target=np.asarray([positive]),
        index=0,
        target_mode="gas_path" if gas else source,
        source=source,
        path_position=0,
        final_phase=False,
        is_original_gas=gas,
        is_current_committed=committed,
        is_recovery=False,
        distance=1.0,
        path_progress_score=progress,
        switch_cost=0.0 if committed else 1.0,
        features={"positive": positive, "negative": negative},
    )


def test_ranker_prefers_gas_unless_non_gas_beats_margin():
    result = rank_contract_candidates(
        [
            candidate("gas", positive=0.60, gas=True),
            candidate("cage", positive=0.62, progress=0.0),
        ],
        DummyScorer(),
        prefer_gas_margin=0.10,
    )

    assert result.selected_candidate.is_original_gas
    assert result.ranking_reason == "prefer_gas_margin"


def test_ranker_selects_non_gas_when_contract_score_is_better():
    result = rank_contract_candidates(
        [
            candidate("gas", positive=0.35, gas=True),
            candidate("cage", positive=0.90, progress=1.0),
        ],
        DummyScorer(),
        prefer_gas_margin=0.05,
    )

    assert result.selected_candidate.source == "cage"
    assert result.best_non_gas_score is not None
    assert result.gas_score is not None
    assert result.best_non_gas_score > result.gas_score


def test_ranker_only_hard_rejects_extreme_negative_risk():
    result = rank_contract_candidates(
        [
            candidate("gas", positive=0.20, negative=0.95, gas=True),
            candidate("cage", positive=0.10, negative=0.96),
            candidate("committed", positive=0.11, negative=0.97, committed=True),
        ],
        DummyScorer(),
        extreme_negative_threshold=0.90,
        min_candidate_coverage=0.50,
    )

    assert result.extreme_reject_count == 3
    assert result.coverage >= 0.50
    assert result.selected_candidate.source in {"gas", "committed"}
