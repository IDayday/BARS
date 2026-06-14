from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from phase3e.edge_risk_calibration import EdgeRiskCalibrationConfig, calibrate_edge_risk, make_planner_certification
from phase3e.policy_likelihood import compute_model_edge_mse, edge_policy_support_scores
from phase3e.repair_edge_certification import evaluate_repair_certified_planning


@dataclass(frozen=True)
class DirectRepairPolicyConfig:
    batch_size: int = 2048
    max_examples: int = 50000
    temperature: float = 0.05
    seed: int = 0
    device: str | None = None
    certification_source: str = "repair_direct_policy_mse"


def repair_bank_segment_indices(
    bank_edge_segments: dict[str, np.ndarray],
    bank_edge_ids: np.ndarray | list[int],
) -> np.ndarray:
    edge_ids = np.asarray(bank_edge_segments["edge_id"], dtype=np.int64)
    wanted = np.asarray(bank_edge_ids, dtype=np.int64)
    if wanted.size == 0 or edge_ids.size == 0:
        return np.empty(0, dtype=np.int64)
    return np.flatnonzero(np.isin(edge_ids, wanted)).astype(np.int64)


def direct_repair_edge_policy_scores(
    dataset: dict[str, Any],
    repair_edge_map: pd.DataFrame,
    bank_option_edges: pd.DataFrame,
    bank_edge_segments: dict[str, np.ndarray],
    model_path: str,
    config: DirectRepairPolicyConfig | None = None,
) -> pd.DataFrame:
    """Evaluate a trained GCBC model directly on selected repair-bank segments."""

    config = config or DirectRepairPolicyConfig()
    if repair_edge_map.empty:
        return pd.DataFrame()
    required = {"edge_id", "bank_edge_id"}
    missing = required.difference(repair_edge_map.columns)
    if missing:
        raise ValueError(f"repair_edge_map missing required columns: {sorted(missing)}")

    repair_map = repair_edge_map.copy()
    repair_map["edge_id"] = repair_map["edge_id"].astype(int)
    repair_map["bank_edge_id"] = repair_map["bank_edge_id"].astype(int)
    bank_ids = repair_map["bank_edge_id"].to_numpy(dtype=np.int64)
    segment_indices = repair_bank_segment_indices(bank_edge_segments, bank_ids)
    if segment_indices.size == 0:
        out = repair_map.copy()
        out["edge_action_mse"] = np.nan
        out["edge_action_mse_ucb"] = np.nan
        out["direct_edge_policy_support_score"] = 0.0
        out["num_policy_eval_samples"] = 0
        return out

    bank_edges = bank_option_edges[bank_option_edges["edge_id"].astype(int).isin(set(bank_ids.tolist()))].copy()
    edge_mse = compute_model_edge_mse(
        dataset=dataset,
        option_edges=bank_edges,
        edge_segments=bank_edge_segments,
        heldout_segment_indices=segment_indices,
        model_path=model_path,
        batch_size=int(config.batch_size),
        max_examples=int(config.max_examples),
        seed=int(config.seed),
        device=config.device,
    )
    scored = edge_policy_support_scores(edge_mse, temperature=float(config.temperature)).rename(
        columns={
            "edge_id": "bank_edge_id",
            "edge_policy_support_score": "direct_edge_policy_support_score",
        }
    )
    scored["bank_edge_id"] = scored["bank_edge_id"].astype(int)
    out = repair_map.merge(scored, on="bank_edge_id", how="left")
    out["edge_action_mse"] = pd.to_numeric(out.get("edge_action_mse", np.nan), errors="coerce")
    out["edge_action_mse_ucb"] = pd.to_numeric(out.get("edge_action_mse_ucb", np.nan), errors="coerce")
    out["direct_edge_policy_support_score"] = pd.to_numeric(
        out.get("direct_edge_policy_support_score", 0.0), errors="coerce"
    ).fillna(0.0)
    out["num_policy_eval_samples"] = pd.to_numeric(
        out.get("num_policy_eval_samples", 0), errors="coerce"
    ).fillna(0).astype(int)
    return out


def apply_direct_policy_scores(
    repair_certification: pd.DataFrame,
    direct_scores: pd.DataFrame,
    calibration_config: EdgeRiskCalibrationConfig,
    certification_source: str = "repair_direct_policy_mse",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace repair-edge transfer policy scores with direct GCBC evidence."""

    cert = repair_certification.copy()
    if "is_repair_edge" not in cert.columns:
        cert["is_repair_edge"] = False
    direct = direct_scores.copy()
    if direct.empty:
        calibrated = calibrate_edge_risk(cert, calibration_config)
        return calibrated, make_planner_certification(calibrated)
    direct = direct.set_index("edge_id")
    repair_mask = cert["is_repair_edge"].astype(bool) & cert["edge_id"].astype(int).isin(direct.index)
    for idx in cert[repair_mask].index:
        edge_id = int(cert.loc[idx, "edge_id"])
        row = direct.loc[edge_id]
        cert.loc[idx, "edge_action_mse"] = row.get("edge_action_mse", np.nan)
        cert.loc[idx, "edge_action_mse_ucb"] = row.get("edge_action_mse_ucb", np.nan)
        cert.loc[idx, "edge_policy_support_score"] = float(row.get("direct_edge_policy_support_score", 0.0))
        cert.loc[idx, "direct_edge_policy_support_score"] = float(row.get("direct_edge_policy_support_score", 0.0))
        cert.loc[idx, "num_policy_eval_samples"] = int(row.get("num_policy_eval_samples", 0))
        cert.loc[idx, "certification_source"] = certification_source
    calibrated = calibrate_edge_risk(cert, calibration_config)
    planner = make_planner_certification(calibrated)
    return calibrated, planner


def direct_vs_transfer_diagnostics(
    repair_certification_before: pd.DataFrame,
    repair_certification_after: pd.DataFrame,
    direct_scores: pd.DataFrame,
) -> dict[str, Any]:
    before = repair_certification_before[_bool_mask(repair_certification_before, "is_repair_edge")].copy()
    after = repair_certification_after[_bool_mask(repair_certification_after, "is_repair_edge")].copy()
    direct = direct_scores.copy()
    out: dict[str, Any] = {
        "num_repair_edges": int(before.shape[0]),
        "num_direct_scored_edges": int(direct["edge_id"].nunique()) if "edge_id" in direct.columns else 0,
        "note": "Direct repair policy MSE is offline supervised evidence, not rollout success.",
    }
    if not direct.empty:
        out["mean_direct_edge_action_mse"] = float(pd.to_numeric(direct["edge_action_mse"], errors="coerce").mean())
        out["median_direct_edge_action_mse"] = float(pd.to_numeric(direct["edge_action_mse"], errors="coerce").median())
        out["mean_direct_policy_support_score"] = float(
            pd.to_numeric(direct["direct_edge_policy_support_score"], errors="coerce").mean()
        )
        out["median_direct_policy_support_score"] = float(
            pd.to_numeric(direct["direct_edge_policy_support_score"], errors="coerce").median()
        )
    if not before.empty and not after.empty:
        before_scores = before.set_index("edge_id")["edge_policy_support_score"]
        after_scores = after.set_index("edge_id")["edge_policy_support_score"]
        common = before_scores.index.intersection(after_scores.index)
        if len(common):
            out["mean_policy_score_delta_direct_minus_transfer"] = float(
                after_scores.loc[common].mean() - before_scores.loc[common].mean()
            )
            out["spearman_transfer_vs_direct_policy_score"] = _safe_corr(
                before_scores.loc[common], after_scores.loc[common], method="spearman"
            )
        transfer_cert_col = "calibrated_certified_binary" if "calibrated_certified_binary" in before.columns else "certified_offline_binary"
        direct_cert_col = "calibrated_certified_binary" if "calibrated_certified_binary" in after.columns else "certified_offline_binary"
        out["transfer_certified_rate"] = float(before.get(transfer_cert_col, False).astype(bool).mean())
        out["direct_certified_rate"] = float(after.get(direct_cert_col, False).astype(bool).mean())
        out["mean_transfer_reliability"] = float(
            pd.to_numeric(before.get("calibrated_edge_reliability_score", before.get("edge_proxy_score", np.nan)), errors="coerce").mean()
        )
        out["mean_direct_reliability"] = float(
            pd.to_numeric(after.get("calibrated_edge_reliability_score", after.get("edge_proxy_score", np.nan)), errors="coerce").mean()
        )
    return out


def _bool_mask(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    values = df[col]
    if values.dtype == bool:
        return values.fillna(False).astype(bool)
    return values.map(lambda x: str(x).strip().lower() in {"1", "true", "yes", "y"}).fillna(False).astype(bool)


def _safe_corr(a: pd.Series, b: pd.Series, method: str) -> float | None:
    corr = pd.to_numeric(a, errors="coerce").corr(pd.to_numeric(b, errors="coerce"), method=method)
    return None if pd.isna(corr) else float(corr)


def evaluate_direct_repair_policy_planning(
    augmented_edges: pd.DataFrame,
    pair_compatibility: pd.DataFrame,
    path_queries: pd.DataFrame,
    direct_planner_certification: pd.DataFrame,
    planner_config: Any,
    methods: list[str],
    max_queries: int | None = None,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return evaluate_repair_certified_planning(
        augmented_edges=augmented_edges,
        pair_compatibility=pair_compatibility,
        path_queries=path_queries,
        planner_certification=direct_planner_certification,
        planner_config=planner_config,
        methods=methods,
        max_queries=max_queries,
        seed=seed,
    )
