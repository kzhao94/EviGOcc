from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_RARE_CLASS_IDS = (4, 5, 6, 7)


@dataclass(frozen=True)
class RASCResult:
    prediction: np.ndarray
    occupied: np.ndarray
    risk: np.ndarray
    commitment_score: np.ndarray
    committed: np.ndarray
    abstained: np.ndarray
    conflict_risk: np.ndarray
    rare_risk: np.ndarray


def reliability_aware_commitment(
    prediction: np.ndarray,
    occupied: np.ndarray,
    semantic_unknown: np.ndarray,
    semantic_conflict: np.ndarray,
    backend_agreement: np.ndarray,
    causal_support: np.ndarray,
    supported_label: np.ndarray,
    *,
    rare_class_ids: tuple[int, ...] = DEFAULT_RARE_CLASS_IDS,
    conflict_weight: float = 0.65,
    unknown_weight: float = 0.35,
) -> RASCResult:
    """Expose semantic commitment without altering occupied geometry."""
    prediction = np.asarray(prediction, dtype=np.uint8)
    occupied = np.asarray(occupied, dtype=bool)
    semantic_unknown = np.asarray(semantic_unknown, dtype=bool)
    semantic_conflict = np.asarray(semantic_conflict, dtype=bool)
    backend_agreement = np.asarray(backend_agreement, dtype=bool)
    causal_support = np.asarray(causal_support, dtype=bool)
    supported_label = np.asarray(supported_label, dtype=np.uint8)
    arrays = (
        occupied,
        semantic_unknown,
        semantic_conflict,
        backend_agreement,
        causal_support,
        supported_label,
    )
    if any(item.shape != prediction.shape for item in arrays):
        raise ValueError("all RASC inputs must share shape")
    if min(conflict_weight, unknown_weight) < 0:
        raise ValueError("RASC weights must be nonnegative")
    conflict_risk = occupied.astype(np.float32) * np.clip(
        conflict_weight * semantic_conflict.astype(np.float32)
        + unknown_weight * semantic_unknown.astype(np.float32),
        0.0,
        1.0,
    )
    predicted_rare = occupied & np.isin(prediction, rare_class_ids)
    rare_supported = (
        predicted_rare & backend_agreement & causal_support & (supported_label == prediction)
    )
    rare_risk = predicted_rare & ~rare_supported
    risk = np.maximum(conflict_risk, rare_risk.astype(np.float32))
    risk *= occupied.astype(np.float32)
    committed = occupied & (risk == 0)
    abstained = occupied & ~committed
    commitment_score = (1.0 - risk) * occupied.astype(np.float32)
    return RASCResult(
        prediction=prediction.copy(),
        occupied=occupied.copy(),
        risk=risk.astype(np.float32),
        commitment_score=commitment_score.astype(np.float32),
        committed=committed,
        abstained=abstained,
        conflict_risk=conflict_risk.astype(np.float32),
        rare_risk=rare_risk.astype(np.float32),
    )
