from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import GridSpec
from .gaussian import accumulate_gaussian_values
from .geometry import PrimitiveSet

DEFAULT_TEMPERATURE = 0.01
DEFAULT_TOP_K = 2
DEFAULT_CONFLICT_MARGIN = 0.02


@dataclass(frozen=True)
class PPSAResult:
    posterior: np.ndarray
    labels: np.ndarray
    known: np.ndarray
    unknown: np.ndarray
    conflict: np.ndarray
    score: np.ndarray
    margin: np.ndarray
    entropy: np.ndarray
    available_sources: np.ndarray


def scores_to_posterior(scores: np.ndarray, temperature: float = DEFAULT_TEMPERATURE) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim < 2 or temperature <= 0 or not np.isfinite(scores).all():
        raise ValueError("scores must be finite with a class axis and positive temperature")
    logits = scores / temperature
    logits -= logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return probabilities.astype(np.float32)


def retain_top_k(posterior: np.ndarray, top_k: int = DEFAULT_TOP_K) -> np.ndarray:
    posterior = np.asarray(posterior, dtype=np.float32)
    if posterior.ndim < 2 or top_k < 1 or top_k > posterior.shape[-1]:
        raise ValueError("invalid posterior or top_k")
    order = np.argsort(-posterior, axis=-1, kind="stable")[..., :top_k]
    retained = np.zeros_like(posterior)
    np.put_along_axis(retained, order, np.take_along_axis(posterior, order, axis=-1), axis=-1)
    mass = retained.sum(axis=-1, keepdims=True)
    return np.divide(retained, mass, out=np.zeros_like(retained), where=mass > 0)


def _decode_posterior(
    posterior: np.ndarray,
    known: np.ndarray,
    conflict: np.ndarray,
    available: np.ndarray,
    class_ids: np.ndarray,
) -> PPSAResult:
    order = np.argsort(-posterior, axis=-1, kind="stable")
    top = np.take_along_axis(posterior, order[..., :1], axis=-1)[..., 0]
    second = np.take_along_axis(posterior, order[..., 1:2], axis=-1)[..., 0]
    labels = np.zeros(known.shape, dtype=np.uint8)
    labels[known] = class_ids[order[..., 0][known]]
    safe = np.clip(posterior, 1e-12, 1.0)
    entropy = -(safe * np.log(safe)).sum(axis=-1)
    return PPSAResult(
        posterior=posterior.astype(np.float32),
        labels=labels,
        known=known,
        unknown=~known,
        conflict=conflict & known,
        score=np.where(known, top, 0).astype(np.float32),
        margin=np.where(known, top - second, 0).astype(np.float32),
        entropy=np.where(known, entropy, 0).astype(np.float32),
        available_sources=available.astype(np.uint8),
    )


def provenance_preserving_arbitration(
    source_posteriors: np.ndarray,
    source_valid: np.ndarray,
    *,
    source_conflict: np.ndarray | None = None,
    class_ids: np.ndarray | None = None,
    conflict_margin: float = DEFAULT_CONFLICT_MARGIN,
) -> PPSAResult:
    """Average only available source posteriors and retain typed conflict."""
    source_posteriors = np.asarray(source_posteriors, dtype=np.float32)
    source_valid = np.asarray(source_valid, dtype=bool)
    if source_posteriors.ndim < 3 or source_valid.shape != source_posteriors.shape[:-1]:
        raise ValueError("expected posteriors [B,...,C] and valid [B,...]")
    if conflict_margin < 0:
        raise ValueError("conflict_margin must be nonnegative")
    class_count = source_posteriors.shape[-1]
    if class_ids is None:
        class_ids = np.arange(1, class_count + 1, dtype=np.uint8)
    class_ids = np.asarray(class_ids, dtype=np.uint8)
    if class_ids.shape != (class_count,):
        raise ValueError("class_ids do not match posterior class axis")
    sums = source_posteriors.sum(axis=-1)
    if not np.allclose(sums[source_valid], 1.0, atol=1e-5):
        raise ValueError("valid source posteriors must sum to one")
    available = source_valid.sum(axis=0)
    posterior = np.divide(
        (source_posteriors * source_valid[..., None]).sum(axis=0),
        available[..., None],
        out=np.zeros_like(source_posteriors[0]),
        where=available[..., None] > 0,
    )
    known = available > 0
    winners = np.argmax(source_posteriors, axis=-1)
    first = np.argmax(source_valid, axis=0)
    reference = np.take_along_axis(winners, first[None], axis=0)[0]
    disagreement = (source_valid & (winners != reference[None])).any(axis=0)
    if source_conflict is None:
        source_conflict = np.zeros_like(source_valid)
    source_conflict = np.asarray(source_conflict, dtype=bool)
    if source_conflict.shape != source_valid.shape:
        raise ValueError("source_conflict must match source_valid")
    ordered = np.sort(posterior, axis=-1)
    low_margin = known & ((ordered[..., -1] - ordered[..., -2]) < conflict_margin)
    conflict = disagreement | (source_conflict & source_valid).any(axis=0) | low_margin
    return _decode_posterior(posterior, known, conflict, available, class_ids)


def gaussian_source_posteriors(
    primitives: PrimitiveSet,
    primitive_posteriors: np.ndarray,
    primitive_valid: np.ndarray,
    grid: GridSpec,
    *,
    primitive_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate backend-conditional primitive posteriors before PPSA."""
    primitive_posteriors = np.asarray(primitive_posteriors, dtype=np.float32)
    primitive_valid = np.asarray(primitive_valid, dtype=bool)
    if primitive_posteriors.ndim != 3 or primitive_posteriors.shape[1] != len(primitives):
        raise ValueError("primitive_posteriors must have shape [B,N,C]")
    if primitive_valid.shape != primitive_posteriors.shape[:2]:
        raise ValueError("primitive_valid must have shape [B,N]")
    outputs = []
    valid_outputs = []
    for backend in range(primitive_posteriors.shape[0]):
        weight = primitive_valid[backend].astype(np.float32)
        if primitive_weight is not None:
            weight *= np.asarray(primitive_weight, dtype=np.float32)
        numerator, denominator = accumulate_gaussian_values(
            primitives, primitive_posteriors[backend], grid, primitive_weight=weight
        )
        posterior = np.divide(
            numerator,
            numerator.sum(axis=-1, keepdims=True),
            out=np.zeros_like(numerator),
            where=numerator.sum(axis=-1, keepdims=True) > 0,
        )
        outputs.append(posterior)
        valid_outputs.append(denominator > 0)
    return np.stack(outputs).astype(np.float32), np.stack(valid_outputs)
