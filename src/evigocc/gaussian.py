from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import GridSpec
from .geometry import PrimitiveSet

DEFAULT_SIGMA_M = 0.20
DEFAULT_SUPPORT_MAHALANOBIS_SQUARED = 9.0
DEFAULT_HAZARD_THRESHOLD = float(np.log(2.0))


@dataclass(frozen=True)
class GaussianReadout:
    occupied: np.ndarray
    observed: np.ndarray
    hazard: np.ndarray
    source_frame_count: np.ndarray
    newest_source_age: np.ndarray


def _support_offsets(grid: GridSpec, sigma: float, support_squared: float) -> np.ndarray:
    radius_m = np.sqrt(support_squared) * sigma
    radius = np.ceil(radius_m / np.asarray(grid.voxel_size)).astype(np.int64)
    axes = [np.arange(-value, value + 1) for value in radius]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


def _iter_support(
    primitives: PrimitiveSet,
    grid: GridSpec,
    sigma: float,
    support_squared: float,
    chunk_size: int,
):
    if sigma <= 0 or support_squared <= 0 or chunk_size < 1:
        raise ValueError("sigma, support, and chunk_size must be positive")
    offsets = _support_offsets(grid, sigma, support_squared)
    minimum = np.asarray(grid.minimum, dtype=np.float64)
    voxel_size = np.asarray(grid.voxel_size, dtype=np.float64)
    shape = np.asarray(grid.shape, dtype=np.int64)
    precision = 1.0 / sigma**2
    for start in range(0, len(primitives), chunk_size):
        end = min(start + chunk_size, len(primitives))
        anchors = primitives.voxel_indices[start:end].astype(np.int64)
        candidates = anchors[:, None, :] + offsets[None, :, :]
        inside = ((candidates >= 0) & (candidates < shape)).all(axis=2)
        centers = minimum + (candidates.astype(np.float64) + 0.5) * voxel_size
        delta = centers - primitives.means[start:end, None, :]
        mahalanobis = (delta * delta).sum(axis=2) * precision
        valid = inside & np.isfinite(mahalanobis) & (mahalanobis <= support_squared)
        primitive_rows, candidate_columns = np.nonzero(valid)
        if not len(primitive_rows):
            continue
        selected = candidates[primitive_rows, candidate_columns]
        flat = np.ravel_multi_index(selected.T, grid.shape)
        gaussian_weight = np.exp(-0.5 * mahalanobis[primitive_rows, candidate_columns])
        yield start, primitive_rows, flat, gaussian_weight


def accumulate_gaussian_values(
    primitives: PrimitiveSet,
    values: np.ndarray,
    grid: GridSpec,
    *,
    primitive_weight: np.ndarray | None = None,
    sigma: float = DEFAULT_SIGMA_M,
    support_squared: float = DEFAULT_SUPPORT_MAHALANOBIS_SQUARED,
    chunk_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate per-primitive values with the paper Gaussian kernel."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] != len(primitives):
        raise ValueError("values must have shape [N] or [N,C]")
    if primitive_weight is None:
        primitive_weight = np.ones(len(primitives), dtype=np.float32)
    primitive_weight = np.asarray(primitive_weight, dtype=np.float32)
    if primitive_weight.shape != (len(primitives),) or (primitive_weight < 0).any():
        raise ValueError("primitive_weight must be nonnegative with shape [N]")
    numerator = np.zeros((grid.size, values.shape[1]), dtype=np.float64)
    denominator = np.zeros(grid.size, dtype=np.float64)
    for start, rows, flat, gaussian_weight in _iter_support(
        primitives, grid, sigma, support_squared, chunk_size
    ):
        source_rows = start + rows
        weight = gaussian_weight * primitive_weight[source_rows]
        denominator += np.bincount(flat, weights=weight, minlength=grid.size)
        for channel in range(values.shape[1]):
            numerator[:, channel] += np.bincount(
                flat,
                weights=weight * values[source_rows, channel],
                minlength=grid.size,
            )
    return numerator.reshape(grid.shape + (values.shape[1],)), denominator.reshape(grid.shape)


def gaussian_hazard_readout(
    primitives: PrimitiveSet,
    grid: GridSpec,
    *,
    evidence_weight: np.ndarray | None = None,
    sigma: float = DEFAULT_SIGMA_M,
    support_squared: float = DEFAULT_SUPPORT_MAHALANOBIS_SQUARED,
    hazard_threshold: float = DEFAULT_HAZARD_THRESHOLD,
    chunk_size: int = 2048,
) -> GaussianReadout:
    """Project metric primitives to occupancy through Poisson hazard readout."""
    if evidence_weight is None:
        evidence_weight = np.ones(len(primitives), dtype=np.float32)
    evidence_weight = np.asarray(evidence_weight, dtype=np.float32)
    if evidence_weight.shape != (len(primitives),) or (evidence_weight < 0).any():
        raise ValueError("evidence_weight must be nonnegative with shape [N]")
    hazard = np.zeros(grid.size, dtype=np.float64)
    age_mask = np.zeros(grid.size, dtype=np.uint8)
    for start, rows, flat, gaussian_weight in _iter_support(
        primitives, grid, sigma, support_squared, chunk_size
    ):
        source_rows = start + rows
        weight = gaussian_weight * evidence_weight[source_rows]
        hazard += np.bincount(flat, weights=weight, minlength=grid.size)
        ages = primitives.ages[source_rows]
        np.bitwise_or.at(age_mask, flat, (1 << ages).astype(np.uint8))
    observed = hazard > 0
    occupied = hazard >= hazard_threshold
    popcount = np.asarray([bin(value).count("1") for value in range(256)], dtype=np.uint8)
    newest_age = np.full(grid.size, 255, dtype=np.uint8)
    for age in range(8):
        selected = (age_mask & (1 << age) != 0) & (newest_age == 255)
        newest_age[selected] = age
    return GaussianReadout(
        occupied=occupied.reshape(grid.shape),
        observed=observed.reshape(grid.shape),
        hazard=hazard.reshape(grid.shape).astype(np.float32),
        source_frame_count=popcount[age_mask].reshape(grid.shape),
        newest_source_age=newest_age.reshape(grid.shape),
    )
