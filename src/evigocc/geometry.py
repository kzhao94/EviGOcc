from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import GridSpec, validate_rigid_transform


@dataclass(frozen=True)
class PrimitiveSet:
    """Metric evidence primitives before grid commitment."""

    means: np.ndarray
    voxel_indices: np.ndarray
    confidence: np.ndarray
    ages: np.ndarray
    source_ids: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.means)
        if self.means.shape != (count, 3) or self.voxel_indices.shape != (count, 3):
            raise ValueError("primitive means and indices must have shape [N, 3]")
        for name in ("confidence", "ages", "source_ids"):
            if getattr(self, name).shape != (count,):
                raise ValueError(f"{name} must contain one value per primitive")
        if not np.isfinite(self.means).all() or not np.isfinite(self.confidence).all():
            raise ValueError("primitive values must be finite")
        if (self.confidence < 0).any():
            raise ValueError("primitive confidence must be nonnegative")
        if (self.ages < 0).any() or (self.ages > 7).any():
            raise ValueError("primitive ages must lie in [0, 7]")

    def __len__(self) -> int:
        return len(self.means)


def backproject_depth(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    pixel_stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project valid metric depth into camera coordinates."""
    depth = np.asarray(depth, dtype=np.float32)
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    if depth.ndim != 2 or intrinsic.shape != (3, 3):
        raise ValueError("depth must be [H,W] and intrinsic must be [3,3]")
    if pixel_stride < 1:
        raise ValueError("pixel_stride must be positive")
    rows, columns = np.mgrid[0 : depth.shape[0] : pixel_stride, 0 : depth.shape[1] : pixel_stride]
    values = depth[rows, columns]
    valid = np.isfinite(values) & (values > 0)
    rows = rows[valid].astype(np.float64)
    columns = columns[valid].astype(np.float64)
    values = values[valid].astype(np.float64)
    x = (columns - intrinsic[0, 2]) * values / intrinsic[0, 0]
    y = (rows - intrinsic[1, 2]) * values / intrinsic[1, 1]
    points = np.stack((x, y, values), axis=1)
    pixels = np.stack((columns, rows), axis=1)
    return points, pixels


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    transform = validate_rigid_transform(transform)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [N, 3]")
    homogeneous = np.concatenate((points, np.ones((len(points), 1))), axis=1)
    return (transform @ homogeneous.T).T[:, :3]


def lift_metric_depth(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    camera_to_target: np.ndarray,
    grid: GridSpec,
    *,
    confidence: np.ndarray | None = None,
    pixel_stride: int = 1,
    age: int = 0,
    source_id: int = 0,
) -> PrimitiveSet:
    """Lift one depth observation and group valid pixels by native voxel.

    The grouped metric mean is retained as the Gaussian center. Semantic
    evidence can subsequently be attached by matching ``source_id`` and pixel
    provenance in a frontend adapter.
    """
    depth = np.asarray(depth, dtype=np.float32)
    if confidence is None:
        confidence = np.ones_like(depth, dtype=np.float32)
    confidence = np.asarray(confidence, dtype=np.float32)
    if confidence.shape != depth.shape:
        raise ValueError("confidence and depth must share shape")
    points_camera, pixels = backproject_depth(depth, intrinsic, pixel_stride)
    points_target = transform_points(points_camera, camera_to_target)
    indices, in_grid = grid.indices(points_target)
    if not in_grid.any():
        return PrimitiveSet(
            means=np.empty((0, 3), dtype=np.float32),
            voxel_indices=np.empty((0, 3), dtype=np.int16),
            confidence=np.empty((0,), dtype=np.float32),
            ages=np.empty((0,), dtype=np.uint8),
            source_ids=np.empty((0,), dtype=np.int32),
        )
    points_target = points_target[in_grid]
    indices = indices[in_grid]
    selected_pixels = pixels[in_grid].astype(np.int64)
    pixel_confidence = confidence[selected_pixels[:, 1], selected_pixels[:, 0]]
    linear = np.ravel_multi_index(indices.T, grid.shape)
    unique, inverse = np.unique(linear, return_inverse=True)
    counts = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
    means = (
        np.stack(
            [
                np.bincount(inverse, weights=points_target[:, axis], minlength=len(unique))
                for axis in range(3)
            ],
            axis=1,
        )
        / counts[:, None]
    )
    mean_confidence = np.bincount(inverse, weights=pixel_confidence, minlength=len(unique)) / counts
    voxel_indices = np.stack(np.unravel_index(unique, grid.shape), axis=1)
    return PrimitiveSet(
        means=means.astype(np.float32),
        voxel_indices=voxel_indices.astype(np.int16),
        confidence=mean_confidence.astype(np.float32),
        ages=np.full(len(unique), age, dtype=np.uint8),
        source_ids=np.full(len(unique), source_id, dtype=np.int32),
    )


def concatenate_primitives(*sets: PrimitiveSet) -> PrimitiveSet:
    if not sets:
        raise ValueError("at least one primitive set is required")
    return PrimitiveSet(
        means=np.concatenate([item.means for item in sets]),
        voxel_indices=np.concatenate([item.voxel_indices for item in sets]),
        confidence=np.concatenate([item.confidence for item in sets]),
        ages=np.concatenate([item.ages for item in sets]),
        source_ids=np.concatenate([item.source_ids for item in sets]),
    )
