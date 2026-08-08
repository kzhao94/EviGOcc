from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GridSpec:
    """Metric axis-aligned voxel contract."""

    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]
    voxel_size: tuple[float, float, float]
    shape: tuple[int, int, int]

    def __post_init__(self) -> None:
        minimum = np.asarray(self.minimum, dtype=np.float64)
        maximum = np.asarray(self.maximum, dtype=np.float64)
        voxel_size = np.asarray(self.voxel_size, dtype=np.float64)
        shape = np.asarray(self.shape, dtype=np.int64)
        if not (minimum.shape == maximum.shape == voxel_size.shape == shape.shape == (3,)):
            raise ValueError("Grid vectors must have three elements")
        if (maximum <= minimum).any() or (voxel_size <= 0).any() or (shape <= 0).any():
            raise ValueError("Grid bounds, voxel size, and shape must be positive")
        expected = (maximum - minimum) / voxel_size
        if not np.allclose(expected, shape, atol=1e-6):
            raise ValueError("Grid bounds and voxel size do not match shape")

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    def indices(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape [N, 3]")
        indices = np.floor(
            (points - np.asarray(self.minimum)) / np.asarray(self.voxel_size)
        ).astype(np.int64)
        valid = np.isfinite(points).all(axis=1)
        valid &= (indices >= 0).all(axis=1)
        valid &= (indices < np.asarray(self.shape)).all(axis=1)
        return indices, valid

    def centers(self, indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 2 or indices.shape[1] != 3:
            raise ValueError("indices must have shape [N, 3]")
        if ((indices < 0) | (indices >= np.asarray(self.shape))).any():
            raise ValueError("indices lie outside the grid")
        return np.asarray(self.minimum) + (indices.astype(np.float64) + 0.5) * np.asarray(
            self.voxel_size
        )


WILDOCC_GRID = GridSpec(
    minimum=(-20.0, -10.0, -2.0),
    maximum=(0.0, 10.0, 6.0),
    voxel_size=(0.2, 0.2, 0.2),
    shape=(100, 100, 40),
)
WILDOCC_CLASS_NAMES = (
    "other",
    "grass",
    "tree",
    "bush",
    "puddle",
    "mud",
    "barrier",
    "rubble",
    "empty",
)
WILDOCC_EVAL_CLASS_IDS = tuple(range(1, 8))
WILDOCC_EMPTY_LABEL = 8

GOOSE_OCC_GRID = GridSpec(
    minimum=(0.0, -16.0, -4.0),
    maximum=(50.0, 16.0, 4.0),
    voxel_size=(0.25, 0.25, 0.25),
    shape=(200, 128, 32),
)
GOOSE_OCC_CLASS_NAMES = (
    "free",
    "artificial_structures",
    "artificial_ground",
    "natural_ground",
    "obstacle",
    "vehicle",
    "vegetation",
    "human",
)
GOOSE_OCCUPIED_CLASS_IDS = tuple(range(1, 8))


def validate_rigid_transform(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("transform must be a finite 4x4 matrix")
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6):
        raise ValueError("transform has an invalid homogeneous row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3):
        raise ValueError("transform rotation is not orthonormal")
    return transform
