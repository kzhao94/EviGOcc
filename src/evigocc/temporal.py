from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import GridSpec, validate_rigid_transform

DEFAULT_HISTORY_FRAMES = 3
DEFAULT_AGE_DECAY = 0.75


@dataclass(frozen=True)
class TemporalObservation:
    prediction: np.ndarray
    evidence: np.ndarray
    source_to_target: np.ndarray


@dataclass(frozen=True)
class CTEAResult:
    prediction: np.ndarray
    occupied: np.ndarray
    conflict: np.ndarray
    unknown: np.ndarray
    semantic_reliability: np.ndarray
    source_frame_count: np.ndarray
    newest_source_age: np.ndarray
    class_mass: np.ndarray


def _project_observation(
    observation: TemporalObservation,
    grid: GridSpec,
    class_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prediction = np.asarray(observation.prediction)
    evidence = np.asarray(observation.evidence, dtype=np.float32)
    transform = validate_rigid_transform(observation.source_to_target)
    if prediction.shape != grid.shape or evidence.shape != grid.shape:
        raise ValueError("temporal prediction and evidence must match the grid")
    occupied = np.isin(prediction, class_ids)
    if not occupied.any():
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.uint8),
            np.empty(0, dtype=np.float32),
        )
    indices = np.stack(np.nonzero(occupied), axis=1)
    centers = grid.centers(indices)
    homogeneous = np.concatenate((centers, np.ones((len(centers), 1))), axis=1)
    target_points = (transform @ homogeneous.T).T[:, :3]
    target_indices, valid = grid.indices(target_points)
    source_tuple = tuple(indices[valid].T)
    linear = np.ravel_multi_index(target_indices[valid].T, grid.shape)
    labels = prediction[source_tuple].astype(np.uint8, copy=False)
    weights = np.maximum(evidence[source_tuple], 1e-6).astype(np.float32, copy=False)
    return linear.astype(np.int64), labels, weights


def causal_temporal_accumulation(
    observations: list[TemporalObservation],
    grid: GridSpec,
    *,
    class_ids: np.ndarray | tuple[int, ...] = tuple(range(1, 8)),
    age_decay: float = DEFAULT_AGE_DECAY,
    max_history: int = DEFAULT_HISTORY_FRAMES,
) -> CTEAResult:
    """Transport current plus strictly causal readouts and accumulate evidence.

    ``observations[0]`` is the current readout. Remaining observations must be
    ordered newest-to-oldest and provide their rigid source-to-target transform.
    """
    if not observations or not 0 < age_decay <= 1 or max_history < 0:
        raise ValueError("CTEA requires observations, valid decay, and history limit")
    observations = observations[: max_history + 1]
    class_ids = np.asarray(class_ids, dtype=np.uint8)
    if class_ids.ndim != 1 or len(class_ids) < 2 or len(np.unique(class_ids)) != len(class_ids):
        raise ValueError("class_ids must be a unique one-dimensional set")
    class_mass = np.zeros((len(class_ids), grid.size), dtype=np.float32)
    current_mass = np.zeros_like(class_mass)
    history_mass = np.zeros_like(class_mass)
    source_count = np.zeros(grid.size, dtype=np.uint8)
    newest_age = np.full(grid.size, 255, dtype=np.uint8)
    for age, observation in enumerate(observations):
        linear, labels, weights = _project_observation(observation, grid, class_ids)
        if not len(linear):
            continue
        weights = weights * np.float32(age_decay**age)
        for offset, class_id in enumerate(class_ids):
            selected = labels == class_id
            if selected.any():
                np.add.at(class_mass[offset], linear[selected], weights[selected])
                target = current_mass if age == 0 else history_mass
                np.add.at(target[offset], linear[selected], weights[selected])
        unique_linear = np.unique(linear)
        source_count[unique_linear] = np.minimum(
            source_count[unique_linear].astype(np.uint16) + 1, 255
        ).astype(np.uint8)
        np.minimum.at(newest_age, unique_linear, np.uint8(age))
    mass_sum = class_mass.sum(axis=0)
    occupied = mass_sum > 0
    winner = np.argmax(class_mass, axis=0)
    top = np.take_along_axis(class_mass, winner[None], axis=0)[0]
    prediction = np.zeros(grid.size, dtype=np.uint8)
    prediction[occupied] = class_ids[winner[occupied]]
    current_sum = current_mass.sum(axis=0)
    history_sum = history_mass.sum(axis=0)
    temporal_disagreement = (
        (current_sum > 0)
        & (history_sum > 0)
        & (np.argmax(current_mass, axis=0) != np.argmax(history_mass, axis=0))
    )
    multi_class = (class_mass > 0).sum(axis=0) > 1
    conflict = occupied & multi_class & temporal_disagreement
    reliability = np.divide(
        top,
        mass_sum,
        out=np.zeros_like(top, dtype=np.float32),
        where=mass_sum > 0,
    )
    return CTEAResult(
        prediction=prediction.reshape(grid.shape),
        occupied=occupied.reshape(grid.shape),
        conflict=conflict.reshape(grid.shape),
        unknown=conflict.reshape(grid.shape).copy(),
        semantic_reliability=reliability.reshape(grid.shape),
        source_frame_count=source_count.reshape(grid.shape),
        newest_source_age=newest_age.reshape(grid.shape),
        class_mass=class_mass.reshape((len(class_ids),) + grid.shape),
    )
