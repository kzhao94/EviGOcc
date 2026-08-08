from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .contracts import GridSpec
from .fusion import gaussian_source_posteriors, provenance_preserving_arbitration
from .gaussian import gaussian_hazard_readout
from .geometry import lift_metric_depth
from .reliability import reliability_aware_commitment
from .temporal import TemporalObservation, causal_temporal_accumulation


def run_demo() -> dict[str, float | int]:
    grid = GridSpec(
        minimum=(-2.0, -2.0, 0.0),
        maximum=(2.0, 2.0, 4.0),
        voxel_size=(0.5, 0.5, 0.5),
        shape=(8, 8, 8),
    )
    depth = np.full((12, 16), 2.0, dtype=np.float32)
    depth[:2] = 0
    intrinsic = np.asarray([[12.0, 0.0, 7.5], [0.0, 12.0, 5.5], [0.0, 0.0, 1.0]])
    primitives = lift_metric_depth(
        depth,
        intrinsic,
        np.eye(4),
        grid,
        pixel_stride=2,
    )
    occupancy = gaussian_hazard_readout(primitives, grid, sigma=0.5)

    class_count = 3
    base_scores = np.zeros((len(primitives), class_count), dtype=np.float32)
    base_scores[:, 0] = 0.8
    base_scores[:, 1] = np.linspace(0.1, 0.9, len(primitives), dtype=np.float32)
    base_scores[:, 2] = 0.2
    second_scores = base_scores[:, ::-1].copy()
    primitive_posteriors = np.stack((base_scores, second_scores))
    primitive_posteriors /= primitive_posteriors.sum(axis=-1, keepdims=True)
    primitive_valid = np.ones((2, len(primitives)), dtype=bool)
    source_posteriors, source_valid = gaussian_source_posteriors(
        primitives, primitive_posteriors, primitive_valid, grid
    )
    ppsa = provenance_preserving_arbitration(
        source_posteriors,
        source_valid,
        class_ids=np.asarray((1, 2, 3), dtype=np.uint8),
    )

    current_prediction = np.where(occupancy.occupied, ppsa.labels, 0).astype(np.uint8)
    current = TemporalObservation(
        prediction=current_prediction,
        evidence=occupancy.hazard,
        source_to_target=np.eye(4),
    )
    history_transform = np.eye(4)
    history_transform[0, 3] = 0.5
    history = TemporalObservation(
        prediction=current_prediction,
        evidence=occupancy.hazard,
        source_to_target=history_transform,
    )
    ctea = causal_temporal_accumulation(
        [current, history], grid, class_ids=(1, 2, 3), age_decay=0.75
    )
    rasc = reliability_aware_commitment(
        prediction=ctea.prediction,
        occupied=ctea.occupied,
        semantic_unknown=ctea.unknown,
        semantic_conflict=ctea.conflict,
        backend_agreement=~ctea.conflict,
        causal_support=ctea.source_frame_count >= 2,
        supported_label=ctea.prediction,
        rare_class_ids=(2, 3),
    )
    if not np.array_equal(rasc.prediction, ctea.prediction):
        raise RuntimeError("RASC changed the dense prediction")
    return {
        "primitives": len(primitives),
        "gaussian_occupied_voxels": int(occupancy.occupied.sum()),
        "ppsa_known_voxels": int(ppsa.known.sum()),
        "ppsa_conflict_voxels": int(ppsa.conflict.sum()),
        "ctea_occupied_voxels": int(ctea.occupied.sum()),
        "rasc_committed_voxels": int(rasc.committed.sum()),
        "rasc_abstained_voxels": int(rasc.abstained.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the dependency-light EviGOcc smoke demo")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_demo()
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
