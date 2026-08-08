import json
from pathlib import Path

import numpy as np

from evigocc.baseline_contracts import ObservationRef, freeocc_contract, ttocc_contract
from evigocc.contracts import GOOSE_OCC_GRID, WILDOCC_GRID, GridSpec
from evigocc.demo import run_demo
from evigocc.fusion import provenance_preserving_arbitration, retain_top_k
from evigocc.gaussian import gaussian_hazard_readout
from evigocc.geometry import PrimitiveSet
from evigocc.metrics import OccupancyMetricAccumulator
from evigocc.reliability import reliability_aware_commitment
from evigocc.temporal import TemporalObservation, causal_temporal_accumulation


def tiny_grid() -> GridSpec:
    return GridSpec((0.0, 0.0, 0.0), (2.0, 2.0, 2.0), (1.0, 1.0, 1.0), (2, 2, 2))


def test_grid_indices_and_centers_round_trip():
    grid = tiny_grid()
    points = np.asarray([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5], [-1.0, 0.0, 0.0]])
    indices, valid = grid.indices(points)
    assert valid.tolist() == [True, True, False]
    np.testing.assert_allclose(grid.centers(indices[valid]), points[valid])


def test_gaussian_hazard_uses_log_two_occupancy_threshold():
    primitives = PrimitiveSet(
        means=np.asarray([[0.5, 0.5, 0.5]], dtype=np.float32),
        voxel_indices=np.asarray([[0, 0, 0]], dtype=np.int16),
        confidence=np.ones(1, dtype=np.float32),
        ages=np.zeros(1, dtype=np.uint8),
        source_ids=np.zeros(1, dtype=np.int32),
    )
    result = gaussian_hazard_readout(primitives, tiny_grid(), sigma=1.0)
    assert result.hazard[0, 0, 0] == 1.0
    assert result.occupied[0, 0, 0]
    assert result.source_frame_count[0, 0, 0] == 1


def test_ppsa_ignores_absent_sources_and_retains_conflict():
    posterior = np.asarray(
        [
            [[[0.8, 0.2], [0.6, 0.4]]],
            [[[0.1, 0.9], [0.0, 0.0]]],
        ],
        dtype=np.float32,
    )
    valid = np.asarray([[[True, True]], [[True, False]]])
    result = provenance_preserving_arbitration(posterior, valid, class_ids=np.asarray((1, 2)))
    np.testing.assert_allclose(result.posterior[0, 1], (0.6, 0.4))
    assert result.conflict[0, 0]
    assert result.available_sources.tolist() == [[2, 1]]


def test_top_k_is_renormalized():
    posterior = np.asarray([[0.5, 0.3, 0.2]], dtype=np.float32)
    retained = retain_top_k(posterior, 2)
    np.testing.assert_allclose(retained.sum(axis=-1), 1.0)
    assert retained[0, 2] == 0


def test_ctea_transports_only_supplied_causal_observations():
    grid = tiny_grid()
    current_prediction = np.zeros(grid.shape, dtype=np.uint8)
    current_prediction[0, 0, 0] = 1
    history_prediction = np.zeros(grid.shape, dtype=np.uint8)
    history_prediction[0, 0, 0] = 2
    evidence = np.ones(grid.shape, dtype=np.float32)
    identity = np.eye(4)
    result = causal_temporal_accumulation(
        [
            TemporalObservation(current_prediction, evidence, identity),
            TemporalObservation(history_prediction, evidence, identity),
        ],
        grid,
        class_ids=(1, 2),
    )
    assert result.prediction[0, 0, 0] == 1
    assert result.conflict[0, 0, 0]
    assert result.source_frame_count[0, 0, 0] == 2


def test_rasc_preserves_prediction_and_occupied_geometry():
    prediction = np.asarray([1, 4, 5], dtype=np.uint8)
    occupied = np.ones(3, dtype=bool)
    result = reliability_aware_commitment(
        prediction,
        occupied,
        semantic_unknown=np.asarray([False, False, True]),
        semantic_conflict=np.asarray([False, False, True]),
        backend_agreement=np.asarray([True, False, True]),
        causal_support=np.asarray([True, True, True]),
        supported_label=prediction,
    )
    np.testing.assert_array_equal(result.prediction, prediction)
    np.testing.assert_array_equal(result.occupied, occupied)
    assert result.committed.tolist() == [True, False, False]


def test_pooled_metrics():
    metric = OccupancyMetricAccumulator(3, (1,), empty_label=2)
    metric.update(np.asarray([1, 2, 1]), np.asarray([1, 2, 2]))
    result = metric.compute()
    assert result["IoU"] == 0.5
    assert result["mIoU"] == 0.5


def test_end_to_end_demo_runs():
    result = run_demo()
    assert result["primitives"] > 0
    assert result["gaussian_occupied_voxels"] > 0
    assert result["ctea_occupied_voxels"] >= result["ppsa_known_voxels"]


def test_machine_readable_grid_contracts_match_source():
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "configs/paper_contracts.json").read_text())
    for key, grid in (("wildocc", WILDOCC_GRID), ("goose_occ", GOOSE_OCC_GRID)):
        record = payload[key]["grid"]
        assert tuple(record["minimum"]) == grid.minimum
        assert tuple(record["maximum"]) == grid.maximum
        assert tuple(record["voxel_size"]) == grid.voxel_size
        assert tuple(record["shape"]) == grid.shape


def test_paper_factorization_recomputes_from_main_rows():
    root = Path(__file__).resolve().parents[1]
    wildocc = json.loads((root / "results/paper_results.json").read_text())["wildocc"]
    methods = wildocc["methods"]
    depth = wildocc["factorization_pp"]["depth_construction"]
    ppsa = wildocc["factorization_pp"]["PPSA_at_fixed_geometry"]
    assert (
        round(methods["Fused-depth field, Talk2DINO only"]["IoU"] - methods["EviGOcc-S"]["IoU"], 4)
        == depth["IoU"]
    )
    assert (
        round(
            methods["Fused-depth field, Talk2DINO only"]["mIoU"] - methods["EviGOcc-S"]["mIoU"], 4
        )
        == depth["mIoU"]
    )
    assert (
        round(
            methods["EviGOcc-X"]["mIoU"] - methods["Fused-depth field, Talk2DINO only"]["mIoU"], 4
        )
        == ppsa["mIoU"]
    )


def test_external_baseline_blocks_are_strictly_causal():
    observations = [ObservationRef(index, "seq", index) for index in range(70)]
    ttocc = ttocc_contract(observations)
    freeocc = freeocc_contract(observations)
    assert len(ttocc[1].context) == 3
    assert max(ttocc[1].context) < min(ttocc[1].frames)
    assert len(freeocc[1].context) == 16
    assert max(freeocc[1].context) < min(freeocc[1].frames)
