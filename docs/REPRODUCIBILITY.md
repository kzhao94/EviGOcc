# Reproducibility Contract

## Prediction boundary

EviGOcc separates prediction from evaluation:

1. Load RGB, calibration, pose, and frozen VFM outputs.
2. Construct metric primitives without occupancy ground truth.
3. Apply Gaussian/native-cell readout and semantic operations.
4. Write predictions and provenance manifests.
5. Load validation occupancy labels only in the evaluator.

Prompts, prediction thresholds, checkpoints, and operating points are fixed
before validation scores are computed. Analysis-only oracle quantities are not
method inputs.

## Frozen paper parameters

`configs/paper_contracts.json` is the machine-readable source for the two grid
contracts and the following values:

| Parameter | WildOcc | GOOSE-OCC |
|---|---:|---:|
| Native voxel edge | 0.20 m | 0.25 m |
| Causal history | up to 3 prior primitives | up to 3 prior readouts |
| Gaussian sigma | 0.20 m | not used in the main native-cell result |
| Gaussian truncation | 3 sigma | not used in the main result |
| Occupancy hazard | log(2) | native occupied cell |
| Semantic temperature | 0.01 | frozen frontend posterior |
| PPSA top-k | 2 | not used |
| PPSA conflict margin | 0.02 | not used |
| CTEA age decay | not used | 0.75 |

## Core API sequence

```python
from evigocc.contracts import WILDOCC_GRID
from evigocc.geometry import lift_metric_depth
from evigocc.gaussian import gaussian_hazard_readout

primitives = lift_metric_depth(
    depth_m,
    intrinsic,
    camera_to_target,
    WILDOCC_GRID,
    confidence=depth_confidence,
    age=0,
    source_id=0,
)
occupancy = gaussian_hazard_readout(primitives, WILDOCC_GRID)
```

Semantic frontends produce one posterior per primitive and source. Accumulate
each source independently with `gaussian_source_posteriors`, then call
`provenance_preserving_arbitration`. Missing source evidence is represented by
`source_valid=False`, not by a zero-probability negative observation.

GOOSE-OCC CTEA takes a current readout followed by strictly causal prior
readouts. Every prior observation supplies a rigid source-to-current transform.
RASC consumes the retained unknown/conflict state and returns a commitment mask
while preserving `prediction` and `occupied` exactly.

## Result interpretation

- WildOcc reports full-grid occupancy IoU and seven-class semantic mIoU.
- GOOSE-OCC reports full-grid, visible, occluded, and range-bin metrics.
- Coarse WildOcc values in the paper are pooled-output diagnostics, not fresh
  Gaussian-field readouts.
- RASC is a reliability/selective-prediction result, not a dense mIoU module.
- The guarded GOOSE FreeOcc row is an adapter-transfer result and does not
  represent FreeOcc in its native embodied setting.

