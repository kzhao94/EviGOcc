# Dataset Setup

No dataset is included in this repository.

## WildOcc

Obtain WildOcc and its underlying RELLIS-3D camera/calibration data from the
official providers. The paper uses 1,243 effective validation frames from two
sequences under the native 0.20 m grid in `configs/paper_contracts.json`.

Required prediction-time inputs per effective frame are:

- forward RGB image;
- camera intrinsic matrix;
- camera-to-ego and ego-to-global transforms;
- metric depth and confidence for the current frame and up to three strictly
  causal prior primitive sets;
- frozen semantic frontend outputs.

Occupancy labels are evaluation-only. Test and ORAD paths must not be mounted in
the prediction process.

## GOOSE-OCC

GOOSE-OCC v1.4.0 is distributed separately from the source GOOSE images. The
annotation archive is published by Science Data Bank at:

`https://doi.org/10.57760/sciencedb.43855`

The implementation and data contract are maintained at:

`https://github.com/kzhao94/GOOSE-OCC`

The paper uses 490 validation frames from three sequences with one windshield
camera and the native 0.25 m grid. Main-method prediction uses RGB, calibration,
pose, Metric3D depth, and Talk2DINO evidence. Official 2D semantic labels,
occupancy labels, and observation masks are not prediction inputs.

Suggested local layout:

```text
data/
  WildOcc/
  GOOSE-OCC/
feature_banks/
  wildocc_dinov2/
  goose_dinov2/
outputs/
```

All three directories are ignored by Git.
