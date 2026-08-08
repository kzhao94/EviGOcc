# EviGOcc

**Training-Free Evidential Gaussian Occupancy for Open-Vocabulary Off-Road Perception**

EviGOcc constructs monocular semantic occupancy at prediction time from frozen
metric-depth and open-vocabulary image priors. It retains metric support,
semantic alternatives, provenance, time, conflict, and unknown state before
committing evidence to a dataset-native grid.

This repository is the source-only reference release for the accompanying
Robotica manuscript. It contains no training on WildOcc or GOOSE-OCC occupancy
labels and bundles no dataset, checkpoint, pretrained weight, prediction cache,
or third-party baseline source.

## Method components

- **Metric Gaussian evidence:** monocular metric depth is lifted into metric
  primitives and read through a Poisson Gaussian-hazard occupancy operator.
- **PPSA:** provenance-preserving semantic arbitration retains source-conditional
  alternatives and averages only sources that provide valid evidence.
- **CTEA:** causal temporal evidence accumulation transports sealed prior
  readouts and discounts them by age.
- **RASC:** reliability-aware semantic commitment ranks conflict and unsupported
  semantics without changing occupied geometry or the base dense prediction.

The public API mirrors these four paper interfaces:

```python
from evigocc import (
    gaussian_hazard_readout,
    provenance_preserving_arbitration,
    causal_temporal_accumulation,
    reliability_aware_commitment,
)
```

## Installation

The dependency-light reference core requires Python 3.8+ and NumPy:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
evigocc-demo
```

For foundation-model feature extraction on CUDA 11.8:

```bash
conda env create -f environment.yml
conda activate evigocc
evigocc-demo
```

The core Gaussian, fusion, temporal, reliability, and metric tests run on CPU.
Foundation-model checkpoints are separate downloads governed by their upstream
licenses.

## Repository layout

```text
src/evigocc/            metric lifting, Gaussian readout, PPSA, CTEA, RASC
configs/                frozen paper contracts and parameters
examples/               dependency-light executable example
tests/                  core algorithm and invariance tests
results/                paper-facing validation values and claim boundaries
adapters/               external baseline provenance and adaptation notes
docs/                   data, VFM, and reproduction documentation
```

## Reproduction levels

1. **Core verification:** install the package, run `pytest`, and run
   `evigocc-demo`. This requires no external assets.
2. **Prepared-evidence inference:** use the APIs documented in
   [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) with metric depth and
   semantic posterior arrays produced by the listed frozen frontends.
3. **Paper validation:** independently obtain WildOcc or GOOSE-OCC and the
   upstream VFM weights, then follow [`docs/DATASETS.md`](docs/DATASETS.md) and
   [`docs/FOUNDATION_MODELS.md`](docs/FOUNDATION_MODELS.md). Validation labels
   are consumed only by the metric accumulator after predictions are written.

The exact paper-facing values are recorded in
[`results/paper_results.json`](results/paper_results.json). WildOcc test,
GOOSE-OCC test, and ORAD are not part of this release.

## External baselines

TT-Occ and FreeOcc are not vendored. Their upstream repositories, pinned
commits, license status, and adaptation contracts are documented in
[`adapters/README.md`](adapters/README.md). This avoids redistributing source
whose upstream repository does not declare a reusable license.

## Data and weights

- WildOcc must be downloaded from its provider. GOOSE-OCC v1.4.0 is archived
  at <https://doi.org/10.57760/sciencedb.43855>; its implementation and data
  contract are maintained at <https://github.com/kzhao94/GOOSE-OCC>.
- Metric3D, DINOv2, Talk2DINO, UniDepth, MoGe-2, and SAM 3 weights are not
  redistributed.
- Generated feature banks, predictions, and evaluation caches belong under
  ignored local directories such as `feature_banks/` and `outputs/`.

## License

Original EviGOcc source code is released under Apache-2.0. Datasets, pretrained
models, and external baselines retain their own terms. See `NOTICE` before
redistributing a derived package.

## Citation

The manuscript is under review. Machine-readable metadata are provided in
[`CITATION.cff`](CITATION.cff); the final BibTeX record will be added after
publication.
