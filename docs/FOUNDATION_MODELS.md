# Foundation Model Inputs

EviGOcc is decoupled from a particular VFM implementation. The paper freezes
one primary depth frontend and one primary semantic frontend for the minimal
configuration; additional sources appear only in controlled WildOcc studies.

## Minimal EviGOcc-S

| Role | Upstream | Output consumed by EviGOcc |
|---|---|---|
| Metric depth | Metric3Dv2 | metric depth map and confidence |
| Dense semantics | DINOv2 + Talk2DINO projection | 768-D dense feature map and text prototypes |

Reference upstream projects:

- Metric3D: `https://github.com/YvanYin/Metric3D`
- DINOv2: `https://github.com/facebookresearch/dinov2`
- Talk2DINO: `https://github.com/lorebianchi98/Talk2DINO`

The frozen Talk2DINO source audit used commit
`1462d0ebbf3c8d2dab623a03d41548c8e6098f34`. The projection is applied as
linear--tanh--linear followed by L2 normalization.

## Controlled WildOcc sources

- UniDepth: `https://github.com/lpiccinelli-eth/UniDepth`
- MoGe/MoGe-2: `https://github.com/microsoft/MoGe`
- FeatUp/MaskCLIP: semantic frontend control
- SAM 3: prompt-response control, not the dense semantic classifier

The fused-depth WildOcc control aligns inverse depth by medians and takes the
median across Metric3D, UniDepth, and MoGe-2. Model count is not claimed as the
method contribution; the paper factorizes this construction change from PPSA.

## Weight policy

Weights must be obtained from the original provider. Do not commit downloaded
checkpoints to this repository. Record upstream version, file checksum, input
resize, normalization, and device in every generated feature-bank manifest.

