# External Baseline Adapters

External source is not copied into this repository. Clone each upstream at the
pinned commit and apply the monocular/native-grid contract described below.
The dependency-free block planner is available as `evigocc.ttocc_contract` and
`evigocc.freeocc_contract`.

| Method | Upstream | Pinned commit | Redistribution status |
|---|---|---|---|
| TT-Occ | `https://github.com/UQMM-Lab/TT-Occ` | `085200fa93c83abe400bcdf37e8a25d2f80dec64` | no top-level license detected; do not vendor |
| FreeOcc (embodied) | `https://github.com/the-masses/FreeOcc` | `565b329270ec0ad8a7666b1e5496a6fbc2d38bab` | Apache-2.0 upstream; kept external here |
| FreeOcc (panoptic) | `https://github.com/andrewcaunes/FreeOcc` | `7003b7a5e6406cc0ebb2098f5b710da2a016db3d` | no top-level license detected; audit only |

## WildOcc contract

- TT-Occ: one real camera, fixed metric depth, native grid, `K=10`, 16-frame
  chunks, three-frame context, eight queries, no synthetic views, no RAFT.
- FreeOcc: registered RGB--depth poses, 64-frame blocks, 16-frame causal map
  warm-up, eight queries, final map excluded from per-frame prediction.

Both adapters use seven evaluated occupied concepts plus one auxiliary
other/void query. The auxiliary query is excluded from mIoU.

## GOOSE-OCC contract

- TT-Occ: one windshield camera, fixed metric depth, native grid, 16-frame
  chunks, three-frame context, Talk2DINO maps, no RAFT.
- FreeOcc: registered RGB--depth poses, 64-frame blocks, 16-frame causal
  context, released mapper with empty-window handling.

The guarded GOOSE FreeOcc row measures transfer under this calibration
contract. Its near-zero coverage must not be interpreted as performance in the
method's native embodied setting.

Additional adapter patches may contain only first-party glue and unified-format
exporters; they must not include upstream files from repositories without an
explicit redistribution license.
