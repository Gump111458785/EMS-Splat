# Public protocol metadata

`ems_occlusion_protocol_v1.csv` is the redistributable definition of the
controlled EMS-Splat occlusion benchmark. It contains no RGB images, 2D/3D
joint coordinates, detector output, camera parameters, or subject metadata.

The masks are generated deterministically at runtime by
`utils/occlusion_utils.py`. Every non-clean run writes
`occlusion_mask_manifest.jsonl`, including mask geometry, pixel count and a
SHA-256 digest. This per-run manifest is the appropriate artifact for auditing
an experiment; it is not bundled because it contains base-dataset scene names.

See `docs/DATASET_CARD.md` for scope, licensing and reporting requirements.
