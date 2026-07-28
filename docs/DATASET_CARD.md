# Dataset and benchmark protocol card

## Name

**EMS Occlusion Protocol Suite, version 1.0**

## Summary

This release defines a deterministic benchmark protocol layered on licensed
multi-view pose datasets. It is not a standalone collection of RGB images or
human-pose annotations. The public contribution is the mask-generation code,
condition table, seed policy, validity-update rule and manifest format.

## Base datasets

| Base dataset | Use | Redistribution in this repository |
|---|---|---|
| Human3.6M | Main controlled and action-wise evaluation | No |
| CMU Panoptic | Independent camera/joint-convention evaluation | No |
| H36M-Occ variants | Separate native/dataset-level diagnostic | No |
| Occlusion-Person | Optional integration only; required assets unavailable | No |

Users must obtain each base dataset from its provider and comply with the
provider's license. The EMS-Splat repository does not redistribute images,
camera calibration, 2D/3D annotations, detector predictions or initial
triangulations.

## Public protocol components

- `data/protocols/ems_occlusion_protocol_v1.csv`: canonical conditions.
- `utils/occlusion_utils.py`: deterministic mask implementation.
- `configs/h36m.yaml` and `configs/panoptic_strict_example.yaml`: configuration
  templates.
- Per-run `occlusion_mask_manifest.jsonl`: generated audit record containing
  scene/camera identity, box geometry, mask shape, masked-pixel count, seed and
  SHA-256 digest.

## Controlled conditions

1. **Rectangle:** one contiguous block at area ratios 0.2, 0.4 and 0.6.
2. **Random block:** the requested area is divided among four blocks at ratios
   0.2, 0.4 and 0.6. Blocks are sampled independently and can overlap, so the
   realized unique-pixel coverage may be lower than the requested ratio.
3. **Body part:** a deterministic padded bounding region covers both arms,
   both legs or the torso according to the H36M 17-joint convention.
4. **Public mask:** an optional manifest-backed arbitrary-mask interface,
   reported separately from procedural observation-space masks.

The controlled benchmark degrades generated joint heatmaps and marks 2D joints
inside a mask invalid. It does not rerun a detector on masked RGB. Results must
therefore be described as **observation-interface degradation**, not native RGB
occlusion performance.

## Seeds and reproducibility

- Canonical mask seed: 2026.
- Repeated severe-condition mask seeds: 2024, 2025 and 2026.
- `training.seed` is a separate per-scene optimization seed.
- Body-part masks are deterministic for fixed observations and padding.

Every baseline/EMS comparison must use the same scene list, observations,
mask manifest and official evaluation conversion. Mask seeds must never be
reported as network-training seeds. For random-block runs, use the manifest's
`masked_pixels` field when auditing realized coverage.

## Generated-data release policy

The procedural code and protocol table are sufficient to regenerate masks
after a user has obtained the base data. We intentionally do not upload
generated RGB frames, heatmaps, joint coordinates, pose banks or detector
outputs because those artifacts derive from datasets or detectors that we do
not own. Sanitized aggregate metrics and skeleton-only visualizations may be
redistributed because they do not reconstruct the source recordings.

## Intended use and limitations

The suite evaluates robustness of calibrated multi-view 3D recovery to missing
or corrupted 2D evidence. It is not an appearance-completion benchmark, a
monocular benchmark or evidence of robustness to arbitrary real occluders.
Semantic body-part masks rely on observed 2D joints and are not area-matched to
the rectangle protocol.

## Suggested data-availability wording

> The EMS Occlusion Protocol Suite, including deterministic mask-generation
> code, condition metadata and audit-manifest definitions, is released with
> the project source. Human3.6M and CMU Panoptic recordings and derived pose
> observations are not redistributed and must be obtained under their
> respective provider terms.
