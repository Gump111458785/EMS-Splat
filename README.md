# EMS-Splat

Official research code release for **EMS-Splat**, an occlusion-robust,
test-time multi-view 3D human pose optimizer built on joint-semantic 3D
Gaussians.

EMS-Splat starts from calibrated multi-view 2D observations and an initial 3D
skeleton. It optimizes one semantic Gaussian per joint using differentiable
heatmap rendering, epipolar geometry, and weak feasible-skeleton constraints.
The optimized Gaussian centers are the final 3D joints. The current
implementation does not use an RGB reconstruction loss.

This repository is derived from
[SkelSplat](https://github.com/laurabragagnolo/SkelSplat) and the
[3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
codebase.

![Framework overview](assets/framework.png)

## Scope

- Calibrated multi-view H36M and CMU Panoptic loaders.
- CPN, MeTRAbs, or ResNet-152 observation branches after preprocessing.
- Rectangle, random-block, semantic body-part, and manifest-backed occlusion.
- Epipolar, length-consistency, soft-symmetry, anatomical-ratio, and
  pose-manifold constraints.
- Official aggregate, per-scene, per-joint, body-part, bone-length, and
  bilateral-symmetry analysis utilities.

This is a per-scene optimizer, not a trained end-to-end RGB pose network.
Detector predictions, datasets, and pose-bank artifacts are intentionally not
distributed.

## Installation

The tested environment uses Python 3.10, PyTorch 2.5.1, and CUDA 11.8.

```bash
git clone https://github.com/Gump111458785/EMS-Splat.git
cd EMS-Splat

conda env create -f environment.yml
conda activate ems-splat
bash scripts/install_extensions.sh
```

The repository vendors the source of the custom rasterizers required by the
three supported dataset conventions. Build artifacts are not committed.

## Data preparation

Follow [docs/DATA_PREPARATION.md](docs/DATA_PREPARATION.md) and
[dataset_tools/README.md](dataset_tools/README.md). The expected top-level
layout is:

```text
data/
  h36m/
    2d_cpn/
    3d_gt/
    triang_cpn/
  panoptic/
    2d_metrabs/
    3d_gt/
    triang_metrabs/
  pose_banks/
    h36m_train_only.npz
```

Human3.6M and CMU Panoptic data are governed by their original licenses and
must be obtained from their providers.

## Leakage-free H36M pose bank

The final H36M configuration requires an explicit pose bank built only from
training subjects S1/S5/S6/S7/S8. S9 and S11 are rejected by an assertion.

```bash
python tools/build_h36m_train_only_pose_bank.py \
  --input-pkl /path/to/h36m_train.pkl \
  --output data/pose_banks/h36m_train_only.npz \
  --manifest data/pose_banks/h36m_train_only_manifest.csv \
  --stride 8 \
  --max-samples 4096
```

The generated NPZ and source annotations must not be committed.

## Run EMS-Splat

H36M clean:

```bash
python train.py --config-name h36m
python eval.py --config-name h36m
```

Rectangle occlusion at ratio 0.6:

```bash
python train.py --config-name h36m \
  test.occlusion.enable=true \
  test.occlusion.type=rectangle \
  test.occlusion.ratio=0.6 \
  test.occlusion.seed=2026
```

Random-block occlusion:

```bash
python train.py --config-name h36m \
  test.occlusion.enable=true \
  test.occlusion.type=random_block \
  test.occlusion.ratio=0.6 \
  test.occlusion.num_blocks=4 \
  test.occlusion.seed=2026
```

Body-part occlusion:

```bash
python train.py --config-name h36m \
  test.occlusion.enable=true \
  test.occlusion.type=body_part \
  test.occlusion.body_part=both_legs
```

See [docs/OCCLUSION_PROTOCOLS.md](docs/OCCLUSION_PROTOCOLS.md) for protocol
boundaries.

## Same-code SkelSplat baseline

The controlled baseline uses the same loader, initialization, optimizer, and
evaluation code while disabling EMS-Splat auxiliary constraints:

```bash
python train.py --config-name h36m \
  training.lambda_epipolar=0 \
  training.lambda_consistency=0 \
  training.lambda_symmetry_target=0 \
  training.lambda_ratio=0 \
  training.use_pose_manifold_prior=false \
  training.lambda_triangulation=0 \
  training.epipolar_fusion_strength=0 \
  training.use_uncertainty_weighting=false
```

## Evaluation and analysis

```bash
python eval.py --config-name h36m

python tools/analyze_corrected_structure_from_npz.py --help
python tools/analyze_per_scene_error_distribution.py --help
python tools/evaluate_ply_exact_manifest.py --help
```

All paper metrics should come from saved official-evaluation artifacts, not
the running means printed in optimization logs.

## Reproducibility notes

- H36M final evaluation uses the 17-joint project convention.
- Panoptic uses a separate 19-joint mapping; H36M structural pairs must not be
  reused for Panoptic.
- `occlusion mask seed` and per-scene `optimization seed` are separate.
- Shared-observation degradation is not equivalent to native RGB detector
  evaluation.
- The final H36M model disables uncertainty weighting and triangulation
  pseudo-targets.

The exact adopted settings are documented in
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## License

The inherited Gaussian Splatting research license in [LICENSE.md](LICENSE.md)
applies. Redistribution and use are restricted to research/non-commercial
purposes. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for bundled
components.

## Citation

The EMS-Splat citation will be added after publication. When using this code,
please also cite SkelSplat and 3D Gaussian Splatting; see
[CITATION.md](CITATION.md).
