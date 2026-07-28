# Reproducibility

## Final H36M configuration

| Component | Setting |
|---|---:|
| 2D observations | CPN |
| Initial guess | triangulated CPN |
| Views | 4 |
| Rendered-view passes | 500 |
| Epipolar loss | `5e-4` |
| 3D length consistency | `1e-5` |
| Soft bilateral symmetry | `5e-4`, margin `0.05` |
| Anatomical ratio | `1e-3`, margin `0.05` |
| Pose manifold | `7e-5`, top-k 4 |
| Pose bank | train subjects only, 4,096 poses |
| Uncertainty weighting | disabled |
| Triangulation pseudo-target | disabled |
| Epipolar heatmap fusion | disabled |
| EMA/temporal prior | disabled |

The pose bank must be explicit. The release configuration refuses a bank
containing test subjects.

## Same-code baseline

Disable epipolar, consistency, symmetry, ratio, pose-manifold,
triangulation-pseudo-target, epipolar-fusion, and uncertainty branches. Keep
the detector, initial guess, loader, iterations, optimizer, and evaluation
unchanged.

## Evaluation

Report MPJPE, root-relative MPJPE, PA-MPJPE, and PCK@150 from saved official
evaluation artifacts. Preserve the evaluated scene manifest, config snapshot,
seed type, and output path.
