# Data preparation

EMS-Splat consumes synchronized calibrated cameras, per-view 2D joints, an
initial 3D skeleton, and 3D ground truth for evaluation.

## On-disk contract

For Human3.6M, each observation branch follows:

```text
data/h36m/
  2d_cpn/S9/<action>/<camera>/poses.npz
  3d_gt/S9/<action>/poses.npz
  triang_cpn/S9/<action>/poses.npz
  cameras/
```

Equivalent branch names can be used for MeTRAbs and ResNet-152. The configured
`frame_step` is applied consistently to GT, 2D observations, and the initial
guess. Length mismatches raise an error.

Panoptic uses its own 19-joint convention and camera names. Configure
`camera_names` and `filtered_nviews` when using filtered detections.

## Pose bank

The final H36M model requires:

```text
data/pose_banks/h36m_train_only.npz
```

Generate it with `tools/build_h36m_train_only_pose_bank.py`. Only S1/S5/S6/S7/S8
are accepted. S9/S11 are prohibited.

## Data licensing

No dataset, detector prediction, initial guess, or pose bank is included.
Human3.6M and CMU Panoptic must be obtained and used under their provider
terms.
