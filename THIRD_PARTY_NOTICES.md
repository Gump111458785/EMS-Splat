# Third-party notices

EMS-Splat is derived from SkelSplat and the 3D Gaussian Splatting codebase.
The repository therefore retains the Gaussian Splatting research-only,
non-commercial license in `LICENSE.md`.

Bundled source components:

| Component | Location | License |
|---|---|---|
| Dataset-specific Gaussian rasterizers | `submodules/diff-gaussian-rasterization-*` | Gaussian Splatting research license, included in each directory |
| Simple KNN | `submodules/simple-knn` | Gaussian Splatting research license |
| Fused SSIM | `submodules/fused-ssim` | See bundled `LICENSE` |
| SSIM utility code | `utils/loss_utils.py` | Includes code derived from `pytorch-ssim` under MIT terms, as noted in `LICENSE.md` |

Datasets and detector predictions are not redistributed. Users must comply
with the original Human3.6M, CMU Panoptic, detector, and benchmark licenses.
