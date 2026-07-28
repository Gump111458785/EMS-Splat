#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python -m pip install --no-build-isolation "$ROOT/submodules/fused-ssim"
python -m pip install --no-build-isolation "$ROOT/submodules/simple-knn"
python -m pip install --no-build-isolation "$ROOT/submodules/diff-gaussian-rasterization-h36m"
python -m pip install --no-build-isolation "$ROOT/submodules/diff-gaussian-rasterization-panoptic"
python -m pip install --no-build-isolation "$ROOT/submodules/diff-gaussian-rasterization-op"

python - <<'PY'
import diff_gaussian_rasterization_h36m
import diff_gaussian_rasterization_op
import diff_gaussian_rasterization_panoptic

print("EMS-Splat CUDA extensions imported successfully")
PY
