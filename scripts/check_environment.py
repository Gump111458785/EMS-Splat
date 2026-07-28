#!/usr/bin/env python3
"""Validate the Python, CUDA, and custom-extension environment."""

import argparse
import importlib
import platform
import shutil
import sys


PACKAGES = (
    "numpy",
    "torch",
    "torchvision",
    "hydra",
    "omegaconf",
    "PIL",
    "scipy",
    "sklearn",
)

EXTENSIONS = (
    "diff_gaussian_rasterization_h36m",
    "diff_gaussian_rasterization_panoptic",
    "diff_gaussian_rasterization_op",
)


def version_of(module):
    return getattr(module, "__version__", "unknown")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-no-cuda",
        action="store_true",
        help="Permit a CPU-only host when checking parsing/analysis dependencies.",
    )
    parser.add_argument(
        "--skip-extensions",
        action="store_true",
        help="Skip custom CUDA-extension imports during a CPU-only check.",
    )
    args = parser.parse_args()

    print(f"Python: {sys.version.split()[0]} ({platform.platform()})")
    print(f"nvcc: {shutil.which('nvcc') or 'not found on PATH'}")

    failed = []
    loaded = {}
    for name in PACKAGES:
        try:
            loaded[name] = importlib.import_module(name)
            print(f"{name}: {version_of(loaded[name])}")
        except Exception as exc:
            failed.append(f"{name}: {exc}")

    torch = loaded.get("torch")
    if torch is not None:
        print(f"PyTorch CUDA runtime: {torch.version.cuda}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        elif not args.allow_no_cuda:
            failed.append("CUDA is unavailable; EMS-Splat optimization requires a CUDA GPU")

    if args.skip_extensions:
        print("Custom CUDA extensions: skipped")
    else:
        for name in EXTENSIONS:
            try:
                importlib.import_module(name)
                print(f"{name}: import OK")
            except Exception as exc:
                failed.append(
                    f"{name}: {exc} (run bash scripts/install_extensions.sh)"
                )

    if failed:
        print("\nEnvironment check failed:", file=sys.stderr)
        for item in failed:
            print(f"- {item}", file=sys.stderr)
        raise SystemExit(1)

    print("\nEMS-Splat environment check passed.")


if __name__ == "__main__":
    main()
