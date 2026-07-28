#!/usr/bin/env python3
"""Evaluate a PLY prediction directory on an audited exact-scene manifest."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import open3d as o3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--occlusion-type", default="clean")
    parser.add_argument("--ratio", type=float, default=0.0)
    parser.add_argument("--body-part", default="none")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--official-eval-script",
        required=True,
        help="Protocol-matched evaluator that accepts --input and --out-dir.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> dict[tuple[int, int, int, int], str]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    mapping = {}
    for row in rows:
        image_id = row.get("raw_image_id", row.get("image_id"))
        key = (
            int(row["subject"]),
            int(row["action"]),
            int(row["subaction"]),
            int(image_id),
        )
        mapping[key] = f"{row['scene_name']}.ply"
    if not mapping:
        raise ValueError(f"No scenes found in manifest: {path}")
    return mapping


def main() -> None:
    args = parse_args()
    prediction_root = Path(args.prediction_root).resolve()
    output = Path(args.out_dir).resolve()
    inference = output / "inference"
    inference.mkdir(parents=True, exist_ok=True)

    cache_file = np.load(args.cache)
    cache = {key: cache_file[key] for key in cache_file.files}
    manifest = load_manifest(Path(args.manifest).resolve())

    predictions = []
    for subject, action, subaction, image_id in zip(
        cache["subject"], cache["action"], cache["subaction"], cache["image_id"]
    ):
        key = (int(subject), int(action), int(subaction), int(image_id))
        if key not in manifest:
            raise KeyError(f"Cache identity is absent from manifest: {key}")
        path = prediction_root / manifest[key]
        if not path.is_file():
            raise FileNotFoundError(path)
        points = np.asarray(o3d.io.read_point_cloud(str(path)).points)
        if points.shape != (17, 3) or not np.isfinite(points).all():
            raise ValueError(f"Invalid prediction {path}: shape={points.shape}")
        predictions.append(points)

    prediction = np.asarray(predictions, dtype=np.float32)
    if prediction.shape != cache["gt_3d"].shape:
        raise ValueError(
            f"Prediction/GT mismatch: {prediction.shape} vs {cache['gt_3d'].shape}"
        )

    npz_path = inference / "official_eval_inputs.npz"
    np.savez(
        npz_path,
        pred_3d=prediction,
        gt_3d=cache["gt_3d"].astype(np.float32),
        subject=cache["subject"],
        action=cache["action"],
        subaction=cache["subaction"],
        image_id=cache["image_id"],
    )
    command = [
        sys.executable,
        str(Path(args.official_eval_script).resolve()),
        "--input",
        str(npz_path),
        "--out-dir",
        str(output),
        "--method",
        args.method,
        "--occlusion-type",
        args.occlusion_type,
        "--ratio",
        str(args.ratio),
        "--body-part",
        args.body_part,
        "--seed",
        str(args.seed),
    ]
    completed = subprocess.run(
        command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    (output / "official_eval.log").write_text(completed.stdout, encoding="utf-8")
    (output / "command.txt").write_text(
        " ".join(shlex.quote(value) for value in sys.argv) + "\n", encoding="utf-8"
    )
    (output / "used_config.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(completed.stdout, end="")


if __name__ == "__main__":
    main()
