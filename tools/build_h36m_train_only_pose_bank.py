#!/usr/bin/env python3
"""Build a leakage-free H36M pose bank from training-subject annotations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


TRAIN_SUBJECTS = (1, 5, 6, 7, 8)
TEST_SUBJECTS = (9, 11)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pkl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=4096)
    parser.add_argument("--camera-id", type=int, default=0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def camera_to_world(pose: np.ndarray, camera: dict) -> np.ndarray:
    rotation = np.asarray(camera["R"], dtype=np.float64)
    translation = np.asarray(camera["T"], dtype=np.float64).reshape(3, 1)
    return (rotation.T.dot(np.asarray(pose, dtype=np.float64).T) + translation).T


def main() -> None:
    args = parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be positive")
    if args.max_samples < 1:
        raise ValueError("--max-samples must be positive")

    with args.input_pkl.open("rb") as handle:
        records = pickle.load(handle)

    pool = []
    seen = set()
    for record in records:
        subject = int(record["subject"])
        if subject not in TRAIN_SUBJECTS:
            continue
        if int(record["camera_id"]) != args.camera_id:
            continue
        image_id = int(record["image_id"])
        if (image_id - 1) % args.stride != 0:
            continue
        key_tuple = (
            subject,
            int(record["action"]),
            int(record["subaction"]),
            image_id,
        )
        if key_tuple in seen:
            raise RuntimeError(f"Duplicate frame in selected camera: {key_tuple}")
        seen.add(key_tuple)
        world = camera_to_world(record["joints_3d_camera"], record["camera"])
        if world.shape != (17, 3):
            raise RuntimeError(f"Unexpected joint shape {world.shape} for {key_tuple}")
        root_relative = world - world[0:1]
        scene_key = (
            f"S{subject}_A{key_tuple[1]:02d}_SA{key_tuple[2]:02d}_"
            f"{image_id - 1:06d}"
        )
        pool.append((key_tuple, scene_key, root_relative.astype(np.float32)))

    pool.sort(key=lambda item: item[0])
    if not pool:
        raise RuntimeError("No training poses matched the requested protocol")

    keep = np.linspace(0, len(pool) - 1, args.max_samples, dtype=np.int64)
    selected = [pool[index] for index in keep]
    poses = np.stack([item[2].reshape(-1) for item in selected]).astype(np.float32)
    scene_keys = np.asarray([item[1] for item in selected])
    subjects = np.asarray([item[0][0] for item in selected], dtype=np.int16)

    if set(subjects.tolist()) & set(TEST_SUBJECTS):
        raise RuntimeError("Test subjects S9/S11 entered the training-only pose bank")
    if float(np.abs(poses.reshape(-1, 17, 3)[:, 0]).max()) != 0.0:
        raise RuntimeError("Pose bank is not exactly root-relative")
    if not np.isfinite(poses).all():
        raise RuntimeError("Pose bank contains non-finite values")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        pose_bank=poses,
        scene_keys=scene_keys,
        subjects=subjects,
        source_subjects=np.asarray(TRAIN_SUBJECTS, dtype=np.int16),
        excluded_subjects=np.asarray(TEST_SUBJECTS, dtype=np.int16),
        source_split=np.asarray("H36M train subjects"),
        joint_convention=np.asarray("H36M 17-joint project convention"),
        root_relative=np.asarray(True),
        stride=np.asarray(args.stride, dtype=np.int32),
        max_samples=np.asarray(args.max_samples, dtype=np.int32),
        source_path=np.asarray(str(args.input_pkl.resolve())),
    )

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["scene_key", "subject", "action", "subaction", "image_id", "source_camera_id"]
        )
        for key_tuple, scene_key, _ in selected:
            writer.writerow([scene_key, *key_tuple, args.camera_id])

    summary = {
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "input": str(args.input_pkl.resolve()),
        "input_sha256": sha256(args.input_pkl),
        "source_subjects": list(TRAIN_SUBJECTS),
        "excluded_test_subjects": list(TEST_SUBJECTS),
        "pool_size_after_stride": len(pool),
        "selected_pose_count": len(selected),
        "stride": args.stride,
        "max_samples": args.max_samples,
        "camera_id": args.camera_id,
        "root_relative": True,
        "joint_count": 17,
        "test_subject_assertion": "passed",
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
