#!/usr/bin/env python3
"""Exercise every procedural occlusion type without private dataset files."""

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.occlusion_utils import (
    apply_joint_occlusion,
    apply_random_block_occlusion,
    apply_rectangle_occlusion,
    make_rng,
)


RATIOS = (0.2, 0.4, 0.6)
BODY_PARTS = ("both_arms", "both_legs", "torso")


def synthetic_h36m_keypoints():
    """Return visible 17-joint points covering a 64x64 observation plane."""
    points = np.array(
        [
            [32, 34],  # pelvis
            [25, 35], [23, 46], [22, 58],  # right leg
            [39, 35], [41, 46], [42, 58],  # left leg
            [32, 27], [32, 20], [32, 12], [32, 7],  # torso/head
            [25, 21], [18, 27], [11, 33],  # right arm
            [39, 21], [46, 27], [53, 33],  # left arm
        ],
        dtype=np.float32,
    )
    return points


def summarize(name, mask, boxes):
    coverage = float(mask.mean())
    print(f"{name:24s} coverage={coverage:.4f} boxes={boxes}")


def main():
    image = np.ones((64, 64), dtype=np.float32)

    for ratio in RATIOS:
        _, mask, box = apply_rectangle_occlusion(
            image,
            ratio,
            rng=make_rng(2026, 0, 0, f"smoke:rectangle:{ratio}"),
            return_mask=True,
        )
        summarize(f"rectangle {ratio}", mask, [box])

    for ratio in RATIOS:
        _, mask, boxes = apply_random_block_occlusion(
            image,
            ratio,
            rng=make_rng(2026, 0, 0, f"smoke:random_block:{ratio}"),
            num_blocks=4,
            return_mask=True,
        )
        summarize(f"random_block {ratio}", mask, boxes)

    keypoints = synthetic_h36m_keypoints()
    for body_part in BODY_PARTS:
        _, mask, box = apply_joint_occlusion(
            image,
            keypoints,
            body_part,
            padding=4,
            return_mask=True,
        )
        summarize(f"body_part {body_part}", mask, [box])

    print("\nAll rectangle, random-block, and body-part smoke checks passed.")


if __name__ == "__main__":
    main()
