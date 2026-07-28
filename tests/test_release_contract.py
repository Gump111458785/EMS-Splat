from pathlib import Path

import numpy as np
import yaml

from utils.occlusion_utils import (
    apply_random_block_occlusion,
    apply_rectangle_occlusion,
    make_rng,
)


ROOT = Path(__file__).resolve().parents[1]


def test_final_h36m_config_is_leakage_safe():
    config = yaml.safe_load((ROOT / "configs" / "h36m.yaml").read_text())
    training = config["training"]

    assert training["use_uncertainty_weighting"] is False
    assert training["use_pose_manifold_prior"] is True
    assert training["pose_bank_allowed_subjects"] == [1, 5, 6, 7, 8]
    assert training["pose_bank_forbidden_subjects"] == [9, 11]
    assert training["lambda_triangulation"] == 0.0
    assert training["lambda_ratio"] == 0.001


def test_rectangle_mask_is_deterministic():
    image = np.ones((64, 64, 3), dtype=np.float32)
    first = apply_rectangle_occlusion(
        image,
        0.4,
        rng=make_rng(2026, 4, 1, "rectangle"),
        return_mask=True,
    )
    second = apply_rectangle_occlusion(
        image,
        0.4,
        rng=make_rng(2026, 4, 1, "rectangle"),
        return_mask=True,
    )

    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert first[2] == second[2]


def test_random_block_returns_requested_block_count():
    image = np.ones((64, 64, 3), dtype=np.float32)
    _, mask, boxes = apply_random_block_occlusion(
        image,
        0.6,
        rng=make_rng(2026, 5, 2, "random-block"),
        num_blocks=4,
        return_mask=True,
    )

    assert mask.any()
    assert len(boxes) == 4
