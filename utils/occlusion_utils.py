import hashlib
import csv
import os
from functools import lru_cache
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from PIL import Image


H36M_BODY_PARTS = {
    "left_arm": [14, 15, 16],
    "right_arm": [11, 12, 13],
    "left_leg": [1, 2, 3],
    "right_leg": [4, 5, 6],
    "both_arms": [11, 12, 13, 14, 15, 16],
    "both_legs": [1, 2, 3, 4, 5, 6],
    "torso": [0, 1, 4, 7, 8, 9, 11, 14],
}


def _stable_int(text: str) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def make_rng(seed: int, scene_id: int, camera_id: int, tag: str = "") -> np.random.Generator:
    full_seed = int(seed) + 1009 * int(scene_id) + 9173 * int(camera_id) + _stable_int(tag)
    return np.random.default_rng(full_seed % (2**32))


def _spatial_shape(image) -> Tuple[int, int]:
    if torch.is_tensor(image):
        if image.ndim < 2:
            raise ValueError(f"Expected image tensor with at least 2 dims, got {image.shape}")
        return int(image.shape[-2]), int(image.shape[-1])
    arr = np.asarray(image)
    if arr.ndim < 2:
        raise ValueError(f"Expected image array with at least 2 dims, got {arr.shape}")
    return int(arr.shape[0]), int(arr.shape[1])


def _fill_region(image, box, mode: str):
    x, y, w, h = box
    if torch.is_tensor(image):
        output = image.clone()
        fill_value = 0.0 if mode == "black" else output.mean()
        output[..., y : y + h, x : x + w] = fill_value
        return output

    output = np.array(image, copy=True)
    fill_value = 0 if mode == "black" else output.mean(axis=(0, 1), keepdims=True)
    output[y : y + h, x : x + w, ...] = fill_value
    return output


def _box_to_mask(height: int, width: int, box) -> np.ndarray:
    x, y, w, h = box
    mask = np.zeros((height, width), dtype=bool)
    if w > 0 and h > 0:
        mask[y : y + h, x : x + w] = True
    return mask


def _sample_rectangle(height: int, width: int, occlusion_ratio: float, rng: np.random.Generator):
    ratio = float(np.clip(occlusion_ratio, 0.0, 1.0))
    if ratio <= 0.0:
        return 0, 0, 0, 0

    target_area = max(1.0, height * width * ratio)
    aspect = float(rng.uniform(0.5, 2.0))
    rect_w = int(round(np.sqrt(target_area * aspect)))
    rect_h = int(round(target_area / max(rect_w, 1)))
    rect_w = int(np.clip(rect_w, 1, width))
    rect_h = int(np.clip(rect_h, 1, height))
    x = int(rng.integers(0, max(width - rect_w + 1, 1)))
    y = int(rng.integers(0, max(height - rect_h + 1, 1)))
    return x, y, rect_w, rect_h


def apply_rectangle_occlusion(image, occlusion_ratio, mode="black", rng=None, return_mask=False):
    """
    Apply one deterministic-area rectangular test-time occluder.

    Supports numpy/PIL images in HxW[xC] format and torch tensors in ...xHxW
    format. The function never mutates the input object.
    """
    if mode not in {"black", "mean"}:
        raise ValueError(f"Unsupported occlusion fill mode: {mode}")
    rng = np.random.default_rng(0) if rng is None else rng
    height, width = _spatial_shape(image)
    box = _sample_rectangle(height, width, occlusion_ratio, rng)
    output = _fill_region(image, box, mode)
    if return_mask:
        return output, _box_to_mask(height, width, box), box
    return output


def apply_random_block_occlusion(
    image,
    occlusion_ratio,
    mode="black",
    rng=None,
    num_blocks=4,
    return_mask=False,
):
    """
    Apply multiple random rectangular blocks with approximately fixed total area.

    This differs from `rectangle` occlusion, which uses one large block. Here the
    configured ratio is split across `num_blocks` smaller blocks, producing a
    scattered missing-observation pattern.
    """
    if mode not in {"black", "mean"}:
        raise ValueError(f"Unsupported occlusion fill mode: {mode}")
    rng = np.random.default_rng(0) if rng is None else rng
    height, width = _spatial_shape(image)
    block_count = max(1, int(num_blocks))
    per_block_ratio = float(np.clip(occlusion_ratio, 0.0, 1.0)) / block_count

    output = image
    full_mask = np.zeros((height, width), dtype=bool)
    boxes = []
    for _ in range(block_count):
        box = _sample_rectangle(height, width, per_block_ratio, rng)
        output = _fill_region(output, box, mode)
        full_mask |= _box_to_mask(height, width, box)
        boxes.append(box)

    if return_mask:
        return output, full_mask, boxes
    return output


def _valid_keypoints(keypoints: np.ndarray, joint_ids: Iterable[int], width: int, height: int):
    points = []
    for joint_id in joint_ids:
        if joint_id < 0 or joint_id >= keypoints.shape[0]:
            continue
        x, y = keypoints[joint_id, :2]
        if np.isfinite(x) and np.isfinite(y) and 0 <= x < width and 0 <= y < height:
            points.append((float(x), float(y)))
    return points


def _joint_bbox(keypoints: np.ndarray, body_part: str, width: int, height: int, padding: int):
    if body_part not in H36M_BODY_PARTS:
        valid = ", ".join(sorted(H36M_BODY_PARTS))
        raise ValueError(f"Unsupported body_part '{body_part}'. Valid options: {valid}")
    points = _valid_keypoints(keypoints, H36M_BODY_PARTS[body_part], width, height)
    if not points:
        return 0, 0, 0, 0
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    x1 = int(np.floor(xs.min() - padding))
    y1 = int(np.floor(ys.min() - padding))
    x2 = int(np.ceil(xs.max() + padding))
    y2 = int(np.ceil(ys.max() + padding))
    x1 = int(np.clip(x1, 0, width - 1))
    y1 = int(np.clip(y1, 0, height - 1))
    x2 = int(np.clip(x2, x1 + 1, width))
    y2 = int(np.clip(y2, y1 + 1, height))
    return x1, y1, x2 - x1, y2 - y1


def apply_joint_occlusion(
    image,
    keypoints,
    body_part,
    mode="black",
    padding=20,
    return_mask=False,
):
    """
    Occlude the image region covered by a Human3.6M body part.

    `keypoints` is expected in H36M 17-joint order with x/y in image pixels.
    """
    if mode not in {"black", "mean"}:
        raise ValueError(f"Unsupported occlusion fill mode: {mode}")
    height, width = _spatial_shape(image)
    keypoints_np = keypoints.detach().cpu().numpy() if torch.is_tensor(keypoints) else np.asarray(keypoints)
    box = _joint_bbox(keypoints_np, body_part, width, height, int(padding))
    output = _fill_region(image, box, mode)
    if return_mask:
        return output, _box_to_mask(height, width, box), box
    return output


def _zero_confidence_inside_mask(poses_2d: torch.Tensor, cam_id: int, mask: np.ndarray) -> torch.Tensor:
    if poses_2d.shape[-1] < 3 or not mask.any():
        return poses_2d
    height, width = mask.shape
    coords = poses_2d[cam_id, :, :2].detach().cpu().numpy()
    xs = np.clip(np.rint(coords[:, 0]).astype(np.int64), 0, width - 1)
    ys = np.clip(np.rint(coords[:, 1]).astype(np.int64), 0, height - 1)
    occluded = torch.as_tensor(mask[ys, xs], device=poses_2d.device, dtype=torch.bool)
    poses_2d[cam_id, occluded, 2] = 0.0
    return poses_2d


@lru_cache(maxsize=8)
def _expected_public_scenes(manifest_path: str):
    if not manifest_path or not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Missing public-occlusion exact manifest: {manifest_path}")
    with open(manifest_path, newline="", encoding="utf-8") as handle:
        scenes = {row["scene_name"] for row in csv.DictReader(handle)}
    if not scenes:
        raise RuntimeError(f"Public-occlusion exact manifest is empty: {manifest_path}")
    return frozenset(scenes)


def _load_public_mask(mask_root: str, scene_name: str, cam_id: int, height: int, width: int):
    mask_path = os.path.join(mask_root, scene_name, f"{cam_id}.png")
    if not os.path.isfile(mask_path):
        raise FileNotFoundError(mask_path)
    mask = np.asarray(Image.open(mask_path).convert("L")) > 0
    transform = "identity"
    if mask.shape == (width, height) and mask.shape != (height, width):
        # The historical H36M loader exposes heatmaps as (width, height), while
        # public RGB masks use the conventional (height, width). Preserve pixel
        # x/y coordinates by copying the shared top-left extent; transposing
        # would move occluders across the image diagonal.
        aligned = np.zeros((height, width), dtype=bool)
        copy_height = min(height, mask.shape[0])
        copy_width = min(width, mask.shape[1])
        aligned[:copy_height, :copy_width] = mask[:copy_height, :copy_width]
        mask = aligned
        transform = "h36m_coordinate_preserving_crop_pad"
    elif mask.shape != (height, width):
        raise RuntimeError(
            f"Public mask shape {mask.shape} does not match observation {(height, width)}: {mask_path}"
        )
    return mask, mask_path, transform


def _apply_arbitrary_mask(image, mask: np.ndarray, mode: str):
    if mode not in {"black", "mean"}:
        raise ValueError(f"Unsupported occlusion fill mode: {mode}")
    if torch.is_tensor(image):
        output = image.clone()
        mask_tensor = torch.as_tensor(mask, device=output.device, dtype=torch.bool)
        fill_value = 0.0 if mode == "black" else output.mean()
        output[..., mask_tensor] = fill_value
        return output
    output = np.array(image, copy=True)
    fill_value = 0 if mode == "black" else output.mean(axis=(0, 1), keepdims=True)
    output[mask, ...] = fill_value
    return output


def apply_test_occlusion_to_observations(
    heatmaps: Dict[str, torch.Tensor],
    poses_2d: torch.Tensor,
    cfg,
    scene_id: int,
    scene_name: str,
):
    """
    Apply configured test-time occlusion to heatmap observations and confidence
    channels. This is intentionally separated from training losses and model code.
    """
    occluded_heatmaps = heatmaps.clone()
    occluded_poses = poses_2d.clone()

    occ_type = str(getattr(cfg, "type", "rectangle")).lower()
    mode = str(getattr(cfg, "mode", "black")).lower()
    seed = int(getattr(cfg, "seed", 0))
    ratio = float(getattr(cfg, "ratio", 0.0))
    body_part = str(getattr(cfg, "body_part", "torso"))
    padding = int(getattr(cfg, "padding", 20))
    num_blocks = int(getattr(cfg, "num_blocks", 4))
    public_mask_root = str(getattr(cfg, "mask_root", ""))
    public_manifest_path = str(getattr(cfg, "manifest_path", ""))
    public_scenes = (
        _expected_public_scenes(public_manifest_path)
        if occ_type == "public_voc"
        else frozenset()
    )

    metadata = []
    for cam_key in sorted(occluded_heatmaps.keys(), key=lambda k: int(k)):
        cam_id = int(cam_key)
        cam_heatmap = occluded_heatmaps[cam_key]
        rng = make_rng(seed, scene_id, cam_id, f"{scene_name}:{occ_type}:{ratio}:{body_part}")

        if occ_type == "rectangle":
            next_heatmap, mask, box = apply_rectangle_occlusion(
                cam_heatmap,
                ratio,
                mode=mode,
                rng=rng,
                return_mask=True,
            )
        elif occ_type == "random_block":
            next_heatmap, mask, box = apply_random_block_occlusion(
                cam_heatmap,
                ratio,
                mode=mode,
                rng=rng,
                num_blocks=num_blocks,
                return_mask=True,
            )
        elif occ_type == "body_part":
            next_heatmap, mask, box = apply_joint_occlusion(
                cam_heatmap,
                occluded_poses[cam_id],
                body_part,
                mode=mode,
                padding=padding,
                return_mask=True,
            )
        elif occ_type == "public_voc":
            if scene_name in public_scenes:
                height, width = _spatial_shape(cam_heatmap)
                mask, mask_path, mask_transform = _load_public_mask(
                    public_mask_root, scene_name, cam_id, height, width
                )
            else:
                height, width = _spatial_shape(cam_heatmap)
                mask = np.zeros((height, width), dtype=bool)
                mask_path = "not-evaluated-scene"
                mask_transform = "none"
            next_heatmap = _apply_arbitrary_mask(cam_heatmap, mask, mode)
            if mask.any():
                ys, xs = np.nonzero(mask)
                box = [
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max() - xs.min() + 1),
                    int(ys.max() - ys.min() + 1),
                ]
            else:
                box = [0, 0, 0, 0]
        else:
            raise ValueError(f"Unsupported test occlusion type: {occ_type}")

        occluded_heatmaps[cam_key] = next_heatmap
        occluded_poses = _zero_confidence_inside_mask(occluded_poses, cam_id, mask)
        item = {"camera": cam_id, "box": box, "mask": mask}
        if occ_type == "public_voc":
            item["mask_path"] = mask_path
            item["mask_transform"] = mask_transform
        metadata.append(item)

    return occluded_heatmaps, occluded_poses, metadata


def save_heatmap_image(heatmap: torch.Tensor, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image = torch.sum(heatmap.detach(), dim=0)
    image = image - image.min()
    denom = image.max().clamp(min=1e-8)
    image = (image / denom * 255.0).cpu().numpy().astype(np.uint8)
    Image.fromarray(image).save(path)
