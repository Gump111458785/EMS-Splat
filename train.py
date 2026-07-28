#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import math
import torch
import numpy as np
from random import randint
from gaussian_renderer import render_functions
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from arguments.config_handler import ConfigHandler
import PIL.Image as Image
from utils.general_utils import generate_heatmaps
from scene.dataset_readers import DataLoader
import hydra
from omegaconf import DictConfig
import sys
import logging
from utils import losses, early_stopping_strategy, consistency_losses
from utils.general_utils import unpack_covariance, OptEarlyStopping
from utils.geometry_utils import (
    cross_modal_epipolar_distances,
    cross_modal_epipolar_loss,
    get_camera_projection_matrix,
    get_fundamental_matrix,
    project_xyz_to_pixels,
    reproject_points,
    weighted_triangulate_points,
)
from utils.occlusion_utils import apply_test_occlusion_to_observations, save_heatmap_image
import matplotlib.pyplot as plt
import json
import hashlib
import torch.nn.functional as F

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

from itertools import combinations

H36M_SYMMETRIC_BONE_PAIRS = [
    ((1, 2), (4, 5)),   # thighs
    ((2, 3), (5, 6)),   # shins
    ((0, 1), (0, 4)),   # pelvis sides
    ((14, 15), (11, 12)),  # upper arms
    ((15, 16), (12, 13)),  # forearms
    ((8, 14), (8, 11)),    # clavicles
]

PANOPTIC_SYMMETRIC_BONE_PAIRS = [
    ((3, 4), (9, 10)),    # upper arms: lShoulder-lElbow vs rShoulder-rElbow
    ((4, 5), (10, 11)),   # forearms: lElbow-lWrist vs rElbow-rWrist
    ((6, 7), (12, 13)),   # thighs: lHip-lKnee vs rHip-rKnee
    ((7, 8), (13, 14)),   # shins: lKnee-lAnkle vs rKnee-rAnkle
    ((0, 3), (0, 9)),     # shoulder offsets from neck
    ((2, 6), (2, 12)),    # hip offsets from body center
]

OCCLUSION_PERSON_SYMMETRIC_BONE_PAIRS = [
    ((1, 2), (4, 5)),      # thighs
    ((2, 3), (5, 6)),      # calves
    ((0, 1), (0, 4)),      # pelvis sides
    ((12, 13), (9, 10)),   # upper arms
    ((13, 14), (10, 11)),  # forearms
    ((8, 12), (8, 9)),     # clavicles
]

H36M_ANATOMICAL_RATIO_GROUPS = [
    # (upper_idx_pair, lower_idx_pair, target_ratio)
    ((14, 15), (15, 16), 1.2),  # right upper/lower arm
    ((11, 12), (12, 13), 1.2),  # left upper/lower arm
    ((1, 2), (2, 3), 1.1),      # right thigh/shin
    ((4, 5), (5, 6), 1.1),      # left thigh/shin
]

OCCLUSION_PERSON_ANATOMICAL_RATIO_GROUPS = [
    ((12, 13), (13, 14), 1.2),  # right upper/lower arm
    ((9, 10), (10, 11), 1.2),   # left upper/lower arm
    ((1, 2), (2, 3), 1.1),      # right thigh/calf
    ((4, 5), (5, 6), 1.1),      # left thigh/calf
]

TRIANGULATION_ENDPOINT_JOINTS = {
    "h36m": [3, 6, 13, 16],  # ankles + wrists
    "occlusion-person": [3, 6, 11, 14],
    "panoptic": [8, 14, 5, 11],
}


def get_triangulation_joint_prior(scene_type, num_joints, endpoint_boost, device, dtype):
    """
    Emphasize unstable end-effectors in the triangulation auxiliary loss.
    """
    prior = torch.ones(num_joints, device=device, dtype=dtype)
    if endpoint_boost <= 1.0:
        return prior

    for key, joint_ids in TRIANGULATION_ENDPOINT_JOINTS.items():
        if key in scene_type:
            valid_ids = [joint_id for joint_id in joint_ids if 0 <= joint_id < num_joints]
            if valid_ids:
                prior[valid_ids] = endpoint_boost
            break
    return prior


def get_symmetry_bone_pairs(scene_type):
    normalized = str(scene_type).lower()
    if "occlusion-person" in normalized:
        return OCCLUSION_PERSON_SYMMETRIC_BONE_PAIRS
    if "panoptic" in normalized:
        return PANOPTIC_SYMMETRIC_BONE_PAIRS
    return H36M_SYMMETRIC_BONE_PAIRS


def get_anatomical_ratio_groups(scene_type):
    if "occlusion-person" in str(scene_type).lower():
        return OCCLUSION_PERSON_ANATOMICAL_RATIO_GROUPS
    return H36M_ANATOMICAL_RATIO_GROUPS


def compute_soft_margin_symmetry_loss(xyz, scene_type="", margin=0.05):
    """
    Soft-margin variant of the bilateral bone-length symmetry prior.
    Deviations smaller than `margin` are tolerated (no penalty).
    """
    penalties = []
    for left, right in get_symmetry_bone_pairs(scene_type):
        left_len = torch.norm(xyz[left[0]] - xyz[left[1]], dim=-1)
        right_len = torch.norm(xyz[right[0]] - xyz[right[1]], dim=-1)
        penalties.append(F.relu(torch.abs(left_len - right_len) - margin))
    if not penalties:
        return torch.tensor(0.0, device=xyz.device, dtype=xyz.dtype)
    return torch.stack(penalties).mean()


def get_symmetry_lambda(current_iter, total_iters, target_lambda, stage1_ratio=0.4, stage2_ratio=0.8):
    """
    Three-phase curriculum:
        1) Stage 1: suppress symmetry term.
        2) Stage 2: linear warm-up towards `target_lambda`.
        3) Stage 3: keep constant at `target_lambda`.
    The default ratios reproduce the requested 0-200-400 schedule when total_iters=500.
    """
    if target_lambda <= 0 or total_iters <= 0:
        return 0.0

    stage1_end = int(stage1_ratio * total_iters)
    stage2_end = int(stage2_ratio * total_iters)
    stage1_end = max(stage1_end, 0)
    stage2_end = max(stage2_end, stage1_end + 1)

    if current_iter <= stage1_end:
        return 0.0
    if current_iter <= stage2_end:
        warmup_span = max(stage2_end - stage1_end, 1)
        warmup_progress = (current_iter - stage1_end) / warmup_span
        return target_lambda * warmup_progress
    return target_lambda


def get_linear_warmup_lambda(current_iter, target_lambda, warmup_start=0, warmup_end=0):
    """
    Linear curriculum for geometry terms.
    - Before `warmup_start`: disabled.
    - Between start/end: linearly ramp to `target_lambda`.
    - After `warmup_end`: keep `target_lambda`.
    """
    if target_lambda <= 0:
        return 0.0

    warmup_start = max(int(warmup_start), 0)
    warmup_end = max(int(warmup_end), warmup_start)

    if warmup_end == 0:
        return target_lambda
    if current_iter <= warmup_start:
        return 0.0
    if current_iter >= warmup_end:
        return target_lambda

    warmup_span = max(warmup_end - warmup_start, 1)
    progress = (current_iter - warmup_start) / warmup_span
    return target_lambda * progress


def set_aux_param_learning_rates(optimizer, freeze_aux_params, base_lrs):
    """
    Freeze or restore the non-xyz Gaussian parameter groups in-place.
    """
    for param_group in optimizer.param_groups:
        group_name = param_group.get("name")
        if group_name in base_lrs:
            param_group["lr"] = 0.0 if freeze_aux_params else base_lrs[group_name]


def soft_anatomical_ratio_loss(xyz, scene_type="", ratio_margin=0.05):
    """
    软边缘解剖比例先验：
    - xyz: (17, 3) 当前 3D 关节点坐标
    - ratio_margin: 可容忍的比例误差范围，避免对 H36M 微小误差过拟合
    """
    ratio_penalties = []

    def bone_length(a, b):
        return torch.norm(xyz[a] - xyz[b], dim=-1)

    for (upper_pair, lower_pair, target) in get_anatomical_ratio_groups(scene_type):
        upper_len = bone_length(*upper_pair)
        lower_len = bone_length(*lower_pair)
        curr_ratio = upper_len / (lower_len + 1e-6)
        deviation = torch.abs(curr_ratio - target)
        ratio_penalties.append(F.relu(deviation - ratio_margin))

    if not ratio_penalties:
        return torch.tensor(0.0, device=xyz.device, dtype=xyz.dtype)
    return torch.stack(ratio_penalties).mean()


def build_epipolar_guidance_heatmaps(
    main_cam,
    train_cameras,
    poses_2d,
    sigma=6.0,
    min_conf=0.05,
    min_views=2,
    eps=1e-6,
):
    """
    Build a soft epipolar prior in the main view from the 2D detections of all other views.
    """
    num_joints = poses_2d.shape[1]
    device = poses_2d.device
    dtype = poses_2d.dtype
    height = main_cam.image_height
    width = main_cam.image_width

    ys = torch.arange(height, device=device, dtype=dtype)
    xs = torch.arange(width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid_x = grid_x.unsqueeze(0)
    grid_y = grid_y.unsqueeze(0)

    prior_sum = torch.zeros((num_joints, height, width), device=device, dtype=dtype)
    weight_sum = torch.zeros((num_joints, 1, 1), device=device, dtype=dtype)
    has_conf_channel = poses_2d.shape[-1] > 2

    for ref_cam in train_cameras:
        if ref_cam.uid == main_cam.uid:
            continue

        ref_points = poses_2d[ref_cam.uid, :, :2]
        if has_conf_channel:
            ref_conf = poses_2d[ref_cam.uid, :, 2]
        else:
            ref_conf = torch.ones(num_joints, device=device, dtype=dtype)

        valid_conf = torch.where(ref_conf >= min_conf, ref_conf, torch.zeros_like(ref_conf))
        if torch.count_nonzero(valid_conf) == 0:
            continue

        ref_to_main = get_fundamental_matrix(ref_cam, main_cam)
        ref_points_h = torch.cat(
            [ref_points, torch.ones((num_joints, 1), device=device, dtype=dtype)],
            dim=1,
        )
        epipolar_lines = (ref_to_main @ ref_points_h.transpose(0, 1)).transpose(0, 1)

        a = epipolar_lines[:, 0].view(num_joints, 1, 1)
        b = epipolar_lines[:, 1].view(num_joints, 1, 1)
        c = epipolar_lines[:, 2].view(num_joints, 1, 1)
        denominator = torch.sqrt(a * a + b * b + eps)
        distances = torch.abs(a * grid_x + b * grid_y + c) / denominator

        line_support = torch.exp(-0.5 * (distances / sigma) ** 2)
        weighted_support = line_support * valid_conf.view(num_joints, 1, 1)

        prior_sum += weighted_support
        weight_sum += valid_conf.view(num_joints, 1, 1)

    prior = prior_sum / torch.clamp(weight_sum, min=eps)
    prior_max = prior.amax(dim=(1, 2), keepdim=True)
    prior = torch.where(prior_max > 0, prior / torch.clamp(prior_max, min=eps), prior)
    valid_joint_mask = (weight_sum.view(num_joints) >= float(min_views) * min_conf).to(dtype)
    valid_ratio = valid_joint_mask.mean()
    return prior, valid_ratio, valid_joint_mask


def fuse_rendered_heatmaps(
    rendered_heatmaps,
    epipolar_prior,
    fusion_strength=0.2,
    valid_joint_mask=None,
):
    """
    Apply a conservative multiplicative boost along epipolar-consistent regions.
    """
    if fusion_strength <= 0:
        return rendered_heatmaps
    if valid_joint_mask is None:
        gate = 1.0
    else:
        gate = valid_joint_mask.view(-1, 1, 1).to(rendered_heatmaps.device, rendered_heatmaps.dtype)
    return rendered_heatmaps * (1.0 + fusion_strength * epipolar_prior * gate)


def extract_heatmap_distribution_stats(heatmaps, temperature=50.0, eps=1e-6):
    """
    Convert each joint heatmap into a soft coordinate plus a confidence score.
    """
    num_joints, height, width = heatmaps.shape
    flat_heatmaps = heatmaps.reshape(num_joints, -1)
    probs = torch.softmax(flat_heatmaps * temperature, dim=-1)

    ys = torch.linspace(0, height - 1, height, device=heatmaps.device, dtype=heatmaps.dtype)
    xs = torch.linspace(0, width - 1, width, device=heatmaps.device, dtype=heatmaps.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    flat_x = grid_x.reshape(1, -1)
    flat_y = grid_y.reshape(1, -1)

    coords_x = (probs * flat_x).sum(dim=-1)
    coords_y = (probs * flat_y).sum(dim=-1)
    coords = torch.stack([coords_x, coords_y], dim=-1)

    entropy = -(probs * torch.log(probs + eps)).sum(dim=-1)
    entropy = entropy / max(math.log(height * width), 1.0)
    peak_prob = probs.max(dim=-1).values
    render_conf = peak_prob * (1.0 - entropy).clamp(min=0.0)
    return coords, render_conf, entropy


def compute_joint_view_uncertainty_weights(
    conf_main,
    conf_ref,
    render_conf=None,
    render_entropy=None,
    epi_distances=None,
    min_weight=0.05,
    render_power=1.0,
    entropy_power=1.0,
    consistency_scale=10.0,
    detach_inputs=True,
    mode="full",
):
    """
    Build a detached per-joint weight from detector confidence, render sharpness,
    and optional epipolar consistency. This keeps bad joints/views from dominating
    the geometry term while avoiding meta-gradients through the weights themselves.
    """
    conf_main = conf_main.reshape(-1)
    conf_ref = conf_ref.reshape(-1)
    base_conf = torch.sqrt(torch.clamp(conf_main * conf_ref, min=0.0))
    weights = base_conf

    if mode in {"full", "render_conf", "render_entropy"}:
        if render_conf is None:
            raise ValueError(f"render_conf is required for uncertainty mode '{mode}'")
        render_conf = render_conf.reshape(-1)
        if detach_inputs:
            render_conf = render_conf.detach()
        render_term = torch.clamp(render_conf, min=0.0, max=1.0).pow(render_power)
        weights = weights * render_term

    if mode in {"full", "entropy", "render_entropy"}:
        if render_entropy is None:
            raise ValueError(f"render_entropy is required for uncertainty mode '{mode}'")
        render_entropy = render_entropy.reshape(-1)
        if detach_inputs:
            render_entropy = render_entropy.detach()
        entropy_term = torch.clamp(1.0 - render_entropy, min=0.0, max=1.0).pow(entropy_power)
        weights = weights * entropy_term

    if mode in {"full", "consistency"} and epi_distances is not None and consistency_scale > 0:
        consistency_term = torch.exp(-epi_distances.detach() / max(consistency_scale, 1e-6))
        weights = weights * consistency_term

    valid_mask = base_conf > 1e-6
    weights = torch.where(valid_mask, torch.clamp(weights, min=min_weight), torch.zeros_like(weights))
    return weights


@torch.no_grad()
def select_reference_view(
    viewpoint_stack,
    main_index,
    poses_2d,
    current_xyz,
    topk=2,
    conf_power=1.0,
    baseline_power=1.0,
    reproj_scale=25.0,
    strategy="random",
    eps=1e-6,
):
    """
    Prefer references with strong detector confidence, wide baseline, and
    low current reprojection error. This avoids wasting geometry updates on
    low-information views while keeping a small amount of diversity.
    """
    other_indices = [i for i in range(len(viewpoint_stack)) if i != main_index]
    if not other_indices:
        return main_index, torch.tensor(0.0, device=current_xyz.device, dtype=current_xyz.dtype), {}

    if strategy == "random":
        ref_index = int(np.random.choice(other_indices))
        return ref_index, torch.tensor(0.0, device=current_xyz.device, dtype=current_xyz.dtype), {}

    main_cam = viewpoint_stack[main_index]
    has_conf_channel = poses_2d.shape[-1] > 2
    if has_conf_channel:
        conf_main = poses_2d[main_cam.uid, :, 2].to(current_xyz.device, current_xyz.dtype)
    else:
        conf_main = torch.ones(poses_2d.shape[1], device=current_xyz.device, dtype=current_xyz.dtype)

    raw_baselines = []
    conf_scores = []
    reproj_scores = []

    for ref_index in other_indices:
        ref_cam = viewpoint_stack[ref_index]
        if has_conf_channel:
            conf_ref = poses_2d[ref_cam.uid, :, 2].to(current_xyz.device, current_xyz.dtype)
        else:
            conf_ref = torch.ones_like(conf_main)

        pair_conf = torch.sqrt(torch.clamp(conf_main * conf_ref, min=0.0))
        conf_scores.append(torch.clamp(pair_conf.mean(), min=eps).pow(conf_power))

        baseline = torch.norm(main_cam.camera_center - ref_cam.camera_center).to(current_xyz.dtype)
        raw_baselines.append(baseline)

        projected_ref = project_xyz_to_pixels(current_xyz.detach(), ref_cam)
        gt_ref = poses_2d[ref_cam.uid, :, :2].to(projected_ref.device, projected_ref.dtype)
        reproj_error = torch.norm(projected_ref - gt_ref, dim=-1)
        ref_weight = conf_ref if has_conf_channel else torch.ones_like(reproj_error)
        reproj_mean = (reproj_error * ref_weight).sum() / torch.clamp(ref_weight.sum(), min=eps)
        reproj_scores.append(torch.exp(-reproj_mean / max(reproj_scale, eps)))

    baseline_tensor = torch.stack(raw_baselines)
    baseline_scores = torch.clamp(
        baseline_tensor / torch.clamp(baseline_tensor.max(), min=eps),
        min=eps,
    ).pow(baseline_power)
    conf_scores = torch.stack(conf_scores)
    reproj_scores = torch.stack(reproj_scores)
    total_scores = conf_scores * baseline_scores * reproj_scores

    k = min(max(int(topk), 1), len(other_indices))
    top_scores, top_positions = torch.topk(total_scores, k=k)
    selection_probs = top_scores / torch.clamp(top_scores.sum(), min=eps)
    sampled_top_pos = torch.multinomial(selection_probs, num_samples=1).item()
    selected_pos = top_positions[sampled_top_pos].item()
    ref_index = other_indices[selected_pos]

    stats = {
        "score": total_scores[selected_pos],
        "conf": conf_scores[selected_pos],
        "baseline": baseline_scores[selected_pos],
        "reproj": reproj_scores[selected_pos],
    }
    return ref_index, total_scores[selected_pos], stats


def compute_hard_joint_adaptive_weights(
    epi_distances,
    conf_main,
    conf_ref,
    running_hardness,
    topk=4,
    boost=1.0,
    momentum=0.8,
    conf_penalty_scale=0.5,
    eps=1e-6,
):
    """
    Focus extra geometry weight on persistently hard joints while keeping the
    average weight near 1.0 so the global loss scale stays stable.
    """
    conf_main = conf_main.reshape(-1)
    conf_ref = conf_ref.reshape(-1)
    pair_conf = torch.sqrt(torch.clamp(conf_main * conf_ref, min=0.0))
    normalized_epi = epi_distances.detach() / torch.clamp(epi_distances.detach().mean(), min=eps)
    current_hardness = normalized_epi * (1.0 + conf_penalty_scale * (1.0 - pair_conf))

    updated_hardness = momentum * running_hardness + (1.0 - momentum) * current_hardness
    adaptive_weights = torch.ones_like(updated_hardness)

    if topk > 0 and boost > 0:
        k = min(int(topk), updated_hardness.numel())
        _, hard_indices = torch.topk(updated_hardness, k=k)
        hard_strength = updated_hardness[hard_indices] / torch.clamp(updated_hardness[hard_indices].mean(), min=eps)
        hard_strength = torch.clamp(hard_strength, min=0.5, max=2.0)
        adaptive_weights[hard_indices] = 1.0 + boost * hard_strength

    adaptive_weights = adaptive_weights / torch.clamp(adaptive_weights.mean(), min=eps)
    hard_ratio = (adaptive_weights > 1.0).float().mean()
    return adaptive_weights, updated_hardness, current_hardness.mean(), hard_ratio


def compute_ema_pose_loss(xyz, ema_xyz, use_root_relative=True, root_weight=0.0):
    """
    Regularize the current pose towards a detached EMA teacher. By default this
    acts on root-relative pose to stabilize articulation without over-constraining
    global translation.
    """
    student_xyz = xyz
    teacher_xyz = ema_xyz.detach()

    if use_root_relative:
        student_xyz = student_xyz - student_xyz[0:1]
        teacher_xyz = teacher_xyz - teacher_xyz[0:1]

    loss = F.smooth_l1_loss(student_xyz, teacher_xyz, reduction="mean")
    if root_weight > 0:
        root_loss = F.smooth_l1_loss(xyz[0:1], ema_xyz[0:1].detach(), reduction="mean")
        loss = loss + root_weight * root_loss
    return loss


def parse_scene_descriptor(scene_name):
    """
    Split a scene name into sequence identity plus frame id.
    Works for names like `S11_Directions 1_000064` and panoptic variants whose
    activity name itself may contain underscores.
    """
    parts = scene_name.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected scene name format: {scene_name}")
    subject = parts[0]
    activity = "_".join(parts[1:-1])
    frame_id = int(parts[-1])
    return subject, activity, frame_id


def compute_temporal_sequence_loss(
    xyz,
    target_xyz,
    use_root_relative=True,
    root_weight=0.0,
    joint_weights=None,
):
    """
    Pull the current pose towards a detached target from neighboring frames in
    the same sequence. Root-relative supervision is the default because it
    stabilizes articulation without pinning global translation too aggressively.
    """
    student_xyz = xyz
    teacher_xyz = target_xyz.detach()

    if use_root_relative:
        student_xyz = student_xyz - student_xyz[0:1]
        teacher_xyz = teacher_xyz - teacher_xyz[0:1]

    per_joint_loss = F.smooth_l1_loss(student_xyz, teacher_xyz, reduction="none").mean(dim=1)
    if joint_weights is not None:
        joint_weights = joint_weights.reshape(-1).to(per_joint_loss.device, per_joint_loss.dtype)
        weight_sum = torch.clamp(joint_weights.sum(), min=1e-6)
        loss = (per_joint_loss * joint_weights).sum() / weight_sum
    else:
        loss = per_joint_loss.mean()
    if root_weight > 0:
        root_loss = F.smooth_l1_loss(xyz[0:1], target_xyz[0:1].detach(), reduction="mean")
        loss = loss + root_weight * root_loss
    return loss


@torch.no_grad()
def build_temporal_sequence_target(
    sequence_state,
    scene_name,
    current_xyz,
    frame_step,
    mode="velocity",
    max_gap_ratio=1.5,
):
    """
    Reuse optimized poses from earlier frames in the same sequence.
    - `prev`: use the previous optimized pose directly.
    - `velocity`: extrapolate from the two previous optimized poses when possible.
    """
    zero = torch.tensor(0.0, device=current_xyz.device, dtype=current_xyz.dtype)
    subject, activity, frame_id = parse_scene_descriptor(scene_name)
    state = sequence_state.get((subject, activity))
    if state is None:
        return None, zero, zero

    prev_frame_id = int(state["prev_frame_id"])
    frame_gap = frame_id - prev_frame_id
    expected_gap = max(int(frame_step), 1)
    max_allowed_gap = max(int(math.ceil(expected_gap * max_gap_ratio)), expected_gap)
    if frame_gap <= 0 or frame_gap > max_allowed_gap:
        return None, zero, zero

    target_xyz = state["prev_xyz"].to(current_xyz.device, current_xyz.dtype)
    has_velocity_target = 0.0

    if mode == "velocity":
        prev_prev_xyz = state.get("prev_prev_xyz")
        prev_prev_frame_id = state.get("prev_prev_frame_id")
        if prev_prev_xyz is not None and prev_prev_frame_id is not None:
            prev_gap = prev_frame_id - int(prev_prev_frame_id)
            if prev_gap > 0:
                dt = float(frame_gap) / float(prev_gap)
                velocity = state["prev_xyz"] - prev_prev_xyz
                target_xyz = state["prev_xyz"] + velocity * dt
                target_xyz = target_xyz.to(current_xyz.device, current_xyz.dtype)
                has_velocity_target = 1.0

    gap_ratio = torch.tensor(
        float(frame_gap) / float(expected_gap),
        device=current_xyz.device,
        dtype=current_xyz.dtype,
    )
    has_velocity_target = torch.tensor(
        has_velocity_target,
        device=current_xyz.device,
        dtype=current_xyz.dtype,
    )
    return target_xyz.detach().clone(), gap_ratio, has_velocity_target


@torch.no_grad()
def update_sequence_state(sequence_state, scene_name, optimized_xyz):
    """
    Store the final optimized pose so later frames in the same sequence can use
    it as a temporal prior.
    """
    subject, activity, frame_id = parse_scene_descriptor(scene_name)
    key = (subject, activity)
    previous_state = sequence_state.get(key)
    next_state = {
        "prev_xyz": optimized_xyz.detach().clone(),
        "prev_frame_id": int(frame_id),
    }
    if previous_state is not None:
        next_state["prev_prev_xyz"] = previous_state.get("prev_xyz").detach().clone()
        next_state["prev_prev_frame_id"] = int(previous_state.get("prev_frame_id"))
    sequence_state[key] = next_state


def build_temporal_uncertainty_gate(
    conf_main,
    conf_ref,
    epi_distances,
    topk=4,
    conf_scale=0.5,
    epi_scale=1.0,
    min_conf=1e-3,
):
    """
    Activate the temporal prior only on the hardest joints in the current
    multiview observation. Hardness combines low detector confidence with high
    epipolar inconsistency.
    """
    conf_main = conf_main.reshape(-1)
    conf_ref = conf_ref.reshape(-1)
    epi_distances = epi_distances.reshape(-1).detach()

    pair_conf = torch.sqrt(torch.clamp(conf_main * conf_ref, min=0.0))
    valid_mask = pair_conf > min_conf
    if not torch.any(valid_mask):
        zeros = torch.zeros_like(pair_conf)
        zero_scalar = torch.tensor(0.0, device=pair_conf.device, dtype=pair_conf.dtype)
        return zeros, zero_scalar, zero_scalar

    valid_epi = epi_distances[valid_mask]
    epi_norm = torch.zeros_like(epi_distances)
    epi_norm[valid_mask] = valid_epi / torch.clamp(valid_epi.mean(), min=1e-6)
    conf_uncertainty = 1.0 - pair_conf
    uncertainty_score = conf_scale * conf_uncertainty + epi_scale * epi_norm
    masked_score = torch.where(
        valid_mask,
        uncertainty_score,
        torch.full_like(uncertainty_score, float("-inf")),
    )

    num_valid = int(valid_mask.sum().item())
    k = min(max(int(topk), 1), num_valid)
    _, selected_indices = torch.topk(masked_score, k=k)
    gate = torch.zeros_like(pair_conf)
    gate[selected_indices] = 1.0

    score_mean = uncertainty_score[valid_mask].mean()
    gate_ratio = gate.mean()
    return gate, score_mean, gate_ratio


def load_pose_array(file_path):
    if not os.path.exists(file_path):
        return None
    data = np.load(file_path, allow_pickle=False)
    for key in ["poses", "poses2d", "boxes", "poses3d", "scores", "joint_errors"]:
        if key in data:
            return data[key]
    return None


def normalize_pose_bank_pose(pose_xyz, use_root_relative=True):
    if use_root_relative:
        pose_xyz = pose_xyz - pose_xyz[0:1]
    return pose_xyz


def load_or_build_pose_bank(
    data_root,
    num_joints,
    nviews,
    frame_stride=8,
    max_samples=4096,
    use_root_relative=True,
    cache=True,
    explicit_path=None,
    allowed_subjects=None,
    forbidden_subjects=None,
):
    """
    Build a dataset-level pose bank from 3D GT poses. This provides a static
    articulation prior that does not rely on temporal continuity across scenes.
    """
    if explicit_path:
        explicit_path = os.path.abspath(os.path.expanduser(str(explicit_path)))
        if not os.path.isfile(explicit_path):
            raise FileNotFoundError(f"Explicit pose bank does not exist: {explicit_path}")
        cached = np.load(explicit_path, allow_pickle=False)
        required_keys = {"pose_bank", "scene_keys", "subjects", "root_relative"}
        missing_keys = required_keys.difference(cached.files)
        if missing_keys:
            raise RuntimeError(
                f"Explicit pose bank is missing provenance keys {sorted(missing_keys)}: {explicit_path}"
            )
        pose_bank = np.asarray(cached["pose_bank"], dtype=np.float32)
        scene_keys = np.asarray(cached["scene_keys"])
        subjects = {int(subject) for subject in np.asarray(cached["subjects"]).reshape(-1)}
        if pose_bank.ndim != 2 or pose_bank.shape[1] != num_joints * 3:
            raise RuntimeError(
                f"Explicit pose bank shape {pose_bank.shape} is incompatible with {num_joints} joints"
            )
        if pose_bank.shape[0] != scene_keys.shape[0]:
            raise RuntimeError("Explicit pose bank and scene_keys have different lengths")
        if bool(np.asarray(cached["root_relative"]).item()) != bool(use_root_relative):
            raise RuntimeError("Explicit pose-bank root-relative setting does not match the config")
        allowed = {int(subject) for subject in (allowed_subjects or [])}
        forbidden = {int(subject) for subject in (forbidden_subjects or [])}
        if allowed and not subjects.issubset(allowed):
            raise RuntimeError(
                f"Explicit pose bank contains subjects outside allowlist: {sorted(subjects - allowed)}"
            )
        if subjects & forbidden:
            raise RuntimeError(
                f"Explicit pose bank contains forbidden subjects: {sorted(subjects & forbidden)}"
            )
        logging.info(
            "Loaded explicit pose bank %s with %d poses from subjects %s",
            explicit_path,
            pose_bank.shape[0],
            sorted(subjects),
        )
        return pose_bank, scene_keys

    cache_name = (
        f"pose_bank_j{num_joints}_n{nviews}_s{int(frame_stride)}"
        f"_m{int(max_samples)}_rr{int(bool(use_root_relative))}.npz"
    )
    cache_path = os.path.join(data_root, cache_name)
    if cache and os.path.exists(cache_path):
        cached = np.load(cache_path, allow_pickle=False)
        return cached["pose_bank"], cached["scene_keys"]

    gt_root = os.path.join(data_root, "3d_gt")
    bank_poses = []
    bank_scene_keys = []
    stride = max(int(frame_stride), 1)

    for subject in sorted(os.listdir(gt_root)):
        subject_path = os.path.join(gt_root, subject)
        if not os.path.isdir(subject_path):
            continue
        for activity in sorted(os.listdir(subject_path)):
            activity_path = os.path.join(subject_path, activity)
            if not os.path.isdir(activity_path):
                continue

            pose_path = os.path.join(activity_path, "poses.npz")
            if "panoptic" in data_root:
                filtered_pose_path = os.path.join(activity_path, f"poses_filtered_{nviews}.npz")
                poses = load_pose_array(filtered_pose_path)
                if poses is None:
                    poses = load_pose_array(pose_path)
            else:
                poses = load_pose_array(pose_path)
            if poses is None:
                continue

            poses = np.asarray(poses, dtype=np.float32)
            if poses.ndim != 3 or poses.shape[1] != num_joints or poses.shape[2] != 3:
                continue

            for frame_id in range(0, poses.shape[0], stride):
                pose_xyz = normalize_pose_bank_pose(poses[frame_id], use_root_relative=use_root_relative)
                bank_poses.append(pose_xyz.reshape(-1))
                bank_scene_keys.append(f"{subject}_{activity}_{frame_id:06d}")

    if not bank_poses:
        raise RuntimeError(f"Could not build pose bank from {gt_root}")

    pose_bank = np.stack(bank_poses, axis=0).astype(np.float32)
    scene_keys = np.asarray(bank_scene_keys)

    if max_samples > 0 and pose_bank.shape[0] > max_samples:
        keep_ids = np.linspace(0, pose_bank.shape[0] - 1, max_samples, dtype=np.int64)
        pose_bank = pose_bank[keep_ids]
        scene_keys = scene_keys[keep_ids]

    if cache:
        np.savez(cache_path, pose_bank=pose_bank, scene_keys=scene_keys)

    return pose_bank, scene_keys


def compute_pose_manifold_loss(
    xyz,
    pose_bank,
    scene_name=None,
    pose_bank_scene_keys=None,
    use_root_relative=True,
    topk=4,
):
    """
    Pull the current pose towards the nearest poses in a dataset-level manifold.
    This constrains articulation without assuming any temporal relation between
    samples.
    """
    current_pose = normalize_pose_bank_pose(xyz, use_root_relative=use_root_relative)
    flat_pose = current_pose.reshape(1, -1)
    distances = ((pose_bank - flat_pose) ** 2).mean(dim=1)

    if scene_name is not None and pose_bank_scene_keys is not None:
        exclude_mask = torch.as_tensor(
            pose_bank_scene_keys == scene_name,
            device=pose_bank.device,
            dtype=torch.bool,
        )
        if torch.any(exclude_mask):
            distances = distances.masked_fill(exclude_mask, float("inf"))

    finite_mask = torch.isfinite(distances)
    if not torch.any(finite_mask):
        zero = torch.tensor(0.0, device=xyz.device, dtype=xyz.dtype)
        return zero, zero, 0

    valid_distances = distances[finite_mask]
    k = min(max(int(topk), 1), int(valid_distances.numel()))
    topk_values, topk_indices = torch.topk(distances, k=k, largest=False)
    target_pose = pose_bank[topk_indices].mean(dim=0).reshape_as(current_pose)
    loss = F.smooth_l1_loss(current_pose, target_pose.detach(), reduction="mean")
    prior_distance = torch.sqrt(topk_values.mean())
    return loss, prior_distance, k


@torch.no_grad()
def build_triangulation_pseudo_target(
    train_cameras,
    poses_2d,
    render,
    gaussians,
    pipe,
    bg,
    model,
    main_cam,
    main_render,
    current_xyz,
    temperature=50.0,
    min_view_weight=1e-3,
    min_views=3,
    reproj_error_threshold=8.0,
):
    pred_points = []
    view_weights = []
    projection_matrices = []
    entropies = []
    has_conf_channel = poses_2d.shape[-1] > 2

    for cam in train_cameras:
        if cam.uid == main_cam.uid:
            rendered_heatmaps = main_render.detach()
        else:
            rendered_heatmaps = render(
                cam,
                gaussians,
                pipe,
                bg,
                use_trained_exp=model.train_test_exp,
                separate_sh=SPARSE_ADAM_AVAILABLE,
            )["render"].detach()

        pred_coords, render_conf, entropy = extract_heatmap_distribution_stats(
            rendered_heatmaps,
            temperature=temperature,
        )

        if has_conf_channel:
            det_conf = poses_2d[cam.uid, :, 2].to(rendered_heatmaps.device, rendered_heatmaps.dtype)
        else:
            det_conf = torch.ones(
                poses_2d.shape[1],
                device=rendered_heatmaps.device,
                dtype=rendered_heatmaps.dtype,
            )

        pred_points.append(pred_coords)
        view_weights.append(torch.clamp(det_conf * render_conf, min=0.0))
        projection_matrices.append(get_camera_projection_matrix(cam))
        entropies.append(entropy)

    pred_points = torch.stack(pred_points, dim=0)
    view_weights = torch.stack(view_weights, dim=0)
    projection_matrices = torch.stack(projection_matrices, dim=0)
    entropies = torch.stack(entropies, dim=0)
    det_points = poses_2d[:, :, :2].to(pred_points.device, pred_points.dtype)

    triangulated_xyz = weighted_triangulate_points(
        projection_matrices,
        pred_points,
        weights=view_weights,
    )

    reprojected_points = reproject_points(projection_matrices, triangulated_xyz)
    reprojection_errors = torch.norm(reprojected_points - det_points, dim=-1)
    weight_sum = torch.clamp(view_weights.sum(dim=0), min=1e-6)
    mean_reprojection_error = (reprojection_errors * view_weights).sum(dim=0) / weight_sum

    enough_views = (view_weights > min_view_weight).sum(dim=0) >= min_views
    low_reprojection_error = mean_reprojection_error <= reproj_error_threshold
    valid_joints = enough_views & low_reprojection_error
    joint_weights = view_weights.mean(dim=0) * torch.exp(
        -mean_reprojection_error / max(reproj_error_threshold, 1e-6)
    )
    joint_weights = torch.where(valid_joints, joint_weights, torch.zeros_like(joint_weights))
    safe_target = torch.where(
        valid_joints[:, None],
        triangulated_xyz,
        current_xyz.detach(),
    )

    valid_ratio = valid_joints.float().mean()
    return safe_target, joint_weights, view_weights.mean(), entropies.mean(), mean_reprojection_error.mean(), valid_ratio


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, key):
        return getattr(cfg, key)
    try:
        return cfg[key]
    except Exception:
        return default


def _cfg_flag(cfg, *keys, default=False):
    for key in keys:
        value = _cfg_get(cfg, key, None)
        if value is not None:
            return bool(value)
    return bool(default)


def _occlusion_ratio_label(occlusion_cfg):
    occ_type = str(_cfg_get(occlusion_cfg, "type", "rectangle")).lower()
    if occ_type == "body_part":
        return str(_cfg_get(occlusion_cfg, "body_part", "torso"))
    return str(_cfg_get(occlusion_cfg, "ratio", 0.0)).replace(".", "p")


def training(dataset, model, opt, pipe, debug, training, dataset_loader, output_dir, log, test_cfg=None):
    
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    opt_criterion = losses[training.loss_function]
    consistency_criterion = consistency_losses[training.consistency_loss]
    render = render_functions[pipe.rendering]
    early_stopping = early_stopping_strategy[training.early_stopping]()

    tb_writer = prepare_output_and_logger(
        output_dir,
        enabled=bool(getattr(training, "enable_tensorboard", True)),
    )
    
    bg_color = [1, 1, 1] if model.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    bg = torch.rand((3), device="cda") if opt.random_background else background

    log.info(f"Training on {len(dataset_loader)} scenes")
    target_lambda_sym = getattr(training, "lambda_symmetry_target", 5e-4)
    symmetry_margin = getattr(training, "symmetry_margin", 0.05)
    lambda_epi = getattr(training, "lambda_epipolar", 5e-4)
    total_iterations = opt.iterations
    lambda_ratio = getattr(training, "lambda_ratio", 1e-3)
    ratio_margin = getattr(training, "ratio_margin", 0.05)
    ratio_warmup = getattr(training, "ratio_warmup", 200)
    fusion_strength = getattr(training, "epipolar_fusion_strength", 0.2)
    fusion_sigma = getattr(training, "epipolar_fusion_sigma", 6.0)
    fusion_conf_threshold = getattr(training, "epipolar_fusion_conf_threshold", 0.05)
    fusion_min_views = getattr(training, "epipolar_fusion_min_views", 2)
    fusion_min_valid_ratio = getattr(training, "epipolar_fusion_min_valid_ratio", 0.5)
    fusion_warmup = getattr(training, "epipolar_fusion_warmup", 0)
    lambda_triang = getattr(training, "lambda_triangulation", 1e-4)
    triangulation_warmup = getattr(training, "triangulation_warmup", 200)
    epipolar_warmup_start = getattr(training, "epipolar_warmup_start", 0)
    epipolar_warmup_end = getattr(training, "epipolar_warmup_end", epipolar_warmup_start)
    triangulation_warmup_start = getattr(training, "triangulation_warmup_start", triangulation_warmup)
    triangulation_warmup_end = getattr(training, "triangulation_warmup_end", triangulation_warmup_start)
    triangulation_temperature = getattr(training, "triangulation_temperature", 50.0)
    triangulation_min_view_weight = getattr(training, "triangulation_min_view_weight", 0.05)
    triangulation_min_views = getattr(training, "triangulation_min_views", 3)
    triangulation_reproj_error = getattr(training, "triangulation_reproj_error", 8.0)
    triangulation_endpoint_boost = getattr(training, "triangulation_endpoint_boost", 1.5)
    xyz_only_warmup = getattr(training, "xyz_only_warmup", 0)
    use_uncertainty_weighting = getattr(training, "use_uncertainty_weighting", False)
    uncertainty_mode = getattr(training, "uncertainty_mode", "full")
    uncertainty_temperature = getattr(training, "uncertainty_temperature", 50.0)
    uncertainty_min_weight = getattr(training, "uncertainty_min_weight", 0.05)
    uncertainty_render_power = getattr(training, "uncertainty_render_power", 1.0)
    uncertainty_entropy_power = getattr(training, "uncertainty_entropy_power", 1.0)
    uncertainty_consistency_scale = getattr(training, "uncertainty_consistency_scale", 10.0)
    hard_joint_adaptive_weighting = getattr(training, "hard_joint_adaptive_weighting", False)
    hard_joint_topk = getattr(training, "hard_joint_topk", 4)
    hard_joint_boost = getattr(training, "hard_joint_boost", 1.0)
    hard_joint_momentum = getattr(training, "hard_joint_momentum", 0.8)
    hard_joint_conf_penalty_scale = getattr(training, "hard_joint_conf_penalty_scale", 0.5)
    hard_joint_warmup = getattr(training, "hard_joint_warmup", 100)
    reference_sampling_strategy = getattr(training, "reference_sampling_strategy", "random")
    reference_sampling_topk = getattr(training, "reference_sampling_topk", 2)
    reference_sampling_conf_power = getattr(training, "reference_sampling_conf_power", 1.0)
    reference_sampling_baseline_power = getattr(training, "reference_sampling_baseline_power", 1.0)
    reference_sampling_reproj_scale = getattr(training, "reference_sampling_reproj_scale", 25.0)
    use_ema_teacher = getattr(training, "use_ema_teacher", False)
    ema_decay = getattr(training, "ema_decay", 0.995)
    ema_warmup_start = getattr(training, "ema_warmup_start", 50)
    ema_warmup_end = getattr(training, "ema_warmup_end", 150)
    lambda_ema = getattr(training, "lambda_ema", 2e-4)
    ema_use_root_relative = getattr(training, "ema_use_root_relative", True)
    ema_root_weight = getattr(training, "ema_root_weight", 0.0)
    use_temporal_sequence_prior = getattr(training, "use_temporal_sequence_prior", False)
    temporal_mode = getattr(training, "temporal_mode", "velocity")
    lambda_temporal = getattr(training, "lambda_temporal", 2e-4)
    temporal_warmup_start = getattr(training, "temporal_warmup_start", 100)
    temporal_warmup_end = getattr(training, "temporal_warmup_end", 200)
    temporal_use_root_relative = getattr(training, "temporal_use_root_relative", True)
    temporal_root_weight = getattr(training, "temporal_root_weight", 0.0)
    temporal_max_gap_ratio = getattr(training, "temporal_max_gap_ratio", 1.5)
    use_temporal_uncertainty_gating = getattr(training, "use_temporal_uncertainty_gating", False)
    temporal_gate_topk = getattr(training, "temporal_gate_topk", 4)
    temporal_gate_conf_scale = getattr(training, "temporal_gate_conf_scale", 0.5)
    temporal_gate_epi_scale = getattr(training, "temporal_gate_epi_scale", 1.0)
    temporal_gate_min_conf = getattr(training, "temporal_gate_min_conf", 1e-3)
    use_pose_manifold_prior = getattr(training, "use_pose_manifold_prior", False)
    lambda_pose_manifold = getattr(training, "lambda_pose_manifold", 5e-5)
    pose_manifold_warmup_start = getattr(training, "pose_manifold_warmup_start", 150)
    pose_manifold_warmup_end = getattr(training, "pose_manifold_warmup_end", 250)
    pose_bank_stride = getattr(training, "pose_bank_stride", 8)
    pose_bank_max_samples = getattr(training, "pose_bank_max_samples", 4096)
    pose_bank_root_relative = getattr(training, "pose_bank_root_relative", True)
    pose_bank_cache = getattr(training, "pose_bank_cache", True)
    pose_bank_path = getattr(training, "pose_bank_path", None)
    pose_bank_allowed_subjects = getattr(training, "pose_bank_allowed_subjects", None)
    pose_bank_forbidden_subjects = getattr(training, "pose_bank_forbidden_subjects", None)
    pose_manifold_topk = getattr(training, "pose_manifold_topk", 4)
    pose_manifold_exclude_self = getattr(training, "pose_manifold_exclude_self", True)
    aux_param_base_lrs = {
        "opacity": opt.opacity_lr,
        "scaling": opt.scaling_lr,
        "rotation": opt.rotation_lr,
    }
    sequence_state = {}
    occlusion_cfg = _cfg_get(test_cfg, "occlusion", None)
    occlusion_enabled = _cfg_flag(occlusion_cfg, "enable", "ENABLE", default=False)
    occlusion_save_vis = _cfg_flag(occlusion_cfg, "save_visualizations", "SAVE_VISUALIZATIONS", default=False)
    occlusion_vis_limit = int(_cfg_get(occlusion_cfg, "vis_num_samples", 10))
    occlusion_vis_saved = 0
    if occlusion_enabled:
        log.info(
            "Test occlusion enabled: type=%s ratio=%s body_part=%s mode=%s seed=%s",
            _cfg_get(occlusion_cfg, "type", "rectangle"),
            _cfg_get(occlusion_cfg, "ratio", 0.0),
            _cfg_get(occlusion_cfg, "body_part", "torso"),
            _cfg_get(occlusion_cfg, "mode", "black"),
            _cfg_get(occlusion_cfg, "seed", 0),
        )
    if use_pose_manifold_prior:
        pose_bank_np, pose_bank_scene_keys = load_or_build_pose_bank(
            dataset.data_root,
            dataset_loader.n_joints,
            dataset.nviews,
            frame_stride=pose_bank_stride,
            max_samples=pose_bank_max_samples,
            use_root_relative=pose_bank_root_relative,
            cache=pose_bank_cache,
            explicit_path=pose_bank_path,
            allowed_subjects=pose_bank_allowed_subjects,
            forbidden_subjects=pose_bank_forbidden_subjects,
        )
        pose_bank = torch.tensor(pose_bank_np, device="cuda", dtype=torch.float32)
        log.info(f"Loaded pose bank with {pose_bank.shape[0]} poses from {dataset.data_root}")
    else:
        pose_bank = None
        pose_bank_scene_keys = None

    for scene_id, scene_data in dataset_loader:
        
        pose_3d, pose_3d_gt, poses_2d, cameras, scene_name = scene_data 
        poses_2d = poses_2d.to("cuda")
        pose_3d_gt = np.asarray(pose_3d_gt, dtype=np.float32)
        pose_3d_gt = torch.tensor(pose_3d_gt, dtype=torch.float32, device="cuda")

        if training.std_dev_noise > 0.0:
            log.info(f"Adding Gaussian noise with std. dev. {training.std_dev_noise} to 3D initial pose")
            rng = np.random.default_rng(seed=0)  # reproducible
            noise = rng.normal(loc=0.0, scale=training.std_dev_noise, size=pose_3d.shape)
            pose_3d = pose_3d + noise

        first_iter = 0
        gaussians = GaussianModel(model.sh_degree, opt.optimizer_type)
        scene = Scene(dataset, model, gaussians, pose_3d, cameras, scene_name, output_dir)
        gaussians.training_setup(opt)

        covariance_3d = unpack_covariance(gaussians.get_covariance())
        heatmaps_cameras = generate_heatmaps(gaussians, poses_2d, scene.getTrainCameras(), covariance_3d, training.dropout, dataset.data_root, dataset.nviews)
        original_heatmaps_cameras = heatmaps_cameras.clone()
        occlusion_vis_record = None
        if occlusion_enabled:
            heatmaps_cameras, poses_2d, occlusion_metadata = apply_test_occlusion_to_observations(
                heatmaps_cameras,
                poses_2d,
                occlusion_cfg,
                scene_id=scene_id,
                scene_name=scene_name,
            )
            mask_manifest_path = os.path.join(output_dir, "occlusion_mask_manifest.jsonl")
            with open(mask_manifest_path, "a", encoding="utf-8") as mask_manifest:
                for item in occlusion_metadata:
                    mask_value = item["mask"]
                    if torch.is_tensor(mask_value):
                        mask_value = mask_value.detach().to("cpu").numpy()
                    mask_np = np.asarray(mask_value, dtype=np.uint8)
                    mask_manifest.write(
                        json.dumps(
                            {
                                "scene_id": int(scene_id),
                                "scene_name": scene_name,
                                "camera": int(item["camera"]),
                                "box": np.asarray(item["box"]).tolist(),
                                "mask_shape": list(mask_np.shape),
                                "masked_pixels": int(mask_np.sum()),
                                "mask_sha256": hashlib.sha256(mask_np.tobytes()).hexdigest(),
                                "mask_path": item.get("mask_path", ""),
                                "mask_transform": item.get("mask_transform", "identity"),
                                "occlusion_type": str(_cfg_get(occlusion_cfg, "type", "rectangle")),
                                "ratio": float(_cfg_get(occlusion_cfg, "ratio", 0.0)),
                                "seed": int(_cfg_get(occlusion_cfg, "seed", 0)),
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
            if occlusion_save_vis and occlusion_vis_saved < occlusion_vis_limit:
                ratio_label = _occlusion_ratio_label(occlusion_cfg)
                sample_dir = os.path.join(
                    output_dir,
                    "occlusion_vis",
                    f"{ratio_label}",
                    f"sample_{occlusion_vis_saved:03d}_{scene_name}",
                )
                save_heatmap_image(original_heatmaps_cameras["0"], os.path.join(sample_dir, "original.jpg"))
                save_heatmap_image(heatmaps_cameras["0"], os.path.join(sample_dir, "occluded.jpg"))
                occlusion_vis_record = sample_dir
                occlusion_vis_saved += 1

        # To visualize the initial guess

        # fig = plt.figure(figsize=(10, 7))
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(pose_3d[:, 0], pose_3d[:, 1], pose_3d[:, 2], color='r', label='Initial Guess Pose', s=20)
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # ax.legend()
        # plt.show()
        
        iter_start = torch.cuda.Event(enable_timing = True)
        iter_end = torch.cuda.Event(enable_timing = True)

        viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_indices = list(range(len(viewpoint_stack)))
        cam_idx_counter = 0

        # to save gt heatmaps
        if debug.save_images:
            save_heatmaps(len(viewpoint_stack), heatmaps_cameras, output_dir, name="heatmap")

        accumulated_loss_total = 0.0

        first_iter += 1  

        grads = []
        accumulated_grads = torch.zeros((len(viewpoint_stack), gaussians.get_xyz.shape[0], gaussians.get_xyz.shape[1]), device="cuda")

        # to compute errors
        errors_all = []
        errors_rel_all = []

        triang_joint_prior = get_triangulation_joint_prior(
            dataset.data_root,
            gaussians.get_xyz.shape[0],
            triangulation_endpoint_boost,
            gaussians.get_xyz.device,
            gaussians.get_xyz.dtype,
        )
        joint_hardness_ema = torch.ones(
            gaussians.get_xyz.shape[0],
            device=gaussians.get_xyz.device,
            dtype=gaussians.get_xyz.dtype,
        )
        ema_xyz = gaussians.get_xyz.detach().clone()
        temporal_target_xyz, temporal_gap_ratio, temporal_has_velocity = build_temporal_sequence_target(
            sequence_state,
            scene_name,
            gaussians.get_xyz,
            frame_step=dataset.frame_step,
            mode=temporal_mode,
            max_gap_ratio=temporal_max_gap_ratio,
        )
        temporal_prior_active = float(temporal_target_xyz is not None)

        # early stopping
        stop = False

        # ======= 【测速：记录开始时间】 =======
        global_start = torch.cuda.Event(enable_timing=True)
        global_end = torch.cuda.Event(enable_timing=True)
        global_start.record()
        # ====================================

        for iteration in range(first_iter, opt.iterations + 1):

            # 【这是你丢失的两行代码，必须原封不动补回来】
            iter_start.record()
            gaussians.update_learning_rate(iteration)
            freeze_aux_params = iteration <= xyz_only_warmup
            set_aux_param_learning_rates(
                gaussians.optimizer,
                freeze_aux_params=freeze_aux_params,
                base_lrs=aux_param_base_lrs,
            )

            # ======= 1. 双视角采样 =======
            idx_main = viewpoint_indices[cam_idx_counter % len(viewpoint_stack)]
            viewpoint_cam_main = viewpoint_stack[idx_main]
            cam_idx_counter += 1

            idx_ref, selected_ref_score, selected_ref_stats = select_reference_view(
                viewpoint_stack,
                idx_main,
                poses_2d,
                gaussians.get_xyz,
                topk=reference_sampling_topk,
                conf_power=reference_sampling_conf_power,
                baseline_power=reference_sampling_baseline_power,
                reproj_scale=reference_sampling_reproj_scale,
                strategy=reference_sampling_strategy,
            )
            viewpoint_cam_ref = viewpoint_stack[idx_ref]
            reference_score_conf = selected_ref_stats.get(
                "conf",
                torch.tensor(0.0, device=gaussians.get_xyz.device, dtype=gaussians.get_xyz.dtype),
            )
            reference_score_baseline = selected_ref_stats.get(
                "baseline",
                torch.tensor(0.0, device=gaussians.get_xyz.device, dtype=gaussians.get_xyz.dtype),
            )
            reference_score_reproj = selected_ref_stats.get(
                "reproj",
                torch.tensor(0.0, device=gaussians.get_xyz.device, dtype=gaussians.get_xyz.dtype),
            )

            # ======= 2. 主视角光栅化渲染 =======
            render_pkg = render(viewpoint_cam_main, gaussians, pipe, bg, use_trained_exp=model.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            # ======= 3. 基础 L2 热力图损失 =======
            c_main = viewpoint_cam_main.uid
            gt_heatmaps = heatmaps_cameras[str(c_main)]
            if iteration >= fusion_warmup and fusion_strength > 0:
                epipolar_prior, fusion_valid_ratio, fusion_valid_joint_mask = build_epipolar_guidance_heatmaps(
                    viewpoint_cam_main,
                    viewpoint_stack,
                    poses_2d,
                    sigma=fusion_sigma,
                    min_conf=fusion_conf_threshold,
                    min_views=fusion_min_views,
                )
                if fusion_valid_ratio.item() >= fusion_min_valid_ratio:
                    fused_image = fuse_rendered_heatmaps(
                        image,
                        epipolar_prior,
                        fusion_strength=fusion_strength,
                        valid_joint_mask=fusion_valid_joint_mask,
                    )
                    fusion_gate = torch.tensor(1.0, device=image.device, dtype=image.dtype)
                else:
                    fused_image = image
                    fusion_gate = torch.tensor(0.0, device=image.device, dtype=image.dtype)
                fusion_prior_strength = epipolar_prior.mean()
                fusion_delta = (fused_image - image).abs().mean()
            else:
                fused_image = image
                fusion_valid_ratio = torch.tensor(0.0, device=image.device, dtype=image.dtype)
                fusion_prior_strength = torch.tensor(0.0, device=image.device, dtype=image.dtype)
                fusion_delta = torch.tensor(0.0, device=image.device, dtype=image.dtype)
                fusion_gate = torch.tensor(0.0, device=image.device, dtype=image.dtype)

            l2_loss, error = opt_criterion(
                fused_image,
                gt_heatmaps,
                poses_2d[c_main, :, :2],
                training.lambda_loss_function,
                reduction="mean",
            )

            # 3D 长度一致性损失
            loss_consistency = consistency_criterion(gaussians.get_xyz, dataset.data_root, reduction="mean") * training.lambda_consistency

            # ======= 4. 极线几何一致性损失 =======
            pts2d_proj_main = project_xyz_to_pixels(gaussians.get_xyz, viewpoint_cam_main)

            has_conf_channel = poses_2d.shape[-1] > 2
            conf_main = poses_2d[c_main, :, 2] if has_conf_channel else torch.ones(
                poses_2d.shape[1], device=poses_2d.device, dtype=poses_2d.dtype
            )
            c_ref = viewpoint_cam_ref.uid
            pts2d_gt_ref = poses_2d[c_ref, :, :2]
            F_main_to_ref = get_fundamental_matrix(viewpoint_cam_main, viewpoint_cam_ref)
            conf_ref = poses_2d[c_ref, :, 2] if has_conf_channel else torch.ones_like(conf_main)
            joint_conf = (conf_main * conf_ref) + 1e-5
            need_temporal_gate = use_temporal_sequence_prior and use_temporal_uncertainty_gating and temporal_target_xyz is not None
            need_epi_distances = use_uncertainty_weighting or hard_joint_adaptive_weighting or need_temporal_gate
            if need_epi_distances:
                epi_distances = cross_modal_epipolar_distances(
                    pts2d_proj_main,
                    pts2d_gt_ref,
                    F_main_to_ref,
                )
            adaptive_joint_weight_mean = torch.tensor(0.0, device=image.device, dtype=image.dtype)
            adaptive_joint_hard_ratio = torch.tensor(0.0, device=image.device, dtype=image.dtype)
            adaptive_joint_hardness = torch.tensor(0.0, device=image.device, dtype=image.dtype)
            if use_uncertainty_weighting:
                render_conf_main = None
                render_entropy_main = None
                if uncertainty_mode in {"full", "render_conf", "entropy", "render_entropy"}:
                    _, render_conf_main, render_entropy_main = extract_heatmap_distribution_stats(
                        image,
                        temperature=uncertainty_temperature,
                    )
                joint_weights = compute_joint_view_uncertainty_weights(
                    conf_main.to(image.device, image.dtype),
                    conf_ref.to(image.device, image.dtype),
                    render_conf_main,
                    render_entropy_main,
                    epi_distances=epi_distances,
                    min_weight=uncertainty_min_weight,
                    render_power=uncertainty_render_power,
                    entropy_power=uncertainty_entropy_power,
                    consistency_scale=uncertainty_consistency_scale,
                    mode=uncertainty_mode,
                )
                uncertainty_weight_mean = joint_weights.mean()
                uncertainty_valid_ratio = (joint_weights > 0).float().mean()
            else:
                joint_weights = joint_conf.to(image.device, image.dtype)
                uncertainty_weight_mean = torch.tensor(0.0, device=image.device, dtype=image.dtype)
                uncertainty_valid_ratio = torch.tensor(0.0, device=image.device, dtype=image.dtype)

            if hard_joint_adaptive_weighting and iteration >= hard_joint_warmup:
                adaptive_joint_weights, joint_hardness_ema, adaptive_joint_hardness, adaptive_joint_hard_ratio = compute_hard_joint_adaptive_weights(
                    epi_distances.to(image.device, image.dtype),
                    conf_main.to(image.device, image.dtype),
                    conf_ref.to(image.device, image.dtype),
                    joint_hardness_ema,
                    topk=hard_joint_topk,
                    boost=hard_joint_boost,
                    momentum=hard_joint_momentum,
                    conf_penalty_scale=hard_joint_conf_penalty_scale,
                )
                joint_weights = joint_weights * adaptive_joint_weights
                adaptive_joint_weight_mean = adaptive_joint_weights.mean()

            if need_epi_distances:
                epi_weight_sum = torch.clamp(joint_weights.sum(), min=1e-6)
                loss_epi = (epi_distances * joint_weights).sum() / epi_weight_sum
            else:
                loss_epi = cross_modal_epipolar_loss(
                    pts2d_proj_main,
                    pts2d_gt_ref,
                    F_main_to_ref,
                    joint_conf=joint_conf,
                )
            current_lambda_epi = get_linear_warmup_lambda(
                iteration,
                lambda_epi,
                warmup_start=epipolar_warmup_start,
                warmup_end=epipolar_warmup_end,
            )

            current_lambda_triang = get_linear_warmup_lambda(
                iteration,
                lambda_triang,
                warmup_start=triangulation_warmup_start,
                warmup_end=triangulation_warmup_end,
            )
            if current_lambda_triang > 0:
                triangulated_xyz, triang_joint_weights, triang_view_conf, triang_entropy, triang_reproj, triang_valid_ratio = build_triangulation_pseudo_target(
                    viewpoint_stack,
                    poses_2d,
                    render,
                    gaussians,
                    pipe,
                    bg,
                    model,
                    viewpoint_cam_main,
                    image,
                    gaussians.get_xyz,
                    temperature=triangulation_temperature,
                    min_view_weight=triangulation_min_view_weight,
                    min_views=triangulation_min_views,
                    reproj_error_threshold=triangulation_reproj_error,
                )
                triangulation_error = F.smooth_l1_loss(
                    gaussians.get_xyz,
                    triangulated_xyz,
                    reduction="none",
                ).mean(dim=1)
                triang_joint_weights = triang_joint_weights * triang_joint_prior
                triang_weight_sum = torch.clamp(triang_joint_weights.sum(), min=1e-6)
                loss_triang = (triangulation_error * triang_joint_weights).sum() / triang_weight_sum
                endpoint_weight_ratio = (
                    triang_joint_weights[triang_joint_prior > 1.0].sum() / triang_weight_sum
                    if torch.any(triang_joint_prior > 1.0)
                    else torch.tensor(0.0, device=gaussians.get_xyz.device, dtype=gaussians.get_xyz.dtype)
                )
            else:
                loss_triang = torch.tensor(0.0, device=gaussians.get_xyz.device, dtype=gaussians.get_xyz.dtype)
                triang_view_conf = torch.tensor(0.0, device=gaussians.get_xyz.device, dtype=gaussians.get_xyz.dtype)
                triang_entropy = torch.tensor(0.0, device=gaussians.get_xyz.device, dtype=gaussians.get_xyz.dtype)
                triang_reproj = torch.tensor(0.0, device=gaussians.get_xyz.device, dtype=gaussians.get_xyz.dtype)
                triang_valid_ratio = torch.tensor(0.0, device=gaussians.get_xyz.device, dtype=gaussians.get_xyz.dtype)
                endpoint_weight_ratio = torch.tensor(0.0, device=gaussians.get_xyz.device, dtype=gaussians.get_xyz.dtype)

            # ======= 5. 软边界对称性先验 =======
            loss_sym = compute_soft_margin_symmetry_loss(
                gaussians.get_xyz,
                scene_type=dataset.data_root,
                margin=symmetry_margin,
            )
            current_lambda_sym = get_symmetry_lambda(iteration, total_iterations, target_lambda_sym)

            # 解剖学比例先验：限制上/下肢比例，iteration>ratio_warmup 后才启用
            if iteration <= ratio_warmup:
                loss_ratio = torch.tensor(0.0, device=gaussians.get_xyz.device, dtype=gaussians.get_xyz.dtype)
            else:
                loss_ratio = soft_anatomical_ratio_loss(
                    gaussians.get_xyz,
                    scene_type=dataset.data_root,
                    ratio_margin=ratio_margin,
                )
            lambda_ratio_curr = lambda_ratio if iteration > ratio_warmup else 0.0
            if freeze_aux_params:
                loss_iso = torch.tensor(0.0, device=gaussians.get_xyz.device, dtype=gaussians.get_xyz.dtype)
                loss_scale_min = torch.tensor(0.0, device=gaussians.get_xyz.device, dtype=gaussians.get_xyz.dtype)
                loss_opacity = torch.tensor(0.0, device=gaussians.get_xyz.device, dtype=gaussians.get_xyz.dtype)
            else:
                scaling_var = torch.var(gaussians._scaling, dim=1, unbiased=False)
                loss_iso = scaling_var.mean()
                loss_scale_min = torch.exp(gaussians._scaling).mean()
                loss_opacity = (1.0 - torch.sigmoid(gaussians._opacity)).mean()
            lambda_iso = 0.05
            lambda_scale_min = 0.01
            lambda_opacity = 0.01
            current_lambda_ema = get_linear_warmup_lambda(
                iteration,
                lambda_ema if use_ema_teacher else 0.0,
                warmup_start=ema_warmup_start,
                warmup_end=ema_warmup_end,
            )
            if current_lambda_ema > 0:
                loss_ema = compute_ema_pose_loss(
                    gaussians.get_xyz,
                    ema_xyz,
                    use_root_relative=ema_use_root_relative,
                    root_weight=ema_root_weight,
                )
            else:
                loss_ema = torch.tensor(0.0, device=gaussians.get_xyz.device, dtype=gaussians.get_xyz.dtype)
            ema_teacher_drift = torch.norm(
                (gaussians.get_xyz.detach() - ema_xyz.detach()),
                dim=1,
            ).mean()
            current_lambda_temporal = get_linear_warmup_lambda(
                iteration,
                lambda_temporal if use_temporal_sequence_prior and temporal_target_xyz is not None else 0.0,
                warmup_start=temporal_warmup_start,
                warmup_end=temporal_warmup_end,
            )
            temporal_gate_weight = torch.tensor(1.0, device=gaussians.get_xyz.device, dtype=gaussians.get_xyz.dtype)
            temporal_gate_ratio = torch.tensor(0.0, device=gaussians.get_xyz.device, dtype=gaussians.get_xyz.dtype)
            temporal_uncertainty_score = torch.tensor(0.0, device=gaussians.get_xyz.device, dtype=gaussians.get_xyz.dtype)
            if need_temporal_gate:
                temporal_gate_weight, temporal_uncertainty_score, temporal_gate_ratio = build_temporal_uncertainty_gate(
                    conf_main.to(image.device, image.dtype),
                    conf_ref.to(image.device, image.dtype),
                    epi_distances.to(image.device, image.dtype),
                    topk=temporal_gate_topk,
                    conf_scale=temporal_gate_conf_scale,
                    epi_scale=temporal_gate_epi_scale,
                    min_conf=temporal_gate_min_conf,
                )
            if current_lambda_temporal > 0:
                loss_temporal = compute_temporal_sequence_loss(
                    gaussians.get_xyz,
                    temporal_target_xyz,
                    use_root_relative=temporal_use_root_relative,
                    root_weight=temporal_root_weight,
                    joint_weights=temporal_gate_weight if need_temporal_gate else None,
                )
                temporal_target_drift = torch.norm(
                    gaussians.get_xyz.detach() - temporal_target_xyz.detach(),
                    dim=1,
                ).mean()
            else:
                loss_temporal = torch.tensor(0.0, device=gaussians.get_xyz.device, dtype=gaussians.get_xyz.dtype)
                temporal_target_drift = torch.tensor(0.0, device=gaussians.get_xyz.device, dtype=gaussians.get_xyz.dtype)
            current_lambda_pose_manifold = get_linear_warmup_lambda(
                iteration,
                lambda_pose_manifold if use_pose_manifold_prior and pose_bank is not None else 0.0,
                warmup_start=pose_manifold_warmup_start,
                warmup_end=pose_manifold_warmup_end,
            )
            if current_lambda_pose_manifold > 0:
                loss_pose_manifold, pose_manifold_distance, pose_manifold_neighbors = compute_pose_manifold_loss(
                    gaussians.get_xyz,
                    pose_bank.to(gaussians.get_xyz.device, gaussians.get_xyz.dtype),
                    scene_name=scene_name if pose_manifold_exclude_self else None,
                    pose_bank_scene_keys=pose_bank_scene_keys if pose_manifold_exclude_self else None,
                    use_root_relative=pose_bank_root_relative,
                    topk=pose_manifold_topk,
                )
            else:
                loss_pose_manifold = torch.tensor(0.0, device=gaussians.get_xyz.device, dtype=gaussians.get_xyz.dtype)
                pose_manifold_distance = torch.tensor(0.0, device=gaussians.get_xyz.device, dtype=gaussians.get_xyz.dtype)
                pose_manifold_neighbors = 0

            # ======= 6. 总 Loss 聚合 =======
            loss = (
                l2_loss
                + loss_consistency
                + current_lambda_epi * loss_epi
                + current_lambda_triang * loss_triang
                + current_lambda_sym * loss_sym
                + lambda_ratio_curr * loss_ratio
                + current_lambda_ema * loss_ema
                + current_lambda_temporal * loss_temporal
                + current_lambda_pose_manifold * loss_pose_manifold
                + lambda_iso * loss_iso
                + lambda_scale_min * loss_scale_min
                + lambda_opacity * loss_opacity
            )

            # 记录到 Tensorboard 中监控
            if iteration % training.accumulation_steps == 0 and tb_writer:
                tb_writer.add_scalar('train_loss/epipolar_loss', loss_epi.item(), iteration)
                tb_writer.add_scalar('train_loss/lambda_epipolar', current_lambda_epi, iteration)
                tb_writer.add_scalar('train_loss/uncertainty_weight_mean', uncertainty_weight_mean.item(), iteration)
                tb_writer.add_scalar('train_loss/uncertainty_valid_ratio', uncertainty_valid_ratio.item(), iteration)
                tb_writer.add_scalar('train_loss/hard_joint_weight_mean', adaptive_joint_weight_mean.item(), iteration)
                tb_writer.add_scalar('train_loss/hard_joint_hard_ratio', adaptive_joint_hard_ratio.item(), iteration)
                tb_writer.add_scalar('train_loss/hard_joint_hardness', adaptive_joint_hardness.item(), iteration)
                tb_writer.add_scalar('train_loss/reference_score', selected_ref_score.item(), iteration)
                tb_writer.add_scalar('train_loss/reference_score_conf', reference_score_conf.item(), iteration)
                tb_writer.add_scalar('train_loss/reference_score_baseline', reference_score_baseline.item(), iteration)
                tb_writer.add_scalar('train_loss/reference_score_reproj', reference_score_reproj.item(), iteration)
                tb_writer.add_scalar('train_loss/epipolar_fusion_prior', fusion_prior_strength.item(), iteration)
                tb_writer.add_scalar('train_loss/epipolar_fusion_delta', fusion_delta.item(), iteration)
                tb_writer.add_scalar('train_loss/epipolar_fusion_valid_ratio', fusion_valid_ratio.item(), iteration)
                tb_writer.add_scalar('train_loss/epipolar_fusion_gate', fusion_gate.item(), iteration)
                tb_writer.add_scalar('train_loss/l2_loss', l2_loss.item(), iteration)
                tb_writer.add_scalar('train_loss/triangulation_loss', loss_triang.item(), iteration)
                tb_writer.add_scalar('train_loss/lambda_triangulation', current_lambda_triang, iteration)
                tb_writer.add_scalar('train_loss/triangulation_confidence', triang_view_conf.item(), iteration)
                tb_writer.add_scalar('train_loss/triangulation_entropy', triang_entropy.item(), iteration)
                tb_writer.add_scalar('train_loss/triangulation_reprojection', triang_reproj.item(), iteration)
                tb_writer.add_scalar('train_loss/triangulation_valid_ratio', triang_valid_ratio.item(), iteration)
                tb_writer.add_scalar('train_loss/triangulation_endpoint_weight_ratio', endpoint_weight_ratio.item(), iteration)
                tb_writer.add_scalar('train_loss/symmetry_loss', loss_sym.item(), iteration)
                tb_writer.add_scalar('train_loss/lambda_symmetry', current_lambda_sym, iteration)
                tb_writer.add_scalar('train_loss/ema_loss', loss_ema.item(), iteration)
                tb_writer.add_scalar('train_loss/lambda_ema', current_lambda_ema, iteration)
                tb_writer.add_scalar('train_loss/ema_teacher_drift', ema_teacher_drift.item(), iteration)
                tb_writer.add_scalar('train_loss/temporal_loss', loss_temporal.item(), iteration)
                tb_writer.add_scalar('train_loss/lambda_temporal', current_lambda_temporal, iteration)
                tb_writer.add_scalar('train_loss/temporal_target_drift', temporal_target_drift.item(), iteration)
                tb_writer.add_scalar('train_loss/temporal_gap_ratio', temporal_gap_ratio.item(), iteration)
                tb_writer.add_scalar('train_loss/temporal_has_velocity', temporal_has_velocity.item(), iteration)
                tb_writer.add_scalar('train_loss/temporal_prior_active', temporal_prior_active, iteration)
                tb_writer.add_scalar('train_loss/temporal_gate_ratio', temporal_gate_ratio.item(), iteration)
                tb_writer.add_scalar('train_loss/temporal_uncertainty_score', temporal_uncertainty_score.item(), iteration)
                tb_writer.add_scalar('train_loss/pose_manifold_loss', loss_pose_manifold.item(), iteration)
                tb_writer.add_scalar('train_loss/lambda_pose_manifold', current_lambda_pose_manifold, iteration)
                tb_writer.add_scalar('train_loss/pose_manifold_distance', pose_manifold_distance.item(), iteration)
                tb_writer.add_scalar('train_loss/pose_manifold_neighbors', pose_manifold_neighbors, iteration)
                tb_writer.add_scalar('train_loss/xyz_only_warm_start', float(freeze_aux_params), iteration)
                tb_writer.add_scalar('train_loss/ratio_loss', loss_ratio.item(), iteration)
                tb_writer.add_scalar('train_loss/iso_loss', loss_iso.item(), iteration)
                tb_writer.add_scalar('train_loss/scale_min_loss', loss_scale_min.item(), iteration)
                tb_writer.add_scalar('train_loss/opacity_loss', loss_opacity.item(), iteration)

            if early_stopping(loss.item()):
                stop = True

            accumulated_loss_total += loss.item()

            if freeze_aux_params:
                grads_xyz = torch.autograd.grad(loss, gaussians.get_xyz, create_graph=True, retain_graph=True)[0]
                grads_scaling = torch.zeros_like(gaussians._scaling)
                grads_rotation = torch.zeros_like(gaussians._rotation)
                grads_opacity = torch.zeros_like(gaussians._opacity)
            else:
                params = [gaussians.get_xyz, gaussians._scaling, gaussians._rotation, gaussians._opacity]
                grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
                grads_xyz = grads[0]
                grads_scaling = grads[1]
                grads_rotation = grads[2]
                grads_opacity = grads[3]

            # grad = torch.autograd.grad(loss, gaussians.get_xyz, create_graph=True, retain_graph=True)[0]
            if gaussians.get_xyz.grad is None:
                gaussians.get_xyz.grad = torch.zeros_like(gaussians.get_xyz)
                gaussians._scaling.grad = torch.zeros_like(gaussians._scaling)
                gaussians._rotation.grad = torch.zeros_like(gaussians._rotation)
                gaussians._opacity.grad = torch.zeros_like(gaussians._opacity)

            accumulated_grads[idx_main, ...] += grads_xyz

            gaussians._scaling.grad = grads_scaling
            gaussians._rotation.grad = grads_rotation
            gaussians._opacity.grad = grads_opacity

            iter_end.record()

            if iteration % training.accumulation_steps == 0 or stop:

                with torch.no_grad():
                    # error computation
                    if "h36m" in dataset.data_root or "dataset_fgy" in dataset.data_root or "occlusion-person" in dataset.data_root:
                        subject, activity, step = scene.scene_name.split("_")
                    elif "panoptic" in dataset.data_root:
                        subject = scene.scene_name.split("_")[0]
                        step = scene.scene_name.split("_")[-1]
                        activity = scene.scene_name.split("_")[1] + "_" + scene.scene_name.split("_")[2]
                    
                    if subject == 'S9' and activity in ['SittingDown 1', 'Waiting 1', 'Greeting']:
                        error = torch.tensor([0.0], device="cuda")
                    else:
                        pred = gaussians.get_xyz.clone()
                        gt = pose_3d_gt
                        error = torch.norm(pred - gt, dim=1)
                        # log.info("Opt - Absolute error: " + str(error))
                        errors_all.append(error)

                    pred_rel = pred - pred[0, ...]
                    gt_rel = gt - gt[0, ...]
                    error_rel = torch.norm(pred_rel - gt_rel, dim=1)
                    errors_rel_all.append(error_rel)

                    torch.cuda.synchronize()
                    training_report(
                        tb_writer, iteration,
                        accumulated_loss_total / training.accumulation_steps,  # averaged loss
                        iter_start.elapsed_time(iter_end),
                        scene, error, error_rel
                    )

                gradients = accumulated_grads
                gradients = gradients.to(gaussians.get_xyz.dtype)
                gradients = gradients.mean(dim=0)
                gaussians.get_xyz.grad = gradients

                with torch.no_grad():
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)
                    if use_ema_teacher:
                        ema_xyz.mul_(ema_decay).add_(gaussians.get_xyz.detach(), alpha=1.0 - ema_decay)
                    # === 新增：清空我们自定义的视角累加梯度池 ===
                    accumulated_grads.zero_()

            # Reset accumulated losses
            accumulated_loss_total = 0.0

            if iteration in debug.save_iterations or stop:
                print(f"Saving iteration {iteration} for scene {scene_name}")
                scene.save_h36m(iteration, scene_name)

            if stop:
                log.info(f"Stopping training for scene {scene_name} at iteration {iteration}")
                break

        # ======= 【测速与资源记录】 =======
        global_end.record()
        torch.cuda.synchronize()  

        total_time_seconds = global_start.elapsed_time(global_end) / 1000.0
        actual_iterations = iteration - first_iter + 1
        
        # 【新增：获取本次训练显存占用峰值】
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
        log.info(f"[{scene_name}] >>> 性能测试报告 <<<")
        log.info(f"[{scene_name}] 实际运行迭代次数: {actual_iterations} iters")
        log.info(f"[{scene_name}] 总耗时 (Wall-clock time): {total_time_seconds:.2f} 秒")
        log.info(f"[{scene_name}] 训练吞吐量 (Speed): {actual_iterations / max(total_time_seconds, 1e-5):.2f} it/s")
        log.info(f"[{scene_name}] 峰值显存占用 (Peak VRAM): {peak_memory_mb:.2f} MB")
        log.info("====================================")
        
        # 记得在重置下一段视频前，清空显存峰值记录
        torch.cuda.reset_peak_memory_stats()
        # ============================================

        # to render on all cameras and save images
        if debug.save_images:
            save_images(scene.getTrainCameras(), gaussians, pipe, model, output_dir, name="render")
        if occlusion_vis_record is not None:
            render_pkg = render(
                scene.getTrainCameras()[0],
                gaussians,
                pipe,
                torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
                use_trained_exp=model.train_test_exp,
                separate_sh=SPARSE_ADAM_AVAILABLE,
            )
            save_heatmap_image(render_pkg["render"], os.path.join(occlusion_vis_record, "prediction.jpg"))

        log.info("Absolute error: " + str(error))
        log.info("Relative error: " + str(error_rel))
        log.info("Mean absolute error: " + str(error.mean()))
        log.info("Mean relative error: " + str(error_rel.mean()))
        update_sequence_state(sequence_state, scene_name, gaussians.get_xyz.detach())

    print("Training completed.")


def prepare_output_and_logger(output_dir, enabled=True):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND and enabled:
        tb_writer = SummaryWriter(output_dir + "/tb")
    elif TENSORBOARD_FOUND:
        print("Tensorboard logging disabled by configuration")
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, loss, elapsed, scene : Scene, error, rel_error):
    torch.cuda.synchronize()
    if "h36m" in scene.scene_type or "dataset_fgy" in scene.scene_type or "occlusion-person" in scene.scene_type:
        subject, activity, step = scene.scene_name.split("_")
    elif "panoptic" in scene.scene_type:
        subject = scene.scene_name.split("_")[0]
        step = scene.scene_name.split("_")[-1]
        activity = scene.scene_name.split("_")[1] + "_" + scene.scene_name.split("_")[2]
    tb_string = f"Subject_{subject}_Activity_{activity}/Step_{step}"
    
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/total_loss', loss, iteration)
        tb_writer.add_scalar(tb_string + "/absolute_error", error.mean(), iteration)
        tb_writer.add_scalar(tb_string + "/relative_error", rel_error.mean(), iteration)

        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def save_images(train_cameras, gaussians, pipe, model, output_dir, name="image"):
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    render = render_functions[pipe.rendering]
    
    for i_camera in range(len(train_cameras)):
        viewpoint_cam = train_cameras[i_camera]
        render_pkg = render(viewpoint_cam, gaussians, pipe, torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"), use_trained_exp=model.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        im = torch.sum(image, dim=0)
        im = (im - torch.min(im)) / (torch.max(im) - torch.min(im))
        im = (im * 255).detach().cpu().numpy().astype(np.uint8)
        im = Image.fromarray(im)
        im.save(f"{output_dir}/images/{name}_{i_camera}.png")


def save_heatmaps(nviews, heatmaps_cameras, output_dir, name="heatmap"):

    os.makedirs(f"{output_dir}/heatmaps", exist_ok=True)

    for i_camera in range(nviews):
        heatmap = heatmaps_cameras[str(i_camera)]
        im = torch.sum(heatmap, dim=0)
        im = (im - torch.min(im)) / (torch.max(im) - torch.min(im))
        im = (im * 255).detach().cpu().numpy().astype(np.uint8)
        im = Image.fromarray(im)
        im.save(f"{output_dir}/heatmaps/{name}_{i_camera}.png")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):

    config = ConfigHandler(cfg)

    output_dir = config.hydra_out
    dataset = cfg.dataset
    train = cfg.training
    debug = cfg.debug
    model = cfg.model
    opt = cfg.optimization
    pipe = cfg.pipeline
    test_cfg = _cfg_get(cfg, "test", _cfg_get(cfg, "TEST", None))

    print(output_dir)

    log = logging.getLogger(__name__)

    if train.dropout:
        print("Dropping out some gt joints during training")

    initial_guess_path = os.path.join(dataset.data_root, "initial_guess", dataset.initial_guess)
    poses_2d_path = os.path.join(dataset.data_root, "2d_" + dataset.poses_2d)

    debug.save_iterations.append(opt.iterations)
    dataset_loader = DataLoader(dataset.data_root, initial_guess_path, poses_2d_path,
                                frame_step=dataset.frame_step, start_id=dataset.start_scene_id, 
                                end_id=dataset.end_scene_id, nviews=dataset.nviews,
                                camera_names=getattr(dataset, "camera_names", None),
                                filtered_nviews=getattr(dataset, "filtered_nviews", None))
    

    # Keep optimization randomness separate from the occlusion-mask seed.
    optimization_seed = int(getattr(train, "seed", 0))
    safe_state(train.quiet, seed=optimization_seed)
    log.info("Optimization RNG seed: %d", optimization_seed)
    print(f"Optimization RNG seed: {optimization_seed}")
    training(dataset, model, opt, pipe, debug, train, dataset_loader, output_dir, log, test_cfg=test_cfg)

if __name__ == "__main__":
    main()
