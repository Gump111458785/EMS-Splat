import torch


def get_camera_projection_matrix(cam):
    """
    Build the 3x4 pinhole projection matrix P = K [R | t] from the runtime camera.
    """
    device = cam.world_view_transform.device
    dtype = cam.world_view_transform.dtype
    world_to_camera = cam.world_view_transform.transpose(0, 1)[:3, :]
    intrinsics = torch.as_tensor(cam.K, device=device, dtype=dtype)
    return intrinsics @ world_to_camera

def get_fundamental_matrix(cam1, cam2):
    """
    计算从 cam1 到 cam2 的基础矩阵 F。
    要求满足： x2^T * F * x1 = 0
    """
    device = cam1.world_view_transform.device
    dtype = cam1.world_view_transform.dtype

    # 1. 提取最新的 World-to-Camera 外参
    W2C_1 = cam1.world_view_transform.transpose(0, 1)
    R1 = W2C_1[:3, :3]
    T1 = W2C_1[:3, 3]

    W2C_2 = cam2.world_view_transform.transpose(0, 1)
    R2 = W2C_2[:3, :3]
    T2 = W2C_2[:3, 3]

    # 2. 相对位姿
    R12 = R2 @ R1.transpose(0, 1)
    T12 = T2 - R12 @ T1

    # 3. 内参
    K1 = torch.as_tensor(cam1.K, device=device, dtype=dtype)
    K2 = torch.as_tensor(cam2.K, device=device, dtype=dtype)

    # 4. 本质矩阵
    zero = torch.zeros((), device=device, dtype=dtype)
    T12_x = torch.stack([
        torch.stack([zero, -T12[2], T12[1]]),
        torch.stack([T12[2], zero, -T12[0]]),
        torch.stack([-T12[1], T12[0], zero])
    ])
    E = T12_x @ R12

    # 5. 基础矩阵
    K2_invT = torch.inverse(K2).transpose(0, 1)
    K1_inv = torch.inverse(K1)
    return K2_invT @ E @ K1_inv

def project_xyz_to_pixels(xyz, cam):
    """
    将 3D 高斯球中心直接进行纯数学投影到 2D 像素坐标系 (无须光栅化渲染)
    """
    N = xyz.shape[0]
    # 齐次坐标
    xyz_h = torch.cat([xyz, torch.ones(N, 1, device=xyz.device)], dim=1)

    # Clip space 投影
    clip_space = xyz_h @ cam.full_proj_transform

    # NDC 空间 [-1, 1]
    ndc_space = clip_space[:, :2] / (clip_space[:, 3:4] + 1e-8)

    # 像素坐标系
    pixel_x = (ndc_space[:, 0] + 1.0) * cam.image_width / 2.0
    pixel_y = (ndc_space[:, 1] + 1.0) * cam.image_height / 2.0

    return torch.stack([pixel_x, pixel_y], dim=1)

def cross_modal_epipolar_distances(pts2d_render_cam1, pts2d_gt_cam2, F12):
    """
    返回逐关节的极线距离 d(x2, l2)，用于后续鲁棒权重/可视化。
    """
    device = pts2d_render_cam1.device
    dtype = pts2d_render_cam1.dtype
    N = pts2d_gt_cam2.shape[0]

    x1_h = torch.cat([pts2d_render_cam1, torch.ones(N, 1, device=device, dtype=dtype)], dim=1)
    l2 = (F12 @ x1_h.transpose(0, 1)).transpose(0, 1)

    a = l2[:, 0]
    b = l2[:, 1]
    c = l2[:, 2]
    u = pts2d_gt_cam2[:, 0]
    v = pts2d_gt_cam2[:, 1]

    numerator = torch.abs(a * u + b * v + c)
    denominator = torch.sqrt(a ** 2 + b ** 2 + 1e-8)
    return numerator / denominator


def cross_modal_epipolar_loss(pts2d_render_cam1, pts2d_gt_cam2, F12, joint_conf=None):
    """
    计算极线距离损失。joint_conf 可选，用于加权平均。
    """
    distances = cross_modal_epipolar_distances(pts2d_render_cam1, pts2d_gt_cam2, F12)
    if joint_conf is not None:
        weights = joint_conf.to(distances.device)
        weights = weights.reshape(-1)
        weighted = distances * weights
        normalizer = torch.clamp(weights.sum(), min=1e-6)
        return weighted.sum() / normalizer
    return distances.mean()


def weighted_triangulate_points(projection_matrices, points_2d, weights=None, eps=1e-6):
    """
    Weighted DLT triangulation.
    projection_matrices: (V, 3, 4)
    points_2d: (V, J, 2)
    weights: (V, J), optional
    returns: (J, 3)
    """
    num_views, num_joints = points_2d.shape[:2]
    if weights is None:
        weights = torch.ones(
            (num_views, num_joints),
            device=points_2d.device,
            dtype=points_2d.dtype,
        )

    triangulated = []
    safe_weights = torch.clamp(weights, min=eps).sqrt()

    for joint_idx in range(num_joints):
        rows = []
        for view_idx in range(num_views):
            proj = projection_matrices[view_idx]
            x_coord = points_2d[view_idx, joint_idx, 0]
            y_coord = points_2d[view_idx, joint_idx, 1]
            weight = safe_weights[view_idx, joint_idx]
            rows.append(weight * (x_coord * proj[2] - proj[0]))
            rows.append(weight * (y_coord * proj[2] - proj[1]))

        design = torch.stack(rows, dim=0)
        _, _, vh = torch.linalg.svd(design, full_matrices=False)
        hom_point = vh[-1]
        denom = hom_point[3:4]
        safe_denom = torch.where(
            denom.abs() < eps,
            torch.full_like(denom, eps),
            denom,
        )
        triangulated.append(hom_point[:3] / safe_denom)

    return torch.stack(triangulated, dim=0)


def reproject_points(projection_matrices, xyz, eps=1e-6):
    """
    Reproject 3D joints with batched projection matrices.
    projection_matrices: (V, 3, 4)
    xyz: (J, 3)
    returns: (V, J, 2)
    """
    ones = torch.ones((xyz.shape[0], 1), device=xyz.device, dtype=xyz.dtype)
    xyz_h = torch.cat([xyz, ones], dim=1)
    projected = torch.einsum("vab,jb->vja", projection_matrices, xyz_h)
    safe_depth = torch.where(
        projected[..., 2:3].abs() < eps,
        torch.full_like(projected[..., 2:3], eps),
        projected[..., 2:3],
    )
    return projected[..., :2] / safe_depth
