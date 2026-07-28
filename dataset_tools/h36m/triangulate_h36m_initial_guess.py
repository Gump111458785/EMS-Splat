import os
import json
import argparse
import numpy as np

SUBJECTS = ["S9", "S11"]
CAMERAS = ["54138969", "55011271", "58860488", "60457274"]


def load_camera_params(cam_json_path, subject):
    """
    从 camera-parameters.json 里取出某个 subject 的 4 个相机投影矩阵 P = K [R | t].

    JSON 结构示例（你贴的这一份）：
    {
      "intrinsics": {
        "54138969": { "calibration_matrix": [...], "distortion": [...] },
        ...
      },
      "extrinsics": {
        "S1": {
          "54138969": { "R": [...], "t": [...] },
          ...
        },
        "S9": { ... },
        "S11": { ... }
      }
    }
    """
    with open(cam_json_path, "r") as f:
        meta = json.load(f)

    intr = meta["intrinsics"]
    extr_all = meta["extrinsics"]

    if subject not in extr_all:
        raise KeyError(
            f"Subject {subject} not found in 'extrinsics' of {cam_json_path}. "
            f"Available subjects: {list(extr_all.keys())}"
        )

    extr = extr_all[subject]
    proj_mats = {}

    for cam_id in CAMERAS:
        if cam_id not in intr:
            print(f"[WARN] Camera {cam_id} not found in 'intrinsics' keys={list(intr.keys())}")
            continue
        if cam_id not in extr:
            print(f"[WARN] Camera {cam_id} not found for subject {subject} in 'extrinsics' keys={list(extr.keys())}")
            continue

        K = np.array(intr[cam_id]["calibration_matrix"], dtype=np.float64)  # 3x3
        R = np.array(extr[cam_id]["R"], dtype=np.float64)                   # 3x3
        t = np.array(extr[cam_id]["t"], dtype=np.float64).reshape(3, 1)     # 3x1

        Rt = np.concatenate([R, t], axis=1)   # 3x4
        P = K @ Rt                            # 3x4
        proj_mats[cam_id] = P

    return proj_mats


def triangulate_point(pts2d, proj_mats):
    """多视角线性三角测量，一个关节在多相机下的 2D 点 -> 3D 点。

    pts2d: list of (u, v)
    proj_mats: list of 3x4 投影矩阵
    """
    A = []
    for (u, v), P in zip(pts2d, proj_mats):
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    A = np.stack(A, axis=0)  # (2M, 4)

    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[-1]  # (4,)
    if np.abs(X_h[3]) < 1e-8:
        return np.zeros(3, dtype=np.float32)
    X = X_h[:3] / X_h[3]
    return X.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default="../../data/h36m",
        help="SkelSplat data/h36m 根目录",
    )
    parser.add_argument(
        "--poses2d_dir",
        type=str,
        default="2d_cpn",
        help="2D 检测结果所在目录（相对于 data_root）",
    )
    parser.add_argument(
        "--camera_json",
        type=str,
        default="3d_gt/cameras/camera-parameters.json",
        help="Human3.6M 相机参数 JSON（相对于 data_root）",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="initial_guess/triang_cpn",
        help="输出初始 3D 结果的目录（相对于 data_root）",
    )
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)
    poses2d_root = os.path.join(data_root, args.poses2d_dir)
    cam_json_path = os.path.join(data_root, args.camera_json)
    out_root = os.path.join(data_root, args.out_dir)

    print("data_root:", data_root)
    print("poses2d_root:", poses2d_root)
    print("camera_json:", cam_json_path)
    print("out_root:", out_root)

    for subject in SUBJECTS:
        print(f"=== Subject {subject} ===")
        subj_2d_root = os.path.join(poses2d_root, subject)
        if not os.path.isdir(subj_2d_root):
            print("  [WARN] 2D dir not found, skip:", subj_2d_root)
            continue

        # 加载该 subject 的 4 个相机投影矩阵
        proj_mats_dict = load_camera_params(cam_json_path, subject)

        # 遍历所有动作（Directions, Directions 1, Walking, ...）
        for action in sorted(os.listdir(subj_2d_root)):
            action_2d_dir = os.path.join(subj_2d_root, action)
            if not os.path.isdir(action_2d_dir):
                continue

            print(f"  Action: {action}")

            # 对 4 个相机分别读取 2D 关键点
            per_cam_poses2d = {}
            num_frames = None
            num_joints = None

            for cam_id in CAMERAS:
                cam_dir = os.path.join(action_2d_dir, cam_id)
                pose_file = os.path.join(cam_dir, "poses.npz")
                if not os.path.isfile(pose_file):
                    print(f"    [WARN] missing 2D for cam {cam_id}:", pose_file)
                    continue
                data = np.load(pose_file)
                if "poses2d" in data:
                    poses2d = data["poses2d"]  # (T, J, 2)
                elif "poses" in data:
                    poses2d = data["poses"]
                else:
                    raise ValueError(f"Unknown key in {pose_file}: {list(data.keys())}")

                if num_frames is None:
                    num_frames = poses2d.shape[0]
                    num_joints = poses2d.shape[1]
                else:
                    num_frames = min(num_frames, poses2d.shape[0])
                    num_joints = min(num_joints, poses2d.shape[1])

                per_cam_poses2d[cam_id] = poses2d

            if num_frames is None:
                print("    [WARN] no 2D data for this action, skip.")
                continue

            print(f"    Use {len(per_cam_poses2d)} cameras, {num_frames} frames, {num_joints} joints")

            poses3d = np.zeros((num_frames, num_joints, 3), dtype=np.float32)
            used_cams = [cid for cid in CAMERAS if cid in per_cam_poses2d and cid in proj_mats_dict]

            if len(used_cams) < 2:
                print("    [WARN] less than 2 cameras with 2D+P, skip triangulation.")
                continue

            proj_mats = [proj_mats_dict[cid] for cid in used_cams]

            for t in range(num_frames):
                for j in range(num_joints):
                    pts2d = []
                    for cid in used_cams:
                        p2d = per_cam_poses2d[cid][t, j]  # (2,)
                        pts2d.append((float(p2d[0]), float(p2d[1])))

                    poses3d[t, j] = triangulate_point(pts2d, proj_mats)

            # 保存结果：和 3d_gt 一样用 key="poses"
            out_dir = os.path.join(out_root, subject, action)
            os.makedirs(out_dir, exist_ok=True)
            out_file = os.path.join(out_dir, "poses.npz")
            np.savez_compressed(out_file, poses=poses3d)
            print("    Saved:", out_file)


if __name__ == "__main__":
    main()