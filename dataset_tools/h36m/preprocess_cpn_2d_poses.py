import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_file",
    type=str,
    default="../../data/h36m/data_2d_h36m_cpn_ft_h36m_dbb.npz",
    help="Path to CPN predictions (.npz from VideoPose3D)."
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="../../data/h36m",
    help="Path to save the reorganized dataset."
)
args = parser.parse_args()

input_file = args.input_file
output_root = args.output_dir
output_2d = os.path.join(output_root, "2d_cpn")

os.makedirs(output_2d, exist_ok=True)

# ---------- 读取 CPN 2D 结果（关键修改部分） ----------
raw = np.load(input_file, allow_pickle=True)

# VideoPose3D 格式：npz 里有 'positions_2d' 和 'metadata'
if isinstance(raw, np.lib.npyio.NpzFile):
    if 'positions_2d' in raw.files:
        data_cpn = raw['positions_2d'].item()
    else:
        # 如果没有 positions_2d，就把唯一的那个 key 当作 positions_2d 使用
        if len(raw.files) == 1:
            data_cpn = raw[raw.files[0]].item()
        else:
            raise RuntimeError(
                f"Unsupported npz structure in {input_file}. "
                f"Keys found: {raw.files}. "
                "Please check which key contains the 2D pose dictionary."
            )
else:
    # .npy + pickle 的老格式
    data_cpn = raw.item()

# ---------- 读取 metadata（可选，只是打印一下） ----------
# 假设你按 VideoPose3D 的脚本把 metadata.npy 单独解出来：
#   <input_file_dir>/<base_name>/metadata.npy
input_dir = os.path.dirname(input_file)
base_name = os.path.splitext(os.path.basename(input_file))[0]  # data_2d_h36m_cpn_ft_h36m_dbb
metadata_path = os.path.join(input_dir, base_name, "metadata.npy")

if os.path.isfile(metadata_path):
    metadata = np.load(metadata_path, allow_pickle=True)
    print("Loaded metadata from:", metadata_path)
    print(metadata)
else:
    print("Warning: metadata file not found at", metadata_path)
    metadata = None

# ---------- 按 subject / activity / camera 切分并保存 ----------
for subject in ["S9", "S11"]:
    subject_path = os.path.join(output_2d, subject)
    os.makedirs(subject_path, exist_ok=True)

    # Process each activity (e.g., Directions, Directions 1, ...)
    for activity in sorted(data_cpn[subject].keys()):
        print("Processing:", subject, activity)
        activity_path = os.path.join(subject_path, activity)
        os.makedirs(activity_path, exist_ok=True)
        poses_2d = data_cpn[subject][activity]

        # 4 个主摄像机的 ID
        for i, cam_name in enumerate(["54138969", "55011271", "58860488", "60457274"]):
            output2d_path = os.path.join(output_2d, subject, activity, cam_name)
            os.makedirs(output2d_path, exist_ok=True)

            poses_cam = poses_2d[i]                  # [T, 17, 2] 或 [T, 34] 之类
            poses_cam = np.array(poses_cam).reshape(-1, 17, 2)

            # 每 64 帧采样一次（和你原来的逻辑一致）
            poses_cam_step = np.array(
                [poses_cam[j] for j in range(0, len(poses_cam), 64)]
            )

            np.savez(os.path.join(output2d_path, "poses.npz"), poses2d=poses_cam_step)