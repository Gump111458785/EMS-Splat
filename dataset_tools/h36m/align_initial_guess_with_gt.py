import os
import numpy as np

DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/h36m"))

GT_ROOT = os.path.join(DATA_ROOT, "3d_gt")
INIT_ROOT = os.path.join(DATA_ROOT, "initial_guess", "triang_cpn")

SUBJECTS = ["S9", "S11"]

print("GT_ROOT   :", GT_ROOT)
print("INIT_ROOT :", INIT_ROOT)

for subject in SUBJECTS:
    subj_gt_root = os.path.join(GT_ROOT, subject)
    subj_init_root = os.path.join(INIT_ROOT, subject)

    if not os.path.isdir(subj_gt_root):
        print("[WARN] no 3d_gt for", subject, "at", subj_gt_root)
        continue
    if not os.path.isdir(subj_init_root):
        print("[WARN] no initial_guess for", subject, "at", subj_init_root)
        continue

    for action in sorted(os.listdir(subj_gt_root)):
        gt_file = os.path.join(subj_gt_root, action, "poses.npz")
        init_file = os.path.join(subj_init_root, action, "poses.npz")

        if not os.path.isfile(gt_file):
            print("[WARN] missing GT:", gt_file)
            continue
        if not os.path.isfile(init_file):
            print("[WARN] missing INIT:", init_file)
            continue

        gt = np.load(gt_file)
        if "poses" in gt:
            poses_gt = gt["poses"]
        else:
            # 保险起见
            key = list(gt.keys())[0]
            poses_gt = gt[key]

        init = np.load(init_file)
        if "poses" in init:
            poses_init = init["poses"]
        else:
            key = list(init.keys())[0]
            poses_init = init[key]

        T_gt = poses_gt.shape[0]
        T_init = poses_init.shape[0]
        T_new = min(T_gt, T_init)

        if T_new < T_init:
            print(f"[CUT] {subject}/{action}: GT={T_gt}, INIT={T_init} -> {T_new}")
            poses_init_new = poses_init[:T_new]
            # 覆盖原文件
            np.savez_compressed(init_file, poses=poses_init_new)
        else:
            print(f"[OK ] {subject}/{action}: GT={T_gt}, INIT={T_init}")

print("Done aligning initial_guess with GT.")