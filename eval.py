
import os
import numpy as np
import open3d as o3d
import hydra
from omegaconf import DictConfig
from arguments.config_handler import ConfigHandler, TriangulationConfigHandler


def compute_pa_mpjpe(pred_coords, gt_coords):
    """Protocol #2 PA-MPJPE with per-frame similarity alignment."""
    aligned = []
    for pred, gt in zip(pred_coords, gt_coords):
        pred_mean = pred.mean(axis=0, keepdims=True)
        gt_mean = gt.mean(axis=0, keepdims=True)
        pred_centered = pred - pred_mean
        gt_centered = gt - gt_mean
        pred_norm = np.sqrt((pred_centered ** 2).sum())
        gt_norm = np.sqrt((gt_centered ** 2).sum())
        if pred_norm < 1e-8 or gt_norm < 1e-8:
            aligned.append(pred.copy())
            continue
        pred_centered /= pred_norm
        gt_centered /= gt_norm
        h = pred_centered.T @ gt_centered
        u, s, vt = np.linalg.svd(h)
        r = vt.T @ u.T
        if np.linalg.det(r) < 0:
            vt[-1, :] *= -1
            s[-1] *= -1
            r = vt.T @ u.T
        scale = (s.sum() * gt_norm) / pred_norm
        aligned.append(scale * (pred - pred_mean) @ r + gt_mean)
    aligned = np.asarray(aligned)
    return np.linalg.norm(aligned - gt_coords, axis=-1).mean()


def compute_pck(pred_coords, gt_coords, threshold=150.0):
    errors = np.linalg.norm(pred_coords - gt_coords, axis=-1)
    return np.mean(errors < threshold) * 100.0


def compute_pck_valid(pred_coords, gt_coords, valid_mask, threshold=150.0):
    errors = np.linalg.norm(pred_coords - gt_coords, axis=-1)
    errors = errors[np.asarray(valid_mask, dtype=bool)]
    return float(np.mean(errors < threshold) * 100.0) if errors.size else float("nan")


def compute_pa_mpjpe_valid(pred_coords, gt_coords, valid_mask):
    valid_mask = np.asarray(valid_mask, dtype=bool)
    frame_mask = valid_mask.all(axis=1)
    if not frame_mask.any():
        return float("nan")
    return compute_pa_mpjpe(pred_coords[frame_mask], gt_coords[frame_mask])


def resolve_eval_range(start_id, end_id, total_len):
    start_id = max(int(start_id), 0)
    if end_id is None or int(end_id) < 0:
        end_id = total_len
    else:
        end_id = min(int(end_id), total_len)
    return start_id, end_id


def panoptic_gt_valid_mask(gt):
    valid = np.isfinite(gt).all(axis=-1)
    valid &= np.linalg.norm(gt, axis=-1) > 1e-6
    return valid


def safe_nanmean(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("nan")


def parse_joint_ids(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value or value.lower() in ("all", "none"):
            return None
        value = value.strip("[]")
        return [int(token.strip()) for token in value.split(",") if token.strip()]
    return [int(joint_id) for joint_id in list(value)]


def align_pred_cpn(pred_coords, gt_coords, image_relpaths):
    # Align the predicted 3D poses with the ground truth
    start_poses = 0
    count = 0
    for i, path in enumerate(image_relpaths):
        if 'S11' in path and 'Directions.' in path:
            start_poses = i
            count += 1
    insert_poses = np.zeros((count, 17, 3))
    new_pred_coords = np.vstack((pred_coords[:start_poses], insert_poses, pred_coords[start_poses:]))
    return new_pred_coords

def get_pred_coords_h36m(ply_dir, sorted_entries, absolute=False, cpn=False):

    activities = []
    pred_coords = []
    for entry in sorted_entries:
        subject, activity, frame = entry
        if absolute:
            if subject == 'S9' and activity in ['SittingDown 1', 'Waiting 1', 'Greeting']:
                continue
        ply_file = f'{ply_dir}/{subject}_{activity}_{frame}'
        pcd = o3d.io.read_point_cloud(ply_file)
        pred_coords.append(np.asarray(pcd.points))
        activities.append(activity.split(" ")[0])

    pred_coords = np.array(pred_coords)
    activities = np.array(activities)
    # print(pred_coords.shape)
    return pred_coords, activities

def get_pred_coords(ply_dir, sorted_entries, absolute=False):

    pred_coords = []
    for entry in sorted_entries:
        subject, activity, frame = entry
        ply_file = f'{ply_dir}/{subject}_{activity}_{frame}'
        pcd = o3d.io.read_point_cloud(ply_file)
        pred_coords.append(np.asarray(pcd.points))

    pred_coords = np.array(pred_coords)
    # print(pred_coords.shape)
    return pred_coords


def get_gt_poses_h36m(gt_path, absolute=False, cpn=False, frame_step=64):
    gt_poses = []
    for subject in sorted(os.listdir(gt_path)):
        if not subject.startswith('S'):
            continue
        for activity in sorted(os.listdir(f'{gt_path}/{subject}')):
            if absolute:
                if subject == 'S9' and activity in ['SittingDown 1', 'Waiting 1', 'Greeting']:
                    continue
            if cpn:
                if subject == 'S11' and activity == 'Directions':
                    continue
            gt_3d = np.load(f'{gt_path}/{subject}/{activity}/poses.npz')['poses']
            gt_sampled = gt_3d[::frame_step]
            gt_poses.append(gt_sampled)
    gt_poses = np.concatenate(gt_poses, axis=0)
    return gt_poses


def get_gt_poses(gt_path, absolute=False, dataset="panoptic", frame_step=1, nviews=4):
    
    gt_poses = []
    for subject in sorted(os.listdir(gt_path)):
        if not subject.startswith('S'):
            continue
        for activity in sorted(os.listdir(f'{gt_path}/{subject}')):
            if dataset == "panoptic":
                gt_3d = np.load(f'{gt_path}/{subject}/{activity}/poses_filtered_{nviews}.npz', allow_pickle=True)['poses']
            else:
                gt_3d = np.load(f'{gt_path}/{subject}/{activity}/poses.npz', allow_pickle=True)['poses3d']
            gt_sampled = gt_3d[::frame_step]
            gt_poses.append(gt_sampled)

    gt_poses = np.concatenate(gt_poses, axis=0)
    return gt_poses

def evaluate(
    gt_path,
    output_path,
    iterations,
    start_id,
    end_id,
    nviews=4,
    cpn=False,
    eval_joint_ids=None,
    frame_step=1,
    pck_threshold=150.0,
    return_results=False,
):

    results = []
    for it in iterations:
        print(f"Results for {it} iterations \n")
        # load the predicted 3D poses
        ply_dir = f'{output_path}/point_cloud/iteration_{it}'
        entries = [entry for entry in os.listdir(ply_dir) if entry.endswith(".ply")]
        name_parts = [entry.split('_') for entry in entries]
        if "panoptic" in gt_path:
            name_parts = [[entry.split("_")[0], entry.split("_")[1] + "_" + entry.split("_")[2], entry.split("_")[-1]] for entry in entries]
            dataset = "panoptic"
        if "occlusion-person" in gt_path:
            name_parts = [[entry.split("_")[0], entry.split("_")[1], entry.split("_")[-1]] for entry in entries]
            dataset = "occlusion-person"

        sorted_entries = sorted(name_parts)


        if "h36m" in gt_path:

            ordered_activities = (
                'Directions Discussion Eating Greeting Phoning Posing Purchases ' +
                'Sitting SittingDown Smoking Photo Waiting Walking WalkDog WalkTogether').split()

            # Absolute MPJPE
            absolute = True
            gt_coords = get_gt_poses_h36m(gt_path, absolute, cpn, frame_step=frame_step)
            pred_coords, activities = get_pred_coords_h36m(ply_dir, sorted_entries, absolute, cpn)
            start_eval, end_eval = resolve_eval_range(start_id, end_id, pred_coords.shape[0])

            print("Evaluating scenes from", start_eval, "to", end_eval)
            abs_error = np.linalg.norm(gt_coords[start_eval:end_eval, ...] - pred_coords[start_eval:end_eval, ...], axis=-1)
            abs_error_mean = safe_nanmean(abs_error)
            print("Absolute MPJPE: ", np.round(abs_error_mean, 2))
            activities_errors = [np.mean(abs_error[a == activities]) for a in ordered_activities]
            print(np.round(activities_errors, 2))

            # Relative MPJPE
            absolute = False
            gt_coords = get_gt_poses_h36m(gt_path, absolute, cpn, frame_step=frame_step)
            pred_coords, activities = get_pred_coords_h36m(ply_dir, sorted_entries, absolute, cpn)
            start_eval, end_eval = resolve_eval_range(start_id, end_id, pred_coords.shape[0])
            # align the root joint
            gt_coords -= gt_coords[:, 0, np.newaxis]
            pred_coords -= pred_coords[:, 0, np.newaxis]
            rel_error = np.linalg.norm(gt_coords[start_eval:end_eval, ...] - pred_coords[start_eval:end_eval, ...], axis=-1)
            rel_error_mean = safe_nanmean(rel_error)
            print("Relative MPJPE: ", np.round(rel_error_mean, 2))
            activities_errors = [np.mean(rel_error[a == activities]) for a in ordered_activities]
            print(np.round(activities_errors, 2))
            pa_mpjpe = compute_pa_mpjpe(pred_coords[start_eval:end_eval, ...], gt_coords[start_eval:end_eval, ...])
            pck = compute_pck(pred_coords[start_eval:end_eval, ...], gt_coords[start_eval:end_eval, ...], threshold=pck_threshold)
            print("PA-MPJPE: ", np.round(pa_mpjpe, 2))
            print(f"PCK@{int(pck_threshold)}mm: ", np.round(pck, 2))
            print("\n")
            results.append({
                "iteration": it,
                "mpjpe": float(abs_error_mean),
                "relative_mpjpe": float(rel_error_mean),
                "pa_mpjpe": float(pa_mpjpe),
                "pck": float(pck),
            })

        else:

            eval_joint_ids_np = None
            if eval_joint_ids is not None:
                eval_joint_ids_np = np.asarray(eval_joint_ids, dtype=np.int64)

            # Absolute MPJPE
            absolute = True
            gt_coords = get_gt_poses(gt_path, absolute, dataset, frame_step=frame_step, nviews=nviews)
            pred_coords = get_pred_coords(ply_dir, sorted_entries, absolute)
            start_eval, end_eval = resolve_eval_range(start_id, end_id, pred_coords.shape[0])
            if eval_joint_ids_np is not None:
                gt_coords = gt_coords[:, eval_joint_ids_np, :]
                pred_coords = pred_coords[:, eval_joint_ids_np, :]

            print("Evaluating scenes from", start_eval, "to", end_eval)
            abs_error = np.linalg.norm(gt_coords[start_eval:end_eval, ...] - pred_coords[start_eval:end_eval, ...], axis=-1)
            abs_valid = None
            if dataset == "panoptic":
                abs_valid = panoptic_gt_valid_mask(gt_coords[start_eval:end_eval, ...])
                abs_error = np.where(abs_valid, abs_error, np.nan)
            abs_error_mean = safe_nanmean(abs_error)
            print("Absolute MPJPE: ", np.round(abs_error_mean, 2))

            # Relative MPJPE
            absolute = False
            gt_coords = get_gt_poses(gt_path, absolute, dataset, frame_step=frame_step, nviews=nviews)
            pred_coords = get_pred_coords(ply_dir, sorted_entries, absolute)
            start_eval, end_eval = resolve_eval_range(start_id, end_id, pred_coords.shape[0])
            if eval_joint_ids_np is not None:
                gt_coords = gt_coords[:, eval_joint_ids_np, :]
                pred_coords = pred_coords[:, eval_joint_ids_np, :]
            rel_valid = None
            if dataset == "panoptic":
                rel_valid = panoptic_gt_valid_mask(gt_coords[start_eval:end_eval, ...])
                rel_valid = rel_valid & rel_valid[:, [0]]
            # align the root joint
            gt_coords -= gt_coords[:, 0, np.newaxis]
            pred_coords -= pred_coords[:, 0, np.newaxis]
            rel_error = np.linalg.norm(gt_coords[start_eval:end_eval, ...] - pred_coords[start_eval:end_eval, ...], axis=-1)
            if dataset == "panoptic":
                rel_error = np.where(rel_valid, rel_error, np.nan)
            rel_error_mean = safe_nanmean(rel_error)
            print("Relative MPJPE: ", np.round(rel_error_mean, 2))
            pred_eval = pred_coords[start_eval:end_eval, ...]
            gt_eval = gt_coords[start_eval:end_eval, ...]
            if dataset == "panoptic" and rel_valid is not None:
                pa_mpjpe = compute_pa_mpjpe_valid(pred_eval, gt_eval, rel_valid)
                pck = compute_pck_valid(pred_eval, gt_eval, rel_valid, threshold=pck_threshold)
            else:
                pa_mpjpe = compute_pa_mpjpe(pred_eval, gt_eval)
                pck = compute_pck(pred_eval, gt_eval, threshold=pck_threshold)
            print("PA-MPJPE: ", np.round(pa_mpjpe, 2))
            print(f"PCK@{int(pck_threshold)}mm: ", np.round(pck, 2))
            print("\n")
            results.append({
                "iteration": it,
                "mpjpe": float(abs_error_mean),
                "relative_mpjpe": float(rel_error_mean),
                "pa_mpjpe": float(pa_mpjpe),
                "pck": float(pck),
            })

    if return_results:
        return results


@hydra.main(version_base=None, config_path="configs", config_name="configs")
def main(cfg: DictConfig):

    if "training" in cfg: 
        config = ConfigHandler(cfg)
    else:
        config = TriangulationConfigHandler(cfg)

    output_path = config.hydra_out
    dataset = cfg.dataset
    start_id = dataset.start_scene_id
    end_id = dataset.end_scene_id
    debug = cfg.debug

    print("Evaluating ", output_path)

    gt_path = os.path.join(dataset.data_root, "3d_gt")
    iterations = debug.save_iterations
    eval_joint_ids = parse_joint_ids(getattr(dataset, "eval_joint_ids", None))
    if eval_joint_ids is None:
        eval_joint_ids = parse_joint_ids(getattr(cfg.training, "eval_joint_ids", None)) if "training" in cfg else None
    if eval_joint_ids is not None:
        print("Eval joint ids:", eval_joint_ids)
    evaluate(
        gt_path,
        output_path,
        iterations,
        start_id,
        end_id,
        dataset.nviews,
        dataset.poses_2d == "cpn",
        eval_joint_ids,
        getattr(dataset, "frame_step", 1),
    )


if __name__ == "__main__":
    main()
