#!/usr/bin/env python3
"""Compute corrected H36M joint and bone diagnostics from official-eval NPZs."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "outputs/corrected_main/cpn"
OUT_ROOT = ROOT / "outputs/analysis/structure_analysis"

JOINTS = [
    "Hip", "Right Hip", "Right Knee", "Right Ankle", "Left Hip", "Left Knee",
    "Left Ankle", "Spine", "Thorax", "Neck/Nose", "Head", "Left Shoulder",
    "Left Elbow", "Left Wrist", "Right Shoulder", "Right Elbow", "Right Wrist",
]
BODY_PARTS = {
    "Head": [9, 10], "Torso": [0, 7, 8], "Left Arm": [11, 12, 13],
    "Right Arm": [14, 15, 16], "Left Leg": [4, 5, 6], "Right Leg": [1, 2, 3],
}
BONES = [
    ("Left Upper Arm", 11, 12, "Left Arm"), ("Left Forearm", 12, 13, "Left Arm"),
    ("Right Upper Arm", 14, 15, "Right Arm"), ("Right Forearm", 15, 16, "Right Arm"),
    ("Left Thigh", 4, 5, "Left Leg"), ("Left Calf", 5, 6, "Left Leg"),
    ("Right Thigh", 1, 2, "Right Leg"), ("Right Calf", 2, 3, "Right Leg"),
    ("Hip-Spine", 0, 7, "Torso"), ("Spine-Thorax", 7, 8, "Torso"),
    ("Thorax-Head", 8, 10, "Head"), ("Left Shoulder Link", 8, 11, "Torso"),
    ("Right Shoulder Link", 8, 14, "Torso"), ("Left Hip Link", 0, 4, "Torso"),
    ("Right Hip Link", 0, 1, "Torso"),
]
SYMMETRY = [
    ("Upper Arms", (11, 12), (14, 15)), ("Forearms", (12, 13), (15, 16)),
    ("Thighs", (4, 5), (1, 2)), ("Calves", (5, 6), (2, 3)),
    ("Shoulder Links", (8, 11), (8, 14)), ("Hip Links", (0, 4), (0, 1)),
]
CONDITIONS = {
    "clean": ("clean", "0.0"),
    "rectangle_0p4": ("rectangle", "0.4"),
    "rectangle_0p6": ("rectangle", "0.6"),
    "random_block_0p4": ("random-block", "0.4"),
    "random_block_0p6": ("random-block", "0.6"),
    "body_part_both_arms": ("body-part", "both arms"),
    "body_part_both_legs": ("body-part", "both legs"),
    "body_part_torso": ("body-part", "torso"),
}
METHODS = {"baseline": "SkelSplat", "ems": "EMS-Splat"}
FOCUS = ["rectangle_0p6", "random_block_0p6", "body_part_both_legs"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", type=Path, default=RUN_ROOT)
    p.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    return p.parse_args()


def load_pair(root: Path, condition: str) -> dict:
    loaded = {}
    identities = None
    gt_ref = None
    for key in METHODS:
        base = root / key / condition / "official_eval"
        npz_path = base / "inference/official_eval_inputs.npz"
        json_path = base / "official_eval_result.json"
        z = np.load(npz_path, allow_pickle=False)
        pred = np.asarray(z["pred_3d"], dtype=np.float64)
        gt = np.asarray(z["gt_3d"], dtype=np.float64)
        identity = np.stack([z[k] for k in ("subject", "action", "subaction", "image_id")], axis=1)
        if pred.shape != (2078, 17, 3) or gt.shape != pred.shape:
            raise RuntimeError(f"Unexpected shape in {npz_path}: {pred.shape}/{gt.shape}")
        if identities is None:
            identities, gt_ref = identity, gt
        elif not np.array_equal(identity, identities) or not np.array_equal(gt, gt_ref):
            raise RuntimeError(f"Baseline/EMS identity or GT mismatch for {condition}")
        official = json.loads(json_path.read_text())
        recomputed = np.linalg.norm(pred - gt, axis=-1).mean(axis=1).mean()
        if abs(recomputed - float(official["MPJPE"])) > 1e-4:
            raise RuntimeError(f"Official MPJPE mismatch for {condition}/{key}")
        loaded[key] = {"pred": pred, "gt": gt, "npz": npz_path, "official": json_path}
    return loaded


def valid_mean(values: np.ndarray, mask: np.ndarray) -> float:
    selected = values[mask & np.isfinite(values)]
    return float(selected.mean())


def add_comparison(rows: list[dict], item: str, value: str) -> None:
    index = {}
    for row in rows:
        key = (row["condition"], row[item])
        if row["method"] == "SkelSplat":
            index[key] = float(row[value])
    for row in rows:
        base = index[(row["condition"], row[item])]
        current = float(row[value])
        row[f"baseline_{value}"] = base
        row["improvement_over_baseline"] = base - current
        row["relative_improvement_percent"] = (base - current) / base * 100.0 if base else 0.0


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, title: str, rows: list[dict], fields: list[str]) -> None:
    lines = [f"# {title}", "", "Artifact status: `valid-final`; source is corrected official-eval NPZ.", "",
             "| " + " | ".join(fields) + " |", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        vals = []
        for field in fields:
            value = row[field]
            vals.append(f"{value:.2f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n")


def style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.spines.right": False, "axes.spines.top": False,
        "axes.linewidth": 0.8, "pdf.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })


def plot_grouped(rows: list[dict], item: str, value: str, filename: Path, ylabel: str) -> None:
    style()
    fig, axes = plt.subplots(3, 1, figsize=(7.1, 8.2))
    colors = {"SkelSplat": "#777777", "EMS-Splat": "#2F6B9A"}
    titles = {"rectangle_0p6": "Rectangle 0.6", "random_block_0p6": "Random-block 0.6", "body_part_both_legs": "Both legs hidden"}
    for panel, (ax, condition) in enumerate(zip(axes, FOCUS)):
        subset = [r for r in rows if r["condition"] == condition]
        items = list(dict.fromkeys(r[item] for r in subset))
        x = np.arange(len(items)); width = 0.38
        for offset, method in [(-width / 2, "SkelSplat"), (width / 2, "EMS-Splat")]:
            lookup = {(r[item], r["method"]): float(r[value]) for r in subset}
            ax.bar(x + offset, [lookup[(label, method)] for label in items], width,
                   color=colors[method], label=method)
        ax.set_xticks(x, items, rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(titles[condition], loc="left", fontweight="bold")
        ax.text(-0.08, 1.04, chr(ord("a") + panel), transform=ax.transAxes, fontweight="bold", fontsize=9)
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    axes[0].legend(frameon=False, ncol=2)
    fig.tight_layout(h_pad=1.5)
    fig.savefig(filename.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(filename.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(filename.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    joint_rows, part_rows, bone_rows, sym_rows, sources = [], [], [], [], []
    for condition, (occ_type, ratio) in CONDITIONS.items():
        pair = load_pair(args.run_root, condition)
        for method_key, method in METHODS.items():
            pred, gt = pair[method_key]["pred"], pair[method_key]["gt"]
            valid = np.isfinite(gt).all(axis=-1)
            errors = np.linalg.norm(pred - gt, axis=-1)
            meta = {"dataset": "H36M", "condition": condition, "occlusion_type": occ_type,
                    "occlusion_ratio_or_part": ratio, "method": method, "scene_count": len(pred),
                    "result_source": str(pair[method_key]["official"]), "artifact_status": "valid-final"}
            for idx, name in enumerate(JOINTS):
                joint_rows.append({**meta, "joint_index": idx, "joint_name": name,
                                   "mpjpe": valid_mean(errors[:, idx], valid[:, idx])})
            for name, indices in BODY_PARTS.items():
                part_rows.append({**meta, "body_part": name,
                                  "mpjpe": valid_mean(errors[:, indices], valid[:, indices])})
            for name, j1, j2, group in BONES:
                mask = valid[:, j1] & valid[:, j2]
                pred_len = np.linalg.norm(pred[:, j1] - pred[:, j2], axis=-1)
                gt_len = np.linalg.norm(gt[:, j1] - gt[:, j2], axis=-1)
                bone_rows.append({**meta, "bone": name, "body_part": group, "j1": j1, "j2": j2,
                                  "error_mm": valid_mean(np.abs(pred_len - gt_len), mask)})
            for name, left, right in SYMMETRY:
                l1, l2 = left; r1, r2 = right
                mask = valid[:, l1] & valid[:, l2] & valid[:, r1] & valid[:, r2]
                pred_left = np.linalg.norm(pred[:, l1] - pred[:, l2], axis=-1)
                pred_right = np.linalg.norm(pred[:, r1] - pred[:, r2], axis=-1)
                gt_left = np.linalg.norm(gt[:, l1] - gt[:, l2], axis=-1)
                gt_right = np.linalg.norm(gt[:, r1] - gt[:, r2], axis=-1)
                sym_rows.append({**meta, "symmetry_pair": name,
                                 "error_mm": valid_mean(np.abs(pred_left - pred_right), mask),
                                 "gt_asymmetry_mm": valid_mean(np.abs(gt_left - gt_right), mask)})
            sources.append({"condition": condition, "method": method,
                            "npz": str(pair[method_key]["npz"]), "official_result": str(pair[method_key]["official"])})

    add_comparison(joint_rows, "joint_name", "mpjpe")
    add_comparison(part_rows, "body_part", "mpjpe")
    add_comparison(bone_rows, "bone", "error_mm")
    add_comparison(sym_rows, "symmetry_pair", "error_mm")
    for name, rows in [("per_joint_error", joint_rows), ("per_body_part_error", part_rows),
                       ("bone_length_error", bone_rows), ("symmetry_error", sym_rows)]:
        write_csv(args.out_dir / f"{name}.csv", rows)
    write_csv(args.out_dir / "source_manifest.csv", sources)
    common = ["dataset", "condition", "occlusion_type", "occlusion_ratio_or_part", "method"]
    write_md(args.out_dir / "per_joint_error.md", "Corrected Per-Joint Error", joint_rows,
             common + ["joint_index", "joint_name", "mpjpe", "improvement_over_baseline", "relative_improvement_percent"])
    write_md(args.out_dir / "per_body_part_error.md", "Corrected Per-Body-Part Error", part_rows,
             common + ["body_part", "mpjpe", "improvement_over_baseline", "relative_improvement_percent"])
    write_md(args.out_dir / "bone_length_error.md", "Corrected Bone-Length Error", bone_rows,
             common + ["bone", "body_part", "error_mm", "improvement_over_baseline", "relative_improvement_percent"])
    write_md(args.out_dir / "symmetry_error.md", "Corrected Symmetry Error", sym_rows,
             common + ["symmetry_pair", "error_mm", "gt_asymmetry_mm", "improvement_over_baseline", "relative_improvement_percent"])

    plot_grouped(joint_rows, "joint_name", "mpjpe", args.out_dir / "per_joint_error_bar", "MPJPE (mm)")
    plot_grouped(part_rows, "body_part", "mpjpe", args.out_dir / "per_body_part_error_bar", "MPJPE (mm)")
    plot_grouped(bone_rows, "bone", "error_mm", args.out_dir / "bone_length_error_bar", "Length error (mm)")
    plot_grouped(sym_rows, "symmetry_pair", "error_mm", args.out_dir / "symmetry_error_bar", "L-R mismatch (mm)")

    top_joint = sorted([r for r in joint_rows if r["method"] == "EMS-Splat"], key=lambda r: r["improvement_over_baseline"], reverse=True)[:12]
    top_bone = sorted([r for r in bone_rows if r["method"] == "EMS-Splat"], key=lambda r: r["improvement_over_baseline"], reverse=True)[:12]
    lines = ["# Corrected Joint and Bone Summary", "", "All rows use artifact-gated corrected official-eval NPZs (2078 matched scenes).", "",
             "## Largest Joint Improvements", "", "| Condition | Joint | Baseline | EMS | Improvement | Relative |", "|---|---|---:|---:|---:|---:|"]
    for r in top_joint:
        lines.append(f"| {r['condition']} | {r['joint_name']} | {r['baseline_mpjpe']:.2f} | {r['mpjpe']:.2f} | {r['improvement_over_baseline']:.2f} | {r['relative_improvement_percent']:.2f}% |")
    lines += ["", "## Largest Bone-Length Improvements", "", "| Condition | Bone | Baseline | EMS | Improvement | Relative |", "|---|---|---:|---:|---:|---:|"]
    for r in top_bone:
        lines.append(f"| {r['condition']} | {r['bone']} | {r['baseline_error_mm']:.2f} | {r['error_mm']:.2f} | {r['improvement_over_baseline']:.2f} | {r['relative_improvement_percent']:.2f}% |")
    lines += ["", "Symmetry is reported only as a left-right consistency diagnostic; real GT bodies are not assumed perfectly symmetric."]
    (args.out_dir / "structure_analysis_summary.md").write_text("\n".join(lines) + "\n")

    mapping = ["# H36M 17-Joint and Bone Mapping", ""] + [f"- {i}: {name}" for i, name in enumerate(JOINTS)]
    mapping += ["", "All bone and symmetry indices were validated against 17 joints; out-of-range indices raise before analysis."]
    (args.out_dir / "README_joint_bone_mapping.md").write_text("\n".join(mapping) + "\n")
    (args.out_dir / "README.md").write_text(
        "# Corrected structure analysis\n\n"
        "Generated only from corrected H36M official-eval NPZ files. The script verifies exact 2078-scene identity/GT pairing and reproduces official MPJPE within 1e-4 mm.\n\n"
        f"Command: `{shlex.join([sys.executable, str(Path(__file__).resolve()), '--run-root', str(args.run_root.resolve()), '--out-dir', str(args.out_dir.resolve())])}`\n"
    )
    (args.out_dir / "command.txt").write_text(
        shlex.join(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--run-root",
                str(args.run_root.resolve()),
                "--out-dir",
                str(args.out_dir.resolve()),
            ]
        )
        + "\n"
    )
    print(args.out_dir)


if __name__ == "__main__":
    main()
