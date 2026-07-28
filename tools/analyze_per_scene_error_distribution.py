#!/usr/bin/env python3
"""Paired per-scene MPJPE diagnostics from official-eval NPZ artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ACTION_NAMES = {
    2: "Directions", 3: "Discussion", 4: "Eating", 5: "Greeting",
    6: "Phoning", 7: "Posing", 8: "Purchases", 9: "Sitting",
    10: "SittingDown", 11: "Smoking", 12: "Photo", 13: "Waiting",
    14: "Walking", 15: "WalkDog", 16: "WalkTogether",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--conditions", nargs="+", default=["rectangle_0p6", "random_block_0p6"]
    )
    parser.add_argument("--artifact-status", default="pose-bank-noncompliant")
    return parser.parse_args()


def load_condition(root: Path, condition: str):
    values = {}
    identities = None
    gt_reference = None
    aggregate_checks = {}
    for method in ("baseline", "ems"):
        condition_root = root / method / condition
        npz_path = condition_root / "inference/official_eval_inputs.npz"
        json_path = condition_root / "official_eval_result.json"
        data = np.load(npz_path, allow_pickle=False)
        prediction = np.asarray(data["pred_3d"], dtype=np.float64)
        ground_truth = np.asarray(data["gt_3d"], dtype=np.float64)
        if prediction.shape != ground_truth.shape or prediction.ndim != 3:
            raise RuntimeError(f"Invalid prediction/GT shapes in {npz_path}")
        identity = np.stack(
            [data["subject"], data["action"], data["subaction"], data["image_id"]], axis=1
        ).astype(np.int64)
        if identities is None:
            identities = identity
            gt_reference = ground_truth
        else:
            if not np.array_equal(identity, identities):
                raise RuntimeError(f"Identity mismatch for {condition}")
            if not np.array_equal(ground_truth, gt_reference):
                raise RuntimeError(f"GT mismatch for {condition}")
        per_scene = np.linalg.norm(prediction - ground_truth, axis=-1).mean(axis=1)
        official = json.loads(json_path.read_text())
        difference = abs(float(per_scene.mean()) - float(official["MPJPE"]))
        if difference > 1e-4:
            raise RuntimeError(
                f"Recomputed MPJPE differs from official result by {difference:.6g} mm: {condition}/{method}"
            )
        values[method] = per_scene
        aggregate_checks[method] = {
            "recomputed": float(per_scene.mean()),
            "official": float(official["MPJPE"]),
            "absolute_difference": difference,
            "npz_path": str(npz_path.resolve()),
            "official_path": str(json_path.resolve()),
        }
    return identities, values, aggregate_checks


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summary_stats(baseline: np.ndarray, ems: np.ndarray) -> dict:
    delta = ems - baseline
    return {
        "scene_count": int(len(baseline)),
        "baseline_mean": float(np.mean(baseline)),
        "ems_mean": float(np.mean(ems)),
        "baseline_median": float(np.median(baseline)),
        "ems_median": float(np.median(ems)),
        "baseline_q25": float(np.percentile(baseline, 25)),
        "ems_q25": float(np.percentile(ems, 25)),
        "baseline_q75": float(np.percentile(baseline, 75)),
        "ems_q75": float(np.percentile(ems, 75)),
        "baseline_q90": float(np.percentile(baseline, 90)),
        "ems_q90": float(np.percentile(ems, 90)),
        "baseline_q95": float(np.percentile(baseline, 95)),
        "ems_q95": float(np.percentile(ems, 95)),
        "fraction_improved": float(np.mean(delta < 0)),
        "fraction_worsened": float(np.mean(delta > 0)),
        "fraction_equal": float(np.mean(delta == 0)),
        "median_paired_delta": float(np.median(delta)),
        "largest_improvement": float(-np.min(delta)),
        "largest_degradation": float(np.max(delta)),
    }


def make_figure(condition_data: dict, output: Path, artifact_status: str) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )
    neutral = "#777777"
    signal = "#2F6B9A"
    conditions = list(condition_data)
    fig, axes = plt.subplots(len(conditions), 2, figsize=(7.1, 2.65 * len(conditions)))
    if len(conditions) == 1:
        axes = np.asarray([axes])
    for row_index, condition in enumerate(conditions):
        baseline = condition_data[condition]["baseline"]
        ems = condition_data[condition]["ems"]
        label = condition.replace("_", " ").replace("0p", "0.").title()

        ax = axes[row_index, 0]
        ems_label = "EMS-Splat" if artifact_status == "valid-final" else "EMS-Splat (legacy)"
        for values, name, color in (
            (baseline, "SkelSplat", neutral),
            (ems, ems_label, signal),
        ):
            ordered = np.sort(values)
            probability = np.arange(1, len(ordered) + 1) / len(ordered)
            ax.plot(ordered, probability, color=color, linewidth=1.6, label=name)
        ax.set_xlabel("Per-scene MPJPE (mm)")
        ax.set_ylabel("Cumulative fraction")
        ax.set_title(f"{label}: error ECDF", loc="left", fontweight="bold")
        ax.grid(axis="both", color="#E6E6E6", linewidth=0.6)
        ax.legend(loc="lower right")

        ax = axes[row_index, 1]
        ax.scatter(baseline, ems, s=7, alpha=0.32, color=signal, edgecolors="none", rasterized=True)
        limit = float(max(np.max(baseline), np.max(ems)))
        ax.plot([0, limit], [0, limit], color="#444444", linewidth=0.9, linestyle="--")
        ax.set_xlim(0, limit * 1.02)
        ax.set_ylim(0, limit * 1.02)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("SkelSplat per-scene MPJPE (mm)")
        ax.set_ylabel("EMS-Splat per-scene MPJPE (mm)")
        ax.set_title(f"{label}: paired scenes", loc="left", fontweight="bold")
        ax.grid(color="#E6E6E6", linewidth=0.6)

    title_prefix = "Corrected per-scene analysis" if artifact_status == "valid-final" else "Provisional per-scene diagnostics"
    fig.suptitle(
        f"{title_prefix} ({artifact_status})",
        x=0.08,
        ha="left",
        fontsize=10,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    summaries = {}
    condition_data = {}
    aggregate_checks = {}
    for condition in args.conditions:
        identities, values, checks = load_condition(args.comparison_root, condition)
        baseline = values["baseline"]
        ems = values["ems"]
        delta = ems - baseline
        condition_data[condition] = values
        summaries[condition] = summary_stats(baseline, ems)
        aggregate_checks[condition] = checks
        for index, identity in enumerate(identities):
            subject, action, subaction, image_id = [int(value) for value in identity]
            all_rows.append(
                {
                    "condition": condition,
                    "subject": subject,
                    "action": ACTION_NAMES.get(action, f"Action{action}"),
                    "action_id": action,
                    "subaction": subaction,
                    "image_id": image_id,
                    "scene_key": f"S{subject}_{ACTION_NAMES.get(action, f'Action{action}')}_{subaction}_{image_id:06d}",
                    "baseline_mpjpe": f"{baseline[index]:.8f}",
                    "ems_mpjpe": f"{ems[index]:.8f}",
                    "delta_ems_minus_baseline": f"{delta[index]:.8f}",
                    "relative_improvement_percent": f"{(-delta[index] / baseline[index] * 100):.8f}",
                    "ems_improved_boolean": str(bool(delta[index] < 0)).lower(),
                    "artifact_status": args.artifact_status,
                }
            )
    write_csv(args.out_dir / "per_scene_error.csv", all_rows)
    (args.out_dir / "per_scene_error_summary.json").write_text(
        json.dumps(
            {
                "artifact_status": args.artifact_status,
                "summaries": summaries,
                "aggregate_checks": aggregate_checks,
            },
            indent=2,
        )
        + "\n"
    )
    readiness = (
        "These diagnostics use artifact-gated corrected predictions."
        if args.artifact_status == "valid-final"
        else "These diagnostics are not paper-ready."
    )
    lines = [
        "# Per-scene error distribution",
        "",
        f"Artifact status: `{args.artifact_status}`. {readiness}",
        "",
        "| Condition | n | Baseline mean | EMS mean | Baseline median | EMS median | Improved | Worsened | Median paired delta | Largest degradation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, stats in summaries.items():
        lines.append(
            f"| {condition} | {stats['scene_count']} | {stats['baseline_mean']:.2f} | "
            f"{stats['ems_mean']:.2f} | {stats['baseline_median']:.2f} | {stats['ems_median']:.2f} | "
            f"{stats['fraction_improved'] * 100:.2f}% | {stats['fraction_worsened'] * 100:.2f}% | "
            f"{stats['median_paired_delta']:.2f} | {stats['largest_degradation']:.2f} |"
        )
    lines.extend(
        [
            "",
            "The recomputed mean of per-scene absolute MPJPE matches each official JSON within `1e-4 mm`.",
            "No p-value or statistical-significance claim is made.",
        ]
    )
    if args.artifact_status != "valid-final":
        lines.append("Corrected predictions must replace these rows.")
    (args.out_dir / "per_scene_error_summary.md").write_text("\n".join(lines) + "\n")
    make_figure(
        condition_data,
        args.out_dir / "per_scene_error_distribution",
        args.artifact_status,
    )
    print(f"Wrote per-scene diagnostics to {args.out_dir}")


if __name__ == "__main__":
    main()
