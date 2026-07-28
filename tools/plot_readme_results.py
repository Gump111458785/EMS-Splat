#!/usr/bin/env python3
"""Render the README robustness figure from the public summary CSV."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


COLORS = {"SkelSplat": "#7D8795", "EMS-Splat": "#0072B2"}
MARKERS = {"SkelSplat": "o", "EMS-Splat": "s"}


def load_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No result rows found in {path}")
    return rows


def ratio_series(rows, occlusion_type):
    grouped = defaultdict(list)
    for row in rows:
        if row["occlusion_type"] in {"clean", occlusion_type}:
            grouped[row["method"]].append(
                (float(row["occlusion_ratio"]), float(row["MPJPE"]))
            )
    return {method: sorted(values) for method, values in grouped.items()}


def body_part_series(rows):
    grouped = defaultdict(dict)
    for row in rows:
        if row["occlusion_type"] == "body_part":
            grouped[row["method"]][row["body_part"]] = float(row["MPJPE"])
    return grouped


def configure_matplotlib():
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def style_axis(ax, title):
    ax.set_title(title, fontsize=10, pad=7)
    ax.set_ylabel("MPJPE (mm)")
    ax.grid(axis="y", color="#D9DEE5", linewidth=0.7)
    ax.set_axisbelow(True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("assets/results/h36m_corrected_summary.csv"),
    )
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=Path("assets/h36m_robustness"),
    )
    args = parser.parse_args()

    configure_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    rows = load_rows(args.csv)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.4, 2.45),
        gridspec_kw={"width_ratios": [1.08, 1.08, 1.0]},
    )

    for ax, occ_type, title in (
        (axes[0], "rectangle", "Contiguous rectangle"),
        (axes[1], "random_block", "Four random blocks"),
    ):
        for method, values in ratio_series(rows, occ_type).items():
            x, y = zip(*values)
            ax.plot(
                x,
                y,
                color=COLORS[method],
                marker=MARKERS[method],
                markersize=4.5,
                linewidth=2.0,
                label=method,
            )
        ax.set_xticks([0.0, 0.2, 0.4, 0.6])
        ax.set_xlabel("Occluded-area ratio")
        ax.set_ylim(bottom=0)
        style_axis(ax, title)

    body_parts = ["both_arms", "both_legs", "torso"]
    labels = ["Both arms", "Both legs", "Torso"]
    body_values = body_part_series(rows)
    x = np.arange(len(body_parts))
    width = 0.36
    for offset, method in ((-width / 2, "SkelSplat"), (width / 2, "EMS-Splat")):
        values = [body_values[method][part] for part in body_parts]
        bars = axes[2].bar(
            x + offset,
            values,
            width,
            color=COLORS[method],
            label=method,
        )
        axes[2].bar_label(bars, fmt="%.0f", padding=2, fontsize=7)
    axes[2].set_xticks(x, labels)
    axes[2].set_ylim(0, 125)
    style_axis(axes[2], "Semantic body-part masks")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=2,
    )
    fig.suptitle(
        "Controlled H36M observation-space occlusion (2,078 scenes)",
        y=1.13,
        fontsize=11,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.02,
        "CPN observations; 500-pass configuration; lower is better.",
        ha="center",
        color="#4B5563",
        fontsize=7,
    )
    fig.tight_layout(w_pad=1.5)

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = args.out_prefix.with_suffix(".png")
    svg_path = args.out_prefix.with_suffix(".svg")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    # Matplotlib emits spaces at the end of multiline SVG path commands.
    # Normalize them so regenerating the figure keeps the Git diff clean.
    svg_lines = svg_path.read_text(encoding="utf-8").splitlines()
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_lines) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
