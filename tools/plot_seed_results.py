#!/usr/bin/env python3
"""Plot seed mean/std MPJPE bars from seed_mean_std.csv."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MEAN_STD_RE = re.compile(r"([0-9.]+)\s*\+/-\s*([0-9.]+)")


def parse_mean_std(text: str):
    match = MEAN_STD_RE.match(text or "")
    if not match:
        return None, None
    return float(match.group(1)), float(match.group(2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot seed mean/std results.")
    parser.add_argument("--csv", type=Path, default=Path("outputs/analysis/seed_mean_std/seed_mean_std.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/analysis/seed_mean_std/seed_mean_std_plot.png"))
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(args.csv)
    rows = list(csv.DictReader(args.csv.open(newline="", encoding="utf-8")))
    labels = []
    means = []
    stds = []
    colors = []
    for row in rows:
        mean, std = parse_mean_std(row.get("MPJPE_mean_std", ""))
        if mean is None:
            continue
        labels.append(f"{row['protocol']}\n{row['method']}")
        means.append(mean)
        stds.append(std)
        colors.append("#777777" if "baseline" in row["method"].lower() else "#2374ab")
    if not labels:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No paper-usable seed aggregates yet", ha="center", va="center")
        ax.axis("off")
        fig.savefig(args.output, dpi=300, bbox_inches="tight")
        return
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 4.5))
    ax.bar(x, means, yerr=stds, capsize=4, color=colors)
    ax.set_ylabel("MPJPE (mm)")
    ax.set_title("Seed Mean +/- Std")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
