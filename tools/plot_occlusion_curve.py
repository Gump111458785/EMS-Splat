import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Plot occlusion robustness curves.")
    parser.add_argument("--csv", default="outputs/occlusion_results.csv")
    parser.add_argument("--output", default="outputs/occlusion_curve.png")
    parser.add_argument("--metric", default="mpjpe", choices=["mpjpe", "relative_mpjpe", "pa_mpjpe", "pck"])
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    csv_path = Path(args.csv)
    out_path = Path(args.output)
    rows = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)

    fig, ax = plt.subplots(figsize=(6, 4))
    for method, group in grouped.items():
        group = sorted(group, key=lambda row: float(row["ratio"]))
        ax.plot(
            [float(row["ratio"]) for row in group],
            [float(row[args.metric]) for row in group],
            marker="o",
            linewidth=2,
            label=method,
        )

    ax.set_xlabel("Occlusion Ratio")
    ylabel = {
        "mpjpe": "MPJPE (mm)",
        "relative_mpjpe": "Root-relative MPJPE (mm)",
        "pa_mpjpe": "PA-MPJPE (mm)",
        "pck": "PCK@150mm (%)",
    }[args.metric]
    ax.set_ylabel(ylabel)
    ax.set_title("Occlusion Robustness")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
