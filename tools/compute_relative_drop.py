import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Compute relative performance degradation under occlusion.")
    parser.add_argument("--csv", default="outputs/occlusion_results.csv")
    parser.add_argument("--output", default="outputs/occlusion_relative_drop.csv")
    parser.add_argument("--metrics", nargs="+", default=["mpjpe", "relative_mpjpe", "pa_mpjpe", "pck"])
    parser.add_argument("--clean-ratio", type=float, default=0.0)
    args = parser.parse_args()

    with open(args.csv, newline="") as f:
        rows_in = list(csv.DictReader(f))
    grouped = defaultdict(list)
    for row in rows_in:
        grouped[row["method"]].append(row)

    rows = []
    for method, group in grouped.items():
        clean_rows = [row for row in group if float(row["ratio"]) == args.clean_ratio]
        if not clean_rows:
            raise ValueError(f"Missing clean ratio {args.clean_ratio} for method {method}")
        clean = clean_rows[0]
        for row in group:
            out = {
                "method": method,
                "occlusion_type": row.get("occlusion_type", ""),
                "body_part": row.get("body_part", ""),
                "ratio": float(row["ratio"]),
            }
            for metric in args.metrics:
                clean_value = float(clean[metric])
                value = float(row[metric])
                if metric == "pck":
                    # PCK is higher-is-better, so degradation is clean-to-occluded drop.
                    drop = (clean_value - value) / clean_value if clean_value != 0 else 0.0
                else:
                    drop = (value - clean_value) / clean_value if clean_value != 0 else 0.0
                out[f"{metric}_relative_drop"] = drop
            rows.append(out)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: (row["method"], row["ratio"]))
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("Relative Performance Degradation")
    for row in rows:
        print(row)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
