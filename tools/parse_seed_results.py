#!/usr/bin/env python3
"""Aggregate seed-sweep official eval JSON files into mean/std tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence


METRICS = ["MPJPE", "Rel_MPJPE", "PA_MPJPE", "PCK_150"]


def fmt_mean_std(values: Sequence[float]) -> str:
    if not values:
        return "n/a"
    mean = sum(values) / len(values)
    if len(values) == 1:
        std = 0.0
    else:
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
    return f"{mean:.2f} +/- {std:.2f}"


def read_rows(root: Path) -> List[Dict[str, object]]:
    rows = []
    for path in sorted(root.rglob("official_eval_result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["result_source"] = str(path)
        rows.append(payload)
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fieldnames} for row in rows])


def aggregate(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    groups = defaultdict(list)
    for row in rows:
        if not row.get("usable_for_paper", False):
            continue
        key = (
            row.get("dataset", ""),
            row.get("protocol", ""),
            row.get("occlusion_type", ""),
            row.get("occlusion_ratio", ""),
            row.get("method", ""),
            row.get("seed_type", ""),
        )
        groups[key].append(row)
    out = []
    for key, items in sorted(groups.items()):
        dataset, protocol, occ_type, ratio, method, seed_type = key
        record = {
            "dataset": dataset,
            "protocol": protocol,
            "occlusion_type": occ_type,
            "occlusion_ratio": ratio,
            "method": method,
            "seed_type": seed_type,
            "num_usable_seeds": len(items),
            "seeds": " ".join(str(item.get("seed", "")) for item in items),
        }
        for metric in METRICS:
            values = [float(item[metric]) for item in items if item.get(metric) not in {None, ""}]
            record[f"{metric}_mean_std"] = fmt_mean_std(values)
            record[f"{metric}_mean"] = sum(values) / len(values) if values else ""
        out.append(record)
    return out


def write_md(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Seed Mean +/- Std\n\n")
        f.write("| Method | Occlusion Type | Ratio | MPJPE ↓ | Rel MPJPE ↓ | PA-MPJPE ↓ | PCK@150 ↑ |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['method']} | {row['occlusion_type']} | {row['occlusion_ratio']} | "
                f"{row['MPJPE_mean_std']} | {row['Rel_MPJPE_mean_std']} | "
                f"{row['PA_MPJPE_mean_std']} | {row['PCK_150_mean_std']} |\n"
            )


def write_summary(path: Path, rows: Sequence[Dict[str, object]], raw_rows: Sequence[Dict[str, object]]) -> None:
    unusable = [row for row in raw_rows if not row.get("usable_for_paper", False)]
    with path.open("w", encoding="utf-8") as f:
        f.write("# Seed Mean/Std Summary\n\n")
        f.write("Seed type: occlusion mask seed for rectangle and random-block protocols. Clean and body-part rows are not paper-usable under this seed type because their seed is not effective.\n\n")
        if rows:
            f.write("## Paper-usable Aggregates\n\n")
            f.write("| Protocol | Method | Seeds | MPJPE | Rel MPJPE | PA-MPJPE | PCK@150 |\n")
            f.write("|---|---|---|---:|---:|---:|---:|\n")
            for row in rows:
                f.write(
                    f"| {row['protocol']} | {row['method']} | {row['seeds']} | {row['MPJPE_mean_std']} | "
                    f"{row['Rel_MPJPE_mean_std']} | {row['PA_MPJPE_mean_std']} | {row['PCK_150_mean_std']} |\n"
                )
        else:
            f.write("No paper-usable aggregates are available yet.\n")
        if unusable:
            f.write("\n## Unusable / Pending Rows\n\n")
            f.write("| Protocol | Method | Seed | Reason |\n")
            f.write("|---|---|---:|---|\n")
            for row in unusable:
                reason = row.get("note", "")
                if not row.get("seed_effective", True):
                    reason = reason or "Seed is not effective for this protocol."
                f.write(f"| {row.get('protocol', '')} | {row.get('method', '')} | {row.get('seed', '')} | {reason} |\n")
        f.write("\n## Paper Wording\n\n")
        f.write(
            "We repeat the robustness evaluation with three random occlusion-mask seeds and report mean and standard deviation. "
            "This wording should be used only for protocols where the mask seed is effective, namely rectangle and random-block occlusion in the current implementation.\n"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse seed-sweep result JSON files.")
    parser.add_argument("--root", type=Path, default=Path("outputs/seed_runs"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/analysis/seed_mean_std"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = read_rows(args.root)
    summary_rows = aggregate(raw_rows)
    raw_fields = [
        "dataset",
        "protocol",
        "occlusion_type",
        "occlusion_ratio",
        "method",
        "seed_type",
        "seed",
        "MPJPE",
        "Rel_MPJPE",
        "PA_MPJPE",
        "PCK_150",
        "result_source",
        "output_dir",
        "log_path",
        "usable_for_paper",
        "note",
    ]
    summary_fields = [
        "dataset",
        "protocol",
        "occlusion_type",
        "occlusion_ratio",
        "method",
        "seed_type",
        "num_usable_seeds",
        "seeds",
        "MPJPE_mean_std",
        "Rel_MPJPE_mean_std",
        "PA_MPJPE_mean_std",
        "PCK_150_mean_std",
    ]
    write_csv(args.out_dir / "seed_runs_raw.csv", raw_rows, raw_fields)
    write_csv(args.out_dir / "seed_mean_std.csv", summary_rows, summary_fields)
    write_md(args.out_dir / "seed_mean_std.md", summary_rows)
    write_summary(args.out_dir / "seed_mean_std_summary.md", summary_rows, raw_rows)
    print(f"Wrote seed mean/std analysis to {args.out_dir}")


if __name__ == "__main__":
    main()
