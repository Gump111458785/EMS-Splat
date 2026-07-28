import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path


METRIC_PATTERNS = {
    "mpjpe": re.compile(r"Absolute MPJPE:\s*([0-9.]+)"),
    "relative_mpjpe": re.compile(r"Relative MPJPE:\s*([0-9.]+)"),
    "pa_mpjpe": re.compile(r"PA-MPJPE:\s*([0-9.]+)"),
    "pck": re.compile(r"PCK@\d+mm:\s*([0-9.]+)"),
}


def parse_method(raw):
    if "=" not in raw:
        raise argparse.ArgumentTypeError("Methods must use NAME=PATH format")
    name, path = raw.split("=", 1)
    return name, os.path.abspath(path)


def parse_metrics(text):
    metrics = {}
    for key, pattern in METRIC_PATTERNS.items():
        matches = pattern.findall(text)
        metrics[key] = float(matches[-1]) if matches else None
    return metrics


def validate_metrics(metrics, eval_log):
    missing = [name for name, value in metrics.items() if value is None]
    if missing:
        raise RuntimeError(f"Missing metrics {missing} in eval output. See {eval_log}")


def expected_scene_count(args):
    if args.end_scene_id is None or args.end_scene_id < 0:
        return None
    start = args.start_scene_id or 0
    return max(args.end_scene_id - start, 0)


def point_cloud_ready(point_cloud_dir, args):
    if not point_cloud_dir.exists():
        return False, 0, expected_scene_count(args)
    ply_count = sum(1 for _ in point_cloud_dir.rglob("*.ply"))
    expected = expected_scene_count(args)
    if expected is None:
        return False, ply_count, expected
    return ply_count >= expected, ply_count, expected


def run_command(cmd, cwd, log_path, dry_run=False):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(cmd)
    if dry_run:
        print(f"[DRY-RUN] cwd={cwd} {printable}")
        return ""
    print(f"[RUN] cwd={cwd} {printable}")
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    Path(log_path).write_text(proc.stdout)
    if proc.returncode != 0:
        print(proc.stdout[-4000:])
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {printable}")
    return proc.stdout


def build_overrides(args, run_dir, ratio):
    save_iterations = f"[{args.iteration}]"
    overrides = [
        f"hydra.run.dir={run_dir}",
        f"dataset.data_root={args.data_root}",
        f"optimization.iterations={args.iteration}",
        f"debug.save_iterations={save_iterations}",
        "debug.save_images=false",
        "++test.occlusion.enable=true",
        f"++test.occlusion.type={args.occlusion_type}",
        f"++test.occlusion.ratio={ratio}",
        f"++test.occlusion.mode={args.mode}",
        f"++test.occlusion.body_part={args.body_part}",
        f"++test.occlusion.padding={args.padding}",
        f"++test.occlusion.num_blocks={args.num_blocks}",
        f"++test.occlusion.seed={args.seed}",
        f"++test.occlusion.save_visualizations={str(args.save_visualizations).lower()}",
        f"++test.occlusion.vis_num_samples={args.vis_num_samples}",
    ]
    if args.start_scene_id is not None:
        overrides.append(f"dataset.start_scene_id={args.start_scene_id}")
    if args.end_scene_id is not None:
        overrides.append(f"dataset.end_scene_id={args.end_scene_id}")
    overrides.extend(args.extra_override)
    return overrides


def parse_method_overrides(raw_overrides):
    parsed = {}
    for raw in raw_overrides:
        if ":" not in raw:
            raise ValueError("--method-override must use METHOD:hydra.override=value")
        method, override = raw.split(":", 1)
        parsed.setdefault(method, []).append(override)
    return parsed


def method_overrides_for(parsed_overrides, method_name):
    """Return wildcard overrides plus overrides for the current method."""
    return parsed_overrides.get("*", []) + parsed_overrides.get(method_name, [])


def build_eval_overrides(args, run_dir, method_overrides):
    overrides = [
        f"hydra.run.dir={run_dir}",
        f"dataset.data_root={args.data_root}",
        f"debug.save_iterations=[{args.iteration}]",
    ]
    if args.start_scene_id is not None:
        overrides.append(f"dataset.start_scene_id={args.start_scene_id}")
    if args.end_scene_id is not None:
        overrides.append(f"dataset.end_scene_id={args.end_scene_id}")
    overrides.extend(args.extra_override)
    overrides.extend(method_overrides)
    return overrides


def main():
    parser = argparse.ArgumentParser(description="Run controlled Human3.6M occlusion robustness evaluation.")
    parser.add_argument("--method", action="append", type=parse_method, default=None,
                        help="Method in NAME=PATH format. Repeat for Baseline and EMS-Splat.")
    parser.add_argument("--config-name", default="h36m")
    parser.add_argument("--data-root", default=os.path.abspath("data/h36m"))
    parser.add_argument("--output-csv", default="outputs/occlusion_results.csv")
    parser.add_argument("--run-root", default="outputs/occlusion_runs")
    parser.add_argument("--occlusion-type", choices=["rectangle", "body_part", "random_block"], default="rectangle")
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.0, 0.2, 0.4, 0.6])
    parser.add_argument("--body-part", default="torso",
                        choices=["left_arm", "right_arm", "both_arms", "left_leg", "right_leg", "both_legs", "torso"])
    parser.add_argument("--mode", choices=["black", "mean"], default="black")
    parser.add_argument("--padding", type=int, default=20)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--iteration", type=int, default=500)
    parser.add_argument("--start-scene-id", type=int, default=None)
    parser.add_argument("--end-scene-id", type=int, default=None)
    parser.add_argument("--save-visualizations", action="store_true")
    parser.add_argument("--vis-num-samples", type=int, default=10)
    parser.add_argument("--extra-override", action="append", default=[],
                        help="Additional Hydra override passed to train.py.")
    parser.add_argument("--method-override", action="append", default=[],
                        help="Method-specific override in METHOD:override=value format.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    script_root = Path(__file__).resolve().parent
    methods = args.method or [("EMS-Splat", str(script_root))]
    method_overrides = parse_method_overrides(args.method_override)
    output_csv = Path(args.output_csv)
    if not output_csv.is_absolute():
        output_csv = script_root / output_csv
    run_root = Path(args.run_root)
    if not run_root.is_absolute():
        run_root = script_root / run_root
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = script_root / data_root
    args.data_root = str(data_root)

    rows = []
    eval_logs_dir = output_csv.parent / "occlusion_eval_logs"
    for method_name, method_path in methods:
        for ratio in args.ratios:
            ratio_label = str(ratio).replace(".", "p")
            if args.occlusion_type == "body_part":
                condition = f"{args.occlusion_type}_{args.body_part}"
            elif args.occlusion_type == "random_block":
                condition = f"{args.occlusion_type}_{ratio_label}_b{args.num_blocks}"
            else:
                condition = f"{args.occlusion_type}_{ratio_label}"
            run_dir = run_root / method_name / condition
            train_log = eval_logs_dir / f"{method_name}_{condition}_train.log"
            eval_log = eval_logs_dir / f"{method_name}_{condition}_eval.log"
            point_cloud_dir = run_dir / "point_cloud" / f"iteration_{args.iteration}"

            overrides = build_overrides(args, str(run_dir), ratio)
            current_method_overrides = method_overrides_for(method_overrides, method_name)
            overrides.extend(current_method_overrides)
            ready, ply_count, expected_count = point_cloud_ready(point_cloud_dir, args)
            if not (args.skip_existing and ready):
                run_command(
                    [sys.executable, "train.py", "--config-name", args.config_name, *overrides],
                    cwd=method_path,
                    log_path=train_log,
                    dry_run=args.dry_run,
                )
            else:
                expected_text = "unknown" if expected_count is None else str(expected_count)
                print(f"[SKIP] existing complete run: {point_cloud_dir} ({ply_count}/{expected_text} ply)")

            eval_overrides = build_eval_overrides(args, str(run_dir), current_method_overrides)
            eval_stdout = run_command(
                [sys.executable, "eval.py", "--config-name", args.config_name, *eval_overrides],
                cwd=method_path,
                log_path=eval_log,
                dry_run=args.dry_run,
            )
            if args.dry_run:
                continue
            metrics = parse_metrics(eval_stdout)
            validate_metrics(metrics, eval_log)
            rows.append({
                "method": method_name,
                "occlusion_type": args.occlusion_type,
                "body_part": args.body_part if args.occlusion_type == "body_part" else "",
                "ratio": ratio,
                "mpjpe": metrics["mpjpe"],
                "relative_mpjpe": metrics["relative_mpjpe"],
                "pa_mpjpe": metrics["pa_mpjpe"],
                "pck": metrics["pck"],
                "run_dir": str(run_dir),
                "eval_log": str(eval_log),
            })

    if args.dry_run:
        return

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    main()
