#!/usr/bin/env python3
"""Validate BIT baseline run artifacts and frame counts."""

import argparse
import csv
import json
import math
from pathlib import Path


REQUIRED_METRICS = (
    "add_mean_mm",
    "add_auc_100mm",
    "success_5deg_50mm",
)


def count_nonempty_lines(path):
    return sum(bool(line.strip()) for line in path.read_text().splitlines())


def validate(run_dir):
    errors = []
    for relative in ("manifest.json", "metrics.json", "run.log", "exit_code.txt",
                     "raw/gt.txt", "raw/pose.txt"):
        if not (run_dir / relative).is_file():
            errors.append(f"missing {relative}")
    if errors:
        return errors

    manifest = json.loads((run_dir / "manifest.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    expected = int(manifest["expected_frames"])
    gt_frames = count_nonempty_lines(run_dir / "raw/gt.txt")
    pose_frames = count_nonempty_lines(run_dir / "raw/pose.txt")

    if int((run_dir / "exit_code.txt").read_text().strip()) != 0:
        errors.append("run exit code is non-zero")

    if gt_frames != expected:
        errors.append(f"GT frames {gt_frames} != expected {expected}")
    if pose_frames != expected:
        errors.append(f"pose frames {pose_frames} != expected {expected}")
    if int(metrics.get("evaluated_frames", -1)) != expected:
        errors.append("metrics evaluated_frames mismatch")
    if not metrics.get("complete", False):
        errors.append("metrics complete is false")

    for key in REQUIRED_METRICS:
        value = metrics.get(key)
        if value is None or not math.isfinite(float(value)):
            errors.append(f"metric {key} is missing or non-finite")

    diagnostics = run_dir / "diagnostics.csv"
    if diagnostics.exists():
        with diagnostics.open(newline="") as stream:
            diagnostic_frames = sum(1 for _ in csv.DictReader(stream))
        if diagnostic_frames != expected:
            errors.append(
                f"diagnostic frames {diagnostic_frames} != expected {expected}"
            )
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = {"schema_version": 1, "runs": [], "valid": 0, "invalid": 0}
    for run_dir in args.run_dirs:
        errors = validate(run_dir)
        report["runs"].append({
            "run_dir": str(run_dir.resolve()),
            "valid": not errors,
            "errors": errors,
        })
        report["valid" if not errors else "invalid"] += 1

    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    raise SystemExit(1 if report["invalid"] else 0)


if __name__ == "__main__":
    main()
