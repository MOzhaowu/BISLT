#!/usr/bin/env python3
"""Aggregate complete BIT metrics by sequence, object, and globally."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


FIELDS = (
    "add_mean_mm",
    "add_auc_100mm",
    "rotation_mean_deg",
    "translation_mean_mm",
    "success_5deg_50mm",
)


def flatten(path):
    data = json.loads(path.read_text())
    return {
        "object": data["object"],
        "sequence": data["sequence"],
        "experiment": path.parent.name,
        "frames": int(data["evaluated_frames"]),
        "add_mean_mm": float(data["add_mean_mm"]),
        "add_auc_100mm": float(data["add_auc_100mm"]),
        "rotation_mean_deg": float(data["rotation_error_deg"]["mean"]),
        "translation_mean_mm": float(data["translation_error_mm"]["mean"]),
        "success_5deg_50mm": float(data["success_5deg_50mm"]),
        "metrics_path": str(path.resolve()),
    }


def summarize(rows):
    frames = sum(row["frames"] for row in rows)
    result = {"sequences": len(rows), "frames": frames}
    for field in FIELDS:
        result[field] = {
            "macro": sum(row[field] for row in rows) / len(rows),
            "frame_weighted": sum(row[field] * row["frames"] for row in rows) / frames,
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--run-glob",
        default="**/run_[0-9]*/metrics.json",
        help="Metrics glob relative to root; defaults to formal run_N directories",
    )
    args = parser.parse_args()

    metric_files = sorted(args.root.glob(args.run_glob))
    rows = [flatten(path) for path in metric_files]
    if not rows:
        raise SystemExit(f"No metrics.json found below {args.root}")

    output_dir = args.output_dir or args.root
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "sequence_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    grouped = defaultdict(list)
    experiments = defaultdict(list)
    for row in rows:
        grouped[row["object"]].append(row)
        experiments[row["experiment"]].append(row)
    summary = {
        "schema_version": 1,
        "overall": summarize(rows),
        "objects": {name: summarize(values) for name, values in sorted(grouped.items())},
        "experiments": {
            name: summarize(values) for name, values in sorted(experiments.items())
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
