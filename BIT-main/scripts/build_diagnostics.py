#!/usr/bin/env python3
"""Merge tracker diagnostics with per-frame pose errors."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluate_baseline import load_poses, project_rotation


def pose_errors(gt_rows, pose_rows):
    result = {}
    for gt, pred in zip(gt_rows, pose_rows):
        if gt[0] != pred[0]:
            raise ValueError(f"Frame index mismatch: {gt[0]} != {pred[0]}")
        relative = project_rotation(pred[1]) @ project_rotation(gt[1]).T
        cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
        rotation = math.degrees(math.acos(cosine))
        translation = float(np.linalg.norm(pred[2] - gt[2]) * 1000.0)
        result[gt[0]] = (rotation, translation)
    return result


def load_cpp_diagnostics(path):
    if not path.exists():
        return {}
    with path.open(newline="") as stream:
        rows = {}
        for row in csv.DictReader(stream):
            if "data_group_valid" not in row and "tracker_valid" in row:
                row["data_group_valid"] = row["tracker_valid"]
            rows[int(row["frame_index"])] = row
        return rows


def load_geometry_versions(path):
    versions = {}
    stable_version = 0
    if not path.exists():
        return versions
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("accepted", False):
            stable_version = int(record["version"])
        stable_version = int(record.get("stable_geometry_version", stable_version))
        versions[str(record["version"])] = str(stable_version)
    return versions


def as_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    raw_dir = args.run_dir / "raw"
    gt_rows = load_poses(raw_dir / "gt.txt")
    pose_rows = load_poses(raw_dir / "pose.txt")
    if len(gt_rows) != len(pose_rows):
        raise ValueError("GT and prediction frame counts differ")

    errors = pose_errors(gt_rows, pose_rows)
    cpp_rows = load_cpp_diagnostics(raw_dir / "diagnostics_cpp.csv")
    geometry_versions = load_geometry_versions(args.run_dir / "model_registry.jsonl")
    fieldnames = [
        "frame_index", "model_version_used", "model_version_after",
        "model_reloaded_after_frame", "tracking_time_ms",
        "geometry_version_used",
        "data_group_valid", "roi_x", "roi_y", "roi_width",
        "roi_height", "view_x", "view_y", "view_z", "min_view_angle_deg",
        "is_reference", "view_candidate", "sent_to_python",
        "rotation_error_deg", "translation_error_mm", "success_5deg_50mm",
    ]

    rows = []
    for frame_index in sorted(errors):
        row = {name: "" for name in fieldnames}
        row.update(cpp_rows.get(frame_index, {}))
        file_version = row.get("model_version_used", row.get("model_version", ""))
        if file_version != "":
            row["geometry_version_used"] = geometry_versions.get(
                str(file_version), str(file_version))
        rotation, translation = errors[frame_index]
        row.update({
            "frame_index": frame_index,
            "rotation_error_deg": rotation,
            "translation_error_mm": translation,
            "success_5deg_50mm": int(rotation <= 5.0 and translation <= 50.0),
        })
        rows.append(row)

    output = args.run_dir / "diagnostics.csv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    first_failure = next(
        (row["frame_index"] for row in rows if not as_bool(row["success_5deg_50mm"])),
        None,
    )
    versions = defaultdict(list)
    for row in rows:
        version = row.get("geometry_version_used", "")
        if version != "":
            versions[str(version)].append(row)

    per_version = {}
    for version, version_rows in versions.items():
        per_version[version] = {
            "frames": len(version_rows),
            "rotation_error_mean_deg": float(np.mean([
                float(row["rotation_error_deg"]) for row in version_rows
            ])),
            "translation_error_mean_mm": float(np.mean([
                float(row["translation_error_mm"]) for row in version_rows
            ])),
            "success_5deg_50mm": float(np.mean([
                int(row["success_5deg_50mm"]) for row in version_rows
            ])),
        }

    summary = {
        "schema_version": 1,
        "frames": len(rows),
        "cpp_diagnostics_available": bool(cpp_rows),
        "first_5deg_50mm_failure_frame": first_failure,
        "model_reloads": sum(
            as_bool(row.get("model_reloaded_after_frame", row.get("model_updated", "")))
            for row in rows
        ),
        "keyframes_sent_to_python": sum(
            as_bool(row.get("sent_to_python", "")) for row in rows
        ),
        "per_geometry_version": per_version,
    }
    (args.run_dir / "diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
