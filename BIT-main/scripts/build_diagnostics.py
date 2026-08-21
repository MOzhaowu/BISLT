#!/usr/bin/env python3
"""Merge tracker diagnostics with per-frame pose errors."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
        "contour_residual", "histogram_separation", "contour_samples", "contour_noise_variance",
        "visible_boundary_ratio", "occlusion_ratio", "effective_contour_ratio",
        "silhouette_iou", "contour_residual_stddev", "contour_residual_p90",
        "contour_residual_centroid_offset", "pose_diagnostics_time_ms",
		"hessian_diagnostics_time_ms",
        "contour_search_lines", "active_contour_lines", "matched_contour_lines",
        "pose_hessian_valid", "hessian_min_eigenvalue",
        "hessian_max_eigenvalue", "hessian_condition",
        "pose_covariance_trace", "object_characteristic_length",
        "pose_teacher_valid", "pose_teacher_probes", "pose_teacher_failures",
        "pose_teacher_failure_rate", "pose_teacher_rotation_rms_deg",
        "pose_teacher_translation_rms_mm",
        "pose_teacher_ablation_probes", "pose_teacher_axis_failure_rate_small",
        "pose_teacher_axis_failure_rate_medium", "pose_teacher_axis_failure_rate_large",
        "pose_teacher_joint_failure_rate_small", "pose_teacher_joint_failure_rate_medium",
        "pose_teacher_joint_failure_rate_large",
        *["hessian_{}{}".format(row, col)
          for row in range(6) for col in range(6)],

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

    def numeric(name):
        return np.array([
            float(row[name]) if str(row.get(name, "")).strip() else np.nan
            for row in rows
        ], dtype=float)

    frame_axis = numeric("frame_index")
    rotation = numeric("rotation_error_deg")
    translation = numeric("translation_error_mm")
    residual = numeric("contour_residual")
    separation = numeric("histogram_separation")
    valid_signals = np.isfinite(residual) & np.isfinite(separation)
    failure_mask = (rotation > 5.0) | (translation > 50.0)
    failure_modes = {
        "total_failure_frames": int(np.sum(failure_mask)),
        "high_contour_residual": int(np.sum(failure_mask & valid_signals & (residual >= np.nanquantile(residual, 0.75)))) if np.any(valid_signals) else 0,
        "low_histogram_separation": int(np.sum(failure_mask & valid_signals & (separation <= np.nanquantile(separation, 0.25)))) if np.any(valid_signals) else 0,
        "after_model_reload": sum(
            bool(failure_mask[i]) and as_bool(row.get("model_reloaded_after_frame", ""))
            for i, row in enumerate(rows)
        ),
    }

    figure, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(frame_axis, rotation, label="rotation (deg)")
    axes[0].plot(frame_axis, translation, label="translation (mm)")
    axes[0].axhline(5.0, color="tab:red", linestyle="--", linewidth=0.8)
    axes[0].axhline(50.0, color="tab:orange", linestyle="--", linewidth=0.8)
    axes[0].legend(loc="upper right")
    axes[1].plot(frame_axis, residual, color="tab:red", label="contour residual")
    axes[1].legend(loc="upper right")
    axes[2].plot(frame_axis, separation, color="tab:green", label="histogram separation")
    axes[2].legend(loc="upper right")
    axes[2].set_xlabel("frame")
    figure.tight_layout()
    figure.savefig(args.run_dir / "diagnostic_timeline.png", dpi=160)
    plt.close(figure)

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
        "diagnostic_signals": {
            "contour_residual_mean": float(np.nanmean(residual)) if np.any(valid_signals) else None,
            "histogram_separation_mean": float(np.nanmean(separation)) if np.any(valid_signals) else None,
            "valid_frames": int(np.sum(valid_signals)),
        },
        "failure_modes": failure_modes,
        "timeline_plot": "diagnostic_timeline.png",

        "per_geometry_version": per_version,
    }
    (args.run_dir / "diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
