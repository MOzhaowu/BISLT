#!/usr/bin/env python3
"""Align SAM uncertainty observations with per-frame 5deg/5cm outcomes."""

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


LABEL_FIELDS = [
    "rotation_error_deg", "translation_error_mm",
    "success_5deg_50mm", "failure_5deg_50mm",
]


def optional_float(row, name):
    value = row.get(name)
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def optional_log(row, name):
    value = optional_float(row, name)
    if value is None or value <= 0:
        return None
    return math.log(value)


def as_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes"}


def load_jsonl(path):
    return [
        json.loads(line) for line in path.read_text().splitlines()
        if line.strip()
    ]


def align_records(observations, diagnostics):
    by_frame = {int(row["frame_index"]): row for row in diagnostics}
    aligned = []
    for observation in observations:
        if "frame_index" not in observation:
            raise ValueError(
                "Uncertainty observation has no frame_index; rerun capture "
                "with frame-aware communication")
        frame_index = int(observation["frame_index"])
        if frame_index not in by_frame:
            raise ValueError(
                "No pose diagnostic for uncertainty frame {}".format(frame_index))
        diagnostic = by_frame[frame_index]
        success = as_bool(diagnostic["success_5deg_50mm"])
        record = dict(observation)
        record.setdefault(
            "temporal_missing", int(record.get("temporal_inconsistency") is None))
        record.setdefault(
            "temporal_or_projection_inconsistency",
            record.get("temporal_inconsistency")
            if record.get("temporal_inconsistency") is not None
            else record.get("projection_inconsistency"))
        record.update({
            "frame_index": frame_index,
            "rotation_error_deg": float(diagnostic["rotation_error_deg"]),
            "translation_error_mm": float(diagnostic["translation_error_mm"]),
            "success_5deg_50mm": int(success),
            "failure_5deg_50mm": int(not success),
            "contour_residual": optional_float(diagnostic, "contour_residual"),
            "histogram_separation": optional_float(
                diagnostic, "histogram_separation"),
            "min_view_angle_deg": optional_float(
                diagnostic, "min_view_angle_deg"),
            "pose_hessian_valid": int(as_bool(
                diagnostic.get("pose_hessian_valid", False))),
            "hessian_min_eigenvalue": optional_float(
                diagnostic, "hessian_min_eigenvalue"),
            "hessian_max_eigenvalue": optional_float(
                diagnostic, "hessian_max_eigenvalue"),
            "hessian_condition": optional_float(
                diagnostic, "hessian_condition"),
            "pose_covariance_trace": optional_float(
                diagnostic, "pose_covariance_trace"),
            "log_hessian_condition": optional_log(
                diagnostic, "hessian_condition"),
            "log_pose_covariance_trace": optional_log(
                diagnostic, "pose_covariance_trace"),
        })
        aligned.append(record)
    return aligned


def mask_iou(predicted, target):
    predicted, target = np.asarray(predicted, bool), np.asarray(target, bool)
    union = np.logical_or(predicted, target).sum()
    return 1.0 if union == 0 else float(
        np.logical_and(predicted, target).sum() / union)


def boundary_f1(predicted, target, tolerance=2):
    predicted, target = np.asarray(predicted, np.uint8), np.asarray(target, np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    predicted_boundary = predicted - cv2.erode(predicted, kernel)
    target_boundary = target - cv2.erode(target, kernel)
    dilation = np.ones((2 * tolerance + 1, 2 * tolerance + 1), np.uint8)
    predicted_near = cv2.dilate(predicted_boundary, dilation)
    target_near = cv2.dilate(target_boundary, dilation)
    predicted_count = np.count_nonzero(predicted_boundary)
    target_count = np.count_nonzero(target_boundary)
    precision = (np.count_nonzero(predicted_boundary & target_near)
                 / predicted_count) if predicted_count else float(target_count == 0)
    recall = (np.count_nonzero(target_boundary & predicted_near)
              / target_count) if target_count else float(predicted_count == 0)
    return 0.0 if precision + recall == 0 else float(
        2 * precision * recall / (precision + recall))


def attach_mask_quality(records, uncertainty_dir, gt_mask_dir):
    enriched = []
    for record in records:
        with np.load(uncertainty_dir / record["archive"]) as archive:
            predicted = np.asarray(archive["masks"][0], dtype=bool)
        gt_path = gt_mask_dir / "{:06d}.png".format(record["frame_index"])
        target = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if target is None:
            raise FileNotFoundError(gt_path)
        iou = mask_iou(predicted, target > 127)
        boundary = boundary_f1(predicted, target > 127)
        item = dict(record)
        item.update({
            "mask_iou": iou,
            "boundary_f1_tolerance2": boundary,
            "mask_failure_iou50": int(iou < 0.5),
            "mask_failure_iou90": int(iou < 0.9),
            "mask_failure_boundary_f1_70": int(boundary < 0.7),
        })
        enriched.append(item)
    return enriched


def select_first_observation_per_frame(records):
    """Keep the first causal observation and avoid duplicate frame weighting."""
    counts = Counter(int(row["frame_index"]) for row in records)
    selected = {}
    for row in sorted(records, key=lambda item: int(item["observation_index"])):
        frame_index = int(row["frame_index"])
        if frame_index not in selected:
            item = dict(row)
            item["observation_count_for_frame"] = counts[frame_index]
            item["frame_selection_policy"] = "first_observation"
            selected[frame_index] = item
    return [selected[index] for index in sorted(selected)]


def write_records(path, records):
    if path.suffix == ".jsonl":
        path.write_text("".join(
            json.dumps(row, sort_keys=True) + "\n" for row in records
        ))
        return
    fieldnames = list(records[0]) if records else [
        "frame_index", "failure_5deg_50mm"
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    uncertainty_dir = args.run_dir / "mask_uncertainty"
    source = uncertainty_dir / "mask_uncertainty.jsonl"
    diagnostics_path = args.run_dir / "diagnostics.csv"
    if not source.exists():
        print(json.dumps({"available": False, "reason": "no uncertainty log"}))
        return
    if not diagnostics_path.exists():
        raise FileNotFoundError(diagnostics_path)

    observations = load_jsonl(source)
    with diagnostics_path.open(newline="") as stream:
        diagnostics = list(csv.DictReader(stream))
    aligned = align_records(observations, diagnostics)
    manifest_path = args.run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if manifest.get("dataset") == "moped":
        workspace = Path(__file__).resolve().parents[2]
        gt_mask_dir = (workspace / "moped" / manifest["object"] /
                       "evaluation" / str(manifest["sequence"]) / "mask")
        aligned = attach_mask_quality(aligned, uncertainty_dir, gt_mask_dir)

    jsonl_path = uncertainty_dir / "mask_uncertainty_labeled.jsonl"
    csv_path = uncertainty_dir / "mask_uncertainty_labeled.csv"
    write_records(jsonl_path, aligned)
    write_records(csv_path, aligned)

    frame_records = select_first_observation_per_frame(aligned)
    identity = {
        name: manifest.get(name)
        for name in ("dataset", "object", "sequence", "seed")
    }
    frame_records = [dict(identity, **record) for record in frame_records]
    frame_jsonl_path = uncertainty_dir / "mask_uncertainty_frame_calibration.jsonl"
    frame_csv_path = uncertainty_dir / "mask_uncertainty_frame_calibration.csv"
    write_records(frame_jsonl_path, frame_records)
    write_records(frame_csv_path, frame_records)

    frame_counts = Counter(row["frame_index"] for row in aligned)
    summary = {
        "schema_version": 1,
        "available": True,
        "observations": len(aligned),
        "unique_frames": len(frame_counts),
        "duplicate_observations": sum(count - 1 for count in frame_counts.values()),
        "failure_observations": sum(row["failure_5deg_50mm"] for row in aligned),
        "failure_frames": len({
            row["frame_index"] for row in aligned if row["failure_5deg_50mm"]
        }),
        "mask_failure_frames": len({
            row["frame_index"] for row in aligned
            if row.get("mask_failure_iou50")
        }),
        "mask_quality_available": all("mask_iou" in row for row in aligned),
        "jsonl": jsonl_path.name,
        "csv": csv_path.name,
        "frame_selection_policy": "first_observation",
        "frame_calibration_jsonl": frame_jsonl_path.name,
        "frame_calibration_csv": frame_csv_path.name,
    }
    (uncertainty_dir / "label_alignment_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
