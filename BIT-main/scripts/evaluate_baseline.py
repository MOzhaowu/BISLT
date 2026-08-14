#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path

import numpy as np


def load_poses(path):
    rows = []
    for line in path.read_text().splitlines():
        values = [float(value) for value in line.split()]
        if len(values) != 13:
            raise ValueError(f"Expected 13 columns in {path}, got {len(values)}")
        index = int(values[0])
        rotation = np.asarray(values[1:10], dtype=np.float64).reshape(3, 3)
        translation = np.asarray(values[10:13], dtype=np.float64)
        rows.append((index, rotation, translation))
    return rows

def project_rotation(matrix):
    """Remove the uniform model scale embedded in BIT's saved 3x3 blocks."""
    u, _, vt = np.linalg.svd(matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def log_metric(text, name):
    match = re.findall(rf"{re.escape(name)}:\s*([-+0-9.eE]+)", text)
    return float(match[-1]) if match else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--object", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--expected-frames", type=int, required=True)
    args = parser.parse_args()

    raw_dir = args.run_dir / "raw"
    gt_rows = load_poses(raw_dir / "gt.txt")
    pose_rows = load_poses(raw_dir / "pose.txt")
    if len(gt_rows) != len(pose_rows):
        raise ValueError("GT and prediction frame counts differ")

    rotation_errors = []
    translation_errors_mm = []
    for gt, pred in zip(gt_rows, pose_rows):
        if gt[0] != pred[0]:
            raise ValueError(f"Frame index mismatch: {gt[0]} != {pred[0]}")
        gt_rotation = project_rotation(gt[1])
        pred_rotation = project_rotation(pred[1])
        relative_rotation = pred_rotation @ gt_rotation.T
        cosine = np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0)
        rotation_errors.append(math.degrees(math.acos(cosine)))
        translation_errors_mm.append(float(np.linalg.norm(pred[2] - gt[2]) * 1000.0))

    rotation_errors = np.asarray(rotation_errors)
    translation_errors_mm = np.asarray(translation_errors_mm)
    success = (rotation_errors <= 5.0) & (translation_errors_mm <= 50.0)
    log_text = (args.run_dir / "run.log").read_text(errors="replace")

    metrics = {
        "dataset": "moped",
        "object": args.object,
        "sequence": args.sequence,
        "expected_frames": args.expected_frames,
        "evaluated_frames": len(gt_rows),
        "complete": len(gt_rows) == args.expected_frames,
        "add_mean_mm": log_metric(log_text, "ADD Error in mm"),
        "add_auc_100mm": log_metric(log_text, "ADD AUC Score in mm"),
        "rotation_error_deg": {
            "mean": float(rotation_errors.mean()),
            "median": float(np.median(rotation_errors)),
            "p90": float(np.percentile(rotation_errors, 90)),
        },
        "translation_error_mm": {
            "mean": float(translation_errors_mm.mean()),
            "median": float(np.median(translation_errors_mm)),
            "p90": float(np.percentile(translation_errors_mm, 90)),
        },
        "success_5deg_50mm": float(success.mean()),
    }
    (args.run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
