#!/usr/bin/env python3
"""LOSO-calibrated SE(3) confidence-ellipsoid coverage and runtime overhead."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import chi2

from build_normalized_pose_uncertainty_dataset import normalize_hessian
from evaluate_baseline import load_poses, project_rotation


LEVELS = (0.50, 0.80, 0.90, 0.95)


def se3_error(gt, pred, length):
    """Left-invariant log(T_pred T_gt^-1), in eta=[omega, rho/L]."""
    gt_rotation, gt_translation = project_rotation(gt[1]), gt[2]
    pred_rotation, pred_translation = project_rotation(pred[1]), pred[2]
    relative_rotation = pred_rotation @ gt_rotation.T
    relative_translation = pred_translation - relative_rotation @ gt_translation
    omega = Rotation.from_matrix(relative_rotation).as_rotvec()
    theta = np.linalg.norm(omega)
    skew = np.asarray([
        [0.0, -omega[2], omega[1]],
        [omega[2], 0.0, -omega[0]],
        [-omega[1], omega[0], 0.0],
    ])
    if theta < 1e-8:
        left_jacobian = np.eye(3) + 0.5 * skew + (skew @ skew) / 6.0
    else:
        left_jacobian = (
            np.eye(3)
            + (1.0 - np.cos(theta)) / theta**2 * skew
            + (theta - np.sin(theta)) / theta**3 * (skew @ skew)
        )
    rho = np.linalg.solve(left_jacobian, relative_translation)
    return np.concatenate((omega, rho / length))


def load_records(run_root, run_glob, relative_floor):
    records = []
    for path in sorted(run_root.glob(run_glob)):
        manifest = json.loads((path.parent / "manifest.json").read_text())
        gt_rows = {row[0]: row for row in load_poses(path.parent / "raw" / "gt.txt")}
        pred_rows = {row[0]: row for row in load_poses(path.parent / "raw" / "pose.txt")}
        with path.open(newline="") as stream:
            diagnostics = list(csv.DictReader(stream))
        group = f"{manifest['object']}/{manifest['sequence']}"
        for row in diagnostics:
            if row.get("pose_hessian_valid", "").lower() not in {"1", "true"}:
                continue
            frame = int(row["frame_index"])
            normalized = normalize_hessian(row, relative_floor)
            covariance = np.asarray([
                [normalized[f"normalized_covariance_{r}{c}"] for c in range(6)]
                for r in range(6)
            ])
            error = se3_error(
                gt_rows[frame], pred_rows[frame],
                normalized["object_characteristic_length"],
            )
            squared_distance = float(error @ np.linalg.pinv(covariance) @ error)
            records.append({
                "group": group,
                "seed": int(manifest["seed"]),
                "frame_index": frame,
                "squared_distance": squared_distance,
                "tracking_time_ms": float(row["tracking_time_ms"]),
                "pose_diagnostics_time_ms": float(row["pose_diagnostics_time_ms"]),
                "teacher_frame": row.get(
                    "pose_teacher_valid", "").lower() in {"1", "true"},
                "hessian_diagnostics_time_ms": float(
                    row["hessian_diagnostics_time_ms"]),
            })
    if not records:
        raise ValueError(f"No valid Hessian records below {run_root}")
    return records


def coverage(records, distance_key):
    values = np.asarray([row[distance_key] for row in records])
    return {
        str(level): float(np.mean(values <= chi2.ppf(level, df=6)))
        for level in LEVELS
    }


def loso_scale(records):
    groups = sorted({row["group"] for row in records})
    output, folds = [], {}
    target_threshold = chi2.ppf(0.95, df=6)
    for held_out in groups:
        training = [
            row["squared_distance"] for row in records
            if row["group"] != held_out
        ]
        scale = max(float(np.quantile(training, 0.95) / target_threshold), 1e-12)
        test = [row for row in records if row["group"] == held_out]
        folds[held_out] = {
            "training_scale": scale,
            "test_frames": len(test),
        }
        for row in test:
            output.append({
                **row,
                "calibrated_squared_distance": row["squared_distance"] / scale,
            })
    return output, folds


def bootstrap(records, value_function, replicates, seed):
    groups = {}
    for row in records:
        groups.setdefault(row["group"], []).append(row)
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        chosen = rng.choice(keys, size=len(keys), replace=True)
        sample = [row for key in chosen for row in groups[key]]
        values.append(value_function(sample))
    values = np.asarray(values)
    return {
        "lower": float(np.percentile(values, 2.5)),
        "median": float(np.percentile(values, 50.0)),
        "upper": float(np.percentile(values, 97.5)),
    }


def overhead(records, field):
    numerator = sum(row[field] for row in records)
    denominator = sum(row["tracking_time_ms"] for row in records)
    return numerator / denominator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--run-glob", default="*/*/qpose_freeze_s*/diagnostics.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--relative-eigenvalue-floor", type=float, default=1e-6)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260821)
    args = parser.parse_args()

    records = load_records(
        args.run_root, args.run_glob, args.relative_eigenvalue_floor)
    calibrated, folds = loso_scale(records)
    calibrated_coverage = coverage(calibrated, "calibrated_squared_distance")
    uncalibrated_coverage = coverage(records, "squared_distance")
    coverage_ci = {}
    for level in LEVELS:
        threshold = chi2.ppf(level, df=6)
        coverage_ci[str(level)] = bootstrap(
            calibrated,
            lambda sample, threshold=threshold: float(np.mean([
                row["calibrated_squared_distance"] <= threshold for row in sample
            ])),
            args.bootstrap_replicates,
            args.bootstrap_seed + int(level * 100),
        )
    production_records = [row for row in records if not row["teacher_frame"]]
    overhead_result = {}
    for scope, scope_records in (
            ("production_frames_excluding_teacher", production_records),
            ("all_frames_including_teacher", records)):
        overhead_result[scope] = {"frames": len(scope_records)}
        for field in ("hessian_diagnostics_time_ms", "pose_diagnostics_time_ms"):
            overhead_result[scope][field] = {
                "fraction_of_tracking_time": overhead(scope_records, field),
                "mean_ms_per_frame": float(np.mean([
                    row[field] for row in scope_records])),
                "sequence_bootstrap_95": bootstrap(
                    scope_records,
                    lambda sample, field=field: overhead(sample, field),
                    args.bootstrap_replicates,
                    args.bootstrap_seed,
                ),
            }
    result = {
        "schema_version": 1,
        "stage": 3,
        "frames": len(records),
        "sequences": len({row["group"] for row in records}),
        "seeds": sorted({row["seed"] for row in records}),
        "coordinate_semantics": (
            "left-invariant log(T_pred*T_gt^-1), "
            "eta=[rotation_vector, translation_log/object_length]"
        ),
        "covariance_semantics": (
            "PSD inverse normalized Hessian, contour-residual-noise scaled"
        ),
        "scale_calibration": (
            "outer LOSO; training 95th Mahalanobis quantile matched to chi2(df=6)"
        ),
        "uncalibrated_coverage": uncalibrated_coverage,
        "loso_calibrated_coverage": calibrated_coverage,
        "coverage_sequence_bootstrap_95": coverage_ci,
        "folds": folds,
        "runtime_overhead": overhead_result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
