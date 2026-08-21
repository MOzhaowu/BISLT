#!/usr/bin/env python3
"""Create dimensionless SE(3) Hessians and unit-noise PSD covariances."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def finite(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite value: {value}")
    return number


def matrix_fields(prefix, matrix):
    return {f"{prefix}_{row}{col}": float(matrix[row, col])
            for row in range(6) for col in range(6)}


def normalize_hessian(row, relative_floor):
    hessian = np.asarray([[finite(row[f"hessian_{r}{c}"]) for c in range(6)]
                          for r in range(6)], dtype=np.float64)
    hessian = 0.5 * (hessian + hessian.T)
    length = finite(row["object_characteristic_length"])
    if length <= 0:
        raise ValueError(f"Invalid characteristic length: {length}")
    # xi=[omega, translation], eta=[omega, translation/L], xi=A*eta.
    # H_eta=A^T H_xi A puts translation and rotation in dimensionless units.
    transform = np.diag([1.0, 1.0, 1.0, length, length, length])
    normalized = transform.T @ hessian @ transform
    normalized = 0.5 * (normalized + normalized.T)
    eigenvalues, eigenvectors = np.linalg.eigh(normalized)
    maximum = max(float(eigenvalues[-1]), 0.0)
    floor = max(maximum * relative_floor, 1e-12)
    clipped = np.maximum(eigenvalues, floor)
    unit_covariance = (eigenvectors * (1.0 / clipped)) @ eigenvectors.T
    unit_covariance = 0.5 * (unit_covariance + unit_covariance.T)
    noise_scale = finite(row.get("contour_noise_variance", 1.0))
    noise_scale = max(noise_scale, 1e-12)
    covariance = unit_covariance * noise_scale
    result = {
        "object_characteristic_length": length,
        "normalized_hessian_min_eigenvalue": float(eigenvalues[0]),
        "normalized_hessian_max_eigenvalue": float(eigenvalues[-1]),
        "normalized_hessian_condition": float(clipped[-1] / clipped[0]),
        "normalized_covariance_trace": float(np.trace(covariance)),
        "normalized_covariance_rotation_trace": float(np.trace(covariance[:3, :3])),
        "normalized_covariance_translation_trace": float(np.trace(covariance[3:, 3:])),
        "normalized_covariance_cross_frobenius": float(np.linalg.norm(covariance[:3, 3:])),
        "psd_eigenvalue_floor": floor,
        "psd_clipped_eigenvalues": int(np.sum(eigenvalues < floor)),
        "covariance_noise_scale": noise_scale,
        "covariance_semantics": "dimensionless_SE3_Bernoulli_residual_scaled",
    }
    result.update(matrix_fields("normalized_hessian", normalized))
    result.update(matrix_fields("normalized_covariance", covariance))
    result.update(matrix_fields("normalized_unit_covariance", unit_covariance))
    for name in ("normalized_hessian_condition", "normalized_covariance_trace",
                 "normalized_covariance_rotation_trace",
                 "normalized_covariance_translation_trace"):
        result[f"log_{name}"] = math.log(max(result[name], 1e-300))
    result["negative_log_normalized_hessian_min_eigenvalue"] = -math.log(
        max(result["normalized_hessian_min_eigenvalue"], floor))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--relative-eigenvalue-floor", type=float, default=1e-6)
    parser.add_argument("--run-glob", default="*/*/hessian_s*/diagnostics.csv")
    args = parser.parse_args()
    records, run_count = [], 0
    for path in sorted(args.run_root.glob(args.run_glob)):
        manifest = json.loads((path.parent / "manifest.json").read_text())
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        required = {"object_characteristic_length", "hessian_00", "hessian_55"}
        if not rows or not required.issubset(rows[0]):
            raise ValueError(f"Run predates full-Hessian capture: {path}")
        run_count += 1
        first_features = None
        for row in rows:
            success = row["success_5deg_50mm"].lower() in {"1", "true"}
            record = {
                "dataset": manifest.get("dataset"), "object": manifest["object"],
                "sequence": str(manifest["sequence"]), "seed": int(manifest["seed"]),
                "frame_index": int(row["frame_index"]),
                "success_5deg_50mm": int(success), "failure_5deg_50mm": int(not success),
                "rotation_error_deg": finite(row["rotation_error_deg"]),
                "translation_error_mm": finite(row["translation_error_mm"]),
                "contour_residual": finite(row["contour_residual"]),
                "histogram_separation": finite(row["histogram_separation"]),
                "contour_noise_variance": finite(row["contour_noise_variance"]),
                "pose_teacher_valid": int(row["pose_teacher_valid"] in {"1", "true", "True"}),
                "pose_teacher_probes": int(row["pose_teacher_probes"]),
                "pose_teacher_failure_rate": (
                    finite(row["pose_teacher_failure_rate"])
                    if row["pose_teacher_valid"] in {"1", "true", "True"} else None),
                "pose_teacher_rotation_rms_deg": (finite(row["pose_teacher_rotation_rms_deg"]) if row["pose_teacher_valid"] in {"1", "true", "True"} else None),
                "pose_teacher_translation_rms_mm": (finite(row["pose_teacher_translation_rms_mm"])
                    if row["pose_teacher_valid"] in {"1", "true", "True"} else None),
            }
            normalized = normalize_hessian(row, args.relative_eigenvalue_floor)
            record.update({
                "pose_teacher_ablation_probes": int(row.get("pose_teacher_ablation_probes", 0)),
                "pose_teacher_axis_failure_rate_small": (finite(row["pose_teacher_axis_failure_rate_small"])
                    if row.get("pose_teacher_ablation_probes", "0") != "0" else None),
                "pose_teacher_axis_failure_rate_medium": (finite(row["pose_teacher_axis_failure_rate_medium"])
                    if row.get("pose_teacher_ablation_probes", "0") != "0" else None),
                "pose_teacher_axis_failure_rate_large": (finite(row["pose_teacher_axis_failure_rate_large"])
                    if row.get("pose_teacher_ablation_probes", "0") != "0" else None),
                "pose_teacher_joint_failure_rate_small": (finite(row["pose_teacher_joint_failure_rate_small"])
                    if row.get("pose_teacher_ablation_probes", "0") != "0" else None),
                "pose_teacher_joint_failure_rate_medium": (finite(row["pose_teacher_joint_failure_rate_medium"])
                    if row.get("pose_teacher_ablation_probes", "0") != "0" else None),
                "pose_teacher_joint_failure_rate_large": (finite(row["pose_teacher_joint_failure_rate_large"])
                    if row.get("pose_teacher_ablation_probes", "0") != "0" else None),
            })
            relative_names = (
                "log_normalized_covariance_trace",
                "log_normalized_covariance_rotation_trace",
                "log_normalized_covariance_translation_trace",
                "negative_log_normalized_hessian_min_eigenvalue",
                "log_normalized_hessian_condition",
            )
            if first_features is None:
                first_features = {name: normalized[name] for name in relative_names}
            for name in relative_names:
                normalized[name + "_relative_to_first"] = (
                    normalized[name] - first_features[name])
            record.update(normalized)
            records.append(record)
    if not records:
        raise ValueError(f"No full-Hessian diagnostics below {args.run_root}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    initial_feature_names = (
        "log_normalized_covariance_trace",
        "log_normalized_covariance_rotation_trace",
        "log_normalized_covariance_translation_trace",
        "negative_log_normalized_hessian_min_eigenvalue",
    )
    by_run = {}
    for record in records:
        by_run.setdefault((record["object"], record["sequence"], record["seed"]), []).append(record)
    for run_records in by_run.values():
        run_records.sort(key=lambda item: item["frame_index"])
        initial = run_records[:min(5, len(run_records))]
        medians = {name: float(np.median([row[name] for row in initial]))
                   for name in initial_feature_names}
        deviations = {name: max(float(np.median([
            abs(row[name] - medians[name]) for row in initial])), 1e-6)
                      for name in initial_feature_names}
        for position, record in enumerate(run_records):
            record["calibration_warmup"] = int(position < len(initial))
            for name in initial_feature_names:
                relative = record[name] - medians[name]
                record[name + "_relative_to_initial_median"] = relative
                record[name + "_initial_robust_z"] = relative / deviations[name]
            record["log_object_characteristic_length"] = math.log(
                record["object_characteristic_length"])
    args.output.write_text("".join(json.dumps(row, sort_keys=True)+"\n" for row in records))
    summary = {"schema_version": 1, "stage": 3, "runs": run_count,
               "sequences": len({(r['object'], r['sequence']) for r in records}),
               "frames": len(records), "failures": sum(r["failure_5deg_50mm"] for r in records),
               "relative_eigenvalue_floor": args.relative_eigenvalue_floor,
               "coordinate_order": ["rx", "ry", "rz", "tx", "ty", "tz"],
               "normalization": "eta=[omega, translation/characteristic_length]",
               "covariance": "inverse of eigenvalue-clipped normalized Hessian; Bernoulli residual scaled",
               "calibration_warmup_frames_per_run": 5}
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
