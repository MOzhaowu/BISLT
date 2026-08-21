#!/usr/bin/env python3
"""Independent nested-LOSO calibration for Stage-3 pose uncertainty."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from fit_stage2_mask_nested_loso import evaluate


MODELS = {
    "normalized_min_eigenvalue": ["negative_log_normalized_hessian_min_eigenvalue"],
    "normalized_covariance_trace": ["log_normalized_covariance_trace"],
    "normalized_covariance_blocks": [
        "log_normalized_covariance_rotation_trace",
        "log_normalized_covariance_translation_trace"],
    "normalized_spectrum": [
        "negative_log_normalized_hessian_min_eigenvalue",
        "log_normalized_hessian_condition"],
    "normalized_covariance_contour": [
        "log_normalized_covariance_rotation_trace",
        "log_normalized_covariance_translation_trace", "contour_residual"],
    "relative_min_eigenvalue": [
        "negative_log_normalized_hessian_min_eigenvalue_relative_to_first"],
    "relative_covariance_trace": [
        "log_normalized_covariance_trace_relative_to_first"],
    "relative_covariance_blocks": [
        "log_normalized_covariance_rotation_trace_relative_to_first",
        "log_normalized_covariance_translation_trace_relative_to_first"],
    "relative_min_eigenvalue_contour": [
        "negative_log_normalized_hessian_min_eigenvalue_relative_to_first",
        "contour_residual"],
    "relative_covariance_contour": [
        "log_normalized_covariance_trace_relative_to_first",
        "contour_residual"],
    "initial_median_covariance_trace": [
        "log_normalized_covariance_trace_relative_to_initial_median"],
    "initial_median_covariance_blocks": [
        "log_normalized_covariance_rotation_trace_relative_to_initial_median",
        "log_normalized_covariance_translation_trace_relative_to_initial_median"],
    "initial_robust_z_covariance_trace": [
        "log_normalized_covariance_trace_initial_robust_z"],
    "initial_median_covariance_contour": [
        "log_normalized_covariance_trace_relative_to_initial_median",
        "contour_residual"],
    "initial_median_covariance_scale": [
        "log_normalized_covariance_trace_relative_to_initial_median",
        "log_object_characteristic_length"],
    "visibility_only": [
        "visible_boundary_ratio", "occlusion_ratio", "effective_contour_ratio",
        "silhouette_iou"],
    "initial_median_covariance_visibility": [
        "log_normalized_covariance_trace_relative_to_initial_median",
        "visible_boundary_ratio", "occlusion_ratio", "effective_contour_ratio",
        "silhouette_iou"],
    "initial_median_covariance_residual_spatial": [
        "log_normalized_covariance_trace_relative_to_initial_median",
        "contour_residual_stddev", "contour_residual_p90",
        "contour_residual_centroid_offset"],
    "initial_median_covariance_object_independent": [
        "log_normalized_covariance_trace_relative_to_initial_median",
        "visible_boundary_ratio", "occlusion_ratio", "effective_contour_ratio",
        "silhouette_iou", "contour_residual_stddev", "contour_residual_p90",
        "contour_residual_centroid_offset"],
    "initial_median_covariance_minimal_boundary": [
        "log_normalized_covariance_trace_relative_to_initial_median",
        "occlusion_ratio", "effective_contour_ratio"],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--l2-grid", default="0.01,0.1,1,10,100")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--include-warmup", action="store_true")
    parser.add_argument("--group-balanced", action="store_true",
                        help="give each training sequence equal total weight")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]
    if not args.include_warmup:
        rows = [row for row in rows if not row.get("calibration_warmup", 0)]
    groups = {}
    for row in rows:
        groups.setdefault(f"{row['object']}/{row['sequence']}", []).append(row)
    candidates = [float(value) for value in args.l2_grid.split(",")]
    results = {}
    for name, features in MODELS.items():
        result, predictions = evaluate(
            groups, features, "failure_5deg_50mm", candidates, args.bins,
            group_balanced=args.group_balanced,
        )
        path = args.output_dir/f"{name}_oof_predictions.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
            writer.writeheader(); writer.writerows(predictions)
        correlations = {}
        for feature in features:
            values = np.asarray([float(row[feature]) for row in rows])
            correlations[feature] = {
                "rotation_spearman": float(spearmanr(
                    values, [row["rotation_error_deg"] for row in rows]).statistic),
                "translation_spearman": float(spearmanr(
                    values, [row["translation_error_mm"] for row in rows]).statistic),
            }
        result["raw_feature_correlations"] = correlations
        results[name] = result
    summary = {"schema_version": 1, "stage": 3,
               "method": "strict nested LOSO L2 logistic calibration",
               "training_weighting": (
                   "equal total weight per sequence" if args.group_balanced
                   else "equal weight per frame"),
               "grouping": "object/sequence; all seeds in same fold",
               "frames": len(rows), "sequences": len(groups),
               "warmup_policy": "included" if args.include_warmup else "exclude first 5 frames per run", "models": results}
    (args.output_dir/"nested_loso_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps({name: result["overall_oof"]
                      for name, result in results.items()}, indent=2))


if __name__ == "__main__":
    main()
