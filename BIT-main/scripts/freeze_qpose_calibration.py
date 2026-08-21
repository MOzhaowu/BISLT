#!/usr/bin/env python3
"""Freeze q_pose with inner-only feature-family/L2 selection and outer LOSO."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from fit_pose_uncertainty_nested_loso import MODELS
from fit_stage2_mask_nested_loso import (
    choose_l2, fit, labels, metrics, predict,
)


CANDIDATE_MODELS = (
    "initial_median_covariance_trace",
    "visibility_only",
    "initial_median_covariance_visibility",
    "initial_median_covariance_residual_spatial",
    "initial_median_covariance_object_independent",
    "initial_median_covariance_minimal_boundary",
)


def select_model(groups, l2_values, group_balanced):
    selection = {}
    for name in CANDIDATE_MODELS:
        selected_l2, scores = choose_l2(
            groups, MODELS[name], "failure_5deg_50mm", l2_values,
            group_balanced,
        )
        selection[name] = {
            "selected_l2": selected_l2,
            "selected_inner_log_loss": scores[str(selected_l2)],
            "inner_log_loss": scores,
        }
    selected_name = min(
        selection,
        key=lambda name: selection[name]["selected_inner_log_loss"],
    )
    return selected_name, selection[selected_name]["selected_l2"], selection


def outer_loso(groups, l2_values, bins, group_balanced):
    predictions, folds = [], {}
    for held_out, test in sorted(groups.items()):
        training_groups = {
            key: rows for key, rows in groups.items() if key != held_out
        }
        selected_name, selected_l2, inner = select_model(
            training_groups, l2_values, group_balanced)
        training = [
            row for rows in training_groups.values() for row in rows
        ]
        model = fit(
            training, MODELS[selected_name], "failure_5deg_50mm",
            selected_l2, group_balanced,
        )
        probability = predict(test, MODELS[selected_name], model)
        target = labels(test, "failure_5deg_50mm").astype(int)
        folds[held_out] = {
            "selected_model": selected_name,
            "selected_features": MODELS[selected_name],
            "selected_l2": selected_l2,
            "inner_selection": inner,
            "metrics": metrics(target, probability, bins),
        }
        for row, label, score in zip(test, target, probability):
            predictions.append({
                "object": row["object"],
                "sequence": row["sequence"],
                "seed": int(row["seed"]),
                "frame_index": int(row["frame_index"]),
                "failure_label": int(label),
                "failure_probability": float(score),
                "held_out_sequence": held_out,
                "selected_model": selected_name,
                "selected_l2": selected_l2,
            })
    target = [row["failure_label"] for row in predictions]
    probability = [row["failure_probability"] for row in predictions]
    fold_aurocs = [
        fold["metrics"]["auroc"] for fold in folds.values()
        if fold["metrics"]["auroc"] is not None
    ]
    return predictions, {
        "overall_oof": metrics(target, probability, bins),
        "sequence_macro_auroc": float(np.mean(fold_aurocs)),
        "mixed_sequence_count": len(fold_aurocs),
        "selected_model_counts": dict(Counter(
            fold["selected_model"] for fold in folds.values())),
        "folds": folds,
    }


def sequence_bootstrap(predictions, bins, replicates, seed):
    groups = {}
    for row in predictions:
        groups.setdefault(row["held_out_sequence"], []).append(row)
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    samples = {"auroc": [], "ece": [], "log_loss": []}
    for _ in range(replicates):
        selected = rng.choice(keys, size=len(keys), replace=True)
        rows = [row for key in selected for row in groups[key]]
        result = metrics(
            [row["failure_label"] for row in rows],
            [row["failure_probability"] for row in rows],
            bins,
        )
        for name in samples:
            if result[name] is not None:
                samples[name].append(result[name])
    return {
        "unit": "sequence",
        "replicates": replicates,
        "seed": seed,
        "confidence_intervals_95": {
            name: {
                "lower": float(np.percentile(values, 2.5)),
                "median": float(np.percentile(values, 50.0)),
                "upper": float(np.percentile(values, 97.5)),
            }
            for name, values in samples.items() if values
        },
    }


def serializable_model(model):
    return {
        name: value.tolist() if isinstance(value, np.ndarray) else value
        for name, value in model.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--l2-grid", default="0.01,0.1,1,10,100")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--include-warmup", action="store_true")
    parser.add_argument("--group-balanced", action="store_true")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260821)
    parser.add_argument("--implementation-commit", default="9f89c12")
    args = parser.parse_args()

    rows = [
        json.loads(line) for line in args.dataset.read_text().splitlines()
        if line.strip()
    ]
    if not args.include_warmup:
        rows = [row for row in rows if not row.get("calibration_warmup", 0)]
    groups = {}
    for row in rows:
        groups.setdefault(f"{row['object']}/{row['sequence']}", []).append(row)
    if len(groups) < 3:
        raise ValueError("Strict nested LOSO needs at least three sequences")
    l2_values = [float(value) for value in args.l2_grid.split(",")]

    predictions, evaluation = outer_loso(
        groups, l2_values, args.bins, args.group_balanced)
    evaluation["sequence_bootstrap"] = sequence_bootstrap(
        predictions, args.bins, args.bootstrap_replicates,
        args.bootstrap_seed,
    )
    selected_name, selected_l2, full_selection = select_model(
        groups, l2_values, args.group_balanced)
    final_model = fit(
        rows, MODELS[selected_name], "failure_5deg_50mm",
        selected_l2, args.group_balanced,
    )
    frozen = {
        "schema_version": 1,
        "stage": 3,
        "name": "q_pose",
        "status": "frozen",
        "implementation_commit": args.implementation_commit,
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "dataset_frames": len(rows),
        "dataset_sequences": len(groups),
        "grouping": "object/sequence; all seeds remain in the same fold",
        "selection": "feature family and L2 selected by inner LOSO only",
        "candidate_models": list(CANDIDATE_MODELS),
        "selected_model": selected_name,
        "selected_features": MODELS[selected_name],
        "selected_l2": selected_l2,
        "full_inner_selection": full_selection,
        "model": serializable_model(final_model),
        "evaluation": evaluation,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "q_pose_calibration_frozen.json").write_text(
        json.dumps(frozen, indent=2, ensure_ascii=False) + "\n")
    with (args.output_dir / "q_pose_outer_loso_predictions.csv").open(
            "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    print(json.dumps({
        "selected_model": selected_name,
        "selected_l2": selected_l2,
        **evaluation["overall_oof"],
        "sequence_macro_auroc": evaluation["sequence_macro_auroc"],
        "bootstrap_95": evaluation["sequence_bootstrap"][
            "confidence_intervals_95"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
