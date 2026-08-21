#!/usr/bin/env python3
"""Strict sequence-grouped nested LOSO evaluation for Stage-2 mask uncertainty."""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from mask_uncertainty import binary_auroc, expected_calibration_error


MODELS = {
    "candidate_only": ["candidate_disagreement"],
    "candidate_projection": ["candidate_disagreement", "projection_inconsistency"],
    "candidate_temporal": ["candidate_disagreement", "temporal_or_projection_inconsistency"],
    "candidate_boundary": ["candidate_disagreement", "boundary_entropy"],
}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def matrix(rows, features):
    return np.asarray([[np.nan if row.get(f) is None else float(row[f])
                        for f in features] for row in rows], dtype=float)


def labels(rows, label):
    return np.asarray([int(row[label]) for row in rows], dtype=float)


def fit(rows, features, label, l2, group_balanced=False):
    x, y = matrix(rows, features), labels(rows, label)
    medians = np.nanmedian(x, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    x = np.where(np.isnan(x), medians, x)
    means, scales = x.mean(axis=0), x.std(axis=0)
    scales[scales < 1e-8] = 1.0
    z = (x - means) / scales
    weights = np.ones(len(rows), dtype=float)
    if group_balanced:
        counts = {}
        for row in rows:
            key = (row["object"], row["sequence"])
            counts[key] = counts.get(key, 0) + 1
        weights = np.asarray([
            1.0 / counts[(row["object"], row["sequence"])] for row in rows
        ])
        weights /= weights.mean()

    def objective(p):
        probability = sigmoid(p[0] + z @ p[1:])
        per_sample = -(y * np.log(probability + 1e-12) +
                       (1-y) * np.log(1-probability + 1e-12))
        nll = np.average(per_sample, weights=weights)
        return nll + 0.5 * l2 * np.sum(p[1:] ** 2) / len(y)

    prevalence = np.clip(np.average(y, weights=weights), 1e-4, 1-1e-4)
    initial = np.zeros(len(features)+1)
    initial[0] = np.log(prevalence/(1-prevalence))
    result = minimize(objective, initial, method="BFGS")
    if not result.success and not np.isfinite(result.fun):
        raise RuntimeError(result.message)
    return dict(parameters=result.x, medians=medians, means=means, scales=scales)


def predict(rows, features, model):
    x = matrix(rows, features)
    x = np.where(np.isnan(x), model["medians"], x)
    return sigmoid(model["parameters"][0] +
                   ((x-model["means"])/model["scales"]) @ model["parameters"][1:])


def log_loss(y, probability):
    y, probability = np.asarray(y), np.clip(probability, 1e-12, 1-1e-12)
    return float(np.mean(-y*np.log(probability) - (1-y)*np.log(1-probability)))


def metrics(y, probability, bins):
    y, probability = np.asarray(y, dtype=int), np.asarray(probability)
    return {"samples": len(y), "failures": int(y.sum()),
            "auroc": binary_auroc(y, probability),
            "ece": expected_calibration_error(y, probability, bins=bins),
            "log_loss": log_loss(y, probability)}


def choose_l2(groups, features, label, candidates, group_balanced=False):
    scores = {}
    for value in candidates:
        losses = []
        for held_out, validation in groups.items():
            training = [row for key, rows in groups.items() if key != held_out
                        for row in rows]
            if len(set(labels(training, label))) < 2:
                continue
            model = fit(training, features, label, value, group_balanced)
            losses.append(log_loss(labels(validation, label),
                                   predict(validation, features, model)))
        scores[str(value)] = float(np.mean(losses)) if losses else None
    valid = [value for value in candidates if scores[str(value)] is not None]
    return min(valid, key=lambda value: scores[str(value)]), scores


def evaluate(groups, features, label, candidates, bins, group_balanced=False):
    predictions, folds = [], {}
    for held_out, test in sorted(groups.items()):
        training_groups = {key: rows for key, rows in groups.items() if key != held_out}
        selected, inner = choose_l2(
            training_groups, features, label, candidates, group_balanced
        )
        training = [row for rows in training_groups.values() for row in rows]
        model = fit(training, features, label, selected, group_balanced)
        probability = predict(test, features, model)
        y = labels(test, label).astype(int)
        folds[held_out] = {"selected_l2": selected, "inner_log_loss": inner,
                           "metrics": metrics(y, probability, bins)}
        for row, target, score in zip(test, y, probability):
            predictions.append({"object": row["object"], "sequence": row["sequence"],
                                "seed": int(row["seed"]), "frame_index": int(row["frame_index"]),
                                "failure_label": int(target),
                                "failure_probability": float(score),
                                "held_out_sequence": held_out})
    y = [row["failure_label"] for row in predictions]
    probability = [row["failure_probability"] for row in predictions]
    fold_aurocs = [fold["metrics"]["auroc"] for fold in folds.values()
                   if fold["metrics"]["auroc"] is not None]
    return {"features": features, "overall_oof": metrics(y, probability, bins),
            "mixed_sequence_macro_auroc": float(np.mean(fold_aurocs)),
            "mixed_sequence_count": len(fold_aurocs), "folds": folds}, predictions


def calibration_points(y, probability, bins):
    y, probability = np.asarray(y), np.asarray(probability)
    points = []
    for index in range(bins):
        low, high = index/bins, (index+1)/bins
        selected = (probability >= low) & (probability < high)
        if index == bins-1:
            selected |= probability == 1
        if selected.any():
            points.append((float(probability[selected].mean()), float(y[selected].mean())))
    return points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="mask_failure_iou90")
    parser.add_argument("--l2-grid", default="0.01,0.1,1,10,100")
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]
    groups = {}
    for row in rows:
        groups.setdefault(f"{row['object']}/{row['sequence']}", []).append(row)
    candidates = [float(value) for value in args.l2_grid.split(",")]
    results, predictions = {}, {}
    for name, features in MODELS.items():
        results[name], predictions[name] = evaluate(groups, features, args.label,
                                                     candidates, args.bins)
    summary = {"schema_version": 1, "stage": 2, "label": args.label,
               "method": "strict nested LOSO; all seeds of a sequence stay in one fold",
               "selection_metric": "inner sequence-macro log loss",
               "sequences": len(groups), "frames": len(rows), "models": results}
    (args.output_dir/"nested_loso_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)+"\n")
    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], "--", color="gray")
    for name, records in predictions.items():
        path = args.output_dir/f"{name}_oof_predictions.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
        points = calibration_points([r["failure_label"] for r in records],
                                    [r["failure_probability"] for r in records], args.bins)
        plt.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=name)
    plt.xlabel("predicted low-quality-mask probability")
    plt.ylabel("observed low-quality-mask frequency")
    plt.grid(alpha=.25); plt.legend(); plt.tight_layout()
    plt.savefig(args.output_dir/"calibration_curve.png", dpi=180); plt.close()
    print(json.dumps({name: result["overall_oof"] for name, result in results.items()}, indent=2))


if __name__ == "__main__":
    main()
