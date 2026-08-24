#!/usr/bin/env python3
"""Evaluate fixed L2 logistic student heads with strict sequence LOSO."""

import argparse
import json
from pathlib import Path

import numpy as np


def sigmoid(values):
    values = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-values))


def fit_logistic(features, labels, l2=1.0, steps=2000, learning_rate=0.05):
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if x.ndim != 2 or len(x) != len(y):
        raise ValueError("invalid feature or label shape")
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2:
        raise ValueError("training fold must contain both classes")
    class_weight = {value: len(y) / (2.0 * count)
                    for value, count in zip(classes, counts)}
    sample_weight = np.asarray([class_weight[value] for value in y])
    weights = np.zeros(x.shape[1], dtype=np.float64)
    bias = 0.0
    for _ in range(int(steps)):
        probabilities = sigmoid(x @ weights + bias)
        residual = (probabilities - y) * sample_weight
        weights -= learning_rate * (
            x.T @ residual / len(y) + float(l2) * weights / len(y))
        bias -= learning_rate * float(np.mean(residual))
    return weights, bias


def fit_preprocessor(features):
    x = np.asarray(features, dtype=np.float64)
    missing = np.isnan(x)
    medians = np.asarray([
        float(np.nanmedian(x[:, column]))
        if not np.all(missing[:, column]) else 0.0
        for column in range(x.shape[1])])
    filled = np.where(missing, medians[None, :], x)
    means = filled.mean(axis=0)
    scales = filled.std(axis=0)
    scales[scales < 1e-12] = 1.0
    return medians, means, scales


def transform(features, preprocessor):
    x = np.asarray(features, dtype=np.float64)
    medians, means, scales = preprocessor
    return (np.where(np.isnan(x), medians[None, :], x) - means) / scales


def binary_auroc(labels, scores):
    y = np.asarray(labels, dtype=bool)
    score = np.asarray(scores, dtype=np.float64)
    positive = score[y]
    negative = score[~y]
    if not len(positive) or not len(negative):
        return None
    comparisons = positive[:, None] - negative[None, :]
    return float((np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0))
                 / comparisons.size)


def binary_metrics(labels, probabilities):
    y = np.asarray(labels, dtype=bool)
    p = np.asarray(probabilities, dtype=np.float64)
    predictions = p >= 0.5
    recalls = []
    for value in (False, True):
        mask = y == value
        recalls.append(float(np.mean(predictions[mask] == value)))
    return {
        "samples": len(y),
        "positives": int(np.sum(y)),
        "auroc": binary_auroc(y, p),
        "accuracy": float(np.mean(predictions == y)),
        "balanced_accuracy": float(np.mean(recalls)),
        "brier": float(np.mean((p - y.astype(float)) ** 2)),
    }


def event_feature_names(events):
    names = sorted(events[0]["event_features"])
    if any(sorted(event["event_features"]) != names for event in events):
        raise ValueError("inconsistent event feature schema")
    return names


def event_matrix(events, names):
    return [[
        np.nan if event["event_features"][name] is None
        else float(event["event_features"][name])
        for name in names] for event in events]


def observation_rows(events, event_names, frame_names):
    rows = []
    for event in events:
        event_values = event_matrix([event], event_names)[0]
        for observation in event["observations"]:
            frame = observation["features"] or {}
            rows.append({
                "group": event["group"],
                "identity": dict(event["identity"],
                                 frame_index=observation["frame_index"]),
                "features": event_values + [
                    float(observation["features_available"]),
                    float(observation["is_anchor"]),
                ] + [
                    np.nan if name not in frame else float(frame[name])
                    for name in frame_names],
                "label": bool(observation["teacher_keep"]),
            })
    return rows


def loso_predictions(rows, feature_names, l2, steps, learning_rate):
    groups = sorted({row["group"] for row in rows})
    if len(groups) < 2:
        raise ValueError("strict LOSO requires at least two groups")
    predictions = []
    folds = []
    for held_out in groups:
        train = [row for row in rows if row["group"] != held_out]
        test = [row for row in rows if row["group"] == held_out]
        train_x = [row["features"] for row in train]
        preprocessor = fit_preprocessor(train_x)
        train_scaled = transform(train_x, preprocessor)
        weights, bias = fit_logistic(
            train_scaled, [row["label"] for row in train],
            l2=l2, steps=steps, learning_rate=learning_rate)
        probabilities = sigmoid(
            transform([row["features"] for row in test], preprocessor)
            @ weights + bias)
        fold_rows = []
        for row, probability in zip(test, probabilities):
            output = dict(row)
            output.pop("features")
            output["probability"] = float(probability)
            fold_rows.append(output)
        predictions.extend(fold_rows)
        folds.append({
            "held_out_group": held_out,
            "train_samples": len(train),
            "test_metrics": binary_metrics(
                [row["label"] for row in fold_rows],
                [row["probability"] for row in fold_rows]),
        })
    return predictions, folds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("teacher_dataset", type=Path)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    args = parser.parse_args()
    events = [
        json.loads(line) for line in args.teacher_dataset.read_text().splitlines()
        if line.strip()]
    protocol = json.loads(args.protocol.read_text())
    locked = set(protocol["teacher"]["locked_report_only_seeds"])
    if locked.intersection(
            event["identity"]["seed"] for event in events):
        raise ValueError("locked independent seed entered student fitting")

    event_names = event_feature_names(events)
    frame_names = list(protocol["online_numeric_frame_features"])
    observation_feature_names = (
        [f"event:{name}" for name in event_names]
        + ["observation:features_available", "observation:is_anchor"]
        + [f"observation:{name}" for name in frame_names])
    observation_data = observation_rows(events, event_names, frame_names)
    observation_predictions, observation_folds = loso_predictions(
        observation_data, observation_feature_names,
        args.l2, args.steps, args.learning_rate)

    event_data = []
    matrix = event_matrix(events, event_names)
    for event, features in zip(events, matrix):
        event_data.append({
            "group": event["group"],
            "identity": event["identity"],
            "features": features,
            "label": bool(event["teacher_labels"]["full_harmful_vs_stable"]),
        })
    event_predictions, event_folds = loso_predictions(
        event_data, event_names, args.l2, args.steps, args.learning_rate)

    result = {
        "schema_version": 1,
        "stage": "stage4_p3",
        "mode": "fixed_logistic_strict_sequence_loso_baseline",
        "data_role": "development_only",
        "hyperparameters": {
            "l2": args.l2,
            "steps": args.steps,
            "learning_rate": args.learning_rate,
        },
        "observation_utility_head": {
            "feature_names": observation_feature_names,
            "folds": observation_folds,
            "oof_metrics": binary_metrics(
                [row["label"] for row in observation_predictions],
                [row["probability"] for row in observation_predictions]),
            "oof_predictions": observation_predictions,
        },
        "event_full_harm_head": {
            "feature_names": event_names,
            "folds": event_folds,
            "oof_metrics": binary_metrics(
                [row["label"] for row in event_predictions],
                [row["probability"] for row in event_predictions]),
            "oof_predictions": event_predictions,
        },
        "limitations": [
            "only two object/sequence groups are available",
            "fixed regularization is used because nested LOSO is not identifiable",
            "seed 68-70 are excluded and may not be used for tuning",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "observation_utility": result["observation_utility_head"]["oof_metrics"],
        "event_full_harm": result["event_full_harm_head"]["oof_metrics"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
