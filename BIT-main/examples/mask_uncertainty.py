"""Mask uncertainty features and calibration metrics for U-BIT stage 2."""

import numpy as np


def _iou(first, second):
    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)
    union = np.logical_or(first, second).sum()
    return 1.0 if union == 0 else float(np.logical_and(first, second).sum() / union)


def candidate_disagreement(masks):
    masks = np.asarray(masks, dtype=bool)
    if len(masks) < 2:
        return 0.0
    distances = [
        1.0 - _iou(masks[i], masks[j])
        for i in range(len(masks)) for j in range(i + 1, len(masks))
    ]
    return float(np.mean(distances))


def boundary_entropy(logits):
    if logits is None:
        return None
    values = np.asarray(logits, dtype=np.float64)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))
    entropy = -probabilities * np.log(probabilities + 1e-12)
    entropy -= (1.0 - probabilities) * np.log(1.0 - probabilities + 1e-12)
    boundary = np.abs(probabilities - 0.5) <= 0.25
    selected = entropy[boundary] if np.any(boundary) else entropy.reshape(-1)
    return float(np.mean(selected) / np.log(2.0))


def compute_mask_uncertainty(masks, scores, logits=None, previous_mask=None,
                             projected_mask=None, selected_index=None):
    masks = np.asarray(masks, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    best_index = int(np.argmax(scores))
    selected_index = best_index if selected_index is None else int(selected_index)
    selected_mask = masks[selected_index]
    features = {
        "sam_score": float(scores[selected_index]),
        "candidate_disagreement": candidate_disagreement(masks),
        "boundary_entropy": boundary_entropy(None if logits is None else logits[selected_index]),
        "temporal_inconsistency": None if previous_mask is None else 1.0 - _iou(selected_mask, previous_mask),
        "projection_inconsistency": None if projected_mask is None else 1.0 - _iou(selected_mask, projected_mask),
    }
    features["temporal_missing"] = int(previous_mask is None)
    features["temporal_or_projection_inconsistency"] = (
        features["temporal_inconsistency"]
        if features["temporal_inconsistency"] is not None
        else features["projection_inconsistency"]
    )
    weighted = []
    for name, weight in (
        ("candidate_disagreement", 0.30),
        ("boundary_entropy", 0.25),
        ("temporal_inconsistency", 0.25),
        ("projection_inconsistency", 0.20),
    ):
        if features[name] is not None:
            weighted.append((weight, features[name]))
    uncertainty = sum(weight * value for weight, value in weighted) / sum(weight for weight, _ in weighted)
    features["uncertainty"] = float(np.clip(uncertainty, 0.0, 1.0))
    features["confidence"] = 1.0 - features["uncertainty"]
    features["best_index"] = best_index
    features["selected_index"] = selected_index
    return features, selected_mask


def binary_auroc(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if len(positive) == 0 or len(negative) == 0:
        return None
    comparisons = (positive[:, None] > negative[None, :]).mean()
    ties = (positive[:, None] == negative[None, :]).mean()
    return float(comparisons + 0.5 * ties)


def expected_calibration_error(labels, probabilities, bins=10):
    labels = np.asarray(labels, dtype=np.float64)
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        selected = (probabilities >= edges[index]) & (probabilities < edges[index + 1])
        if index == bins - 1:
            selected |= probabilities == 1.0
        if np.any(selected):
            error += selected.mean() * abs(probabilities[selected].mean() - labels[selected].mean())
    return float(error)
