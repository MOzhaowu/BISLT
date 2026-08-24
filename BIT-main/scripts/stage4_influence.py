"""Pure utilities for first-order observation influence policies."""

import numpy as np


def cosine_alignments(validation_gradient, observation_gradients, epsilon=1e-12):
    """Return normalized g_val dot g_i scores.

    A positive score predicts a validation-loss decrease after a gradient step
    on the observation under the first-order approximation.
    """
    validation = np.asarray(validation_gradient, dtype=np.float64).ravel()
    validation_norm = np.linalg.norm(validation)
    output = []
    for gradient in observation_gradients:
        observation = np.asarray(gradient, dtype=np.float64).ravel()
        if observation.shape != validation.shape:
            raise ValueError("gradient shapes do not match")
        denominator = validation_norm*np.linalg.norm(observation)
        output.append(
            0.0 if denominator <= epsilon
            else float(np.dot(validation, observation)/denominator))
    return output


def influence_rescue_weights(rows, base_weights, alignments, margin=0.0):
    """Restore only P1-v2-demoted parent accepts with positive influence."""
    if not (len(rows) == len(base_weights) == len(alignments)):
        raise ValueError("rows, weights, and alignments must have equal length")
    output = []
    for row, base_weight, alignment in zip(rows, base_weights, alignments):
        eligible = (
            row is not None
            and row["baseline_decision"] == "accept"
            and row["decision"] == "downweight"
        )
        output.append(
            1.0 if eligible and float(alignment) > float(margin)
            else float(base_weight))
    return output


def gradient_matching_weights(
        validation_gradient, observation_gradients, base_weights,
        locked_zero=None, ridge=1.0, epsilon=1e-12):
    """Fit bounded weights whose normalized train gradient matches validation.

    Solves ridge least squares before clipping to [0, 1]. Parent-rejected
    observations may be fixed at zero through locked_zero.
    """
    validation = np.asarray(validation_gradient, dtype=np.float64).ravel()
    observations = [
        np.asarray(gradient, dtype=np.float64).ravel()
        for gradient in observation_gradients]
    if len(observations) != len(base_weights):
        raise ValueError("gradients and weights must have equal length")
    if any(gradient.shape != validation.shape for gradient in observations):
        raise ValueError("gradient shapes do not match")
    locked = ([False]*len(observations) if locked_zero is None
              else [bool(value) for value in locked_zero])
    if len(locked) != len(observations):
        raise ValueError("locked_zero and gradients must have equal length")
    output = np.zeros(len(observations), dtype=np.float64)
    active = [index for index, value in enumerate(locked) if not value]
    validation_norm = np.linalg.norm(validation)
    if not active or validation_norm <= epsilon:
        return [
            0.0 if locked[index] else float(base_weights[index])
            for index in range(len(observations))]
    target = validation/validation_norm
    columns = []
    for index in active:
        norm = np.linalg.norm(observations[index])
        columns.append(
            np.zeros_like(target) if norm <= epsilon
            else observations[index]/norm)
    matrix = np.stack(columns, axis=1)
    prior = np.asarray([base_weights[index] for index in active],
                       dtype=np.float64)
    system = matrix.T@matrix+float(ridge)*np.eye(len(active))
    rhs = matrix.T@target+float(ridge)*prior
    fitted = np.linalg.solve(system, rhs)
    output[active] = np.clip(fitted, 0.0, 1.0)
    return output.tolist()


def short_loo_softmax_weights(rows, marginal_utilities, epsilon=1e-8):
    """Map short-horizon leave-one-out utilities to robust relative weights."""
    if len(rows) != len(marginal_utilities):
        raise ValueError("rows and utilities must have equal length")
    locked = [
        row is not None and row["baseline_decision"] == "reject"
        for row in rows]
    active = [index for index, value in enumerate(locked) if not value]
    output = np.zeros(len(rows), dtype=np.float64)
    if not active:
        return output.tolist()
    values = np.asarray(
        [marginal_utilities[index] for index in active], dtype=np.float64)
    center = float(np.median(values))
    scale = max(float(np.median(np.abs(values-center))), float(epsilon))
    logits = np.clip((values-float(np.max(values)))/scale, -50.0, 0.0)
    output[active] = np.exp(logits)
    return output.tolist()

def short_loo_filter_weights(rows, marginal_utilities, margin=0.0):
    """Keep only observations with positive short-horizon marginal utility."""
    if len(rows) != len(marginal_utilities):
        raise ValueError("rows and utilities must have equal length")
    return [
        0.0 if (
            row is not None and row["baseline_decision"] == "reject"
        ) else float(float(utility) > float(margin))
        for row, utility in zip(rows, marginal_utilities)
    ]


def monotone_validation_commit(
        stable_metrics, candidate_metrics, min_iou_improvement=0.0,
        max_pose_regression=None, max_temporal_regression=None):
    """Decide whether a candidate may replace stable validation geometry."""
    stable_iou = float(stable_metrics["mean_iou_loss"])
    candidate_iou = float(candidate_metrics["mean_iou_loss"])
    if not all(np.isfinite(value) for value in (stable_iou, candidate_iou)):
        raise ValueError("validation losses must be finite")
    iou_improvement = stable_iou-candidate_iou
    checks = {
        "iou_monotone": iou_improvement >= float(min_iou_improvement),
    }
    if max_pose_regression is not None:
        checks["pose_guard"] = (
            float(candidate_metrics["pose_inconsistency"])
            - float(stable_metrics["pose_inconsistency"])
            <= float(max_pose_regression))
    if max_temporal_regression is not None:
        checks["temporal_guard"] = (
            float(candidate_metrics["temporal_std"])
            - float(stable_metrics["temporal_std"])
            <= float(max_temporal_regression))
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "committed": not failed,
        "reason": "trust_region_passed" if not failed else failed[0],
        "checks": {name: bool(value) for name, value in checks.items()},
        "actual_iou_improvement": iou_improvement,
    }

def select_lowest_validation_geometry(stable_metrics, proposals):
    """Choose the lowest finite validation-IoU geometry, preferring stable ties."""
    candidates = [("stable", stable_metrics), *proposals.items()]
    scored = []
    for priority, (name, metrics) in enumerate(candidates):
        loss = float(metrics["mean_iou_loss"])
        if not np.isfinite(loss):
            raise ValueError("validation losses must be finite")
        scored.append((loss, priority, name, metrics))
    loss, _, name, metrics = min(scored)
    stable_loss = float(stable_metrics["mean_iou_loss"])
    return {
        "selected_source": name,
        "selected_metrics": metrics,
        "selected_iou_loss": loss,
        "stable_iou_loss": stable_loss,
        "actual_iou_improvement": stable_loss-loss,
        "committed": name != "stable",
    }

def select_multimetric_validation_geometry(
        stable_metrics, full_metrics, loo_metrics=None,
        max_pose_regression=0.0, max_temporal_regression=0.0,
        epsilon=1e-12):
    """Select lowest-IoU geometry among candidates non-regressive vs full."""
    candidates = [("full", full_metrics), ("stable", stable_metrics)]
    if loo_metrics is not None:
        candidates.append(("loo", loo_metrics))
    feasible = []
    audit = {}
    full_pose = float(full_metrics["pose_inconsistency"])
    full_temporal = float(full_metrics["temporal_std"])
    for priority, (name, metrics) in enumerate(candidates):
        pose_regression = float(metrics["pose_inconsistency"])-full_pose
        temporal_regression = float(metrics["temporal_std"])-full_temporal
        passed = (
            pose_regression <= float(max_pose_regression)+float(epsilon)
            and temporal_regression <= float(max_temporal_regression)+float(epsilon)
        )
        audit[name] = {
            "feasible": bool(passed),
            "pose_regression_vs_full": pose_regression,
            "temporal_regression_vs_full": temporal_regression,
        }
        if passed:
            loss = float(metrics["mean_iou_loss"])
            if not np.isfinite(loss):
                raise ValueError("validation losses must be finite")
            feasible.append((loss, priority, name, metrics))
    loss, _, name, metrics = min(feasible)
    return {
        "selected_source": name,
        "selected_metrics": metrics,
        "selected_iou_loss": loss,
        "committed": name != "stable",
        "feasibility": audit,
    }
