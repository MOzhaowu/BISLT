"""Differentiable regularizers for Stage-4 geometry candidates."""

import math

import torch


def cosine_similarity(left, right, epsilon=1e-12):
    denominator = left.norm() * right.norm()
    if float(denominator.detach().item()) <= epsilon:
        return 0.0
    return float((torch.dot(left, right) / denominator).detach().item())


def project_direction_to_halfspaces(direction, normals, cycles=20,
                                    epsilon=1e-12):
    """Project a step direction onto intersections normal.dot(step) >= 0.

    Dykstra's cyclic projections return the Euclidean projection onto the
    intersection, rather than a single-order PCGrad approximation.
    """
    usable = [
        normal for normal in normals
        if float(normal.square().sum().detach().item()) > epsilon
    ]
    if not usable:
        return direction.clone()
    projected = direction.clone()
    residuals = [torch.zeros_like(direction) for _ in usable]
    for _ in range(int(cycles)):
        for index, normal in enumerate(usable):
            candidate = projected + residuals[index]
            dot = torch.dot(candidate, normal)
            correction = torch.clamp(-dot, min=0.0) / (
                normal.square().sum() + epsilon)
            updated = candidate + correction * normal
            residuals[index] = candidate - updated
            projected = updated
    return projected


def per_frame_iou_losses(predictions, targets):
    """Return differentiable silhouette IoU loss for every frame."""
    dims = tuple(range(1, predictions.ndimension()))
    intersection = (predictions * targets).sum(dims)
    union = (
        predictions + targets - predictions * targets
    ).sum(dims) + 1e-6
    return 1.0 - intersection / union


def weighted_mean(values, weights):
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


def candidate_regularization_terms(nominal_losses, perturbed_losses, weights):
    """Return gate-aligned pose inconsistency and cross-frame stability."""
    if nominal_losses.ndimension() != 1:
        raise ValueError("nominal_losses must be one-dimensional")
    if weights.shape != nominal_losses.shape:
        raise ValueError("weights must match nominal_losses")
    mean = weighted_mean(nominal_losses, weights)
    variance = weighted_mean((nominal_losses - mean).square(), weights)
    if nominal_losses.numel() < 2:
        temporal_std = nominal_losses.new_zeros(())
    else:
        temporal_std = torch.sqrt(variance.clamp_min(0.0) + 1e-12)

    if perturbed_losses is None:
        pose_inconsistency = nominal_losses.new_zeros(())
    else:
        if (perturbed_losses.ndimension() != 2
                or perturbed_losses.shape[0] != nominal_losses.shape[0]):
            raise ValueError(
                "perturbed_losses must have shape [frames, perturbations]")
        correction_gain = torch.relu(
            nominal_losses - perturbed_losses.min(dim=1).values)
        pose_inconsistency = weighted_mean(correction_gain, weights)
    return pose_inconsistency, temporal_std


def regularized_candidate_loss(base_iou_loss, laplacian_loss,
                               pose_inconsistency, temporal_std,
                               pose_weight=0.0, temporal_weight=0.0,
                               laplacian_weight=0.1):
    return (
        base_iou_loss
        + float(laplacian_weight) * laplacian_loss
        + float(pose_weight) * pose_inconsistency
        + float(temporal_weight) * temporal_std
    )


def se3_axis_perturbations(rotation_delta_deg, translation_delta,
                           device, dtype):
    """Build the same 12 signed local SE(3) probes used by gate evaluation."""
    angle = math.radians(float(rotation_delta_deg))
    c, s = math.cos(angle), math.sin(angle)
    deltas = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            delta = torch.eye(4, device=device, dtype=dtype)
            i, j = (1, 2) if axis == 0 else ((0, 2) if axis == 1 else (0, 1))
            signed_s = sign * s
            delta[i, i], delta[j, j] = c, c
            delta[i, j], delta[j, i] = -signed_s, signed_s
            deltas.append(delta)
    for axis in range(3):
        for sign in (-1.0, 1.0):
            delta = torch.eye(4, device=device, dtype=dtype)
            delta[axis, 3] = sign * float(translation_delta)
            deltas.append(delta)
    return torch.stack(deltas)
