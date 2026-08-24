#!/usr/bin/env python3
"""Re-optimize recorded candidates under frozen Stage-4 observation policies."""

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "examples"))
sys.path.insert(0, str(PROJECT / "scripts"))

import soft_renderer as sr
from replay_stage4_p2_observation_policies import (
    attach_parent_weights,
    load_csv,
    load_jsonl,
    policy_weight,
    split_batches,
)
from run_example import Model, evaluate_model_gate_metrics
from sam_utils import LoadConfigSafety
from stage4_candidate_regularization import (
    candidate_regularization_terms,
    cosine_similarity,
    per_frame_iou_losses,
    project_direction_to_halfspaces,
    regularized_candidate_loss,
    se3_axis_perturbations,
    weighted_mean,
)
from stage4_influence import (
    cosine_alignments,
    gradient_matching_weights,
    influence_rescue_weights,
    monotone_validation_commit,
    select_lowest_validation_geometry,
    short_loo_filter_weights,
    short_loo_softmax_weights,
)
from validation_selection import pose_signature


POLICIES = ("original", "p1v2_soft", "p1v2_soft_floor50",
            "p1v2_hard_nonaccept", "p1v2_influence_rescue",
            "p1v2_gradient_match", "p1v2_short_loo_softmax",
            "p1v2_short_loo_trust_region",
            "p1v2_short_loo_dual_trust_region",
            "p1v2_short_loo_dual_regularized")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def training_target(mask, size):
    mask = np.flip(np.asarray(mask), axis=0)
    height, width = mask.shape[-2:]
    if width > height:
        mask = np.pad(mask, ((width-height, 0), (0, 0)))
    return cv2.resize(mask, (size, size)).astype(np.float32)/255.0


def weighted_iou_loss(prediction, target, weights):
    dims = tuple(range(1, prediction.ndimension()))
    intersection = (prediction*target).sum(dims)
    union = (prediction+target-prediction*target).sum(dims)+1e-6
    losses = 1.0-intersection/union
    return (losses*weights).sum()/weights.sum().clamp_min(1e-6)


def load_archive(run_dir, record):
    archive = np.load(run_dir/"mask_uncertainty"/record["archive"])
    required = {"training_mask", "pose", "intrinsic"}
    missing = required-set(archive.files)
    if missing:
        raise ValueError(f"missing replay arrays {missing} in {record['archive']}")
    return {
        "frame_index": int(archive["frame_index"]),
        "mask": np.asarray(archive["training_mask"]),
        "pose": np.asarray(archive["pose"], dtype=np.float32),
        "intrinsic": np.asarray(archive["intrinsic"], dtype=np.float32),
    }


def validation_archives(run_dir, records, signatures):
    archives = [load_archive(run_dir, record) for record in records]
    selected = []
    for signature in signatures:
        match = next((item for item in archives
                      if pose_signature(item["pose"]) == tuple(signature)), None)
        if match is None:
            raise ValueError(f"validation pose not found in {run_dir}")
        selected.append(match)
    return selected


def flatten_gradient(loss, parameters):
    gradients = torch.autograd.grad(loss, tuple(parameters), allow_unused=False)
    return torch.cat([gradient.reshape(-1) for gradient in gradients])


def flatten_autograd_gradient(loss, parameters, retain_graph):
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=retain_graph, allow_unused=True)
    return torch.cat([
        (torch.zeros_like(parameter) if gradient is None else gradient).reshape(-1)
        for parameter, gradient in zip(parameters, gradients)
    ])


def apply_flat_parameter_step(parameters, direction, learning_rate):
    offset = 0
    with torch.no_grad():
        for parameter in parameters:
            count = parameter.numel()
            parameter.add_(
                direction[offset:offset+count].reshape_as(parameter),
                alpha=-float(learning_rate))
            offset += count
    if offset != direction.numel():
        raise ValueError("flat direction does not match model parameters")


def render_iou_gradient(model, items, config, transform, lighting, rasterizer):
    device = next(model.parameters()).device
    size = int(config["renderer_img_size"])
    poses = torch.from_numpy(np.stack([item["pose"] for item in items])).to(device)
    intrinsics = torch.from_numpy(np.stack([
        item["intrinsic"] for item in items])).to(device)
    targets = torch.from_numpy(np.stack([
        training_target(item["mask"], size) for item in items])).to(device)
    transform.set_img_size(max(int(config["height"]), int(config["width"])))
    transform.set_K_list(intrinsics)
    transform.set_T(poses)
    mesh, _, _ = model(len(items))
    predictions = rasterizer(transform(lighting(mesh)))[:, 3]
    weights = torch.ones(len(items), dtype=torch.float32, device=device)
    loss = weighted_iou_loss(predictions, targets, weights)
    return flatten_gradient(loss, model.parameters())


def first_order_influence(stable_path, training, validation, config, seed):
    """Estimate each observation's validation utility at the stable geometry."""
    started = time.perf_counter()
    set_seed(seed)
    device = torch.device("cuda")
    size = int(config["renderer_img_size"])
    model = Model(str(stable_path)).to(device)
    transform = sr.LookAt(viewing_angle=float(config["viewing_angle"]))
    lighting = sr.Lighting()
    rasterizer = sr.SoftRasterizer(
        image_size=size, sigma_val=1e-4, aggr_func_rgb="hard",
        near=0.1, far=1000)
    validation_gradient = render_iou_gradient(
        model, validation, config, transform, lighting, rasterizer)
    observation_gradients = [
        render_iou_gradient(
            model, [item], config, transform, lighting, rasterizer)
        for item in training
    ]
    validation_array = validation_gradient.detach().cpu().numpy()
    observation_arrays = [
        gradient.detach().cpu().numpy() for gradient in observation_gradients]
    alignments = cosine_alignments(validation_array, observation_arrays)
    dots = [
        float(np.dot(validation_array.astype(np.float64),
                     gradient.astype(np.float64)))
        for gradient in observation_arrays
    ]
    torch.cuda.synchronize()
    return {
        "cosine_alignments": alignments,
        "gradient_dots": dots,
        "validation_gradient_norm": float(np.linalg.norm(validation_array)),
        "observation_gradient_norms": [
            float(np.linalg.norm(gradient)) for gradient in observation_arrays],
        "elapsed_ms": (time.perf_counter()-started)*1000.0,
        "_validation_gradient": validation_array,
        "_observation_gradients": observation_arrays,
    }


def optimize(stable_path, training, validation, weights, config, seed,
             iterations=None, model=None, regularization=None):
    set_seed(seed)
    device = torch.device("cuda")
    size = int(config["renderer_img_size"])
    if model is None:
        model = Model(str(stable_path)).to(device)
    transform = sr.LookAt(viewing_angle=float(config["viewing_angle"]))
    lighting = sr.Lighting()
    rasterizer = sr.SoftRasterizer(
        image_size=size, sigma_val=1e-4, aggr_func_rgb="hard",
        near=0.1, far=1000)
    poses = torch.from_numpy(np.stack([item["pose"] for item in training])).to(device)
    intrinsics = torch.from_numpy(np.stack([
        item["intrinsic"] for item in training])).to(device)
    targets = torch.from_numpy(np.stack([
        training_target(item["mask"], size) for item in training])).to(device)
    validation_target_arrays = np.stack([
        training_target(item["mask"], size) for item in validation])
    validation_reg_targets = torch.from_numpy(
        validation_target_arrays).to(device)
    validation_reg_poses = torch.from_numpy(np.stack([
        item["pose"] for item in validation])).to(device)
    validation_reg_intrinsics = torch.from_numpy(np.stack([
        item["intrinsic"] for item in validation])).to(device)
    tensor_weights = torch.as_tensor(weights, dtype=torch.float32, device=device)
    transform.set_img_size(max(int(config["height"]), int(config["width"])))
    transform.set_K_list(intrinsics)
    transform.set_T(poses)
    learning_rate = 0.01
    parameters = tuple(model.parameters())
    regularization = regularization or {}
    pose_weight = float(regularization.get("pose_weight", 0.0))
    temporal_weight = float(regularization.get("temporal_weight", 0.0))
    control_iou_weight = float(regularization.get("control_iou_weight", 0.0))
    regularization_scope = regularization.get(
        "scope", "training_support_frames")
    if regularization_scope not in (
            "training_support_frames", "validation_control_frames"):
        raise ValueError(f"unknown regularization scope: {regularization_scope}")
    projection_primary = regularization.get(
        "projection_primary", "training_objective")
    if projection_primary not in (
            "training_objective", "validation_control_iou"):
        raise ValueError(f"unknown projection primary: {projection_primary}")
    if (projection_primary == "validation_control_iou"
            and regularization_scope != "validation_control_frames"):
        raise ValueError(
            "validation_control_iou primary requires validation control scope")
    pose_threshold = float(regularization.get("pose_threshold", 0.0))
    temporal_threshold = float(regularization.get("temporal_threshold", 0.0))
    optimizer_mode = regularization.get("optimizer_mode", "weighted_sum")
    if optimizer_mode not in ("weighted_sum", "constrained_projection"):
        raise ValueError(f"unknown optimizer mode: {optimizer_mode}")
    projection_cycles = int(regularization.get("projection_cycles", 20))
    if projection_cycles < 1:
        raise ValueError("projection_cycles must be at least 1")
    activation_epsilon = float(
        regularization.get("constraint_activation_epsilon", 1e-8))
    optimizer = (
        torch.optim.Adam(parameters, learning_rate, betas=(0.5, 0.99))
        if optimizer_mode == "weighted_sum" else None)
    adam_first_moment = None
    adam_second_moment = None
    pose_probe_interval = int(regularization.get("pose_probe_interval", 1))
    if pose_probe_interval < 1:
        raise ValueError("pose_probe_interval must be at least 1")
    validation_config = config.get("model_validation", {})
    perturbations = None
    if pose_weight > 0.0:
        perturbations = se3_axis_perturbations(
            validation_config.get("rotation_delta_deg", 2.0),
            validation_config.get("translation_delta", 0.005),
            device, poses.dtype)
    last_regularization = None
    projection_trace = []
    best_perturbation_indices = None
    if regularization_scope == "validation_control_frames":
        regularization_poses = validation_reg_poses
        regularization_intrinsics = validation_reg_intrinsics
        regularization_targets = validation_reg_targets
        regularization_weights = torch.ones(
            len(validation), dtype=torch.float32, device=device)
    else:
        regularization_poses = poses
        regularization_intrinsics = intrinsics
        regularization_targets = targets
        regularization_weights = tensor_weights
    iteration_count = (int(config["iteration_nums_in_each_deform"])
                       if iterations is None else int(iterations))
    for iteration in range(iteration_count):
        transform.set_K_list(intrinsics)
        transform.set_T(poses)
        mesh, laplacian_loss, _ = model(len(training))
        predictions = rasterizer(transform(lighting(mesh)))[:, 3]
        nominal_losses = per_frame_iou_losses(predictions, targets)
        base_iou_loss = weighted_mean(nominal_losses, tensor_weights)
        regularization_nominal_losses = nominal_losses
        if regularization_scope == "validation_control_frames":
            transform.set_K_list(regularization_intrinsics)
            transform.set_T(regularization_poses)
            regularization_mesh, _, _ = model(len(validation))
            regularization_predictions = rasterizer(
                transform(lighting(regularization_mesh)))[:, 3]
            regularization_nominal_losses = per_frame_iou_losses(
                regularization_predictions, regularization_targets)
        perturbed_losses = None
        if perturbations is not None:
            if (best_perturbation_indices is None
                    or iteration % pose_probe_interval == 0):
                probe_columns = []
                with torch.no_grad():
                    for delta in perturbations:
                        probe_poses = torch.matmul(
                            delta[None, :, :], regularization_poses)
                        transform.set_K_list(regularization_intrinsics)
                        transform.set_T(probe_poses)
                        probe_mesh, _, _ = model(len(regularization_poses))
                        probe_predictions = rasterizer(
                            transform(lighting(probe_mesh)))[:, 3]
                        probe_columns.append(per_frame_iou_losses(
                            probe_predictions, regularization_targets))
                best_perturbation_indices = torch.stack(
                    probe_columns, dim=1).argmin(dim=1)
            selected_deltas = perturbations[best_perturbation_indices]
            perturbed_poses = torch.matmul(
                selected_deltas, regularization_poses)
            transform.set_K_list(regularization_intrinsics)
            transform.set_T(perturbed_poses)
            perturbed_mesh, _, _ = model(len(regularization_poses))
            perturbed_predictions = rasterizer(
                transform(lighting(perturbed_mesh)))[:, 3]
            perturbed_losses = per_frame_iou_losses(
                perturbed_predictions, regularization_targets)[:, None]
        raw_pose_term, raw_temporal_term = candidate_regularization_terms(
            regularization_nominal_losses, perturbed_losses,
            regularization_weights)
        control_iou_loss = weighted_mean(
            regularization_nominal_losses, regularization_weights)
        pose_term = torch.relu(
            raw_pose_term - raw_pose_term.new_tensor(pose_threshold))
        temporal_term = torch.relu(
            raw_temporal_term
            - raw_temporal_term.new_tensor(temporal_threshold))
        loss = regularized_candidate_loss(
            base_iou_loss, laplacian_loss, pose_term, temporal_term,
            pose_weight=pose_weight, temporal_weight=temporal_weight)
        loss = loss + control_iou_weight * control_iou_loss
        if optimizer_mode == "constrained_projection":
            primary_loss = (
                control_iou_loss
                if projection_primary == "validation_control_iou"
                else base_iou_loss + 0.1 * laplacian_loss)
            primary_gradient = flatten_autograd_gradient(
                primary_loss, parameters, retain_graph=True)
            active_constraints = []
            if float(pose_term.detach().item()) > activation_epsilon:
                active_constraints.append((
                    "pose", flatten_autograd_gradient(
                        raw_pose_term, parameters, retain_graph=True)))
            if float(temporal_term.detach().item()) > activation_epsilon:
                active_constraints.append((
                    "temporal", flatten_autograd_gradient(
                        raw_temporal_term, parameters, retain_graph=True)))
            combined_gradient = flatten_autograd_gradient(
                loss, parameters, retain_graph=False)
            if adam_first_moment is None:
                adam_first_moment = torch.zeros_like(combined_gradient)
                adam_second_moment = torch.zeros_like(combined_gradient)
            adam_first_moment.mul_(0.5).add_(combined_gradient, alpha=0.5)
            adam_second_moment.mul_(0.99).addcmul_(
                combined_gradient, combined_gradient, value=0.01)
            step = iteration + 1
            first_unbiased = adam_first_moment / (1.0 - 0.5 ** step)
            second_unbiased = adam_second_moment / (1.0 - 0.99 ** step)
            adam_direction = first_unbiased / (
                second_unbiased.sqrt() + 1e-8)
            named_normals = [("primary", primary_gradient)] + active_constraints
            projected_direction = project_direction_to_halfspaces(
                adam_direction, [normal for _, normal in named_normals],
                cycles=projection_cycles)
            before_dots = {
                name: float(torch.dot(
                    adam_direction, normal).detach().item())
                for name, normal in named_normals}
            after_dots = {
                name: float(torch.dot(
                    projected_direction, normal).detach().item())
                for name, normal in named_normals}
            projection_tolerance = 1e-7
            constraint_gradients = dict(active_constraints)
            projection_trace.append({
                "iteration": iteration + 1,
                "active_constraints": [
                    name for name, _ in active_constraints],
                "primary_pose_cosine": (
                    cosine_similarity(
                        primary_gradient, constraint_gradients["pose"])
                    if "pose" in constraint_gradients else None),
                "primary_temporal_cosine": (
                    cosine_similarity(
                        primary_gradient, constraint_gradients["temporal"])
                    if "temporal" in constraint_gradients else None),
                "pose_temporal_cosine": (
                    cosine_similarity(
                        constraint_gradients["pose"],
                        constraint_gradients["temporal"])
                    if all(name in constraint_gradients
                           for name in ("pose", "temporal")) else None),
                "direction_dots_before": before_dots,
                "direction_dots_after": after_dots,
                "conflicting_directions_before": [
                    name for name, value in before_dots.items()
                    if value < -projection_tolerance],
                "projection_feasible": all(
                    value >= -projection_tolerance
                    for value in after_dots.values()),
                "projection_norm": float((
                    projected_direction-adam_direction).norm().detach().item()),
                "direction_norm": float(
                    adam_direction.norm().detach().item()),
            })
            apply_flat_parameter_step(
                parameters, projected_direction, learning_rate)
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        last_regularization = {
            "base_iou_loss": float(base_iou_loss.detach().item()),
            "control_iou_loss": float(control_iou_loss.detach().item()),
            "pose_inconsistency": float(raw_pose_term.detach().item()),
            "temporal_std": float(raw_temporal_term.detach().item()),
            "pose_violation": float(pose_term.detach().item()),
            "temporal_violation": float(temporal_term.detach().item()),
            "total_loss": float(loss.detach().item()),
        }

    validation_masks = np.stack([
        np.tile(target, (4, 1, 1)) for target in validation_target_arrays])
    validation_poses = torch.from_numpy(np.stack([
        item["pose"] for item in validation]))
    validation_intrinsics = torch.from_numpy(np.stack([
        item["intrinsic"] for item in validation]))
    metrics = evaluate_model_gate_metrics(
        model, validation_masks, validation_poses, validation_intrinsics,
        transform, lighting, rasterizer,
        float(config.get("model_validation", {}).get("rotation_delta_deg", 2.0)),
        float(config.get("model_validation", {}).get("translation_delta", 0.005)),
    )
    metrics["optimization_regularization"] = {
        "scope": regularization_scope,
        "pose_weight": pose_weight,
        "temporal_weight": temporal_weight,
        "control_iou_weight": control_iou_weight,
        "projection_primary": projection_primary,
        "optimizer_mode": optimizer_mode,
        "projection_cycles": projection_cycles,
        "constraint_activation_epsilon": activation_epsilon,
        "pose_threshold": pose_threshold,
        "temporal_threshold": temporal_threshold,
        "pose_probe_interval": pose_probe_interval,
        "rotation_delta_deg": float(validation_config.get(
            "rotation_delta_deg", 2.0)),
        "translation_delta": float(validation_config.get(
            "translation_delta", 0.005)),
        "last_iteration": last_regularization,
        "gradient_projection_trace": projection_trace,
        "gradient_projection_summary": {
            "steps": len(projection_trace),
            "projected_steps": sum(
                row["projection_norm"] > 1e-12
                for row in projection_trace),
            "pose_active_steps": sum(
                "pose" in row["active_constraints"]
                for row in projection_trace),
            "temporal_active_steps": sum(
                "temporal" in row["active_constraints"]
                for row in projection_trace),
            "all_steps_feasible": all(
                row["projection_feasible"] for row in projection_trace),
        },
    }
    return metrics


def short_horizon_loo(stable_path, training, validation, config, seed, steps):
    """Estimate marginal utility with short full and leave-one-out optimizations."""
    started = time.perf_counter()
    base_model = Model(str(stable_path)).to(torch.device("cuda"))
    full = optimize(
        stable_path, training, validation, [1.0]*len(training),
        config, seed, iterations=steps, model=copy.deepcopy(base_model))
    leave_one_out_losses = []
    for omitted in range(len(training)):
        subset = [
            item for index, item in enumerate(training) if index != omitted]
        metrics = optimize(
            stable_path, subset, validation, [1.0]*len(subset),
            config, seed, iterations=steps,
            model=copy.deepcopy(base_model))
        leave_one_out_losses.append(float(metrics["mean_iou_loss"]))
    full_loss = float(full["mean_iou_loss"])
    return {
        "steps": int(steps),
        "full_validation_iou_loss": full_loss,
        "leave_one_out_validation_iou_losses": leave_one_out_losses,
        "marginal_utilities": [
            loss-full_loss for loss in leave_one_out_losses],
        "elapsed_ms": (time.perf_counter()-started)*1000.0,
    }

def dual_trust_record(policy_record, training_frames, stable_metrics,
                      full_metrics, loo_metrics, record):
    """Build an audited lowest-loss decision across stable/full/LOO geometry."""
    proposals = {"full": full_metrics}
    if loo_metrics is not None:
        proposals["loo"] = loo_metrics
    choice = select_lowest_validation_geometry(stable_metrics, proposals)
    final_metrics = choice["selected_metrics"]
    source = choice["selected_source"]
    return dict(
        policy_record,
        status="committed" if choice["committed"] else "rolled_back",
        training_frames=training_frames,
        committed=choice["committed"],
        commit_reason=f"selected_{source}",
        selected_source=source,
        trust_region_checks={"selected_lowest_validation_iou": True},
        actual_iou_improvement=choice["actual_iou_improvement"],
        proposal_iou_losses={
            name: float(metrics["mean_iou_loss"])
            for name, metrics in [("stable", stable_metrics),
                                  ("full", full_metrics)]
            + ([] if loo_metrics is None else [("loo", loo_metrics)])
        },
        same_run_original_mean_iou_loss=full_metrics["mean_iou_loss"],
        same_run_original_pose_inconsistency=full_metrics[
            "pose_inconsistency"],
        same_run_original_temporal_std=full_metrics["temporal_std"],
        same_run_original_uncertainty=full_metrics["uncertainty"],
        proposed_mean_iou_loss=(
            None if loo_metrics is None else loo_metrics["mean_iou_loss"]),
        proposed_pose_inconsistency=(
            None if loo_metrics is None else loo_metrics[
                "pose_inconsistency"]),
        proposed_temporal_std=(
            None if loo_metrics is None else loo_metrics["temporal_std"]),
        proposed_uncertainty=(
            None if loo_metrics is None else loo_metrics["uncertainty"]),
        proposed_optimization_regularization=(
            None if loo_metrics is None else loo_metrics.get(
                "optimization_regularization")),
        stable_mean_iou_loss=stable_metrics["mean_iou_loss"],
        stable_pose_inconsistency=stable_metrics["pose_inconsistency"],
        stable_temporal_std=stable_metrics["temporal_std"],
        stable_uncertainty=stable_metrics["uncertainty"],
        mean_iou_loss=final_metrics["mean_iou_loss"],
        pose_inconsistency=final_metrics["pose_inconsistency"],
        temporal_std=final_metrics["temporal_std"],
        uncertainty=final_metrics["uncertainty"],
        recorded_candidate_iou_loss=record["candidate_validation_iou_loss"],
        recorded_stable_iou_loss=record["stable_validation_iou_loss"],
    )


def indexed_dual_references(path):
    if path is None:
        return {}
    rows = load_jsonl(path)
    return {
        (row["object"], row["sequence"], int(row["seed"]),
         int(row["candidate_version"])): row
        for row in rows
        if row.get("policy") == "p1v2_short_loo_dual_trust_region"
    }


def reference_metrics(row, prefix):
    return {
        "mean_iou_loss": float(row[f"{prefix}mean_iou_loss"]),
        "pose_inconsistency": float(row[f"{prefix}pose_inconsistency"]),
        "temporal_std": float(row[f"{prefix}temporal_std"]),
        "uncertainty": float(row[f"{prefix}uncertainty"]),
    }


def summarize(results):
    """Summarize replay fidelity and counterfactual policy effects by event."""
    events = {}
    for row in results:
        key = (row["object"], row["sequence"], row["seed"],
               row["candidate_version"])
        events.setdefault(key, {})[row["policy"]] = row

    fidelity = []
    summaries = {}
    for policy in POLICIES[1:]:
        iou_deltas = []
        stable_deltas = []
        uncertainty_deltas = []
        blocked = 0
        committed = 0
        rolled_back = 0
        for policies in events.values():
            original = policies.get("original")
            candidate = policies.get(policy)
            if not original or original["status"] != "optimized" or not candidate:
                continue
            fidelity.append(abs(
                original["mean_iou_loss"]-
                original["recorded_candidate_iou_loss"]))
            if candidate["status"] == "blocked":
                blocked += 1
                continue
            committed += int(candidate.get("committed") is True)
            rolled_back += int(candidate.get("committed") is False)
            iou_deltas.append(
                candidate["mean_iou_loss"]-candidate.get(
                    "same_run_original_mean_iou_loss", original["mean_iou_loss"]))
            stable_deltas.append(
                candidate["mean_iou_loss"]-
                candidate.get("stable_mean_iou_loss",
                              candidate["recorded_stable_iou_loss"]))
            uncertainty_deltas.append(
                candidate["uncertainty"]-candidate.get(
                    "same_run_original_uncertainty", original["uncertainty"]))
        summaries[policy] = {
            "optimized_events": len(iou_deltas),
            "blocked_events": blocked,
            "committed_events": committed,
            "rolled_back_events": rolled_back,
            "mean_iou_loss_delta_vs_original": (
                float(np.mean(iou_deltas)) if iou_deltas else None),
            "improved_iou_events": sum(delta < 0 for delta in iou_deltas),
            "mean_iou_loss_delta_vs_recorded_stable": (
                float(np.mean(stable_deltas)) if stable_deltas else None),
            "better_than_recorded_stable_events": sum(
                delta < 0 for delta in stable_deltas),
            "mean_uncertainty_delta_vs_original": (
                float(np.mean(uncertainty_deltas))
                if uncertainty_deltas else None),
        }
    return {
        "schema": "stage4_p2_geometry_replay_summary_v1",
        "stage": "stage4_p2",
        "mode": "exact_counterfactual_geometry_replay",
        "events": len(events),
        "max_abs_original_replay_iou_error": max(fidelity, default=None),
        "policies": summaries,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-glob", default="p2_capture_s*")
    parser.add_argument("--candidate-frames", type=Path, required=True)
    parser.add_argument("--parent-frames", type=Path, required=True)
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policies", default=",".join(POLICIES))
    parser.add_argument("--reference-results", type=Path)
    parser.add_argument("--loo-utility-margin", type=float, default=0.0)
    parser.add_argument("--trust-min-iou-improvement", type=float, default=0.0)
    parser.add_argument("--loo-steps", type=int, default=25)
    parser.add_argument("--pose-consistency-weight", type=float, default=0.0)
    parser.add_argument("--temporal-stability-weight", type=float, default=0.0)
    parser.add_argument("--pose-probe-interval", type=int, default=1)
    parser.add_argument("--proposal-steps", type=int)
    parser.add_argument("--dual-reference-results", type=Path)
    parser.add_argument(
        "--regularization-scope",
        choices=("training_support_frames", "validation_control_frames"),
        default="training_support_frames")
    parser.add_argument(
        "--candidate-optimizer",
        choices=("weighted_sum", "constrained_projection"),
        default="weighted_sum")
    parser.add_argument("--projection-cycles", type=int, default=20)
    parser.add_argument(
        "--constraint-activation-epsilon", type=float, default=1e-8)
    parser.add_argument("--control-iou-weight", type=float, default=0.0)
    parser.add_argument(
        "--projection-primary",
        choices=("training_objective", "validation_control_iou"),
        default="training_objective")
    args = parser.parse_args()
    policies = tuple(item.strip() for item in args.policies.split(",")
                     if item.strip())
    unknown = set(policies)-set(POLICIES)
    if unknown:
        raise ValueError(f"unknown policies: {sorted(unknown)}")
    if (args.pose_consistency_weight < 0.0
            or args.temporal_stability_weight < 0.0
            or args.control_iou_weight < 0.0):
        raise ValueError("regularizer weights must be non-negative")
    if args.projection_cycles < 1:
        raise ValueError("projection_cycles must be at least 1")
    if (args.regularization_scope == "validation_control_frames"
            and args.dual_reference_results is None):
        raise ValueError(
            "validation_control_frames requires frozen dual reference metrics")
    frozen = json.loads(args.frozen_config.read_text())
    threshold = float(frozen["method"]["demotion_threshold"])
    decisions = attach_parent_weights(
        load_csv(args.candidate_frames), load_csv(args.parent_frames))
    results = (load_jsonl(args.reference_results)
               if args.reference_results else [])
    dual_references = indexed_dual_references(args.dual_reference_results)
    for registry_path in sorted(args.run_root.glob(
            f"*/*/{args.run_glob}/model_registry.jsonl")):
        relative = registry_path.relative_to(args.run_root)
        object_name, sequence, run_tag = relative.parts[:3]
        seed = int(run_tag.rsplit("s", 1)[1])
        run_dir = registry_path.parent
        config = LoadConfigSafety(str(
            PROJECT/"config"/"moped"/object_name/"evaluation"/f"{sequence}.yml"))
        registry = load_jsonl(registry_path)
        records = load_jsonl(
            run_dir/"mask_uncertainty"/"mask_uncertainty.jsonl")
        batches = split_batches(records)
        for event_index, (record, batch) in enumerate(zip(registry, batches)):
            version = int(record["version"])
            if version == 1 or not record["validation_pose_signatures"]:
                continue
            indices = [int(value) for value in record["training_indices"]]
            training = [load_archive(run_dir, batch[index]) for index in indices]
            decision_rows = [decisions.get(
                (object_name, sequence, seed, item["frame_index"]))
                for item in training]
            previous_records = [item for group in batches[:event_index+1]
                                for item in group]
            validation = validation_archives(
                run_dir, previous_records, record["validation_pose_signatures"])
            stable_version = int(record["base_stable_geometry_version"])
            stable_path = (run_dir/"model_candidates"/
                           f"candidate_v{stable_version:03d}.obj")
            dual_reference = dual_references.get(
                (object_name, sequence, seed, version))
            if (args.dual_reference_results is not None
                    and dual_reference is None):
                continue
            influence = None
            if any(policy in policies for policy in (
                    "p1v2_influence_rescue", "p1v2_gradient_match")):
                influence = first_order_influence(
                    stable_path, training, validation, config, seed)
            loo = None
            if (dual_reference is not None
                    and "p1v2_short_loo_dual_regularized" in policies):
                loo = {
                    "steps": int(dual_reference["loo_steps"]),
                    "full_validation_iou_loss": float(
                        dual_reference["loo_full_validation_iou_loss"]),
                    "leave_one_out_validation_iou_losses":
                        dual_reference["loo_validation_iou_losses"],
                    "marginal_utilities":
                        dual_reference["loo_marginal_utilities"],
                    "elapsed_ms": float(dual_reference["loo_elapsed_ms"]),
                }
            elif any(policy in policies for policy in (
                    "p1v2_short_loo_softmax", "p1v2_short_loo_trust_region",
                    "p1v2_short_loo_dual_trust_region",
                    "p1v2_short_loo_dual_regularized")):
                loo = short_horizon_loo(
                    stable_path, training, validation, config, seed,
                    args.loo_steps)
            stable_metrics = None
            if (dual_reference is not None
                    and "p1v2_short_loo_dual_regularized" in policies):
                stable_metrics = reference_metrics(dual_reference, "stable_")
            elif any(policy in policies for policy in (
                    "p1v2_short_loo_trust_region",
                    "p1v2_short_loo_dual_trust_region",
                    "p1v2_short_loo_dual_regularized")):
                stable_metrics = optimize(
                    stable_path, training, validation, [1.0]*len(training),
                    config, seed, iterations=0)
            common = {
                "object": object_name,
                "sequence": sequence,
                "seed": seed,
                "candidate_version": version,
                "validation_frames": len(validation),
                "training_frame_indices": [
                    item["frame_index"] for item in training],
                "influence_cosine_alignments": (
                    influence["cosine_alignments"] if influence else None),
                "influence_gradient_dots": (
                    influence["gradient_dots"] if influence else None),
                "influence_validation_gradient_norm": (
                    influence["validation_gradient_norm"] if influence else None),
                "influence_observation_gradient_norms": (
                    influence["observation_gradient_norms"] if influence else None),
                "influence_elapsed_ms": (
                    influence["elapsed_ms"] if influence else None),
                "loo_steps": loo["steps"] if loo else None,
                "loo_full_validation_iou_loss": (
                    loo["full_validation_iou_loss"] if loo else None),
                "loo_validation_iou_losses": (
                    loo["leave_one_out_validation_iou_losses"]
                    if loo else None),
                "loo_marginal_utilities": (
                    loo["marginal_utilities"] if loo else None),
                "loo_elapsed_ms": loo["elapsed_ms"] if loo else None,
            }
            for policy in policies:
                if policy == "p1v2_influence_rescue":
                    base_weights = [
                        policy_weight("p1v2_soft", row, threshold)
                        for row in decision_rows]
                    weights = influence_rescue_weights(
                        decision_rows, base_weights,
                        influence["cosine_alignments"], margin=0.0)
                elif policy == "p1v2_gradient_match":
                    base_weights = [
                        policy_weight("p1v2_soft", row, threshold)
                        for row in decision_rows]
                    locked_zero = [
                        row is not None and row["baseline_decision"] == "reject"
                        for row in decision_rows]
                    weights = gradient_matching_weights(
                        influence["_validation_gradient"],
                        influence["_observation_gradients"],
                        base_weights, locked_zero=locked_zero, ridge=1.0)
                elif policy == "p1v2_short_loo_softmax":
                    weights = short_loo_softmax_weights(
                        decision_rows, loo["marginal_utilities"])
                elif policy in ("p1v2_short_loo_trust_region",
                               "p1v2_short_loo_dual_trust_region",
                               "p1v2_short_loo_dual_regularized"):
                    weights = short_loo_filter_weights(
                        decision_rows, loo["marginal_utilities"],
                        margin=args.loo_utility_margin)
                else:
                    weights = [
                        policy_weight(policy, row, threshold)
                        for row in decision_rows]

                selected = [(item, weight) for item, weight in zip(training, weights)
                            if weight > 0]
                policy_record = dict(
                    common, policy=policy, policy_weights=weights,
                    effective_training_mass=sum(weights))
                if len(selected) < 2:
                    if policy in ("p1v2_short_loo_dual_trust_region",
                                  "p1v2_short_loo_dual_regularized"):
                        full_metrics = (
                            reference_metrics(
                                dual_reference, "same_run_original_")
                            if dual_reference is not None
                            else optimize(
                                stable_path, training, validation,
                                [1.0]*len(training), config, seed,
                                iterations=args.proposal_steps))
                        results.append(dual_trust_record(
                            policy_record, len(selected), stable_metrics,
                            full_metrics, None, record))
                    elif policy == "p1v2_short_loo_trust_region":
                        results.append(dict(
                            policy_record, status="rolled_back",
                            training_frames=len(selected), committed=False,
                            commit_reason="loo_prescreen_insufficient_observations",
                            trust_region_checks={
                                "minimum_training_frames": False},
                            actual_iou_improvement=None,
                            proposed_mean_iou_loss=None,
                            mean_iou_loss=stable_metrics["mean_iou_loss"],
                            pose_inconsistency=stable_metrics[
                                "pose_inconsistency"],
                            temporal_std=stable_metrics["temporal_std"],
                            uncertainty=stable_metrics["uncertainty"],
                            stable_mean_iou_loss=stable_metrics["mean_iou_loss"],
                            recorded_candidate_iou_loss=record[
                                "candidate_validation_iou_loss"],
                            recorded_stable_iou_loss=record[
                                "stable_validation_iou_loss"],
                        ))
                    else:
                        results.append(dict(
                            policy_record, status="blocked",
                            training_frames=len(selected)))
                    continue
                metrics = optimize(
                    stable_path, [item for item, _ in selected], validation,
                    [weight for _, weight in selected], config, seed,
                    iterations=args.proposal_steps,
                    regularization=(
                        {
                            "pose_weight": args.pose_consistency_weight,
                            "temporal_weight": args.temporal_stability_weight,
                            "pose_probe_interval": args.pose_probe_interval,
                            "scope": args.regularization_scope,
                            "pose_threshold": (
                                dual_reference[
                                    "same_run_original_pose_inconsistency"]
                                if args.regularization_scope
                                == "validation_control_frames" else 0.0),
                            "temporal_threshold": (
                                dual_reference[
                                    "same_run_original_temporal_std"]
                                if args.regularization_scope
                                == "validation_control_frames" else 0.0),
                            "optimizer_mode": args.candidate_optimizer,
                            "projection_cycles": args.projection_cycles,
                            "constraint_activation_epsilon":
                                args.constraint_activation_epsilon,
                            "control_iou_weight": args.control_iou_weight,
                            "projection_primary": args.projection_primary,
                        }
                        if policy == "p1v2_short_loo_dual_regularized"
                        else None))
                if policy in ("p1v2_short_loo_dual_trust_region",
                              "p1v2_short_loo_dual_regularized"):
                    full_metrics = (
                        reference_metrics(
                            dual_reference, "same_run_original_")
                        if dual_reference is not None
                        else optimize(
                            stable_path, training, validation,
                            [1.0]*len(training), config, seed,
                            iterations=args.proposal_steps))
                    results.append(dual_trust_record(
                        policy_record, len(selected), stable_metrics,
                        full_metrics, metrics, record))
                    continue
                if policy == "p1v2_short_loo_trust_region":
                    trust = monotone_validation_commit(
                        stable_metrics, metrics,
                        min_iou_improvement=args.trust_min_iou_improvement)
                    final_metrics = metrics if trust["committed"] else stable_metrics
                    results.append(dict(
                        policy_record,
                        status=("committed" if trust["committed"]
                                else "rolled_back"),
                        training_frames=len(selected),
                        committed=trust["committed"],
                        commit_reason=trust["reason"],
                        trust_region_checks=trust["checks"],
                        actual_iou_improvement=trust[
                            "actual_iou_improvement"],
                        loo_utility_margin=args.loo_utility_margin,
                        trust_min_iou_improvement=(
                            args.trust_min_iou_improvement),
                        proposed_mean_iou_loss=metrics["mean_iou_loss"],
                        proposed_pose_inconsistency=metrics[
                            "pose_inconsistency"],
                        proposed_temporal_std=metrics["temporal_std"],
                        proposed_uncertainty=metrics["uncertainty"],
                        stable_mean_iou_loss=stable_metrics["mean_iou_loss"],
                        stable_pose_inconsistency=stable_metrics[
                            "pose_inconsistency"],
                        stable_temporal_std=stable_metrics["temporal_std"],
                        stable_uncertainty=stable_metrics["uncertainty"],
                        mean_iou_loss=final_metrics["mean_iou_loss"],
                        pose_inconsistency=final_metrics[
                            "pose_inconsistency"],
                        temporal_std=final_metrics["temporal_std"],
                        uncertainty=final_metrics["uncertainty"],
                        recorded_candidate_iou_loss=record[
                            "candidate_validation_iou_loss"],
                        recorded_stable_iou_loss=record[
                            "stable_validation_iou_loss"],
                    ))
                    continue
                results.append(dict(
                    policy_record,
                    status="optimized",
                    training_frames=len(selected),
                    mean_iou_loss=metrics["mean_iou_loss"],
                    pose_inconsistency=metrics["pose_inconsistency"],
                    temporal_std=metrics["temporal_std"],
                    uncertainty=metrics["uncertainty"],
                    recorded_candidate_iou_loss=record[
                        "candidate_validation_iou_loss"],
                    recorded_stable_iou_loss=record[
                        "stable_validation_iou_loss"],
                ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(
        json.dumps(row, sort_keys=True) for row in results)+"\n")
    summary = summarize(results)
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n")
    print(json.dumps({
        "rows": len(results), "output": str(args.output),
        "summary": str(summary_path), "metrics": summary,
    }, indent=2))


if __name__ == "__main__":
    main()
