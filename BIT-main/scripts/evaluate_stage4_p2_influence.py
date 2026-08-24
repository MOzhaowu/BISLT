#!/usr/bin/env python3
"""Evaluate frozen Stage-4 P2 influence policies on independent replay."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = ("mean_iou_loss", "pose_inconsistency", "temporal_std", "uncertainty")


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def evaluate(rows, gate_summary, protocol, policy, bootstrap_seed=20260823,
             bootstrap_replicates=20000):
    events = defaultdict(dict)
    for row in rows:
        key = (row["object"], row["sequence"], int(row["seed"]),
               int(row["candidate_version"]))
        events[key][row["policy"]] = row
    paired = []
    blocked = 0
    for key, policies in sorted(events.items()):
        baseline = policies.get("original")
        candidate = policies.get(policy)
        if candidate is None:
            continue
        has_embedded_baseline = all(
            candidate.get(f"same_run_original_{metric}") is not None
            for metric in METRICS)
        if baseline is None and not has_embedded_baseline:
            continue
        if candidate["status"] == "blocked":
            blocked += 1
            continue
        baseline_values = {}
        for metric in METRICS:
            value = candidate.get(f"same_run_original_{metric}")
            if value is None:
                value = baseline[metric]
            baseline_values[metric] = float(value)
        deltas = {metric: float(candidate[metric])-baseline_values[metric]
                  for metric in METRICS}
        elapsed_ms = candidate.get("influence_elapsed_ms")
        if elapsed_ms is None:
            elapsed_ms = candidate.get("loo_elapsed_ms")
        paired.append({
            "object": key[0], "sequence": key[1], "seed": key[2],
            "candidate_version": key[3], "deltas": deltas,
            "marginal_utility_elapsed_ms": elapsed_ms,
            "status": candidate["status"],
            "committed": candidate.get("committed"),
            "commit_reason": candidate.get("commit_reason"),
            "baseline_values": baseline_values,
        })
    iou = np.asarray([row["deltas"]["mean_iou_loss"] for row in paired])
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap = (
        rng.choice(iou, size=(bootstrap_replicates, len(iou)), replace=True)
        .mean(axis=1) if len(iou) else np.asarray([]))
    seeds = {}
    for seed in sorted({row["seed"] for row in paired}):
        values = [row["deltas"]["mean_iou_loss"] for row in paired
                  if row["seed"] == seed]
        seeds[str(seed)] = {
            "events": len(values),
            "mean_iou_loss_delta": float(np.mean(values)),
            "improved_events": sum(value < 0 for value in values),
        }
    means = {
        metric: (float(np.mean([
            row["deltas"][metric] for row in paired])) if paired else None)
        for metric in METRICS}
    acceptance = protocol["acceptance"]
    gate = gate_summary["candidate"]
    checks = {
        "mean_iou_loss": (
            means["mean_iou_loss"]
            <= acceptance["mean_iou_loss_delta_vs_original_max"]),
        "improved_event_fraction": (
            sum(value < 0 for value in iou)/len(iou)
            >= acceptance["improved_event_fraction_min"] if len(iou) else False),
        "pose_inconsistency": (
            means["pose_inconsistency"]
            <= acceptance["mean_pose_inconsistency_delta_max"]),
        "temporal_stability": (
            means["temporal_std"]
            <= acceptance["mean_temporal_std_delta_max"]),
        "safe_reject_rate": (
            gate["safe_reject_rate"] <= acceptance["safe_reject_rate_max"]),
        "safe_nonaccept_rate": (
            gate["safe_nonaccept_rate"]
            <= acceptance["safe_nonaccept_rate_max"]),
        "seed_direction_consistent": all(
            value["mean_iou_loss_delta"] < 0 for value in seeds.values()),
    }
    if "strict_seed_fraction_min" in acceptance:
        checks.pop("seed_direction_consistent")
        tolerance = float(acceptance.get("comparison_epsilon", 1e-12))
        seed_deltas = [
            value["mean_iou_loss_delta"] for value in seeds.values()]
        strict_seed_fraction = (
            sum(value < -tolerance for value in seed_deltas)
            / len(seed_deltas) if seed_deltas else 0.0)
        checks.update({
            "iou_eventwise_nonregression": all(
                row["deltas"]["mean_iou_loss"]
                <= acceptance["max_event_iou_loss_delta"]+tolerance
                for row in paired),
            "pose_eventwise_nonregression": all(
                row["deltas"]["pose_inconsistency"]
                <= acceptance["max_event_pose_delta"]+tolerance
                for row in paired),
            "temporal_eventwise_nonregression": all(
                row["deltas"]["temporal_std"]
                <= acceptance["max_event_temporal_delta"]+tolerance
                for row in paired),
            "bootstrap_iou_upper_bound": (
                bool(len(bootstrap)) and float(np.quantile(bootstrap, 0.975))
                <= acceptance["bootstrap_iou_upper_bound_max"]+tolerance),
            "strict_seed_fraction": strict_seed_fraction
            >= acceptance["strict_seed_fraction_min"],
            "seed_nonregression": all(
                value <= acceptance["max_seed_iou_delta"]+tolerance
                for value in seed_deltas),
        })
    checks = {name: bool(value) for name, value in checks.items()}
    return {
        "schema": "stage4_p2_influence_independent_evaluation_v1",
        "policy": policy,
        "events": len(events),
        "paired_optimized_events": len(paired),
        "blocked_events": blocked,
        "committed_events": sum(
            row["committed"] is True for row in paired),
        "rolled_back_events": sum(
            row["committed"] is False for row in paired),
        "improved_iou_events": int(sum(value < 0 for value in iou)),
        "mean_deltas": means,
        "mean_iou_delta_bootstrap_ci95": (
            [float(np.quantile(bootstrap, 0.025)),
             float(np.quantile(bootstrap, 0.975))]
            if len(bootstrap) else None),
        "mean_marginal_utility_elapsed_ms": (
            float(np.mean([
                row["marginal_utility_elapsed_ms"] for row in paired
                if row["marginal_utility_elapsed_ms"] is not None]))
            if any(row["marginal_utility_elapsed_ms"] is not None
                   for row in paired) else None),
        "by_seed": seeds,
        "gate": gate,
        "acceptance_checks": checks,
        "pass": all(checks.values()),
        "paired_events": paired,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("geometry_jsonl", type=Path)
    parser.add_argument("gate_summary", type=Path)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--policy", default="p1v2_gradient_match")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(
        load_jsonl(args.geometry_jsonl),
        json.loads(args.gate_summary.read_text()),
        json.loads(args.protocol.read_text()),
        args.policy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
