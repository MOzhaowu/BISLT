#!/usr/bin/env python3
"""Validate projected-candidate results and emit an auditable freeze record."""

import argparse
import hashlib
import json
from pathlib import Path


SOURCE_POLICY = "p1v2_short_loo_dual_regularized"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path):
    return [
        json.loads(line) for line in path.read_text().splitlines()
        if line.strip()
    ]


def summarize_projection(rows):
    source = [row for row in rows if row.get("policy") == SOURCE_POLICY]
    proposals = [
        row for row in source if row.get("proposed_mean_iou_loss") is not None
    ]
    feasible = [
        row for row in proposals
        if (row["proposed_pose_inconsistency"]
            <= row["same_run_original_pose_inconsistency"] + 1e-12)
        and (row["proposed_temporal_std"]
             <= row["same_run_original_temporal_std"] + 1e-12)
    ]
    summaries = [
        row["proposed_optimization_regularization"][
            "gradient_projection_summary"]
        for row in proposals
    ]
    return {
        "events": len(source),
        "loo_proposals": len(proposals),
        "loo_feasible_events": len(feasible),
        "optimizer_steps": sum(row["steps"] for row in summaries),
        "projected_steps": sum(row["projected_steps"] for row in summaries),
        "pose_active_steps": sum(row["pose_active_steps"] for row in summaries),
        "temporal_active_steps": sum(
            row["temporal_active_steps"] for row in summaries),
        "all_projected_steps_feasible": all(
            row["all_steps_feasible"] for row in summaries),
    }


def finalize(replay_path, evaluation_path, protocol_path, mode):
    replay = summarize_projection(load_jsonl(replay_path))
    evaluation = json.loads(evaluation_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    checks = {
        "multimetric_evaluation": evaluation["pass"] is True,
        "all_projected_steps_feasible":
            replay["all_projected_steps_feasible"] is True,
    }
    if mode == "development":
        checks["loo_feasible_events"] = (
            replay["loo_feasible_events"]
            >= int(protocol["acceptance"]["loo_feasible_events_min"]))
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "stage": "stage4_p2",
        "mode": mode,
        "status": (
            "frozen_development_pass"
            if passed and mode == "development"
            else ("independent_validation_pass" if passed else "rejected")
        ),
        "pass": passed,
        "checks": checks,
        "projection": replay,
        "evaluation": {
            "events": evaluation["paired_optimized_events"],
            "improved_iou_events": evaluation["improved_iou_events"],
            "mean_deltas": evaluation["mean_deltas"],
            "mean_iou_delta_bootstrap_ci95":
                evaluation["mean_iou_delta_bootstrap_ci95"],
            "by_seed": evaluation["by_seed"],
            "acceptance_checks": evaluation["acceptance_checks"],
        },
        "sha256": {
            "replay": sha256(replay_path),
            "evaluation": sha256(evaluation_path),
            "protocol": sha256(protocol_path),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("replay", type=Path)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--mode", choices=("development", "independent"),
                        required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = finalize(
        args.replay, args.evaluation, args.protocol, args.mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
