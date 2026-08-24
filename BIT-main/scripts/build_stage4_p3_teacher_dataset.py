#!/usr/bin/env python3
"""Build a leakage-audited event/observation dataset from the frozen P2 teacher."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


IDENTITY_FIELDS = ("object", "sequence", "seed", "frame_index")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_csv_index(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    output = {}
    for row in rows:
        key = (
            row["object"], row["sequence"], int(row["seed"]),
            int(row["frame_index"]))
        if key in output:
            raise ValueError(f"duplicate frame identity in {path}: {key}")
        output[key] = row
    return output


def load_jsonl(path):
    return [
        json.loads(line) for line in path.read_text().splitlines()
        if line.strip()]


def seed_set_valid(observed, expected, allow_zero_event_seeds=False):
    return (set(observed).issubset(expected)
            if allow_zero_event_seeds
            else sorted(observed) == sorted(expected))


def finite_float(row, field):
    value = float(row[field])
    if not np.isfinite(value):
        raise ValueError(f"non-finite {field}: {value}")
    return value


def aggregate(values, operation):
    if not values:
        return None
    if operation == "mean":
        return float(np.mean(values))
    if operation == "min":
        return float(np.min(values))
    if operation == "max":
        return float(np.max(values))
    raise ValueError(f"unsupported aggregate: {operation}")


def merged_frame_features(key, p1v2_rows, gate_rows, feature_spec):
    p1v2 = p1v2_rows.get(key)
    gate = gate_rows.get(key)
    if p1v2 is None or gate is None:
        return None
    merged = dict(gate)
    merged.update(p1v2)
    return {field: finite_float(merged, field) for field in feature_spec}


def build_event(row, p1v2_rows, gate_rows, protocol):
    indices = [int(value) for value in row["training_frame_indices"]]
    utilities = [float(value) for value in row["loo_marginal_utilities"]]
    if len(indices) != len(utilities):
        raise ValueError("training indices and teacher utilities differ in length")
    feature_spec = protocol["online_numeric_frame_features"]
    observations = []
    decisions = []
    for frame_index, utility in zip(indices, utilities):
        key = (row["object"], row["sequence"], int(row["seed"]), frame_index)
        features = merged_frame_features(
            key, p1v2_rows, gate_rows, feature_spec)
        p1v2 = p1v2_rows.get(key)
        decision = None if p1v2 is None else p1v2["decision"]
        if decision is not None:
            decisions.append(decision)
        observations.append({
            "frame_index": frame_index,
            "features_available": features is not None,
            "is_anchor": frame_index == int(protocol[
                "missing_feature_policy"]["anchor_frame_index"]),
            "features": features,
            "teacher_marginal_utility": utility,
            "teacher_keep": utility > 0.0,
        })

    available = [item for item in observations if item["features"] is not None]
    event_features = {
        "candidate_version": int(row["candidate_version"]),
        "candidate_frame_count": len(observations),
        "validation_frame_count": int(row["validation_frames"]),
        "available_feature_fraction": len(available) / len(observations),
        "anchor_fraction": sum(item["is_anchor"] for item in observations)
        / len(observations),
    }
    for decision in ("accept", "downweight", "reject"):
        event_features[f"{decision}_fraction"] = (
            decisions.count(decision) / len(decisions) if decisions else 0.0)
    for field, operations in feature_spec.items():
        values = [item["features"][field] for item in available]
        for operation in operations:
            event_features[f"{field}_{operation}"] = aggregate(values, operation)

    full_iou = float(row["same_run_original_mean_iou_loss"])
    stable_iou = float(row["stable_mean_iou_loss"])
    selected_iou = float(row["mean_iou_loss"])
    feasibility = row.get("multimetric_feasibility") or {}
    return {
        "identity": {
            "object": row["object"],
            "sequence": row["sequence"],
            "seed": int(row["seed"]),
            "candidate_version": int(row["candidate_version"]),
        },
        "group": f'{row["object"]}/{row["sequence"]}',
        "event_features": event_features,
        "observations": observations,
        "teacher_labels": {
            "selected_source": row["selected_source"],
            "commit_candidate": bool(row["committed"]),
            "select_loo": row["selected_source"] == "loo",
            "fallback_stable": row["selected_source"] == "stable",
            "full_harmful_vs_stable": full_iou > stable_iou,
            "selected_iou_gain_vs_full": full_iou - selected_iou,
            "selected_pose_gain_vs_full": float(
                row["same_run_original_pose_inconsistency"])
                - float(row["pose_inconsistency"]),
            "selected_temporal_gain_vs_full": float(
                row["same_run_original_temporal_std"])
                - float(row["temporal_std"]),
            "loo_available": row.get("proposed_mean_iou_loss") is not None,
            "candidate_feasibility": {
                name: bool(values["feasible"])
                for name, values in feasibility.items()},
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("selected_results", type=Path)
    parser.add_argument("p1v2_frames", type=Path)
    parser.add_argument("parent_gate", type=Path)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-zero-event-seeds", action="store_true",
        help="Allow expected seeds with no teacher events; run completeness "
             "must be audited separately.")
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text())
    expected_hashes = {
        args.selected_results: protocol["teacher"]["selected_results_sha256"],
        args.p1v2_frames: protocol["teacher"]["p1v2_frames_sha256"],
        args.parent_gate: protocol["teacher"]["parent_gate_sha256"],
    }
    for path, expected in expected_hashes.items():
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"hash mismatch for {path}: {actual}")

    rows = [
        row for row in load_jsonl(args.selected_results)
        if row.get("policy") == protocol["teacher"]["policy"]]
    if not rows:
        raise ValueError("no frozen teacher rows")
    observed_seeds = sorted({int(row["seed"]) for row in rows})
    expected_seeds = sorted(protocol["teacher"]["development_seeds"])
    if not seed_set_valid(
            observed_seeds, expected_seeds,
            allow_zero_event_seeds=args.allow_zero_event_seeds):
        raise ValueError(
            f"teacher seeds differ: {observed_seeds} != {expected_seeds}")
    locked = set(protocol["teacher"]["locked_report_only_seeds"])
    if locked.intersection(observed_seeds):
        raise ValueError("locked independent seed entered student dataset")

    p1v2_rows = load_csv_index(args.p1v2_frames)
    gate_rows = load_csv_index(args.parent_gate)
    events = [
        build_event(row, p1v2_rows, gate_rows, protocol) for row in rows]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(event, sort_keys=True) for event in events) + "\n")

    observations = [item for event in events for item in event["observations"]]
    summary = {
        "schema_version": 1,
        "stage": "stage4_p3",
        "mode": "offline_teacher_distillation_dataset",
        "events": len(events),
        "observations": len(observations),
        "available_observations": sum(
            item["features_available"] for item in observations),
        "missing_anchor_observations": sum(
            not item["features_available"] and item["is_anchor"]
            for item in observations),
        "teacher_keep_observations": sum(
            item["teacher_keep"] for item in observations),
        "teacher_selection_counts": dict(sorted(Counter(
            event["teacher_labels"]["selected_source"]
            for event in events).items())),
        "groups": sorted({event["group"] for event in events}),
        "seeds": observed_seeds,
        "dataset_sha256": sha256(args.output),
        "source_sha256": {
            "selected_results": sha256(args.selected_results),
            "p1v2_frames": sha256(args.p1v2_frames),
            "parent_gate": sha256(args.parent_gate),
            "protocol": sha256(args.protocol),
        },
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
