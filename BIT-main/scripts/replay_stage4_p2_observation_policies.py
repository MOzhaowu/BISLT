#!/usr/bin/env python3
"""Replay frozen Stage-4 decisions over recorded candidate-training events.

This is the first P2 replay layer. It reconstructs which communication frames
trained each recorded model candidate and compares hard filtering with soft
weighting. It measures observation exposure and bad-candidate coverage; it does
not claim counterfactual tracking accuracy without re-optimizing geometry.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


POLICIES = (
    "original",
    "parent_soft",
    "parent_hard_nonaccept",
    "p1v2_soft",
    "p1v2_soft_floor25",
    "p1v2_soft_floor50",
    "p1v2_soft_floor75",
    "p1v2_hard_nonaccept",
)


def identity(row):
    return (row["object"], str(row["sequence"]).zfill(2),
            int(row["seed"]), int(row["frame_index"]))


def load_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def split_batches(records):
    batches = []
    for record in records:
        batch_index = int(record["batch_index"])
        if batch_index == 0:
            batches.append([])
        if not batches or batch_index != len(batches[-1]):
            raise ValueError("non-contiguous mask batch indices")
        batches[-1].append(record)
    return batches


def policy_weight(policy, row, threshold):
    if row is None:
        return 1.0
    parent_decision = row["baseline_decision"]
    candidate_decision = row["decision"]
    parent_weight = float(row["parent_weight"])
    score = float(row["pose_risk_drawup"])
    if policy == "original":
        return 1.0
    if policy == "parent_soft":
        return parent_weight
    if policy == "parent_hard_nonaccept":
        return float(parent_decision == "accept")
    if policy == "p1v2_hard_nonaccept":
        return float(candidate_decision == "accept")
    if policy == "p1v2_soft" or policy.startswith("p1v2_soft_floor"):
        floor = (float(policy.removeprefix("p1v2_soft_floor"))/100.0
                 if policy.startswith("p1v2_soft_floor") else 0.0)
        if parent_decision == "reject":
            return 0.0
        if parent_decision == "downweight":
            return max(floor, parent_weight)
        if candidate_decision == "downweight":
            return max(floor, min(1.0, threshold/max(score, threshold)))
        return 1.0
    raise KeyError(policy)


def failure_delta(diagnostics, version, window=3):
    update = next((int(row["frame_index"]) for row in diagnostics
                   if int(float(row.get("model_reloaded_after_frame", 0))) == 1
                   and int(row["model_version_after"]) == version), None)
    if update is None:
        return None, None, None
    by_frame = {int(row["frame_index"]): row for row in diagnostics}

    def rate(indices):
        values = [1-int(by_frame[index]["success_5deg_50mm"])
                  for index in indices if index in by_frame]
        return sum(values)/len(values) if values else None

    pre = rate(range(max(0, update-window+1), update+1))
    post = rate(range(update+1, update+window+1))
    delta = None if pre is None or post is None else post-pre
    return update, pre, delta


def attach_parent_weights(candidate_rows, parent_rows):
    parent = {identity(row): row for row in parent_rows}
    output = {}
    for row in candidate_rows:
        key = identity(row)
        if key not in parent:
            raise ValueError(f"missing parent gate row for {key}")
        output[key] = dict(row, parent_weight=parent[key]["mask_pose_weight"])
    return output


def replay_events(run_root, decisions, threshold, run_glob):
    output = []
    pattern = f"*/*/{run_glob}/model_registry.jsonl"
    for registry_path in sorted(run_root.glob(pattern)):
        relative = registry_path.relative_to(run_root)
        object_name, sequence, run_tag = relative.parts[:3]
        seed = int(run_tag.rsplit("s", 1)[1])
        run_dir = registry_path.parent
        registry = load_jsonl(registry_path)
        mask_records = load_jsonl(
            run_dir/"mask_uncertainty"/"mask_uncertainty.jsonl")
        batches = split_batches(mask_records)
        if len(batches) != len(registry):
            raise ValueError(
                f"candidate/batch count mismatch in {run_dir}: "
                f"{len(registry)} != {len(batches)}")
        diagnostics = load_csv(run_dir/"diagnostics.csv")
        for record, batch in zip(registry, batches):
            version = int(record["version"])
            if version == 1:
                continue
            training_indices = [int(value)
                                for value in record["training_indices"]]
            if any(index >= len(batch) for index in training_indices):
                raise ValueError(f"training index outside batch in {run_dir}")
            frames = [int(batch[index]["frame_index"])
                      for index in training_indices]
            aligned = [decisions.get(
                (object_name, sequence, seed, frame)) for frame in frames]
            update_frame, pre_failure, failure_change = failure_delta(
                diagnostics, version)
            stable_loss = record.get("stable_validation_iou_loss")
            candidate_loss = record.get("candidate_validation_iou_loss")
            degraded = (stable_loss is not None and candidate_loss is not None
                        and float(candidate_loss) > float(stable_loss))
            for policy in POLICIES:
                weights = [policy_weight(policy, row, threshold)
                           for row in aligned]
                labeled = [(row, weight) for row, weight in zip(aligned, weights)
                           if row is not None]
                unsafe = [(row, weight) for row, weight in labeled
                          if int(row["unsafe_observation"])]
                safe = [(row, weight) for row, weight in labeled
                        if not int(row["unsafe_observation"])]
                output.append({
                    "object": object_name,
                    "sequence": sequence,
                    "seed": seed,
                    "candidate_version": version,
                    "policy": policy,
                    "training_frames": len(frames),
                    "training_frame_indices": ";".join(map(str, frames)),
                    "aligned_labeled_frames": len(labeled),
                    "warmup_unscored_frames": sum(row is None for row in aligned),
                    "unsafe_frames": len(unsafe),
                    "safe_frames": len(safe),
                    "effective_training_mass": sum(weights),
                    "effective_unsafe_mass": sum(weight for _, weight in unsafe),
                    "effective_safe_mass": sum(weight for _, weight in safe),
                    "filtered_frames": sum(weight == 0 for weight in weights),
                    "generation_feasible": int(sum(weight > 0 for weight in weights) >= 2),
                    "candidate_accepted": int(bool(record["accepted"])),
                    "candidate_degraded": int(degraded),
                    "validation_iou_change": (
                        "" if stable_loss is None or candidate_loss is None
                        else float(candidate_loss)-float(stable_loss)),
                    "update_frame": "" if update_frame is None else update_frame,
                    "pre_update_failure_rate": (
                        "" if pre_failure is None else pre_failure),
                    "post_minus_pre_failure_rate": (
                        "" if failure_change is None else failure_change),
                })
    return output


def summarize(rows):
    variants = {}
    for policy in POLICIES:
        selected = [row for row in rows if row["policy"] == policy]
        degraded = [row for row in selected if row["candidate_degraded"]]
        unsafe_frames = sum(row["unsafe_frames"] for row in selected)
        safe_frames = sum(row["safe_frames"] for row in selected)
        variants[policy] = {
            "candidate_events": len(selected),
            "degraded_candidate_events": len(degraded),
            "blocked_candidate_events": sum(
                not row["generation_feasible"] for row in selected),
            "blocked_degraded_candidate_events": sum(
                not row["generation_feasible"] for row in degraded),
            "degraded_candidate_block_recall": (
                sum(not row["generation_feasible"] for row in degraded)
                / len(degraded) if degraded else None),
            "unsafe_observation_exposure_reduction": (
                1-sum(row["effective_unsafe_mass"] for row in selected)
                / unsafe_frames if unsafe_frames else None),
            "safe_observation_exposure_reduction": (
                1-sum(row["effective_safe_mass"] for row in selected)
                / safe_frames if safe_frames else None),
            "mean_effective_training_mass": (
                sum(row["effective_training_mass"] for row in selected)
                / len(selected) if selected else None),
        }
    return {
        "schema_version": 1,
        "stage": 4,
        "mode": "P2_recorded_observation_policy_replay",
        "scope": (
            "candidate-training exposure and generation feasibility only; "
            "counterfactual geometry and tracking metrics not inferred"),
        "events": len(rows)//len(POLICIES),
        "policies": variants,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--candidate-frames", type=Path, required=True)
    parser.add_argument("--parent-frames", type=Path, required=True)
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--run-glob", default="p1v2_reconfirm_s*")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.frozen_config.read_text())
    decisions = attach_parent_weights(
        load_csv(args.candidate_frames), load_csv(args.parent_frames))
    rows = replay_events(
        args.run_root, decisions,
        float(config["method"]["demotion_threshold"]), args.run_glob)
    if not rows:
        raise ValueError("no replayable candidate events")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir/"p2_observation_replay_events.csv").open(
            "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    (args.output_dir/"p2_observation_replay_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
