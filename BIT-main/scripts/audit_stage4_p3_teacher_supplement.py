#!/usr/bin/env python3
"""Audit the preregistered Stage 4 P3 teacher supplement."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def capture_status(capture_root, groups, seeds):
    runs = []
    for seed in seeds:
        for group in groups:
            object_name, sequence = group.split("/", 1)
            run = capture_root / object_name / sequence / f"p2_capture_s{seed}"
            errors = []
            for name in ("manifest.json", "completeness.json", "exit_code.txt"):
                if not (run / name).is_file():
                    errors.append(f"missing_{name}")
            if not errors:
                manifest = json.loads((run / "manifest.json").read_text())
                complete = json.loads((run / "completeness.json").read_text())
                if (manifest.get("object") != object_name
                        or str(manifest.get("sequence")).zfill(2) != sequence
                        or int(manifest.get("seed", -1)) != seed):
                    errors.append("manifest_identity_mismatch")
                if complete.get("invalid") != 0 or complete.get("valid") != 1:
                    errors.append("completeness_invalid")
                if (run / "exit_code.txt").read_text().strip() != "0":
                    errors.append("nonzero_exit_code")
            runs.append({"group": group, "seed": seed, "path": str(run),
                         "valid": not errors, "errors": errors})
    return runs


def missing_feature_counts(events, warmup_frames):
    counts = Counter()
    for event in events:
        for observation in event["observations"]:
            if observation["features_available"]:
                counts["available"] += 1
                continue
            counts["missing"] += 1
            if observation["is_anchor"]:
                counts["missing_anchor"] += 1
            elif int(observation["frame_index"]) < warmup_frames:
                counts["missing_pose_warmup_nonanchor"] += 1
            else:
                counts["missing_unexplained"] += 1
    return dict(sorted(counts.items()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--supplement-dataset", type=Path, required=True)
    parser.add_argument("--combined-dataset", type=Path, required=True)
    parser.add_argument("--combined-audit", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text())
    parent = protocol["parent"]
    workspace = Path(__file__).resolve().parents[2]
    parent_dataset = workspace / parent["combined_teacher_dataset"]
    parent_audit = workspace / parent["quality_audit"]
    if sha256(parent_dataset) != parent["combined_teacher_dataset_sha256"]:
        raise ValueError("parent combined teacher dataset hash mismatch")
    if sha256(parent_audit) != parent["quality_audit_sha256"]:
        raise ValueError("parent quality audit hash mismatch")

    design = protocol["supplement_design"]
    groups = design["groups"]
    seeds = design["development_seeds"]
    supplement_events = load_jsonl(args.supplement_dataset)
    combined_events = load_jsonl(args.combined_dataset)
    combined_audit = json.loads(args.combined_audit.read_text())
    observed_supplement_seeds = sorted({
        int(event["identity"]["seed"]) for event in supplement_events})
    observed_supplement_groups = sorted({
        event["group"] for event in supplement_events})

    runs = capture_status(args.capture_root, groups, seeds)
    mixed_groups = set(combined_audit["checks"][
        "groups_with_both_full_safe_harmful"]["groups"])
    parent_mixed = set(parent["observed_mixed_groups"])
    targeted = set(groups)
    newly_mixed = sorted((mixed_groups - parent_mixed) & targeted)
    locked = protocol["locked_sets"]
    locked_seeds = set(locked["previous_p2_report_only_seeds"])
    locked_seeds.update(locked["formula_confirmation_seeds"])
    locked_seeds.update(locked["external_independent_seeds"])
    observed_combined_seeds = set(combined_audit["seeds"])
    retained_checks = {
        name: bool(combined_audit["checks"][name]["pass"])
        for name in (
            "expected_sequence_groups_present",
            "minimum_sequence_groups_with_events",
            "minimum_total_sequence_groups",
            "minimum_observations",
            "groups_with_both_keep_drop",
        )
    }
    primary = protocol["primary_acceptance"]
    checks = {
        "all_planned_runs_accounted_for": {
            "actual": sum(run["valid"] for run in runs),
            "required": design["planned_runs"],
            "pass": len(runs) == design["planned_runs"] and all(
                run["valid"] for run in runs),
        },
        "supplement_seed_set": {
            "actual": observed_supplement_seeds,
            "required_subset_of": seeds,
            "pass": set(observed_supplement_seeds).issubset(seeds),
        },
        "supplement_event_groups": {
            "actual": observed_supplement_groups,
            "required_subset_of": groups,
            "pass": set(observed_supplement_groups).issubset(targeted),
        },
        "newly_mixed_previously_single_label_groups": {
            "actual": len(newly_mixed), "groups": newly_mixed,
            "required": primary[
                "newly_mixed_previously_single_label_groups_min"],
            "pass": len(newly_mixed) >= primary[
                "newly_mixed_previously_single_label_groups_min"],
        },
        "total_groups_with_both_full_safe_harmful": {
            "actual": len(mixed_groups), "groups": sorted(mixed_groups),
            "required": primary["total_groups_with_both_full_safe_harmful_min"],
            "pass": len(mixed_groups) >= primary[
                "total_groups_with_both_full_safe_harmful_min"],
        },
        "locked_seed_leakage": {
            "actual": sorted(locked_seeds & observed_combined_seeds),
            "locked_seeds": sorted(locked_seeds),
            "pass": not (locked_seeds & observed_combined_seeds),
        },
        "retained_student_development_gates": {
            "details": retained_checks,
            "pass": all(retained_checks.values()),
        },
    }
    report = {
        "schema_version": 1,
        "stage": "stage4_p3",
        "mode": "preregistered_teacher_supplement_audit",
        "accepted_for_student_development": all(
            check["pass"] for check in checks.values()),
        "checks": checks,
        "capture_runs": runs,
        "supplement": {
            "events": len(supplement_events),
            "dataset_sha256": sha256(args.supplement_dataset),
        },
        "combined": {
            "events": len(combined_events),
            "observations": sum(len(event["observations"])
                                for event in combined_events),
            "dataset_sha256": sha256(args.combined_dataset),
            "quality_audit_sha256": sha256(args.combined_audit),
            "missing_features": missing_feature_counts(
                combined_events,
                protocol["causal_feature_policy"][
                    "pose_warmup_frames_per_run"]),
        },
        "protocol_sha256": sha256(args.protocol),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
