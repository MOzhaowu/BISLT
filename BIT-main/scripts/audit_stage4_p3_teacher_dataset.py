#!/usr/bin/env python3
"""Combine and quality-audit frozen Stage 4 P3 teacher datasets."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path):
    return [
        json.loads(line) for line in path.read_text().splitlines()
        if line.strip()
    ]


def event_key(event):
    identity = event["identity"]
    return (
        identity["object"], identity["sequence"], int(identity["seed"]),
        int(identity["candidate_version"]),
    )


def label_counts(events, selector):
    counts = Counter(bool(selector(event)) for event in events)
    return {"negative": counts[False], "positive": counts[True]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--base-protocol", type=Path, required=True)
    parser.add_argument("--expansion-protocol", type=Path, required=True)
    parser.add_argument("--combined-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = json.loads(args.base_protocol.read_text())
    expansion = json.loads(args.expansion_protocol.read_text())
    events = []
    source_hashes = {}
    seen = set()
    duplicates = []
    for path in args.datasets:
        source_hashes[str(path)] = sha256(path)
        for event in load_jsonl(path):
            key = event_key(event)
            if key in seen:
                duplicates.append(key)
            seen.add(key)
            events.append(event)

    if not events:
        raise ValueError("no teacher events")
    if duplicates:
        raise ValueError(f"duplicate teacher event identities: {duplicates[:5]}")

    events.sort(key=event_key)
    args.combined_output.parent.mkdir(parents=True, exist_ok=True)
    args.combined_output.write_text(
        "\n".join(json.dumps(event, sort_keys=True) for event in events) + "\n")

    groups = sorted({event["group"] for event in events})
    seeds = sorted({int(event["identity"]["seed"]) for event in events})
    expected_groups = sorted(set(
        expansion["existing_teacher_groups"]
        + expansion["development_expansion"]["groups"]
    ))
    locked_seeds = set(base["teacher"]["locked_report_only_seeds"])
    locked_seeds.update(
        expansion["formula_confirmation"]["seeds"])
    locked_seeds.update(
        expansion["external_independent_validation"]["seeds"])

    by_group = defaultdict(list)
    for event in events:
        by_group[event["group"]].append(event)
    group_summary = {}
    mixed_keep_drop = []
    mixed_full_safety = []
    total_observations = 0
    available_observations = 0
    for group in groups:
        group_events = by_group[group]
        observations = [
            observation for event in group_events
            for observation in event["observations"]
        ]
        keep = Counter(bool(item["teacher_keep"]) for item in observations)
        harmful = Counter(bool(
            event["teacher_labels"]["full_harmful_vs_stable"])
            for event in group_events)
        if keep[False] and keep[True]:
            mixed_keep_drop.append(group)
        if harmful[False] and harmful[True]:
            mixed_full_safety.append(group)
        total_observations += len(observations)
        available_observations += sum(
            bool(item["features_available"]) for item in observations)
        group_summary[group] = {
            "events": len(group_events),
            "seeds": sorted({
                int(event["identity"]["seed"]) for event in group_events}),
            "observations": len(observations),
            "available_observations": sum(
                bool(item["features_available"]) for item in observations),
            "keep_labels": {
                "drop": keep[False], "keep": keep[True]},
            "full_harm_labels": {
                "safe": harmful[False], "harmful": harmful[True]},
        }

    gate = expansion["student_development_gate"]
    checks = {
        "expected_sequence_groups_present": {
            "actual": groups,
            "expected": expected_groups,
            "pass": groups == expected_groups,
        },
        "minimum_sequence_groups_with_events": {
            "actual": len(groups),
            "required": int(gate["minimum_sequence_groups_with_events"]),
            "pass": len(groups) >= int(
                gate["minimum_sequence_groups_with_events"]),
        },
        "minimum_total_sequence_groups": {
            "actual": len(groups),
            "required": int(expansion["development_expansion"][
                "minimum_total_sequence_groups"]),
            "pass": len(groups) >= int(expansion["development_expansion"][
                "minimum_total_sequence_groups"]),
        },
        "minimum_observations": {
            "actual": total_observations,
            "required": int(gate["minimum_observations"]),
            "pass": total_observations >= int(gate["minimum_observations"]),
        },
        "groups_with_both_keep_drop": {
            "actual": len(mixed_keep_drop),
            "groups": mixed_keep_drop,
            "required": int(gate[
                "both_keep_drop_labels_required_in_at_least_groups"]),
            "pass": len(mixed_keep_drop) >= int(gate[
                "both_keep_drop_labels_required_in_at_least_groups"]),
        },
        "groups_with_both_full_safe_harmful": {
            "actual": len(mixed_full_safety),
            "groups": mixed_full_safety,
            "required": int(gate[
                "both_full_safe_harmful_labels_required_in_at_least_groups"]),
            "pass": len(mixed_full_safety) >= int(gate[
                "both_full_safe_harmful_labels_required_in_at_least_groups"]),
        },
        "locked_seed_leakage": {
            "actual": sorted(locked_seeds.intersection(seeds)),
            "locked_seeds": sorted(locked_seeds),
            "pass": not locked_seeds.intersection(seeds),
        },
    }
    report = {
        "schema_version": 1,
        "stage": "stage4_p3",
        "mode": "combined_teacher_dataset_quality_audit",
        "student_development_ready": all(
            check["pass"] for check in checks.values()),
        "events": len(events),
        "observations": total_observations,
        "available_observations": available_observations,
        "groups": groups,
        "seeds": seeds,
        "observation_keep_labels": label_counts(
            [item for event in events for item in event["observations"]],
            lambda item: item["teacher_keep"]),
        "event_full_harm_labels": label_counts(
            events,
            lambda event: event["teacher_labels"][
                "full_harmful_vs_stable"]),
        "teacher_selection_counts": dict(sorted(Counter(
            event["teacher_labels"]["selected_source"]
            for event in events).items())),
        "checks": checks,
        "per_group": group_summary,
        "combined_dataset_sha256": sha256(args.combined_output),
        "source_dataset_sha256": source_hashes,
        "protocol_sha256": {
            "base": sha256(args.base_protocol),
            "expansion": sha256(args.expansion_protocol),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
