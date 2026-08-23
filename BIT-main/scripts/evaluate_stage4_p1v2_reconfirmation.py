#!/usr/bin/env python3
"""Evaluate the preregistered fixed P1-v2 gate on untouched seeds."""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_stage4_p1v2_hierarchical_gate import (
    apply_demotion,
    decision_metrics,
    load_rows,
    sequence_key,
)


def group_key(row):
    return (row["object"], row["sequence"], row["seed"])


def protected_unsafe_recall(rows, decision_field):
    unsafe = [row for row in rows if row["unsafe_observation"]]
    if not unsafe:
        return None
    return sum(row[decision_field] != "accept" for row in unsafe)/len(unsafe)


def bootstrap(rows, replicates, seed):
    groups = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)
    keys = sorted(groups)
    generator = np.random.default_rng(seed)
    deltas = []
    for _ in range(replicates):
        sample = [groups[keys[index]]
                  for index in generator.integers(0, len(keys), len(keys))]
        sample_rows = [row for group in sample for row in group]
        baseline = protected_unsafe_recall(sample_rows, "baseline_decision")
        candidate = protected_unsafe_recall(sample_rows, "decision")
        if baseline is not None and candidate is not None:
            deltas.append(candidate-baseline)
    values = np.asarray(deltas, dtype=float)
    return {
        "unit": "object_sequence_seed",
        "replicates": replicates,
        "seed": seed,
        "protected_unsafe_delta_ci95": np.quantile(
            values, [0.025, 0.975]).tolist(),
        "probability_positive": float(np.mean(values > 0)),
        "probability_negative": float(np.mean(values < 0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gate_csv", type=Path)
    parser.add_argument("pose_jsonl", type=Path)
    parser.add_argument("frozen_config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.frozen_config.read_text())
    rows = load_rows(args.gate_csv, args.pose_jsonl)
    method = config["method"]
    threshold = float(method["demotion_threshold"])
    score_field = method["score_field"]
    predictions = apply_demotion(
        rows, [row[score_field] for row in rows], threshold)
    baseline_rows = [dict(row, decision=row["baseline_decision"]) for row in rows]

    baseline = decision_metrics(baseline_rows)
    candidate = decision_metrics(predictions)
    targets = {}
    for target in config["reconfirmation"]["target_sequences"]:
        target_rows = [row for row in predictions if sequence_key(row) == target]
        target_baseline = [dict(row, decision=row["baseline_decision"])
                           for row in target_rows]
        targets[target] = {
            "baseline": decision_metrics(target_baseline),
            "candidate": decision_metrics(target_rows),
        }

    seeds = {}
    mixed_directions = []
    for key in sorted({group_key(row) for row in predictions}):
        seed_rows = [row for row in predictions if group_key(row) == key]
        unsafe = sum(row["unsafe_observation"] for row in seed_rows)
        safe = len(seed_rows)-unsafe
        baseline_recall = protected_unsafe_recall(
            seed_rows, "baseline_decision")
        candidate_recall = protected_unsafe_recall(seed_rows, "decision")
        mixed = unsafe > 0 and safe > 0
        if mixed:
            mixed_directions.append(candidate_recall-baseline_recall)
        seeds[f"{key[0]}/{key[1]}/seed{key[2]}"] = {
            "frames": len(seed_rows),
            "unsafe": unsafe,
            "safe": safe,
            "mixed_labels": mixed,
            "baseline_protected_unsafe_recall": baseline_recall,
            "candidate_protected_unsafe_recall": candidate_recall,
            "candidate_safe_nonaccept_rate": decision_metrics(
                seed_rows)["safe_nonaccept_rate"],
        }

    bootstrap_result = bootstrap(
        predictions,
        int(config["reconfirmation"]["bootstrap_replicates"]),
        int(config["reconfirmation"]["bootstrap_seed"]),
    )
    acceptance = config["acceptance"]
    checks = {
        "safe_reject_rate": candidate["safe_reject_rate"]
        <= acceptance["safe_reject_rate_max"],
        "safe_nonaccept_rate": candidate["safe_nonaccept_rate"]
        <= acceptance["safe_nonaccept_rate_max"],
        "protected_unsafe_not_below_parent":
            candidate["protected_unsafe_recall"]
            >= baseline["protected_unsafe_recall"],
        "both_targets_not_below_parent": all(
            item["candidate"]["protected_unsafe_recall"]
            >= item["baseline"]["protected_unsafe_recall"]
            for item in targets.values()),
        "at_least_one_target_strictly_improved": any(
            item["candidate"]["protected_unsafe_recall"]
            > item["baseline"]["protected_unsafe_recall"]
            for item in targets.values()),
        "minimum_mixed_label_seeds": len(mixed_directions)
        >= config["reconfirmation"]["minimum_mixed_label_seeds"],
        "mixed_label_seed_direction_consistent": bool(mixed_directions)
        and all(value >= 0 for value in mixed_directions),
        "bootstrap_no_significant_reverse_change":
            bootstrap_result["protected_unsafe_delta_ci95"][1] >= 0,
    }
    summary = {
        "schema_version": 1,
        "stage": 4,
        "mode": "P1_v2_fixed_formula_independent_reconfirmation",
        "frozen_method": method,
        "baseline": baseline,
        "candidate": candidate,
        "targets": targets,
        "seeds": seeds,
        "mixed_label_seed_count": len(mixed_directions),
        "bootstrap": bootstrap_result,
        "checks": checks,
        "enter_p2": all(checks.values()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir/"reconfirmation_frames.csv").open(
            "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    (args.output_dir/"reconfirmation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
