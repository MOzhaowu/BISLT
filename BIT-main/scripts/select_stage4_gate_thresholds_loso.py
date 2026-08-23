#!/usr/bin/env python3
"""Select Stage-4 shadow-gate thresholds with sequence-level LOSO."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from build_stage4_shadow_gate import DECISIONS, binary_auroc, gate_decision


def read_rows(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["base_reliability"] = float(row["base_reliability"])
        row["unsafe_observation"] = int(row["unsafe_observation"])
    return rows


def rates(rows, accept_threshold, reject_threshold):
    decisions = [gate_decision(row["base_reliability"], accept_threshold,
                               reject_threshold)[0] for row in rows]
    unsafe = [(row, decision) for row, decision in zip(rows, decisions)
              if row["unsafe_observation"]]
    safe = [(row, decision) for row, decision in zip(rows, decisions)
            if not row["unsafe_observation"]]

    def decision_rate(selected, predicate):
        return (sum(predicate(decision) for _, decision in selected) / len(selected)
                if selected else None)

    return {
        "unsafe_reject_recall": decision_rate(
            unsafe, lambda decision: decision == "reject"),
        "unsafe_protected_recall": decision_rate(
            unsafe, lambda decision: decision != "accept"),
        "safe_reject_rate": decision_rate(
            safe, lambda decision: decision == "reject"),
        "safe_nonaccept_rate": decision_rate(
            safe, lambda decision: decision != "accept"),
    }


def select_thresholds(rows, accept_grid, reject_grid, max_safe_reject,
                      max_safe_nonaccept):
    candidates = []
    for accept in accept_grid:
        for reject in reject_grid:
            if reject >= accept:
                continue
            result = rates(rows, accept, reject)
            if any(value is None for value in result.values()):
                continue
            feasible = (result["safe_reject_rate"] <= max_safe_reject and
                        result["safe_nonaccept_rate"] <= max_safe_nonaccept)
            objective = (result["unsafe_reject_recall"] +
                         0.25 * result["unsafe_protected_recall"])
            candidates.append((feasible, objective,
                               -result["safe_nonaccept_rate"],
                               -result["safe_reject_rate"],
                               -accept, -reject, accept, reject, result))
    feasible = [candidate for candidate in candidates if candidate[0]]
    if not feasible:
        raise ValueError("No threshold pair satisfies the safe-frame constraints")
    selected = max(feasible)
    return selected[6], selected[7], selected[8]


def evaluate_loso(rows, accept_grid, reject_grid, max_safe_reject,
                  max_safe_nonaccept):
    groups = sorted({f"{row['object']}/{str(row['sequence']).zfill(2)}"
                     for row in rows})
    predictions = []
    folds = {}
    for held_out in groups:
        train = [row for row in rows
                 if f"{row['object']}/{str(row['sequence']).zfill(2)}" != held_out]
        test = [row for row in rows
                if f"{row['object']}/{str(row['sequence']).zfill(2)}" == held_out]
        accept, reject, training_rates = select_thresholds(
            train, accept_grid, reject_grid, max_safe_reject,
            max_safe_nonaccept)
        fold_rows = []
        for row in test:
            decision, weight = gate_decision(
                row["base_reliability"], accept, reject)
            output = dict(row)
            output.update({
                "threshold_held_out_sequence": held_out,
                "selected_accept_threshold": accept,
                "selected_reject_threshold": reject,
                "loso_shadow_decision": decision,
                "loso_shadow_weight": weight,
            })
            predictions.append(output)
            fold_rows.append(output)
        folds[held_out] = {
            "selected_accept_threshold": accept,
            "selected_reject_threshold": reject,
            "training_rates": training_rates,
            "test_rates": rates(test, accept, reject),
        }
    return predictions, folds


def summarize(rows, folds, constraints):
    unsafe = [row for row in rows if row["unsafe_observation"]]
    safe = [row for row in rows if not row["unsafe_observation"]]

    def rate(selected, predicate):
        return (sum(predicate(row["loso_shadow_decision"]) for row in selected) /
                len(selected) if selected else None)

    counts = Counter(row["loso_shadow_decision"] for row in rows)
    return {
        "schema_version": 1,
        "stage": 4,
        "mode": "shadow_threshold_loso",
        "method": "thresholds selected without the held-out object/sequence",
        "objective": "maximize unsafe reject recall + 0.25 * unsafe protected recall",
        "training_constraints": constraints,
        "frames": len(rows),
        "sequences": len(folds),
        "decision_counts": {name: counts[name] for name in DECISIONS},
        "metrics": {
            "unsafe_observations": len(unsafe),
            "safe_observations": len(safe),
            "unsafe_risk_auroc": binary_auroc(
                [row["unsafe_observation"] for row in rows],
                [1.0 - row["base_reliability"] for row in rows]),
            "reject_unsafe_recall": rate(
                unsafe, lambda decision: decision == "reject"),
            "protected_unsafe_recall": rate(
                unsafe, lambda decision: decision != "accept"),
            "safe_reject_rate": rate(
                safe, lambda decision: decision == "reject"),
            "safe_nonaccept_rate": rate(
                safe, lambda decision: decision != "accept"),
        },
        "folds": folds,
    }


def parse_grid(value):
    return [float(item) for item in value.split(",") if item]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("shadow_frames", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--accept-grid", default="0.15,0.2,0.25,0.3,0.4,0.5")
    parser.add_argument("--reject-grid", default="0.01,0.025,0.05,0.075,0.1,0.15")
    parser.add_argument("--max-safe-reject", type=float, default=0.10)
    parser.add_argument("--max-safe-nonaccept", type=float, default=0.50)
    args = parser.parse_args()
    rows = read_rows(args.shadow_frames)
    predictions, folds = evaluate_loso(
        rows, parse_grid(args.accept_grid), parse_grid(args.reject_grid),
        args.max_safe_reject, args.max_safe_nonaccept)
    summary = summarize(predictions, folds, {
        "max_safe_reject_rate": args.max_safe_reject,
        "max_safe_nonaccept_rate": args.max_safe_nonaccept,
    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "threshold_loso_frames.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    (args.output_dir / "threshold_loso_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
