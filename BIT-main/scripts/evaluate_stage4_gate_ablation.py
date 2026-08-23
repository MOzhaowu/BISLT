#!/usr/bin/env python3
"""Evaluate Stage-4 gate feature ablations with threshold LOSO.

Every held-out object/sequence receives thresholds derived only from the other
sequences. Candidate thresholds are training-score quantiles so reliability
proxies with different numeric scales can be compared fairly.
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from build_stage4_shadow_gate import DECISIONS, binary_auroc, gate_decision


VARIANTS = {
    "mask_only": lambda row: row["q_mask"],
    "pose_only": lambda row: row["q_pose"],
    "mask_pose": lambda row: row["q_mask"] * row["q_pose"],
    "visibility_only": lambda row: row["q_visibility_audit"],
    "information_only": lambda row: row["q_information_audit"],
    "mask_pose_visibility": lambda row: (
        row["q_mask"] * row["q_pose"] * row["q_visibility_audit"]),
    "mask_pose_information": lambda row: (
        row["q_mask"] * row["q_pose"] * row["q_information_audit"]),
    "mask_pose_visibility_information": lambda row: (
        row["q_mask"] * row["q_pose"] * row["q_visibility_audit"] *
        row["q_information_audit"]),
    "causal_visibility_only": lambda row: row["q_visibility_causal_relative"],
    "causal_information_only": lambda row: row["q_information_causal_relative"],
    "mask_pose_causal_visibility": lambda row: (
        row["q_mask"] * row["q_pose"] * row["q_visibility_causal_relative"]),
    "mask_pose_causal_information": lambda row: (
        row["q_mask"] * row["q_pose"] * row["q_information_causal_relative"]),
}


def sequence_key(row):
    return f"{row['object']}/{str(row['sequence']).zfill(2)}"


def read_rows(path):
    numeric = (
        "q_mask", "q_pose", "q_visibility_audit", "q_information_audit",
    )
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for field in numeric:
            row[field] = float(row[field])
        row["unsafe_observation"] = int(row["unsafe_observation"])
    return add_causal_relative_features(rows)


def add_causal_relative_features(rows):
    """Normalize audit features by the run's causal running maximum."""
    maxima = {}
    ordered = sorted(rows, key=lambda row: (
        row["object"], str(row["sequence"]).zfill(2), int(row["seed"]),
        int(row["frame_index"])))
    for row in ordered:
        key = (row["object"], str(row["sequence"]).zfill(2), int(row["seed"]))
        previous = maxima.get(key, (
            row["q_visibility_audit"], row["q_information_audit"]))
        visibility_max = max(previous[0], row["q_visibility_audit"])
        information_max = max(previous[1], row["q_information_audit"])
        maxima[key] = (visibility_max, information_max)
        row["q_visibility_causal_relative"] = (
            row["q_visibility_audit"] / visibility_max if visibility_max else 1.0)
        row["q_information_causal_relative"] = (
            row["q_information_audit"] / information_max if information_max else 1.0)
    return rows


def add_variant_score(rows, variant):
    score = VARIANTS[variant]
    return [dict(row, gate_reliability=float(score(row))) for row in rows]


def decision_rates(rows, accept, reject, decision_field=None):
    decisions = ([row[decision_field] for row in rows] if decision_field else
                 [gate_decision(row["gate_reliability"], accept, reject)[0]
                  for row in rows])
    unsafe = [decision for row, decision in zip(rows, decisions)
              if row["unsafe_observation"]]
    safe = [decision for row, decision in zip(rows, decisions)
            if not row["unsafe_observation"]]

    def rate(selected, predicate):
        return (sum(predicate(value) for value in selected) / len(selected)
                if selected else None)

    return {
        "reject_unsafe_recall": rate(unsafe, lambda value: value == "reject"),
        "protected_unsafe_recall": rate(unsafe, lambda value: value != "accept"),
        "safe_reject_rate": rate(safe, lambda value: value == "reject"),
        "safe_nonaccept_rate": rate(safe, lambda value: value != "accept"),
    }


def quantile_grid(rows, quantiles):
    values = np.asarray([row["gate_reliability"] for row in rows], dtype=float)
    return sorted(set(float(value) for value in np.quantile(values, quantiles)))


def select_thresholds(rows, accept_quantiles, reject_quantiles,
                      max_safe_reject, max_safe_nonaccept):
    accept_grid = quantile_grid(rows, accept_quantiles)
    reject_grid = quantile_grid(rows, reject_quantiles)
    candidates = []
    for accept in accept_grid:
        for reject in reject_grid:
            if reject >= accept:
                continue
            result = decision_rates(rows, accept, reject)
            if any(value is None for value in result.values()):
                continue
            feasible = (result["safe_reject_rate"] <= max_safe_reject and
                        result["safe_nonaccept_rate"] <= max_safe_nonaccept)
            objective = (result["reject_unsafe_recall"] +
                         0.25 * result["protected_unsafe_recall"])
            candidates.append((
                feasible, objective, -result["safe_nonaccept_rate"],
                -result["safe_reject_rate"], -accept, -reject,
                accept, reject, result,
            ))
    feasible = [candidate for candidate in candidates if candidate[0]]
    if not feasible:
        raise ValueError("no threshold pair satisfies safe-frame constraints")
    selected = max(feasible)
    return selected[6], selected[7], selected[8]


def threshold_loso(rows, accept_quantiles, reject_quantiles,
                   max_safe_reject, max_safe_nonaccept):
    groups = sorted({sequence_key(row) for row in rows})
    predictions, folds = [], {}
    for held_out in groups:
        train = [row for row in rows if sequence_key(row) != held_out]
        test = [row for row in rows if sequence_key(row) == held_out]
        accept, reject, train_rates = select_thresholds(
            train, accept_quantiles, reject_quantiles,
            max_safe_reject, max_safe_nonaccept)
        fold_predictions = []
        for row in test:
            decision, weight = gate_decision(
                row["gate_reliability"], accept, reject)
            output = dict(row)
            output.update({
                "held_out_sequence": held_out,
                "selected_accept_threshold": accept,
                "selected_reject_threshold": reject,
                "ablation_decision": decision,
                "ablation_weight": weight,
            })
            predictions.append(output)
            fold_predictions.append(output)
        folds[held_out] = {
            "samples": len(test),
            "unsafe": sum(row["unsafe_observation"] for row in test),
            "selected_accept_threshold": accept,
            "selected_reject_threshold": reject,
            "training_rates": train_rates,
            "test_rates": decision_rates(
                fold_predictions, accept, reject, "ablation_decision"),
            "test_risk_auroc": binary_auroc(
                [row["unsafe_observation"] for row in test],
                [1.0 - row["gate_reliability"] for row in test]),
        }
    return predictions, folds


def evaluate_predictions(rows):
    counts = Counter(row["ablation_decision"] for row in rows)
    rates = decision_rates(rows, 0.0, 0.0, "ablation_decision")
    mixed_aurocs = []
    for sequence in sorted({sequence_key(row) for row in rows}):
        selected = [row for row in rows if sequence_key(row) == sequence]
        auroc = binary_auroc(
            [row["unsafe_observation"] for row in selected],
            [1.0 - row["gate_reliability"] for row in selected])
        if auroc is not None:
            mixed_aurocs.append(auroc)
    return {
        "risk_auroc": binary_auroc(
            [row["unsafe_observation"] for row in rows],
            [1.0 - row["gate_reliability"] for row in rows]),
        "mixed_sequence_macro_risk_auroc": (
            float(np.mean(mixed_aurocs)) if mixed_aurocs else None),
        "mixed_sequence_count": len(mixed_aurocs),
        "decision_counts": {name: counts[name] for name in DECISIONS},
        **rates,
    }


def paired_sequence_bootstrap(all_predictions, baseline_name, replicates, seed):
    by_variant_group = {}
    for name, rows in all_predictions.items():
        groups = defaultdict(list)
        for row in rows:
            groups[sequence_key(row)].append(row)
        by_variant_group[name] = groups
    group_names = sorted(by_variant_group[baseline_name])
    rng = np.random.default_rng(seed)
    metrics = ("risk_auroc", "reject_unsafe_recall",
               "protected_unsafe_recall", "safe_reject_rate",
               "safe_nonaccept_rate")
    deltas = {name: {metric: [] for metric in metrics}
              for name in all_predictions if name != baseline_name}
    for _ in range(replicates):
        chosen = rng.choice(group_names, size=len(group_names), replace=True)
        baseline_rows = [row for group in chosen
                         for row in by_variant_group[baseline_name][group]]
        baseline = evaluate_predictions(baseline_rows)
        for name in deltas:
            selected_rows = [row for group in chosen
                             for row in by_variant_group[name][group]]
            selected = evaluate_predictions(selected_rows)
            for metric in metrics:
                if selected[metric] is not None and baseline[metric] is not None:
                    deltas[name][metric].append(selected[metric] - baseline[metric])
    return {
        name: {
            metric: {
                "lower": float(np.percentile(values, 2.5)),
                "median": float(np.percentile(values, 50.0)),
                "upper": float(np.percentile(values, 97.5)),
            }
            for metric, values in metric_values.items() if values
        }
        for name, metric_values in deltas.items()
    }


def write_predictions(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_quantiles(value):
    result = [float(item) for item in value.split(",") if item]
    if not result or min(result) < 0.0 or max(result) > 1.0:
        raise argparse.ArgumentTypeError("quantiles must be within [0, 1]")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("shadow_frames", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--accept-quantiles", type=parse_quantiles,
                        default=parse_quantiles("0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80"))
    parser.add_argument("--reject-quantiles", type=parse_quantiles,
                        default=parse_quantiles("0.00,0.05,0.10,0.20,0.30,0.40"))
    parser.add_argument("--max-safe-reject", type=float, default=0.10)
    parser.add_argument("--max-safe-nonaccept", type=float, default=0.50)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260822)
    args = parser.parse_args()
    source = read_rows(args.shadow_frames)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions, variants = {}, {}
    for name in VARIANTS:
        rows, folds = threshold_loso(
            add_variant_score(source, name), args.accept_quantiles,
            args.reject_quantiles, args.max_safe_reject,
            args.max_safe_nonaccept)
        predictions[name] = rows
        variants[name] = {"metrics": evaluate_predictions(rows), "folds": folds}
        write_predictions(args.output_dir / f"{name}_loso_frames.csv", rows)
    summary = {
        "schema_version": 1,
        "stage": 4,
        "mode": "P1_feature_ablation_threshold_loso",
        "label": "mask IoU<0.90 OR pose failure 5deg/5cm",
        "threshold_selection": (
            "training-fold score quantiles; held-out object/sequence unseen"),
        "constraints": {
            "max_training_safe_reject_rate": args.max_safe_reject,
            "max_training_safe_nonaccept_rate": args.max_safe_nonaccept,
        },
        "frames": len(source),
        "sequences": len({sequence_key(row) for row in source}),
        "variants": variants,
        "paired_sequence_bootstrap_delta_vs_mask_pose":
            paired_sequence_bootstrap(
                predictions, "mask_pose", args.bootstrap_replicates,
                args.bootstrap_seed),
        "bootstrap": {
            "unit": "object/sequence", "replicates": args.bootstrap_replicates,
            "seed": args.bootstrap_seed,
        },
    }
    (args.output_dir / "stage4_gate_ablation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({name: value["metrics"]
                      for name, value in variants.items()},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
