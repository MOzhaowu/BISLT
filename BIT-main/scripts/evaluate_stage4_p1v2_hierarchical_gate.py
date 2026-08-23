#!/usr/bin/env python3
"""Develop a leakage-safe Stage-4 P1-v2 hierarchical demotion gate.

The frozen q_mask*q_pose decision is the parent decision. A learned local-risk
model may only demote ``accept`` to ``downweight``; it can never promote a frame
or create a hard reject. Both regularization and the demotion threshold are
selected inside the outer sequence LOSO fold.

The independent-confirmation data is intentionally treated as development data
after its one-shot decision has been archived. Results from this script are not
an independent confirmation and must be re-tested on unseen mixed-label seeds.
"""

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_stage4_shadow_gate import binary_auroc
from fit_stage2_mask_nested_loso import fit, log_loss, predict


FEATURE_SETS = {
    "base_causal": [
        "base_risk_logit",
        "visibility_drawdown",
        "information_drawdown",
    ],
    "causal_risk_drawup": [
        "base_risk_drawup",
    ],
    "causal_risk_context": [
        "base_risk_drawup",
        "visibility_drawdown",
        "information_drawdown",
    ],
    "causal_pose_drawup": [
        "pose_risk_drawup",
    ],
    "causal_pose_context": [
        "pose_risk_drawup",
        "visibility_drawdown",
        "information_drawdown",
    ],
    "dense_pose_percentile": [
        "dense_pose_risk_percentile",
    ],
    "dense_pose_context": [
        "dense_pose_risk_percentile",
        "dense_pose_risk_robust_z",
        "dense_pose_risk_drawup",
    ],
    "base_local_residual": [
        "base_risk_logit",
        "log_contour_noise_variance",
        "contour_tail_over_noise",
        "contour_residual_centroid_offset",
        "contour_match_deficit",
    ],
    "base_local_hessian": [
        "base_risk_logit",
        "covariance_trace_relative_initial",
        "hessian_condition_relative_first",
    ],
    "base_local_combined": [
        "base_risk_logit",
        "visibility_drawdown",
        "information_drawdown",
        "log_contour_noise_variance",
        "contour_tail_over_noise",
        "contour_residual_centroid_offset",
        "contour_match_deficit",
        "covariance_trace_relative_initial",
        "hessian_condition_relative_first",
    ],
}

DIRECT_SCORE_SETS = {
    "direct_pose_drawup": "pose_risk_drawup",
    "direct_visibility_drawdown": "visibility_drawdown",
    "direct_information_drawdown": "information_drawdown",
    "direct_dense_pose_percentile": "dense_pose_risk_percentile",
}


def sequence_key(row):
    return f"{row['object']}/{str(row['sequence']).zfill(2)}"


def frame_key(row):
    return (row["object"], str(row["sequence"]).zfill(2),
            int(row["seed"]), int(row["frame_index"]))


def finite(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def add_causal_risk_features(rows):
    """Add run-relative q_base risk changes using no future observations."""
    state = {}
    for row in sorted(rows, key=lambda item: (
            item["object"], item["sequence"], item["seed"],
            item["frame_index"])):
        run = (row["object"], row["sequence"], row["seed"])
        risk = row["base_risk_logit"]
        pose_risk = row.get("pose_risk_logit", risk)
        if run not in state:
            state[run] = {"first": risk, "minimum": risk,
                          "pose_first": pose_risk, "pose_minimum": pose_risk}
        row["base_risk_delta_from_first"] = risk-state[run]["first"]
        row["base_risk_drawup"] = max(0.0, risk-state[run]["minimum"])
        row["pose_risk_delta_from_first"] = pose_risk-state[run]["pose_first"]
        row["pose_risk_drawup"] = max(
            0.0, pose_risk-state[run]["pose_minimum"])
        state[run]["minimum"] = min(state[run]["minimum"], risk)
        state[run]["pose_minimum"] = min(state[run]["pose_minimum"], pose_risk)
    return rows


def add_dense_pose_context(rows):
    """Add run-adaptive pose features using strictly preceding dense frames."""
    histories = defaultdict(list)
    feature = "log_normalized_covariance_trace_relative_to_initial_median"
    for row in sorted(rows, key=lambda item: (
            item["object"], str(item["sequence"]).zfill(2), item["seed"],
            item["frame_index"])):
        run = (row["object"], str(row["sequence"]).zfill(2), row["seed"])
        value = finite(row.get(feature))
        history = histories[run]
        if history:
            values = np.asarray(history, dtype=float)
            median = float(np.median(values))
            mad = float(np.median(np.abs(values-median)))
            scale = max(1.4826*mad, 1e-6)
            row["dense_pose_risk_percentile"] = (
                sum(previous <= value for previous in history)+0.5
            )/(len(history)+1.0)
            row["dense_pose_risk_robust_z"] = float(np.clip(
                (value-median)/scale, -20.0, 20.0))
            row["dense_pose_risk_drawup"] = max(0.0, value-min(history))
        else:
            row["dense_pose_risk_percentile"] = 0.5
            row["dense_pose_risk_robust_z"] = 0.0
            row["dense_pose_risk_drawup"] = 0.0
        history.append(value)
    return rows


def load_rows(confirmation_csv, pose_jsonl):
    pose_rows = []
    with pose_jsonl.open() as stream:
        for line in stream:
            if line.strip():
                pose_rows.append(json.loads(line))
    pose = {frame_key(row): row for row in add_dense_pose_context(pose_rows)}

    rows = []
    with confirmation_csv.open(newline="") as stream:
        for source in csv.DictReader(stream):
            key = frame_key(source)
            q_pose = np.clip(float(source["q_pose"]), 1e-8, 1-1e-8)
            if key not in pose:
                raise ValueError(f"missing pose diagnostics for {key}")
            diagnostic = pose[key]
            q_base = np.clip(float(source["mask_pose_reliability"]), 1e-8, 1-1e-8)
            noise = max(finite(diagnostic.get("contour_noise_variance")), 1e-12)
            search = max(finite(diagnostic.get("contour_search_lines")), 1.0)
            matched = finite(diagnostic.get("matched_contour_lines"))
            row = {
                "object": source["object"],
                "sequence": str(source["sequence"]).zfill(2),
                "seed": int(source["seed"]),
                "pose_risk_logit": float(np.log((1-q_pose)/q_pose)),
                "frame_index": int(source["frame_index"]),
                "unsafe_observation": int(source["unsafe_observation"]),
                "baseline_decision": source["mask_pose_decision"],
                "base_risk_logit": float(np.log((1-q_base)/q_base)),
                "visibility_drawdown": 1.0-float(source["q_visibility_causal_relative"]),
                "information_drawdown": 1.0-float(source["q_information_causal_relative"]),
                "log_contour_noise_variance": float(np.log(noise)),
                "contour_tail_over_noise": (
                    finite(diagnostic.get("contour_residual_p90")) / math.sqrt(noise)),
                "contour_residual_centroid_offset": finite(
                    diagnostic.get("contour_residual_centroid_offset")),
                "contour_match_deficit": float(np.clip(1.0-matched/search, 0.0, 1.0)),
                "covariance_trace_relative_initial": finite(diagnostic.get(
                    "log_normalized_covariance_trace_relative_to_initial_median")),
                "hessian_condition_relative_first": finite(diagnostic.get(
                    "log_normalized_hessian_condition_relative_to_first")),
                "dense_pose_risk_percentile": finite(
                    diagnostic.get("dense_pose_risk_percentile"), 0.5),
                "dense_pose_risk_robust_z": finite(
                    diagnostic.get("dense_pose_risk_robust_z")),
                "dense_pose_risk_drawup": finite(
                    diagnostic.get("dense_pose_risk_drawup")),
            }
            rows.append(row)
    return add_causal_risk_features(rows)


def decision_metrics(rows, decision_field="decision"):
    unsafe = [row for row in rows if row["unsafe_observation"]]
    safe = [row for row in rows if not row["unsafe_observation"]]

    def rate(selected, predicate):
        return (sum(predicate(row[decision_field]) for row in selected)/len(selected)
                if selected else None)

    return {
        "frames": len(rows),
        "unsafe": len(unsafe),
        "safe": len(safe),
        "decision_counts": dict(Counter(row[decision_field] for row in rows)),
        "protected_unsafe_recall": rate(unsafe, lambda value: value != "accept"),
        "safe_reject_rate": rate(safe, lambda value: value == "reject"),
        "safe_nonaccept_rate": rate(safe, lambda value: value != "accept"),
    }


def apply_demotion(rows, probabilities, threshold):
    output = []
    for row, probability in zip(rows, probabilities):
        decision = row["baseline_decision"]
        if decision == "accept" and probability >= threshold:
            decision = "downweight"
        output.append(dict(row, local_failure_probability=float(probability),
                           demotion_threshold=float(threshold), decision=decision))
    return output


def select_demotion_threshold(rows, probabilities, max_safe_nonaccept):
    """Maximize protected unsafe frames under a hard safe-nonaccept budget."""
    values = sorted(set(float(value) for value in probabilities))
    candidates = [float("inf"), *values]
    feasible = []
    for threshold in candidates:
        predictions = apply_demotion(rows, probabilities, threshold)
        metrics = decision_metrics(predictions)
        if (metrics["safe_nonaccept_rate"] is not None and
                metrics["safe_nonaccept_rate"] <= max_safe_nonaccept):
            feasible.append((metrics["protected_unsafe_recall"],
                             -metrics["safe_nonaccept_rate"], threshold, metrics))
    if not feasible:
        raise ValueError("no demotion threshold satisfies safe-nonaccept constraint")
    selected = max(feasible)
    return selected[2], selected[3]


def choose_l2_nested(groups, features, candidates):
    scores = {}
    for value in candidates:
        losses = []
        for held_out, validation in groups.items():
            training = [row for key, group in groups.items() if key != held_out
                        for row in group]
            if len({row["unsafe_observation"] for row in training}) < 2:
                continue
            model = fit(training, features, "unsafe_observation", value,
                        group_balanced=True)
            losses.append(log_loss(
                [row["unsafe_observation"] for row in validation],
                predict(validation, features, model)))
        scores[str(value)] = float(np.mean(losses)) if losses else None
    valid = [value for value in candidates if scores[str(value)] is not None]
    if not valid:
        raise ValueError("no valid inner LOSO regularization candidate")
    return min(valid, key=lambda value: scores[str(value)]), scores


def inner_oof_probabilities(groups, features, l2):
    rows, probabilities = [], []
    for held_out, validation in groups.items():
        training = [row for key, group in groups.items() if key != held_out
                    for row in group]
        model = fit(training, features, "unsafe_observation", l2,
                    group_balanced=True)
        rows.extend(validation)
        probabilities.extend(predict(validation, features, model))
    return rows, probabilities


def nested_loso(rows, features, l2_candidates, max_safe_nonaccept):
    groups = defaultdict(list)
    for row in rows:
        groups[sequence_key(row)].append(row)
    predictions, folds = [], {}
    for held_out, test in sorted(groups.items()):
        training_groups = {key: value for key, value in groups.items()
                           if key != held_out}
        l2, inner_scores = choose_l2_nested(
            training_groups, features, l2_candidates)
        inner_rows, inner_probability = inner_oof_probabilities(
            training_groups, features, l2)
        threshold, training_metrics = select_demotion_threshold(
            inner_rows, inner_probability, max_safe_nonaccept)
        training = [row for group in training_groups.values() for row in group]
        model = fit(training, features, "unsafe_observation", l2,
                    group_balanced=True)
        fold_predictions = apply_demotion(
            test, predict(test, features, model), threshold)
        predictions.extend(fold_predictions)
        folds[held_out] = {
            "selected_l2": l2,
            "inner_log_loss": inner_scores,
            "selected_demotion_threshold": threshold,
            "inner_oof_metrics": training_metrics,
            "test_metrics": decision_metrics(fold_predictions),
        }
    return predictions, folds


def direct_score_loso(rows, score_field, max_safe_nonaccept):
    """Evaluate a directed causal risk score without fitting object weights."""
    groups = defaultdict(list)
    for row in rows:
        groups[sequence_key(row)].append(row)
    predictions, folds = [], {}
    for held_out, test in sorted(groups.items()):
        training = [row for key, group in groups.items() if key != held_out
                    for row in group]
        training_scores = [row[score_field] for row in training]
        threshold, training_metrics = select_demotion_threshold(
            training, training_scores, max_safe_nonaccept)
        fold_predictions = apply_demotion(
            test, [row[score_field] for row in test], threshold)
        predictions.extend(fold_predictions)
        folds[held_out] = {
            "score_field": score_field,
            "selected_demotion_threshold": threshold,
            "training_metrics": training_metrics,
            "test_metrics": decision_metrics(fold_predictions),
        }
    return predictions, folds



def summarize(rows):
    metrics = decision_metrics(rows)
    metrics["risk_auroc"] = binary_auroc(
        [row["unsafe_observation"] for row in rows],
        [row["local_failure_probability"] for row in rows])
    metrics["sequences"] = {
        key: decision_metrics([row for row in rows if sequence_key(row) == key])
        for key in sorted({sequence_key(row) for row in rows})
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("confirmation_frames", type=Path)
    parser.add_argument("pose_dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--l2-grid", default="0.01,0.1,1,10,100")
    parser.add_argument("--max-safe-nonaccept", type=float, default=0.50)
    args = parser.parse_args()
    rows = load_rows(args.confirmation_frames, args.pose_dataset)
    l2_candidates = [float(value) for value in args.l2_grid.split(",")]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline_rows = [dict(row, decision=row["baseline_decision"]) for row in rows]
    variants = {}
    for name, features in FEATURE_SETS.items():
        predictions, folds = nested_loso(
            rows, features, l2_candidates, args.max_safe_nonaccept)
        variants[name] = {"features": features, "metrics": summarize(predictions),
                          "folds": folds}
        with (args.output_dir/f"{name}_nested_loso_frames.csv").open(
                "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
            writer.writeheader()
            writer.writerows(predictions)

    for name, score_field in DIRECT_SCORE_SETS.items():
        predictions, folds = direct_score_loso(
            rows, score_field, args.max_safe_nonaccept)
        variants[name] = {
            "score_field": score_field,
            "metrics": summarize(predictions),
            "folds": folds,
        }
        with (args.output_dir/f"{name}_loso_frames.csv").open(
                "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
            writer.writeheader()
            writer.writerows(predictions)

    summary = {
        "schema_version": 1,
        "stage": 4,
        "mode": "P1_v2_development_nested_loso_hierarchical_demotion",
        "independent_confirmation": False,
        "warning": "confirmation-v1 labels are now development-only",
        "frames": len(rows),
        "sequences": len({sequence_key(row) for row in rows}),
        "constraints": {
            "parent_decision": "frozen q_mask*q_pose",
            "allowed_transition": "accept -> downweight only",
            "max_inner_safe_nonaccept_rate": args.max_safe_nonaccept,
        },
        "baseline": decision_metrics(baseline_rows),
        "variants": variants,
    }
    (args.output_dir/"p1v2_hierarchical_gate_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps({name: value["metrics"] for name, value in variants.items()},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
