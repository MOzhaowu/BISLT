#!/usr/bin/env python3
"""Evaluate the frozen Stage-4 confirmation protocol exactly once."""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from build_stage4_shadow_gate import binary_auroc, gate_decision, geometric_mean


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(row):
    return (row["object"], str(row["sequence"]).zfill(2),
            int(row["seed"]), int(row["frame_index"]))


def sigmoid(value):
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))


def predict_failure(row, frozen):
    features = frozen.get("features", frozen.get("selected_features"))
    model = frozen["model"]
    values = []
    for index, feature in enumerate(features):
        value = row.get(feature)
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = float(model["medians"][index])
        if not math.isfinite(value):
            value = float(model["medians"][index])
        values.append(value)
    linear = float(model["parameters"][0])
    for index, value in enumerate(values):
        scale = max(float(model["scales"][index]), 1e-12)
        standardized = (value - float(model["means"][index])) / scale
        linear += standardized * float(model["parameters"][index + 1])
    return sigmoid(linear)


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def load_mask_rows(run_root):
    rows = []
    pattern = "*/*/confirm_s*/mask_uncertainty/mask_uncertainty_frame_calibration.jsonl"
    for path in sorted(run_root.glob(pattern)):
        rows.extend(load_jsonl(path))
    return rows


def visibility_reliability(row):
    return geometric_mean([
        float(row["visible_boundary_ratio"]),
        float(row["effective_contour_ratio"]),
        1.0 - float(row["occlusion_ratio"]),
    ])


def information_reliability(row):
    try:
        value = float(row["min_view_angle_deg"])
    except (TypeError, ValueError):
        return 1.0
    return max(0.0, min(1.0, value / 20.0)) if math.isfinite(value) else 1.0


def align_rows(mask_rows, pose_rows, q_mask_model, q_pose_model):
    masks = {key(row): row for row in mask_rows}
    poses = {key(row): row for row in pose_rows
             if not int(row.get("calibration_warmup", 0))}
    rows = []
    for identity in sorted(masks.keys() & poses.keys()):
        mask, pose = masks[identity], poses[identity]
        p_mask = predict_failure(mask, q_mask_model)
        p_pose = predict_failure(pose, q_pose_model)
        row = {
            "object": identity[0], "sequence": identity[1],
            "seed": identity[2], "frame_index": identity[3],
            "p_mask_failure": p_mask, "p_pose_failure": p_pose,
            "q_mask": 1.0 - p_mask, "q_pose": 1.0 - p_pose,
            "q_visibility": visibility_reliability(pose),
            "q_information": information_reliability(mask),
            "mask_failure_iou90": int(mask["mask_failure_iou90"]),
            "pose_failure_5deg_50mm": int(pose["failure_5deg_50mm"]),
        }
        row["unsafe_observation"] = int(
            row["mask_failure_iou90"] or row["pose_failure_5deg_50mm"])
        rows.append(row)
    maxima = {}
    for row in rows:
        run = (row["object"], row["sequence"], row["seed"])
        previous = maxima.get(run, (row["q_visibility"], row["q_information"]))
        vis_max = max(previous[0], row["q_visibility"])
        info_max = max(previous[1], row["q_information"])
        maxima[run] = (vis_max, info_max)
        row["q_visibility_causal_relative"] = (
            row["q_visibility"] / vis_max if vis_max else 1.0)
        row["q_information_causal_relative"] = (
            row["q_information"] / info_max if info_max else 1.0)
    return rows


def method_reliability(row, method):
    base = row["q_mask"] * row["q_pose"]
    if method == "mask_pose":
        return base
    if method == "mask_pose_causal_visibility":
        return base * row["q_visibility_causal_relative"]
    if method == "mask_pose_causal_information":
        return base * row["q_information_causal_relative"]
    raise KeyError(method)


def attach_decisions(rows, methods):
    output = []
    for row in rows:
        item = dict(row)
        for name, config in methods.items():
            reliability = method_reliability(row, name)
            decision, weight = gate_decision(
                reliability, float(config["accept_threshold"]),
                float(config["reject_threshold"]))
            item[f"{name}_reliability"] = reliability
            item[f"{name}_decision"] = decision
            item[f"{name}_weight"] = weight
        output.append(item)
    return output


def metrics(rows, method):
    unsafe = [row for row in rows if row["unsafe_observation"]]
    safe = [row for row in rows if not row["unsafe_observation"]]
    field = f"{method}_decision"

    def rate(selected, predicate):
        return (sum(predicate(row[field]) for row in selected) / len(selected)
                if selected else None)

    return {
        "frames": len(rows), "unsafe": len(unsafe), "safe": len(safe),
        "risk_auroc": binary_auroc(
            [row["unsafe_observation"] for row in rows],
            [1.0 - row[f"{method}_reliability"] for row in rows]),
        "reject_unsafe_recall": rate(unsafe, lambda value: value == "reject"),
        "protected_unsafe_recall": rate(unsafe, lambda value: value != "accept"),
        "safe_reject_rate": rate(safe, lambda value: value == "reject"),
        "safe_nonaccept_rate": rate(safe, lambda value: value != "accept"),
    }


def bootstrap_delta(rows, candidate, baseline, replicates, seed):
    groups = defaultdict(list)
    for row in rows:
        groups[f"{row['object']}/{row['sequence']}"].append(row)
    names = sorted(groups)
    rng = np.random.default_rng(seed)
    fields = ("risk_auroc", "protected_unsafe_recall", "safe_reject_rate",
              "safe_nonaccept_rate")
    values = {field: [] for field in fields}
    for _ in range(replicates):
        selected = rng.choice(names, size=len(names), replace=True)
        sample = [row for name in selected for row in groups[name]]
        candidate_metrics = metrics(sample, candidate)
        baseline_metrics = metrics(sample, baseline)
        for field in fields:
            left, right = candidate_metrics[field], baseline_metrics[field]
            if left is not None and right is not None:
                values[field].append(left - right)
    return {field: {
        "lower": float(np.percentile(items, 2.5)),
        "median": float(np.percentile(items, 50.0)),
        "upper": float(np.percentile(items, 97.5)),
    } for field, items in values.items() if items}


def subset(rows, sequences):
    selected = set(sequences)
    return [row for row in rows if f"{row['object']}/{row['sequence']}" in selected]


def candidate_acceptance(rows, candidate, baseline, protocol):
    limits = protocol["acceptance"]
    overall = metrics(rows, candidate)
    base = metrics(rows, baseline)
    target_sequences = protocol["primary_confirmation"]["target_sequences"]
    target = {name: {
        "candidate": metrics(subset(rows, [name]), candidate),
        "baseline": metrics(subset(rows, [name]), baseline),
    } for name in target_sequences}
    target_not_below = all(
        value["candidate"]["protected_unsafe_recall"] is not None and
        value["baseline"]["protected_unsafe_recall"] is not None and
        value["candidate"]["protected_unsafe_recall"] >=
        value["baseline"]["protected_unsafe_recall"]
        for value in target.values())
    target_strict = any(
        value["candidate"]["protected_unsafe_recall"] is not None and
        value["baseline"]["protected_unsafe_recall"] is not None and
        value["candidate"]["protected_unsafe_recall"] >
        value["baseline"]["protected_unsafe_recall"]
        for value in target.values())
    seed_direction = {}
    target_rows = subset(rows, target_sequences)
    for seed in protocol["primary_confirmation"]["target_seeds"]:
        selected = [row for row in target_rows if row["seed"] == seed]
        seed_direction[str(seed)] = {
            "candidate": metrics(selected, candidate),
            "baseline": metrics(selected, baseline),
        }
    seeds_consistent = all(
        result["candidate"]["protected_unsafe_recall"] is not None and
        result["baseline"]["protected_unsafe_recall"] is not None and
        result["candidate"]["protected_unsafe_recall"] >=
        result["baseline"]["protected_unsafe_recall"]
        for result in seed_direction.values())
    checks = {
        "safe_reject_rate": overall["safe_reject_rate"] is not None and
            overall["safe_reject_rate"] <= limits["safe_reject_rate_max"],
        "safe_nonaccept_rate": overall["safe_nonaccept_rate"] is not None and
            overall["safe_nonaccept_rate"] <= limits["safe_nonaccept_rate_max"],
        "protected_not_below_baseline":
            overall["protected_unsafe_recall"] is not None and
            base["protected_unsafe_recall"] is not None and
            overall["protected_unsafe_recall"] >= base["protected_unsafe_recall"],
        "target_sequences_not_below": target_not_below,
        "target_sequence_strict_improvement": target_strict,
        "target_seed_direction_consistent": seeds_consistent,
    }
    return {"passed": all(checks.values()), "checks": checks,
            "target_sequences": target, "target_seeds": seed_direction}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--pose-dataset", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260822)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    formula_path = (args.workspace_root / "BIT-main" / "scripts" /
                    "evaluate_stage4_gate_ablation.py")
    formula_hash = sha256(formula_path)
    if formula_hash != protocol["formula_implementation_sha256"]:
        raise ValueError(f"formula implementation hash mismatch: {formula_hash}")
    model_files = {}
    for name, config in protocol["models"].items():
        path = args.workspace_root / config["path"]
        actual = sha256(path)
        if actual != config["sha256"]:
            raise ValueError(f"{name} hash mismatch: {actual}")
        model_files[name] = json.loads(path.read_text())
    rows = align_rows(load_mask_rows(args.run_root),
                      load_jsonl(args.pose_dataset),
                      model_files["q_mask"], model_files["q_pose"])
    rows = attach_decisions(rows, protocol["methods"])
    external = protocol["primary_confirmation"]["external_sequences"]
    targets = protocol["primary_confirmation"]["target_sequences"]
    sets = {"all": rows, "external": subset(rows, external),
            "targets": subset(rows, targets)}
    method_metrics = {name: {set_name: metrics(selected, name)
                            for set_name, selected in sets.items()}
                      for name in protocol["methods"]}
    baseline = "mask_pose"
    candidates = {}
    for name in protocol["methods"]:
        if name == baseline:
            continue
        candidates[name] = {
            "acceptance": candidate_acceptance(rows, name, baseline, protocol),
            "bootstrap_delta_vs_baseline": bootstrap_delta(
                rows, name, baseline, args.bootstrap_replicates,
                args.bootstrap_seed),
        }
    unsafe = sum(row["unsafe_observation"] for row in rows)
    trigger = protocol["fallback_confirmation"]["trigger_if"]
    expansion_reasons = []
    if len(rows) < trigger["aligned_post_warmup_frames_lt"]:
        expansion_reasons.append("aligned_post_warmup_frames")
    if len(rows) - unsafe < trigger["safe_frames_lt"]:
        expansion_reasons.append("safe_frames")
    if unsafe < trigger["unsafe_frames_lt"]:
        expansion_reasons.append("unsafe_frames")
    summary = {
        "schema_version": 1, "stage": 4, "mode": "independent_confirmation",
        "frames": len(rows), "unsafe": unsafe, "safe": len(rows) - unsafe,
        "fallback_required": bool(expansion_reasons),
        "fallback_reasons": expansion_reasons,
        "methods": method_metrics, "candidates": candidates,
        "enter_p2": any(value["acceptance"]["passed"]
                        for value in candidates.values()),
        "bootstrap": {"unit": "object/sequence",
                      "replicates": args.bootstrap_replicates,
                      "seed": args.bootstrap_seed},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "confirmation_frames.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader(); writer.writerows(rows)
    (args.output_dir / "confirmation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
