#!/usr/bin/env python3
"""Build the Stage-4 shadow gate without changing BIT tracking decisions.

The inputs are out-of-fold probabilities frozen in Stage 2 and Stage 3.  In
this script q_mask and q_pose are *reliabilities* (one minus the corresponding
failure probability).  The primary v1 gate intentionally excludes visibility
and information proxies until their value has been demonstrated out of fold.
"""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


KEY_FIELDS = ("object", "sequence", "seed", "frame_index")
DECISIONS = ("accept", "downweight", "reject")


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def finite_float(value, default=None):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def row_key(row):
    return (
        row["object"], str(row["sequence"]).zfill(2),
        int(row["seed"]), int(row["frame_index"]),
    )


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def geometric_mean(values):
    values = [clamp(value) for value in values if value is not None]
    if not values:
        return 1.0
    if any(value == 0.0 for value in values):
        return 0.0
    return math.exp(sum(math.log(value) for value in values) / len(values))


def visibility_reliability(row):
    """Object-independent visibility proxy, logged but not gated in v1."""
    return geometric_mean([
        finite_float(row.get("visible_boundary_ratio")),
        finite_float(row.get("effective_contour_ratio")),
        None if finite_float(row.get("occlusion_ratio")) is None else
        1.0 - finite_float(row.get("occlusion_ratio")),
    ])


def information_reliability(row, novelty_angle_deg=20.0):
    """View-novelty proxy, logged but not gated in v1."""
    angle = finite_float(row.get("min_view_angle_deg"))
    if angle is None:
        return 1.0
    return clamp(angle / novelty_angle_deg)


def gate_decision(reliability, accept_threshold=0.8, reject_threshold=0.3):
    if not 0.0 <= reject_threshold < accept_threshold <= 1.0:
        raise ValueError("thresholds must satisfy 0 <= reject < accept <= 1")
    if reliability >= accept_threshold:
        return "accept", 1.0
    if reliability < reject_threshold:
        return "reject", 0.0
    return "downweight", reliability / accept_threshold


def binary_auroc(labels, scores):
    positives = [score for label, score in zip(labels, scores) if label]
    negatives = [score for label, score in zip(labels, scores) if not label]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += positive > negative
            wins += 0.5 * (positive == negative)
    return wins / (len(positives) * len(negatives))


def future_pose_labels(key, pose_rows, horizon):
    prefix = key[:3]
    frame = key[3]
    future = [pose_rows[(prefix[0], prefix[1], prefix[2], index)]
              for index in range(frame + 1, frame + horizon + 1)
              if (prefix[0], prefix[1], prefix[2], index) in pose_rows]
    labels = [int(row["failure_5deg_50mm"]) for row in future]
    return {
        "future_pose_frames": len(labels),
        "future_pose_failure_any": int(any(labels)) if labels else "",
        "future_pose_failure_rate": sum(labels) / len(labels) if labels else "",
    }


def diagnostic_run_key(path, root):
    relative = path.relative_to(root)
    parts = relative.parts
    if len(parts) < 4:
        return None
    object_name, sequence, run_name = parts[0], parts[1], parts[2]
    marker = "_s"
    if marker not in run_name:
        return None
    try:
        seed = int(run_name.rsplit(marker, 1)[1])
    except ValueError:
        return None
    return object_name, sequence.zfill(2), seed


def load_diagnostics(root):
    runs = {}
    if root is None:
        return runs
    for path in sorted(root.glob("*/*/*/diagnostics.csv")):
        key = diagnostic_run_key(path, root)
        if key is not None:
            runs[key] = {int(row["frame_index"]): row for row in read_csv(path)}
    return runs


def update_outcome(key, runs, horizon, outcome_window):
    frames = runs.get(key[:3], {})
    start = key[3]
    candidates = []
    for index in range(start, start + horizon + 1):
        row = frames.get(index)
        if row and int(float(row.get("model_reloaded_after_frame", 0))) == 1:
            candidates.append(index)
    if not candidates:
        return {
            "linked_update": 0, "linked_update_frame": "",
            "pre_update_failure_rate": "", "post_update_failure_rate": "",
            "update_degraded": "", "clearly_bad_update": "",
        }
    update_frame = candidates[0]

    def failure_rates(indices):
        values = []
        for index in indices:
            row = frames.get(index)
            if row is None:
                continue
            success = finite_float(row.get("success_5deg_50mm"))
            if success is not None:
                values.append(1.0 - success)
        return sum(values) / len(values) if values else None

    pre = failure_rates(range(max(0, update_frame - outcome_window + 1),
                              update_frame + 1))
    post = failure_rates(range(update_frame + 1,
                               update_frame + outcome_window + 1))
    degraded = "" if pre is None or post is None else int(post > pre)
    clearly_bad = "" if pre is None or post is None else int(pre == 0 and post > 0)
    return {
        "linked_update": 1,
        "linked_update_frame": update_frame,
        "pre_update_failure_rate": "" if pre is None else pre,
        "post_update_failure_rate": "" if post is None else post,
        "update_degraded": degraded,
        "clearly_bad_update": clearly_bad,
    }


def build_rows(mask_predictions, pose_predictions, pose_dataset,
               mask_dataset, diagnostic_runs, accept_threshold,
               reject_threshold, future_horizon, update_horizon,
               outcome_window):
    mask = {row_key(row): row for row in mask_predictions}
    pose = {row_key(row): row for row in pose_predictions}
    pose_features = {row_key(row): row for row in pose_dataset}
    mask_features = {row_key(row): row for row in mask_dataset}
    aligned = sorted(mask.keys() & pose.keys())
    rows = []
    for key in aligned:
        p_mask = clamp(float(mask[key]["failure_probability"]))
        p_pose = clamp(float(pose[key]["failure_probability"]))
        q_mask = 1.0 - p_mask
        q_pose = 1.0 - p_pose
        base_reliability = q_mask * q_pose
        features = dict(pose_features.get(key, {}))
        features.update(mask_features.get(key, {}))
        q_visibility = visibility_reliability(features)
        q_information = information_reliability(features)
        audit_reliability = base_reliability * q_visibility * q_information
        decision, weight = gate_decision(
            base_reliability, accept_threshold, reject_threshold)
        mask_failure = int(mask[key]["failure_label"])
        pose_failure = int(pose[key]["failure_label"])
        row = {
            "object": key[0], "sequence": key[1], "seed": key[2],
            "frame_index": key[3],
            "p_mask_failure": p_mask, "p_pose_failure": p_pose,
            "q_mask": q_mask, "q_pose": q_pose,
            "q_visibility_audit": q_visibility,
            "q_information_audit": q_information,
            "base_reliability": base_reliability,
            "full_proxy_reliability_audit": audit_reliability,
            "shadow_decision": decision, "shadow_weight": weight,
            "mask_failure_iou90": mask_failure,
            "pose_failure_5deg_50mm": pose_failure,
            "unsafe_observation": int(mask_failure or pose_failure),
        }
        row.update(future_pose_labels(key, pose_features, future_horizon))
        row.update(update_outcome(
            key, diagnostic_runs, update_horizon, outcome_window))
        rows.append(row)
    return rows


def fraction(rows, field):
    values = [int(row[field]) for row in rows if row.get(field, "") != ""]
    return sum(values) / len(values) if values else None


def summarize(rows, accept_threshold, reject_threshold, future_horizon,
              update_horizon, outcome_window):
    by_decision = {}
    for decision in DECISIONS:
        selected = [row for row in rows if row["shadow_decision"] == decision]
        by_decision[decision] = {
            "frames": len(selected),
            "fraction": len(selected) / len(rows) if rows else 0.0,
            "unsafe_observation_rate": fraction(selected, "unsafe_observation"),
            "current_pose_failure_rate": fraction(
                selected, "pose_failure_5deg_50mm"),
            "future_pose_failure_any_rate": fraction(
                selected, "future_pose_failure_any"),
            "linked_updates": sum(int(row["linked_update"]) for row in selected),
            "clearly_bad_updates": sum(
                int(row["clearly_bad_update"]) for row in selected
                if row["clearly_bad_update"] != ""),
        }
    unsafe = [row for row in rows if row["unsafe_observation"]]
    safe = [row for row in rows if not row["unsafe_observation"]]
    labels = [row["unsafe_observation"] for row in rows]
    risk = [1.0 - row["base_reliability"] for row in rows]
    return {
        "schema_version": 1,
        "stage": 4,
        "mode": "shadow",
        "primary_policy": "q_mask_times_q_pose",
        "probability_semantics": {
            "p_mask_failure": "OOF probability of mask IoU < 0.90",
            "p_pose_failure": "OOF probability of failing 5deg/5cm",
            "q_mask": "1 - p_mask_failure",
            "q_pose": "1 - p_pose_failure",
            "base_reliability": "q_mask * q_pose",
            "visibility_and_information": "audit only; excluded from v1 decisions",
        },
        "thresholds": {"accept": accept_threshold, "reject": reject_threshold},
        "alignment": {
            "frames": len(rows),
            "sequences": len({(row["object"], row["sequence"]) for row in rows}),
            "seeds": sorted({row["seed"] for row in rows}),
            "future_horizon_frames": future_horizon,
            "update_link_horizon_frames": update_horizon,
            "update_outcome_window_frames": outcome_window,
        },
        "metrics": {
            "unsafe_observations": len(unsafe),
            "safe_observations": len(safe),
            "unsafe_risk_auroc": binary_auroc(labels, risk),
            "reject_unsafe_recall": (
                sum(row["shadow_decision"] == "reject" for row in unsafe) /
                len(unsafe) if unsafe else None),
            "protected_unsafe_recall": (
                sum(row["shadow_decision"] != "accept" for row in unsafe) /
                len(unsafe) if unsafe else None),
            "safe_reject_rate": (
                sum(row["shadow_decision"] == "reject" for row in safe) /
                len(safe) if safe else None),
            "safe_nonaccept_rate": (
                sum(row["shadow_decision"] != "accept" for row in safe) /
                len(safe) if safe else None),
        },
        "by_decision": by_decision,
    }


def write_csv(path, rows):
    if not rows:
        raise ValueError("No aligned q_mask/q_pose rows were found")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-predictions", type=Path, required=True)
    parser.add_argument("--pose-predictions", type=Path, required=True)
    parser.add_argument("--mask-dataset", type=Path, required=True)
    parser.add_argument("--pose-dataset", type=Path, required=True)
    parser.add_argument("--diagnostics-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--accept-threshold", type=float, default=0.8)
    parser.add_argument("--reject-threshold", type=float, default=0.3)
    parser.add_argument("--future-horizon", type=int, default=5)
    parser.add_argument("--update-horizon", type=int, default=10)
    parser.add_argument("--outcome-window", type=int, default=3)
    args = parser.parse_args()
    if min(args.future_horizon, args.update_horizon, args.outcome_window) < 1:
        parser.error("all horizons must be positive")

    rows = build_rows(
        read_csv(args.mask_predictions), read_csv(args.pose_predictions),
        read_jsonl(args.pose_dataset), read_csv(args.mask_dataset),
        load_diagnostics(args.diagnostics_root), args.accept_threshold,
        args.reject_threshold, args.future_horizon, args.update_horizon,
        args.outcome_window,
    )
    summary = summarize(
        rows, args.accept_threshold, args.reject_threshold,
        args.future_horizon, args.update_horizon, args.outcome_window)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "shadow_gate_frames.csv", rows)
    (args.output_dir / "shadow_gate_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
