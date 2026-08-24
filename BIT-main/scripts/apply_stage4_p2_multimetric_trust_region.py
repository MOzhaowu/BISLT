#!/usr/bin/env python3
"""Apply a frozen multi-metric feasible-set selector to dual replay rows."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

from stage4_influence import select_multimetric_validation_geometry


SOURCE_POLICY = "p1v2_short_loo_dual_trust_region"
TARGET_POLICY = "p1v2_short_loo_multimetric_trust_region"


def metrics_from_row(row, prefix):
    return {
        "mean_iou_loss": float(row[f"{prefix}mean_iou_loss"]),
        "pose_inconsistency": float(row[f"{prefix}pose_inconsistency"]),
        "temporal_std": float(row[f"{prefix}temporal_std"]),
        "uncertainty": float(row[f"{prefix}uncertainty"]),
    }


def apply_selector(row, max_pose_regression=0.0,
                   max_temporal_regression=0.0,
                   target_policy=TARGET_POLICY):
    stable = metrics_from_row(row, "stable_")
    full = metrics_from_row(row, "same_run_original_")
    loo = None
    if row.get("proposed_mean_iou_loss") is not None:
        loo = metrics_from_row(row, "proposed_")
    choice = select_multimetric_validation_geometry(
        stable, full, loo,
        max_pose_regression=max_pose_regression,
        max_temporal_regression=max_temporal_regression)
    final = choice["selected_metrics"]
    output = dict(row)
    output.update({
        "policy": target_policy,
        "parent_policy": row["policy"],
        "status": ("committed" if choice["committed"] else "rolled_back"),
        "committed": choice["committed"],
        "commit_reason": f"selected_feasible_{choice['selected_source']}",
        "selected_source": choice["selected_source"],
        "multimetric_feasibility": choice["feasibility"],
        "max_pose_regression_vs_full": float(max_pose_regression),
        "max_temporal_regression_vs_full": float(max_temporal_regression),
        "mean_iou_loss": float(final["mean_iou_loss"]),
        "pose_inconsistency": float(final["pose_inconsistency"]),
        "temporal_std": float(final["temporal_std"]),
        "uncertainty": float(final["uncertainty"]),
    })
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-pose-regression", type=float, default=0.0)
    parser.add_argument("--max-temporal-regression", type=float, default=0.0)
    parser.add_argument("--source-policy", default=SOURCE_POLICY)
    parser.add_argument("--target-policy", default=TARGET_POLICY)
    args = parser.parse_args()
    rows = [
        json.loads(line) for line in args.input_jsonl.read_text().splitlines()
        if line.strip()]
    source = [row for row in rows if row["policy"] == args.source_policy]
    selected = [
        apply_selector(
            row, args.max_pose_regression, args.max_temporal_regression,
            args.target_policy)
        for row in source]
    output = rows+selected
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in output)+"\n")
    counts = Counter(row["selected_source"] for row in selected)
    print(json.dumps({
        "events": len(selected),
        "selection_counts": dict(sorted(counts.items())),
        "output": str(args.output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
