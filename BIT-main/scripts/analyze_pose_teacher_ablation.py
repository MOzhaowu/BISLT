#!/usr/bin/env python3
"""Summarize Stage-3 pose-teacher radius and joint-perturbation ablations."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


RATE_FIELDS = [
    "pose_teacher_axis_failure_rate_small",
    "pose_teacher_axis_failure_rate_medium",
    "pose_teacher_axis_failure_rate_large",
    "pose_teacher_joint_failure_rate_small",
    "pose_teacher_joint_failure_rate_medium",
    "pose_teacher_joint_failure_rate_large",
]


def finite_pairs(rows, x_field, y_field):
    pairs = []
    for row in rows:
        x, y = row.get(x_field), row.get(y_field)
        if x is None or y is None:
            continue
        x, y = float(x), float(y)
        if np.isfinite(x) and np.isfinite(y):
            pairs.append((x, y))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--proxy",
        default="log_normalized_covariance_trace_relative_to_initial_median",
    )
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.dataset.read_text().splitlines()
        if line.strip()
    ]
    sampled = [
        row
        for row in rows
        if int(row.get("pose_teacher_valid", 0))
        and int(row.get("pose_teacher_ablation_probes", 0)) > 0
    ]
    rates = {}
    for field in RATE_FIELDS:
        pairs = finite_pairs(sampled, args.proxy, field)
        x = np.asarray([pair[0] for pair in pairs])
        y = np.asarray([pair[1] for pair in pairs])
        correlation = spearmanr(x, y).statistic if len(pairs) >= 2 else np.nan
        rates[field] = {
            "frames": len(pairs),
            "mean_failure_rate": float(y.mean()) if len(y) else None,
            "spearman_to_proxy": (
                float(correlation) if np.isfinite(correlation) else None
            ),
        }

    summary = {
        "schema_version": 1,
        "stage": 3,
        "runs": len({(row["object"], row["sequence"], row["seed"]) for row in rows}),
        "teacher_frames": len(sampled),
        "ablation_probes": int(
            sum(int(row.get("pose_teacher_ablation_probes", 0)) for row in sampled)
        ),
        "proxy": args.proxy,
        "rates": rates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
