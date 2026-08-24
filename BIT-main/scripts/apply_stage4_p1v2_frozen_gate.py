#!/usr/bin/env python3
"""Apply the frozen P1-v2 hierarchical demotion gate to arbitrary runs."""

import argparse
import csv
import json
from pathlib import Path

from evaluate_stage4_p1v2_hierarchical_gate import (
    apply_demotion,
    decision_metrics,
    load_rows,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gate_csv", type=Path)
    parser.add_argument("pose_jsonl", type=Path)
    parser.add_argument("frozen_config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.frozen_config.read_text())
    rows = load_rows(args.gate_csv, args.pose_jsonl)
    score_field = config["method"]["score_field"]
    threshold = float(config["method"]["demotion_threshold"])
    predictions = apply_demotion(
        rows, [row[score_field] for row in rows], threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    baseline = [dict(row, decision=row["baseline_decision"]) for row in rows]
    summary = {
        "schema_version": 1,
        "stage": 4,
        "mode": "frozen_P1_v2_gate_application",
        "score_field": score_field,
        "demotion_threshold": threshold,
        "baseline": decision_metrics(baseline),
        "candidate": decision_metrics(predictions),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
