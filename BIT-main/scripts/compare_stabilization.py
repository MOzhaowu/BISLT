#!/usr/bin/env python3
"""Compare an unstabilized BIT run with a stabilized run."""

import argparse
import json
from pathlib import Path


def load_metrics(run_dir):
    return json.loads((run_dir / "metrics.json").read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("stabilized", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    baseline = load_metrics(args.baseline)
    stabilized = load_metrics(args.stabilized)
    report = {
        "baseline": str(args.baseline.resolve()),
        "stabilized": str(args.stabilized.resolve()),
        "frames": stabilized["evaluated_frames"],
        "metrics": {
            "add_auc_100mm": {
                "baseline": baseline["add_auc_100mm"],
                "stabilized": stabilized["add_auc_100mm"],
                "delta": stabilized["add_auc_100mm"] - baseline["add_auc_100mm"],
            },
            "rotation_mean_deg": {
                "baseline": baseline["rotation_error_deg"]["mean"],
                "stabilized": stabilized["rotation_error_deg"]["mean"],
                "delta": stabilized["rotation_error_deg"]["mean"] - baseline["rotation_error_deg"]["mean"],
            },
            "translation_mean_mm": {
                "baseline": baseline["translation_error_mm"]["mean"],
                "stabilized": stabilized["translation_error_mm"]["mean"],
                "delta": stabilized["translation_error_mm"]["mean"] - baseline["translation_error_mm"]["mean"],
            },
            "success_5deg_50mm": {
                "baseline": baseline["success_5deg_50mm"],
                "stabilized": stabilized["success_5deg_50mm"],
                "delta": stabilized["success_5deg_50mm"] - baseline["success_5deg_50mm"],
            },
        },
    }
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
