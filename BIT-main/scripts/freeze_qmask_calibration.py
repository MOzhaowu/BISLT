#!/usr/bin/env python3
"""Freeze the Stage-2 candidate-disagreement mask calibration model."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from fit_stage2_mask_nested_loso import choose_l2, fit


FEATURES = ["candidate_disagreement"]
LABEL = "mask_failure_iou90"


def serializable(model):
    return {name: value.tolist() if isinstance(value, np.ndarray) else value
            for name, value in model.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--l2-grid", default="0.01,0.1,1,10,100")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.dataset.read_text().splitlines()
            if line.strip()]
    groups = {}
    for row in rows:
        groups.setdefault(f"{row['object']}/{row['sequence']}", []).append(row)
    candidates = [float(value) for value in args.l2_grid.split(",")]
    selected_l2, inner_scores = choose_l2(
        groups, FEATURES, LABEL, candidates)
    model = fit(rows, FEATURES, LABEL, selected_l2)
    frozen = {
        "schema_version": 1,
        "stage": 2,
        "name": "q_mask",
        "status": "frozen_for_stage4_confirmation",
        "probability_semantics": "P(mask IoU < 0.90)",
        "features": FEATURES,
        "label": LABEL,
        "selected_l2": selected_l2,
        "inner_sequence_loso_log_loss": inner_scores,
        "model": serializable(model),
        "dataset_frames": len(rows),
        "dataset_sequences": len(groups),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(frozen, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(frozen, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
