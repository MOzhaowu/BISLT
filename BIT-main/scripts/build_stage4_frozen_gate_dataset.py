#!/usr/bin/env python3
"""Build aligned Stage-4 frame data with frozen q_mask/q_pose models."""

import argparse
import csv
import json
from pathlib import Path

from evaluate_stage4_confirmation import (
    align_rows,
    attach_decisions,
    load_jsonl,
    sha256,
)


def load_mask_rows(run_root, run_glob):
    pattern = (
        f"*/*/{run_glob}/mask_uncertainty/"
        "mask_uncertainty_frame_calibration.jsonl")
    rows = []
    for path in sorted(run_root.glob(pattern)):
        rows.extend(load_jsonl(path))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-glob", required=True)
    parser.add_argument("--pose-dataset", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text())
    frozen = {}
    for name, config in protocol["models"].items():
        path = args.workspace_root / config["path"]
        actual = sha256(path)
        if actual != config["sha256"]:
            raise ValueError(f"{name} hash mismatch: {actual}")
        frozen[name] = json.loads(path.read_text())

    mask_rows = load_mask_rows(args.run_root, args.run_glob)
    pose_rows = load_jsonl(args.pose_dataset)
    rows = attach_decisions(
        align_rows(mask_rows, pose_rows, frozen["q_mask"], frozen["q_pose"]),
        protocol["methods"])
    if not rows:
        raise ValueError("no aligned post-warmup frames")

    identities = {
        (row["object"], row["sequence"], row["seed"], row["frame_index"])
        for row in rows}
    if len(identities) != len(rows):
        raise ValueError("duplicate aligned frame identities")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": 1,
        "mode": "frozen_stage4_gate_dataset",
        "run_glob": args.run_glob,
        "frames": len(rows),
        "unsafe": sum(row["unsafe_observation"] for row in rows),
        "safe": sum(not row["unsafe_observation"] for row in rows),
        "runs": len({(row["object"], row["sequence"], row["seed"])
                     for row in rows}),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
