#!/usr/bin/env python3
"""Merge Stage-4 gate/pose development inputs with identity deduplication."""

import argparse
import csv
import json
from pathlib import Path


def identity(row):
    return (row["object"], str(row["sequence"]).zfill(2),
            int(row["seed"]), int(row["frame_index"]))


def merge_csv(paths, output):
    rows, seen, fields = [], set(), None
    for path in paths:
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if fields is None:
                fields = reader.fieldnames
            elif reader.fieldnames != fields:
                raise ValueError(f"CSV schema mismatch: {path}")
            for row in reader:
                key = identity(row)
                if key in seen:
                    raise ValueError(f"duplicate gate frame: {key}")
                seen.add(key)
                rows.append(row)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def merge_jsonl(paths, output):
    rows, seen = [], set()
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = identity(row)
            if key in seen:
                raise ValueError(f"duplicate pose frame: {key}")
            seen.add(key)
            rows.append(row)
    output.write_text("".join(
        json.dumps(row, ensure_ascii=False)+"\n" for row in rows))
    return len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-csv", type=Path, action="append", required=True)
    parser.add_argument("--pose-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--output-gate", type=Path, required=True)
    parser.add_argument("--output-pose", type=Path, required=True)
    args = parser.parse_args()
    args.output_gate.parent.mkdir(parents=True, exist_ok=True)
    gate_frames = merge_csv(args.gate_csv, args.output_gate)
    pose_frames = merge_jsonl(args.pose_jsonl, args.output_pose)
    print(json.dumps({"gate_frames": gate_frames, "pose_frames": pose_frames},
                     indent=2))


if __name__ == "__main__":
    main()
