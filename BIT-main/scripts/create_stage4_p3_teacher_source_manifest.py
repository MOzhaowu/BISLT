#!/usr/bin/env python3
"""Bind generated teacher sources to hashes for the distillation builder."""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_protocol", type=Path)
    parser.add_argument("selected_results", type=Path)
    parser.add_argument("p1v2_frames", type=Path)
    parser.add_argument("parent_gate", type=Path)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.base_protocol.read_text())
    protocol["status"] = "generated_teacher_source_manifest"
    protocol["teacher"].update({
        "git_commit": args.git_commit,
        "policy": args.policy,
        "development_seeds": sorted(args.seeds),
        "selected_results_sha256": sha256(args.selected_results),
        "p1v2_frames_sha256": sha256(args.p1v2_frames),
        "parent_gate_sha256": sha256(args.parent_gate),
        "source_manifest_parent_protocol_sha256": sha256(args.base_protocol),
    })
    locked = set(protocol["teacher"]["locked_report_only_seeds"])
    if locked.intersection(args.seeds):
        raise ValueError("locked report-only seed requested for student source")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "seeds": sorted(args.seeds),
        "policy": args.policy,
        "sha256": sha256(args.output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
