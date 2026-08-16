#!/usr/bin/env python3
"""Create a reproducibility manifest for one BIT experiment run."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def command_output(command, cwd):
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ).stdout.strip()
    except OSError:
        return None


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--expected-frames", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parents[1]
    config = args.config.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "moped",
        "object": args.object,
        "sequence": args.sequence,
        "expected_frames": args.expected_frames,
        "seed": args.seed,
        "config": {
            "path": str(config),
            "sha256": sha256(config),
        },
        "experiment": {
            "run_tag": os.environ.get("BIT_RUN_TAG"),
            "config_glob": os.environ.get("BIT_CONFIG_GLOB"),
            "consume_once": os.environ.get("BIT_CONSUME_ONCE", "1").lower()
            not in ("0", "false", "no"),
            "model_validation_override": json.loads(
                os.environ.get("BIT_MODEL_VALIDATION_JSON", "null")
            ),
        },
        "code": {
            "git_commit": command_output(["git", "rev-parse", "HEAD"], project_dir),
            "git_status_porcelain": command_output(
                ["git", "status", "--short", "--", "BIT-main"], project_dir.parent
            ),
        },
        "runtime": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "hostname": platform.node(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu": command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                ],
                project_dir,
            ),
        },
    }
    output = args.run_dir / "manifest.json"
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(output)


if __name__ == "__main__":
    main()
