#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
results_root="${BIT_PRIORITY1_RESULTS_ROOT:-$project_dir/../baseline/moped/priority1_ablation}"
python_bin="${BIT_PYTHON:-/root/miniconda3/envs/BIT_Track/bin/python}"

cd "$project_dir"

for object_name in black_drill cheezit duplo_dude; do
    "$python_bin" scripts/run_gate_sweep.py \
        --presets original sync iou multi \
        --thresholds 0.01 \
        --seeds 41 43 \
        --config-glob "$object_name/evaluation/0[0-2].yml" \
        --output "$results_root/manifests/${object_name}_seeds_41_43.json" \
        --results-root "$results_root" \
        --execute
done

