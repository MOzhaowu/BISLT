#!/usr/bin/env bash
set -u -o pipefail

runner_pid="${1:?usage: $0 RUNNER_PID}"
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_dir="$(cd "$project_dir/.." && pwd)"
python_bin="${BIT_PYTHON:-/root/miniconda3/envs/BIT_Track/bin/python}"
run_root="${BIT_STAGE4_P1V2_DEV_ROOT:-$workspace_dir/baseline/moped/stage4_p1v2_target_development}"
analysis_dir="$run_root/analysis"
v1_dir="$workspace_dir/baseline/moped/stage4_confirmation_v1/analysis/final"
protocol="$project_dir/config/stage4_confirmation_v1.json"

while kill -0 "$runner_pid" 2>/dev/null; do
    sleep 30
done

completed=$(find "$run_root" -type f -name metrics.json | wc -l)
if [ "$completed" -ne 12 ]; then
    echo "[FAIL] P1-v2 development has $completed/12 completed runs"
    exit 1
fi

mkdir -p "$analysis_dir"
"$python_bin" "$project_dir/scripts/build_normalized_pose_uncertainty_dataset.py" \
    "$run_root" --run-glob '*/*/p1v2_dev_s*/diagnostics.csv' \
    --output "$analysis_dir/qpose_p1v2_development.jsonl"
"$python_bin" "$project_dir/scripts/build_stage4_frozen_gate_dataset.py" \
    --run-root "$run_root" --run-glob 'p1v2_dev_s*' \
    --pose-dataset "$analysis_dir/qpose_p1v2_development.jsonl" \
    --protocol "$protocol" --workspace-root "$workspace_dir" \
    --output "$analysis_dir/frozen_gate_p1v2_development.csv"
"$python_bin" "$project_dir/scripts/merge_stage4_development_inputs.py" \
    --gate-csv "$v1_dir/confirmation_frames.csv" \
    --gate-csv "$analysis_dir/frozen_gate_p1v2_development.csv" \
    --pose-jsonl "$v1_dir/qpose_confirmation.jsonl" \
    --pose-jsonl "$analysis_dir/qpose_p1v2_development.jsonl" \
    --output-gate "$analysis_dir/combined_gate_development.csv" \
    --output-pose "$analysis_dir/combined_qpose_development.jsonl"
"$python_bin" "$project_dir/scripts/evaluate_stage4_p1v2_hierarchical_gate.py" \
    "$analysis_dir/combined_gate_development.csv" \
    "$analysis_dir/combined_qpose_development.jsonl" \
    --output-dir "$analysis_dir/hierarchical_gate_expanded" \
    >"$analysis_dir/hierarchical_gate_expanded.log" 2>&1

echo "[DONE] P1-v2 expanded nested LOSO: $analysis_dir/hierarchical_gate_expanded"
