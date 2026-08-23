#!/usr/bin/env bash
set -u -o pipefail

runner_pid="${1:?usage: $0 RUNNER_PID}"
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_dir="$(cd "$project_dir/.." && pwd)"
python_bin="${BIT_PYTHON:-/root/miniconda3/envs/BIT_Track/bin/python}"
run_root="${BIT_STAGE4_P1V2_RECONFIRM_ROOT:-$workspace_dir/baseline/moped/stage4_p1v2_reconfirmation}"
analysis_dir="$run_root/analysis"
parent_protocol="$project_dir/config/stage4_confirmation_v1.json"
frozen_config="$project_dir/config/stage4_p1v2_reconfirmation.json"

while kill -0 "$runner_pid" 2>/dev/null; do
    sleep 30
done

completed=$(find "$run_root" -type f -name metrics.json | wc -l)
if [ "$completed" -ne 12 ]; then
    echo "[FAIL] P1-v2 reconfirmation has $completed/12 completed runs"
    exit 1
fi

mkdir -p "$analysis_dir"
"$python_bin" "$project_dir/scripts/build_normalized_pose_uncertainty_dataset.py" \
    "$run_root" --run-glob '*/*/p1v2_reconfirm_s*/diagnostics.csv' \
    --output "$analysis_dir/qpose_reconfirmation.jsonl"
"$python_bin" "$project_dir/scripts/build_stage4_frozen_gate_dataset.py" \
    --run-root "$run_root" --run-glob 'p1v2_reconfirm_s*' \
    --pose-dataset "$analysis_dir/qpose_reconfirmation.jsonl" \
    --protocol "$parent_protocol" --workspace-root "$workspace_dir" \
    --output "$analysis_dir/frozen_gate_reconfirmation.csv"
"$python_bin" "$project_dir/scripts/evaluate_stage4_p1v2_reconfirmation.py" \
    "$analysis_dir/frozen_gate_reconfirmation.csv" \
    "$analysis_dir/qpose_reconfirmation.jsonl" "$frozen_config" \
    --output-dir "$analysis_dir/final" \
    >"$analysis_dir/reconfirmation.log" 2>&1

echo "[DONE] P1-v2 independent reconfirmation: $analysis_dir/final"
