#!/usr/bin/env bash
set -u -o pipefail

primary_pid="${1:?usage: $0 PRIMARY_PID}"
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_dir="$(cd "$project_dir/.." && pwd)"
python_bin="${BIT_PYTHON:-/root/miniconda3/envs/BIT_Track/bin/python}"
run_root="${BIT_STAGE4_CONFIRM_ROOT:-$workspace_dir/baseline/moped/stage4_confirmation_v1}"
analysis_dir="$run_root/analysis/final"
protocol="$project_dir/config/stage4_confirmation_v1.json"

while kill -0 "$primary_pid" 2>/dev/null; do
    sleep 30
done

primary_count=$(find "$run_root" -type f -name metrics.json | wc -l)
if [ "$primary_count" -ne 21 ]; then
    echo "[FAIL] primary confirmation has $primary_count/21 completed runs"
    exit 1
fi

build_and_evaluate() {
    mkdir -p "$analysis_dir"
    "$python_bin" "$project_dir/scripts/build_normalized_pose_uncertainty_dataset.py" \
        "$run_root" --run-glob '*/*/confirm_s*/diagnostics.csv' \
        --output "$analysis_dir/qpose_confirmation.jsonl"
    "$python_bin" "$project_dir/scripts/evaluate_stage4_confirmation.py" \
        --run-root "$run_root" \
        --pose-dataset "$analysis_dir/qpose_confirmation.jsonl" \
        --protocol "$protocol" --workspace-root "$workspace_dir" \
        --output-dir "$analysis_dir" --bootstrap-replicates 2000 \
        --bootstrap-seed 20260822
}

build_and_evaluate
fallback_required=$("$python_bin" -c \
    "import json; print(int(json.load(open('$analysis_dir/confirmation_summary.json'))['fallback_required']))")

if [ "$fallback_required" -eq 1 ]; then
    echo "[EXPAND] preregistered sample/label threshold triggered"
    BIT_STAGE4_CONFIRM_ROOT="$run_root" \
        bash "$project_dir/scripts/run_stage4_confirmation.sh" fallback
    final_count=$(find "$run_root" -type f -name metrics.json | wc -l)
    if [ "$final_count" -ne 36 ]; then
        echo "[FAIL] expanded confirmation has $final_count/36 completed runs"
        exit 1
    fi
    build_and_evaluate
fi

echo "[DONE] stage4 confirmation analysis: $analysis_dir/confirmation_summary.json"
