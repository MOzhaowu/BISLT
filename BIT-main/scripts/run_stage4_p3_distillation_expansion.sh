#!/usr/bin/env bash
set -Eeuo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_dir="$(cd "$project_dir/.." && pwd)"
python_bin="${BIT_PYTHON:-/root/miniconda3/envs/BIT_Track/bin/python}"
capture_root="${BIT_STAGE4_P3_CAPTURE_ROOT:-$workspace_dir/baseline/moped/stage4_p3_distillation/capture}"
analysis_root="${BIT_STAGE4_P3_ANALYSIS_ROOT:-$workspace_dir/baseline/moped/stage4_p3_distillation/expansion_s71_73}"
seeds="71 72 73"
cases="black_drill/00 cheezit/00 cheezit/02 cheezit/04 duplo_dude/02 toy_plane/03"
teacher_policy="p1v2_short_loo_control_iou_distillation_expansion"
git_commit="$(git -C "$workspace_dir" rev-parse HEAD)"

mkdir -p "$analysis_root"
printf '%s\n' "$BASHPID" > "$analysis_root/pipeline.pid"

mark() {
    printf '%s %s\n' "$(date --iso-8601=seconds)" "$1" |
        tee "$analysis_root/pipeline.state"
}

on_error() {
    local code="$?"
    mark "failed_exit_${code}_line_${BASH_LINENO[0]}"
    exit "$code"
}
trap on_error ERR

mark "capture_running"
BIT_STAGE4_P2_CAPTURE_ROOT="$capture_root" \
BIT_STAGE4_P2_CAPTURE_SEEDS="$seeds" \
BIT_STAGE4_P2_CAPTURE_CASES="$cases" \
    bash "$project_dir/scripts/run_stage4_p2_replay_capture.sh"

mark "aligned_inputs_building"
for seed in $seeds; do
    seed_root="$analysis_root/seed_$seed"
    mkdir -p "$seed_root"
    "$python_bin" "$project_dir/scripts/build_normalized_pose_uncertainty_dataset.py" \
        "$capture_root" \
        --run-glob "*/*/p2_capture_s$seed/diagnostics.csv" \
        --output "$seed_root/qpose.jsonl"
    "$python_bin" "$project_dir/scripts/build_stage4_frozen_gate_dataset.py" \
        --run-root "$capture_root" \
        --run-glob "p2_capture_s$seed" \
        --pose-dataset "$seed_root/qpose.jsonl" \
        --protocol "$project_dir/config/stage4_confirmation_v1.json" \
        --workspace-root "$workspace_dir" \
        --output "$seed_root/parent_gate.csv"
done

"$python_bin" "$project_dir/scripts/merge_stage4_development_inputs.py" \
    --gate-csv "$analysis_root/seed_71/parent_gate.csv" \
    --gate-csv "$analysis_root/seed_72/parent_gate.csv" \
    --gate-csv "$analysis_root/seed_73/parent_gate.csv" \
    --pose-jsonl "$analysis_root/seed_71/qpose.jsonl" \
    --pose-jsonl "$analysis_root/seed_72/qpose.jsonl" \
    --pose-jsonl "$analysis_root/seed_73/qpose.jsonl" \
    --output-gate "$analysis_root/parent_gate.csv" \
    --output-pose "$analysis_root/qpose.jsonl"
"$python_bin" "$project_dir/scripts/apply_stage4_p1v2_frozen_gate.py" \
    "$analysis_root/parent_gate.csv" \
    "$analysis_root/qpose.jsonl" \
    "$project_dir/config/stage4_p1v2_reconfirmation.json" \
    --output "$analysis_root/p1v2_frames.csv"

mark "dual_reference_running"
dual_previous=""
for seed in $seeds; do
    dual_output="$analysis_root/dual_through_s$seed.jsonl"
    reference_args=()
    if [[ -n "$dual_previous" ]]; then
        reference_args=(--reference-results "$dual_previous")
    fi
    "$python_bin" "$project_dir/scripts/replay_stage4_p2_geometry.py" \
        --run-root "$capture_root" \
        --run-glob "p2_capture_s$seed" \
        --candidate-frames "$analysis_root/p1v2_frames.csv" \
        --parent-frames "$analysis_root/parent_gate.csv" \
        --frozen-config "$project_dir/config/stage4_p1v2_reconfirmation.json" \
        --output "$dual_output" \
        --policies p1v2_short_loo_dual_trust_region \
        --loo-steps 25 "${reference_args[@]}"
    dual_previous="$dual_output"
done

mark "teacher_projection_200_running"
replay="$analysis_root/replay.jsonl"
selected="$analysis_root/selected.jsonl"
"$python_bin" "$project_dir/scripts/replay_stage4_p2_geometry.py" \
    --run-root "$capture_root" \
    --run-glob 'p2_capture_s*' \
    --candidate-frames "$analysis_root/p1v2_frames.csv" \
    --parent-frames "$analysis_root/parent_gate.csv" \
    --frozen-config "$project_dir/config/stage4_p1v2_reconfirmation.json" \
    --output "$replay" \
    --policies p1v2_short_loo_dual_regularized \
    --dual-reference-results "$dual_previous" \
    --proposal-steps 200 \
    --pose-consistency-weight 1.0 \
    --temporal-stability-weight 1.0 \
    --control-iou-weight 1.0 \
    --pose-probe-interval 1 \
    --regularization-scope validation_control_frames \
    --candidate-optimizer constrained_projection \
    --projection-primary validation_control_iou \
    --projection-cycles 20 \
    --constraint-activation-epsilon 1e-8
"$python_bin" "$project_dir/scripts/apply_stage4_p2_multimetric_trust_region.py" \
    "$replay" --output "$selected" \
    --source-policy p1v2_short_loo_dual_regularized \
    --target-policy "$teacher_policy"

mark "teacher_dataset_building"
source_manifest="$analysis_root/teacher_source_manifest.json"
teacher_dataset="$analysis_root/teacher_events.jsonl"
"$python_bin" "$project_dir/scripts/create_stage4_p3_teacher_source_manifest.py" \
    "$project_dir/config/stage4_p3_distillation_protocol.json" \
    "$selected" "$analysis_root/p1v2_frames.csv" "$analysis_root/parent_gate.csv" \
    --policy "$teacher_policy" --seeds 71 72 73 \
    --git-commit "$git_commit" --output "$source_manifest"
"$python_bin" "$project_dir/scripts/build_stage4_p3_teacher_dataset.py" \
    "$selected" "$analysis_root/p1v2_frames.csv" "$analysis_root/parent_gate.csv" \
    "$source_manifest" --output "$teacher_dataset"
"$python_bin" "$project_dir/scripts/evaluate_stage4_p3_student_baseline.py" \
    "$teacher_dataset" "$source_manifest" \
    --output "$analysis_root/student_loso_baseline.json"

mark "complete"
