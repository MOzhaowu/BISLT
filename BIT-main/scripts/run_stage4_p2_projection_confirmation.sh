#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_dir="$(cd "$project_dir/.." && pwd)"
capture_root="$workspace_dir/baseline/moped/stage4_p2_offline_replay/capture"
analysis_root="$capture_root/analysis"
development_root="$analysis_root/development_s59_64"
candidate_root="$development_root/candidate_regularization"
confirmation_root="$candidate_root/confirmation_200"
independent_root="$analysis_root/independent_s68_70_projection"
protocol="$project_dir/config/stage4_p2_control_iou_projection_protocol.json"
dual_development="$development_root/geometry_replay_dual_trust_v1.jsonl"
python_bin="${BIT_PYTHON:-/root/miniconda3/envs/BIT_Track/bin/python}"

mkdir -p "$confirmation_root" "$independent_root"
printf '%s\n' "$BASHPID" > "$confirmation_root/pipeline.pid"

mark() {
    printf '%s %s\n' "$(date --iso-8601=seconds)" "$1" |
        tee "$confirmation_root/pipeline.state"
}

run_projected_replay() {
    local output="$1"
    local run_glob="$2"
    local candidate_frames="$3"
    local parent_frames="$4"
    local dual_reference="$5"
    "$python_bin" "$project_dir/scripts/replay_stage4_p2_geometry.py" \
        --run-root "$capture_root" \
        --run-glob "$run_glob" \
        --candidate-frames "$candidate_frames" \
        --parent-frames "$parent_frames" \
        --frozen-config "$project_dir/config/stage4_p1v2_reconfirmation.json" \
        --output "$output" \
        --policies p1v2_short_loo_dual_regularized \
        --dual-reference-results "$dual_reference" \
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
}

apply_and_evaluate() {
    local replay="$1"
    local selected="$2"
    local evaluation="$3"
    local gate_summary="$4"
    local target_policy="$5"
    "$python_bin" "$project_dir/scripts/apply_stage4_p2_multimetric_trust_region.py" \
        "$replay" --output "$selected" \
        --source-policy p1v2_short_loo_dual_regularized \
        --target-policy "$target_policy"
    "$python_bin" "$project_dir/scripts/evaluate_stage4_p2_influence.py" \
        "$selected" "$gate_summary" \
        "$project_dir/config/stage4_p2_multimetric_trust_region_protocol.json" \
        --policy "$target_policy" --output "$evaluation"
}

mark "development_200_running"
development_replay="$confirmation_root/replay.jsonl"
development_selected="$confirmation_root/selected.jsonl"
development_evaluation="$confirmation_root/evaluation.json"
run_projected_replay \
    "$development_replay" 'p2_capture_s*' \
    "$development_root/p1v2_frames.csv" \
    "$development_root/parent_gate.csv" \
    "$dual_development"
apply_and_evaluate \
    "$development_replay" "$development_selected" "$development_evaluation" \
    "$development_root/p1v2_frames.summary.json" \
    p1v2_short_loo_control_iou_projected_200

mark "development_200_finalizing"
"$python_bin" "$project_dir/scripts/finalize_stage4_p2_projection.py" \
    "$development_replay" "$development_evaluation" "$protocol" \
    --mode development \
    --output "$project_dir/config/stage4_p2_control_iou_projection_frozen.json"

mark "development_frozen_independent_capture_running"
BIT_STAGE4_P2_CAPTURE_SEEDS="68 69 70" \
    bash "$project_dir/scripts/run_stage4_p2_replay_capture.sh"

mark "independent_inputs_building"
for seed in 68 69 70; do
    seed_root="$independent_root/seed_$seed"
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
    --gate-csv "$independent_root/seed_68/parent_gate.csv" \
    --gate-csv "$independent_root/seed_69/parent_gate.csv" \
    --gate-csv "$independent_root/seed_70/parent_gate.csv" \
    --pose-jsonl "$independent_root/seed_68/qpose.jsonl" \
    --pose-jsonl "$independent_root/seed_69/qpose.jsonl" \
    --pose-jsonl "$independent_root/seed_70/qpose.jsonl" \
    --output-gate "$independent_root/parent_gate.csv" \
    --output-pose "$independent_root/qpose.jsonl"
"$python_bin" "$project_dir/scripts/apply_stage4_p1v2_frozen_gate.py" \
    "$independent_root/parent_gate.csv" \
    "$independent_root/qpose.jsonl" \
    "$project_dir/config/stage4_p1v2_reconfirmation.json" \
    --output "$independent_root/p1v2_frames.csv"

mark "independent_dual_reference_running"
dual_previous=""
for seed in 68 69 70; do
    dual_output="$independent_root/dual_through_s$seed.jsonl"
    reference_args=()
    if [[ -n "$dual_previous" ]]; then
        reference_args=(--reference-results "$dual_previous")
    fi
    "$python_bin" "$project_dir/scripts/replay_stage4_p2_geometry.py" \
        --run-root "$capture_root" \
        --run-glob "p2_capture_s$seed" \
        --candidate-frames "$independent_root/p1v2_frames.csv" \
        --parent-frames "$independent_root/parent_gate.csv" \
        --frozen-config "$project_dir/config/stage4_p1v2_reconfirmation.json" \
        --output "$dual_output" \
        --policies p1v2_short_loo_dual_trust_region \
        --loo-steps 25 "${reference_args[@]}"
    dual_previous="$dual_output"
done

mark "independent_projection_200_running"
independent_replay="$independent_root/replay.jsonl"
independent_selected="$independent_root/selected.jsonl"
independent_evaluation="$independent_root/evaluation.json"
run_projected_replay \
    "$independent_replay" 'p2_capture_s*' \
    "$independent_root/p1v2_frames.csv" \
    "$independent_root/parent_gate.csv" \
    "$dual_previous"
apply_and_evaluate \
    "$independent_replay" "$independent_selected" "$independent_evaluation" \
    "$independent_root/p1v2_frames.summary.json" \
    p1v2_short_loo_control_iou_projected_independent
"$python_bin" "$project_dir/scripts/finalize_stage4_p2_projection.py" \
    "$independent_replay" "$independent_evaluation" "$protocol" \
    --mode independent \
    --output "$independent_root/final_evaluation.json"

mark "complete"
