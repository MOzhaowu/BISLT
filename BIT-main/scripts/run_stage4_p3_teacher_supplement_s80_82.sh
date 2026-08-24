#!/usr/bin/env bash
set -Eeuo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_dir="$(cd "$project_dir/.." && pwd)"
python_bin="${BIT_PYTHON:-/root/miniconda3/envs/BIT_Track/bin/python}"
protocol="$project_dir/config/stage4_p3_teacher_supplement_s80_82_protocol.json"
capture_root="$workspace_dir/baseline/moped/stage4_p3_distillation/supplement_s80_82/capture"
analysis_root="$workspace_dir/baseline/moped/stage4_p3_distillation/supplement_s80_82/analysis"
combined_root="$workspace_dir/baseline/moped/stage4_p3_distillation/combined_s59_82"
seeds="80 81 82"
cases="black_drill/00 cheezit/00 cheezit/02 cheezit/04 duplo_dude/02 toy_plane/03"
teacher_policy="p1v2_short_loo_control_iou_distillation_supplement_s80_82"
git_commit="$(git -C "$workspace_dir" rev-parse HEAD)"

mkdir -p "$analysis_root" "$combined_root"
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

mark "preflight"
"$python_bin" -m unittest \
    tests.test_stage4_p3_teacher_supplement_protocol \
    tests.test_stage4_p3_teacher_supplement_audit
"$python_bin" -c '
import json, pathlib, sys
p=json.loads(pathlib.Path(sys.argv[1]).read_text())
assert p["status"] == "preregistered_not_started"
assert p["supplement_design"]["development_seeds"] == [80, 81, 82]
assert p["supplement_design"]["groups"] == sys.argv[2].split()
assert p["supplement_design"]["planned_runs"] == 18
assert p["supplement_design"]["no_early_stopping_after_intermediate_labels"]
assert p["supplement_design"]["student_fitting_during_supplement"] is False
locked=p["locked_sets"]
assert not ({80,81,82} & set(locked["formula_confirmation_seeds"] + locked["external_independent_seeds"]))
' "$protocol" "$cases"
if find "$workspace_dir/baseline/moped/stage4_p3_distillation" \
    -type d \( -name 'p2_capture_s74' -o -name 'p2_capture_s75' \
    -o -name 'p2_capture_s76' -o -name 'p2_capture_s77' \
    -o -name 'p2_capture_s78' -o -name 'p2_capture_s79' \) \
    -print -quit | grep -q .; then
    echo "locked seed 74-79 capture exists; refusing supplement launch" >&2
    exit 3
fi

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
    --gate-csv "$analysis_root/seed_80/parent_gate.csv" \
    --gate-csv "$analysis_root/seed_81/parent_gate.csv" \
    --gate-csv "$analysis_root/seed_82/parent_gate.csv" \
    --pose-jsonl "$analysis_root/seed_80/qpose.jsonl" \
    --pose-jsonl "$analysis_root/seed_81/qpose.jsonl" \
    --pose-jsonl "$analysis_root/seed_82/qpose.jsonl" \
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
    --policy "$teacher_policy" --seeds 80 81 82 \
    --git-commit "$git_commit" --output "$source_manifest"
"$python_bin" "$project_dir/scripts/build_stage4_p3_teacher_dataset.py" \
    "$selected" "$analysis_root/p1v2_frames.csv" "$analysis_root/parent_gate.csv" \
    "$source_manifest" --allow-zero-event-seeds --output "$teacher_dataset"

mark "combined_quality_audit"
combined_dataset="$combined_root/teacher_events.jsonl"
combined_audit="$combined_root/quality_audit.json"
"$python_bin" "$project_dir/scripts/audit_stage4_p3_teacher_dataset.py" \
    "$workspace_dir/baseline/moped/stage4_p3_distillation/development_s59_64/teacher_events.jsonl" \
    "$workspace_dir/baseline/moped/stage4_p3_distillation/expansion_s71_73/teacher_events.jsonl" \
    "$teacher_dataset" \
    --base-protocol "$project_dir/config/stage4_p3_distillation_protocol.json" \
    --expansion-protocol "$project_dir/config/stage4_p3_distillation_expansion_protocol.json" \
    --combined-output "$combined_dataset" \
    --output "$combined_audit"
"$python_bin" "$project_dir/scripts/audit_stage4_p3_teacher_supplement.py" \
    --protocol "$protocol" \
    --supplement-dataset "$teacher_dataset" \
    --combined-dataset "$combined_dataset" \
    --combined-audit "$combined_audit" \
    --capture-root "$capture_root" \
    --output "$combined_root/supplement_acceptance.json"

if "$python_bin" -c '
import json, pathlib, sys
report=json.loads(pathlib.Path(sys.argv[1]).read_text())
raise SystemExit(0 if report["accepted_for_student_development"] else 1)
' "$combined_root/supplement_acceptance.json"; then
    mark "complete_gate_passed"
else
    mark "complete_gate_failed"
fi
