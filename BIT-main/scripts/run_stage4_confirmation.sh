#!/usr/bin/env bash
set -u -o pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_dir="$(cd "$project_dir/.." && pwd)"
mode="${1:-primary}"
results_root="${BIT_STAGE4_CONFIRM_ROOT:-$workspace_dir/baseline/moped/stage4_confirmation_v1}"

run_case() {
    local object="$1"
    local sequence="$2"
    local seed="$3"
    echo "[CONFIRM] $object/$sequence seed=$seed"
    BIT_BASELINE_ROOT="$results_root" \
    BIT_CONFIG_GLOB="$object/evaluation/$sequence.yml" \
    BIT_RUN_TAG="confirm_s$seed" \
    BIT_RANDOM_SEED="$seed" \
    BIT_CONSUME_ONCE=1 \
    BIT_POSE_TEACHER_STRIDE=0 \
    BIT_POSE_TEACHER_ABLATION=0 \
    BIT_SEQUENCE_TIMEOUT="${BIT_SEQUENCE_TIMEOUT:-45m}" \
    BIT_METRICS_GLOB='**/metrics.json' \
        bash "$project_dir/scripts/run_moped_baseline.sh"
}

run_primary() {
    local object seed
    for seed in 41 42 43; do
        for object in duster graphics_card orange_drill pouch remote; do
            run_case "$object" 00 "$seed" || return 1
        done
    done
    for seed in 44 45 46; do
        run_case duplo_dude 01 "$seed" || return 1
        run_case toy_plane 02 "$seed" || return 1
    done
}

run_fallback() {
    local object seed
    for seed in 41 42 43; do
        for object in duster graphics_card orange_drill pouch remote; do
            run_case "$object" 02 "$seed" || return 1
        done
    done
}

case "$mode" in
    primary) run_primary ;;
    fallback) run_fallback ;;
    all) run_primary && run_fallback ;;
    *) echo "usage: $0 [primary|fallback|all]" >&2; exit 2 ;;
esac
