#!/usr/bin/env bash
set -u -o pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_dir="$(cd "$project_dir/.." && pwd)"
results_root="${BIT_STAGE4_P1V2_DEV_ROOT:-$workspace_dir/baseline/moped/stage4_p1v2_target_development}"

run_case() {
    local object="$1"
    local sequence="$2"
    local seed="$3"
    echo "[P1V2-DEV] $object/$sequence seed=$seed"
    BIT_BASELINE_ROOT="$results_root" \
    BIT_CONFIG_GLOB="$object/evaluation/$sequence.yml" \
    BIT_RUN_TAG="p1v2_dev_s$seed" \
    BIT_RANDOM_SEED="$seed" \
    BIT_CONSUME_ONCE=1 \
    BIT_POSE_TEACHER_STRIDE=0 \
    BIT_POSE_TEACHER_ABLATION=0 \
    BIT_SEQUENCE_TIMEOUT="${BIT_SEQUENCE_TIMEOUT:-45m}" \
    BIT_METRICS_GLOB='**/metrics.json' \
        bash "$project_dir/scripts/run_moped_baseline.sh"
}

for seed in 47 48 49 50 51 52; do
    run_case duplo_dude 01 "$seed" || exit 1
    run_case toy_plane 02 "$seed" || exit 1
done
