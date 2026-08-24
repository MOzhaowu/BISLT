#!/usr/bin/env bash
set -u -o pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_dir="$(cd "$project_dir/.." && pwd)"
results_root="${BIT_STAGE4_P2_CAPTURE_ROOT:-$workspace_dir/baseline/moped/stage4_p2_offline_replay/capture}"
seeds="${BIT_STAGE4_P2_CAPTURE_SEEDS:-59}"
cases="${BIT_STAGE4_P2_CAPTURE_CASES:-duplo_dude/01 toy_plane/02}"

run_case() {
    local object="$1"
    local sequence="$2"
    local seed="$3"
    echo "[P2-CAPTURE] $object/$sequence seed=$seed"
    BIT_BASELINE_ROOT="$results_root" \
    BIT_CONFIG_GLOB="$object/evaluation/$sequence.yml" \
    BIT_RUN_TAG="p2_capture_s$seed" \
    BIT_RANDOM_SEED="$seed" \
    BIT_CONSUME_ONCE=1 \
    BIT_POSE_TEACHER_STRIDE=0 \
    BIT_POSE_TEACHER_ABLATION=0 \
    BIT_SEQUENCE_TIMEOUT="${BIT_SEQUENCE_TIMEOUT:-45m}" \
    BIT_METRICS_GLOB='**/metrics.json' \
        bash "$project_dir/scripts/run_moped_baseline.sh"
}

for seed in $seeds; do
    for case_name in $cases; do
        object="${case_name%/*}"
        sequence="${case_name#*/}"
        if [[ -z "$object" || -z "$sequence" || "$object" == "$sequence" ]]; then
            echo "invalid BIT_STAGE4_P2_CAPTURE_CASES entry: $case_name" >&2
            exit 2
        fi
        if [[ ! -f "$project_dir/config/moped/$object/evaluation/$sequence.yml" ]]; then
            echo "missing sequence config: $object/evaluation/$sequence.yml" >&2
            exit 2
        fi
        run_case "$object" "$sequence" "$seed" || exit 1
    done
done
