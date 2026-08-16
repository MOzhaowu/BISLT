#!/usr/bin/env bash
set -u -o pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_dir="$(cd "$project_dir/.." && pwd)"
python_bin="${BIT_PYTHON:-/root/miniconda3/envs/BIT_Track/bin/python}"
baseline_root="${BIT_BASELINE_ROOT:-$workspace_dir/baseline/moped}"
timeout_per_sequence="${BIT_SEQUENCE_TIMEOUT:-30m}"
run_number="${BIT_RUN_NUMBER:-1}"
run_tag="${BIT_RUN_TAG:-run_$run_number}"
config_glob="${BIT_CONFIG_GLOB:-*/evaluation/*.yml}"
random_seed="${BIT_RANDOM_SEED:-42}"

cd "$project_dir" || exit 1

mapfile -t configs < <(find config/moped -path "config/moped/$config_glob" -type f | sort)
total="${#configs[@]}"
completed=0
failed=0

for config in "${configs[@]}"; do
    relative="${config#config/moped/}"
    object="${relative%%/*}"
    sequence_file="${relative##*/}"
    sequence="${sequence_file%.yml}"
    sequence_root="$baseline_root/$object/$sequence"
    run_dir="$sequence_root/$run_tag"

    if [ -f "$run_dir/metrics.json" ]; then
        echo "[SKIP] $object/$sequence already has metrics"
        completed=$((completed + 1))
        continue
    fi

    expected_frames=$(find "$workspace_dir/moped/$object/evaluation/$sequence/color" -maxdepth 1 -type f -name '*.jpg' ! -name '._*' | wc -l)
    mkdir -p "$run_dir"
    "$python_bin" scripts/create_run_manifest.py "$run_dir" \
        --config "$config" --object "$object" --sequence "$sequence" \
        --expected-frames "$expected_frames" --seed "$random_seed" >/dev/null
    start_epoch=$(date +%s)
    echo "[RUN $((completed + failed + 1))/$total] $object/$sequence ($expected_frames frames)"
    dataset_sequence_dir="$workspace_dir/moped/$object/evaluation/$sequence"
    reference_count=$(find "$dataset_sequence_dir/refers/pose" -maxdepth 1 -type f -name '*.pose' 2>/dev/null | wc -l || true)
    if [ "$reference_count" -ne "$expected_frames" ]; then
        mkdir -p "$run_dir/reference_generation"
        echo "[REF] $object/$sequence ($reference_count/$expected_frames present)"
        set +e
        timeout "$timeout_per_sequence" xvfb-run -a -s '-screen 0 1280x1024x24' \
            sh gen_refers.sh moped "$object/evaluation/$sequence.yml" \
            >"$run_dir/reference_generation/run.log" 2>&1
        reference_status=$?
        set -e
        reference_count=$(find "$dataset_sequence_dir/refers/pose" -maxdepth 1 -type f -name '*.pose' 2>/dev/null | wc -l || true)
        if [ "$reference_status" -ne 0 ] || [ "$reference_count" -ne "$expected_frames" ]; then
            echo "[FAIL-REF] $object/$sequence exit=$reference_status files=$reference_count/$expected_frames"
            failed=$((failed + 1))
            continue
        fi
    fi


    set +e
    timeout "$timeout_per_sequence" xvfb-run -a -s '-screen 0 1280x1024x24' \
        "$python_bin" examples/run_example.py moped "$object/evaluation/$sequence.yml" \
        >"$run_dir/run.log" 2>&1
    status=$?
    set -e
    end_epoch=$(date +%s)
    printf '%s\n' "$((end_epoch - start_epoch))" >"$run_dir/elapsed_seconds.txt"
    printf '%s\n' "$status" >"$run_dir/exit_code.txt"

    result_relative=$(grep -oE '\.\./result/Summer/[^/[:space:]]+/[^/[:space:]]+' "$run_dir/run.log" | head -1 || true)
    if [ "$status" -eq 0 ] && [ -n "$result_relative" ] && [ -d "$project_dir/$result_relative" ]; then
        mkdir -p "$run_dir/raw"
        cp -a "$project_dir/$result_relative/." "$run_dir/raw/"
        registry_file="$dataset_sequence_dir/cmc/model_registry.jsonl"
        if [ -f "$registry_file" ]; then
            cp "$registry_file" "$run_dir/model_registry.jsonl"
        fi
        if [ -d "$dataset_sequence_dir/cmc/model_candidates" ]; then
            cp -a "$dataset_sequence_dir/cmc/model_candidates" "$run_dir/"
        fi
        if "$python_bin" scripts/evaluate_baseline.py "$run_dir" \
            --object "$object" --sequence "$sequence" --expected-frames "$expected_frames" \
            >"$run_dir/evaluation.log" 2>&1 && \
           "$python_bin" scripts/build_diagnostics.py "$run_dir" \
            >"$run_dir/diagnostics.log" 2>&1 && \
           "$python_bin" scripts/check_run_completeness.py "$run_dir" \
            >"$run_dir/completeness.json" 2>&1; then
            echo "[OK] $object/$sequence in $((end_epoch - start_epoch))s"
            completed=$((completed + 1))
            continue
        fi
    fi

    echo "[FAIL] $object/$sequence exit=$status; see $run_dir/run.log"
    failed=$((failed + 1))
done

echo "Batch finished: completed=$completed failed=$failed total=$total"
if find "$baseline_root" -name metrics.json -type f -print -quit 2>/dev/null | grep -q .; then
    mkdir -p "$baseline_root/summary"
    "$python_bin" scripts/aggregate_results.py "$baseline_root" \
        --run-glob "${BIT_METRICS_GLOB:-**/run_[0-9]*/metrics.json}" \
        --output-dir "$baseline_root/summary" >"$baseline_root/summary/aggregate.log"
fi
test "$failed" -eq 0
