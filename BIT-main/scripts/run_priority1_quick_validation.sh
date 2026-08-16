#!/usr/bin/env bash
set -u -o pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
results_root="${BIT_PRIORITY1_RESULTS_ROOT:-$project_dir/../baseline/moped/priority1_ablation}"
python_bin="${BIT_PYTHON:-/root/miniconda3/envs/BIT_Track/bin/python}"
max_attempts="${BIT_MAX_ATTEMPTS:-2}"
failed_cases="$results_root/control/quick_validation.failed"

cd "$project_dir" || exit 1
mkdir -p "$results_root/control" "$results_root/manifests"
: >"$failed_cases"

for seed in 41 43; do
    for object_name in black_drill cheezit duplo_dude; do
        for sequence in 00 01 02; do
            run_tag="multi_t0.01_s${seed}"
            metrics="$results_root/$object_name/$sequence/$run_tag/metrics.json"
            if [ -f "$metrics" ]; then
                echo "[SKIP] $object_name/$sequence $run_tag"
                continue
            fi

            succeeded=0
            for attempt in $(seq 1 "$max_attempts"); do
                echo "[ATTEMPT $attempt/$max_attempts] $object_name/$sequence $run_tag"
                if "$python_bin" scripts/run_gate_sweep.py \
                    --presets multi --thresholds 0.01 --seeds "$seed" \
                    --config-glob "$object_name/evaluation/$sequence.yml" \
                    --output "$results_root/manifests/${object_name}_${sequence}_${run_tag}.json" \
                    --results-root "$results_root" --execute; then
                    succeeded=1
                    break
                fi
            done
            if [ "$succeeded" -ne 1 ]; then
                echo "$object_name/$sequence $run_tag" | tee -a "$failed_cases"
            fi
        done
    done
done

"$python_bin" scripts/aggregate_results.py "$results_root" \
    --run-glob '**/metrics.json' --output-dir "$results_root/summary"

test ! -s "$failed_cases"
