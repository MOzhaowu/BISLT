#!/usr/bin/env python3
"""Create a reproducible U-BIT gate ablation matrix and optionally execute it."""

import argparse
import itertools
import json
import os
import subprocess
from pathlib import Path


PRESETS = {
    'original': {
        'consume_once': False,
        'model_validation': {'enabled': False},
    },
    'sync': {
        'consume_once': True,
        'model_validation': {'enabled': False},
    },
    'iou': {
        'consume_once': True,
        'model_validation': {
            'enabled': True,
            'weights': {'iou': 1.0, 'pose_consistency': 0.0,
                        'temporal_stability': 0.0, 'uncertainty': 0.0},
        },
    },
    'multi': {
        'consume_once': True,
        'model_validation': {
            'enabled': True,
            'weights': {'iou': 1.0, 'pose_consistency': 0.25,
                        'temporal_stability': 0.25, 'uncertainty': 0.25},
        },
    },
}


def cases(args):
    for preset, threshold, seed in itertools.product(
            args.presets, args.thresholds, args.seeds):
        preset_config = PRESETS[preset]
        config = dict(preset_config['model_validation'])
        config['min_improvement'] = threshold
        yield {
            'name': f'{preset}_t{threshold:g}_s{seed}',
            'preset': preset,
            'seed': seed,
            'consume_once': preset_config['consume_once'],
            'model_validation': config,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--presets', nargs='+', choices=PRESETS, default=list(PRESETS))
    parser.add_argument('--thresholds', nargs='+', type=float, default=[0.0, 0.0025, 0.005, 0.01])
    parser.add_argument('--seeds', nargs='+', type=int, default=[41, 42, 43])
    parser.add_argument('--output', type=Path, default=Path('experiments/gate_sweep.json'))
    parser.add_argument(
        '--config-glob', default='cheezit/evaluation/04.yml',
        help='Config path glob below config/moped passed to the batch runner.',
    )
    parser.add_argument('--execute', action='store_true')
    parser.add_argument(
        '--results-root', type=Path, default=Path('../baseline/moped/gate_sweep'),
    )
    args = parser.parse_args()

    matrix = list(cases(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'schema_version': 1, 'cases': matrix}, indent=2) + '\n')
    print(f'wrote {len(matrix)} cases to {args.output}')
    if not args.execute:
        return
    for case in matrix:
        env = os.environ.copy()
        env['BIT_MODEL_VALIDATION_JSON'] = json.dumps(case['model_validation'])
        env['BIT_CONSUME_ONCE'] = '1' if case['consume_once'] else '0'
        env['BIT_RANDOM_SEED'] = str(case['seed'])
        env['BIT_RUN_TAG'] = case['name']
        env['BIT_CONFIG_GLOB'] = args.config_glob
        env['BIT_BASELINE_ROOT'] = str(args.results_root.resolve())
        env['BIT_METRICS_GLOB'] = '**/metrics.json'
        subprocess.run(['bash', 'scripts/run_moped_baseline.sh'], env=env, check=True)
    subprocess.run([
        os.environ.get('BIT_PYTHON', 'python3'), 'scripts/aggregate_results.py',
        str(args.results_root), '--run-glob', '**/metrics.json',
        '--output-dir', str(args.results_root / 'summary'),
    ], check=True)


if __name__ == '__main__':
    main()
