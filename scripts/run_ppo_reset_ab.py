"""Freeze and execute the paired v16 reset-only continuation experiment."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'diagnostics/ppo_reset_ab'
SOURCE = ROOT/'checkpoints/joint_tracking_pos_v16/checkpoint'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    if (OUT/'config_comparison.json').exists():
        raise SystemExit('Experiment already exists; refusing to overwrite frozen configuration.')
    OUT.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in (1, 2, 3):
        modes = ('legacy', 'projected') if seed % 2 else ('projected', 'legacy')
        for mode in modes:
            name = f'ppo_reset_ab_20261104_s{seed}_{mode}'
            checkpoint = ROOT/'checkpoints'/name/'checkpoint'
            if checkpoint.parent.exists():
                raise SystemExit(f'Output already exists: {checkpoint.parent}')
            run_dir = OUT/f'seed_{seed}'/mode
            run_dir.mkdir(parents=True)
            args = [sys.executable, '-u', str(ROOT/'scripts/train_uz05.py'),
                    '--stage', 'joint_tracking_v16', '--checkpoint', str(SOURCE),
                    '--version', name, '--updates', '200', '--rollout-steps', '512',
                    '--batch-size', '512', '--num-envs', '8', '--vec-env', 'subproc',
                    '--device', 'cpu', '--learning-rate', '0.00003', '--disable-adaptive-lr',
                    '--entropy-coef', '0.001', '--preserve-action-std', '--seed', str(seed),
                    '--vecnormalize-path', str(SOURCE)+'_vecnormalize.pkl',
                    '--save-interval', '50', '--stand-level', '0',
                    '--tensorboard-log', str(run_dir/'tensorboard'),
                    '--initial-state-audit', str(run_dir/'initial_state.json'),
                    '--reset-audit-dir', str(run_dir/'resets')]
            if mode == 'legacy': args.append('--legacy-reset')
            runs.append(dict(seed=seed, mode=mode, checkpoint=str(checkpoint),
                             directory=str(run_dir), command=args))
    files = [SOURCE.with_suffix('.zip'), Path(str(SOURCE)+'_vecnormalize.pkl'),
             Path(str(SOURCE)+'_contract.json'), SOURCE.parent/'training_config.json',
             ROOT/'uz05_修正后(3).xml']
    files += [ROOT/p for p in ('scripts/train_uz05.py', 'scripts/uz05/env.py',
               'scripts/uz05/model.py', 'scripts/uz05/spec.py', 'scripts/uz05/actuators.py',
               'scripts/uz05/reset_projection.py', 'scripts/uz05/experiment_audit.py',
               'scripts/uz05/simple_rewards.py', 'scripts/run_ppo_reset_ab.py')]
    config = {
        'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'stage': 'joint_tracking_v16', 'paired_training_seeds': [1, 2, 3],
        'updates_per_run': 200, 'steps_per_update': 4096, 'steps_per_run': 819200,
        'total_training_steps': 4915200,
        'source_checkpoint': str(SOURCE), 'source_config': json.loads((SOURCE.parent/'training_config.json').read_text()),
        'only_treatment': 'RobotParams.closed_chain_reset (includes bounded contact rejection/resampling)',
        'unchanged': ['v16 reward', 'action semantics', 'PPO/network hyperparameters',
                      '8 environments', 'randomization', 'pushes', 'full fixed command sampling'],
        'normalization': 'Load same saved reward statistics, norm_obs=False; no reset-vecnormalize.',
        'resume_limitations': ['No saved environment/episode/RNG/rollout state',
                              'Local timestep/progress restarts on learn; constant schedules unchanged',
                              'PPO epoch counter and Adam moments retained and audited before learning'],
        'evaluation': {
            'reset': 'projected for every policy', 'duration_s': 24,
            'seeds': [51000, 51001, 51002, 51003, 51004],
            'primary_tasks': ['stand', 'forward', 'reverse', 'motion', 'stop2_reverse'],
            'additional_tasks': ['reverse10', 'height2_reverse'],
            'checkpoints': [0, 50, 100, 150, 200],
            'intermediate_tasks': ['stand'],
            'height_m': .27, 'held_speed_m_s': 1.,
            'domain_randomization': False, 'pushes': False, 'moving_starts': False,
            'observation_noise': 'existing v16 unchanged',
            'success': 'Existing evaluator: termination_reason == time_limit at 24 s; not a tracking-quality guarantee.',
            'tracking': 'Report all-episode RMS/MAE including failures; separate post-1s metrics and survival.',
            'scope': 'All commanded speeds/heights within v16 training ranges; exact fixed schedules are held-out protocols.',
            'common_swing_band_m': .08,
            'common_swing_band_basis': 'existing support_response/wheel_fore_aft band, descriptive only',
        },
        'decision_rule': {
            'A': 'All 3 seed primary-task mean survival differences >= +0.10, no task median difference <= -0.10; common-offset RMS improves >=10% for all 3 seeds.',
            'C': 'All 3 seed primary-task mean survival differences <= -0.10, or stand/stop2_reverse survival median difference <= -0.20 with all 3 differences negative.',
            'B': 'Otherwise: no stable practical superiority/retreat; retain physical consistency and investigate control/reward. Small n=3 does not establish statistical equivalence.',
        },
        'hashes': {str(p.relative_to(ROOT)): sha(p) for p in files}, 'runs': runs,
    }
    (OUT/'config_comparison.json').write_text(json.dumps(config, indent=2, ensure_ascii=False)+'\n')
    (OUT/'commands.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\n'+
                                  '\n'.join(shlex.join(r['command']) for r in runs)+'\n')
    return config


def execute(config):
    for relative, expected in config['hashes'].items():
        if sha(ROOT/relative) != expected:
            raise SystemExit(f'Frozen source changed: {relative}')
    environment = os.environ.copy()
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'TORCH_NUM_THREADS'):
        environment[key] = '1'
    for run in config['runs']:
        directory = Path(run['directory'])
        if (directory/'completed.json').exists():
            print('already completed:', run['seed'], run['mode'], flush=True); continue
        if Path(run['checkpoint']).parent.exists():
            raise SystemExit(f'Partial run exists, refusing automatic overwrite: {run["checkpoint"]}')
        started = time.time()
        (OUT/'status.json').write_text(json.dumps({'active': run, 'started': started}, indent=2)+'\n')
        print('start:', run['seed'], run['mode'], flush=True)
        with (directory/'train.log').open('w') as log:
            result = subprocess.run(run['command'], cwd=ROOT, env=environment,
                                    stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise SystemExit(f'Training failed ({result.returncode}): {directory}/train.log')
        (directory/'completed.json').write_text(json.dumps({'seconds': time.time()-started,
            'checkpoint_sha256': sha(Path(run['checkpoint']).with_suffix('.zip'))}, indent=2)+'\n')
        print('completed:', run['seed'], run['mode'], flush=True)
    (OUT/'status.json').write_text(json.dumps({'training_complete': True}, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Skip completed runs; never overwrite partial runs')
    args = parser.parse_args()
    config = json.loads((OUT/'config_comparison.json').read_text()) if args.resume else prepare()
    if args.execute: execute(config)
