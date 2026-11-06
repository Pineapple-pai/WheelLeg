"""Frozen conservative continuation. Run from the repository with wheelleg-train."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'diagnostics/ppo_low_band_refinement'
OUT.mkdir(parents=True, exist_ok=True)
SOURCE = ROOT / 'checkpoints/joint_tracking_pos_v16_model20261104/checkpoint'
PYTHON = sys.executable

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

protected = [Path(str(SOURCE) + s) for s in ('.zip', '_vecnormalize.pkl', '_contract.json')]
protected += [ROOT / p for p in ('models/uz05_20261104/uz05_rl.xml', 'scripts/uz05/env.py',
    'scripts/uz05/actuators.py', 'scripts/uz05/spec.py', 'scripts/uz05/command_sampling.py', 'scripts/train_uz05.py')]
runs = []
for seed in (71, 72, 73):
    folder = OUT / f'seed_{seed}'
    version = f'low_band_lr3e5_300_s{seed}_20261105'
    cp = ROOT / 'checkpoints' / version / 'checkpoint'
    cmd = [PYTHON, '-u', str(ROOT / 'scripts/train_uz05.py'), '--stage', 'joint_tracking_v16',
        '--checkpoint', str(SOURCE), '--version', version, '--updates', '300',
        '--rollout-steps', '256', '--batch-size', '256', '--num-envs', '4', '--vec-env', 'subproc',
        '--device', 'cpu', '--learning-rate', '0.00003', '--disable-adaptive-lr',
        '--entropy-coef', '0.001', '--preserve-action-std', '--seed', str(seed),
        '--command-sampling', 'low_band', '--save-interval', '100', '--log-interval', '25',
        '--initial-state-audit', str(folder / 'initial_state.json'),
        '--tensorboard-log', str(folder / 'tensorboard')]
    evaluations = [[PYTHON, str(ROOT / 'scripts/eval_new_model_training.py'), str(cp) + f'_iter_{step}',
        '--output', str(folder / f'evaluation_{step}'), '--launch-speeds', '.25', '.5',
        '--seeds', '64000', '64001', '64002'] for step in (100, 300)]
    runs.append(dict(seed=seed, folder=str(folder), checkpoint=str(cp), command=cmd, evaluations=evaluations))
plan = dict(source=str(SOURCE), protected={str(p): digest(p) for p in protected}, runs=runs,
    purpose='At100 updates compare only LR3e-5 vs existing paired low_band LR1e-4; at300 assess duration separately.',
    gate='All4low tasks combined3/3 in >=2training seeds, stand3/3 in all3; no automatic speed expansion.',
    transitions=3*300*4*256)
plan_path = OUT / 'experiment_config.json'
if plan_path.exists():
    assert json.loads(plan_path.read_text()) == plan, 'Frozen configuration changed'
else:
    plan_path.write_text(json.dumps(plan, indent=2) + '\n')
(OUT / 'commands.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\n' + '\n'.join(
    shlex.join(cmd) for r in runs for cmd in [r['command'], *r['evaluations']]) + '\n')

def execute(run):
    folder = Path(run['folder']); folder.mkdir(exist_ok=True)
    marker = folder / 'training_completed.json'
    if marker.exists():
        assert digest(run['checkpoint'] + '.zip') == json.loads(marker.read_text())['sha256']
    else:
        assert not Path(run['checkpoint'] + '.zip').exists(), 'Refuse to overwrite'
        with (folder / 'train.log').open('w') as log:
            subprocess.run(run['command'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        marker.write_text(json.dumps(dict(sha256=digest(run['checkpoint'] + '.zip'))) + '\n')
    for step, cmd in zip((100, 300), run['evaluations']):
        if not (folder / f'evaluation_{step}/results.json').exists():
            with (folder / f'evaluation_{step}.log').open('w') as log:
                subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    print('Completed seed', run['seed'], flush=True)

if '--execute' in sys.argv:
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(execute, run) for run in runs]
        for future in as_completed(futures): future.result()
    assert all(digest(p) == h for p, h in plan['protected'].items())
    (OUT / 'integrity.json').write_text(json.dumps(dict(protected_unchanged=True)) + '\n')
else:
    print('Plan frozen; add --execute to run.')
