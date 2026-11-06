"""Plan or sequentially execute the frozen 2x2 reward-simplification experiment.

Default is plan-only. --execute --smoke runs two small updates per arm and one
evaluation profile; it is an integration check, not an experiment result.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from uz05.spec import STAGE_BY_NAME, env_params_for_stage, precision_rewards_for_stage

ROOT = Path(__file__).resolve().parents[1]
ARMS = {'legacy': 'joint_tracking_v15', 'minimal': 'joint_tracking_simple',
        'minimal_axle': 'joint_tracking_simple_axle',
        'support': 'joint_tracking_simple_support',
        'support_axle': 'joint_tracking_simple_support_axle'}
EXTRA_ARMS = {'tight_axle':'joint_tracking_simple_tight_axle',
              'tight_survival':'joint_tracking_simple_tight_survival',
              'responsive':'joint_tracking_simple_responsive',
              'responsive_survival':'joint_tracking_simple_responsive_survival'}
PROFILES = ['reverse10','reverse10_backfirst','stop_go10','stop_go10_backfirst',
            'motion','height2_forward','height2_reverse','turn']


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_plan(args):
    pilot = getattr(args,'pilot',False)
    folder = ROOT/'logs'/args.name
    checkpoint = None if args.from_scratch else Path(args.checkpoint).expanduser().resolve().with_suffix('')
    source_hashes = {}
    if checkpoint is not None:
        for suffix in ('.zip','_contract.json','_vecnormalize.pkl'):
            path = Path(f'{checkpoint}{suffix}')
            source_hashes[str(path)] = file_hash(path)
    runs = []
    for seed in args.seeds:
        for arm in args.arms:
            stage = {**ARMS,**EXTRA_ARMS}[arm]
            version = f'{args.name}_{arm}_seed{seed}'
            output = ROOT/'checkpoints'/version/'checkpoint'
            command = ['bash',str(ROOT/'scripts/train_joint_quiet_v23.sh'),
                '--stage',stage,'--version',version,'--seed',str(seed),
                '--updates',str(2 if args.smoke else args.updates),
                '--num-envs',str(args.num_envs),'--learning-rate','0.00003','--reset-vecnormalize']
            if checkpoint is not None:
                command += ['--checkpoint',str(checkpoint),'--preserve-action-std']
            if args.smoke:
                command += ['--num-envs','2','--rollout-steps','64','--batch-size','64']
            conda = ['conda','run','--no-capture-output','-n','sim','python']
            transition = conda+[str(ROOT/'scripts/eval_joint_tracking_transitions.py'),str(output),
                '--profiles',*(['reverse10'] if args.smoke else
                    ['reverse10','reverse10_backfirst','motion','stop2_forward','stop2_reverse','height2_reverse'] if pilot else PROFILES),
                '--episodes',str(1 if args.smoke or pilot else 5),'--seed','31000' if pilot else '30000',
                '--output',str(folder/f'{version}_transitions.json')]
            commands = [{'kind':'train','argv':command,'log':str(folder/f'{version}_train.log')},
                        {'kind':'transitions','argv':transition,'log':str(folder/f'{version}_transitions.log')}]
            if not args.smoke and not pilot:
                commands.append({'kind':'robust_transitions',
                    'argv':transition[:-2]+['--domain-randomization','--perturbations',
                        '--output',str(folder/f'{version}_robust.json')],
                    'log':str(folder/f'{version}_robust.log')})
            if not args.smoke:
                for speed in ((2,) if pilot else (1,2)):
                    commands.append({'kind':f'held_speed{speed}', 'argv':conda+[
                        str(ROOT/'scripts/eval_joint_tracking.py'),str(output),
                        '--episodes-per-case','1' if pilot else '5','--seed','31000' if pilot else '30000','--duration','24',
                        '--heights',*(['.27'] if pilot else ['.22','.27','.32']),'--speed',str(speed),
                        '--cases',*(['stand','forward','reverse'] if pilot else ['stand','forward','reverse','yaw_left','yaw_right']),
                        '--output',str(folder/f'{version}_held{speed}.json')],
                        'log':str(folder/f'{version}_held{speed}.log')})
            params = env_params_for_stage(stage)
            runs.append({'arm':arm,'seed':seed,'stage':stage,'version':version,
                         'stage_config':asdict(STAGE_BY_NAME[stage]),
                         'effective_reward':asdict(precision_rewards_for_stage(STAGE_BY_NAME[stage],params)),
                         'commands':commands})
    return {'schema':'reward_simplification_experiment_v1','smoke_only':args.smoke,'pilot_only':pilot,
            'initialization':'from_scratch' if args.from_scratch else 'matched_checkpoint_warm_start',
            'source_checkpoint_hashes':source_hashes,
            'source_code_hashes':{str(path.relative_to(ROOT)):file_hash(path) for path in (
                ROOT/'scripts/uz05/spec.py',ROOT/'scripts/uz05/env.py',ROOT/'scripts/uz05/simple_rewards.py',
                ROOT/'scripts/train_uz05.py',ROOT/'scripts/train_joint_quiet_v23.sh',
                ROOT/'scripts/eval_joint_tracking_transitions.py',ROOT/'scripts/eval_joint_tracking.py')},
            'runs':runs,'comparison_rule':'Use physical metrics and completion first; do not compare training rewards across definitions.'}


def summarize_transition(report):
    results = report['results']
    segments = [s for r in results for s in r['segments']]
    expected = sum(len(report['profiles'][r['profile']]) for r in results)
    def mean(values):
        values = [v for v in values if v is not None]
        return sum(values)/len(values) if values else None
    reversals = [s for s in segments if s['initial_vx_m_s']*s['target_vx_yaw_height'][0]<0
                 and abs(s['initial_vx_m_s'])>.1 and abs(s['target_vx_yaw_height'][0])>.1]
    return {'episodes':len(results),'completed_episodes':sum(r['reason']=='time_limit' for r in results),
            'planned_segments':expected,'observed_segments':len(segments),
            'reversal_segments_observed':len(reversals),
            'reversals_with_t90':sum(s['vx_response_t90_s'] is not None for s in reversals),
            'observed_reversal_t90_mean_s':mean(s['vx_response_t90_s'] for s in reversals),
            'observed_segment_vx_mae_mean_m_s':mean(s['vx_mae_m_s'] for s in segments),
            'observed_segment_axle_peak_mean_m':mean(s['axle_line_offset_peak_m'] for s in segments),
            'observed_segment_support_pitch_rms_mean_deg':mean(s['support_pitch_error_rms_deg'] for s in segments),
            'note':'Conditional on observed segments; missing/unsettled segments are not successes. Read per-profile raw reports.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--pilot',action='store_true',help='small exploratory screening; not a formal multi-seed comparison')
    p.add_argument('--arms',nargs='+',choices=list(ARMS)+list(EXTRA_ARMS),default=list(ARMS))
    p.add_argument('--num-envs',type=int,default=8)
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    p.add_argument('--updates',type=int,default=None)
    p.add_argument('--checkpoint',default=str(ROOT/'checkpoints/joint_tracking_pos_v15/checkpoint'))
    p.add_argument('--from-scratch',action='store_true')
    p.add_argument('--name',default='reward_simplification_v1')
    args = p.parse_args()
    if args.smoke and args.pilot:p.error('--smoke and --pilot are mutually exclusive')
    if args.updates is None:args.updates=50 if args.pilot else 4000
    if args.updates<1 or args.num_envs<1 or any(seed<0 for seed in args.seeds):
        p.error('updates must be positive and seeds nonnegative')
    if Path(args.name).name != args.name or args.name in ('.','..'):
        p.error('name must be a single directory name')
    if len(set(args.arms))!=len(args.arms) or len(set(args.seeds))!=len(args.seeds):
        p.error('duplicate arms/seeds are not allowed')
    if args.smoke:
        args.name += '_smoke'
    if args.pilot:
        args.name += '_pilot'
    folder = ROOT/'logs'/args.name
    manifest = folder/'manifest.json'
    try:
        # dataclass tuples become JSON arrays; compare the serialized representation.
        plan = json.loads(json.dumps(build_plan(args)))
    except FileNotFoundError as error:
        p.error(str(error))
    if manifest.exists() and json.loads(manifest.read_text()) != plan:
        p.error(f'existing plan differs: {manifest}; choose a new --name')
    if args.execute:
        for run in plan['runs']:
            if (ROOT/'checkpoints'/run['version']).exists():
                p.error(f'checkpoint directory already exists for {run["version"]}')
    folder.mkdir(parents=True,exist_ok=True)
    manifest.write_text(json.dumps(plan,indent=2,ensure_ascii=False)+'\n')
    print(f'Plan: {manifest} ({len(plan["runs"])} runs; execute={args.execute}; smoke={args.smoke})',flush=True)
    if not args.execute:
        return
    summary = []
    for run in plan['runs']:
        for task in run['commands']:
            print(f'{run["version"]}: {task["kind"]}',flush=True)
            with open(task['log'],'w') as log:
                subprocess.run(task['argv'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        report_path = folder/f'{run["version"]}_transitions.json'
        summary.append({'arm':run['arm'],'seed':run['seed'],
                        **summarize_transition(json.loads(report_path.read_text()))})
        (folder/'summary.json').write_text(json.dumps({'smoke_only':args.smoke,'pilot_only':args.pilot,'runs':summary},indent=2)+'\n')


if __name__ == '__main__':
    main()
