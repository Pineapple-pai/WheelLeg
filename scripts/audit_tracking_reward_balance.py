"""Compare v3/v4 reward contributions on identical frozen-policy trajectories.

This is an offline coefficient audit, not a claim that the v4 policy is trained.
Fixed commands start at reset so v2/v3 reference origins coincide for this test.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from train_uz05 import PPO, require_compatible_contract
import numpy as np
from uz05.env import UZ05Env
from uz05.spec import EnvParams, STAGE_BY_NAME, precision_rewards_for_stage

TASK_FIELDS = {'upright': 'tracking_upright_weight', 'track_vx': 'tracking_vx_weight',
               'track_yaw': 'tracking_yaw_weight', 'track_height': 'tracking_height_weight',
               'termination': 'tracking_termination_weight'}
CONTROL = ('action_rate', 'wheel_target_change', 'joint_target_change', 'effort', 'effort_change', 'saturation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoints', nargs='+', type=Path)
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--seed', type=int, default=20000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    params = EnvParams()
    params.domain_randomization.enabled = False
    old, new = STAGE_BY_NAME['joint_tracking_v3'], STAGE_BY_NAME['joint_tracking_v4']
    a, b = (asdict(precision_rewards_for_stage(stage, params)) for stage in (old, new))
    ratios = {key: b[key]/a[key] for key in b if key in a and a[key] != 0}
    ratios.update({key: getattr(new, field)/getattr(old, field) for key, field in TASK_FIELDS.items()})
    env = UZ05Env('joint_tracking_v3', params=params)
    rows = []
    for checkpoint in args.checkpoints:
        checkpoint = checkpoint.with_suffix('')
        contract = json.loads(Path(f'{checkpoint}_contract.json').read_text())
        require_compatible_contract(checkpoint, contract['stage'])
        model = PPO.load(checkpoint, device='cpu')
        for case, vx, yaw in (('stand', 0, 0), ('forward', .3, 0), ('yaw', 0, 1)):
            env.reset(seed=args.seed)
            env.command[:] = [vx, 0, yaw, .27, 0]
            env.command_target[:] = env.command
            env.command_steps_left = args.steps + 1
            obs = env._obs()
            samples = []
            reason = 'running'
            for _ in range(min(args.steps, env.stage.episode_steps)):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, truncated, info = env.step(action)
                samples.append(info['reward_terms'])
                reason = info['termination_reason']
                if done or truncated:
                    break
            mean = {key: float(np.mean([t[key] for t in samples])) for key in samples[0]}
            scaled = {key: value*ratios.get(key, 1) for key, value in mean.items()}
            comparison = {}
            for label, values in (('v3', mean), ('v4', scaled)):
                task = sum(values[key] for key in TASK_FIELDS if key != 'termination')
                control = sum(values[key] for key in CONTROL)
                comparison[label] = {'task_positive_mean': task, 'control_cost_mean': control,
                                     'hold_cost_mean': values['hold_position']+values['hold_heading'],
                                     'control_to_task_ratio': abs(control)/max(task, 1e-9),
                                     'total_mean': sum(values.values()), 'terms': values}
            row = {'checkpoint': str(checkpoint), 'case': case, 'seed': args.seed,
                   'steps': len(samples), 'reason': reason, 'comparison': comparison}
            rows.append(row)
            print(json.dumps(row), flush=True)
    env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'kind': 'fixed_trajectory_reward_audit', 'results': rows}, indent=2)+'\n')


if __name__ == '__main__':
    main()
