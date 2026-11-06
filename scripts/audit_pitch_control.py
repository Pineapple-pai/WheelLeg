"""Audit v6 reward scale using a frozen v5 policy; never train or supply actions to PPO."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from train_uz05 import PPO
from uz05.env import UZ05Env
from uz05.spec import env_params_for_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='checkpoints/joint_tracking_pos_v5/checkpoint')
    parser.add_argument('--output', type=Path, default=Path('logs/joint_tracking_v6_reward_audit.json'))
    args = parser.parse_args()
    model = PPO.load(args.checkpoint, device='cpu')
    results = []
    for case, vx in [('stand', 0), ('forward', .3), ('reverse', -.3)]:
        params = env_params_for_stage('joint_tracking_v6')
        params.domain_randomization.enabled = False
        params.robust.pushes_enabled = False
        env = UZ05Env('joint_tracking_v6', params=params)
        env.reset(seed=20000)
        env.command[:] = [vx, 0, 0, .27, 0]
        env.command_target[:] = env.command
        env.command_steps_left = 3001
        obs = env._obs()
        rows = []
        for step in range(1250):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if step >= 125:
                t = info['reward_terms']
                task = sum(t[k] for k in ('upright','pitch_alignment','track_vx','track_yaw','track_height'))
                control = sum(t[k] for k in ('action_rate','wheel_target_change','joint_target_change','wheel_acceleration','effort','effort_change','saturation'))
                rows.append([task,control,t['pitch_alignment']+t['pitch_error'],info['pitch'],info['wheel_target_change_sq'],info['joint_target_change_sq'],info['wheel_acceleration_sq']])
            if terminated or truncated:
                break
        x = np.array(rows)
        results.append({'case':case,'steps':step+1,'reason':info['termination_reason'],
                        'task_mean':float(x[:,0].mean()) if len(x) else None,
                        'control_mean':float(x[:,1].mean()) if len(x) else None,
                        'control_to_task_ratio':float(abs(x[:,1].mean())/max(x[:,0].mean(),1e-9)) if len(x) else None,
                        'net_pitch_term_mean':float(x[:,2].mean()) if len(x) else None,
                        'pitch_mean_deg':float(np.degrees(x[:,3].mean())) if len(x) else None,
                        'wheel_target_change_rms':float(np.sqrt(x[:,4].mean())) if len(x) else None,
                        'joint_target_change_rms':float(np.sqrt(x[:,5].mean())) if len(x) else None,
                        'wheel_acceleration_rms_rad_s2':float(np.sqrt(x[:,6].mean())) if len(x) else None})
        env.close()
    env = UZ05Env('joint_tracking_v6')
    angles = []
    for angle in (0,3,5,8,10,12,20):
        terms = env._pitch_terms(0,np.radians(angle))
        angles.append({'pitch_deg':angle,**terms,'net_pitch_term':terms['pitch_alignment']+terms['pitch_error']})
    env.close()
    report = {'policy_checkpoint':args.checkpoint,'evaluation_stage':'joint_tracking_v6',
              'note':'Frozen v5 policy under v6 noise/rewards; scale audit only, not v6 learned performance.',
              'angle_sweep':angles,'results':results}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
