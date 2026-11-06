"""Compare deterministic and sampled PPO control under held-out random command episodes."""
from __future__ import annotations
import argparse
from dataclasses import replace
from collections import Counter
import json
from pathlib import Path

from train_uz05 import PPO, require_compatible_contract
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from uz05.env import UZ05Env
from uz05.spec import EnvParams, env_params_for_stage


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--nominal', action='store_true', help='disable randomized dynamics and scheduled pushes')
    p.add_argument('--moving-starts', action='store_true', help='include the training mixture of moving initial states')
    p.add_argument('--episodes', type=int, default=10)
    p.add_argument('--seed', type=int, default=30000)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.episodes < 1:
        p.error('--episodes must be positive')
    checkpoint = args.checkpoint.with_suffix('')
    contract = json.loads(Path(f'{checkpoint}_contract.json').read_text())
    stage = contract['stage']
    require_compatible_contract(checkpoint, stage)
    params = env_params_for_stage(stage)
    params.domain_randomization.enabled = contract['domain_randomization'] and not args.nominal
    if args.nominal:
        params.robust.pushes_enabled = False
    raw = UZ05Env(stage, params=params)
    if not args.moving_starts:
        raw.stage = replace(raw.stage,moving_start_probability=0.0)
    normalized = VecNormalize.load(f'{checkpoint}_vecnormalize.pkl', DummyVecEnv([lambda: raw]))
    normalized.training = False
    model = PPO.load(checkpoint, device='cpu')
    output = {'checkpoint': str(checkpoint), 'seed_base': args.seed, 'nominal': args.nominal,
              'moving_start_probability':raw.stage.moving_start_probability,'results': []}
    for deterministic in (True, False):
        errors, targets, torque_sat, geometry, episodes, terms = [], [], [], [], [], []
        delta_squares = []
        hold_samples, output_samples, support_samples = [], [], []
        for index in range(args.episodes):
            seed = args.seed + index
            torch.manual_seed(seed)
            obs, _ = raw.reset(seed=seed)
            previous = None
            total_reward = 0.0
            for step in range(raw.stage.episode_steps):
                action, _ = model.predict(normalized.normalize_obs(obs), deterministic=deterministic)
                obs, reward, terminated, truncated, info = raw.step(action)
                support_samples.append([info["leg_drop_difference_m"], info["leg_differential_velocity_m_s"], info["wheel_dual_support"], info["push_force_n"]])
                output_samples.append([info["joint_target_change_sq"], info["wheel_saturation_fraction"],
                                       info["effort_change_sq"]])
                if info["zero_command"]:
                    hold_samples.append([info["hold_planar_error_m"], info["hold_heading_error_rad"]])
                errors.append([info['tracking_vx_error'], info['tracking_yaw_error'], info['tracking_height_error']])
                target = np.array([info['wheel_target_left'], info['wheel_target_right']])
                targets.append(target)
                if previous is not None:
                    delta_squares.append(float(np.square(target-previous).mean()))
                previous = target
                torque_sat.append(info['joint_saturation_per_joint'])
                geometry.append([info['wheel_stagger_m'], info['wheel_offset_max_m']])
                terms.append(info['reward_terms'])
                total_reward += reward
                if terminated or truncated:
                    episodes.append({'seed': seed, 'steps': step+1, 'reason': info['termination_reason'], 'reward': total_reward})
                    break
        errors = np.asarray(errors)
        geometry = np.asarray(geometry)
        result = {
            'leg_drop_difference_rms_m': float(np.sqrt(np.mean(np.asarray(support_samples)[:,0]**2))),
            'leg_differential_velocity_rms_m_s': float(np.sqrt(np.mean(np.asarray(support_samples)[:,1]**2))),
            'dual_wheel_support_fraction': float(np.mean(np.asarray(support_samples)[:,2])),
            'push_active_fraction': float(np.mean(np.asarray(support_samples)[:,3] > 0)),
            'zero_command_peak_planar_drift_m': float(np.max(np.asarray(hold_samples)[:,0])) if hold_samples else None,
            'zero_command_peak_heading_error_rad': float(np.max(np.asarray(hold_samples)[:,1])) if hold_samples else None,
            'zero_command_within_5cm_fraction': float(np.mean(np.asarray(hold_samples)[:,0] <= .05)) if hold_samples else None,
            'joint_target_change_rms_rad': float(np.sqrt(np.mean(np.asarray(output_samples)[:,0]))),
            'wheel_saturation_fraction': float(np.mean(np.asarray(output_samples)[:,1])),
            'effort_change_rms': float(np.sqrt(np.mean(np.asarray(output_samples)[:,2]))),
            'deterministic': deterministic, 'episodes': episodes, 'samples': len(errors),
            'survival_rate': float(np.mean([e['reason']=='time_limit' for e in episodes])),
            'episode_length_mean': float(np.mean([e['steps'] for e in episodes])),
            'reward_mean': float(np.mean([e['reward'] for e in episodes])),
            'mae_vx_yaw_height': np.abs(errors).mean(axis=0).tolist(),
            'rmse_vx_yaw_height': np.sqrt(np.square(errors).mean(axis=0)).tolist(),
            'wheel_target_delta_rms_rad_s': float(np.sqrt(np.mean(delta_squares))),
            'wheel_target_abs_mean_rad_s': np.abs(targets).mean(axis=0).tolist(),
            'joint_torque_saturation_fraction': np.mean(torque_sat, axis=0).tolist(),
            'stance_valid_fraction': float(np.mean((geometry[:,0] <= raw.stage.wheel_stagger_limit_m)
                                                  & (geometry[:,1] <= raw.stage.wheel_offset_limit_m))),
            'wheel_stagger_max_m': float(geometry[:,0].max()),
            'reward_terms_per_step': {k: float(np.mean([t[k] for t in terms])) for k in terms[0]},
            'termination_counts': dict(Counter(e['reason'] for e in episodes)),
        }
        output['results'].append(result)
        print(json.dumps(result), flush=True)
    normalized.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2)+'\n')


if __name__ == '__main__':
    main()
