"""Fixed projected-reset, deterministic evaluation and joint motion recording."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from eval_joint_tracking_transitions import PROFILES
from train_uz05 import require_compatible_contract
from uz05.env import UZ05Env
from uz05.spec import env_params_for_stage

SCHEDULES = {
    'stand': [(0, 0, 0, .27)], 'forward': [(0, 1, 0, .27)],
    'reverse': [(0, -1, 0, .27)],
    **{k: PROFILES[k] for k in ('motion', 'stop2_reverse', 'reverse10', 'height2_reverse')},
}
COLUMNS = ['time_s', 'pitch_rad', 'roll_rad', 'pitch_rate_rad_s', 'vx_m_s', 'command_vx_m_s',
           'height_error_m', 'axle_error_m', 'wheel_x_left_m', 'wheel_x_right_m',
           'common_x_m', 'differential_x_m', 'joint_saturation', 'wheel_saturation',
           'joint_target_delta_sq', 'hold_distance_m', 'joint_target_step_rate_rms_rad_s',
           *[f'actual_q_{i}' for i in range(4)], *[f'target_q_{i}' for i in range(4)],
           *[f'actual_qd_{i}' for i in range(4)],
           *[f'wheel_target_{i}' for i in range(2)], *[f'wheel_actual_{i}' for i in range(2)],
           *[f'wheel_torque_nm_{i}' for i in range(2)],
           *[f'joint_saturation_{i}' for i in range(4)],
           *[f'wheel_saturation_{i}' for i in range(2)]]


def wrap(x):
    return np.arctan2(np.sin(x), np.cos(x))


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def longest_full_saturation(samples, dt=.008):
    longest = run = 0
    for value in samples:
        run = run+1 if value >= 1.-1e-9 else 0
        longest = max(longest, run)
    return longest*dt


def lag_correlation(a, b, maxlag=62, dt=.008):
    values = []
    for lag in range(-min(maxlag, len(a)//4), min(maxlag, len(a)//4)+1):
        x, y = (a[-lag:], b[:lag]) if lag < 0 else ((a[:-lag], b[lag:]) if lag else (a, b))
        if len(x) >= 10 and np.std(x) > 1e-9 and np.std(y) > 1e-9:
            values.append((float(np.corrcoef(x, y)[0, 1]), lag*dt))
    if not values: return None, None
    return max(values, key=lambda p: abs(p[0]))


def episode_metrics(a, reason):
    v = lambda key: a[:, COLUMNS.index(key)]
    targets = np.column_stack([v(f'target_q_{i}') for i in range(4)])
    actual = np.column_stack([v(f'actual_q_{i}') for i in range(4)])
    target_delta = np.diff(targets, axis=0)
    circular_delta = wrap(target_delta)
    lag_error = wrap(targets - actual)
    settled = a[a[:, 0] > 1.]
    if len(settled) == 0: settled = a
    common = v('common_x_m')
    corr, lag = lag_correlation(common, v('pitch_rad'))
    acceleration = np.gradient(v('vx_m_s'), .008)
    corr_acc, lag_acc = lag_correlation(common, acceleration)
    stop = a[(a[:, 0] > 10.) & (np.abs(v('command_vx_m_s')) < .01)]
    return {
        'survival_s': float(a[-1, 0]), 'success': int(reason == 'time_limit'), 'termination_reason': reason,
        'pitch_rms_deg': float(np.degrees(rms(v('pitch_rad')))),
        'roll_rms_deg': float(np.degrees(rms(v('roll_rad')))),
        'height_mae_m': float(np.mean(np.abs(v('height_error_m')))),
        'height_rmse_m': rms(v('height_error_m')), 'axle_mean_m': float(v('axle_error_m').mean()),
        'axle_p95_m': float(np.percentile(v('axle_error_m'), 95)),
        'vx_rmse_m_s': rms(v('vx_m_s')-v('command_vx_m_s')),
        'joint_target_rate_rms_rad_s': rms(circular_delta/.008) if len(circular_delta) else 0.,
        'joint_saturation_fraction': float(v('joint_saturation').mean()),
        'wheel_saturation_fraction': float(v('wheel_saturation').mean()),
        'longest_joint_full_cycle_saturation_s': max(longest_full_saturation(v(f'joint_saturation_{i}')) for i in range(4)),
        'longest_wheel_full_cycle_saturation_s': max(longest_full_saturation(v(f'wheel_saturation_{i}')) for i in range(2)),
        'common_x_rms_m': rms(common), 'common_x_peak_m': float(np.abs(common).max()),
        'common_x_outside_008_fraction': float((np.abs(common) > .08).mean()),
        'differential_x_rms_m': rms(v('differential_x_m')),
        'settled_pitch_rms_deg': float(np.degrees(rms(settled[:, COLUMNS.index('pitch_rad')]))),
        'settled_roll_rms_deg': float(np.degrees(rms(settled[:, COLUMNS.index('roll_rad')]))),
        'settled_common_x_rms_m': rms(settled[:, COLUMNS.index('common_x_m')]),
        'joint_target_lag_rms_rad': rms(lag_error),
        'joint_target_lag_p95_rad': float(np.percentile(np.abs(lag_error), 95)),
        'target_numeric_jump_count': int((np.abs(target_delta) > np.pi).sum()),
        'target_numeric_only_jump_count': int(((np.abs(target_delta)>np.pi) & (np.abs(circular_delta)<.2)).sum()),
        'target_circular_step_peak_rad': float(np.abs(circular_delta).max()) if len(circular_delta) else 0.,
        'common_pitch_lag_correlation': corr, 'common_pitch_lag_s': lag,
        'common_acceleration_lag_correlation': corr_acc, 'common_acceleration_lag_s': lag_acc,
        'stop_window_vx_rmse_m_s': rms(stop[:, COLUMNS.index('vx_m_s')]) if len(stop) else None,
        'stop_window_hold_peak_m': float(stop[:, COLUMNS.index('hold_distance_m')].max()) if len(stop) else None,
        'final_second_vx_mean_m_s': float(v('vx_m_s')[-125:].mean()),
        'final_second_common_x_mean_m': float(common[-125:].mean()),
    }


def evaluate(checkpoint, tasks, seeds, destination, save_traces=True):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    result_path = destination/'episodes.json'
    if result_path.exists():
        return json.loads(result_path.read_text())
    checkpoint = str(checkpoint)
    require_compatible_contract(checkpoint, 'joint_tracking_v16')
    params = env_params_for_stage('joint_tracking_v16')
    params.robot.closed_chain_reset = True
    params.domain_randomization.enabled = False
    params.robust.pushes_enabled = False
    raw = UZ05Env('joint_tracking_v16', params=params)
    raw.stage = replace(raw.stage, moving_start_probability=0., episode_steps=3000)
    normalizer = VecNormalize.load(checkpoint+'_vecnormalize.pkl', DummyVecEnv([lambda: raw]))
    normalizer.training = False; normalizer.norm_reward = False
    model = PPO.load(checkpoint, device='cpu')
    rows = []
    try:
        for task in tasks:
            schedule = SCHEDULES[task]
            for seed in seeds:
                obs, _ = raw.reset(seed=seed)
                phase, samples = -1, []
                for step in range(3000):
                    time_s = step * params.control_dt
                    index = max(i for i, entry in enumerate(schedule) if time_s+1e-9 >= entry[0])
                    if index != phase:
                        phase = index
                        _, vx, yaw, height = schedule[index]
                        raw.command[:] = [vx, 0, yaw, height, 0]
                        raw.command_target[:] = raw.command
                        raw.command_steps_left = 3001
                        obs = raw._obs(current_action=raw.previous_action)
                    action, _ = model.predict(normalizer.normalize_obs(obs), deterministic=True)
                    obs, _, terminated, truncated, info = raw.step(action)
                    s, control = raw.sim, raw._previous_control
                    xl, xr = info['wheel_fore_aft_left_m'], info['wheel_fore_aft_right_m']
                    samples.append([(step+1)*params.control_dt, info['pitch'], info['roll'], info['pitch_rate'],
                                    info['tracking_vx_error']+raw.command[0], raw.command[0],
                                    info['tracking_height_error'], info['axle_line_offset_m'], xl, xr,
                                    (xl+xr)/2, xl-xr, info['joint_saturation_fraction'],
                                    info['wheel_saturation_fraction'], info['joint_target_change_sq'],
                                    info['hold_planar_error_m'],
                                    np.sqrt(info['joint_target_change_sq'])/params.control_dt,
                                    *s.joint_positions(), *control['joint_target'], *s.joint_velocities(),
                                    *control['wheel_target'], *s.wheel_velocities(), *control['wheel_torque'],
                                    *info['joint_saturation_per_joint'],
                                    *np.mean(np.abs(control['motor_wheel_current_samples']) >= .99*params.wheel.current_limit_a, axis=0)])
                    if terminated or truncated: break
                a = np.asarray(samples)
                assert a.shape[1] == len(COLUMNS)
                row = {'task': task, 'eval_seed': seed, **episode_metrics(a, info['termination_reason'])}
                rows.append(row)
                if save_traces:
                    np.savez_compressed(destination/f'{task}_{seed}.npz', samples=a, columns=np.array(COLUMNS))
            print(f'evaluated {destination.name} {task}', flush=True)
    finally:
        normalizer.close()
    result_path.write_text(json.dumps(rows, indent=2)+'\n')
    return rows
