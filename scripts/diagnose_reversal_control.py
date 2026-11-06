"""Paired, evaluation-only reversal diagnostics. Never trains or saves weights."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
OUT = ROOT/'diagnostics/reversal_control'
SEEDS = list(range(51000, 51005))
# Weak: static torque/Kp error exhausted in one policy interval; medium: rated
# speed; strong: half rated. These are probes, not a calibrated hardware model.
ARMS = {'step': (None, 0.), 'slew_weak': (25., 0.),
        'slew_medium': (10.47, 0.), 'slew_strong': (5.235, 0.),
        'ramp_020': (None, .2), 'ramp_050': (None, .5), 'ramp_100': (None, 1.)}
POLICIES = {'v16': ROOT/'checkpoints/joint_tracking_pos_v16/checkpoint',
            **{f's{s}_{m}': ROOT/f'checkpoints/ppo_reset_ab_20261104_s{s}_{m}/checkpoint'
               for s in (1, 2, 3) for m in ('legacy', 'projected')}}
CORE = ['scripts/uz05/'+n+'.py' for n in ('actuators', 'env', 'spec', 'model')]+['scripts/train_uz05.py', 'uz05_修正后(3).xml']

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze():
    from ppo_reset_ab_evaluation import SCHEDULES
    OUT.mkdir(parents=True, exist_ok=True)
    config = {'stage': 'joint_tracking_v16', 'policies': {k: str(v) for k,v in POLICIES.items()},
              'checkpoint_sha256': {k: digest(str(v)+'.zip') for k,v in POLICIES.items()},
              'normalizer_sha256': {k: digest(str(v)+'_vecnormalize.pkl') for k,v in POLICIES.items()},
              'core_sha256': {p: digest(ROOT/p) for p in CORE},
              'evaluation_seeds': SEEDS, 'arms': ARMS, 'tasks': {k: SCHEDULES[k] for k in ('motion','stop2_reverse')},
              'duration_s': 24, 'control_dt_s': .008, 'physics_dt_s': .002,
              'success': 'Existing termination_reason == time_limit (survival only). Tracking errors reported separately.',
              'event_windows_s': {'pre': [-.5, 0], 'early': [0, .2], 'response': [0, .25]},
              'diagnostic_thresholds': {'large_target_step_rad': .2, 'pitch_departure_rad': .15,
                                         'pitch_rate_departure_rad_s': 1.},
              'large_step_basis': '20 Nm / Kp 100 = 0.2 rad static torque budget; Kd can consume that budget.',
              'slew_basis': '25 rad/s = (20/100)/.008; 10.47 rated parameter; 5.235 half rated.',
              'limiter': 'Every policy tick, circular delta, initialized from actual q. Applied action feeds previous_action. qd_target stays zero.',
              'ramp': 'motion: only event 11s; reverse-stop: launch at 3s and stop at 10s. No other command changes. Sample t at policy tick, endpoint first tick >= duration.',
              'environment': 'Projected reset, nominal dynamics, no pushes, no moving start, original observation noise, deterministic PPO.',
              'matched_windows': 'Report pre/early/.25s windows separately, counts and exposure. Do not compare only whole-episode RMS.',
              'interpretation': 'Interventions change closed-loop action/observation history; no causal mechanism is assumed from correlation.'}
    path = OUT/'experiment_plan.json'
    if path.exists():
        old = json.loads(path.read_text())
        old.pop('frozen_utc', None)
        if old != json.loads(json.dumps(config)):
            raise RuntimeError('Frozen experiment differs; refuse reuse/overwrite')
    else:
        config['frozen_utc'] = datetime.now(timezone.utc).isoformat()
        path.write_text(json.dumps(config, indent=2)+'\n')
    (OUT/'commands.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd '+str(ROOT)+'\n/home/p/NavRL/isaac-training/yes/condabin/conda run --no-capture-output -n sim python scripts/diagnose_reversal_control.py --execute --workers 3\n/home/p/NavRL/isaac-training/yes/condabin/conda run --no-capture-output -n sim python scripts/analyze_reversal_control.py\n')


def run_job(policy, arm, task):
    import mujoco
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from train_uz05 import require_compatible_contract
    from uz05.env import UZ05Env
    from uz05.spec import env_params_for_stage
    from uz05.reversal_diagnostics import CircularTargetLimiter, command_at, replace_observed_command, wrap
    from ppo_reset_ab_evaluation import SCHEDULES, COLUMNS, episode_metrics
    dest = OUT/policy/arm/task
    dest.mkdir(parents=True, exist_ok=True)
    if (dest/'episodes.json').exists():
        return json.loads((dest/'episodes.json').read_text())
    checkpoint = str(POLICIES[policy])
    require_compatible_contract(checkpoint, 'joint_tracking_v16')
    params = env_params_for_stage('joint_tracking_v16')
    params.robot.closed_chain_reset = True
    params.domain_randomization.enabled = False
    params.robust.pushes_enabled = False
    raw = UZ05Env('joint_tracking_v16', params=params)
    raw.stage = replace(raw.stage, moving_start_probability=0., episode_steps=3000)
    norm = VecNormalize.load(checkpoint+'_vecnormalize.pkl', DummyVecEnv([lambda: raw]))
    norm.training = False; norm.norm_reward = False
    model = PPO.load(checkpoint, device='cpu')
    original_compute = raw.actuators.compute
    motor = []
    # Store inputs/outputs of each real motor computation, without recomputing it.
    def compute(action, *, state, dt):
        c = original_compute(action, state=state, dt=dt)
        motor.append(np.r_[state['joint_pos'], state['joint_vel'], c['joint_target'],
                            c['leg_torque_demand'], c['leg_torque'], state['wheel_vel'],
                            c['wheel_target'], c['wheel_current'], c['wheel_torque']])
        return c
    raw.actuators.compute = compute
    mc = ([f'q_{i}' for i in range(4)]+[f'qd_{i}' for i in range(4)]+[f'qd_target_position_{i}' for i in range(4)]+
          [f'leg_demand_nm_{i}' for i in range(4)]+[f'leg_torque_nm_{i}' for i in range(4)]+
          [f'wheel_velocity_{i}' for i in range(2)]+[f'wheel_target_{i}' for i in range(2)]+
          [f'wheel_current_a_{i}' for i in range(2)]+[f'wheel_torque_nm_{i}' for i in range(2)])
    extras = ['limiter_joint_fraction', 'requested_target_step_peak_rad', 'applied_target_step_peak_rad',
              *[f'policy_target_{i}' for i in range(4)], *[f'applied_target_{i}' for i in range(4)],
              *[f'normal_force_n_{i}' for i in range(2)], *[f'friction_ratio_{i}' for i in range(2)],
              *[f'contact_mu_{i}' for i in range(2)], *[f'wheel_point_slip_m_s_{i}' for i in range(2)],
              *[f'wheel_qfrc_actuator_nm_{i}' for i in range(2)],
              *[f'wheel_actuator_force_{i}' for i in range(2)],
              *[f'wheel_ctrl_{i}' for i in range(2)], 'wheel_torque_mapping_error_nm']
    rows = []
    try:
        for seed in SEEDS:
            obs, _ = raw.reset(seed=seed)
            rate, ramp = ARMS[arm]
            limiter = CircularTargetLimiter(raw.sim.joint_positions(), rate, params.control_dt)
            prev_requested = raw.sim.joint_positions().copy()
            prev_applied = prev_requested.copy()
            phase, samples, extra_samples, motors = -1, [], [], []
            for step in range(3000):
                time_s = step*params.control_dt
                index, command = command_at(SCHEDULES[task], time_s, ramp, task)
                raw.command[:] = command; raw.command_target[:] = command
                raw.command_steps_left = 3001
                if index != phase:
                    phase = index
                    obs = raw._obs(current_action=raw.previous_action)
                else:
                    # command is noise-free; no extra observations/RNG draws.
                    obs = replace_observed_command(obs, command, raw._observation_encoder.clip_observations)
                action, _ = model.predict(norm.normalize_obs(obs), deterministic=True)
                action = np.clip(np.asarray(action), -1., 1.)
                applied, active = limiter.apply(action)
                requested = action[2:]*np.pi
                requested_step = np.abs(wrap(requested-prev_requested)).max()
                applied_step = np.abs(wrap(applied[2:]*np.pi-prev_applied)).max()
                prev_requested, prev_applied = requested.copy(), applied[2:]*np.pi
                motor.clear()
                obs, _, terminated, truncated, info = raw.step(applied)
                motors.append(np.asarray(motor))
                s, c = raw.sim, raw._previous_control
                xl, xr = info['wheel_fore_aft_left_m'], info['wheel_fore_aft_right_m']
                samples.append([(step+1)*params.control_dt, info['pitch'], info['roll'], info['pitch_rate'],
                                info['tracking_vx_error']+raw.command[0], raw.command[0],
                                info['tracking_height_error'], info['axle_line_offset_m'], xl, xr,
                                (xl+xr)/2, xl-xr, info['joint_saturation_fraction'], info['wheel_saturation_fraction'],
                                info['joint_target_change_sq'], info['hold_planar_error_m'],
                                np.sqrt(info['joint_target_change_sq'])/params.control_dt,
                                *s.joint_positions(), *c['joint_target'], *s.joint_velocities(),
                                *c['wheel_target'], *s.wheel_velocities(), *c['wheel_torque'],
                                *info['joint_saturation_per_joint'],
                                *np.mean(np.abs(c['motor_wheel_current_samples'])>=.99*params.wheel.current_limit_a, axis=0)])
                normal, tangent, mu, slip = [np.zeros(2) for _ in range(4)]
                # Solved contact force and velocity at the actual contact point.
                for ci in range(s.data.ncon):
                    ct = s.data.contact[ci]
                    for side, geom in enumerate(s.wheel_geoms):
                        if geom not in (ct.geom1, ct.geom2): continue
                        other = ct.geom2 if geom == ct.geom1 else ct.geom1
                        if s.model.geom_bodyid[other] != 0: continue
                        force = np.zeros(6); mujoco.mj_contactForce(s.model, s.data, ci, force)
                        normal[side] += abs(force[0]); tangent[side] += np.linalg.norm(force[1:3])
                        mu[side] = max(mu[side], float(ct.friction[0]))
                        jp = np.zeros((3, s.model.nv)); jr = np.zeros_like(jp)
                        mujoco.mj_jac(s.model, s.data, jp, jr, ct.pos, int(s.model.geom_bodyid[geom]))
                        velocity = jp@s.data.qvel
                        frame = np.asarray(ct.frame).reshape(3, 3)
                        slip[side] = max(slip[side], float(np.linalg.norm(frame[1:]@velocity)))
                actual_torque = s.data.qfrc_actuator[s.wheel_dof_adr].copy()
                mismatch = np.max(np.abs(actual_torque-c['wheel_torque']))
                extra_samples.append([np.mean(active), requested_step, applied_step, *requested, *(applied[2:]*np.pi),
                                      *normal, *(tangent/np.maximum(normal, 1e-12)), *mu, *slip, *actual_torque,
                                      *s.data.actuator_force[s.wheel_actuator_ids],
                                      *s.data.ctrl[s.wheel_actuator_ids], mismatch])
                if terminated or truncated: break
            a = np.asarray(samples); e = np.asarray(extra_samples); m = np.asarray(motors)
            assert a.shape[1] == len(COLUMNS) and e.shape[1] == len(extras) and m.shape[1:] == (4,len(mc))
            assert np.isfinite(a).all() and np.isfinite(e).all() and np.isfinite(m).all()
            row = {'policy': policy, 'arm': arm, 'task': task, 'eval_seed': seed,
                   **episode_metrics(a, info['termination_reason']),
                   'total_mass_kg': float(s.model.body_mass.sum()),
                   'limiter_intervention_fraction': float(e[:,0].mean()),
                   'torque_mapping_max_error_nm': float(e[:,-1].max())}
            rows.append(row)
            np.savez_compressed(dest/f'{seed}.npz', samples=a, columns=np.array(COLUMNS),
                                extra=e, extra_columns=np.array(extras), motor=m, motor_columns=np.array(mc))
        (dest/'episodes.json').write_text(json.dumps(rows, indent=2)+'\n')
    finally:
        norm.close()
    return rows


def main():
    p = argparse.ArgumentParser(); p.add_argument('--execute', action='store_true'); p.add_argument('--workers', type=int, default=3)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args(); freeze()
    if not args.execute: return
    jobs = [(policy, arm, task) for policy in POLICIES for arm in ARMS for task in ('motion','stop2_reverse')]
    if args.smoke: jobs = [('v16','step','motion')]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_job, *job): job for job in jobs}
        for f in as_completed(futures):
            rows = f.result()
            print(f'DONE {futures[f]} {sum(r["success"] for r in rows)}/5', flush=True)
    frozen = json.loads((OUT/'experiment_plan.json').read_text())
    assert all(digest(ROOT/p) == h for p,h in frozen['core_sha256'].items())
    assert all(digest(str(POLICIES[k])+'.zip') == h for k,h in frozen['checkpoint_sha256'].items())
    (OUT/'integrity_verification.json').write_text(json.dumps({'core_and_checkpoints_unchanged': True, 'completed_jobs': len(jobs)}, indent=2)+'\n')

if __name__ == '__main__': main()
