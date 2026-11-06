"""Audit chassis pitch, total-COM lean, wheel axes, and policy joint targets."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import mujoco
import numpy as np

from train_uz05 import PPO, require_compatible_contract
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from eval_joint_tracking_transitions import PROFILES
from uz05.env import UZ05Env
from uz05.model import HIP_JOINTS, WHEEL_JOINTS, wheel_axis_metrics
from uz05.spec import env_params_for_stage


def axle_geometry(centers, axes):
    """Distance between parallel axle lines, and angle between their directions."""
    return wheel_axis_metrics(centers, axes)


def circular_pair_difference(angles):
    delta = np.asarray(angles)[:2] - np.asarray(angles)[2:]
    return np.arctan2(np.sin(delta), np.cos(delta))


def summarize(rows):
    if not rows:
        return None
    result = {'samples': len(rows)}
    for key in rows[0]:
        a = np.asarray([r[key] for r in rows])
        result[key] = {'mean': a.mean(axis=0).tolist(), 'min': a.min(axis=0).tolist(),
                       'max': a.max(axis=0).tolist()}
    moving = [r for r in rows if abs(r['acceleration_heading_m_s2']) > .3]
    result['accelerating_samples'] = len(moving)
    result['opposite_chassis_pitch_fraction_when_accelerating'] = (
        float(np.mean([r['pitch_deg'] * r['acceleration_heading_m_s2'] < 0 for r in moving]))
        if moving else None)
    result['opposite_com_lean_fraction_when_accelerating'] = (
        float(np.mean([r['com_lean_deg'] * r['acceleration_heading_m_s2'] < 0 for r in moving]))
        if moving else None)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint', type=Path)
    profiles = {**PROFILES, 'held_forward2': [(0,2,0,.27)], 'held_reverse2': [(0,-2,0,.27)],
                'settled_forward2': [(0,0,0,.27),(3,2,0,.27)],
                'settled_reverse2': [(0,0,0,.27),(3,-2,0,.27)]}
    p.add_argument('--profiles', nargs='+', choices=list(profiles),
                   default=['reverse10','reverse10_backfirst','held_forward2','held_reverse2'])
    p.add_argument('--seed', type=int, default=20000)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve().with_suffix('')
    contract = json.loads(Path(f'{checkpoint}_contract.json').read_text())
    stage = contract['stage']
    require_compatible_contract(checkpoint, stage)
    params = env_params_for_stage(stage)
    params.domain_randomization.enabled = False
    params.robust.pushes_enabled = False
    raw = UZ05Env(stage, params=params)
    raw.stage = replace(raw.stage, moving_start_probability=0., episode_steps=3000)
    norm = VecNormalize.load(f'{checkpoint}_vecnormalize.pkl', DummyVecEnv([lambda: raw]))
    norm.training = False
    norm.norm_reward = False
    policy = PPO.load(checkpoint, device='cpu')
    sim = raw.sim
    joint_id = lambda name: mujoco.mj_name2id(sim.model,mujoco.mjtObj.mjOBJ_JOINT,name)
    wheel_ids = [joint_id(n) for n in WHEEL_JOINTS]
    hip_ids = [joint_id(n) for n in HIP_JOINTS]
    results = []
    try:
        for name in args.profiles:
            obs, _ = raw.reset(seed=args.seed)
            mujoco.mj_subtreeVel(sim.model, sim.data)
            previous_com_velocity = sim.data.subtree_linvel[sim.chassis_body].copy()
            phase = -1
            samples, segments = [], []
            schedule = profiles[name]
            def finish():
                if not samples:
                    return
                segments.append({'start_s':schedule[phase][0], 'command':list(schedule[phase][1:]),
                    'first_second':summarize(samples[:125]), 'last_second':summarize(samples[-125:]),
                    'whole_segment':summarize(samples)})
            for step in range(3000):
                t = step * params.control_dt
                next_phase = max(i for i,item in enumerate(schedule) if t >= item[0])
                if next_phase != phase:
                    finish()
                    samples, phase = [], next_phase
                    _, vx, yaw, height = schedule[phase]
                    raw.command[:] = [vx,0,yaw,height,0]
                    raw.command_target[:] = raw.command
                    raw.command_steps_left = 3001
                    obs = raw._obs(current_action=raw.previous_action)
                action, _ = policy.predict(norm.normalize_obs(obs), deterministic=True)
                obs, _, done, truncated, info = raw.step(action)
                mujoco.mj_subtreeVel(sim.model, sim.data)
                com = sim.data.subtree_com[sim.chassis_body].copy()
                com_velocity = sim.data.subtree_linvel[sim.chassis_body].copy()
                heading = sim.rotation_matrix()[:,0].copy()
                heading[2] = 0
                heading /= np.linalg.norm(heading)
                acceleration = float((com_velocity-previous_com_velocity) @ heading / params.control_dt)
                previous_com_velocity = com_velocity
                centers = sim.data.site_xpos[sim.wheel_sites]
                # Geometric wheel contact-line proxy; force centre is not assumed here.
                support = centers.mean(axis=0).copy()
                support[2] -= sim.wheel_radius
                lean = float(np.degrees(np.arctan2((com-support) @ heading, com[2]-support[2])))
                axes = sim.data.xaxis[wheel_ids]
                targets = raw._previous_control['joint_target']
                pair_target = np.degrees(circular_pair_difference(targets))
                pair_actual = np.degrees(circular_pair_difference(sim.joint_positions()))
                row = {'time_s':(step+1)*params.control_dt,
                    'pitch_deg':float(np.degrees(info['pitch'])), 'com_lean_deg':lean,
                    'acceleration_heading_m_s2':acceleration,
                    'vx_m_s':float(raw.command[0]+info['tracking_vx_error']),
                    'wheel_stagger_m':info['wheel_stagger_m'],
                    'wheel_offset_m':info['wheel_offset_max_m'],
                    'leg_length_difference_m':info['leg_length_left']-info['leg_length_right'],
                    'joint_target_pair_difference_deg':pair_target.tolist(),
                    'joint_actual_pair_difference_deg':pair_actual.tolist(),
                    'joint_target_deg':np.degrees(np.arctan2(np.sin(targets),np.cos(targets))).tolist(),
                    'wheel_fore_aft_left_right_m':[info[f'wheel_fore_aft_{s}_m'] for s in ('left','right')],
                    **axle_geometry(centers,axes)}
                samples.append(row)
                if done or truncated:
                    break
            finish()
            result = {'profile':name,'seed':args.seed,'reason':info['termination_reason'],
                      'steps':step+1,'segments':segments}
            results.append(result)
            print(f'{name}: steps={step+1} {info["termination_reason"]}', flush=True)
        raw.reset(seed=args.seed)
        hip_axes = sim.data.xaxis[hip_ids]
        axis_pair_dots = [float(hip_axes[i] @ hip_axes[i+2]) for i in range(2)]
    finally:
        norm.close()
    report = {'checkpoint':str(checkpoint),'stage':stage,'seed':args.seed,
              'conditions':'nominal dynamics, no pushes, sensor noise retained, near-static reset',
              'pitch_convention':'+X forward,+Y left,+Z up; positive pitch tilts chassis +Z toward +X',
              'com_lean_reference':'total robot COM relative to mean wheel centres minus wheel radius; not force CoP',
              'acceleration_reference':'finite difference of whole-robot COM velocity along horizontal chassis heading',
              'hip_axis_pair_dot_products_at_reset':axis_pair_dots,
              'results':results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
