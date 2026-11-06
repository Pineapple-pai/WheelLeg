"""Measure response time, height sag and attitude during held-command switches."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from train_uz05 import PPO, require_compatible_contract
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from uz05.env import UZ05Env
from uz05.spec import env_params_for_stage

# time, vx, yaw rate, base height. Fixed protocols, not training curricula.
PROFILES = {
    'stop2_forward': [(0,0,0,.27),(3,2,0,.27),(10,0,0,.27)],
    'stop2_reverse': [(0,0,0,.27),(3,-2,0,.27),(10,0,0,.27)],
    'reverse17': [(0,0,0,.27),(3,.5,0,.27),(7,1.7,0,.27),(12,-1.7,0,.27),(18,1.7,0,.27),(22,0,0,.27)],
    'reverse15': [(0,0,0,.27),(3,.5,0,.27),(7,1.5,0,.27),(12,-1.5,0,.27),(18,1.5,0,.27),(22,0,0,.27)],
    'reverse05': [(0,0,0,.27),(3,.5,0,.27),(8,-.5,0,.27),(13,.5,0,.27),(18,0,0,.27)],
    'reverse10': [(0,0,0,.27),(3,1,0,.27),(8,-1,0,.27),(14,1,0,.27),(20,0,0,.27)],
    'reverse10_backfirst': [(0,0,0,.27),(3,-1,0,.27),(8,1,0,.27),(14,-1,0,.27),(20,0,0,.27)],
    'stop_go10': [(0,0,0,.27),(3,1,0,.27),(7,0,0,.27),(11,-1,0,.27),(15,0,0,.27),(19,1,0,.27)],
    'stop_go10_backfirst': [(0,0,0,.27),(3,-1,0,.27),(7,0,0,.27),(11,1,0,.27),(15,0,0,.27),(19,-1,0,.27)],
    'reverse09': [(0,0,0,.27),(3,.5,0,.27),(7,.9,0,.27),(11,-.9,0,.27),(17,.9,0,.27),(21,0,0,.27)],
    'reverse09_backfirst': [(0,0,0,.27),(3,-.5,0,.27),(7,-.9,0,.27),(11,.9,0,.27),(17,-.9,0,.27),(21,0,0,.27)],
    'motion': [(0,0,0,.27),(3,1,0,.27),(7,2,0,.27),(11,-2,0,.27),(16,0,1,.27),(20,0,0,.27)],
    'height': [(0,0,0,.27),(3,0,0,.22),(7,0,0,.32),(11,1,0,.32),(15,1,0,.22),(19,0,0,.27)],
    'turn': [(0,1,0,.27),(4,1,1,.27),(8,1,-1,.27),(12,-1,0,.27),(16,0,0,.27),(20,2,.8,.27)],
    'height2_forward': [(0,0,0,.27),(3,2,0,.27),(8,2,0,.22),(13,2,0,.32),(18,0,0,.32)],
    'height2_reverse': [(0,0,0,.27),(3,-2,0,.27),(8,-2,0,.22),(13,-2,0,.32),(18,0,0,.32)],
    'turn2': [(0,0,0,.27),(3,2,0,.27),(8,0,1,.27),(13,-2,0,.27),(18,0,-1,.27)],
}


def settling_time(values, target, initial, dt, absolute_tolerance, hold_seconds=.25):
    """First entry into a 10%-step/absolute band maintained for at least 0.25s."""
    tolerance=max(absolute_tolerance,.1*abs(target-initial))
    required=int(np.ceil(hold_seconds/dt));run=0
    for index,value in enumerate(values):
        run=run+1 if abs(value-target)<=tolerance else 0
        if run>=required:
            return (index-required+2)*dt
    return None


def velocity_response(values, target, initial, dt):
    """Separate braking, direction change, and tight tracking after a reversal."""
    values = np.asarray(values)
    reversing = initial * target < 0 and abs(initial) > .1 and abs(target) > .1
    crossed = np.flatnonzero(values * initial <= 0) if reversing else []
    t_zero = float((crossed[0] + 1) * dt) if len(crossed) else None
    t_90 = settling_time(values, target, initial, dt, .05)
    # initial=target disables the step-relative band: this is an absolute 0.05m/s band.
    t_tight = settling_time(values, target, target, dt, .05)
    return {
        'initial_vx_m_s': initial,
        'vx_zero_crossing_s': t_zero,
        'vx_reverse_acceleration_to_t90_s': (max(0., t_90-t_zero)
                                            if t_zero is not None and t_90 is not None else None),
        'vx_settling_005_m_s_s': t_tight,
        'vx_integral_absolute_error_m': float(np.abs(values-target).sum()*dt),
        'vx_at_elapsed_m_s': {str(t):float(values[int(round(t/dt))-1])
                               for t in (.25, .5, 1., 2.) if int(round(t/dt)) <= len(values)},
    }


def support_response(offsets, joint_positions, dt, band=.08):
    """Measure both legs individually; a symmetric outward swing must be visible."""
    offsets = np.asarray(offsets)
    joint_positions = np.asarray(joint_positions)
    excess = np.any(np.abs(offsets) > band, axis=1)
    longest = run = 0
    for invalid in excess:
        run = run + 1 if invalid else 0
        longest = max(longest, run)
    peak = int(np.argmax(np.max(np.abs(offsets), axis=1)))
    recovery = settling_time(np.max(np.abs(offsets[peak:]), axis=1), 0, 0, dt, band)
    return {
        'wheel_fore_aft_left_right_min_m': offsets.min(axis=0).tolist(),
        'wheel_fore_aft_left_right_max_m': offsets.max(axis=0).tolist(),
        'wheel_fore_aft_band_m': band,
        'wheel_fore_aft_outside_band_fraction': float(excess.mean()),
        'wheel_fore_aft_longest_outside_band_s': longest * dt,
        'wheel_fore_aft_recovery_after_peak_s': recovery,
        'peak_wheel_offset_elapsed_s': (peak + 1) * dt,
        'joint_position_at_peak_wheel_offset_rad':
            np.arctan2(np.sin(joint_positions[peak]), np.cos(joint_positions[peak])).tolist(),
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint',type=Path)
    p.add_argument('--profiles',nargs='+',choices=list(PROFILES),default=list(PROFILES))
    p.add_argument('--episodes',type=int,default=3)
    p.add_argument('--seed',type=int,default=20000)
    p.add_argument('--domain-randomization',action='store_true')
    p.add_argument('--perturbations',action='store_true')
    p.add_argument('--moving-starts',action='store_true',help='include the training mixture of moving initial states')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.episodes<1:p.error('--episodes must be positive')
    checkpoint=args.checkpoint.with_suffix('')
    contract=json.loads(Path(f'{checkpoint}_contract.json').read_text())
    stage=contract['stage'];require_compatible_contract(checkpoint,stage)
    params=env_params_for_stage(stage)
    params.domain_randomization.enabled=args.domain_randomization
    params.robust.pushes_enabled=args.perturbations
    raw=UZ05Env(stage,params=params)
    if not args.moving_starts:
        raw.stage=replace(raw.stage,moving_start_probability=0.0)
    raw.stage=replace(raw.stage,episode_steps=int(24/params.control_dt))
    norm=VecNormalize.load(f'{checkpoint}_vecnormalize.pkl',DummyVecEnv([lambda:raw]))
    norm.training=False;norm.norm_reward=False
    model=PPO.load(checkpoint,device='cpu')
    results=[]
    for name in args.profiles:
        schedule=PROFILES[name]
        for episode in range(args.episodes):
            obs,_=raw.reset(seed=args.seed+episode)
            segments=[];rows=[];reward_rows=[];phase=-1;initial=None
            def finish():
                if not rows:return
                a=np.asarray(rows);start,vx,yaw,height=schedule[phase]
                last=a[-min(len(a),int(1/params.control_dt)):]
                segments.append({'start_s':start,'target_vx_yaw_height':[vx,yaw,height],
                    **velocity_response(a[:,0],vx,initial[0],params.control_dt),
                    **support_response(a[:,16:18],a[:,18:22],params.control_dt),
                    'observed_seconds':len(a)*params.control_dt,
                    'vx_response_t90_s':settling_time(a[:,0],vx,initial[0],params.control_dt,.05),
                    'yaw_response_t90_s':settling_time(a[:,1],yaw,initial[1],params.control_dt,.10),
                    'height_settling_time_s':settling_time(a[:,2],height,initial[2],params.control_dt,.005),
                    'height_min_m':float(a[:,2].min()),'height_max_m':float(a[:,2].max()),
                    'height_shortfall_peak_m':float(np.maximum(height-a[:,2],0).max()),
                    'pitch_peak_abs_deg':float(np.degrees(np.abs(a[:,3]).max())),
                    'pitch_rate_peak_abs_rad_s':float(np.abs(a[:,4]).max()),
                    'vx_mae_m_s':float(np.abs(a[:,0]-vx).mean()),'last_second_vx_mean_m_s':float(last[:,0].mean()),
                    'last_second_height_mean_m':float(last[:,2].mean()),
                    'last_second_pitch_mean_deg':float(np.degrees(last[:,3].mean())),
                    'leg_forward_reach_peak_left_right_m':np.max(a[:,6:8],axis=0).tolist(),
                    'leg_backward_reach_peak_left_right_m':np.max(a[:,8:10],axis=0).tolist(),
                    'leg_length_difference_rms_m':float(np.sqrt(np.mean(a[:,10]**2))),
                    'leg_length_difference_peak_m':float(np.abs(a[:,10]).max()),
                    'roll_rms_deg':float(np.degrees(np.sqrt(np.mean(a[:,11]**2)))),
                    'wheel_stagger_peak_m':float(a[:,12].max()),
                    'wheel_offset_peak_m':float(a[:,13].max()),
                    'last_second_wheel_fore_aft_mean_left_right_m':last[:,16:18].mean(axis=0).tolist(),
                    'last_second_wheel_offset_peak_m':float(last[:,13].max()),
                    'axle_line_offset_peak_m':float(a[:,22].max()),
                    'last_second_axle_line_offset_mean_m':float(last[:,22].mean()),
                    'axle_offset_above_1cm_fraction':float(np.mean(a[:,22]>.01)),
                    'axle_parallel_error_peak_deg':float(a[:,23].max()),
                    'support_com_lean_mean_deg':float(np.degrees(a[:,24].mean())),
                    'support_pitch_error_rms_deg':float(np.degrees(np.sqrt(np.mean(a[:,25]**2)))),
                    'joint_target_pair_difference_rms_deg':np.degrees(np.sqrt(np.mean(a[:,26:28]**2,axis=0))).tolist(),
                    'joint_actual_pair_difference_rms_deg':np.degrees(np.sqrt(np.mean(a[:,28:30]**2,axis=0))).tolist(),
                    'wheel_saturation_fraction':float(a[:,14].mean()),
                    'joint_saturation_fraction':float(a[:,15].mean()),
                    'reward_terms_mean':{key:float(np.mean([r[key] for r in reward_rows]))
                                         for key in reward_rows[0]},
                    'hold_position_peak_m':float(a[:,5].max()) if vx==0 and yaw==0 else None})
            for step in range(raw.stage.episode_steps):
                t=step*params.control_dt
                next_phase=max(i for i,x in enumerate(schedule) if t>=x[0])
                if next_phase!=phase:
                    finish();rows=[];reward_rows=[];phase=next_phase
                    _,vx,yaw,height=schedule[phase]
                    initial=[float(raw.sim.body_frame(raw.sim.base_lin_vel_world)[0]),
                             float(raw.sim.body_frame(raw.sim.base_ang_vel_world)[2]),
                             float(raw.sim.base_pos[2]-raw.sim.support_height(float(raw.sim.base_pos[0]),float(raw.sim.base_pos[1])))]
                    raw.command[:]=[vx,0,yaw,height,0];raw.command_target[:]=raw.command
                    raw.command_steps_left=raw.stage.episode_steps+1
                    obs=raw._obs(current_action=raw.previous_action)
                action,_=model.predict(norm.normalize_obs(obs),deterministic=True)
                obs,_,terminated,truncated,info=raw.step(action)
                envelope=raw.sim.leg_envelope()
                target=raw._previous_control['joint_target']
                target_delta=target[:2]-target[2:]
                actual=raw.sim.joint_positions()
                actual_delta=actual[:2]-actual[2:]
                rows.append([raw.command[0]+info['tracking_vx_error'],info['yaw_rate'],raw.command[3]+info['tracking_height_error'],
                             info['pitch'],info['pitch_rate'],info['hold_planar_error_m'],
                             *[envelope[f'leg_{direction}_reach_{side}_m'] for direction in ('forward','backward') for side in ('left','right')],
                             info['leg_length_left']-info['leg_length_right'],info['roll'],
                             info['wheel_stagger_m'],info['wheel_offset_max_m'],
                             info['wheel_saturation_fraction'],info['joint_saturation_fraction'],
                             info['wheel_fore_aft_left_m'],info['wheel_fore_aft_right_m'],
                             *actual,info['axle_line_offset_m'],info['axle_parallel_error_deg'],
                             info['support_com_lean_rad'],info['support_pitch_error_rad'],
                             *np.arctan2(np.sin(target_delta),np.cos(target_delta)),
                             *np.arctan2(np.sin(actual_delta),np.cos(actual_delta))])
                reward_rows.append(info['reward_terms'])
                if terminated or truncated:break
            finish()
            r={'profile':name,'seed':args.seed+episode,'steps':step+1,
               'reason':info['termination_reason'],'segments':segments}
            results.append(r)
            print(f"{name} seed={args.seed+episode} steps={step+1} {info['termination_reason']}",flush=True)
    norm.close()
    report={'checkpoint':str(checkpoint),'stage':stage,'duration_seconds':24,
            'domain_randomization':args.domain_randomization,'perturbations':args.perturbations,
            'moving_start_probability':raw.stage.moving_start_probability,
            'response_definition':'10% of initial-to-target step or absolute band, held >=0.25s; None means not settled before segment ended.',
            'profiles':PROFILES,'results':results}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
