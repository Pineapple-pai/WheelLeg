"""Fixed-policy, no-learning assessment of the new-model training checkpoints."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
from train_uz05 import require_compatible_contract
from uz05.env import UZ05Env
from uz05.spec import env_params_for_stage,MODEL_XML
from run_m3508_validation import PROFILES,TASKS,THRESHOLDS
from eval_joint_tracking_transitions import PROFILES as EXTRA_PROFILES
from ppo_reset_ab_evaluation import COLUMNS,episode_metrics
from uz05.m3508_metrics import score_interval

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main(*, diagnostic_transform=None, diagnostic_metadata=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tasks',nargs='+',default=None)
    p.add_argument('--launch-speeds',nargs='+',type=float,default=None,
                   help='Add paired 0->+/-speed launch tasks at t=3s, held until24s')
    p.add_argument('--seeds',nargs='+',type=int,default=list(range(62000,62005)))
    p.add_argument('--domain-randomization',action='store_true')
    args=p.parse_args();torch.set_num_threads(1)
    schedules={**EXTRA_PROFILES,**PROFILES}
    intervals=list(TASKS)
    args.tasks=args.tasks or (['stand'] if args.launch_speeds else list(PROFILES))
    for speed in args.launch_speeds or []:
        if not np.isfinite(speed) or speed<=0:p.error('launch speeds must be finite and positive')
        for sign in (1,-1):
            goal=sign*speed
            name=f'launch_{"p" if sign>0 else "m"}{speed:g}'
            schedules[name]=[(0,0,0,.27),(3,goal,0,.27)]
            intervals.append((name,name,3.,24.,goal))
            args.tasks.append(name)
    cp=str(args.checkpoint);require_compatible_contract(args.checkpoint,'joint_tracking_v16')
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'results.json').exists():raise RuntimeError('Use a new output directory')
    protected={x:digest(cp+x) for x in ('.zip','_vecnormalize.pkl','_contract.json')}
    config=dict(checkpoint=cp,protected=protected,model_xml_sha256=digest(MODEL_XML),
                tasks=args.tasks,seeds=args.seeds,thresholds=THRESHOLDS,
                schedules={k:schedules[k] for k in args.tasks},
                domain_randomization=args.domain_randomization,no_learning=True,
                observation_noise=True,moving_starts=False,pushes=False,duration_s=24)
    if diagnostic_transform is not None:
        if not diagnostic_metadata:
            raise ValueError('Diagnostic transforms require explicit provenance')
        config['diagnostic_intervention'] = diagnostic_metadata
    (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    params=env_params_for_stage('joint_tracking_v16')
    params.domain_randomization.enabled=args.domain_randomization
    params.robust.pushes_enabled=False
    e=UZ05Env('joint_tracking_v16',params=params)
    e.stage=replace(e.stage,moving_start_probability=0.,episode_steps=3000)
    norm=VecNormalize.load(cp+'_vecnormalize.pkl',DummyVecEnv([lambda:e]))
    norm.training=False;norm.norm_reward=False
    policy=PPO.load(cp,device='cpu');episodes=[];scored=[];segments=[]
    extra_cols=['normal_force_n_0','normal_force_n_1','yaw_rate_rad_s','command_yaw_rad_s','command_height_m']
    for task in args.tasks:
        schedule=schedules[task]
        for seed in args.seeds:
            obs,_=e.reset(seed=seed);samples=[];extras=[];phase=-1
            for step in range(3000):
                tick=step*params.control_dt
                index=max(i for i,x in enumerate(schedule) if tick+1e-9>=x[0])
                if phase!=index:
                    phase=index;_,vx,yaw,h=schedule[index]
                    e.command[:]=[vx,0,yaw,h,0];e.command_target[:]=e.command
                    e.command_steps_left=3001;obs=e._obs(current_action=e.previous_action)
                policy_obs = obs if diagnostic_transform is None else diagnostic_transform(obs.copy())
                action,_=policy.predict(norm.normalize_obs(policy_obs),deterministic=True)
                obs,reward,terminated,truncated,info=e.step(action)
                assert np.isfinite(obs).all() and np.isfinite(reward)
                s,c=e.sim,e._previous_control
                xl,xr=info['wheel_fore_aft_left_m'],info['wheel_fore_aft_right_m']
                samples.append([(step+1)*params.control_dt,info['pitch'],info['roll'],info['pitch_rate'],
                    info['tracking_vx_error']+vx,vx,info['tracking_height_error'],info['axle_line_offset_m'],
                    xl,xr,(xl+xr)/2,xl-xr,info['joint_saturation_fraction'],info['wheel_saturation_fraction'],
                    info['joint_target_change_sq'],info['hold_planar_error_m'],np.sqrt(info['joint_target_change_sq'])/params.control_dt,
                    *s.joint_positions(),*c['joint_target'],*s.joint_velocities(),*c['wheel_target'],
                    *s.wheel_velocities(),*c['wheel_torque'],*info['joint_saturation_per_joint'],
                    *np.mean(np.abs(c['motor_wheel_current_samples'])>=.99*params.wheel.current_limit_a,axis=0)])
                force=s.contact_state(terrain_only=True)[0]
                extras.append([*force,info['yaw_rate'],yaw,h])
                if terminated or truncated:break
            a=np.asarray(samples);x=np.asarray(extras)
            assert a.shape[1]==len(COLUMNS) and x.shape[1]==len(extra_cols)
            cols={k:a[:,i] for i,k in enumerate(COLUMNS)}
            cols.update({k:x[:,i] for i,k in enumerate(extra_cols)})
            reason=info['termination_reason']
            row=dict(task=task,seed=seed,**episode_metrics(a,reason))
            episodes.append(row)
            for name,protocol,start,end,goal in intervals:
                if protocol==task:
                    scored.append(dict(task=name,protocol=task,seed=seed,
                        **score_interval(cols,start,end,goal,THRESHOLDS,reason)))
            # Per-command exposure and last-two-second estimates. Missing tails
            # remain null rather than being filled with a pre-failure mean.
            for i,(start,vx0,yaw0,h0) in enumerate(schedule):
                end=schedule[i+1][0] if i+1<len(schedule) else 24.
                t=cols['time_s'];mask=(t>start+1e-9)&(t<=end+1e-9)
                complete=bool(t[-1]>=end-1e-9 and (reason=='time_limit' or t[-1]>end+1e-9))
                tail=mask&(t>end-2+1e-9)
                tail_ok=complete and tail.sum()==250
                get=lambda name:float(np.mean(cols[name][tail])) if tail_ok else None
                target=np.mean(np.column_stack([cols['wheel_target_0'],cols['wheel_target_1']]),axis=1)*(-.055)
                wheel=np.mean(np.column_stack([cols['wheel_actual_0'],cols['wheel_actual_1']]),axis=1)*(-.055)
                band=.05*abs(vx0) if vx0 else .05
                se=(cols['vx_m_s'][tail]-vx0) if tail_ok else np.array([])
                segments.append(dict(task=task,seed=seed,start_s=start,end_s=end,vx_goal=vx0,yaw_goal=yaw0,height_goal=h0,
                    complete=complete,exposure_s=float(mask.sum()*.008),steady_vx=get('vx_m_s'),
                    steady_speed_rmse=float(np.sqrt(np.mean(se**2))) if tail_ok else None,
                    steady_inband_fraction=float(np.mean(np.abs(se)<=band)) if tail_ok else None,
                    steady_yaw=get('yaw_rate_rad_s'),steady_height_error=get('height_error_m'),
                    target_equivalent_vx=float(target[tail].mean()) if tail_ok else None,
                    wheel_equivalent_vx=float(wheel[tail].mean()) if tail_ok else None,
                    steady_pitch_deg=float(np.degrees(cols['pitch_rad'][tail].mean())) if tail_ok else None,
                    steady_wheel_sat=get('wheel_saturation'),steady_leg_sat=get('joint_saturation')))
            np.savez_compressed(args.output/f'{task}_{seed}.npz',samples=a,columns=COLUMNS,extra=x,extra_columns=extra_cols)
            print(f'{args.output.name} {task} {seed}: {row["survival_s"]:.3f}s {reason}',flush=True)
    for suffix,h in protected.items():assert digest(cp+suffix)==h
    result=dict(config=config,episodes=episodes,task_metrics=scored,segments=segments)
    (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    norm.close()

if __name__=='__main__':main()
