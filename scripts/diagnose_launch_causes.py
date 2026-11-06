"""Paired fixed-policy interventions; no training or production control changes."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from train_uz05 import require_compatible_contract
from uz05.env import UZ05Env
from uz05.spec import env_params_for_stage, MODEL_XML, ACTOR_OBS_BLOCKS, OBS_SCALE
from ppo_reset_ab_evaluation import COLUMNS, episode_metrics
from uz05.m3508_metrics import score_interval
from run_m3508_validation import THRESHOLDS

ROOT=Path(__file__).resolve().parents[1]
ARMS={'baseline':{}, 'ramp_a05':{'ramp':.5}, 'ramp_a15':{'ramp':1.5},
      'wheel_slew100':{'slew':100.}, 'wheel_slew300':{'slew':300.},
      'leg_hold05':{'hold':.5}, 'leg_hold10':{'hold':1.}, 'leg_hold21':{'hold':21.}}
DEFAULT_ARMS=list(ARMS)
ARMS.update(wheel_slew20={'slew':20.},wheel_slew50={'slew':50.})
ARMS['ramp_a025']={'ramp':.25}

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'diagnostics/launch_cause_ablation')
    p.add_argument('--arms',nargs='+',choices=list(ARMS),default=DEFAULT_ARMS)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/joint_tracking_pos_v16_model20261104/checkpoint')
    p.add_argument('--speeds',nargs='+',type=float,default=[.5,-.5])
    p.add_argument('--seeds',nargs='+',type=int,default=list(range(64000,64005)))
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    if 'baseline' not in args.arms:p.error('baseline is required for paired prefix checks')
    arms={name:ARMS[name] for name in args.arms}
    source=args.checkpoint.resolve()
    if any(not np.isfinite(v) or v==0 for v in args.speeds):p.error('speeds must be finite and nonzero')
    task_name=lambda goal: ('p' if goal>0 else 'm')+format(abs(goal),'g').replace('.','')
    require_compatible_contract(source,'joint_tracking_v16')
    protected={str(p):digest(p) for p in [MODEL_XML,*[Path(str(source)+s) for s in ('.zip','_vecnormalize.pkl','_contract.json')],
        *[ROOT/f'scripts/uz05/{s}.py' for s in ('env','actuators','spec')]]}
    config=dict(source=str(source),protected=protected,arms=arms,seeds=args.seeds,
        speeds=args.speeds,duration=24.,launch=3.,thresholds=THRESHOLDS,
        nominal=True,pushes=False,moving_start=False,noise=True,
        scoring='Final user goal throughout3-24s; ramp-reference error separately retained.',
        action_semantics='Applied action fed to previous_action; leg hold is last pre-launch policy target, never zero joints.',
        interpretation='Fixed-policy closed-loop intervention; cannot uniquely identify network credit assignment.')
    cfg=out/'experiment_config.json'
    if cfg.exists():assert json.loads(cfg.read_text())==config
    else:cfg.write_text(json.dumps(config,indent=2)+'\n')
    params=env_params_for_stage('joint_tracking_v16');params.domain_randomization.enabled=False;params.robust.pushes_enabled=False
    e=UZ05Env('joint_tracking_v16',params=params);e.stage=replace(e.stage,moving_start_probability=0.,episode_steps=3000)
    norm=VecNormalize.load(str(source)+'_vecnormalize.pkl',DummyVecEnv([lambda:e]));norm.training=False;norm.norm_reward=False
    policy=PPO.load(str(source),device='cpu')
    offset=0
    for name,width in ACTOR_OBS_BLOCKS:
        if name=='command':break
        offset+=width
    extra_cols=['normal_force_n_0','normal_force_n_1','yaw_rate_rad_s','command_yaw_rad_s','command_height_m',
                *[f'raw_action_{i}' for i in range(6)],*[f'applied_action_{i}' for i in range(6)]]
    records=[]
    for arm,intervention in arms.items():
        folder=out/arm;folder.mkdir(exist_ok=True)
        for goal in args.speeds:
            task=task_name(goal)
            for seed in config['seeds']:
                path=folder/f'{task}_{seed}.json'
                if path.exists():records.append(json.loads(path.read_text()));continue
                obs,_=e.reset(seed=seed);samples=[];extras=[];last=None;hold=None
                for step in range(3000):
                    tick=step*.008;launched=step>=375
                    ref=(np.sign(goal)*min(abs(goal),intervention['ramp']*(tick-3.))
                         if launched and 'ramp' in intervention else goal if launched else 0.)
                    e.command[:]=[ref,0.,0.,.27,0.];e.command_target[:]=e.command;e.command_steps_left=3001
                    if step in (0,375):obs=e._obs(current_action=e.previous_action)
                    # Command block has zero observation noise; do not consume extra sensor RNG draws for ramp arms.
                    obs[offset:offset+5]=np.asarray(e.command)*np.asarray(OBS_SCALE['command'])
                    raw,_=policy.predict(norm.normalize_obs(obs),deterministic=True);action=np.array(raw,copy=True)
                    if step==375:hold=last[2:].copy()
                    if launched and 'slew' in intervention:
                        delta=intervention['slew']*.008/48.
                        action[:2]=np.clip(action[:2],last[:2]-delta,last[:2]+delta)
                    if launched and tick<3.+intervention.get('hold',0.):action[2:]=hold
                    obs,reward,terminated,truncated,info=e.step(action);last=action.copy()
                    s,c=e.sim,e._previous_control;xl,xr=info['wheel_fore_aft_left_m'],info['wheel_fore_aft_right_m']
                    samples.append([(step+1)*.008,info['pitch'],info['roll'],info['pitch_rate'],
                        info['tracking_vx_error']+ref,ref,info['tracking_height_error'],info['axle_line_offset_m'],xl,xr,
                        (xl+xr)/2,xl-xr,info['joint_saturation_fraction'],info['wheel_saturation_fraction'],
                        info['joint_target_change_sq'],info['hold_planar_error_m'],np.sqrt(info['joint_target_change_sq'])/.008,
                        *s.joint_positions(),*c['joint_target'],*s.joint_velocities(),*c['wheel_target'],*s.wheel_velocities(),
                        *c['wheel_torque'],*info['joint_saturation_per_joint'],
                        *np.mean(np.abs(c['motor_wheel_current_samples'])>=.99*params.wheel.current_limit_a,axis=0)])
                    extras.append([*s.contact_state(terrain_only=True)[0],info['yaw_rate'],0.,.27,*raw,*action])
                    assert np.isfinite(obs).all() and np.isfinite(reward)
                    if terminated or truncated:break
                a=np.asarray(samples);x=np.asarray(extras);assert a.shape[1]==len(COLUMNS)
                cols={k:a[:,i] for i,k in enumerate(COLUMNS)};cols.update({k:x[:,i] for i,k in enumerate(extra_cols)})
                reason=info['termination_reason'];motion=a[:,0]>3.
                row=dict(arm=arm,task=task,seed=seed,goal=goal,**episode_metrics(a,reason),
                    **{'task_'+k:v for k,v in score_interval(cols,3.,24.,goal,THRESHOLDS,reason).items()},
                    motion_pitch_peak_deg=float(np.degrees(np.abs(cols['pitch_rad'][motion])).max()),
                    first05_wheel_sat=float(cols['wheel_saturation'][motion & (a[:,0]<=3.5)].mean()),
                    first05_leg_sat=float(cols['joint_saturation'][motion & (a[:,0]<=3.5)].mean()))
                np.savez_compressed(folder/f'{task}_{seed}.npz',samples=a,columns=COLUMNS,extra=x,extra_columns=extra_cols)
                path.write_text(json.dumps(row,indent=2)+'\n');records.append(row)
                print(arm,task,seed,round(row['survival_s'],3),reason,flush=True)
    df=pd.DataFrame(records)
    for key in ('task_steady_mean_vx_m_s','task_steady_rmse_m_s'):
        if key not in df:df[key]=np.nan
    df.to_csv(out/'episodes.csv',index=False)
    summary=df.groupby(['arm','task'],sort=False).agg(n=('seed','size'),survived=('success','sum'),
        posture=('task_posture_success','sum'),tracking=('task_tracking_success','sum'),combined=('task_composite_success','sum'),
        mean_survival_s=('survival_s','mean'),steady_vx=('task_steady_mean_vx_m_s','mean'),
        steady_rmse=('task_steady_rmse_m_s','mean'),pitch_peak_mean=('motion_pitch_peak_deg','mean'))
    summary.to_csv(out/'summary.csv');print(summary.to_string())
    # Paired pre-launch prefixes and source baseline reproduction, including noise draws.
    for arm in arms:
        for task in map(task_name,args.speeds):
            for seed in config['seeds']:
                a=np.load(out/arm/f'{task}_{seed}.npz')['samples'];b=np.load(out/'baseline'/f'{task}_{seed}.npz')['samples']
                np.testing.assert_array_equal(a[:375],b[:375])
    historical=(source==ROOT/'checkpoints/joint_tracking_pos_v16_model20261104/checkpoint'
                and args.speeds==[.5,-.5] and args.seeds==list(range(64000,64005)))
    if historical:
        for task,oldtask in [('p05','launch_p0.5'),('m05','launch_m0.5')]:
            for seed in config['seeds']:
                a=np.load(out/'baseline'/f'{task}_{seed}.npz')['samples']
                b=np.load(ROOT/'diagnostics/ppo_launch_sampling/baseline_low'/f'{oldtask}_{seed}.npz')['samples']
                np.testing.assert_allclose(a,b,rtol=0,atol=1e-12)
    assert all(digest(p)==h for p,h in protected.items())
    (out/'verification.json').write_text(json.dumps(dict(protected_unchanged=True,paired_prelaunch_exact=True,
        historical_baseline_reproduced=historical,episodes=len(records)),indent=2)+'\n')
    norm.close()

if __name__=='__main__':main()
