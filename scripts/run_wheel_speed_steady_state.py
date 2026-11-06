"""Pilot-first, opt-in wheel-speed probes. Never trains PPO or edits XML."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
from dataclasses import replace,asdict
from pathlib import Path
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_m3508_validation import CHECKPOINT,PROFILES,TASKS,THRESHOLDS,CORE,digest
from uz05.m3508_diagnostics import reference_at
OUT=ROOT/'diagnostics/wheel_speed_steady_state'
SEEDS=list(range(51000,51005));PILOT=SEEDS[:3]
CONDITIONS={
 'legacy_A':{'model':'A_legacy','damping':.02,'scheme':'P','reference':'step'},
 'base':{'model':'B_rated','damping':.02,'scheme':'P','reference':'step'},
 'd010':{'model':'B_rated','damping':.01,'scheme':'P','reference':'step'},
 'd005':{'model':'B_rated','damping':.005,'scheme':'P','reference':'step'},
 'PI':{'model':'B_rated','damping':.02,'scheme':'PI','reference':'step'},
 'P_FF':{'model':'B_rated','damping':.02,'scheme':'P_FF','reference':'step'},
 'ref15':{'model':'B_rated','damping':.02,'scheme':'P','reference':'accel_1.5'}}


def jobs():
    return [(c,p) for c in CONDITIONS for p in (PROFILES if c!='ref15' else ['motion','forward2','stop2_reverse'])]


def freeze():
    from uz05.spec import env_params_for_stage
    OUT.mkdir(parents=True,exist_ok=True);p=env_params_for_stage('joint_tracking_v16')
    cfg={'conditions':CONDITIONS,'pilot_seeds':PILOT,'confirmation_seeds':SEEDS,
         'checkpoint_sha256':digest(str(CHECKPOINT)+'.zip'),'normalizer_sha256':digest(str(CHECKPOINT)+'_vecnormalize.pkl'),
         'protected':{f:digest(ROOT/f) for f in CORE+['scripts/uz05/reset_projection.py']},
         'profiles':PROFILES,'scored_tasks':TASKS,'thresholds':THRESHOLDS,
         'PI':{'ki_a_per_rad':.5,'integral_time_s':2.5,'integral_current_limit_a':4.,
               'antiwindup':'Conditional integration blocks outward saturation; integral-current clamp; reset zeros state.',
               'basis':'Kp/Ki=2.5s, slow integral relative to500Hz motor loop. 4A covers simulated b*48/0.246=3.902A.'},
         'P_FF':{'formula':'I_ff=b_compiled*omega_target/0.246 before shared total10A clamp',
                 'meaning':'Verified simulated viscous damping, not hardware-calibrated damping.'},
         'original_wheel_params':asdict(p.wheel),'original_joint_params':asdict(p.joint),
         'fixed_robot':{'targets_rad_s':[20.,-20.,36.36,-36.36],'seeds':PILOT,'duration_s':5.,'legs':'four zero absolute targets, unchanged PD; no PPO inference or external stabilization'},
         'isolated':{'targets_rad_s':[20.,-20.,36.36,-36.36],'damping':[.02,.01,.005],
                     'schemes':['P','PI','P_FF'],'duration_s':20.,'steady_window_s':2.,
                     'plant':'Two wheels extracted mass/inertia/COM/axis/armature/gear from compiled original; fixed axles, no base/legs/contact/equality/gravity; XML built only in memory.'},
         'expansion_gate':'Independent B P wheel steady-state agrees with analytical error within1e-5rad/s AND any intervention reduces steady wheel target RMS error by>=25% in all3 paired seeds for BOTH forward2/reverse2, while preserving stand/forward2/reverse2 posture successes vsbase. Then expand ALL conditions/tasks to5seeds.',
         'environment':'Projected reset; nominal dynamics except explicit wheel damping; observation noise retained; no pushes/moving start; deterministic original v16;24s.',
         'comparisons':'base->d010/d005 changes only wheel damping; base->PI/FF changes only speed loop; base->ref15 changes only reference; no combined changes.',
         'kinematics':'Signed contact components are end-cycle normal-force weighted per wheel. Missing contact gives NaN, not zero. Exact rolling geometry identity checked.',
         'no_training':True}
    path=OUT/'experiment_config.json'
    if path.exists():
        old=json.loads(path.read_text());old.pop('frozen_utc',None)
        if old!=json.loads(json.dumps(cfg)):raise RuntimeError('Frozen plan mismatch')
    else:cfg['frozen_utc']=datetime.now(timezone.utc).isoformat();path.write_text(json.dumps(cfg,indent=2)+'\n')
    (OUT/'commands.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd '+str(ROOT)+'\n/home/p/NavRL/isaac-training/yes/condabin/conda run --no-capture-output -n sim python scripts/wheel_speed_motor_bench.py\n/home/p/NavRL/isaac-training/yes/condabin/conda run --no-capture-output -n sim python scripts/run_wheel_speed_steady_state.py --pilot --workers 3\n/home/p/NavRL/isaac-training/yes/condabin/conda run --no-capture-output -n sim python scripts/analyze_wheel_speed_steady_state.py --pilot-gate\n# Only if expansion_gate.json passes:\n/home/p/NavRL/isaac-training/yes/condabin/conda run --no-capture-output -n sim python scripts/run_wheel_speed_steady_state.py --confirm --workers 3\n/home/p/NavRL/isaac-training/yes/condabin/conda run --no-capture-output -n sim python scripts/analyze_wheel_speed_steady_state.py\n')


def evaluate(condition,task,seeds,fixed_target=None,*,checkpoint_override=None,destination=None,torque_model_override=None):
    config=CONDITIONS[condition];model_name=torque_model_override or config['model'];reference=config['reference']
    import mujoco
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
    from train_uz05 import require_compatible_contract
    from uz05.env import UZ05Env
    from uz05.spec import env_params_for_stage
    from uz05.wheel_speed_diagnostics import SpeedLoopProbe,contact_diagnostics
    from uz05.reversal_diagnostics import replace_observed_command
    from ppo_reset_ab_evaluation import COLUMNS,episode_metrics
    dest=Path(destination) if destination is not None else (OUT/condition/task if fixed_target is None else OUT/'fixed_robot'/str(fixed_target))
    dest.mkdir(parents=True,exist_ok=True)
    existing=json.loads((dest/'episodes.json').read_text()) if (dest/'episodes.json').exists() else []
    completed={r['eval_seed'] for r in existing}
    if set(seeds)<=completed:return [r for r in existing if r['eval_seed'] in seeds]
    checkpoint=str(checkpoint_override or CHECKPOINT)
    source_contract=json.loads(Path(checkpoint+'_contract.json').read_text())
    source_model='B_rated' if source_contract['actuator_config']['wheel']['efficiency']==1.0 else 'A_legacy'
    require_compatible_contract(checkpoint,'joint_tracking_v16',source_model)
    p=env_params_for_stage('joint_tracking_v16');p.robot.closed_chain_reset=True
    p.domain_randomization.enabled=False;p.robust.pushes_enabled=False
    raw=UZ05Env('joint_tracking_v16',params=p)
    steps=3000 if fixed_target is None else 625
    raw.stage=replace(raw.stage,moving_start_probability=0.,episode_steps=steps)
    raw.sim.model.dof_damping[raw.sim.wheel_dof_adr]=config['damping']
    raw._nominal_model['damping'][raw.sim.wheel_dof_adr]=config['damping']
    raw.actuators.wheel=SpeedLoopProbe(raw.actuators.wheel.params,config['scheme'],[config['damping']]*2,model_name)
    normalizer=None;policy=None
    if fixed_target is None:
        normalizer=VecNormalize.load(checkpoint+'_vecnormalize.pkl',DummyVecEnv([lambda:raw]))
        normalizer.training=False;normalizer.norm_reward=False
        policy=PPO.load(checkpoint,device='cpu')
    original=raw.actuators.compute;motors=[];tick=0.
    def compute(action,*,state,dt):
        raw.actuators.wheel.time_s=tick+len(motors)*.002
        c=original(action,state=state,dt=dt)
        esc=raw.actuators.wheel
        motors.append(np.r_[c['leg_torque_demand'],c['leg_torque'],c['wheel_current'],c['wheel_torque'],
                             c['wheel_target']-state['wheel_vel'],esc.effective_cap_nm,
                             c['wheel_target'],state['wheel_vel'],esc.p_current,esc.integral_current,
                             esc.feedforward_current,esc.raw_current,esc.antiwindup_blocked,
                             -esc.damping*state['wheel_vel']])
        return c
    raw.actuators.compute=compute
    mc=[*[f'leg_demand_nm_{i}' for i in range(4)],*[f'leg_torque_nm_{i}' for i in range(4)],
        *[f'wheel_control_current_a_{i}' for i in range(2)],*[f'wheel_torque_nm_{i}' for i in range(2)],
        *[f'wheel_speed_error_rad_s_{i}' for i in range(2)],'effective_cap_nm',
        *[f'{name}_{i}' for name in ['wheel_target','wheel_actual','p_current','integral_current','ff_current','raw_current','antiwindup_blocked','viscous_predicted_nm'] for i in range(2)]]
    ec=['requested_vx_m_s',*[f'normal_force_n_{i}' for i in range(2)],
        *[f'contact_slip_m_s_{i}' for i in range(2)],*[f'friction_utilization_{i}' for i in range(2)],
        *[f'wheel_qfrc_actuator_nm_{i}' for i in range(2)],*[f'wheel_ctrl_{i}' for i in range(2)],
        *[f'wheel_actuator_force_{i}' for i in range(2)],'mapping_error_nm']
    rows=existing.copy();extra_contact_names=None
    try:
      for seed in seeds:
        if seed in completed:continue
        obs,_=raw.reset(seed=seed);phase=-1;samples=[];extras=[];all_motor=[];previous_ref=0.
        for step in range(steps):
            tick=step*p.control_dt
            index,command,requested=reference_at(PROFILES[task],tick,reference,previous_ref,p.control_dt,task)
            previous_ref=command[0];raw.command[:]=command;raw.command_target[:]=command;raw.command_steps_left=3001
            raw.actuators.wheel.begin_control(tick,requested)
            if index!=phase:
                phase=index;obs=raw._obs(current_action=raw.previous_action)
            else:obs=replace_observed_command(obs,command,raw._observation_encoder.clip_observations)
            if fixed_target is None:action,_=policy.predict(normalizer.normalize_obs(obs),deterministic=True)
            else:action=np.r_[np.full(2,fixed_target/48.),np.zeros(4)]
            motors.clear();obs,_,terminated,truncated,info=raw.step(action)
            all_motor.append(np.array(motors));s=raw.sim;c=raw._previous_control
            xl,xr=info['wheel_fore_aft_left_m'],info['wheel_fore_aft_right_m']
            samples.append([(step+1)*p.control_dt,info['pitch'],info['roll'],info['pitch_rate'],
                info['tracking_vx_error']+command[0],command[0],info['tracking_height_error'],
                info['axle_line_offset_m'],xl,xr,(xl+xr)/2,xl-xr,
                info['joint_saturation_fraction'],info['wheel_saturation_fraction'],
                info['joint_target_change_sq'],info['hold_planar_error_m'],np.sqrt(info['joint_target_change_sq'])/p.control_dt,
                *s.joint_positions(),*c['joint_target'],*s.joint_velocities(),*c['wheel_target'],
                *s.wheel_velocities(),*c['wheel_torque'],*info['joint_saturation_per_joint'],
                *np.mean(np.abs(c['motor_wheel_current_samples'])>=9.9,axis=0)])
            contact=contact_diagnostics(s)
            if extra_contact_names is None:
                extra_contact_names=[k for k in contact if not k.startswith(('normal_force_n','contact_slip_m_s','friction_utilization'))]
            normal=np.array([contact[f'normal_force_n_{i}'] for i in range(2)])
            slip=np.array([contact[f'contact_slip_m_s_{i}'] for i in range(2)])
            utilization=np.array([contact[f'friction_utilization_{i}'] for i in range(2)])
            actual=s.data.qfrc_actuator[s.wheel_dof_adr].copy();error=float(np.abs(actual-c['wheel_torque']).max())
            extras.append([requested,*normal,*slip,*utilization,*actual,
                           *s.data.ctrl[s.wheel_actuator_ids],*s.data.actuator_force[s.wheel_actuator_ids],error,
                           *[contact[k] for k in extra_contact_names]])
            if terminated or truncated:break
        a,e,m=np.asarray(samples),np.asarray(extras),np.asarray(all_motor)
        assert a.shape[1]==len(COLUMNS) and e.shape[1]==len(ec)+len(extra_contact_names) and m.shape[1:]==(4,len(mc))
        assert all(np.isfinite(x).all() for x in (a,m))
        assert np.isfinite(e[:,:len(ec)]).all()
        row={'condition':condition,'model':model_name,'reference':reference,'protocol':task,'eval_seed':seed,
             **episode_metrics(a,info['termination_reason']),'mapping_error_max_nm':float(e[:,ec.index('mapping_error_nm')].max())}
        rows.append(row)
        np.savez_compressed(dest/f'{seed}.npz',samples=a,columns=np.array(COLUMNS),extra=e,extra_columns=np.array(ec+extra_contact_names),motor=m,motor_columns=np.array(mc))
      (dest/'episodes.json').write_text(json.dumps(rows,indent=2)+'\n')
    finally:
        if normalizer is not None:normalizer.close()
        else:raw.close()
    return [r for r in rows if r['eval_seed'] in seeds]


def verify_protected():
    cfg=json.loads((OUT/'experiment_config.json').read_text())
    assert all(digest(ROOT/p)==h for p,h in cfg['protected'].items())
    assert digest(str(CHECKPOINT)+'.zip')==cfg['checkpoint_sha256']
    assert digest(str(CHECKPOINT)+'_vecnormalize.pkl')==cfg['normalizer_sha256']


def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');p.add_argument('--confirm',action='store_true');p.add_argument('--workers',type=int,default=3)
    args=p.parse_args();freeze()
    if args.confirm:
        gate=json.loads((OUT/'expansion_gate.json').read_text())
        if not gate['passed']:raise RuntimeError('Predeclared expansion gate failed')
    if args.pilot or args.confirm:
        seeds=SEEDS if args.confirm else PILOT
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            fs={pool.submit(evaluate,c,task,seeds):(c,task) for c,task in jobs()}
            for f in as_completed(fs):
                r=f.result();print('DONE',fs[f],sum(x['success'] for x in r),'/',len(r),'survived',flush=True)
        verify_protected()
        (OUT/('confirmation_complete.json' if args.confirm else 'pilot_complete.json')).write_text(json.dumps({'seeds':seeds,'jobs':len(jobs()),'protected_unchanged':True},indent=2)+'\n')

if __name__=='__main__':main()
