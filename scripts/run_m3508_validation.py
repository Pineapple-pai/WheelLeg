"""Frozen single-factor diagnostic comparisons; never calls PPO.learn()."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace, asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from diagnose_reversal_control import digest, CORE
from uz05.m3508_diagnostics import MODELS, reference_at
from ppo_reset_ab_evaluation import SCHEDULES
OUT=ROOT/'diagnostics/m3508_torque_validation'
CHECKPOINT=ROOT/'checkpoints/joint_tracking_pos_v16/checkpoint'
SEEDS=list(range(51000,51005))
PROFILES={
 'stand':[(0,0,0,.27)], 'forward2':[(0,0,0,.27),(3,2,0,.27)],
 'reverse2':[(0,0,0,.27),(3,-2,0,.27)],
 **{k:SCHEDULES[k] for k in ('stop2_reverse','reverse10','motion')}}
REFERENCES=['step','ramp_0.5','ramp_1.0','ramp_1.5','ramp_2.0','accel_1.5','accel_2.0','accel_3.0']
# Labels, source protocol, start, end, final requested velocity. Never score
# tracking against the temporarily smoothed reference in place of the goal.
TASKS=[('stand','stand',0.,24.,0.),('forward2','forward2',3.,24.,2.),
       ('reverse2','reverse2',3.,24.,-2.),('reverse_launch','reverse2',3.,8.,-2.),
       ('reverse_stop','stop2_reverse',10.,24.,0.),
       ('moderate_reversal','reverse10',14.,20.,1.),('high_reversal','motion',11.,16.,-2.)]
THRESHOLDS={'relative_speed_band':.05,'zero_speed_band_m_s':.05,'settle_window_s':1.,
 'settle_occupancy':.95,'response_deadline_s':5.,'steady_window_s':2.,
 'pitch_rms_deg':10.,'roll_rms_deg':5.,'height_mae_m':.03,
 'pitch_excursion_deg':15.,'roll_excursion_deg':10.,'height_excursion_m':.06,
 'max_excursion_duration_s':.25,'max_both_wheels_unloaded_s':.10,
 'common_offset_descriptor_m':.08}


def freeze():
    from uz05.spec import env_params_for_stage
    p=env_params_for_stage('joint_tracking_v16')
    OUT.mkdir(parents=True,exist_ok=True)
    cfg={'checkpoint':str(CHECKPOINT),'checkpoint_sha256':digest(str(CHECKPOINT)+'.zip'),
         'normalizer_sha256':digest(str(CHECKPOINT)+'_vecnormalize.pkl'),
         'protected_files':{x:digest(ROOT/x) for x in CORE+['scripts/uz05/reset_projection.py']},
         'models':{n:m.to_dict() for n,m in MODELS.items()},'references':REFERENCES,
         'profiles':PROFILES,'tasks':TASKS,'thresholds':THRESHOLDS,'seeds':SEEDS,
         'stage':'joint_tracking_v16','control_dt':p.control_dt,'physics_dt':.002,
         'original_wheel_params':asdict(p.wheel),'original_joint_params':asdict(p.joint),
         'environment':'Projected reset, nominal dynamics, observation noise retained, deterministic policy, no pushes/moving start, 24s.',
         'comparisons':'Torque effects: all four models with step. Reference effects: within A and within B, changing only reference. Never attribute A-step to B-ramp difference to one factor.',
         'reference_scope':'motion event11 only; forward2 event3; stop2_reverse events3 and10. Other profiles step only.',
         'C_burst':'B-rated outside 0.5s after requested-vx transitions; boost gain0.3 Nm/control-A, 10A unchanged; equivalent12.195A using0.246 is a hypothesis, not measured current.',
         'D_bound':'0.369 Nm/control-A up to3.69; equivalent15A using0.246. No thermal/back-EMF model; no normal hardware operating claim.',
         'scoring':'Posture and tracking separate per fixed task interval. All requested intervals must be fully observed. Failures are not excluded. Whole-episode survival separately. Tracking: final2s RMS<=band and95% inband; 1s rolling window with RMS<=band and95% inband completed by5s, capped by task end.',
         'posture':'Local interval RMS/height limits plus no sustained excursions or >0.1s both wheels<1N, survive full interval. Endpoints and exposure reported.',
         'no_training':True}
    path=OUT/'experiment_config.json'
    if path.exists():
        old=json.loads(path.read_text());old.pop('frozen_utc',None)
        if old!=json.loads(json.dumps(cfg)):raise RuntimeError('Frozen plan changed; refuse result reuse')
    else:
        cfg['frozen_utc']=datetime.now(timezone.utc).isoformat();path.write_text(json.dumps(cfg,indent=2)+'\n')
    (OUT/'commands.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd '+str(ROOT)+'\n/home/p/NavRL/isaac-training/yes/condabin/conda run --no-capture-output -n sim python scripts/run_m3508_validation.py --bench\n/home/p/NavRL/isaac-training/yes/condabin/conda run --no-capture-output -n sim python scripts/run_m3508_validation.py --execute --workers 3\n/home/p/NavRL/isaac-training/yes/condabin/conda run --no-capture-output -n sim python scripts/analyze_m3508_validation.py\n')


def bench():
    from copy import deepcopy
    import mujoco
    from uz05.env import UZ05Env
    from uz05.spec import env_params_for_stage
    from uz05.m3508_diagnostics import DiagnosticWheelEsc
    import csv
    p=env_params_for_stage('joint_tracking_v16');p.domain_randomization.enabled=False
    env=UZ05Env('joint_tracking_v16',params=p);env.reset(seed=51000)
    s=env.sim;rows=[]
    for name in MODELS:
        esc=DiagnosticWheelEsc(deepcopy(p.wheel),name)
        for time_s in (0.,.002,.498,.5,.6):
            esc.reset();esc.begin_control(0.,2.);esc.time_s=time_s
            for value in (-10.,-5.,0.,5.,10.):
                current=np.array([value,-value]);torque=esc.torque_from_current(current)
                s.data.ctrl[:]=0;s.data.ctrl[s.wheel_actuator_ids]=esc.motor_command(torque)
                mujoco.mj_forward(s.model,s.data)
                actual=s.data.qfrc_actuator[s.wheel_dof_adr].copy()
                assert np.allclose(torque,actual,atol=1e-9)
                rows.append({'model':name,'time_s':time_s,'control_current_left_a':value,
                             'cap_nm':esc.effective_cap_nm,'expected_left_nm':torque[0],
                             'actual_left_nm':actual[0],'ctrl_left':s.data.ctrl[s.wheel_actuator_ids[0]],
                             'actuator_force_left':s.data.actuator_force[s.wheel_actuator_ids[0]],
                             'error_nm':float(np.abs(torque-actual).max())})
        # Inherited current loop reaches exactly10A at8rad/s for every model.
        esc.reset();np.testing.assert_array_equal(esc.current(np.array([8.,-8.]),np.zeros(2),.002),[10.,-10.])
    with (OUT/'motor_bench.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    (OUT/'bench_verification.json').write_text(json.dumps({'passed':True,'cases':len(rows),
        'max_mapping_error_nm':max(r['error_nm'] for r in rows),
        'xml_wheel_gear':s.model.actuator_gear[s.wheel_actuator_ids,0].tolist(),
        'xml_wheel_ctrlrange':s.model.actuator_ctrlrange[s.wheel_actuator_ids].tolist(),
        'unchanged_speed_loop_current_at_8rad_s_error_a':10.},indent=2)+'\n')
    env.close()


def evaluate(model_name,reference,task):
    import mujoco
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
    from train_uz05 import require_compatible_contract
    from uz05.env import UZ05Env
    from uz05.spec import env_params_for_stage
    from uz05.m3508_diagnostics import DiagnosticWheelEsc
    from uz05.reversal_diagnostics import replace_observed_command
    from ppo_reset_ab_evaluation import COLUMNS,episode_metrics
    dest=OUT/model_name/reference/task;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'episodes.json').exists():return json.loads((dest/'episodes.json').read_text())
    checkpoint=str(CHECKPOINT);require_compatible_contract(checkpoint,'joint_tracking_v16')
    p=env_params_for_stage('joint_tracking_v16');p.robot.closed_chain_reset=True
    p.domain_randomization.enabled=False;p.robust.pushes_enabled=False
    raw=UZ05Env('joint_tracking_v16',params=p)
    raw.stage=replace(raw.stage,moving_start_probability=0.,episode_steps=3000)
    raw.actuators.wheel=DiagnosticWheelEsc(raw.actuators.wheel.params,model_name)
    normalizer=VecNormalize.load(checkpoint+'_vecnormalize.pkl',DummyVecEnv([lambda:raw]))
    normalizer.training=False;normalizer.norm_reward=False
    policy=PPO.load(checkpoint,device='cpu');original=raw.actuators.compute;motors=[];tick=0.
    def compute(action,*,state,dt):
        raw.actuators.wheel.time_s=tick+len(motors)*.002
        c=original(action,state=state,dt=dt)
        motors.append(np.r_[c['leg_torque_demand'],c['leg_torque'],c['wheel_current'],c['wheel_torque'],
                             c['wheel_target']-state['wheel_vel'],raw.actuators.wheel.effective_cap_nm])
        return c
    raw.actuators.compute=compute
    mc=[*[f'leg_demand_nm_{i}' for i in range(4)],*[f'leg_torque_nm_{i}' for i in range(4)],
        *[f'wheel_control_current_a_{i}' for i in range(2)],*[f'wheel_torque_nm_{i}' for i in range(2)],
        *[f'wheel_speed_error_rad_s_{i}' for i in range(2)],'effective_cap_nm']
    ec=['requested_vx_m_s',*[f'normal_force_n_{i}' for i in range(2)],
        *[f'contact_slip_m_s_{i}' for i in range(2)],*[f'friction_utilization_{i}' for i in range(2)],
        *[f'wheel_qfrc_actuator_nm_{i}' for i in range(2)],*[f'wheel_ctrl_{i}' for i in range(2)],
        *[f'wheel_actuator_force_{i}' for i in range(2)],'mapping_error_nm']
    rows=[]
    try:
      for seed in SEEDS:
        obs,_=raw.reset(seed=seed);phase=-1;samples=[];extras=[];all_motor=[];previous_ref=0.
        for step in range(3000):
            tick=step*p.control_dt
            index,command,requested=reference_at(PROFILES[task],tick,reference,previous_ref,p.control_dt,task)
            previous_ref=command[0];raw.command[:]=command;raw.command_target[:]=command;raw.command_steps_left=3001
            raw.actuators.wheel.begin_control(tick,requested)
            if index!=phase:
                phase=index;obs=raw._obs(current_action=raw.previous_action)
            else:obs=replace_observed_command(obs,command,raw._observation_encoder.clip_observations)
            action,_=policy.predict(normalizer.normalize_obs(obs),deterministic=True)
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
            normal,tangent,mu,slip=[np.zeros(2) for _ in range(4)]
            for ci in range(s.data.ncon):
                ct=s.data.contact[ci]
                for side,geom in enumerate(s.wheel_geoms):
                    if geom not in (ct.geom1,ct.geom2):continue
                    other=ct.geom2 if geom==ct.geom1 else ct.geom1
                    if s.model.geom_bodyid[other]!=0:continue
                    force=np.zeros(6);mujoco.mj_contactForce(s.model,s.data,ci,force)
                    normal[side]+=abs(force[0]);tangent[side]+=np.linalg.norm(force[1:3]);mu[side]=max(mu[side],ct.friction[0])
                    jp=np.zeros((3,s.model.nv));jr=np.zeros_like(jp)
                    mujoco.mj_jac(s.model,s.data,jp,jr,ct.pos,int(s.model.geom_bodyid[geom]))
                    slip[side]=max(slip[side],float(np.linalg.norm(np.asarray(ct.frame).reshape(3,3)[1:]@(jp@s.data.qvel))))
            actual=s.data.qfrc_actuator[s.wheel_dof_adr].copy();error=float(np.abs(actual-c['wheel_torque']).max())
            extras.append([requested,*normal,*slip,*(tangent/np.maximum(normal*mu,1e-12)),*actual,
                           *s.data.ctrl[s.wheel_actuator_ids],*s.data.actuator_force[s.wheel_actuator_ids],error])
            if terminated or truncated:break
        a,e,m=np.asarray(samples),np.asarray(extras),np.asarray(all_motor)
        assert a.shape[1]==len(COLUMNS) and e.shape[1]==len(ec) and m.shape[1:]==(4,len(mc))
        assert all(np.isfinite(x).all() for x in (a,e,m))
        row={'model':model_name,'reference':reference,'protocol':task,'eval_seed':seed,
             **episode_metrics(a,info['termination_reason']),'mapping_error_max_nm':float(e[:,-1].max())}
        rows.append(row)
        np.savez_compressed(dest/f'{seed}.npz',samples=a,columns=np.array(COLUMNS),extra=e,extra_columns=np.array(ec),motor=m,motor_columns=np.array(mc))
      (dest/'episodes.json').write_text(json.dumps(rows,indent=2)+'\n')
    finally:normalizer.close()
    return rows


def jobs():
    primary=[(m,'step',t) for m in MODELS for t in PROFILES]
    reference=[(m,r,t) for m in ('A_legacy','B_rated') for r in REFERENCES[1:] for t in ('motion','forward2','stop2_reverse')]
    return primary+reference


def verify():
    cfg=json.loads((OUT/'experiment_config.json').read_text())
    assert all(digest(ROOT/p)==h for p,h in cfg['protected_files'].items())
    assert digest(str(CHECKPOINT)+'.zip')==cfg['checkpoint_sha256']
    assert digest(str(CHECKPOINT)+'_vecnormalize.pkl')==cfg['normalizer_sha256']
    matches=[]
    for task in ('motion','stop2_reverse'):
      for seed in SEEDS:
        old=np.load(ROOT/f'diagnostics/reversal_control/v16/step/{task}/{seed}.npz')['samples']
        new=np.load(OUT/f'A_legacy/step/{task}/{seed}.npz')['samples']
        assert np.array_equal(old,new),(task,seed)
        matches.append([task,seed])
    for duration in (.5,1.):
      for seed in SEEDS:
        old_arm='ramp_050' if duration==.5 else 'ramp_100'
        old=np.load(ROOT/f'diagnostics/reversal_control/v16/{old_arm}/motion/{seed}.npz')['samples']
        new=np.load(OUT/f'A_legacy/ramp_{duration:.1f}/motion/{seed}.npz')['samples']
        assert np.array_equal(old,new),(duration,seed)
    (OUT/'integrity_verification.json').write_text(json.dumps({'protected_files_checkpoint_normalizer_unchanged':True,
        'legacy_10_step_traces_exact':True,'legacy_10_ramp_traces_exact':True,'completed_jobs':len(jobs())},indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--bench',action='store_true');p.add_argument('--execute',action='store_true')
    p.add_argument('--workers',type=int,default=3);p.add_argument('--smoke',action='store_true')
    args=p.parse_args();freeze()
    if args.bench:bench()
    if args.execute:
        work=jobs() if not args.smoke else [('A_legacy','step','motion')]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            pending={pool.submit(evaluate,*job):job for job in work}
            for f in as_completed(pending):
                result=f.result();print('DONE',pending[f],f'{sum(r["success"] for r in result)}/5 survived',flush=True)
        if not args.smoke:verify()

if __name__=='__main__':main()
