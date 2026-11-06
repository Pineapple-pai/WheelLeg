"""Score a frozen v6 policy with v7 geometry costs; this script never trains PPO."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from train_uz05 import PPO
from uz05.env import UZ05Env
from uz05.spec import env_params_for_stage, precision_rewards_for_stage, STAGE_BY_NAME


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',default='checkpoints/joint_tracking_pos_v6/checkpoint')
    parser.add_argument('--output',type=Path,default=Path('logs/joint_tracking_v7_symmetry_audit.json'))
    args=parser.parse_args()
    model=PPO.load(args.checkpoint,device='cpu')
    old=precision_rewards_for_stage(STAGE_BY_NAME['joint_tracking_v6'],env_params_for_stage('joint_tracking_v6'))
    cases=[(name,vx,0,h) for h in (.245,.27,.295) for name,vx in [('stand',0),('forward',.3)]]
    cases.extend([('yaw_left',0,1,.27),('mixed',.3,-1,.27)])
    results=[]
    for case,vx,yaw,h in cases:
        params=env_params_for_stage('joint_tracking_v7')
        params.domain_randomization.enabled=False;params.robust.pushes_enabled=False
        env=UZ05Env('joint_tracking_v7',params=params)
        env.reset(seed=20000)
        env.command[:]=[vx,0,yaw,h,0];env.command_target[:]=env.command;env.command_steps_left=3001
        obs=env._obs();rows=[];trace=[]
        for step in range(750):
            action,_=model.predict(obs,deterministic=True)
            obs,_,terminated,truncated,i=env.step(action)
            trace.append([(step+1)*.008,env.sim.body_frame(env.sim.base_lin_vel_world)[0],i['yaw_rate'],i['base_height'],*[i[f'hip_pos_{j}'] for j in range(4)],i['pitch']])
            if step>=125:
                t=i['reward_terms']
                task=sum(t[k] for k in ('upright','pitch_alignment','track_vx','track_yaw','track_height'))
                symmetry=sum(t[k] for k in ('leg_length_difference','fore_aft_difference','roll_error','leg_drop_difference'))
                excess=max(0,abs(i['leg_drop_difference_m'])-old.leg_drop_deadband_m)/old.leg_drop_scale_m
                old_drop=-old.leg_drop_difference*float(env._robust_square(excess))
                severity=(max(0,i['wheel_stagger_m']-env.stage.wheel_stagger_soft_m)/.08)**2+(max(0,i['wheel_offset_max_m']-env.stage.wheel_offset_soft_m)/.10)**2
                old_roll=env.stage.tracking_upright_weight/3*np.exp(-(i['roll']/old.roll_sigma_rad)**2)*np.exp(-severity)
                change=symmetry-old_drop+t['upright']-old_roll
                rows.append([task,symmetry,change,i['leg_length_left']-i['leg_length_right'],i['wheel_fore_aft_left_m']-i['wheel_fore_aft_right_m'],i['roll']])
            if terminated or truncated:break
        a=np.asarray(rows)
        results.append({'case':case,'height_m':h,'steps':step+1,'reason':i['termination_reason'],
            'task_mean':float(a[:,0].mean()) if len(a) else None,
            'symmetry_cost_mean':float(a[:,1].mean()) if len(a) else None,
            'symmetry_to_task_ratio':float(abs(a[:,1].mean())/max(a[:,0].mean(),1e-9)) if len(a) else None,
            'posture_score_change_vs_v6_mean':float(a[:,2].mean()) if len(a) else None,
            'leg_length_difference_rms_m':float(np.sqrt(np.square(a[:,3]).mean())) if len(a) else None,
            'fore_aft_difference_rms_m':float(np.sqrt(np.square(a[:,4]).mean())) if len(a) else None,
            'roll_rms_deg':float(np.degrees(np.sqrt(np.square(a[:,5]).mean()))) if len(a) else None})
        if case=='stand' and h==.27 and args.checkpoint=='checkpoints/joint_tracking_pos_v6/checkpoint':
            baseline=Path('logs/joint_tracking_pos_v6_traces.npz')
            if baseline.is_file():
                expected=np.load(baseline)['Zero_command'][:125]
                np.testing.assert_allclose(np.asarray(trace)[:125],expected,rtol=0,atol=1e-10)
        env.close()
    report={'checkpoint':args.checkpoint,'stage':'joint_tracking_v7','seed':20000,'settle_seconds':1,
        'note':'Frozen v6 policy; same-trajectory v6/v7 posture scoring, not learned v7 performance. No training or teacher data generated.',
        'results':results}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
