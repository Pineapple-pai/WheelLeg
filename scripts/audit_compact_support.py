"""Compare v10/v11 rewards on identical frozen-policy trajectories, offline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from train_uz05 import require_compatible_contract
from eval_joint_tracking_transitions import PROFILES
from uz05.env import UZ05Env
from uz05.spec import env_params_for_stage


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('--profiles',nargs='+',choices=list(PROFILES),default=['reverse05','reverse15'])
    parser.add_argument('--seed',type=int,default=20000)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    checkpoint=args.checkpoint.with_suffix('')
    require_compatible_contract(checkpoint,'joint_tracking_v10')
    require_compatible_contract(checkpoint,'joint_tracking_v11')
    envs=[]
    for stage in ('joint_tracking_v10','joint_tracking_v11'):
        p=env_params_for_stage(stage)
        p.domain_randomization.enabled=False;p.robust.pushes_enabled=False
        envs.append(UZ05Env(stage,params=p))
    old,new=envs
    norm=VecNormalize.load(f'{checkpoint}_vecnormalize.pkl',DummyVecEnv([lambda:old]))
    norm.training=False;norm.norm_reward=False
    policy=PPO.load(checkpoint,device='cpu')
    results=[];max_state_error=0.
    try:
        for name in args.profiles:
            oldobs,_=old.reset(seed=args.seed)
            newobs,_=new.reset(seed=args.seed)
            schedule=PROFILES[name];phase=-1;rows=[];segments=[]
            def finish():
                if not rows:return
                a=np.asarray(rows)
                segments.append({
                    'target_vx_yaw_height':list(schedule[phase][1:]),'samples':len(a),
                    'v10_reward_mean':float(a[:,0].mean()),'v11_reward_mean':float(a[:,1].mean()),
                    'additional_cost_to_v10_positive_task_ratio':float(np.mean(a[:,0]-a[:,1])/max(1e-9,a[:,2].mean())),
                    'leg_reach_cost_mean':float(a[:,3].mean()),
                    'leg_forward_reach_peak_m':float(a[:,4].max()),
                    'leg_backward_reach_peak_m':float(a[:,5].max()),
                })
            for step in range(old.stage.episode_steps):
                t=step*old.params.control_dt
                next_phase=max(i for i,x in enumerate(schedule) if t>=x[0])
                if next_phase!=phase:
                    finish();rows=[];phase=next_phase
                    _,vx,yaw,height=schedule[phase]
                    for env in envs:
                        env.command[:]=[vx,0,yaw,height,0]
                        env.command_target[:]=env.command
                        env.command_steps_left=env.stage.episode_steps+1
                    oldobs=old._obs(current_action=old.previous_action)
                    newobs=new._obs(current_action=new.previous_action)
                action,_=policy.predict(norm.normalize_obs(oldobs),deterministic=True)
                oldobs,r0,d0,tr0,i0=old.step(action)
                newobs,r1,d1,tr1,i1=new.step(action)
                error=max(float(np.max(np.abs(oldobs-newobs))),
                          float(np.max(np.abs(old.sim.data.qpos-new.sim.data.qpos))))
                max_state_error=max(max_state_error,error)
                if error>1e-10 or (d0,tr0)!=(d1,tr1):
                    raise AssertionError('reward-only change unexpectedly altered observations/physics')
                positive=sum(i0['reward_terms'][k] for k in ('upright','pitch_alignment','track_vx','track_vx_coarse','track_yaw','track_height'))
                rows.append([r0,r1,positive,
                             sum(i1['reward_terms'][k] for k in ('leg_forward_reach','leg_backward_reach')),
                             max(i1[f'leg_forward_reach_{side}_m'] for side in ('left','right')),
                             max(i1[f'leg_backward_reach_{side}_m'] for side in ('left','right'))])
                if d0 or tr0:break
            finish()
            results.append({'profile':name,'steps':step+1,'reason':i0['termination_reason'],'segments':segments})
    finally:
        norm.close();new.close()
    report={'kind':'offline_frozen_policy_reward_audit_not_v11_performance',
            'checkpoint':str(checkpoint),'seed':args.seed,'max_state_or_observation_error':max_state_error,
            'results':results}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
