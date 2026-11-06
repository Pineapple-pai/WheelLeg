"""Re-score one frozen-policy trajectory with all four minimal reward variants."""
from dataclasses import replace
import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from train_uz05 import PPO, require_compatible_contract
from eval_joint_tracking_transitions import PROFILES
from uz05.env import UZ05Env
from uz05.spec import SimpleTrackingRewards
from uz05.simple_rewards import tracking_terms


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint',type=Path)
    p.add_argument('--profiles',nargs='+',choices=list(PROFILES),default=['reverse10','motion'])
    p.add_argument('--seed',type=int,default=30000)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    checkpoint=args.checkpoint.expanduser().resolve().with_suffix('')
    contract=json.loads(Path(f'{checkpoint}_contract.json').read_text())
    stage=contract['stage'];require_compatible_contract(checkpoint,stage)
    env=UZ05Env(stage);env.params.domain_randomization.enabled=False;env.params.robust.pushes_enabled=False
    env.stage=replace(env.stage,moving_start_probability=0.,episode_steps=3000)
    norm=VecNormalize.load(f'{checkpoint}_vecnormalize.pkl',DummyVecEnv([lambda:env]))
    norm.training=False;norm.norm_reward=False
    policy=PPO.load(checkpoint,device='cpu')
    results=[];max_invariant_error=0.
    try:
        for name in args.profiles:
            obs,_=env.reset(seed=args.seed);phase=-1;schedule=PROFILES[name]
            rates={key:[] for key in ('minimal','minimal_axle','support','support_axle')}
            failures={key:0. for key in rates}
            for step in range(3000):
                t=step*env.params.control_dt
                next_phase=max(i for i,s in enumerate(schedule) if t>=s[0])
                if phase!=next_phase:
                    phase=next_phase;_,vx,yaw,height=schedule[phase]
                    env.command[:]=[vx,0,yaw,height,0];env.command_target[:]=env.command
                    env.command_steps_left=3001;obs=env._obs(current_action=env.previous_action)
                previous=env.previous_action.copy()
                action,_=policy.predict(norm.normalize_obs(obs),deterministic=True)
                obs,_,done,truncated,info=env.step(action)
                delta=action-previous
                delta[2:]=np.arctan2(np.sin(np.pi*delta[2:]),np.cos(np.pi*delta[2:]))/np.pi
                scores={}
                for arm in rates:
                    terms=tracking_terms(SimpleTrackingRewards(),vx_error=info['tracking_vx_error'],
                        yaw_rate_error=info['tracking_yaw_error'],height_error=info['tracking_height_error'],
                        roll=info['roll'],pitch_error=info['support_pitch_error_rad'] if arm.startswith('support') else info['pitch'],
                        heading_error=env._heading_error() if abs(env.command[2])<=.01 else 0.,
                        stop_distance=info['hold_planar_error_m'],action_delta=delta,terminated=done,
                        axle_offset=info['axle_line_offset_m'],axle_enabled=arm.endswith('_axle'),dt=env.params.control_dt)
                    failures[arm]+=terms.pop('failure')
                    scores[arm]={k:v/env.params.control_dt for k,v in terms.items()}
                    rates[arm].append(scores[arm])
                for key in ('task_tracking','action_smoothness'):
                    max_invariant_error=max(max_invariant_error,
                        max(v[key] for v in scores.values())-min(v[key] for v in scores.values()))
                for a,b in (('minimal','minimal_axle'),('support','support_axle')):
                    max_invariant_error=max(max_invariant_error,abs(scores[a]['body_attitude']-scores[b]['body_attitude']))
                if done or truncated:break
            summaries={}
            for arm,rows in rates.items():
                summaries[arm]={'continuous_reward_rates':{key:{
                    'mean':float(np.mean([row[key] for row in rows])),
                    'min':float(np.min([row[key] for row in rows])),
                    'max':float(np.max([row[key] for row in rows]))} for key in rows[0]},
                    'terminal_cost_total':failures[arm]}
            results.append({'profile':name,'steps':step+1,'reason':info['termination_reason'],'arms':summaries})
            print(f'{name}: {step+1} steps {info["termination_reason"]}',flush=True)
    finally:norm.close()
    if max_invariant_error>1e-12:raise AssertionError('comparison changed a non-factor reward group')
    report={'kind':'frozen_policy_rescoring_not_trained_policy_performance','checkpoint':str(checkpoint),
            'seed':args.seed,'max_nonfactor_reward_rate_difference':max_invariant_error,'results':results}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
