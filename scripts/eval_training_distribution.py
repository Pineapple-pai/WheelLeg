"""Deterministic-policy evaluation using the unchanged training distribution."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
from train_uz05 import require_compatible_contract,environment_contract
from uz05.env import UZ05Env
from uz05.spec import env_params_for_stage

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--episodes',type=int,default=30)
    args=parser.parse_args();torch.set_num_threads(1)
    cp=str(args.checkpoint);require_compatible_contract(args.checkpoint,'joint_tracking_v16')
    contract=json.loads(Path(cp+'_contract.json').read_text())
    expected=json.loads(json.dumps(environment_contract('joint_tracking_v16',True,True)))
    assert contract['environment_config']==expected
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'results.json').exists():raise RuntimeError('Refuse result overwrite')
    paths=[Path(cp+s) for s in ['.zip','_vecnormalize.pkl','_contract.json']]
    protected={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    e=UZ05Env('joint_tracking_v16',params=env_params_for_stage('joint_tracking_v16'))
    norm=VecNormalize.load(cp+'_vecnormalize.pkl',DummyVecEnv([lambda:e]))
    norm.training=False;norm.norm_reward=False
    policy=PPO.load(cp,device='cpu');rows=[]
    columns=['time_s','vx','command_vx','pitch_rad','roll_rad','height_error_m','push_force_n','command_yaw','yaw_error']
    for seed in range(63000,63000+args.episodes):
        obs,_=e.reset(seed=seed);samples=[]
        initial_vx=float(e.sim.body_frame(e.sim.base_lin_vel_world)[0])
        for step in range(e.stage.episode_steps):
            action,_=policy.predict(norm.normalize_obs(obs),deterministic=True)
            obs,reward,term,trunc,info=e.step(action)
            assert np.isfinite(obs).all() and np.isfinite(reward)
            samples.append([(step+1)*.008,info['tracking_vx_error']+e.command[0],e.command[0],
                info['pitch'],info['roll'],info['tracking_height_error'],info['push_force_n'],e.command[2],info['tracking_yaw_error']])
            if term or trunc:break
        a=np.asarray(samples)
        rows.append(dict(seed=seed,survival_s=float(a[-1,0]),reason=info['termination_reason'],
            initial_vx=initial_vx,vx_rmse=float(np.sqrt(np.mean((a[:,1]-a[:,2])**2))),
            pitch_rms_deg=float(np.degrees(np.sqrt(np.mean(a[:,3]**2)))),
            height_mae_m=float(np.mean(np.abs(a[:,5]))),push_fraction=float(np.mean(a[:,6]>0))))
        np.savez_compressed(args.output/f'{seed}.npz',samples=a,columns=columns)
        print(seed,rows[-1]['survival_s'],rows[-1]['reason'],flush=True)
    for p in paths:assert hashlib.sha256(p.read_bytes()).hexdigest()==protected[str(p)]
    (args.output/'results.json').write_text(json.dumps(dict(checkpoint=cp,deterministic=True,
        no_learning=True,training_environment_contract_matched=True,protected=protected,
        domain_randomization=True,moving_start_probability=e.stage.moving_start_probability,
        pushes_enabled=e.params.robust.pushes_enabled,episodes=rows),indent=2)+'\n')
    norm.close()

if __name__=='__main__':main()
