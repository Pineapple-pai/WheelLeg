"""Read-only preflight for source state under each frozen adaptation setup."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','TORCH_NUM_THREADS'):os.environ[k]='1'
import json
from pathlib import Path
import torch
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
from train_uz05 import TrainingPPO,make_env,require_compatible_contract
from uz05.experiment_audit import initial_state_audit
from run_ppo_v17_torque_ab import prepare,OUT

def main():
    torch.set_num_threads(1);cfg=prepare();audits=[]
    for run in cfg['runs']:
        require_compatible_contract(Path(cfg['source_checkpoint']),'joint_tracking_v16',run['mode'],run['mode']=='B_rated')
        v=DummyVecEnv([make_env('joint_tracking_v16',i,run['seed'],0,wheel_torque_model=run['mode']) for i in range(8)])
        env=VecNormalize.load(cfg['source_checkpoint']+'_vecnormalize.pkl',v);env.training=True;env.norm_reward=True
        model=TrainingPPO.load(cfg['source_checkpoint'],env=env,device='cpu',seed=run['seed'],n_steps=512,batch_size=512,
          learning_rate=3e-5,gamma=.997,gae_lambda=.97,ent_coef=.001)
        model.learning_rate=3e-5
        for group in model.policy.optimizer.param_groups:group['lr']=3e-5
        path=Path(run['directory'])/'preflight_state.json';initial_state_audit(model,env,path)
        a=json.loads(path.read_text());audits.append(a)
        env.close()
    assert all(a==audits[0] for a in audits)
    actual=OUT/'seed_1/A_legacy/initial_state.json'
    assert json.loads(actual.read_text())==audits[0]
    (OUT/'preflight_verification.json').write_text(json.dumps({'all_six_source_states_equal':True,'matches_actual_first_training_initial_state':True,'state':audits[0]},indent=2)+'\n')
    print('Verified all6sourceActorCritic/Adam/normalization/schedules/counts; exact match actual initial state')
if __name__=='__main__':main()
