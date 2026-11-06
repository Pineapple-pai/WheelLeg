"""Explicit parameter initialization on a new plant; never relabel a source checkpoint."""
import argparse,hashlib,json,os
from pathlib import Path
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(key,'1')
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
from train_uz05 import TrainingPPO,make_env,write_contract,require_compatible_contract
from uz05.spec import REPO_ROOT,MODEL_XML,ORIGINAL_MODEL_XML
from uz05.experiment_audit import fingerprint

INTERFACE_KEYS=('contract_version','actuator_config','observation_reference','observation_shape',
 'actor_observation_shape','critic_privileged_shape','action_shape','actor_blocks',
 'critic_privileged_blocks','observation_scale','action_blocks','precision_objective')
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def validate_interface(source,target):
 mismatch=[k for k in INTERFACE_KEYS if json.dumps(source.get(k),sort_keys=True)!=json.dumps(target.get(k),sort_keys=True)]
 if mismatch:raise ValueError('Transfer requires identical interface, actuator map and objective: '+', '.join(mismatch))

def initialize(source,version,allow_cross_model_initialization=False,reset_optimizer=False):
 if not allow_cross_model_initialization:raise ValueError('Explicit --allow-cross-model-initialization is required')
 if not version or Path(version).name!=version or version in ('.','..'):raise ValueError('version must be one directory name')
 source=Path(source).resolve()
 if source.suffix=='.zip':source=source.with_suffix('')
 protected={str(source)+s:digest(str(source)+s) for s in ('.zip','_contract.json','_vecnormalize.pkl')}
 contract=json.loads(Path(str(source)+'_contract.json').read_text())
 known={digest(p):str(p) for p in (REPO_ROOT/'uz05_修正后(3).xml',ORIGINAL_MODEL_XML,MODEL_XML)}
 if contract['model_xml_sha256'] not in known:raise ValueError('Source plant is not one of the audited local models')
 if contract['model_xml_sha256']==digest(MODEL_XML):raise ValueError('Same plant: use ordinary --checkpoint continuation')
 stage=('joint_tracking_startup_precision' if contract['stage']=='joint_tracking_startup_precision' else 'joint_tracking_startup')
 folder=REPO_ROOT/'checkpoints'/version
 if folder.exists():raise FileExistsError(f'Refuse overwrite: {folder}')
 folder.mkdir();target=folder/'checkpoint';norm=None
 try:
  target_contract_path=write_contract(target,stage,0,version,command_sampling='startup_balanced')
  target_contract=json.loads(target_contract_path.read_text());validate_interface(contract,target_contract)
  env=DummyVecEnv([make_env(stage,0,0,0,command_sampling='startup_balanced')])
  norm=VecNormalize.load(str(source)+'_vecnormalize.pkl',env)
  if norm.norm_obs:raise ValueError('This transfer supports the audited non-normalized observations only')
  norm.training=True
  model=TrainingPPO.load(str(source),env=norm,device='cpu')
  actor_before=fingerprint(model.policy.state_dict());source_adam=fingerprint(model.policy.optimizer.state_dict())
  if reset_optimizer:model.policy.optimizer.state.clear()
  adam=fingerprint(model.policy.optimizer.state_dict());rms=fingerprint(vars(norm.ret_rms))
  provenance=dict(operation='explicit_cross_model_parameter_initialization',source_checkpoint=str(source),
   source_model_path=known[contract['model_xml_sha256']],source_model_sha256=contract['model_xml_sha256'],
   target_model_path=str(MODEL_XML),target_model_sha256=digest(MODEL_XML),source_files=protected,
   source_timesteps=int(model.num_timesteps),new_plant_training_transitions=0,
   source_policy_sha256=actor_before,source_optimizer_sha256=source_adam,optimizer_sha256=adam,reward_rms_sha256=rms,
   reset_optimizer=bool(reset_optimizer),
   retained=['Actor','Critic','six action stds','reward RMS']+([] if reset_optimizer else ['Adam moments and counters']),
   restarted=['target environment states and RNG','rollout buffer','target-plant transition counter']+(['Adam moments and counters'] if reset_optimizer else []),
   not_claimed=['source trained on target plant','physical equivalence','post-transfer success'],
   source_stage=contract['stage'],target_stage=stage,
   target_sampler='startup_balanced',interface_and_objective_keys_checked=list(INTERFACE_KEYS))
  model.initialization_provenance=provenance
  model.num_timesteps=0;model._total_timesteps=0;model._current_progress_remaining=1.
  model.save(str(target));norm.save(str(target)+'_vecnormalize.pkl')
  target_contract['initialization_provenance']=provenance
  target_contract_path.write_text(json.dumps(target_contract,indent=2,ensure_ascii=False)+'\n')
  require_compatible_contract(target,stage)
  restored=TrainingPPO.load(str(target),device='cpu')
  assert fingerprint(restored.policy.state_dict())==actor_before
  assert fingerprint(restored.policy.optimizer.state_dict())==adam
  assert restored.num_timesteps==0
  assert all(digest(p)==h for p,h in protected.items())
  provenance['output_files']={s:digest(str(target)+s) for s in ('.zip','_contract.json','_vecnormalize.pkl')}
  Path(str(target)+'_initialization.json').write_text(json.dumps(provenance,indent=2,ensure_ascii=False)+'\n')
  print('Initialized, not trained:',target,flush=True)
  return target
 except Exception:
  # Retain any partial output for diagnosis, never mistake it for a complete initialization.
  (folder/'INITIALIZATION_FAILED').write_text('Initialization did not complete. Do not train from this directory.\n')
  raise
 finally:
  if norm is not None:norm.close()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--version',required=True)
 p.add_argument('--allow-cross-model-initialization',action='store_true');p.add_argument('--reset-optimizer',action='store_true',help='Explicitly discard source Adam moments/counters while retaining Actor/Critic/std/RMS')
 a=p.parse_args();initialize(a.source,a.version,a.allow_cross_model_initialization,a.reset_optimizer)
if __name__=='__main__':main()
