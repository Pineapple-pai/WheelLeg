"""Pure PPO reward/exploration factorial; no teacher, calibration or action wrapper."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,sys,argparse
from run_startup_precision_ab import execute,evaluate
from initialize_model_transfer import digest
from uz05.spec import MODEL_XML
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'diagnostics/pure_ppo_precision_20261106'
SOURCE=ROOT/'checkpoints/armature001_transfer_legacy_20261106/checkpoint'

def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=711)
 p.add_argument('--arms',nargs='+',choices=['control','wheel_fine','reward_fine','both_fine'],
                default=['control','wheel_fine','reward_fine','both_fine']);a=p.parse_args()
 OUT.mkdir(exist_ok=True)
 runs=[]
 for arm,stage,wstd in [('control','joint_tracking_startup_precision','.02'),
   ('wheel_fine','joint_tracking_startup_precision','.003'),
   ('reward_fine','joint_tracking_startup_precision_fine','.02'),
   ('both_fine','joint_tracking_startup_precision_fine','.003')]:
  if arm not in a.arms:continue
  folder=OUT/f'seed_{a.seed}'/arm;version=f'pure_ppo_{arm}_s{a.seed}_20261106';cp=ROOT/'checkpoints'/version/'checkpoint'
  if cp.parent.exists():raise FileExistsError(cp.parent)
  cmd=[sys.executable,'-u',str(ROOT/'scripts/train_uz05.py'),'--stage',stage,'--command-sampling','startup_balanced',
    '--checkpoint',str(SOURCE),'--action-std','.02','--wheel-action-std',wstd,'--reset-vecnormalize','--version',version,
    '--updates','200','--rollout-steps','512','--batch-size','512','--num-envs','8','--vec-env','subproc','--device','cpu',
    '--learning-rate','.00001','--disable-adaptive-lr','--entropy-coef','0','--seed',str(a.seed),
    '--save-interval','50','--log-interval','25','--initial-state-audit',str(folder/'initial_state.json'),
    '--tensorboard-log',str(folder/'tensorboard')]
  ev=[sys.executable,str(ROOT/'scripts/eval_new_model_training.py'),str(cp),'--output',str(folder/'evaluation'),
      '--launch-speeds','.1','.25','.5','--seeds','66600','66601','66602']
  runs.append(dict(seed=a.seed,arm=arm,directory=str(folder),checkpoint=str(cp),command=cmd,evaluate=ev))
 protected=[MODEL_XML,*[Path(str(SOURCE)+s) for s in ('.zip','_vecnormalize.pkl','_contract.json')],
  *[ROOT/f'scripts/uz05/{s}.py' for s in ('spec','env','actuators','command_sampling')],ROOT/'scripts/train_uz05.py']
 cfg=dict(runs=runs,protected={str(p):digest(p) for p in protected},
   intervention='2x2 actual velocity reward scale0.05/0.01 and initial wheel std0.02/0.003; four leg std0.02. All learned freely by PPO.',
   fixed='Same Actor/Critic/Adam; reward RMS reset in all arms; lr1e-5, entropy0,8x512x200. Noise and domain randomization ON.',
   prohibited='No teacher, action targets, command remap, residual, extra controller, action filter or frozen action channels.',
   gate='Screen all arms, report failures. Full task acceptance and independent replication required before long-run recommendation.')
 f=OUT/f'plan_s{a.seed}.json';f.write_text(json.dumps(cfg,indent=2)+'\n')
 with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(execute,runs))
 assert all(digest(p)==h for p,h in cfg['protected'].items())
 (OUT/f'integrity_s{a.seed}.json').write_text(json.dumps(dict(protected_unchanged=True,new_transitions=len(runs)*200*8*512))+'\n')

if __name__=='__main__':main()
