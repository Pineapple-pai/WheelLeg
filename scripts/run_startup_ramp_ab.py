"""Paired continuation test: preserve reward RMS/Adam/std; change reference sampling only."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import json,sys
from run_startup_precision_ab import digest,execute
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'diagnostics/precision_ramp_ab'
SOURCE=ROOT/'checkpoints/joint_precision_20261105_184905/checkpoint'

def main():
 OUT.mkdir(exist_ok=True);runs=[]
 for seed in (101,102,103):
  for arm,sampler in [('control','startup_balanced'),('mixed_ramp','startup_mixed_ramp')]:
   folder=OUT/f'seed_{seed}'/arm;cp=ROOT/'checkpoints'/f'precision_ramp_ab_{arm}_s{seed}_20261105'/'checkpoint'
   command=[sys.executable,'-u',str(ROOT/'scripts/train_uz05.py'),'--stage','joint_tracking_startup_precision',
    '--command-sampling',sampler,'--checkpoint',str(SOURCE),'--preserve-action-std',
    '--vecnormalize-path',str(SOURCE)+'_vecnormalize.pkl','--version',cp.parent.name,
    '--updates','100','--rollout-steps','256','--batch-size','256','--num-envs','4','--vec-env','subproc',
    '--device','cpu','--learning-rate','.00005','--disable-adaptive-lr','--entropy-coef','.001',
    '--seed',str(seed),'--save-interval','100','--log-interval','10',
    '--initial-state-audit',str(folder/'initial_state.json'),'--tensorboard-log',str(folder/'tensorboard')]
   evaluate=[sys.executable,str(ROOT/'scripts/eval_new_model_training.py'),str(cp),'--output',str(folder/'evaluation'),
    '--launch-speeds','.1','.25','.5','--seeds','65400','65401','65402']
   runs.append(dict(seed=seed,arm=arm,directory=str(folder),checkpoint=str(cp),command=command,evaluate=evaluate))
 protected=[Path(str(SOURCE)+s) for s in ('.zip','_vecnormalize.pkl','_contract.json')]
 protected += [ROOT/f'scripts/uz05/{s}.py' for s in ('env','spec','actuators','command_sampling')]
 config=dict(source=str(SOURCE),runs=runs,protected={str(p):digest(p) for p in protected},
  intervention='Same goals/probabilities and rewards; half launches use0.25m/s^2 reference ramp, half step.',
  budget='6x100x4x256=614400additional transitions',
  evaluation='Unchanged step commands, nominal with sensor noise; seeds65400..402, all7tasks. No ramp in acceptance tests.',
  gate='Retain only if improvement repeats across >=2training seeds; report all seeds and forward/stand regressions. No lucky seed promotion.')
 p=OUT/'experiment_config.json'
 if p.exists():assert json.loads(p.read_text())==config
 else:p.write_text(json.dumps(config,indent=2)+'\n')
 with ThreadPoolExecutor(max_workers=2) as pool:
  for f in as_completed([pool.submit(execute,r) for r in runs]):f.result()
 assert all(digest(p)==h for p,h in config['protected'].items())
 (OUT/'integrity.json').write_text(json.dumps(dict(protected_unchanged=True))+'\n')
if __name__=='__main__':main()
