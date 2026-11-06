"""Sampling-only follow-up to the frozen velocity-scale trial."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import json,shlex,sys
from run_startup_precision_ab import digest,eval_command,execute
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'diagnostics/startup_precision_low'
SOURCE=ROOT/'checkpoints/joint_stand_translation_20261105_161432/checkpoint_iter_1000'

def prepare():
 from uz05.spec import MODEL_XML
 from uz05.command_sampling import sampling_contract
 runs=[]
 for seed in (91,92,93):
  folder=OUT/f'seed_{seed}';cp=ROOT/'checkpoints'/f'startup_precision_low_s{seed}_20261105'/'checkpoint'
  cmd=[sys.executable,'-u',str(ROOT/'scripts/train_uz05.py'),'--stage','joint_tracking_startup_precision',
   '--command-sampling','startup_low','--checkpoint',str(SOURCE),'--preserve-action-std','--reset-vecnormalize',
   '--version',cp.parent.name,'--updates','100','--rollout-steps','256','--batch-size','256',
   '--num-envs','4','--vec-env','subproc','--device','cpu','--learning-rate','.0001','--disable-adaptive-lr',
   '--entropy-coef','.001','--seed',str(seed),'--save-interval','50','--log-interval','10',
   '--initial-state-audit',str(folder/'initial_state.json'),'--tensorboard-log',str(folder/'tensorboard')]
  runs.append(dict(seed=seed,arm='precision_low',directory=str(folder),checkpoint=str(cp),command=cmd,evaluate=eval_command(cp,folder/'evaluation')))
 protected=[MODEL_XML,*[Path(str(SOURCE)+s) for s in ('.zip','_vecnormalize.pkl','_contract.json')],
  *[ROOT/f'scripts/uz05/{n}.py' for n in ('spec','env','actuators','command_sampling')],ROOT/'scripts/train_uz05.py']
 cfg=dict(source=str(SOURCE),protected={str(p):digest(p) for p in protected},runs=runs,sampling=sampling_contract('startup_low'),
  comparison='Compare to precision arm of startup_precision_ab on paired train/eval seeds. Only sampling changes; source policy/Adam/std/freshRMS and optimizer budget identical.',
  budget=307200,eval_seeds=[65200,65201,65202],
  gate='All4trained low tasks combined3/3 in>=2training seeds, stand3/3 in all3. +/-0.5 remains out-of-training regression only; no speed expansion unless low gate passes.')
 OUT.mkdir(exist_ok=True);p=OUT/'experiment_config.json'
 if p.exists():assert json.loads(p.read_text())==cfg
 else:p.write_text(json.dumps(cfg,indent=2)+'\n')
 (OUT/'commands.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd '+shlex.quote(str(ROOT))+'\n'+'\n'.join(shlex.join(c) for r in runs for c in (r['command'],r['evaluate']))+'\n')
 return cfg

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');args=p.parse_args();cfg=prepare()
 if not args.execute:print('Frozen plan:',OUT/'experiment_config.json');return
 with ThreadPoolExecutor(max_workers=2) as pool:
  for f in as_completed([pool.submit(execute,r) for r in cfg['runs']]):f.result()
 assert all(digest(p)==h for p,h in cfg['protected'].items())
 (OUT/'integrity.json').write_text(json.dumps(dict(protected_unchanged=True))+'\n')
if __name__=='__main__':main()
