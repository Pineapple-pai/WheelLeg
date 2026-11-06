"""Paired velocity-error scale experiment; reward RMS reset in both arms."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import hashlib,json,shlex,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'diagnostics/startup_precision_ab'
SOURCE=ROOT/'checkpoints/joint_stand_translation_20261105_161432/checkpoint_iter_1000'
ARMS={'control':'joint_tracking_startup','precision':'joint_tracking_startup_precision'}
SEEDS=[91,92,93];EVAL_SEEDS=['65200','65201','65202']
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def eval_command(cp,folder):return [sys.executable,str(ROOT/'scripts/eval_new_model_training.py'),str(cp),'--output',str(folder),'--launch-speeds','.1','.25','.5','--seeds',*EVAL_SEEDS]
def prepare():
 from uz05.spec import MODEL_XML
 protected=[Path(str(SOURCE)+s) for s in ('.zip','_vecnormalize.pkl','_contract.json')]+[MODEL_XML,*[ROOT/f'scripts/uz05/{n}.py' for n in ('spec','env','actuators','command_sampling')],ROOT/'scripts/train_uz05.py']
 runs=[]
 for seed in SEEDS:
  for arm,stage in ARMS.items():
   folder=OUT/f'seed_{seed}'/arm;version=f'startup_precision_ab_{arm}_s{seed}_20261105';cp=ROOT/'checkpoints'/version/'checkpoint'
   cmd=[sys.executable,'-u',str(ROOT/'scripts/train_uz05.py'),'--stage',stage,'--command-sampling','startup_balanced',
    '--checkpoint',str(SOURCE),'--preserve-action-std','--reset-vecnormalize','--version',version,
    '--updates','100','--rollout-steps','256','--batch-size','256','--num-envs','4','--vec-env','subproc',
    '--device','cpu','--learning-rate','.0001','--disable-adaptive-lr','--entropy-coef','.001',
    '--seed',str(seed),'--save-interval','50','--log-interval','10',
    '--initial-state-audit',str(folder/'initial_state.json'),'--tensorboard-log',str(folder/'tensorboard')]
   runs.append(dict(seed=seed,arm=arm,directory=str(folder),checkpoint=str(cp),command=cmd,evaluate=eval_command(cp,folder/'evaluation')))
 cfg=dict(source=str(SOURCE),protected={str(p):digest(p) for p in protected},runs=runs,seeds=SEEDS,
  baseline=eval_command(SOURCE,OUT/'source_evaluation'),evaluation_seeds=EVAL_SEEDS,
  intervention='Only active vx_error_scale_m_s0.15->0.05. Physics, objectives, sampler, action mapping, optimizer settings unchanged.',
  normalization='Both arms reset reward RMS to separate reward-scale effect from reset effect; source Actor/Critic/Adam/std retained.',
  budget='100updates*4env*256=102400transitions per arm/seed;614400total.',
  gate='All6nonzero tasks combined3/3 in>=2training seeds and stand3/3 in all3; no automatic speed expansion.',
  adoption='Report all seeds. Retain candidate only if speed benefit replicates with standing and posture retained; no lucky-seed promotion.')
 OUT.mkdir(exist_ok=True);p=OUT/'experiment_config.json'
 if p.exists():assert json.loads(p.read_text())==cfg
 else:p.write_text(json.dumps(cfg,indent=2)+'\n')
 (OUT/'commands.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd '+shlex.quote(str(ROOT))+'\n'+'\n'.join(shlex.join(c) for c in [cfg['baseline'],*[c for r in runs for c in (r['command'],r['evaluate'])]])+'\n')
 return cfg

def evaluate(command,log):
 folder=Path(command[command.index('--output')+1]);p=folder/'results.json'
 if p.exists():
  cfg=json.loads(p.read_text())['config'];assert cfg['checkpoint']==command[2]
  for s,h in cfg['protected'].items():assert digest(command[2]+s)==h
 else:
  with log.open('w') as stream:subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)

def execute(run):
 folder=Path(run['directory']);folder.mkdir(parents=True,exist_ok=True);cp=run['checkpoint'];p=folder/'training_completed.json'
 if p.exists():
  for s,h in json.loads(p.read_text()).items():assert digest(cp+s)==h
 else:
  assert not Path(cp+'.zip').exists(),'Refuse overwrite'
  with (folder/'train.log').open('w') as stream:subprocess.run(run['command'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
  p.write_text(json.dumps({s:digest(cp+s) for s in ('.zip','_vecnormalize.pkl','_contract.json')})+'\n')
 evaluate(run['evaluate'],folder/'evaluation.log');print('Completed',run['seed'],run['arm'],flush=True)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');args=p.parse_args();cfg=prepare()
 if not args.execute:print('Frozen plan:',OUT/'experiment_config.json');return
 evaluate(cfg['baseline'],OUT/'source_evaluation.log')
 with ThreadPoolExecutor(max_workers=2) as pool:
  for f in as_completed([pool.submit(execute,r) for r in cfg['runs']]):f.result()
 assert all(digest(p)==h for p,h in cfg['protected'].items())
 (OUT/'integrity.json').write_text(json.dumps(dict(protected_unchanged=True))+'\n')
if __name__=='__main__':main()
