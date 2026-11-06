"""Frozen snapshot/initialization review on the target0.001 plant; no learning."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import json,subprocess,sys
from initialize_model_transfer import digest
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'diagnostics/armature001_training_review'
def main():
 jobs=[];source=ROOT/'checkpoints/joint_startup_armature001_20261106_080550/checkpoint'
 for step in [250,500,750]:
  jobs.append((f'scratch_iter_{step}',Path(str(source)+f'_iter_{step}'),['--tasks','stand']))
 for label in ['legacy','stand','precision']:
  jobs.append((f'transfer_{label}',ROOT/f'checkpoints/armature001_transfer_{label}_20261106/checkpoint',['--launch-speeds','.1','.25','.5']))
 cfg=dict(seeds=[66000,66001,66002],jobs=[dict(name=n,checkpoint=str(cp),args=args,protected={s:digest(str(cp)+s) for s in ('.zip','_contract.json','_vecnormalize.pkl')}) for n,cp,args in jobs],
  protocol='Same nominal target0.001 model; sensor noise retained;24s; unchanged5% score. Scratch snapshots stand only to separate prelaunch failures.',
  gate='Choose initialization by standing and bidirectional posture coverage, report speed errors and all sources. No training or best-evaluation-seed selection.')
 p=OUT/'review_plan.json'
 if p.exists():assert json.loads(p.read_text())==cfg
 else:p.write_text(json.dumps(cfg,indent=2)+'\n')
 def run(job):
  folder=OUT/job['name'];result=folder/'results.json'
  if not result.exists():
   cmd=[sys.executable,str(ROOT/'scripts/eval_new_model_training.py'),job['checkpoint'],'--output',str(folder),*job['args'],'--seeds',*map(str,cfg['seeds'])]
   with (OUT/(job['name']+'.log')).open('w') as f:subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
  for s,h in job['protected'].items():assert digest(job['checkpoint']+s)==h
  print('Completed',job['name'],flush=True)
 with ThreadPoolExecutor(max_workers=2) as pool:
  for f in as_completed([pool.submit(run,j) for j in cfg['jobs']]):f.result()
 (OUT/'review_integrity.json').write_text(json.dumps(dict(checkpoints_unchanged=True,evaluation_episodes=72))+'\n')
if __name__=='__main__':main()
