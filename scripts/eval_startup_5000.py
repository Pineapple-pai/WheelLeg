"""Assess interrupted startup training checkpoints without learning or overwriting."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'diagnostics/startup_5000_review'
SOURCE=ROOT/'checkpoints/joint_stand_translation_20261105_161432/checkpoint'

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 OUT.mkdir(exist_ok=True);jobs=[]
 for step,dr in [(1000,False),(3000,False),(5000,False),(5000,True)]:
  name=f'iter_{step}'+('_dr' if dr else '')
  cp=str(SOURCE)+f'_iter_{step}';folder=OUT/name
  command=[sys.executable,str(ROOT/'scripts/eval_new_model_training.py'),cp,'--output',str(folder),
   '--launch-speeds','.1','.25','.5','--seeds','65100','65101','65102']+(['--domain-randomization'] if dr else [])
  jobs.append(dict(name=name,checkpoint=cp,command=command,protected={s:digest(cp+s) for s in ('.zip','_vecnormalize.pkl','_contract.json')}))
 plan=dict(jobs=jobs,thresholds='Existing frozen posture and5%tracking thresholds; no changes.',
  source_baseline=str(ROOT/'diagnostics/startup_warmstart/source_evaluation/results.json'),
  interpretation='5101iterations logged; latest persisted snapshot5000. Evaluate stored snapshots, not missing final checkpoint.zip.')
 p=OUT/'experiment_config.json'
 if p.exists():assert json.loads(p.read_text())==plan
 else:p.write_text(json.dumps(plan,indent=2)+'\n')
 def run(job):
  result=OUT/job['name']/'results.json'
  if not result.exists():
   with (OUT/(job['name']+'.log')).open('w') as stream:subprocess.run(job['command'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
  for suffix,h in job['protected'].items():assert digest(job['checkpoint']+suffix)==h
  print('Completed',job['name'],flush=True)
 with ThreadPoolExecutor(max_workers=2) as pool:
  for f in as_completed([pool.submit(run,j) for j in jobs]):f.result()
 (OUT/'integrity.json').write_text(json.dumps(dict(checkpoints_unchanged=True))+'\n')
if __name__=='__main__':main()
