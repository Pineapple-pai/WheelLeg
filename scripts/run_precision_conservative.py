"""Single-factor learning-rate follow-up against frozen 5e-5 balanced controls."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import json,sys
from run_startup_precision_ab import digest,execute
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'diagnostics/precision_conservative'

def main():
 old=json.loads((ROOT/'diagnostics/precision_ramp_ab/experiment_config.json').read_text());OUT.mkdir(exist_ok=True);runs=[]
 for r in old['runs']:
  if r['arm']!='control':continue
  seed=r['seed'];folder=OUT/f'seed_{seed}';cp=ROOT/'checkpoints'/f'precision_conservative_s{seed}_20261105'/'checkpoint'
  command=r['command'].copy()
  for key,value in {'--version':cp.parent.name,'--learning-rate':'.00001','--initial-state-audit':str(folder/'initial_state.json'),'--tensorboard-log':str(folder/'tensorboard')}.items():command[command.index(key)+1]=value
  evaluate=r['evaluate'].copy();evaluate[2]=str(cp);evaluate[evaluate.index('--output')+1]=str(folder/'evaluation')
  runs.append(dict(seed=seed,arm='conservative',directory=str(folder),checkpoint=str(cp),command=command,evaluate=evaluate))
 config=dict(runs=runs,protected=old['protected'],source=old['source'],intervention='Only LR5e-5->1e-5 versus balanced control; same seed/source/RMS/std/Adam/goals.',budget='3x100x4x256=307200transitions',
  gate='Report all3seeds; source +0.25 posture and velocity must not be silently traded away. No best-seed promotion.')
 p=OUT/'experiment_config.json'
 if p.exists():assert json.loads(p.read_text())==config
 else:p.write_text(json.dumps(config,indent=2)+'\n')
 with ThreadPoolExecutor(max_workers=2) as pool:
  for f in as_completed([pool.submit(execute,r) for r in runs]):f.result()
 assert all(digest(p)==h for p,h in config['protected'].items())
 (OUT/'integrity.json').write_text(json.dumps(dict(protected_unchanged=True))+'\n')
if __name__=='__main__':main()
