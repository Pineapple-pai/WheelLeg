"""Continue all three replicated precision candidates; do not select a lucky seed."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import json,sys,time
from initialize_model_transfer import digest
from run_startup_precision_ab import execute
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'diagnostics/armature001_precision_continuation'
def main():
 OUT.mkdir(exist_ok=True);old=json.loads((ROOT/'diagnostics/armature001_transfer_precision_ab/experiment_config.json').read_text());runs=[]
 for r in old['runs']:
  if r['arm']!='precision_reset':continue
  lineage=r['seed'];folder=OUT/f'seed_{lineage}';cp=ROOT/'checkpoints'/f'armature001_precision_cont_s{lineage}_20261106'/'checkpoint';cmd=r['command'].copy()
  cmd.remove('--reset-vecnormalize');i=cmd.index('--action-std');del cmd[i:i+2]
  for key,value in {'--checkpoint':r['checkpoint'],'--version':cp.parent.name,'--updates':'150','--seed':str(lineage+1000),'--save-interval':'50','--initial-state-audit':str(folder/'initial_state.json'),'--tensorboard-log':str(folder/'tensorboard')}.items():cmd[cmd.index(key)+1]=value
  cmd+=['--preserve-action-std','--vecnormalize-path',r['checkpoint']+'_vecnormalize.pkl']
  ev=r['evaluate'].copy();ev[2]=str(cp);ev[ev.index('--output')+1]=str(folder/'evaluation')
  runs.append(dict(seed=lineage,arm='continued_precision',directory=str(folder),checkpoint=str(cp),command=cmd,evaluate=ev,
   source=r['checkpoint'],protected={s:digest(r['checkpoint']+s) for s in ('.zip','_contract.json','_vecnormalize.pkl')}))
 cfg=dict(runs=runs,protected=old['protected'],source_phase='precision_reset50updates, all3lineages retained',
  budget='3x150x8x256=921600additional transitions; each lineage total409600over50+150iterations.',
  interpretation='Fixed objective and plant; keep learned std/Adam/RMS. Explicit restart with fresh environment seed1201..1203; not bit-for-bit equivalent to a monolithic200-iteration run.',
  gate='Report all3lineages, keep all standing/bidirectional posture coverage, require replicated speed progress. Domain-randomization holdout follows; no best-seed promotion.')
 p=OUT/'experiment_config.json'
 if p.exists():assert json.loads(p.read_text())==cfg
 else:p.write_text(json.dumps(cfg,indent=2)+'\n')
 print('Frozen150iteration continuation; waiting for the optimizer control.',flush=True)
 for _ in range(120):
  if (ROOT/'diagnostics/armature001_freshadam/integrity.json').exists():break
  time.sleep(5)
 else:raise TimeoutError('Optimizer comparison did not finish in10minutes')
 with ThreadPoolExecutor(max_workers=2) as pool:
  for f in as_completed([pool.submit(execute,r) for r in runs]):f.result()
 assert all(digest(p)==h for p,h in cfg['protected'].items())
 for run in runs:
  assert all(digest(run['source']+s)==h for s,h in run['protected'].items())
 (OUT/'integrity.json').write_text(json.dumps(dict(protected_unchanged=True,new_training_transitions=921600))+'\n')
if __name__=='__main__':main()
