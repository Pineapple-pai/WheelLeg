"""Single-factor stale-optimizer test against the frozen precision/cap/reset-RMS arm."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import json,subprocess,sys,time
from initialize_model_transfer import digest
from run_startup_precision_ab import execute
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'diagnostics/armature001_freshadam'
SOURCE=ROOT/'checkpoints/armature001_transfer_freshadam_20261106/checkpoint'
def main():
 OUT.mkdir(exist_ok=True);old=json.loads((ROOT/'diagnostics/armature001_transfer_precision_ab/experiment_config.json').read_text());runs=[]
 for r in old['runs']:
  if r['arm']!='precision_reset':continue
  seed=r['seed'];folder=OUT/f'seed_{seed}';cp=ROOT/'checkpoints'/f'armature001_freshadam_s{seed}_20261106'/'checkpoint';cmd=r['command'].copy()
  for key,value in {'--checkpoint':str(SOURCE),'--version':cp.parent.name,'--initial-state-audit':str(folder/'initial_state.json'),'--tensorboard-log':str(folder/'tensorboard')}.items():cmd[cmd.index(key)+1]=value
  ev=r['evaluate'].copy();ev[2]=str(cp);ev[ev.index('--output')+1]=str(folder/'evaluation')
  runs.append(dict(seed=seed,arm='freshadam',directory=str(folder),checkpoint=str(cp),command=cmd,evaluate=ev))
 protected=dict(old['protected']);protected.update({str(SOURCE)+s:digest(str(SOURCE)+s) for s in ('.zip','_contract.json','_vecnormalize.pkl','_initialization.json')})
 protected[str(ROOT/'scripts/initialize_model_transfer.py')]=digest(ROOT/'scripts/initialize_model_transfer.py')
 cfg=dict(runs=runs,protected=protected,control='armature001_transfer_precision_ab/precision_reset',
  intervention='Only source Adam moments/counters cleared at explicit initialization; identical mean Actor/Critic/std/initial reset rewardRMS, seed, precision objective and8x256 samples.',
  budget='3x50x8x256=307200transitions,63matched evaluations',gate='Require standing and bidirectional posture retained; velocity gains in>=2training seeds. Report all; no lucky-seed adoption.')
 p=OUT/'experiment_config.json'
 if p.exists():assert json.loads(p.read_text())==cfg
 else:p.write_text(json.dumps(cfg,indent=2)+'\n')
 print('Frozen fresh-Adam comparison; waiting for precision experiment.',flush=True)
 for _ in range(120):
  if (ROOT/'diagnostics/armature001_transfer_precision_ab/integrity.json').exists():break
  time.sleep(5)
 else:raise TimeoutError('Precision experiment not completed in10minutes')
 with ThreadPoolExecutor(max_workers=2) as pool:
  for f in as_completed([pool.submit(execute,r) for r in runs]):f.result()
 assert all(digest(p)==h for p,h in cfg['protected'].items())
 (OUT/'integrity.json').write_text(json.dumps(dict(protected_unchanged=True,new_training_transitions=307200))+'\n')
if __name__=='__main__':main()
