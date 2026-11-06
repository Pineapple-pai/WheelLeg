"""PPO-only matched continuation: change command sampling, not runtime control."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from run_startup_precision_ab import execute,evaluate
from initialize_model_transfer import digest
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'diagnostics/pure_ppo_stratified_20261106'

def main():
 OUT.mkdir(exist_ok=True)
 original=json.loads((ROOT/'diagnostics/pure_ppo_precision_20261106/continuation_s711/plan.json').read_text())['run']
 runs=[]
 for seed in (1711,1712):
  folder=OUT/f'seed_{seed}';cp=ROOT/f'checkpoints/pure_ppo_stratified_s{seed}_20261106/checkpoint'
  if cp.parent.exists():raise FileExistsError(cp.parent)
  cmd=original['command'].copy()
  for key,value in {'--command-sampling':'startup_stratified','--version':cp.parent.name,'--seed':str(seed),
    '--initial-state-audit':str(folder/'initial_state.json'),'--tensorboard-log':str(folder/'tensorboard')}.items():cmd[cmd.index(key)+1]=value
  ev=original['evaluate'].copy();ev[2]=str(cp);ev[ev.index('--output')+1]=str(folder/'evaluation')
  runs.append(dict(seed=seed,arm='stratified',directory=str(folder),checkpoint=str(cp),command=cmd,evaluate=ev))
 protected=[ROOT/'scripts/train_uz05.py',*[ROOT/f'scripts/uz05/{k}.py' for k in ('spec','env','actuators','command_sampling')]]
 cfg=dict(runs=runs,protected={str(p):digest(p) for p in protected},
  source='checkpoints/pure_ppo_reward_fine_s711_20261106/checkpoint',
  intervention='Only training sampler switches random independent goals to rank-offset cyclic strata. Same goal values and25%stand/12.5%each nonzero marginal, same24s/3s step timing. Inference and scoring commands untouched.',
  control='pure_ppo_precision_20261106/continuation_s711: same source and seed1711, original random sampler, rejected',
  budget='2x200x8x512=1638400additional PPO transitions',
  gate='Require both seeds retain200-update source successes and show progress; preselected1711 must pass DR against both source200 and original initialization.')
 (OUT/'plan.json').write_text(json.dumps(cfg,indent=2)+'\n')
 def run(r):
  execute(r)
  ev=r['evaluate'].copy();ev[ev.index('--output')+1]=str(Path(r['directory'])/'evaluation_dr')
  ev[ev.index('--seeds')+1:]=['66700','66701','66702'];ev.append('--domain-randomization')
  evaluate(ev,Path(r['directory'])/'evaluation_dr.log')
 with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run,runs))
 assert all(digest(p)==h for p,h in cfg['protected'].items())
 (OUT/'integrity.json').write_text(json.dumps(dict(protected_unchanged=True,no_assistance=True))+'\n')

if __name__=='__main__':main()
