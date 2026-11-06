"""Compare stratified pure-PPO continuation against both source and random control."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from review_completed_transfer import read_verified
from check_transfer_progress import evaluate_progress
from initialize_model_transfer import digest
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'diagnostics/pure_ppo_stratified_20261106';PREV=ROOT/'diagnostics/pure_ppo_precision_20261106'

def main():
 sources={'nominal':read_verified(PREV/'seed_711/reward_fine/evaluation/results.json'),
          'dr':read_verified(PREV/'holdout_dr/reward_fine_s711/results.json')}
 initial={'nominal':read_verified(PREV/'source_evaluation/results.json'),'dr':read_verified(PREV/'holdout_dr/source/results.json')}
 control={'nominal':read_verified(PREV/'continuation_s711/evaluation/results.json'),'dr':read_verified(PREV/'holdout_dr/reward_cont_s711/results.json')}
 gates={};records=[];fits={}
 for seed in (1711,1712):
  for mode,suffix in [('nominal','evaluation'),('dr','evaluation_dr')]:
   path=OUT/f'seed_{seed}/{suffix}/results.json'
   if not path.exists():continue
   d=read_verified(path);assert 'diagnostic_intervention' not in d['config']
   key=f'{seed}/{mode}'
   gates[key]={name:evaluate_progress(base,d) for name,base in [('source200',sources[mode]),('initial',initial[mode]),('random400',control[mode])]}
   for r in d['task_metrics']:records.append(dict(train_seed=seed,mode=mode,goal=d['config']['schedules'][r['task']][-1][1],**r))
   seg=[r for r in d['segments'] if r['end_s']==24 and r['vx_goal']!=0]
   if all(r['steady_vx'] is not None for r in seg):
    gain,bias=np.polyfit([r['vx_goal'] for r in seg],[r['steady_vx'] for r in seg],1)
    fits[key]=dict(gain=float(gain),bias=float(bias))
 if records:
  df=pd.DataFrame(records);df.to_csv(OUT/'per_episode.csv',index=False)
  agg=df.groupby(['train_seed','mode','goal']).agg(n=('seed','size'),posture=('posture_success','sum'),passed=('composite_success','sum'),
    vx=('steady_mean_vx_m_s','mean'),rmse=('steady_rmse_m_s','mean'),std=('steady_std_vx_m_s','mean'),occupancy=('steady_inband_fraction','mean'))
  agg.to_csv(OUT/'per_speed.csv');print(agg.to_string())
 (OUT/'gates.json').write_text(json.dumps(gates,indent=2)+'\n');(OUT/'fits.json').write_text(json.dumps(fits,indent=2)+'\n')
 ready=len(gates)==4 and all(gates[f'{s}/nominal']['source200']['accepted_for_further_training'] for s in (1711,1712)) and all(gates['1711/dr'][x]['accepted_for_further_training'] for x in ('source200','initial'))
 cp=ROOT/'checkpoints/pure_ppo_stratified_s1711_20261106/checkpoint'
 result=dict(ready_for_bounded_long_training=ready,recipe='reward_fine_stratified_continuation',evidence=gates,
  approved_sources={str(cp):{s:digest(str(cp)+s) for s in ('.zip','_contract.json','_vecnormalize.pkl')}} if ready else {},
  fully_passed_all_tasks=len(gates)==4 and all(gates[k]['source200']['fully_passed_all_tasks'] for k in gates),
  limitation='Training readiness, not deployment; unchanged full task acceptance. Ordinary PPO inference only.')
 (OUT/'readiness.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(fits,indent=2));print('Ready:',ready)
 for key,d in gates.items():
  print(key,{n:(r['accepted_for_further_training'],round(r['relative_improvement'],4),r['regressions']) for n,r in d.items()})

if __name__=='__main__':main()
