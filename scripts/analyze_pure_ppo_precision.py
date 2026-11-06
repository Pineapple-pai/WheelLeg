"""Audit PPO-only runs and report every speed/seed before proposing continuation."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from review_completed_transfer import read_verified
from check_transfer_progress import evaluate_progress
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'diagnostics/pure_ppo_precision_20261106'

def main():
 records=[];fits={};audits={};gates={}
 baseline=read_verified(OUT/'source_evaluation/results.json')
 for plan in sorted(OUT.glob('plan_s*.json')):
  cfg=json.loads(plan.read_text())
  for run in cfg['runs']:
   folder=Path(run['directory']);path=folder/'evaluation/results.json'
   if not path.exists():continue
   d=read_verified(path)
   gates[f"{run['seed']}/{run['arm']}"]=evaluate_progress(baseline,d)
   assert 'diagnostic_intervention' not in d['config']
   a=json.loads((folder/'initial_state.json').read_text());audits[run['arm']]=a
   stage=json.loads((Path(run['checkpoint']).parent/'training_config.json').read_text())
   assert stage['environment_config']['observation_noise']['enabled']
   assert stage['environment_config']['domain_randomization']['enabled']
   expected_std=.003 if run['arm'] in ('wheel_fine','both_fine') else .02
   np.testing.assert_allclose(a['action_std'][:2],expected_std,rtol=1e-6)
   np.testing.assert_allclose(a['action_std'][2:],.02,rtol=1e-6)
   expected_scale=.01 if run['arm'] in ('reward_fine','both_fine') else .05
   assert stage['precision_objective']['vx_error_scale_m_s']==expected_scale
   for r in d['task_metrics']:
    records.append(dict(arm=run['arm'],train_seed=run['seed'],goal=d['config']['schedules'][r['task']][-1][1],**r))
   seg=[r for r in d['segments'] if r['end_s']==24 and r['vx_goal']!=0]
   if all(r['steady_vx'] is not None for r in seg):
    gain,bias=np.polyfit([r['vx_goal'] for r in seg],[r['steady_vx'] for r in seg],1)
    fits[f"{run['seed']}/{run['arm']}"]=dict(gain=float(gain),bias=float(bias))
 if not records:print('No complete evaluations yet');return
 for key in ('actor_mean_sha256','critic_sha256','optimizer_sha256','reward_normalization_sha256'):
  assert len({a[key] for a in audits.values()})==1,key
 df=pd.DataFrame(records);df.to_csv(OUT/'per_episode.csv',index=False)
 agg=df.groupby(['train_seed','arm','goal']).agg(n=('seed','size'),posture=('posture_success','sum'),
  passed=('composite_success','sum'),vx=('steady_mean_vx_m_s','mean'),rmse=('steady_rmse_m_s','mean'),
  std=('steady_std_vx_m_s','mean'),occupancy=('steady_inband_fraction','mean'))
 agg.to_csv(OUT/'per_speed.csv');(OUT/'fits.json').write_text(json.dumps(fits,indent=2)+'\n')
 (OUT/'gates.json').write_text(json.dumps(gates,indent=2)+'\n')
 print(agg.to_string());print(json.dumps(fits,indent=2))
 print(json.dumps({k:{x:v[x] for x in ('accepted_for_further_training','fully_passed_all_tasks','relative_improvement','per_goal_regression_within_half_band')} for k,v in gates.items()},indent=2))

if __name__=='__main__':main()
