"""Analyze the sampling follow-up against the previously frozen precision arm."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'diagnostics/startup_precision_low';OLD=ROOT/'diagnostics/startup_precision_ab'
LOW=['launch_p0.1','launch_m0.1','launch_p0.25','launch_m0.25']
def main():
 cfg=json.loads((OUT/'experiment_config.json').read_text());rows=[];audits=[]
 for run in cfg['runs']:
  seed=run['seed'];folder=Path(run['directory']);a=json.loads((folder/'initial_state.json').read_text());audits.append(a)
  b=json.loads((OLD/f'seed_{seed}/precision/initial_state.json').read_text());assert a==b
  for arm,path in [('balanced',OLD/f'seed_{seed}/precision/evaluation/results.json'),('low',folder/'evaluation/results.json')]:
   data=json.loads(path.read_text())
   for r in data['task_metrics']:
    ep=next(e for e in data['episodes'] if e['task']==r['task'] and e['seed']==r['seed'])
    rows.append(dict(arm=arm,train_seed=seed,**r,survived=ep['success'],survival_s=ep['survival_s']))
 assert all(a==audits[0] for a in audits)
 df=pd.DataFrame(rows);assert len(df)==126
 for key in ('steady_mean_vx_m_s','steady_rmse_m_s'):
  if key not in df:df[key]=np.nan
 df.to_csv(OUT/'task_metrics.csv',index=False)
 cols=dict(n=('seed','size'),survived=('survived','sum'),posture=('posture_success','sum'),tracking=('tracking_success','sum'),combined=('composite_success','sum'),steady_vx=('steady_mean_vx_m_s','mean'),steady_rmse=('steady_rmse_m_s','mean'))
 agg=df.groupby(['arm','task'],sort=False).agg(**cols);agg.to_csv(OUT/'summary.csv')
 df.groupby(['arm','train_seed','task'],sort=False).agg(**cols).to_csv(OUT/'per_training_seed.csv')
 pairs=[]
 for r in df[df.arm=='low'].to_dict('records'):
  b=df[(df.arm=='balanced')&(df.train_seed==r['train_seed'])&(df.task==r['task'])&(df.seed==r['seed'])].iloc[0]
  pairs.append({k:r[k] for k in ('train_seed','seed','task')}|{k+'_delta':r[k]-b[k] for k in ('survived','posture_success','composite_success')}|
   dict(steady_rmse_delta=r.get('steady_rmse_m_s',np.nan)-b.get('steady_rmse_m_s',np.nan)))
 pd.DataFrame(pairs).to_csv(OUT/'paired_comparisons.csv',index=False)
 gate={}
 for arm in ('balanced','low'):
  stands=[];passes=[]
  for seed in (91,92,93):
   a=df[(df.arm==arm)&(df.train_seed==seed)];stands.append(int(a[a.task=='stand'].composite_success.sum())==3)
   if int(a[a.task.isin(LOW)].composite_success.sum())==12:passes.append(seed)
  gate[arm]=dict(stand_all_training_seeds=all(stands),trained_tasks_all_pass_seeds=passes,expand_speed=len(passes)>=2 and all(stands))
 (OUT/'expansion_gate.json').write_text(json.dumps(gate,indent=2)+'\n')
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in cfg['protected'].items())
 (OUT/'verification.json').write_text(json.dumps(dict(protected_unchanged=True,initial_states_match_prior_precision_arm=True,new_evaluation_records=63),indent=2)+'\n')
 print(agg.to_string());print(json.dumps(gate,indent=2))
if __name__=='__main__':main()
