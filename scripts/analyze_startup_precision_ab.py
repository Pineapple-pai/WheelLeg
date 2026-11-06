"""Verify the single-factor trial and report all matched seeds, including failures."""
from pathlib import Path
import hashlib,json,re
import numpy as np
import pandas as pd
from stable_baselines3.common.running_mean_std import RunningMeanStd
from train_uz05 import TrainingPPO
from uz05.experiment_audit import fingerprint
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'diagnostics/startup_precision_ab'
LOW=['launch_p0.1','launch_m0.1','launch_p0.25','launch_m0.25']
def main():
 cfg=json.loads((OUT/'experiment_config.json').read_text());rows=[];audits=[];logs=[];positions=[]
 sources=[('source',0,OUT/'source_evaluation')]
 for run in cfg['runs']:
  folder=Path(run['directory']);a=json.loads((folder/'initial_state.json').read_text());a.pop('stage');audits.append(a)
  sources.append((run['arm'],run['seed'],folder/'evaluation'))
  cp=json.loads(Path(run['checkpoint']+'_contract.json').read_text())
  expected=.05 if run['arm']=='precision' else .15
  assert cp['precision_objective']['vx_error_scale_m_s']==expected
  assert cp['model_xml_sha256']==json.loads(Path(cfg['source']+'_contract.json').read_text())['model_xml_sha256']
  for block in re.split(r'Learning iteration\s+', (folder/'train.log').read_text())[1:]:
   values=dict(re.findall(r'^\s*(Mean episode length|Mean reward|Episode/survive_rate|Posture/pitch_rms_deg):\s*([-+\d.eE]+)',block,re.M))
   logs.append(dict(arm=run['arm'],train_seed=run['seed'],update=int(block.split('/')[0]),**{k:float(v) for k,v in values.items()}))
 assert all(a==audits[0] for a in audits)
 model=TrainingPPO.load(cfg['source'],device='cpu')
 assert fingerprint(model.policy.state_dict())==audits[0]['policy_sha256']
 assert fingerprint(model.policy.optimizer.state_dict())==audits[0]['optimizer_sha256']
 assert fingerprint(vars(RunningMeanStd(shape=())))==audits[0]['reward_normalization_sha256']
 for arm,seed,folder in sources:
  data=json.loads((folder/'results.json').read_text())
  for r in data['task_metrics']:
   ep=next(e for e in data['episodes'] if e['task']==r['task'] and e['seed']==r['seed'])
   rows.append(dict(arm=arm,train_seed=seed,**r,survived=ep['success'],survival_s=ep['survival_s']))
  for seed0 in map(int,cfg['evaluation_seeds']):
   z=np.load(folder/f'stand_{seed0}.npz');a=z['samples'];c=list(z['columns']);mask=a[:,0]>22.
   positions.append(dict(arm=arm,train_seed=seed,seed=seed0,tail_position_mean_m=float(a[mask,c.index('hold_distance_m')].mean()) if mask.any() else None))
 df=pd.DataFrame(rows);assert len(df)==147
 for key in ('steady_mean_vx_m_s','steady_rmse_m_s'):
  if key not in df:df[key]=np.nan
 df.to_csv(OUT/'task_metrics.csv',index=False);pd.DataFrame(logs).to_csv(OUT/'training_trend.csv',index=False);pd.DataFrame(positions).to_csv(OUT/'stand_positions.csv',index=False)
 cols=dict(n=('seed','size'),survived=('survived','sum'),posture=('posture_success','sum'),tracking=('tracking_success','sum'),combined=('composite_success','sum'),mean_survival_s=('survival_s','mean'),steady_vx=('steady_mean_vx_m_s','mean'),steady_rmse=('steady_rmse_m_s','mean'))
 agg=df.groupby(['arm','task'],sort=False).agg(**cols);agg.to_csv(OUT/'summary.csv')
 df.groupby(['arm','train_seed','task'],sort=False).agg(**cols).to_csv(OUT/'per_training_seed.csv')
 pairs=[];seed_comparisons=[]
 for r in df[df.arm=='precision'].to_dict('records'):
  b=df[(df.arm=='control')&(df.train_seed==r['train_seed'])&(df.seed==r['seed'])&(df.task==r['task'])].iloc[0]
  pairs.append({k:r[k] for k in ('train_seed','seed','task')}|
   {k+'_delta':r[k]-b[k] for k in ('posture_success','tracking_success','composite_success','survived','survival_s')}|
   dict(steady_rmse_delta=r.get('steady_rmse_m_s',np.nan)-b.get('steady_rmse_m_s',np.nan)))
 pd.DataFrame(pairs).to_csv(OUT/'paired_comparisons.csv',index=False)
 for seed in cfg['seeds']:
  for arm in ('control','precision'):
   a=df[(df.arm==arm)&(df.train_seed==seed)&df.task.isin(LOW)]
   seed_comparisons.append(dict(train_seed=seed,arm=arm,n=len(a),complete=int(a.interval_complete.sum()),combined=int(a.composite_success.sum()),steady_rmse_mean=float(a.steady_rmse_m_s.mean())))
 pd.DataFrame(seed_comparisons).to_csv(OUT/'low_speed_seed_comparison.csv',index=False)
 gates={}
 for arm in ('control','precision'):
  standing=[];passing=[]
  for seed in cfg['seeds']:
   a=df[(df.arm==arm)&(df.train_seed==seed)]
   standing.append(int(a[a.task=='stand'].composite_success.sum())==3)
   if int(a[a.task!='stand'].composite_success.sum())==18:passing.append(seed)
  gates[arm]=dict(stand_all_training_seeds=all(standing),all_low_tasks_passing_training_seeds=passing,expand_speed=all(standing) and len(passing)>=2)
 (OUT/'expansion_gate.json').write_text(json.dumps(gates,indent=2)+'\n')
 archived=[]
 for name,expected in cfg['protected'].items():
  path=Path(name)
  if hashlib.sha256(path.read_bytes()).hexdigest()==expected:continue
  snapshot=OUT/'source_snapshot'/path.relative_to(ROOT)
  assert snapshot.exists() and hashlib.sha256(snapshot.read_bytes()).hexdigest()==expected,name
  archived.append(name)
 (OUT/'verification.json').write_text(json.dumps(dict(
  protected_unchanged_during_recorded_run=json.loads((OUT/'integrity.json').read_text())['protected_unchanged'],
  current_source_files_differ_but_archived_hashes_match=archived,
  source_actor_critic_optimizer_std_preserved=True,initial_states_equal_except_stage=True,
  both_arms_fresh_reward_rms=True,evaluation_records=len(df)),indent=2)+'\n')
 print(agg.to_string());print(pd.DataFrame(seed_comparisons).to_string(index=False));print(json.dumps(gates,indent=2))
if __name__=='__main__':main()
