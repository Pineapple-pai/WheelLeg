"""Summarize completed precision training and paired follow-up without relaxing thresholds."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'diagnostics/precision_completed_review';AB=ROOT/'diagnostics/precision_ramp_ab'

def rows(path,**labels):
 d=json.loads(Path(path).read_text());result=[]
 for m in d['task_metrics']:
  ep=next(e for e in d['episodes'] if (e['task'],e['seed'])==(m['task'],m['seed']))
  result.append(dict(**labels,**m,survived=ep['success'],survival_s=ep['survival_s'],termination_reason=ep['termination_reason']))
 return result

def summary(df,groups):
 return df.groupby(groups,sort=False).agg(n=('seed','size'),survived=('survived','sum'),posture=('posture_success','sum'),
  tracking=('tracking_success','sum'),combined=('composite_success','sum'),steady_vx=('steady_mean_vx_m_s','mean'),
  steady_rmse=('steady_rmse_m_s','mean'),survival_s=('survival_s','mean'))

def main():
 latest=ROOT/'diagnostics/joint_precision_20261105_184905/evaluation'
 r=rows(latest/'results.json',snapshot='final1000_nominal5')
 for p in sorted(OUT.glob('iter_*/results.json')):r+=rows(p,snapshot=p.parent.name)
 df=pd.DataFrame(r);df.to_csv(OUT/'checkpoint_metrics.csv',index=False);agg=summary(df,['snapshot','task']);agg.to_csv(OUT/'checkpoint_summary.csv')
 # Exact reproduction of baseline trajectories, including observation-noise draws.
 for goal in (.25,-.25,.5,-.5):
  task=('p' if goal>0 else 'm')+format(abs(goal),'g').replace('.','')
  original=f'launch_{"p" if goal>0 else "m"}{abs(goal):g}'
  for seed in (65300,65301,65302):
   a=np.load(OUT/f'interventions/baseline/{task}_{seed}.npz')['samples'];b=np.load(latest/f'{original}_{seed}.npz')['samples']
   np.testing.assert_array_equal(a,b)
 verify=dict(baseline_reproduces_latest_exact=True)
 if (AB/'integrity.json').exists():
  cfg=json.loads((AB/'experiment_config.json').read_text());r=[];audits=[]
  for run in cfg['runs']:
   folder=Path(run['directory']);audits.append(json.loads((folder/'initial_state.json').read_text()))
   r+=rows(folder/'evaluation/results.json',arm=run['arm'],train_seed=run['seed'])
  assert all(a==audits[0] for a in audits)
  assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in cfg['protected'].items())
  df=pd.DataFrame(r);assert len(df)==126;df.to_csv(AB/'task_metrics.csv',index=False)
  s=summary(df,['arm','task']);s.to_csv(AB/'summary.csv');summary(df,['arm','train_seed','task']).to_csv(AB/'per_seed.csv');print(s.to_string())
  # Compare steady errors only on the same complete episodes; early failures remain failures.
  pairs=[]
  for seed in (101,102,103):
   a=df[(df.arm=='control')&(df.train_seed==seed)].set_index(['task','seed']);b=df[(df.arm=='mixed_ramp')&(df.train_seed==seed)].set_index(['task','seed'])
   paired=(a.steady_complete==1)&(b.steady_complete==1)
   pairs.append(dict(train_seed=seed,common_complete=int(paired.sum()),
    paired_rmse_delta=float((b.loc[paired,'steady_rmse_m_s']-a.loc[paired,'steady_rmse_m_s']).mean()),
    posture_delta=int(b.posture_success.sum()-a.posture_success.sum()),composite_delta=int(b.composite_success.sum()-a.composite_success.sum())))
  (AB/'paired_summary.json').write_text(json.dumps(pairs,indent=2)+'\n');print(pairs)
  verify.update(initial_training_states_equal=True,protected_unchanged=True,ab_evaluation_episodes=126)
 (OUT/'verification.json').write_text(json.dumps(verify,indent=2)+'\n')
 conservative=ROOT/'diagnostics/precision_conservative'
 if (conservative/'integrity.json').exists():
  cfg=json.loads((conservative/'experiment_config.json').read_text());r=[]
  from train_uz05 import TrainingPPO
  from uz05.experiment_audit import fingerprint
  source_model=TrainingPPO.load(cfg['source'],device='cpu')
  expected_optimizer={}
  for lr in (1e-5,5e-5):
   for group in source_model.policy.optimizer.param_groups:group['lr']=lr
   expected_optimizer[lr]=fingerprint(source_model.policy.optimizer.state_dict())
  for run in cfg['runs']:
   folder=Path(run['directory']);a=json.loads((folder/'initial_state.json').read_text())
   b=json.loads((AB/f'seed_{run["seed"]}/control/initial_state.json').read_text())
   assert a.pop('learning_rate_schedule_0_05_1')==[1e-5]*3
   assert b.pop('learning_rate_schedule_0_05_1')==[5e-5]*3
   assert a.pop('optimizer_sha256')==expected_optimizer[1e-5]
   assert b.pop('optimizer_sha256')==expected_optimizer[5e-5]
   assert a==b
   r+=rows(folder/'evaluation/results.json',arm='conservative',train_seed=run['seed'])
  df=pd.DataFrame(r);assert len(df)==63;df.to_csv(conservative/'task_metrics.csv',index=False)
  s=summary(df,['arm','task']);s.to_csv(conservative/'summary.csv');summary(df,['arm','train_seed','task']).to_csv(conservative/'per_seed.csv');print(s.to_string())
  assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in cfg['protected'].items())
  (conservative/'verification.json').write_text(json.dumps(dict(initial_states_equal_except_learning_rate=True,protected_unchanged=True,evaluation_episodes=63))+'\n')
 print(agg.to_string())
if __name__=='__main__':main()
