"""Analyze target-plant collapse, explicit initialization and paired adaptation."""
from pathlib import Path
import json,re,hashlib
import numpy as np
import pandas as pd
from analyze_precision_completed import rows,summary
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'diagnostics/armature001_training_review'

def analyze_ab(folder):
 cfg=json.loads((folder/'experiment_config.json').read_text());r=[];audits=[]
 for run in cfg['runs']:
  d=Path(run['directory']);audits.append((run['seed'],run['arm'],json.loads((d/'initial_state.json').read_text())))
  r+=rows(d/'evaluation/results.json',arm=run['arm'],train_seed=run['seed'])
 df=pd.DataFrame(r);assert len(df)==126;df.to_csv(folder/'task_metrics.csv',index=False)
 agg=summary(df,['arm','task']);agg.to_csv(folder/'summary.csv');summary(df,['arm','train_seed','task']).to_csv(folder/'per_seed.csv')
 first=audits[0][2]
 for seed,arm,a in audits:
  for key in ['actor_mean_sha256','critic_sha256','optimizer_sha256','reward_normalization_sha256']:
   assert a[key]==first[key],(arm,key)
  if arm=='preserve':assert a['action_std'][0]>.06
  else:np.testing.assert_allclose(a['action_std'],.02,rtol=1e-6)
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in cfg['protected'].items())
 (folder/'verification.json').write_text(json.dumps(dict(initial_mean_actor_critic_adam_rms_equal=True,only_declared_std_or_reward_intervention=True,protected_unchanged=True,evaluation_episodes=126))+'\n')
 print(agg.to_string())
 # All failure counts retained; paired speed deltas use only the same complete episodes.
 arms=list(dict.fromkeys(df.arm));pairs=[]
 for seed in [201,202,203]:
  a=df[(df.train_seed==seed)&(df.arm==arms[0])].set_index(['task','seed']);b=df[(df.train_seed==seed)&(df.arm==arms[1])].set_index(['task','seed'])
  both=(a.steady_complete==1)&(b.steady_complete==1)&(a.index.get_level_values('task')!='stand')
  pairs.append(dict(train_seed=seed,common_complete_motion=int(both.sum()),motion_rmse_delta=float((b.loc[both,'steady_rmse_m_s']-a.loc[both,'steady_rmse_m_s']).mean()),posture_delta=int(b.posture_success.sum()-a.posture_success.sum()),composite_delta=int(b.composite_success.sum()-a.composite_success.sum())))
 (folder/'paired_summary.json').write_text(json.dumps(pairs,indent=2)+'\n');print(pairs)

def main():
 r=rows(ROOT/'diagnostics/joint_startup_armature001_20261106_080550/evaluation/results.json',model='scratch_final')
 for p in OUT.glob('*/results.json'):r+=rows(p,model=p.parent.name)
 df=pd.DataFrame(r);df.to_csv(OUT/'model_metrics.csv',index=False);summary(df,['model','task']).to_csv(OUT/'model_summary.csv')
 source_path=ROOT/'diagnostics/armature001_transfer_ab/source_evaluation/results.json'
 if source_path.exists():
  data=json.loads(source_path.read_text())
  seg=[s for s in data['segments'] if s['complete'] and s['end_s']==24 and s['vx_goal']!=0]
  pd.DataFrame(seg).to_csv(OUT/'steady_speed_decomposition.csv',index=False)
  x=np.array([s['vx_goal'] for s in seg]);fits={}
  for key in ['steady_vx','target_equivalent_vx','wheel_equivalent_vx']:
   y=np.array([s[key] for s in seg]);gain,bias=np.polyfit(x,y,1);pred=gain*x+bias
   fits[key]=dict(gain=float(gain),bias_m_s=float(bias),r_squared=float(1-((y-pred)**2).sum()/((y-y.mean())**2).sum()))
  fits['interpretation']='Diagnostic steady affine fit only; no runtime command or action correction applied.'
  (OUT/'steady_speed_fit.json').write_text(json.dumps(fits,indent=2)+'\n')
 log=(ROOT/'logs/joint_startup_armature001_20261106_080550.log').read_text()
 metrics={k:[float(x) for x in re.findall(re.escape(k)+r':\s*([-\d.]+)',log)] for k in ['Mean episode length','Mean reward','Posture/pitch_rms_deg','Control/joint_saturation_fraction','Episode/survive_rate']}
 (OUT/'scratch_training_scalars.json').write_text(json.dumps(metrics,indent=2)+'\n')
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axs=plt.subplots(2,2,figsize=(10,6))
 for ax,(key,scale,label) in zip(axs.flat,[('Mean episode length',.008,'Mean episode length (s)'),('Posture/pitch_rms_deg',1,'Training pitch RMS (deg)'),('Control/joint_saturation_fraction',1,'Joint saturation fraction'),('Episode/survive_rate',1,'Training rolling survival fraction')]):
  values=np.array(metrics[key])*scale;ax.plot(np.arange(1,len(values)+1),values,lw=.8);ax.set_ylabel(label);ax.set_xlabel('PPO iterations');ax.grid(alpha=.2)
 axs[0,0].axhline(3,color='red',ls='--',label='Motion command at3s');axs[0,0].legend(fontsize=8)
 fig.suptitle('From-scratch0.001 run: fails before commanded motion');fig.tight_layout();fig.savefig(OUT/'scratch_learning.png',dpi=150)
 for folder in ['armature001_transfer_ab','armature001_transfer_precision_ab']:
  p=ROOT/'diagnostics'/folder
  if (p/'integrity.json').exists():analyze_ab(p)
 fresh=ROOT/'diagnostics/armature001_freshadam'
 if (fresh/'integrity.json').exists():
  from check_transfer_progress import evaluate_progress
  cfg=json.loads((fresh/'experiment_config.json').read_text());rr=[];pairs=[]
  base=json.loads((ROOT/'diagnostics/armature001_transfer_ab/source_evaluation/results.json').read_text())
  for run in cfg['runs']:
   folder=Path(run['directory']);a=json.loads((folder/'initial_state.json').read_text())
   old=ROOT/f'diagnostics/armature001_transfer_precision_ab/seed_{run["seed"]}/precision_reset'
   b=json.loads((old/'initial_state.json').read_text())
   for key in ['actor_mean_sha256','critic_sha256','action_std','reward_normalization_sha256','learning_rate_schedule_0_05_1','clip_schedule_0_05_1']:
    assert a[key]==b[key],key
   assert a['optimizer_step_values']==[] and b['optimizer_step_values']==[1260000.]
   data=json.loads((folder/'evaluation/results.json').read_text())
   gate=evaluate_progress(base,data);(folder/'progress_vs_source.json').write_text(json.dumps(gate,indent=2)+'\n')
   for arm,path in [('retained_adam',old/'evaluation/results.json'),('fresh_adam',folder/'evaluation/results.json')]:rr+=rows(path,arm=arm,train_seed=run['seed'])
  df=pd.DataFrame(rr);assert len(df)==126;df.to_csv(fresh/'task_metrics.csv',index=False)
  agg=summary(df,['arm','task']);agg.to_csv(fresh/'summary.csv');summary(df,['arm','train_seed','task']).to_csv(fresh/'per_seed.csv');print(agg.to_string())
  for seed in (201,202,203):
   a=df[(df.train_seed==seed)&(df.arm=='retained_adam')].set_index(['task','seed']);b=df[(df.train_seed==seed)&(df.arm=='fresh_adam')].set_index(['task','seed'])
   both=(a.steady_complete==1)&(b.steady_complete==1)&(a.index.get_level_values('task')!='stand')
   pairs.append(dict(seed=seed,common_complete_motion=int(both.sum()),rmse_delta=float((b.loc[both,'steady_rmse_m_s']-a.loc[both,'steady_rmse_m_s']).mean()),posture_delta=int(b.posture_success.sum()-a.posture_success.sum()),composite_delta=int(b.composite_success.sum()-a.composite_success.sum())))
  (fresh/'paired_summary.json').write_text(json.dumps(pairs,indent=2)+'\n');print(pairs)
  assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in cfg['protected'].items())
  (fresh/'verification.json').write_text(json.dumps(dict(initial_parameters_std_rms_equal=True,only_optimizer_state_changed=True,protected_unchanged=True))+'\n')
 for name in ['armature001_precision_continuation','armature001_fine_exploration']:
  folder=ROOT/'diagnostics'/name
  if not (folder/'integrity.json').exists():continue
  from check_transfer_progress import evaluate_progress
  cfg=json.loads((folder/'experiment_config.json').read_text());rr=[];gates=[]
  for run in cfg['runs']:
   d=Path(run['directory'])
   for protocol,suffix,baseline in [('nominal','evaluation',ROOT/'diagnostics/armature001_transfer_ab/source_evaluation/results.json'),('randomized','evaluation_dr',OUT/'transfer_legacy_dr/results.json')]:
    path=d/suffix/'results.json'
    if not path.exists():continue
    data=json.loads(path.read_text());base=json.loads(baseline.read_text());gate=evaluate_progress(base,data)
    (d/f'progress_{protocol}_vs_source.json').write_text(json.dumps(gate,indent=2)+'\n')
    gates.append(dict(seed=run['seed'],protocol=protocol,**gate))
    rr+=rows(path,plant_test=protocol,train_seed=run['seed'])
   if 'source' in run:
    assert all(hashlib.sha256(Path(run['source']+s).read_bytes()).hexdigest()==h for s,h in run['protected'].items())
  df=pd.DataFrame(rr);df.to_csv(folder/'task_metrics.csv',index=False)
  agg=summary(df,['plant_test','task']);agg.to_csv(folder/'summary.csv');summary(df,['plant_test','train_seed','task']).to_csv(folder/'per_seed.csv');print(name,agg.to_string())
  (folder/'progress_summary.json').write_text(json.dumps(gates,indent=2)+'\n')
  if name=='armature001_fine_exploration':
   pairs=[]
   for run in cfg['runs']:
    d=Path(run['directory']);old=ROOT/f'diagnostics/armature001_transfer_precision_ab/seed_{run["seed"]}/precision_reset'
    a=json.loads((d/'initial_state.json').read_text());b=json.loads((old/'initial_state.json').read_text())
    for key in ['actor_mean_sha256','critic_sha256','optimizer_sha256','reward_normalization_sha256']:assert a[key]==b[key],key
    np.testing.assert_allclose(a['action_std'][:2],.02,rtol=1e-6);np.testing.assert_allclose(a['action_std'][2:],.002,rtol=1e-6)
    x=pd.DataFrame(rows(old/'evaluation/results.json')).set_index(['task','seed']);y=pd.DataFrame(rows(d/'evaluation/results.json')).set_index(['task','seed'])
    mask=x.index.get_level_values('task')!='stand'
    pairs.append(dict(seed=run['seed'],paired_motion_rmse_delta=float((y.loc[mask,'steady_rmse_m_s']-x.loc[mask,'steady_rmse_m_s']).mean()),posture=int(y.posture_success.sum()),motion_success=int(y.loc[mask,'composite_success'].sum())))
   (folder/'paired_summary.json').write_text(json.dumps(pairs,indent=2)+'\n')
  assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in cfg['protected'].items())
  (folder/'verification.json').write_text(json.dumps(dict(protected_unchanged=True,evaluation_episodes=len(df)))+'\n')
if __name__=='__main__':main()
