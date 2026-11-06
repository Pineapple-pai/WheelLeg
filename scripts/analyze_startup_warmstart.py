"""Verify inheritance and summarize every warm-start seed and milestone."""
from pathlib import Path
import hashlib
import json
import pickle
import re
import numpy as np
import pandas as pd
from train_uz05 import TrainingPPO
from uz05.experiment_audit import fingerprint

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'diagnostics/startup_warmstart'

def main():
    cfg=json.loads((OUT/'experiment_config.json').read_text());rows=[];audits=[];logs=[]
    sources=[('source',0,OUT/'source_evaluation/results.json'),('source_dr',0,OUT/'source_dr/results.json')]
    for run in cfg['runs']:
        folder=Path(run['directory']);audits.append(json.loads((folder/'initial_state.json').read_text()))
        sources += [(f'warm_{step}',run['seed'],folder/f'evaluation_{step}/results.json') for step in (50,150)]
        sources += [('warm_150_dr',run['seed'],folder/'evaluation_dr/results.json')]
        for block in re.split(r'Learning iteration\s+', (folder/'train.log').read_text())[1:]:
            update=int(block.split('/')[0]);values=dict(re.findall(r'^\s*(Mean episode length|Mean reward|Episode/survive_rate|Posture/pitch_rms_deg|Control/joint_saturation_fraction):\s*([-+\d.eE]+)',block,re.M))
            logs.append(dict(train_seed=run['seed'],update=update,**{k:float(v) for k,v in values.items()}))
    assert all(a==audits[0] for a in audits),'Warm initial states differ'
    model=TrainingPPO.load(cfg['source'],device='cpu')
    assert fingerprint(model.policy.state_dict())==audits[0]['policy_sha256']
    assert fingerprint(model.policy.optimizer.state_dict())==audits[0]['optimizer_sha256']
    with Path(cfg['source']+'_vecnormalize.pkl').open('rb') as stream:
        source_normalization=pickle.load(stream)
    assert audits[0]['reward_normalization_sha256']==fingerprint(vars(source_normalization.ret_rms))
    assert audits[0]['source_num_timesteps_before_learn']==24576000
    for arm,seed,path in sources:
        data=json.loads(path.read_text())
        for task in data['task_metrics']:
            ep=next(e for e in data['episodes'] if e['task']==task['task'] and e['seed']==task['seed'])
            rows.append(dict(arm=arm,train_seed=seed,**task,survived=ep['success'],survival_s=ep['survival_s'],termination_reason=ep['termination_reason']))
    df=pd.DataFrame(rows)
    assert len(df)==231
    for key in ('steady_mean_vx_m_s','steady_rmse_m_s'):
        if key not in df:df[key]=np.nan
    df.to_csv(OUT/'task_metrics.csv',index=False);pd.DataFrame(logs).to_csv(OUT/'training_trend.csv',index=False)
    cols=dict(n=('seed','size'),survived=('survived','sum'),posture=('posture_success','sum'),tracking=('tracking_success','sum'),
        combined=('composite_success','sum'),mean_survival_s=('survival_s','mean'),steady_vx=('steady_mean_vx_m_s','mean'),steady_rmse=('steady_rmse_m_s','mean'))
    agg=df.groupby(['arm','task'],sort=False).agg(**cols);agg.to_csv(OUT/'summary.csv')
    per=df.groupby(['arm','train_seed','task'],sort=False).agg(**cols);per.to_csv(OUT/'per_training_seed.csv')
    paired=[]
    for r in df[df.arm.str.startswith('warm')].to_dict('records'):
        arm='source_dr' if r['arm'].endswith('_dr') else 'source'
        b=df[(df.arm==arm)&(df.task==r['task'])&(df.seed==r['seed'])].iloc[0]
        paired.append({k:r[k] for k in ('arm','train_seed','task','seed')}|
            {k+'_delta':r[k]-b[k] for k in ('survived','posture_success','tracking_success','composite_success','survival_s')})
    pd.DataFrame(paired).to_csv(OUT/'paired_source_comparisons.csv',index=False)
    gates={}
    for arm in ('warm_50','warm_150','warm_150_dr'):
        passes=[];stand=[]
        for seed in cfg['seeds']:
            a=df[(df.arm==arm)&(df.train_seed==seed)]
            stand.append(int(a[a.task=='stand'].composite_success.sum())==3)
            if int(a[a.task!='stand'].composite_success.sum())==18:passes.append(seed)
        gates[arm]=dict(stand_all_training_seeds=all(stand),all_low_tasks_passing_training_seeds=passes,
            expand_speed=len(passes)>=2 and all(stand))
    (OUT/'expansion_gate.json').write_text(json.dumps(gates,indent=2)+'\n')
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in cfg['protected'].items())
    (OUT/'verification.json').write_text(json.dumps(dict(protected_unchanged=True,initial_states_identical=True,
        source_actor_critic_optimizer_std_and_normalization_preserved=True,new_training_transitions=cfg['transitions'],
        evaluation_records=len(df),gates=gates),indent=2)+'\n')
    print(agg.to_string());print(json.dumps(gates,indent=2))

if __name__=='__main__':main()
