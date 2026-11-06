"""Analyze all frozen sampling arms; no selection by lucky training seeds."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'diagnostics/ppo_launch_sampling'
LOW=['launch_p0.25','launch_m0.25','launch_p0.5','launch_m0.5']

def main():
    cfg=json.loads((OUT/'experiment_config.json').read_text())
    records=[];episodes=[];segments=[];audits=[];verified=[]
    baseline=json.loads((OUT/'baseline_low/results.json').read_text())
    for run in [dict(arm='source6000',seed=0,directory=str(OUT/'baseline_low'),source=True),*cfg['runs']]:
        folder=Path(run['directory'])
        data=baseline if run.get('source') else json.loads((folder/'evaluation/results.json').read_text())
        if not run.get('source'):
            audits.append(json.loads((folder/'initial_state.json').read_text()))
            cp=Path(run['checkpoint'])
            con=json.loads(Path(str(cp)+'_contract.json').read_text())
            src=json.loads(Path(cfg['source']+'_contract.json').read_text())
            for key in ['model_xml_sha256','actuator_config','environment_config','precision_objective',
                        'observation_shape','actor_blocks','critic_privileged_blocks','action_shape']:
                assert con[key]==src[key],(run['arm'],key)
            assert con['command_sampling']==cfg['arms'][run['arm']]
            verified.append(dict(seed=run['seed'],arm=run['arm'],contracts_match=True))
        tag=dict(arm=run['arm'],train_seed=run['seed'])
        for target,key in [(records,'task_metrics'),(episodes,'episodes'),(segments,'segments')]:
            target.extend(dict(**tag,**r) for r in data[key])
    assert len(audits)==9 and all(a==audits[0] for a in audits)
    for p,h in cfg['protected'].items():
        path=Path(p) if Path(p).is_absolute() else ROOT/p
        assert hashlib.sha256(path.read_bytes()).hexdigest()==h,p
    df=pd.DataFrame(records);ep=pd.DataFrame(episodes);sg=pd.DataFrame(segments)
    df.to_csv(OUT/'task_metrics.csv',index=False);ep.to_csv(OUT/'episode_metrics.csv',index=False)
    sg.to_csv(OUT/'segment_metrics.csv',index=False)
    per=df.groupby(['arm','train_seed','task'],sort=False).agg(n=('seed','count'),
        posture=('posture_success','sum'),tracking=('tracking_success','sum'),combined=('composite_success','sum'),
        complete=('interval_complete','sum'),steady_vx=('steady_mean_vx_m_s','mean'),
        steady_rmse=('steady_rmse_m_s','mean'),pitch_rms=('pitch_rms_deg','mean'))
    per.to_csv(OUT/'per_training_seed_summary.csv')
    agg=df.groupby(['arm','task'],sort=False).agg(n=('seed','count'),posture=('posture_success','sum'),
        tracking=('tracking_success','sum'),combined=('composite_success','sum'),complete=('interval_complete','sum'),
        steady_vx=('steady_mean_vx_m_s','mean'),steady_rmse=('steady_rmse_m_s','mean'))
    agg.to_csv(OUT/'condition_summary.csv')
    survival=ep.groupby(['arm','task'],sort=False).agg(n=('seed','count'),survived=('success','sum'),
        mean_survival_s=('survival_s','mean'),min_survival_s=('survival_s','min'),max_survival_s=('survival_s','max'))
    survival.to_csv(OUT/'survival_summary.csv')
    pairs=[]
    for r in df[df.arm.isin(['low_band','launch_focus'])].to_dict('records'):
        b=df[(df.arm=='original')&(df.train_seed==r['train_seed'])&(df.seed==r['seed'])&(df.task==r['task'])].iloc[0]
        pairs.append({k:r[k] for k in ['arm','train_seed','seed','task']}|
            {k+'_delta':r[k]-b[k] for k in ['posture_success','tracking_success','composite_success','interval_complete']}|
            {'steady_rmse_delta':r.get('steady_rmse_m_s',np.nan)-b.get('steady_rmse_m_s',np.nan)})
    pd.DataFrame(pairs).to_csv(OUT/'paired_comparisons.csv',index=False)
    source_pairs=[]
    for r in df[df.arm!='source6000'].to_dict('records'):
        b=df[(df.arm=='source6000')&(df.seed==r['seed'])&(df.task==r['task'])].iloc[0]
        source_pairs.append({k:r[k] for k in ['arm','train_seed','seed','task']}|
            {k+'_delta':r[k]-b[k] for k in ['posture_success','tracking_success','composite_success','interval_complete']}|
            {'steady_rmse_delta':r.get('steady_rmse_m_s',np.nan)-b.get('steady_rmse_m_s',np.nan)})
    pd.DataFrame(source_pairs).to_csv(OUT/'paired_source_comparisons.csv',index=False)
    gates={}
    for arm in cfg['arms']:
        pass_seeds=[];stand=[]
        for seed in cfg['seeds']:
            a=df[(df.arm==arm)&(df.train_seed==seed)]
            st=a[a.task=='stand'];stand.append(int(st.composite_success.sum())==3)
            low=a[a.task.isin(LOW)]
            if len(low)==12 and int(low.composite_success.sum())==12:pass_seeds.append(seed)
        gates[arm]=dict(low_all_pass_training_seeds=pass_seeds,stand_all_training_seeds=all(stand),
                        expand_to_1m_s=len(pass_seeds)>=2 and all(stand))
    (OUT/'expansion_gate.json').write_text(json.dumps(gates,indent=2)+'\n')
    (OUT/'verification.json').write_text(json.dumps(dict(protected_files_unchanged=True,
        all9initial_actor_critic_optimizer_normalization_std_states_equal=True,initial_state=audits[0],
        contracts=verified,total_short_training_transitions=9*100*4*256,
        episodes=len(ep),comparable_eval_seeds=[64000,64001,64002],
        baseline_additional_eval_seeds=[64003,64004]),indent=2)+'\n')
    plt.rcParams.update({'font.size':10})
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    tasks=['stand',*LOW];x=np.arange(len(tasks))
    for ax,col,label in zip(axes,['posture','tracking','combined'],['Posture pass (%)','Tracking pass (%)','Combined pass (%)']):
        for i,arm in enumerate(['source6000','original','low_band','launch_focus']):
            a=agg.loc[arm].reindex(tasks)
            ax.bar(x+(i-1.5)*.2,100*a[col]/a.n,width=.19,label=arm)
        ax.set(xticks=x,xticklabels=['stand','+0.25','-0.25','+0.5','-0.5'],ylim=(0,110),ylabel=label)
        ax.grid(axis='y',alpha=.2)
    axes[-1].legend(fontsize=8);fig.savefig(OUT/'success_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for ax,task in zip(axes.flat,LOW):
        for arm,folder in [('source6000',OUT/'baseline_low'),
                           *[(a,OUT/'seed_71'/a/'evaluation') for a in cfg['arms']]]:
            z=np.load(folder/f'{task}_64000.npz');a=z['samples'];c=list(z['columns'])
            ax.plot(a[:,c.index('time_s')],a[:,c.index('vx_m_s')],label=arm)
        goal=float(task.split('launch_')[1][1:])*(1 if '_p' in task else -1)
        ax.axhline(goal,color='black',ls='--',lw=1);ax.axvline(3,color='gray',ls=':')
        ax.set(title=task,xlabel='Time(s)',ylabel='Body vx(m/s)');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.savefig(OUT/'launch_velocity_examples.png',dpi=160);plt.close(fig)
    print(agg.to_string());print(survival.to_string());print(json.dumps(gates,indent=2))

if __name__=='__main__':main()
