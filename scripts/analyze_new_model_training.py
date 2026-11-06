"""Summarize saved evaluations only; no environment or policy mutation."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'diagnostics/new_model_training_effect'

def main():
    episodes=[];metrics=[];segments=[]
    for tag in ['final_nominal','final_randomized','iter_1000','iter_3000','iter_5000','iter_5500','paired_legacy_policy_seeds']:
        d=json.loads((OUT/tag/'results.json').read_text())
        for suffix,h in d['config']['protected'].items():
            assert hashlib.sha256(Path(d['config']['checkpoint']+suffix).read_bytes()).hexdigest()==h
        for target,source in [(episodes,'episodes'),(metrics,'task_metrics'),(segments,'segments')]:
            target.extend(dict(setting=tag,**r) for r in d[source])
    ep=pd.DataFrame(episodes);mt=pd.DataFrame(metrics);sg=pd.DataFrame(segments)
    ep.to_csv(OUT/'episode_metrics.csv',index=False)
    mt.to_csv(OUT/'task_metrics.csv',index=False)
    sg.to_csv(OUT/'segment_metrics.csv',index=False)
    summary=ep.assign(survived=ep.termination_reason.eq('time_limit')).groupby(['setting','task'],sort=False).agg(
        n=('seed','count'),survived=('survived','sum'),mean_survival_s=('survival_s','mean'),
        min_survival_s=('survival_s','min'),max_survival_s=('survival_s','max'))
    summary.to_csv(OUT/'survival_summary.csv')
    strict=mt.groupby(['setting','task'],sort=False).agg(n=('seed','count'),
        posture=('posture_success','sum'),tracking=('tracking_success','sum'),combined=('composite_success','sum'),
        mean_pitch_rms_deg=('pitch_rms_deg','mean'),mean_roll_rms_deg=('roll_rms_deg','mean'),
        mean_height_mae_m=('height_mae_m','mean'),steady_vx=('steady_mean_vx_m_s','mean'))
    strict.to_csv(OUT/'strict_summary.csv')
    old=json.loads((ROOT/'diagnostics/new_model_validation/new/results.json').read_text())
    oldidx={(r['task'],r['seed']):r for r in old['episodes']}
    paired=[]
    for r in ep[ep.setting=='paired_legacy_policy_seeds'].to_dict('records'):
        b=oldidx[(r['task'],r['seed'])]
        paired.append(dict(task=r['task'],seed=r['seed'],old_policy_survival_s=b['steps']*.008,
            trained_policy_survival_s=r['survival_s'],delta_s=r['survival_s']-b['steps']*.008))
    pd.DataFrame(paired).to_csv(OUT/'paired_old_policy_comparison.csv',index=False)
    scalars=json.loads((OUT/'training_scalars.json').read_text())
    plt.rcParams.update({'font.size':10})
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for ax,key,label,scale in zip(axes.flat,
        ['rollout/ep_rew_mean','rollout/ep_len_mean','Episode/survive_rate','Tracking/vx_rmse'],
        ['Episode reward','Mean episode duration (s)','Training rolling survival (%)','Velocity RMSE (m/s)'],[1,.008,100,1]):
        x=np.array([r['step']/4096 for r in scalars[key]])
        y=np.array([r['value']*scale for r in scalars[key]])
        ax.plot(x,y,alpha=.25,lw=.6,color='#3978a8')
        ax.plot(x,pd.Series(y).rolling(100,min_periods=1).mean(),color='#145a86',lw=1.5,label='100-update mean')
        ax.set(xlabel='PPO iteration',ylabel=label);ax.grid(alpha=.2)
        if key=='rollout/ep_len_mean':ax.axhline(24,color='#b34e36',ls='--',lw=1)
    fig.suptitle('New model: learning improved late, movement remains unstable')
    fig.savefig(OUT/'training_curves.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(11,8),layout='constrained')
    for task,color in [('forward2','#2176a5'),('reverse2','#bd5b3e')]:
        z=np.load(OUT/'final_nominal'/f'{task}_62000.npz');a=z['samples'];c=list(z['columns'])
        get=lambda n:a[:,c.index(n)]
        axes[0].plot(get('time_s'),get('vx_m_s'),color=color,label=f'{task}: actual')
        axes[0].plot(get('time_s'),get('command_vx_m_s'),color=color,ls='--',alpha=.6)
        axes[1].plot(get('time_s'),np.degrees(get('pitch_rad')),color=color,label=task)
        axes[2].plot(get('time_s'),get('wheel_saturation'),color=color,label=task)
    for ax,label in zip(axes,['Velocity (m/s)','Pitch (deg)','Wheel saturated substep fraction']):
        ax.set(xlim=(2.8,3.5),xlabel='Time (s)',ylabel=label);ax.axvline(3,color='gray',ls=':');ax.grid(alpha=.2);ax.legend()
    axes[1].axhline(-np.degrees(.6),color='gray',ls='--',lw=1)
    fig.suptitle('Final policy: command at 3 s, both directions terminate before settling')
    fig.savefig(OUT/'launch_transients.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4),layout='constrained')
    labels=['stand','forward2','reverse2','stop2_reverse','reverse10','motion']
    x=np.arange(len(labels))
    for offset,setting,label,color in [(-.18,'final_nominal','Nominal','#28749a'),(.18,'final_randomized','Domain randomized','#d39438')]:
        frame=summary.loc[setting].reindex(labels)
        ax.bar(x+offset,100*frame.survived/frame.n,width=.35,label=label,color=color)
    ax.set(xticks=x,xticklabels=labels,ylabel='24 s survival (%)',ylim=(0,110))
    ax.legend();ax.grid(axis='y',alpha=.2)
    fig.savefig(OUT/'survival_comparison.png',dpi=160);plt.close(fig)
    natural=json.loads((OUT/'training_distribution/results.json').read_text())
    for p,h in natural['protected'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    nr=pd.DataFrame(natural['episodes']);nr.to_csv(OUT/'training_distribution_metrics.csv',index=False)
    audit=dict(fixed_protocol_episodes=len(ep),training_distribution_episodes=len(nr),
        total_episodes=len(ep)+len(nr),source_checkpoints_and_normalizers_unchanged=True,
        no_learning=True,natural_survival_count=int(nr.reason.eq('time_limit').sum()),
        natural_mean_survival_s=float(nr.survival_s.mean()),natural_median_survival_s=float(nr.survival_s.median()),
        natural_reasons=nr.reason.value_counts().to_dict(),
        training_window_means={})
    for key in ['Episode/survive_rate','rollout/ep_len_mean','Tracking/vx_rmse','rollout/ep_rew_mean','train/approx_kl','train/clip_fraction','train/explained_variance']:
        audit['training_window_means'][key]={}
        for a,b in [(0,500),(2500,3000),(5500,6000),(5900,6000)]:
            values=[r['value'] for r in scalars[key] if a*4096<r['step']<=b*4096]
            audit['training_window_means'][key][f'{a+1}-{b}']=float(np.mean(values))
    (OUT/'analysis_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(summary.to_string())
    print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
