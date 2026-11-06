"""Milestone, paired-seed and cross-dynamics analysis of frozen v17 runs."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3.common.save_util import load_from_zip_file
from run_ppo_v17_torque_ab import OUT,ROOT,OLD_SEEDS,NEW_SEEDS,MILESTONES,prepare,verify
from run_m3508_validation import PROFILES,TASKS,THRESHOLDS
from uz05.m3508_metrics import score_interval
from analyze_m3508_validation import controls,table
from analyze_wheel_speed_steady_state import decomposition
from analyze_ppo_reset_ab import get_training


def load(path):
    with np.load(path) as z:
        samples=z['samples'];extra=z['extra']
        c={k:samples[:,i] for i,k in enumerate(z['columns'])}
        c.update({k:extra[:,i] for i,k in enumerate(z['extra_columns'])})
        c['motor']=z['motor'];c['motor_columns']=list(z['motor_columns'])
    return c


def locations(cfg):
    for model in ['A_legacy','B_rated']:
        yield {'train_seed':0,'train_model':'v16','update':0,'eval_model':model},OUT/'evaluation'/'baseline'/model
    for run in cfg['runs']:
      for update in MILESTONES:
        for model in (['A_legacy','B_rated'] if update==200 else ['B_rated']):
          yield {'train_seed':run['seed'],'train_model':run['mode'],'update':update,'eval_model':model},OUT/'evaluation'/f'seed_{run["seed"]}'/run['mode']/f'update_{update}'/model


def collect(cfg):
    rows=[];steady=[];episodes=[];max_map=max_identity=max_current=max_cap_excess=max_contact_identity=max_passive=0.
    for meta,root in locations(cfg):
      for protocol in PROFILES:
        records=json.loads((root/protocol/'episodes.json').read_text())
        expected=set(OLD_SEEDS+NEW_SEEDS if meta['update'] in [0,200] else OLD_SEEDS)
        assert {r['eval_seed'] for r in records}==expected,(meta,protocol)
        for record in records:
          seed=record['eval_seed'];c=load(root/protocol/f'{seed}.npz');t=c['time_s'];motor=c['motor'];mc=c['motor_columns']
          identity={**meta,'eval_seed':seed,'cohort':'historical' if seed in OLD_SEEDS else 'independent','protocol':protocol}
          episodes.append({**record,**identity})
          max_map=max(max_map,record['mapping_error_max_nm'])
          for i in range(2):
              v=c[f'kinematic_identity_error_{i}'];v=v[np.isfinite(v)]
              if len(v):max_contact_identity=max(max_contact_identity,float(np.abs(v).max()))
              max_passive=max(max_passive,float(np.abs(c[f'wheel_passive_nm_{i}']-c[f'wheel_viscous_predicted_nm_{i}']).max()))
          current=motor[:,:,[mc.index(f'wheel_control_current_a_{i}') for i in range(2)]]
          torque=motor[:,:,[mc.index(f'wheel_torque_nm_{i}') for i in range(2)]]
          cap=1.722 if meta['eval_model']=='A_legacy' else 2.46
          max_current=max(max_current,float(np.abs(current).max()));max_cap_excess=max(max_cap_excess,float(np.abs(torque).max()-cap))
          for name,source,start,end,goal in TASKS:
            if source!=protocol:continue
            mask=(t>start+1e-9)&(t<=end+1e-9)
            row={**identity,'task':name,'survival_s':record['survival_s'],'termination_reason':record['termination_reason'],
              'episode_success':record['success'],**score_interval(c,start,end,goal,THRESHOLDS,record['termination_reason']),**controls(c,mask)}
            # Explicit stopping rebound and common-motion exposure, never fill unobserved windows.
            row['forward_rebound_m_s']=float(max(0,c['vx_m_s'][mask].max())) if name=='reverse_stop' and mask.any() else None
            rows.append(row)
            if row.get('steady_complete')==1:
                sm=(t>end-2+1e-9)&(t<=end+1e-9)
                dec=decomposition(c,sm,goal);max_identity=max(max_identity,dec['macro_identity_max_error_m_s'])
                steady.append({**identity,'task':name,**dec})
    assert max_map<1e-9 and max_identity<1e-9 and max_current<=10+1e-9 and max_cap_excess<1e-9 and max_contact_identity<1e-9 and max_passive<1e-9
    df=pd.DataFrame(rows);ss=pd.DataFrame(steady);ep=pd.DataFrame(episodes)
    df.to_csv(OUT/'evaluation_metrics.csv',index=False);ss.to_csv(OUT/'steady_speed_decomposition.csv',index=False)
    ep.to_csv(OUT/'episode_metrics.csv',index=False)
    joint=['train_seed','train_model','update','eval_model','eval_seed','cohort','task','survival_s','interval_complete','exposure_s',
      'common_x_rms_m','common_x_peak_m','common_outside_008_longest_s','differential_x_rms_m','joint_target_step_max_rad','joint_target_step_p95_rad',
      'joint_target_rate_rms_rad_s','joint_tracking_rmse_rad','leg_saturation_fraction','wheel_saturation_fraction','pitch_rms_deg','forward_rebound_m_s']
    df[joint].to_csv(OUT/'joint_motion_metrics.csv',index=False)
    (OUT/'physics_checks.json').write_text(json.dumps({'episodes':len(ep),'task_intervals':len(df),'max_current_a':max_current,
      'max_mapping_error_nm':max_map,'max_torque_above_cap_nm':max_cap_excess,'macro_identity_error_m_s':max_identity,'point_contact_identity_error_m_s':max_contact_identity,'passive_torque_error_nm':max_passive,'passed':True},indent=2)+'\n')
    return df,ss,ep


def summarize(df):
    group=['train_seed','train_model','update','eval_model','cohort','task']
    result=df.groupby(group,sort=False).agg(n=('eval_seed','count'),posture=('posture_success','mean'),tracking=('tracking_success','mean'),
      combined=('composite_success','mean'),survived=('episode_success','mean'),survival_s=('survival_s','mean'),
      steady_vx=('steady_mean_vx_m_s','mean'),steady_rmse=('steady_rmse_m_s','mean'),steady_complete=('steady_complete','sum'),
      response_s=('response_time_s','mean'),pitch_rms=('pitch_rms_deg','mean'),roll_rms=('roll_rms_deg','mean'),height_mae=('height_mae_m','mean'),
      common_peak=('common_x_peak_m','mean'),common_rms=('common_x_rms_m','mean'),common_duration=('common_outside_008_longest_s','mean'),
      wheel_sat=('wheel_saturation_fraction','mean'),leg_sat=('leg_saturation_fraction','mean'),rebound=('forward_rebound_m_s','mean')).reset_index()
    result.to_csv(OUT/'milestone_summary.csv',index=False)
    regress=[]
    baseline=result[result['update']==0].set_index(['eval_model','cohort','task'])
    for r in result[result['update']==200].to_dict('records'):
        ref=baseline.loc[(r['eval_model'],r['cohort'],r['task'])]
        regress.append({**{k:r[k] for k in group},**{k+'_vs_v16':r[k]-ref[k] for k in ['posture','tracking','combined','survival_s','steady_rmse','pitch_rms','common_peak']}})
    pd.DataFrame(regress).to_csv(OUT/'cross_dynamics_regressions.csv',index=False)
    result[result['update'].isin([0,200])].to_csv(OUT/'cross_dynamics_results.csv',index=False)
    pairs=[];a=result[result.train_model=='A_legacy'].set_index(['train_seed','update','eval_model','cohort','task'])
    keys=['posture','tracking','combined','survival_s','steady_vx','steady_rmse','pitch_rms','roll_rms','height_mae','common_peak','common_rms','common_duration','wheel_sat','leg_sat','rebound']
    for r in result[result.train_model=='B_rated'].to_dict('records'):
        reference=a.loc[tuple(r[k] for k in ['train_seed','update','eval_model','cohort','task'])]
        pairs.append({**{k:r[k] for k in ['train_seed','update','eval_model','cohort','task']},**{k+'_delta':r[k]-reference[k] for k in keys}})
    pd.DataFrame(pairs).to_csv(OUT/'paired_results.csv',index=False)
    return result,pd.DataFrame(pairs)


def audit(cfg):
    verify(cfg);audits=[json.loads((Path(r['directory'])/'initial_state.json').read_text()) for r in cfg['runs']]
    assert all(x==audits[0] for x in audits)
    counters=[];actual=[]
    from run_m3508_validation import digest
    from copy import deepcopy
    source=cfg['source_training_config']
    stage=deepcopy(cfg['training_stage']);assert stage.pop('recovery_sampling') is False;assert stage.pop('simple_reward_profile')=='baseline'
    assert stage==source['stage']
    env=deepcopy(cfg['training_environment']);assert env['robot'].pop('closed_chain_reset') is True;assert env['robot'].pop('reset_projection_attempts')==8
    assert env==source['environment_config']
    assert cfg['actuator_models']['A_legacy']==source['actuators']
    for run in cfg['runs']:
        done=json.loads((Path(run['directory'])/'completed.json').read_text())
        assert digest(run['checkpoint']+'.zip')==done['checkpoint_sha256']
        tc=json.loads((Path(run['checkpoint']).parent/'training_config.json').read_text());actual.append(tc)
        assert tc['actuators']==cfg['actuator_models'][run['mode']]
        assert tc['stage']==cfg['training_stage'] and tc['environment_config']==cfg['training_environment']
        assert tc['precision_objective']==source['precision_objective']
        assert tc['arguments']['num_envs']*tc['arguments']['rollout_steps']==4096
        for update in MILESTONES:
            path=run['checkpoint']+(f'_iter_{update}' if update<200 else '')
            contract=json.loads(Path(path+'_contract.json').read_text());assert contract['actuator_config']==cfg['actuator_models'][run['mode']]
            for ev in (['A_legacy','B_rated'] if update==200 else ['B_rated']):
                for protocol in PROFILES:
                    metadata=json.loads((OUT/'evaluation'/f'seed_{run["seed"]}'/run['mode']/f'update_{update}'/ev/protocol/'evaluation_config.json').read_text())
                    assert metadata['checkpoint_sha256']==digest(path+'.zip')
                    assert metadata['normalizer_sha256']==digest(path+'_vecnormalize.pkl')
            d,p,_=load_from_zip_file(path+'.zip',device='cpu')
            optimizer=p['policy.optimizer'];steps=sorted(set(float(v['step']) for v in optimizer['state'].values()))
            assert d['num_timesteps']==update*4096
            assert d['_n_updates']==audits[0]['ppo_epoch_update_count']+update*d['n_epochs']
            expected_steps=audits[0]['optimizer_step_values'][0]+update*d['n_epochs']*(4096//d['batch_size'])
            assert steps==[expected_steps]
            counters.append({'seed':run['seed'],'model':run['mode'],'update':update,'environment_steps':d['num_timesteps'],
              'cumulative_ppo_epochs':d['_n_updates'],'adam_steps':steps,'n_epochs':d['n_epochs'],'batch_size':d['batch_size']})
    ignored={'seed','wheel_torque_model','adapt_wheel_torque_model','version','tensorboard_log','initial_state_audit'}
    normalized=[{k:v for k,v in a['arguments'].items() if k not in ignored} for a in actual]
    assert all(x==normalized[0] for x in normalized)
    (OUT/'actual_config_verification.json').write_text(json.dumps({'all_actual_configs_match_frozen_plan':True,'only_treatment':'Wheel map efficiency .70 vs1.00','unchanged_arguments':normalized[0],'source_v16_config_verified':True,'added_stage_defaults':{'recovery_sampling':False,'simple_reward_profile':'baseline (inactive because simple_reward=False)'},'already_authorized_reset_additions':{'closed_chain_reset':True,'reset_projection_attempts':8}},indent=2)+'\n')
    (OUT/'resume_state_verification.json').write_text(json.dumps({'all_six_initial_states_identical':True,'initial_state':audits[0],
      'milestone_counters':counters,'protected_files_unchanged':True,'source_code_hashes_verified':True},indent=2)+'\n')


def decide(summary,pairs):
    b=summary[(summary.eval_model=='B_rated')&(summary.cohort=='historical')]
    base=b[b['update']==0].set_index('task');retreats=[]
    for model in ['A_legacy','B_rated']:
      for task in ['stand','forward2','reverse2','reverse_launch','reverse_stop','high_reversal']:
        for metric in (['posture','tracking'] if task=='reverse_stop' else ['posture']):
          for seed in [1,2,3]:
            q=b[(b.train_model==model)&(b.train_seed==seed)&(b.task==task)].sort_values('update')
            prev=0
            for r in q.to_dict('records'):
                if r[metric]<=base.loc[task,metric]-.20+1e-9:
                    retreats.append({'train_model':model,'train_seed':seed,'task':task,'metric':metric,
                        'earliest_observed_update':r['update'],'onset_after_update':prev,'value':r[metric],'v16_value':float(base.loc[task,metric])});break
                prev=r['update']
    pd.DataFrame(retreats).to_csv(OUT/'earliest_regressions.csv',index=False)
    c=False
    for task in ['stand','forward2','reverse2','reverse_launch','reverse_stop','high_reversal']:
      for metric in (['posture','tracking'] if task=='reverse_stop' else ['posture']):
        q=b[(b.train_model=='B_rated')&(b['update']==200)&(b.task==task)]
        if sum(q[metric]<=base.loc[task,metric]-.20+1e-9)>=2:c=True
    p=pairs[(pairs.eval_model=='B_rated')&(pairs.cohort=='historical')&(pairs['update']==200)&pairs.task.isin(['forward2','reverse2'])]
    improvement=p.groupby('train_seed').steady_rmse_delta.mean()
    gains=p.groupby('train_seed').tracking_delta.mean()
    good=bool(len(improvement)==3 and (improvement<=-.02).all() and (gains>0).all())
    # Explicit holdout confirmation and preservation for every primary capability.
    hold=pairs[(pairs.eval_model=='B_rated')&(pairs.cohort=='independent')&(pairs['update']==200)&pairs.task.isin(['forward2','reverse2'])]
    good &= bool((hold.groupby('train_seed').steady_rmse_delta.mean()<0).all())
    for seed in [1,2,3]:
      for task in ['stand','forward2','reverse2','reverse_launch','reverse_stop','high_reversal']:
        q=b[(b.train_model=='B_rated')&(b.train_seed==seed)&(b['update']==200)&(b.task==task)].iloc[0]
        good &= bool(q.posture>=base.loc[task,'posture']-1e-9)
        if task=='reverse_stop':good &= bool(q.tracking>=base.loc[task,'tracking']-1e-9)
    decision='C' if c else ('A' if good else 'B')
    result={'decision':decision,'B_regression_rule_met':bool(c),'B_superiority_rule_met':bool(good),
      'historical_paired_forward_reverse_rmse_delta_by_train_seed':improvement.to_dict(),
      'historical_paired_tracking_delta_by_train_seed':gains.to_dict(),
      'interpretation':'Frozen practical rule; small n=3, not a statistical equivalence test; regression takes precedence.'}
    (OUT/'decision.json').write_text(json.dumps(result,indent=2)+'\n')
    return result,pd.DataFrame(retreats)


def plots(training,summary,ss):
    colors={'A_legacy':'#4878cf','B_rated':'#e67e22'}
    fig,axes=plt.subplots(3,3,figsize=(15,11),layout='constrained')
    for ax,key in zip(axes.flat,['reward','episode_length','policy_loss','value_loss','approx_kl','pitch_rms_deg','roll_rms_deg','training_survival_rate','height_rmse_m']):
        for (model,seed),g in training.groupby(['mode','train_seed']):
            if key in g:ax.plot(g['update'],g[key],color=colors[model],alpha=.75,lw=.8,label=f'{model} s{seed}')
        ax.set_title(key);ax.set_xlabel('PPO update');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7,ncol=2);fig.savefig(OUT/'training_curves.png',dpi=160);plt.close(fig)
    for filename,metrics in [('speed_tracking_comparison.png',['steady_vx','tracking','combined']),
                             ('posture_comparison.png',['posture','pitch_rms','roll_rms']),
                             ('common_leg_motion.png',['common_peak','common_rms','common_duration'])]:
        tasks=['forward2','reverse2','reverse_launch','reverse_stop','high_reversal'];fig,axes=plt.subplots(len(metrics),len(tasks),figsize=(19,10),layout='constrained')
        for j,task in enumerate(tasks):
          for i,metric in enumerate(metrics):
            ax=axes[i,j];q=summary[(summary.task==task)&(summary.eval_model=='B_rated')&(summary.cohort=='historical')]
            base=q[q['update']==0].iloc[0];ax.axhline(base[metric],color='k',ls=':',label='v16 baseline')
            for (model,seed),g in q[q['update']>0].groupby(['train_model','train_seed']):
                g=g.sort_values('update');x=np.r_[0,g['update']];y=np.r_[base[metric],g[metric]]
                ax.plot(x,y,color=colors[model],marker=['o','s','^'][seed-1],ms=3,lw=.9,alpha=.8,label=f'{model} s{seed}')
            if metric=='steady_vx' and task in ['forward2','reverse2']:
                band=(1.9,2.1) if task=='forward2' else (-2.1,-1.9);ax.axhspan(*band,color='green',alpha=.1)
            if metric in ['posture','tracking','combined']:ax.set_ylim(-.05,1.05)
            if metric=='steady_vx' and task=='reverse_stop':ax.set_ylim(-.06,.06);ax.axhspan(-.05,.05,color='green',alpha=.1)
            ax.set_title(task+' '+metric,fontsize=9);ax.set_xlabel('update');ax.grid(alpha=.2)
        fig.suptitle('Nominal B evaluation / historical seeds; unobserved steady windows remain missing',fontsize=11)
        axes[0,0].legend(fontsize=6);fig.savefig(OUT/filename,dpi=160);plt.close(fig)
    # Every seed shown in cross dynamics; no pooling 8 eval episodes as independent training repeats.
    q=summary[(summary['update'].isin([0,200]))&(summary.cohort=='independent')]
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for ax,task in zip(axes,['forward2','reverse2']):
      for j,ev in enumerate(['A_legacy','B_rated']):
        base=q[(q.train_model=='v16')&(q.eval_model==ev)&(q.task==task)].iloc[0]
        ax.scatter([j-.22],[base.steady_vx],color='k',marker='x',label='v16' if j==0 else None)
        for k,model in enumerate(['A_legacy','B_rated']):
            g=q[(q.train_model==model)&(q.eval_model==ev)&(q.task==task)]
            ax.scatter(np.full(len(g),j+k*.18),g.steady_vx,color=colors[model],label=model if j==0 else None)
      ax.set_xticks([0,1],['eval A','eval B']);ax.set_title(task+' independent seeds');ax.grid(alpha=.2)
    axes[0].legend();fig.savefig(OUT/'cross_dynamics_comparison.png',dpi=160);plt.close(fig)




def matched_windows(cfg):
    rows=[]
    for meta,root in locations(cfg):
        if meta['eval_model']!='B_rated' or meta['update'] not in [0,200]:continue
        records=json.loads((root/'motion/episodes.json').read_text())
        for record in records:
            seed=record['eval_seed'];c=load(root/'motion'/f'{seed}.npz');t=c['time_s']
            mask=(t>11+1e-9)&(t<=11.2+1e-9);complete=bool(t[-1]>=11.2-1e-9)
            r={**meta,'eval_seed':seed,'cohort':'historical' if seed in OLD_SEEDS else 'independent',
              'window_complete':int(complete),'survival_s':record['survival_s']}
            if complete:
                r.update(common_peak_m=float(np.abs(c['common_x_m'][mask]).max()),common_rms_m=float(np.sqrt(np.mean(c['common_x_m'][mask]**2))),
                  pitch_rms_deg=float(np.sqrt(np.mean(np.degrees(c['pitch_rad'][mask])**2))),**controls(c,mask))
            rows.append(r)
    data=pd.DataFrame(rows);data.to_csv(OUT/'matched_reversal_window.csv',index=False)
    result=data.groupby(['train_seed','train_model','update','cohort']).agg(n=('eval_seed','count'),complete=('window_complete','sum'),
      common_peak=('common_peak_m','mean'),common_rms=('common_rms_m','mean'),pitch_rms=('pitch_rms_deg','mean'),
      joint_step_max=('joint_target_step_max_rad','mean'),wheel_sat=('wheel_saturation_fraction','mean')).reset_index()
    result.to_csv(OUT/'matched_reversal_summary.csv',index=False)
    return result


def trajectory_plots():
    # Fixed first historical eval seed, all three trained B policies, no success-based selection.
    seed=OLD_SEEDS[0];palette=['k','#e67e22','#2ca02c','#9467bd']
    def traces(protocol):
        yield 'v16 eval B',load(OUT/'evaluation/baseline/B_rated'/protocol/f'{seed}.npz')
        for train_seed in [1,2,3]:
            yield f'B trained s{train_seed}',load(OUT/'evaluation'/f'seed_{train_seed}'/'B_rated/update_200/B_rated'/protocol/f'{seed}.npz')
    fig,axes=plt.subplots(3,2,figsize=(15,11),layout='constrained')
    for row,protocol in enumerate(['forward2','reverse2','motion']):
        for j,(label,c) in enumerate(traces(protocol)):
            t=c['time_s'];vt=-.055*np.column_stack([c[f'wheel_target_{i}'] for i in range(2)]).mean(axis=1)
            vw=-.055*np.column_stack([c[f'wheel_actual_{i}'] for i in range(2)]).mean(axis=1)
            axes[row,0].plot(t,c['vx_m_s'],color=palette[j],label=label,lw=1)
            axes[row,1].plot(t,vt,color=palette[j],ls=':',lw=.8,label=label+' target')
            axes[row,1].plot(t,vw,color=palette[j],lw=.8)
            if j==0:axes[row,0].plot(t,c['command_vx_m_s'],color='grey',ls='--',label='user command')
        for ax in axes[row]:ax.set_xlabel('time(s)');ax.grid(alpha=.2)
        axes[row,0].set_title(protocol+' body velocity(m/s)');axes[row,1].set_title('Equivalent wheel target(dotted) / actual(solid), m/s')
    axes[0,0].legend(fontsize=8);axes[0,1].legend(fontsize=7);fig.savefig(OUT/'outer_loop_trajectories.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(4,2,figsize=(15,12),layout='constrained')
    for col,(protocol,event) in enumerate([('motion',11),('stop2_reverse',10)]):
        for j,(label,c) in enumerate(traces(protocol)):
            t=c['time_s']-event;mask=(t>=-.5)&(t<=5)
            current=c['motor'][:,:,[c['motor_columns'].index(f'wheel_control_current_a_{i}') for i in range(2)]].mean(axis=(1,2))
            for ax,value in zip(axes[:,col],[c['vx_m_s'],np.degrees(c['pitch_rad']),c['common_x_m'],current]):
                ax.plot(t[mask],value[mask],color=palette[j],label=label,lw=1)
            if j==0:axes[0,col].plot(t[mask],c['command_vx_m_s'][mask],color='grey',ls='--',label='user command')
        for row,name in enumerate(['body vx(m/s)','pitch(deg)','common wheel x(m)','mean control current(A)']):
            ax=axes[row,col];ax.set_title(protocol+' '+name);ax.set_xlabel('time from event(s)');ax.grid(alpha=.2);ax.axvline(0,color='grey',lw=.5)
    axes[0,0].legend(fontsize=8);fig.savefig(OUT/'reversal_trajectories.png',dpi=160);plt.close(fig)


def report(cfg,training,df,ss,ep,summary,pairs,decision,regressions,matched):
    hist=summary[(summary.eval_model=='B_rated')&(summary.cohort=='historical')]
    final=hist[hist['update']==200]
    select=['train_seed','train_model','task','posture','tracking','combined','survival_s','steady_vx','steady_rmse','pitch_rms','common_peak','rebound']
    held=summary[(summary.eval_model=='B_rated')&(summary.cohort=='independent')&(summary['update'].isin([0,200]))]
    speed=ss[ss.task.isin(['forward2','reverse2'])].groupby(['train_seed','train_model','update','eval_model','cohort','task']).mean(numeric_only=True).reset_index()
    speed.to_csv(OUT/'speed_decomposition_summary.csv',index=False)
    comparisons=pairs[(pairs['update']==200)&(pairs.eval_model=='B_rated')]
    pairedsummary=comparisons.groupby(['cohort','task']).agg(rmse_delta_median=('steady_rmse_delta','median'),rmse_delta_min=('steady_rmse_delta','min'),rmse_delta_max=('steady_rmse_delta','max'),
      tracking_delta_median=('tracking_delta','median'),posture_delta_median=('posture_delta','median'),common_peak_delta_median=('common_peak_delta','median')).reset_index()
    pairedsummary.to_csv(OUT/'paired_summary.csv',index=False)
    result=decision['decision'];explanation={'A':'满足事前跨种子改善与能力保留门槛，可继续研究B适应；仍未完成实机标定。',
      'B':'没有达到稳定优于对照的门槛，不直接扩展4000更新。',
      'C':'B续训达到事前能力退化门槛，应保留原v16并停止盲目追加训练。'}[result]
    initial=json.loads((OUT/'resume_state_verification.json').read_text())['initial_state']
    timing=[]
    for run in cfg['runs']:
        seconds=json.loads((Path(run['directory'])/'completed.json').read_text())['seconds']
        timing.append({'train_seed':run['seed'],'train_model':run['mode'],'environment_steps':819200,'wall_seconds':seconds,'steps_per_second':819200/seconds})
    timing=pd.DataFrame(timing);timing.to_csv(OUT/'training_efficiency.csv',index=False)
    # Each question includes measured tables below; expose per training seed and independent cohort.
    def success_counts(task,metric,cohort):
        q=summary[(summary.train_model=='B_rated')&(summary.eval_model=='B_rated')&(summary['update']==200)&(summary.cohort==cohort)&(summary.task==task)].sort_values('train_seed')
        return [f'{int(round(r[metric]*r["n"]))}/{int(r["n"])}' for r in q.to_dict('records')]
    meanfinal=final.groupby(['train_model','task']).mean(numeric_only=True)
    bfinal=speed[(speed['update']==200)&(speed.train_model=='B_rated')&(speed.eval_model=='B_rated')&(speed.cohort=='historical')]
    baseline=speed[(speed['update']==0)&(speed.eval_model=='B_rated')&(speed.cohort=='historical')]
    headline=[]
    for seed in [1,2,3]:
        g=final[(final.train_model=='B_rated')&(final.train_seed==seed)].set_index('task')
        headline.append({'train_seed':seed,'forward_m_s':g.loc['forward2','steady_vx'],'reverse_m_s':g.loc['reverse2','steady_vx'],
          'forward_tracking':f"{int(g.loc['forward2','tracking']*5)}/5",'reverse_tracking':f"{int(g.loc['reverse2','tracking']*5)}/5",
          'high_reversal_posture':f"{int(g.loc['high_reversal','posture']*5)}/5"})
    headline=pd.DataFrame(headline);headline.to_csv(OUT/'headline_results.csv',index=False)
    lines=['# UZ05 v17 第一阶段：PPO动力学适应A/B',
      f'最终决策：**{result}**。{explanation} 没有根据评估结果调整训练超参数或挑选最佳中间checkpoint，以下最终结论均基于Update200。',
      'B续训的最终结果（统一名义B动力学、历史五个评估种子；每行是一个独立训练种子）：',table(headline),
      '三个B训练种子均通过正向5%跟踪，但反向均未通过；站立、后退起步与停车在B下保留。两个种子的高速阶跃换向退化，独立三个评估种子复现同样的成功/失败方向。具体数值及回归A动力学结果见下文，不能用正向提速替代综合能力保留。',
      f'三组配对训练种子1/2/3，每组200更新×4096步，共4,915,200新增环境步。完成{len(ep)}个唯一评估回合、{len(df)}个任务区间，其中60条Update0历史轨迹经源哈希核对后复用，其余实际运行。历史评估种子51000–51004；独立种子61000–61002。所有里程碑均评估全部七项任务；中间里程碑使用统一B动力学及历史种子，Update0和200使用A/B两种动力学及两组种子。',
      '## 唯一变量与续训状态',
      'A轮端映射I×0.246×0.70；B为I×0.246，均保留原电流上限10A与软件力矩硬限幅2.46N·m。默认训练行为仍为A，B必须显式选择，加载异映射checkpoint必须显式允许适应。实际训练继续原相对力矩随机化0.95–1.05，因此A实际随机输出上限约1.636–1.808N·m，B约2.337–2.46N·m；名义上限分别1.722/2.46。B高增益样本被原2.46硬限幅截断，不提高限幅。',
      '原P环Kp1.25、Ki0；腿PD100/4；名义轮阻尼0.02及原0.9–1.1随机化；闭链一致Reset、奖励、动作、原阶跃命令分布、推力和其他域随机化不变。v17是实验版本名，训练stage仍为joint_tracking_v16。与源配置比较，新增recovery_sampling=False及simple_reward_profile=baseline默认字段，前者关闭、后者在simple_reward=False时不生效；闭链Reset字段为前阶段已完成的共同修复。其余原有stage/随机化/执行器A/奖励配置逐项一致。未加入PI、前馈、低阻尼或参考平滑。efficiency字段在这里仅作为输出映射乘数，不表示已测得100%机械效率。B是额定点近似假设，不代表真实标定。',
      f'六组训练前Actor/Critic、Adam、归一化和LR/clip调度、更新计数指纹完全一致。源累计PPO epoch={initial["ppo_epoch_update_count"]}、Adam step={initial["optimizer_step_values"]}。观察归一化关闭，继承相同奖励归一化统计并各自继续更新；评估关闭奖励归一化。',
      '未恢复MuJoCo/环境RNG/全局RNG/rollout状态及各环境折扣return。learn局部timesteps/progress重新计数；源固定LR=3e-5、clip=0.2不受进度重启影响。保留累计epoch与Adam状态，不声称逐轨迹完全断点续训。各里程碑实际计数见resume_state_verification.json。每组完成200更新后，用对应时刻保存的中间权重评估；部分评估与后续独立训练并行。评估没有反馈调整训练。',
      '## 运行规模与耗时',table(timing),
      '部分训练与独立评估并行，系统负载不同；这些是实际运行耗时，不用于证明B模型本身更快或更慢。',
      '## 成功标准与缺失处理',
      '速度沿用历史协议的机身坐标前向分量。沿用上一阶段冻结标准：非零速度±5%、零速±0.05m/s；1s窗口≥95%在带内且RMS≤带，须事件后5s内完成，最后2s复核。姿态要求Pitch RMS≤10°、Roll RMS≤5°、高度MAE≤3cm、严重偏离连续≤0.25s、双轮卸载≤0.1s并存活完整任务区间。姿态/跟踪/综合分别报告。失败导致末窗缺失时留NaN，不填零、不剔除失败；RMS须结合存活和暴露时间解读。24s整段存活不等于局部姿态或速度成功。',
      '## 统一B动力学下的最终逐训练种子结果',table(final[select]),
      '## 独立评估种子确认（与历史种子分开）',table(held[select]),
      '## 返回旧A动力学的回归检查',
      '以下最终策略均用名义A评估，与v16的A基线比较；A中间里程碑未评估，因此若仅在A发现退化，只能定位Update0–200，不能声称50更新时已发生。完整双模型逐种子表见cross_dynamics_results.csv，基线差值见cross_dynamics_regressions.csv。',
      table(summary[(summary.eval_model=='A_legacy')&(summary.cohort=='historical')&(summary['update'].isin([0,200]))][select]),
      '## 配对差值：B训练−A训练，评估动力学统一为B',
      '先按同一训练种子对评估回合取均值，再报告三个训练重复的差值中位数与范围。不能把每个评估回合当作独立训练种子。',table(pairedsummary),
      '## 退化最早出现的测量区间',
      '相对同一评估动力学B下的原v16，能力成功率下降至少20个百分点记录为退化。下面只定位到相邻评估里程碑，不能认定首次失败恰发生于保存更新。主停止规则要求至少两个训练种子的最终B策略退化。首次退化不代表后续一直失败：例如A seed1在150更新退化、200更新恢复；所有里程碑均保留。',
      table(regressions) if len(regressions) else '未触发该退化描述阈值。',
      '## 共同摆腿：匹配换向后相同0.2秒',
      '下面使用11.000–11.200s的相同窗口，仅完整覆盖者计算连续指标，缺失不填零。它是事后描述性暴露核对，不更改事前成功标准；完整任务指标仍保留。',
      table(matched[matched.cohort=='historical']),
      '## 外环目标与低层差额',
      '恒等式v_cmd−v_body=(v_cmd−Vt)+(Vt−Vw)+(Vw−v_body)，Vt/Vw=−0.055×左右目标/实际轮速均值。第三项保留接触几何/腿运动含义，不能全部叫滑移。所有中间与最终分解见steady_speed_decomposition.csv；无完整末2s时缺失。38.73rad/s仅为B纯P固定轮轴参考，未强制策略输出。',
      table(pd.concat([baseline,bfinal])[['train_seed','train_model','update','task','steady_mean_vx_m_s','wheel_target_mean_rad_s','policy_shortfall_m_s','speed_loop_shortfall_m_s','rolling_kinematic_shortfall_m_s']]),
      '## 对十二个问题的回答']
    for number,question in enumerate(['B训练是否改善稳态轮速目标','正向是否稳定达到至少1.9m/s','反向是否稳定达到至少−1.9m/s','误差改善体现在哪层','是否保留站立、后退起步及停车','高速阶跃换向姿态是否保持','共同摆腿改善或恶化','训练种子是否一致','能否区分训练适应与动力学变化','是否值得完整训练','是否开展速度参考课程实验','效果不佳应先分析什么'],1):
        if number==1:
            pieces=[]
            for task in ['forward2','reverse2']:
                ref=baseline[baseline.task==task].iloc[0].wheel_target_mean_rad_s
                vals=bfinal[bfinal.task==task].sort_values('train_seed').wheel_target_mean_rad_s.tolist()
                pieces.append(f'{task}：v16 B下{ref:.3f}rad/s，B续训seed1/2/3为'+str([round(v,3) for v in vals]))
            answer='；'.join(pieces)+'。须同时检查实际速度及姿态，目标变大不自动代表改善。'
        elif number in [2,3]:
            task='forward2' if number==2 else 'reverse2';q=final[(final.train_model=='B_rated')&(final.task==task)].sort_values('train_seed')
            answer='历史种子下各训练种子稳态均值='+str(q.steady_vx.round(4).tolist())+'m/s，完整5%跟踪成功率='+str(q.tracking.round(3).tolist())+'。独立种子结果见上表；不以均值过线替代完整成功标准。'
        elif number==4:
            parts=[]
            for task in ['forward2','reverse2']:
                ref=baseline[baseline.task==task].iloc[0]
                q=bfinal[bfinal.task==task].sort_values('train_seed')
                parts.append(task+'外层差额变化='+str((q.policy_shortfall_m_s-ref.policy_shortfall_m_s).round(4).tolist())+'m/s，内层差额变化='+str((q.speed_loop_shortfall_m_s-ref.speed_loop_shortfall_m_s).round(4).tolist())+'m/s')
            answer='；'.join(parts)+'。正反向保留符号；内层仍是同一个P环，轮速升高时绝对静差也可能增大。改善不是PI/前馈消除低层误差，而是策略给定与新闭环状态改变。'
        elif number==5:
            parts=[]
            for task in ['stand','reverse_launch','reverse_stop']:
                parts.append(task+'姿态：历史'+str(success_counts(task,'posture','historical'))+'，独立'+str(success_counts(task,'posture','independent')))
            parts.append('停车跟踪：历史'+str(success_counts('reverse_stop','tracking','historical'))+'，独立'+str(success_counts('reverse_stop','tracking','independent')))
            answer='；'.join(parts)+'。这些基本任务在B下得到保留，高速换向另有退化；返回A的兼容性另见交叉表。'
        elif number==6:
            retreat=regressions[(regressions.train_model=='B_rated')&(regressions.task=='high_reversal')&(regressions.metric=='posture')]
            answer='没有跨三个种子保持。历史种子姿态成功数='+str(success_counts('high_reversal','posture','historical'))+'，独立种子='+str(success_counts('high_reversal','posture','independent'))+'；原v16分别5/5和3/3。最早观测退化区间='+str([{k:r[k] for k in ['train_seed','onset_after_update','earliest_observed_update']} for r in retreat.to_dict('records')])+'。'
        elif number==7:
            q=final[(final.train_model=='B_rated')&(final.task=='high_reversal')].sort_values('train_seed');ref=hist[(hist['update']==0)&(hist.task=='high_reversal')].iloc[0]
            early=matched[(matched.train_model=='B_rated')&(matched.cohort=='historical')].sort_values('train_seed');eb=matched[(matched.train_model=='v16')&(matched.cohort=='historical')].iloc[0]
            answer=f'完整高速区间v16共同偏移峰值{ref.common_peak:.4f}m，B续训三种子为{q.common_peak.round(4).tolist()}m；但失败暴露时间不同。相同0.2s窗口的v16峰值{eb.common_peak:.4f}m，B三种子为{early.common_peak.round(4).tolist()}m。即使短窗峰值下降，也不能等同于换向稳定改善；两种子随后倒地。共同RMS和超8cm时长另表保留。'
        elif number==8:answer='正向跟踪及平均正反向稳态RMSE改善的方向一致，但反向改善幅度和高速稳定性不一致。B−A的平均正反向RMSE配对差（seed1/2/3）='+str([round(v,4) for v in decision['historical_paired_forward_reverse_rmse_delta_by_train_seed'].values()])+'m/s，负值代表改善。n=3训练重复较小，不能据此证明综合能力稳定提高。'
        elif number==9:answer='可以部分区分。Update0双模型体现纯动力学变化；统一B下v16/A训/B训比较体现适应差异；最终A→A/A→B/B→A/B→B完整保留于cross_dynamics_results.csv。核心结论没有只比较A→A与B→B。B续训三个策略返回A后，±2m/s起步姿态均为历史0/5、独立0/3，零速站立仍保留；后退停车协议常在3s开始后退时已失败，尚未到10s停车，不能把它解释为单独制动失败。A续训seed2在统一B中也失去高速能力，因此不能把所有续训退化都归因于B映射本身。'
        elif number==10:answer=('可继续研究有限B适应，但仍先复核独立种子与能力保留，不据此认定硬件正确。' if result=='A' else '不建议直接延长至4000更新；保留原v16，先分析本次未改善/退化原因。')
        elif number==11:answer='暂不与本轮混合。1.5m/s²参考的既有诊断优势仍有效；应先解决或解释速度目标不足及能力退化，再作为单独课程消融，不能把阶跃成功与平滑命令成功混报。本轮未额外运行平滑参考。'
        else:answer='优先分开检查−2m/s稳态与+2→−2切换的实际加权奖励：track_vx高斯项、vx_error_dense连续误差项、track_vx_coarse，以及stance_geometry对速度奖励的quality门控，再对照pitch/height、动作变化和控制成本。当前已有robust_square连续误差项，不能未经测量就断言速度奖励完全没有梯度。还应核对原阶跃命令分布对极端换向的覆盖与续训遗忘；即时奖励对速度的局部导数不等于PPO策略参数梯度。本次只定位下一步核查对象，未证明某一项是根因，也未修改奖励。'
        lines.append(f'{number}. **{question}？** {answer}')
    lines+=['## 曲线与复现','![训练](training_curves.png)','![速度跟踪](speed_tracking_comparison.png)','![姿态](posture_comparison.png)','![共同摆腿](common_leg_motion.png)','![交叉动力学新种子](cross_dynamics_comparison.png)',
      '下面时序固定使用第一个历史评估种子51000，展示全部三个B训练种子与v16 B基线；没有按成功与否挑轨迹，失败后不补造尾部。统计结论仍以所有配对与独立种子为准。',
      '![外环目标与实际响应](outer_loop_trajectories.png)','![换向与停车瞬态](reversal_trajectories.png)',
      '完整命令见commands.sh；训练权重独立保存在checkpoints/ppo_v17_torque_ab_*，未覆盖v16。离线统计曾优化重复解压读取并重跑汇总，未重训或重评；最终完整日志为analysis_final.log。原XML、奖励、动作/低层控制器源文件哈希保持不变；新增训练开关及评估入口哈希冻结在experiment_config.json。initial_state.json、各组train.log和TensorBoard保留。所有NPZ记录125Hz策略状态与500Hz电机子步。',
      '默认A回归还额外与上一轮相同种子的Projected Reset续训对照：三个训练种子的最终policy及optimizer状态均逐元素一致，见legacy_training_reproduction.json。这不是额外训练，也不代表B结果有效，只验证默认路径没有被新开关改变。全套142项测试通过，随后新增的正式B与诊断B轨迹一致性测试也通过（映射测试组共3项）；测试与预检日志均保留。',
      'training_metrics.csv优先读取TensorBoard，依SB3延后一轮dump对齐；最后一轮未刷新的对应项仅用终端舍入值，不存在的字段保留缺失。三种子结果不作统计显著性或实机泛化证明。']
    (OUT/'report.md').write_text('\n\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser();p.parse_args();cfg=prepare();assert (OUT/'evaluation_complete.json').exists()
    audit(cfg);training=get_training(cfg);training.to_csv(OUT/'training_metrics.csv',index=False)
    df,ss,ep=collect(cfg);summary,pairs=summarize(df);decision,regressions=decide(summary,pairs)
    matched=matched_windows(cfg);plots(training,summary,ss);trajectory_plots();report(cfg,training,df,ss,ep,summary,pairs,decision,regressions,matched)
    print('COMPLETE',len(ep),'episodes',len(df),'task intervals; decision',decision,flush=True)
if __name__=='__main__':main()
