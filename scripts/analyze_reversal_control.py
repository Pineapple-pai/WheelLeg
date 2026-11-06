"""Analyze paired reversal probes, including censored event windows and raw traces."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from diagnose_reversal_control import OUT, POLICIES, ARMS, SEEDS, ROOT, digest
from ppo_reset_ab_evaluation import wrap, rms, longest_full_saturation


def longest(x,dt):
    return max(longest_full_saturation(x[:,i],dt) for i in range(x.shape[1]))


def load(policy, arm, task, seed):
    z=np.load(OUT/policy/arm/task/f'{seed}.npz')
    a,e,m=z['samples'],z['extra'],z['motor']
    c={k:a[:,i] for i,k in enumerate(z['columns'])}
    c.update({k:e[:,i] for i,k in enumerate(z['extra_columns'])})
    # Data at command application time, motor substeps before integration.
    c['target_step']=np.r_[0.,np.max(np.abs(wrap(np.diff(a[:,[list(z['columns']).index(f'target_q_{i}') for i in range(4)]],axis=0))),axis=1)]
    targets=a[:,[list(z['columns']).index(f'target_q_{i}') for i in range(4)]]
    c['target_deltas']=np.vstack([np.zeros((1,4)),wrap(np.diff(targets,axis=0))])
    c['target_rate']=c['target_step']/.008
    c['motor']=m
    c['motor_columns']=list(z['motor_columns'])
    c['motor_time']=c['time_s'][:,None]-.008+np.arange(4)[None,:]*.002
    return c


def measures(c,mask):
    if not mask.any(): return {}
    a=c['motor'][mask].reshape(-1,c['motor'].shape[-1])
    mc=c['motor_columns']
    x=lambda prefix,n: a[:,[mc.index(f'{prefix}_{i}') for i in range(n)]]
    q,qd,qt=x('q',4),x('qd',4),x('qd_target_position',4)
    lt,ld=x('leg_torque_nm',4),x('leg_demand_nm',4)
    wt,wa,wtau=x('wheel_target',2),x('wheel_velocity',2),x('wheel_torque_nm',2)
    satleg=np.abs(lt)>=19.8; satwheel=np.abs(wtau)>=.99*1.722
    targets=np.column_stack([c[f'target_q_{i}'][mask] for i in range(4)])
    target_rates=c['target_deltas'][mask]/.008
    wheelsteps=np.diff(wt.reshape(-1,4,2)[:,0],axis=0)
    return {'samples':int(mask.sum()),'exposure_s':float(mask.sum()*.008),
            'target_step_max_rad':float(c['target_step'][mask].max()),
            'target_step_p95_rad':float(np.percentile(c['target_step'][mask],95)),
            'target_joint_step_p95_rad':float(np.percentile(np.abs(target_rates)*.008,95)) if target_rates.size else 0.,
            'target_rate_rms_rad_s':rms(target_rates) if target_rates.size else 0.,
            'requested_step_max_rad':float(c['requested_target_step_peak_rad'][mask].max()),
            'limiter_fraction':float(c['limiter_joint_fraction'][mask].mean()),
            'joint_tracking_rms_rad':rms(wrap(qt-q)), 'joint_tracking_p95_rad':float(np.percentile(np.abs(wrap(qt-q)),95)),
            'actual_joint_speed_p95_rad_s':float(np.percentile(np.abs(qd),95)),
            'actual_joint_speed_max_rad_s':float(np.abs(qd).max()),
            'leg_torque_peak_nm':float(np.abs(lt).max()),'leg_demand_peak_nm':float(np.abs(ld).max()),
            'leg_saturation_fraction':float(satleg.mean()),'leg_saturation_longest_s':longest(satleg,.002),
            'wheel_torque_peak_nm':float(np.abs(wtau).max()),'wheel_saturation_fraction':float(satwheel.mean()),
            'wheel_saturation_longest_s':longest(satwheel,.002),
            'wheel_tracking_rms_rad_s':rms(wt-wa),'wheel_tracking_p95_rad_s':float(np.percentile(np.abs(wt-wa),95)),
            'wheel_target_step_max_rad_s':float(np.abs(wheelsteps).max()) if wheelsteps.size else 0.,
            'common_x_rms_m':rms(c['common_x_m'][mask]), 'common_x_peak_m':float(np.abs(c['common_x_m'][mask]).max()),
            'pitch_rms_deg':float(np.degrees(rms(c['pitch_rad'][mask]))),
            'pitch_rate_peak_rad_s':float(np.abs(c['pitch_rate_rad_s'][mask]).max()),
            'vx_rmse_m_s':rms(c['vx_m_s'][mask]-c['command_vx_m_s'][mask]),
            'normal_force_mean_n':float(np.column_stack([c[f'normal_force_n_{i}'][mask] for i in range(2)]).mean()),
            'wheel_no_contact_fraction':float(np.column_stack([c[f'normal_force_n_{i}'][mask]<1 for i in range(2)]).mean()),
            'friction_utilization_p95':float(np.percentile(np.column_stack([c[f'friction_ratio_{i}'][mask]/np.maximum(c[f'contact_mu_{i}'][mask],1e-12) for i in range(2)]),95)),
            'contact_slip_p95_m_s':float(np.percentile(np.column_stack([c[f'wheel_point_slip_m_s_{i}'][mask] for i in range(2)]),95))}


def collect():
    episodes=[]; events=[]; distributions=[]
    for policy in POLICIES:
      for arm in ARMS:
       for task in ('motion','stop2_reverse'):
        episodes.extend(json.loads((OUT/policy/arm/task/'episodes.json').read_text()))
        for seed in SEEDS:
         c=load(policy,arm,task,seed); t=c['time_s']; episode=episodes[-5+SEEDS.index(seed)]
         focal=11. if task=='motion' else 3.
         for name,mask in [('all',np.ones(len(t),bool)),('pre',(t>focal-.5)&(t<=focal)),('early',(t>focal)&(t<=focal+.2+1e-9))]:
          if mask.any():
           values=np.abs(c['target_deltas'][mask]).ravel()
           distributions.append({'policy':policy,'arm':arm,'task':task,'eval_seed':seed,'window':name,
             **{f'p{p}_rad':float(np.percentile(values,p)) for p in (50,90,95,99,100)},
             'mean_rad':float(values.mean()),'fraction_over_02':float((values>.2).mean()),
             'samples':len(values),'complete':bool(name!='early' or t[-1]>=focal+.2-1e-9)})
         for key,value in measures(c,np.ones(len(t),bool)).items():
             episode['control_'+key]=value
         episode['failure_time_s']=None if episode['success'] else episode['survival_s']
         for event in ([11.] if task=='motion' else [3.,10.]):
          b={'policy':policy,'arm':arm,'task':task,'eval_seed':seed,'event_s':event,
             'survival_s':episode['survival_s'],'success':episode['success'],
             'termination_reason':episode['termination_reason'], 'reached_event':bool(t[-1]>event+1e-8),
             'post_event_survival_s':max(0.,t[-1]-event), 'failure_before_event':bool(t[-1]<=event+1e-8)}
          for window,lo,hi in [('pre',-.5,0),('early',0,.2),('response',0,.25)]:
           mask=(t>event+lo+1e-9)&(t<=event+hi+1e-9)
           b.update({window+'_'+k:v for k,v in measures(c,mask).items()})
           b[window+'_complete']=bool(t[-1]+1e-9>=event+hi)
          post=(t>event+1e-9)&(t<=event+1.+1e-9)
          first=lambda mask:float(t[np.flatnonzero(mask)[0]]) if mask.any() else np.nan
          b['first_large_step_time_s']=first(post&(c['target_step']>.2))
          b['first_pitch_departure_time_s']=first((t>event-.5)&(t<=event+1)&((np.abs(c['pitch_rad'])>.15)|(np.abs(c['pitch_rate_rad_s'])>1)))
          b['pre_pitch_departure']=bool(((t>event-.5)&(t<=event)&((np.abs(c['pitch_rad'])>.15)|(np.abs(c['pitch_rate_rad_s'])>1))).any())
          if post.any():
           indices=np.flatnonzero(post); peak=indices[np.argmax(c['target_step'][post])]
           b['peak_target_time_s']=float(t[peak]); b['peak_target_step_rad']=float(c['target_step'][peak])
           b['peak_target_to_failure_s']=float(t[-1]-t[peak]) if not episode['success'] else np.nan
           b['first_positive_wheel_target_time_s']=first(post&((c['wheel_target_0']+c['wheel_target_1'])/2>0))
           b['first_positive_wheel_actual_time_s']=first(post&((c['wheel_actual_0']+c['wheel_actual_1'])/2>0))
           b['first_negative_vx_time_s']=first(post&(c['vx_m_s']<0))
          events.append(b)
    df=pd.DataFrame(episodes); ev=pd.DataFrame(events)
    pd.DataFrame(distributions).to_csv(OUT/'target_step_distributions.csv',index=False)
    df.to_csv(OUT/'evaluation_metrics.csv',index=False); ev.to_csv(OUT/'event_metrics.csv',index=False)
    summary=df.groupby(['policy','arm','task'])[df.select_dtypes('number').columns.difference(['eval_seed'])].mean().reset_index()
    summary.to_csv(OUT/'summary.csv',index=False)
    base=df[df.arm=='step'].set_index(['policy','task','eval_seed'])
    pairs=[]
    fields=['success','survival_s','control_target_step_max_rad','control_target_rate_rms_rad_s',
            'control_joint_tracking_rms_rad','control_leg_saturation_fraction','control_wheel_saturation_fraction',
            'control_wheel_tracking_rms_rad_s','common_x_peak_m','pitch_rms_deg']
    for row in df[df.arm!='step'].to_dict('records'):
        reference=base.loc[(row['policy'],row['task'],row['eval_seed'])]
        pairs.append({**{k:row[k] for k in ['policy','arm','task','eval_seed']},
                      **{k+'_delta':row[k]-reference[k] for k in fields}})
    pd.DataFrame(pairs).to_csv(OUT/'paired_results.csv',index=False)
    return df,ev,summary


def plots(df,ev):
    plt.rcParams.update({'font.size':9})
    colors=plt.cm.tab10(np.arange(7))
    fig,axs=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for ax,task in zip(axs,('motion','stop2_reverse')):
        table=df[df.task==task].pivot_table(index='policy',columns='arm',values='success').reindex(index=POLICIES,columns=ARMS)
        ax.imshow(table,vmin=0,vmax=1,cmap='YlGn',aspect='auto')
        ax.set_xticks(range(7),[s.replace('slew_','q:').replace('ramp_','r:') for s in ARMS],rotation=35,ha='right')
        ax.set_yticks(range(7),list(POLICIES)); ax.set_title(task+' survival / 5 (24 s)')
        for i in range(7):
            for j in range(7):ax.text(j,i,f'{round(table.iloc[i,j]*5)}/5',ha='center',va='center')
    fig.savefig(OUT/'intervention_success.png',dpi=170);plt.close(fig)
    # All 35 unmodified motion failures, all seeds; no extrapolation beyond death.
    keys=['target_step','actual_qd_0','joint_saturation','wheel_saturation','wheel_target_0','wheel_actual_0','common_x_m','pitch_rad']
    fig,axs=plt.subplots(4,2,figsize=(14,12),layout='constrained')
    for k,ax in zip(keys,axs.flat):
        for pi,policy in enumerate(POLICIES):
            for si,seed in enumerate(SEEDS):
                c=load(policy,'step','motion',seed); t=c['time_s']-11;mask=(t>=-.25)&(t<=.65)
                ax.plot(t[mask],c[k][mask],color=colors[pi],alpha=.35,lw=.8,label=policy if si==0 else None)
        ax.axvline(0,c='black',lw=.6); ax.set_title(k); ax.set_xlabel('time from reversal (s)');ax.grid(alpha=.2)
    axs.flat[0].legend(fontsize=7,ncol=2); fig.savefig(OUT/'event_aligned_all_failures.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(4,2,figsize=(14,12),layout='constrained')
    names=['leg unclipped demand peak (Nm)','leg clipped torque peak (Nm)','wheel error RMS (rad/s)',
           'wheel normal force sum (N)','contact slip max (m/s)','friction ratio / mu max','vx (m/s)','pitch rate (rad/s)']
    for pi,policy in enumerate(POLICIES):
        for seed in SEEDS:
            c=load(policy,'step','motion',seed);t=c['time_s']-11; mask=(t>=-.25)&(t<=.65)
            m=c['motor']; mc=c['motor_columns']
            get=lambda prefix,n:m[:,:,[mc.index(f'{prefix}_{i}') for i in range(n)]]
            values=[np.abs(get('leg_demand_nm',4)).max(axis=(1,2)), np.abs(get('leg_torque_nm',4)).max(axis=(1,2)),
                    np.sqrt(np.mean((get('wheel_target',2)-get('wheel_velocity',2))**2,axis=(1,2))),
                    c['normal_force_n_0']+c['normal_force_n_1'],np.maximum(c['wheel_point_slip_m_s_0'],c['wheel_point_slip_m_s_1']),
                    np.maximum(c['friction_ratio_0']/np.maximum(c['contact_mu_0'],1e-12),c['friction_ratio_1']/np.maximum(c['contact_mu_1'],1e-12)),
                    c['vx_m_s'],c['pitch_rate_rad_s']]
            for ax,y in zip(axs.flat,values):ax.plot(t[mask],y[mask],color=colors[pi],alpha=.35,lw=.7)
    for ax,name in zip(axs.flat,names):ax.set_title(name);ax.axvline(0,c='k',lw=.5);ax.grid(alpha=.2);ax.set_xlabel('time from reversal (s)')
    fig.savefig(OUT/'saturation_contact_response.png',dpi=170);plt.close(fig)
    # Paired event statistics: show every seed instead of only pooled means.
    fig,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    metrics=['early_target_step_max_rad','early_joint_tracking_rms_rad','early_leg_saturation_fraction',
             'early_wheel_tracking_rms_rad_s','early_common_x_peak_m','post_event_survival_s']
    for ax,metric in zip(axs.flat,metrics):
        for pi,policy in enumerate(POLICIES):
            for si,seed in enumerate(SEEDS):
                s=ev[(ev.policy==policy)&(ev.task=='motion')&(ev.eval_seed==seed)].set_index('arm').reindex(ARMS)
                ax.plot(np.arange(7)+.03*(pi-3),s[metric],'.-',color=colors[pi],alpha=.25,lw=.5)
        ax.set_xticks(range(7),list(ARMS),rotation=40,ha='right',fontsize=7);ax.set_title(metric);ax.grid(alpha=.2)
    fig.savefig(OUT/'paired_event_comparison.png',dpi=170);plt.close(fig)
    # Detailed case is illustrative, supplement to the all-episode plot.
    fig,axs=plt.subplots(5,2,figsize=(14,13),layout='constrained')
    for col,task in enumerate(('motion','stop2_reverse')):
        event=11 if task=='motion' else 3
        for pi,policy in enumerate(('v16','s2_legacy','s2_projected')):
            c=load(policy,'step',task,51003); t=c['time_s']-event; mask=(t>=-.3)&(t<=.65)
            axs[0,col].plot(t[mask],c['target_q_0'][mask],color=colors[pi],label=policy+' target')
            axs[0,col].plot(t[mask],c['actual_q_0'][mask],color=colors[pi],ls='--',label=policy+' actual')
            axs[1,col].plot(t[mask],c['wheel_target_0'][mask],color=colors[pi])
            axs[1,col].plot(t[mask],c['wheel_actual_0'][mask],color=colors[pi],ls='--')
            axs[2,col].plot(t[mask],c['wheel_torque_nm_0'][mask],color=colors[pi])
            axs[3,col].plot(t[mask],c['common_x_m'][mask],color=colors[pi])
            axs[4,col].plot(t[mask],np.degrees(c['pitch_rad'][mask]),color=colors[pi])
        for row,title in enumerate(('q0 target/actual (rad)','wheel target/actual (rad/s)','wheel torque (Nm)','common wheel x (m)','pitch (deg)')):
            axs[row,col].set_title(task+': '+title);axs[row,col].axvline(0,c='k',lw=.5);axs[row,col].grid(alpha=.2)
        axs[0,col].legend(fontsize=6)
    fig.savefig(OUT/'joint_wheel_response.png',dpi=170);plt.close(fig)


def table(frame,columns=None):
    if columns: frame=frame[columns]
    # Avoid optional tabulate dependency.
    def fmt(v):
        if isinstance(v,(float,np.floating)):return f'{v:.4g}'
        return str(v)
    return '| '+' | '.join(map(str,frame.columns))+' |\n|'+'|'.join(['---']*len(frame.columns))+'|\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))


def report(df,ev,summary):
    base=ev[(ev.arm=='step')&(ev.task=='motion')]
    motion=df[(df.task=='motion')&(df.arm=='step')]
    launch=ev[(ev.arm=='step')&(ev.task=='stop2_reverse')&(ev.event_s==3)]
    bypolicy=base.groupby('policy',sort=False).agg(peak_step=('early_target_step_max_rad','mean'),
        p95_step=('early_target_step_p95_rad','mean'),target_rate=('early_target_rate_rms_rad_s','mean'),
        q_lag=('early_joint_tracking_rms_rad','mean'),qd_p95=('early_actual_joint_speed_p95_rad_s','mean'),
        leg_sat=('early_leg_saturation_fraction','mean'),leg_sat_s=('response_leg_saturation_longest_s','mean'),
        wheel_err=('early_wheel_tracking_rms_rad_s','mean'),wheel_sat=('early_wheel_saturation_fraction','mean'),
        post_survival=('post_event_survival_s','mean')).reset_index()
    (OUT/'baseline_event_summary.csv').write_text(bypolicy.to_csv(index=False))
    successes=df.pivot_table(index=['policy','task'],columns='arm',values='success').reindex(columns=ARMS)
    successes.to_csv(OUT/'success_rates.csv')
    ordering=(base.first_large_step_time_s<base.first_pitch_departure_time_s).sum()
    mass=float(df.total_mass_kg.iloc[0]); accel=2*1.722/(mass*.055)
    launch_summary=launch.groupby('policy',sort=False).agg(survival_s=('survival_s','mean'),
        target_peak=('early_target_step_max_rad','mean'),target_rate=('early_target_rate_rms_rad_s','mean'),
        q_lag=('early_joint_tracking_rms_rad','mean'),leg_sat=('early_leg_saturation_fraction','mean'),
        wheel_sat=('early_wheel_saturation_fraction','mean'),wheel_error=('early_wheel_tracking_rms_rad_s','mean'),
        common_peak=('early_common_x_peak_m','mean')).reset_index()
    timing=base[['policy','eval_seed','first_large_step_time_s','first_pitch_departure_time_s','peak_target_time_s','survival_s','termination_reason']]
    timing.to_csv(OUT/'failure_timing.csv',index=False)
    # Evidence-only generated section; conclusions are written after inspection.
    lines=['# UZ05 高速换向控制链诊断', '',
      '本轮仅进行评估，没有 PPO 训练，没有修改正式奖励、控制器、动作空间或训练配置。',
      '7 个策略 × 7 个条件 × 2 个任务 × 5 个评估种子，共 490 回合。全部使用 Projected Reset、名义动力学、原观测噪声、无推力/移动起步。种子 51000–51004。成功沿用 24 s time_limit，仅表示存活，跟踪质量另报。',
      '基线五个原始 v16 高速回合与此前记录逐元素完全相同；没有额外噪声采样或控制更新。完整命令见 commands.sh，冻结协议见 experiment_plan.json，完整性校验见 integrity_verification.json。', '',
      '## 控制链与单位', '',
      '- PPO 输出 6 维，顺序左轮、右轮、左 link2/link4、右 link2/link4。确定性 predict 输出经 Box 裁剪到 ±1；轮目标 action×48 rad/s，四腿 action×π rad，按实际角度的最近等效周期映射。',
      '- 策略频率 125 Hz（8 ms）；保持该动作，PD/轮速环每 2 ms 更新一次（500 Hz）。腿 PD 为 τ=100·wrap(q_policy−q)−4·q̇，目标速度为零，软件限幅 ±20 N·m。XML 的 ±39 N·m 不先触发。',
      '- 10.47 rad/s 是项目中的额定速度参数，当前没有被腿控制器使用作速度限制；模型没有实现实机完整的速度—力矩曲线。高速实际速度不能直接解释为硬件可实现。',
      '- 轮速误差先裁剪 ±8 rad/s，乘 1.25 A/(rad/s)，限流 ±10 A；Ki=0。轮端 τ=I×0.246×0.70，因此当前有效上限 1.722 N·m/轮。日志 wheel_torque_nm 为输出轴力矩，不是电机转子力矩。',
      '- ctrl=轮端τ/(268/17)，XML gear=268/17；actuator_force 是传动前标量，qfrc_actuator 是作用于轮关节的广义力矩。两者勿混用。',
      f'- 全部回合 qfrc_actuator 与记录轮端 τ 的最大差为 {df.torque_mapping_max_error_nm.max():.3g} N·m，换算闭合。软件最大 ctrl={1.722/(268/17):.6f}，低于 XML ±0.285448。没有发现减速比重复放大/缩小。',
      '- spec 的 0.246 N·m/A 已由输出侧额定点反推，随后再次乘 0.70 会使 10 A 从 2.46 降为 1.722 N·m。这是需要实机标定核实的模型一致性疑点，本轮保留现状，不把 70% 擅自移除。',
      f'- 总质量 {mass:.5f} kg，半径 0.055 m。忽略轮惯量、腿部动量交换、阻尼及姿态约束，满轮驱动力对应平移加速度约 {accel:.3f} m/s²，改变 4 m/s 至少约 {4/accel:.3f} s（理想轮驱估计，不是完整多体系统严格下界）。',
      '- +2→−2 m/s 的一步阶跃等效 500 m/s² 的参考变化率；0.2/0.5/1 s 斜坡分别 20/8/4 m/s²。命令不是机器人必须瞬时实现的速度约束。理想纯滚动轮速从 −36.36 到 +36.36 rad/s；实际还包含腿和机身运动及滑移。',
      '- 地面/轮接触滑动摩擦系数名义为 1；记录接触实际 μ、切向/法向力比与接触点切向滑移速度。摩擦利用率和失去接触需同时看，不能仅从轮速误差断定打滑。', '',
      '## 原始阶跃：全部 35 个高速失败回合', '',
      '下表每个策略为五个种子的均值。指标统一在换向后前 0.2 s，持续饱和另用前 0.25 s。目标 P95 是每周期四关节最大变化量的 P95；逐关节汇总 P95 另在 event_metrics.csv。目标差全部取最短周期差。',
      table(bypolicy), '',
      f'35 回合中 {(base.early_target_step_max_rad>.2).sum()}/35 在前 0.2 s 发生 >0.2 rad 目标步变；0.2 rad 是零速度时耗尽 20 N·m 的静态位置误差预算，不是失稳阈值。',
      f'相对预先固定的姿态偏离观察阈值（|pitch|>0.15 rad 或 |pitch rate|>1 rad/s），{ordering}/35 的首次大步目标早于首次偏离；有 {base.pre_pitch_departure.sum()}/35 在事件前 0.5 s 已触及阈值。阈值只用于时间排序，没有改变环境终止或成功标准。',
      '逐回合 first_large_step_time、first_pitch_departure_time、peak_target_time 和 failure_time 见 failure_timing.csv；采样分辨率 8 ms，同周期事件无法细分先后。时间先后与限速干预只能支持机制线索，不能单凭相关性证明根因。', '',
      '## 限速与斜坡结果', '',
      '每个单元格为 5 个种子存活率。step 为原始基线；slew_weak/medium/strong 为 25/10.47/5.235 rad/s，对应每周期 0.2/0.08376/0.04188 rad；以实际初始 q 初始化，沿最短周期方向限幅，每 8 ms 更新一次，目标速度保持零。执行后的动作进入 previous_action 观测。介入比例按四关节×全部周期统计，事件窗口另报。',
      'ramp_020/050/100 为 0.2/0.5/1.0 s。高速任务仅 11 s +2→−2 斜坡，之前完全同轨迹；后退停车在 3 s 起步和 10 s 停车使用斜坡。后者同时减轻起步与停车难度，不能当作孤立停车修复。斜坡的完成时刻为不早于指定时长的首个 8 ms 采样点。',
      table(successes.reset_index()), '',
      '完整逐种子结果、失败时刻和配对变化见 evaluation_metrics.csv、event_metrics.csv 和 paired_results.csv。early_complete/response_complete 标记是否完整观察到统一窗口；未到达事件的回合 reached_event=False，不把其 post_event_survival=0 当作经历了换向。不同存活时间的全段 RMS 不单独用于改善判断。', '',
      '## 原始 v16 与续训的普通后退起步/停车', '',
      '以下为 3 s 的 0→−2 m/s 起步后 0.2 s 统一窗口；停车在 10 s，早于 10 s 失败不能解释为制动失败。', table(launch_summary), '',
      '## 图与原始记录', '',
      '![所有高速基线回合的事件对齐曲线](event_aligned_all_failures.png)',
      '![配对事件窗口指标](paired_event_comparison.png)',
      '![力矩、轮速误差与接触](saturation_contact_response.png)',
      '![介入条件存活率](intervention_success.png)',
      '![代表种子关节及轮响应](joint_wheel_response.png)', '',
      '每回合 NPZ 同时保存 125 Hz 状态/目标以及每周期四个 500 Hz 电机计算输入、未限幅需求、限幅后力矩和电流。关节顺序同 PPO 的四腿通道。motor 的 qd_target_position 列名指位置目标，不是速度前馈；所有速度目标仍为零。电机计算发生在子步积分之前，samples 状态为周期结束后，时轴差异已记录。接触力/滑移是周期结束时快照，没有冒充完整 500 Hz 接触数据。',
      '本次沿用既有五个种子进行配对诊断，属于对既有结果的机制探索；n=5 不是更多独立训练种子，不能当作大样本泛化验证。']
    (OUT/'report.md').write_text('\n\n'.join(lines)+'\n')


def audit():
    plan=json.loads((OUT/'experiment_plan.json').read_text())
    assert all(digest(ROOT/p)==h for p,h in plan['core_sha256'].items())
    assert all(digest(str(POLICIES[k])+'.zip')==h for k,h in plan['checkpoint_sha256'].items())
    assert all(digest(str(POLICIES[k])+'_vecnormalize.pkl')==h for k,h in plan['normalizer_sha256'].items())
    original_matches=[]; ramp_matches=[]; largest_excess=0.
    for policy in POLICIES:
      for task in ('motion','stop2_reverse'):
       for seed in SEEDS:
        new=np.load(OUT/policy/'step'/task/f'{seed}.npz')['samples']
        if policy=='v16':oldpath=ROOT/f'diagnostics/ppo_reset_ab/baseline/{task}_{seed}.npz'
        else:
            training_seed,mode=policy.split('_')
            oldpath=ROOT/f'diagnostics/ppo_reset_ab/seed_{training_seed[1:]}/{mode}/eval_200/{task}_{seed}.npz'
        old=np.load(oldpath)['samples']
        assert new.shape==old.shape and np.array_equal(new,old), (policy,task,seed)
        original_matches.append([policy,task,seed])
        for arm,(rate,ramp) in ARMS.items():
            z=np.load(OUT/policy/arm/task/f'{seed}.npz');e=z['extra'];ec=list(z['extra_columns'])
            if rate is not None:
                excess=float(e[:,ec.index('applied_target_step_peak_rad')].max()-rate*.008)
                largest_excess=max(largest_excess,excess)
                assert excess<1e-10,(policy,task,seed,arm,excess)
            if ramp and task=='motion':
                mask=new[:,0]<=11+1e-9
                other=z['samples'];other=other[other[:,0]<=11+1e-9]
                assert np.array_equal(new[mask],other),(policy,seed,arm)
                ramp_matches.append([policy,seed,arm])
    result={'core_checkpoint_normalizer_unchanged':True,'all_70_unmodified_baseline_traces_exactly_reproduce_previous_experiment':True,
            'all_105_motion_ramp_pre_event_traces_exactly_equal_baseline':True,
            'limiter_bound_max_excess_rad':largest_excess,'baseline_trace_count':len(original_matches),
            'ramp_pre_event_trace_count':len(ramp_matches),
            'diagnostic_source_sha256':{p:digest(ROOT/p) for p in ['scripts/diagnose_reversal_control.py','scripts/analyze_reversal_control.py','scripts/uz05/reversal_diagnostics.py','tests/test_reversal_diagnostics.py','scripts/reversal_control_findings.py']}}
    (OUT/'diagnostic_audit.json').write_text(json.dumps(result,indent=2)+'\n')


def main():
    audit(); df,ev,summary=collect(); plots(df,ev);report(df,ev,summary)
    from reversal_control_findings import append_findings
    append_findings(OUT,df,ev,load,POLICIES,SEEDS,table)
    print('Analyzed',len(df),'episodes;',len(ev),'events',flush=True)

if __name__=='__main__':main()
