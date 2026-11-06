"""Read-only analysis of motor-model and reference probes; no fitting to outcomes."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_m3508_validation import OUT,ROOT,MODELS,REFERENCES,SEEDS,PROFILES,TASKS,THRESHOLDS,CHECKPOINT,digest,verify,jobs
from uz05.m3508_metrics import score_interval,longest_run,rms
from ppo_reset_ab_evaluation import lag_correlation,wrap


def load(model,ref,protocol,seed):
    with np.load(OUT/model/ref/protocol/f'{seed}.npz') as z:
        a,e=z['samples'],z['extra']
        c={k:a[:,i] for i,k in enumerate(z['columns'])}
        c.update({k:e[:,i] for i,k in enumerate(z['extra_columns'])})
        c['motor']=z['motor'].copy();c['motor_columns']=list(z['motor_columns'])
    return c


def controls(c,mask):
    if not mask.any():return {}
    a=c['motor'][mask];mc=c['motor_columns']
    get=lambda prefix,n:a[:,:,[mc.index(f'{prefix}_{i}') for i in range(n)]]
    leg=get('leg_torque_nm',4);wheel=get('wheel_torque_nm',2)
    cap=a[:,:,mc.index('effective_cap_nm')][:,:,None]
    leg_sat=np.abs(leg)>=19.8;wheel_sat=np.abs(wheel)>=.99*cap
    longest=lambda x:max(longest_run(x[:,:,i].ravel(),.002) for i in range(x.shape[-1]))
    qt=np.column_stack([c[f'target_q_{i}'] for i in range(4)])
    delta=np.vstack([np.zeros((1,4)),wrap(np.diff(qt,axis=0))])[mask]
    actual=np.column_stack([c[f'actual_q_{i}'] for i in range(4)])[mask]
    return {'leg_saturation_fraction':float(leg_sat.mean()),'wheel_saturation_fraction':float(wheel_sat.mean()),
      'leg_saturation_longest_s':longest(leg_sat),'wheel_saturation_longest_s':longest(wheel_sat),
      'leg_torque_peak_nm':float(np.abs(leg).max()),'leg_demand_peak_nm':float(np.abs(get('leg_demand_nm',4)).max()),
      'wheel_torque_peak_nm':float(np.abs(wheel).max()),'wheel_control_current_peak_a':float(np.abs(get('wheel_control_current_a',2)).max()),
      'wheel_equivalent_current_peak_a':float(np.abs(wheel).max()/.246),
      'wheel_torque_above_rated_fraction':float((np.abs(wheel)>2.46+1e-9).mean()),
      'wheel_torque_above_rated_longest_s':longest(np.abs(wheel)>2.46+1e-9),
      'wheel_speed_error_rmse_rad_s':rms(get('wheel_speed_error_rad_s',2)),
      'joint_target_step_max_rad':float(np.abs(delta).max()),'joint_target_step_p95_rad':float(np.percentile(np.abs(delta),95)),
      'joint_target_rate_rms_rad_s':rms(delta/.008),'joint_tracking_rmse_rad':rms(wrap(qt[mask]-actual)),
      'normal_force_mean_n':float(np.column_stack([c[f'normal_force_n_{i}'][mask] for i in range(2)]).mean()),
      'wheel_unloaded_fraction':float(np.column_stack([c[f'normal_force_n_{i}'][mask]<1 for i in range(2)]).mean()),
      'contact_slip_p95_m_s':float(np.percentile(np.column_stack([c[f'contact_slip_m_s_{i}'][mask] for i in range(2)]),95)),
      'friction_utilization_p95':float(np.percentile(np.column_stack([c[f'friction_utilization_{i}'][mask] for i in range(2)]),95)),
      'common_over008_and_wheel_saturated_fraction':float(((np.abs(c['common_x_m'][mask])>.08)&wheel_sat.any(axis=(1,2))).mean()),
      'mapping_error_max_nm':float(c['mapping_error_nm'][mask].max())}


def collect():
    episodes=[];task_rows=[];max_current=0.;max_leg=0.;max_torque=0.;largest_cap_excess=0.
    for model,ref,protocol in jobs():
        raw=json.loads((OUT/model/ref/protocol/'episodes.json').read_text())
        episodes.extend(raw)
        for e in raw:
            c=load(model,ref,protocol,e['eval_seed']);t=c['time_s']
            physical=c['motor'];mc=c['motor_columns']
            current=physical[:,:,[mc.index(f'wheel_control_current_a_{i}') for i in range(2)]]
            leg=physical[:,:,[mc.index(f'leg_torque_nm_{i}') for i in range(4)]]
            torque=physical[:,:,[mc.index(f'wheel_torque_nm_{i}') for i in range(2)]]
            cap=physical[:,:,mc.index('effective_cap_nm')][:,:,None]
            max_current=max(max_current,float(np.abs(current).max()));max_leg=max(max_leg,float(np.abs(leg).max()))
            max_torque=max(max_torque,float(np.abs(torque).max()))
            largest_cap_excess=max(largest_cap_excess,float((np.abs(torque)-cap).max()))
            assert np.isfinite(physical).all() and max_current<=10.+1e-10 and max_leg<=20.+1e-10
            assert largest_cap_excess<1e-9 and e['mapping_error_max_nm']<1e-9
            for name,source,start,end,goal in TASKS:
                if source!=protocol:continue
                mask=(t>start+1e-9)&(t<=end+1e-9)
                row={'model':model,'reference':ref,'task':name,'protocol':protocol,'eval_seed':e['eval_seed'],
                     'survival_s':e['survival_s'],'episode_survival_success':e['success'],
                     'termination_reason':e['termination_reason'],'requested_goal_m_s':goal,
                     **score_interval(c,start,end,goal,THRESHOLDS,e['termination_reason']),**controls(c,mask)}
                early=(t>start+1e-9)&(t<=start+.2+1e-9)
                row.update({'early_'+k:v for k,v in controls(c,early).items()})
                row['early_complete']=int(t[-1]>=start+.2-1e-9)
                row['early_common_x_peak_m']=float(np.abs(c['common_x_m'][early]).max()) if early.any() else np.nan
                corr,lag=lag_correlation(c['common_x_m'][mask],c['pitch_rad'][mask]) if mask.sum()>=10 else (None,None)
                row['common_pitch_lag_correlation']=corr;row['common_pitch_lag_s']=lag
                task_rows.append(row)
    (OUT/'control_bounds_verification.json').write_text(json.dumps({'episodes_checked':len(episodes),
        'max_control_current_a':max_current,'max_leg_torque_nm':max_leg,'max_wheel_torque_nm':max_torque,
        'max_effective_cap_excess_nm':largest_cap_excess,'all_mapping_errors_below_1e_9_nm':True},indent=2)+'\n')
    df=pd.DataFrame(task_rows);ep=pd.DataFrame(episodes)
    df.to_csv(OUT/'task_metrics.csv',index=False);ep.to_csv(OUT/'episode_metrics.csv',index=False)
    keys=['model','reference','task','eval_seed','survival_s','episode_survival_success','interval_complete','exposure_s']
    velocity=['tracking_success','composite_success','requested_goal_m_s','speed_band_m_s','response_time_s','response_window_start_delay_s',
              'steady_complete','steady_tracking_success','steady_mean_vx_m_s','steady_bias_m_s','steady_mae_m_s',
              'steady_rmse_m_s','steady_std_vx_m_s','steady_inband_fraction','steady_1p90_to_1p95','vx_goal_rmse_m_s','vx_reference_rmse_m_s']
    posture=['posture_success','pitch_rms_deg','pitch_std_deg','pitch_peak_deg','pitch_rate_peak_rad_s','roll_rms_deg',
             'height_mae_m','height_peak_m','longest_pitch_excursion_s','longest_roll_excursion_s','longest_height_excursion_s',
             'longest_unsupported_s','common_x_rms_m','common_x_peak_m','common_outside_008_longest_s','common_steady_rms_m','differential_x_rms_m']
    saturation=[k for k in df if any(s in k for s in ('saturation','torque','current','slip','force','friction','unloaded','target_step','tracking_rmse'))]
    df[keys+velocity].to_csv(OUT/'velocity_tracking_metrics.csv',index=False)
    df[keys+posture].to_csv(OUT/'posture_stability_metrics.csv',index=False)
    df[keys+saturation].to_csv(OUT/'saturation_metrics.csv',index=False)
    agg=df.groupby(['model','reference','task'],sort=False).agg(
        n=('eval_seed','count'),intervals_complete=('interval_complete','sum'),posture=('posture_success','sum'),
        tracking=('tracking_success','sum'),combined=('composite_success','sum'),
        episode_survival=('episode_survival_success','sum'),survival_s=('survival_s','mean'),
        speed_bias=('steady_bias_m_s','mean'),speed_rmse=('steady_rmse_m_s','mean'),steady_mean_vx=('steady_mean_vx_m_s','mean'),
        steady_complete_count=('steady_complete','sum'),response_s=('response_time_s','mean'),
        pitch_rms_deg=('pitch_rms_deg','mean'),roll_rms_deg=('roll_rms_deg','mean'),height_mae_m=('height_mae_m','mean'),
        common_peak_m=('common_x_peak_m','mean'),differential_rms_m=('differential_x_rms_m','mean'),
        leg_saturation=('leg_saturation_fraction','mean'),wheel_saturation=('wheel_saturation_fraction','mean'),
        slip_p95=('contact_slip_p95_m_s','mean'),early_complete_count=('early_complete','sum'),
        early_common_peak=('early_common_x_peak_m','mean'),early_target_max=('early_joint_target_step_max_rad','mean'),
        early_wheel_saturation=('early_wheel_saturation_fraction','mean'),early_leg_saturation=('early_leg_saturation_fraction','mean')).reset_index()
    agg.to_csv(OUT/'condition_summary.csv',index=False)
    agg[agg.reference=='step'].to_csv(OUT/'torque_model_comparison.csv',index=False)
    pairs=[]
    fields=['posture_success','tracking_success','composite_success','survival_s','steady_rmse_m_s','pitch_rms_deg',
            'common_x_peak_m','wheel_saturation_fraction','leg_saturation_fraction','early_common_x_peak_m']
    a=df[(df.model=='A_legacy')&(df.reference=='step')].set_index(['task','eval_seed'])
    for row in df[(df.reference=='step')&(df.model!='A_legacy')].to_dict('records'):
        b=a.loc[(row['task'],row['eval_seed'])]
        pairs.append({**{k:row[k] for k in ['model','reference','task','eval_seed']},
                      **{k+'_delta':row.get(k,np.nan)-b[k] for k in fields}})
    pd.DataFrame(pairs).to_csv(OUT/'paired_torque_results.csv',index=False)
    pairs=[]
    for row in df[df.reference!='step'].to_dict('records'):
        b=df[(df.model==row['model'])&(df.reference=='step')&(df.task==row['task'])&(df.eval_seed==row['eval_seed'])].iloc[0]
        pairs.append({**{k:row[k] for k in ['model','reference','task','eval_seed']},
                      **{k+'_delta':row.get(k,np.nan)-b[k] for k in fields}})
    pd.DataFrame(pairs).to_csv(OUT/'paired_reference_results.csv',index=False)
    return df,ep,agg


def table(df):
    def fmt(x):
        if isinstance(x,(float,np.floating)):return '—' if np.isnan(x) else f'{x:.4g}'
        return str(x)
    return '| '+' | '.join(map(str,df.columns))+' |\n|'+'|'.join(['---']*len(df.columns))+'|\n'+'\n'.join('| '+' | '.join(fmt(x) for x in r)+' |' for r in df.itertuples(index=False,name=None))


def plots(df,agg):
    plt.rcParams.update({'font.size':9})
    labels=[x[0] for x in TASKS];colors=plt.cm.tab10(np.arange(8))
    fig,axes=plt.subplots(3,1,figsize=(13,10),layout='constrained')
    for ax,col in zip(axes,['posture','tracking','combined']):
        for j,model in enumerate(MODELS):
            d=agg[(agg.model==model)&(agg.reference=='step')].set_index('task').reindex(labels)
            ax.bar(np.arange(7)+(j-1.5)*.2,d[col]/5,width=.19,label=model,color=colors[j])
        ax.set_ylim(0,1.12);ax.set_yticks([0,.2,.4,.6,.8,1]);ax.set_ylabel(col+' success');ax.set_xticks(range(7),labels)
        ax.grid(axis='y',alpha=.2)
    axes[0].legend(ncol=4);fig.savefig(OUT/'torque_comparison.png',dpi=170);plt.close(fig)
    for model in ['A_legacy','B_rated']:
        fig,axes=plt.subplots(3,1,figsize=(13,10),layout='constrained')
        for ax,task in zip(axes,['forward2','stop2_reverse','motion']):
            for j,ref in enumerate(REFERENCES):
                arrays=[]
                for seed in SEEDS:
                    c=load(model,ref,task,seed);arr=np.full(3000,np.nan);arr[:len(c['vx_m_s'])]=c['vx_m_s'];arrays.append(arr)
                data=np.array(arrays);n=np.isfinite(data).sum(axis=0);med=np.full(3000,np.nan)
                good=n>0;med[good]=np.nanmedian(data[:,good],axis=0)
                ax.plot(np.arange(1,3001)*.008,med,label=ref,color=colors[j])
            # Requested goal, not a smoothed reference: all curves share it.
            t=np.arange(1,3001)*.008
            goal=np.array([PROFILES[task][max(i for i,e in enumerate(PROFILES[task]) if x-.008+1e-9>=e[0])][1] for x in t])
            ax.step(t,goal,where='post',color='black',ls='--',label='requested goal')
            ax.set_title(model+' '+task);ax.set_xlabel('time (s)');ax.set_ylabel('vx (m/s)');ax.grid(alpha=.2)
        axes[0].legend(ncol=5,fontsize=7)
        name='velocity_tracking.png' if model=='B_rated' else 'velocity_tracking_legacy.png'
        fig.savefig(OUT/name,dpi=170);plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(14,11),layout='constrained')
    for row,task in enumerate(['forward2','stop2_reverse','motion']):
      for j,model in enumerate(MODELS):
        for seed in SEEDS:
            c=load(model,'step',task,seed);t=c['time_s']
            axes[row,0].plot(t,c['vx_m_s'],color=colors[j],alpha=.5,lw=.8,label=model if seed==SEEDS[0] else None)
            axes[row,1].plot(t,np.degrees(c['pitch_rad']),color=colors[j],alpha=.5,lw=.8)
        axes[row,0].set_title(task+' velocity (m/s)');axes[row,1].set_title(task+' pitch (deg)')
      for ax in axes[row]:ax.grid(alpha=.2);ax.set_xlabel('time(s)')
    axes[0,0].legend(ncol=2);fig.savefig(OUT/'pitch_response.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained')
    for j,model in enumerate(MODELS):
      for seed in SEEDS:
        c=load(model,'step','motion',seed);mask=(c['time_s']>=10.5)&(c['time_s']<=16)
        axes[0,0].plot(c['time_s'][mask]-11,c['common_x_m'][mask],color=colors[j],alpha=.5,label=model if seed==SEEDS[0] else None)
        axes[1,0].plot(c['time_s'][mask]-11,c['differential_x_m'][mask],color=colors[j],alpha=.5)
    for col,model in enumerate(['A_legacy','B_rated']):
      for j,ref in enumerate(REFERENCES):
        # All paired seeds, lightly drawn; no failed-trajectory extrapolation.
        for seed in SEEDS:
            c=load(model,ref,'motion',seed);mask=(c['time_s']>=10.5)&(c['time_s']<=16)
            axes[col,1].plot(c['time_s'][mask]-11,c['common_x_m'][mask],color=colors[j],alpha=.25,lw=.8,label=ref if seed==SEEDS[0] else None)
      axes[col,1].set_title(model+' references: common x (m)')
    axes[0,0].set_title('Torque models: common x (m)');axes[1,0].set_title('Torque models: left-right x (m)')
    for ax in axes.flat:ax.set_xlabel('time from reversal (s)');ax.axvline(0,color='k',lw=.5);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7);axes[0,1].legend(ncol=2,fontsize=6)
    fig.savefig(OUT/'common_leg_motion.png',dpi=170);plt.close(fig)


def audit():
    verify()
    prefixes=0
    for model in ['A_legacy','B_rated']:
      for ref in REFERENCES[1:]:
       for protocol,event in [('motion',11.),('forward2',3.),('stop2_reverse',3.)]:
        for seed in SEEDS:
            a=load(model,'step',protocol,seed);b=load(model,ref,protocol,seed)
            for key in a:
                if not isinstance(a[key],np.ndarray):continue
                ma=a['time_s']<=event+1e-9;mb=b['time_s']<=event+1e-9
                assert np.array_equal(a[key][ma],b[key][mb]),(model,ref,protocol,seed,key)
            prefixes+=1
    files=['scripts/run_m3508_validation.py','scripts/analyze_m3508_validation.py','scripts/uz05/m3508_diagnostics.py',
           'scripts/uz05/m3508_metrics.py','tests/test_m3508_diagnostics.py','tests/test_m3508_metrics.py','scripts/m3508_validation_findings.py']
    (OUT/'analysis_audit.json').write_text(json.dumps({'all_210_reference_pre_event_prefixes_equal':prefixes,
        'diagnostic_source_sha256':{p:digest(ROOT/p) for p in files}},indent=2)+'\n')


def write_report(df,ep,agg):
    # Detailed conclusions appended after final inspection, using a reproducible
    # findings module. This report also stands alone if only artifacts are read.
    step=agg[agg.reference=='step']
    high=agg[agg.task=='high_reversal']
    lines=['# M3508 力矩模型与稳定优先控制：配对诊断报告','',
      '本轮没有 PPO 重训，没有修改正式奖励、六维动作、PD/速度环增益、闭链 Reset、XML、质量/惯量/碰撞或原始 checkpoint。使用原始 v16，5 个固定种子 51000–51004，Projected Reset、名义动力学、原观测噪声、无移动起步/推力。',
      f'共 {len(ep)} 个 24 s 回合，{len(df)} 个固定任务区间。七项任务来自六个协议，后退起步单独评价 reverse2 的 3–8 s，因此不是额外独立重复。所有原始记录在各模型/参考/协议目录的 NPZ 中。',
      '## 参数审计与模型边界','',
      '完整物理定义、公式、官方来源与未核实项见 [motor_parameter_audit.md](motor_parameter_audit.md)。旧 1.722=10×0.246×0.70；0.246 明确来自输出额定点 2.46/10。在“有效输出比例”语义下再乘效率是重复折损，但产品表的电流口径仍未知，不能由此证明 B 是已标定的真实电机模型。',
      'A 完全复现历史模型；B 为 2.46 N·m 额定点近似；C 在请求 vx 变化后 0.5 s 提升至 3 N·m，其余同 B；D 是 3.69 N·m 乐观上界。C 的时长、增益与 D 的全速度能力都是假设。C/D 保留 10 A 控制变量限制，显式改变输出增益至 0.300/0.369 N·m/control-A，相当于按 0.246 反推 12.195/15 A；这些不是硬件测量电流。',
      '100 项 MuJoCo 电机映射检查通过，bench_verification.json 记录误差。ctrl=输出力矩/gear，actuator_force 为传动前标量，qfrc_actuator 为实际输出轴广义力矩；减速比没有重复使用。所有实际回合再次检查映射。XML gear 始终为 268/17。',
      '## 事前固定的独立成功标准','',
      '- 姿态成功：完整存活到任务区间末；区间 Pitch RMS≤10°、Roll RMS≤5°、高度 MAE≤3 cm；连续 |Pitch|>15°、|Roll|>10°、|高度误差|>6 cm 不超过 0.25 s；两轮同时法向力<1 N 不超过 0.10 s。原环境跌倒终止仍生效。这是本轮冻结的候选评价协议，不是厂家姿态安全规范。',
      '- 跟踪成功：非零命令误差带 ±5%；零命令独立用 ±0.05 m/s。1 s 窗口 RMS≤误差带且≥95% 采样在带内，窗口末必须在事件后 5 s 内（或任务区间末，取更早者）；区间最后 2 s 也必须 RMS≤带且≥95% 在带内。对最终请求速度评分，不能用尚未完成的斜坡参考替代目标。',
      '- 综合成功：姿态与跟踪同时满足。Episode 24 s 存活另报，不能代替这两个指标。首次稳定窗口的 response_time 是窗口**完成**时刻，response_window_start_delay 是进入该窗口的时刻，两者相差 1 s。',
      '- 任务区间：stand 0–24 s；forward2/reverse2 为 3–24 s；reverse_launch 为 3–8 s；reverse_stop 为 10–24 s；moderate_reversal 为 14–20 s；high_reversal 为 11–16 s。局部任务成功不代表随后其他阶段也成功，24 s 存活单独保留。',
      '失败回合全部保留。未到达某区间或未覆盖完整区间，成功置零、不可观测指标留空；不补造尾部轨迹。连续误差均值必须结合 intervals_complete、exposure 和全回合结果解释。',
      '## 单变量力矩对照：原始阶跃','',
      '各行 n=5，posture/tracking/combined 为对应成功数；episode_survival 是整段 24 s 存活数。响应时间均值仅包含达到完整稳定窗口的回合，缺失不能理解成零响应时间。speed_bias/RMSE 为名义最后 2 s 观测值均值，完整样本数见 steady_complete_count；未完整覆盖的回合不会进入稳态跟踪成功。',
      table(step[['model','task','n','intervals_complete','posture','tracking','combined','episode_survival','steady_complete_count','speed_bias','speed_rmse','pitch_rms_deg','common_peak_m']]),
      '## 固定力矩下的速度参考对照','',
      '分别在 A 内和 B 内对照；每一次比较只改变参考。高速协议仅平滑 11 s 换向；forward2 平滑 3 s 前进起步；stop2_reverse 平滑 3 s 后退起步和 10 s 停车。其他命令保持原阶跃，不能把 A-step 与 B-ramp 的差异称为一个因素的效果。斜坡 0.5/1/1.5/2 s；加速度上限 1.5/2/3 m/s²。后者是候选值，未声称已验证硬件能力。',
      table(high[['model','reference','n','posture','tracking','combined','episode_survival','steady_complete_count','steady_mean_vx','speed_rmse','response_s','early_common_peak']]),
      '加速度限制器按每个 8 ms 周期更新；对于 +2→−2，理想参考时长分别约 2.667/2/1.333 s。固定时长斜坡的加速度随命令跨度改变，因此同一“1 s 斜坡”在 0→2 与 +2→−2 上不是同一加速度。当前所有参考在加速/制动方向使用相同数值；方向差异从 forward2/后退起步/停车/换向分别分析。',
      '## 曲线与共同摆腿','',
      '曲线不对失败后的轨迹外推；单回合/逐种子结果见 CSV，不能用不同存活长度的全段 RMS 单独认定改善。common_x 为机体系左右轮心相对髋轴前后偏移均值，differential_x 为两侧差值。共同偏移>8 cm 的持续时长只是描述指标，没有加入奖励或成功惩罚；大幅短时动态补偿不能自动认定为错误。',
      '![三类成功率](torque_comparison.png)','![额定近似下的参考与实际速度](velocity_tracking.png)',
      '![旧模型下的参考与实际速度](velocity_tracking_legacy.png)','![实际速度与Pitch](pitch_response.png)',
      '![共同摆腿与左右差值](common_leg_motion.png)',
      '## 复现与文件','',
      'experiment_config.json 在评估前冻结模型、电流假设、协议、种子和评分阈值。commands.sh 给出映射检查、评估与分析命令；完整运行复用已完成组，拒绝配置变化后复用旧结果。motor_bench.csv/bench_verification.json 为实测仿真映射。',
      'torque_model_comparison.csv 为每模型每任务汇总；velocity_tracking_metrics.csv、posture_stability_metrics.csv、saturation_metrics.csv 为逐种子指标；task_metrics.csv 是全部联合字段。paired_torque_results.csv 是同任务同种子的其他力矩模型减 A；paired_reference_results.csv 是同一模型同种子下参考减阶跃。C-B 的关系须结合 C 只在短时增强的定义解释。',
      'NPZ samples 为周期结束状态/目标，motor 为每周期四个实际电机计算，extra 包括请求速度、轮地快照/滑移及实际 qfrc/ctrl/actuator_force。实际轮端驱动力矩与轮地接触/被动阻尼之和不同。接触力是 125 Hz 快照，电机输出是 500 Hz 子步；没有混淆为硬件实测数据。']
    (OUT/'report.md').write_text('\n\n'.join(lines)+'\n')


def main():
    audit();df,ep,agg=collect();plots(df,agg);write_report(df,ep,agg)
    from m3508_validation_findings import append_findings
    append_findings(OUT,df,ep,agg,load,MODELS,SEEDS,table)
    print('Analyzed',len(ep),'episodes;',len(df),'task intervals',flush=True)

if __name__=='__main__':main()
