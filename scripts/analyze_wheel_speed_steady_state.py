"""Paired scoring and exact velocity-error decomposition, without PPO training."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_wheel_speed_steady_state import OUT,ROOT,SEEDS,PILOT,CONDITIONS,PROFILES,TASKS,THRESHOLDS,jobs,verify_protected,digest
from uz05.m3508_metrics import score_interval,rms
from analyze_m3508_validation import controls,table


def load(condition,protocol,seed):
    with np.load(OUT/condition/protocol/f'{seed}.npz') as z:
        a,e=z['samples'],z['extra']
        c={k:a[:,i] for i,k in enumerate(z['columns'])}
        c.update({k:e[:,i] for i,k in enumerate(z['extra_columns'])})
        c['motor']=z['motor'].copy();c['motor_columns']=list(z['motor_columns'])
    return c


def decomposition(c,mask,goal):
    mc=c['motor_columns'];motor=c['motor'][mask]
    get=lambda prefix,n:motor[:,:,[mc.index(f'{prefix}_{i}') for i in range(n)]]
    target=np.column_stack([c[f'wheel_target_{i}'][mask] for i in range(2)])
    actual=np.column_stack([c[f'wheel_actual_{i}'][mask] for i in range(2)])
    vt=-.055*target.mean(axis=1);va=-.055*actual.mean(axis=1);body=c['vx_m_s'][mask]
    outer=goal-vt;inner=vt-va;geometry=va-body
    torque=get('wheel_torque_nm',2);speed_error=get('wheel_speed_error_rad_s',2)
    passive=np.column_stack([c[f'wheel_passive_nm_{i}'][mask] for i in range(2)])
    pred=np.column_stack([c[f'wheel_viscous_predicted_nm_{i}'][mask] for i in range(2)])
    result={'steady_mean_vx_m_s':float(body.mean()),'goal_minus_body_m_s':float((goal-body).mean()),
      'target_equivalent_vx_m_s':float(vt.mean()),'actual_wheel_equivalent_vx_m_s':float(va.mean()),
      'policy_shortfall_m_s':float(outer.mean()),'speed_loop_shortfall_m_s':float(inner.mean()),
      'rolling_kinematic_shortfall_m_s':float(geometry.mean()),
      'macro_identity_max_error_m_s':float(np.abs((goal-body)-(outer+inner+geometry)).max()),
      'wheel_target_mean_rad_s':float(target.mean()),'wheel_actual_mean_rad_s':float(actual.mean()),
      'wheel_error_mean_rad_s':float(speed_error.mean()),'wheel_error_rms_rad_s':rms(speed_error),
      'motor_abs_torque_nm':float(np.abs(torque).mean()),'passive_abs_torque_nm':float(np.abs(passive).mean()),
      'passive_model_max_error_nm':float(np.abs(passive-pred).max()),
      'viscous_power_per_wheel_w':float((-passive*actual).mean()),
      'motor_power_per_wheel_w':float((torque*get('wheel_actual',2)).mean()),
      'wheel_saturation_fraction':float((np.abs(get('wheel_control_current_a',2))>=9.9).mean()),
      'integral_abs_current_a':float(np.abs(get('integral_current',2)).mean()),
      'ff_abs_current_a':float(np.abs(get('ff_current',2)).mean()),
      'antiwindup_block_fraction':float(get('antiwindup_blocked',2).mean()),
      'steady_pitch_std_deg':float(np.degrees(c['pitch_rad'][mask]).std()),
      'steady_vx_std_m_s':float(body.std())}
    components={name:np.column_stack([c[f'{name}_{i}'][mask] for i in range(2)]) for name in ['signed_slip_body_vx','normal_body_vx','spin_body_vx','carrier_body_vx','center_minus_base_vx','kinematic_identity_error']}
    valid=np.ones(len(body),dtype=bool)
    for value in components.values():valid&=np.isfinite(value).all(axis=1)
    valid&=(np.column_stack([c[f'normal_force_n_{i}'][mask] for i in range(2)])>1.).all(axis=1)
    result['both_contact_valid_fraction']=float(valid.mean())
    if valid.any():
        v=lambda key:components[key][valid].mean(axis=1)
        spin_geometry=v('spin_body_vx')-.055*actual[valid].mean(axis=1)
        carrier=v('carrier_body_vx');center=v('center_minus_base_vx');slip=v('signed_slip_body_vx');normal=v('normal_body_vx')
        rebuilt=spin_geometry+carrier+center-slip-normal
        result.update(contact_spin_geometry_m_s=float(spin_geometry.mean()),contact_carrier_rotation_m_s=float(carrier.mean()),
            contact_center_relative_motion_m_s=float(center.mean()),contact_signed_slip_m_s=float(slip.mean()),
            contact_normal_motion_m_s=float(normal.mean()),
            contact_reconstruction_max_error_m_s=float(np.abs(rebuilt-geometry[valid]).max()),
            point_identity_max_error_m_s=float(np.abs(components['kinematic_identity_error'][valid]).max()))
    return result


def collect(seeds):
    episodes=[];tasks=[];steady=[]
    passive_error=kin_error=torque_error=max_current=max_integral=0.
    for condition,protocol in jobs():
      records=json.loads((OUT/condition/protocol/'episodes.json').read_text())
      selected=[r for r in records if r['eval_seed'] in seeds]
      assert {r['eval_seed'] for r in selected}==set(seeds),(condition,protocol)
      episodes.extend(selected)
      for e in selected:
        c=load(condition,protocol,e['eval_seed']);t=c['time_s'];mc=c['motor_columns'];motor=c['motor']
        torque_error=max(torque_error,e['mapping_error_max_nm'])
        current=motor[:,:,[mc.index(f'wheel_control_current_a_{i}') for i in range(2)]]
        integral=motor[:,:,[mc.index(f'integral_current_{i}') for i in range(2)]]
        max_current=max(max_current,float(np.abs(current).max()));max_integral=max(max_integral,float(np.abs(integral).max()))
        assert max_current<=10+1e-9 and max_integral<=4+1e-9
        for i in range(2):
            passive_error=max(passive_error,float(np.abs(c[f'wheel_passive_nm_{i}']-c[f'wheel_viscous_predicted_nm_{i}']).max()))
            errors=c[f'kinematic_identity_error_{i}'];finite=np.isfinite(errors)
            if finite.any():kin_error=max(kin_error,float(np.abs(errors[finite]).max()))
        for name,source,start,end,goal in TASKS:
            if source!=protocol:continue
            mask=(t>start+1e-9)&(t<=end+1e-9)
            r={'condition':condition,'protocol':protocol,'task':name,'eval_seed':e['eval_seed'],
              'survival_s':e['survival_s'],'episode_survival_success':e['success'],
              'termination_reason':e['termination_reason'],**score_interval(c,start,end,goal,THRESHOLDS,e['termination_reason']),
              **controls(c,mask)}
            early=(t>start+1e-9)&(t<=start+.2+1e-9)
            r.update({'early_'+k:v for k,v in controls(c,early).items()});r['early_complete']=int(t[-1]>=start+.2-1e-9)
            r['early_common_peak_m']=float(np.abs(c['common_x_m'][early]).max()) if early.any() else None
            tasks.append(r)
            if r.get('steady_complete')==1:
                sm=(t>end-2+1e-9)&(t<=end+1e-9)
                result=decomposition(c,sm,goal)
                assert result['macro_identity_max_error_m_s']<1e-9
                assert result.get('contact_reconstruction_max_error_m_s',0.)<1e-9
                steady.append({'condition':condition,'task':name,'eval_seed':e['eval_seed'],**result})
    assert torque_error<1e-9 and passive_error<1e-9 and kin_error<1e-9
    checks={'episodes':len(episodes),'tasks':len(tasks),'max_control_current_a':max_current,'max_integral_current_a':max_integral,
        'max_torque_mapping_error_nm':torque_error,'max_actual_passive_minus_bomega_nm':passive_error,
        'max_point_kinematic_identity_error_m_s':kin_error,'all_checks_passed':True}
    (OUT/'physics_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    df=pd.DataFrame(tasks);ss=pd.DataFrame(steady);ep=pd.DataFrame(episodes)
    df.to_csv(OUT/'task_metrics.csv',index=False);ss.to_csv(OUT/'steady_state_decomposition.csv',index=False);ep.to_csv(OUT/'episode_metrics.csv',index=False)
    summary=df.groupby(['condition','task'],sort=False).agg(n=('eval_seed','count'),
      posture=('posture_success','sum'),tracking=('tracking_success','sum'),combined=('composite_success','sum'),
      survived=('episode_survival_success','sum'),complete=('interval_complete','sum'),survival_s=('survival_s','mean'),
      steady_vx=('steady_mean_vx_m_s','mean'),speed_bias=('steady_bias_m_s','mean'),speed_rmse=('steady_rmse_m_s','mean'),
      response_s=('response_time_s','mean'),pitch_rms=('pitch_rms_deg','mean'),roll_rms=('roll_rms_deg','mean'),
      common_peak=('common_x_peak_m','mean'),common_duration=('common_outside_008_longest_s','mean'),
      wheel_sat=('wheel_saturation_fraction','mean'),leg_sat=('leg_saturation_fraction','mean'),
      slip_p95=('contact_slip_p95_m_s','mean'),early_common_peak=('early_common_peak_m','mean')).reset_index()
    summary.to_csv(OUT/'condition_summary.csv',index=False)
    pairs=[];base=df[df.condition=='base'].set_index(['task','eval_seed'])
    for r in df[df.condition!='base'].to_dict('records'):
        reference=base.loc[(r['task'],r['eval_seed'])]
        pairs.append({**{k:r[k] for k in ['condition','task','eval_seed']},
          **{k+'_delta':r.get(k,np.nan)-reference[k] for k in ['posture_success','tracking_success','composite_success','survival_s','steady_rmse_m_s','pitch_rms_deg','common_x_peak_m','wheel_saturation_fraction']}})
    pd.DataFrame(pairs).to_csv(OUT/'paired_results.csv',index=False)
    return df,ss,ep,summary


def gate(df,ss):
    bench=json.loads((OUT/'isolated_verification.json').read_text());detail={};passing=[]
    for candidate in ['d010','d005','PI','P_FF']:
        changes=[];preserved=True
        for task in ['forward2','reverse2']:
          for seed in PILOT:
            a=ss[(ss.condition=='base')&(ss.task==task)&(ss.eval_seed==seed)]
            b=ss[(ss.condition==candidate)&(ss.task==task)&(ss.eval_seed==seed)]
            ratio=float(b.iloc[0].wheel_error_rms_rad_s/a.iloc[0].wheel_error_rms_rad_s) if len(a)==1 and len(b)==1 else None
            changes.append({'task':task,'seed':seed,'wheel_error_ratio':ratio})
        for task in ['stand','forward2','reverse2']:
          for seed in PILOT:
            a=df[(df.condition=='base')&(df.task==task)&(df.eval_seed==seed)].iloc[0]
            b=df[(df.condition==candidate)&(df.task==task)&(df.eval_seed==seed)].iloc[0]
            preserved&=bool(b.posture_success>=a.posture_success)
        passed=all(x['wheel_error_ratio'] is not None and x['wheel_error_ratio']<=.75 for x in changes) and preserved
        detail[candidate]={'paired_wheel_errors':changes,'basic_posture_preserved':preserved,'passed':passed}
        if passed:passing.append(candidate)
    result={'passed':bool(bench['p_theory_verified'] and passing),'passing_interventions':passing,
            'details':detail,'theory_verified':bench['p_theory_verified'],
            'meaning':'Clear low-level effect across3pairedseeds, not proof of body tracking improvement. Expand ALL conditions uniformly.'}
    (OUT/'expansion_gate.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PILOT GATE',result['passed'],passing,flush=True)


def transparency(seeds):
    comparisons=[]
    for condition,oldmodel,ref in [('legacy_A','A_legacy','step'),('base','B_rated','step'),('ref15','B_rated','accel_1.5')]:
      protocols=PROFILES if condition!='ref15' else ['motion','forward2','stop2_reverse']
      for protocol in protocols:
       for seed in seeds:
        with np.load(OUT/condition/protocol/f'{seed}.npz') as z:a=z['samples']
        with np.load(ROOT/f'diagnostics/m3508_torque_validation/{oldmodel}/{ref}/{protocol}/{seed}.npz') as z:b=z['samples']
        assert np.array_equal(a,b),(condition,protocol,seed)
        comparisons.append([condition,protocol,seed])
    verify_protected()
    files=['scripts/run_wheel_speed_steady_state.py','scripts/wheel_speed_motor_bench.py','scripts/analyze_wheel_speed_steady_state.py',
           'scripts/uz05/wheel_speed_diagnostics.py','tests/test_wheel_speed_diagnostics.py','scripts/wheel_speed_findings.py']
    (OUT/'integrity_verification.json').write_text(json.dumps({'protected_files_unchanged':True,
       'baseline_and_reference_exact_trace_count':len(comparisons),'source_sha256':{p:digest(ROOT/p) for p in files}},indent=2)+'\n')


def plots(df,ss,summary,seeds):
    colors=plt.cm.tab10(np.arange(7));names=list(CONDITIONS)
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for ax,target in zip(axes.flat,[20.,-20.,36.36,-36.36]):
      for j,scheme in enumerate(['P','PI','P_FF']):
        with np.load(OUT/f'isolated/B_rated_0.02_{scheme}_{target}.npz') as z:a=z['samples']
        ax.plot(a[:,0],a[:,2],color=colors[j],label=scheme)
      ax.axhline(target,color='k',ls='--');ax.set_xlim(0,20);ax.set_title(f'Fixed target {target} rad/s');ax.set_xlabel('time(s)');ax.grid(alpha=.2)
    axes[0,0].legend();fig.savefig(OUT/'isolated_speed_control.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(14,11),layout='constrained')
    for row,task in enumerate(['forward2','reverse2','motion']):
      for j,condition in enumerate(names):
        if condition=='ref15' and task=='reverse2':continue
        for seed in seeds:
            c=load(condition,task,seed);t=c['time_s']
            axes[row,0].plot(t,c['vx_m_s'],color=colors[j],alpha=.35,lw=.7,label=condition if seed==seeds[0] else None)
            axes[row,1].plot(t,np.degrees(c['pitch_rad']),color=colors[j],alpha=.35,lw=.7)
      axes[row,0].set_title(task+' body velocity (m/s)');axes[row,1].set_title(task+' pitch (deg)')
      for ax in axes[row]:ax.grid(alpha=.2);ax.set_xlabel('time(s)')
    axes[0,0].legend(ncol=4,fontsize=7);fig.savefig(OUT/'body_velocity_posture.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for ax,task in zip(axes,['forward2','reverse2']):
        data=ss[ss.task==task].groupby('condition')[['policy_shortfall_m_s','speed_loop_shortfall_m_s','rolling_kinematic_shortfall_m_s']].mean().reindex(names)
        data=data.dropna(how='all');sign=1 if task=='forward2' else -1
        bottom=np.zeros(len(data))
        for col,color,label in zip(data.columns,['#4878cf','#ef8c3b','#6acc65'],['PPO target','speed loop','contact / kinematics']):
            values=sign*data[col].to_numpy();ax.bar(np.arange(len(data)),values,bottom=bottom,label=label,color=color);bottom+=values
        ax.axhline(.1,c='k',ls='--',label='5% band');ax.set_xticks(range(len(data)),data.index,rotation=25);ax.set_title(task+' directional speed deficit (m/s)');ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8);fig.savefig(OUT/'steady_error_decomposition.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for j,condition in enumerate(names):
        for seed in seeds:
            c=load(condition,'motion',seed);t=c['time_s']-11;mask=(t>=-.3)&(t<=3)
            mc=c['motor_columns'];m=c['motor'];error=m[:,:,[mc.index(f'wheel_speed_error_rad_s_{i}') for i in range(2)]]
            current=m[:,:,[mc.index(f'integral_current_{i}') for i in range(2)]]
            axes[0,0].plot(t[mask],c['common_x_m'][mask],color=colors[j],alpha=.35,label=condition if seed==seeds[0] else None)
            axes[1,0].plot(t[mask],np.degrees(c['pitch_rad'][mask]),color=colors[j],alpha=.35)
            axes[0,1].plot(t[mask],np.sqrt(np.mean(error**2,axis=(1,2)))[mask],color=colors[j],alpha=.35)
            axes[1,1].plot(t[mask],current.mean(axis=(1,2))[mask],color=colors[j],alpha=.35)
    for ax,title in zip(axes.flat,['common x(m)','wheel error RMS(rad/s)','pitch(deg)','integral current(A)']):
        ax.set_title(title);ax.axvline(0,color='k',lw=.5);ax.set_xlabel('time from reversal(s)');ax.grid(alpha=.2)
    axes[0,0].legend(ncol=3,fontsize=7);fig.savefig(OUT/'reversal_control_comparison.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(13,10),layout='constrained')
    task_names=[x[0] for x in TASKS]
    for ax,key in zip(axes,['posture','tracking','combined']):
        for j,condition in enumerate(names):
            q=summary[summary.condition==condition].set_index('task').reindex(task_names)
            ax.bar(np.arange(7)+(j-3)*.115,q[key]/len(seeds),width=.11,color=colors[j],label=condition)
        ax.set_xticks(range(7),task_names);ax.set_ylim(0,1.13);ax.set_ylabel(key+' success');ax.grid(axis='y',alpha=.2)
    axes[0].legend(ncol=7,fontsize=7);fig.savefig(OUT/'success_comparison.png',dpi=170);plt.close(fig)


def write_report(df,ss,ep,summary,seeds):
    selected=summary[summary.task.isin(['stand','forward2','reverse2','moderate_reversal','high_reversal','reverse_stop'])]
    steady=ss[ss.task.isin(['forward2','reverse2'])].groupby(['condition','task'],sort=False).mean(numeric_only=True).reset_index()
    steady.to_csv(OUT/'steady_summary.csv',index=False)
    bench=pd.read_csv(OUT/'isolated_metrics.csv');powers=pd.read_csv(OUT/'damping_power.csv')
    primary=bench[(bench.model=='B_rated')&(bench.damping==.02)]
    full=json.loads((OUT/'fixed_robot_summary.json').read_text())
    lines=['# UZ05 轮毂电机稳态速度误差诊断','',
      f'完成 {len(ep)} 个 PPO 固定权重配对回合，{len(df)} 个任务区间，另有 12 个不使用 PPO 决策的整机固定轮速试验、40 个独立轮轴试验。先用 3 个种子完成小规模诊断，扩展规则见 expansion_gate.json；本报告使用种子 {seeds}。未训练 PPO、未改正式 XML/奖励/动作映射/腿部 PD 或原 checkpoint。',
      '## 事前冻结的条件与成功标准','',
      'base 为 B_rated、阻尼 0.02、原 P 环和原阶跃；d010/d005 只改变轮阻尼；PI 只加入 Ki=0.5 A/rad，积分电流限幅±4 A、条件抗饱和；P_FF 只加入 b·ω_target/0.246 前馈；ref15 只改变候选速度参考为 1.5 m/s²。legacy_A 为历史 1.722 N·m 模型。所有总电流仍≤10 A，B 轮端输出≤2.46 N·m。未同时改变阻尼、PI 和参考。',
      '沿用上一轮事前固定的姿态/跟踪标准：Pitch RMS≤10°、Roll RMS≤5°、高度 MAE≤3 cm，严重姿态/高度偏离连续≤0.25 s、两轮同时卸载≤0.10 s；速度误差带为非零命令±5%、零速±0.05 m/s，1 s窗口≥95%在带内且RMS≤带，须5s内完成，最后2s继续满足。综合成功要求姿态和跟踪同时通过。完整阈值、任务区间与失败处理见 experiment_config.json。速度指向最终用户命令，不是尚未完成的参考。',
      '## 阻尼来源与物理一致性','',
      '实际训练模型为根目录 uz05_修正后(3).xml。左右轮 joint 显式 damping=0.02，编译后的 dof_damping 仍为0.02；该值位于输出关节，不再乘 gear。原训练的域随机化对轮与主动腿阻尼采用0.9–1.1倍缩放，本次名义对照关闭随机化。compiled_joint_losses.csv 与 damping_audit.json 保存实际参数。',
      '其他主动/被动腿关节也有0.02阻尼。wheel frictionloss=0、stiffness=0，armature=0.0001是惯性而非耗散。全模型流体密度/黏性为0，motor无速度偏置/电气损耗动态；实际接触 condim=3，切向接触摩擦可能耗散，但所列旋转/滚动摩擦系数不会在condim=3中成为独立滚阻模型。闭链/接触约束力、重力与主动腿Kd也不能混为轮关节被动阻尼。',
      'A 的0.70是恒定输出映射折减，B已不使用它；no_load_current_a=0.63未参与执行器动力学计算。0.246来自输出侧额定点，若0.02想代表同一内部损耗，存在重复扣损的疑点；若它代表额外轮轴/轮胎/机构损耗则未必重复。现有源文件没有明确的实机阻尼标定证据，不能宣布0.02必然错误。',
      '不同转速的 τ_b=bω、P_b=bω² 如下；额定571rpm、空载587rpm与空载0.63A均需确认测量口径。',table(powers),
      '587rpm时，0.02/0.01/0.005分别耗散约75.57/37.79/18.89W每轮。若0.63A是24V母线电流，总输入只有15.12W；若将其全部分配给黏性机械损耗，则b上界约0.0040，尚未扣其他电损。若它是转矩电流且有效Kt=0.246，则对应输出约0.155N·m，与0.02在该转速的1.229N·m也不同。这些是条件性一致性检验，不能跨电流定义拟合硬件参数；0.005同样不自动等于真实标定值。',
      '## 固定轮速：先整机、后独立轮轴','',
      f'无PPO决策的整机固定轮速测试：目标±20、±36.36rad/s，四腿固定零位、原PD，无额外稳定力；{sum(r["success"] for r in full)}/12运行至5s。失败记录和接触/滑移保存在fixed_robot，不把短时倒地轨迹当作稳态电机测量。',
      '独立轮轴从真实编译模型复制两轮质量、惯量、COM、惯性坐标系、关节轴、armature、gear和限幅，并使用原2ms Euler积分；两根轴固定，移除浮动机身、腿、闭链、接触及重力。因此它检验速度环和已声明阻尼，不验证整机平衡或真实带载能力。isolated_plant.json保存质量矩阵和复制参数。',
      'B 的输出增益Kt=0.246，Kp=1.25，速度环力矩增益K=0.3075 N·m/(rad/s)。未饱和、无其他负载时，ω∞=K/(K+b)·ω_target，e∞=b/(K+b)·ω_target。b=0.02时误差比例6.1069%，目标20/36.36对应误差1.22137/2.22046rad/s；纯滚动等效线速度损失约0.0672/0.1221m/s。',
      table(primary[['controller','target_rad_s','steady_omega_rad_s','steady_error_rad_s','theory_difference_rad_s','steady_motor_torque_nm','steady_passive_torque_nm','overshoot_rad_s','steady_omega_std_rad_s']]),
      'P理论在全部阻尼和正反向目标下已验证；PI是20s有限时域结果，不冒充精确零误差。PI/FF均保持额定电流/力矩限制。独立轮轴有毫秒级起步超调，不能据此声称整机无超调；瞬态全部保存。',
      '## 整机逐条件结果','',
      '每行n为配对种子数；posture/tracking/combined是成功数，survived是24s整段存活数。局部任务区间与24s存活分别保留；缺失稳态不填零。失败越早的全段RMS不能单独用于优劣比较。',
      table(selected[['condition','task','n','posture','tracking','combined','survived','steady_vx','speed_rmse','response_s','pitch_rms','roll_rms','common_peak','common_duration','wheel_sat','leg_sat']]),
      '## 外环、内环与接触几何分解','',
      '定义Vt=−0.055·mean(ω_target)、Vw=−0.055·mean(ω_actual)，则恒等式：v_cmd−v_body=(v_cmd−Vt)+(Vt−Vw)+(Vw−v_body)。分别称PPO目标不足、速度环差额、接触/运动换算差额。Vt/Vw是等效纯滚动速度，不能当作实际机身速度。负向命令下这些差额通常为负，比较大小时保留方向。',
      table(steady[['condition','task','steady_mean_vx_m_s','target_equivalent_vx_m_s','policy_shortfall_m_s','speed_loop_shortfall_m_s','rolling_kinematic_shortfall_m_s','wheel_error_rms_rad_s','motor_abs_torque_nm','passive_abs_torque_nm','wheel_saturation_fraction']]),
      '进一步用接触点Jacobian分解：v_body=切向滑移X+法向速度X−轮关节自转X−载体角速度X−(轮心速度X−机身速度X)。自转与名义半径换算之差包含接触杠杆/轴姿态，载体与轮心项包含姿态和腿运动。按各轮法向力加权，再对两轮平均；无接触留NaN。分解列和恒等式残差在steady_state_decomposition.csv中，不能把整个第三项都叫打滑。',
      '## 曲线','',
      '![独立轮轴P/PI/前馈](isolated_speed_control.png)','![机身速度与姿态](body_velocity_posture.png)',
      '![稳态误差来源](steady_error_decomposition.png)','![换向控制与共同摆腿](reversal_control_comparison.png)',
      '![三类成功率](success_comparison.png)',
      '## 复现与审计','',
      'experiment_config.json为事前冻结协议；commands.sh为分阶段复现命令；expansion_gate.json记录扩展理由。isolated_metrics.csv、fixed_robot_summary.json、task_metrics.csv、episode_metrics.csv、steady_state_decomposition.csv、paired_results.csv和condition_summary.csv分别保存底层/整机/分解/配对结果。每回合NPZ记录125Hz状态、500Hz实际电机计算及接触快照。',
      'physics_checks.json检查实际qfrc_passive=−bω、电流/积分限幅、轮端实际力矩及接触运动学恒等式；integrity_verification.json核对旧模型、checkpoint、归一化文件与历史基线轨迹。未声称完成实机标定；0.01/0.005、Ki与前馈只作为独立诊断配置。']
    text='\n\n'.join(lines)+'\n'
    while '\n\n\n' in text:text=text.replace('\n\n\n','\n\n')
    (OUT/'report.md').write_text(text)


def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot-gate',action='store_true');args=p.parse_args()
    seeds=PILOT if args.pilot_gate or not (OUT/'confirmation_complete.json').exists() else SEEDS
    if args.pilot_gate and not (OUT/'pilot_complete.json').exists():raise RuntimeError('Pilot not complete')
    df,ss,ep,summary=collect(seeds);transparency(seeds)
    if args.pilot_gate:gate(df,ss);return
    plots(df,ss,summary,seeds);write_report(df,ss,ep,summary,seeds)
    from wheel_speed_findings import append_findings
    append_findings(OUT,df,ss,ep,summary,seeds,load,table)
    print('Analyzed',len(ep),'PPO episodes',len(df),'task intervals',flush=True)

if __name__=='__main__':main()
