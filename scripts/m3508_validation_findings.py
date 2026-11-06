"""Reproducible interpretation and additional steady-state diagnostics."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def append_findings(out,df,ep,agg,load,models,seeds,table):
    # Reconfirm model values from the actual compiled MuJoCo model. Never mutate.
    from uz05.model import UZ05Model,TerrainSpec
    sim=UZ05Model(TerrainSpec());m=sim.model;ids=sim.wheel_actuator_ids
    damping=m.dof_damping[sim.wheel_dof_adr].copy();radius=sim.wheel_radius
    static={'mass_kg':float(m.body_mass.sum()),'wheel_damping_nm_per_rad_s':damping.tolist(),
      'wheel_frictionloss_nm':m.dof_frictionloss[sim.wheel_dof_adr].tolist(),
      'wheel_actuator_gainprm':m.actuator_gainprm[ids].tolist(),'wheel_actuator_biasprm':m.actuator_biasprm[ids].tolist(),
      'wheel_actuator_dyntype':m.actuator_dyntype[ids].tolist(),
      'no_load_rad_s_from_587rpm':float(587*2*np.pi/60),
      'modeled_wheel_viscous_power_at_587rpm_w':float(damping[0]*(587*2*np.pi/60)**2)}
    (out/'static_model_audit.json').write_text(json.dumps(static,indent=2)+'\n')
    steady=[];peaks=[]
    for row in df[df.task.isin(['forward2','reverse2','high_reversal'])].to_dict('records'):
        c=load(row['model'],row['reference'],row['protocol'],row['eval_seed'])
        t=c['time_s'];end=24. if row['task']!='high_reversal' else 16.
        start=3. if row['task']!='high_reversal' else 11.
        mask=(t>end-2+1e-9)&(t<=end+1e-9)
        mc=c['motor_columns'];motor=c['motor']
        get=lambda prefix,n:motor[:,:,[mc.index(f'{prefix}_{i}') for i in range(n)]]
        wt=np.column_stack([c[f'wheel_target_{i}'] for i in range(2)])
        actual=wt[:,None,:]-get('wheel_speed_error_rad_s',2)
        torque=get('wheel_torque_nm',2)
        cap=motor[:,:,mc.index('effective_cap_nm')][:,:,None]
        if row.get('steady_complete',0)==1 and mask.any():
            steady.append({**{k:row[k] for k in ['model','reference','task','eval_seed']},
                'vx_m_s':float(c['vx_m_s'][mask].mean()),'wheel_target_rad_s':float(wt[mask].mean()),
                'wheel_actual_rad_s':float(actual[mask].mean()),
                'wheel_target_equivalent_vx_m_s':float(-radius*wt[mask].mean()),
                'wheel_error_rmse_rad_s':float(np.sqrt(np.mean(get('wheel_speed_error_rad_s',2)[mask]**2))),
                'wheel_control_current_abs_mean_a':float(np.abs(get('wheel_control_current_a',2)[mask]).mean()),
                'wheel_torque_abs_mean_nm':float(np.abs(torque[mask]).mean()),
                'wheel_viscous_torque_abs_mean_nm':float(np.abs(actual[mask]*damping).mean()),
                'mean_motor_minus_viscous_torque_nm':float((torque[mask]-actual[mask]*damping).mean()),
                'wheel_saturation_fraction':float((np.abs(torque[mask])>=.99*cap[mask]).mean()),
                'steady_pitch_std_deg':float(np.degrees(c['pitch_rad'][mask]).std()),
                'steady_roll_std_deg':float(np.degrees(c['roll_rad'][mask]).std()),
                'steady_vx_std_m_s':float(c['vx_m_s'][mask].std())})
        if row['task']=='high_reversal':
            for window_name,window_end in [('first_0p5s',11.5),('full_task',16.)]:
                early=(t>11+1e-9)&(t<=window_end+1e-9)
                if early.any():
                    i=np.flatnonzero(early)[np.argmax(np.abs(c['common_x_m'][early]))]
                    peaks.append({**{k:row[k] for k in ['model','reference','eval_seed']},
                        'window':window_name,'window_end_s':window_end,
                        'common_peak_m':float(c['common_x_m'][i]),'peak_delay_s':float(t[i]-11),
                        'pitch_at_peak_deg':float(np.degrees(c['pitch_rad'][i])),
                        'wheel_saturated_at_peak':int((np.abs(torque[i])>=.99*cap[i]).any()),
                        'actual_vx_at_peak_m_s':float(c['vx_m_s'][i])})
    steady=pd.DataFrame(steady);steady.to_csv(out/'steady_motor_detail.csv',index=False)
    peaks=pd.DataFrame(peaks);peaks.to_csv(out/'common_swing_events.csv',index=False)
    steady_summary=steady[steady.reference=='step'].groupby(['model','task'],sort=False).agg(
        vx=('vx_m_s','mean'),target_v_equivalent=('wheel_target_equivalent_vx_m_s','mean'),
        wheel_error=('wheel_error_rmse_rad_s','mean'),drive_nm=('wheel_torque_abs_mean_nm','mean'),
        viscous_nm=('wheel_viscous_torque_abs_mean_nm','mean'),wheel_sat=('wheel_saturation_fraction','mean'),
        pitch_std=('steady_pitch_std_deg','mean'),vx_std=('steady_vx_std_m_s','mean')).reset_index()
    steady_summary.to_csv(out/'steady_motor_summary.csv',index=False)
    # Additional C-vs-B contrast: same rated baseline, only transient boost differs.
    paired=[]
    a=df[(df.model=='B_rated')&(df.reference=='step')].set_index(['task','eval_seed'])
    for row in df[(df.model=='C_burst')&(df.reference=='step')].to_dict('records'):
        base=a.loc[(row['task'],row['eval_seed'])]
        paired.append({**{k:row[k] for k in ['task','eval_seed']},
          **{key+'_C_minus_B':row.get(key,np.nan)-base[key] for key in ['posture_success','tracking_success',
            'composite_success','pitch_rms_deg','common_x_peak_m','early_wheel_saturation_fraction','steady_rmse_m_s']}})
    pd.DataFrame(paired).to_csv(out/'paired_burst_results.csv',index=False)
    # Transparent plot of tolerance vs the actual steady means of each seed.
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,task,goal in zip(axes,['forward2','reverse2'],[2.,-2.]):
        ax.axhspan(goal-.1,goal+.1,color='green',alpha=.15,label='5% band')
        ax.axhline(goal,color='k',ls='--')
        for j,name in enumerate(models):
            values=df[(df.model==name)&(df.reference=='step')&(df.task==task)].steady_mean_vx_m_s
            ax.plot(j+np.linspace(-.07,.07,len(values)),values,'.',label=name)
        ax.set_xticks(range(4),list(models),rotation=15);ax.set_title(task+' steady mean (m/s)');ax.grid(alpha=.2)
    fig.savefig(out/'steady_speed_band.png',dpi=170);plt.close(fig)
    high=agg[agg.task=='high_reversal']
    source=agg[(agg.reference=='step')&(agg.task.isin(['stand','forward2','reverse2','moderate_reversal']))]
    early=df[(df.reference=='step')&(df.task=='high_reversal')].groupby('model',sort=False).agg(
       complete=('early_complete','sum'),target_peak=('early_joint_target_step_max_rad','mean'),
       leg_sat=('early_leg_saturation_fraction','mean'),wheel_sat=('early_wheel_saturation_fraction','mean'),
       common_peak=('early_common_x_peak_m','mean')).reset_index()
    early.to_csv(out/'matched_reversal_window.csv',index=False)
    # Supplement the same already-fixed 3-8s reverse-launch criterion using
    # the stop protocol's identical prefix, including shaped-reference cases.
    from uz05.m3508_metrics import score_interval
    from run_m3508_validation import THRESHOLDS
    directional=[]
    for episode in ep[ep.protocol=='stop2_reverse'].to_dict('records'):
        c=load(episode['model'],episode['reference'],'stop2_reverse',episode['eval_seed'])
        directional.append({**{k:episode[k] for k in ['model','reference','eval_seed']},
            **score_interval(c,3.,8.,-2.,THRESHOLDS,episode['termination_reason'])})
    directional=pd.DataFrame(directional);directional.to_csv(out/'reverse_launch_reference_metrics.csv',index=False)
    directional_summary=directional.groupby(['model','reference'],sort=False).agg(
        posture=('posture_success','sum'),tracking=('tracking_success','sum'),combined=('composite_success','sum'),
        steady_vx=('steady_mean_vx_m_s','mean'),pitch_rms=('pitch_rms_deg','mean'),common_peak=('common_x_peak_m','mean')).reset_index()
    directional_summary.to_csv(out/'reverse_launch_reference_summary.csv',index=False)
    text='''
## 对十二个问题的回答

1. **原始 1.722 N·m 的来源明确。** 软件速度环最多给出 10 A 控制变量，乘输出侧额定点比例 0.246，再乘 0.70，得到 1.722。原 ±2.46 N·m 裁剪未先触发。所有模型的 ctrl→actuator_force→qfrc_actuator 换算均已用实际 MuJoCo 模型核验，最大误差约 1.10e−11 N·m。

2. **在当前“输出侧有效比例”的变量语义下存在重复折损；硬件标定仍未完成。** 0.246 正是 2.46/10，不是独立实测的绕组电流转矩常数。最大效率不是全工况恒定系数；而产品表的电流测点未知，所以不能直接宣布“真实轮端应永远等于 current×0.246”。B 取消该额外折减仅作诊断，不覆盖 A。

3. **2.46 N·m 可作额定工作点的条件性近似。** 前提是输入当前控制变量与额定电流口径兼容。它提供有用的仿真能力基准，但未实现反电动势、全速度—力矩包络或温升，不能证明额定高速连续能力。B 在本次固定策略/种子下显著改善了高速阶跃姿态恢复，支持优先核实这一动力学能力差异。

4. **3 N·m 作为显式限时假设。** C 仅在请求 vx 改变后的 0.5 s 把等效输出比例改为 0.300 N·m/control-A，之后退回 B；速度环与 10 A 控制变量上限不变。3 N·m 按 0.246 反推约 12.195 A，仅为归一化能力说明。0.5 s 和更高输出增益没有厂家热数据支持，不能作为实机峰值许可。实际评估中 C 高速区间的单轮连续超过 2.46 N·m 最长为 0.196 s。D 是更乐观的堵转上界仿真，并非连续模型。

5. **原始 v16 的新旧表现已分开比较。** A 的高速阶跃 0/5 存活/姿态成功；B/C/D 均 5/5 存活和姿态成功，普通站立、前进、后退与停车均保持 5/5 存活。但所有模型在 ±2 稳态与中等换向任务上均未通过新的 5% 跟踪要求：稳定不等于准确。站立与后退停车的零速跟踪均为 5/5。

6. **提高力矩改善了高速换向的姿态恢复，没有完成综合任务。** B/C/D 的高速跟踪和综合成功均为 0/5。这里的 B 改变了整个轮端电流—力矩映射，既增加有效增益也增加上限，不能把收益只归因于一个截断阈值。B 稳态后退约 −1.719 m/s，仍离 −2 m/s 有约 0.281 m/s 的偏差。C 比 B 的能力提升没有产生新的成功率收益，D 也没有把稳态误差带入容差范围。不能据此选用最大的力矩。

7. **未观察到新的整体跌倒或严重持续振荡，但存在小幅代价。** B/C/D 的基础六协议均 5/5 存活。中等换向区间 Pitch RMS 从 A 的约 1.195° 升至 B/C/D 的 1.474°/1.546°/1.707°，共同偏移峰值也略增；这些区间 RMS 含瞬态，不能全解释成稳态振荡。稳态 Pitch/速度标准差与饱和另列，不能只报告有利指标。n=5 同一策略的配对诊断不足以证明跨条件无退化。

8. **目前不能在 +2 指令下稳定达到 1.90～1.95 m/s。** A/B/C/D 的前进稳态均值分别约 1.809/1.813/1.813/1.816 m/s；后退约 −1.714/−1.719/−1.719/−1.724 m/s。没有达到允许的 5% 误差带。提高上限没有显著改善稳态速度，而且这些稳态窗口的轮饱和为零，所以不能把持续低速简单归因为最大力矩不足。

9. **当前没有一种参考同时满足新姿态与速度跟踪目标。** 在 B 内，测试的所有参考都通过高速姿态标准，但跟踪均 0/5。更缓参考显著减小瞬态共同摆腿与关节目标步幅；1.5–2 m/s² 可作为稳定优先的后续候选参考，而不是已验证的完整解决方案。A 的 0.5 s 斜坡虽 5/5 存活，却仅 1/5 通过姿态标准：另四回合 |Pitch|>15° 持续 0.256–0.312 s，超过事前固定的 0.25 s。更保守参考没有解决固定策略的稳态偏差。

10. **力矩和参考对共同摆腿的作用不同。** 高速阶跃的共同偏移峰值均值 A≈22.69 cm、B≈23.44 cm、C≈23.54 cm、D≈23.32 cm：提高驱动能力没有消除大幅共同前摆。固定 B，2 s 斜坡/2 m/s² 参考降至约 6.93 cm，1.5 m/s² 降至约 5.03 cm。它与瞬态轮饱和、Pitch 变化同时出现，但相关性不证明必要性。B 的首周期目标变化仍约 1.21 rad，却能恢复，进一步说明单个目标大步并非唯一充分解释。是否为必需的平衡补偿，还需要受力分析或独立反事实；不应惩罚所有摆腿。

11. **当前没有充分依据先修改奖励。** 先核实力矩/电流和被动损耗，再区分低层速度 P 环误差与既有策略的稳态响应。新的 5% 标准明确揭示当前偏差不可接受，但单靠固定权重的动力学/参考实验不能证明哪个奖励项造成偏差。下一次若短训仍在已校准模型上稳定低估速度，再独立评估跟踪目标或权重，不能同时堆叠 Pitch、对称与摆腿惩罚。

12. **优先动力学/损耗标定，其次参考与课程，动作接口后置。** 保留 A 作为回归基线、B 作为待标定的额定点候选；不把 C/D 作为正式连续模型。首先查清 10 A 口径和原轮阻尼，再选定与部署一致的参考。未来首个短程 PPO 消融只改变已确认的动力学模型，沿用现有动作/奖励与相同续训状态；参考生成器、逐级课程分别再做独立消融，持续回归站立、后退起步、停车及阶跃压力任务。只有这些对照仍显示绝对周期目标接口的独立问题，才研究增量/残差/速度前馈。本轮不执行任何 PPO 训练，也不发布新的正式训练配置。

## 稳态误差：轮阻尼、P 环与策略闭环

'''
    text+='实际模型审计确认轮关节 `dof_damping=0.02 N·m/(rad/s)`。在约 31–33 rad/s 的稳态轮速下，黏性阻力矩约 0.63–0.66 N·m，几乎等于记录的驱动力矩；下表给出 500 Hz 电机输入/输出的稳态均值。该阻尼保持不变，它是另一个必须与实物核实的损耗来源，不属于本轮力矩开关。\n\n'+table(steady_summary)+'\n\n'
    text+='速度环 Ki=0，维持非零负载必须保留转速误差。B 比 A 的轮速误差从约 3.10 降至 2.19 rad/s，但前进实际速度仅从 1.809 升至 1.813 m/s，同时策略轮速目标从约 −36.10 回撤至 −35.25 rad/s（纯滚动等效 1.985→1.939 m/s）。D 的速度环误差进一步下降，策略目标也继续变化。固定 PPO 是有状态反馈的外环，不是固定轮速给定器；这些数据说明只抬高驱动上限无法代替稳态闭环问题的检查，不能由此证明策略“故意忽略命令”。\n\n'
    text+=f'当前 0.02 阻尼在 587 rpm 处对应每轮约 {static["modeled_wheel_viscous_power_at_587rpm_w"]:.2f} W 的机械耗散。旧 spec 还写有 0.63 A 空载电流，其测点同样未核实；若它是 24 V 母线输入，则总输入仅 15.12 W，与该仿真耗散不能同时作为同一空载工作点。这里是条件性的一致性疑点，不足以直接把阻尼改成另一个数字。本轮未改变阻尼、效率默认值、速度环或奖励。\n\n'
    text+='![实际稳态速度与5%容差](steady_speed_band.png)\n\n'
    text+='## 共同摆腿的配对事件记录\n\n'
    text+='下表统一使用换向后前 0.2 s，避免 A 失败更早造成全段指标偏小。target_peak 为周期角差最大值均值，饱和为电机子步比例。完整 signed common peak、事件时刻、Pitch 及峰值时是否有轮饱和见 common_swing_events.csv；瞬态最大目标步幅仍远大于额定速度在一个 8 ms 周期内可走的角度。它是有限响应执行器的阶跃目标，不是周期包装误差。\n\n'+table(early)+'\n\n'
    text+='不能把所有前后运动都视为异常：在 B 下，阶跃后大幅共同前摆仍存在但机身恢复了，而缓参考同时减小了这种补偿需求。若要评价持续异常，应同时看 common_outside_008_longest_s、common_steady_rms_m、支撑/姿态与跟踪，不能单看瞬时峰值。相关和重合统计只提供机制线索。\n\n'
    text+='## 加速方向与制动的补充对照\n\n'
    text+='将事前固定的后退起步 3–8 s 标准应用于 stop2_reverse 协议的同一起步片段，补充评价反向加速参考；不修改主表350个区间、阈值或既有成功计数。完整结果见 reverse_launch_reference_metrics.csv，均值汇总如下。前进参考见主表forward2，制动见reverse_stop，高速换向见high_reversal。\n\n'+table(directional_summary)+'\n\n'
    text+='前进与后退的稳态偏差不同，当前参考实验没有证明应采用不同的硬件加速度上限。它同时受到策略与速度环/阻尼影响，不能直接归因为电机正反向能力不对称。较缓参考减小两方向的瞬态补偿，但没有消除任一方向的5%稳态跟踪失败；应先完成双方向损耗与输出标定，再决定是否需要方向相关参考。\n\n'
    text+='## 下一轮的执行边界\n\n'
    text+='本轮已完成仿真审计、100 点实际力矩映射检查、330 配对回合及参考对照，尚未完成实机电流/轮端扭矩/损耗标定。135 项自动测试通过；旧模型的 10 条阶跃与 10 条斜坡轨迹逐元素复现此前记录，210 条参考试验在介入前与各自固定力矩基线一致。所有旧 XML、Reset、正式执行器/奖励/训练文件、checkpoint 和归一化文件保持原哈希。\n\n'
    text+='C 的短时结果、D 的上界结果以及 B 的高速存活都不是实机连续能力证明。没有任何非零速度方案通过本轮全部候选跟踪标准，因而现在不建议直接完整 PPO 重训或以新力矩上限替换旧默认。下一阶段应先取得供应商电流定义和同步硬件数据，核对电流—输出力矩及空载/负载损耗，再进行单变量短程适应训练；正式训练命令待这些模型选择有依据后另行确定。\n'
    p=out/'report.md';original=p.read_text()
    intro=('**主要结论：2.46 N·m 额定点近似让原始 v16 的高速阶跃姿态成功从 0/5 提升至 5/5，但跟踪/综合成功仍为 0/5。+2 m/s 时实际稳态约 1.81 m/s，−2 m/s 时约 −1.72 m/s，均未达到允许的 5% 误差。较缓参考显著减少共同摆腿，但不能消除稳态偏差。优先核实电流与轮阻尼/速度环损耗，保留旧模型，不立即完整重训。硬件模型尚未完成实测标定。**')
    original=original.replace('# M3508 力矩模型与稳定优先控制：配对诊断报告','# M3508 力矩模型与稳定优先控制：配对诊断报告\n\n'+intro,1)
    result=original+'\n\n'+text
    while '\n\n\n' in result:result=result.replace('\n\n\n','\n\n')
    p.write_text(result)
