"""Additional event measures and interpretation, without changing criteria."""
import json
import numpy as np
import pandas as pd


def append_findings(out,df,ss,ep,summary,seeds,load,table):
    steady=ss[ss.task.isin(['forward2','reverse2'])].groupby(['condition','task'],sort=False).mean(numeric_only=True).reset_index()
    event_rows=[]
    for e in ep[ep.protocol.isin(['motion','stop2_reverse'])].to_dict('records'):
        c=load(e['condition'],e['protocol'],e['eval_seed']);t=c['time_s']
        start,end=(11.,16.) if e['protocol']=='motion' else (10.,24.)
        mask=(t>start+1e-9)&(t<=end+1e-9);early=(t>start+1e-9)&(t<=start+.5+1e-9)
        if not mask.any():continue
        mc=c['motor_columns'];m=c['motor'];get=lambda prefix:m[:,:,[mc.index(f'{prefix}_{i}') for i in range(2)]]
        i=get('integral_current');p=get('p_current');before=np.flatnonzero(t<=start+1e-9)
        when=lambda test:float(t[np.flatnonzero(mask&test)[0]]-start) if (mask&test).any() else np.nan
        event_rows.append({'condition':e['condition'],'protocol':e['protocol'],'eval_seed':e['eval_seed'],
          'initial_integral_current_a':float(i[before[-1],-1].mean()) if len(before) else np.nan,
          'first_05_integral_opposes_p_fraction':float((i[early]*p[early]<-1e-9).mean()) if early.any() else np.nan,
          'posture_pitch_peak_deg':float(np.degrees(np.abs(c['pitch_rad'][mask])).max()),
          'forward_rebound_after_stop_m_s':float(max(0.,c['vx_m_s'][mask].max())) if e['protocol']=='stop2_reverse' else np.nan,
          'body_first_negative_delay_s':when(c['vx_m_s']<0) if e['protocol']=='motion' else np.nan,
          'negative_90percent_command_delay_s':when(c['vx_m_s']<=-1.8) if e['protocol']=='motion' else np.nan,
          'survival_s':e['survival_s'],'termination_reason':e['termination_reason']})
    events=pd.DataFrame(event_rows);events.to_csv(out/'controller_transients.csv',index=False)
    event_summary=events.groupby(['condition','protocol'],sort=False).mean(numeric_only=True).reset_index()
    event_summary.to_csv(out/'controller_transient_summary.csv',index=False)
    # Fixed-robot labels are reconstructed from named trace folders, not list order.
    fixed=[]
    for target in [20.,-20.,36.36,-36.36]:
        rows=json.loads((out/'fixed_robot'/str(target)/'episodes.json').read_text())
        for r in rows:fixed.append({'target_rad_s':target,'eval_seed':r['eval_seed'],'survival_s':r['survival_s'],'termination_reason':r['termination_reason'],'steady_window_available':False})
    fixed=pd.DataFrame(fixed);fixed.to_csv(out/'fixed_robot_metrics.csv',index=False)
    # Same frozen reverse-launch window additionally applied to stop protocol.
    from uz05.m3508_metrics import score_interval
    from run_wheel_speed_steady_state import THRESHOLDS
    rear=[]
    for e in ep[ep.protocol=='stop2_reverse'].to_dict('records'):
        c=load(e['condition'],e['protocol'],e['eval_seed'])
        rear.append({'condition':e['condition'],'eval_seed':e['eval_seed'],**score_interval(c,3,8,-2.,THRESHOLDS,e['termination_reason'])})
    pd.DataFrame(rear).to_csv(out/'reverse_launch_reference_metrics.csv',index=False)
    # No posthoc change to the main success classification or expansion gate.
    high=summary[summary.task=='high_reversal']
    basef=steady[(steady.condition=='base')&(steady.task=='forward2')].iloc[0]
    baser=steady[(steady.condition=='base')&(steady.task=='reverse2')].iloc[0]
    ff=steady[(steady.condition=='P_FF')&(steady.task=='forward2')].iloc[0]
    ratios={'forward_base':{'PPO_target_fraction':float(basef.policy_shortfall_m_s/(2-basef.steady_mean_vx_m_s)),
                           'inner_loop_fraction':float(basef.speed_loop_shortfall_m_s/(2-basef.steady_mean_vx_m_s)),
                           'kinematics_fraction':float(basef.rolling_kinematic_shortfall_m_s/(2-basef.steady_mean_vx_m_s))},
            'reverse_base':{'PPO_target_fraction':float(baser.policy_shortfall_m_s/(-2-baser.steady_mean_vx_m_s)),
                           'inner_loop_fraction':float(baser.speed_loop_shortfall_m_s/(-2-baser.steady_mean_vx_m_s)),
                           'kinematics_fraction':float(baser.rolling_kinematic_shortfall_m_s/(-2-baser.steady_mean_vx_m_s))}}
    (out/'baseline_error_fractions.json').write_text(json.dumps(ratios,indent=2)+'\n')
    text='''
## 对十个问题的回答

1. **0.02 存在物理一致性疑点，但不能认定错误。** 它来自实际训练XML的显式设置，名义编译值无误；仓库历史可追到引入该模型的 cc563dc，但没有发现厂家阻尼标定来源。所有14个铰链都采用同量级阻尼，不足以证明轮轴真实损耗就是这个数。按输出关节坐标，它在587rpm需要约1.229N·m/75.57W；与空载0.63A的两种电流口径都存在条件性不一致。0.01与0.005只是敏感性点，不是校准答案。A的效率折减、输出侧有效Kt和额外阻尼是否重复表达同一损耗，需要厂家/测量定义，现阶段无法唯一判断。

2. **纯P速度环的稳态误差已被独立验证。** B基线的无其他负载固定轮速误差为6.1069%，20与36.36rad/s分别少1.22137与2.22046rad/s，反向同样成立；全部B纯P试验与理论稳态值的最大差约6.2e−12rad/s。整机B基线在恒定±2命令下，低层平均差额折算约正向0.1184m/s、反向−0.1124m/s。驱动输出约0.63–0.66N·m，主要抵消同坐标下的被动阻尼，稳态电流/力矩未触限。

3. **降低阻尼明显改善轮速环误差，机身跟踪却只改善很少。** b=0.01/0.005时，纯P无负载误差比例降到3.1496%/1.6000%；整机前进稳态约1.818/1.820m/s、后退约−1.726/−1.729m/s，均未达5%。两组保留基础任务及高速换向姿态成功，但后退稳态姿态偏置增加，换向Pitch RMS也略升，不能把降低阻尼当作无代价的真实修正。

4. **小积分能减少低层偏差，但不是机身跟踪修复。** 固定轮轴PI在20s末的最大残余误差约0.00181rad/s，未突破10A/2.46N·m限制；整机恒速窗口的平均轮速误差接近零，但因125Hz策略目标持续变化，500Hz子步RMS仍约0.39–0.43rad/s。机身仅约+1.823/−1.733m/s。Reset清零、条件抗饱和和±4A积分电流限制均已验证。

5. **经模拟模型核实的阻尼前馈同样消除了低层平均误差。** 实际qfrc_passive与−bω逐周期吻合；I_ff=bω_target/0.246补偿这个明确的模拟负载，先与P项求和再做共享10A限幅。独立轮轴误差接近数值精度；整机仍约+1.823/−1.733m/s。这里核实的是MuJoCo损耗，不是实机阻尼或Kt标定，不能直接把该前馈系数部署到硬件。

6. **PPO会改变输出目标，且抵消了大部分低层改进。** B基线正/反向的等效轮速目标约+1.9386/−1.8394m/s；PI/前馈下约+1.8296…1.8298/−1.7413…−1.7415m/s。虽然轮速更准确地跟踪其目标，外环目标本身更低。固定权重仍根据车身、轮速、姿态和上一动作反馈决策，不是固定给定器。数据证明了新的闭环平衡点和目标不足，不能进一步断言策略有“主动忽略命令”的意图，也不能仅凭这些输出定位是哪一个奖励项或网络近似误差造成的。

7. **+2/−2的5%要求仍未满足。** 所有非零恒速方案的跟踪和综合成功均为0/5；姿态成功与24s存活已分别报告。最好约+1.823/−1.733m/s，仍未到用户接受的±1.90…1.95m/s区间。正向/反向误差并非主要来自持续滑移：基线接触/运动换算差额只有约+0.00694/−0.00783m/s，其中纵向带符号滑移的均值贡献仅约0.00025/−0.0004m/s，主要余项为接触点杠杆/轴姿态投影。不能把整个换算差额叫轮胎打滑，也不能据此宣称瞬态没有滑移。

8. **没有新增倒地，但PI有明显换向姿态代价。** 除历史A模型高速阶跃仍0/5外，B基线及阻尼/PI/前馈候选均保留5/5高速姿态成功。PI的换向Pitch RMS约4.67°，高于B的2.70°；前馈约2.80°。积分在换向时保留原方向负载偏置，P项反向后可能与其对抗，记录见controller_transients.csv，这种正常状态记忆与windup不同。独立轮轴也存在毫秒级超调；各方案的停车回弹、峰值和末窗波动另列，不能声称增加积分或前馈完全没有瞬态代价。1.5m/s²参考仍最显著减小共同摆腿，但本轮没有把它与低阻尼/PI/前馈组合。

9. **暂不额外修改速度奖励。** 当前已经量化底层偏差，也证实减小它不能让固定外环自动达到目标；PPO需要学习新闭环下的稳态目标补偿和换向协调。这个结论不是“奖励已经正确”的证明，也不是必须立刻加权的证明。先在固定物理假设、动作接口和现有奖励下做策略适应对照；若短训仍稳定低估目标，再独立分析速度奖励梯度与其他代价的取舍，避免同时提高Pitch或增加多项惩罚。

10. **具备有限的v17仿真单变量短训依据，尚不具备实机标定完成的依据。** 不再需要假设“轮电机一直饱和”才能解释问题：在已声明的B模型下，轮速/力矩余量足够，剩余偏差主要在外环输出。首个v17候选应优先只做A→B轮端映射适应，原P环、0.02阻尼、动作/奖励、采样课程和参考保持不变，使用同一v16状态的配对短训；不要一次加入低阻尼、PI和参考平滑。B仍是额定点假设，硬件用途必须先核实电流与损耗。若以后比较P与前馈，两臂必须固定同一阻尼与同一参考，作为另一项独立消融。本轮没有启动任何PPO训练。

## 稳态误差的主要来源与几何核对

'''
    text+=f'在B基线正向差额中，内环约占{ratios["forward_base"]["inner_loop_fraction"]:.1%}、外环约{ratios["forward_base"]["PPO_target_fraction"]:.1%}、其余换算约{ratios["forward_base"]["kinematics_fraction"]:.1%}；反向则外环约{ratios["reverse_base"]["PPO_target_fraction"]:.1%}、内环约{ratios["reverse_base"]["inner_loop_fraction"]:.1%}。因此不能给正反向套用同一个单一来源结论。\n\n'
    text+='修正低层后，外环差额增大成为主要剩余项；这不是证明低层改进无效，而是说明需要同时观察固定策略如何调整给定。本轮实际变化可以在下表直接核对，数值为五个配对种子的名义末2s均值。forward2/reverse2保持命令21s，比5s的高速任务区间更适合确定稳态；较缓换向参考的最后2s仍可能包含收敛过程。\n\n'
    text+=table(steady[['condition','task','steady_mean_vx_m_s','target_equivalent_vx_m_s','wheel_error_mean_rad_s','wheel_error_rms_rad_s','motor_abs_torque_nm','passive_abs_torque_nm','viscous_power_per_wheel_w','contact_signed_slip_m_s','contact_spin_geometry_m_s']])+'\n\n'
    text+='接触重构的最大残差低于1e−12m/s，支持单位、坐标系和采样时刻一致。表中的“自转几何项”来自真实接触点杠杆与轮轴/车身方向，不是推断真实轮半径已改变；轮心和载体项区分了运动换算与切向滑移，没有通过改变半径消除速度偏差。\n\n'
    text+='## 换向与停车瞬态\n\n'+table(event_summary[['condition','protocol','initial_integral_current_a','first_05_integral_opposes_p_fraction','posture_pitch_peak_deg','forward_rebound_after_stop_m_s','body_first_negative_delay_s','negative_90percent_command_delay_s']])+'\n\n'
    text+='未达到负向90%速度或完整5%稳定窗口的响应时间保留缺失，不填成零。表中“初始积分”是事件前最后一个电机子步的两轮均值；对抗比例只统计事件后0.5s的电机子步。相关的姿态代价需要与整个闭环变化一起解释，不能单凭同向/反向统计认定倒地因果。\n\n'
    text+='## v17应学习什么，以及短训边界\n\n'
    text+='在无额外负载、纯滚动估算下，要用B纯P环使轮速达到2/0.055≈36.364rad/s，目标需约38.73rad/s；稳态阻尼力矩约0.727N·m、电流约2.96A，均在当前48rad/s目标范围及10A/2.46N·m限制内。这只是稳态能力估算，不证明整机所有姿态/瞬态都可行，但说明无需先扩大动作空间或突破力矩限制来补偿已量化的静差。\n\n'
    text+='建议下一轮作为明确假设的仿真短训：三组配对训练种子、同一v16 Actor/Critic/优化器/归一化/调度状态，A与B仅切换轮端映射；在较短里程碑评估外环目标是否接近所需补偿、5%跟踪是否改善，并持续回归站立、后退起步、停车与换向。既有奖励先保持，不能把短训和奖励修改混在一起。若偏差不改善，先查策略目标及代价取舍，不盲目增加更新数。\n\n'
    text+='1.5m/s²继续作为独立候选参考：其共同摆腿优势已复现，但尚未验证与PI/前馈组合的结果；不直接组合进v17。诊断控制器没有被正式训练代码导入，未来任何训练开关都需单独定义并保存其控制契约和域随机化语义。本轮只交付诊断与方案，不生成或执行训练命令。\n\n'
    text+='## 验证与可复现性\n\n'
    text+='先完成117个小规模PPO回合，四项干预均满足事前扩展门槛，再统一增加两个种子，总计195个唯一回合；没有只扩展成功组，也没有更换阈值。40个独立轮轴案例全部保留，12个固定整机失败也保留。旧A/B及候选参考共75条轨迹与上一轮逐元素一致，源XML/Reset/正式控制与奖励/原checkpoint和归一化保持原哈希。\n\n'
    text+='本轮针对性测试不调用PPO.learn：检查P数值复现、积分抗饱和/限幅/Reset、前馈共享电流限幅，以及真实模型的被动力矩和接触运动学恒等式。tests.log保存测试输出；physics_checks.json保存全部实际试验核对。源文件哈希及完整命令见integrity_verification.json和commands.sh。产品参数和阻尼仍待实机口径校准，仿真结果不替代该标定。\n'
    report=out/'report.md';original=report.read_text()
    intro=('**结论：纯P环与0.02阻尼确实造成约6.1%的无负载轮速静差；但PI/前馈几乎消除平均轮速偏差后，机身仍约+1.823/−1.733m/s，未达到5%要求。PPO会把等效目标回撤到约+1.830/−1.741m/s，剩余偏差主要来自外环给定，稳态滑移不是主因。0.02有条件性的物理一致性疑点，尚不能判错或把0.005当作真实标定。未训练PPO、未改变正式配置。**')
    original=original.replace('# UZ05 轮毂电机稳态速度误差诊断','# UZ05 轮毂电机稳态速度误差诊断\n\n'+intro,1)
    original+='\n\n'+text
    while '\n\n\n' in original:original=original.replace('\n\n\n','\n\n')
    report.write_text(original)
