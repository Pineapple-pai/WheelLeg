"""Post-experiment interpretation; does not change frozen evaluation thresholds."""
import numpy as np
import pandas as pd


def append_findings(out, df, ev, load, policies, seeds, table):
    baseline=ev[(ev.arm=='step')&(ev.task=='motion')]
    relevant=ev[ev.task=='motion']
    comparison=relevant.groupby('arm',sort=False).agg(
        completed_early_windows=('early_complete','sum'),survival=('post_event_survival_s','mean'),
        target_max=('early_target_step_max_rad','mean'),target_p95=('early_target_joint_step_p95_rad','mean'),
        target_rate=('early_target_rate_rms_rad_s','mean'),actual_qd_p95=('early_actual_joint_speed_p95_rad_s','mean'),
        joint_error=('early_joint_tracking_rms_rad','mean'),leg_saturation=('early_leg_saturation_fraction','mean'),
        wheel_error=('early_wheel_tracking_rms_rad_s','mean'),wheel_saturation=('early_wheel_saturation_fraction','mean'),
        common_peak=('early_common_x_peak_m','mean'),pitch_rms=('early_pitch_rms_deg','mean'),
        intervention=('early_limiter_fraction','mean')).reset_index()
    comparison.to_csv(out/'intervention_event_summary.csv',index=False)
    contact=baseline.groupby('policy',sort=False).agg(
        normal_force=('early_normal_force_mean_n','mean'),no_contact=('early_wheel_no_contact_fraction','mean'),
        friction_p95=('early_friction_utilization_p95','mean'),slip_p95=('early_contact_slip_p95_m_s','mean'),
        leg_demand=('early_leg_demand_peak_nm','mean'),common_peak=('early_common_x_peak_m','mean')).reset_index()
    contact.to_csv(out/'contact_summary.csv',index=False)
    # Describe actual tracking separately from survival. Limit to the translation
    # interval, before yaw command at 16s; do not extrapolate failed trajectories.
    tracking=[]; first_responses=[]
    for policy in policies:
      for arm in ('step','ramp_020','ramp_050','ramp_100'):
       for seed in seeds:
        c=load(policy,arm,'motion',seed);t=c['time_s'];mask=(t>11)&(t<=16+1e-9)
        if arm=='step':
            i=np.flatnonzero(t>11+1e-9)[0];pre=i-1
            motor=c['motor'][i,0];names=c['motor_columns']
            early=(t>11)&(t<=11.2+1e-9)
            peak=np.flatnonzero(early)[np.argmax(np.abs(c['common_x_m'][early]))]
            event=baseline[(baseline.policy==policy)&(baseline.eval_seed==seed)].iloc[0]
            first_responses.append({'policy':policy,'eval_seed':seed,
                'pre_wheel_target_rad_s':np.mean([c[f'wheel_target_{j}'][pre] for j in range(2)]),
                'first_wheel_target_rad_s':np.mean([c[f'wheel_target_{j}'][i] for j in range(2)]),
                'first_end_wheel_actual_rad_s':np.mean([c[f'wheel_actual_{j}'][i] for j in range(2)]),
                'first_substep_wheel_torque_nm':np.mean([motor[names.index(f'wheel_torque_nm_{j}')] for j in range(2)]),
                'first_positive_target_delay_s':event['first_positive_wheel_target_time_s']-11,
                'signed_common_peak_m':c['common_x_m'][peak],
                **{f'first_signed_joint_step_{j}':float(np.arctan2(np.sin(c[f'target_q_{j}'][i]-c[f'target_q_{j}'][pre]),np.cos(c[f'target_q_{j}'][i]-c[f'target_q_{j}'][pre]))) for j in range(4)}})
        when=lambda condition:float(t[np.flatnonzero(mask&condition)[0]]-11) if (mask&condition).any() else np.nan
        tracking.append({'policy':policy,'arm':arm,'eval_seed':seed,
                         'negative_vx_delay_s':when(c['vx_m_s']<0),
                         'vx_below_minus_1p8_delay_s':when(c['vx_m_s']<=-1.8),
                         'window_exposure_s':float(mask.sum()*.008),
                         'window_vx_rmse_m_s':float(np.sqrt(np.mean((c['vx_m_s'][mask]-c['command_vx_m_s'][mask])**2)))})
    first_responses=pd.DataFrame(first_responses);first_responses.to_csv(out/'first_response.csv',index=False)
    first_summary=first_responses.groupby('policy',sort=False)[['pre_wheel_target_rad_s','first_wheel_target_rad_s','first_end_wheel_actual_rad_s','first_substep_wheel_torque_nm','first_positive_target_delay_s','signed_common_peak_m']].mean().reset_index()
    first_summary.to_csv(out/'first_response_summary.csv',index=False)
    tracking=pd.DataFrame(tracking);tracking.to_csv(out/'reversal_speed_tracking.csv',index=False)
    tracking_summary=tracking.groupby('arm',sort=False).agg(reached_negative=('negative_vx_delay_s','count'),
        negative_delay=('negative_vx_delay_s','mean'),reached_minus_1p8=('vx_below_minus_1p8_delay_s','count'),
        minus_1p8_delay=('vx_below_minus_1p8_delay_s','mean'),translation_exposure=('window_exposure_s','mean'),
        vx_rmse=('window_vx_rmse_m_s','mean')).reset_index()
    tracking_summary.to_csv(out/'reversal_speed_summary.csv',index=False)
    success=df.groupby(['task','arm']).success.sum()
    units=('表中 target_max/target_p95/joint_error 为 rad，target_rate/actual_qd_p95 为 rad/s；'
           'target_p95 是逐关节变化量 P95。饱和与介入为比例，common_peak 为 m，pitch_rms 为度。'
           '这些是相同前 0.2 s 窗口的各回合统计再平均；后续存活是单独指标。')
    text='''
## 最终回答与下一步

本次结果支持优先处理速度参考与闭环控制的可实现性。**不支持直接给现有策略加关节限速，也不支持立即增加奖励或开始完整重训。**

1. **高速失败普遍伴随大步关节目标变化。** 原始 v16 与六个续训策略的 35/35 基线回合，在 11.008 s 都出现全局最大目标变化；各策略五回合峰值均值 1.215–1.562 rad，最坏 1.568663 rad，即 196.08 rad/s 的单周期参考变化率。实际关节速度 P95 仅约 12.73–17.58 rad/s，换向后 0.2 s 的周期跟踪误差 RMS 为 0.464–0.717 rad。所有基线回合数值跨 π 跳变次数为零，是真实策略目标变化，不是周期包装伪影。原始 v16 已存在这个问题，续训各组的平均峰值均更大，但不是续训才首次产生。

2. **突变先于明显姿态偏离和倒地，并非倒地后才出现的补偿。** 35/35 首次大步目标在 11.008 s，固定姿态偏离观察阈值首次在 11.024 s，早 16 ms；事件前 0.5 s 均未触及该阈值。终止发生在 11.272–11.568 s，30 回合 tilt_limit、5 回合 wheel_stance_limit。这里的“偏离”是预设观察阈值，不是物理失稳起点的证明；系统在前几个子步已经开始响应。目标突变与命令直接切换的时间吻合，不能由此断言它是唯一原因。

3. **腿部有明确瞬态饱和，但不是始终饱和。** 原始阶跃前 0.2 s，各策略腿电机的物理子步饱和占比均值约 26.35%–53.90%；未限幅需求峰值均值约 125–172 N·m，而实际裁剪为 ±20 N·m。前 0.25 s 单电机最长连续饱和均值约 46–186 ms；完整基线高速回合最大连续饱和 238 ms。限速确实限制了参考步幅，但没有使任何一个高速回合通过，说明“裁剪目标大步变化”不足以修复这套闭环。

4. **轮毂速度环也进入能力受限区，瞬时换向要求不可直接跟踪。** 前 0.2 s 轮速误差 RMS 均值约 9.76–17.00 rad/s，超过速度环 8 rad/s 的满电流误差尺度；轮电机子步饱和占比约 27.6%–51.4%。完整基线高速回合中最长连续轮饱和 424 ms。当前 1.722 N·m 的输出上限与减速比换算一致，稳态 ±2 m/s 的理想轮速约 ±36.36 rad/s，小于策略 ±48 rad/s 上限；问题不只是最高转速。瞬态同时存在共同摆腿、轮地卸载与滑移，不能把全部误差归结为轮速 PI 增益或摩擦系数。速度环 Ki 实际为零。

5. **三档关节目标限速没有改善高速任务成功率，并存在明确代价。** 无限制及三档限速均为 0/35；平均换向后存活从 0.440 s 变为弱 0.441 s、中 0.500 s、强 0.564 s，延迟倒地没有形成完成任务的能力。普通后退停车从基线 10/35 变为弱 6/35、中 0/35、强 0/35；原始 v16 从 5/5 退化至中/强均 0/5。所有高速限速回合都存活到了 11 s，故高速失败结果没有被换向前倒地混淆。限速器在全程的介入并不频繁，但事件期介入明显，见下表。它还改变执行动作反馈和移动目标的相位，不宜据此否定所有可能的动作接口设计。Kd 对零目标速度施加阻尼，单纯降低位置目标速度也会削弱快速姿态恢复；本轮没有改变速度前馈。

6. **速度命令斜坡明显提高了固定测试中的存活。** 高速任务：阶跃 0/35，0.2 s 为 0/35，0.5 s 为 10/35（仅原始 v16 与 Projected seed2 各 5/5），1.0 s 为 35/35。三类策略都从 1.0 s 斜坡受益，且 105 条斜坡轨迹在 11 s 之前与各自基线逐元素相同。这支持命令突变/参考变化率是重要触发条件，不能解释成已解决原始高速阶跃任务。后退任务同时斜坡化起步和停车，分别为 27/35、35/35、35/35，不能把此结果单独归因于停车制动。

7. **原始 v16 普通后退停车的优势主要是保留了急后退起步恢复能力。** 四个续训策略的 20/20 回合在 3 s 的 0→−2 起步后约 0.44–0.79 s 就失败，根本没到 10 s 停车。Legacy seed1 的五回合到了停车后才失败，Projected seed3 与原始 v16 均完成五回合。原始起步目标步幅峰值均值 0.785 rad，续训约 0.847–0.937 rad；失败组中的多个策略伴随更大的轮速误差和更高的腿/轮饱和。Projected seed3 同样有更大步幅却存活，因此幅度并非充分解释。统一初始化、权重/归一化文件完整性检查与逐元素复现排除了这次评估实现漂移：确有续训后的能力退化，但尚不能证明由某个奖励项、Reset 开关或单一 PPO 机制导致。

8. **下一步先校准执行器模型并明确参考指令契约，再单独设计课程/动作接口消融。** 首先核实输出侧 0.246 N·m/A 与额外 0.70 效率的含义、实机速度—力矩能力和低层响应，保留本次控制链为对照。随后把可实现的加减速参考及阶跃压力测试明确分开；若进入训练实验，优先测试从平滑加减速到更陡换向的课程，并持续回归普通后退起步与停车，避免再次丢失已有能力。关节增量/残差或带速度参考的动作接口应作为独立变量与原绝对周期位置接口对照；本轮硬限速结果不支持直接部署，也不支持立即开展“已验证有效的限速器重训”。奖励结构放在这些控制与参考问题之后检查，本轮没有依据增加 Pitch 或左右对称惩罚。

## 统一窗口对照、接触与速度跟踪

'''
    text+='11 s 的第一周期，七个策略并没有立即把轮目标翻成后退方向：各策略平均目标从约 −36…−38 变为 −43…−48 rad/s，第一电机子步仍输出负向驱动力矩。正轮速目标约在切换后 0.176–0.232 s 才出现。与此同时，腿部目标大步改变，轮心共同向车身前方移动约 0.210–0.259 m（各策略峰值均值），第一个周期末实际轮速均值已达约 −45…−54 rad/s。正轮角速度对应车身后退，但该神经策略输出是平衡闭环的动作，不是把 vx 直接线性解码成轮速；初始“继续前驱”不是已证实的符号映射错误。后续速度误差与正向力矩饱和既包含目标方向延迟，也包含执行器和机构响应。\n\n'+table(first_summary)+'\n\n'
    text+=units+'\n\n'+table(comparison)+'\n\n'
    text+='下表是原始阶跃换向后前 0.2 s 的轮地快照统计：normal_force 为每轮平均 N；no_contact 是轮法向力 <1 N 的快照占比；friction_p95 为切/法向力比除以实际 μ；slip_p95 是接触点切向滑移速度 m/s；leg_demand 为未限幅需求峰值 N·m，common_peak 为 m。\n\n'+table(contact)+'\n\n'
    text+='前 0.2 s 有 10.0%–19.6% 的轮快照卸载至 <1 N；滑移 P95 均值 0.38–0.64 m/s，而摩擦利用率 P95 均值约 0.47–0.59。说明存在卸载和滑移，但不足以断言全程摩擦受限；完整摩擦锥形式和瞬时峰值亦应考虑，不能只用一个利用率百分位归因。\n\n'
    text+='下面单独检查 11–16 s 平移阶段。负速度和 −1.8 m/s 的首次达到时延从 11 s 算起；均值仅在达到该状态的回合计算，并明确列出达到数。它不是在该速度持续稳定的判定。失败后的数据不补齐，不把 0.44 s 的失败轨迹 RMSE 与完整 5 s 的斜坡轨迹 RMSE 当作公平优劣比较。\n\n'+table(tracking_summary)+'\n\n'
    text+='**1 s 斜坡的 35 回合都变成了负速度，但只有 14/35 在随后 5 s 内达到过 −1.8 m/s；它仍没有证明快速、准确的速度跟踪。**负速度首次达到的平均时延约 0.817 s；达到 −1.8 的 14 回合平均约 2.166 s，其余回合不能填成已达到。35/35 仅说明这七个既有策略在五个已用评估种子上的存活改善；这些种子不是新的独立训练重复，也不是未见条件的泛化证明。1 s 参考斜坡并不保证实际速度在 1 s 内完成反向，精确实际响应见 reversal_speed_tracking.csv。\n\n'
    text+='## 验证与复现产物\n\n'
    text+='全套 123 项测试通过（tests.log），包含周期边界/多圈限速、无干预动作保持、命令观测只替换无噪声字段以及斜坡时序检查。所有 70 条无干预轨迹与此前 A/B 轨迹完全相同；105 条高速斜坡在事件前完全相同。限速步幅最大数值超差 <4e−16 rad。源代码、七个 checkpoint 及七份归一化文件均保持原哈希。\n\n'
    text+='文件索引：experiment_plan.json 为事前冻结协议；evaluation_metrics.csv 为 490 回合全段指标/失败时刻；event_metrics.csv 为 735 个事件及完整窗口标记；target_step_distributions.csv 为逐回合 P50/P90/P95/P99/最大值；paired_results.csv 为同策略同种子的干预减基线；failure_timing.csv 为全部高速基线的时间排序；intervention_event_summary.csv、contact_summary.csv、reversal_speed_tracking.csv 为补充统计；各策略/条件/任务目录保留 NPZ 与 episodes.json。diagnostic_audit.json 是完整性和透明性校验；commands.sh 保存可复现命令。重新执行评估会复用已完成记录，分析命令只读取结果，不训练策略。\n'
    path=out/'report.md'
    original=path.read_text()
    intro=('完成 **490 回合配对评估**。高速原始阶跃与三档关节限速均为 **0/35**；1 s 换向命令斜坡为 **35/35**，但降低了测试难度。**大步目标、腿/轮饱和和共同摆腿同时存在；直接加关节限速不足以修复，而且会损害普通后退停车。下一步优先校准执行器并设计可实现的参考/课程对照，不立即改奖励或完整重训。**')
    original=original.replace('# UZ05 高速换向控制链诊断','# UZ05 高速换向控制链诊断\n\n'+intro,1)
    combined=original+'\n\n'+text
    while '\n\n\n' in combined:combined=combined.replace('\n\n\n','\n\n')
    path.write_text(combined)
