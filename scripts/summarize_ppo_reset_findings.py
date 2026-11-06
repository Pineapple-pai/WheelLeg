"""Post-hoc trajectory explanation; does not change preregistered A/B metrics."""
from pathlib import Path
import numpy as np
import pandas as pd

MARKER = '\n## 补充：失败阶段与共同摆腿的探索性分析\n'


def supplement(out):
    out = Path(out)
    data = pd.read_csv(out/'evaluation_metrics.csv')
    final = data[(data['update'] == 200) & (data['mode'] != 'baseline')]
    baseline = data[data['mode'] == 'baseline']
    lines = [MARKER, '',
        '以下是在完整结果出来后的解释性分析，不改变预注册的成功标准、主指标或 B 决策。', '',
        '### 与原始 v16 比较', '',
        '| 任务 | 原始 v16 | Legacy 续训后三种子合计 | Projected 续训后三种子合计 |',
        '|---|---:|---:|---:|']
    for task in ('stand','forward','reverse','motion','stop2_reverse','reverse10','height2_reverse'):
        samples = [baseline[baseline.task == task]] + [
            final[(final.task == task) & (final['mode'] == m)] for m in ('legacy','projected')]
        cells = [f'{int(x.success.sum())}/{len(x)}' for x in samples]
        lines.append('| '+task+' | '+' | '.join(cells)+' |')
    lines += ['', '普通后退停车原始 v16 为 5/5，续训 Legacy 为 0/15、Projected 为 5/15，'
        '因此不能只比较两组之间的差值而忽略两组相对原始模型的退步。'
        '汇总 15 回合不是 15 个独立训练种子；仍只有三个训练重复。', '',
        '失败阶段按既定命令事件划分，未改变整段成功判定：', '',
        '| seed/mode | 高速协议平均存活秒 | 后退停车平均存活秒 | 高位后退停车平均存活秒 |',
        '|---|---:|---:|---:|']
    phases = []
    for _, row in final.iterrows():
        if row.task == 'motion':
            phase = 'before_reversal' if row.survival_s < 11 else ('reversal' if row.survival_s < 16 else 'turn_or_stop')
        elif row.task == 'stop2_reverse':
            phase = 'before_stop_command' if row.survival_s < 10 else 'after_stop_command'
        elif row.task == 'height2_reverse':
            phase = 'before_first_height_change' if row.survival_s < 8 else ('height_changes' if row.survival_s < 18 else 'after_high_stop')
        else: continue
        phases.append(dict(train_seed=row.train_seed, mode=row['mode'], task=row.task,
                           eval_seed=row.eval_seed, survival_s=row.survival_s, success=row.success,
                           phase='complete' if row.success else phase))
    pd.DataFrame(phases).to_csv(out/'failure_phases.csv', index=False)
    for seed in (1,2,3):
        for mode in ('legacy','projected'):
            a = final[(final.train_seed == seed) & (final['mode'] == mode)]
            v = [a[a.task == t].survival_s.mean() for t in ('motion','stop2_reverse','height2_reverse')]
            lines.append(f'| {seed}/{mode} | '+' | '.join(f'{x:.3f}' for x in v)+' |')
    lines += ['', '高速完整换向发生在 11s，两组都在其后约 0.27–0.49s 失败。'
        '普通后退停车命令在 10s，但 Projected seed1/2 和 Legacy seed2/3 '
        '在 3s 开始 −2m/s 加速后不到 0.8s 就失败，尚未进入停车阶段；'
        '这些回合不能当作孤立的制动控制失败。', '',
        '### 共同后摆仍然存在', '',
        '选取两组全部存活的 reverse10 协议分析，避免较早失败导致 RMS 人为变小。'
        '该协议在 14s 从 −1m/s 切到 +1m/s。']
    events = []
    for seed in (1,2,3):
        for mode in ('legacy','projected'):
            for es in range(51000,51005):
                z = np.load(out/f'seed_{seed}'/mode/'eval_200'/f'reverse10_{es}.npz')
                a, c = z['samples'], z['columns'].tolist()
                i = int(np.argmax(np.abs(a[:,c.index('common_x_m')])))
                t = a[i,0]; change = max([0]+[x for x in (3,8,14,20) if x <= t])
                events.append(dict(train_seed=seed, mode=mode, eval_seed=es, peak_time_s=t,
                    since_command_change_s=t-change, left_x_m=a[i,c.index('wheel_x_left_m')],
                    right_x_m=a[i,c.index('wheel_x_right_m')], pitch_deg=np.degrees(a[i,c.index('pitch_rad')]),
                    body_acceleration_m_s2=np.gradient(a[:,c.index('vx_m_s')],.008)[i]))
    ev = pd.DataFrame(events); ev.to_csv(out/'common_swing_events.csv', index=False)
    count = int((ev.since_command_change_s <= 1).sum())
    same = int((ev.left_x_m*ev.right_x_m > 0).sum())
    lines += [f'30 个回合中，{count}/30 的共同偏移全局峰值落在命令切换后 1s 内，'
        f'{same}/30 的左右轮心同向偏移；平均峰值发生在 14.136–14.160s，'
        '轮心相对髋轴共同后移约 18–21cm，左右差 RMS 仅约 3–8mm。'
        '这明确区分了共同后摆与左右不对称。', '',
        '| seed | reverse10 共同偏移 RMS Legacy→Projected m | 配对变化 m |', '|---|---:|---:|']
    for seed in (1,2,3):
        vals = [final[(final.train_seed == seed) & (final['mode'] == mode) &
                      (final.task == 'reverse10')].common_x_rms_m.mean() for mode in ('legacy','projected')]
        lines.append(f'| {seed} | {vals[0]:.5f}→{vals[1]:.5f} | {vals[1]-vals[0]:+.5f} |')
    lines += ['', '在这个时长完全相同的成功协议中，Projected 三个训练种子的共同摆动 RMS 都略增，'
        '没有减少共同摆腿。它发生在 Reset 十多秒之后，说明仅修复初始化没有消除该切换行为。', '',
        '共同位移与 pitch 的最大绝对滞后相关系数均值约 +0.31…+0.47，'
        'pitch 领先共同位移约 16–40ms；与实际机体系速度差分所得加速度的对应值约 −0.61…−0.67，'
        '加速度滞后约 0.11–0.14s。它们是同一闭环系统内的相关关系，不能证明共同摆腿导致 pitch 或相反。'
        '峰值时实际加速度不一定已变正，不能只按目标命令方向判断瞬时倾角是否合理。', '',
        '### 目标大步变化与执行器响应', '',
        f'210 个最终评估回合的数值目标跨 π 跳变总数为 {int(final.target_numeric_jump_count.sum())}。'
        '因此本次未发现周期坐标跨界导致的 >π 目标跳变；这不等于策略目标足够平滑。']
    worst = final.loc[final.target_circular_step_peak_rad.idxmax()]
    z = np.load(out/f'seed_{int(worst.train_seed)}'/worst['mode']/'eval_200'/f'{worst.task}_{int(worst.eval_seed)}.npz')
    a,c = z['samples'],z['columns'].tolist()
    target = a[:,[c.index(f'target_q_{j}') for j in range(4)]]
    delta = np.arctan2(np.sin(np.diff(target,axis=0)),np.cos(np.diff(target,axis=0)))
    index,joint = np.unravel_index(np.abs(delta).argmax(),delta.shape)
    lines += [f'最大真实周期目标单步变化为 {abs(delta[index,joint]):.4f} rad（约 '
        f'{np.degrees(abs(delta[index,joint])):.1f}°），发生于 Projected seed2、motion、eval seed '
        f'{int(worst.eval_seed)} 的 {a[index+1,0]:.3f}s、joint {joint}：正好是 +2→−2m/s 命令切换后的第一控制周期。'
        '这是策略输出目标本身的大步变化，不是 ±π 包装伪影。', '',
        'reverse10 目标与实际关节位置的周期误差 RMS 约 0.125–0.131rad，绝对误差 P95 约 0.149–0.154rad。'
        '这显示有限响应滞后，但不能在没有参照轨迹/带宽要求时直接断言它“严重”。'
        f'全部最终评估中，某腿电机最长完整周期连续饱和 {final.longest_joint_full_cycle_saturation_s.max():.3f}s，'
        f'轮电机为 {final.longest_wheel_full_cycle_saturation_s.max():.3f}s。'
        '平均饱和比例并不表示电机始终触限，切换瞬间的短时饱和仍值得检查。', '',
        '下一步优先核查切换瞬间的关节目标变化、可实现速度/力矩和姿态耦合；'
        '本次数据不能证明其中某一项是根因，也不支持直接提高 pitch 权重或叠加对称奖励。', '',
        '![最大目标变化案例](switch_target_jump.png)', '',
        '### 审计与运行记录', '',
        'actual_config_verification.json 验证三个实际配置对除 Reset 开关及输出元数据外一致，'
        '且原始 v16 的动作/执行器与奖励配置保持不变。完整 119 项测试通过。'
        '报告整理的 CSV 列名读取错误修复后，复用了原有 335 回合数据，没有重新训练或改变评估。', '']
    report = out/'report.md'
    original = report.read_text().split(MARKER)[0]
    report.write_text(original+'\n'.join(lines)+'\n')
    _plot_jump(out,int(worst.train_seed),int(worst.eval_seed))


def _plot_jump(out, seed, es):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4,2,figsize=(14,12))
    colors = {'legacy':'#2878b5','projected':'#d27a15'}
    for mode,color in colors.items():
        z=np.load(out/f'seed_{seed}'/mode/'eval_200'/f'motion_{es}.npz');a=z['samples'];c=z['columns'].tolist()
        for j in range(4):
            ax=axes.ravel()[j]
            ax.plot(a[:,0],a[:,c.index(f'actual_q_{j}')],color=color,label=f'{mode} actual')
            ax.plot(a[:,0],a[:,c.index(f'target_q_{j}')],'--',color=color,label=f'{mode} target')
            ax.set_title(f'Joint {j} (rad)')
        for ax,key in ((axes[2,0],'common_x_m'),(axes[2,1],'pitch_rad')):
            ax.plot(a[:,0],a[:,c.index(key)],color=color,label=mode);ax.set_title(key)
        wheel=lambda name:np.mean(a[:,[c.index(f'{name}_{i}') for i in range(2)]],axis=1)
        axes[3,0].plot(a[:,0],wheel('wheel_actual'),color=color,label=f'{mode} actual')
        axes[3,0].plot(a[:,0],wheel('wheel_target'),'--',color=color,label=f'{mode} target')
        axes[3,0].set_title('Mean left/right wheel speed (rad/s)')
        axes[3,1].plot(a[:,0],wheel('wheel_torque_nm'),color=color,label=mode)
        axes[3,1].set_title('Mean left/right output torque (Nm)')
    for ax in axes.ravel():
        ax.set_xlim(10.5,11.55);ax.axvline(11,color='gray',linestyle=':');ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.suptitle(f'Post-hoc maximum target-step case: train seed {seed}, eval seed {es}, +2 to -2 m/s at 11 s; lines end on failure')
    fig.tight_layout();fig.savefig(out/'switch_target_jump.png',dpi=170);plt.close(fig)


if __name__ == '__main__':
    supplement(Path(__file__).resolve().parents[1]/'diagnostics/ppo_reset_ab')
