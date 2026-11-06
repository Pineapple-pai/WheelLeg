# v6：pitch 近零收益、偏移代价与轮腿输出平滑

启动脚本现默认 v7，见 [v7 软对称目标](joint_tracking_v7.md)。复现本页 v6 训练需设置
`VERSION=joint_tracking_pos_v6` 并显式添加 `--stage joint_tracking_v6`。

默认阶段 `joint_tracking_v6`，输出 `checkpoints/joint_tracking_pos_v6/`。
正式 6000 轮训练及复测已完成，见 [v6 实测分析与回放](joint_tracking_pos_v6_analysis.md)。
固定命令名义/抗扰均为 54/54 跑满，pitch 明显改善；运动切换后的零命令保持与残余高频输出仍未完全达标。
v5 及以前的阶段、参数、checkpoint 保留；v5 前 125 步已保存轨迹回归对比一致。

## 设计依据

[v5 诊断](joint_tracking_pos_v5_pitch_oscillation.md)发现，前进/后退存在约 −12°/+10° 的持续俯仰偏置，
站立时则是带噪观测导致轮速和关节目标跳变。尤其平地上的地形预览每帧注入独立 ±5cm 噪声，
即使确定性 PPO 回放也会改变输出。单纯提高角速度惩罚无法解决持续倾斜，
单看低 pitch 波动或双轮接地也无法验收高频输出。

因此本次同步修改固定奖励与观测噪声配置，保持纯 PPO 六维动作：两轮独立速度目标，
四关节周期绝对位置目标，360° 旋转处理不变。没有动作滤波、教师、残差、课程、退火或平衡外环。
PPO 周期仍为 8ms，位置/轮速执行环每 2ms 更新；额定增益与限幅不变。

## pitch 奖励与惩罚

把原最高 3 分的直立预算拆为 roll 最高 1 分和 pitch 最高 2 分，不额外提高满分收益。
角度代入公式时使用弧度：

```text
roll_reward  = exp(-(roll / 12°)^2)
pitch_bonus  = 2 * exp(-(pitch / 5°)^2)
pitch_cost   = -0.4 * H(max(0, abs(pitch) - 3°) / 7°)
H(x)         = x²（x <= 1），否则 2*x - 1
```

pitch 正负方向完全对称；±3° 内无额外偏移代价，近零收益随角度平滑变化。
超过死区逐渐扣分，大角度代价线性增长，避免二次代价无限放大。
所有系数从训练开始固定，在站立、行驶、转向与受扰时都计算。
不硬限制 pitch 或关节动作；动态倾身与恢复仍有动作空间，但恢复效果需要训练后验收。

| pitch | pitch 奖励＋代价净值 |
|---|---:|
| 0° | +2.000 |
| ±3° | +1.395 |
| ±5° | +0.703 |
| ±8° | −0.049 |
| ±10° | −0.363 |
| ±12° | −0.622 |
| ±20° | −1.543 |

表中只包含 pitch 项，roll、速度、高度等另算。姿态几何无效时仍会衰减正向任务收益，
不能用斜撑获得完整 pitch 奖励。

速度跟踪保持权重 2.5，但 vx 指数尺度由 0.25 收紧到 0.15m/s：
当命令 +0.3m/s、实际原地不动时，此项收益约 0.046，而 v5 约 0.592。
目的是同时约束直立与跟踪，避免仅站直、持续欠速。
yaw、高度权重及漂移、支撑、饱和等目标保留。

## 高频输出与观测噪声

固定轮速目标变化权重从 0.04 调至 0.08，关节目标变化权重从 0.015 调至 0.03，
低于此前过强的 v3 配置。周期关节目标仍按最短角差计算，跨 ±π 不产生假跳变。

另增加实际轮速变化代价：每个物理子步采样轮速，计算
`-0.02 * mean(H(wheel_acceleration / 150rad/s²))`。
使用全部四个子步的变化，不先平均成一个轮速差，避免正反变化相消。
这是仿真训练代价，不修改输出指令、不设置变化率硬限幅。

观测噪声保持开启，调整后的物理量半宽如下；这些是新的固定假设范围，尚未用实机数据标定：

| 输入 | v5 每帧均匀噪声半宽 | v6 |
|---|---:|---:|
| 地形预览 | ±5cm | ±2mm |
| 机身高度 | ±5cm | ±5mm |
| 机身线速度 | ±0.05m/s | ±0.01m/s |
| 机身角速度 | ±0.1rad/s | ±0.02rad/s |
| 关节/轮速度 | ±0.75rad/s | ±0.10rad/s |
| 重力投影分量 | ±0.025 | ±0.005 |
| 水平位置 | ±5mm | ±2mm |

关节角和航向噪声保持原配置。没有关闭域随机化或外推力。
高度、vx、yaw、长短命令混合及所有鲁棒性分布沿用 v5，从第一步同时训练，回合最长 24 秒。

## 量级审计与验证

31 项测试通过，覆盖 pitch 净值与正负对称性、轮速子步变化不能相消、噪声单位与独立地形配置，
以及六维位置/速度动作没有被覆盖。完成从头 PPO 30 次更新、保存和独立评估链路检查，
以及 v5 到 v6 的两次更新续训链路检查。
短跑模型仍会很快倒下，仅证明链路正常，不作为性能结论。

冻结 v5 最终策略，在 v6 名义环境中分别运行站立、前进和后退 10 秒，排除首秒：

| 模式 | pitch 均值 | pitch 项净值均值 | 控制代价 / 正向任务收益 |
|---|---:|---:|---:|
| 站立 | −0.014° | +1.973 | 0.147% |
| 前进 | −11.14° | −0.516 | 0.211% |
| 后退 | +10.18° | −0.382 | 0.204% |

这说明旧策略的持续大倾角会收到明确的 pitch 代价，新增控制代价未主导这些轨迹。
噪声、奖励同时改变；这不是同轨迹的单项消融，更不是 v6 已学习后的表现。
[完整审计数据](../logs/joint_tracking_v6_reward_audit.json)可通过以下命令重现：

```bash
conda run --no-capture-output -n sim python scripts/audit_pitch_control.py
```

## 正式训练

默认从头训练，避免直接延续旧模型的倾斜姿态与方向能力偏置：

```bash
cd /home/p/下载/WheelLeg
mkdir -p logs
bash scripts/train_joint_quiet_v23.sh 2>&1 | tee logs/joint_tracking_pos_v6.log
```

默认 6000 次更新、8 环境、每环境 rollout 512，共 24,576,000 个采样步。
学习率固定 1e-4、熵系数固定 0.001；初始动作标准差 0.06，随后由 PPO 自己学习。
模型保存在 `checkpoints/joint_tracking_pos_v6/`，不会覆盖旧版本。

v5 与 v6 观测参考和动作接口相同，技术上可迁移，但跨奖励版本必须重置奖励归一化，
训练入口会拒绝省略 `--reset-vecnormalize` 的跨版本续训。
默认命令没有加载 v5，审计也不会给训练提供教师动作。

新增日志包括 `Posture/pitch_mean_deg`、`Posture/pitch_rms_deg`、
`Posture/pitch_within_5deg_fraction`、`Reward/pitch_cost_mean` 和
`Control/wheel_acceleration_rms_rad_s2`。
`Reward/task_mean` 包含 pitch 正收益，pitch 负代价单列，输出变化代价计入控制分组。
checkpoint 和 training_config 记录实际奖励与噪声参数。

## 验收

```bash
conda run --no-capture-output -n sim python scripts/eval_joint_tracking.py \
  checkpoints/joint_tracking_pos_v6/checkpoint --duration 24 --episodes-per-case 3 \
  --output logs/joint_tracking_pos_v6_eval.json

conda run --no-capture-output -n sim python scripts/eval_joint_tracking.py \
  checkpoints/joint_tracking_pos_v6/checkpoint --duration 24 --episodes-per-case 3 \
  --domain-randomization --perturbations --output logs/joint_tracking_pos_v6_stress_eval.json

bash scripts/replay_joint_tracking.sh --keyboard --nominal \
  --checkpoint checkpoints/joint_tracking_pos_v6/checkpoint
```

不能只按 reward 选择模型。至少同时检查分模式 24 秒存活、前进实际速度、稳定段 pitch RMS 与
±5° 采样比例、站定漂移和输出变化；按高度分别检查正反方向。
评估新增 `settled_pitch_*` 与 `settled_wheel_acceleration_rms_rad_s2`，均排除首秒。
这些量不能排除之后发生的起停或推力恢复，需要结合完整轨迹判断；
平均 pitch 正负相消也可能掩盖大幅摆动，因此必须同时看 RMS 与范围内比例。

上述评估命令为小样本筛查，候选模型需扩大种子验证。
本轮最终结果、失败案例和回放以 [v6 实测报告](joint_tracking_pos_v6_analysis.md)为准。
