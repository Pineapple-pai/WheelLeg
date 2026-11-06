# v5：双腿起伏、抗扰恢复与持续前进

启动脚本现默认 v6，见 [v6 修改与训练命令](joint_tracking_v6.md)。复现本页 v5 训练时设置
`VERSION=joint_tracking_pos_v5` 并显式传入 `--stage joint_tracking_v5`。

阶段为 `joint_tracking_v5`，启动脚本默认输出 `checkpoints/joint_tracking_pos_v5/`。
这是代码与训练分布修订。6000 轮训练已完成，详见 [v5 实测分析与回放](joint_tracking_pos_v5_analysis.md)；
名义固定命令仅 27/54 跑满 24 秒，整体稳定性尚未达标。
v1–v4 阶段与 checkpoint 保留。

## 诊断与修改

v4 在固定前进任务下仅 1/30 回合跑满 12 秒，而随机切换命令的 20/20 存活不能反映这个短板。
种子 20000、0.27m 零命令回放中，排除首秒后每条腿的长度跨度约 2.2cm，
左右腿长度差的变化率 RMS 约 0.18m/s。详见
[v4 结果](joint_tracking_pos_v4_analysis.md)和
[腿部复现数据](../logs/joint_tracking_pos_v4_ground_leg_audit.json)。

v5 同时处理以下问题，所有分布与权重从第一次采样起固定启用：

* **执行器时序**：PPO 仍每 8ms 输出一次六维位置/速度目标，但关节 PD 和轮速环改为每个
  2ms 物理步读取最新状态并重新计算输出。旧实现将首次计算的力矩保持四个物理步。
  不提高额定力矩、电流上限或增益，也不增加姿态平衡外环。
* **腿部支撑**：约束实际轮心相对髋部的竖向落差之差、差动速度、缺失的地面轮支撑，
  并在站立时约束机身竖向速度。落差差异留 1cm 死区；这些是软代价，允许恢复时动作。
  不指定关节角度、关节运动相位或左右接触力相等，也不设置必须运动的下限。
* **持续前进**：运动任务的累计起点位移与航向误差观测置为零，避免速度跟踪依赖越来越大的
  世界坐标位移。零命令仍锁定进入站立时的位置/航向参考，后续高度变化不会刷新锚点。
* **训练覆盖**：站立、纯 vx、纯 yaw、混合模式各约 25%，高度 0.24–0.30m、
  vx ±0.5m/s、yaw ±1.5rad/s 从开始同时采样。零命令保持 12–24 秒；运动命令一半保持
  12–24 秒、一半保持 2–5 秒；回合上限 3000 步，即 24 秒。
* **鲁棒性**：固定范围随机质量、惯量、重心、摩擦、阻尼和执行器参数；每隔约 3–6 秒
  随机施加 5–15N 水平推力，持续 80–152ms。方向独立于机器人状态，是环境扰动，
  不提供恢复动作或恢复力。初态倾角 ±0.07rad、角速度 ±0.12rad/s、速度 ±0.08m/s。
* **实现问题**：执行器使用参数副本，避免每次 reset 的随机增益继续乘在上一回合增益上；
  reset 高度改为 0.315m，避免初始轮碰撞体深陷地面。轮支撑统计只计地面接触，
  避免把轮与机器人零件的接触计为支撑。

策略仍是纯 PPO：左右轮独立速度目标和四关节周期绝对位置目标，共六维。
关节按最短等价位置误差执行、跨 ±π 连续处理，允许 360° 旋转；不锁关节。
策略不输出力矩；仿真内的 PD/电调把位置和速度目标转换成物理驱动力矩。
没有教师、残差、课程、退火、动作滤波或额外的平衡控制器。

固定域参数：质量倍率 0.95–1.10，惯量 0.90–1.10，水平重心偏移 ±8mm、竖直 ±5mm，
摩擦 0.85–1.15，关节阻尼 0.90–1.10，PD/轮速增益和轮驱动转矩系数 0.95–1.05。
未实现执行器延迟，因此延迟参数固定为零，不计作已训练的鲁棒性。

## 固定奖励

保留 v4 任务权重：直立 3、vx 跟踪 2.5、yaw 跟踪 2、高度跟踪 2，失败代价 30。
保留轮目标变化 0.04、关节目标变化 0.015、实际输出幅值 0.05、实际输出变化 0.025、
饱和 0.03。没有进一步用强平滑代价压制恢复动作。

零命令位置/航向代价分别为 1.0/0.35；站定误差还会衰减零命令的速度跟踪收益，
避免缓慢漂移持续获取完整跟踪分。直立与高度收益仍保留，支持恢复。
新增支撑代价权重：落差差异 0.25、差动速度 0.08、每缺一个轮支撑 0.35、
站立机身竖向速度 0.08。完整尺度写在 `SupportPrecisionRewards` 中。

实际输出变化、幅值和饱和按全部 2ms 子步统计，不先把正负力矩平均，
避免往复饱和被平均成低输出。新旧执行频率不同，输出变化指标需结合饱和和轨迹判断。
奖励定义和回合上限已经变化，总 reward/length 不能直接与 v4 比大小。

## 训练

从头训练，不能直接续接 v4：观测参考语义与执行器时序已变化，契约会拒绝不匹配的 checkpoint。

```bash
cd /home/p/下载/WheelLeg
mkdir -p logs
bash scripts/train_joint_quiet_v23.sh 2>&1 | tee logs/joint_tracking_pos_v5.log
```

默认 6000 次更新、8 个环境、每环境 rollout 512 步，共 24,576,000 个采样步。
学习率固定 1e-4、熵系数固定 0.001；初始动作标准差 0.06，之后由 PPO 自身学习，
无手动标准差退火。每 100 轮保存 checkpoint，末尾保存在 `checkpoints/joint_tracking_pos_v5/checkpoint`。
终端和日志文件同时输出，使用无缓冲 Python 与 `conda --no-capture-output`。

新增日志 `Support/leg_drop_difference_rms_m`、`Support/leg_differential_velocity_rms`、
`Support/dual_wheel_fraction` 和 `Robust/push_active_fraction`。
原有任务误差、奖励分组、零命令漂移、输出变化和饱和指标继续记录。

## 训练后验收与回放

先做名义动力学的 24 秒恒定命令评估，覆盖三个高度及站立、正反行驶、正反旋转和混合运动：

```bash
conda run --no-capture-output -n sim python scripts/eval_joint_tracking.py \
  checkpoints/joint_tracking_pos_v5/checkpoint --duration 24 --episodes-per-case 10 \
  --output logs/joint_tracking_pos_v5_nominal_eval.json
```

再启用动力学随机化与外推力，评估相同恒定命令：

```bash
conda run --no-capture-output -n sim python scripts/eval_joint_tracking.py \
  checkpoints/joint_tracking_pos_v5/checkpoint --duration 24 --episodes-per-case 10 \
  --domain-randomization --perturbations \
  --output logs/joint_tracking_pos_v5_stress_eval.json

conda run --no-capture-output -n sim python scripts/eval_joint_tracking_random.py \
  checkpoints/joint_tracking_pos_v5/checkpoint --episodes 20 \
  --output logs/joint_tracking_pos_v5_random_eval.json
```

随机评估默认采用模型所属阶段的训练环境；加 `--nominal` 可关闭域随机化和推力。
除了存活与跟踪，重点检查零命令 5cm/5°、持续前进 24 秒、输出变化及饱和。
建议各模式名义存活率至少 95%，且零命令稳定段双轮接地比例至少 95%；
腿部差动速度和长度跨度应较 v4 明显下降，同时保持有扰动时的恢复能力。
这组数值是待验证目标，并非已达到的成绩。
`settled_*` 指标排除每回合首秒：包括逐回合腿长跨度的平均值、竖向差动速度 RMS、双轮接地比例。
前述旧版审计的 0.18m/s 是腿长差动速度，而新日志为竖向落差差动速度，二者不同。
使用当前评估器复测的 [v4 同定义基线](../logs/joint_tracking_pos_v4_support_baseline.json)
（种子 20000、0.27m、站立 12 秒）为：排除首秒后竖向差动速度 RMS 0.173m/s、
双轮接地比例 80.4%、两腿长度跨度约 2.2cm。新模型应与这组同定义数据比较。

数字键盘回放（5 站立，8 前进，2 后退，4/6 转向，1/3 改高度，R 重置）：

```bash
bash scripts/replay_joint_tracking.sh --keyboard --nominal \
  --checkpoint checkpoints/joint_tracking_pos_v5/checkpoint
```

去掉 `--nominal` 可观察含随机参数和外推力的恢复。键盘模式每回合最多 24 秒；
非交互前进可用 `--case forward --duration 24 --once`。

配置发布时完成 26 项测试、30 轮 PPO 短跑、模型保存与两种固定命令评估入口检查。
短跑模型不到一秒即倒下，不能作为训练效果证明；后续正式 6000 轮训练的结果以
[v5 实测报告](joint_tracking_pos_v5_analysis.md)为准。
