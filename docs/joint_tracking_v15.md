# v15：约束两腿一起前后摆的支撑位置

v15完整4000次更新已完成，当前默认为[v16](joint_tracking_v16.md)。
v15仍有反向车身倾斜和左右轮轴错位，复测及后续修改见v16说明。

## v14复现

最终4000轮 `checkpoints/joint_tracking_pos_v14/checkpoint`，名义动力学、无推力、
保留观测噪声、初始近静止、高度0.27m。`reverse10`、`reverse10_backfirst` 和
`motion`协议各两个种子20000/20001，均跑满24s，但跑满不代表姿态合格。
报告：`logs/joint_tracking_pos_v14_posture_diagnosis.json`。
更细的轮位时序统计及峰值时关节角见
`logs/joint_tracking_pos_v14_support_recovery_diagnosis.json`（种子20000）。

以下为种子20000、`reverse10`协议。轮位是机体系中轮心相对对应髋部的X偏移，
不是机器人在地面上的总位移。关节角按±π展开显示，顺序为L1/L2/R1/R2。

| 切换 | 轮位最大绝对偏移 | 超出±8cm连续时长 | 峰值时关节角 |
|---|---|---|---|
| 站立→+1m/s | 15.2cm | 0.504s | −34.5/−38.7/−34.7/−39.6° |
| +1→−1m/s | 13.9cm | 0.608s | +34.8/+37.7/+35.5/+37.6° |
| −1→+1m/s | 22.8cm | 0.800s | −44.2/−57.0/−44.5/−62.4° |
| +1m/s→站立 | 13.7cm | 0.360s | +31.9/+33.6/+35.2/+33.8° |

−1→+1m/s切换时，左右轮前后错位峰值约1.9cm，轮位绝对偏移却达22.8cm。
两腿一起向后摆时，左右差值仍然很小；真实腿长相等也不等于轮子位于髋部下方。
这解释了对称性奖励无法单独约束这种姿态。现有轮位成本有±8cm软死区，
在此换向段平均仅约−0.08；轮位超过20cm需要连续25帧才终止，短时大幅伸出仍可能存活。
连杆前伸范围另有约束，但不等同于独立约束每侧轮心的前后位置。

本次复现主要是切换中的短时大幅摆腿，−1→+1段末秒轮位最大偏移已回到约3.85cm。
不能把它描述为所有时间都处于同一个异常稳态，也不能据有限样本认定唯一根因。

## 新的几何目标

v15继承v14，新增 `wheel_fore_aft_alignment`：

```text
x_side = 机体系中对应wheel_site相对hip_site的X偏移
excess_side = max(0, abs(x_side) - 0.03) / 0.04
cost = -mean(HuberSquare(excess_left), HuberSquare(excess_right))
```

HuberSquare近零为平方、大误差为线性。每侧先算成本再平均，防止两腿同向或反向伸出时抵消。
±3cm内成本为零；两侧各偏移6cm成本为−0.5625，各偏移15cm为−5.0，
各偏移22.8cm为−8.9。前后对称、左右对称，在前进、后退、转向、停车、站立均固定启用。
只增加这一项，不继续叠加关节角目标或摆速成本，便于区分新几何目标的作用。

这是软成本，不保证轮位永远在3cm内。保留允许加减速时必要的构型变化；
不硬夹关节角，不强制左右动作相同，不改电机参数、命令阶跃或终止边界。
v14的速度误差权重、换向采样、实际腿长差、roll/pitch、高度、漂移、输出平滑和连杆范围约束保持。
无命令爬坡、动作滤波、教师、残差或手写平衡外环；旧v14物理轨迹与回放定义保留。

该目标可能改变加减速时利用腿部摆动的策略，因此训练后必须同时检查速度响应和存活，
不能以姿态收紧为由接受速度跟踪明显退化。

## 训练与验证

```bash
bash scripts/train_joint_quiet_v23.sh \
  --stage joint_tracking_v15 --version joint_tracking_pos_v15 \
  --checkpoint checkpoints/joint_tracking_pos_v14/checkpoint \
  --reset-vecnormalize --preserve-action-std \
  --learning-rate 0.00003 --updates 4000 \
  2>&1 | tee logs/joint_tracking_pos_v15.log
```

`joint_tracking_pos_v15_smoke`是开发时10次更新的运行检查，
不作为姿态或速度已改善的证据。
86项测试通过；从v14继承的10次更新（40960步）正常完成并保存模型、归一化量及契约。
短训仍记录到倾角、轮位或触地终止，不能作为策略已达标的证据。
shell语法与`git diff --check`通过。额外的`stop_go10`和`stop_go10_backfirst`
名义动力学基线各1个种子也跑满24s，报告为`logs/joint_tracking_pos_v14_stop_go_support_diagnosis.json`。

```bash
conda run --no-capture-output -n sim python scripts/eval_joint_tracking_transitions.py \
  checkpoints/joint_tracking_pos_v15/checkpoint \
  --profiles reverse10 reverse10_backfirst stop_go10 stop_go10_backfirst motion height2_forward height2_reverse \
  --episodes 5 --output logs/joint_tracking_pos_v15_support_eval.json

conda run --no-capture-output -n sim python scripts/eval_joint_tracking.py \
  checkpoints/joint_tracking_pos_v15/checkpoint --episodes-per-case 5 \
  --duration 24 --heights .22 .27 .32 --speed 2 --cases stand forward reverse \
  --output logs/joint_tracking_pos_v15_held_eval.json
```

切换报告新增每侧轮位的有符号最小/最大值、任一侧超出±8cm的比例及最长连续时长、
轮位峰值时的关节角、峰值之后首次双侧回到±8cm并保持0.25s的恢复耗时、末秒轮位。
8cm作为跨版本可比的诊断带，不等于v15新成本的3cm死区。
除摆腿幅度与持续时间，还比较T90、±0.05m/s稳定时间、误差积分、pitch/roll、
真实腿长差、高度、停车漂移和饱和情况；未存活或未回稳时明确报告失败。
