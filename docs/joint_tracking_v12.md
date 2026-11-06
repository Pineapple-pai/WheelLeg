# v12：高速后退、换向响应与直行航向

当前默认阶段为 [v13](joint_tracking_v13.md)，本文保留 v12 历史设置。

## v11 复测

使用最终 v11 checkpoint，名义动力学、无推力扰动、保留观测噪声，目标高度 0.27m。
固定速度每回合 24s，种子 20000/20001；稳态统计排除首秒。

| 命令 | 跑满 24s | 实际速度均值 | 实际腿长差 RMS | pitch 均值 |
|---|---|---|---|---|
| +1.7m/s | 2/2 | +1.549m/s | 0.40mm | −0.93° |
| −1.7m/s | 0/2 | −1.529m/s | 6.59mm | −9.45° |

后退统计包含失稳，不能解释为稳态能力。种子 20000 在 2.48s 触发倾角终止，
末段 pitch 约 −36.8°、腿长差约 21.8mm。前进虽不倒，峰值航向变化仍达 12.5°。
截图中的瞬时 yaw rate 接近零，也不能说明此前没有累积转向。

低速 ±0.5m/s 直接换向达到新目标 90% 约需 0.77–0.91s。
±1.5m/s 协议在 12s 开始正转反后，于 12.88s 倾倒；左腿连杆前伸达到约 0.134m，
已超过正常运动的约 0.10m，但原 0.13m 软阈值对此约束较弱。
失稳轨迹中闭链约束残差没有持续增长，不能将现象归因于闭链数值发散。
这些是有限种子复现结果，尚不能证明唯一原因。

原始报告：
`logs/joint_tracking_pos_v11_speed17_diagnosis.json`、
`logs/joint_tracking_pos_v11_reversal_diagnosis.json`、
`logs/joint_tracking_pos_v11_failure_trace_summary.json`。

## 固定修改

- **增加困难状态覆盖。** 平移命令方向固定为 60% 后退、40% 前进；绝对速度
  0–0.4 / 0.4–1 / 1–1.5 / 1.5–2m/s 的采样比例为 20/25/25/30%。
  35% 回合从 0.4–2m/s 的运动初态开始，方向同样偏重后退，轮速按滚动关系初始化；
  其余 65% 保留近静止初态。命令独立采样，让 PPO 接触高速恢复和制动状态。
  这只修改 reset 状态，不在运行中添加控制动作。
- **直接换向与响应。** 符合条件的纯平移命令重采样时，直接反向概率由 25% 提至 35%，
  高度保持原命令，幅度直接翻转。普通运动命令保持时间从 1–4s 改为 2–5s，
  让短换向后也有时间完成制动及跟踪；12–24s 长命令覆盖保留。
  速度连续误差成本权重从 0.8 提至 1.0，保留原平衡、姿态和平滑目标。
- **抑制累积转向。** 直行航向成本权重 0.6→1.0、尺度 5°→3°；
  yaw rate 连续误差成本权重 0.35→0.5、尺度 0.25→0.20rad/s。
  站立航向成本权重 0.35→0.5、尺度 5°→3°。主动转向时不惩罚航向偏移。
- **实际支撑几何。** 实际腿长差成本权重 0.5→0.6、尺度 5→4mm，保留 2mm 死区。
  连杆前伸软阈值 0.13→0.115m，伸出成本尺度 0.04→0.03m；后伸阈值仍为 0.27m。
  保留实际前后错位、roll、pitch 和高度目标，不强制左右关节角相同。

上述目标和采样比例从训练开始固定启用。动作仍为两轮独立速度及四关节周期位置，
执行器增益和额定限值保持 v11。没有教师、残差、课程、退火、动作滤波或命令爬坡。
近水平是软目标，允许加减速、抗扰时必要的短暂倾斜。

## 续训命令

```bash
cd /home/p/下载/WheelLeg
mkdir -p logs
bash scripts/train_joint_quiet_v23.sh \
  --stage joint_tracking_v12 --version joint_tracking_pos_v12 \
  --checkpoint checkpoints/joint_tracking_pos_v11/checkpoint \
  --reset-vecnormalize --preserve-action-std \
  --learning-rate 0.00003 --updates 4000 \
  2>&1 | tee logs/joint_tracking_pos_v12.log
```

继承 v11 策略和探索标准差，重新统计新分布的归一化量；固定学习率，不做退火。
默认启动阶段已改为 v12。终端不增加诊断行，运动初态占比仅写入 TensorBoard。

## 验证与训练后评估

71 项测试和 10 次更新的续训运行检查通过，普通评估、运动初态评估和换向评估入口均可运行。
v11 相同种子的前 310 帧轨迹回归检查通过。短续训模型在 reverse17 换向测试仍会倒下，
因此这些检查仅验证实现与运行，不能证明高速后退、换向和直行已经达标。

```bash
conda run --no-capture-output -n sim python scripts/eval_joint_tracking.py \
  checkpoints/joint_tracking_pos_v12/checkpoint --episodes-per-case 5 \
  --duration 24 --heights .27 --speed 1.7 --cases forward reverse \
  --output logs/joint_tracking_pos_v12_speed17_eval.json

conda run --no-capture-output -n sim python scripts/eval_joint_tracking_transitions.py \
  checkpoints/joint_tracking_pos_v12/checkpoint \
  --profiles reverse05 reverse15 reverse17 --episodes 3 \
  --output logs/joint_tracking_pos_v12_reversal_eval.json

bash scripts/replay_joint_tracking.sh \
  --checkpoint checkpoints/joint_tracking_pos_v12/checkpoint \
  --keyboard --nominal --speed .5 --height .27
```

普通评估和回放默认从近静止重置；显式加 `--moving-starts` 才使用训练的运动初态混合。
验收应比较两个方向的生存率、实际速度、换向 T90、航向漂移、腿长差和连杆前伸，
再检查高度、pitch、动作变化和饱和率，不能只比较训练 reward/length。
原生侧边栏和 HUD 保留，数字 7 站立，1/2 前后，3/4 转向，5/6 高度，± 调整，R 重置。
