# v9：直行偏航、速度欠跟踪与运动升沉

启动脚本现默认 v11，见 [v11 复现、修改与续训](joint_tracking_v11.md)。
复现本页训练需显式指定 `--stage joint_tracking_v9` 并设置独立 `VERSION`。

v8 的 6000 轮终点仍未达到平移目标。以下是名义动力学、无推力扰动、保留观测噪声、
高度 0.27m、种子 20000/20001、每回合最多 24s 的固定命令复现。
命令从 reset 后施加，键盘中从已站稳姿态开始的结果可能不同。
速度和 pitch 统计排除首秒；偏航峰值记录整回合连续展开后的角度。

| 目标 vx | 实际 vx 均值 | 跑满 24s | pitch RMS | 累计偏航绝对峰值 |
|---|---|---|---|---|
| +0.5m/s | +0.306m/s | 2/2 | 14.85° | 258.2° |
| −0.5m/s | −0.230m/s | 2/2 | 2.19° | 268.9° |
| +0.7m/s | +0.327m/s | 1/2 | 20.28° | 315.4° |
| −0.7m/s | −0.266m/s | 2/2 | 2.33° | 312.1° |

原始报告：`logs/joint_tracking_pos_v8_speed05_diagnosis.json`、
`logs/joint_tracking_pos_v8_speed07_diagnosis.json`。
这是有限种子的复现，不是整体成功率估计。前进 +0.7m/s 时，两腿首秒后长度跨度平均约
6.8cm/8.1cm，同时高度标准差约 1.9cm，已构成明显姿态失稳。

## 原因与改动

1. **直行航向不可观测。** v8 在全部运动模式将航向误差置零，小 yaw 角速度可以长期累计。
   v9 进入纯平移时记录当前航向，保留真实误差观测，增加 `−0.6 H(航向误差/5°)` 软成本。
   前后切换、速度调整、改变高度保留参考；停止或转弯后重新进入平移时重新取参考。
   转弯命令不受直行航向惩罚，停止仍使用原停止锚点。参考只用于观测和奖励，不输出控制修正。
2. **较大速度误差下奖励改善信号弱。** 原精细高斯 sigma=0.15m/s，在误差 0.4m/s 处
   跟踪正奖励已接近零。v9 使用权重 3.5、sigma=0.20m/s 的精细跟踪，加权重 1.5、
   sigma=0.70m/s 的宽范围奖励，速度 Huber 成本权重从 0.15 增至 0.8。
   两种正奖励都受原有支撑几何质量约束，停止时也受站位质量约束。
   新增 yaw 角速度成本 `−0.35 H(yaw_rate_error/0.25)`，增强瞬时纠偏。
3. **运动时也约束升沉。** 保留 pitch 近零奖励、pitch 偏离、实际腿长差、前后构型差和 roll
   软成本；pitch 角速度成本权重 0.04→0.08，新增运动中世界系竖直速度成本
   `−0.08 H(vz/0.15)` 及两腿平均长度变化率成本 `−0.04 H(mean_leg_rate/0.20)`。
   不要求两腿关节角相同，不锁关节，正常升降和恢复运动仍可发生。
4. **固定分布覆盖中速直行。** 从第一批环境起，停止/纯平移/纯转弯/混合运动比例固定为
   20%/45%/15%/20%；平移速度正负各半，幅值 0–0.4、0.4–1.0、1.0–2.0m/s
   比例固定为 25%/40%/35%。长命令概率固定 50%，高度仍从 0.22–0.32m 同时采样。
   不使用课程、退火、命令斜坡、动作滤波、教师、残差或额外平衡控制。

`H(x)=0.5x²`（|x|≤1），其余为 `|x|−0.5`，是软成本而非硬约束。
奖励形状检查见 `logs/joint_tracking_v9_reward_shape_audit.json`，只证明目标形状，不代表模型表现。
六维动作、±48rad/s 轮速目标范围、四个圆周关节位置目标和执行器额定限值均保留。

## 训练与验证

v9 改变了运动时航向观测含义，契约会拒绝把 v8 checkpoint 当作 v9 直接续训。
从头训练，默认输出新目录：

```bash
cd /home/p/下载/WheelLeg
mkdir -p logs
bash scripts/train_joint_quiet_v23.sh 2>&1 | tee logs/joint_tracking_pos_v9.log
```

默认 6000 轮、固定学习率 1e−4，终端继续保持精简，新增航向诊断写入 TensorBoard。
`Tracking/straight_heading_rms_deg` 仅统计纯平移帧；`Reward/straight_tracking_cost_mean`
记录新增航向/角速度成本。不能据 reward/length 单独验收。

本次通过 55 项测试、30 轮初步训练及奖励门控修订后的 10 轮 subproc 训练。
短训练仅验证数据流和数值可运行，尚未产生可用 v9 策略，也未启动正式长训练。

训练后先分别验收 ±0.5、±0.7m/s，再检查 ±1、±2m/s；用相同种子对比存活、
首秒后实际均速、yaw 角速度均值/累计偏航、pitch RMS/峰值、腿长跨度和高度波动。
新评估字段不依赖 v8 被置零的航向误差观测，可直接发现真实累计偏航。

```bash
conda run --no-capture-output -n sim python scripts/eval_joint_tracking.py \
  checkpoints/joint_tracking_pos_v9/checkpoint --episodes-per-case 5 \
  --duration 24 --heights .27 --speed .7 --cases forward reverse \
  --output logs/joint_tracking_pos_v9_speed07_eval.json

conda run --no-capture-output -n sim python scripts/eval_joint_tracking_transitions.py \
  checkpoints/joint_tracking_pos_v9/checkpoint --episodes 3 \
  --output logs/joint_tracking_pos_v9_transition_eval.json

bash scripts/replay_joint_tracking.sh \
  --checkpoint checkpoints/joint_tracking_pos_v9/checkpoint \
  --keyboard --nominal --speed .5 --height .27
```

原生查看器及顶部屏幕文字保持不变，`7` 站立、`R` 重置。
复现 v8 训练时需显式指定 `--stage joint_tracking_v8` 并使用独立 `VERSION`。
已有 v8 模型回放仍按其原观测语义运行。
