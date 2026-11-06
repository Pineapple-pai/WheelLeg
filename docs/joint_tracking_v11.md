# v11：实际连杆伸出、pitch 与左右支撑一致性

当前默认训练为 [v12](joint_tracking_v12.md)，本文保留 v11 配置和历史复测。

## v10 实测

名义动力学、无推力扰动、保留观测噪声，高度目标 0.27m，种子 20000/20001，
固定命令每回合最多 24s。速度/高度/pitch 统计排除首秒：

| vx 命令 | 实际 vx | 跑满24s | 高度均值 | pitch 均值 | 实际腿长差 RMS |
|---|---|---|---|---|---|
| +0.5m/s | +0.485m/s | 2/2 | 0.2742m | −2.95° | 2.19mm |
| −0.5m/s | −0.503m/s | 2/2 | 0.2845m | +6.79° | 1.97mm |
| +1.5m/s | +1.411m/s | 2/2 | 0.2694m | −4.24° | 9.24mm |
| −1.5m/s | −1.444m/s | 0/2 | 0.2728m | −28.52° | 48.99mm |

−1.5m/s 的均值含失稳过程，不能解释为稳态能力。v10 的高速前进取得进展，
高速后退及直接换向尚未同样通过。+1.5m/s 时 roll RMS 约 0.98°，与左右实际腿长差一起说明
画面中的支撑偏差并非仅是透视效果。有限种子测试用于复现，不代表总体成功率。

固定换向协议 `reverse15` 中，连杆向前伸出量从正常约 0.10m 增至左 0.240m、右 0.201m，
最后触发支撑几何终止。原有 `wheel_offset`/`wheel_stagger` 只看轮心相对髋的位置，
不能完整描述中间连杆的伸出。

原始报告：`logs/joint_tracking_pos_v10_speed05_diagnosis.json`、
`logs/joint_tracking_pos_v10_speed15_diagnosis.json`、
`logs/joint_tracking_pos_v10_geometry_reversal_diagnosis.json`。

## 固定软目标

1. **实际连杆包络。** 对各腿所有物理碰撞胶囊，计算机体系 X 方向相对该侧髋轴的完整投影，
   包含线段端点和半径。新目标不依赖显示网格，也不将轮心位置代替连杆伸出量。
   前伸超过 0.13m、后伸超过 0.27m 后分别开始 `−0.4 mean(H(excess/0.04))` 成本。
   前后阈值不同来自该机构的实际结构：正常后伸包络已经约 0.255m；采用相同阈值会惩罚正常构型。
   两者都是软成本，恢复时可短暂越出，不裁剪关节动作或关节角。
2. **近水平 pitch。** 保留 v10 的近零正奖励，负成本死区从 3° 改为 1°，尺度从 7° 改为 3°，
   权重从 0.4 改为 0.8。约 4.25° 的 pitch 成本从约 0.006 增为 0.467。
   成本正负对称，仍允许加减速和恢复时短暂倾斜，不要求任何时刻 pitch 严格等于零。
3. **实际腿长差及 roll。** 腿长差死区 3→2mm，尺度 10→5mm，权重 0.35→0.5。
   roll 正奖励 sigma 5→3°，负成本死区 1→0.5°、尺度 4→2°、权重 0.25→0.4。
   保留 v10 原有前后构型错位、垂向支撑差和差动速度目标，不重复叠加更强的同类成本。
   所有量都使用实际空间几何；左右四个关节目标仍独立，不强制关节角相同。
4. **前后高度偏差。** 双向高度 Huber 成本从 `−0.4 H(error/0.01)` 改为
   `−0.6 H(error/0.008)`。高度命令相同，两个运动方向都朝同一高度收敛。
   不引入方向相关的目标高度或目标 pitch。

H 是标准 Huber 函数：|x|≤1 为 0.5x²，其余为 |x|−0.5。
速度/yaw 跟踪预算、±2m/s 与 0.22–0.32m 命令范围、固定命令混合与直接换向采样、
PPO γ=0.997/GAE λ=0.97、六维动作及执行器限值均保持 v10 设定。
没有教师、残差、课程、退火、动作滤波、手工姿态补偿或硬关节锁定。

## 检查与局限

65 项测试通过，覆盖连杆伸出测量、世界姿态/关节整圈旋转不变性、软成本、左右互换对称性、
继承契约和动作独立性。使用同一旧 PPO 策略、相同动作、相同随机种子审计 v10/v11，
观测及 qpos 最大差为 0；这证明修改目标未悄悄改变物理或策略输入，不代表 v11 已训练达标。

固定旧轨迹中，正常 +1.5m/s 阶段新增连杆伸出成本为零；其余新增姿态成本约为旧正任务奖励
的 16.9%。错误的后退姿态及换向伸出受到更明显惩罚。系数仍需训练后复验，不能保证不影响速度。
审计结果为 `logs/joint_tracking_v11_frozen_reward_audit.json`，可用下列脚本重现：

```bash
conda run --no-capture-output -n sim python scripts/audit_compact_support.py \
  checkpoints/joint_tracking_pos_v10/checkpoint \
  --output logs/joint_tracking_v11_frozen_reward_audit.json
```

已做短续训数据流检查，未启动正式长训练，也没有将短训 checkpoint 作为合格策略。
终端继续保持精简；连杆伸出峰值和成本写入 TensorBoard，前后高度偏差及 pitch 均值保留分组记录。

## 续训

v11 的观测、动作和执行器与 v10 相同，可继承已训练的 v10；
新奖励重置奖励归一化，保留已学到的六维探索标准差并继续由 PPO 学习。
使用较低且固定的学习率做姿态优化，保留原有高速能力是验收条件。

```bash
cd /home/p/下载/WheelLeg
mkdir -p logs
bash scripts/train_joint_quiet_v23.sh \
  --checkpoint checkpoints/joint_tracking_pos_v10/checkpoint \
  --reset-vecnormalize --preserve-action-std \
  --learning-rate 0.00003 --updates 3000 \
  2>&1 | tee logs/joint_tracking_pos_v11.log
```

模型保存到 `checkpoints/joint_tracking_pos_v11/`。不据训练 reward/length 判断姿态合格。
训练后同种子分别比较 ±0.5、±1.5m/s、前后换向，再扩大到 2m/s、多高度和扰动：

```bash
conda run --no-capture-output -n sim python scripts/eval_joint_tracking.py \
  checkpoints/joint_tracking_pos_v11/checkpoint --episodes-per-case 5 \
  --duration 24 --heights .27 --speed 1.5 --cases forward reverse \
  --output logs/joint_tracking_pos_v11_speed15_eval.json

conda run --no-capture-output -n sim python scripts/eval_joint_tracking_transitions.py \
  checkpoints/joint_tracking_pos_v11/checkpoint --profiles reverse05 reverse15 --episodes 3 \
  --output logs/joint_tracking_pos_v11_geometry_reversal_eval.json
```

同时查看实际均速、存活、pitch 均值/RMS/峰值、实际腿长差、roll、前后高度均值和换向连杆伸出峰值。
原生 MuJoCo、侧边栏、顶部屏幕文字及 `7` 站立保持不变。
