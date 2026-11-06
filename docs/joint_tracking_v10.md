# v10：平移中的高度/pitch 一致性与持续平衡

启动脚本现默认 v11，见 [v11 实测、修改与续训](joint_tracking_v11.md)。
复现本页训练需显式指定 `--stage joint_tracking_v10` 并设置独立 `VERSION`。

## v9 复现

名义动力学、无推力扰动、保留观测噪声、高度目标 0.27m，种子 20000/20001。
固定速度测试每回合最多 24 秒，统计稳态时排除首秒：

| 命令 | 实际 vx 均值 | 跑满 24 秒 | pitch RMS |
|---|---|---|---|
| +0.5m/s | +0.487m/s | 2/2 | 2.50° |
| −0.5m/s | −0.480m/s | 2/2 | 2.21° |
| +0.9m/s | +0.911m/s | 0/2 | 8.83° |
| −0.9m/s | −0.917m/s | 0/2 | 3.93° |

0.5m/s 时累计偏航峰值分别约 2.19°/0.64°，v9 的低速跟踪及航向控制已有实际改善。
0.9m/s 已能达到目标速度，但前进平均仅持续 4.26s、后退 9.19s，说明持续平衡尚未达标。
该组轮输出饱和占比低于 0.5%，不能将问题简单归结为轮速范围不足。

固定同高度换向协议：站立 3s→前进 0.5m/s 5s→后退 0.5m/s 5s→前进 5s→站立。
末秒前进高度约 0.258m、pitch −2.5°，后退高度约 0.287m、pitch +2.2°。
目标高度相同，却形成了约 3cm 的方向相关高度偏差；后退切前进时高度最低约 0.229m。
加速至 ±0.9m/s 后再换向的两个协议中，两种开始方向均未跑满 24s。

原始结果：

- `logs/joint_tracking_pos_v9_speed05_diagnosis.json`
- `logs/joint_tracking_pos_v9_speed09_diagnosis.json`
- `logs/joint_tracking_pos_v9_reversal_diagnosis.json`

有限种子的测试只用于复现，不能作为全局成功率。前后运动可有短暂不同的加减速倾角；
训练目标是相同高度命令得到相近实际高度，稳态 pitch 靠近零，而非强迫两种方向的关节角完全相同。

## 固定修改

1. **独立的高度/pitch 联合软成本。** 定义
   `Q=0.30+0.70 exp(−0.5[(高度误差/0.015)²+(pitch/7°)²])`，增加 `−5(1−Q)` 成本，
   幅度有界于 [−3.5,0]。保留 v9 原有速度正奖励及几何/站位质量约束，
   不按 Q 缩减速度奖励，恢复时朝目标速度改善的奖励信号仍然保留。
2. **双向高度误差成本。** 新增 `−0.4 H(高度误差/0.01)`，同时惩罚偏高和偏低；
   下沉成本权重 0.5→1.0、尺度 1.5cm→1cm，强化严重下沉的成本。
   保留原有高度跟踪、升沉速度和腿部软目标，不直接规定腿角或腿长。
3. **固定增加直接换向样本。** 非首帧命令重采样时，若上一命令是 |vx|>0.1m/s 的纯平移，
   固定 25% 概率采样同高度、同幅值、反向 vx，保持 1–4 秒。
   其他样本仍覆盖完整 ±2m/s、0.22–0.32m 及 yaw 分布，从第一批环境启用。
   只是用户命令分布，不指定动作，也不做命令斜坡或自动恢复控制。
4. **固定 PPO 收益视野。** γ=0.99→0.997，GAE λ=0.95→0.97。
   125Hz 下，4 秒后回报的折扣由约 0.66% 变为 22.3%，使迟发失稳更有机会影响当前策略。
   这说明折扣差异，不证明它是全部失稳原因。PPO、rollout buffer、VecNormalize 的 γ 同步设置。

H 为标准 Huber 软成本。解析检查见 `logs/joint_tracking_v10_objective_audit.json`，
不代表策略性能。全部系数固定，无课程、退火、教师、残差、动作滤波或额外平衡控制。
仍是两路独立轮速目标和四路独立圆周关节位置目标，动作范围和执行器额定限值未改变。

## 训练命令

v10 与 v9 的观测含义、六维动作和执行器契约相同，可继承 v9 的 PPO 权重继续训练。
新奖励及折扣使用新的奖励归一化统计；保留已学到的六维探索标准差，之后仍由 PPO 自由学习。

```bash
cd /home/p/下载/WheelLeg
mkdir -p logs
bash scripts/train_joint_quiet_v23.sh \
  --checkpoint checkpoints/joint_tracking_pos_v9/checkpoint \
  --reset-vecnormalize --preserve-action-std \
  --learning-rate 0.00005 --updates 4000 \
  2>&1 | tee logs/joint_tracking_pos_v10.log
```

保存到 `checkpoints/joint_tracking_pos_v10/`，不覆盖 v9。学习率从头到尾固定 5e−5。
若不传 checkpoint，则从头训练 v10；`--preserve-action-std` 必须与 checkpoint 一起使用。
终端保持精简，`Tracking/pose_quality`、新增姿态成本及前后方向分别统计的高度偏差、
pitch 均值、vx 误差细项写入 TensorBoard。

## 验收

```bash
conda run --no-capture-output -n sim python scripts/eval_joint_tracking.py \
  checkpoints/joint_tracking_pos_v10/checkpoint --episodes-per-case 5 \
  --duration 24 --heights .27 --speed .9 --cases forward reverse \
  --output logs/joint_tracking_pos_v10_speed09_eval.json

conda run --no-capture-output -n sim python scripts/eval_joint_tracking_transitions.py \
  checkpoints/joint_tracking_pos_v10/checkpoint \
  --profiles reverse05 reverse09 reverse09_backfirst --episodes 3 \
  --output logs/joint_tracking_pos_v10_reversal_eval.json
```

先对比同种子的 ±0.5、±0.9m/s 和换向，再扩大到 ±1、±2m/s、多高度、随机参数及推力。
同时看存活、实际速度、稳态高度均值/波动、pitch 均值/RMS/峰值及换向最低高度。
不能通过降低实际速度、增加累计偏航或牺牲原有站立来换取“更稳定”的单一指标。
本次测试及短续训结果另记于本页末尾；正式长训练效果待验收。


## 本次验证范围

59 项测试通过，包括：姿态成本双向对称且有界、姿态偏差不削弱原有速度正奖励、
直接换向保持高度、动作不被覆盖、v9 续训兼容及旧版折扣参数不变。
实际续训检查确认初始六维标准差与源 checkpoint 完全一致；保存后的 PPO、rollout buffer、
VecNormalize 的 γ 均为 0.997，GAE λ 为 0.97。
另修正了载入 checkpoint 后继承旧 TensorBoard 路径的问题，新运行的数据写入其自身版本目录。

第一次将速度正奖励乘以姿态质量的 100 轮试跑虽改善了存活，但实际 ±0.9m/s 仅约 ±0.75m/s，
且换向最低高度仍可降至约 0.189m，因此没有采用乘性方案作为最终奖励。
当前采用独立姿态软成本，避免直接削弱速度改善信号。

独立成本方案的首个 100 轮试跑也仅说明可运行及有限种子存活改善：
实际 ±0.9m/s 约 +0.779/−0.762m/s，后退平均高度仍约 0.307m，不能认定速度或高度已达标。
随后继续相同固定配置的短续训，结果以对应 JSON 为准；所有试验 checkpoint 均与正式版本分开保存。
