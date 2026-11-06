# v4：平衡、跟踪优先，所有目标固定启用

启动脚本现默认 v5；本页保留 v4 记录。复现本页训练时必须显式添加
`--stage joint_tracking_v4 --disable-domain-randomization --updates 5000`，并设置 `VERSION=joint_tracking_pos_v4`。

阶段 `joint_tracking_v4`，默认输出 `checkpoints/joint_tracking_pos_v4/`。
v2/v3 模型、阶段与参数均保留。此次调整不改六维速度/周期位置接口，不更改执行器增益和额定限幅。

## 权重

| 项目 | v3 | v4 |
|---|---:|---:|
| 直立正奖励 | 2.0 | 3.0 |
| vx 跟踪正奖励 | 2.0 | 2.5 |
| yaw 跟踪正奖励 | 1.5 | 2.0 |
| 高度跟踪正奖励 | 2.0 | 2.0 |
| 终止惩罚 | 20 | 30 |
| 零命令位置漂移 | 1.5 | 0.8 |
| 零命令航向漂移 | 0.4 | 0.25 |
| 轮速目标变化 | 0.12 | 0.04 |
| 周期关节目标变化 | 0.08 | 0.015 |
| 归一化实际输出幅值 | 0.15 | 0.05 |
| 归一化实际输出变化 | 0.10 | 0.025 |
| 饱和代价（含关节限幅前需求） | 0.20 | 0.03 |

代价的归一化尺度、零命令锚点、周期角处理和姿态几何约束均沿用 v3。
正奖励每步理论上限从 7.5 增至 9.5，新旧总回报不再同尺度。

所有系数从第一次采样起固定，不随更新轮次、成功率或 episode 长度变化。
漂移项只在零速度任务下产生代价，平滑和饱和项在所有模式下计算；没有关闭后再启用的课程。
没有教师、残差、退火、动作滤波、手写平衡控制、动作变化硬限幅或锁定关节。

实现位置：`StageSpec` 中固定的任务权重与 `RebalancedPrecisionRewards` 中的正则权重。
`precision_rewards_for_stage` 明确选择版本，写出的 checkpoint 契约与 `training_config.json` 记录实际生效参数。

## 为什么这样改

离线权重量级审计使用固定策略与相同轨迹，将同一批状态/动作分别按 v3/v4 权重计分。
例如 v2 最终策略、种子 20000、0.27m 零命令的前 600 步：

| 分组（每步平均） | v3 计分 | v4 计分 |
|---|---:|---:|
| 正向任务收益 | 7.140 | 9.110 |
| 控制代价 | −5.689 | −1.044 |
| 漂移代价 | −1.192 | −0.639 |
| 控制代价 / 正向任务收益 | 79.7% | 11.5% |

这说明 v3 会重罚一段本来能保持平衡的控制轨迹，容易与恢复平衡的动作冲突。
v4 让这些动作保留主要收益，同时对不必要的快速变化和过大输出继续扣分。
审计也观察到 v3 第 2600 轮在三个固定命令下约 99–125 步就失败，控制代价却很低；
低正则代价本身不是成功，不能代替存活和跟踪验收。

[审计数据](../logs/joint_tracking_v4_reward_audit.json)包含 v2 最终、v3 第 1900 和第 2600 轮，
每个模型的站立、前进和旋转三段轨迹。它只校验固定权重的相对量级，不是新策略效果证明，
也未用于教师训练或动作生成。

## 探索和训练

学习率固定 1e-4；熵系数固定 **0.001**；初始动作标准差 **0.06**，之后由 PPO 自己学习。
熵系数不随训练变化，不采用标准差退火。这样给探索保留较小的固定激励，
避免完全关闭熵正则时过早收缩；它仍不能保证不会发生探索不足或输出抖动。

高度、正反 vx、正反 yaw 和组合命令从第一批采样同时训练。
命令范围、模式概率、零命令 4–12 秒及运动 2–5 秒的固定保持时间分布均沿用 v3。

默认从头训练：

```bash
cd /home/p/下载/WheelLeg
bash scripts/train_joint_quiet_v23.sh 2>&1 | tee logs/joint_tracking_pos_v4.log
```

也可以从中期 v3 第 1900 轮继续同一 PPO；它与 v4 的观测参考和动作接口一致。
必须重置奖励归一化统计量，因为奖励定义已经变化。这不是完整效果已经验证的初始化方案：
该中期模型在审计旋转片段仍会失稳。

```bash
VERSION=joint_tracking_pos_v4_resume1900 bash scripts/train_joint_quiet_v23.sh \
  --checkpoint checkpoints/joint_tracking_pos_v3/checkpoint_iter_1900 \
  --reset-vecnormalize \
  2>&1 | tee logs/joint_tracking_pos_v4_resume1900.log
```

默认 5000 次更新、8 环境、每环境 512 步，所有目标始终固定启用。
初始标准差设置仅在启动时应用，不在训练中反复改写。

## 日志与验收

新增分组日志：`Reward/task_mean`、`Reward/hold_cost_mean`、`Reward/control_cost_mean`、
`Reward/control_to_task_ratio`。按本轮全部采样步统计实际奖励贡献，便于发现代价重新主导训练。
原有 `Hold/*`、`Control/*`、跟踪误差、姿态和存活率继续输出。

首先检查存活与跟踪是否恢复，再同时评估 5cm/5° 零命令站定、输出变化和饱和。
这是评估优先顺序，训练目标从第一步就全部启用，不是分阶段课程。
不能用低饱和、低变化但很快倒下的策略通过验收，也不能只比较新旧 reward 总数。

已完成 16 项行为/契约检查、20 次更新的从头 PPO 短跑、两次更新的 v3 续训检查、
中途及最终 checkpoint 保存，以及独立固定/随机命令评估入口验证。
这些检查只证明训练链路和版本参数工作，短跑模型不具备合格平衡，正式长训练尚未启动。

训练完成后：

```bash
conda run --no-capture-output -n sim python scripts/eval_joint_tracking.py \
  checkpoints/joint_tracking_pos_v4/checkpoint --episodes-per-case 10 \
  --output logs/joint_tracking_pos_v4_eval.json

conda run --no-capture-output -n sim python scripts/eval_joint_tracking_random.py \
  checkpoints/joint_tracking_pos_v4/checkpoint --episodes 10 \
  --output logs/joint_tracking_pos_v4_random_eval.json

bash scripts/replay_joint_tracking.sh --keyboard \
  --checkpoint checkpoints/joint_tracking_pos_v4/checkpoint
```
