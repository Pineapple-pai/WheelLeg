# v3：零命令站定、输出变化和持续饱和

本页保留 v3 历史配置；启动脚本当前默认 [v4](joint_tracking_v4.md)，不会覆盖 v3 模型。

训练阶段 `joint_tracking_v3`，默认输出 `checkpoints/joint_tracking_pos_v3/`。
v1/v2 的奖励和旧模型保留，不覆盖已训练的 v2。
本次完成的是目标实现与训练链路验证，尚未获得训练合格的 v3 模型。

## 固定训练目标

六维 PPO 动作保持为左右轮速度目标和四关节周期绝对位置目标。
执行器参数保持不变：关节位置 PD、轮子速度环、原额定力矩/电流限制。
无教师、残差、动作滤波、手写平衡修正、动作变化硬限幅或学习率/奖励退火。
没有固定关节角度或限制累计关节转数。

1. **零命令站定**：进入零速度命令时记录当前位置与航向。
   从运动切到停止时，以切换位置作为锚点；继续零命令或只改高度不会更新锚点。
   按平面位置误差和包裹至 ±π 的航向误差给 PPO 奖励，运动时关闭这两项。
   actor/critic 的现有相对位置块与 actor 航向块同时使用这个参考，确保误差可观测。
   世界坐标不需要回到每回合起点，切换时也不修改机器人位置、速度或动作。
2. **输出变化**：分别惩罚真实单位的相邻轮速目标变化、周期关节位置目标变化，
   以及归一化实际关节力矩/轮电流变化；首步不人为当作从零目标跳变。
   关节跨 ±π 或相差整圈的等价目标不会产生虚假的大变化惩罚。
3. **持续饱和**：从额定输出的 80% 开始计入惩罚，每步累计。
   关节使用限幅前 PD 力矩需求，因此 20 Nm 和 60 Nm 的请求即使都被执行器限到
   20 Nm，也会受到不同程度惩罚；同时提高实际输出幅值惩罚。

参数集中在 `scripts/uz05/spec.py` 的 `PrecisionRewards`，并写入配置和 checkpoint 契约。
位置惩罚有 1 cm 死区、5 cm 归一化尺度，航向尺度为 5°。
轮速变化尺度为 2 rad/s，关节目标变化尺度为 0.08 rad；它们是奖励归一化尺度，**不是输出硬上限**。
大误差使用二次到线性的连续代价，保留极端值的区分，不将大请求全部压成同一个奖励。

## 采样与初始化

高度、正反 vx、正反 yaw 和组合命令仍从第一次采样同时进入训练，命令范围与 v2 一致。
模式采样概率仍各 25%；v3 的零命令保持时间固定分布为 4–12 秒，其余为 2–5 秒，
增加长时间站定样本。它不随迭代进度变化。

默认从随机 PPO 权重训练，初始动作标准差 0.05、熵系数固定 0、学习率固定 1e-4。
标准差仍是 PPO 可学习参数，不使用人工标准差退火。
v3 的位置/航向参考语义已改变，契约会拒绝直接将 v2 checkpoint 当作 v3 续训。
这避免不明确的接口语义迁移；v2 仍可用其原阶段回放。

## 启动

```bash
cd /home/p/下载/WheelLeg
bash scripts/train_joint_quiet_v23.sh 2>&1 | tee logs/joint_tracking_pos_v3.log
```

默认 5000 次更新、8 环境、每环境 512 步，共 20,480,000 个采样步。
每 100 次更新保存策略、归一化统计量和契约。
日志增加 `Hold/*` 与 `Control/*`：平面漂移、航向误差、5 cm 内采样比例、轮速/关节目标变化、
实际输出变化，以及关节/轮子饱和比例。它们是本轮采样的统计，不只打印最后一帧。

## 验收

先与 v2 确定性基线对比：零命令最坏前后漂移 32.4 cm；随机变命令轮速变化 RMS 7.22 rad/s；
关节饱和比例约 82%–86%。新评估同时记录二维漂移和航向，不能仅比较总 reward。
新旧奖励定义不同，回报大小不可直接作为性能增减证据。

本轮目标建议：12 秒零命令平面漂移 ≤5 cm、航向误差 ≤5°；轮速目标变化 RMS <2 rad/s；
关节饱和采样比例 <30%；同时保留姿态、速度跟踪和至少 95% 存活率。
这些是验收目标，尚未通过训练验证可同时达到。需要覆盖固定命令、随机变命令及速度范围边界。

```bash
conda run --no-capture-output -n sim python scripts/eval_joint_tracking.py \
  checkpoints/joint_tracking_pos_v3/checkpoint --episodes-per-case 10 \
  --output logs/joint_tracking_pos_v3_eval.json

conda run --no-capture-output -n sim python scripts/eval_joint_tracking_random.py \
  checkpoints/joint_tracking_pos_v3/checkpoint --episodes 10 \
  --output logs/joint_tracking_pos_v3_random_eval.json

bash scripts/replay_joint_tracking.sh --keyboard \
  --checkpoint checkpoints/joint_tracking_pos_v3/checkpoint
```

已通过 14 项行为/契约测试、20 轮 PPO 短跑、checkpoint 保存与两种评估入口。
短跑仅为程序验证，样本策略仍会倒下；低输出变化和低饱和必须与存活及跟踪同时评估，
不能把不动作但倒下的模型视为改进。
