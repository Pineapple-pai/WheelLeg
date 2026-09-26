# UZ-05 reward 与 sim2real 训练策略

参考项目：[scutrobotlab/wheeled-legged_RL](https://github.com/scutrobotlab/wheeled-legged_RL)。

## 结论

可以借鉴开源项目的 reward 形式、课程和延迟注入，但不能直接复制权重。
两台轮腿车的质量、轮半径、减速比、电流环、关节极性、控制频率和碰撞形状都不同，
reward 系数的数值没有可迁移性。本工程只迁移以下原则：

- 主任务用稠密状态奖励：速度/高度/姿态的平滑误差函数，不只奖励终点。
- 动作幅值、变化率、功率、滑移和关节限位只做次级约束，不能压过任务奖励。
- 终止惩罚显著大于单步最坏惩罚，防止“主动摔倒止损”。
- 训练按能力门控推进，不按固定 iteration 盲目升级。
- actor 只看实物可得信号；critic 可在训练时看仿真特权量。
- 训练末段注入观测延迟、动作延迟和扭矩-转速包络。

## Reward 层级

1. **生存与安全**：`alive` / `termination` / 关节限位 / 腿长限位 / 非期望接触。
2. **主任务**：站立/零命令保持的 `upright + station`，平移的 `track_vx`，变高的 `leg_length`。站立漂移奖励和课程漂移验收只用于零命令回合；移动回合用速度误差、方向和存活验收。
3. **连续信用分配**：`upright_progress` / `track_vx_progress` / `leg_length_progress`。
4. **实物可执行性**：轮腿滑移、电流/功率、动作一阶与二阶差分、关节速度。

训练全过程只根据速度、位移、姿态、轮滑和执行器状态等结果计算回报；
部署模式只负责施加真实执行器约束，不改变 PPO 的动作权限或控制链路。

## 训练顺序

1. 在标称模型上学会站立，再按速度范围学低速平移。
2. 每一级只在存活率、姿态和正/反向跟踪同时达标时推进。
3. 名义能力通过后，用 `train_sim2real_curriculum.py` 分三段打开执行器包络与延迟。
4. 最后 checkpoint 必须在同一 deployment profile 下做确定性评估，不能用零延迟评估代替。
5. 上车顺序为悬空轮、支撑架、限流低权限、全权限；每级都要有独立急停。

```bash
# 名义能力通过后，用实测的控制周期延迟进行三段微调
conda run --no-capture-output -n sim python \
  scripts/train_sim2real_curriculum.py \
  --checkpoint checkpoints/<accepted>/checkpoint \
  --stage low_speed \
  --final-observation-delay 1 3 \
  --final-actuator-delay 0 1
```

## 实物数据门槛

以下数据没有实测前，只能说“鲁棒仿真”，不能宣称“已贴近实物”：

- IMU 和编码器从采样到 policy 输入的 p50/p95 延迟与噪声频谱。
- policy 输出到电流/位置响应的 p50/p95 延迟。
- 轮电机电流-轮端扭矩-转速曲线，包括死区、饱和和正反向差异。
- 四个腿关节的位置阶跃响应、摩擦/死区、连续与峰值扭矩。
- 整机质量、质心和惯量，轮胎纵/横向摩擦，轮半径及负载变形。

域随机化区间应覆盖上述测量的置信区间，而不是越大越好。
