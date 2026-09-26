# WheelLeg · UZ-05 轮腿机器人强化学习

这是 UZ-05 两轮平衡轮腿机器人的 MuJoCo + Stable-Baselines3 PPO 项目。
当前版本采用开源轮腿项目的直接目标接口：PPO 全权输出四个腿关节目标和左右轮速目标。

## 控制链

```text
PPO action[0:2]  -> 左右轮绝对角速度目标 -> 轮速闭环 -> 电流/力矩
PPO action[2:6]  -> 四个腿关节绝对位置目标 -> 位置/速度 PD -> 关节力矩
```

代码中没有高层动作生成、动作残差、教师动作、通道混合或额外姿态控制路径。
站立和平移都由同一个 PPO 策略从观测、奖励和动力学中学习，便于直接迁移到实物控制周期。

## 快速开始

```bash
cd /home/p/下载/WheelLeg

# 从零训练站立策略
conda run --no-capture-output -n sim python scripts/train_uz05.py \
  --stage stand --version ppo_direct_stand_v1 --updates 800

# 直接评估
conda run --no-capture-output -n sim python scripts/eval_uz05.py \
  --checkpoint checkpoints/ppo_direct_stand_v1/checkpoint \
  --stage stand --episodes 20

# 在站立 checkpoint 上训练低速平移
conda run --no-capture-output -n sim python scripts/train_low_speed_curriculum.py \
  --base-checkpoint checkpoints/ppo_direct_stand_v1/checkpoint

# 本地回放。指定 checkpoint 时使用 PPO；不指定时使用手动轮速目标
MUJOCO_GL=egl conda run --no-capture-output -n sim \
  python -u scripts/web_replay_uz05.py \
  --checkpoint checkpoints/ppo_direct_stand_v1/checkpoint
```

## 固定接口

| 项目 | 维度/频率 | 定义 |
|---|---:|---|
| actor 观测 | 38 | IMU、速度估计、腿/轮编码器、命令、上一动作、模式 |
| critic 特权观测 | 56 | 真值速度、相对位移、接触、地形、力矩、域参数 |
| 总观测 | 94 | 非对称 actor-critic |
| PPO 动作 | 6 | 左轮速、右轮速、4 个腿关节绝对位置目标 |
| PPO推理 | 125 Hz | ONNX部署，每次输出保持4个底层周期 |
| 电机底层控制 | 500 Hz | 每2 ms刷新DM MIT PD和C620轮速PI |

动作缩放统一定义在 `scripts/uz05/spec.py`：轮速目标为 ±8 rad/s，腿关节绝对位置目标为
±1.2 rad。动作不会按训练阶段缩小，也不会被其他策略或控制器覆盖。

第一阶段 `stand` 固定腿长目标为 `0.18413 m`、车身高度目标为 `0.23825 m`，
线速度和偏航速度命令均为 0；该阶段不进行腿长/高度切换，只学习小漂移下的独立平衡。

## 训练结构

训练按 `stand → low_speed → high_speed → steering → rotation → airborne → stairs → jump → recovery`
推进。奖励使用姿态、腿长、速度跟踪、位移误差、动作平滑、滑移、功率和安全终止项。
训练末段可用 `--deployment-mode` 注入观测/执行延迟及轮端转速-力矩包络。

核心代码：

```text
scripts/uz05/spec.py       接口、观测、动作、奖励和域随机化
scripts/uz05/actuators.py  腿 PD 与轮速闭环
scripts/uz05/env.py        MuJoCo 环境、观测、奖励和终止
scripts/train_uz05.py      PPO 训练入口
scripts/eval_uz05.py       确定性评估入口
scripts/web_replay_uz05.py 本地网页回放
```

几何与执行器参数仍需使用实物测量数据校准；仿真通过不代表已经完成实物验收。

## ONNX部署

实机只使用确定性actor的38维可用观测，不加载SB3或critic：

```bash
conda run --no-capture-output -n sim python scripts/export_onnx.py \
  --checkpoint checkpoints/ppo_direct_stand_fixed0184_v4/checkpoint \
  --output deployment/models/stand.onnx

conda run --no-capture-output -n sim python scripts/run_onnx_policy.py \
  --model deployment/models/stand.onnx

conda run --no-capture-output -n sim python scripts/eval_uz05.py \
  --onnx deployment/models/stand.onnx --stage stand --deployment-mode
```

部署依赖见 `requirements-deploy.txt`。协议边界、未实测阻塞项和CAN参考打包分别见
`deployment/uz05_interface.json`、`deployment/README.md` 与
`scripts/uz05/deployment.py`。在上实物前可运行：

```bash
conda run --no-capture-output -n sim python scripts/audit_checkpoint_actuators.py \
  --checkpoint checkpoints/ppo_direct_stand_fixed0184_v4/checkpoint \
  --deployment-mode --episodes 20
```
