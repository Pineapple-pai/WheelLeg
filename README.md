# WheelLeg

这是一个从 OmniDrones 裁剪出来的轮腿机器人强化学习训练项目。当前仓库只保留轮腿训练需要的环境、机器人模型、资产、算法和通用工具；无人机、吊载、平台、多机编队等无关任务已经移除。

## 轮腿代码位置

核心代码在这些位置：

- `omni_drones/envs/twowheel/balance.py`：轮腿强化学习环境 `TwoWheelBalance`，包括观测、奖励、reset、终止条件、评估命令等。
- `omni_drones/robots/twowheel.py`：轮腿机器人封装 `TwoWheelRobot`，包括 USD 加载、关节索引、动作映射、轮速/腿部目标控制。
- `omni_drones/robots/assets/twowheel_uz05/`：轮腿机器人资产，包含 USD、mesh、URDF/ROS 辅助文件。
- `cfg/task/TwoWheel*.yaml`：轮腿任务配置。不同文件对应不同奖励、初始化、鲁棒性和站立/移动实验版本。
- `scripts/train.py`：训练入口。
- `scripts/play.py`：加载 checkpoint 播放/评估入口。
- `scripts/diagnose_twowheel.py`、`scripts/sweep_twowheel_stand.py`：轮腿诊断和参数扫描脚本。
- `scripts/sweep_uz05_closed_loop.py`：四个主动轴的无重力渐进闭链扫描。
- `omni_drones/learning/`：PPO、MAPPO、SAC、TD3 等学习算法实现。
- `omni_drones/utils/`、`omni_drones/views/`、`omni_drones/envs/isaac_env.py`：Isaac Sim、TorchRL 和环境基类所需公共代码。

## 环境要求

项目仍依赖 Isaac Sim、PyTorch/TorchRL、Hydra、WandB 等原环境。不要删除或重建已有 conda 环境；本次清理只处理仓库文件，不卸载环境包。

如果已有环境名是 `sim`：

```bash
conda activate sim
pip install -e .
```

如需让 Isaac Sim 使用指定安装路径，可在环境变量中设置：

```bash
export ISAACSIM_PATH=/path/to/isaacsim
```

## 训练

默认训练任务已经改为 `TwoWheelBalance`。

从仓库根目录运行：

```bash
python scripts/train.py
```

常用覆盖参数：

```bash
python scripts/train.py headless=true task=TwoWheelStandQuietRobustV19 total_frames=150000000
python scripts/train.py headless=false task=TwoWheelBalance env.num_envs=64
```

## 站立与移动联合课程

`TwoWheelBalance` 仅保留为固定腿、2 维轮动作回归基线。`TwoWheelClosedLoopV5`
在同一策略和同一次训练中联合学习站立、前进、后退和转向，不需要人工切换站立/平移 checkpoint。

V5 的联合命令分布：

- `30%` 精确零命令，`50%` 纯前后平移，`10%` 原地 yaw，`10%` 平移加 yaw。
- 正负命令按区间分层采样，避免普通均匀随机造成某个速度段样本偏多。
- 所有命令类别从第 1 轮就开启，仅速度幅值随训练从 `0.08 m/s` 扩大到 `0.50 m/s`。
- 命令每 `4 s` 重采样；观测始终包含 `vx_cmd` 和 `yaw_rate_cmd`。
- 零速时启用出生位置保持，平移时只约束侧向偏移，不把车拉回原点。

checkpoint 默认保存到当前项目的 `checkpoints/<checkpoint_version>/`。也可以用环境变量改保存位置：

```bash
export WHEELLEG_CHECKPOINT_ROOT=/path/to/checkpoints
python3 scripts/train.py checkpoint_version=v1
```

## 四关节闭链训练

CAD 修复模型的四个主动腿轴为 `[LL, Lcrank, RL, Rcrank]`，而不是闭链内部的 `L12/L45/R12/R45`。V5 使用 `2` 个轮速目标、`4` 个关节位置目标和 `4` 个关节速度目标，共 `10` 维动作；其他腿关节由 PhysX 闭环约束被动求解。轮速 drive 的输出力矩上限仍是 `2.46 Nm`，关节 drive 上限仍是 `20 Nm`。

`TwoWheelClosedLoopV2/V3` 曾收敛到约 `0.052 m` 的低车身局部最优。动作到高度扫描后确认，零位本身就是合理站姿。`TwoWheelClosedLoopV4` 改为可部署的残差架构：额定轮端力矩 PD 负责安全平衡，PPO 学习小幅轮端残差和四关节动作。

训练前先运行小角度闭链扫描：

```bash
conda run --no-capture-output -n sim python -u scripts/sweep_uz05_closed_loop.py \
  headless=true \
  wandb.mode=disabled \
  task=TwoWheelClosedLoop
```

从头训练第一版站立策略并保存为 `v1`：

```bash
conda run --no-capture-output -n sim python -u scripts/train.py \
  headless=true \
  wandb.mode=disabled \
  task=TwoWheelClosedLoopV4 \
  task.train_stage=stand \
  algo=ppo \
  task.env.num_envs=512 \
  max_iters=3000 \
  total_frames=49152000 \
  eval_interval=1000 \
  save_interval=1000 \
  checkpoint_version=v1 \
  algo.initial_std=0.02 \
  algo.min_std=0.005 \
  algo.max_std=0.05 \
  algo.actor_lr=0.00005 \
  algo.critic_lr=0.0003
```

站立阶段每 `1000` 轮同时输出 `eval_stand` 和 `eval_robust`。两者都必须满足 `episode_len=999`、`roll_error < 0.05`、`height_error < 0.005`、`abs(forward_velocity) < 0.02`，且轮端基线动作不能长期饱和，才能进入平移课程。平移配置还需要单独放开残差权限、去掉站立位置保持项并重新验收，当前不直接复用旧 V3 平移命令。

联合训练 V5（从头训练，checkpoint 保存为 `v2`）：

```bash
conda run --no-capture-output -n sim python -u scripts/train.py \
  headless=true \
  wandb.mode=disabled \
  task=TwoWheelClosedLoopV5 \
  algo=ppo \
  task.env.num_envs=512 \
  max_iters=3000 \
  total_frames=49152000 \
  eval_interval=1000 \
  save_interval=1000 \
  checkpoint_version=v2 \
  algo.initial_std=0.03 \
  algo.min_std=0.005 \
  algo.max_std=0.08 \
  algo.actor_lr=0.00005 \
  algo.critic_lr=0.0003
```

每 `1000` 轮会独立输出 `eval_stand`、`eval_forward`、`eval_backward`、`eval_turn` 和 `eval_robust`。`state.command_abs_vx` 和 `state.command_abs_yaw_rate` 是当前 batch 的命令绝对值均值，`state.command_vx_min/max` 是当前课程范围。V5 动作维度已变为 `10`，不能直接加载 V4 的 6 维 actor checkpoint。

闭链奖励配置借鉴两个开源项目：使用指数速度跟踪、姿态和高度门控、零指令静止约束、左右腿对称、动作一阶/二阶差分以及执行器负载代理。平移阶段有 `25%` 的零速样本，姿态退火下限为 `35%`，用于避免平移微调遗忘站立。跳跃任务仍未开放：可靠腾空/接触检测、落地冲击约束和实物峰值持续时间尚未完成，现在迁移跳跃奖励会产生错误监督。

## 播放和评估

```bash
conda run --no-capture-output -n sim python -u scripts/play.py task=TwoWheelClosedLoopV4 checkpoint_version=v1 headless=false
```

如果 `algo.checkpoint_path` 没有显式指定，`scripts/play.py` 会从 `checkpoints/<checkpoint_version>/` 自动选择最新 checkpoint。

## 保留和删除原则

已保留：

- 轮腿环境、轮腿机器人封装、轮腿资产。
- 轮腿任务配置 `cfg/task/TwoWheel*.yaml`。
- 训练、播放、诊断、轮腿扫描脚本。
- 强化学习算法和 Isaac/TorchRL 公共工具。
- `setup.py`、`conda_setup/`、`cfg/base/`、`cfg/algo/` 等训练环境需要的文件。

已移除：

- 无人机、吊载、平台、运输、倒立摆、编队、森林、弹球等非轮腿任务环境和配置。
- 无人机资产、控制器、传感器、示例脚本、论文实验脚本和原 OmniDrones 文档素材。

## 备注

`cfg/task/TwoWheel*.yaml` 中的机器人资产路径已使用仓库内资产，不再依赖旧目录 `/home/p/RAL26/OmniDrones`。

当前 USD/关节设置：

- `Lwhl`、`Rwhl` 保持无限旋转，用于轮速控制。
- USD 中两轮轴均沿世界 `+X`，所以实际前进方向是世界 `Y`，平衡角是绕 `X` 的 `roll`。
- V4 保留 `[-2.46, 2.46] Nm` 直接力矩动作作为回归基线；V5 按实物接口改为轮速目标，但 PhysX drive 的输出力矩仍限制为 `2.46 Nm`。
- `TwoWheelBalance` 当前只暴露两个轮动作，其余关节固定在 URDF 零位。
- Isaac Sim 4.0 的 articulation tensor view 不提供可靠的逐 link 接触力；直立和倒地高度范围又有重叠，因此终止主要使用 roll/pitch 和最低高度，不启用 `body_contact_height` 代理。
- CAD 零位轮胎支撑高度为 `0.08917 m`，V4 reset 使用该值，稳态 base 高度约 `0.0864 m`。
- 旧基线其它 revolute 关节 authored 为 `[-1.57, 1.57] deg`，并不是预期的 `rad` 宽限位。新的 `cad_repaired.usda` 将四个连续主动轴及闭链被动轴设为无限转动；辅助轴仍使用临时 `[-90, 90] deg`。连续关节仍需实测软件安全工作区，不能把无限转动直接作为训练动作范围。
- 所有轮腿任务配置都显式设置了腿部 drive 和被动关节 drive，不再隐式依赖 USD 中导出的超大默认 stiffness。
- `lock_passive_joints: true` 会把非受控关节锁在 reset 时的初始位置，减少训练时机构自由漂移。

原始模型位于 `/home/p/下载/UZ-05-open总装11.57/urdf/UZ-05-open总装11.57.urdf`。它由 SolidWorks 自动导出，包含错误固定的 `Rcrank` 和一个无法表达机械闭环的开链树。左右轮刚体原点不同，但计入 STL 局部偏置后的轮胎几何中心实际对称，不能使用 `balanced.usd` 的简单镜像补丁。后续腿部训练仍需依据实物电机编码器零位和闭环约束建立简化运动学模型。
