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

checkpoint 默认保存到当前项目的 `checkpoints/<checkpoint_version>/`。也可以用环境变量改保存位置：

```bash
export WHEELLEG_CHECKPOINT_ROOT=/path/to/checkpoints
python scripts/train.py checkpoint_version=v1
```

## 播放和评估

```bash
python scripts/play.py task=TwoWheelBalance checkpoint_version=v1 headless=false
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
- 其它 revolute 关节已写入宽限位 `[-1.57, 1.57]`，便于 PhysX 和任务代码里的 DOF limit penalty 生效。
- 所有轮腿任务配置都显式设置了腿部 drive 和被动关节 drive，不再隐式依赖 USD 中导出的超大默认 stiffness。
- `lock_passive_joints: true` 会把非受控关节锁在 reset 时的初始位置，减少训练时机构自由漂移。
