# WheelLeg 文档

本仓库当前只保留轮腿训练项目。主要说明见根目录 `README.md`。`TwoWheelBalance` 保留两轮固定腿基线；`TwoWheelClosedLoop` 使用左右轮和 `LL/Lcrank/RL/Rcrank` 四个主动腿轴。

轮腿相关代码位置：

- `omni_drones/envs/twowheel/balance.py`
- `omni_drones/robots/twowheel.py`
- `omni_drones/robots/assets/twowheel_uz05/`
- `cfg/task/TwoWheel*.yaml`
- `scripts/train.py`
- `scripts/play.py`
- `scripts/diagnose_twowheel.py`
- `scripts/audit_twowheel_model.py`：只读检查 URDF/USD 的关节、驱动、碰撞与部署阻塞项。
- [模型与实物部署审计](model_deployment_audit.md)
- `scripts/sweep_twowheel_stand.py`
- `scripts/sweep_uz05_closed_loop.py`

最小训练命令：

```bash
conda activate sim
pip install -e .
python scripts/train.py task=TwoWheelBalance
```

当前 `TwoWheelClosedLoopV5` 在同一次训练中联合学习站立、前后平移和 yaw：

```bash
conda run --no-capture-output -n sim python -u scripts/train.py headless=true wandb.mode=disabled task=TwoWheelClosedLoopV5 algo=ppo task.env.num_envs=512 max_iters=3000 total_frames=49152000 eval_interval=1000 save_interval=1000 checkpoint_version=v2 algo.initial_std=0.03 algo.min_std=0.005 algo.max_std=0.08 algo.actor_lr=0.00005 algo.critic_lr=0.0003
```

V5 是 10 维动作：2 个轮速目标、4 个主动关节位置目标和 4 个关节速度目标。命令使用分层采样，固定保留 `30%` 精确零命令，并按训练帧数扩大速度范围。

USD 设置要点：

- 轮关节 `Lwhl/Rwhl` 不设角度限位，保持可连续旋转。
- 两轮转轴沿世界 `X`，实际前进方向为世界 `Y`，前后平衡角为 `roll`。
- 轮速 drive 使用零 stiffness、有限 damping 和最大力矩。
- `LL/Lcrank/RL/Rcrank` 和闭链被动轴不设有限角度限位；`Yaw/P/P12/P23` 保留临时有限限位。
- 原始 URDF 仍是开链装配树；`cad_repaired.usda` 使用四条被动 PhysX 回边表示左右闭链。
- `TwoWheelClosedLoop` 只驱动 `LL/Lcrank/RL/Rcrank`，不锁死闭链内部关节。
