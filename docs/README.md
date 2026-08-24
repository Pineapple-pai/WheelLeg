# WheelLeg 文档

本仓库当前只保留轮腿训练项目。主要说明见根目录 `README.md`。当前阶段是固定腿姿的两轮站立和平移训练，策略动作只有左右轮。

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

最小训练命令：

```bash
conda activate sim
pip install -e .
python scripts/train.py task=TwoWheelBalance
```

两阶段训练：

```bash
python -u scripts/train.py headless=true wandb.mode=disabled task=TwoWheelBalance task.env.num_envs=512 max_iters=3000 eval_interval=1000 save_interval=1000 checkpoint_version=v6 algo.initial_std=0.08 algo.min_std=0.01 algo.max_std=0.20
python scripts/train.py task=TwoWheelBalance task.train_stage=move algo.checkpoint_path=checkpoints/v6/checkpoint_final.pt checkpoint_version=v7
```

`v1` 到 `v5` 是旧的 10 维动作 checkpoint，不与当前 2 维轮动作策略兼容。新站立训练从 `v6` 开始。

USD 设置要点：

- 轮关节 `Lwhl/Rwhl` 不设角度限位，保持可连续旋转。
- 两轮转轴沿世界 `X`，实际前进方向为世界 `Y`，前后平衡角为 `roll`。
- 轮速 drive 使用零 stiffness、有限 damping 和最大力矩。
- 非轮 revolute 关节设有 `[-1.57, 1.57]` 宽限位。
- 原始 URDF 是 SolidWorks 导出的开链装配树，当前锁定所有非轮关节；闭环腿机构完成实物标定前不参与策略动作。
