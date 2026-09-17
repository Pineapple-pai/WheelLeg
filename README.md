# WheelLeg · UZ-05 轮腿机器人强化学习

UZ-05 两轮平衡轮腿机器人的训练项目。仿真用 **MuJoCo**（`diagnostics/uz05.xml`），
算法用 **Stable-Baselines3 PPO**，跑在 conda 环境 `sim` 里。

> 历史说明：仓库曾包含一套基于 Isaac Sim / OmniDrones 的旧轮腿实现
> （`omni_drones/`、`cfg/`、`tools/`、`setup.py`），已于 2026-09 移除，只保留 MuJoCo 版。
> 需要找回旧代码：`git checkout HEAD -- omni_drones cfg tools setup.py`
> （`_deps/`、`conda_setup/` 未纳入 git，备份在 `/home/p/WheelLeg_isaac_backup_20260917/`）。

## 快速开始

```bash
cd /home/p/下载/WheelLeg

# 站立三级课程（辅助 1.0→0.6 → 0.6→0.2 → 0.2→0.0，逐级继承，最后一级验收 ±5cm）
NUM_ENVS=8 TAG=v1 bash scripts/train_uz05_all.sh stand \
  > logs/train_stand_v1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

tail -35 logs/train_stand_v1_*.log     # 每行一个量
pkill -f train_uz0[5]                  # 停止
```

后续阶段（自动继承上一阶段 checkpoint，无需任何迁移）：

```bash
TAG=v1 bash scripts/train_uz05_all.sh low_speed   # 单阶段
TAG=v1 bash scripts/train_uz05_all.sh             # 全部 9 个阶段
```

## 目录结构

```
scripts/
  train_uz05.py                 训练入口（PPO + 非对称 actor-critic + 辅助退火 + 日志）
  train_uz05_all.sh             分阶段课程脚本（自动串阶段、自动继承）
  uz05/
    spec.py                     ★ 单一事实来源：接口/奖励/执行器/课程/域随机化
    model.py                    MJCF 加载、索引、接触、地形、域随机化写入
    actuators.py                关节 PD + 轮速→电调→电流→力矩
    env.py                      环境：观测、奖励、终止、课程
  validate_uz05_deployment_data.py   实机实测数据校验（等电调/电机数据）
diagnostics/uz05.xml            机器人 MJCF（含 29 个纯显示网格，加载时自动裁剪）
model_source/uz05/              SolidWorks 模型源（腿 / 底盘 / 总装 / 修复版）
calibration/                    平衡控制器标定原始数据
docs/uz05_refactor.md           ★ 重构说明：接口、坐标系、执行器、对齐开源、资源上限
checkpoints/<version>/          每次训练一个独立文件夹（策略 + 归一化统计 + 契约）
runs/<version>/                 tensorboard
logs/                           终端输出留存
```

## 接口（所有阶段固定，因此 checkpoint 永久可继承）

| | 维度 | 内容 |
|---|---|---|
| 观测 | **226** | actor 170（34 维/帧 × 历史 5 帧，实机全可测） + 特权 56（仅 critic） |
| 动作 | **6** | 轮速共模 ×8.0、轮速差模 ×6.0、4 个腿关节位置偏置 ×0.35 |
| 控制 | 120 Hz | 关节 = 位置+速度+PD；轮 = 速度目标 → 电调 → 电流 → 力矩 |

坐标系全部**机器人坐标系**（重力投影 / 机体角速度 / 机体前向速度），
不使用任何世界系绝对量，与实机陀螺仪一致。

## 训练阶段

`stand → low_speed → high_speed → steering → rotation → airborne → stairs → jump → recovery`

前 5 个阶段用**官方 StackForce 16 项奖励**（权重逐项照抄，可与开源曲线直接对比），
后 4 个阶段追加自研塑形项。详见 [docs/uz05_refactor.md](docs/uz05_refactor.md) §7–§8。

**验收（站立 S2，三者同时满足）**：`survive_rate ≥ 0.95` 且 `drift_tail_cm < 5` 且 `assist_scale = 0`
（终端直接打印 `accept_5cm: 1`）。硬约束：**不允许锁死姿态来控制姿态**，
平衡必须由策略协调关节与轮子实现，辅助控制器必须退火到 0。

## 资源上限（别踩，之前把整机跑死过）

- subproc 环境 = 每环境一个独立进程，`--num-envs` 默认 8，代码里自动夹到 `min(核数/2, 8)`
- 线程数统一为 1（否则 8 进程 × 16 线程会抢满 16 核）
- MJCF 里的高模显示网格在加载时被裁掉（单进程 2.0 GB → 37 MB），要渲染时设 `UZ05_VISUAL=1`

## 文档

- [重构说明与训练配置](docs/uz05_refactor.md) —— 接口、坐标系、执行器、开源对齐、资源上限
- [模型与实物部署审计](docs/model_deployment_audit.md) —— 机械/驱动事实（Isaac 时期，路径已失效）
- [修改建议书](修改.md) —— 待办与设计参考

## 待补的实测数据

`scripts/uz05/spec.py` 里这几处仍是手册值或占位值，等实测数据替换：

| 参数 | 当前值 | 来源 |
|---|---|---|
| `JointActuatorParams.kp / kd` | 100 / 4 | 旧版验证值 |
| `WheelEscParams.speed_kp_a_per_rad_s` | 0.081 | 由旧 C620 增益换算 |
| `RobotParams.leg_length_min / max` | 0.150 / 0.340 | 占位 |
| `RobotParams.hip_joint_limit` | 1.2 rad | 占位 |
| `DomainRandomization.*` | 12 项区间 | 参考开源，需用实测误差替换 |
