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

# ★ 站立验收（零指令站定）：腿 + 轮协同平衡控制器，不需要 checkpoint
conda run -n sim python scripts/accept_stand.py --episodes 5 --steps 2000 --episode-steps 2000
#   → pitch RMS ≈ 0.15°、峰峰 ≈ 0.9°、pitch 速率 0.009 rad/s、
#     漂移峰值 ≈ 2.4 cm、末段漂移 0.000 cm（旧 checkpoint：3.68° / 5.9° / 0.089 / 1.10 cm）

# 证明"不能只靠轮、也不能只靠腿"
conda run -n sim python scripts/diag_mask_ablation.py
#   → 只腿 39 步倒、只轮 75 步倒、腿+轮 1500 步站住

# ★ 腿长变化验收（腿负责机体高度；内部腿长 0.15~0.27 m）
conda run -n sim python scripts/accept_leg_length.py --steps 4000 --settle 2500 \
    --targets 0.150 0.180 0.210 0.240 0.270
# 随机腿长课程验收（每 episode 采样一个目标）
conda run -n sim python scripts/diag_leg_curriculum.py --episodes 15
#   → 15/15 通过：腿长误差均值 2.4 mm、最大 2.9 mm
#   ⚠️ 训练与网页回放上限统一为 0.270 m，给动态切换预留恢复余量

# 0.15~0.27 m 动态腿长续训（配置已收入代码 profile）
conda run --no-capture-output -n sim python scripts/train_uz05_mujoco.py --profile v29

# ★ 交互式网页回放（不需要 checkpoint）
MUJOCO_GL=egl conda run --no-capture-output -n sim \
  python -u scripts/web_replay_uz05.py --controller-only
#   浏览器打开 http://127.0.0.1:8087/
#   页面实时显示 pitch RMS / 峰峰值 / 速率、漂移峰值、轮电流 RMS 与抖动 dI，
#   以及"腿偏置(低频) vs 轮电流(高频)"分工条；时间轴同画漂移(绿) + pitch(橙)。
#   对照旧策略（1.62 Hz 点头）：--port 8088 --checkpoint \
#     checkpoints/ppo_stand_s2_pitchquiet_v1/checkpoint_iter_40

# 站立三级课程（腿解锁 + 协同平衡控制器 + 逐级继承，最后一级验收 ±5cm）
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

> **站姿 pitch 点头 / 前后漂移的修复**见 [docs/stand_pitch_fix.md](docs/stand_pitch_fix.md)。
> 一句话：旧版站立训练**把腿动作锁死**（`lock_stand_leg_actions=True`），策略在物理上
> 没有腿可用，只能靠轮子硬撑位置环，于是产生 1.6 Hz 的 pitch 极限环；同时轮/腿平衡律
> 的**符号全部推导反了**。现在站立默认解锁腿动作，环境里内置一个已标定的
> **腿 + 轮协同平衡控制器**（`uz05/balance.py`：腿管低频姿态/位置/支撑，轮管高频
> pitch 与速度），策略在受限残差内微调。

## 目录结构

```
scripts/
  train_uz05.py                 训练入口（PPO + 非对称 actor-critic + 课程退火 + 日志）
  train_uz05_all.sh             分阶段课程脚本（自动串阶段、自动继承）
  accept_stand.py               ★ 站立验收：零指令下 pitch 振荡 / 漂移 / 电流颤振
  diag_mask_ablation.py         ★ 通道消融：证明"只靠腿/只靠轮"都不行
  optimize_coord_robust.py      鲁棒性优化 + 30 工况容量包线验证
  tune_impulse.py               冲量恢复（刹车—回弹）专项调参
  diag_full_compare.py          54 工况配置对比
  diag_residual_noise.py        探索噪声容忍度（定位 RL 失败真因）
  diag_impulse_trace.py         冲量响应逐帧轨迹
  diag_train_env.py             训练包装一致性检查
  accept_leg_length.py          ★ 腿长变化验收（跟踪 / 漂移 / 平衡）
  diag_leg_ws.py                腿机构工作空间标定 (q2,q4) → 腿长
  diag_leg_mirror.py            左右腿镜像约定标定
  diag_leg_sign.py              差模符号与静态增益标定
  diag_leg_max.py               腿长可达上限
  diag_leg_ceiling.py           腿长上限成因（几何饱和 vs 行程限位）
  diag_leg_curriculum.py        随机腿长课程验收
  diag_leg_track.py             腿长跟踪扫描
  web_replay_uz05.py            ★ 交互式网页回放（协同控制器 / 策略对照）
web_replay/                     回放前端（index.html / app.js / styles.css）
  distill_coord.py              把协同平衡控制器行为克隆进策略网络
  coord_refine.py               协同控制器增益精修 + 鲁棒性验证
  tune_coord3.py                协同控制器结构化调参（网格 + (1+1)-ES）
  diag_osc.py / diag_coord.py   系统辨识：符号、权限、开环/闭环响应
  diag_long_stand.py            长时程（60 s）站定漂移测试
  diag_disturbance.py           扰动恢复（速度冲量 / 持续外力）
  uz05/
    spec.py                     ★ 单一事实来源：接口/奖励/执行器/课程/域随机化
    balance.py                  ★ 腿 + 轮协同动态平衡控制器（含腿长环）
    model.py                    MJCF 加载、索引、接触、地形、域随机化写入
    actuators.py                关节 PD + 轮速→电调→电流→力矩
    env.py                      环境：观测、奖励、终止、课程
  validate_uz05_deployment_data.py   实机实测数据校验（等电调/电机数据）
diagnostics/uz05.xml            机器人 MJCF（含 29 个纯显示网格，加载时自动裁剪）
model_source/uz05/              SolidWorks 模型源（腿 / 底盘 / 总装 / 修复版）
calibration/                    平衡控制器标定原始数据
docs/stand_pitch_fix.md         ★ 站姿 pitch 点头 / 漂移的根因与修复
docs/leg_length.md              ★ 腿长变化：机构学、差模语义、验收、可达上限
docs/FIX_SUMMARY.md             站定修复总结
docs/uz05_refactor.md           重构说明：接口、坐标系、执行器、对齐开源、资源上限
checkpoints/<version>/          每次训练一个独立文件夹（策略 + 归一化统计 + 契约）
runs/<version>/                 tensorboard
logs/                           终端输出留存
```

## 接口（所有阶段固定，因此 checkpoint 永久可继承）

| | 维度 | 内容 |
|---|---|---|
| 观测 | **94** | actor 38（实机可得） + 特权 56（仅 critic），历史长度 1 |
| 动作 | **6** | 轮电流共模、轮电流差模、4 个腿关节位置偏置 |
| 控制 | 125 Hz | 关节 = 位置目标 + PD；轮 = PPO 电流残差 + 训练阶段平衡辅助 |

坐标系全部**机器人坐标系**（重力投影 / 机体角速度 / 机体前向速度），
不使用任何世界系绝对量，与实机陀螺仪一致。

## 训练阶段

`stand → low_speed → high_speed → steering → rotation → airborne → stairs → jump → recovery`

所有阶段使用针对 UZ-05 的自定义稠密奖励；站立阶段直接优化姿态、速度和相对起点位置误差。

**验收（站立 S2）**：零指令、零辅助下 `survive_rate = 1.0`、末段漂移 `≈ 0`、
pitch RMS `≈ 0.15°`（旧 checkpoint 是 3.68°）。硬约束：**不允许锁死姿态来控制姿态**，
平衡必须由**腿与轮协同**实现 —— 已用 `diag_mask_ablation.py` 实测证明：
只靠腿 39 步倒、只靠轮 75 步倒、腿+轮 1500 步站住。


**腿长（腿负责机体高度）**：腿通道语义已拆成**共模/差模**（共模改腿的俯仰角、
差模改腿长），腿长环与平衡环解耦。站立阶段的
`leg_length_range = (0.150, 0.270)` 会**每 episode 随机采样内部腿长目标**
（由已有的高度命令换算，不新增接口维度 → 旧 checkpoint 仍可加载）。
实测：固定目标 6/6（最大误差 2.8 mm）、随机课程 15/15（最大误差
2.9 mm），pitch RMS 约 0.003–0.005°，漂移很小。
代码内部腿长加轮半径约 `55 mm` 才是车底高度；因此训练区间对应车底约
`205~325 mm`。当前训练上限 `0.270 m` 避开高位形饱和区。详见
[docs/leg_extension_scan.md](docs/leg_extension_scan.md)。
详见 [docs/leg_length.md](docs/leg_length.md)。

**站立默认配置**（`STAND_LEVELS`，可被 CLI 覆盖）：

| 项 | 默认 | 说明 |
|---|---|---|
| `--unlock-stand-legs` | **开** | 腿动作参与平衡。锁腿会让策略永远学不到用腿 |
| `--stand-leg-action-limit` | 1.0 | ±0.35 rad 关节偏置 |
| `--coord-start/--coord-end` | 1.0 → 1.0 | 协同平衡控制器全权，策略学受限残差 |
| `--coord-residual-scale` | 0.05 0.02 0.05 | 残差预算（实测 \|a\|≥0.2 就会触发漂移终止） |

> ⚠️ **实测结论：本任务的站定精度超出了 PPO 稳定探索的范围。**
> 腿的 CoP 权限约 30 (rad/s²)/rad，策略动作均值只要偏 0.1 就是宏观漂移。
> 在 `coord_mix → 0`（策略完全自主）下训练，存活率会被熵/探索噪声从 1.00 打到
> 0.1~0.4；残差预算放到 0.15/0.30 同样被推坏。这个精度问题需要
> 更低噪声的优化器（如确定性策略梯度 + 动作平滑正则）或更小的探索方差，
> 尚未完成。当前可用的站定方案是**协同控制器本身**（见下）。

## 资源上限（别踩，之前把整机跑死过）

- subproc 环境 = 每环境一个独立进程，`--num-envs` 默认 8，代码里自动夹到 `min(核数/2, 8)`
- 线程数统一为 1（否则 8 进程 × 16 线程会抢满 16 核）
- MJCF 里的高模显示网格在加载时被裁掉（单进程 2.0 GB → 37 MB），要渲染时设 `UZ05_VISUAL=1`

## 文档

- [站姿 pitch 点头 / 漂移的根因与修复](docs/stand_pitch_fix.md) —— 符号辨识、锁腿问题、协同控制器、验收数据
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
