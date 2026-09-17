# UZ-05 重构说明与训练配置

> 本次重构针对的目标：让**平移能真正训起来**，并让后续 8 个能力可以**逐级继承**，
> 而不是每加一个能力就重做一次接口。

---

## 1. 先说结论：为什么之前平移一直训不好

在对 `diagnostics/uz05.xml` 做平衡控制器标定时，发现旧 v28 的轮端外环是：

```python
ctrl = -1.82*(pitch + 0.36*pitch_rate) - 1.475*boundary_error(x) - 1.70*(vx - vx_cmd)
```

**这是一个位置控制器，不是速度控制器。** 实测三种外环的稳态速度：

| 指令 vx | 旧设计（含位置项） | 纯速度环 | **速度环 + 积分** ✅ |
|---|---|---|---|
| +0.15 | +0.003 | +0.089 | **+0.156** |
| +0.30 | +0.007 | +0.241 | **+0.311** |
| +0.60 | +0.015 | +0.546 | **+0.625** |
| −0.15 | −0.005 | −0.215 | **−0.153** |
| −0.30 | −0.008 | −0.367 | **−0.307** |
| −0.60 | −0.015 | −0.672 | **−0.613** |

旧设计下，"速度指令"只产生一个**位置偏移**，机体到位后 `vx → 0`。
而奖励却在考核**瞬时速度 `vx` 与指令的差** —— 控制器和奖励目标长期互相抵消。

**这就是平移训不起来的根因。** 本版把外环改成 **速度 PI**，全程（含反向）跟踪误差 < 5%。

---

## 2. 重构后的结构

```
scripts/
  train_uz05.py            # 训练入口（分阶段）
  uz05/
    __init__.py
    spec.py                # ★ 单一事实来源：能力/观测/动作/奖励/执行器/物理参数
    model.py               # MJCF 加载（diagnostics/uz05.xml）、索引、接触、地形
    actuators.py           # 关节 PD + 轮电调（可替换为实测参数）
    env.py                 # 环境：观测全量常驻，奖励全量实现
```

原来 2800 行的单文件被拆成 4 个模块，`spec.py` 是唯一的参数与规格来源。

---

## 3. 接口设计：固定维度 → 永久继承

**观测 226 维（actor 170 + 特权 56）、动作 6 维，在所有 9 个阶段完全固定。**

### actor 观测：34 维/帧 × 历史 5 帧 = 170 维（实机全部可测）

| 观测块 | 维度 | 内容 | 归一化 |
|---|---|---|---|
| gravity | 3 | 重力投影（加速度计），对齐开源 `projected_gravity_b` | ×1.0 |
| base_ang_vel | 3 | 机体角速度（陀螺仪），对齐 `root_ang_vel_b` | ×0.25 |
| leg_joint_pos | 4 | 4 个主动关节角（编码器） | ×1.0 |
| leg_joint_vel | 4 | 关节角速度（编码器差分） | ×0.05 |
| wheel_joint_vel | 2 | 轮速（C620 回传 rpm） | ×0.05 |
| command | 5 | vx, vy, yaw_rate, **离地高度**, 跳跃 | ×[2.0, 1.0, 0.25, **5.0**, 1.0] |
| previous_action | 6 | 上一帧动作，对齐开源 `actions` 通道 | ×1.0 |
| phase | 2 | 相位时钟 sin/cos | ×1.0 |
| mode_onehot | 5 | normal / airborne / stair / recover / jump | ×1.0 |

> `obs_history_length = 5` 与官方配置一致；历史长度只影响 actor 输入拼接，
> **不改变任何观测块的语义**，所以旧 checkpoint 与新建 checkpoint 之间依旧零迁移。

### 特权观测（仅 critic）：56 维

| 观测块 | 维度 | 内容 |
|---|---|---|
| base_lin_vel | 3 | 机体系线速度（实机需状态估计） |
| base_pos_rel | 2 | 相对出发点位移（实机需里程计） |
| leg_length | 4 | 腿长 + 变化率（可由关节角推出，故不进 actor） |
| wheel_contact_force | 2 | 轮法向力（实机无力传感器） |
| contact_flag | 4 | 轮接地×2 / 机身触地 / 腾空 |
| terrain_scan | 17 | 前方地形高度扫描（相对支撑面） |
| dof_acc | 6 | 6 个主动关节角加速度（对齐官方 `dof_acc`，缩放 0.0025） |
| torques | 6 | 6 个主动关节实际力矩（对齐官方 `torque`，缩放 0.05） |
| **domain_params** | **12** | **本局域随机化参数（质量/惯量/质心/摩擦/阻尼/增益/延迟…）** |

最后一块是对齐开源的关键：**critic 知道"这一局我控制的是台什么机器"，actor 不知道**。
12 项参数已在 `DomainRandomization.sample()` 里归一化到 `[-1, 1]`，
因此即使以后调整随机化区间，这一块的分布也不会突变。
`info["domain"]` 同时输出原始物理值，便于复盘。

**关键决定：所有观测块从第一天起就存在，并且始终输出真实值。**
capability 开关只控制 **奖励 / 课程 / 终止**，不改变观测维度、也不改变观测分布。

由此得到三条性质：

1. **任何阶段的 checkpoint 都能直接加载到任何其它阶段**，`PPO.load` 即可，**零迁移**；
2. 旧能力被完整保留（观测分布不变，只是奖励目标变了）；
3. 不存在"打开开关瞬间观测分布突变"的问题。

> ⚠️ 与历史的 **73 维接口不兼容**（已决定弃用 v23–v39 的旧 checkpoint 重新训练）。
> 接口契约写在 `<checkpoint>_contract.json` 里，每次训练都会覆写。

### 动作（6 维，覆盖全部 8 个能力）

| 通道 | 语义 | 缩放（对齐官方） | 服务的能力 |
|---|---|---|---|
| 0 | 轮速共模目标 `wheel_common` | ×8.0 rad/s（官方 `vel_action_scale`） | 平移、平衡 |
| 1 | 轮速差模目标 `wheel_differential` | ×6.0 rad/s | 转向 / 旋转 |
| 2:6 | 4 个腿关节位置偏置 `hip_position_offset` | ×0.35 rad（官方 `pos_action_scale`） | 腿长、下蹲、蹬伸、台阶、自救 |

> 速度前馈通道（旧版 6:10）已删除：官方只用"位置偏置 + 阻尼"，
> 多一路速度前馈既没有实测增益支撑，也让策略更容易用高频抖动骗奖励。

---

## 4. 坐标系：全部机体系

按实物用陀螺仪的要求，**不使用任何世界系绝对量**：

| 量 | 实现 |
|---|---|
| 姿态 | 重力在机体系的投影（等效加速度计） |
| 角速度 | 机体坐标系的 ω（等效陀螺仪） |
| 速度 | **机体前向速度** `vx_body`，不是世界 `qvel[0]` |
| 高度 | 腿长 / 相对支撑面，不用世界 z |
| 地形 | 前方扫描相对当前支撑面 |

**已实测确认的物理约定**：

- 前进方向 = 机体 **+X**；轮轴 = 世界 **−Y**；两轮位于 y = ±0.2103
- 平衡角 = 绕 Y 的 **pitch**；yaw = 绕 Z
- **正轮力矩驱动车身朝 −X**，所以前进（vx>0）需要负的轮力矩

---

## 5. 执行器接口

### 关节：位置 + 速度 + PD（DM-J8009P，24 V 供电）

```python
torque = kp*(q_target - q) + kd*(qd_target - qd)     # 限幅 ±20 Nm
q_target = 固定零位 + action[2:6] * 0.35             # 对齐官方 pos_action_scale
```

> ⚠️ 目标必须相对**固定零位**。写成"当前位姿 + 偏置"会让 `action=0` 时位置误差恒为 0，
> PD 退化成纯阻尼，腿会直接软掉（重构过程中踩过这个坑）。
> 与官方控制律 `torques = p_gains*(pos_ref + default_dof_pos - dof_pos) + d_gains*(vel_ref - dof_vel)`
> 完全同构，官方同样规定腿关节 `vel_ref = 0`。

### 轮：速度指令 → 电调 → 期望电流 → 力矩（M3508 + C620）

```python
error   = clip(w_target - w_actual, ±5 rad/s)
current = kp_a * error + ki_a * ∫error                # 限幅 ±10 A
torque  = torque_per_amp_joint * current              # 0.246 Nm/A
ctrl    = torque / gear_ratio                         # 15.7647
```

> **C620 本身没有速度环**：CAN 下发的是力矩电流（−16384..16384 ↔ −20..20 A），
> 速度环跑在我们自己的控制器里，反馈用 CAN 的转速回传。
> 官方同样是"轮子只给速度目标、位置目标置 0，靠阻尼增益跟踪"
> （`pos_ref[:, 2] = pos_ref[:, 5] = 0`），与本实现一致。

**后续填实测数据的位置**（全部集中在 `spec.py`）：

| 参数 | 当前值 | 来源 |
|---|---|---|
| `WheelEscParams.torque_per_amp_joint` | 0.246 Nm/A | 手册 2.46 Nm / 10 A |
| `WheelEscParams.speed_kp_a_per_rad_s` | 0.081 | 由旧 C620 增益换算 |
| `WheelEscParams.current_limit_a` | 10.0 | 额定 |
| `JointActuatorParams.kp / kd` | 100 / 4 | 旧版验证值 |
| `RobotParams.leg_length_min / max` | 0.150 / 0.340 | **[待实测替换]** |

---

## 6. 平衡控制器标定记录

外环（`BalanceParams`，关节侧力矩单位）：

```python
tau = -28.7*(pitch + 0.36*pitch_rate) - 26.8*(vx - vx_cmd) + 7.9*∫(vx_cmd - vx)
```

| 项 | 值 | 说明 |
|---|---|---|
| `pitch_kp` | 28.7 Nm/rad | = 1.82 ctrl 单位 × 15.7647（旧版验证值） |
| `pitch_kd` | 0.36 s | 旧版验证值 |
| `body_speed_kp` | 26.8 Nm/(m/s) | 实测标定（本轮新增积分项） |
| `body_speed_ki` | 7.9 Nm/(m/s·s) | 实测标定 |

**俯仰环符号验证**：初始 pitch=+0.10 rad，`tau = −Kp·pitch` 收敛（0.100→0.047），
`tau = +Kp·pitch` 发散 —— 确认符号为负。

**稳态测量**：

| 量 | 值 |
|---|---|
| 平衡稳态高度 | **0.2645 m** |
| 平衡稳态腿长（hip 站→轮站） | **0.2098 m** |
| 重置高度 | 0.280 m |

---

## 7. 八阶段训练配置

### 命令与课程参数（`spec.py` 的 `STAGES`）

| 阶段 | vx 范围 | yaw 范围 | 零指令概率 | 反向概率 | 初倾角 | 终止倾角 | 地形 |
|---|---|---|---|---|---|---|---|
| `stand` | — | — | 1.00 | — | 0.03 | 0.30 | 平地 |
| `low_speed` | 0.05–0.30 | — | 0.45 | 0.35 | 0.05 | 0.30 | 平地 |
| `high_speed` | 0.10–1.20 | — | 0.30 | 0.40 | 0.05 | 0.30 | 平地 |
| `steering` | 0.05–0.90 | 0.3–1.2 | 0.20 | 0.35 | 0.06 | 0.30 | 平地 |
| `rotation` | 0–0.40 | 0.8–3.0 | 0.15 | — | 0.06 | 0.30 | 平地 |
| `airborne` | 0–0.90 | 0–1.5 | 0.15 | — | **0.60** | **1.20**（持续 25 步） | 平地 |
| `stairs` | 0.30–1.00 | 0–0.8 | 0.10 | 0.30 | 0.10 | 0.80（15 步） | **阶梯** |
| `jump` | 0–1.00 | 0–1.0 | 0.15 | — | 0.20 | 1.00（20 步） | 平地 |
| `recovery` | 0–0.80 | 0–1.0 | 0.25 | — | **3.10（全姿态）** | **3.20**（60 步） | 平地 |

### 奖励组开关（仅自研奖励阶段使用）

> ⚠️ 现在**只有 `airborne / stairs / jump / recovery` 这 4 个阶段**用下面这套自研奖励组
> （`StageSpec.official_rewards = False`）。`stand / low_speed / high_speed / steering / rotation`
> 五个阶段全部走官方 16 项（§8.1），此时 `reward_groups` 不生效、`RewardWeights` 被忽略。
>
> 站立阶段的 ±5 cm 要求不靠奖励，而靠**终止条件**（`station_hard_limit`，按 S0/S1/S2 逐级收紧）+
> 终端里的 `drift_tail_cm` / `survive_rate` 验收统计，因此换成官方奖励后该要求依然成立。

每组在 `RewardWeights` 里有独立权重，阶段只决定**开启哪些组**：

| 组 | stand | low | high | steer | rot | air | stairs | jump | recover |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| posture | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| regularization | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| limits | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| track_vx | | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| track_yaw | | | | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| stand_still | | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| airborne | | | | | | ✓ | ✓ | ✓ | ✓ |
| terrain | | | | | | | ✓ | | |
| jump | | | | | | | | ✓ | |
| recovery | | | | | | | | | ✓ |

### 训练命令

全部逻辑集中在 `scripts/train_uz05_all.sh`（自动串阶段、自动继承上一阶段 checkpoint）：

```bash
cd /home/p/下载/WheelLeg

TAG=v1 bash scripts/train_uz05_all.sh stand        # 站立三级课程 s0 → s1 → s2
TAG=v1 bash scripts/train_uz05_all.sh low_speed    # 只跑一个阶段（自动找上一阶段）
TAG=v1 bash scripts/train_uz05_all.sh              # 全部 9 个阶段依次跑

NUM_ENVS=8 TAG=v1 bash scripts/train_uz05_all.sh stand   # 改并行环境数（默认 8，上限见 §10）
```

单阶段等价的手写命令：

```bash
conda run --no-capture-output -n sim python -u scripts/train_uz05.py \
  --stage stand --stand-level 0 --assist-adaptive \
  --updates 800 --rollout-steps 256 --batch-size 512 \
  --num-envs 8 --vec-env subproc --version stand_s0_v1
```

产物（每个版本一个独立文件夹，全部绝对路径）：

```
checkpoints/<version>/checkpoint.zip                 # 策略
checkpoints/<version>/checkpoint_vecnormalize.pkl    # 奖励归一化统计量（续训自动复用）
checkpoints/<version>/checkpoint_contract.json       # 接口契约
runs/<version>/                                      # tensorboard
logs/train_stand_v1_<时间戳>.log                     # 终端输出留存
```

> **继承是自动的**：所有阶段共享同一套 214/6 接口，`--checkpoint` 直接指向上一个阶段即可，
> 不存在维度迁移、也没有"打开开关导致观测分布突变"的问题。
> 终端每次迭代打印 ~25 行（每行一个量）：姿态、腿长、轮速、轮电流、关节力矩、
> 指令/实际速度、站位误差与尾均值、存活率、辅助强度、是否满足 ±5 cm 验收。

---

## 8. 与开源（StackForce SimReady / Wheel-Legged-Gym）的对齐

对齐基准：`/home/p/下载/Stackforce-simready-uz_05_wheel_legged_chassis-wheel-legged-gym/`。
其中 URDF 是**占位桩**（只有一个 `chassis` link）、`num_observations=12 / num_actions=1` 也是占位值，
**只有约定和尺度可信**；真实约定来自基类 `legged_robot_config.py`（27 维观测 / 6 维动作 /
`obs_history_length=5`）与厂商配置 `uz_05_wheel_legged_chassis_config.py`。

| 项 | 官方值 | 本项目 | 说明 |
|---|---|---|---|
| `pos_action_scale` | 0.35 | 0.35（唯一来源 `ACTION_SPEC`） | 关节位置偏置缩放 |
| `vel_action_scale` | 8.0 | 8.0（唯一来源 `ACTION_SPEC`） | 轮速共模缩放 |
| `decimation` | 4 | 4 | 两边控制周期不同：官方 = 1/60×4 = 0.0667 s，我们 0.008 s |
| `episode_length_s` | 20 | 20 | |
| `termination_grace_time_s` | 2.0 | 2.0 | |
| `resampling_time` | 10.0 | 10.0 | 指令重采样 |
| `tracking_sigma` | 0.25 | 0.25 | 速度跟踪 exp 核宽度 |
| 指令范围 | vx −0.3..0.6、yaw ±0.4、height 0.2..0.45 | 同（站立先固定 0.2645） | 高度=离地高度 |
| 关节观测缩放 | `dof_vel 0.05` / `dof_pos 1.0` | 同 | |
| 角速度/线速度缩放 | `ang_vel 0.25` / `lin_vel 2.0` | 同 | |
| 地形扫描缩放 | `height_measurements 5.0` | 同 | |
| 命令缩放 | `[lin_vel 2.0, ang_vel 0.25, height 5.0]` | `[2.0, 1.0, 0.25, 5.0, 1.0]` | 多 vy / 跳跃两维 |
| `clip_observations` | 100.0 | 100.0 | |
| 观测噪声 | `add_noise=True`、`noise_level=0.5`、`noise_scales{ang_vel .2, gravity .05, dof_pos .01, dof_vel 1.5}` | 同 | 加在归一化后、入历史队列前；命令与上一帧动作不加 |
| **奖励** | 16 项（见 §8.1） | **逐项照抄** | 5 个移动类阶段用官方项，台阶/跳跃/自救段用自研项 |
| 域随机化 | 只开摩擦 0.6..1.2 | 摩擦 0.70..1.30 + 11 项自选 | **有意偏离**：上实机需要，官方导出里其余项显式关闭 |
| 网络 | `[256,128,64]` ELU | 同 | |
| PPO | `entropy_coef 0.01`、`lr 1e-3`、`n_steps 24`、4096 环境、1500 iter | entropy 0.01、lr 1e-4 起自适应、rollout 256、8 环境 | 训练规模由硬件决定 |
| 控制律 | 腿 `p*(pos_ref+default−pos) − d*qd`；**轮 `d*(vel_ref − qd)`**（官方 UZ-05 子类重载，轮子没有位置项） | 同构 | 我们轮子多一个积分项（官方增益是空字典 → 全 0，无法照抄） |

> ⚠️ 官方导出里 `cfg.control.stiffness/damping` 都是**空字典**，基类把 `p_gains/d_gains`
> 初始化成 0 后只在字典里查值 ⇒ **官方公开的增益全是 0**。所以增益只能用我们自己的实测/标定值，
> 但"腿=位置PD、轮=速度误差×阻尼增益"这个**结构**完全一致。

### 8.1 奖励对齐（官方 16 项，权重照抄）

实现在 `env._reward()` 的 `official_rewards` 分支，权重表在 `spec.OfficialRewards`。

| 项 | 权重 | 形式（官方口径） |
|---|---|---|
| `tracking_lin_vel` | 1 | `exp(−(vx−vx_cmd)²/0.25)` |
| `tracking_ang_vel` | 0.5 | `exp(−(ω_z−ω_cmd)²/0.25)` |
| `base_height` | −1 | `|h − h_cmd|`（**绝对误差**，不是平方） |
| `nominal_state` | −0.1 | `(θ_L − θ_R)²`（两腿倾角差） |
| `lin_vel_z` | −2 | `v_z²` |
| `ang_vel_xy` | −0.05 | `ω_x² + ω_y²` |
| `orientation` | −1 | `g_x² + g_y²` |
| `dof_vel` | 0 | 关闭 |
| `dof_acc` | −2.5e-7 | `Σ q̈²`（按官方控制周期 0.0667 s 差分，见下） |
| `torques` | −1e-5 | `Σ τ²` |
| `action_rate` | −0.01 | `Σ (a − a_prev)²` |
| `action_smooth` | −0.01 | `Σ (a − 2a_prev + a_prev2)²` |
| `collision` | −1 | 机身接触计数（阈值 0.1 N） |
| `dof_pos_limits` | −1 | 出软限位量（`soft_dof_pos_limit = 0.9`） |
| `termination` | 0 | 官方为 0 |
| `custom_reward` | 0 | 未使用 |

三处必须注意的非显然细节：

1. **`dof_acc` 的差分窗口**：官方是"一个控制周期内的速度差"，其控制周期 0.0667 s，
   我们的控制周期 0.008 s。直接按 0.008 s 差分会把加速度放大 8.3 倍、**平方后惩罚放大 ≈69 倍**
   （实测该单项 −0.45，压过 +1.0 的跟踪项）。因此取 8 步窗口（≈0.064 s）差分，
   权重保持官方 −2.5e-7 不变。见 `spec.DOF_ACC_WINDOW`。
2. **`nominal_state` 在官方是失效项**：它依赖 `cfg.asset.l1/l2/offset`，而导出配置里这三个都是 0
   ⇒ 官方该项恒为 0。我们按同样公式、用真实腿几何算倾角，所以它是"有意义的版本"。
3. **`termination = 0`**：只有当再次引入"单步大额惩罚"时，才需要把它提到 `50 × 最坏单步`
   （`RewardWeights` 的说明里记着那次教训）。官方口径下最坏单步只有约 −3，保持 0 即可。

**结论**：动作语义、轮子控制方式、观测缩放/噪声、奖励集合都已与官方一致，
策略接口可以直接搬到 Isaac Gym 版官方环境里跑。

---

## 9. 性能与资源上限（曾把整机跑死，务必保留）

`diagnostics/uz05.xml` 里有 30 个 mesh geom，其中 **29 个是纯显示网格**
（`body_000..011.stl` 每个 10 MB ≈ 20 万面、`gimal_00X.obj` 各 8 MB），
`contype=0 conaffinity=0`，质量走独立的 `<inertial>`，对物理零影响。

| 场景 | 单进程内存 | 8 个 subproc 环境 |
|---|---|---|
| 保留显示网格 | **≈2.0 GB** | ≈16 GB → **14 GB 机器直接卡死** |
| 删掉显示网格（当前默认） | **≈37 MB** | ≈0.3 GB |

处理方式：`model._strip_visual_geoms()` 在编译前删掉纯显示 mesh geom 及其 mesh 资产，
**加载前后逐 body 对比 mass / inertia / ipos 完全一致，80 步轨迹 qpos/qvel/obs 偏差 0.000e+00**（已实测）。
渲染（回放、截图）时用 `UZ05_VISUAL=1` 保留高模。

其他护栏（`train_uz05.py` + `train_uz05_all.sh`）：

- `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1` + `torch.set_num_threads(1)`
  ——否则 8 个环境进程 × 16 线程 = 128 线程抢 16 个核；
- `--num-envs` 自动夹到 `min(核数/2, 8)`，要超配得显式设 `ALLOW_OVERSUBSCRIBE=1`；
- 训练进程 `nice -n 10`，桌面优先。

---

## 10. 训练前必须确认的三件事（已做）

1. **机构能站住** —— 扫平衡增益：`Kp ≥ 5, Kd ≥ 0.5` 可站稳，稳态高度 0.2645 m；
2. **速度能跟踪** —— 换成速度 PI 后全程误差 < 5%（含反向）；
3. **腿能撑住** —— 关节 PD 目标相对固定零位（不是当前位置）。

这三件事各自都是一个"看起来在训练、实际学不到东西"的陷阱，**建议每换一次模型都重跑一遍**。

---

## 11. 待办

| # | 事项 | 说明 |
|---|---|---|
| 1 | **填入电调/电机实测数据** | 见 §5 表格；目前全部为手册值或旧版验证值 |
| 2 | **实测腿长硬限位** | `RobotParams.leg_length_min / max` 现为占位 0.150 / 0.340 |
| 3 | 台阶地形参数 | `TerrainSpec` 台阶高度/深度目前 0.06 / 0.30 m |
| 4 | 跳跃安全前置 | DM-J8009P 峰值持续时间未知，实机前必须实测 |
| 5 | 可恢复集扫描 | 训练自救前先界定物理上可恢复的初始姿态范围 |
| 6 | ~~域随机化~~ | ✅ 已实现：质量/惯量/质心/摩擦/阻尼/增益/延迟 12 项，且进特权观测 |
| 8 | ~~奖励/观测噪声对齐~~ | ✅ 官方 16 项权重照抄（§8.1）、`add_noise 0.5`、`clip 100`、特权补 `dof_acc`/`torques` |
| 7 | 提交代码 | 当前工作区仍未提交 |

---

*本文档描述的是重构后的实现。所有标定数值均为 MuJoCo 实测结果。*
