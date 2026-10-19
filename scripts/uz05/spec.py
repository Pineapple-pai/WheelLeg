"""UZ-05 轮腿机器人 —— 能力规格与接口定义（单一事实来源）。

设计原则
--------
1. actor 与 critic 观测分开定义：actor 输入带传感器噪声的当前估计量，critic 另有
   仿真特权量。观测布局由 ``observations.py`` 按这里的单一事实来源严格校验。
2. **动作恒为 6 维**：两路轮速目标和四路腿关节位置偏置。
3. **机器人轴定义为机体系**（x 前 / y 左 / z 上）；观测来源与实机信号等价性仍需
   逐块验证，不能只根据名称推断。
4. 奖励项由阶段配置和奖励权重控制；官方奖励分支会直接返回，不使用自定义奖励组。

物理约定（已用 MuJoCo 实测确认）
--------------------------------
- 前进方向 = 机体 +X（世界 +X，初始姿态为单位四元数）
- 轮轴 = 世界 −Y，两轮位于 y = ±0.2103（左右）
- 平衡角 = 绕 Y 的 pitch；yaw = 绕 Z
- **正轮力矩驱动车身朝 −X**，所以前进（vx>0）需要轮速目标为负
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# 路径
# --------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_XML = REPO_ROOT / "uz05_修正后(3).xml"
MESH_DIR = Path(os.environ.get(
    "UZ05_MESH_DIR", str(REPO_ROOT / "model_source" / "uz05" / "mjcf_meshes")
)).expanduser()

# --------------------------------------------------------------------------
# 能力（按训练顺序）
# --------------------------------------------------------------------------
CAPABILITY_ORDER = (
    "stand",       # 站立
    "low_speed",   # 低速平移
    "high_speed",  # 高速平移
    "steering",    # 平移转向
    "rotation",    # 原地旋转
    "airborne",    # 空中落下保持平衡
    "stairs",      # 上下台阶
    "jump",        # 跳跃
    "recovery",    # 翻到自救
)


@dataclass(frozen=True)
class StageSpec:
    """一个 capability 阶段的完整训练设定。"""

    name: str
    # 命令范围
    vx_range: tuple[float, float] = (0.0, 0.0)
    vy_range: tuple[float, float] = (0.0, 0.0)
    yaw_range: tuple[float, float] = (0.0, 0.0)
    # 离地高度命令（机体系 z，不是腿长）。官方范围 [0.2, 0.45]，
    # 站立阶段先固定在额定高度，后续阶段再逐步放宽到官方全范围。
    base_height_range: tuple[float, float] = (0.2645, 0.2645)
    zero_command_prob: float = 0.5
    reverse_prob: float = 0.0
    jump_prob: float = 0.0
    # 奖励：True = 用官方 16 项（权重照抄），False = 用我们自己的塑形项
    official_rewards: bool = True
    # 初始状态分布
    init_tilt: float = 0.01
    init_tilt_rate: float = 0.01
    init_vel: float = 0.02
    init_height_offset: float = 0.0
    init_airborne_prob: float = 0.0
    init_airborne_height: tuple[float, float] = (0.0, 0.0)
    # 奖励组开关（对应 REWARD_GROUPS）
    reward_groups: tuple[str, ...] = ()
    # 终止
    tilt_limit: float = 0.30
    tilt_limit_hold_steps: int = 0
    # 其它
    terrain: str = "flat"
    episode_steps: int = 1000


# 训练课程：每一级都包含前面全部能力（累积）
STAGES: tuple[StageSpec, ...] = (
    StageSpec(
        name="stand",
        # 站立要优化站位、绝对航向和动作平滑，使用下方站立专用 shaping。
        official_rewards=False,
        vx_range=(0.0, 0.0), zero_command_prob=1.0,
        init_tilt=0.03, init_tilt_rate=0.02, init_vel=0.02,
        # Allow 40 control frames (320 ms) to recover after a transient tilt
        # crossing. The positive and negative pitch boundaries are identical.
        tilt_limit_hold_steps=40,
        reward_groups=("posture", "regularization", "limits", "stand_still"),
        episode_steps=1000,
    ),
    StageSpec(
        name="low_speed",
        official_rewards=True,
        vx_range=(0.05, 0.30), zero_command_prob=0.45, reverse_prob=0.35,
        init_tilt=0.05, init_tilt_rate=0.03, init_vel=0.05,
        reward_groups=("posture", "regularization", "limits", "track_vx", "stand_still"),
        episode_steps=1000,
    ),
    StageSpec(
        name="high_speed",
        official_rewards=True,
        vx_range=(0.10, 1.20), zero_command_prob=0.30, reverse_prob=0.40,
        init_tilt=0.05, init_tilt_rate=0.04, init_vel=0.08,
        reward_groups=("posture", "regularization", "limits", "track_vx", "stand_still"),
        episode_steps=1000,
    ),
    StageSpec(
        name="steering",
        official_rewards=True,
        vx_range=(0.05, 0.90), yaw_range=(0.3, 1.2), zero_command_prob=0.20, reverse_prob=0.35,
        init_tilt=0.06, init_tilt_rate=0.05, init_vel=0.10,
        reward_groups=("posture", "regularization", "limits", "track_vx", "track_yaw", "stand_still"),
        episode_steps=1000,
    ),
    StageSpec(
        name="rotation",
        official_rewards=True,
        vx_range=(0.0, 0.40), yaw_range=(0.8, 3.0), zero_command_prob=0.15,
        init_tilt=0.06, init_tilt_rate=0.06, init_vel=0.10,
        reward_groups=("posture", "regularization", "limits", "track_vx", "track_yaw", "stand_still"),
        episode_steps=1000,
    ),
    StageSpec(
        name="airborne",
        official_rewards=False,   # 空中段需要自研的收腿/着陆项
        vx_range=(0.0, 0.90), yaw_range=(0.0, 1.5), zero_command_prob=0.15,
        init_tilt=0.60, init_tilt_rate=0.60, init_vel=0.40,
        init_airborne_prob=0.35, init_airborne_height=(0.05, 0.35),
        reward_groups=("posture", "regularization", "limits",
                       "track_vx", "track_yaw", "stand_still", "airborne"),
        tilt_limit=1.20, tilt_limit_hold_steps=25,
        episode_steps=1000,
    ),
    StageSpec(
        name="stairs",
        official_rewards=False,   # 台阶段需要自研的地形项
        vx_range=(0.30, 1.00), yaw_range=(0.0, 0.8), zero_command_prob=0.10, reverse_prob=0.30,
        init_tilt=0.10, init_tilt_rate=0.10, init_vel=0.20,
        reward_groups=("posture", "regularization", "limits",
                       "track_vx", "track_yaw", "stand_still", "airborne", "terrain"),
        tilt_limit=0.80, tilt_limit_hold_steps=15,
        terrain="stairs",
        episode_steps=1200,
    ),
    StageSpec(
        name="jump",
        official_rewards=False,   # 跳跃段需要自研的腾空项
        vx_range=(0.0, 1.00), yaw_range=(0.0, 1.0), zero_command_prob=0.15, jump_prob=0.25,
        init_tilt=0.20, init_tilt_rate=0.20, init_vel=0.30,
        init_airborne_prob=0.20, init_airborne_height=(0.05, 0.25),
        reward_groups=("posture", "regularization", "limits",
                       "track_vx", "track_yaw", "stand_still", "airborne", "jump"),
        tilt_limit=1.00, tilt_limit_hold_steps=20,
        episode_steps=1200,
    ),
    StageSpec(
        name="recovery",
        official_rewards=False,   # 自救段需要自研的翻转项
        vx_range=(0.0, 0.80), yaw_range=(0.0, 1.0), zero_command_prob=0.25,
        init_tilt=3.10, init_tilt_rate=1.50, init_vel=0.50,
        init_airborne_prob=0.15, init_airborne_height=(0.05, 0.30),
        reward_groups=("posture", "regularization", "limits",
                       "track_vx", "track_yaw", "stand_still", "airborne", "recovery"),
        tilt_limit=3.20, tilt_limit_hold_steps=60,
        episode_steps=1500,
    ),
)

STAGE_BY_NAME = {stage.name: stage for stage in STAGES}
DEFAULT_STAGE = "stand"


@dataclass(frozen=True)
class StandLevel:
    """站立分级课程：逐级收紧漂移和航向要求，最终验收 ±5 cm / ±5°。"""

    level: int
    name: str
    deadband: float          # 奖励死区（m）
    hard_limit: float        # 超出即终止（m）
    yaw_hard_limit: float    # 相对初始航向超出即终止（rad）
    tilt_limit: float        # 终止倾角（rad）
    init_tilt: float
    note: str


STAND_LEVELS: tuple[StandLevel, ...] = (
    StandLevel(0, "S0_balance", 0.02, 0.40, 0.7854, 0.30, 0.02,
               "从近直立初态学会站住：漂移硬限 ±40 cm、偏航 ±45°"),
    StandLevel(1, "S1_tighten", 0.03, 0.20, 0.2618, 0.25, 0.04,
               "纯 PPO 收紧到漂移硬限 ±20 cm、偏航 ±15°"),
    StandLevel(2, "S2_accept", 0.00, 0.05, 0.0873, 0.20, 0.08,
               "验收：纯 PPO 全程漂移 ±5 cm、偏航 ±5°；倾角 11.5° 为恢复边界"),
)
STAND_LEVEL_BY_INDEX = {lv.level: lv for lv in STAND_LEVELS}

# --------------------------------------------------------------------------
# 观测（94 维：当前 actor 帧 38 + critic 特权观测 56）
# --------------------------------------------------------------------------
# 观测拆分为「actor 可见」与「仅 critic 的特权部分」——对齐开源的非对称 actor-critic。
#
#   actor  : 状态估计器、IMU、编码器和高度/地形传感器可提供的输入；不放仿真阶段标签
#   critic : actor 全部 + 仿真真值（线速度、相对位移、接触力、地形…）
#
# actor 使用当前 38 维观测，不堆叠历史帧；critic 专属 56 维。
# 相对初始航向的 sin/cos 使零命令下也能观测和修正累计偏航。
# 无历史观测的 MLP 接口使用 uz05_iface_v5（含执行器契约）。
ACTOR_OBS_BLOCKS: tuple[tuple[str, int], ...] = (
    ("gravity", 3),          # 重力投影（加速度计）—— 对齐开源 projected_gravity_b
    ("base_ang_vel", 3),     # 机体角速度（陀螺仪）—— 对齐 root_ang_vel_b
    ("heading_error_sincos", 2),  # 相对 episode 初始航向误差的 sin/cos
    ("base_lin_vel", 3),     # 机体系速度估计（IMU + 轮速/里程计融合）
    ("base_height_rel", 1),  # 相对当前地面的高度测量（高度传感器）
    ("base_pos_rel", 2),     # 相对起点的水平里程计估计（站定漂移反馈）
    ("terrain_preview", 3),  # 前方稀疏高度测量（深度/测距传感器）
    ("leg_joint_pos", 4),    # 关节位置（编码器）
    ("leg_joint_vel", 4),    # 关节速度（编码器差分）
    ("wheel_joint_vel", 2),  # 轮速（C620 回传 rpm）
    ("command", 5),          # vx, vy, yaw_rate, 离地高度目标, 跳跃
    ("previous_action", 6),   # = ACTION_DIM，对齐开源 actions 通道
)

PRIV_OBS_BLOCKS: tuple[tuple[str, int], ...] = (
    ("base_lin_vel_true", 3),  # 精确仿真机体系线速度（critic 特权）
    ("base_pos_rel_true", 2),  # 精确仿真相对位移（critic 特权）
    ("leg_length", 4),       # 腿长 + 变化率（可由关节角推出，故不进 actor）
    ("wheel_contact_force", 2),  # 轮法向力（实机无力传感器）
    ("contact_flag", 4),         # 轮接地×2 / 机身触地 / 腾空
    ("terrain_scan", 17),    # 前方地形扫描（实机需深度传感器）
    # 官方特权观测里的两项（critic 专用）：关节加速度与关节力矩
    ("dof_acc", 6),          # 6 个主动关节的角加速度（对齐 dof_acc 缩放 0.0025）
    ("torques", 6),          # 6 个主动关节的实际力矩（对齐 torque 缩放 0.05）
    # 域随机化参数（每 episode 采样一次，整局不变）。对齐开源：
    # critic 知道"这一局我控制的是台什么机器"，actor 不知道。
    # 12 项见 DomainRandomization.PARAM_NAMES，均已归一化到 [-1, 1]。
    ("domain_params", 12),
)

OBS_BLOCKS: tuple[tuple[str, int], ...] = ACTOR_OBS_BLOCKS + PRIV_OBS_BLOCKS

# --------------------------------------------------------------------------
# 观测归一化（标量或按分量向量）
#
# 对齐 StackForce SimReady 导出配置 ``uz_05_wheel_legged_chassis_config.py``：
#   ang_vel 0.25 / dof_vel 0.05 / lin_vel 2.0 / height_measurements 5.0 / torque 0.05
# 不归一化的话轮速 ±60 rad/s 会和重力投影 ±1 一起进网络，条件数极差。
# --------------------------------------------------------------------------
OBS_SCALE: dict[str, object] = {
    "gravity": 1.0,                 # 已是单位向量
    "base_ang_vel": 0.25,           # 对齐开源 ang_vel
    "heading_error_sincos": 1.0,    # 单位圆上的相对航向编码
    "base_lin_vel": 2.0,            # 对齐开源 lin_vel（actor 状态估计）
    "base_lin_vel_true": 2.0,       # 精确仿真值（critic）
    "base_height_rel": 5.0,         # 对齐开源 height measurements
    "base_pos_rel": 1.0,            # 相对起点里程计，m
    "base_pos_rel_true": 1.0,       # 精确仿真值（critic）
    "terrain_preview": 5.0,         # 前方高度测量，m
    "leg_joint_pos": 1.0,           # 对齐开源 dof_pos
    "leg_joint_vel": 0.05,          # 对齐开源 dof_vel
    "wheel_joint_vel": 0.05,        # 同 dof_vel：60 rad/s → 3.0
    "command": [2.0, 1.0, 0.25, 5.0, 1.0],   # 对齐官方 commands_scale：
                                             # [lin_vel 2.0, ang_vel 0.25, height 5.0]
                                             # 命令顺序 [vx, vy, yaw, 离地高度, 跳跃]
    "previous_action": 1.0,         # 已归一化到 ±1
    "leg_length": 1.0,              # m，±0.25 有界
    "wheel_contact_force": 0.01,    # N：100 N → 1.0（对齐开源 max_contact_force）
    "contact_flag": 1.0,            # 0/1 标志必须保持原样
    "terrain_scan": 5.0,            # 对齐开源 height_measurements
    "dof_acc": 0.0025,              # 对齐开源 dof_acc（仅 critic）
    "torques": 0.05,                # 对齐开源 torque（仅 critic）
    "domain_params": 1.0,           # 已归一化到 [-1, 1]
}

# 记录每个张量块进入策略/critic 前的测量来源，训练 contract 与文档共用。
# actor 的高度/地形值目前是带噪仿真测量代理，后续需要用真实传感器标定替换。
OBS_SOURCE: dict[str, str] = {
    "gravity": "IMU 姿态融合 / 重力方向估计",
    "base_ang_vel": "机体陀螺仪角速度",
    "heading_error_sincos": "相对 episode 初始航向的航向估计 sin/cos（当前为 MuJoCo 代理）",
    "base_lin_vel": "IMU、轮速与里程计融合的机体系速度估计（当前为带噪 MuJoCo 代理）",
    "base_height_rel": "相对支撑面的向下测距/高度估计（当前为带噪 MuJoCo 代理）",
    "base_pos_rel": "相对起点的水平里程计估计（当前为带噪 MuJoCo 代理）",
    "terrain_preview": "前向稀疏深度/测距高度（当前为带噪解析地形代理）",
    "leg_joint_pos": "腿关节编码器",
    "leg_joint_vel": "腿关节编码器差分或驱动器速度估计",
    "wheel_joint_vel": "左右轮编码器/电调速度反馈",
    "command": "上层速度、高度和跳跃命令",
    "previous_action": "策略上一控制周期动作缓存",
    "base_lin_vel_true": "MuJoCo 机体系线速度真值",
    "base_pos_rel_true": "MuJoCo 相对起点位置真值",
    "leg_length": "MuJoCo 腿部几何及其控制周期差分",
    "wheel_contact_force": "MuJoCo 接触法向力",
    "contact_flag": "MuJoCo 轮/机身接触标签",
    "terrain_scan": "MuJoCo 解析地形高度扫描",
    "dof_acc": "MuJoCo 主动关节角加速度估计",
    "torques": "MuJoCo 执行器输出力矩",
    "domain_params": "训练 episode 域随机化参数",
}
ACTOR_FRAME_DIM = sum(width for _, width in ACTOR_OBS_BLOCKS)   # 单帧 actor 观测
PRIV_OBS_DIM = sum(width for _, width in PRIV_OBS_BLOCKS)
# 开源最终采用无历史观测的 MLP；FrameStack5/GRU 只作为对照实验。
ACTOR_OBS_DIM = ACTOR_FRAME_DIM
OBS_DIM = ACTOR_OBS_DIM + PRIV_OBS_DIM

# 官方控制周期 = sim.dt(1/60) × decimation(4) = 0.0667 s，我们是 0.008 s。
# 官方的 dof_acc 是"一个控制周期内的速度差"：直接用 0.008 s 差分，幅值大 8.3 倍、
# 平方后惩罚大 ≈69 倍（实测 -0.45 对比跟踪项 +1.0，会直接压过主奖励）。
# 所以按官方周期取 8 步窗口差分，权重仍然保持官方的 -2.5e-7 不动。
DOF_ACC_WINDOW = 8


def observation_noise_vector(noise: "ObservationNoise") -> dict[str, object]:
    """每个观测块的噪声半宽（加的是 ``±vec`` 均匀噪声），对齐官方公式。

    官方：``vec = noise_scales.X × noise_level × obs_scales.X``，
    加在**归一化之后**的 actor 观测上；命令、上一帧动作、特权观测都不加噪声。
    """
    zero = 0.0
    if not noise.enabled:
        return {name: zero for name, _ in OBS_BLOCKS}

    def scale(key: str) -> float:
        value = OBS_SCALE[key]
        if isinstance(value, (list, tuple)):
            value = value[0]
        return float(value)

    return {
        "gravity": noise.gravity * noise.level,
        "base_ang_vel": noise.ang_vel * noise.level * scale("base_ang_vel"),
        "heading_error_sincos": noise.heading * noise.level,
        "base_lin_vel": noise.lin_vel * noise.level * scale("base_lin_vel"),
        "base_height_rel": noise.height_measurements * noise.level * scale("base_height_rel"),
        "base_pos_rel": noise.position * noise.level * scale("base_pos_rel"),
        "terrain_preview": noise.height_measurements * noise.level * scale("terrain_preview"),
        "leg_joint_pos": noise.dof_pos * noise.level * scale("leg_joint_pos"),
        "leg_joint_vel": noise.dof_vel * noise.level * scale("leg_joint_vel"),
        "wheel_joint_vel": noise.dof_vel * noise.level * scale("wheel_joint_vel"),
        "command": zero,            # 控制命令不加传感噪声
        "previous_action": zero,    # 内部动作缓存不加传感噪声
        # 特权观测（critic）不加噪声
        "base_lin_vel_true": zero, "base_pos_rel_true": zero, "leg_length": zero,
        "wheel_contact_force": zero, "contact_flag": zero, "terrain_scan": zero,
        "dof_acc": zero, "torques": zero, "domain_params": zero,
    }

# --------------------------------------------------------------------------
# 动作（固定 6 维）
# --------------------------------------------------------------------------
# 对齐开源 action_space = 6：4 个腿关节位置 + 2 个轮速目标。
# 原来的 4 个「腿关节速度前馈」通道已移除（开源没有，且实测贡献很小）。
ACTION_SPEC: tuple[tuple[str, int, float], ...] = (
    ("wheel_common", 1, 8.0),           # rad/s —— 共模轮速目标（官方 vel_action_scale = 8.0）
    ("wheel_differential", 1, 6.0),     # rad/s —— 差模轮速目标（转向/旋转）
    ("hip_position_offset", 4, 0.35),   # rad   —— 4 个主动关节位置偏置（官方 pos_action_scale = 0.35）
)
ACTION_DIM = sum(width for _, width, _ in ACTION_SPEC)

# --------------------------------------------------------------------------
# 奖励组（全部实现，权重为 0 即关闭）
# --------------------------------------------------------------------------
REWARD_GROUPS = (
    "posture",          # 姿态/高度/腿长
    "regularization",   # 动作平滑/力矩/速度/对称
    "limits",           # 关节与腿长限位软壁垒
    "track_vx",         # 前向速度跟踪
    "track_yaw",        # yaw 角速度跟踪
    "stand_still",      # 零指令静止
    "airborne",         # 腾空/落地
    "terrain",          # 台阶辅助
    "jump",             # 跳跃相位
    "recovery",         # 自救进度
)

# --------------------------------------------------------------------------
# 物理与执行器参数（后续用电调/电机实测数据替换，见 docs）
# --------------------------------------------------------------------------


@dataclass
class JointActuatorParams:
    """关节：位置 + 速度 + PD 控制器（DM-J8009P-2EC，24 V 供电）。

    实测参数（厂家手册）::

        额定电压 24 V（支持 24–48 V）   额定/峰值电流 20 / 50 A
        额定/峰值扭矩 20 / 40 N·m       额定转速 100 rpm = 10.47 rad/s
        24 V 空载最大转速 160 rpm = 16.76 rad/s
        减速比 9:1   极对数 21   相电感 80 µH   相电阻 0.145 Ω
        编码器 14 bit ×2（磁编单圈绝对）   CAN 1 Mbps
        控制模式 MIT / 速度 / 位置
        保护：驱动过温 120 °C 退出使能；电机过温建议 ≤100 °C；
              过压建议 ≤52 V；过流建议 ≤39 A；欠压 ≥15 V

    力矩常数（输出侧）= 20 N·m / 20 A = **1.0 N·m/A**。

    ⚠️ MJCF 的 ctrlrange 是 ±39 N·m，**超过实机额定 20 N·m**，
    所以这里必须自己限幅，不能依赖模型。
    """

    kp: float = 100.0            # Nm/rad
    kd: float = 4.0              # Nm/(rad/s)
    torque_limit: float = 20.0   # Nm —— 实机额定（峰值 40，暂不开放）
    torque_per_amp: float = 1.0  # Nm/A（输出侧）
    velocity_limit: float = 10.47  # rad/s —— 额定 100 rpm
    peak_torque_limit: float = 40.0
    over_temperature_c: float = 100.0


@dataclass
class WheelEscParams:
    """轮：M3508（特制减速箱）+ C620。

    实测参数::

        M3508 减速箱改 + C620（24 V）
          空载转速 587 rpm = 61.47 rad/s    空载电流 0.63 A
          额定转速 571 rpm = 59.79 rad/s    额定扭矩 2.46 N·m
          额定电流 10 A                     最大效率 70 %
          堵转扭矩 3.69 N·m                 重量 336.4 g
        C620 电调
          CAN 指令 = **转矩电流**，-16384~16384 ↔ -20~20 A
          最大持续电流 20 A；反馈含机械角/转速(rpm)/实际转矩电流/温度

    ⚠️ **C620 没有内置速度环**：它接收的是电流（转矩）指令。
    速度环跑在我们的控制器上，用电调回传的 rpm 作反馈 —— 见
    :class:`~uz05.actuators.WheelEscController`。

    ⚠️ 表中「堵转电流 2.5 A」与其余数据矛盾（0.246 N·m/A × 2.5 A = 0.62 N·m
    ≠ 3.69 N·m），按额定点反推 3.69 N·m 需要约 15 A。该值**判为无效**，
    力矩常数一律取额定点 0.246 N·m/A，堵转扭矩仅作硬上限。
    """

    # 仿真速度环：满轮速误差 8 rad/s 对应额定持续电流 10 A。
    # 旧的 0.081 × 5 只允许 0.405 A，轮端力矩不足以平衡机身。
    # 该增益仍需用实机 C620 电流/转速数据标定，不能直接视作硬件实测值。
    speed_kp_a_per_rad_s: float = 1.25    # A / (rad/s)
    speed_ki_a_per_rad_s2: float = 0.0
    speed_error_limit: float = 8.0        # rad/s (8 × 1.25 A = 10 A rated current)
    integral_limit_a: float = 8.0
    # 电流 → 输出侧力矩
    torque_per_amp_joint: float = 0.246   # Nm/A（额定点 2.46 Nm / 10 A）
    current_limit_a: float = 10.0         # 额定；C620 持续上限 20 A
    esc_current_limit_a: float = 20.0     # C620 硬上限
    # 输出侧限制
    joint_torque_limit: float = 2.46      # Nm 额定
    stall_torque_limit: float = 3.69      # Nm 堵转硬上限
    velocity_limit: float = 59.79         # rad/s 额定 571 rpm
    no_load_velocity: float = 61.47       # rad/s 587 rpm
    no_load_current_a: float = 0.63       # 空载电流（可作阻力矩）
    efficiency: float = 0.70
    gear_ratio: float = 268.0 / 17.0


@dataclass
class RobotParams:
    """机体与几何参数。"""

    wheel_radius: float = 0.055
    leg_length_min: float = 0.150           # 实物硬限位下限 [待实测替换]
    leg_length_max: float = 0.340           # 实物硬限位上限 [待实测替换]
    hip_joint_limit: float = 1.2            # 髋关节机械行程 ±1.2 rad [待实测替换]
    tendon_limit: float = 0.388             # MJCF tendon 上限
    reset_height: float = 0.268          # 贴近平衡高度，减小落地瞬态漂移
    nominal_stand_height: float = 0.2645  # 实测平衡稳态高度
    nominal_leg_length: float = 0.2098    # 实测平衡稳态腿长（tendon 口径）
    control_substeps: int = 4
    # 轮速目标限幅：**不在这里定义**，一律取 ACTION_SPEC 的缩放
    # （以前这里写 10.0、ACTION_SPEC 写 8.0，实际生效的是 10.0，
    #  和官方 vel_action_scale = 8.0 不一致 —— 已删掉这组重复定义）
    command_accel_limit: float = 0.75       # m/s^2
    command_yaw_accel_limit: float = 4.0    # rad/s^2


@dataclass
class RewardWeights:
    """全部奖励项的权重。关闭 = 0。

    ⚠️ **铁律：终止惩罚必须远大于"单步最坏惩罚"。**

    否则策略会学会**主动摔倒**来止损 —— 实测过一次：``wheel_differential``
    每步最多 -20，而 ``termination`` 只有 -20（一次性），结果 ``survive_rate``
    在 36 轮内从 0.67 崩到 0.00，而 ``reward_total`` 反而在"变好"。

    经验取法：``termination >= 50 × 单步最坏惩罚``。
    当前单步最坏约 -3.6（station -3.0 + wheel -0.5 + 其他），
    故取 200（约 55 步）。
    """

    # --- posture ---
    upright: float = 3.0
    height: float = 2.0
    leg_length: float = 1.0
    joint_neutral: float = 0.5
    # --- regularization ---
    action_rate: float = 0.05
    action_smooth: float = 1.20
    leg_action: float = 0.30
    leg_velocity_action: float = 0.0    # 动作通道已移除
    joint_velocity: float = 0.02
    joint_torque: float = 1e-4
    wheel_power: float = 1e-4
    leg_symmetry: float = 20.0
    wheel_differential: float = 0.02
    # --- limits ---
    joint_limit: float = 5.0
    leg_length_limit: float = 20.0
    # --- tracking ---
    track_vx: float = 6.0
    track_vx_tight: float = 1.5
    track_vx_square: float = 3.0
    track_vx_gap: float = 4.0
    track_vy: float = 1.6
    track_yaw: float = 2.0
    track_yaw_square: float = 0.5
    wrong_direction: float = 4.0
    # --- stand still ---
    station: float = 60.0
    station_vel: float = 4.0
    stand_vx: float = 12.0
    stand_yaw: float = 4.0
    stand_heading: float = 150.0
    stand_ang_vel: float = 2.0
    stand_tilt: float = 100.0
    stand_wheel_speed: float = 0.02
    stand_body_contact: float = 12.0
    # MuJoCo one-step sweep: positive common wheel speed produces positive
    # pitch acceleration. These terms reward PPO for opposing pitch and rate.
    stand_balance_action: float = 24.0
    stand_balance_rate: float = 4.0
    stand_heading_action: float = 12.0
    stand_heading_rate: float = 2.0
    # The negative-pitch replay showed negative mean hip offsets reinforcing
    # the fall. Penalize only that state/action combination so positive-pitch
    # recovery and ordinary leg control retain the baseline reward.
    stand_negative_pitch_hip_action: float = 12.0
    # --- airborne ---
    airborne_upright: float = 2.0
    airborne_leg_retract: float = 1.0
    landing_impact: float = 2.0
    undesired_contact: float = 5.0
    # --- terrain ---
    terrain_progress: float = 2.0
    front_wheel_height: float = 1.0
    # --- jump ---
    jump_height: float = 5.0
    jump_phase_time: float = 0.5
    # --- recovery ---
    recovery_upright: float = 10.0
    recovery_progress: float = 5.0
    # --- terminal ---
    termination: float = 200.0
    alive: float = 0.0


@dataclass
class DomainRandomization:
    """域随机化范围。每 episode 在 reset 时采样一次，整局保持不变。

    默认值参考开源 wheelbipe V14 的 EventCfg（质量 0.9~1.3、惯量 0.8~1.2、
    COM ±0.02~0.04、摩擦 0.5~1.2、关节增益 0.75~1.25、执行器 0.8~1.1）。
    **上实机前必须用实测误差替换这些范围，不能凭猜。**
    """

    enabled: bool = True
    base_mass_scale: tuple[float, float] = (0.90, 1.25)
    base_inertia_scale: tuple[float, float] = (0.80, 1.20)
    base_com_offset_m: tuple[float, float] = (0.04, 0.04)   # x/y 对称范围
    base_com_offset_z_m: float = 0.02
    wheel_friction_scale: tuple[float, float] = (0.70, 1.30)
    joint_damping_scale: tuple[float, float] = (0.75, 1.25)
    joint_kp_scale: tuple[float, float] = (0.80, 1.20)
    joint_kd_scale: tuple[float, float] = (0.80, 1.20)
    wheel_torque_scale: tuple[float, float] = (0.85, 1.15)   # 力矩常数
    wheel_speed_gain_scale: tuple[float, float] = (0.80, 1.20)
    actuator_delay_steps: tuple[int, int] = (0, 1)

    # 喂给 critic 的参数顺序（12 项）
    PARAM_NAMES: tuple[str, ...] = (
        "base_mass_scale", "base_inertia_scale",
        "base_com_x", "base_com_y", "base_com_z",
        "wheel_friction_scale", "joint_damping_scale",
        "joint_kp_scale", "joint_kd_scale",
        "wheel_torque_scale", "wheel_speed_gain_scale", "actuator_delay",
    )

    def sample(self, rng):
        """返回 ``{参数名: (原始值, 归一化值)}``。归一化到 [-1, 1]。"""
        def norm(value, lo, hi):
            if hi <= lo:
                return 0.0
            return float(2.0 * (value - lo) / (hi - lo) - 1.0)

        out = {}
        m = float(rng.uniform(*self.base_mass_scale))
        out["base_mass_scale"] = (m, norm(m, *self.base_mass_scale))
        i = float(rng.uniform(*self.base_inertia_scale))
        out["base_inertia_scale"] = (i, norm(i, *self.base_inertia_scale))
        cx = float(rng.uniform(-self.base_com_offset_m[0], self.base_com_offset_m[0]))
        cy = float(rng.uniform(-self.base_com_offset_m[1], self.base_com_offset_m[1]))
        cz = float(rng.uniform(-self.base_com_offset_z_m, self.base_com_offset_z_m))
        out["base_com_x"] = (cx, norm(cx, -self.base_com_offset_m[0], self.base_com_offset_m[0]))
        out["base_com_y"] = (cy, norm(cy, -self.base_com_offset_m[1], self.base_com_offset_m[1]))
        out["base_com_z"] = (cz, norm(cz, -self.base_com_offset_z_m, self.base_com_offset_z_m))
        f = float(rng.uniform(*self.wheel_friction_scale))
        out["wheel_friction_scale"] = (f, norm(f, *self.wheel_friction_scale))
        d = float(rng.uniform(*self.joint_damping_scale))
        out["joint_damping_scale"] = (d, norm(d, *self.joint_damping_scale))
        kp = float(rng.uniform(*self.joint_kp_scale))
        out["joint_kp_scale"] = (kp, norm(kp, *self.joint_kp_scale))
        kd = float(rng.uniform(*self.joint_kd_scale))
        out["joint_kd_scale"] = (kd, norm(kd, *self.joint_kd_scale))
        t = float(rng.uniform(*self.wheel_torque_scale))
        out["wheel_torque_scale"] = (t, norm(t, *self.wheel_torque_scale))
        g = float(rng.uniform(*self.wheel_speed_gain_scale))
        out["wheel_speed_gain_scale"] = (g, norm(g, *self.wheel_speed_gain_scale))
        delay = int(rng.integers(self.actuator_delay_steps[0], self.actuator_delay_steps[1] + 1))
        out["actuator_delay"] = (float(delay), norm(delay, *self.actuator_delay_steps))
        return out


@dataclass
class OfficialRewards:
    """官方 UZ-05 奖励（StackForce SimReady 导出，**逐项 1:1 对应**）。

    来源：``uz_05_wheel_legged_chassis_config.py`` 的 ``rewards.scales``
    ＋ ``envs/base/legged_robot.py`` 里对应的 ``_reward_*`` 实现。

    权重**照抄，不要改**：一旦改了，跑出来的曲线就没法和开源对比。
    需要额外塑形时另加自己的项（``RewardWeights``），不要动这里的数。

    几处必须知道的官方细节：

    - ``base_height`` 的目标来自 ``commands[:, 2]``（**离地高度**，不是腿长），
      范围 ``[0.2, 0.45]`` m；
    - ``nominal_state`` 官方代码里依赖 ``cfg.asset.l1/l2/offset``，而导出的
      UZ-05 配置里这三个是 0 → 该项恒为 0（失效）。我们按同样的公式
      ``(theta_L - theta_R)^2`` 实现，但用**真实腿几何**算腿倾角，
      因此它在我们这里是"两条腿倾角不一致"的惩罚（有意义版）；
    - ``termination`` 官方是 0。只有当引入"单步大额惩罚"时才需要把它提到
      ``50 × 最坏单步``（见 RewardWeights 的说明），否则保持 0；
    - ``collision`` 只统计 ``penalize_contacts_on = ["chassis"]`` 的接触，
      阈值 0.1 N；
    - ``dof_pos_limits`` 用软限位（``soft_dof_pos_limit = 0.9``），且只检查
      4 个腿关节（官方代码只取索引 0,1,3,4）。
    """

    tracking_lin_vel: float = 1.0
    tracking_ang_vel: float = 0.5
    base_height: float = -1.0
    nominal_state: float = -0.1
    lin_vel_z: float = -2.0
    ang_vel_xy: float = -0.05
    orientation: float = -1.0
    dof_vel: float = 0.0            # 官方 0 = 关闭
    dof_acc: float = -2.5e-7
    torques: float = -1e-5
    action_rate: float = -0.01
    action_smooth: float = -0.01
    collision: float = -1.0
    dof_pos_limits: float = -1.0
    termination: float = 0.0
    custom_reward: float = 0.0
    # 形式参数（官方 rewards 段）
    tracking_sigma: float = 0.25
    soft_dof_pos_limit: float = 0.9
    max_contact_force: float = 100.0
    collision_force_threshold: float = 0.1


@dataclass
class ObservationNoise:
    """观测噪声（对齐官方 ``noise`` 段：``add_noise=True, noise_level=0.5``）。

    官方做法：噪声加在**归一化之后**的当前 actor 观测上，均匀分布
    ``±(noise_scales.X × noise_level × obs_scales.X)``；命令与上一帧动作
    **不加噪声**；特权观测不加噪声。策略不堆叠历史观测帧。

    ``noise_scales`` 取自官方基类配置，``noise_level`` 取自 UZ-05 厂商配置。
    """

    enabled: bool = True
    level: float = 0.5
    ang_vel: float = 0.2
    heading: float = 0.01
    gravity: float = 0.05
    dof_pos: float = 0.01
    dof_vel: float = 1.5
    lin_vel: float = 0.1
    height_measurements: float = 0.1
    position: float = 0.01
    clip_observations: float = 100.0


@dataclass
class EnvParams:
    robot: RobotParams = field(default_factory=RobotParams)
    joint: JointActuatorParams = field(default_factory=JointActuatorParams)
    wheel: WheelEscParams = field(default_factory=WheelEscParams)
    rewards: RewardWeights = field(default_factory=RewardWeights)
    official_rewards: OfficialRewards = field(default_factory=OfficialRewards)
    noise: ObservationNoise = field(default_factory=ObservationNoise)
    domain_randomization: DomainRandomization = field(default_factory=DomainRandomization)
    control_dt: float = 0.008       # 4 × 2 ms
    terminate_on_drift: float = 0.60
    terminate_lateral_vel: float = 0.80
    # 站定要求（由 stand 分级课程覆盖）
    station_deadband: float = 0.05
    station_hard_limit: float = 0.12


def active_rewards(stage: StageSpec) -> dict[str, bool]:
    """把阶段的奖励组展开成 ``{组名: 是否启用}``。"""
    enabled = set(stage.reward_groups)
    return {group: (group in enabled) for group in REWARD_GROUPS}
