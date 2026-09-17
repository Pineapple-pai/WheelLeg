"""UZ-05 轮腿机器人 —— 能力规格与接口定义（单一事实来源）。

设计原则
--------
1. **观测全量常驻且始终实时计算**：所有观测块从第一天就存在，永远输出真实值。
   capability 开关只控制奖励/课程/终止，不改变观测维度与分布。
   => 任何 checkpoint 在任何 capability 组合下都能直接加载，**永远不需要迁移**。
2. **动作恒为 10 维**：已覆盖全部 8 个能力（含腿长、跳跃、自救），不随能力扩容。
3. **坐标系全部用机体系**（x 前 / y 左 / z 上），只使用实机陀螺仪与编码器可测量的量：
   姿态用重力投影、角速度用机体角速度、速度用机体前向速度、高度用腿长/相对支撑面。
4. **奖励全部实现**，由权重表开关；关闭项权重为 0。

物理约定（已用 MuJoCo 实测确认）
--------------------------------
- 前进方向 = 机体 +X（世界 +X，初始姿态为单位四元数）
- 轮轴 = 世界 −Y，两轮位于 y = ±0.2103（左右）
- 平衡角 = 绕 Y 的 pitch；yaw = 绕 Z
- **正轮力矩驱动车身朝 −X**，所以前进（vx>0）需要轮速目标为负
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# 路径
# --------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_XML = REPO_ROOT / "diagnostics" / "uz05.xml"
MESH_DIR = Path("/home/p/WheelLegMJCFReference/meshes/stl")

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
        official_rewards=True,
        vx_range=(0.0, 0.0), zero_command_prob=1.0,
        init_tilt=0.03, init_tilt_rate=0.02, init_vel=0.02,
        reward_groups=("posture", "regularization", "limits"),
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
    """站立分级课程：逐级收紧前后漂移要求，最终验收 ±5 cm。"""

    level: int
    name: str
    deadband: float          # 奖励死区（m）
    hard_limit: float        # 超出即终止（m）
    tilt_limit: float        # 终止倾角（rad）
    init_tilt: float
    station_kp: float        # 站定位置环增益（Nm/m）
    assist_start: float      # 本阶段起始辅助强度（1.0 = 全辅助）
    assist_end: float        # 本阶段结束辅助强度（0.0 = 完全靠策略）
    note: str


STAND_LEVELS: tuple[StandLevel, ...] = (
    StandLevel(0, "S0_balance", 0.08, 0.40, 0.30, 0.05, 100.0, 1.00, 0.60,
               "先学会不摔：辅助 1.0→0.6，漂移硬限放宽到 ±40 cm"),
    StandLevel(1, "S1_tighten", 0.05, 0.20, 0.25, 0.04, 120.0, 0.60, 0.20,
               "收紧到 ±20 cm 硬限，辅助退到 0.20"),
    StandLevel(2, "S2_accept", 0.04, 0.12, 0.22, 0.03, 120.0, 0.20, 0.00,
               "验收：辅助退到 0，前后漂移 ±5 cm 内，超差即终止"),
)
STAND_LEVEL_BY_INDEX = {lv.level: lv for lv in STAND_LEVELS}

# --------------------------------------------------------------------------
# 观测（固定 73 维，全部常驻且始终实时计算）
# --------------------------------------------------------------------------
# 观测拆分为「actor 可见」与「仅 critic 的特权部分」——对齐开源的非对称 actor-critic。
#
#   actor  : 只放**实机测得到**的量（编码器 / 陀螺仪 / 加速度计 / 电调回传 / 内部时钟）
#   critic : actor 全部 + 仿真真值（线速度、相对位移、接触力、地形…）
#
# 关键：旧 73 维 checkpoint 与当前接口**不兼容**（被删掉的通道是活信号，不是恒零占位），
# 已决定弃用旧 checkpoint 重新训练。接口从此固定，后续只增能力、不改维度。
ACTOR_OBS_BLOCKS: tuple[tuple[str, int], ...] = (
    ("gravity", 3),          # 重力投影（加速度计）—— 对齐开源 projected_gravity_b
    ("base_ang_vel", 3),     # 机体角速度（陀螺仪）—— 对齐 root_ang_vel_b
    ("leg_joint_pos", 4),    # 关节位置（编码器）
    ("leg_joint_vel", 4),    # 关节速度（编码器差分）
    ("wheel_joint_vel", 2),  # 轮速（C620 回传 rpm）
    ("command", 5),          # vx, vy, yaw_rate, 腿长, 跳跃
    ("previous_action", 6),   # = ACTION_DIM，对齐开源 actions 通道
    ("phase", 2),            # 相位时钟 sin/cos
    ("mode_onehot", 5),      # normal / airborne / stair / recover / jump
)

PRIV_OBS_BLOCKS: tuple[tuple[str, int], ...] = (
    ("base_lin_vel", 3),     # 机体系线速度（实机需状态估计）
    ("base_pos_rel", 2),     # 相对出发点位移（实机需里程计）
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
    "yaw": 1.0,                     # rad，有界
    "base_ang_vel": 0.25,           # 对齐开源 ang_vel
    "base_lin_vel": 2.0,            # 对齐开源 lin_vel（仅 critic）
    "base_pos_rel": 1.0,            # m，±0.6 有界（仅 critic）
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
    "phase": 1.0,
    "mode_onehot": 1.0,
}
ACTOR_FRAME_DIM = sum(width for _, width in ACTOR_OBS_BLOCKS)   # 单帧 actor 观测
PRIV_OBS_DIM = sum(width for _, width in PRIV_OBS_BLOCKS)
# 观测历史堆叠帧数（对齐开源 obs_history_len；1 = 不堆叠）
OBS_HISTORY = 5
ACTOR_OBS_DIM = ACTOR_FRAME_DIM * OBS_HISTORY
OBS_DIM = ACTOR_OBS_DIM + PRIV_OBS_DIM

OBS_SLICES: dict[str, slice] = {}
_start = 0
for _name, _width in OBS_BLOCKS:
    OBS_SLICES[_name] = slice(_start, _start + _width)
    _start += _width

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
        "leg_joint_pos": noise.dof_pos * noise.level * scale("leg_joint_pos"),
        "leg_joint_vel": noise.dof_vel * noise.level * scale("leg_joint_vel"),
        "wheel_joint_vel": noise.dof_vel * noise.level * scale("wheel_joint_vel"),
        "command": zero,            # 官方 noise_vec[6:8] = 0
        "previous_action": zero,    # 官方 noise_vec[20:26] = 0
        "phase": zero,
        "mode_onehot": zero,
        # 特权观测（critic）不加噪声
        "base_lin_vel": zero, "base_pos_rel": zero, "leg_length": zero,
        "wheel_contact_force": zero, "contact_flag": zero, "terrain_scan": zero,
        "dof_acc": zero, "torques": zero, "domain_params": zero,
    }

# --------------------------------------------------------------------------
# 动作（固定 10 维）
# --------------------------------------------------------------------------
# 对齐开源 action_space = 6：4 个腿关节位置 + 2 个轮速目标。
# 原来的 4 个「腿关节速度前馈」通道已移除（开源没有，且实测贡献很小）。
ACTION_SPEC: tuple[tuple[str, int, float], ...] = (
    ("wheel_common", 1, 8.0),           # rad/s —— 共模轮速目标（官方 vel_action_scale = 8.0）
    ("wheel_differential", 1, 6.0),     # rad/s —— 差模轮速目标（转向/旋转）
    ("hip_position_offset", 4, 0.35),   # rad   —— 4 个主动关节位置偏置（官方 pos_action_scale = 0.35）
)
ACTION_DIM = sum(width for _, width, _ in ACTION_SPEC)
ACTION_SLICES: dict[str, slice] = {}
_start = 0
for _name, _width, _ in ACTION_SPEC:
    ACTION_SLICES[_name] = slice(_start, _start + _width)
    _start += _width

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

    # 速度环（输出期望电流）—— 参数待实测标定
    speed_kp_a_per_rad_s: float = 0.081   # A / (rad/s)
    speed_ki_a_per_rad_s2: float = 0.0
    speed_error_limit: float = 5.0        # rad/s
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
class BalanceParams:
    """轮端外层平衡 + 速度环（输出关节力矩需求，折算成电流叠加到电调）。

    **全部为实测标定值**（单位：关节侧 Nm / (rad, rad/s, m/s)）：
    ``tau = -kp*(pitch + kd*pitch_rate) - kv*(vx - vx_cmd) + ki*∫(vx_cmd - vx)``

    标定记录（MuJoCo, 2500 步, 稳态速度）::

       目标 vx   +0.15   +0.30   +0.60   -0.15   -0.30   -0.60
       实测 vx   +0.156  +0.311  +0.625  -0.153  -0.307  -0.613

    ⚠️ 历史教训：旧 v28 外层用的是 ``+ station_kp * x_error``（位置环）。
    那使“速度指令”退化成位置偏移，稳态 vx≈0，导致速度跟踪奖励与实际行为
    长期相互抵消 —— 这是平移一直训不好的根因。本版改为**速度环 + 积分**，
    不再使用位置项。
    """

    pitch_kp: float = 28.7       # = 1.82 ctrl 单位 × gear 15.7647
    pitch_kd: float = 0.36       # s
    body_speed_kp: float = 26.8  # = 1.70 ctrl 单位 × gear
    body_speed_ki: float = 7.9   # = 0.50 ctrl 单位 × gear
    integral_limit: float = 30.0 # Nm
    overspeed_brake_kp: float = 0.0
    overspeed_margin: float = 0.03
    # 站定位置环（仅零指令时生效）。符号已实测确认：
    # 正轮力矩 → 车身加速度 −X，故 x 偏正时需要正力矩把它推回。
    # 实测（锁死姿态纯平移）：ctrl=+0.30 → Δvx=−1.72 m/s / 0.64 s。
    station_kp: float = 50.0        # Nm/m（关节侧，负号在控制器里）
    station_deadband: float = 0.05  # m —— ±5 cm 内不干预
    station_kd: float = 0.0         # Nm/(m/s)，闭环实测 0 最好
    # ★ 辅助退火系数：整个手写外环（俯仰 PD + 速度 PI + 站定环）都乘以它。
    #   1.0 = 全辅助（仅用于把策略"扶起来"），0.0 = 完全靠策略。
    #   **验收必须在 assist_scale = 0 下进行**，否则等于靠外部控制器维持姿态。
    assist_scale: float = 1.0


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
    station: float = 20.0
    station_vel: float = 2.0
    stand_vx: float = 6.0
    stand_yaw: float = 1.5
    stand_wheel_speed: float = 0.005
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

    官方做法：噪声加在**归一化之后**的 actor 观测上，均匀分布
    ``±(noise_scales.X × noise_level × obs_scales.X)``；命令与上一帧动作
    **不加噪声**；特权观测不加噪声；加噪后才推进历史队列
    （所以 5 帧历史里每帧的噪声独立）。

    ``noise_scales`` 取自官方基类配置，``noise_level`` 取自 UZ-05 厂商配置。
    """

    enabled: bool = True
    level: float = 0.5
    ang_vel: float = 0.2
    gravity: float = 0.05
    dof_pos: float = 0.01
    dof_vel: float = 1.5
    lin_vel: float = 0.1
    height_measurements: float = 0.1
    clip_observations: float = 100.0


@dataclass
class EnvParams:
    robot: RobotParams = field(default_factory=RobotParams)
    joint: JointActuatorParams = field(default_factory=JointActuatorParams)
    wheel: WheelEscParams = field(default_factory=WheelEscParams)
    balance: BalanceParams = field(default_factory=BalanceParams)
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
