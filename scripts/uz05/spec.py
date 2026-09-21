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

import numpy as np

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
    # 离地高度命令（机体系 z，不是腿长）。
    # 站立阶段先固定在额定高度，后续阶段再逐步放宽到官方全范围。
    base_height_range: tuple[float, float] = (0.23825, 0.23825)
    # ★ 腿长变化训练：是否在站立阶段随机化腿长目标，以及可达范围。
    #   腿长目标 = 高度命令 − 轮半径（实测 base_z ≈ leg_length + 0.0541）。
    #   默认关闭，保持历史站定行为不变；打开后每 episode 采样一个腿长目标，
    #   腿长环跟踪它、奖励按它计误差。
    #   上限避开高位形的负载饱和区；不要把无载几何上限直接当成站立训练上限。
    leg_length_range: tuple[float, float] | None = None
    zero_command_prob: float = 0.5
    reverse_prob: float = 0.0
    jump_prob: float = 0.0
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
        vx_range=(0.0, 0.0), zero_command_prob=1.0,
        base_height_range=(0.23825, 0.23825),
        init_tilt=0.03, init_tilt_rate=0.02, init_vel=0.02,
        # ★ 腿长变化训练：每 episode 在 0.15~0.27 m 内采样一个腿长目标。
        #   代码内部腿长是髋轴到轮心距离；加轮半径约 55 mm 后得到车底高度。
        #   0.270 m 在当前负载闭环中远离饱和区，为快速切换和域随机化留出恢复余量。
        leg_length_range=(0.150, 0.270),
        # Stationary standing must explicitly suppress yaw-rate and wheel
        # spin. Without this group a small differential action can accumulate
        # yaw for thousands of steps while position drift still looks good.
        reward_groups=("posture", "regularization", "limits", "stand_still"),
        episode_steps=1000,
    ),
    StageSpec(
        name="low_speed",
        # Low-speed translation is symmetric forward/backward.  Keep yaw
        # locked here; steering gets its own command/reward stage later.
        vx_range=(0.05, 0.30), zero_command_prob=0.45, reverse_prob=0.50,
        init_tilt=0.05, init_tilt_rate=0.03, init_vel=0.05,
        reward_groups=("posture", "regularization", "limits", "track_vx", "yaw_lock", "stand_still"),
        episode_steps=1000,
    ),
    StageSpec(
        name="high_speed",
        vx_range=(0.10, 1.20), zero_command_prob=0.30, reverse_prob=0.40,
        init_tilt=0.05, init_tilt_rate=0.04, init_vel=0.08,
        reward_groups=("posture", "regularization", "limits", "track_vx", "yaw_lock", "stand_still"),
        episode_steps=1000,
    ),
    StageSpec(
        name="steering",
        vx_range=(0.05, 0.90), yaw_range=(0.3, 1.2), zero_command_prob=0.20, reverse_prob=0.35,
        init_tilt=0.06, init_tilt_rate=0.05, init_vel=0.10,
        reward_groups=("posture", "regularization", "limits", "track_vx", "track_yaw", "stand_still"),
        episode_steps=1000,
    ),
    StageSpec(
        name="rotation",
        vx_range=(0.0, 0.40), yaw_range=(0.8, 3.0), zero_command_prob=0.15,
        init_tilt=0.06, init_tilt_rate=0.06, init_vel=0.10,
        reward_groups=("posture", "regularization", "limits", "track_vx", "track_yaw", "stand_still"),
        episode_steps=1000,
    ),
    StageSpec(
        name="airborne",
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
    """站立分级课程：逐级收紧漂移/倾角终止条件，并把教师辅助退火到 0。

    ⚠️ 与旧版的关键区别（旧版是"站不稳"的直接原因之一）：

    1. **动作权限不再逐级变化**。旧版 S0/S1/S2 把 ``policy_current_scale_a``
       分别缩到 2 / 4 / 8 A，而策略动作空间始终是 [-1,1] —— 于是一个在 S1 训好的
       checkpoint 加载到 S2 后输出被放大 2 倍，直接自激。现在全级别统一 8 A。
    2. **``hard_limit`` 只是"摔倒/跑飞"判据，不是验收判据**。实测标定过的参考
       平衡律在 5 cm 处的瞬时峰值也有 5.0~5.1 cm（初始扰动换来的），把终止线
       压在 5 cm 会让"标准答案"本身也被判死，episode 永远活不满 → 学不到东西。
       验收仍然是**末段 200 步平均漂移 < 5 cm**（见 ``station_tail_mean``）。
    3. **``hard_hold_steps``**：瞬时越界要先连续超限若干步才终止，避免单帧尖峰。
    """

    level: int
    name: str
    deadband: float          # 位置环死区（m）——只用于辅助教师
    hard_limit: float        # 漂移超过即终止（m）；仅作安全判据
    hard_hold_steps: int     # 连续超限多少步才终止
    tilt_limit: float        # 终止倾角（rad）
    init_tilt: float
    station_kp: float        # 辅助教师的外环位置增益（rad/m）
    assist_start: float      # 本阶段起始辅助强度（1.0 = 全教师）
    assist_end: float        # 本阶段结束辅助强度（0.0 = 完全靠策略）
    leg_action_limit: float  # 4 个腿关节位置残差的最大幅度（归一化）
    dr_scale: float          # 域随机化课程：所有区间向标称值收缩的比例
    note: str
    # ★ 腿+轮协同平衡控制器（uz05.balance.CoordinatedBalance）的混合系数课程。
    #   1.0 = 完全由协同控制器驱动（策略学残差），0.0 = 完全靠策略。
    #   这是"腿参与平衡"能否被学到的关键：站立阶段必须解锁腿动作，并让一个
    #   已标定的腿+轮协同律先稳住机体，策略再接管。
    coord_start: float = 0.0
    coord_end: float = 0.0


STAND_LEVELS: tuple[StandLevel, ...] = (
    # All stand levels use the same physical upright boundary.  A loose S0
    # limit let PPO collect full episodes while resting at roughly 27 deg of
    # pitch, which is a fallen posture rather than a transferable balance skill.
    #
    # ★ coord_* 是"腿+轮协同平衡控制器"（uz05.balance.CoordinatedBalance）的
    #   混合系数：1.0 = 控制器全权驱动 + 策略在受限残差内微调；0.0 = 完全靠策略。
    #
    #   默认三级都钉在 1.0 —— 这是**站定精度要求**决定的。实测腿的 CoP 权限约
    #   30 (rad/s²)/rad，策略动作均值只要偏 0.1 就是宏观漂移；标准 PPO 的熵项
    #   与高斯探索会把标定好的基线逐步推坏（实测存活率 1.00 → 0.14）。
    #   默认配置 = "协同控制器负责平衡 + 策略学小残差"，兼顾精度与安全。
    #   想训练完全自主的策略（更高风险），显式退火：
    #     --coord-start 1.0 --coord-end 0.0 --coord-residual-scale 1 1 1
    #
    #   站立阶段**不再锁定腿动作**——旧版锁腿使得站立策略永远学不会用腿，
    #   只能靠轮子硬撑，结果就是 1.6 Hz 的 pitch 点头极限环。
    StandLevel(0, "S0_balance", 0.03, 0.30, 0, 0.30, 0.03, 27.9, 0.00, 0.00, 0.00, 0.25,
               "腿+轮协同控制器稳住机体（腿管低频姿态/位置、轮管高频 pitch），策略学受限残差；"
               "倾角 < 0.30 rad，漂移硬限 ±30 cm",
               coord_start=1.0, coord_end=1.0),
    StandLevel(1, "S1_tighten", 0.03, 0.18, 5, 0.30, 0.03, 27.9, 0.00, 0.00, 0.00, 0.60,
               "协同控制器 + 策略残差，域随机化 60%，漂移硬限 ±18 cm",
               coord_start=1.0, coord_end=1.0),
    StandLevel(2, "S2_accept", 0.02, 0.10, 10, 0.30, 0.03, 27.9, 0.00, 0.00, 0.00, 1.00,
               "验收：协同控制器 + 策略残差、无辅助、域随机化 100%，稳态漂移 < 5 cm、全程峰值 < 5 cm",
               coord_start=1.0, coord_end=1.0),
)
STAND_LEVEL_BY_INDEX = {lv.level: lv for lv in STAND_LEVELS}

# --------------------------------------------------------------------------
# 观测（固定接口，全部常驻且始终实时计算）
# --------------------------------------------------------------------------
# 观测拆分为「actor 可见」与「仅 critic 的特权部分」——对齐开源的非对称 actor-critic。
#
#   actor  : 只放**实机测得到**的量（编码器 / 陀螺仪 / 加速度计 / 电调回传 / 内部时钟）
#   critic : actor 全部 + 仿真真值（线速度、相对位移、接触力、地形…）
#
# ``station_error`` 是相对起点的前向里程计误差。要训练闭环定点，actor 必须看到它；
# 只把它留给 critic 时，策略无法区分向前和向后的静态位置误差。
ACTOR_OBS_BLOCKS: tuple[tuple[str, int], ...] = (
    ("gravity", 3),          # 重力投影（加速度计）—— 对齐开源 projected_gravity_b
    ("base_ang_vel", 3),     # 机体角速度（陀螺仪）—— 对齐 root_ang_vel_b
    # 站立轮控必须知道质心前后速度；它可由轮速/IMU 融合状态估计获得，不能只给 critic。
    ("base_lin_vel_actor", 3),
    ("leg_joint_pos", 4),    # 关节位置（编码器）
    ("leg_joint_vel", 4),    # 关节速度（编码器差分）
    ("wheel_joint_vel", 2),  # 轮速（C620 回传 rpm）
    ("command", 5),          # vx, vy, yaw_rate, 腿长, 跳跃
    ("station_error", 1),    # 前向里程计相对起点误差（m）
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
# 不归一化的话轮速 ±60 rad/s 会和重力投影 ±1 一起进网络，条件数极差。
# --------------------------------------------------------------------------
OBS_SCALE: dict[str, object] = {
    "gravity": 1.0,                 # 已是单位向量
    "yaw": 1.0,                     # rad，有界
    "base_ang_vel": 0.25,           # 对齐开源 ang_vel
    "base_lin_vel": 2.0,            # 对齐开源 lin_vel（仅 critic）
    "base_lin_vel_actor": 2.0,      # 同一状态估计量，供 actor 闭环站定
    "base_pos_rel": 1.0,            # m，±0.6 有界（仅 critic）
    "leg_joint_pos": 1.0,           # 对齐开源 dof_pos
    "leg_joint_vel": 0.05,          # 对齐开源 dof_vel
    "wheel_joint_vel": 0.05,        # 同 dof_vel：60 rad/s → 3.0
    "command": [2.0, 1.0, 0.25, 5.0, 1.0],   # 对齐官方 commands_scale：
                                             # [lin_vel 2.0, ang_vel 0.25, height 5.0]
                                             # 命令顺序 [vx, vy, yaw, 离地高度, 跳跃]
    "station_error": 10.0,      # 0.05 m -> 0.5；位置环不会被大轮速量纲淹没
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
OBS_HISTORY = 1
ACTOR_OBS_DIM = ACTOR_FRAME_DIM * OBS_HISTORY
OBS_DIM = ACTOR_OBS_DIM + PRIV_OBS_DIM

OBS_SLICES: dict[str, slice] = {}
_start = 0
for _name, _width in OBS_BLOCKS:
    OBS_SLICES[_name] = slice(_start, _start + _width)
    _start += _width

# 取八步窗口求关节加速度，避免控制频率下的差分噪声主导特权观测。
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
    "command": zero,
    "station_error": zero,      # 保持里程计零点与符号，避免位置反馈被随机翻转
    "previous_action": zero,
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
    "yaw_lock",         # 非转向平移时抑制 yaw 角速度
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
    # ★ PPO 单位动作（[-1,1]）对应的轮电流（A）。**全课程统一 8.0**，
    #   不再按站立分级缩放：旧版 S0/S1/S2 分别乘 0.25/0.5/1.0，等于让
    #   同一个策略输出在不同级别下放大 2~4 倍，跨级继承必然自激。
    #   8 A 单轮 ⇒ 1.38 Nm 关节侧（含 0.7 效率），俯仰权限可达 ±0.16 rad。
    policy_current_scale_a: float = 8.0
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
    """轮端平衡教师：**级联结构 + 力矩预算**（输出期望轮电流，A）。

    结构（本版最重要的改动之一）::

        theta_ref = clip(kv*(vx - vx_cmd) + kx*x_error, ±theta_max)   # 外环给俯仰指令
        tau       = kp*(theta_ref - pitch) - kd_eff*pitch_rate        # 内环俯仰 PD
        current   = clip(tau / torque_per_amp, ±teacher_current_limit)

    **为什么不再用"各项直接相加到力矩"的老形式**：
    老形式 ``tau = -kp*(pitch + kd*pitch_rate) - kv*(vx-vx_cmd) - kx*x`` 数学上
    与级联等价（``kv_eff = kp*kv``），但每个增益都直接写成力矩，于是：:

    * ``kd = 0.36 s`` ⇒ 微分增益 ``kp*kd = 10.3 Nm/(rad/s)``。轮端可用力矩只有
      ``8 A × 0.246 × 0.7 = 1.38 Nm``，姿态角速度一超过 0.13 rad/s 电流就饱和，
      实测退化成 **bang-bang 颤振**（电流在 ±8 A 之间交替），既不消振也不出力；
    * ``station_kp = 100 Nm/m`` ⇒ 5 cm 位置误差直接要 5 Nm = 20 A，**远超电流上限**，
      内环俯仰环失去权限，实测俯仰角标准差 0.085 rad（≈5°）并周期性撞倾角上限；
    * 死区 ±5 cm + 高增益 ⇒ 极限环，稳态漂移恰好被顶到 4.4 cm 左右（刚好卡在
      ±5 cm 验收线上，看上去"永远差一点"）。

    现在外环只输出**有界的俯仰指令** ``theta_max``，内环需要的最大力矩
    ``kp*(theta_max + 允许的 pitch)`` 被限制在电流预算内，俯仰环永远保有权限。

    增益依据（MuJoCo 实测）：m=15.8 kg，轮半径 r=0.055 m，质心高 h≈0.26 m，
    维持俯仰角所需轮力矩 ``tau ≈ m*g*r*theta = 8.53*theta`` Nm（关节侧）。
    实测扫参最优（完整域随机化、20 个随机初始条件、外环符号取 +1）：
    ``pitch_kp = 28.7``、``kd_eff = 5.0 Nm/(rad/s)``、外环 ``kv = 13.9 s``、
    ``station_kp = 27.9 rad/m``（等效力矩增益 800 Nm/m）、``theta_max = 0.6 rad``：
    **survive 20/20、末段漂移 1.7 cm、峰值 3.1 cm**，平均电流 5.4 A。

    ⚠️ 外环增益"高到饱和"反而是对的：0.5 cm 位置误差就要 4 Nm，远超 3.44 Nm
    的轮端权限，于是外环实际上是一个**继电式（滑模）位置环**，靠切换满力矩把
    漂移压在厘米级。增益调小（kx=100 Nm/m）反而因为"温和的饱和"产生 12~14 cm
    的极限环 —— 实测扫过 kx ∈ {5…800} Nm/m、kv ∈ {5…400} Nm/(m/s)，单调改善。

    ⚠️ **两个增益的绝对下限/上限都要守**：
    * ``pitch_kp`` 必须 > 8.53（``m*g*r``）才可能稳定；实测 10 只有 17% 余量，
      大扰动下恢复不过来（实测漂到 0.6 m/s 匀速跑飞）；20 才够。
    * ``kd_eff = kp*kd`` 超过 ~2 Nm/(rad/s) 就会让 1.38 Nm 的电流权限在正常
      角速度下饱和，退化成 bang-bang 颤振。所以 kp 提高时 **kd 必须同比降**。

    注意：本类现在是**教师**，不是"必须依赖的辅助"。
    ``assist_scale = 0`` 时它完全不参与控制，站立必须由策略实现。
    """

    pitch_kp: float = 28.7        # Nm/rad（关节侧；物理下限 8.53，3.4 倍余量）
    pitch_kd: float = 0.174       # s ⇒ kd_eff = pitch_kp*pitch_kd = 5.0 Nm/(rad/s)
    body_speed_kp: float = 400.0  # Nm/(m/s) ⇒ 外环 kv = 400/28.7 = 13.9 s
    body_speed_ki: float = 0.0    # Nm/(m/s·s)；实测加积分反而更容易极限环
    integral_limit: float = 6.0   # Nm
    overspeed_brake_kp: float = 0.0
    overspeed_margin: float = 0.03
    # 站定位置环（仅零指令时生效）。符号实测（锁死姿态开环）：
    # 正轮力矩 → Δvx<0；闭环下"要朝 +X 修正必须先向 +X 倾"，
    # 故 x_error<0 时需要 theta_ref>0 ⇒ theta_ref = +kx*x_error。
    station_kp: float = 27.9      # 外环增益（rad/m）⇒ 等效力矩增益 kp*kx = 800 Nm/m
    # 外环符号（实测标定，见 actuators.BalanceController 的推导）：
    #   speed_sign   : θ_ref 对速度误差 (vx − vx_cmd) 的符号
    #   station_sign : θ_ref 对位置误差 x_error 的符号
    # 实测 +1 是负反馈（漂移收敛），−1 是正反馈（实测以 0.7 m/s 匀速漂走）。
    speed_sign: float = 1.0
    station_sign: float = 1.0
    station_deadband: float = 0.0
    station_kd: float = 0.0       # rad/(m/s)，外环速度阻尼（0 = 关闭）
    # ★ 外环俯仰指令限幅 = 力矩预算。kp*theta_max = 1.2 Nm ≈ 4.9 A，
    #   保证内环俯仰 PD 始终还剩至少一半电流权限。
    theta_max: float = 0.60       # rad
    teacher_current_limit: float = 8.0   # A，教师输出限幅（= 策略单通道权限）
    # ★ 辅助系数：1.0 = 完全由教师控制，0.0 = 完全由策略控制。
    #   实际下发 ``current = assist*teacher + (1-assist)*policy``（线性混合），
    #   而不是旧版的"策略补残差"——旧版在 assist=1 时残差目标恒为 0（学不到任何
    #   平衡技能），在 assist 低时目标又超过策略权限（S0 只有 2 A）而不可达。
    assist_scale: float = 1.0


@dataclass
class RobotParams:
    """机体与几何参数。"""

    wheel_radius: float = 0.055
    leg_length_min: float = 0.150           # 实物硬限位下限 [待实测替换]
    leg_length_max: float = 0.340           # 实物硬限位上限 [待实测替换]
    hip_joint_limit: float = 1.2            # 髋关节机械行程 ±1.2 rad [待实测替换]
    tendon_limit: float = 0.388             # MJCF tendon 上限

    # ------------------------------------------------------------------
    # ★ 负载平衡位形（MuJoCo 实测：域随机化关闭、辅助 1.0、末段 500 步均值）
    #
    # 为什么必须整条 qpos 一起给：UZ-05 的腿是**闭环五连杆**（MJCF 每腿 5 个铰链
    # + 2 个 ``<connect>`` 等式约束），只有 2 个关节被电机驱动，其余是被动关节。
    # 旧版重置只写 4 个主动关节、被动关节留在 0，等于把机器人放在"等式约束违反"
    # 的位形上 —— 第一个物理步约束会把整机弹一下，实测造成 5~10 cm 初始漂移和
    # 一次姿态冲击。这是"明明有辅助也站不稳"最主要的可避免来源。
    #
    # 下面 14 个值对应 ``qpos[7:]``，顺序（MJCF 定义）::
    #
    #   L_chassis_link2, L_link2_link5, L_link5_link3, L_link5_link1,
    #   L_link1_link6, L_wheel, L_chassis_link4,
    #   R_chassis_link2, R_link2_link5, R_link5_link3, R_link5_link1,
    #   R_link1_link6, R_wheel, R_chassis_link4
    #
    # 轮子转角置 0（旋转对称，无影响）。
    # ------------------------------------------------------------------
    reset_joint_pos: tuple[float, ...] = (
        -0.19450, 0.27196, -0.15812, -0.27196, 0.27200, 0.0, 0.19129,
        -0.19141, 0.26949, -0.15698, -0.26947, 0.26959, 0.0, 0.19057,
    )
    # 4 个主动关节（hip_qpos_adr 顺序 L2, L4, R2, R4）在负载平衡时的实际角度。
    # 用途：奖励的"中立位"参考。**不是** PD 的目标 —— 见 pd_neutral_joint_pos。
    stand_joint_pos: tuple[float, float, float, float] = (
        -0.19450, 0.19129, -0.19141, 0.19057,
    )
    # 腿位置 PD 的零位（action[2:6] = 0 时下发的位置目标）——**站姿指令**。
    #
    # ⚠️ 关节 PD 是有限刚度：负载下关节从"目标"再偏出约 0.10 rad 才产生支撑
    # 力矩（kp=100 ⇒ 10 Nm）。所以：
    #   目标 = pd_neutral_joint_pos（本值，一条"下蹲指令"）
    #   实际 = stand_joint_pos（负载平衡角，比目标再低 0.094 rad）
    # 把目标直接写成 stand_joint_pos 会让腿再塌 0.1 rad（实测腿长 0.207→0.182），
    # 基准位形随之改变；把目标写成 0 则站得更高（腿长 0.207）但更不稳。
    #
    # 为什么选"蹲"：实测同样增益下，高站姿（腿长 0.207 m）尾漂 5.1 cm，
    # 蹲姿（腿长 0.184 m）尾漂 2.0 cm —— 摆短了，倒立摆更稳。
    pd_neutral_joint_pos: tuple[float, float, float, float] = (
        -0.10055, 0.09896, -0.09886, 0.09842,
    )
    reset_height: float = 0.23825          # 实测负载平衡高度（与上面关节位形一致）
    nominal_stand_height: float = 0.23825  # 实测平衡稳态高度
    nominal_leg_length: float = 0.18413    # 实测平衡稳态腿长（hip site→wheel site）
    control_substeps: int = 4
    # 轮速目标限幅：**不在这里定义**，一律取 ACTION_SPEC 的缩放
    # （以前这里写 10.0、ACTION_SPEC 写 8.0，实际生效的是 10.0，
    #  和官方 vel_action_scale = 8.0 不一致 —— 已删掉这组重复定义）
    command_accel_limit: float = 0.75       # m/s^2
    command_yaw_accel_limit: float = 4.0    # rad/s^2


@dataclass
class RewardWeights:
    """全部奖励项的权重。关闭 = 0。**本版按站立任务重新标定过。**

    ⚠️ **铁律：终止惩罚必须远大于"单步最坏惩罚"。**

    否则策略会学会**主动摔倒**来止损 —— 实测过一次：``wheel_differential``
    每步最多 -20，而 ``termination`` 只有 -20（一次性），结果 ``survive_rate``
    在 36 轮内从 0.67 崩到 0.00，而 ``reward_total`` 反而在"变好"。

    本版单步最坏约 -6（station -2、station_vel -1、height -1.5、其余 -1.5），
    ``termination = 200``（约 33 步）仍然远大于它。

    ⚠️ **``alive`` 必须为正**：站立阶段每步都该有正的生存收益，否则
    "早点摔倒少挨罚"会重新出现。取 0.5，相当于 400 步的寿命价值。
    """

    # --- posture ---
    upright: float = 8.0            # exp(-(tilt/0.25)^2)，避免恒定低头姿态成为存活解
    # State-potential shaping for PPO credit assignment.  This is computed
    # only from the measured upright state, never from a teacher action.
    upright_progress: float = 4.0
    height: float = 2.0             # 归一化高度误差，防止折腿降高
    leg_length: float = 2.0         # 归一化腿长误差，防止折腿降高
    # 腿长误差逐步减小的势函数奖励。只依赖目标和编码器推算腿长，实物可得；
    # 默认关闭，由动态腿长训练 profile 显式打开。
    leg_length_progress: float = 0.0
    # 目标附近的腿长速度惩罚，用于让策略学习提前刹车而不是越过目标再回弹。
    leg_length_rate: float = 0.0
    leg_length_rate_sigma_m_s: float = 0.04
    joint_neutral: float = 1.0      # mean((q - stand_q)^2)
    # --- regularization ---
    action_rate: float = 0.05
    leg_action: float = 0.10
    leg_velocity_action: float = 0.0    # 动作通道已移除
    joint_velocity: float = 0.01
    joint_torque: float = 1e-4
    wheel_power: float = 1e-4
    leg_symmetry: float = 20.0
    wheel_differential: float = 0.05
    # 站立时轮子不该持续空转（旧版叫 wheel_differential，语义是共模轮速）。
    wheel_speed: float = 0.02
    # ★ 轮电流变化的平滑度。这是**直接压制"高频点头"的项**：pitch 点头的
    #   直接原因是共模轮电流逐帧大幅翻转（bang-bang），单步电流本身不大，
    #   差分却很大。权重必须远小于 upright 等主项，否则会把"正常配平"也一起
    #   罚掉（实测：权重 2.0 时标定协同控制器单步被罚 -4.3，完全盖过姿态奖励）。
    wheel_current_rate: float = 0.15
    # ★ 轮电流二阶差分（"电流抖动"）。比一阶差分更专一的颤振判据：平滑的
    #   配平动作二阶差分很小，而 bang-bang 翻转会非常大。
    wheel_current_jerk: float = 3.0
    # ★ 腿动作变化的平滑度（腿做低频，不该抖）。
    leg_action_rate: float = 2.0
    # --- limits ---
    joint_limit: float = 5.0
    leg_length_limit: float = 20.0
    # --- tracking ---
    track_vx: float = 6.0
    track_vx_tight: float = 1.5
    track_vx_square: float = 3.0
    track_vx_gap: float = 4.0
    # Translation stages should preserve heading; steering stages omit this
    # group and use track_yaw instead.
    yaw_lock: float = 1.5
    yaw_lock_sigma: float = 0.20
    track_vy: float = 1.6
    track_yaw: float = 2.0
    track_yaw_square: float = 0.5
    wrong_direction: float = 4.0
    # --- stand still ---
    # 有界二次惩罚：超过 sigma 后不再线性放大，避免"位置项压过姿态项"。
    station: float = 3.0            # -3*min((x/0.04)^2, 1)
    station_sigma_m: float = 0.04
    # Potential reward for reducing odometry error.  It has no action target
    # and supplies dense feedback for braking an otherwise stable overshoot.
    station_progress: float = 40.0
    station_vel: float = 1.0        # -1*min((vx/0.15)^2, 1)
    station_vel_sigma_m_s: float = 0.15
    stand_vx: float = 1.0
    stand_yaw: float = 1.5
    stand_wheel_speed: float = 0.08
    # Zero-command mode may use common-mode wheel torque for balance, but it
    # should remain small and quiet instead of driving the body into a pitch
    # limit cycle. Differential and leg channels are hard-gated in the env.
    stand_common_action: float = 0.10
    stand_pitch_rate: float = 0.20
    stand_pitch_rate_sigma: float = 0.15
    stand_action: float = 5.0
    # ★ 站立"别动"的核心项：机体前后加速度（由 pitch 与轮力矩直接决定）。
    #   验收是"机身保持不动"，所以要对**运动本身**下重手，而不只是位置误差。
    stand_accel: float = 0.5
    # 实测标定协同控制器在 125 Hz 上的 body_accel RMS ≈ 0.75 m/s²
    # （离散控制的固有抖动，不是漂移）。sigma 必须明显大于它，否则这一项
    # 会一直饱和，把"正常配平"当成振荡来罚。
    stand_accel_sigma: float = 2.0      # m/s^2
    # ---------------------------------------------------------------
    # ★ 教师跟踪项（本版新增，取代旧的"残差教师"）。
    #
    # 教师的期望电流是**完整的**平衡律输出，而不是"策略该补的那部分"：
    #   reward = w * exp(-mean((I_policy - I_teacher)^2) / sigma^2)
    # 这样目标在任何辅助强度下都是同一个状态函数，策略可以一次学会，
    # 退火只是把"实际下发"从教师平滑交到策略，不存在目标随辅助漂移的问题。
    # sigma = 3 A 覆盖常见残差（稳态 1~2 A），远离目标时也不会把 value 撑爆。
    # ---------------------------------------------------------------
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
    alive: float = 0.5


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

    def scaled(self, scale: float) -> "DomainRandomization":
        """把所有随机化区间按 ``scale`` 向标称值收缩（0 = 完全关闭，1 = 全量）。

        站立课程用它做**域随机化课程**：S0 先把扰动收窄到 25%（教师能覆盖），
        S1 60%，S2 100%。区间始终围绕标称值收缩，因此 12 维特权观测的归一化
        口径不变；critic 依然知道"这一局是什么机器"。
        """
        s = float(np.clip(scale, 0.0, 1.0))

        def shrink(lo, hi):
            mid = 0.5 * (lo + hi)
            return (mid + s * (lo - mid), mid + s * (hi - mid))

        return DomainRandomization(
            enabled=self.enabled and s > 0.0,
            base_mass_scale=shrink(*self.base_mass_scale),
            base_inertia_scale=shrink(*self.base_inertia_scale),
            base_com_offset_m=shrink(*self.base_com_offset_m),
            base_com_offset_z_m=self.base_com_offset_z_m * s,
            wheel_friction_scale=shrink(*self.wheel_friction_scale),
            joint_damping_scale=shrink(*self.joint_damping_scale),
            joint_kp_scale=shrink(*self.joint_kp_scale),
            joint_kd_scale=shrink(*self.joint_kd_scale),
            wheel_torque_scale=shrink(*self.wheel_torque_scale),
            wheel_speed_gain_scale=shrink(*self.wheel_speed_gain_scale),
            actuator_delay_steps=(0, int(round(self.actuator_delay_steps[1] * s))),
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
class ObservationNoise:
    """观测噪声（对齐官方 ``noise`` 段：``add_noise=True, noise_level=0.5``）。

    官方做法：噪声加在**归一化之后**的 actor 观测上，均匀分布
    ``±(noise_scales.X × noise_level × obs_scales.X)``；命令与上一帧动作
    **不加噪声**；特权观测不加噪声；加噪后才推进历史队列，
    因此增加历史帧时每帧噪声仍相互独立。

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
    # Temporal correlation of sensor noise. Independent white noise caused the
    # stand policy to chase frame-to-frame pitch-rate changes at 125 Hz.
    temporal_alpha: float = 0.15


@dataclass
class EnvParams:
    robot: RobotParams = field(default_factory=RobotParams)
    joint: JointActuatorParams = field(default_factory=JointActuatorParams)
    wheel: WheelEscParams = field(default_factory=WheelEscParams)
    balance: BalanceParams = field(default_factory=BalanceParams)
    rewards: RewardWeights = field(default_factory=RewardWeights)
    noise: ObservationNoise = field(default_factory=ObservationNoise)
    domain_randomization: DomainRandomization = field(default_factory=DomainRandomization)
    control_dt: float = 0.008       # 4 × 2 ms
    # 侧向速度终止阈值：两轮差动本身会产生偏航，旧值 0.80 m/s 在 S0 权限很小
    # 时也容易被探索噪声触发，放宽到 1.5 m/s（真正的侧滑远大于此）。
    terminate_on_drift: float = 1.00
    terminate_lateral_vel: float = 1.50
    # 站定要求（由 stand 分级课程覆盖）
    station_deadband: float = 0.03
    station_hard_limit: float = 0.10
    station_hold_steps: int = 10
    # ★ 腿长命令范围。当前训练与网页回放统一为 0.150~0.270 m；
    #   上限避开高位形的负载饱和区，给动态切换留恢复余量。
    leg_length_target_min: float = 0.150
    leg_length_target_max: float = 0.270
    # 站立时的高度安全线（相对额定高度的比例）。站立目标是动态直立，
    # 不能把低趴/折腿当成可接受姿态。
    height_fail_ratio: float = 0.75
    leg_length_fail_ratio: float = 0.60


def active_rewards(stage: StageSpec) -> dict[str, bool]:
    """把阶段的奖励组展开成 ``{组名: 是否启用}``。"""
    enabled = set(stage.reward_groups)
    return {group: (group in enabled) for group in REWARD_GROUPS}
