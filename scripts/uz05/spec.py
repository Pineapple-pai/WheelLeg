"""UZ-05 的物理接口、观测、动作和训练阶段定义。

控制接口刻意保持简单并与开源轮腿实现一致：PPO 直接给出全部六个主动
执行器的目标，执行器只负责把目标变成真实的关节 PD 和轮速闭环。
这里不放任何高层平衡律或动作混合逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# Native MuJoCo wheel-joint rotation is opposite to positive body-forward x.
# The policy action uses body-forward-positive wheel speed; convert only at
# the actuator boundary so the action convention is shared with deployment.
WHEEL_ANGULAR_TO_BODY_X = -1.0
WHEEL_ACTION_FRAME = "body_forward_positive"
REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_XML = REPO_ROOT / "diagnostics" / "uz05.xml"
MESH_DIR = Path("/home/p/WheelLegMJCFReference/meshes/stl")

# Frozen deployment timing.  PPO owns targets at 125 Hz while the physical
# actuator interfaces are refreshed at 500 Hz.
POLICY_RATE_HZ = 125
MOTOR_CONTROL_RATE_HZ = 500
MOTOR_TICKS_PER_POLICY = MOTOR_CONTROL_RATE_HZ // POLICY_RATE_HZ
POLICY_DT = 1.0 / POLICY_RATE_HZ
MOTOR_CONTROL_DT = 1.0 / MOTOR_CONTROL_RATE_HZ

CAPABILITY_ORDER = (
    "stand", "low_speed", "high_speed", "steering", "rotation",
    "airborne", "stairs", "jump", "recovery",
)


@dataclass(frozen=True)
class StageSpec:
    name: str
    vx_range: tuple[float, float] = (0.0, 0.0)
    vy_range: tuple[float, float] = (0.0, 0.0)
    yaw_range: tuple[float, float] = (0.0, 0.0)
    base_height_range: tuple[float, float] = (0.23825, 0.23825)
    leg_length_range: tuple[float, float] | None = None
    zero_command_prob: float = 0.5
    reverse_prob: float = 0.0
    jump_prob: float = 0.0
    init_tilt: float = 0.01
    init_tilt_rate: float = 0.01
    init_vel: float = 0.02
    init_height_offset: float = 0.0
    init_airborne_prob: float = 0.0
    init_airborne_height: tuple[float, float] = (0.0, 0.0)
    reward_groups: tuple[str, ...] = ()
    tilt_limit: float = 0.30
    tilt_limit_hold_steps: int = 0
    terrain: str = "flat"
    episode_steps: int = 1000


STAGES: tuple[StageSpec, ...] = (
    StageSpec(
        "stand", vx_range=(0.0, 0.0), zero_command_prob=1.0,
        base_height_range=(0.23825, 0.23825),
        leg_length_range=None, init_tilt=0.03,
        init_tilt_rate=0.02, init_vel=0.02,
        reward_groups=("posture", "regularization", "limits", "stand_still"),
    ),
    StageSpec(
        "low_speed", vx_range=(0.05, 0.30), zero_command_prob=0.45,
        reverse_prob=0.50, init_tilt=0.05, init_tilt_rate=0.03, init_vel=0.05,
        reward_groups=("posture", "regularization", "limits", "track_vx",
                       "yaw_lock", "stand_still"),
    ),
    StageSpec(
        "high_speed", vx_range=(0.10, 1.20), zero_command_prob=0.30,
        reverse_prob=0.40, init_tilt=0.05, init_tilt_rate=0.04, init_vel=0.08,
        reward_groups=("posture", "regularization", "limits", "track_vx",
                       "yaw_lock", "stand_still"),
    ),
    StageSpec(
        "steering", vx_range=(0.05, 0.90), yaw_range=(0.3, 1.2),
        zero_command_prob=0.20, reverse_prob=0.35, init_tilt=0.06,
        init_tilt_rate=0.05, init_vel=0.10,
        reward_groups=("posture", "regularization", "limits", "track_vx",
                       "track_yaw", "stand_still"),
    ),
    StageSpec(
        "rotation", vx_range=(0.0, 0.40), yaw_range=(0.8, 3.0),
        zero_command_prob=0.15, init_tilt=0.06, init_tilt_rate=0.06,
        init_vel=0.10,
        reward_groups=("posture", "regularization", "limits", "track_vx",
                       "track_yaw", "stand_still"),
    ),
    StageSpec(
        "airborne", vx_range=(0.0, 0.90), yaw_range=(0.0, 1.5),
        zero_command_prob=0.15, init_tilt=0.60, init_tilt_rate=0.60,
        init_vel=0.40, init_airborne_prob=0.35, init_airborne_height=(0.05, 0.35),
        reward_groups=("posture", "regularization", "limits", "track_vx",
                       "track_yaw", "stand_still", "airborne"),
        tilt_limit=1.20, tilt_limit_hold_steps=25,
    ),
    StageSpec(
        "stairs", vx_range=(0.30, 1.00), yaw_range=(0.0, 0.8),
        zero_command_prob=0.10, reverse_prob=0.30, init_tilt=0.10,
        init_tilt_rate=0.10, init_vel=0.20,
        reward_groups=("posture", "regularization", "limits", "track_vx",
                       "track_yaw", "stand_still", "airborne", "terrain"),
        tilt_limit=0.80, tilt_limit_hold_steps=15, terrain="stairs",
        episode_steps=1200,
    ),
    StageSpec(
        "jump", vx_range=(0.0, 1.00), yaw_range=(0.0, 1.0),
        zero_command_prob=0.15, jump_prob=0.25, init_tilt=0.20,
        init_tilt_rate=0.20, init_vel=0.30, init_airborne_prob=0.20,
        init_airborne_height=(0.05, 0.25),
        reward_groups=("posture", "regularization", "limits", "track_vx",
                       "track_yaw", "stand_still", "airborne", "jump"),
        tilt_limit=1.00, tilt_limit_hold_steps=20, episode_steps=1200,
    ),
    StageSpec(
        "recovery", vx_range=(0.0, 0.80), yaw_range=(0.0, 1.0),
        zero_command_prob=0.25, init_tilt=3.10, init_tilt_rate=1.50,
        init_vel=0.50, init_airborne_prob=0.15,
        init_airborne_height=(0.05, 0.30),
        reward_groups=("posture", "regularization", "limits", "track_vx",
                       "track_yaw", "stand_still", "airborne", "recovery"),
        tilt_limit=3.20, tilt_limit_hold_steps=60, episode_steps=1500,
    ),
)
STAGE_BY_NAME = {stage.name: stage for stage in STAGES}
DEFAULT_STAGE = "stand"


@dataclass(frozen=True)
class StandLevel:
    level: int
    name: str
    deadband: float
    hard_limit: float
    hard_hold_steps: int
    tilt_limit: float
    dr_scale: float
    note: str


STAND_LEVELS: tuple[StandLevel, ...] = (
    StandLevel(0, "S0_open", 0.03, 0.30, 0, 0.30, 0.25,
               "宽松安全边界，PPO 直接学习六个主动目标"),
    StandLevel(1, "S1_train", 0.03, 0.18, 5, 0.30, 0.60,
               "中等域随机化与漂移边界"),
    StandLevel(2, "S2_accept", 0.02, 0.10, 10, 0.30, 1.00,
               "完整域随机化与实物部署接口验收"),
)
STAND_LEVEL_BY_INDEX = {level.level: level for level in STAND_LEVELS}


ACTOR_OBS_BLOCKS: tuple[tuple[str, int], ...] = (
    ("gravity", 3), ("base_ang_vel", 3), ("base_lin_vel_actor", 3),
    ("leg_joint_pos", 4), ("leg_joint_vel", 4), ("wheel_joint_vel", 2),
    ("command", 5), ("station_error", 1), ("previous_action", 6),
    ("phase", 2), ("mode_onehot", 5),
)
PRIV_OBS_BLOCKS: tuple[tuple[str, int], ...] = (
    ("base_lin_vel", 3), ("base_pos_rel", 2), ("leg_length", 4),
    ("wheel_contact_force", 2), ("contact_flag", 4), ("terrain_scan", 17),
    ("dof_acc", 6), ("torques", 6), ("domain_params", 12),
)
OBS_BLOCKS = ACTOR_OBS_BLOCKS + PRIV_OBS_BLOCKS
OBS_HISTORY = 1
ACTOR_FRAME_DIM = sum(width for _, width in ACTOR_OBS_BLOCKS)
ACTOR_OBS_DIM = ACTOR_FRAME_DIM * OBS_HISTORY
PRIV_OBS_DIM = sum(width for _, width in PRIV_OBS_BLOCKS)
OBS_DIM = ACTOR_OBS_DIM + PRIV_OBS_DIM
OBS_SLICES: dict[str, slice] = {}
_offset = 0
for _name, _width in OBS_BLOCKS:
    OBS_SLICES[_name] = slice(_offset, _offset + _width)
    _offset += _width

OBS_SCALE: dict[str, object] = {
    "gravity": 1.0, "base_ang_vel": 0.25, "base_lin_vel": 2.0,
    "base_lin_vel_actor": 2.0, "base_pos_rel": 1.0, "leg_joint_pos": 1.0,
    "leg_joint_vel": 0.05, "wheel_joint_vel": 0.05,
    "command": [2.0, 1.0, 0.25, 5.0, 1.0], "station_error": 10.0,
    "previous_action": 1.0, "leg_length": 1.0,
    "wheel_contact_force": 0.01, "contact_flag": 1.0,
    "terrain_scan": 5.0, "dof_acc": 0.0025, "torques": 0.05,
    "domain_params": 1.0, "phase": 1.0, "mode_onehot": 1.0,
}
DOF_ACC_WINDOW = 8


def observation_noise_vector(noise: "ObservationNoise") -> dict[str, object]:
    zero = 0.0
    if not noise.enabled:
        return {name: zero for name, _ in OBS_BLOCKS}
    return {
        "gravity": noise.gravity * noise.level,
        "base_ang_vel": noise.ang_vel * noise.level * OBS_SCALE["base_ang_vel"],
        "base_lin_vel_actor": noise.lin_vel * noise.level * OBS_SCALE["base_lin_vel_actor"],
        "leg_joint_pos": noise.dof_pos * noise.level,
        "leg_joint_vel": noise.dof_vel * noise.level * OBS_SCALE["leg_joint_vel"],
        "wheel_joint_vel": noise.dof_vel * noise.level * OBS_SCALE["wheel_joint_vel"],
        "command": zero, "station_error": zero, "previous_action": zero,
        "phase": zero, "mode_onehot": zero,
        "base_lin_vel": zero, "base_pos_rel": zero, "leg_length": zero,
        "wheel_contact_force": zero, "contact_flag": zero, "terrain_scan": zero,
        "dof_acc": zero, "torques": zero, "domain_params": zero,
    }


# [left/right body-forward-positive wheel speed, four absolute hip targets].
ACTION_SPEC: tuple[tuple[str, int, float], ...] = (
    ("wheel_velocity_target", 2, 8.0),
    ("hip_position_target", 4, 1.2),
)
ACTION_DIM = sum(width for _, width, _ in ACTION_SPEC)
ACTION_SLICES: dict[str, slice] = {}
_offset = 0
for _name, _width, _scale in ACTION_SPEC:
    ACTION_SLICES[_name] = slice(_offset, _offset + _width)
    _offset += _width

REWARD_GROUPS = (
    "posture", "regularization", "limits", "track_vx", "yaw_lock",
    "track_yaw", "stand_still", "airborne", "terrain", "jump", "recovery",
)


@dataclass
class JointActuatorParams:
    control_mode: str = "mit"
    kp: float = 100.0
    kd: float = 4.0
    torque_limit: float = 20.0
    torque_per_amp: float = 1.0
    velocity_limit: float = 10.47
    velocity_target: float = 0.0
    torque_feedforward: float = 0.0
    mit_position_range: tuple[float, float] = (-12.5, 12.5)
    mit_velocity_range: tuple[float, float] = (-30.0, 30.0)
    mit_kp_range: tuple[float, float] = (0.0, 500.0)
    mit_kd_range: tuple[float, float] = (0.0, 5.0)
    mit_torque_range: tuple[float, float] = (-20.0, 20.0)


@dataclass
class WheelEscParams:
    # The PPO action is a wheel-speed target, so this PI loop must have
    # enough authority to create the balancing torque.  The previous
    # 0.081 A/(rad/s) gain could produce at most about 0.20 Nm through the
    # proportional path after the C620/current and efficiency model, which
    # made direct-policy standing effectively under-actuated.
    speed_kp_a_per_rad_s: float = 0.60
    speed_ki_a_per_rad_s2: float = 0.30
    speed_error_limit: float = 14.0
    integral_limit_a: float = 2.0
    torque_per_amp_joint: float = 0.246
    current_limit_a: float = 10.0
    velocity_limit: float = 59.79
    no_load_velocity: float = 61.47
    stall_torque_limit: float = 3.69
    joint_torque_limit: float = 2.46
    efficiency: float = 0.70
    gear_ratio: float = 268.0 / 17.0
    c620_full_scale_current_a: float = 20.0
    c620_full_scale_command: int = 16384
    torque_speed_envelope_enabled: bool = False


@dataclass
class RobotParams:
    wheel_radius: float = 0.055
    leg_length_min: float = 0.150
    leg_length_max: float = 0.340
    hip_joint_limit: float = 1.2
    tendon_limit: float = 0.388
    reset_joint_pos: tuple[float, ...] = (
        -0.19450, 0.27196, -0.15812, -0.27196, 0.27200, 0.0, 0.19129,
        -0.19141, 0.26949, -0.15698, -0.26947, 0.26959, 0.0, 0.19057,
    )
    stand_joint_pos: tuple[float, float, float, float] = (
        -0.19450, 0.19129, -0.19141, 0.19057,
    )
    reset_height: float = 0.23825
    nominal_stand_height: float = 0.23825
    nominal_leg_length: float = 0.18413
    control_substeps: int = MOTOR_TICKS_PER_POLICY
    command_accel_limit: float = 0.75
    command_yaw_accel_limit: float = 4.0


@dataclass
class RewardWeights:
    upright: float = 8.0
    upright_progress: float = 4.0
    height: float = 2.0
    leg_length: float = 2.0
    leg_length_progress: float = 0.0
    leg_length_rate: float = 0.0
    leg_length_rate_sigma_m_s: float = 0.04
    joint_neutral: float = 1.0
    action_rate: float = 0.05
    leg_action: float = 0.10
    joint_velocity: float = 0.01
    joint_torque: float = 1e-4
    wheel_power: float = 1e-4
    wheel_slip: float = 0.14
    pitch_speed_coupling: float = 1.0
    pitch_speed_sigma_rad: float = 0.10
    pitch_speed_gain_rad_per_m_s: float = 0.40
    leg_symmetry: float = 20.0
    wheel_differential: float = 0.05
    wheel_speed: float = 0.02
    wheel_current_rate: float = 0.15
    wheel_current_jerk: float = 3.0
    leg_action_rate: float = 2.0
    joint_limit: float = 5.0
    leg_length_limit: float = 20.0
    track_vx: float = 6.0
    # Absolute error floor for low-speed tracking; the relative term below
    # widens the reward band only when larger commands need it.
    low_speed_track_sigma_m_s: float = 0.05
    # Gradually widen the Gaussian tracking band for larger velocity targets
    # so PPO still receives a useful magnitude gradient near 0.2–0.3 m/s.
    low_speed_track_sigma_relative: float = 0.35
    track_vx_wide_sigma: float = 1.0
    track_vx_tight: float = 1.5
    track_vx_tight_sigma: float = 0.10
    track_vx_square: float = 3.0
    track_vx_gap: float = 4.0
    # Kept as named diagnostics/config fields for compatibility; the primary
    # bounded `track_vx` term handles magnitude error without duplicate costs.
    track_vx_error: float = 0.0
    track_vx_forward_error: float = 0.0
    track_vx_reverse_error: float = 0.0
    # Dense signed alignment gives PPO a useful gradient around vx=0 without
    # reintroducing multiple copies of the same velocity-error penalty.
    track_vx_progress: float = 2.0
    # Legacy shaping fields remain for checkpoint/config compatibility.  The
    # current low-speed task uses one tracking term plus posture safety.
    low_speed_startup_progress: float = 0.0
    low_speed_cruise_progress: float = 0.0
    low_speed_stable_motion: float = 3.0
    low_speed_stable_pitch_sigma_rad: float = 0.12
    low_speed_stable_pitch_rate_sigma_rad_s: float = 0.8
    low_speed_stable_penalty_cap: float = 4.0
    low_speed_posture_gate_gain: float = 1.0
    low_speed_speed_window_steps: int = 32
    low_speed_startup_steps: int = 64
    yaw_lock: float = 1.5
    yaw_lock_sigma: float = 0.20
    track_vy: float = 1.6
    track_yaw: float = 2.0
    track_yaw_square: float = 0.5
    wrong_direction: float = 4.0
    station: float = 3.0
    station_sigma_m: float = 0.04
    station_progress: float = 40.0
    station_vel: float = 1.0
    station_vel_sigma_m_s: float = 0.15
    stand_vx: float = 1.0
    stand_yaw: float = 1.5
    stand_wheel_speed: float = 0.08
    stand_common_action: float = 0.10
    stand_pitch_rate: float = 0.20
    stand_pitch_rate_sigma: float = 0.15
    stand_action: float = 0.05
    stand_accel: float = 0.5
    stand_accel_sigma: float = 2.0
    # Additional shaping used only by the zero-command stand stage.  Keeping
    # these separate from shared regularizers avoids suppressing jump takeoff.
    stand_wheel_contact: float = 3.0
    stand_leg_target_rate: float = 0.5
    stand_leg_target_rate_sigma_rad_s: float = 10.0
    stand_leg_length_rate: float = 0.35
    airborne_upright: float = 2.0
    airborne_leg_retract: float = 1.0
    landing_impact: float = 2.0
    undesired_contact: float = 5.0
    terrain_progress: float = 2.0
    front_wheel_height: float = 1.0
    jump_height: float = 5.0
    jump_phase_time: float = 0.5
    recovery_upright: float = 10.0
    recovery_progress: float = 5.0
    termination: float = 200.0
    alive: float = 0.5


@dataclass
class DomainRandomization:
    enabled: bool = True
    base_mass_scale: tuple[float, float] = (0.90, 1.25)
    base_inertia_scale: tuple[float, float] = (0.80, 1.20)
    base_com_offset_m: tuple[float, float] = (0.04, 0.04)
    base_com_offset_z_m: float = 0.02
    wheel_friction_scale: tuple[float, float] = (0.70, 1.30)
    joint_damping_scale: tuple[float, float] = (0.75, 1.25)
    joint_kp_scale: tuple[float, float] = (0.80, 1.20)
    joint_kd_scale: tuple[float, float] = (0.80, 1.20)
    wheel_torque_scale: tuple[float, float] = (0.85, 1.15)
    wheel_speed_gain_scale: tuple[float, float] = (0.80, 1.20)
    actuator_delay_steps: tuple[int, int] = (0, 1)
    observation_delay_steps: tuple[int, int] = (1, 3)
    PARAM_NAMES: tuple[str, ...] = (
        "base_mass_scale", "base_inertia_scale", "base_com_x", "base_com_y",
        "base_com_z", "wheel_friction_scale", "joint_damping_scale",
        "joint_kp_scale", "joint_kd_scale", "wheel_torque_scale",
        "wheel_speed_gain_scale", "actuator_delay",
    )

    def scaled(self, scale: float) -> "DomainRandomization":
        s = float(np.clip(scale, 0.0, 1.0))

        def shrink(lo, hi):
            mid = 0.5 * (lo + hi)
            return mid + s * (lo - mid), mid + s * (hi - mid)

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
            observation_delay_steps=(0, int(round(self.observation_delay_steps[1] * s))),
        )

    def sample(self, rng):
        def norm(value, lo, hi):
            return 0.0 if hi <= lo else float(2 * (value - lo) / (hi - lo) - 1)

        out = {}
        for name, bounds in (
            ("base_mass_scale", self.base_mass_scale),
            ("base_inertia_scale", self.base_inertia_scale),
            ("wheel_friction_scale", self.wheel_friction_scale),
            ("joint_damping_scale", self.joint_damping_scale),
            ("joint_kp_scale", self.joint_kp_scale),
            ("joint_kd_scale", self.joint_kd_scale),
            ("wheel_torque_scale", self.wheel_torque_scale),
            ("wheel_speed_gain_scale", self.wheel_speed_gain_scale),
        ):
            value = float(rng.uniform(*bounds))
            out[name] = (value, norm(value, *bounds))
        for name, limit in (("base_com_x", self.base_com_offset_m[0]),
                            ("base_com_y", self.base_com_offset_m[1]),
                            ("base_com_z", self.base_com_offset_z_m)):
            value = float(rng.uniform(-limit, limit))
            out[name] = (value, norm(value, -limit, limit))
        delay = int(rng.integers(self.actuator_delay_steps[0],
                                 self.actuator_delay_steps[1] + 1))
        out["actuator_delay"] = (float(delay), norm(
            delay, *self.actuator_delay_steps))
        obs_delay = int(rng.integers(self.observation_delay_steps[0],
                                     self.observation_delay_steps[1] + 1))
        out["observation_delay"] = (float(obs_delay), norm(
            obs_delay, *self.observation_delay_steps))
        return out


@dataclass
class ObservationNoise:
    enabled: bool = True
    level: float = 0.5
    ang_vel: float = 0.2
    gravity: float = 0.05
    dof_pos: float = 0.01
    dof_vel: float = 1.5
    lin_vel: float = 0.1
    clip_observations: float = 100.0
    temporal_alpha: float = 0.15


@dataclass
class EnvParams:
    robot: RobotParams = field(default_factory=RobotParams)
    joint: JointActuatorParams = field(default_factory=JointActuatorParams)
    wheel: WheelEscParams = field(default_factory=WheelEscParams)
    rewards: RewardWeights = field(default_factory=RewardWeights)
    noise: ObservationNoise = field(default_factory=ObservationNoise)
    domain_randomization: DomainRandomization = field(default_factory=DomainRandomization)
    control_dt: float = POLICY_DT
    motor_control_dt: float = MOTOR_CONTROL_DT
    terminate_on_drift: float = 1.00
    terminate_lateral_vel: float = 1.50
    station_deadband: float = 0.03
    station_hard_limit: float = 0.10
    station_hold_steps: int = 10
    leg_length_target_min: float = 0.150
    leg_length_target_max: float = 0.270
    height_fail_ratio: float = 0.75
    leg_length_fail_ratio: float = 0.60

    def __post_init__(self) -> None:
        ratio = self.control_dt / self.motor_control_dt
        if not np.isclose(ratio, self.robot.control_substeps):
            raise ValueError(
                "control_dt / motor_control_dt must equal robot.control_substeps"
            )
        if self.joint.control_mode.lower() != "mit":
            raise ValueError("UZ-05 deployment contract requires DM MIT mode")


def active_rewards(stage: StageSpec) -> dict[str, bool]:
    enabled = set(stage.reward_groups)
    return {group: group in enabled for group in REWARD_GROUPS}
