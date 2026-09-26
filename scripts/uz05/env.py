"""UZ-05 MuJoCo environment with direct PPO actuator targets."""

from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation

from .actuators import HIP_POSITION_SCALE, ActuatorBank, JointPD, WheelSpeedLoop
from .model import TerrainSpec, UZ05Model
from .spec import (
    ACTION_DIM,
    ACTOR_OBS_BLOCKS,
    ACTOR_OBS_DIM,
    ACTOR_FRAME_DIM,
    ACTION_SLICES,
    DEFAULT_STAGE,
    DOF_ACC_WINDOW,
    EnvParams,
    OBS_BLOCKS,
    OBS_DIM,
    OBS_HISTORY,
    OBS_SCALE,
    STAND_LEVELS,
    STAGE_BY_NAME,
    StageSpec,
    active_rewards,
    observation_noise_vector,
    WHEEL_ANGULAR_TO_BODY_X,
)

TERRAIN_SCAN_POINTS = 17
MODE_NAMES = ("normal", "airborne", "stair", "recover", "jump")
_BLOCK_ORDER = tuple(name for name, _ in OBS_BLOCKS)
_BLOCK_WIDTH = {name: width for name, width in OBS_BLOCKS}
_ACTOR_BLOCK_NAMES = frozenset(name for name, _ in ACTOR_OBS_BLOCKS)
_NOISE_VEC = observation_noise_vector(EnvParams().noise)


class UZ05Env(gym.Env):
    """Two-wheel legged robot task.

    The policy action is sent to the two physical interfaces directly.  There
    is no hidden action replacement in standing or motion stages.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        stage: str = DEFAULT_STAGE,
        params: EnvParams | None = None,
        seed: int | None = None,
        stand_level: int = 2,
        init_scale: float = 1.0,
        height_switch_steps: int = 0,
        height_switch_prob: float = 0.0,
        height_target_rate_m_s: float = 0.0,
        leg_length_reward_weight: float | None = None,
        leg_length_progress_reward_weight: float | None = None,
        leg_length_rate_reward_weight: float | None = None,
        low_speed_stable_motion_weight: float | None = None,
        low_speed_posture_gate_gain: float | None = None,
        low_speed_track_sigma_relative: float | None = None,
        extra_init_tilt: float = 0.0,
        extra_init_tilt_rate: float = 0.0,
        extra_init_vel: float = 0.0,
        vx_range_override: tuple[float, float] | None = None,
        zero_command_prob_override: float | None = None,
        reverse_prob_override: float | None = None,
        command_accel_limit_override: float | None = None,
        vx_reward_scale: float = 1.0,
        low_speed_reverse_reward_scale: float = 1.0,
        low_speed_wrong_direction_weight: float = 0.0,
        low_speed_wheel_differential_weight: float | None = None,
        tilt_limit_override: float | None = None,
        deployment_mode: bool = False,
        observation_delay_steps: tuple[int, int] | None = None,
        actuator_delay_steps: tuple[int, int] | None = None,
        tilt_hold_steps: int | None = None,
        wheel_speed_kp: float | None = None,
        wheel_speed_ki: float | None = None,
        wheel_integral_limit: float | None = None,
        episode_steps_override: int | None = None,
    ):
        super().__init__()
        if stage not in STAGE_BY_NAME:
            raise ValueError(f"unknown stage {stage!r}; known: {sorted(STAGE_BY_NAME)}")
        self.stage: StageSpec = STAGE_BY_NAME[stage]
        self.episode_steps_override = None
        self.set_episode_steps(episode_steps_override)
        self.params = params or EnvParams()
        self.rng = np.random.default_rng(seed)
        self.deployment_mode = bool(deployment_mode)
        self.tilt_hold_steps = (
            int(self.stage.tilt_limit_hold_steps)
            if tilt_hold_steps is None
            else max(0, int(tilt_hold_steps))
        )
        if wheel_speed_kp is not None:
            self.params.wheel.speed_kp_a_per_rad_s = max(0.0, float(wheel_speed_kp))
        if wheel_speed_ki is not None:
            self.params.wheel.speed_ki_a_per_rad_s2 = max(0.0, float(wheel_speed_ki))
        if wheel_integral_limit is not None:
            self.params.wheel.integral_limit_a = max(1e-6, float(wheel_integral_limit))
        self.observation_delay_steps_override = self._validate_delay_range(
            observation_delay_steps, "observation_delay_steps"
        )
        self.actuator_delay_steps_override = self._validate_delay_range(
            actuator_delay_steps, "actuator_delay_steps"
        )
        if self.deployment_mode:
            self.params.wheel.torque_speed_envelope_enabled = True

        self.stand_level = next(
            (level for level in STAND_LEVELS if level.level == int(stand_level)),
            STAND_LEVELS[-1],
        )
        self.params.station_deadband = self.stand_level.deadband
        self.params.station_hard_limit = self.stand_level.hard_limit
        self.params.station_hold_steps = self.stand_level.hard_hold_steps
        self.init_scale = float(np.clip(init_scale, 0.0, 1.0))
        self.extra_init_tilt = float(extra_init_tilt)
        self.extra_init_tilt_rate = float(extra_init_tilt_rate)
        self.extra_init_vel = float(extra_init_vel)
        self.vx_range_override = self._validate_vx_range(vx_range_override)
        self.zero_command_prob_override = (
            None if zero_command_prob_override is None
            else float(np.clip(zero_command_prob_override, 0.0, 1.0))
        )
        self.reverse_prob_override = (
            None if reverse_prob_override is None
            else float(np.clip(reverse_prob_override, 0.0, 1.0))
        )
        self.command_accel_limit = float(
            self.params.robot.command_accel_limit
            if command_accel_limit_override is None
            else command_accel_limit_override
        )
        if self.command_accel_limit <= 0.0:
            raise ValueError("command_accel_limit_override must be positive")
        self.vx_reward_scale = float(np.clip(vx_reward_scale, 0.0, 1.0))
        self.low_speed_reverse_reward_scale = max(0.0, float(low_speed_reverse_reward_scale))
        self.low_speed_wrong_direction_weight = max(
            0.0, float(low_speed_wrong_direction_weight)
        )
        self.low_speed_wheel_differential_weight = (
            None if low_speed_wheel_differential_weight is None
            else max(0.0, float(low_speed_wheel_differential_weight))
        )
        self.height_switch_steps = max(0, int(height_switch_steps))
        self.height_switch_prob = float(np.clip(height_switch_prob, 0.0, 1.0))
        self.height_target_rate_m_s = max(0.0, float(height_target_rate_m_s))
        if leg_length_reward_weight is not None:
            self.params.rewards.leg_length = max(0.0, float(leg_length_reward_weight))
        if leg_length_progress_reward_weight is not None:
            self.params.rewards.leg_length_progress = max(
                0.0, float(leg_length_progress_reward_weight)
            )
        if leg_length_rate_reward_weight is not None:
            self.params.rewards.leg_length_rate = max(
                0.0, float(leg_length_rate_reward_weight)
            )
        if low_speed_stable_motion_weight is not None:
            self.params.rewards.low_speed_stable_motion = max(
                0.0, float(low_speed_stable_motion_weight)
            )
        if low_speed_posture_gate_gain is not None:
            self.params.rewards.low_speed_posture_gate_gain = max(
                0.0, float(low_speed_posture_gate_gain)
            )
        if low_speed_track_sigma_relative is not None:
            self.params.rewards.low_speed_track_sigma_relative = max(
                0.0, float(low_speed_track_sigma_relative)
            )

        default_tilt = self.stand_level.tilt_limit if stage == "stand" else self.stage.tilt_limit
        self.tilt_limit = default_tilt if tilt_limit_override is None else max(
            0.05, float(tilt_limit_override)
        )
        if self.stage.leg_length_range is not None:
            lo, hi = self.stage.leg_length_range
            self._leg_height_range = (
                max(lo, self.params.leg_length_target_min)
                + self.params.robot.wheel_radius,
                min(hi, self.params.leg_length_target_max)
                + self.params.robot.wheel_radius,
            )
        else:
            self._leg_height_range = None
        self.leg_length_override: float | None = None

        self.sim = UZ05Model(TerrainSpec(kind=self.stage.terrain))
        if not np.isclose(self.sim.model.opt.timestep, self.params.motor_control_dt):
            raise ValueError(
                "MuJoCo timestep must match the 500 Hz motor control period: "
                f"{self.sim.model.opt.timestep} != {self.params.motor_control_dt}"
            )
        self.actuators = ActuatorBank(
            robot=self.params.robot,
            joint=JointPD(self.params.joint, quantize_mit=self.deployment_mode),
            wheel=WheelSpeedLoop(self.params.wheel, quantize_c620=self.deployment_mode),
        )
        self._nominal_actuator = {
            "joint_kp": float(self.params.joint.kp),
            "joint_kd": float(self.params.joint.kd),
            "wheel_torque_per_amp_joint": float(self.params.wheel.torque_per_amp_joint),
            "wheel_speed_kp_a_per_rad_s": float(
                self.params.wheel.speed_kp_a_per_rad_s
            ),
        }
        self.action_space = spaces.Box(-1.0, 1.0, (ACTION_DIM,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        self.reward_flags = active_rewards(self.stage)

        self.steps = 0
        self.previous_action = np.zeros(ACTION_DIM, dtype=np.float64)
        self.command = np.zeros(5, dtype=np.float64)
        self.command_target = np.zeros(5, dtype=np.float64)
        self.nominal_xy = np.zeros(2, dtype=np.float64)
        self.prev_leg_lengths = self.sim.leg_lengths()
        self._leg_rate_for_control = 0.0
        self.tilt_exceed_steps = 0
        self.station_exceed_steps = 0
        self.airborne_prev = 0.0
        self.airborne_steps = 0
        self.jump_phase = 0
        self.jump_timer = 0.0
        self.phase_clock = 0.0
        self.last_landing_impact = 0.0
        self._upright_potential = 1.0
        self._leg_error_potential = 0.0
        self._station_potential = 0.0
        self._prev_body_vx = 0.0
        self._last_step_body_vx = 0.0
        self._commanded_x = 0.0
        self._translation_episode_active = False
        self._station_max_abs = 0.0
        self._motion_max_abs = 0.0
        self._station_window: deque = deque(maxlen=200)
        self._motion_window: deque = deque(maxlen=200)
        self._motion_vx_window: deque = deque(
            maxlen=max(1, int(self.params.rewards.low_speed_speed_window_steps))
        )
        self._actor_history: deque = deque(maxlen=OBS_HISTORY)
        self._sensor_frame_history: deque = deque(maxlen=16)
        self._action_delay_history: deque = deque(maxlen=16)
        self._observation_delay_steps = 0
        self._actuator_delay_steps = 0
        self._sensor_noise_state: np.ndarray | None = None
        self._dof_vel_hist: deque = deque(maxlen=DOF_ACC_WINDOW + 1)
        self._last_leg_torque = np.zeros(4, dtype=np.float64)
        self._last_wheel_torque = np.zeros(2, dtype=np.float64)
        self._last_wheel_current = np.zeros(2, dtype=np.float64)
        self._height_switch_count = 0
        self._height_last_switch_step = -1
        self._height_settle_steps = -1
        self._raw_cache: dict[str, np.ndarray] = {}
        self._obs_cache: dict[str, np.ndarray] = {}
        self._last_reward_terms: dict[str, float] = {}
        self._info: dict[str, Any] = {}
        self._nominal_model = self.sim.nominal()
        self.domain: dict = {}

    @staticmethod
    def _validate_delay_range(value, name: str):
        if value is None:
            return None
        if len(value) != 2:
            raise ValueError(f"{name} must be (min, max)")
        lo, hi = (int(v) for v in value)
        if lo < 0 or hi < lo:
            raise ValueError(f"{name} must satisfy 0 <= min <= max")
        return lo, hi

    @staticmethod
    def _validate_vx_range(value):
        if value is None:
            return None
        lo, hi = (float(v) for v in value)
        if lo < 0.0 or hi < lo:
            raise ValueError("vx range must satisfy 0 <= min <= max")
        return lo, hi

    def set_init_scale(self, value: float) -> None:
        self.init_scale = float(np.clip(value, 0.0, 1.0))

    def set_episode_steps(self, value: int | None) -> None:
        if value is not None and int(value) <= 0:
            raise ValueError("episode_steps_override must be positive")
        self.episode_steps_override = None if value is None else int(value)

    def set_vx_range(self, value) -> None:
        self.vx_range_override = self._validate_vx_range(value)

    def set_zero_command_prob(self, value: float) -> None:
        self.zero_command_prob_override = float(np.clip(value, 0.0, 1.0))

    def set_reverse_prob(self, value: float) -> None:
        self.reverse_prob_override = float(np.clip(value, 0.0, 1.0))

    def set_vx_reward_scale(self, value: float) -> None:
        self.vx_reward_scale = float(np.clip(value, 0.0, 1.0))

    def get_vx_range(self):
        return tuple(self.vx_range_override or self.stage.vx_range)

    def get_init_scale(self) -> float:
        return float(self.init_scale)

    def _sample_command(self) -> None:
        height_range = self._leg_height_range or self.stage.base_height_range
        zero_prob = (
            self.stage.zero_command_prob
            if self.zero_command_prob_override is None
            else self.zero_command_prob_override
        )
        reverse_prob = (
            self.stage.reverse_prob
            if self.reverse_prob_override is None
            else self.reverse_prob_override
        )
        vx_range = self.vx_range_override or self.stage.vx_range
        if self.rng.random() < zero_prob:
            target = np.zeros(5, dtype=np.float64)
        else:
            vx = float(self.rng.uniform(*vx_range))
            if reverse_prob and self.rng.random() < reverse_prob:
                vx = -vx
            vy = float(self.rng.uniform(*self.stage.vy_range)) if self.stage.vy_range[1] else 0.0
            yaw = float(self.rng.uniform(*self.stage.yaw_range))
            if self.rng.random() < 0.5:
                yaw = -yaw
            target = np.array([vx, vy, yaw, 0.0, 0.0], dtype=np.float64)
        target[3] = float(self.rng.uniform(*height_range))
        if self.stage.jump_prob and self.rng.random() < self.stage.jump_prob:
            target[4] = 1.0
        self.command_target = target
        self.command[:] = target
        self.command[0] = 0.0
        self.command[2] = 0.0
        self._translation_episode_active = (
            self.stage.name == "low_speed" and abs(float(target[0])) > 0.01
        )

    def _maybe_switch_height_target(self) -> None:
        if (self.stage.name == "stand" or self.height_switch_steps <= 0
                or self.height_switch_prob <= 0.0
                or self.steps <= 0 or self.steps % self.height_switch_steps):
            return
        if self.rng.random() > self.height_switch_prob:
            return
        height_range = self._leg_height_range or self.stage.base_height_range
        self.command_target[3] = float(self.rng.uniform(*height_range))
        self._height_switch_count += 1
        self._height_last_switch_step = int(self.steps)
        self._height_settle_steps = -1

    def _advance_command(self) -> None:
        dt = self.params.control_dt
        dv = self.command_accel_limit * dt
        self.command[0] += np.clip(self.command_target[0] - self.command[0], -dv, dv)
        dyaw = self.params.robot.command_yaw_accel_limit * dt
        self.command[2] += np.clip(self.command_target[2] - self.command[2], -dyaw, dyaw)
        if self.height_target_rate_m_s > 0.0:
            dh = self.height_target_rate_m_s * dt
            self.command[3] += np.clip(self.command_target[3] - self.command[3], -dh, dh)
        else:
            self.command[3] = self.command_target[3]
        self.command[1] = self.command_target[1]
        self.command[4] = self.command_target[4]
        if self.command[4] > 0.5 and self.jump_phase == 0:
            self.jump_phase, self.jump_timer = 1, 0.24

    @property
    def command_active(self) -> bool:
        return bool(abs(self.command[0]) > 0.01 or abs(self.command[2]) > 0.01)

    @property
    def motion_control_active(self) -> bool:
        if self.stage.name == "low_speed":
            return self._translation_episode_active
        return self.command_active

    @property
    def leg_length_command(self) -> float:
        if self.leg_length_override is not None:
            return float(self.leg_length_override)
        if self.stage.name == "stand":
            return float(self.params.robot.nominal_leg_length)
        return float(self.command[3] - self.params.robot.wheel_radius)

    @property
    def leg_length_range(self):
        if self._leg_height_range is None:
            return None
        return (
            float(self._leg_height_range[0] - self.params.robot.wheel_radius),
            float(self._leg_height_range[1] - self.params.robot.wheel_radius),
        )

    def set_leg_length_command(self, value: float | None) -> float:
        if value is None:
            self.leg_length_override = None
        else:
            lo, hi = self.leg_length_range or (
                self.params.leg_length_target_min,
                self.params.leg_length_target_max,
            )
            self.leg_length_override = float(np.clip(value, lo, hi))
            self.command_target[3] = self.leg_length_override + self.params.robot.wheel_radius
            self.command[3] = self.command_target[3]
        return self.leg_length_command

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        stage = self.stage
        self.sim.reset(self.params.robot.reset_height)
        airborne = bool(stage.init_airborne_prob and self.rng.random() < stage.init_airborne_prob)
        if airborne:
            self.sim.data.qpos[2] += float(self.rng.uniform(*stage.init_airborne_height))
        limit = stage.init_tilt * self.init_scale
        roll, pitch, yaw = self.rng.uniform(-limit, limit, 3)
        pitch += self.extra_init_tilt
        quat = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_quat()
        self.sim.data.qpos[3:7] = [quat[3], quat[0], quat[1], quat[2]]
        speed = stage.init_vel * self.init_scale
        self.sim.data.qvel[:3] = self.rng.uniform(-speed, speed, 3)
        rate = stage.init_tilt_rate * self.init_scale
        self.sim.data.qvel[3:6] = self.rng.uniform(-rate, rate, 3)
        self.sim.data.qvel[4] += self.extra_init_tilt_rate
        self.sim.data.qvel[0] += self.extra_init_vel
        self.sim.data.qpos[7:] = np.asarray(self.params.robot.reset_joint_pos, dtype=np.float64)
        self.sim.data.qpos[self.sim.hip_qpos_adr] += self.rng.uniform(-0.02, 0.02, 4) * self.init_scale
        self.sim.data.qvel[self.sim.hip_dof_adr] = self.rng.uniform(-0.1, 0.1, 4) * self.init_scale

        dr = self.params.domain_randomization.scaled(self.stand_level.dr_scale)
        if dr.enabled:
            self.domain = dr.sample(self.rng)
            self.sim.apply_randomization(self._nominal_model, self.domain)
            self.actuators.joint.params.kp = self._nominal_actuator["joint_kp"] * self.domain["joint_kp_scale"][0]
            self.actuators.joint.params.kd = self._nominal_actuator["joint_kd"] * self.domain["joint_kd_scale"][0]
            self.actuators.wheel.params.torque_per_amp_joint = self._nominal_actuator["wheel_torque_per_amp_joint"] * self.domain["wheel_torque_scale"][0]
            self.actuators.wheel.params.speed_kp_a_per_rad_s = self._nominal_actuator["wheel_speed_kp_a_per_rad_s"] * self.domain["wheel_speed_gain_scale"][0]
        else:
            self.domain = {name: (0.0, 0.0) for name in dr.PARAM_NAMES}
            self.sim.apply_randomization(self._nominal_model, {
                name: (1.0 if "scale" in name else 0.0, 0.0)
                for name in dr.PARAM_NAMES
            })
            self.actuators.joint.params.kp = self._nominal_actuator["joint_kp"]
            self.actuators.joint.params.kd = self._nominal_actuator["joint_kd"]
            self.actuators.wheel.params.torque_per_amp_joint = self._nominal_actuator["wheel_torque_per_amp_joint"]
            self.actuators.wheel.params.speed_kp_a_per_rad_s = self._nominal_actuator["wheel_speed_kp_a_per_rad_s"]
        if self.deployment_mode:
            obs_range = self.observation_delay_steps_override or dr.observation_delay_steps
            act_range = self.actuator_delay_steps_override or dr.actuator_delay_steps
            self._observation_delay_steps = int(self.rng.integers(obs_range[0], obs_range[1] + 1))
            self._actuator_delay_steps = int(self.rng.integers(act_range[0], act_range[1] + 1))
        else:
            self._observation_delay_steps = self._actuator_delay_steps = 0
        self.sim.forward()

        self.nominal_xy = self.sim.data.qpos[:2].copy()
        self.steps = 0
        self._height_switch_count = 0
        self._height_last_switch_step = -1
        self._height_settle_steps = -1
        self.previous_action[:] = 0.0
        self.prev_leg_lengths = self.sim.leg_lengths()
        self._leg_rate_for_control = 0.0
        self._dof_vel_hist.clear()
        self._dof_vel_hist.append(np.concatenate([
            self.sim.joint_velocities(), self.sim.wheel_velocities()
        ]))
        self._last_leg_torque[:] = 0.0
        self._last_wheel_torque[:] = 0.0
        self._last_wheel_current[:] = 0.0
        self.tilt_exceed_steps = self.station_exceed_steps = 0
        self.jump_phase = 0
        self.jump_timer = 0.0
        self.phase_clock = 0.0
        self.last_landing_impact = 0.0
        self._actor_history.clear()
        self._sensor_frame_history.clear()
        self._action_delay_history.clear()
        self._sensor_noise_state = None
        self._station_window.clear()
        self._motion_window.clear()
        self._motion_vx_window.clear()
        self._station_max_abs = self._motion_max_abs = 0.0
        self._commanded_x = 0.0
        self._prev_body_vx = self._last_step_body_vx = 0.0
        self._station_potential = 0.0
        self.actuators.reset()
        self._sample_command()
        _, _, _, airborne_now = self.sim.contact_state()
        self.airborne_prev = airborne_now
        rpy = Rotation.from_quat([
            self.sim.base_quat[1], self.sim.base_quat[2],
            self.sim.base_quat[3], self.sim.base_quat[0]
        ]).as_euler("xyz")
        self._upright_potential = float(np.exp(-(rpy[0] ** 2 + rpy[1] ** 2) / 0.25 ** 2))
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        sim = self.sim
        rot = Rotation.from_quat([
            sim.base_quat[1], sim.base_quat[2], sim.base_quat[3], sim.base_quat[0]
        ])
        rpy = rot.as_euler("xyz")
        gravity = rot.inv().apply(np.array([0.0, 0.0, -1.0]))
        lin_vel = sim.body_frame(sim.base_lin_vel_world)
        ang_vel = sim.body_frame(sim.base_ang_vel_world)
        lengths = sim.leg_lengths()
        rates = (lengths - self.prev_leg_lengths) / self.params.control_dt
        wheel_force, wheel_hit, body_hit, airborne = sim.contact_state()
        dof_now = np.concatenate([sim.joint_velocities(), sim.wheel_velocities()])
        if len(self._dof_vel_hist) >= 2:
            span = (len(self._dof_vel_hist) - 1) * self.params.control_dt
            dof_acc = (self._dof_vel_hist[0] - self._dof_vel_hist[-1]) / span
        else:
            dof_acc = np.zeros(6, dtype=np.float64)
        blocks = {
            "gravity": gravity,
            "base_ang_vel": ang_vel,
            "base_lin_vel_actor": lin_vel,
            "base_lin_vel": lin_vel,
            "base_pos_rel": sim.data.qpos[:2] - self.nominal_xy,
            "leg_joint_pos": sim.joint_positions(),
            "leg_joint_vel": sim.joint_velocities(),
            "wheel_joint_vel": sim.wheel_velocities(),
            "command": self.command,
            "station_error": np.array([sim.data.qpos[0] - self.nominal_xy[0]]),
            "previous_action": self.previous_action,
            "leg_length": np.concatenate([lengths, rates]),
            "wheel_contact_force": wheel_force,
            "contact_flag": np.concatenate([wheel_hit, [body_hit, airborne]]),
            "terrain_scan": sim.terrain_scan(float(sim.base_pos[0]), np.zeros(TERRAIN_SCAN_POINTS)),
            "dof_acc": dof_acc,
            "torques": np.concatenate([self._last_leg_torque, self._last_wheel_torque]),
            "domain_params": np.array([
                self.domain[name][1] for name in self.params.domain_randomization.PARAM_NAMES
            ], dtype=np.float64),
            "phase": np.array([np.sin(self.phase_clock), np.cos(self.phase_clock)]),
            "mode_onehot": self._mode_onehot(airborne, float(sim.base_pos[0])),
        }
        self._raw_cache = blocks
        scaled = {
            name: np.asarray(value, dtype=np.float64) * np.asarray(OBS_SCALE.get(name, 1.0), dtype=np.float64)
            for name, value in blocks.items()
        }
        self._obs_cache = scaled
        frame = np.concatenate([scaled[name] for name in _BLOCK_ORDER if name in _ACTOR_BLOCK_NAMES]).astype(np.float64)
        clip = float(self.params.noise.clip_observations)
        if clip > 0.0:
            np.clip(frame, -clip, clip, out=frame)
        noise_vec = np.concatenate([
            np.full(_BLOCK_WIDTH[name], _NOISE_VEC.get(name, 0.0))
            for name in _BLOCK_ORDER if name in _ACTOR_BLOCK_NAMES
        ])
        if np.any(noise_vec):
            white = self.rng.uniform(-1.0, 1.0, frame.shape) * noise_vec
            if self._sensor_noise_state is None:
                self._sensor_noise_state = white.copy()
            else:
                alpha = float(self.params.noise.temporal_alpha)
                self._sensor_noise_state = (1.0 - alpha) * self._sensor_noise_state + alpha * white
            frame += self._sensor_noise_state
        if self.deployment_mode:
            self._sensor_frame_history.append(frame.copy())
            if len(self._sensor_frame_history) > self._observation_delay_steps:
                frame = list(self._sensor_frame_history)[-self._observation_delay_steps - 1].copy()
        if not self._actor_history:
            for _ in range(OBS_HISTORY):
                self._actor_history.append(frame.copy())
        else:
            self._actor_history.append(frame.copy())
        history = np.concatenate(list(self._actor_history))
        privileged = np.concatenate([scaled[name] for name in _BLOCK_ORDER if name not in _ACTOR_BLOCK_NAMES])
        return np.concatenate([history, privileged]).astype(np.float32)

    def _mode_onehot(self, airborne: float, base_x: float) -> np.ndarray:
        onehot = np.zeros(len(MODE_NAMES), dtype=np.float64)
        if self.jump_phase:
            onehot[MODE_NAMES.index("jump")] = 1.0
        elif self.stage.name == "recovery" and abs(self.sim.data.qpos[2] - self.params.robot.nominal_stand_height) > 0.05:
            onehot[MODE_NAMES.index("recover")] = 1.0
        elif airborne > 0.5:
            onehot[MODE_NAMES.index("airborne")] = 1.0
        elif self.sim.terrain_scan(base_x, np.zeros(1))[0] > 0.01:
            onehot[MODE_NAMES.index("stair")] = 1.0
        else:
            onehot[MODE_NAMES.index("normal")] = 1.0
        return onehot

    def step(self, action):
        sim = self.sim
        policy_action = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
        if policy_action.size != ACTION_DIM:
            raise ValueError(f"expected action dimension {ACTION_DIM}, got {policy_action.size}")
        self._maybe_switch_height_target()
        self._advance_command()
        self._commanded_x += float(self.command[0]) * self.params.control_dt
        self._leg_error_potential = abs(float(sim.leg_lengths().mean()) - self.leg_length_command)
        applied_action = policy_action.copy()
        quat = sim.base_quat
        rpy = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler("xyz")
        body_vel = sim.body_frame(sim.base_lin_vel_world)
        # PPO updates the target at 125 Hz.  The held target is converted to
        # fresh MIT PD and C620 speed-loop commands four times at 500 Hz.
        control = None
        state = None
        for _ in range(self.params.robot.control_substeps):
            applied_action = policy_action.copy()
            if self.deployment_mode:
                # Actuator delay is expressed in 500 Hz motor ticks.  Repeated
                # targets in this queue model CAN/driver latency without
                # changing the 125 Hz policy contract.
                self._action_delay_history.append(policy_action.copy())
                if len(self._action_delay_history) <= self._actuator_delay_steps:
                    applied_action[:] = 0.0
                else:
                    applied_action = list(self._action_delay_history)[
                        -self._actuator_delay_steps - 1
                    ].copy()
            state = {
                "joint_pos": sim.joint_positions(),
                "joint_vel": sim.joint_velocities(),
                "wheel_vel": sim.wheel_velocities(),
            }
            control = self.actuators.compute(
                applied_action, state=state, dt=self.params.motor_control_dt
            )
            sim.data.ctrl[:4] = control["leg_ctrl"]
            sim.data.ctrl[4:] = control["wheel_ctrl"]
            sim.step()
        assert control is not None and state is not None
        self._last_leg_torque = np.asarray(control["leg_torque"], dtype=np.float64).copy()
        self._last_wheel_torque = np.asarray(control["wheel_torque"], dtype=np.float64).copy()
        self.steps += 1
        station_offset_xy = np.asarray(sim.data.qpos[:2] - self.nominal_xy, dtype=np.float64)
        station_error = float(station_offset_xy[0])
        motion_error = station_error - self._commanded_x
        station_drift = float(np.linalg.norm(station_offset_xy))
        motion_drift = float(np.linalg.norm([motion_error, station_offset_xy[1]]))
        self._station_max_abs = max(self._station_max_abs, station_drift)
        self._motion_max_abs = max(self._motion_max_abs, motion_drift)
        self._station_window.append(station_drift)
        self._motion_window.append(motion_drift)
        self.phase_clock += self.params.control_dt * 2.0 * np.pi / 0.6
        self._advance_jump_phase()
        self._dof_vel_hist.append(np.concatenate([sim.joint_velocities(), sim.wheel_velocities()]))
        obs = self._obs()
        terminated, truncated, reason = self._termination()
        reward, terms = self._reward(control, state, policy_action, terminated)
        self.previous_action = policy_action.copy()
        lengths = sim.leg_lengths()
        self._leg_rate_for_control = float((lengths - self.prev_leg_lengths).mean() / self.params.control_dt)
        self.prev_leg_lengths = lengths
        self._last_reward_terms = terms
        self._last_wheel_current = np.asarray(control["wheel_current"], dtype=np.float64).copy()
        leg_error = float(lengths.mean() - self.leg_length_command)
        if self._height_last_switch_step >= 0 and self._height_settle_steps < 0 and abs(leg_error) <= 0.005:
            self._height_settle_steps = self.steps - self._height_last_switch_step
        body_after = float(sim.body_frame(sim.base_lin_vel_world)[0])
        rpy_after = Rotation.from_quat([
            sim.base_quat[1], sim.base_quat[2], sim.base_quat[3], sim.base_quat[0]
        ]).as_euler("xyz")
        wheel_velocity_body_forward = (
            WHEEL_ANGULAR_TO_BODY_X * sim.wheel_velocities()
        )
        wheel_target_body_forward = (
            WHEEL_ANGULAR_TO_BODY_X
            * np.asarray(control["wheel_target"], dtype=np.float64)
        )
        wheel_vx = float(np.mean(wheel_velocity_body_forward)) * sim.wheel_radius
        self._info = {
            "reward_terms": terms, "reward_unattributed": 0.0,
            "termination_reason": reason,
            "domain": {key: value[0] for key, value in self.domain.items()},
            "mode": MODE_NAMES[int(np.argmax(self._obs_cache["mode_onehot"]))],
            **self.actuators.diagnostics, **state,
            "roll": float(rpy_after[0]), "pitch": float(rpy_after[1]), "yaw": float(rpy_after[2]),
            "pitch_rate": float(sim.body_frame(sim.base_ang_vel_world)[1]),
            "yaw_rate": float(sim.body_frame(sim.base_ang_vel_world)[2]),
            "body_vx": float(body_vel[0]), "body_vx_after": body_after,
            "body_vy": float(sim.body_frame(sim.base_lin_vel_world)[1]),
            "command_vx": float(self.command[0]), "command_target_vx": float(self.command_target[0]),
            "command_yaw": float(self.command[2]), "vx_reward_scale": float(self.vx_reward_scale),
            "vx_tracking_error": body_after - float(self.command[0]),
            "vx_tracking_error_abs": abs(body_after - float(self.command[0])),
            "body_accel": (body_after - self._last_step_body_vx) / self.params.control_dt,
            "base_height": float(sim.data.qpos[2]),
            "station_error": station_error, "station_error_abs": abs(station_error),
            "station_drift_m": station_drift,
            "station_within_5cm": float(station_drift <= 0.05),
            "station_max_abs": float(self._station_max_abs),
            "station_tail_mean": float(np.mean(self._station_window)),
            "commanded_x_displacement": float(self._commanded_x),
            "motion_position_error": motion_error, "motion_position_error_abs": abs(motion_error),
            "motion_drift_m": motion_drift,
            "motion_max_abs": float(self._motion_max_abs),
            "motion_tail_mean": float(np.mean(self._motion_window)),
            "episode_steps": int(self.steps), "deployment_mode": bool(self.deployment_mode),
            "policy_rate_hz": float(1.0 / self.params.control_dt),
            "motor_control_rate_hz": float(1.0 / self.params.motor_control_dt),
            "motor_ticks_per_policy": int(self.params.robot.control_substeps),
            "observation_delay_steps": int(self._observation_delay_steps),
            "actuator_delay_motor_ticks": int(self._actuator_delay_steps),
            "actuator_delay_ms": float(
                1000.0 * self._actuator_delay_steps * self.params.motor_control_dt
            ),
            "desired_actuator_action_abs": float(np.abs(policy_action).mean()),
            "applied_actuator_action_abs": float(np.abs(applied_action).mean()),
            "motion_velocity_error": float(self.command[0] - body_vel[0]),
            "wheel_body_vx": float(wheel_vx),
            "wheel_slip_m_s": float(abs(wheel_vx - body_after)),
            "wheel_target_forward_left": float(wheel_target_body_forward[0]),
            "wheel_target_forward_right": float(wheel_target_body_forward[1]),
            "wheel_target_skew_rad_s": float(abs(
                wheel_target_body_forward[0] - wheel_target_body_forward[1]
            )),
            "wheel_speed_forward_left": float(wheel_velocity_body_forward[0]),
            "wheel_speed_forward_right": float(wheel_velocity_body_forward[1]),
            "wheel_speed_skew_rad_s": float(abs(
                wheel_velocity_body_forward[0] - wheel_velocity_body_forward[1]
            )),
            "wheel_angular_to_body_x_sign": float(WHEEL_ANGULAR_TO_BODY_X),
            "leg_length_left": float(lengths[0]), "leg_length_right": float(lengths[1]),
            "leg_length_rate_mean": float(self._raw_cache["leg_length"][2:].mean()),
            "leg_length_target": float(self.leg_length_command),
            "leg_length_error": leg_error, "leg_length_error_mm": leg_error * 1000.0,
            "height_target": float(self.command[3]), "height_final_target": float(self.command_target[3]),
            "height_switch_count": int(self._height_switch_count),
            "height_last_switch_step": int(self._height_last_switch_step),
            "height_settle_steps": int(self._height_settle_steps),
            "airborne": float(self._raw_cache["contact_flag"][3]),
            "body_contact": float(self._raw_cache["contact_flag"][2]),
            "wheel_contact_left": float(self._raw_cache["contact_flag"][0]),
            "wheel_contact_right": float(self._raw_cache["contact_flag"][1]),
            "leg_torque_vector": np.asarray(control["leg_torque"], dtype=np.float32),
            "wheel_current_vector": np.asarray(control["wheel_current"], dtype=np.float32),
            "wheel_current_raw": np.asarray(control["wheel_current_raw"], dtype=np.int32),
            "mit_p_des": np.asarray(control["mit_p_des"], dtype=np.float32),
        }
        self._last_step_body_vx = body_after
        return obs, float(reward), bool(terminated), bool(truncated), self._info

    def _advance_jump_phase(self) -> None:
        if self.jump_phase == 0:
            return
        self.jump_timer -= self.params.control_dt
        if self.jump_timer <= 0.0:
            self.jump_phase = 0 if self.jump_phase >= 3 else self.jump_phase + 1
            self.jump_timer = 0.24

    def _reward(self, control: dict, state: dict, action: np.ndarray,
                terminated: bool) -> tuple[float, dict[str, float]]:
        w = self.params.rewards
        flags = self.reward_flags
        sim = self.sim
        rpy = Rotation.from_quat([
            sim.base_quat[1], sim.base_quat[2], sim.base_quat[3], sim.base_quat[0]
        ]).as_euler("xyz")
        roll, pitch = float(rpy[0]), float(rpy[1])
        lin = sim.body_frame(sim.base_lin_vel_world)
        ang = sim.body_frame(sim.base_ang_vel_world)
        vx, vy, yaw_rate = float(lin[0]), float(lin[1]), float(ang[2])
        if self.stage.name == "low_speed":
            self._motion_vx_window.append(vx)
        lengths = sim.leg_lengths()
        leg_mean = float(lengths.mean())
        leg_error = leg_mean - self.leg_length_command
        height_error = float(sim.data.qpos[2] - self.command[3])
        station_offset_xy = np.asarray(sim.data.qpos[:2] - self.nominal_xy, dtype=np.float64)
        station_error = float(station_offset_xy[0])
        station_drift = float(np.linalg.norm(station_offset_xy))
        motion_error = station_error - self._commanded_x
        _, wheel_hit, body_hit, airborne = sim.contact_state()
        q, qd = state["joint_pos"], state["joint_vel"]
        terms: dict[str, float] = {}
        upright = float(np.exp(-(roll * roll + pitch * pitch) / 0.25 ** 2))
        terms["upright"] = w.upright * upright
        terms["upright_progress"] = w.upright_progress * (upright - self._upright_potential)
        self._upright_potential = upright
        terms["height"] = -w.height * min((height_error / 0.02) ** 2, 4.0)
        terms["leg_length"] = -w.leg_length * min((leg_error / 0.015) ** 2, 4.0)
        terms["leg_length_progress"] = w.leg_length_progress * np.clip(
            self._leg_error_potential - abs(leg_error), -0.02, 0.02
        )
        terms["leg_length_rate"] = -w.leg_length_rate * min(
            (self._leg_rate_for_control / max(w.leg_length_rate_sigma_m_s, 1e-6)) ** 2, 4.0
        )
        terms["joint_neutral"] = -w.joint_neutral * float(
            np.square(q - np.asarray(self.params.robot.stand_joint_pos)).mean()
        )
        prev = self.previous_action
        terms["action_rate"] = -w.action_rate * float(np.square(action - prev).mean())
        terms["leg_action"] = -w.leg_action * float(np.square(action[2:]).mean())
        terms["joint_velocity"] = -w.joint_velocity * min(float(np.square(qd).mean()), 400.0)
        terms["joint_torque"] = -w.joint_torque * min(float(np.square(control["leg_torque"]).mean()), 1600.0)
        terms["wheel_power"] = -w.wheel_power * min(float(np.square(control["wheel_current"]).mean()), 400.0)
        terms["wheel_current_rate"] = -w.wheel_current_rate * float(np.square(
            control["wheel_current"] - self._last_wheel_current
        ).mean())
        terms["wheel_current_jerk"] = 0.0
        terms["leg_action_rate"] = -w.leg_action_rate * float(np.square(action[2:] - prev[2:]).mean())
        if self.stage.name in {"stand", "low_speed"} and not self.motion_control_active:
            # Any zero-command episode should rehearse the stand contact and
            # smooth-leg behavior, including the zero-command portion of the
            # low-speed curriculum.  Other tasks such as jump/recovery remain
            # unaffected even when their translation command is zero.
            terms["stand_wheel_contact"] = -w.stand_wheel_contact * (
                1.0 - float(np.mean(wheel_hit))
            )
            hip_target_rate = (
                (action[2:6] - prev[2:6]) * HIP_POSITION_SCALE
                / self.params.control_dt
            )
            terms["stand_leg_target_rate"] = -w.stand_leg_target_rate * min(
                float(np.square(
                    hip_target_rate / max(w.stand_leg_target_rate_sigma_rad_s, 1e-6)
                ).mean()),
                4.0,
            )
            leg_length_rates = (
                lengths - self.prev_leg_lengths
            ) / self.params.control_dt
            terms["stand_leg_length_rate"] = -w.stand_leg_length_rate * min(
                float(np.square(
                    leg_length_rates / max(w.leg_length_rate_sigma_m_s, 1e-6)
                ).mean()),
                4.0,
            )
        else:
            terms["stand_wheel_contact"] = 0.0
            terms["stand_leg_target_rate"] = 0.0
            terms["stand_leg_length_rate"] = 0.0
        terms["leg_symmetry"] = -w.leg_symmetry * float((lengths[0] - lengths[1]) ** 2)
        wheel_velocities = sim.wheel_velocities()
        wheel_differential_weight = 0.0
        if not self.motion_control_active:
            wheel_differential_weight = w.wheel_differential
        elif (
            self.stage.name == "low_speed"
            and abs(float(self.command[2])) <= 0.01
            and self.low_speed_wheel_differential_weight is not None
        ):
            wheel_differential_weight = self.low_speed_wheel_differential_weight
        terms["wheel_differential"] = -wheel_differential_weight * float(
            (wheel_velocities[0] - wheel_velocities[1]) ** 2
        )
        wheel_body_vx = WHEEL_ANGULAR_TO_BODY_X * float(np.mean(wheel_velocities)) * sim.wheel_radius
        terms["wheel_slip"] = -w.wheel_slip * min(abs(wheel_body_vx - vx) / 0.15, 2.0) if self.motion_control_active else 0.0
        terms["wheel_speed"] = -w.wheel_speed * min(float(np.square(sim.wheel_velocities()).mean()), 25.0) if not self.motion_control_active else 0.0
        leg_min, leg_max = self.params.robot.leg_length_min, self.params.robot.leg_length_max
        over = max(0.0, leg_mean - leg_max) + max(0.0, leg_min - leg_mean)
        terms["leg_length_limit"] = -w.leg_length_limit * over * over
        joint_over = np.clip(np.abs(q) - self.params.robot.hip_joint_limit, 0.0, None)
        terms["joint_limit"] = -w.joint_limit * float(np.square(joint_over).mean())

        terms.update({
            "track_vx": 0.0, "track_vx_error": 0.0, "track_vx_forward_error": 0.0,
            "track_vx_reverse_error": 0.0, "track_vx_progress": 0.0,
            "wrong_direction": 0.0, "track_vy": -w.track_vy * abs(vy),
            "yaw_lock": 0.0, "track_yaw": 0.0,
            "motion_position_error": 0.0, "motion_position_progress": 0.0,
            "pitch_speed_coupling": 0.0,
            "low_speed_startup_progress": 0.0,
            "low_speed_cruise_progress": 0.0,
            "low_speed_stable_motion": 0.0,
        })
        if flags["track_vx"] and abs(self.command[0]) > 0.01:
            cmd = float(self.command[0])
            low_speed_motion = self.stage.name == "low_speed" and self._translation_episode_active
            if low_speed_motion:
                # Use one absolute-error tracking objective in m/s.  This
                # avoids making a stationary robot look acceptable merely
                # because a small command turns into a normalized error near 1.
                scored_vx = float(vx)
                sigma = max(
                    float(w.low_speed_track_sigma_m_s),
                    max(float(w.low_speed_track_sigma_relative), 0.0) * abs(cmd),
                    1e-4,
                )
                velocity_error = scored_vx - cmd
                terms["track_vx"] = self.vx_reward_scale * w.track_vx * np.exp(
                    -((velocity_error / sigma) ** 2)
                )
                # Bidirectional low-speed commands need a signed gradient at
                # zero velocity.  The absolute-error term is symmetric, so
                # opposite commands can cancel while the policy converges to
                # standing still.  Reuse the configured alignment reward to
                # tell PPO which way each command should move.
                velocity_alignment = np.clip(
                    float(np.sign(cmd)) * scored_vx / max(abs(cmd), 0.04),
                    -1.0,
                    1.0,
                )
                terms["track_vx_progress"] = (
                    self.vx_reward_scale * w.track_vx_progress * velocity_alignment
                )
                if cmd < 0.0:
                    # Early bidirectional training can be dominated by the
                    # already-solved forward command.  Scale only the reverse
                    # task reward so its successful and failed trajectories
                    # contribute comparable PPO signal; posture and effort
                    # rewards retain their original scale.
                    terms["track_vx"] *= self.low_speed_reverse_reward_scale
                    terms["track_vx_progress"] *= self.low_speed_reverse_reward_scale
                wrong_direction_speed = max(
                    0.0, -float(np.sign(cmd)) * scored_vx
                )
                wrong_direction_ratio = wrong_direction_speed / max(abs(cmd), 0.04)
                terms["wrong_direction"] = (
                    -self.vx_reward_scale
                    * self.low_speed_wrong_direction_weight
                    * min(wrong_direction_ratio ** 2, 4.0)
                )
                pitch_excess = max(
                    0.0,
                    abs(pitch) / max(w.low_speed_stable_pitch_sigma_rad, 1e-6) - 1.0,
                )
                pitch_rate_excess = max(
                    0.0,
                    abs(float(ang[1]))
                    / max(w.low_speed_stable_pitch_rate_sigma_rad_s, 1e-6) - 1.0,
                )
                posture_penalty = min(
                    pitch_excess ** 2 + pitch_rate_excess ** 2,
                    max(w.low_speed_stable_penalty_cap, 0.0),
                )
                terms["low_speed_stable_motion"] = (
                    -self.vx_reward_scale * w.low_speed_stable_motion * posture_penalty
                )
                # The low-speed ablation has one task reward.  Keep posture
                # safety separate from speed tracking; do not add startup,
                # cruise, pitch-target, or duplicate direction shaping here.
                terms["low_speed_startup_progress"] = 0.0
                terms["low_speed_cruise_progress"] = 0.0
                terms["pitch_speed_coupling"] = 0.0
            else:
                sustained_vx = vx
                scored_vx = vx
                posture_penalty = 0.0
                error = (scored_vx - cmd) / max(abs(cmd), 0.06)
                abs_error = abs(error)
                terms["track_vx"] = self.vx_reward_scale * (
                    w.track_vx * np.exp(-((error / w.track_vx_wide_sigma) ** 2))
                    + w.track_vx_tight * np.exp(-((error / w.track_vx_tight_sigma) ** 2))
                    - w.track_vx_square * (0.25 * error) ** 2
                    - w.track_vx_gap * max(0.0, abs_error - 1.0) ** 2
                )
                velocity_alignment = np.clip(
                    float(np.sign(cmd)) * vx / max(abs(cmd), 0.04),
                    -1.0, 1.0,
                )
                terms["track_vx_progress"] = (
                    self.vx_reward_scale * w.track_vx_progress * velocity_alignment
                )
                wrong_speed = max(0.0, -float(np.sign(cmd)) * vx)
                wrong_direction_ratio = wrong_speed / max(abs(cmd), 0.04)
                terms["wrong_direction"] = -self.vx_reward_scale * w.wrong_direction * min(
                    wrong_direction_ratio ** 2, 4.0
                )
                pitch_target = np.clip(
                    -w.pitch_speed_gain_rad_per_m_s * (cmd - vx), -0.14, 0.14
                )
                terms["pitch_speed_coupling"] = w.pitch_speed_coupling * np.exp(
                    -(((pitch - pitch_target) / w.pitch_speed_sigma_rad) ** 2)
                )
            terms["track_vx_forward_error"] = 0.0
            terms["track_vx_reverse_error"] = 0.0
        # Translation episodes are optimized against velocity commands.  Keep
        # path-position error as a diagnostic only; applying station/drift
        # reward while moving can overwhelm velocity tracking.  Station
        # keeping penalties below are reserved for zero-command episodes.
        if flags["yaw_lock"] and self.motion_control_active and abs(self.command[2]) <= 0.01:
            terms["yaw_lock"] = -w.yaw_lock * min((yaw_rate / w.yaw_lock_sigma) ** 2, 4.0)
        if flags["track_yaw"] and abs(self.command[2]) > 0.01:
            yaw_error = (yaw_rate - self.command[2]) / max(abs(self.command[2]), 0.15)
            terms["track_yaw"] = w.track_yaw * np.exp(-((yaw_error / 0.5) ** 2)) - w.track_yaw_square * (0.25 * yaw_error) ** 2
        if not self.motion_control_active:
            if self.stage.name == "low_speed":
                # Zero-command episodes must hold the full planar position.
                # Keep the penalty bounded, but smooth its tail so PPO still
                # gets a gradient after the old 4 cm hard saturation point.
                position_error = station_drift
                normalized_drift = position_error / max(w.station_sigma_m, 1e-6)
                terms["station"] = -w.station * (
                    1.0 - np.exp(-(normalized_drift ** 2))
                )
                terms["station_vel"] = -w.station_vel * min(
                    (vx / w.station_vel_sigma_m_s) ** 2, 1.0
                )
                terms["station_progress"] = w.station_progress * (
                    self._station_potential - position_error
                )
                self._station_potential = position_error
            else:
                terms["station"] = -w.station * min((station_error / w.station_sigma_m) ** 2, 1.0)
                terms["station_vel"] = -w.station_vel * min((vx / w.station_vel_sigma_m_s) ** 2, 1.0)
                terms["station_progress"] = w.station_progress * (self._station_potential - abs(station_error))
                self._station_potential = abs(station_error)
            terms["stand_action"] = -w.stand_action * float(np.square(action).mean())
        else:
            terms["station"] = terms["station_vel"] = terms["station_progress"] = terms["stand_action"] = 0.0
        if flags["stand_still"] and not self.motion_control_active:
            terms["stand_vx"] = -w.stand_vx * abs(vx)
            terms["stand_yaw"] = -w.stand_yaw * abs(yaw_rate)
            terms["stand_wheel"] = -w.stand_wheel_speed * min(float(np.square(sim.wheel_velocities()).mean()), 25.0)
            terms["stand_common_action"] = -w.stand_common_action * float(np.square(action[:2]).mean())
            terms["stand_pitch_rate"] = -w.stand_pitch_rate * min((float(ang[1]) / w.stand_pitch_rate_sigma) ** 2, 4.0)
            accel = (vx - self._prev_body_vx) / self.params.control_dt
            terms["stand_accel"] = -w.stand_accel * min((accel / w.stand_accel_sigma) ** 2, 4.0)
        else:
            for key in ("stand_vx", "stand_yaw", "stand_wheel", "stand_common_action", "stand_pitch_rate", "stand_accel"):
                terms[key] = 0.0
        self._prev_body_vx = vx
        for key in ("airborne_upright", "airborne_leg_retract", "undesired_contact", "landing_impact",
                    "terrain_progress", "front_wheel_height", "jump_height", "jump_phase_time",
                    "recovery_upright", "recovery_progress"):
            terms[key] = 0.0
        if flags["airborne"]:
            terms["airborne_upright"] = w.airborne_upright * upright if airborne > 0.5 else 0.0
            terms["airborne_leg_retract"] = -w.airborne_leg_retract * max(0.0, leg_mean - self.params.robot.nominal_leg_length) if airborne > 0.5 else 0.0
            terms["undesired_contact"] = -w.undesired_contact * body_hit
            terms["landing_impact"] = -w.landing_impact * self.last_landing_impact
        if flags["terrain"]:
            scan = self._raw_cache["terrain_scan"]
            terms["terrain_progress"] = w.terrain_progress * float((scan > 0.01).mean())
            terms["front_wheel_height"] = -w.front_wheel_height * max(0.0, float(scan[0]) - 0.05)
        if flags["jump"] and self.jump_phase:
            terms["jump_height"] = w.jump_height * max(0.0, float(sim.data.qpos[2]) - self.params.robot.reset_height)
            terms["jump_phase_time"] = -w.jump_phase_time * abs(self.jump_timer)
        if flags["recovery"]:
            terms["recovery_upright"] = w.recovery_upright * upright
            terms["recovery_progress"] = w.recovery_progress * max(0.0, 1.0 - abs(float(sim.data.qpos[2]) - self.params.robot.nominal_stand_height) / 0.15)
        positive = {"upright", "upright_progress", "track_vx", "track_vx_progress", "track_yaw",
                    "low_speed_startup_progress", "low_speed_cruise_progress",
                    "pitch_speed_coupling", "terrain_progress", "airborne_upright", "recovery_upright",
                    "recovery_progress", "jump_height"}
        continuous = sum(value for key, value in terms.items() if key not in positive)
        if continuous < -10.0:
            factor = 10.0 / abs(continuous)
            for key in terms:
                if key not in positive:
                    terms[key] *= factor
        terms["alive"] = w.alive
        terms["termination"] = -w.termination if terminated else 0.0
        return float(sum(terms.values())), terms

    def _termination(self) -> tuple[bool, bool, str]:
        rpy = Rotation.from_quat([
            self.sim.base_quat[1], self.sim.base_quat[2], self.sim.base_quat[3], self.sim.base_quat[0]
        ]).as_euler("xyz")
        tilt = max(abs(float(rpy[0])), abs(float(rpy[1])))
        reasons: list[str] = []
        if tilt > self.tilt_limit:
            self.tilt_exceed_steps += 1
            if self.tilt_exceed_steps > self.tilt_hold_steps:
                reasons.append("tilt_limit")
        else:
            self.tilt_exceed_steps = 0
        if self.sim.data.qpos[2] < self.params.robot.nominal_stand_height * self.params.height_fail_ratio:
            reasons.append("height_limit")
        if float(self.sim.leg_lengths().mean()) < self.params.robot.nominal_leg_length * self.params.leg_length_fail_ratio:
            reasons.append("leg_length_limit")
        station_abs = abs(float(self.sim.data.qpos[0] - self.nominal_xy[0]))
        if not self.motion_control_active and station_abs > self.params.station_hard_limit:
            self.station_exceed_steps += 1
            if self.station_exceed_steps > self.params.station_hold_steps:
                reasons.append("station_limit")
        else:
            self.station_exceed_steps = 0
        if self._translation_episode_active:
            drift = float(np.hypot(
                self.sim.data.qpos[0] - self.nominal_xy[0] - self._commanded_x,
                self.sim.data.qpos[1] - self.nominal_xy[1],
            ))
        else:
            drift = float(np.linalg.norm(self.sim.data.qpos[:2] - self.nominal_xy))
        if drift > self.params.terminate_on_drift:
            reasons.append("drift_limit")
        if abs(float(self.sim.body_frame(self.sim.base_lin_vel_world)[1])) > self.params.terminate_lateral_vel:
            reasons.append("lateral_velocity_limit")
        terminated = bool(reasons)
        episode_steps = self.episode_steps_override or self.stage.episode_steps
        truncated = self.steps >= episode_steps
        return terminated, truncated, ", ".join(reasons) if reasons else ("time_limit" if truncated else "running")


def control_diag(bank: ActuatorBank) -> dict[str, float]:
    return dict(bank.diagnostics)
