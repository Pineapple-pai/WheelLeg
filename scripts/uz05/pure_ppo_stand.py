"""Isolated, fixed-difficulty Pure PPO standing task for the UZ-05.

This module has no policy assistance, action blending, pose feedback branch,
curriculum, observation noise, or domain randomization.  The only feedback
loops are the fixed leg-joint PD and wheel-speed current controller that turn
the policy's six targets into MuJoCo actuator forces.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .actuators import JointPDController, WheelEscController
from .model import TerrainSpec, UZ05Model
from .observations import ObservationEncoder
from .spec import (
    ACTOR_OBS_DIM,
    OBS_DIM,
    PRIV_OBS_DIM,
    EnvParams,
    DOF_ACC_WINDOW,
)


TASK_ID = "pure_ppo_stand_v1"
ACTION_PROTOCOL = "stand_offset_v1"
ACTION_ORDER = (
    "wheel_left",
    "wheel_right",
    "leg_left_1",
    "leg_left_2",
    "leg_right_1",
    "leg_right_2",
)
ACTION_DIM = 6
POLICY_DT_S = 0.008
MOTOR_DT_S = 0.002
MOTOR_TICKS_PER_POLICY_STEP = 4
POLICY_RATE_HZ = 125
MOTOR_RATE_HZ = 500
EPISODE_STEPS = 1000
RESET_HEIGHT_M = 0.23825
NOMINAL_LEG_LENGTH_M = 0.18413
Q_REF = np.asarray([-0.19450, 0.19129, -0.19141, 0.19057], dtype=np.float64)
WHEEL_SIGN = np.asarray([1.0, 1.0], dtype=np.float64)
WHEEL_SPEED_SCALE_RAD_S = 8.0
LEG_POSITION_SCALE_RAD = 0.5
JOINT_POSITION_LIMIT_RAD = 1.2
PITCH_INIT_LIMIT_RAD = np.deg2rad(3.0)
PITCH_RATE_INIT_LIMIT_RAD_S = 0.1
TILT_FAILURE_LIMIT_RAD = np.deg2rad(25.0)
HEIGHT_FAILURE_LIMIT_M = (0.12, 0.45)
DRIFT_FAILURE_LIMIT_M = 0.40


@dataclass(frozen=True)
class StandRewardConfig:
    """Fixed per-policy-step reward coefficients for the first experiment."""

    upright: float = 1.0
    alive: float = 0.1
    pitch_rate: float = 0.05
    horizontal_velocity: float = 0.05
    position_drift: float = 0.10
    height_error: float = 0.05
    leg_length_error: float = 0.05
    action_rate: float = 0.005
    effort: float = 0.005
    termination: float = 5.0
    upright_sigma_rad: float = 0.10
    pitch_rate_scale_rad_s: float = 0.10
    horizontal_velocity_scale_m_s: float = 0.10
    position_drift_scale_m: float = 0.10
    height_error_scale_m: float = 0.05
    leg_length_error_scale_m: float = 0.05


REWARD_CONFIG = StandRewardConfig()


def pure_ppo_config() -> dict[str, Any]:
    """Serializable, immutable-by-convention experiment constants."""
    return {
        "task_id": TASK_ID,
        "action_protocol": ACTION_PROTOCOL,
        "action_order": list(ACTION_ORDER),
        "action_dim": ACTION_DIM,
        "action_low": [-1.0] * ACTION_DIM,
        "action_high": [1.0] * ACTION_DIM,
        "wheel_sign": WHEEL_SIGN.tolist(),
        "wheel_speed_scale_rad_s": WHEEL_SPEED_SCALE_RAD_S,
        "leg_position_scale_rad": LEG_POSITION_SCALE_RAD,
        "joint_position_limit_rad": JOINT_POSITION_LIMIT_RAD,
        "q_ref_rad": Q_REF.tolist(),
        "reset_height_m": RESET_HEIGHT_M,
        "nominal_leg_length_m": NOMINAL_LEG_LENGTH_M,
        "policy_dt_s": POLICY_DT_S,
        "motor_dt_s": MOTOR_DT_S,
        "motor_ticks_per_policy_step": MOTOR_TICKS_PER_POLICY_STEP,
        "policy_rate_hz": POLICY_RATE_HZ,
        "motor_rate_hz": MOTOR_RATE_HZ,
        "episode_steps": EPISODE_STEPS,
        "initial_pitch_rad": [-PITCH_INIT_LIMIT_RAD, PITCH_INIT_LIMIT_RAD],
        "initial_pitch_rate_rad_s": [
            -PITCH_RATE_INIT_LIMIT_RAD_S,
            PITCH_RATE_INIT_LIMIT_RAD_S,
        ],
        "initial_roll_yaw_rad": [0.0, 0.0],
        "domain_randomization": False,
        "observation_noise": False,
        "observation_delay_steps": 0,
        "action_delay_steps": 0,
        "pushes_during_training": False,
        "actor_observation_dim": ACTOR_OBS_DIM,
        "critic_observation_dim": PRIV_OBS_DIM,
        "observation_dim": OBS_DIM,
        "reward_timing": "one reward per 8ms policy step; not multiplied by dt",
        "reward": asdict(REWARD_CONFIG),
        "termination": {
            "abs_roll_or_pitch_rad": TILT_FAILURE_LIMIT_RAD,
            "base_height_m": list(HEIGHT_FAILURE_LIMIT_M),
            "planar_drift_m": DRIFT_FAILURE_LIMIT_M,
            "body_contact": "any detected chassis/gimbal contact",
            "episode_cap_policy_steps": EPISODE_STEPS,
        },
    }


def map_stand_offset_action(
    action: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map one normalized policy action to two wheel and four leg targets.

    This function intentionally takes no observation or simulator state.  Its
    outputs can therefore depend only on the action and the fixed protocol.
    The returned action is clipped to the declared normalized action space.
    """
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (ACTION_DIM,):
        raise ValueError(f"expected action shape ({ACTION_DIM},), got {action.shape}")
    if not np.all(np.isfinite(action)):
        raise ValueError("action contains NaN or infinity")
    clipped = np.clip(action, -1.0, 1.0)
    wheel_target = WHEEL_SIGN * WHEEL_SPEED_SCALE_RAD_S * clipped[:2]
    joint_target = np.clip(
        Q_REF + LEG_POSITION_SCALE_RAD * clipped[2:6],
        -JOINT_POSITION_LIMIT_RAD,
        JOINT_POSITION_LIMIT_RAD,
    )
    return clipped, wheel_target, joint_target


def compute_stand_reward(
    *,
    roll_rad: float,
    pitch_rad: float,
    pitch_rate_rad_s: float,
    horizontal_velocity_m_s: float,
    position_drift_m: float,
    height_error_m: float,
    leg_length_error_m: float,
    action: np.ndarray,
    previous_action: np.ndarray,
    normalized_effort: float,
    terminated: bool,
    config: StandRewardConfig = REWARD_CONFIG,
) -> tuple[float, dict[str, float], dict[str, float]]:
    """Return reward, raw measurements, and signed weighted contributions.

    No term compares policy output with a target/controller action.  Action is
    read only for a small action-rate regularizer; effort is measured from the
    low-level actuator torques.
    """
    action = np.asarray(action, dtype=np.float64)
    previous_action = np.asarray(previous_action, dtype=np.float64)
    tilt_sq = float(roll_rad * roll_rad + pitch_rad * pitch_rad)
    raw = {
        "upright": float(np.exp(-tilt_sq / (config.upright_sigma_rad**2))),
        "alive": 1.0,
        "pitch_rate_cost": float(
            (pitch_rate_rad_s / config.pitch_rate_scale_rad_s) ** 2
        ),
        "horizontal_velocity_cost": float(
            (horizontal_velocity_m_s / config.horizontal_velocity_scale_m_s) ** 2
        ),
        "position_drift_cost": float(
            (position_drift_m / config.position_drift_scale_m) ** 2
        ),
        "height_error_cost": float(
            (height_error_m / config.height_error_scale_m) ** 2
        ),
        "leg_length_error_cost": float(
            (leg_length_error_m / config.leg_length_error_scale_m) ** 2
        ),
        "action_rate_cost": float(np.mean(np.square(action - previous_action))),
        "effort_cost": float(max(0.0, normalized_effort)),
        "terminal_failure": float(bool(terminated)),
    }
    weighted = {
        "upright": config.upright * raw["upright"],
        "alive": config.alive * raw["alive"],
        "pitch_rate": -config.pitch_rate * raw["pitch_rate_cost"],
        "horizontal_velocity": -config.horizontal_velocity * raw[
            "horizontal_velocity_cost"
        ],
        "position_drift": -config.position_drift * raw["position_drift_cost"],
        "height_error": -config.height_error * raw["height_error_cost"],
        "leg_length_error": -config.leg_length_error * raw[
            "leg_length_error_cost"
        ],
        "action_rate": -config.action_rate * raw["action_rate_cost"],
        "effort": -config.effort * raw["effort_cost"],
        "termination": -config.termination * raw["terminal_failure"],
    }
    return float(sum(weighted.values())), raw, weighted


class PurePPOStandActuator:
    """Fixed actuator mapping plus only the motor-level PD/P controllers."""

    def __init__(self, params: EnvParams):
        self.joint = JointPDController(params.joint)
        self.wheel = WheelEscController(params.wheel)

    def reset(self) -> None:
        self.wheel.reset()

    def compute(
        self,
        action: np.ndarray,
        *,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        wheel_vel: np.ndarray,
        dt: float = POLICY_DT_S,
    ) -> dict[str, np.ndarray]:
        clipped, wheel_target, joint_target = map_stand_offset_action(action)
        joint_torque = self.joint.torque(
            joint_target, np.zeros(4), joint_pos, joint_vel
        )
        wheel_current = self.wheel.current(wheel_target, wheel_vel, dt)
        wheel_torque = self.wheel.torque_from_current(wheel_current)
        return {
            "action": clipped,
            "joint_target": joint_target,
            "wheel_target": wheel_target,
            "joint_torque": joint_torque,
            "wheel_current": wheel_current,
            "wheel_torque": wheel_torque,
            "joint_ctrl": joint_torque,
            "wheel_ctrl": self.wheel.motor_command(wheel_torque),
        }


class PurePPOStandEnv(gym.Env):
    """Fixed flat-ground 8s stand task with a 38D actor / 94D critic input."""

    metadata = {"render_modes": []}

    def __init__(self, seed: int | None = None, params: EnvParams | None = None):
        super().__init__()
        self.params = params if params is not None else EnvParams()
        # These settings are forced before constructing the per-instance encoder.
        self.params.domain_randomization.enabled = False
        self.params.noise.enabled = False
        self.params.noise.level = 0.0
        self.rng = np.random.default_rng(seed)
        self.sim = UZ05Model(TerrainSpec(kind="flat"))
        if not np.isclose(self.sim.model.opt.timestep, MOTOR_DT_S, atol=1e-12):
            raise RuntimeError(
                f"expected MuJoCo timestep {MOTOR_DT_S}, got {self.sim.model.opt.timestep}"
            )
        if not np.isclose(POLICY_DT_S, MOTOR_TICKS_PER_POLICY_STEP * MOTOR_DT_S):
            raise RuntimeError("policy/motor timing contract is inconsistent")
        self.observation_encoder = ObservationEncoder(
            self.params.noise, self.params.noise.clip_observations
        )
        self.actuators = PurePPOStandActuator(self.params)
        self.action_space = spaces.Box(-1.0, 1.0, (ACTION_DIM,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), dtype=np.float32)
        self._passive_qpos_adr = self._find_passive_joint_addresses()
        self._passive_reset_qpos = self._solve_reset_closure()
        self.steps = 0
        self.previous_action = np.zeros(ACTION_DIM, dtype=np.float64)
        self.nominal_xy = np.zeros(2, dtype=np.float64)
        self.nominal_yaw = 0.0
        self._prev_leg_lengths = np.zeros(2, dtype=np.float64)
        self._dof_vel_hist: deque[np.ndarray] = deque(maxlen=DOF_ACC_WINDOW + 1)
        self._last_joint_torque = np.zeros(4, dtype=np.float64)
        self._last_wheel_torque = np.zeros(2, dtype=np.float64)
        self._episode_samples: dict[str, list[Any]] = {}
        self._episode_return = 0.0
        self._episode_start_pitch = 0.0
        self._episode_start_pitch_rate = 0.0
        self.reset(seed=seed)

    def _find_passive_joint_addresses(self) -> list[int]:
        active = set(self.sim.hip_qpos_adr + self.sim.wheel_qpos_adr)
        passive = []
        for joint_id in range(self.sim.model.njnt):
            if int(self.sim.model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_HINGE):
                continue
            address = int(self.sim.model.jnt_qposadr[joint_id])
            if address not in active:
                passive.append(address)
        if len(passive) != 8:
            raise RuntimeError(f"expected 8 passive hinge coordinates, found {len(passive)}")
        return passive

    def _equality_residual(self) -> np.ndarray:
        mask = self.sim.data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY
        return np.asarray(self.sim.data.efc_pos[mask], dtype=np.float64).copy()

    def _solve_reset_closure(self) -> np.ndarray:
        """Find one fixed passive-joint pose that closes all four link loops."""
        self.sim.reset(RESET_HEIGHT_M)
        data = self.sim.data
        data.qpos[self.sim.hip_qpos_adr] = Q_REF

        def residual(passive_qpos: np.ndarray) -> np.ndarray:
            data.qpos[self.sim.hip_qpos_adr] = Q_REF
            data.qpos[self._passive_qpos_adr] = passive_qpos
            mujoco.mj_forward(self.sim.model, data)
            return self._equality_residual()

        solution = least_squares(
            residual,
            np.zeros(len(self._passive_qpos_adr), dtype=np.float64),
            max_nfev=100,
            xtol=1e-11,
            ftol=1e-11,
            gtol=1e-11,
        )
        final_residual = residual(solution.x)
        if not solution.success or not np.all(np.isfinite(solution.x)):
            raise RuntimeError(f"failed to close reset linkage: {solution.message}")
        if final_residual.size != 12 or np.max(np.abs(final_residual)) > 1e-8:
            raise RuntimeError(
                "reset pose does not close all four link loops: "
                f"max residual={np.max(np.abs(final_residual), initial=0.0):.3e} m"
            )
        return np.asarray(solution.x, dtype=np.float64)

    @staticmethod
    def _rpy(sim: UZ05Model) -> np.ndarray:
        quat = sim.base_quat
        return Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler("xyz")

    def _reset_episode_samples(self) -> None:
        self._episode_samples = {
            "pitch": [],
            "roll": [],
            "pitch_rate": [],
            "drift": [],
            "wheel_contact": [],
            "body_contact": [],
            "action": [],
            "leg_torque": [],
            "wheel_current": [],
            "wheel_torque": [],
            "reward": [],
            "reward_raw": [],
            "reward_components": [],
        }
        self._episode_return = 0.0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.sim.reset(RESET_HEIGHT_M)
        options = options or {}
        pitch_bounds = options.get("pitch_bounds_rad", (-PITCH_INIT_LIMIT_RAD, PITCH_INIT_LIMIT_RAD))
        pitch_rate_bounds = options.get(
            "pitch_rate_bounds_rad_s",
            (-PITCH_RATE_INIT_LIMIT_RAD_S, PITCH_RATE_INIT_LIMIT_RAD_S),
        )
        pitch = (
            float(options["initial_pitch_rad"])
            if "initial_pitch_rad" in options
            else float(self.rng.uniform(*pitch_bounds))
        )
        pitch_rate = (
            float(options["initial_pitch_rate_rad_s"])
            if "initial_pitch_rate_rad_s" in options
            else float(self.rng.uniform(*pitch_rate_bounds))
        )
        if (
            "pitch_bounds_rad" not in options
            and "initial_pitch_rad" not in options
            and not -PITCH_INIT_LIMIT_RAD <= pitch <= PITCH_INIT_LIMIT_RAD
        ):
            raise ValueError("training pitch sample is outside the fixed ±3 degree range")
        if (
            "pitch_rate_bounds_rad_s" not in options
            and "initial_pitch_rate_rad_s" not in options
            and not -PITCH_RATE_INIT_LIMIT_RAD_S <= pitch_rate <= PITCH_RATE_INIT_LIMIT_RAD_S
        ):
            raise ValueError("training pitch-rate sample is outside the fixed ±0.1 rad/s range")
        quat_xyzw = Rotation.from_euler("xyz", [0.0, pitch, 0.0]).as_quat()
        self.sim.data.qpos[3:7] = [
            quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]
        ]
        self.sim.data.qpos[self.sim.hip_qpos_adr] = Q_REF
        self.sim.data.qpos[self._passive_qpos_adr] = self._passive_reset_qpos
        self.sim.data.qvel[:] = 0.0
        # Free-joint angular velocity is local; body Y is the requested pitch axis.
        self.sim.data.qvel[4] = pitch_rate
        mujoco.mj_forward(self.sim.model, self.sim.data)

        self.steps = 0
        self.nominal_xy = self.sim.data.qpos[:2].copy()
        self.nominal_yaw = 0.0
        self.previous_action[:] = 0.0
        self._prev_leg_lengths = self.sim.leg_lengths().copy()
        self._dof_vel_hist = deque(maxlen=DOF_ACC_WINDOW + 1)
        self._dof_vel_hist.append(
            np.concatenate([self.sim.joint_velocities(), self.sim.wheel_velocities()])
        )
        self._last_joint_torque[:] = 0.0
        self._last_wheel_torque[:] = 0.0
        self.actuators.reset()
        self._episode_start_pitch = pitch
        self._episode_start_pitch_rate = pitch_rate
        self._reset_episode_samples()
        obs = self._get_observation()
        info = {
            "initial_pitch_rad": pitch,
            "initial_pitch_rate_rad_s": pitch_rate,
            "reset_height_m": RESET_HEIGHT_M,
            "reset_leg_length_m": self.sim.leg_lengths().tolist(),
            "reset_passive_joint_qpos": self._passive_reset_qpos.tolist(),
            "reset_equality_residual_max_m": float(
                np.max(np.abs(self._equality_residual()), initial=0.0)
            ),
            "domain_randomization_enabled": False,
            "observation_noise_enabled": False,
            "observation_noise_width_max": max(
                (float(np.max(np.asarray(v))) for v in self.observation_encoder.noise.values()),
                default=0.0,
            ),
        }
        return obs, info

    def _get_observation(self) -> np.ndarray:
        sim = self.sim
        rpy = self._rpy(sim)
        rotation = Rotation.from_quat(
            [sim.base_quat[1], sim.base_quat[2], sim.base_quat[3], sim.base_quat[0]]
        )
        gravity_body = rotation.inv().apply(np.array([0.0, 0.0, -1.0]))
        lin_vel_body = sim.body_frame(sim.base_lin_vel_world)
        ang_vel_body = sim.body_frame(sim.base_ang_vel_world)
        leg_lengths = sim.leg_lengths()
        leg_rate = (leg_lengths - self._prev_leg_lengths) / POLICY_DT_S
        dof_vel = np.concatenate([sim.joint_velocities(), sim.wheel_velocities()])
        if len(self._dof_vel_hist) >= 2:
            span = (len(self._dof_vel_hist) - 1) * POLICY_DT_S
            dof_acc = (self._dof_vel_hist[-1] - self._dof_vel_hist[0]) / span
        else:
            dof_acc = np.zeros(6, dtype=np.float64)
        wheel_force, wheel_hit, body_hit, airborne = sim.contact_state()
        yaw = float(rpy[2])
        heading_error = float(np.arctan2(np.sin(yaw - self.nominal_yaw), np.cos(yaw - self.nominal_yaw)))
        base_x, base_y = float(sim.base_pos[0]), float(sim.base_pos[1])
        base_height = float(sim.base_pos[2] - sim.support_height(base_x, base_y))
        blocks = {
            "gravity": gravity_body,
            "base_ang_vel": ang_vel_body,
            "heading_error_sincos": np.array(
                [np.sin(heading_error), np.cos(heading_error)], dtype=np.float64
            ),
            "base_lin_vel": lin_vel_body,
            "base_height_rel": np.array([base_height], dtype=np.float64),
            "base_pos_rel": sim.data.qpos[:2] - self.nominal_xy,
            "terrain_preview": sim.terrain_preview(
                sim.base_pos, yaw, np.array([0.10, 0.50, 0.90], dtype=np.float64)
            ),
            "leg_joint_pos": sim.joint_positions(),
            "leg_joint_vel": sim.joint_velocities(),
            "wheel_joint_vel": sim.wheel_velocities(),
            "command": np.array([0.0, 0.0, 0.0, RESET_HEIGHT_M, 0.0]),
            "previous_action": self.previous_action,
            "base_lin_vel_true": lin_vel_body,
            "base_pos_rel_true": sim.data.qpos[:2] - self.nominal_xy,
            "leg_length": np.concatenate([leg_lengths, leg_rate]),
            "wheel_contact_force": wheel_force,
            "contact_flag": np.concatenate([wheel_hit, [body_hit, airborne]]),
            "terrain_scan": sim.terrain_scan(base_x, np.zeros(17, dtype=np.float64)),
            "dof_acc": dof_acc,
            "torques": np.concatenate([self._last_joint_torque, self._last_wheel_torque]),
            "domain_params": np.zeros(12, dtype=np.float64),
        }
        observation, _, _ = self.observation_encoder.encode(blocks, self.rng)
        return observation

    def _state(self) -> dict[str, Any]:
        rpy = self._rpy(self.sim)
        ang_vel_body = self.sim.body_frame(self.sim.base_ang_vel_world)
        lin_vel_body = self.sim.body_frame(self.sim.base_lin_vel_world)
        wheel_force, wheel_hit, body_hit, airborne = self.sim.contact_state()
        leg_lengths = self.sim.leg_lengths()
        height = float(
            self.sim.base_pos[2]
            - self.sim.support_height(float(self.sim.base_pos[0]), float(self.sim.base_pos[1]))
        )
        drift = float(np.linalg.norm(self.sim.data.qpos[:2] - self.nominal_xy))
        return {
            "roll": float(rpy[0]),
            "pitch": float(rpy[1]),
            "yaw": float(rpy[2]),
            "roll_rate": float(ang_vel_body[0]),
            "pitch_rate": float(ang_vel_body[1]),
            "yaw_rate": float(ang_vel_body[2]),
            "vx": float(lin_vel_body[0]),
            "vy": float(lin_vel_body[1]),
            "vz": float(lin_vel_body[2]),
            "height": height,
            "drift": drift,
            "leg_lengths": leg_lengths,
            "wheel_force": wheel_force,
            "wheel_hit": wheel_hit,
            "body_hit": float(body_hit),
            "airborne": float(airborne),
        }

    def _failure_reason(self, state: dict[str, Any], finite: bool) -> str | None:
        if not finite:
            return "numerical_failure"
        if max(abs(state["roll"]), abs(state["pitch"])) > TILT_FAILURE_LIMIT_RAD:
            return "tilt_limit"
        if state["body_hit"] > 0.5:
            return "body_contact"
        if not HEIGHT_FAILURE_LIMIT_M[0] <= state["height"] <= HEIGHT_FAILURE_LIMIT_M[1]:
            return "height_limit"
        if state["drift"] > DRIFT_FAILURE_LIMIT_M:
            return "drift_limit"
        return None

    @staticmethod
    def _longest_true_run(mask: np.ndarray) -> int:
        longest = current = 0
        for value in np.asarray(mask, dtype=bool).reshape(-1):
            current = current + 1 if value else 0
            longest = max(longest, current)
        return longest

    def _episode_summary(self, terminated: bool, truncated: bool, reason: str) -> dict[str, Any]:
        samples = self._episode_samples
        pitches = np.asarray(samples["pitch"], dtype=np.float64)
        rolls = np.asarray(samples["roll"], dtype=np.float64)
        rates = np.asarray(samples["pitch_rate"], dtype=np.float64)
        drift = np.asarray(samples["drift"], dtype=np.float64)
        contacts = np.asarray(samples["wheel_contact"], dtype=np.float64)
        actions = np.asarray(samples["action"], dtype=np.float64)
        leg_tau = np.asarray(samples["leg_torque"], dtype=np.float64)
        wheel_i = np.asarray(samples["wheel_current"], dtype=np.float64)
        wheel_tau = np.asarray(samples["wheel_torque"], dtype=np.float64)
        reward_terms = samples["reward_components"]
        current_sat = np.abs(wheel_i) >= 0.99 * self.params.wheel.current_limit_a
        wheel_tau_sat = np.abs(wheel_tau) >= 0.99 * self.params.wheel.joint_torque_limit
        leg_tau_sat = np.abs(leg_tau) >= 0.99 * self.params.joint.torque_limit
        wheel_sat = current_sat | wheel_tau_sat
        return {
            "initial_pitch_rad": self._episode_start_pitch,
            "initial_pitch_rate_rad_s": self._episode_start_pitch_rate,
            "steps": int(self.steps),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "termination_reason": reason,
            "episode_return": float(self._episode_return),
            "pitch_rms_rad": float(np.sqrt(np.mean(np.square(pitches)))) if pitches.size else 0.0,
            "pitch_peak_rad": float(np.max(np.abs(pitches), initial=0.0)),
            "roll_rms_rad": float(np.sqrt(np.mean(np.square(rolls)))) if rolls.size else 0.0,
            "roll_peak_rad": float(np.max(np.abs(rolls), initial=0.0)),
            "pitch_rate_rms_rad_s": float(np.sqrt(np.mean(np.square(rates)))) if rates.size else 0.0,
            "drift_max_m": float(np.max(drift, initial=0.0)),
            "wheel_left_contact_fraction": float(np.mean(contacts[:, 0])) if contacts.size else 0.0,
            "wheel_right_contact_fraction": float(np.mean(contacts[:, 1])) if contacts.size else 0.0,
            "body_contact_steps": int(sum(samples["body_contact"])),
            "action_mean": actions.mean(axis=0).tolist() if actions.size else [0.0] * ACTION_DIM,
            "action_std": actions.std(axis=0).tolist() if actions.size else [0.0] * ACTION_DIM,
            "action_saturation_fraction": (
                np.mean(np.abs(actions) >= 0.99, axis=0).tolist() if actions.size else [0.0] * ACTION_DIM
            ),
            "leg_torque_peak_nm": np.max(np.abs(leg_tau), axis=0).tolist() if leg_tau.size else [0.0] * 4,
            "leg_torque_saturation_fraction": (
                np.mean(leg_tau_sat, axis=0).tolist() if leg_tau.size else [0.0] * 4
            ),
            "leg_torque_saturation_max_consecutive_steps": (
                [self._longest_true_run(leg_tau_sat[:, i]) for i in range(4)]
                if leg_tau_sat.size else [0] * 4
            ),
            "wheel_current_peak_a": np.max(np.abs(wheel_i), axis=0).tolist() if wheel_i.size else [0.0] * 2,
            "wheel_current_saturation_fraction": (
                np.mean(current_sat, axis=0).tolist() if current_sat.size else [0.0] * 2
            ),
            "wheel_torque_peak_nm": np.max(np.abs(wheel_tau), axis=0).tolist() if wheel_tau.size else [0.0] * 2,
            "wheel_torque_saturation_fraction": (
                np.mean(wheel_tau_sat, axis=0).tolist() if wheel_tau_sat.size else [0.0] * 2
            ),
            "wheel_saturation_fraction": (
                np.mean(wheel_sat, axis=0).tolist() if wheel_sat.size else [0.0] * 2
            ),
            "wheel_saturation_max_consecutive_steps": (
                [self._longest_true_run(wheel_sat[:, i]) for i in range(2)]
                if wheel_sat.size else [0] * 2
            ),
            "reward_component_mean": {
                key: float(np.mean([row[key] for row in reward_terms]))
                for key in (reward_terms[0].keys() if reward_terms else [])
            },
            "reward_raw_mean": {
                key: float(np.mean([row[key] for row in samples["reward_raw"]]))
                for key in (samples["reward_raw"][0].keys() if samples["reward_raw"] else [])
            },
        }

    def step(self, action: np.ndarray):
        action, _, _ = map_stand_offset_action(action)
        state_before = {
            "joint_pos": self.sim.joint_positions(),
            "joint_vel": self.sim.joint_velocities(),
            "wheel_vel": self.sim.wheel_velocities(),
        }
        control = self.actuators.compute(
            action,
            joint_pos=state_before["joint_pos"],
            joint_vel=state_before["joint_vel"],
            wheel_vel=state_before["wheel_vel"],
            dt=POLICY_DT_S,
        )
        self.sim.data.ctrl[:4] = control["joint_ctrl"]
        self.sim.data.ctrl[4:] = control["wheel_ctrl"]
        self._last_joint_torque = control["joint_torque"].copy()
        self._last_wheel_torque = control["wheel_torque"].copy()
        for _ in range(MOTOR_TICKS_PER_POLICY_STEP):
            self.sim.step()
        self.steps += 1

        finite = bool(
            np.isfinite(self.sim.data.qpos).all()
            and np.isfinite(self.sim.data.qvel).all()
            and np.isfinite(self.sim.data.qacc).all()
            and np.isfinite(self.sim.data.ctrl).all()
        )
        if finite:
            state = self._state()
            reason = self._failure_reason(state, finite=True)
        else:
            state = {
                "roll": 0.0, "pitch": 0.0, "yaw": 0.0, "pitch_rate": 0.0,
                "roll_rate": 0.0, "yaw_rate": 0.0, "vx": 0.0, "vy": 0.0,
                "vz": 0.0, "height": RESET_HEIGHT_M, "drift": 0.0,
                "leg_lengths": np.full(2, NOMINAL_LEG_LENGTH_M),
                "wheel_force": np.zeros(2), "wheel_hit": np.zeros(2),
                "body_hit": 0.0, "airborne": 1.0,
            }
            reason = self._failure_reason(state, finite=False)
        terminated = reason is not None
        truncated = self.steps >= EPISODE_STEPS and not terminated

        torque_limit = np.asarray(
            [self.params.joint.torque_limit] * 4
            + [self.params.wheel.joint_torque_limit] * 2,
            dtype=np.float64,
        )
        torque = np.concatenate([control["joint_torque"], control["wheel_torque"]])
        normalized_effort = float(np.mean(np.square(torque / torque_limit)))
        reward, raw_reward, weighted_reward = compute_stand_reward(
            roll_rad=state["roll"],
            pitch_rad=state["pitch"],
            pitch_rate_rad_s=state["pitch_rate"],
            horizontal_velocity_m_s=float(np.hypot(state["vx"], state["vy"])),
            position_drift_m=state["drift"],
            height_error_m=state["height"] - RESET_HEIGHT_M,
            leg_length_error_m=float(np.mean(state["leg_lengths"])) - NOMINAL_LEG_LENGTH_M,
            action=action,
            previous_action=self.previous_action,
            normalized_effort=normalized_effort,
            terminated=terminated,
        )
        self._episode_return += reward
        self._episode_samples["pitch"].append(state["pitch"])
        self._episode_samples["roll"].append(state["roll"])
        self._episode_samples["pitch_rate"].append(state["pitch_rate"])
        self._episode_samples["drift"].append(state["drift"])
        self._episode_samples["wheel_contact"].append(state["wheel_hit"].copy())
        self._episode_samples["body_contact"].append(int(state["body_hit"] > 0.5))
        self._episode_samples["action"].append(action.copy())
        self._episode_samples["leg_torque"].append(control["joint_torque"].copy())
        self._episode_samples["wheel_current"].append(control["wheel_current"].copy())
        self._episode_samples["wheel_torque"].append(control["wheel_torque"].copy())
        self._episode_samples["reward"].append(float(reward))
        self._episode_samples["reward_raw"].append(raw_reward)
        self._episode_samples["reward_components"].append(weighted_reward)

        current_dof_vel = np.concatenate(
            [self.sim.joint_velocities(), self.sim.wheel_velocities()]
        )
        self._dof_vel_hist.append(current_dof_vel.copy())
        self._prev_leg_lengths = state["leg_lengths"].copy()
        self.previous_action = action.copy()

        if finite:
            observation = self._get_observation()
        else:
            observation = np.zeros(OBS_DIM, dtype=np.float32)
        info: dict[str, Any] = {
            "termination_reason": reason or ("time_limit" if truncated else "running"),
            "pitch_rad": state["pitch"],
            "roll_rad": state["roll"],
            "pitch_rate_rad_s": state["pitch_rate"],
            "base_vx_m_s": state["vx"],
            "base_vy_m_s": state["vy"],
            "base_height_m": state["height"],
            "leg_length_m": state["leg_lengths"].tolist(),
            "planar_drift_m": state["drift"],
            "wheel_contact": state["wheel_hit"].tolist(),
            "body_contact": state["body_hit"],
            "action_applied": action.tolist(),
            "wheel_target_rad_s": control["wheel_target"].tolist(),
            "joint_target_rad": control["joint_target"].tolist(),
            "wheel_current_a": control["wheel_current"].tolist(),
            "wheel_torque_nm": control["wheel_torque"].tolist(),
            "joint_torque_nm": control["joint_torque"].tolist(),
            "reward_raw": raw_reward,
            "reward_components": weighted_reward,
            "reward_total": float(reward),
            "sim_finite": finite,
            "policy_steps": self.steps,
        }
        if terminated or truncated:
            info["episode_metrics"] = self._episode_summary(
                terminated, truncated, reason or "time_limit"
            )
        return observation, reward, terminated, truncated, info

    def apply_pitch_rate_perturbation(self, delta_rad_s: float) -> None:
        """Apply an evaluation-only instantaneous body-Y rate disturbance."""
        if not np.isfinite(delta_rad_s):
            raise ValueError("pitch-rate perturbation must be finite")
        self.sim.data.qvel[4] += float(delta_rad_s)
        mujoco.mj_forward(self.sim.model, self.sim.data)
        current = np.concatenate([self.sim.joint_velocities(), self.sim.wheel_velocities()])
        self._dof_vel_hist.clear()
        self._dof_vel_hist.append(current)

    def refresh_observation(self) -> np.ndarray:
        """Return actor/critic observations after an evaluation-only disturbance."""
        return self._get_observation()

    def close(self) -> None:
        return None
