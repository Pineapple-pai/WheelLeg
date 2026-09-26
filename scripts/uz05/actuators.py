"""把 PPO 的物理目标变成 MuJoCo 执行器输入。

动作路径只有两层：

* 腿：PPO 直接给四个绝对关节位置目标，执行器跑位置/速度 PD；
* 轮：PPO 给左右轮绝对角速度目标，执行器跑速度环，再换算为电流和力矩。

没有额外的姿态反馈、动作混合或高层动作生成，因而动作权限和实物部署
接口完全一致。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .spec import (ACTION_SPEC, WHEEL_ANGULAR_TO_BODY_X, JointActuatorParams,
                   RobotParams, WheelEscParams)

WHEEL_SPEED_SCALE = float(ACTION_SPEC[0][2])
HIP_POSITION_SCALE = float(ACTION_SPEC[1][2])


def quantize_linear(value, lower: float, upper: float, bits: int):
    """Round-trip a value through the unsigned linear mapping used by DM MIT."""
    clipped = np.clip(np.asarray(value, dtype=np.float64), lower, upper)
    levels = (1 << bits) - 1
    encoded = np.rint((clipped - lower) * levels / (upper - lower)).astype(np.int64)
    decoded = encoded.astype(np.float64) * (upper - lower) / levels + lower
    return decoded, encoded


@dataclass
class JointPD:
    params: JointActuatorParams = field(default_factory=JointActuatorParams)
    quantize_mit: bool = False

    def command(self, target: np.ndarray, q: np.ndarray, qd: np.ndarray) -> dict:
        """Apply the DM MIT command semantics used by the real motor driver."""
        p = self.params
        p_des = np.asarray(target, dtype=np.float64)
        v_des = np.full_like(p_des, p.velocity_target)
        kp = np.full_like(p_des, p.kp)
        kd = np.full_like(p_des, p.kd)
        t_ff = np.full_like(p_des, p.torque_feedforward)
        if self.quantize_mit:
            p_des, _ = quantize_linear(p_des, *p.mit_position_range, 16)
            v_des, _ = quantize_linear(v_des, *p.mit_velocity_range, 12)
            kp, _ = quantize_linear(kp, *p.mit_kp_range, 12)
            kd, _ = quantize_linear(kd, *p.mit_kd_range, 12)
            t_ff, _ = quantize_linear(t_ff, *p.mit_torque_range, 12)
        demand = kp * (p_des - q) + kd * (v_des - qd) + t_ff
        torque = np.clip(demand, -p.torque_limit, p.torque_limit)
        return {
            "torque": torque,
            "p_des": p_des,
            "v_des": v_des,
            "kp": kp,
            "kd": kd,
            "t_ff": t_ff,
        }

    def torque(self, target: np.ndarray, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        return self.command(target, q, qd)["torque"]


@dataclass
class WheelSpeedLoop:
    params: WheelEscParams = field(default_factory=WheelEscParams)
    quantize_c620: bool = False
    integral: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.integral = np.zeros(2, dtype=np.float64)

    def reset(self) -> None:
        self.integral[:] = 0.0

    def current(self, target: np.ndarray, actual: np.ndarray, dt: float) -> np.ndarray:
        target = np.clip(target, -self.params.velocity_limit, self.params.velocity_limit)
        error = np.clip(target - actual, -self.params.speed_error_limit,
                        self.params.speed_error_limit)
        self.integral = np.clip(
            self.integral + error * dt,
            -self.params.integral_limit_a,
            self.params.integral_limit_a,
        )
        demand = self.params.speed_kp_a_per_rad_s * error
        demand += self.params.speed_ki_a_per_rad_s2 * self.integral
        demand = np.clip(demand, -self.params.current_limit_a,
                         self.params.current_limit_a)
        if self.quantize_c620:
            scale = self.params.c620_full_scale_command / self.params.c620_full_scale_current_a
            raw = np.rint(demand * scale)
            raw = np.clip(
                raw,
                -self.params.c620_full_scale_command,
                self.params.c620_full_scale_command,
            )
            demand = raw / scale
        return demand

    def current_command_raw(self, current: np.ndarray) -> np.ndarray:
        scale = self.params.c620_full_scale_command / self.params.c620_full_scale_current_a
        raw = np.rint(np.asarray(current, dtype=np.float64) * scale)
        return np.clip(
            raw,
            -self.params.c620_full_scale_command,
            self.params.c620_full_scale_command,
        ).astype(np.int32)

    def torque_limit_at_speed(self, speed: np.ndarray) -> np.ndarray:
        p = self.params
        speed_abs = np.abs(np.asarray(speed, dtype=np.float64))
        rated = max(float(p.velocity_limit), 1e-6)
        no_load = max(float(p.no_load_velocity), rated + 1e-6)
        stall = float(p.stall_torque_limit)
        rated_torque = float(p.joint_torque_limit)
        limit = np.where(
            speed_abs <= rated,
            stall + (rated_torque - stall) * speed_abs / rated,
            rated_torque * np.clip((no_load - speed_abs) / (no_load - rated), 0.0, 1.0),
        )
        return np.maximum(limit, 0.0)

    def torque_from_current(self, current: np.ndarray,
                            speed: np.ndarray | None = None) -> np.ndarray:
        p = self.params
        torque = current * p.torque_per_amp_joint * p.efficiency
        if p.torque_speed_envelope_enabled and speed is not None:
            limit = self.torque_limit_at_speed(speed)
            torque = np.clip(torque, -limit, limit)
        return np.clip(torque, -p.joint_torque_limit, p.joint_torque_limit)

    def motor_command(self, torque: np.ndarray) -> np.ndarray:
        return torque / self.params.gear_ratio


@dataclass
class ActuatorBank:
    robot: RobotParams = field(default_factory=RobotParams)
    joint: JointPD = field(default_factory=JointPD)
    wheel: WheelSpeedLoop = field(default_factory=WheelSpeedLoop)
    diagnostics: dict = field(default_factory=dict, init=False)

    def reset(self) -> None:
        self.wheel.reset()
        self.diagnostics = {}

    def compute(self, action: np.ndarray, *, state: dict, dt: float) -> dict:
        """Apply the six PPO targets without changing their authority."""
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
        if action.size != 6:
            raise ValueError(f"expected six action values, got {action.size}")

        # Convert body-forward-positive policy targets to native wheel-joint
        # velocity before the 500 Hz PI loop.  This is a fixed axis convention,
        # not an extra controller or policy residual.
        wheel_target = action[:2] * WHEEL_SPEED_SCALE * WHEEL_ANGULAR_TO_BODY_X
        hip_target = action[2:6] * HIP_POSITION_SCALE
        mit = self.joint.command(
            hip_target,
            np.asarray(state["joint_pos"], dtype=np.float64),
            np.asarray(state["joint_vel"], dtype=np.float64),
        )
        leg_torque = mit["torque"]
        wheel_velocity = np.asarray(state["wheel_vel"], dtype=np.float64)
        wheel_current = self.wheel.current(wheel_target, wheel_velocity, dt)
        wheel_current_raw = self.wheel.current_command_raw(wheel_current)
        wheel_torque = self.wheel.torque_from_current(
            wheel_current, speed=wheel_velocity
        )
        self.diagnostics = {
            "wheel_target_left": float(wheel_target[0]),
            "wheel_target_right": float(wheel_target[1]),
            "wheel_speed_abs": float(np.abs(wheel_velocity).mean()),
            "wheel_velocity_error_left": float(wheel_target[0] - wheel_velocity[0]),
            "wheel_velocity_error_right": float(wheel_target[1] - wheel_velocity[1]),
            "wheel_current_left": float(wheel_current[0]),
            "wheel_current_right": float(wheel_current[1]),
            "wheel_torque_left": float(wheel_torque[0]),
            "wheel_torque_right": float(wheel_torque[1]),
            "wheel_target_abs": float(np.abs(wheel_target).mean()),
            "wheel_current_abs": float(np.abs(wheel_current).mean()),
            "wheel_torque_abs_mean": float(np.abs(wheel_torque).mean()),
            "leg_target_abs": float(np.abs(hip_target).mean()),
            "leg_torque_abs_mean": float(np.abs(leg_torque).mean()),
            "leg_torque_peak_abs": float(np.abs(leg_torque).max()),
            "mit_position_quantization_abs_max": float(
                np.abs(mit["p_des"] - hip_target).max()
            ),
            "action_abs": float(np.abs(action).mean()),
        }
        return {
            "leg_ctrl": leg_torque,
            "wheel_ctrl": self.wheel.motor_command(wheel_torque),
            "leg_torque": leg_torque,
            "wheel_torque": wheel_torque,
            "wheel_current": wheel_current,
            "wheel_current_raw": wheel_current_raw,
            "wheel_target": wheel_target,
            "hip_target": hip_target,
            "mit_p_des": mit["p_des"],
            "mit_v_des": mit["v_des"],
            "mit_kp": mit["kp"],
            "mit_kd": mit["kd"],
            "mit_t_ff": mit["t_ff"],
        }
