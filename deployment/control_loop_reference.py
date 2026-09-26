"""Hardware-independent reference for the 500 Hz / 125 Hz control boundary.

This is not a hardware driver.  It turns calibrated feedback and a 38-D actor
observation into CAN payloads so the embedded implementation has one testable
behavioral reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from uz05.deployment import DmMitCommand, action_to_physical_targets, encode_c620_group
from uz05.onnx_policy import OnnxPolicy
from uz05.spec import MOTOR_CONTROL_DT, JointActuatorParams, WheelEscParams


@dataclass(frozen=True)
class CalibratedHardware:
    dm_can_ids: tuple[int, int, int, int]
    dm_zero_offsets_rad: tuple[float, float, float, float]
    dm_policy_to_raw_sign: tuple[int, int, int, int]
    c620_ids: tuple[int, int]
    # Raw rotor sign -> simulation's native wheel-joint sign.  The fixed
    # body-forward -> joint sign is already applied by action_to_physical_targets.
    wheel_raw_to_policy_sign: tuple[int, int]
    wheel_gear_ratio: float

    def __post_init__(self) -> None:
        if any(sign not in (-1, 1) for sign in self.dm_policy_to_raw_sign):
            raise ValueError("all DM direction signs must be calibrated to -1 or 1")
        if any(sign not in (-1, 1) for sign in self.wheel_raw_to_policy_sign):
            raise ValueError("both wheel direction signs must be calibrated to -1 or 1")
        if len(set(self.dm_can_ids)) != 4 or len(set(self.c620_ids)) != 2:
            raise ValueError("motor CAN IDs must be unique within each motor family")
        if not (all(1 <= motor_id <= 4 for motor_id in self.c620_ids)
                or all(5 <= motor_id <= 8 for motor_id in self.c620_ids)):
            raise ValueError("both C620 IDs must belong to the same 0x200/0x1FF group")
        if self.wheel_gear_ratio <= 0.0:
            raise ValueError("wheel gear ratio must be measured and positive")


@dataclass(frozen=True)
class MotorFeedback:
    wheel_rotor_rpm: np.ndarray


@dataclass(frozen=True)
class ControlFrames:
    dm_frames: tuple[tuple[int, bytes], ...]
    c620_group_id: int
    c620_payload: bytes
    policy_action: np.ndarray
    wheel_target_rad_s: np.ndarray
    leg_target_rad: np.ndarray


class ControlLoopCore:
    def __init__(
        self,
        onnx_model: str | Path,
        hardware: CalibratedHardware,
        *,
        wheel_current_limit_a: float,
        leg_torque_limit_nm: float,
        policy_timeout_motor_ticks: int = 8,
    ):
        if wheel_current_limit_a <= 0.0 or leg_torque_limit_nm <= 0.0:
            raise ValueError("bring-up current and torque limits must be positive")
        self.policy = OnnxPolicy(onnx_model)
        self.hardware = hardware
        self.wheel = WheelEscParams(current_limit_a=wheel_current_limit_a)
        self.leg = JointActuatorParams(
            torque_limit=leg_torque_limit_nm,
            mit_torque_range=(-leg_torque_limit_nm, leg_torque_limit_nm),
        )
        self.integral = np.zeros(2, dtype=np.float64)
        self.action = np.zeros(6, dtype=np.float64)
        self.motor_tick_count = 0
        self.policy_timeout_motor_ticks = max(1, int(policy_timeout_motor_ticks))
        self.ticks_since_policy = self.policy_timeout_motor_ticks + 1

    def policy_tick(self, actor_observation: np.ndarray) -> np.ndarray:
        """Run from the independent 125 Hz task and latch a complete action."""
        self.action = self.policy.predict(actor_observation).astype(np.float64)
        self.ticks_since_policy = 0
        return self.action.copy()

    def motor_tick(self, feedback: MotorFeedback) -> ControlFrames:
        """Run at 500 Hz; never invokes ONNX or waits for the policy task."""
        if self.ticks_since_policy > self.policy_timeout_motor_ticks:
            raise RuntimeError("policy timeout: enter the hardware safe-stop path")
        wheel_target, leg_target_policy = action_to_physical_targets(self.action)

        signs = np.asarray(self.hardware.wheel_raw_to_policy_sign, dtype=np.float64)
        rotor_rpm = np.asarray(feedback.wheel_rotor_rpm, dtype=np.float64)
        wheel_speed_joint = signs * rotor_rpm * (2.0 * np.pi / 60.0) / self.hardware.wheel_gear_ratio
        error = np.clip(
            wheel_target - wheel_speed_joint,
            -self.wheel.speed_error_limit,
            self.wheel.speed_error_limit,
        )
        self.integral = np.clip(
            self.integral + error * MOTOR_CONTROL_DT,
            -self.wheel.integral_limit_a,
            self.wheel.integral_limit_a,
        )
        policy_current = (
            self.wheel.speed_kp_a_per_rad_s * error
            + self.wheel.speed_ki_a_per_rad_s2 * self.integral
        )
        policy_current = np.clip(
            policy_current, -self.wheel.current_limit_a, self.wheel.current_limit_a
        )
        raw_current = signs * policy_current

        dm_frames = []
        leg_target_raw = []
        for motor_id, zero, sign, target in zip(
            self.hardware.dm_can_ids,
            self.hardware.dm_zero_offsets_rad,
            self.hardware.dm_policy_to_raw_sign,
            leg_target_policy,
        ):
            raw_target = float(zero + sign * target)
            leg_target_raw.append(raw_target)
            payload = DmMitCommand(
                p_des=raw_target,
                v_des=self.leg.velocity_target,
                kp=self.leg.kp,
                kd=self.leg.kd,
                t_ff=self.leg.torque_feedforward,
            ).encode(self.leg)
            dm_frames.append((motor_id, payload))

        c620_group = 0x200 if max(self.hardware.c620_ids) <= 4 else 0x1FF
        c620_payload = encode_c620_group(
            dict(zip(self.hardware.c620_ids, raw_current)), group=c620_group
        )
        output = ControlFrames(
            dm_frames=tuple(dm_frames),
            c620_group_id=c620_group,
            c620_payload=c620_payload,
            policy_action=self.action.copy(),
            wheel_target_rad_s=wheel_target,
            leg_target_rad=np.asarray(leg_target_raw),
        )
        self.motor_tick_count += 1
        self.ticks_since_policy += 1
        return output
