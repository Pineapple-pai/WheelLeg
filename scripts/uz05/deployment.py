"""Pure protocol helpers shared by simulation checks and deployment code.

This module intentionally performs no CAN I/O.  Motor IDs, zero offsets and
installation signs remain hardware-specific and must be filled after bench
testing.
"""

from __future__ import annotations

from dataclasses import dataclass
import struct

import numpy as np

from .actuators import HIP_POSITION_SCALE, WHEEL_SPEED_SCALE
from .spec import (WHEEL_ANGULAR_TO_BODY_X, JointActuatorParams, RobotParams,
                   WheelEscParams)


def float_to_uint(value: float, lower: float, upper: float, bits: int) -> int:
    value = float(np.clip(value, lower, upper))
    return int(round((value - lower) * ((1 << bits) - 1) / (upper - lower)))


def uint_to_float(value: int, lower: float, upper: float, bits: int) -> float:
    mask = (1 << bits) - 1
    return float((int(value) & mask) * (upper - lower) / mask + lower)


@dataclass(frozen=True)
class DmMitCommand:
    p_des: float
    v_des: float = 0.0
    kp: float = 100.0
    kd: float = 4.0
    t_ff: float = 0.0

    def encode(self, params: JointActuatorParams | None = None) -> bytes:
        p = params or JointActuatorParams()
        pos = float_to_uint(self.p_des, *p.mit_position_range, 16)
        vel = float_to_uint(self.v_des, *p.mit_velocity_range, 12)
        kp = float_to_uint(self.kp, *p.mit_kp_range, 12)
        kd = float_to_uint(self.kd, *p.mit_kd_range, 12)
        torque = float_to_uint(self.t_ff, *p.mit_torque_range, 12)
        return bytes((
            pos >> 8,
            pos & 0xFF,
            vel >> 4,
            ((vel & 0xF) << 4) | (kp >> 8),
            kp & 0xFF,
            kd >> 4,
            ((kd & 0xF) << 4) | (torque >> 8),
            torque & 0xFF,
        ))


def c620_current_to_raw(current_a: float, params: WheelEscParams | None = None) -> int:
    p = params or WheelEscParams()
    current = float(np.clip(
        current_a, -p.c620_full_scale_current_a, p.c620_full_scale_current_a
    ))
    return int(round(current * p.c620_full_scale_command / p.c620_full_scale_current_a))


def encode_c620_group(currents_by_id: dict[int, float], group: int = 0x200) -> bytes:
    """Encode IDs 1-4 (0x200) or 5-8 (0x1FF) as four signed currents."""
    if group not in (0x200, 0x1FF):
        raise ValueError("C620 group must be 0x200 or 0x1FF")
    ids = range(1, 5) if group == 0x200 else range(5, 9)
    raw = [c620_current_to_raw(currents_by_id.get(motor_id, 0.0)) for motor_id in ids]
    return struct.pack(">hhhh", *raw)


def action_to_physical_targets(
    action: np.ndarray, robot: RobotParams | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Convert the six PPO actions into native joint-speed and position targets."""
    values = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
    if values.size != 6:
        raise ValueError(f"expected six actions, got {values.size}")
    robot = robot or RobotParams()
    wheel_target = values[:2] * WHEEL_SPEED_SCALE * WHEEL_ANGULAR_TO_BODY_X
    leg_target = values[2:] * HIP_POSITION_SCALE
    return wheel_target, leg_target
