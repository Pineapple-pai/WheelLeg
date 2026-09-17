"""执行器模型：关节位置-速度-PD，轮子速度→电调→电流→力矩。

两条链路都与实机接口一一对应：

* 关节  DM-J8009P：上位机给「位置 + 速度」指令，驱动器内部跑 PD 输出力矩。
* 轮子  M3508 + C620：上位机给「转速」指令，电调内部跑速度环并把输出折算成
  期望电流；电流经电机转矩常数变成力矩。

参数集中在 :mod:`uz05.spec`，后续用电调/电机实测数据替换即可，不需要改这里。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .spec import (
    ACTION_SPEC,
    BalanceParams,
    JointActuatorParams,
    RobotParams,
    WheelEscParams,
)

# 动作缩放只有一个来源：spec.ACTION_SPEC。
# （历史上这里硬编码 hip_scale = 0.50、而 spec 写 0.35，实际生效的是 0.50，
#   于是"文档说对齐官方 0.35、代码其实没对齐"——现在改成直接读 spec。）
ACTION_SCALE: dict[str, float] = {name: scale for name, _, scale in ACTION_SPEC}
POS_ACTION_SCALE = ACTION_SCALE["hip_position_offset"]   # 官方 pos_action_scale = 0.35
WHEEL_COMMON_SCALE = ACTION_SCALE["wheel_common"]        # 官方 vel_action_scale = 8.0
WHEEL_DIFF_SCALE = ACTION_SCALE["wheel_differential"]


@dataclass
class JointPDController:
    """位置 + 速度 + PD 的关节驱动器。"""

    params: JointActuatorParams = field(default_factory=JointActuatorParams)

    def torque(self, q_target: np.ndarray, qd_target: np.ndarray,
               q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        demand = self.params.kp * (q_target - q) + self.params.kd * (qd_target - qd)
        return np.clip(demand, -self.params.torque_limit, self.params.torque_limit)


@dataclass
class WheelEscController:
    """速度指令 → 电调速度环 → 期望电流（安培）。"""

    params: WheelEscParams = field(default_factory=WheelEscParams)
    integral: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.integral = np.zeros(2, dtype=np.float64)

    def reset(self) -> None:
        self.integral[:] = 0.0

    def current(self, target: np.ndarray, actual: np.ndarray, dt: float) -> np.ndarray:
        """返回两轮的期望电流（A，限幅后）。"""
        target = np.clip(target, -self.params.velocity_limit, self.params.velocity_limit)
        error = np.clip(target - actual, -self.params.speed_error_limit,
                        self.params.speed_error_limit)
        self.integral = np.clip(
            self.integral + error * dt, -self.params.integral_limit_a,
            self.params.integral_limit_a,
        )
        demand = self.params.speed_kp_a_per_rad_s * error
        if self.params.speed_ki_a_per_rad_s2:
            demand = demand + self.params.speed_ki_a_per_rad_s2 * self.integral
        return np.clip(demand, -self.params.current_limit_a, self.params.current_limit_a)

    def torque_from_current(self, current: np.ndarray) -> np.ndarray:
        """电流 → 关节侧力矩（Nm）。"""
        torque = (current * self.params.torque_per_amp_joint * self.params.efficiency)
        return np.clip(torque, -self.params.joint_torque_limit,
                       self.params.joint_torque_limit)

    def motor_command(self, joint_torque: np.ndarray) -> np.ndarray:
        """关节侧力矩 → MuJoCo motor 的 ctrl（= 力矩 / 减速比）。"""
        return joint_torque / self.params.gear_ratio


@dataclass
class BalanceController:
    """轮端外层平衡反馈，输出关节力矩需求，再折算成电流叠加到电调。

    符号约定（实测）：正轮力矩驱动车身朝 −X，前进（vx>0）需要负力矩。
    """

    params: BalanceParams = field(default_factory=BalanceParams)

    integral: float = field(default=0.0, init=False)

    def reset(self) -> None:
        self.integral = 0.0

    def torque(self, pitch: float, pitch_rate: float, body_vx: float,
               command_vx: float, dt: float, station_error: float = 0.0,
               command_active: bool = False) -> float:
        """俯仰 PD + 速度 PI + 零指令时的站定位置环。

        整条外环乘以 ``assist_scale``（辅助退火系数）。退到 0 时本函数输出恒为 0，
        姿态与位置完全由策略通过关节+轮子学习得到 —— 不允许靠外部硬约束维持姿态。
        符号与增益为实测标定值（见 :class:`BalanceParams`）。
        """
        if self.params.assist_scale <= 1e-6:
            self.integral = 0.0
            return 0.0
        error_v = body_vx - command_vx
        self.integral = float(np.clip(
            self.integral - error_v * dt,
            -self.params.integral_limit, self.params.integral_limit,
        ))
        torque = -self.params.pitch_kp * (pitch + self.params.pitch_kd * pitch_rate)
        torque -= self.params.body_speed_kp * error_v
        torque += self.params.body_speed_ki * self.integral
        if not command_active and self.params.station_kp:
            # 死区外才出力。**符号为实测标定：负增益**。
            # 闭环实测（900 步，死区 5 cm）::
            #     Kx = -50 → 漂移 8.1 cm    Kx = -18 → 9.9 cm
            #     Kx =  +6 → 25 cm 并触发 station_limit
            # 注意这与"锁死姿态下的开环测量"结论相反 —— 闭环下摆动力学主导，
            # 一律以闭环实测为准。
            boundary = float(np.copysign(
                max(abs(station_error) - self.params.station_deadband, 0.0), station_error
            ))
            torque -= self.params.station_kp * boundary
            torque -= self.params.station_kd * body_vx
        torque *= self.params.assist_scale
        if (self.params.overspeed_brake_kp and body_vx * command_vx > 0.0
                and abs(body_vx) > abs(command_vx) + self.params.overspeed_margin):
            torque += self.params.overspeed_brake_kp * error_v
        return float(torque)


@dataclass
class ActuatorBank:
    """把策略动作翻译成 MuJoCo 的 ``data.ctrl``。"""

    robot: RobotParams = field(default_factory=RobotParams)
    joint: JointPDController = field(default_factory=JointPDController)
    wheel: WheelEscController = field(default_factory=WheelEscController)
    balance: BalanceController = field(default_factory=BalanceController)

    # 最近一步的诊断量
    diagnostics: dict = field(default_factory=dict, init=False)

    def reset(self) -> None:
        self.wheel.reset()
        self.balance.reset()
        self.diagnostics = {}

    # ------------------------------------------------------------------
    @staticmethod
    def decode_wheel_basis(action_common: float, action_differential: float,
                           common_limit: float, differential_limit: float) -> np.ndarray:
        """归一化 [common, differential] → 左右轮速目标（rad/s）。"""
        common = float(np.clip(action_common, -1.0, 1.0)) * common_limit
        differential = float(np.clip(action_differential, -1.0, 1.0)) * differential_limit
        return np.array([common + differential, common - differential], dtype=np.float64)

    def compute(
        self,
        action: np.ndarray,
        *,
        state: dict,
        dt: float,
    ) -> dict:
        """返回 ``{"leg_ctrl": (4,), "wheel_ctrl": (2,), ...}`` 诊断字典。

        关节指令 = **固定零位** + 动作偏置（不能写成“当前位姿 + 偏置”，否则
        action=0 时位置误差恒为 0，PD 退化成纯阻尼，腿会直接软掉）。
        """
        # 动作 6 维（对齐开源）：[0:2] 轮速目标，[2:6] 4 个腿关节位置偏置。
        # 缩放全部来自 spec.ACTION_SPEC（官方 pos_action_scale = 0.35）。
        q_target = (
            state["nominal_joint_pos"]
            + np.asarray(action[2:6], dtype=np.float64) * POS_ACTION_SCALE
        )
        qd_target = np.zeros(4, dtype=np.float64)
        leg_torque = self.joint.torque(q_target, qd_target, state["joint_pos"], state["joint_vel"])

        wheel_target = self.decode_wheel_basis(
            action[0], action[1], WHEEL_COMMON_SCALE, WHEEL_DIFF_SCALE,
        )
        # 注意：**不**在零指令时屏蔽策略的轮速目标。
        # 姿态与位置都必须由策略自己学，屏蔽会让策略失去唯一的控制手段。
        esc_current = self.wheel.current(wheel_target, state["wheel_vel"], dt)

        balance_torque = self.balance.torque(
            state["pitch"], state["pitch_rate"], state["body_vx"],
            state["command_vx"], dt,
            station_error=state.get("station_error", 0.0),
            command_active=state["command_active"],
        )
        balance_current = np.full(2, balance_torque / self.wheel.params.torque_per_amp_joint)

        total_current = np.clip(
            esc_current + balance_current,
            -self.wheel.params.current_limit_a, self.wheel.params.current_limit_a,
        )
        wheel_torque = self.wheel.torque_from_current(total_current)

        self.diagnostics = {
            "wheel_target_left": float(wheel_target[0]),
            "wheel_target_right": float(wheel_target[1]),
            "wheel_current_left": float(total_current[0]),
            "wheel_current_right": float(total_current[1]),
            "wheel_torque_left": float(wheel_torque[0]),
            "wheel_torque_right": float(wheel_torque[1]),
            "balance_torque": float(balance_torque),
            "leg_torque_abs_mean": float(np.abs(leg_torque).mean()),
        }
        return {
            "leg_ctrl": leg_torque,
            "wheel_ctrl": self.wheel.motor_command(wheel_torque),
            "leg_torque": leg_torque,
            "wheel_torque": wheel_torque,
            "wheel_current": total_current,
            "wheel_target": wheel_target,
        }

    def wheel_target_hold(self, state: dict) -> np.ndarray:
        """零指令下把轮速目标设为 0（由电调速度环主动刹车）。"""
        return np.zeros(2, dtype=np.float64)
