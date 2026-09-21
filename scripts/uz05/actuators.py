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
# ★ 腿**差模**通道的独立缩放（rad / 归一化动作）。
#
# 为什么需要单独一个缩放：差模负责腿长，而腿机构能伸到 0.3765 m（无载实测），
# 但共模那 0.35 rad/动作 的缩放套在差模上时，只能给出 ±0.35 rad 关节差，
# 对应约 ±91 mm 腿长 —— 于是腿长被**动作缩放**卡在 ~0.26 m，而不是机构极限。
# 实测（`diag_leg_load_limit.py`）：关节差模 0.6 rad 时带载腿长 0.376 m。
# 取 0.7 rad 覆盖 0.15~0.35 m 全行程（q2 与 q4 反向，各自仍不超 0.7 rad）。
LEG_DIFF_ACTION_SCALE = 0.70
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
    """轮端平衡**教师**：级联结构（外环俯仰指令 + 内环俯仰 PD + 力矩预算）。

    输出的是**期望轮电流（A）**，与环境里"策略输出电流"在同一层，方便线性混合::

        current = assist * teacher_current + (1 - assist) * policy_current

    符号约定（MuJoCo 实测，见 spec.BalanceParams）：
    正轮电流 → 车身加速度 −X、俯仰角增大。因此"想朝 +X 修正"要先向 +θ 倾，
    外环两条支路都是**正号**。
    """

    params: BalanceParams = field(default_factory=BalanceParams)

    integral: float = field(default=0.0, init=False)

    def reset(self) -> None:
        self.integral = 0.0

    def current(self, pitch: float, pitch_rate: float, body_vx: float,
                command_vx: float, dt: float, station_error: float = 0.0,
                command_active: bool = False) -> float:
        """返回教师期望电流（A）。

        ⚠️ **即使 assist_scale = 0 也照常计算**：此时它不参与实际控制（由
        :class:`ActuatorBank` 的线性混合决定），但奖励里的 ``teacher_track``
        仍然需要它作为"完整平衡律应该输出多少"的参考。若在这里按 assist 提前
        返回 0，退火到 0 之后跟踪奖励就会把策略推向"输出 0"——而不是"自己把
        平衡律做出来"。
        """
        error_v = body_vx - command_vx
        self.integral = float(np.clip(
            self.integral + error_v * dt,
            -self.params.integral_limit, self.params.integral_limit,
        ))
        # 外环：把"速度/位置误差"翻译成一个**有界的俯仰指令**。
        #
        # 符号推导（静平衡，实测 τ_hold(θ) = −8.53·θ Nm）：
        # 内环令 τ = kp(θ_ref − θ)，稳态要求 τ = τ_hold ⇒
        #   θ = (kp·θ_ref + kv·error_v + kx·x) / (kp − 8.53)
        # 想让 x>0 时车身后仰（θ<0，随后 v̇<0 往回走）就必须取**负号**：
        #   θ_ref = −(kv·error_v + kx·x_error)
        # 这与实测标定的加性形式 ``tau = -kp*θ - kv*vx - kx*x`` 完全同构。
        theta_ref = self.params.speed_sign * (
            self.params.body_speed_kp * error_v
            + self.params.body_speed_ki * self.integral
        ) / self.params.pitch_kp
        if not command_active and self.params.station_kp:
            theta_ref += self.params.station_sign * self.params.station_kp * station_error
            theta_ref += self.params.station_sign * self.params.station_kd * body_vx
        theta_ref = float(np.clip(theta_ref, -self.params.theta_max, self.params.theta_max))

        # 内环：俯仰 PD。微分增益 = pitch_kp * pitch_kd（有界，不会打饱和）。
        torque = (self.params.pitch_kp * (theta_ref - pitch)
                  - self.params.pitch_kp * self.params.pitch_kd * pitch_rate)
        current = torque / WheelEscParams.torque_per_amp_joint
        return float(np.clip(current, -self.params.teacher_current_limit,
                             self.params.teacher_current_limit))


@dataclass
class ActuatorBank:
    """把策略动作翻译成 MuJoCo 的 ``data.ctrl``。"""

    robot: RobotParams = field(default_factory=RobotParams)
    joint: JointPDController = field(default_factory=JointPDController)
    wheel: WheelEscController = field(default_factory=WheelEscController)
    balance: BalanceController = field(default_factory=BalanceController)

    # 最近一步的诊断量
    diagnostics: dict = field(default_factory=dict, init=False)
    # ★ 腿通道语义：[common, diff]（True）还是直接 4 关节（False，历史行为）
    leg_differential: bool = True

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

        关节指令 = **固定中立位** + 动作偏置（不能写成“当前位姿 + 偏置”，否则
        action=0 时位置误差恒为 0，PD 退化成纯阻尼，腿会直接软掉）。
        中立位是实测负载平衡位形（``RobotParams.stand_joint_pos``），不是 0。
        """
        # 动作 6 维：[0] common-mode 轮电流，[1] differential 轮电流，
        # [2:6] 4 个腿关节位置偏置（**共模 / 差模**基，见下）。
        #
        # ★ 腿通道语义（第二轮腿长训练引入）：
        #   a[2] = 左腿共模偏置，a[3] = 左腿差模偏置
        #   a[4] = 右腿共模偏置，a[5] = 右腿差模偏置
        #   下发：q2 = neutral2 + (common + diff)·scale
        #         q4 = neutral4 + (common − diff)·scale
        #   共模改腿的俯仰角（**不影响腿长**），差模改腿长
        #   （实测 dL/d(q2−q4) ≈ −0.1237 m/rad）。
        #   旧语义是 4 个通道直接对应 4 个关节，无法改变腿长 —— 保留
        #   `leg_differential=False` 以复现历史行为。
        action = np.asarray(action, dtype=np.float64)
        common_l, diff_l = action[2], action[3]
        common_r, diff_r = action[4], action[5]
        if not self.leg_differential:
            common_l = diff_l = action[2]
            common_r = diff_r = action[3]
        q_target = np.asarray(state["nominal_joint_pos"], dtype=np.float64).copy()
        q_target[0] += common_l * POS_ACTION_SCALE + diff_l * LEG_DIFF_ACTION_SCALE
        q_target[1] += common_l * POS_ACTION_SCALE - diff_l * LEG_DIFF_ACTION_SCALE
        q_target[2] += common_r * POS_ACTION_SCALE + diff_r * LEG_DIFF_ACTION_SCALE
        q_target[3] += common_r * POS_ACTION_SCALE - diff_r * LEG_DIFF_ACTION_SCALE
        qd_target = np.zeros(4, dtype=np.float64)
        leg_torque = self.joint.torque(q_target, qd_target, state["joint_pos"], state["joint_vel"])

        # PPO 站立 checkpoint 的动作契约是受限轮电流（common / differential
        # 两个基）。速度环 ``current()`` 保留给后续从零训练的速度目标课程；
        # 不能在旧 checkpoint 上静默切换动作语义。
        policy_current = self.decode_wheel_basis(
            action[0], action[1],
            self.wheel.params.policy_current_scale_a,
            self.wheel.params.policy_current_scale_a,
        )
        total_current = np.clip(
            policy_current,
            -self.wheel.params.current_limit_a, self.wheel.params.current_limit_a,
        )
        wheel_torque = self.wheel.torque_from_current(total_current)

        self.diagnostics = {
            "wheel_target_left": float(policy_current[0]),
            "wheel_target_right": float(policy_current[1]),
            "wheel_current_left": float(total_current[0]),
            "wheel_current_right": float(total_current[1]),
            "wheel_torque_left": float(wheel_torque[0]),
            "wheel_torque_right": float(wheel_torque[1]),
            "balance_torque": 0.0,
            "balance_current": np.zeros(2, dtype=np.float64),
            "leg_torque_abs_mean": float(np.abs(leg_torque).mean()),
        }
        return {
            "leg_ctrl": leg_torque,
            "wheel_ctrl": self.wheel.motor_command(wheel_torque),
            "leg_torque": leg_torque,
            "wheel_torque": wheel_torque,
            "wheel_current": total_current,
            "wheel_target": policy_current,
            "balance_current": np.zeros(2, dtype=np.float64),
        }

    def wheel_target_hold(self, state: dict) -> np.ndarray:
        """零指令下把轮速目标设为 0（由电调速度环主动刹车）。"""
        return np.zeros(2, dtype=np.float64)
