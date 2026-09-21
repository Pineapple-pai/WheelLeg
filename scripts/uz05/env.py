"""UZ-05 强化学习环境：观测全量常驻，奖励全量实现，能力只做开关。"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation

from .actuators import ActuatorBank, BalanceController, JointPDController, WheelEscController
from .balance import CoordinatedBalance, CoordinatedBalanceParams
from .model import TerrainSpec, UZ05Model
from .spec import (
    ACTION_DIM,
    ACTOR_OBS_BLOCKS,
    OBS_BLOCKS,
    ACTOR_OBS_DIM,
    ACTOR_FRAME_DIM,
    OBS_HISTORY,
    PRIV_OBS_DIM,
    STAND_LEVELS,
    ACTION_SLICES,
    DEFAULT_STAGE,
    DOF_ACC_WINDOW,
    EnvParams,
    OBS_DIM,
    OBS_SCALE,
    OBS_SLICES,
    STAGE_BY_NAME,
    STAGES,
    StageSpec,
    active_rewards,
    observation_noise_vector,
)

TERRAIN_SCAN_POINTS = 17
MODE_NAMES = ("normal", "airborne", "stair", "recover", "jump")
# 每个观测块的噪声半宽（对齐官方公式；由 spec 统一计算，环境里只查表）
_NOISE_VEC = observation_noise_vector(EnvParams().noise)


class UZ05Env(gym.Env):
    """两轮轮腿机器人的本体感知强化学习环境。

    观测（94 维 = actor 38 + 特权 56）在任何 capability 下都完整存在
    且实时更新；同为 94/6 接口的 checkpoint 可在不同阶段直接加载。
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        stage: str = DEFAULT_STAGE,
        params: EnvParams | None = None,
        seed: int | None = None,
        stand_level: int = 2,
        assist_scale: float = 1.0,
        init_scale: float = 1.0,
        action_filter_alpha: float = 1.0,
        pitch_angle_correction_a_per_rad: float = 5.0,
        pitch_rate_damping_a_per_rad_s: float = 2.0,
        # ★ 默认解锁腿：站立平衡必须由腿 + 轮协同完成。锁腿会让策略永远
        #   学不会用腿（旧版就是因此在 1.6 Hz 上点头）。需要复现历史行为时
        #   显式传 True。
        lock_stand_leg_actions: bool = False,
        stand_leg_action_limit: float = 1.0,
        coord_mix: float | None = None,
        coord_params: CoordinatedBalanceParams | None = None,
        # 策略残差的**安全预算**。实测（零指令、完整域随机化）协同控制器对
        # 常量残差的容忍度：|a| < 0.1 全部存活，|a| ≥ 0.2 直接触发漂移终止；
        # 对零均值高斯噪声的容忍度约 0.03（PPO 的 log_std = -3.5）。
        # 所以默认给一个很紧的预算：策略只能微调，不能推翻已标定的平衡。
        coord_residual_scale: tuple[float, float, float] = (0.05, 0.02, 0.05),
        coord_channels: str = "all",
        height_switch_steps: int = 0,
        height_switch_prob: float = 0.0,
        height_rate_limit: float | None = None,
        height_rate_gain: float | None = None,
        height_rate_damping: float | None = None,
        height_retract_rate_damping: float | None = None,
        height_low_target_brake_damping_scale: float | None = None,
        height_rate_brake_threshold: float | None = None,
        height_reference_jump_reset_m: float | None = None,
        height_brake_error_m: float | None = None,
        height_target_rate_feedforward_scale: float | None = None,
        height_hold_error_m: float | None = None,
        height_hold_rate_limit: float | None = None,
        height_hold_rate_gain: float | None = None,
        height_filter_alpha: float | None = None,
        height_retract_rate_limit: float | None = None,
        height_retract_slow_rate_limit: float | None = None,
        height_retract_slow_error_m: float | None = None,
        height_retract_rate_gain: float | None = None,
        height_retract_feedforward_scale: float | None = None,
        leg_feedforward_scale: float | None = None,
        height_target_rate_m_s: float | None = None,
        leg_residual_slew_rate: float = 0.30,
        leg_residual_brake_error_m: float | None = None,
        project_leg_length_residual: bool = False,
        leg_length_reward_weight: float | None = None,
        leg_length_progress_reward_weight: float | None = None,
        leg_length_rate_reward_weight: float | None = None,
        # ★ 压力测试用：在标准 reset 之上额外注入的初始倾角（rad）、倾角速率
        #   （rad/s）与整机前向速度冲量（m/s）。默认全 0 = 不影响正常训练/评估；
        #   调参脚本用它把初始条件推到课程分布之外（例如 ±3° 倾角 + 0.2 m/s
        #   冲量），把"能不能扛住"直接变成优化目标。
        extra_init_tilt: float = 0.0,
        extra_init_tilt_rate: float = 0.0,
        extra_init_vel: float = 0.0,
        vx_range_override: tuple[float, float] | None = None,
        zero_command_prob_override: float | None = None,
        reverse_prob_override: float | None = None,
    ):
        super().__init__()
        if stage not in STAGE_BY_NAME:
            raise ValueError(f"unknown stage {stage!r}; known: {sorted(STAGE_BY_NAME)}")
        self.stage: StageSpec = STAGE_BY_NAME[stage]
        self.params = params or EnvParams()
        self.rng = np.random.default_rng(seed)
        # 站立分级课程：收紧漂移死区/硬限，并把辅助强度退火到 0
        self.stand_level = STAND_LEVELS[0] if stand_level is None else next(
            (lv for lv in STAND_LEVELS if lv.level == int(stand_level)), STAND_LEVELS[-1]
        )
        self.params.station_deadband = self.stand_level.deadband
        self.params.station_hard_limit = self.stand_level.hard_limit
        self.params.station_hold_steps = self.stand_level.hard_hold_steps
        self.params.balance.station_deadband = self.stand_level.deadband
        self.params.balance.station_kp = self.stand_level.station_kp
        self.params.balance.assist_scale = float(np.clip(assist_scale, 0.0, 1.0))
        self.init_scale = float(np.clip(init_scale, 0.0, 1.0))
        self.extra_init_tilt = float(extra_init_tilt)
        self.extra_init_tilt_rate = float(extra_init_tilt_rate)
        self.extra_init_vel = float(extra_init_vel)
        if vx_range_override is None:
            self.vx_range_override = None
        else:
            vx_lo, vx_hi = (float(v) for v in vx_range_override)
            if vx_lo < 0.0 or vx_hi < vx_lo:
                raise ValueError(
                    "vx_range_override 必须满足 0 <= min <= max，"
                    f"得到 {vx_range_override!r}"
                )
            self.vx_range_override = (vx_lo, vx_hi)
        self.zero_command_prob_override = (
            None if zero_command_prob_override is None
            else float(np.clip(zero_command_prob_override, 0.0, 1.0))
        )
        self.reverse_prob_override = (
            None if reverse_prob_override is None
            else float(np.clip(reverse_prob_override, 0.0, 1.0))
        )
        # A small first-order filter removes the high-frequency common-wheel
        # command chatter that otherwise shows up as pitch nodding at 125 Hz.
        # Keep it configurable so rollouts can compare the raw policy (1.0)
        # with the filtered deployment path without changing the action space.
        self.action_filter_alpha = float(np.clip(action_filter_alpha, 0.0, 1.0))
        self.pitch_angle_correction_a_per_rad = float(pitch_angle_correction_a_per_rad)
        self.pitch_rate_damping_a_per_rad_s = float(pitch_rate_damping_a_per_rad_s)
        self.lock_stand_leg_actions = bool(lock_stand_leg_actions)
        self.stand_leg_action_limit = float(np.clip(stand_leg_action_limit, 0.0, 1.0))
        # ★ 腿+轮协同平衡控制器：站立时把低频姿态/位置/高度交给腿、高频 pitch
        #   修正与速度交给轮。coord_mix 是它与策略的混合系数（1 = 控制器全权）。
        self.coordinated = CoordinatedBalance(coord_params)
        self._coord_gains = self.coordinated.p
        self.coord_residual_scale = tuple(float(v) for v in coord_residual_scale)
        # 策略腿残差是实物关节目标的一部分，必须有独立的物理变化率限制。
        # 这里的单位是“最终归一化动作/s”；0.30 对应约 0.105 rad/s。
        self.leg_residual_slew_rate = max(0.0, float(leg_residual_slew_rate))
        # ``None`` preserves the old conservative behavior: once the measured
        # leg is moving toward the target, the learned residual is fully
        # gated. A positive value allows a sign-safe residual while the leg is
        # still far from the target, and gates it only inside this braking band.
        self.leg_residual_brake_error_m = (
            None if leg_residual_brake_error_m is None else
            max(0.0, float(leg_residual_brake_error_m))
        )
        # Height tracking has only one physically meaningful learned degree of
        # freedom: the left/right symmetric differential-leg mode (a3/a5).
        # Projecting onto that mode prevents PPO from bending the body with the
        # common leg channels or creating an artificial left/right mismatch.
        self.project_leg_length_residual = bool(project_leg_length_residual)
        self._leg_residual_applied = np.zeros(4, dtype=np.float64)
        self._leg_policy_height_residual = 0.0
        self.coord_mix = (self.stand_level.coord_start if coord_mix is None
                          else float(np.clip(coord_mix, 0.0, 1.0)))
        # ★ 通道级分工掩码：让"腿 + 轮协同"可以**分开训练**。
        #   coord_channels 指定哪些通道交给控制器（其余交给策略）：
        #     "all"        控制器全权（默认）
        #     "legs"       腿通道由控制器给，轮通道（共模+差模）由策略学
        #     "wheels"     轮通道由控制器给，腿通道由策略学
        #     "none"       完全靠策略
        #   这样策略能拿到**明确的回报梯度**（而不是被控制器把状态钉死），
        #   是"让 PPO 真正学会协同"的可训练分解。
        self._leg_height_range: tuple[float, float] | None = None
        # 显式腿长目标覆盖（None = 由高度命令换算）
        self.leg_length_override: float | None = None
        self.coord_channels = str(coord_channels).lower()
        if self.coord_channels not in ("all", "legs", "wheels", "none"):
            raise ValueError(f"coord_channels 必须是 all/legs/wheels/none，得到 "
                             f"{coord_channels!r}")
        self._mask_wheel = 1.0 if self.coord_channels in ("all", "wheels") else 0.0
        self._mask_leg = 1.0 if self.coord_channels in ("all", "legs") else 0.0
        if self.coord_channels == "none":
            self.coord_mix = 0.0
        self.tilt_limit = self.stand_level.tilt_limit if stage == "stand" else self.stage.tilt_limit
        # ★ 课程**不再**逐级改变动作权限。策略动作空间恒为 [-1, 1]，物理幅度由
        #   ACTION_SPEC（8 A / 0.35 rad）唯一决定。旧版把电流权限按 2/4/8 A
        #   分级缩放，导致跨级继承时策略输出被整体放大 2~4 倍而自激。
        self.action_limit = 1.0
        self.action_delta_limit = 1.0 if stage != "stand" else 1.0
        self.leg_action_limit = self.stand_level.leg_action_limit if stage == "stand" else 1.0

        # ★ 腿长变化训练：stage 指定了范围就打开腿长环，并把腿长范围换算成
        #   高度命令范围（base_z ≈ leg_length + wheel_radius）。
        if self.stage.leg_length_range is not None:
            lo, hi = self.stage.leg_length_range
            wheel_r = float(self.params.robot.wheel_radius)
            self.coordinated.enable_leg_length(
                max(lo, self.params.leg_length_target_min),
                min(hi, self.params.leg_length_target_max))
            self._leg_height_range = (lo + wheel_r, hi + wheel_r)
        if height_rate_limit is not None:
            self.coordinated.p.height_rate_limit = max(0.0, float(height_rate_limit))
        if height_rate_gain is not None:
            self.coordinated.p.height_rate_gain = max(0.0, float(height_rate_gain))
        if height_rate_damping is not None:
            self.coordinated.p.height_rate_damping = max(0.0, float(height_rate_damping))
        if height_retract_rate_damping is not None:
            self.coordinated.p.height_retract_rate_damping = max(
                0.0, float(height_retract_rate_damping)
            )
        if height_low_target_brake_damping_scale is not None:
            self.coordinated.p.height_low_target_brake_damping_scale = max(
                1.0, float(height_low_target_brake_damping_scale)
            )
        if height_rate_brake_threshold is not None:
            self.coordinated.p.height_rate_brake_threshold = max(
                0.0, float(height_rate_brake_threshold)
            )
        if height_reference_jump_reset_m is not None:
            self.coordinated.p.height_reference_jump_reset_m = max(
                0.0, float(height_reference_jump_reset_m)
            )
        if height_brake_error_m is not None:
            self.coordinated.p.height_brake_error_m = max(
                0.0, float(height_brake_error_m)
            )
        if height_target_rate_feedforward_scale is not None:
            self.coordinated.p.height_target_rate_feedforward_scale = max(
                0.0, float(height_target_rate_feedforward_scale)
            )
        if height_hold_error_m is not None:
            self.coordinated.p.height_hold_error_m = max(0.0, float(height_hold_error_m))
        if height_hold_rate_limit is not None:
            self.coordinated.p.height_hold_rate_limit = max(0.0, float(height_hold_rate_limit))
        if height_hold_rate_gain is not None:
            self.coordinated.p.height_hold_rate_gain = max(0.0, float(height_hold_rate_gain))
        if height_filter_alpha is not None:
            self.coordinated.p.height_filter_alpha = float(np.clip(
                height_filter_alpha, 0.0, 1.0
            ))
        if height_retract_rate_limit is not None:
            self.coordinated.p.height_retract_rate_limit = max(
                0.0, float(height_retract_rate_limit)
            )
        if height_retract_slow_rate_limit is not None:
            self.coordinated.p.height_retract_slow_rate_limit = max(
                0.0, float(height_retract_slow_rate_limit)
            )
        if height_retract_slow_error_m is not None:
            self.coordinated.p.height_retract_slow_error_m = max(
                0.0, float(height_retract_slow_error_m)
            )
        if height_retract_rate_gain is not None:
            self.coordinated.p.height_retract_rate_gain = max(
                0.0, float(height_retract_rate_gain)
            )
        if height_retract_feedforward_scale is not None:
            self.coordinated.p.height_retract_feedforward_scale = float(np.clip(
                height_retract_feedforward_scale, 0.0, 1.0
            ))
        if leg_feedforward_scale is not None:
            self.coordinated.p.leg_feedforward_scale = float(np.clip(
                leg_feedforward_scale, 0.0, 1.0
            ))
        self.height_target_rate_m_s = (0.0 if height_target_rate_m_s is None else
                                       max(0.0, float(height_target_rate_m_s)))
        self.height_switch_steps = max(0, int(height_switch_steps))
        self.height_switch_prob = float(np.clip(height_switch_prob, 0.0, 1.0))
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

        terrain = TerrainSpec(kind=self.stage.terrain)
        self.sim = UZ05Model(terrain)
        # 必须把 EnvParams 里的执行器参数真正接进去，否则 assist_scale /
        # 电流限幅 / 关节力矩限幅全都会静默使用 dataclass 默认值。
        self.actuators = ActuatorBank(
            robot=self.params.robot,
            joint=JointPDController(self.params.joint),
            wheel=WheelEscController(self.params.wheel),
            balance=BalanceController(self.params.balance),
        )
        # Actuator controllers hold the same parameter objects as EnvParams.
        # Therefore domain randomization must always start from immutable
        # scalar baselines. Multiplying ``self.params.joint.kp`` in-place on
        # every reset compounds random factors across episodes and eventually
        # destroys an otherwise stable plant (v32: 99% height_limit).
        self._nominal_actuator = {
            "joint_kp": float(self.params.joint.kp),
            "joint_kd": float(self.params.joint.kd),
            "wheel_torque_per_amp_joint": float(
                self.params.wheel.torque_per_amp_joint
            ),
            "wheel_speed_kp_a_per_rad_s": float(
                self.params.wheel.speed_kp_a_per_rad_s
            ),
        }

        self.action_space = spaces.Box(-1.0, 1.0, (ACTION_DIM,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)

        self.reward_flags = active_rewards(self.stage)

        # 运行时状态
        self.steps = 0
        self.previous_action = np.zeros(ACTION_DIM, dtype=np.float64)
        # Keep policy history separate from the final controller+residual
        # command.  Feeding the final command back into the policy filter or
        # action regularizers creates a recursive residual and makes PPO pay
        # for the immutable controller baseline.
        self.previous_policy_action = np.zeros(ACTION_DIM, dtype=np.float64)
        self.prev_policy_action_prev = np.zeros(ACTION_DIM, dtype=np.float64)
        self.command = np.zeros(5, dtype=np.float64)
        self.command_target = np.zeros(5, dtype=np.float64)
        self.nominal_xy = np.zeros(2, dtype=np.float64)
        self.prev_leg_lengths = self.sim.leg_lengths()
        # 腿长速度供下一控制周期的收缩制动使用。prev_leg_lengths 在
        # step() 末尾更新，因此直接在控制器入口相减会恒为 0。
        self._leg_rate_for_control = 0.0
        self.tilt_exceed_steps = 0
        self.station_exceed_steps = 0
        self.airborne_prev = 0.0
        self.airborne_steps = 0
        self.jump_phase = 0
        self.jump_timer = 0.0
        self.phase_clock = 0.0
        self.last_landing_impact = 0.0
        self._last_reward_terms: dict[str, float] = {}
        self._upright_potential = 1.0
        self._station_potential = 0.0
        self._station_max_abs = 0.0
        self._prev_body_vx = 0.0
        self._last_step_body_vx = 0.0
        self._station_window: deque = deque(maxlen=200)   # 末段稳态漂移窗口
        self._station_potential = 0.0
        self._actor_history: deque = deque(maxlen=OBS_HISTORY)   # actor 观测历史堆叠
        self._sensor_noise_state: np.ndarray | None = None
        self._nominal_model = self.sim.nominal()                  # 域随机化基准值
        self.domain: dict = {}
        self._raw_cache: dict[str, np.ndarray] = {}
        self._info: dict[str, Any] = {}
        self._height_switch_count = 0
        self._height_last_switch_step = -1
        self._height_settle_steps = -1

    # ==================================================================
    # 辅助退火
    # ==================================================================
    def set_assist_scale(self, value: float) -> None:
        """设置手写平衡外环的强度（1.0 全辅助 → 0.0 完全靠策略）。

        验收必须在 0.0 下进行：不允许靠外部控制器或姿态硬约束维持平衡。
        """
        self.params.balance.assist_scale = float(np.clip(value, 0.0, 1.0))

    def get_assist_scale(self) -> float:
        return float(self.params.balance.assist_scale)

    def set_init_scale(self, value: float) -> None:
        """Set the reset-disturbance curriculum for future episodes."""
        self.init_scale = float(np.clip(value, 0.0, 1.0))

    def set_vx_range(self, value) -> None:
        """设置未来 episode 的 |vx| 命令范围，用于渐进式平移课程。"""
        vx_lo, vx_hi = (float(v) for v in value)
        if vx_lo < 0.0 or vx_hi < vx_lo:
            raise ValueError(
                "vx_range 必须满足 0 <= min <= max，"
                f"得到 {value!r}"
            )
        self.vx_range_override = (vx_lo, vx_hi)

    def get_vx_range(self) -> tuple[float, float]:
        return tuple(self.vx_range_override or self.stage.vx_range)

    def get_init_scale(self) -> float:
        return float(self.init_scale)

    def set_coord_mix(self, value: float) -> None:
        """设置腿+轮协同平衡控制器的混合系数（1 = 控制器全权 → 0 = 完全靠策略）。"""
        self.coord_mix = float(np.clip(value, 0.0, 1.0))

    def get_coord_mix(self) -> float:
        return float(self.coord_mix)

    def set_coord_residual_scale(self, value) -> None:
        """更新策略残差权限；供 rollout 边界处的安全课程调用。"""
        values = tuple(float(v) for v in value)
        if len(values) != 3:
            raise ValueError("coord_residual_scale 必须是 (wheel, diff, leg)")
        self.coord_residual_scale = tuple(float(np.clip(v, 0.0, 1.0)) for v in values)

    def get_coord_residual_scale(self) -> tuple[float, float, float]:
        return tuple(self.coord_residual_scale)

    # ==================================================================
    # 命令
    # ==================================================================
    def _sample_command(self) -> None:
        stage = self.stage
        # ★ 腿长变化训练：高度命令在 leg_length_range 换算出的范围内采样，
        #   否则用固定的 base_height_range。
        height_range = self._leg_height_range or stage.base_height_range
        # command[3] = **离地高度目标**（官方 commands[:,2] 口径），不是腿长
        nominal_height = 0.5 * (height_range[0] + height_range[1])
        # ★ 腿长变化课程：站立阶段 zero_command_prob = 1.0，几乎所有 episode 都走
        #   下面这个“零指令”分支。腿长目标仍是独立课程变量，但 reset 时先对齐
        #   当前构型，随后由 episode 内 height_switch 采样新的区间目标。
        vary_leg = self._leg_height_range is not None
        # 动态腿长训练从 reset() 产生的**当前机构长度**开始，而不是把
        # 目标随机到区间另一端后再让执行器追赶。这样第一步没有人为的
        # 30~80 mm 初始误差，episode 内的 height_switch 才是实际训练的
        # 变高/变低过渡；网页回放也遵循同一“从当前姿态出发”的语义。
        reset_leg = None
        if vary_leg:
            reset_leg = float(np.mean(self.sim.leg_lengths()))
            reset_leg = float(np.clip(reset_leg,
                                      self.coordinated.p.leg_length_target_min,
                                      self.coordinated.p.leg_length_target_max))
        zero_command_prob = (
            stage.zero_command_prob
            if self.zero_command_prob_override is None
            else self.zero_command_prob_override
        )
        reverse_prob = (
            stage.reverse_prob
            if self.reverse_prob_override is None
            else self.reverse_prob_override
        )
        vx_range = self.vx_range_override or stage.vx_range
        if self.rng.random() < zero_command_prob:
            target = np.zeros(5, dtype=np.float64)
            target[3] = (reset_leg + float(self.params.robot.wheel_radius)
                         if reset_leg is not None else nominal_height)
        else:
            vx = float(self.rng.uniform(*vx_range))
            if reverse_prob and self.rng.random() < reverse_prob:
                vx = -vx
            vy = float(self.rng.uniform(*stage.vy_range)) if stage.vy_range[1] else 0.0
            yaw = float(self.rng.uniform(*stage.yaw_range))
            if self.rng.random() < 0.5:
                yaw = -yaw
            height = (reset_leg + float(self.params.robot.wheel_radius)
                      if reset_leg is not None
                      else float(self.rng.uniform(*height_range)))
            target = np.array([vx, vy, yaw, height, 0.0])
        jump = 1.0 if (stage.jump_prob and self.rng.random() < stage.jump_prob) else 0.0
        target[4] = jump
        self.command_target = target
        self.command[0] = 0.0  # 平滑逼近
        self.command[3] = target[3]

    def _maybe_switch_height_target(self) -> None:
        """在 episode 内切换高度目标，让腿长环学习动态跟踪。

        运行中只更新参考值，不调用 ``snap_leg_length``；reset 才允许预置，
        这样切换响应速度和过渡扰动都会真实进入训练分布。
        """
        if (self.height_switch_steps <= 0 or self.height_switch_prob <= 0.0
                or self.steps <= 0 or self.steps % self.height_switch_steps):
            return
        if self.rng.random() > self.height_switch_prob:
            return
        height_range = self._leg_height_range or self.stage.base_height_range
        height = float(self.rng.uniform(*height_range))
        self.command_target[3] = height
        self.leg_length_override = None
        self._height_switch_count += 1
        self._height_last_switch_step = int(self.steps)
        self._height_settle_steps = -1

    def _advance_command(self) -> None:
        dt = self.params.control_dt
        dvx = self.params.robot.command_accel_limit * dt
        self.command[0] = float(np.clip(self.command_target[0] - self.command[0], -dvx, dvx) + self.command[0])
        self.command[1] = self.command_target[1]
        dyaw = self.params.robot.command_yaw_accel_limit * dt
        self.command[2] = float(np.clip(self.command_target[2] - self.command[2], -dyaw, dyaw) + self.command[2])
        if self.height_target_rate_m_s > 0.0:
            dh = self.height_target_rate_m_s * dt
            self.command[3] = float(
                np.clip(self.command_target[3] - self.command[3], -dh, dh)
                + self.command[3]
            )
        else:
            self.command[3] = self.command_target[3]
        if self.coordinated.p.height_diff_limit > 0.0:
            self.coordinated.set_leg_length_ref(self.leg_length_command)
        if self.command_target[4] > 0.5 and self.jump_phase == 0:
            self.jump_phase = 1
            self.jump_timer = 0.24
        self.command[4] = self.command_target[4]

    @property
    def command_active(self) -> bool:
        return bool(abs(self.command[0]) > 0.01 or abs(self.command[2]) > 0.01)

    # ==================================================================
    # 重置
    # ==================================================================
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.sim.reset(self.params.robot.reset_height)
        stage = self.stage

        # 落地起步 or 空中起步
        airborne = stage.init_airborne_prob and self.rng.random() < stage.init_airborne_prob
        height_offset = 0.0
        if airborne:
            height_offset = float(self.rng.uniform(*stage.init_airborne_height))
        self.sim.data.qpos[2] += height_offset + float(
            self.rng.uniform(-stage.init_height_offset, stage.init_height_offset)
        )

        # 姿态
        init_scale = self.init_scale
        limit = stage.init_tilt * init_scale
        roll, pitch, yaw = self.rng.uniform(-limit, limit, 3)
        # 压力测试注入：把初始条件推到课程分布之外
        pitch += self.extra_init_tilt
        quat = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_quat()
        self.sim.data.qpos[3:7] = [quat[3], quat[0], quat[1], quat[2]]

        # 速度
        scale = stage.init_vel * init_scale
        self.sim.data.qvel[:3] = self.rng.uniform(-scale, scale, 3)
        rate = stage.init_tilt_rate * init_scale
        self.sim.data.qvel[3:6] = self.rng.uniform(-rate, rate, 3)
        if self.extra_init_tilt_rate:
            self.sim.data.qvel[4] += self.extra_init_tilt_rate
        if self.extra_init_vel:
            self.sim.data.qvel[0] += self.extra_init_vel

        # ★ 关节：**整条** qpos[7:] 用实测负载平衡位形（含被动关节）。
        #   只写 4 个主动关节、被动关节留 0 会让闭环等式约束在第一步猛拉一下，
        #   实测造成 5~10 cm 初始漂移 —— 这是"站不稳"最主要的可避免来源。
        self.sim.data.qpos[7:] = np.asarray(self.params.robot.reset_joint_pos, dtype=np.float64)
        # 主动关节叠加一点小扰动（被动关节由约束决定，不再乱写）
        self.sim.data.qpos[self.sim.hip_qpos_adr] += self.rng.uniform(
            -0.02, 0.02, 4
        ) * init_scale
        self.sim.data.qvel[self.sim.hip_dof_adr] = self.rng.uniform(
            -0.1, 0.1, 4
        ) * init_scale
        # ★ 腿长课程：在 qpos 定稿、sim.forward() 之前采样腿长目标并把差模
        #   预置到位。放这里（而不是 reset 末尾）的原因：末尾预置会让腿在
        #   已经算好运动学之后再被瞬移，第一步就是一次冲击 —— 实测某个
        #   episode 在 92 步触发 station_limit。
        # ---- 域随机化：每 episode 采样一次，整局不变 ----
        # 按站立分级收缩区间（域随机化课程：S0 25% → S1 60% → S2 100%）
        dr = self.params.domain_randomization.scaled(self.stand_level.dr_scale)
        if dr.enabled:
            self.domain = dr.sample(self.rng)
            self.sim.apply_randomization(self._nominal_model, self.domain)
            self.actuators.joint.params.kp = (
                self._nominal_actuator["joint_kp"]
                * self.domain["joint_kp_scale"][0]
            )
            self.actuators.joint.params.kd = (
                self._nominal_actuator["joint_kd"]
                * self.domain["joint_kd_scale"][0]
            )
            self.actuators.wheel.params.torque_per_amp_joint = (
                self._nominal_actuator["wheel_torque_per_amp_joint"]
                * self.domain["wheel_torque_scale"][0]
            )
            self.actuators.wheel.params.speed_kp_a_per_rad_s = (
                self._nominal_actuator["wheel_speed_kp_a_per_rad_s"]
                * self.domain["wheel_speed_gain_scale"][0]
            )
        else:
            self.domain = {name: (0.0, 0.0) for name in dr.PARAM_NAMES}
            self.sim.apply_randomization(self._nominal_model, {k: (1.0 if "scale" in k else 0.0, 0.0)
                                                              for k in dr.PARAM_NAMES})
            self.actuators.joint.params.kp = self._nominal_actuator["joint_kp"]
            self.actuators.joint.params.kd = self._nominal_actuator["joint_kd"]
            self.actuators.wheel.params.torque_per_amp_joint = (
                self._nominal_actuator["wheel_torque_per_amp_joint"]
            )
            self.actuators.wheel.params.speed_kp_a_per_rad_s = (
                self._nominal_actuator["wheel_speed_kp_a_per_rad_s"]
            )
        self.sim.forward()

        self.nominal_xy = self.sim.data.qpos[:2].copy()
        self.steps = 0
        self._height_switch_count = 0
        self._height_last_switch_step = -1
        self._height_settle_steps = -1
        self.previous_action[:] = 0.0
        self.previous_policy_action[:] = 0.0
        self.prev_policy_action_prev[:] = 0.0
        self._leg_residual_applied[:] = 0.0
        self._leg_policy_height_residual = 0.0
        self.prev_leg_lengths = self.sim.leg_lengths()
        self._leg_rate_for_control = 0.0
        # 6 个主动关节速度历史（算 dof_acc，官方特权观测项；窗口见 spec.DOF_ACC_WINDOW）
        self._dof_vel_hist = deque(maxlen=DOF_ACC_WINDOW + 1)
        self._dof_vel_hist.append(
            np.concatenate([self.sim.joint_velocities(), self.sim.wheel_velocities()])
        )
        self.prev_action_prev = np.zeros(ACTION_DIM, dtype=np.float64)   # 上上帧实际动作
        self._last_leg_torque = np.zeros(4, dtype=np.float64)            # 上一步关节力矩
        self._last_wheel_torque = np.zeros(2, dtype=np.float64)
        self.tilt_exceed_steps = 0
        self.station_exceed_steps = 0
        self.airborne_prev = 0.0
        self.airborne_steps = 0
        self.jump_phase = 0
        self.jump_timer = 0.0
        self.phase_clock = 0.0
        self.last_landing_impact = 0.0
        self._prev_body_vx = 0.0
        self._last_step_body_vx = 0.0
        self._station_max_abs = 0.0
        self._station_window: deque = deque(maxlen=200)   # 末段稳态漂移窗口
        self._actor_history: deque = deque(maxlen=OBS_HISTORY)   # actor 观测历史堆叠
        self._sensor_noise_state = None
        initial_rpy = Rotation.from_quat([
            self.sim.base_quat[1], self.sim.base_quat[2],
            self.sim.base_quat[3], self.sim.base_quat[0],
        ]).as_euler("xyz")
        initial_tilt_sq = float(initial_rpy[0] ** 2 + initial_rpy[1] ** 2)
        self._upright_potential = float(np.exp(-initial_tilt_sq / (0.25 ** 2)))
        self.actuators.reset()
        self.coordinated.reset()
        self._coord_action = np.zeros(ACTION_DIM, dtype=np.float64)
        _, _, _, airborne_now = self.sim.contact_state()
        self.airborne_prev = airborne_now
        # command 必须每 episode 重新采样（它决定腿长目标与高度命令）。
        # 顺序：coordinated.reset() 之后再 set_leg_length_ref，
        # 否则 reset 会把刚设好的参考清掉。
        self.leg_length_override = None
        self._sample_command()
        # 只有 episode 起点允许把差模预置到目标；运行中的目标变化必须
        # 经过 _step_leg_modes 的速率限制，不能每帧 snap 前馈，否则高到低
        # 切换会表现为腿长瞬间塌下、随后再由积分器慢慢拉回。
        self._sync_leg_length_command(snap=True)
        return self._obs(), {}

    # ==================================================================
    # 腿长命令（腿负责机体高度）
    # ==================================================================
    @property
    def leg_length_command(self) -> float:
        """当前 episode 的腿长目标（m）。

        `command[3]` 是**离地高度**命令（actor 可见）。腿长与高度近似满足
        `base_z ≈ leg_length + wheel_radius`（实测 0.2383 ≈ 0.1842 + 0.0541），
        所以腿长目标 = 高度命令 − 轮半径，不引入新的量纲。

        若设置了 ``leg_length_override``（见 :meth:`set_leg_length_command`），
        优先用它 —— 供网页交互/脚本直接给腿长目标，不必换算高度。
        """
        if self.leg_length_override is not None:
            return float(self.leg_length_override)
        return float(self.command[3]) - float(self.params.robot.wheel_radius)

    @property
    def leg_length_range(self) -> tuple[float, float] | None:
        """腿长可达范围（m）；未启用腿长环时为 None。"""
        if self.coordinated.p.height_diff_limit <= 0.0:
            return None
        return (float(self.coordinated.p.leg_length_target_min),
                float(self.coordinated.p.leg_length_target_max))

    def set_leg_length_command(self, value: float | None) -> float:
        """直接设置腿长目标（m），返回实际生效值。

        * ``value=None``：清除覆盖，回到"由高度命令换算"。
        * 目标会被限到已启用的腿长范围（``enable_leg_length`` 设定的
          ``leg_length_target_min/max``）。腿长环未启用时不生效。
        * 同时刷新前馈目标；下一次 ``step`` 会按速率限幅平滑过去，
          不会瞬跳（瞬跳会把机体推倒）。
        """
        if self.coordinated.p.height_diff_limit <= 0.0:
            self.leg_length_override = None
            return self.leg_length_command
        if value is None:
            self.leg_length_override = None
        else:
            lo, hi = (self.coordinated.p.leg_length_target_min,
                      self.coordinated.p.leg_length_target_max)
            self.leg_length_override = float(np.clip(value, lo, hi))
        # ★ 只在目标**真正变化**时刷新。`_sync_leg_length_command` 会重置
        #   积分器（合理：换目标就要重新收敛），但调用方（网页回放）每步都会
        #   重发同一个目标 —— 那样等于每步清零积分，腿永远追不上目标
        #   （实测卡在 0.2316 m，目标 0.300）。
        if abs(self.coordinated.leg_length_ref - self.leg_length_command) > 1e-9:
            self._sync_leg_length_command()
        return self.leg_length_command

    def _sync_leg_length_command(self, *, snap: bool = False) -> None:
        """把高度命令同步给协同控制器的腿长环。

        腿长环默认关闭（`height_diff_limit = 0`）；只有显式调用
        `enable_leg_length()` 打开后才会跟踪。目标会被限到
        `leg_length_target_min/max`（按机构实测可达范围设定）。
        ``snap=True`` 只用于 reset 初始构型；运行中保持默认值，
        让目标变化经过腿长速率限制。
        """
        if self.coordinated.p.height_diff_limit <= 0.0:
            return
        self.coordinated.set_leg_length_ref(self.leg_length_command)
        # episode 起点预置到位：此时还没有平衡状态可破坏，瞬跳是安全的。
        # 运行中的 set_leg_length_command() 不允许 snap，必须由腿长闭环
        # 按 rate_limit 平滑追踪。
        if snap:
            self.coordinated.snap_leg_length()

    # ==================================================================
    # 观测（全部常驻、始终实时）
    # ==================================================================
    def _obs(self) -> np.ndarray:
        sim = self.sim
        quat = sim.base_quat
        rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        rpy = rot.as_euler("xyz")
        gravity_body = rot.inv().apply(np.array([0.0, 0.0, -1.0]))
        lin_vel_body = sim.body_frame(sim.base_lin_vel_world)
        ang_vel_body = sim.body_frame(sim.base_ang_vel_world)
        leg_lengths = sim.leg_lengths()
        leg_rate = (leg_lengths - self.prev_leg_lengths) / self.params.control_dt
        wheel_force, wheel_hit, body_hit, airborne = sim.contact_state()
        base_x = float(sim.base_pos[0])
        mode = self._mode_onehot(airborne, base_x)

        dof_vel_now = np.concatenate([sim.joint_velocities(), sim.wheel_velocities()])
        # 官方口径 dof_acc：一个**控制周期**内的速度差（我们用 8 步窗口对齐官方
        # 的 0.0667 s，见 spec.DOF_ACC_WINDOW），否则惩罚会被放大 69 倍
        hist = self._dof_vel_hist
        if len(hist) >= 2:
            span = (len(hist) - 1) * self.params.control_dt
            dof_acc = (hist[0] - hist[-1]) / span
        else:
            dof_acc = np.zeros(6, dtype=np.float64)
        # 最近一次下发的 6 个主动关节力矩（腿 4 + 轮 2），官方特权观测项
        torques = np.concatenate([self._last_leg_torque, self._last_wheel_torque])

        blocks = {
            "gravity": gravity_body,
            "yaw": np.array([rpy[2]]),
            "base_lin_vel": lin_vel_body,
            "base_lin_vel_actor": lin_vel_body,
            "base_ang_vel": ang_vel_body,
            "base_pos_rel": sim.data.qpos[:2] - self.nominal_xy,
            "leg_joint_pos": sim.joint_positions(),
            "leg_joint_vel": sim.joint_velocities(),
            "wheel_joint_vel": sim.wheel_velocities(),
            "command": self.command,
            "station_error": np.array([sim.data.qpos[0] - self.nominal_xy[0]]),
            "previous_action": self.previous_action,
            "leg_length": np.concatenate([leg_lengths, leg_rate]),
            "wheel_contact_force": wheel_force,
            "contact_flag": np.concatenate([wheel_hit, [body_hit, airborne]]),
            "terrain_scan": sim.terrain_scan(base_x, np.zeros(TERRAIN_SCAN_POINTS)),
            "dof_acc": dof_acc,
            "torques": torques,
            # 特权：本 episode 采样到的域随机化参数（归一化，critic 专用）
            "domain_params": np.array(
                [self.domain[name][1] for name in
                 self.params.domain_randomization.PARAM_NAMES], dtype=np.float64
            ),
            "phase": np.array([np.sin(self.phase_clock), np.cos(self.phase_clock)]),
            "mode_onehot": mode,
        }
        # 原始物理量留一份：奖励与终端日志必须用真实单位，不能用归一化后的值
        self._raw_cache = blocks
        # 归一化（对齐开源 obs_scales）；标量或按分量向量
        blocks = {
            name: np.asarray(value, dtype=np.float64)
            * np.asarray(OBS_SCALE.get(name, 1.0), dtype=np.float64)
            for name, value in blocks.items()
        }
        self._obs_cache = blocks
        # 1) 单帧 actor 观测（归一化 → 裁剪 → 加噪，顺序与官方一致）
        frame = np.concatenate(
            [blocks[name] for name in _BLOCK_ORDER if name in _ACTOR_BLOCK_NAMES]
        ).astype(np.float64)
        clip = float(self.params.noise.clip_observations)
        if clip > 0:
            np.clip(frame, -clip, clip, out=frame)
        # 噪声半宽按 block 展开（官方公式见 spec.observation_noise_vector）
        noise_vec = np.concatenate([
            np.full(_BLOCK_WIDTH[name], _NOISE_VEC.get(name, 0.0))
            for name in _BLOCK_ORDER if name in _ACTOR_BLOCK_NAMES
        ])
        if np.any(noise_vec):
            white_noise = self.rng.uniform(-1.0, 1.0, frame.shape) * noise_vec
            if self._sensor_noise_state is None:
                self._sensor_noise_state = white_noise.copy()
            else:
                self._sensor_noise_state = (
                    (1.0 - float(self.params.noise.temporal_alpha))
                    * self._sensor_noise_state
                    + float(self.params.noise.temporal_alpha) * white_noise
                )
            frame += self._sensor_noise_state
        # 2) 历史堆叠（最早 → 最新）；不足时用最早一帧填充。
        #    注意：噪声在入栈之前加，因此增加历史帧时各帧噪声仍相互独立。
        if len(self._actor_history) == 0:
            for _ in range(OBS_HISTORY):
                self._actor_history.append(frame.copy())
        else:
            self._actor_history.append(frame.copy())
        history = np.concatenate(list(self._actor_history))
        # 3) 特权部分（单帧）
        privileged = np.concatenate(
            [blocks[name] for name in _BLOCK_ORDER if name not in _ACTOR_BLOCK_NAMES]
        )
        return np.concatenate([history, privileged]).astype(np.float32)

    def _mode_onehot(self, airborne: float, base_x: float) -> np.ndarray:
        onehot = np.zeros(len(MODE_NAMES), dtype=np.float64)
        if self.jump_phase:
            onehot[MODE_NAMES.index("jump")] = 1.0
        elif self.stage.name == "recovery" and abs(
            self.sim.data.qpos[2] - self.params.robot.nominal_stand_height
        ) > 0.05:
            onehot[MODE_NAMES.index("recover")] = 1.0
        elif airborne > 0.5:
            onehot[MODE_NAMES.index("airborne")] = 1.0
        elif self.sim.terrain_scan(base_x, np.zeros(1))[0] > 0.01:
            onehot[MODE_NAMES.index("stair")] = 1.0
        else:
            onehot[MODE_NAMES.index("normal")] = 1.0
        return onehot

    # ==================================================================
    # 步进
    # ==================================================================
    def step(self, action):
        sim = self.sim
        policy_action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        action = policy_action.copy()
        self._maybe_switch_height_target()
        # Update the smoothed command before selecting the action mode. A zero
        # command is a standing mode even inside low_speed/steering stages.
        self._advance_command()
        # 以“本步已经斜坡化后的目标”为基准计算进度，避免目标本身移动时产生
        # 虚假的奖励或惩罚。腿长可由实物编码器正运动学得到。
        self._leg_error_potential = abs(
            float(self.sim.leg_lengths().mean()) - self.leg_length_command
        )
        if not self.command_active:
            # Keep the fixed 6-D interface for checkpoint compatibility. In
            # standing mode the unused differential wheel basis and leg
            # residuals are locked; only the common wheel balance target stays
            # available to prevent a two-wheel robot from falling over.
            action[1] = 0.0
            if self.lock_stand_leg_actions:
                action[2:] = 0.0
            else:
                action[2:] = np.clip(
                    action[2:], -self.stand_leg_action_limit,
                    self.stand_leg_action_limit,
                )
        else:
            action[2:] = np.clip(action[2:], -self.leg_action_limit, self.leg_action_limit)
        if self.stage.name == "stand" and self.action_filter_alpha < 1.0:
            action = self.previous_policy_action + self.action_filter_alpha * (
                action - self.previous_policy_action
            )
        if self.stage.name == "stand" and (
                self.pitch_angle_correction_a_per_rad
                or self.pitch_rate_damping_a_per_rad_s):
            quat_now = self.sim.base_quat
            pitch_now = float(Rotation.from_quat([
                quat_now[1], quat_now[2], quat_now[3], quat_now[0]
            ]).as_euler("xyz")[1])
            pitch_rate_now = float(self.sim.base_ang_vel_world[1])
            current_scale = max(float(self.params.wheel.policy_current_scale_a), 1e-6)
            action[0] -= self.pitch_angle_correction_a_per_rad * pitch_now / current_scale
            action[0] -= self.pitch_rate_damping_a_per_rad_s * pitch_rate_now / current_scale
            action[0] = float(np.clip(action[0], -1.0, 1.0))
        # PPO 的 log-prob 对应它采样到的动作。站立课程在这里**不做** slew 限制，
        # 否则奖励/实际控制作用于另一个动作，梯度会系统性失配；其余能力课程
        # 保留动作变化率约束。
        if self.stage.name != "stand":
            action = self.previous_policy_action + np.clip(
                action - self.previous_policy_action,
                -self.action_delta_limit, self.action_delta_limit
            )
        # This is the action PPO can actually change.  Keep it distinct from
        # the coordinated controller's final actuator command for rewards and
        # temporal filtering.
        learned_action = action.copy()
        quat = self.sim.base_quat
        rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        rpy = rot.as_euler("xyz")
        pitch, pitch_rate = float(rpy[1]), float(self.sim.base_ang_vel_world[1])
        body_vx = float(self.sim.body_frame(self.sim.base_lin_vel_world)[0])
        station_error = float(self.sim.data.qpos[0] - self.nominal_xy[0])

        # ---- 腿 + 轮协同动态平衡：控制器基线 + 策略残差 -------------------
        # 控制器只在"无运动指令"（站定）时接管。混合系数 coord_mix 由课程
        # 从 1 退到 0，退火完成后策略必须自己实现同样的协同。
        self._coord_action = np.zeros(ACTION_DIM, dtype=np.float64)
        if not self.command_active and self.coord_mix > 0.0:
            leg_length = float(self.sim.leg_lengths().mean())
            leg_rate = self._leg_rate_for_control
            coord_action, _ = self.coordinated(
                pitch=pitch, pitch_rate=pitch_rate, body_vx=body_vx,
                station_error=station_error,
                yaw_rate=float(self.sim.body_frame(self.sim.base_ang_vel_world)[2]),
                dt=self.params.control_dt,
                leg_length=leg_length, leg_length_rate=float(leg_rate),
            )
            self._coord_action = coord_action.copy()
            r_wheel, r_diff, r_leg = self.coord_residual_scale
            if self.project_leg_length_residual:
                # Actuator semantics: a2/a4 are pitch/common mode; a3/a5 are
                # leg-length differential mode.  The policy chooses only an
                # assistance magnitude; encoder error supplies the safe sign.
                # Thus a learned bias can accelerate convergence but can never
                # drive the legs away from the requested length.  Near target,
                # or while already moving quickly toward it, the residual
                # fades out and leaves braking to the calibrated PI loop.
                raw_magnitude = max(0.0, float(0.5 * (action[3] + action[5])))
                leg_error_now = leg_length - self.coordinated.leg_length_ref
                deadband = max(0.0, float(self.coordinated.p.height_deadband_m))
                error_gate = float(np.clip(
                    (abs(leg_error_now) - deadband) / 0.02, 0.0, 1.0
                ))
                moving_toward_target = leg_error_now * leg_rate < 0.0
                if (moving_toward_target and abs(leg_rate) >
                        self.coordinated.p.height_rate_brake_threshold
                        and (self.leg_residual_brake_error_m is None
                             or abs(leg_error_now) <= self.leg_residual_brake_error_m)):
                    error_gate = 0.0
                height_residual = (-float(np.sign(leg_error_now))
                                   * raw_magnitude * error_gate)
                self._leg_policy_height_residual = height_residual
                leg_policy_residual = np.array(
                    [0.0, height_residual, 0.0, height_residual], dtype=np.float64
                )
            else:
                self._leg_policy_height_residual = float(
                    0.5 * (action[3] + action[5])
                )
                leg_policy_residual = action[2:6]
            desired_leg_residual = leg_policy_residual * r_leg
            residual_rate = self.leg_residual_slew_rate
            if (self.project_leg_length_residual
                    and abs(self._leg_policy_height_residual) < 1e-12):
                # Removing assistance is a braking action. Permit a faster but
                # still hardware-bounded withdrawal (<= 0.35 rad/s at the
                # configured joint action scale) so stale residual cannot carry
                # the mechanism through the target.
                residual_rate = max(residual_rate, 1.0)
            residual_step = residual_rate * self.params.control_dt
            self._leg_residual_applied += np.clip(
                desired_leg_residual - self._leg_residual_applied,
                -residual_step, residual_step,
            )
            # 通道级掩码：未被控制器覆盖的通道整权交给策略（残差缩放 1.0）
            wheel_scale = r_wheel if self._mask_wheel > 0.5 else 1.0
            diff_scale = r_diff if self._mask_wheel > 0.5 else 1.0
            leg_scale = r_leg if self._mask_leg > 0.5 else 1.0
            leg_residual = (self._leg_residual_applied if self._mask_leg > 0.5
                            else action[2:6] * leg_scale)
            masked_residual = np.concatenate([
                action[0:1] * wheel_scale, action[1:2] * diff_scale,
                leg_residual,
            ])
            mask = np.concatenate([
                np.full(2, self._mask_wheel), np.full(4, self._mask_leg),
            ])
            action = np.clip(self.coord_mix * mask * coord_action + masked_residual,
                             -1.0, 1.0)

        state = {
            # 腿位置 PD 的零位（= 0；负载下的实际平衡角由 PD 刚度决定，
            # 不是 PD 目标，见 spec.RobotParams.pd_neutral_joint_pos 的说明）。
            "nominal_joint_pos": np.asarray(
                self.params.robot.pd_neutral_joint_pos, dtype=np.float64
            ),
            "joint_pos": self.sim.joint_positions(),
            "joint_vel": self.sim.joint_velocities(),
            "wheel_vel": self.sim.wheel_velocities(),
            "pitch": pitch,
            "pitch_rate": pitch_rate,
            "body_vx": body_vx,
            "command_vx": float(self.command[0]),
            "command_yaw": float(self.command[2]),
            "station_error": station_error,
            "command_active": self.command_active,
        }
        control = self.actuators.compute(action, state=state, dt=self.params.control_dt)
        self.sim.data.ctrl[:4] = control["leg_ctrl"]
        self.sim.data.ctrl[4:] = control["wheel_ctrl"]
        # 记录本步下发的实际力矩（官方特权观测 torques）
        self._last_leg_torque = np.asarray(control["leg_torque"], dtype=np.float64).copy()
        self._last_wheel_torque = np.asarray(control["wheel_torque"], dtype=np.float64).copy()
        for _ in range(self.params.robot.control_substeps):
            self.sim.step()

        self.steps += 1
        self._station_max_abs = max(
            self._station_max_abs,
            abs(float(self.sim.data.qpos[0] - self.nominal_xy[0])),
        )
        self._station_window.append(
            abs(float(self.sim.data.qpos[0] - self.nominal_xy[0]))
        )
        self.phase_clock += self.params.control_dt * 2.0 * np.pi / 0.6
        self._advance_jump_phase()

        # 推进 dof_acc 的速度历史（必须在 _obs() 之前，最后一格是当前速度）
        self._dof_vel_hist.append(
            np.concatenate([self.sim.joint_velocities(), self.sim.wheel_velocities()])
        )
        obs = self._obs()
        terminated, truncated, reasons = self._termination()
        # Rewards and the next observation must describe the action that was
        # actually applied; otherwise action shaping is invisible to PPO and
        # the learned policy can recreate the same oscillation.
        reward_action = learned_action
        reward, terms = self._reward(control, state, reward_action, terminated)
        self.prev_action_prev = self.previous_action.copy()
        self.previous_action = action.copy()
        self.prev_policy_action_prev = self.previous_policy_action.copy()
        self.previous_policy_action = learned_action
        current_leg_lengths = self.sim.leg_lengths()
        self._leg_rate_for_control = float(
            (current_leg_lengths - self.prev_leg_lengths).mean()
            / max(self.params.control_dt, 1e-6)
        )
        self.prev_leg_lengths = current_leg_lengths

        self._last_reward_terms = terms
        leg_target = float(self.leg_length_command)
        leg_error = float(np.mean(self.sim.leg_lengths()) - leg_target)
        if (self._height_last_switch_step >= 0 and self._height_settle_steps < 0
                and abs(leg_error) <= 0.005):
            self._height_settle_steps = int(self.steps - self._height_last_switch_step)
        self._info = {
            "reward_terms": terms,
            "reward_unattributed": 0.0,
            "termination_reason": reasons,
            # 域随机化：原始值，供日志/复盘（critic 看到的是归一化版）
            "domain": {k: v[0] for k, v in self.domain.items()},
            "mode": MODE_NAMES[int(np.argmax(self._obs_cache["mode_onehot"]))],
            **control_diag(self.actuators),
            **state,
            # ---- 姿态 ----
            "roll": float(rpy_all[0]) if (rpy_all := Rotation.from_quat([
                sim.base_quat[1], sim.base_quat[2], sim.base_quat[3], sim.base_quat[0]
            ]).as_euler("xyz")) is not None else 0.0,
            "yaw": float(rpy_all[2]),
            "yaw_rate": float(sim.body_frame(sim.base_ang_vel_world)[2]),
            "body_vy": float(sim.body_frame(sim.base_lin_vel_world)[1]),
            # 机体前后加速度（站立"别动"的验收量；速度差分）
            # ⚠️ `**state` 里已有 spec 定义的 pitch_rate，这里必须放在它之后；
            #    且差分要用**上一步末**的速度 `_last_step_body_vx`（在 step() 末尾
            #    更新），不能用 self._prev_body_vx —— 后者在 _reward 里已被本步
            #    速度覆盖，差分恒为 0（踩过一次）。
            "body_accel": float(
                (float(sim.body_frame(sim.base_lin_vel_world)[0]) - self._last_step_body_vx)
                / max(self.params.control_dt, 1e-6)
            ),
            "base_height": float(self.sim.data.qpos[2]),
            "pitch_rate_body": float(self.sim.base_ang_vel_world[1]),
            "station_error": float(self.sim.data.qpos[0] - self.nominal_xy[0]),
            "station_error_abs": float(abs(self.sim.data.qpos[0] - self.nominal_xy[0])),
            "station_within_5cm": float(abs(self.sim.data.qpos[0] - self.nominal_xy[0]) <= 0.05),
            "station_max_abs": self._station_max_abs,
            "station_tail_abs": abs(float(self.sim.data.qpos[0] - self.nominal_xy[0])),
            # 末段 200 步的漂移均值 —— 真正的"稳态漂移"，用于验收统计
            "station_tail_mean": float(np.mean(self._station_window))
            if self._station_window else 0.0,
            "episode_steps": int(self.steps),
            "assist_scale": self.params.balance.assist_scale,
            "coord_mix": float(self.coord_mix),
            "coord_current": float(self.coordinated.last.get("current", 0.0)),
            "coord_leg_offset": float(self.coordinated.last.get("leg_offset", 0.0)),
            "coord_action_abs": float(np.abs(self._coord_action).mean()),
            "coord_residual_wheel_scale": float(self.coord_residual_scale[0]),
            "coord_residual_diff_scale": float(self.coord_residual_scale[1]),
            "coord_residual_leg_scale": float(self.coord_residual_scale[2]),
            "leg_residual_applied_abs": float(np.abs(self._leg_residual_applied).mean()),
            "policy_leg_residual_command": float(
                self._leg_policy_height_residual
            ),
            "leg_length_left": float(self._obs_cache["leg_length"][0]),
            "leg_length_right": float(self._obs_cache["leg_length"][1]),
            "leg_length_rate_mean": float(self._obs_cache["leg_length"][2:].mean()),
            "leg_length_target": leg_target,
            "leg_length_error": leg_error,
            "leg_length_error_mm": leg_error * 1000.0,
            "height_target": float(self.command[3]),
            "height_final_target": float(self.command_target[3]),
            "height_switch_count": int(self._height_switch_count),
            "height_last_switch_step": int(self._height_last_switch_step),
            "height_settle_steps": int(self._height_settle_steps),
            # ---- 关节 ----
            "hip_pos_0": float(self.sim.joint_positions()[0]),
            "hip_pos_1": float(self.sim.joint_positions()[1]),
            "hip_pos_2": float(self.sim.joint_positions()[2]),
            "hip_pos_3": float(self.sim.joint_positions()[3]),
            "hip_vel_0": float(self.sim.joint_velocities()[0]),
            "hip_vel_1": float(self.sim.joint_velocities()[1]),
            "hip_vel_2": float(self.sim.joint_velocities()[2]),
            "hip_vel_3": float(self.sim.joint_velocities()[3]),
            "wheel_vel_left": float(self.sim.wheel_velocities()[0]),
            "wheel_vel_right": float(self.sim.wheel_velocities()[1]),
            # ---- 接触 ----
            "wheel_force_left": float(self._raw_cache["wheel_contact_force"][0]),
            "wheel_force_right": float(self._raw_cache["wheel_contact_force"][1]),
            "airborne": float(self._raw_cache["contact_flag"][3]),
            "body_contact": float(self._raw_cache["contact_flag"][2]),
        }
        # 记录本步末的机体前向速度，供下一帧 info["body_accel"] 差分
        self._last_step_body_vx = float(self.sim.body_frame(self.sim.base_lin_vel_world)[0])
        return obs, float(reward), bool(terminated), bool(truncated), self._info

    def _advance_jump_phase(self) -> None:
        if self.jump_phase == 0:
            return
        self.jump_timer -= self.params.control_dt
        if self.jump_timer <= 0.0:
            self.jump_phase = 0 if self.jump_phase >= 3 else self.jump_phase + 1
            self.jump_timer = 0.24

    # ==================================================================
    # 奖励（全部实现，权重为 0 即关闭）
    # ==================================================================
    def _reward(self, control: dict, state: dict, action: np.ndarray,
                terminated: bool) -> tuple[float, dict[str, float]]:
        w = self.params.rewards
        flags = self.reward_flags
        sim = self.sim
        terms: dict[str, float] = {}

        rpy = Rotation.from_quat(
            [sim.base_quat[1], sim.base_quat[2], sim.base_quat[3], sim.base_quat[0]]
        ).as_euler("xyz")
        roll, pitch = float(rpy[0]), float(rpy[1])
        tilt_sq = roll * roll + pitch * pitch

        lin_vel_body = sim.body_frame(sim.base_lin_vel_world)
        ang_vel_body = sim.body_frame(sim.base_ang_vel_world)
        vx, vy = float(lin_vel_body[0]), float(lin_vel_body[1])
        yaw_rate = float(ang_vel_body[2])

        leg_lengths = sim.leg_lengths()
        leg_mean = float(leg_lengths.mean())
        leg_diff = float(leg_lengths[0] - leg_lengths[1])
        # command[3] 是离地高度；腿长目标由它换算（见 leg_length_command）。
        # ★ 腿长变化训练：误差指向**本 episode 的腿长命令**，而不是固定标称值，
        #   否则策略会因为"腿长等于命令值"而被罚。
        leg_error = leg_mean - self.leg_length_command
        height_error = float(sim.data.qpos[2] - self.command[3])
        station_error = float(sim.data.qpos[0] - self.nominal_xy[0])
        wheel_force, wheel_hit, body_hit, airborne = sim.contact_state()

        joint_pos = state["joint_pos"]
        joint_vel = state["joint_vel"]

        # ---------------------------------------------------------- posture
        # 姿态用有界 exp：平滑、量级可控，且远离直立时梯度自然衰减。
        # 旧版此处写死 20.0（σ≈0.22 rad），与权重 3.0 一起只贡献 ~+3，
        # 相对 -8 的位置惩罚几乎可以忽略，姿态根本没被"奖励"到。
        # 对倾角单调衰减：倾角越大奖励越低，避免低头趴伏的假站立解。
        upright = np.exp(-tilt_sq / (0.25 ** 2))
        terms["upright"] = w.upright * upright
        # Dense, action-agnostic credit assignment: reducing measured tilt
        # earns reward immediately, increasing it loses reward immediately.
        # The potential is reset from the actual randomized start pose.
        terms["upright_progress"] = w.upright_progress * (
            upright - self._upright_potential
        )
        self._upright_potential = float(upright)
        # Use physical tolerances rather than raw metres. A raw quadratic term
        # is almost zero at a 5-7 cm crouch, so PPO can fold the legs to gain
        # upright reward. The clipped normalized form makes that shortcut costly
        # while preserving a smooth gradient around the target.
        height_norm = height_error / 0.02
        leg_norm = leg_error / 0.015
        terms["height"] = -w.height * min(height_norm * height_norm, 4.0)
        terms["leg_length"] = -w.leg_length * min(leg_norm * leg_norm, 4.0)
        terms["leg_length_progress"] = w.leg_length_progress * (
            self._leg_error_potential - abs(leg_error)
        )
        current_leg_rate = float(
            (leg_lengths - self.prev_leg_lengths).mean()
            / max(self.params.control_dt, 1e-6)
        )
        near_target = float(np.exp(-0.5 * (leg_error / 0.012) ** 2))
        rate_norm = current_leg_rate / max(w.leg_length_rate_sigma_m_s, 1e-6)
        terms["leg_length_rate"] = (
            -w.leg_length_rate * min(rate_norm * rate_norm, 4.0) * near_target
        )
        terms["posture_symmetry"] = -w.leg_symmetry * leg_diff * leg_diff
        # 关节中立位是实测负载平衡位形，不是 0。
        stand_q = np.asarray(self.params.robot.stand_joint_pos, dtype=np.float64)
        terms["joint_neutral"] = -w.joint_neutral * float(
            np.square(np.asarray(joint_pos) - stand_q).mean()
        )

        # ------------------------------------------------- regularization
        terms["action_rate"] = -w.action_rate * float(
            np.square(action - self.previous_policy_action).mean()
        )
        terms["leg_action"] = -w.leg_action * float(np.square(action[2:6]).mean())
        # ★ 频段分工的**直接激励**：轮电流逐帧变化（高频颤振）与腿动作逐帧变化
        #   分别受罚。pitch 点头的物理来源就是共模轮电流 bang-bang 翻转，
        #   单步幅值不大但差分极大，这一项把"抖"和"出力"区分开。
        terms["wheel_current_rate"] = -w.wheel_current_rate * float(
            np.square(action[0] - self.previous_policy_action[0])
        )
        terms["wheel_current_jerk"] = -w.wheel_current_jerk * float(
            np.square(
                action[0] - 2.0 * self.previous_policy_action[0]
                + self.prev_policy_action_prev[0]
            )
        )
        terms["leg_action_rate"] = -w.leg_action_rate * float(
            np.square(action[2:6] - self.previous_policy_action[2:6]).mean()
        )
        terms["leg_velocity_action"] = 0.0   # 动作通道已移除
        terms["joint_velocity"] = -w.joint_velocity * min(float(np.square(joint_vel).mean()), 400.0)
        terms["joint_torque"] = -w.joint_torque * min(float(np.square(control["leg_torque"]).mean()), 1600.0)
        terms["wheel_power"] = -w.wheel_power * min(float(np.square(control["wheel_current"]).mean()), 400.0)
        # 站立（零指令）时轮子不该持续转动：这是"原地站住"的直接刻画。
        terms["wheel_differential"] = 0.0
        terms["wheel_speed"] = (
            0.0 if self.command_active
            else -w.wheel_speed * min(float(np.square(sim.wheel_velocities()).mean()), 25.0)
        )

        # ------------------------------------------------------------- limits
        leg_min, leg_max = self.params.robot.leg_length_min, self.params.robot.leg_length_max
        over = max(0.0, leg_mean - leg_max) + max(0.0, leg_min - leg_mean)
        terms["leg_length_limit"] = -w.leg_length_limit * over * over
        joint_over = np.clip(np.abs(joint_pos) - 1.2, 0.0, None)
        terms["joint_limit"] = -w.joint_limit * float(np.square(joint_over).mean())

        # ---------------------------------------------------------- tracking
        terms["track_vx"] = 0.0
        terms["wrong_direction"] = 0.0
        terms["track_vy"] = -w.track_vy * abs(vy)
        terms["yaw_lock"] = 0.0
        if flags["track_vx"] and abs(self.command[0]) > 0.01:
            cmd = float(self.command[0])
            error_norm = (vx - cmd) / max(abs(cmd), 0.06)
            terms["track_vx"] = (
                w.track_vx * float(np.exp(-((error_norm / 0.5) ** 2)))
                + w.track_vx_tight * float(np.exp(-((error_norm / 0.1) ** 2)))
                - w.track_vx_square * float((0.25 * error_norm) ** 2)
                - (w.track_vx_gap if abs(error_norm) > 1.0 else 0.0)
            )
            terms["wrong_direction"] = -w.wrong_direction * max(0.0, -vx * cmd)

        # Translation stages do not command yaw. Penalize yaw *rate* only
        # while a translational command is active; steering/rotation stages
        # omit this reward group and keep the differential wheel channel free.
        if flags["yaw_lock"] and self.command_active and abs(self.command[2]) <= 0.01:
            yaw_rate_norm = yaw_rate / max(w.yaw_lock_sigma, 1e-6)
            terms["yaw_lock"] = -w.yaw_lock * min(yaw_rate_norm * yaw_rate_norm, 4.0)

        terms["track_yaw"] = 0.0
        if flags["track_yaw"] and abs(self.command[2]) > 0.01:
            cmd_yaw = float(self.command[2])
            yaw_error = (yaw_rate - cmd_yaw) / max(abs(cmd_yaw), 0.15)
            terms["track_yaw"] = (
                w.track_yaw * float(np.exp(-((yaw_error / 0.5) ** 2)))
                - w.track_yaw_square * float((0.25 * yaw_error) ** 2)
            )

        # ------------------------------------------------------------ station
        # 位置误差是 actor 的里程计输入，因此位置和速度惩罚可形成闭环。
        # 用**有界**二次项（饱和到 1.0）：旧版饱和在 4.0，权重 2.0 ⇒ 单步最多 -8，
        # 远大于姿态项，策略学到的是"用任何手段减小 x"而不是"站住"。
        if not self.command_active:
            position_error = station_error / w.station_sigma_m
            velocity_error = vx / w.station_vel_sigma_m_s
            terms["station"] = -w.station * min(position_error * position_error, 1.0)
            terms["station_vel"] = -w.station_vel * min(velocity_error * velocity_error, 1.0)
            station_abs = abs(station_error)
            terms["station_progress"] = w.station_progress * (
                self._station_potential - station_abs
            )
            self._station_potential = station_abs
        else:
            terms["station"] = terms["station_vel"] = terms["station_progress"] = 0.0

        # --------------------------------------------------- 站立：PPO 直接塑形（无教师/学生目标）
        if self.stage.name == "stand" and not self.command_active:
            terms["teacher_track"] = 0.0
            # 平衡主要由 common-mode 轮电流学习；差模和腿残差持续正则化。
            terms["stand_action"] = -w.stand_action * (
                float(np.square(action[1])) + float(np.square(action[2:]).mean())
            )
        else:
            terms["teacher_track"] = 0.0
            terms["stand_action"] = 0.0

        # -------------------------------------------------------- stand still
        if flags["stand_still"] and not self.command_active:
            terms["stand_vx"] = -w.stand_vx * abs(vx)
            terms["stand_yaw"] = -w.stand_yaw * abs(yaw_rate)
            terms["stand_wheel"] = -w.stand_wheel_speed * min(
                float(np.square(sim.wheel_velocities()).mean()), 25.0
            )
            terms["stand_common_action"] = -w.stand_common_action * float(
                np.square(action[0])
            )
            posture_gate = float(np.exp(
                -0.5 * (height_error / 0.02) ** 2
                -0.5 * (leg_error / 0.015) ** 2
            ))
            pitch_rate_norm = float(ang_vel_body[1]) / max(
                w.stand_pitch_rate_sigma, 1e-6
            )
            terms["stand_pitch_rate"] = -w.stand_pitch_rate * min(
                pitch_rate_norm * pitch_rate_norm, 4.0
            ) * posture_gate
            # ★ "机身保持不动"的直接刻画：机体前后加速度。位置误差可以靠慢慢
            #   挪回来补偿，加速度不能——它对高频敏感，正好压住来回摆动。
            accel = (vx - self._prev_body_vx) / max(self.params.control_dt, 1e-6)
            accel_norm = accel / max(w.stand_accel_sigma, 1e-6)
            terms["stand_accel"] = -w.stand_accel * min(accel_norm * accel_norm, 4.0)
        else:
            terms["stand_vx"] = terms["stand_yaw"] = terms["stand_wheel"] = 0.0
            terms["stand_common_action"] = 0.0
            terms["stand_pitch_rate"] = 0.0
            terms["stand_accel"] = 0.0
        self._prev_body_vx = vx

        # ----------------------------------------------------------- airborne
        if flags["airborne"]:
            if airborne > 0.5:
                terms["airborne_upright"] = w.airborne_upright * upright
                terms["airborne_leg_retract"] = -w.airborne_leg_retract * max(
                    0.0, leg_mean - self.params.robot.nominal_leg_length
                )
            else:
                terms["airborne_upright"] = 0.0
                terms["airborne_leg_retract"] = 0.0
            terms["undesired_contact"] = -w.undesired_contact * body_hit
            terms["landing_impact"] = -w.landing_impact * self.last_landing_impact
        else:
            terms["airborne_upright"] = terms["airborne_leg_retract"] = 0.0
            terms["undesired_contact"] = terms["landing_impact"] = 0.0

        # ------------------------------------------------------------ terrain
        if flags["terrain"]:
            scan = self._raw_cache["terrain_scan"]
            terms["terrain_progress"] = w.terrain_progress * float((scan > 0.01).mean())
            terms["front_wheel_height"] = -w.front_wheel_height * max(
                0.0, float(scan[0]) - 0.05
            )
        else:
            terms["terrain_progress"] = terms["front_wheel_height"] = 0.0

        # --------------------------------------------------------------- jump
        if flags["jump"] and self.jump_phase:
            terms["jump_height"] = w.jump_height * max(0.0, float(sim.data.qpos[2]) - self.params.robot.reset_height)
            terms["jump_phase_time"] = -w.jump_phase_time * abs(self.jump_timer)
        else:
            terms["jump_height"] = terms["jump_phase_time"] = 0.0

        # ----------------------------------------------------------- recovery
        if flags["recovery"]:
            terms["recovery_upright"] = w.recovery_upright * upright
            terms["recovery_progress"] = w.recovery_progress * max(
                0.0, 1.0 - abs(float(sim.data.qpos[2]) - self.params.robot.nominal_stand_height) / 0.15
            )
        else:
            terms["recovery_upright"] = terms["recovery_progress"] = 0.0

        # ---------------------------------------------------------- alive/term
        # 兜底：连续惩罚项总和不得低于 -10/步。
        # 这一步是"防 hack 保险丝"——万一将来加项忘了收紧 clip，
        # 也不会出现"摔倒比站着更划算"的激励反转。
        positive_terms = ("upright", "track_vx", "track_yaw", "airborne_upright",
                          "recovery_upright", "recovery_progress", "jump_height",
                          "alive")
        continuous = sum(v for k, v in terms.items() if k not in positive_terms)
        if continuous < -10.0:
            scale = 10.0 / abs(continuous)
            for k in list(terms):
                if k not in positive_terms:
                    terms[k] *= scale
        terms["alive"] = w.alive
        terms["termination"] = -w.termination if terminated else 0.0

        return float(sum(terms.values())), terms

    # ==================================================================
    # 终止
    # ==================================================================
    def _termination(self) -> tuple[bool, bool, str]:
        sim = self.sim
        rpy = Rotation.from_quat(
            [sim.base_quat[1], sim.base_quat[2], sim.base_quat[3], sim.base_quat[0]]
        ).as_euler("xyz")
        tilt = max(abs(float(rpy[0])), abs(float(rpy[1])))
        # 低趴能暂时避开倾倒，却不是可接受的动态站立构型。
        reachable_height = self.params.robot.nominal_stand_height * self.params.height_fail_ratio
        reasons: list[str] = []

        if tilt > self.tilt_limit:
            self.tilt_exceed_steps += 1
            if self.tilt_exceed_steps > self.stage.tilt_limit_hold_steps:
                reasons.append("tilt_limit")
        else:
            self.tilt_exceed_steps = 0

        if sim.data.qpos[2] < reachable_height:
            reasons.append("height_limit")

        # A low folded configuration can keep the base above the height line
        # while no longer being a usable standing posture. Treat it as a
        # physical failure instead of allowing PPO to collect alive reward.
        leg_mean = float(sim.leg_lengths().mean())
        leg_floor = self.params.robot.nominal_leg_length * self.params.leg_length_fail_ratio
        if leg_mean < leg_floor:
            reasons.append("leg_length_limit")

        # 漂移终止：**连续超限若干步**才判死。单帧尖峰（落地冲击、约束恢复）
        # 不算失败。注意它只是"跑飞/摔倒"的安全判据，不是 ±5 cm 验收判据 ——
        # 验收看的是末段 200 步平均漂移 ``station_tail_mean``。
        station_abs = float(abs(sim.data.qpos[0] - self.nominal_xy[0]))
        if not self.command_active and station_abs > self.params.station_hard_limit:
            self.station_exceed_steps += 1
            if self.station_exceed_steps > self.params.station_hold_steps:
                reasons.append("station_limit")
        else:
            self.station_exceed_steps = 0
        drift = float(np.linalg.norm(sim.data.qpos[:2] - self.nominal_xy))
        if drift > self.params.terminate_on_drift:
            reasons.append("drift_limit")

        lin_vel_body = sim.body_frame(sim.base_lin_vel_world)
        if abs(float(lin_vel_body[1])) > self.params.terminate_lateral_vel:
            reasons.append("lateral_velocity_limit")

        terminated = len(reasons) > 0
        truncated = self.steps >= self.stage.episode_steps
        return terminated, truncated, ", ".join(reasons) if reasons else (
            "time_limit" if truncated else "running"
        )


def control_diag(bank: ActuatorBank) -> dict[str, float]:
    return dict(bank.diagnostics)

_BLOCK_ORDER = tuple(name for name, _ in OBS_BLOCKS)
_BLOCK_WIDTH = {name: width for name, width in OBS_BLOCKS}
_ACTOR_BLOCK_NAMES = frozenset(name for name, _ in ACTOR_OBS_BLOCKS)
