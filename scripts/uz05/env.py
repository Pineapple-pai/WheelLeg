"""UZ-05 强化学习环境：观测全量常驻，奖励全量实现，能力只做开关。"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation

from .actuators import ActuatorBank, JointPDController, WheelEscController
from .model import TerrainSpec, UZ05Model
from .observations import ObservationEncoder
from .spec import (
    ACTION_DIM,
    STAND_LEVELS,
    DEFAULT_STAGE,
    DOF_ACC_WINDOW,
    EnvParams,
    OBS_DIM,
    STAGE_BY_NAME,
    StageSpec,
    active_rewards,
    precision_rewards_for_stage,
    env_params_for_stage,
)

TERRAIN_SCAN_POINTS = 17


class UZ05Env(gym.Env):
    """两轮轮腿机器人的本体感知强化学习环境。

    观测总计 94 维：当前 actor 观测 38 维，critic 特权观测 56 维；不堆叠历史帧，动作固定为 6 维。
    观测收集与编码由独立 ObservationEncoder 负责，actor 不接收仿真阶段标签。
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        stage: str = DEFAULT_STAGE,
        params: EnvParams | None = None,
        seed: int | None = None,
        stand_level: int = 2,
    ):
        super().__init__()
        if stage not in STAGE_BY_NAME:
            raise ValueError(f"unknown stage {stage!r}; known: {sorted(STAGE_BY_NAME)}")
        self.stage: StageSpec = STAGE_BY_NAME[stage]
        self.params = params or env_params_for_stage(self.stage)
        self.rng = np.random.default_rng(seed)
        self._observation_encoder = ObservationEncoder(
            self.params.noise, self.params.noise.clip_observations
        )
        # 站立分级课程只收紧 episode 终止边界；所有动作均来自 PPO。
        self.stand_level = STAND_LEVELS[0] if stand_level is None else next(
            (lv for lv in STAND_LEVELS if lv.level == int(stand_level)), STAND_LEVELS[-1]
        )
        self.params.station_deadband = self.stand_level.deadband
        self.params.station_hard_limit = self.stand_level.hard_limit
        self.tilt_limit = self.stand_level.tilt_limit if stage == "stand" else self.stage.tilt_limit
        self.yaw_limit = self.stand_level.yaw_hard_limit

        terrain = TerrainSpec(kind=self.stage.terrain)
        self.sim = UZ05Model(terrain)
        if self.stage.motor_substep_control and not np.isclose(
            self.params.control_dt, self.sim.model.opt.timestep * self.params.robot.control_substeps
        ):
            raise ValueError("control_dt must equal timestep * control_substeps")
        # 把环境中的真实执行器参数接入 wheel speed / joint position 驱动模型。
        self.actuators = ActuatorBank(
            continuous_position=self.stage.continuous_position,
            wheel_velocity_scale_rad_s=self.stage.wheel_velocity_scale_rad_s,
            robot=self.params.robot,
            joint=JointPDController(deepcopy(self.params.joint)),
            wheel=WheelEscController(deepcopy(self.params.wheel)),
        )

        self.action_space = spaces.Box(-1.0, 1.0, (ACTION_DIM,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)

        self.reward_flags = active_rewards(self.stage)

        # 运行时状态
        self.steps = 0
        self.previous_action = np.zeros(ACTION_DIM, dtype=np.float64)
        self.command = np.zeros(5, dtype=np.float64)
        self.command_target = np.zeros(5, dtype=np.float64)
        self.nominal_xy = np.zeros(2, dtype=np.float64)
        self.hold_xy = np.zeros(2, dtype=np.float64)
        self.hold_yaw = 0.0
        self._hold_zero_active = False
        self._initial_rolling_vx = 0.0
        self._straight_active = False
        self.straight_yaw = 0.0
        self._previous_control = None
        self._previous_leg_drop = np.zeros(2)
        self._support_diag = {}
        self._push_force = np.zeros(3)
        self._push_remaining = 0
        self._next_push_step = 0
        self.prev_leg_lengths = self.sim.leg_lengths()
        self.tilt_exceed_steps = 0
        self.stance_invalid_steps = 0
        self.body_contact_exceed_steps = 0
        self.airborne_prev = 0.0
        self.airborne_steps = 0
        self.jump_phase = 0
        self.jump_timer = 0.0
        self.last_landing_impact = 0.0
        self._last_reward_terms: dict[str, float] = {}
        self._tracking_quality = 1.0
        self._station_max_abs = 0.0
        self._yaw_max_abs = 0.0
        self._roll_max_abs = 0.0
        self._pitch_max_abs = 0.0
        self._ang_vel_sq_sum = 0.0
        self._ang_vel_samples = 0
        self._station_window: deque = deque(maxlen=200)   # 末段稳态漂移窗口
        self._nominal_model = self.sim.nominal()                  # 域随机化基准值
        self.domain: dict = {}
        self._raw_cache: dict[str, np.ndarray] = {}
        self._info: dict[str, Any] = {}

    # ==================================================================
    # 命令
    # ==================================================================
    def _sample_command(self) -> None:
        stage = self.stage
        if stage.continuous_position:
            # Fixed mixture from the first reset, independent of training progress.
            # v8: equal modes; v9: fixed 20/45/15/20 stop/straight/turn/mixed.
            mode = (int(self.rng.choice(4, p=(.20, .45, .15, .20)))
                    if stage.straight_tracking else int(self.rng.integers(4)))
            target = np.zeros(5, dtype=np.float64)
            target[3] = self.rng.uniform(*stage.base_height_range)
            if stage.recovery_sampling and self.rng.random() < .40:
                target[3] = self.rng.uniform(.30, stage.base_height_range[1])
            if mode in (1, 3):
                if stage.transition_tracking:
                    # All speed bands are present from the first reset, no curriculum.
                    sign = (int(self.rng.choice([-1,1],p=(stage.translation_reverse_probability,
                                                         1-stage.translation_reverse_probability)))
                            if stage.responsive_tracking else int(self.rng.choice([-1,1])))
                    limit = -stage.vx_range[0] if sign < 0 else stage.vx_range[1]
                    if stage.straight_tracking:
                        edges=stage.velocity_band_fractions
                        bands=tuple((a*limit,b*limit) for a,b in zip(edges[:-1],edges[1:]))
                        band=int(self.rng.choice(len(bands),p=stage.velocity_band_probabilities))
                    else:
                        bands = ((0.0, .25 * limit), (.25 * limit, .625 * limit), (.625 * limit, limit))
                        band = int(self.rng.integers(3))
                    target[0] = self.rng.uniform(*bands[band]) * sign
                else:
                    target[0] = self.rng.uniform(*stage.vx_range)
            if mode in (2, 3):
                target[2] = self.rng.uniform(*stage.yaw_range)
            if stage.transition_tracking and self.steps > 0:
                change = int(self.rng.integers(4))
                if change < 2:  # 50% motion-only: expose sag without changing height target.
                    target[3] = self.command_target[3]
                elif change == 2:  # 25% height-only, including while translating/turning.
                    target[:3] = self.command_target[:3]
            reversal = False
            if stage.coordinated_tracking and self.steps > 0:
                # All simple-reward arms keep exactly the v15 command distribution.
                p = (self.params.precision_centered if stage.simple_reward else
                     precision_rewards_for_stage(stage, self.params))
                eligible = abs(self.command_target[0]) > .1 and abs(self.command_target[2]) <= .01
                if stage.braking_tracking and eligible:
                    stop_probability = p.stop_command_probability
                    reversal_probability = p.reversal_command_probability
                    height_probability = getattr(p,'height_command_probability',0.)
                    turn_probability = getattr(p,'turn_command_probability',0.)
                    high_speed = stage.recovery_sampling and abs(self.command_target[0]) >= 1.3
                    if high_speed:
                        # Fixed distribution; no training-progress curriculum or action helper.
                        stop_probability, reversal_probability = ((.40,.35)
                            if self.command_target[3] >= .30 else (.20,.50))
                        height_probability, turn_probability = ((.10,.05)
                            if self.command_target[3] >= .30 else (.15,.05))
                    event = self.rng.random()
                    if event < stop_probability:
                        target[:] = self.command_target
                        target[:3] = 0.0
                    elif event < stop_probability + reversal_probability:
                        target[:] = self.command_target
                        target[0] *= -1
                        if (stage.switching_tracking
                                and self.rng.random() < (0. if high_speed else p.reversal_mid_speed_probability)):
                            target[0] = np.sign(target[0]) * self.rng.uniform(*p.reversal_mid_speed_range_m_s)
                        reversal = True
                    elif (stage.switching_tracking and event < stop_probability
                          + reversal_probability + height_probability):
                        target[:] = self.command_target
                        # Exercise the full height step at the held translation speed.
                        midpoint = sum(stage.base_height_range) / 2
                        target[3] = stage.base_height_range[int(target[3] <= midpoint)]
                    elif (stage.switching_tracking and event < stop_probability
                          + reversal_probability + height_probability + turn_probability):
                        target[:] = self.command_target
                        target[0] = 0.0
                        target[2] = self.rng.uniform(*stage.yaw_range)
                elif (eligible
                        and self.rng.random() < p.reversal_command_probability):
                    # A sampled user command, never a prescribed posture/action.
                    target[:] = self.command_target
                    target[0] *= -1
                    reversal = True
            self.command_target = target
            self.command[:] = target
            zero_like = bool(np.max(np.abs(target[[0, 2]])) <= 0.01)
            hold_steps = (stage.zero_command_hold_steps if stage.precision_control
                          and (mode == 0 or (stage.robust_support and zero_like)) else stage.command_hold_steps)
            if stage.transition_tracking:
                hold_steps = stage.zero_command_hold_steps if zero_like else stage.command_hold_steps
            if reversal:
                hold_steps = stage.command_hold_steps
            elif stage.transition_tracking and self.rng.random() < stage.long_command_prob:
                hold_steps = stage.long_command_hold_steps
            elif not stage.transition_tracking and stage.robust_support and not zero_like and self.rng.random() < stage.long_command_prob:
                hold_steps = stage.long_command_hold_steps
            self.command_steps_left = int(self.rng.integers(*hold_steps))
            return
        # command[3] = **离地高度目标**（官方 commands[:,2] 口径），不是腿长
        nominal_height = 0.5 * (stage.base_height_range[0] + stage.base_height_range[1])
        if self.rng.random() < stage.zero_command_prob:
            target = np.zeros(5, dtype=np.float64)
            target[3] = nominal_height          # 站立时高度命令 = 额定离地高度
        else:
            vx = float(self.rng.uniform(*stage.vx_range))
            if stage.reverse_prob and self.rng.random() < stage.reverse_prob:
                vx = -vx
            vy = float(self.rng.uniform(*stage.vy_range)) if stage.vy_range[1] else 0.0
            yaw = float(self.rng.uniform(*stage.yaw_range))
            if self.rng.random() < 0.5:
                yaw = -yaw
            height = float(self.rng.uniform(*stage.base_height_range))
            target = np.array([vx, vy, yaw, height, 0.0])
        jump = 1.0 if (stage.jump_prob and self.rng.random() < stage.jump_prob) else 0.0
        target[4] = jump
        self.command_target = target
        self.command[0] = 0.0  # 平滑逼近
        self.command[3] = target[3]

    def _advance_command(self) -> None:
        if self.stage.continuous_position:
            if self.command_steps_left <= 0:
                self._sample_command()
            self.command_steps_left -= 1
            return
        dt = self.params.control_dt
        dvx = self.params.robot.command_accel_limit * dt
        self.command[0] = float(np.clip(self.command_target[0] - self.command[0], -dvx, dvx) + self.command[0])
        self.command[1] = self.command_target[1]
        dyaw = self.params.robot.command_yaw_accel_limit * dt
        self.command[2] = float(np.clip(self.command_target[2] - self.command[2], -dyaw, dyaw) + self.command[2])
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
        limit = self.stand_level.init_tilt if stage.name == "stand" else stage.init_tilt
        roll = float(self.rng.uniform(-limit, limit))
        if stage.name == "stand" and self.rng.random() < 0.67:
            # The paired replay exposed a repeatable backward-tilt failure;
            # oversample that side while preserving positive-tilt coverage.
            pitch = -float(self.rng.uniform(0.0, limit))
        else:
            pitch = float(self.rng.uniform(-limit, limit))
        # The S2 yaw hard limit is only 5 degrees; sampling yaw from the same
        # 0.08 rad tilt range starts episodes almost at termination. Keep yaw
        # perturbations within one quarter of that boundary during stand.
        yaw_init_limit = (
            min(limit, 0.25 * self.yaw_limit) if stage.name == "stand" else limit
        )
        yaw = float(self.rng.uniform(-yaw_init_limit, yaw_init_limit))
        quat = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_quat()
        self.sim.data.qpos[3:7] = [quat[3], quat[0], quat[1], quat[2]]

        # 速度
        scale = stage.init_vel
        self.sim.data.qvel[:3] = self.rng.uniform(-scale, scale, 3)
        rate = stage.init_tilt_rate
        self.sim.data.qvel[3:6] = self.rng.uniform(-rate, rate, 3)

        # 关节
        self.sim.data.qpos[self.sim.hip_qpos_adr] = self.rng.uniform(-0.05, 0.05, 4)
        self.sim.data.qvel[self.sim.hip_dof_adr] = self.rng.uniform(-0.2, 0.2, 4)
        self._initial_rolling_vx = 0.0
        if stage.moving_start_probability > 0 and self.rng.random() < stage.moving_start_probability:
            sign=int(self.rng.choice([-1,1],p=(stage.translation_reverse_probability,
                                             1-stage.translation_reverse_probability)))
            self._initial_rolling_vx=float(self.rng.uniform(*stage.moving_start_speed_range))*sign
            # Initial state only: a freely moving base and consistent wheel spin.
            # No reference action, force, or pose is supplied after reset.
            self.sim.data.qvel[:3]=Rotation.from_quat(quat).apply([self._initial_rolling_vx,0.,0.])
            self.sim.data.qvel[self.sim.wheel_dof_adr]=-self._initial_rolling_vx/self.sim.wheel_radius
        # ---- 域随机化：每 episode 采样一次，整局不变 ----
        dr = self.params.domain_randomization
        if dr.enabled:
            self.domain = dr.sample(self.rng)
            self.sim.apply_randomization(self._nominal_model, self.domain)
            self.actuators.joint.params.kp = self.params.joint.kp * self.domain["joint_kp_scale"][0]
            self.actuators.joint.params.kd = self.params.joint.kd * self.domain["joint_kd_scale"][0]
            self.actuators.wheel.params.torque_per_amp_joint = (
                self.params.wheel.torque_per_amp_joint * self.domain["wheel_torque_scale"][0]
            )
            self.actuators.wheel.params.speed_kp_a_per_rad_s = (
                self.params.wheel.speed_kp_a_per_rad_s * self.domain["wheel_speed_gain_scale"][0]
            )
        else:
            self.actuators.joint.params.kp = self.params.joint.kp
            self.actuators.joint.params.kd = self.params.joint.kd
            self.actuators.wheel.params.torque_per_amp_joint = self.params.wheel.torque_per_amp_joint
            self.actuators.wheel.params.speed_kp_a_per_rad_s = self.params.wheel.speed_kp_a_per_rad_s
            self.domain = {name: (0.0, 0.0) for name in dr.PARAM_NAMES}
            self.sim.apply_randomization(self._nominal_model, {k: (1.0 if "scale" in k else 0.0, 0.0)
                                                              for k in dr.PARAM_NAMES})
        self.sim.forward()

        self.reset_projection_info = {'enabled': self.params.robot.closed_chain_reset,
                                      'resamples': 0, 'iterations': 0, 'rejections': []}
        if self.params.robot.closed_chain_reset:
            from .reset_projection import ProjectionError
            attempts = self.params.robot.reset_projection_attempts
            if attempts < 1:
                raise ValueError('reset_projection_attempts must be positive')
            for attempt in range(attempts):
                try:
                    result = self.sim.reset_projector.project(self.sim.data)
                    geometry = self.sim.wheel_stance()
                    if any(not .15 < geometry[f'leg_drop_{side}_m'] < .34
                           or abs(geometry[f'wheel_fore_aft_{side}_m']) > .10
                           for side in ('left', 'right')):
                        raise ProjectionError('invalid local wheel-leg assembly')
                    # Robust stages deliberately start clear of terrain. Legacy
                    # low-height stages retain their original ground placement.
                    if self.stage.robust_support:
                        for c in self.sim.data.contact:
                            if c.dist < -1e-5:
                                raise ProjectionError(f'initial penetration {c.dist:g} m')
                    self.reset_projection_info.update(result, resamples=attempt)
                    break
                except ProjectionError as exc:
                    self.reset_projection_info['rejections'].append(str(exc))
                    if attempt + 1 == attempts:
                        raise RuntimeError(f'closed-chain reset failed after {attempts} attempts: {exc}') from exc
                    self.sim.data.qpos[self.sim.hip_qpos_adr] = self.rng.uniform(-.05, .05, 4)
                    self.sim.data.qvel[self.sim.hip_dof_adr] = self.rng.uniform(-.2, .2, 4)
                    self.sim.data.qvel[self.sim.reset_projector.dofs] = 0.

        self.nominal_xy = self.sim.data.qpos[:2].copy()
        reset_quat = self.sim.base_quat
        reset_rot = Rotation.from_quat([
            reset_quat[1], reset_quat[2], reset_quat[3], reset_quat[0]
        ])
        self.nominal_yaw = float(reset_rot.as_euler("xyz")[2])
        self.hold_xy = self.nominal_xy.copy()
        self.hold_yaw = self.nominal_yaw
        self._hold_zero_active = False
        self._straight_active = False
        self.straight_yaw = self.nominal_yaw
        self._previous_control = None
        self.steps = 0
        self.previous_action[:] = 0.0
        self.prev_leg_lengths = self.sim.leg_lengths()
        # 6 个主动关节速度历史（算 dof_acc，官方特权观测项；窗口见 spec.DOF_ACC_WINDOW）
        self._dof_vel_hist = deque(maxlen=DOF_ACC_WINDOW + 1)
        self._dof_vel_hist.append(
            np.concatenate([self.sim.joint_velocities(), self.sim.wheel_velocities()])
        )
        self.prev_action_prev = np.zeros(ACTION_DIM, dtype=np.float64)   # 上上帧动作
        self._last_leg_torque = np.zeros(4, dtype=np.float64)            # 上一步关节力矩
        self._last_wheel_torque = np.zeros(2, dtype=np.float64)
        self.tilt_exceed_steps = 0
        self.stance_invalid_steps = 0
        self.body_contact_exceed_steps = 0
        self.airborne_prev = 0.0
        self.airborne_steps = 0
        self.jump_phase = 0
        self.jump_timer = 0.0
        self.last_landing_impact = 0.0
        self._station_max_abs = 0.0
        self._yaw_max_abs = 0.0
        self._roll_max_abs = 0.0
        self._pitch_max_abs = 0.0
        self._ang_vel_sq_sum = 0.0
        self._ang_vel_samples = 0
        self._station_window: deque = deque(maxlen=200)   # 末段稳态漂移窗口
        self.actuators.reset()
        geometry = self.sim.wheel_stance()
        self._previous_leg_drop = np.array([geometry["leg_drop_left_m"], geometry["leg_drop_right_m"]])
        self._push_force[:] = 0.0
        self._push_remaining = 0
        self._next_push_step = (int(self.rng.integers(*self.params.robust.push_interval_steps))
                                if self.params.robust.pushes_enabled else 0)
        _, _, _, airborne_now = self.sim.contact_state(terrain_only=self.stage.robust_support)
        self.airborne_prev = airborne_now
        self._sample_command()
        return self._obs(), {}

    # ==================================================================
    # 观测状态收集；打包、缩放和加噪由 observations.py 负责
    # ==================================================================
    def _obs(self, current_action: np.ndarray | None = None) -> np.ndarray:
        self._sync_hold_reference()
        sim = self.sim
        quat = sim.base_quat
        rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        gravity_body = rot.inv().apply(np.array([0.0, 0.0, -1.0]))
        lin_vel_body = sim.body_frame(sim.base_lin_vel_world)
        ang_vel_body = sim.body_frame(sim.base_ang_vel_world)
        leg_lengths = sim.leg_lengths()
        leg_rate = (leg_lengths - self.prev_leg_lengths) / self.params.control_dt
        support = sim.wheel_stance()
        drop = np.array([support["leg_drop_left_m"], support["leg_drop_right_m"]])
        drop_rate = (drop - self._previous_leg_drop) / self.params.control_dt
        self._support_diag = {**support, **sim.leg_envelope(), **sim.wheel_axle_geometry(),
                              **sim.support_posture(),
                              "leg_differential_velocity_m_s": float(drop_rate[0] - drop_rate[1])}
        wheel_force, wheel_hit, body_hit, airborne = sim.contact_state(terrain_only=self.stage.robust_support)
        base_x = float(sim.base_pos[0])
        base_height_rel = float(sim.base_pos[2] - sim.support_height(base_x, float(sim.base_pos[1])))
        yaw = float(rot.as_euler("xyz")[2])
        yaw_error = self._heading_error(yaw)
        terrain_preview = sim.terrain_preview(
            sim.base_pos, yaw, np.array([0.10, 0.50, 0.90], dtype=np.float64)
        )

        # 官方口径 dof_acc：一个**控制周期**内的速度差（我们用 8 步窗口对齐官方
        # 的 0.0667 s，见 spec.DOF_ACC_WINDOW），否则惩罚会被放大 69 倍
        hist = self._dof_vel_hist
        if len(hist) >= 2:
            span = (len(hist) - 1) * self.params.control_dt
            # ``hist[0]`` is oldest and ``hist[-1]`` is newest.
            dof_acc = (hist[-1] - hist[0]) / span
        else:
            dof_acc = np.zeros(6, dtype=np.float64)
        # 最近一次下发的 6 个主动关节力矩（腿 4 + 轮 2），官方特权观测项
        torques = np.concatenate([self._last_leg_torque, self._last_wheel_torque])

        blocks = {
            "gravity": gravity_body,
            "base_ang_vel": ang_vel_body,
            "heading_error_sincos": np.array([np.sin(yaw_error), np.cos(yaw_error)]),
            "base_lin_vel": lin_vel_body,
            "base_height_rel": np.array([base_height_rel], dtype=np.float64),
            "base_pos_rel": sim.data.qpos[:2] - self._position_reference(),
            "terrain_preview": terrain_preview,
            "leg_joint_pos": (
                np.arctan2(np.sin(sim.joint_positions()), np.cos(sim.joint_positions()))
                if self.stage.continuous_position else sim.joint_positions()
            ),
            "leg_joint_vel": sim.joint_velocities(),
            "wheel_joint_vel": sim.wheel_velocities(),
            "command": self.command,
            "previous_action": self.previous_action if current_action is None else current_action,
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
            # critic receives exact simulator values; actor counterparts above
            # go through the sensor noise model in ObservationEncoder.
            "base_lin_vel_true": lin_vel_body,
            "base_pos_rel_true": sim.data.qpos[:2] - self._position_reference(),
        }
        observation, self._raw_cache, self._obs_cache = self._observation_encoder.encode(
            blocks, self.rng
        )
        return observation

    def _sync_hold_reference(self) -> None:
        if not self.stage.precision_control:
            return
        if self.stage.straight_tracking:
            # Capture once on entry to straight translation. Speed reversal and
            # height changes keep it; after a turn/stop capture the new heading.
            straight = self.command_active and abs(self.command[2]) <= 0.01
            if straight and not self._straight_active:
                q = self.sim.base_quat
                self.straight_yaw = float(Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz")[2])
            self._straight_active = bool(straight)
        zero = not self.command_active
        if zero and not self._hold_zero_active:
            self.hold_xy = self.sim.base_pos[:2].copy()
            q = self.sim.base_quat
            self.hold_yaw = float(Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz")[2])
        # Repeated stop commands and height changes do not move this reference.
        self._hold_zero_active = zero

    def _position_reference(self) -> np.ndarray:
        if self.stage.robust_support and self.command_active:
            # Velocity tracking must not depend on accumulated episode position.
            return self.sim.base_pos[:2].copy()
        return self.hold_xy if self.stage.precision_control and self._hold_zero_active else self.nominal_xy

    def _heading_error(self, yaw: float | None = None) -> float:
        """Wrapped heading error against the stage's hold/straight reference."""
        if self.stage.robust_support and self.command_active and not (
                self.stage.straight_tracking and abs(self.command[2]) <= 0.01):
            return 0.0
        if yaw is None:
            quat = self.sim.base_quat
            rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
            yaw = float(rot.as_euler("xyz")[2])
        reference = (self.straight_yaw if self.stage.straight_tracking and self.command_active
                     and abs(self.command[2]) <= 0.01 else
                     self.hold_yaw if self.stage.precision_control and self._hold_zero_active
                     else float(getattr(self, "nominal_yaw", 0.0)))
        delta = float(yaw) - reference
        return float(np.arctan2(np.sin(delta), np.cos(delta)))

    def _diagnostic_mode(self, airborne: float, base_x: float) -> str:
        """Return a simulator-only label for logs; it is never sent to the actor."""
        if self.jump_phase:
            return "jump"
        elif self.stage.name == "recovery" and abs(
            self.sim.data.qpos[2] - self.params.robot.nominal_stand_height
        ) > 0.05:
            return "recover"
        elif airborne > 0.5:
            return "airborne"
        elif self.sim.terrain_scan(base_x, np.zeros(1))[0] > 0.01:
            return "stair"
        return "normal"

    # ==================================================================
    # 步进
    # ==================================================================
    def _prepare_push(self):
        p = self.params.robust
        if not p.pushes_enabled:
            self._push_force[:] = 0.0
            self._push_remaining = 0
            return
        if self._push_remaining == 0 and self.steps >= self._next_push_step:
            angle = float(self.rng.uniform(-np.pi, np.pi))
            magnitude = float(self.rng.uniform(*p.push_force_n))
            self._push_force[:] = [magnitude * np.cos(angle), magnitude * np.sin(angle), 0.0]
            self._push_remaining = int(self.rng.integers(*p.push_duration_steps))
            self._next_push_step = self.steps + int(self.rng.integers(*p.push_interval_steps))
        if self._push_remaining > 0:
            self._push_remaining -= 1
        else:
            self._push_force[:] = 0.0

    def step(self, action):
        sim = self.sim
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self._advance_command()
        self._sync_hold_reference()

        quat = self.sim.base_quat
        rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        rpy = rot.as_euler("xyz")
        pitch = float(rpy[1])
        pitch_rate = float(self.sim.body_frame(self.sim.base_ang_vel_world)[1])
        body_vx = float(self.sim.body_frame(self.sim.base_lin_vel_world)[0])
        station_error = float(self.sim.data.qpos[0] - self.nominal_xy[0])

        state = {
            "nominal_joint_pos": np.zeros(4, dtype=np.float64),
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
        self._prepare_push()
        motor_controls = []
        wheel_velocity_samples = [sim.wheel_velocities().copy()] if self.stage.pitch_control else []
        for substep in range(self.params.robot.control_substeps):
            if substep == 0 or self.stage.motor_substep_control:
                motor_state = state
                dt = self.params.control_dt
                if self.stage.motor_substep_control:
                    motor_state = {**state, "joint_pos": sim.joint_positions(),
                                   "joint_vel": sim.joint_velocities(), "wheel_vel": sim.wheel_velocities()}
                    dt = float(sim.model.opt.timestep)
                control = self.actuators.compute(action, state=motor_state, dt=dt)
                sim.data.ctrl[sim.hip_actuator_ids] = control["leg_ctrl"]
                sim.data.ctrl[sim.wheel_actuator_ids] = control["wheel_ctrl"]
                motor_controls.append(control)
            # Scheduled pushes are external disturbances, never restoring forces.
            sim.data.xfrc_applied[sim.chassis_body, :3] += self._push_force
            try:
                sim.step()
            finally:
                sim.data.xfrc_applied[sim.chassis_body, :3] -= self._push_force
            if self.stage.pitch_control:
                wheel_velocity_samples.append(sim.wheel_velocities().copy())
        if self.stage.motor_substep_control:
            # mj_step integrates qpos last; refresh sites/IMU/contact proxies at
            # that new state before building the policy observation.
            sim.data.xfrc_applied[sim.chassis_body, :3] += self._push_force
            try:
                sim.forward()
            finally:
                sim.data.xfrc_applied[sim.chassis_body, :3] -= self._push_force
        self._last_leg_torque = np.asarray(control["leg_torque"], dtype=np.float64).copy()
        self._last_wheel_torque = np.asarray(control["wheel_torque"], dtype=np.float64).copy()
        if self.stage.motor_substep_control:
            # Do not hide within-cycle saturation or sign changes by averaging torque.
            control = dict(control)
            control["motor_leg_torque_samples"] = np.stack([v["leg_torque"] for v in motor_controls])
            control["motor_wheel_current_samples"] = np.stack([v["wheel_current"] for v in motor_controls])
            control["motor_leg_demand_samples"] = np.stack([v["leg_torque_demand"] for v in motor_controls])
        if self.stage.pitch_control:
            control["wheel_acceleration_samples"] = np.diff(wheel_velocity_samples, axis=0) / float(sim.model.opt.timestep)

        self.steps += 1
        self._station_max_abs = max(
            self._station_max_abs,
            abs(float(self.sim.data.qpos[0] - self.nominal_xy[0])),
        )
        yaw_error = self._heading_error()
        self._yaw_max_abs = max(self._yaw_max_abs, abs(yaw_error))
        self._station_window.append(
            abs(float(self.sim.data.qpos[0] - self.nominal_xy[0]))
        )
        post_quat = sim.base_quat
        post_rot = Rotation.from_quat([
            post_quat[1], post_quat[2], post_quat[3], post_quat[0]
        ])
        post_rpy = post_rot.as_euler("xyz")
        post_ang_vel_body = sim.body_frame(sim.base_ang_vel_world)
        self._roll_max_abs = max(self._roll_max_abs, abs(float(post_rpy[0])))
        self._pitch_max_abs = max(self._pitch_max_abs, abs(float(post_rpy[1])))
        self._ang_vel_sq_sum += float(np.square(post_ang_vel_body).sum())
        self._ang_vel_samples += 1
        self._advance_jump_phase()

        # 推进 dof_acc 的速度历史（必须在 _obs() 之前，最后一格是当前速度）
        self._dof_vel_hist.append(
            np.concatenate([self.sim.joint_velocities(), self.sim.wheel_velocities()])
        )
        obs = self._obs(current_action=action)
        terminated, truncated, reasons = self._termination()
        reward, terms = self._reward(control, state, action, terminated)
        output_metrics = self._output_metrics(control)
        self._previous_control = {key: np.asarray(value).copy() for key, value in control.items()}
        self.prev_action_prev = self.previous_action.copy()
        self.previous_action = action
        self.prev_leg_lengths = self.sim.leg_lengths()
        self._previous_leg_drop = np.array([self._support_diag["leg_drop_left_m"], self._support_diag["leg_drop_right_m"]])

        self._last_reward_terms = terms
        self._info = {
            **self._support_diag,
            "push_force_n": float(np.linalg.norm(self._push_force)),
            "wheel_dual_support": float(np.all(self._raw_cache["contact_flag"][:2] > 0.5)),
            **output_metrics,
            "wheel_acceleration_sq": float(np.square(control["wheel_acceleration_samples"]).mean()) if self.stage.pitch_control else 0.0,
            "joint_torque_applied": control["leg_torque"].copy(),
            "joint_saturation_per_joint": np.mean(np.abs(self._effort_samples(control)[:, :4]) >= .99, axis=0),
            "zero_command": float(not self.command_active),
            "hold_planar_error_m": (float(np.linalg.norm(sim.base_pos[:2] - self._position_reference()))
                                    if not self.command_active else 0.0),
            "hold_heading_error_rad": abs(self._heading_error()) if not self.command_active else 0.0,
            "tracking_vx_error": float(sim.body_frame(sim.base_lin_vel_world)[0] - self.command[0]),
            "tracking_yaw_error": float(post_ang_vel_body[2] - self.command[2]),
            "tracking_height_error": float(self._raw_cache["base_height_rel"][0] - self.command[3]),
            "command_height": float(self.command[3]),
            "reward_terms": terms,
            "tracking_pose_quality": self._tracking_quality,
            "initial_rolling_vx": self._initial_rolling_vx,
            "reward_unattributed": 0.0,
            "termination_reason": reasons,
            # 域随机化：原始值，供日志/复盘（critic 看到的是归一化版）
            "domain": {k: v[0] for k, v in self.domain.items()},
            "mode": self._diagnostic_mode(
                float(self._raw_cache["contact_flag"][3]), float(sim.base_pos[0])
            ),
            **control_diag(self.actuators),
            **state,
            # ---- 姿态 ----
            "roll": float(post_rpy[0]),
            "pitch": float(post_rpy[1]),
            "yaw": float(post_rpy[2]),
            "roll_max_abs": self._roll_max_abs,
            "pitch_max_abs": self._pitch_max_abs,
            "yaw_error": yaw_error,
            "yaw_error_abs": abs(yaw_error),
            "straight_heading_active": self.stage.straight_tracking and self._straight_active,
            "yaw_max_abs": self._yaw_max_abs,
            "roll_rate": float(post_ang_vel_body[0]),
            "pitch_rate": float(post_ang_vel_body[1]),
            "yaw_rate": float(post_ang_vel_body[2]),
            "angular_velocity_rms": float(np.sqrt(
                self._ang_vel_sq_sum / max(1, self._ang_vel_samples)
            )),
            "body_vy": float(sim.body_frame(sim.base_lin_vel_world)[1]),
            "base_height": float(self.sim.data.qpos[2]),
            "station_error": float(self.sim.data.qpos[0] - self.nominal_xy[0]),
            "station_error_abs": float(abs(self.sim.data.qpos[0] - self.nominal_xy[0])),
            "station_within_5cm": float(abs(self.sim.data.qpos[0] - self.nominal_xy[0]) <= 0.05),
            "station_max_abs": self._station_max_abs,
            "station_tail_abs": abs(float(self.sim.data.qpos[0] - self.nominal_xy[0])),
            # 末段 200 步的漂移均值 —— 真正的"稳态漂移"，用于验收统计
            "station_tail_mean": float(np.mean(self._station_window))
            if self._station_window else 0.0,
            "episode_steps": int(self.steps),
            "leg_length_left": float(self._obs_cache["leg_length"][0]),
            "leg_length_right": float(self._obs_cache["leg_length"][1]),
            "leg_length_rate_mean": float(self._obs_cache["leg_length"][2:].mean()),
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
    def _leg_inclination_angles(self) -> np.ndarray:
        """左右腿相对竖直方向的倾角（机体系、矢状面，坐下为 0，前倾为正）。

        官方 ``nominal_state = (theta_L - theta_R)^2`` 用的就是腿倾角之差；
        官方导出里 ``asset.l1/l2/offset`` 都是 0，theta0 恒为 0（该项失效），
        所以我们按同样的物理含义、用真实腿几何来算。
        """
        rot = self.sim.rotation_matrix()
        angles = np.zeros(2, dtype=np.float64)
        for side, (hip, wheel) in enumerate(zip(self.sim.hip_sites, self.sim.wheel_sites)):
            vec = rot.T @ (self.sim.data.site_xpos[wheel] - self.sim.data.site_xpos[hip])
            angles[side] = float(np.arctan2(vec[0], -vec[2]))   # 相对"竖直向下"
        return angles

    def _dof_pos_limit_violation(self, joint_pos: np.ndarray, soft_factor: float) -> float:
        """官方 ``dof_pos_limits``：超出软限位多少就罚多少（只算腿关节）。

        官方先把 URDF 限位按 ``soft_dof_pos_limit`` 收窄成软限位，再求和越界量。
        我们 MJCF 里髋关节是无范围铰链（限位靠腿长 tendon），所以机械行程取
        ``RobotParams.hip_joint_limit``（**占位值，待实测替换**）。
        """
        limit = float(self.params.robot.hip_joint_limit) * float(soft_factor)
        q = np.asarray(joint_pos, dtype=np.float64)
        return float(np.sum(np.maximum(0.0, -limit - q)) + np.sum(np.maximum(0.0, q - limit)))

    @staticmethod
    def _robust_square(x):
        """Quadratic near zero, linear for large errors; never clips away gradients."""
        x = np.abs(np.asarray(x, dtype=np.float64))
        return np.where(x <= 1.0, x * x, 2.0 * x - 1.0)

    def _normalized_effort(self, control):
        return np.concatenate([
            control["leg_torque"] / self.params.joint.torque_limit,
            control["wheel_current"] / self.params.wheel.current_limit_a,
        ])

    def _effort_samples(self, control):
        if "motor_leg_torque_samples" in control:
            return np.concatenate([control["motor_leg_torque_samples"] / self.params.joint.torque_limit,
                                   control["motor_wheel_current_samples"] / self.params.wheel.current_limit_a], axis=1)
        return self._normalized_effort(control)[None, :]

    def _effort_deltas(self, control):
        samples = self._effort_samples(control)
        if self._previous_control is not None:
            samples = np.vstack([self._normalized_effort(self._previous_control), samples])
        return np.diff(samples, axis=0) if len(samples) > 1 else np.zeros((1, 6))

    def _output_metrics(self, control):
        old = self._previous_control
        wheel_delta = control["wheel_target"] - old["wheel_target"] if old is not None else np.zeros(2)
        joint_delta = control["joint_target"] - old["joint_target"] if old is not None else np.zeros(4)
        joint_delta = np.arctan2(np.sin(joint_delta), np.cos(joint_delta))
        effort_delta = self._effort_deltas(control)
        effort_samples = self._effort_samples(control)
        return {
            "wheel_target_change_sq": float(np.square(wheel_delta).mean()),
            "joint_target_change_sq": float(np.square(joint_delta).mean()),
            "effort_change_sq": float(np.square(effort_delta).mean()),
            "joint_saturation_fraction": float(np.mean(np.abs(effort_samples[:, :4]) >= .99)),
            "wheel_saturation_fraction": float(np.mean(np.abs(effort_samples[:, 4:]) >= .99)),
        }

    def _precision_terms(self, control):
        p = precision_rewards_for_stage(self.stage, self.params)
        terms = {"hold_position": 0.0, "hold_heading": 0.0,
                 "wheel_target_change": 0.0, "joint_target_change": 0.0, "effort_change": 0.0}
        if not self.command_active:
            error = float(np.linalg.norm(self.sim.base_pos[:2] - self.hold_xy))
            terms["hold_position"] = -p.hold_position * float(self._robust_square(
                max(0.0, error - p.hold_position_deadband_m) / p.hold_position_scale_m))
            if self.stage.braking_tracking:
                # Same fixed anchor and local gradient; bound the far-field cost
                # so unavoidable braking distance cannot dominate survival/tracking.
                cap = p.hold_position_cost_cap
                terms["hold_position"] = cap * float(np.tanh(terms["hold_position"] / cap))
            terms["hold_heading"] = -p.hold_heading * float(self._robust_square(
                self._heading_error() / p.hold_heading_scale_rad))
        terms["effort"] = -p.effort * float(np.square(self._effort_samples(control)).mean())
        # The pre-clamp joint PD request preserves a learning signal above saturation.
        demand = np.concatenate([
            np.abs(control.get("motor_leg_demand_samples", control["leg_torque_demand"][None, :])) / self.params.joint.torque_limit,
            np.abs(control.get("motor_wheel_current_samples", control["wheel_current"][None, :])) / self.params.wheel.current_limit_a,
        ], axis=1)
        excess = np.maximum(0.0, demand - p.saturation_soft_fraction) / p.saturation_scale_fraction
        terms["saturation"] = -p.saturation * float(self._robust_square(excess).mean())
        if self._previous_control is not None:
            old = self._previous_control
            wheel_delta = (control["wheel_target"] - old["wheel_target"]) / p.wheel_target_change_scale_rad_s
            joint_delta = control["joint_target"] - old["joint_target"]
            joint_delta = np.arctan2(np.sin(joint_delta), np.cos(joint_delta)) / p.joint_target_change_scale_rad
            terms["wheel_target_change"] = -p.wheel_target_change * float(self._robust_square(wheel_delta).mean())
            terms["joint_target_change"] = -p.joint_target_change * float(self._robust_square(joint_delta).mean())
            terms["effort_change"] = -p.effort_change * float(np.square(self._effort_deltas(control)).mean())
        if self.stage.motor_substep_control:
            terms["effort_change"] = -p.effort_change * float(np.square(self._effort_deltas(control)).mean())
        if self.stage.pitch_control:
            terms["wheel_acceleration"] = -p.wheel_acceleration * float(self._robust_square(
                control["wheel_acceleration_samples"] / p.wheel_acceleration_scale_rad_s2).mean())
        return terms

    def _pitch_terms(self, roll: float, pitch: float) -> dict[str, float]:
        p = precision_rewards_for_stage(self.stage, self.params)
        # Split the old 3-point upright budget: roll 1, pitch 2 at zero.
        # A smooth near-zero bonus plus symmetric cost, no hard angle/action clamp.
        weight = self.stage.tracking_upright_weight
        excess = max(0.0, abs(pitch) - p.pitch_deadband_rad) / p.pitch_error_scale_rad
        return {
            "upright": weight / 3.0 * float(np.exp(-(roll / p.roll_sigma_rad)**2)),
            "pitch_alignment": weight * 2.0 / 3.0 * float(np.exp(-(pitch / p.pitch_sigma_rad)**2)),
            "pitch_error": -p.pitch_error * float(self._robust_square(excess)),
        }

    def _symmetry_terms(self, leg_lengths, fore_aft_left: float,
                        fore_aft_right: float, roll: float) -> dict[str, float]:
        p = precision_rewards_for_stage(self.stage, self.params)
        def cost(error, deadband, scale, weight):
            return -weight * float(self._robust_square(max(0.0, abs(error) - deadband) / scale))
        return {
            "leg_length_difference": cost(leg_lengths[0] - leg_lengths[1], p.leg_length_deadband_m,
                                          p.leg_length_scale_m, p.leg_length_difference),
            "fore_aft_difference": cost(fore_aft_left - fore_aft_right, p.fore_aft_deadband_m,
                                       p.fore_aft_scale_m, p.fore_aft_difference),
            "roll_error": cost(roll, p.roll_deadband_rad, p.roll_error_scale_rad, p.roll_error),
        }

    def _transition_terms(self, height_error: float, vx_error: float, pitch_rate: float):
        p = precision_rewards_for_stage(self.stage, self.params)
        return {
            "height_shortfall": -p.height_shortfall * float(self._robust_square(
                max(0.0, -height_error - p.height_shortfall_deadband_m) / p.height_shortfall_scale_m)),
            "vx_error_dense": -p.vx_error_dense * float(self._robust_square(vx_error / p.vx_error_scale_m_s)),
            "pitch_rate": -p.pitch_rate * float(self._robust_square(pitch_rate / p.pitch_rate_scale_rad_s)),
        }

    def _straight_terms(self, vx_error, yaw_error, vertical_velocity, leg_common_rate):
        p = precision_rewards_for_stage(self.stage, self.params)
        straight = self.command_active and abs(self.command[2]) <= 0.01
        return {
            "track_vx_coarse": p.vx_coarse_bonus * float(np.exp(-(vx_error / p.vx_coarse_sigma_m_s)**2)),
            "yaw_error_dense": -p.yaw_error_dense * float(self._robust_square(yaw_error / p.yaw_error_scale_rad_s)),
            "straight_heading": (-p.straight_heading * float(self._robust_square(
                self._heading_error() / p.straight_heading_scale_rad)) if straight else 0.0),
            "motion_vertical_velocity": (-p.motion_vertical_velocity * float(self._robust_square(
                vertical_velocity / p.motion_vertical_velocity_scale_m_s)) if self.command_active else 0.0),
            "leg_common_velocity": -p.leg_common_velocity * float(self._robust_square(
                leg_common_rate / p.leg_common_velocity_scale_m_s)),
        }

    def _tracking_pose_quality(self, height_error, pitch):
        p = precision_rewards_for_stage(self.stage, self.params)
        severity = ((height_error / p.tracking_height_sigma_m)**2
                    + (pitch / p.tracking_pitch_sigma_rad)**2)
        return float(p.tracking_pose_floor + (1-p.tracking_pose_floor)*np.exp(-.5*severity))

    def _leg_reach_terms(self, envelope):
        p = precision_rewards_for_stage(self.stage, self.params)
        terms = {}
        for direction, limit in (('forward',p.leg_forward_reach_soft_m),
                                 ('backward',p.leg_backward_reach_soft_m)):
            reach = np.array([envelope[f'leg_{direction}_reach_{side}_m'] for side in ('left','right')])
            excess = np.maximum(0.,reach-limit)/p.leg_reach_scale_m
            terms[f'leg_{direction}_reach'] = -p.leg_reach_cost*float(self._robust_square(excess).mean())
        return terms

    def _centered_support_terms(self, geometry):
        p = precision_rewards_for_stage(self.stage, self.params)
        offsets = np.array([geometry[f'wheel_fore_aft_{side}_m'] for side in ('left', 'right')])
        excess = np.maximum(0., np.abs(offsets) - p.wheel_fore_aft_deadband_m) / p.wheel_fore_aft_scale_m
        # Cost each side before averaging: equal or opposite offsets cannot cancel.
        return {'wheel_fore_aft_alignment': -p.wheel_fore_aft_alignment
                * float(self._robust_square(excess).mean())}

    def _aligned_support_terms(self, geometry, control):
        p = precision_rewards_for_stage(self.stage, self.params)
        excess = max(0., geometry['axle_line_offset_m'] - p.axle_offset_deadband_m) / p.axle_offset_scale_m
        terms = {'wheel_axle_alignment': -p.axle_alignment * float(self._robust_square(excess)),
                 'paired_joint_target_alignment': 0.0}
        # The mirrored MJCF uses the same signed axes for corresponding joints.
        # Weak command cost only in stand/straight motion; actual axle cost is primary.
        if abs(self.command[2]) <= .01:
            target = control['joint_target']
            delta = target[:2] - target[2:]
            delta = np.arctan2(np.sin(delta), np.cos(delta))
            excess = np.maximum(0., np.abs(delta) - p.paired_target_deadband_rad) / p.paired_target_scale_rad
            terms['paired_joint_target_alignment'] = -p.paired_target_alignment * float(self._robust_square(excess).mean())
        return terms

    def _simple_reward(self, control, action, terminated, stage=None):
        from .simple_rewards import tracking_terms
        stage = stage or self.stage
        sim = self.sim
        rpy = Rotation.from_quat([sim.base_quat[1],sim.base_quat[2],sim.base_quat[3],sim.base_quat[0]]).as_euler('xyz')
        delta = np.asarray(action)-self.previous_action
        delta[2:] = np.arctan2(np.sin(np.pi*delta[2:]),np.cos(np.pi*delta[2:]))/np.pi
        heading = self._heading_error() if abs(self.command[2]) <= .01 else 0.
        stop_distance = float(np.linalg.norm(sim.base_pos[:2]-self.hold_xy)) if not self.command_active else 0.
        pitch_error = (self._support_diag['support_pitch_error_rad']
                       if stage.simple_pitch_reference == 'support' else float(rpy[1]))
        height = float(self._raw_cache['base_height_rel'][0])
        self._tracking_quality = 1.0  # No posture/geometry multiplication of task reward.
        return tracking_terms(precision_rewards_for_stage(stage,self.params),
            vx_error=float(sim.body_frame(sim.base_lin_vel_world)[0]-self.command[0]),
            yaw_rate_error=float(sim.body_frame(sim.base_ang_vel_world)[2]-self.command[2]),
            height_error=height-self.command[3],roll=float(rpy[0]),pitch_error=pitch_error,
            heading_error=heading,stop_distance=stop_distance,action_delta=delta,
            terminated=terminated,axle_offset=self._support_diag['axle_line_offset_m'],
            axle_enabled=stage.simple_axle_cost,dt=self.params.control_dt)

    def _reward(self, control: dict, state: dict, action: np.ndarray,
                terminated: bool) -> tuple[float, dict[str, float]]:
        if self.stage.simple_reward:
            terms = self._simple_reward(control, action, terminated)
            return float(sum(terms.values())), terms
        w = self.params.rewards
        flags = self.reward_flags
        sim = self.sim
        terms: dict[str, float] = {}

        rpy = Rotation.from_quat(
            [sim.base_quat[1], sim.base_quat[2], sim.base_quat[3], sim.base_quat[0]]
        ).as_euler("xyz")
        roll, pitch = float(rpy[0]), float(rpy[1])
        pitch_objective_error = (self._support_diag['support_pitch_error_rad']
                                 if self.stage.aligned_support else pitch)
        tilt_sq = roll * roll + pitch * pitch

        lin_vel_body = sim.body_frame(sim.base_lin_vel_world)
        ang_vel_body = sim.body_frame(sim.base_ang_vel_world)
        vx, vy = float(lin_vel_body[0]), float(lin_vel_body[1])
        yaw_rate = float(ang_vel_body[2])

        leg_lengths = sim.leg_lengths()
        leg_mean = float(leg_lengths.mean())
        leg_diff = float(leg_lengths[0] - leg_lengths[1])
        # command[3] is base height above the local support surface, not leg length.
        base_height_rel = float(
            sim.base_pos[2] - sim.support_height(float(sim.base_pos[0]), float(sim.base_pos[1]))
        )
        height_error = base_height_rel - float(self.command[3])
        leg_error = leg_mean - float(self.params.robot.nominal_leg_length)
        station_error = float(sim.data.qpos[0] - self.nominal_xy[0])
        wheel_force, wheel_hit, body_hit, airborne = sim.contact_state(terrain_only=self.stage.robust_support)

        joint_pos = state["joint_pos"]
        joint_vel = state["joint_vel"]

        if self.stage.continuous_position:
            # The same continuous tracking objective includes zero velocities.
            # No action sign or minimum motion is prescribed. v16 adds a weak
            # paired-target cost while keeping all six actions independent.
            upright = float(np.exp(-tilt_sq / 0.12))
            action_delta = action - self.previous_action
            action_delta[2:] = np.arctan2(
                np.sin(np.pi * action_delta[2:]), np.cos(np.pi * action_delta[2:])
            ) / np.pi
            normalized_efforts = np.concatenate([
                control["leg_torque"] / self.params.joint.torque_limit,
                control["wheel_torque"] / self.params.wheel.joint_torque_limit,
            ])
            terms = {
                "upright": self.stage.tracking_upright_weight * upright,
                "track_vx": self.stage.tracking_vx_weight * float(np.exp(-((vx - self.command[0]) / self.stage.tracking_vx_sigma) ** 2)),
                "track_yaw": self.stage.tracking_yaw_weight * float(np.exp(-((yaw_rate - self.command[2]) / 0.5) ** 2)),
                "track_height": self.stage.tracking_height_weight * float(np.exp(-(height_error / 0.025) ** 2)),
                "velocity_error": -0.5 * (vx - self.command[0]) ** 2
                                  -0.15 * (yaw_rate - self.command[2]) ** 2,
                "height_error": -2.0 * (height_error / 0.05) ** 2,
                "lateral_velocity": -0.5 * vy ** 2,
                "vertical_velocity": -0.1 * float(lin_vel_body[2] ** 2),
                "angular_velocity_xy": -0.05 * float(np.square(ang_vel_body[:2]).sum()),
                "action_rate": -0.02 * float(np.square(action_delta).mean()),
                "effort": -0.01 * float(np.square(normalized_efforts).mean()),
                "body_contact": -5.0 * body_hit,
                "termination": -self.stage.tracking_termination_weight if terminated else 0.0,
            }
            if self.stage.pitch_control:
                terms.update(self._pitch_terms(roll, pitch_objective_error))
            if self.stage.straight_tracking:
                terms.update(self._straight_terms(
                    vx - self.command[0], yaw_rate - self.command[2], sim.base_lin_vel_world[2],
                    float((leg_lengths - self.prev_leg_lengths).mean() / self.params.control_dt)))
            if self.stage.stance_geometry:
                stance = sim.wheel_stance()
                stagger = max(0.0, stance["wheel_stagger_m"] - self.stage.wheel_stagger_soft_m)
                offset = max(0.0, stance["wheel_offset_max_m"] - self.stage.wheel_offset_soft_m)
                severity = (stagger / 0.08) ** 2 + (offset / 0.10) ** 2
                quality = float(np.exp(-severity))
                # A static diagonal prop must not collect full task rewards.
                # No joint target or action direction is prescribed.
                for key in ("upright", "track_vx", "track_yaw", "track_height"):
                    terms[key] *= quality
                if self.stage.straight_tracking:
                    terms["track_vx_coarse"] *= quality
                if self.stage.pitch_control:
                    terms["pitch_alignment"] *= quality
                terms["stance_geometry"] = -min(6.0, severity)
                terms["effort"] *= 5.0
            if self.stage.precision_control:
                terms.update(self._precision_terms(control))
            if self.stage.robust_support:
                p = precision_rewards_for_stage(self.stage, self.params)
                if not self.command_active and not self.stage.braking_tracking:
                    position = max(0.0, float(np.linalg.norm(sim.base_pos[:2] - self.hold_xy)) - p.hold_position_deadband_m)
                    station_quality = float(np.exp(-(position / p.hold_task_position_scale_m) ** 2
                                                   - (self._heading_error() / p.hold_task_heading_scale_rad) ** 2))
                    # Zero speed without holding position/heading is incomplete tracking.
                    terms["track_vx"] *= station_quality
                    terms["track_yaw"] *= station_quality
                    if self.stage.straight_tracking:
                        terms["track_vx_coarse"] *= station_quality
                difference = max(0.0, abs(self._support_diag["leg_drop_difference_m"]) - p.leg_drop_deadband_m)
                terms["leg_drop_difference"] = -p.leg_drop_difference * float(self._robust_square(difference / p.leg_drop_scale_m))
                terms["leg_differential_velocity"] = -p.leg_differential_velocity * float(self._robust_square(
                    self._support_diag["leg_differential_velocity_m_s"] / p.leg_differential_velocity_scale_m_s))
                terms["wheel_support"] = -p.missing_wheel_support * float(2.0 - np.sum(wheel_hit))
                terms["stand_vertical_velocity"] = (-p.stand_vertical_velocity * float(self._robust_square(
                    lin_vel_body[2] / p.stand_vertical_velocity_scale_m_s)) if not self.command_active else 0.0)
            if self.stage.symmetric_support:
                terms.update(self._symmetry_terms(leg_lengths, self._support_diag["wheel_fore_aft_left_m"],
                                                  self._support_diag["wheel_fore_aft_right_m"], roll))
            if self.stage.transition_tracking:
                terms.update(self._transition_terms(height_error, vx - self.command[0], ang_vel_body[1]))
            if self.stage.coordinated_tracking:
                p = precision_rewards_for_stage(self.stage, self.params)
                quality = self._tracking_pose_quality(height_error, pitch_objective_error)
                self._tracking_quality = quality
                # Separate pose cost preserves the original speed incentive,
                # including away from the desired height/pitch during recovery.
                terms["tracking_pose_cost"] = -p.tracking_pose_cost_weight * (1-quality)
                terms["height_error_dense"] = -p.height_error_dense * float(self._robust_square(
                    height_error / p.height_error_scale_m))
            if self.stage.compact_support:
                terms.update(self._leg_reach_terms(self._support_diag))
            if self.stage.centered_support:
                terms.update(self._centered_support_terms(self._support_diag))
            if self.stage.aligned_support:
                terms.update(self._aligned_support_terms(self._support_diag, control))
            return float(sum(terms.values())), terms

        # ==================================================================
        # 官方奖励（阶段开关 official_rewards=True 时只走这条路）
        #
        # 权重与形式逐项照抄 StackForce 导出（见 spec.OfficialRewards），
        # 目的：训练曲线能和开源直接对比。需要额外塑形时不要改这里的数，
        # 而是给阶段关掉官方项、改用下面那套 RewardWeights。
        # ==================================================================
        if self.stage.official_rewards:
            o = self.params.official_rewards
            sigma = o.tracking_sigma

            # tracking（exp(-e^2/0.25)，官方 sigma = 0.25）
            terms["tracking_lin_vel"] = o.tracking_lin_vel * float(
                np.exp(-((vx - float(self.command[0])) ** 2) / sigma)
            )
            terms["tracking_ang_vel"] = o.tracking_ang_vel * float(
                np.exp(-((yaw_rate - float(self.command[2])) ** 2) / sigma)
            )
            # 离地高度跟踪：官方是 |h - h_cmd| 的绝对误差惩罚（不是平方）
            terms["base_height"] = o.base_height * abs(
                float(sim.data.qpos[2]) - float(self.command[3])
            )
            # 官方 nominal_state = (theta_L - theta_R)^2：两腿倾角差
            # （官方 asset.l1/l2 为 0 导致该项恒 0，我们用真实腿几何算）
            leg_angles = self._leg_inclination_angles()
            terms["nominal_state"] = o.nominal_state * float(
                (leg_angles[0] - leg_angles[1]) ** 2
            )
            # 姿态与角速度
            terms["lin_vel_z"] = o.lin_vel_z * float(lin_vel_body[2] ** 2)
            terms["ang_vel_xy"] = o.ang_vel_xy * float(
                ang_vel_body[0] ** 2 + ang_vel_body[1] ** 2
            )
            gravity_body = np.asarray(self._raw_cache["gravity"], dtype=np.float64)
            terms["orientation"] = o.orientation * float(
                gravity_body[0] ** 2 + gravity_body[1] ** 2
            )
            # 正则化
            terms["dof_vel"] = o.dof_vel * float(
                np.sum(np.square(np.concatenate([joint_vel, state["wheel_vel"]])))
            )
            dof_acc = np.asarray(self._raw_cache["dof_acc"], dtype=np.float64)
            terms["dof_acc"] = o.dof_acc * float(np.sum(np.square(dof_acc)))
            torque_all = np.concatenate([
                np.asarray(control["leg_torque"], dtype=np.float64),
                np.asarray(control["wheel_torque"], dtype=np.float64),
            ])
            terms["torques"] = o.torques * float(np.sum(np.square(torque_all)))
            terms["action_rate"] = o.action_rate * float(
                np.sum(np.square(self.previous_action - action))
            )
            terms["action_smooth"] = o.action_smooth * float(
                np.sum(np.square(action - 2.0 * self.previous_action + self.prev_action_prev))
            )
            # 碰撞（只算机身，阈值 0.1 N）与软限位
            terms["collision"] = o.collision * (
                1.0 if body_hit > 0.5 else 0.0
            )
            terms["dof_pos_limits"] = o.dof_pos_limits * self._dof_pos_limit_violation(
                joint_pos, o.soft_dof_pos_limit
            )
            terms["termination"] = -o.termination if terminated else 0.0
            terms["custom_reward"] = 0.0
            terms["alive"] = 0.0
            return float(sum(terms.values())), terms

        # ---------------------------------------------------------- posture
        if flags["posture"] or flags["recovery"]:
            upright = np.exp(-(20.0 * tilt_sq))
        else:
            upright = np.exp(-(20.0 * tilt_sq))
        terms["upright"] = w.upright * upright
        # Meter-scale squared errors were almost zero at the observed collapsed
        # pose (height 0.18 m, legs 0.12 m), so PPO could gain by resting on the
        # chassis. Normalize stand errors by 5 cm to give the requested stance a
        # useful gradient while preserving the old scale for other stages.
        height_scale = 0.05 if self.stage.name == "stand" else 1.0
        leg_scale = 0.05 if self.stage.name == "stand" else 1.0
        terms["height"] = -w.height * (height_error / height_scale) ** 2
        terms["leg_length"] = -w.leg_length * (leg_error / leg_scale) ** 2
        terms["posture_symmetry"] = -w.leg_symmetry * leg_diff * leg_diff
        terms["joint_neutral"] = -w.joint_neutral * float(np.square(joint_pos).mean())

        # ------------------------------------------------- regularization
        terms["action_rate"] = -w.action_rate * float(np.square(action - self.previous_action).mean())
        action_accel = action - 2.0 * self.previous_action + self.prev_action_prev
        terms["action_smooth"] = -w.action_smooth * float(np.square(action_accel).mean())
        terms["leg_action"] = -w.leg_action * float(np.square(action[2:6]).mean())
        terms["leg_velocity_action"] = 0.0   # 动作通道已移除
        terms["joint_velocity"] = -w.joint_velocity * min(float(np.square(joint_vel).mean()), 400.0)
        terms["joint_torque"] = -w.joint_torque * min(float(np.square(control["leg_torque"]).mean()), 1600.0)
        terms["wheel_power"] = -w.wheel_power * min(float(np.square(control["wheel_current"]).mean()), 400.0)
        terms["wheel_differential"] = (
            0.0 if self.command_active
            else -w.wheel_differential * min(float(np.square(np.mean(sim.wheel_velocities()))), 25.0)
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

        terms["track_yaw"] = 0.0
        if flags["track_yaw"] and abs(self.command[2]) > 0.01:
            cmd_yaw = float(self.command[2])
            yaw_error = (yaw_rate - cmd_yaw) / max(abs(cmd_yaw), 0.15)
            terms["track_yaw"] = (
                w.track_yaw * float(np.exp(-((yaw_error / 0.5) ** 2)))
                - w.track_yaw_square * float((0.25 * yaw_error) ** 2)
            )

        # ------------------------------------------------------------ station
        # 前后漂移要求：死区内零惩罚，死区外线性惩罚（有界）。
        station_over = max(0.0, abs(station_error) - self.params.station_deadband)
        if not self.command_active:
            terms["station"] = -w.station * min(station_over, 0.15)
            terms["station_vel"] = -w.station_vel * abs(vx)
        else:
            terms["station"] = terms["station_vel"] = 0.0

        # -------------------------------------------------------- stand still
        if flags["stand_still"] and not self.command_active:
            terms["stand_vx"] = -w.stand_vx * abs(vx)
            terms["stand_yaw"] = -w.stand_yaw * abs(yaw_rate)
            heading_error = self._heading_error(float(rpy[2]))
            terms["stand_heading"] = -w.stand_heading * heading_error * heading_error
            terms["stand_ang_vel"] = -w.stand_ang_vel * float(
                np.square(ang_vel_body).sum()
            )
            terms["stand_tilt"] = -w.stand_tilt * tilt_sq
            terms["stand_wheel"] = -w.stand_wheel_speed * min(
                float(np.square(sim.wheel_velocities()).mean()), 25.0
            )
            # In this MJCF, a positive common wheel-speed target increases
            # pitch (verified by paired one-step torque probes). Give PPO a
            # direct corrective signal instead of relying only on eventual
            # fall/termination feedback.
            pitch_balance_weight = w.stand_balance_action * (1.5 if pitch < 0.0 else 1.0)
            terms["stand_balance_action"] = -pitch_balance_weight * (
                float(np.clip(pitch, -0.4, 0.4)) * float(action[0])
            )
            terms["stand_balance_rate"] = -w.stand_balance_rate * (
                float(np.clip(ang_vel_body[1], -2.0, 2.0)) * float(action[0])
            )
            terms["stand_hip_action"] = 0.0  # leg_action regularizer is shared with other stages
            negative_hip_action = max(0.0, -float(np.mean(action[2:6])))
            negative_pitch_factor = float(np.clip(-pitch / 0.08, 0.0, 1.0))
            negative_pitch_rate_factor = float(
                np.clip(-ang_vel_body[1] / 0.30, 0.0, 1.0)
            )
            terms["stand_negative_pitch_hip_action"] = (
                -w.stand_negative_pitch_hip_action * negative_pitch_factor
                * negative_pitch_rate_factor * negative_hip_action * negative_hip_action
            )
            heading_error = self._heading_error(float(rpy[2]))
            terms["stand_heading_action"] = -w.stand_heading_action * (
                heading_error * float(action[1])
            )
            terms["stand_heading_rate"] = -w.stand_heading_rate * (
                float(np.clip(ang_vel_body[2], -2.0, 2.0)) * float(action[1])
            )
            # A chassis/gimbal touch is a collapsed stand, even if the wheels
            # keep the body from crossing the tilt termination threshold.
            terms["stand_body_contact"] = -w.stand_body_contact * body_hit
        else:
            terms["stand_vx"] = terms["stand_yaw"] = terms["stand_heading"] = 0.0
            terms["stand_tilt"] = 0.0
            terms["stand_ang_vel"] = terms["stand_wheel"] = 0.0
            terms["stand_body_contact"] = 0.0
            terms["stand_balance_action"] = terms["stand_balance_rate"] = 0.0
            terms["stand_hip_action"] = 0.0
            terms["stand_negative_pitch_hip_action"] = 0.0
            terms["stand_heading_action"] = terms["stand_heading_rate"] = 0.0

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
        # Bound regularizers, never rescale the task objective. The old global
        # cap forced every collapsed stand to exactly -10 reward/step, erasing
        # the height, upright, and body-contact gradients needed to leave that
        # local optimum.
        regularizers = {
            "posture_symmetry", "joint_neutral", "action_rate", "action_smooth",
            "leg_action", "joint_velocity", "joint_torque", "wheel_power",
            "wheel_differential", "stand_wheel",
        }
        if self.stage.name == "stand":
            regularization = sum(terms.get(key, 0.0) for key in regularizers)
            if regularization < -5.0:
                scale = 5.0 / abs(regularization)
                for key in regularizers:
                    if key in terms:
                        terms[key] *= scale
        else:
            # Keep the existing safety bound for the other capability stages.
            protected = {
                "upright", "track_vx", "track_yaw", "airborne_upright",
                "recovery_upright", "recovery_progress", "jump_height", "alive",
            }
            continuous = sum(v for k, v in terms.items() if k not in protected)
            if continuous < -10.0:
                scale = 10.0 / abs(continuous)
                for key in terms:
                    if key not in protected:
                        terms[key] *= scale
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
        reachable_height = self.params.robot.leg_length_min * 0.6
        reasons: list[str] = []

        # Keep the positive and negative boundary identical despite Euler-angle
        # floating-point roundoff at exactly the configured limit.
        if tilt > self.tilt_limit + 1e-9:
            self.tilt_exceed_steps += 1
            if self.tilt_exceed_steps >= self.stage.tilt_limit_hold_steps:
                reasons.append("tilt_limit")
        else:
            self.tilt_exceed_steps = 0

        # Chassis/gimbal support is a fall, even when the wheels keep the base
        # just inside the angular limit. Require a few consecutive frames to
        # ignore one-step contact solver noise.
        if self.stage.name == "stand" or self.stage.continuous_position:
            _, _, body_hit, _ = sim.contact_state(terrain_only=self.stage.robust_support)
            self.body_contact_exceed_steps = (
                self.body_contact_exceed_steps + 1 if body_hit > 0.5 else 0
            )
            if self.body_contact_exceed_steps >= 3:
                reasons.append("body_contact")
        else:
            self.body_contact_exceed_steps = 0

        if self.stage.stance_geometry:
            stance = sim.wheel_stance()
            invalid = (stance["wheel_stagger_m"] > self.stage.wheel_stagger_limit_m
                       or stance["wheel_offset_max_m"] > self.stage.wheel_offset_limit_m)
            self.stance_invalid_steps = (
                self.stance_invalid_steps + 1
                if invalid and self.steps > self.stage.stance_reset_grace_steps else 0
            )
            if self.stance_invalid_steps >= self.stage.stance_invalid_hold_steps:
                reasons.append("wheel_stance_limit")

        yaw_error = self._heading_error(float(rpy[2]))
        if self.stage.name == "stand" and abs(yaw_error) > self.yaw_limit:
            reasons.append("yaw_limit")

        if sim.data.qpos[2] < reachable_height:
            reasons.append("height_limit")

        station_abs = float(abs(sim.data.qpos[0] - self.nominal_xy[0]))
        if not self.stage.continuous_position and not self.command_active and station_abs > self.params.station_hard_limit:
            reasons.append("station_limit")
        drift = float(np.linalg.norm(sim.data.qpos[:2] - self.nominal_xy))
        if not self.stage.continuous_position and drift > self.params.terminate_on_drift:
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
