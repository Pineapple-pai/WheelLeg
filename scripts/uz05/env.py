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

    观测（214 维 = actor 34×历史5 + 特权 44）在任何 capability 下都完整存在
    且实时更新，因此**任何 checkpoint 在任何阶段都能直接加载，永远不需要迁移**。
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        stage: str = DEFAULT_STAGE,
        params: EnvParams | None = None,
        seed: int | None = None,
        stand_level: int = 2,
        assist_scale: float = 1.0,
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
        self.params.balance.station_deadband = self.stand_level.deadband
        self.params.balance.station_kp = self.stand_level.station_kp
        self.params.balance.assist_scale = float(np.clip(assist_scale, 0.0, 1.0))
        self.tilt_limit = self.stand_level.tilt_limit if stage == "stand" else self.stage.tilt_limit

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

        self.action_space = spaces.Box(-1.0, 1.0, (ACTION_DIM,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)

        self.reward_flags = active_rewards(self.stage)

        # 运行时状态
        self.steps = 0
        self.previous_action = np.zeros(ACTION_DIM, dtype=np.float64)
        self.command = np.zeros(5, dtype=np.float64)
        self.command_target = np.zeros(5, dtype=np.float64)
        self.nominal_xy = np.zeros(2, dtype=np.float64)
        self.prev_leg_lengths = self.sim.leg_lengths()
        self.tilt_exceed_steps = 0
        self.airborne_prev = 0.0
        self.airborne_steps = 0
        self.jump_phase = 0
        self.jump_timer = 0.0
        self.phase_clock = 0.0
        self.last_landing_impact = 0.0
        self._last_reward_terms: dict[str, float] = {}
        self._station_max_abs = 0.0
        self._station_window: deque = deque(maxlen=200)   # 末段稳态漂移窗口
        self._actor_history: deque = deque(maxlen=OBS_HISTORY)   # actor 观测历史堆叠
        self._nominal_model = self.sim.nominal()                  # 域随机化基准值
        self.domain: dict = {}
        self._raw_cache: dict[str, np.ndarray] = {}
        self._info: dict[str, Any] = {}

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

    # ==================================================================
    # 命令
    # ==================================================================
    def _sample_command(self) -> None:
        stage = self.stage
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
        limit = stage.init_tilt
        roll, pitch, yaw = self.rng.uniform(-limit, limit, 3)
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
            self.domain = {name: (0.0, 0.0) for name in dr.PARAM_NAMES}
            self.sim.apply_randomization(self._nominal_model, {k: (1.0 if "scale" in k else 0.0, 0.0)
                                                              for k in dr.PARAM_NAMES})
        self.sim.forward()

        self.nominal_xy = self.sim.data.qpos[:2].copy()
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
        self.airborne_prev = 0.0
        self.airborne_steps = 0
        self.jump_phase = 0
        self.jump_timer = 0.0
        self.phase_clock = 0.0
        self.last_landing_impact = 0.0
        self._station_max_abs = 0.0
        self._station_window: deque = deque(maxlen=200)   # 末段稳态漂移窗口
        self._actor_history: deque = deque(maxlen=OBS_HISTORY)   # actor 观测历史堆叠
        self.actuators.reset()
        _, _, _, airborne_now = self.sim.contact_state()
        self.airborne_prev = airborne_now
        self._sample_command()
        return self._obs(), {}

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
            "base_ang_vel": ang_vel_body,
            "base_pos_rel": sim.data.qpos[:2] - self.nominal_xy,
            "leg_joint_pos": sim.joint_positions(),
            "leg_joint_vel": sim.joint_velocities(),
            "wheel_joint_vel": sim.wheel_velocities(),
            "command": self.command,
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
            frame += self.rng.uniform(-1.0, 1.0, frame.shape) * noise_vec
        # 2) 历史堆叠（最早 → 最新）；不足时用最早一帧填充。
        #    注意：噪声在入栈之前加，所以 5 帧历史各自独立带噪（与官方一致）。
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
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self._advance_command()

        quat = self.sim.base_quat
        rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        rpy = rot.as_euler("xyz")
        pitch, pitch_rate = float(rpy[1]), float(self.sim.base_ang_vel_world[1])
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
        reward, terms = self._reward(control, state, action, terminated)
        self.prev_action_prev = self.previous_action.copy()
        self.previous_action = action
        self.prev_leg_lengths = self.sim.leg_lengths()

        self._last_reward_terms = terms
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
            "assist_scale": self.params.balance.assist_scale,
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
        leg_error = leg_mean - float(self.command[3])
        height_error = float(sim.data.qpos[2] - self.params.robot.nominal_stand_height)
        station_error = float(sim.data.qpos[0] - self.nominal_xy[0])
        wheel_force, wheel_hit, body_hit, airborne = sim.contact_state()

        joint_pos = state["joint_pos"]
        joint_vel = state["joint_vel"]

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
        terms["height"] = -w.height * height_error * height_error
        terms["leg_length"] = -w.leg_length * leg_error * leg_error
        terms["posture_symmetry"] = -w.leg_symmetry * leg_diff * leg_diff
        terms["joint_neutral"] = -w.joint_neutral * float(np.square(joint_pos).mean())

        # ------------------------------------------------- regularization
        terms["action_rate"] = -w.action_rate * float(np.square(action - self.previous_action).mean())
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
            terms["stand_wheel"] = -w.stand_wheel_speed * min(
                float(np.square(sim.wheel_velocities()).mean()), 25.0
            )
        else:
            terms["stand_vx"] = terms["stand_yaw"] = terms["stand_wheel"] = 0.0

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
        continuous = sum(v for k, v in terms.items()
                         if k not in ("upright", "track_vx", "track_yaw",
                                      "airborne_upright", "recovery_upright",
                                      "recovery_progress", "jump_height", "alive"))
        if continuous < -10.0:
            scale = 10.0 / abs(continuous)
            for k in list(terms):
                if k not in ("upright", "track_vx", "track_yaw", "airborne_upright",
                             "recovery_upright", "recovery_progress", "jump_height", "alive"):
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
        reachable_height = self.params.robot.leg_length_min * 0.6
        reasons: list[str] = []

        if tilt > self.tilt_limit:
            self.tilt_exceed_steps += 1
            if self.tilt_exceed_steps > self.stage.tilt_limit_hold_steps:
                reasons.append("tilt_limit")
        else:
            self.tilt_exceed_steps = 0

        if sim.data.qpos[2] < reachable_height:
            reasons.append("height_limit")

        station_abs = float(abs(sim.data.qpos[0] - self.nominal_xy[0]))
        if not self.command_active and station_abs > self.params.station_hard_limit:
            reasons.append("station_limit")
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
