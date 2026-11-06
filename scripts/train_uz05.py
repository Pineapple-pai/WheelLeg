"""训练入口：分阶段（capability）训练 UZ-05 轮腿机器人。

**继承机制**：当前环境观测为 94 维（当前 actor 观测 38 + critic 特权量 56），动作 6 维。
actor 不使用历史帧堆叠。接口、MJCF 哈希和执行器参数相同的阶段可以直接继承；旧版 236/6、226/6 与 73 维策略均不兼容。
能力阶段不改变张量形状。

站立课程只用 PPO 学习平衡：没有外环辅助、残差策略、教师策略或辅助退火。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict
from collections import deque
from pathlib import Path

# ---- 线程数必须在 import torch 之前卡死 ----
# subproc 环境的子进程会继承这些变量。不限制的话：32 个环境进程 × 每进程 16 个
# OMP/BLAS 线程 = 500+ 线程抢 16 个核，机器直接卡死。
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize

from uz05.env import UZ05Env
from uz05.policy import AsymmetricActorCriticPolicy
from uz05.command_sampling import SAMPLING_NAMES, STARTUP_STAGES, STARTUP_SAMPLERS, sampling_contract
from uz05.spec import (
    ACTION_DIM, ACTION_SPEC, JOINT_TRACKING_ACTION_SPEC, ACTOR_FRAME_DIM, ACTOR_OBS_BLOCKS, ACTOR_OBS_DIM,
    CAPABILITY_ORDER, OBS_DIM, OBS_SCALE, OBS_SOURCE, PRIV_OBS_BLOCKS, PRIV_OBS_DIM,
    EnvParams, MODEL_XML, REPO_ROOT, STAGE_BY_NAME, STAGES, precision_rewards_for_stage, env_params_for_stage, action_spec_for_stage,
)

CONTRACT_VERSION = "uz05_iface_v5"


def model_xml_sha256() -> str:
    """Fingerprint the exact MJCF used to train a checkpoint."""
    return hashlib.sha256(MODEL_XML.read_bytes()).hexdigest()


def actuator_contract(stage: str = "stand", wheel_torque_model: str = "A_legacy") -> dict:
    """Capture actuator and control-period semantics for checkpoint safety."""
    params = EnvParams()
    if wheel_torque_model == "B_rated":
        params.wheel.efficiency = 1.0
    elif wheel_torque_model != "A_legacy":
        raise ValueError(wheel_torque_model)
    return {
        "joint": asdict(params.joint),
        **({"actuator_update_mode": "motor_substep", "actuator_dt_s": params.control_dt / params.robot.control_substeps}
           if STAGE_BY_NAME[stage].motor_substep_control else {}),
        **({"action_mode": "independent_wheel_velocity_circular_joint_position_v1",
            "wheel_velocity_scale_rad_s": STAGE_BY_NAME[stage].wheel_velocity_scale_rad_s,
            "joint_position_scale_rad": JOINT_TRACKING_ACTION_SPEC[1][2],
            "joint_angle_observation": "wrapped_at_pi"}
           if STAGE_BY_NAME[stage].continuous_position else {}),
        "wheel": asdict(params.wheel),
        "control_dt_s": params.control_dt,
    }

def observation_reference(stage: str) -> str:
    spec = STAGE_BY_NAME[stage]
    if spec.straight_tracking:
        return "stop_anchor_straight_heading_entry_else_zero_v1"
    return ("stop_anchor_else_zero" if spec.robust_support else
            "zero_command_entry" if spec.precision_control else "episode_initial")


def can_preserve_reward_statistics(source_contract: dict, target_stage: str) -> bool:
    source_stage = source_contract.get('stage')
    if source_stage == target_stage:
        return True
    # Startup narrows tasks/initial states; its objective is the v16 objective.
    # Keep all other stage transitions behind the normalization-reset guard.
    if {source_stage, target_stage} != {'joint_tracking_v16', 'joint_tracking_startup'}:
        return False
    expected = asdict(precision_rewards_for_stage(STAGE_BY_NAME[target_stage], EnvParams()))
    # Contract JSON stores tuple-valued ranges as lists.
    return (json.dumps(source_contract.get('precision_objective'), sort_keys=True)
            == json.dumps(expected, sort_keys=True))


def environment_contract(stage: str, domain_randomization: bool, closed_chain_reset: bool = True) -> dict:
    params = env_params_for_stage(stage)
    params.domain_randomization.enabled = bool(domain_randomization)
    params.robot.closed_chain_reset = bool(closed_chain_reset)
    return {"robot": asdict(params.robot),
            "domain_randomization": asdict(params.domain_randomization),
            "disturbance": asdict(params.robust),
            "observation_noise": asdict(params.noise)}


def make_env(stage: str, rank: int, seed: int, stand_level: int,
             domain_randomization: bool = True, closed_chain_reset: bool = True,
             reset_audit_dir: str = '', wheel_torque_model: str = 'A_legacy',
             command_sampling: str = 'original'):
    def _init():
        if command_sampling != 'original':
            if reset_audit_dir:raise ValueError('reset-audit-dir is not supported with experimental sampling')
            from uz05.command_sampling import SamplingEnv
            env = SamplingEnv(stage=stage, seed=seed + rank, stand_level=stand_level,
                              command_sampling=command_sampling, sampling_rank=rank)
        elif reset_audit_dir:
            from uz05.experiment_audit import AuditedResetEnv
            env = AuditedResetEnv(stage=stage, seed=seed + rank, stand_level=stand_level,
                                  audit_path=Path(reset_audit_dir)/f'worker_{rank}.csv')
        else:
            env = UZ05Env(stage=stage, seed=seed + rank, stand_level=stand_level)
        env.command_sampling = command_sampling
        if wheel_torque_model == "B_rated":
            env.params.wheel.efficiency = 1.0
            env.actuators.wheel.params.efficiency = 1.0
        elif wheel_torque_model != "A_legacy":
            raise ValueError(wheel_torque_model)
        env.params.domain_randomization.enabled = bool(domain_randomization)
        env.params.robot.closed_chain_reset = bool(closed_chain_reset)
        env.reset(seed=seed + rank)
        return env
    return _init


def write_contract(path: str | Path, stage: str, total_updates: int,
                   version: str = "", domain_randomization: bool = True,
                   closed_chain_reset: bool = True, wheel_torque_model: str = "A_legacy",
                   command_sampling: str = 'original', initialization_provenance: dict | None = None) -> Path:
    """写出可审阅的观测/动作接口契约。"""
    path = Path(path)
    if path.suffix == ".zip":
        path = path.with_suffix("")
    contract = {
        "contract_version": CONTRACT_VERSION,
        "stage": stage,
        "version": version,
        "checkpoint_root": "checkpoints",
        "stage_order": list(CAPABILITY_ORDER),
        "model_xml_sha256": model_xml_sha256(),
        "model_xml_path": str(MODEL_XML),
        "command_sampling": sampling_contract(command_sampling),
        "actuator_config": actuator_contract(stage, wheel_torque_model),
        "observation_reference": observation_reference(stage),
        "environment_config": environment_contract(stage, domain_randomization, closed_chain_reset),
        "precision_objective": asdict(precision_rewards_for_stage(STAGE_BY_NAME[stage], EnvParams())) if STAGE_BY_NAME[stage].precision_control else None,
        "observation_shape": OBS_DIM,
        "actor_observation_shape": ACTOR_OBS_DIM,
        "actor_frame_shape": ACTOR_FRAME_DIM,
        "actor_frame_count": 1,
        "history_stacking": False,
        "critic_privileged_shape": PRIV_OBS_DIM,
        "actor_blocks": [{"name": name, "width": width,
                           "scale": OBS_SCALE[name], "source": OBS_SOURCE[name]}
                         for name, width in ACTOR_OBS_BLOCKS],
        "critic_privileged_blocks": [{"name": name, "width": width,
                                      "scale": OBS_SCALE[name], "source": OBS_SOURCE[name]}
                                      for name, width in PRIV_OBS_BLOCKS],
        "observation_scale": OBS_SCALE,
        "actor_hidden_simulator_label_leaks": [],
        "policy_assistance": "none; PPO actions only",
        "domain_randomization": bool(domain_randomization),
        "actor_simulated_sensor_proxies": [
            "base_lin_vel", "base_height_rel", "base_pos_rel", "terrain_preview"
        ],
        "action_shape": ACTION_DIM,
        "action_blocks": [
            {"name": name, "width": width, "scale": scale} for name, width, scale in (
                action_spec_for_stage(stage)
            )
        ],
        "stage_config": asdict(STAGE_BY_NAME[stage]),
        "interface_frozen": True,
        "note": (
            "观测与动作张量形状在所有 capability 阶段固定；接口版本、MJCF 哈希和执行器参数都相同的 "
            "checkpoint 才可直接继承。actor 不接收仿真任务标签；速度、位移、高度和前向地形值仍是带噪"
            "仿真代理，部署前需要用传感器状态估计替换并标定。"
        ),
        "total_updates": int(total_updates),
    }
    if initialization_provenance is not None:
        contract['initialization_provenance'] = initialization_provenance
    target = Path(f"{path}_contract.json")
    target.write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n")
    return target


def require_compatible_contract(checkpoint: Path, stage: str = "stand",
                                wheel_torque_model: str = "A_legacy",
                                adapt_wheel_torque_model: bool = False) -> None:
    """拒绝缺少契约、接口不同或训练模型文件不同的 checkpoint。"""
    contract_path = Path(f"{checkpoint}_contract.json")
    if not contract_path.is_file():
        raise SystemExit(
            f"checkpoint 缺少接口/模型契约 {contract_path}；"
            "请从当前接口和 MJCF 重新训练，或使用带匹配契约的 checkpoint。"
        )
    try:
        contract = json.loads(contract_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"无法读取 checkpoint 接口契约 {contract_path}: {error}")
    expected_reference = observation_reference(stage)
    if contract.get("observation_reference", "episode_initial") != expected_reference:
        raise SystemExit("checkpoint 的站位/航向参考与当前模式不同；请从头训练，或使用相同观测参考的 checkpoint 续训。")
    found = contract.get("contract_version")
    if found != CONTRACT_VERSION:
        raise SystemExit(
            f"checkpoint 接口为 {found!r}，当前代码要求 {CONTRACT_VERSION!r}: {checkpoint}\n"
            "观测接口/执行器模型版本已改变；请使用 v5 checkpoint，或去掉 --checkpoint 从头训练。"
        )
    expected = {
        "model_xml_sha256": model_xml_sha256(),
        "actuator_config": actuator_contract(stage, wheel_torque_model),
        "observation_shape": OBS_DIM,
        "actor_observation_shape": ACTOR_OBS_DIM,
        "critic_privileged_shape": PRIV_OBS_DIM,
        "action_shape": ACTION_DIM,
    }
    if adapt_wheel_torque_model:
        alternate = "A_legacy" if wheel_torque_model == "B_rated" else "B_rated"
        if contract.get("actuator_config") == actuator_contract(stage, alternate):
            expected["actuator_config"] = contract["actuator_config"]
    mismatches = {
        key: (contract.get(key), value)
        for key, value in expected.items()
        if contract.get(key) != value
    }
    if mismatches:
        raise SystemExit(
            f"checkpoint 契约与当前模型/张量不一致 {mismatches}: {checkpoint}\n"
            "请使用当前 v5 checkpoint，或去掉 --checkpoint 从头训练。"
        )


class TrainingPPO(PPO):
    """Report each iteration after its PPO update, with matching timing/metrics."""

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["_progress_logger"]

    def train(self) -> None:
        started = time.perf_counter()
        super().train()
        progress = getattr(self, "_progress_logger", None)
        if progress is not None:
            progress.log_update(time.perf_counter() - started)


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class ProgressLogger(BaseCallback):
    """Print aligned iteration summaries after learning; verbose adds robot state."""

    # Hide secondary terminal rows only; logger.record calls still retain all metrics.
    DETAIL_ROWS = frozenset({
        "Mean entropy loss", "Learning rate",
        "Tracking/wheel_target_delta_rms",
        "Reward/task_mean", "Reward/hold_cost_mean", "Reward/control_cost_mean",
        "Support/leg_drop_difference_rms_m", "Robust/push_active_fraction",
        "Control/effort_change_rms", "Posture/pitch_mean_deg",
        "Reward/pitch_cost_mean", "Reward/symmetry_cost_mean",
        "Posture/wheel_stagger_m", "Posture/wheel_offset_max_m",
    })

    # 核心量：(显示名, info 键, 格式)
    CORE = (
        ("roll_rad",       "roll",                 "+.4f"),
        ("pitch_rad",      "pitch",                "+.4f"),
        ("yaw_error_rad",  "yaw_error",            "+.4f"),
        ("roll_rate_rad_s", "roll_rate",            "+.3f"),
        ("pitch_rate_rad_s", "pitch_rate",          "+.3f"),
        ("yaw_rate_rad_s",   "yaw_rate",            "+.3f"),
        ("height_m",       "base_height",          ".4f"),
        ("vx_body_m_s",    "body_vx",              "+.4f"),
        ("command_vx_m_s", "command_vx",           "+.4f"),
        ("station_err_m",  "station_error",        "+.4f"),
        ("station_tail_m", "station_tail_abs",     ".4f"),
        ("within_5cm",     "station_within_5cm",   ".0f"),
        ("leg_length_L_m", "leg_length_left",      ".4f"),
        ("leg_length_R_m", "leg_length_right",     ".4f"),
        ("hip_pos_abs",    "_hip_pos_abs",         ".4f"),
        ("hip_vel_abs",    "_hip_vel_abs",         ".4f"),
        ("wheel_cur_L_A",  "wheel_current_left",   "+.2f"),
        ("wheel_cur_R_A",  "wheel_current_right",  "+.2f"),
        ("leg_torque_Nm",  "leg_torque_abs_mean",  ".2f"),
        ("airborne",       "airborne",             ".0f"),
        ("termination",    "termination_reason",   "s"),
    )

    # 完整量：仅在 --log-verbose 时输出
    EXTRA = (
        ("station_max_m",    "station_max_abs",      ".4f"),
        ("roll_peak_deg",    "roll_max_abs",         ".3f"),
        ("pitch_peak_deg",   "pitch_max_abs",        ".3f"),
        ("gyro_rms_rad_s",   "angular_velocity_rms", ".3f"),
        ("wheel_vel_L",      "wheel_vel_left",       "+.2f"),
        ("wheel_vel_R",      "wheel_vel_right",      "+.2f"),
        ("yaw_rad",          "yaw",                  "+.4f"),
        ("yaw_rate_rad_s",   "yaw_rate",             "+.4f"),
        ("vy_body_m_s",      "body_vy",              "+.4f"),
        ("leg_length_rate",  "leg_length_rate_mean", "+.4f"),
        ("hip_pos_0",        "hip_pos_0",            "+.4f"),
        ("hip_pos_1",        "hip_pos_1",            "+.4f"),
        ("hip_pos_2",        "hip_pos_2",            "+.4f"),
        ("hip_pos_3",        "hip_pos_3",            "+.4f"),
        ("hip_vel_0",        "hip_vel_0",            "+.4f"),
        ("hip_vel_1",        "hip_vel_1",            "+.4f"),
        ("hip_vel_2",        "hip_vel_2",            "+.4f"),
        ("hip_vel_3",        "hip_vel_3",            "+.4f"),
        ("wheel_target_L",   "wheel_target_left",    "+.3f"),
        ("wheel_target_R",   "wheel_target_right",   "+.3f"),
        ("wheel_torque_L",   "wheel_torque_left",    "+.4f"),
        ("wheel_torque_R",   "wheel_torque_right",   "+.4f"),
        ("wheel_force_L_N",  "wheel_force_left",     ".2f"),
        ("wheel_force_R_N",  "wheel_force_right",    ".2f"),
        ("body_contact",     "body_contact",         ".0f"),
        ("mode",             "mode",                 "s"),
    )

    REWARD_TOP = 3        # 只打印贡献最大的几项

    def __init__(self, stage, save_prefix, save_interval, version="",
                 verbose=False, log_interval=1, adaptive_lr=True, total_updates=0,
                 tilt_limit_deg=8.6):
        super().__init__()
        self.stage = stage
        self.save_prefix = save_prefix
        self.save_interval = max(1, int(save_interval))
        self.version = version
        self.verbose = verbose
        self.log_interval = max(1, int(log_interval))
        self.total_updates = max(1, int(total_updates or 1))
        self.tilt_limit_deg = float(tilt_limit_deg)
        self.term_counts: dict[str, int] = {}
        self.last_info: dict = {}
        self._start = 0.0
        # 验收统计：滚动窗口内的 episode 结果
        #   (是否活满整集, 末段漂移均值, 最大漂移, 最大偏航, 最大倾角, 角速度 RMS)
        self.episodes: deque = deque(maxlen=50)
        self.launch_reached: deque = deque(maxlen=50)
        # RSL-RL 自适应学习率（对齐开源 desired_kl=0.01）
        self.desired_kl = 0.01
        self.lr_min, self.lr_max = 1.0e-5, 2.0e-4
        self.adaptive_lr = bool(adaptive_lr)
        self.best_drift_cm = float("nan")
        self.reward_window = []
        self.control_window = []
        self.tracking_window = []
        self.wheel_target_deltas = []
        self._wheel_targets = {}
        self._iteration = 0
        self._collection_seconds = 0.0

    def _on_training_start(self) -> None:
        import time
        self._start = time.perf_counter()
        self.model._progress_logger = self

    def _on_rollout_start(self) -> None:
        self._rollout_start = time.perf_counter()
        self.reward_window.clear()
        self.control_window.clear()
        self.tracking_window.clear()
        self.wheel_target_deltas.clear()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos") or []
        dones = self.locals.get("dones")
        for index, info in enumerate(infos):
            reason = info.get("termination_reason", "running")
            if reason not in ("running", "time_limit"):
                self.term_counts[reason] = self.term_counts.get(reason, 0) + 1
            self.last_info = info
            if STAGE_BY_NAME[self.stage].simple_reward:
                t = info['reward_terms']
                for key,value in t.items():
                    self.logger.record_mean(f'Reward/{key}_mean',value)
            if STAGE_BY_NAME[self.stage].rebalanced_control and not STAGE_BY_NAME[self.stage].simple_reward:
                t = info.get("reward_terms", {})
                self.reward_window.append([
                    sum(t.get(k, 0.0) for k in ("upright", "pitch_alignment", "track_vx", "track_vx_coarse", "track_yaw", "track_height")),
                    sum(t.get(k, 0.0) for k in ("hold_position", "hold_heading")),
                    sum(t.get(k, 0.0) for k in ("action_rate", "wheel_target_change", "joint_target_change", "wheel_acceleration", "pitch_rate", "motion_vertical_velocity", "leg_common_velocity", "effort", "effort_change", "saturation")),
                ])
            self.control_window.append({key: info.get(key, 0.0) for key in (
                "wheel_target_change_sq", "joint_target_change_sq", "effort_change_sq",
                "joint_saturation_fraction", "wheel_saturation_fraction", "zero_command",
                "hold_planar_error_m", "hold_heading_error_rad", "leg_drop_difference_m",
                "leg_differential_velocity_m_s", "wheel_dual_support", "push_force_n",
                "pitch", "wheel_acceleration_sq")})
            self.control_window[-1]["pitch_cost"] = info.get("reward_terms", {}).get("pitch_error", 0.0)
            if STAGE_BY_NAME[self.stage].transition_tracking:
                self.control_window[-1].update({
                    "tracking_deficit_cost": sum(info["reward_terms"].get(k, 0.0) for k in ("height_shortfall", "vx_error_dense")),
                    "height_shortfall_sq": max(0.0, -info["tracking_height_error"])**2,
                    "pitch_rate_sq": info["pitch_rate"]**2,
                })
            if STAGE_BY_NAME[self.stage].straight_tracking:
                self.control_window[-1].update({
                    "straight_heading_active": info["straight_heading_active"],
                    "heading_error_sq": info["yaw_error"]**2,
                    "straight_tracking_cost": sum(info["reward_terms"].get(k, 0.0) for k in ("straight_heading", "yaw_error_dense")),
                })
            if STAGE_BY_NAME[self.stage].coordinated_tracking:
                self.control_window[-1].update({
                    "tracking_pose_quality": info["tracking_pose_quality"],
                    "coordinated_pose_cost": sum(info["reward_terms"].get(k,0.0) for k in ("tracking_pose_cost", "height_error_dense")),
                    "command_vx": info["command_vx"], "command_yaw": info["command_yaw"],
                    "height_error": info["tracking_height_error"], "vx_error": info["tracking_vx_error"],
                })
            if STAGE_BY_NAME[self.stage].compact_support:
                self.control_window[-1].update({key:info[key] for key in (
                    "leg_forward_reach_left_m", "leg_forward_reach_right_m",
                    "leg_backward_reach_left_m", "leg_backward_reach_right_m")})
                self.control_window[-1]["leg_reach_cost"] = sum(info["reward_terms"].get(k,0.0) for k in ("leg_forward_reach","leg_backward_reach"))
            if STAGE_BY_NAME[self.stage].responsive_tracking:
                self.control_window[-1]["moving_start"] = abs(info["initial_rolling_vx"]) > .1
            if STAGE_BY_NAME[self.stage].symmetric_support:
                self.control_window[-1].update({key: info[key] for key in (
                    "roll", "leg_length_left", "leg_length_right",
                    "wheel_fore_aft_left_m", "wheel_fore_aft_right_m")})
                self.control_window[-1]["symmetry_cost"] = sum(info.get("reward_terms", {}).get(k, 0.0)
                    for k in ("leg_length_difference", "fore_aft_difference", "roll_error", "leg_drop_difference"))
            if "wheel_target_left" in info and "wheel_target_right" in info:
                target = np.array([info["wheel_target_left"], info["wheel_target_right"]])
                previous = self._wheel_targets.get(index)
                if previous is not None:
                    self.wheel_target_deltas.append(float(np.square(target - previous).mean()))
                self._wheel_targets[index] = target
                if dones is not None and bool(dones[index]):
                    self._wheel_targets.pop(index, None)
            if STAGE_BY_NAME[self.stage].continuous_position:
                self.tracking_window.append([
                    info["tracking_vx_error"], info["tracking_yaw_error"], info["tracking_height_error"]
                ])
            # 只在该 episode 真正结束时记录一次
            if dones is not None and index < len(dones) and bool(dones[index]):
                if self.stage in STARTUP_STAGES:
                    launch_step = round(3. / actuator_contract(self.stage)['control_dt_s'])
                    self.launch_reached.append(info.get('episode_steps', 0) > launch_step)
                survived = reason == "time_limit"
                tail = float(info.get("station_tail_mean", float("nan")))
                self.episodes.append((
                    survived,
                    tail,
                    float(info.get("station_max_abs", float("nan"))),
                    float(info.get("yaw_max_abs", float("nan"))),
                    max(float(info.get("roll_max_abs", float("nan"))),
                        float(info.get("pitch_max_abs", float("nan")))),
                    float(info.get("angular_velocity_rms", float("nan"))),
                ))
        return True

    def _adapt_learning_rate(self) -> None:
        """RSL-RL schedule="adaptive"：KL 偏大降 lr，偏小升 lr。"""
        if not self.adaptive_lr:
            return
        kl = self.logger.name_to_value.get("train/approx_kl")
        if kl is None or not np.isfinite(kl):
            return
        lr = float(self.model.learning_rate)
        if kl > 2.0 * self.desired_kl:
            lr = max(lr / 1.5, self.lr_min)
        elif kl < 0.5 * self.desired_kl:
            lr = min(lr * 1.5, self.lr_max)
        else:
            return
        self.model.learning_rate = lr
        for group in self.model.policy.optimizer.param_groups:
            group["lr"] = lr

    def _on_rollout_end(self) -> None:
        self._collection_seconds = time.perf_counter() - self._rollout_start
        self._adapt_learning_rate()

    def log_update(self, learning_seconds: float) -> None:
        self._iteration += 1
        update = self._iteration
        if update % self.log_interval == 0 or update == self.total_updates:
            elapsed = time.perf_counter() - self._start
            iteration_seconds = self._collection_seconds + learning_seconds
            steps = self.model.n_steps * self.training_env.num_envs
            rows = []

            def row(label, value, fmt=".4f"):
                if self.verbose or label not in self.DETAIL_ROWS:
                    rows.append((label, format(float(value), fmt)))

            values = self.logger.name_to_value
            for label, key in (
                ("Mean action noise std", "train/std"),
                ("Mean surrogate loss", "train/policy_gradient_loss"),
                ("Mean value_function loss", "train/value_loss"),
                ("Mean entropy loss", "train/entropy_loss"),
                ("Mean kl loss", "train/approx_kl"),
                ("Mean ratio_clip_fraction", "train/clip_fraction"),
                ("Explained variance", "train/explained_variance"),
                ("Learning rate", "train/learning_rate"),
            ):
                value = values.get(key)
                if value is not None:
                    row(label, value, ".6f" if key.endswith("learning_rate") else ".4f")
            episodes = list(self.model.ep_info_buffer)
            if episodes:
                row("Mean reward", np.mean([e["r"] for e in episodes]), ".2f")
                row("Mean episode length", np.mean([e["l"] for e in episodes]), ".2f")
            if self.tracking_window:
                errors = np.asarray(self.tracking_window)
                # Absolute errors avoid hiding failures through sign cancellation.
                for name, value in zip(("vx", "yaw", "height"), np.abs(errors).mean(axis=0)):
                    key = f"Tracking/tracking_{name}_error"
                    row(key, value)
                    self.logger.record(key, value)
                for name, value in zip(("vx", "yaw", "height"), np.sqrt(np.square(errors).mean(axis=0))):
                    self.logger.record(f"Tracking/{name}_rmse", value)
            if self.wheel_target_deltas:
                value = np.sqrt(np.mean(self.wheel_target_deltas))
                row("Tracking/wheel_target_delta_rms", value)
                self.logger.record("Tracking/wheel_target_delta_rms", value)
            if self.reward_window:
                task, hold, control = np.mean(self.reward_window, axis=0)
                for label, value in (("Reward/task_mean", task), ("Reward/hold_cost_mean", hold),
                                     ("Reward/control_cost_mean", control),
                                     ("Reward/control_to_task_ratio", abs(control) / max(task, 1e-9))):
                    row(label, value)
                    self.logger.record(label, value)
            if STAGE_BY_NAME[self.stage].robust_support and self.control_window:
                for label, value in (
                    ("Support/leg_drop_difference_rms_m", np.sqrt(np.mean([v["leg_drop_difference_m"]**2 for v in self.control_window]))),
                    ("Support/leg_differential_velocity_rms", np.sqrt(np.mean([v["leg_differential_velocity_m_s"]**2 for v in self.control_window]))),
                    ("Support/dual_wheel_fraction", np.mean([v["wheel_dual_support"] for v in self.control_window])),
                    ("Robust/push_active_fraction", np.mean([v["push_force_n"] > 0 for v in self.control_window])),
                ):
                    row(label, value)
                    self.logger.record(label, value)
            if STAGE_BY_NAME[self.stage].precision_control and self.control_window:
                for label, key, rms in (
                    ("Control/wheel_target_change_rms", "wheel_target_change_sq", True),
                    ("Control/joint_target_change_rms", "joint_target_change_sq", True),
                    ("Control/effort_change_rms", "effort_change_sq", True),
                    ("Control/joint_saturation_fraction", "joint_saturation_fraction", False),
                    ("Control/wheel_saturation_fraction", "wheel_saturation_fraction", False),
                ):
                    value = float(np.mean([v[key] for v in self.control_window]))
                    value = np.sqrt(value) if rms else value
                    row(label, value)
                    self.logger.record(label, value)
                if STAGE_BY_NAME[self.stage].pitch_control:
                    pitches = np.array([v["pitch"] for v in self.control_window])
                    for label, value in (
                        ("Posture/pitch_mean_deg", np.degrees(pitches.mean())),
                        ("Posture/pitch_rms_deg", np.degrees(np.sqrt(np.square(pitches).mean()))),
                        ("Posture/pitch_within_5deg_fraction", np.mean(np.abs(pitches) <= np.radians(5))),
                        ("Reward/pitch_cost_mean", np.mean([v["pitch_cost"] for v in self.control_window])),
                        ("Control/wheel_acceleration_rms_rad_s2", np.sqrt(np.mean([v["wheel_acceleration_sq"] for v in self.control_window]))),
                    ):
                        row(label, value)
                        self.logger.record(label, value)
                if STAGE_BY_NAME[self.stage].symmetric_support:
                    for label, value in (
                        ("Support/leg_length_difference_rms_m", np.sqrt(np.mean([(v["leg_length_left"]-v["leg_length_right"])**2 for v in self.control_window]))),
                        ("Support/fore_aft_difference_rms_m", np.sqrt(np.mean([(v["wheel_fore_aft_left_m"]-v["wheel_fore_aft_right_m"])**2 for v in self.control_window]))),
                        ("Posture/roll_rms_deg", np.degrees(np.sqrt(np.mean([v["roll"]**2 for v in self.control_window])))),
                        ("Posture/roll_within_2deg_fraction", np.mean([abs(v["roll"]) <= np.radians(2) for v in self.control_window])),
                        ("Reward/symmetry_cost_mean", np.mean([v["symmetry_cost"] for v in self.control_window])),
                    ):
                        row(label, value)
                        self.logger.record(label, value)
                zero = [v for v in self.control_window if v["zero_command"]]
                if STAGE_BY_NAME[self.stage].transition_tracking:
                    # Detailed additions go to TensorBoard without growing terminal rows.
                    self.logger.record("Reward/tracking_deficit_cost_mean", np.mean([v["tracking_deficit_cost"] for v in self.control_window]))
                    self.logger.record("Tracking/height_shortfall_rms_m", np.sqrt(np.mean([v["height_shortfall_sq"] for v in self.control_window])))
                    self.logger.record("Posture/pitch_rate_rms_rad_s", np.sqrt(np.mean([v["pitch_rate_sq"] for v in self.control_window])))
                if STAGE_BY_NAME[self.stage].straight_tracking:
                    straight = [v for v in self.control_window if v["straight_heading_active"]]
                    if straight:
                        self.logger.record("Tracking/straight_heading_rms_deg", np.degrees(np.sqrt(np.mean([v["heading_error_sq"] for v in straight]))))
                    self.logger.record("Reward/straight_tracking_cost_mean", np.mean([v["straight_tracking_cost"] for v in self.control_window]))
                if STAGE_BY_NAME[self.stage].coordinated_tracking:
                    motion_fraction = np.mean([abs(v['command_vx']) > .01 for v in self.control_window])
                    row('Training/nonzero_vx_frame_fraction', motion_fraction)
                    self.logger.record('Training/nonzero_vx_frame_fraction', motion_fraction)
                    self.logger.record("Tracking/pose_quality", np.mean([v["tracking_pose_quality"] for v in self.control_window]))
                    self.logger.record("Reward/coordinated_pose_cost_mean", np.mean([v["coordinated_pose_cost"] for v in self.control_window]))
                    for direction, sign in (("forward",1), ("reverse",-1)):
                        samples = [v for v in self.control_window if sign*v["command_vx"] > .1 and abs(v["command_yaw"]) <= .01]
                        if samples:
                            self.logger.record(f"Tracking/{direction}_height_bias_m", np.mean([v["height_error"] for v in samples]))
                            self.logger.record(f"Posture/{direction}_pitch_mean_deg", np.degrees(np.mean([v["pitch"] for v in samples])))
                            self.logger.record(f"Tracking/{direction}_vx_mae", np.mean([abs(v["vx_error"]) for v in samples]))
                if STAGE_BY_NAME[self.stage].compact_support:
                    for direction in ('forward','backward'):
                        reaches = [v[f'leg_{direction}_reach_{side}_m'] for v in self.control_window for side in ('left','right')]
                        self.logger.record(f"Support/leg_{direction}_reach_peak_m", max(reaches))
                    self.logger.record("Reward/leg_reach_cost_mean", np.mean([v["leg_reach_cost"] for v in self.control_window]))
                if STAGE_BY_NAME[self.stage].responsive_tracking:
                    self.logger.record("Robust/moving_start_frame_fraction", np.mean([v["moving_start"] for v in self.control_window]))
                if zero:
                    for label, value in (
                        ("Hold/position_mae_m", np.mean([v["hold_planar_error_m"] for v in zero])),
                        ("Hold/position_peak_m", max(v["hold_planar_error_m"] for v in zero)),
                        ("Hold/heading_mae_rad", np.mean([v["hold_heading_error_rad"] for v in zero])),
                        ("Hold/within_5cm_fraction", np.mean([v["hold_planar_error_m"] <= .05 for v in zero])),
                    ):
                        row(label, value)
                        self.logger.record(label, value)
            if STAGE_BY_NAME[self.stage].continuous_position:
                for key in ("wheel_stagger_m", "wheel_offset_max_m"):
                    if key in self.last_info:
                        row(f"Posture/{key}", self.last_info[key])
                        self.logger.record(f"Posture/{key}", self.last_info[key])
            if self.episodes:
                rate = np.mean([e[0] for e in self.episodes])
                row("Episode/survive_rate", rate)
                self.logger.record("Episode/survive_rate", rate)
                if self.launch_reached:
                    reached = np.mean(self.launch_reached)
                    row('Training/reached_launch_episode_fraction', reached)
                    self.logger.record('Training/reached_launch_episode_fraction', reached)
            else:
                rows.append(("Episode/survive_rate", "n/a (no completed episodes)"))

            if self.stage == "stand" and self.episodes:
                completed = [e for e in self.episodes if e[0]]
                if completed:
                    peaks = np.max(np.asarray(completed)[:, 2:6], axis=0)
                    drift, yaw, tilt, gyro = peaks
                    row("Episode/drift_peak_cm", drift * 100)
                    row("Episode/yaw_peak_deg", np.degrees(yaw))
                    row("Episode/tilt_peak_deg", np.degrees(tilt))
                    row("Episode/gyro_rms_rad_s", gyro)
                    accepted = (rate >= 0.95 and np.all(np.isfinite(peaks))
                                and drift <= 0.05 and np.degrees(yaw) <= 5
                                and np.degrees(tilt) <= self.tilt_limit_deg and gyro <= 0.20)
                else:
                    accepted = False
                row("Episode/accept_stand", int(accepted), ".0f")
            if self.verbose:
                info = dict(self.last_info)
                for label, key, fmt in self.CORE + self.EXTRA:
                    if key in info:
                        value = info[key]
                        rows.append((label, str(value) if fmt == "s" else format(float(value), fmt)))
                for key, value in (info.get("reward_terms") or {}).items():
                    row(f"Reward/{key}", value)

            print("#" * 80, flush=True)
            print(f" Learning iteration {update}/{self.total_updates} ".center(80), flush=True)
            print(f"{'Computation':>40}: {steps / max(iteration_seconds, 1e-9):.0f} steps/s "
                  f"(collection: {self._collection_seconds:.3f}s, learning {learning_seconds:.3f}s)", flush=True)
            for label, value in rows:
                print(f"{label:>40}: {value}", flush=True)
            print("-" * 80, flush=True)
            print(f"{'Total timesteps':>40}: {self.num_timesteps}", flush=True)
            print(f"{'Iteration time':>40}: {iteration_seconds:.2f}s", flush=True)
            print(f"{'Time elapsed':>40}: {format_duration(elapsed)}", flush=True)
            eta = elapsed / update * max(0, self.total_updates - update)
            print(f"{'ETA':>40}: {format_duration(eta)}\n", flush=True)
        if self.save_prefix and update % self.save_interval == 0:
            self._save_checkpoint(update)

    def _save_checkpoint(self, update: int) -> None:
        path = f"{self.save_prefix}_iter_{update}"
        self.model.save(path)
        self.training_env.save(f"{path}_vecnormalize.pkl")
        write_contract(path, self.stage, update, self.version,
                       domain_randomization=self.training_env.get_attr("params")[0].domain_randomization.enabled,
                       closed_chain_reset=self.training_env.get_attr("params")[0].robot.closed_chain_reset,
                       wheel_torque_model="B_rated" if self.training_env.get_attr("params")[0].wheel.efficiency == 1.0 else "A_legacy",
                       command_sampling=self.training_env.get_attr('command_sampling')[0],
                       initialization_provenance=getattr(self.model,'initialization_provenance',None))

    def _on_training_end(self) -> None:
        if self.term_counts:
            total = sum(self.term_counts.values())
            summary = "   ".join(f"{k}={v}({100.0 * v / total:.0f}%)"
                                 for k, v in sorted(self.term_counts.items(), key=lambda kv: -kv[1]))
            print(f"termination_summary: {summary}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="UZ-05 分阶段强化学习训练")
    parser.add_argument("--stage", default="stand", choices=list(CAPABILITY_ORDER))
    parser.add_argument("--checkpoint", default="", help="从该 checkpoint 继承（任意阶段均可）")
    parser.add_argument("--wheel-torque-model", choices=("A_legacy", "B_rated"), default="A_legacy",
                        help="Explicit wheel output map; B is an uncalibrated rated-point approximation")
    parser.add_argument("--adapt-wheel-torque-model", action="store_true",
                        help="Explicitly allow loading the alternate A/B wheel map for adaptation only")
    parser.add_argument('--command-sampling', choices=SAMPLING_NAMES, default='original',
                        help='Opt-in command/reset sampling experiment; physics/reward/controller unchanged')
    parser.add_argument("--updates", type=int, default=1500)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--vec-env", choices=("dummy", "subproc"), default="subproc")
    parser.add_argument("--disable-adaptive-lr", action="store_true")
    parser.add_argument("--legacy-reset", action="store_true",
                        help="Use original inconsistent reset for paired A/B experiments only")
    parser.add_argument("--initial-state-audit", default='', help="Optional pre-learn state fingerprints JSON")
    parser.add_argument("--reset-audit-dir", default='', help="Optional per-worker initial-state CSVs")
    parser.add_argument("--disable-domain-randomization", action="store_true",
                        help="train the specified nominal MJCF dynamics without episode parameter randomization")
    parser.add_argument("--version", default="",
                        help="本次训练的版本名；checkpoint 存到 checkpoints/<version>/。"
                             "默认 <stage>_v4")
    parser.add_argument("--out", default="", help="显式指定 checkpoint 前缀（覆盖 --version）")
    parser.add_argument("--tensorboard-log", default="")
    parser.add_argument("--save-interval", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)   # 对齐开源
    parser.add_argument("--entropy-coef", type=float, default=0.005,
                        help="PPO entropy coefficient; reduce for low-noise zero-command refinement")
    parser.add_argument("--action-std", type=float, default=None,
                        help="initial/resume policy std; lower values reduce exploration noise")
    parser.add_argument("--wheel-action-std", type=float, default=None,
                        help="Explicit initial wheel exploration std, overriding the stage default; PPO learns it thereafter")
    parser.add_argument("--preserve-action-std", action="store_true",
                        help="续训保留 checkpoint 的六维探索标准差，不重新初始化")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vecnormalize-path", default="",
                        help="显式指定奖励归一化统计量；默认自动找 <checkpoint>_vecnormalize.pkl")
    parser.add_argument("--reset-vecnormalize", action="store_true",
                        help="不继承旧奖励归一化统计；奖励函数变化后续训时使用")
    parser.add_argument("--log-verbose", action="store_true",
                        help="展开完整迭代指标及机器人状态量（默认省略次要和重复指标）")
    parser.add_argument("--log-interval", type=int, default=1,
                        help="每 N 次迭代打印一次")
    parser.add_argument("--stand-level", type=int, default=0,
                        help="站立分级 0=S0_balance / 1=S1_tighten / 2=S2_accept(±5cm/±5° 验收)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.wheel_action_std is not None:
        if not np.isfinite(args.wheel_action_std) or args.wheel_action_std <= 0:
            parser.error('--wheel-action-std must be finite and positive')
        if args.preserve_action_std:
            parser.error('--wheel-action-std conflicts with --preserve-action-std')
    if (args.stage in STARTUP_STAGES) != (args.command_sampling in STARTUP_SAMPLERS):
        parser.error('startup stages require --command-sampling startup_balanced or startup_low')
    if args.preserve_action_std and not args.checkpoint:
        parser.error("--preserve-action-std requires --checkpoint")
    training_stage = STAGE_BY_NAME[args.stage]

    if STAGE_BY_NAME[args.stage].continuous_position:
        args.disable_adaptive_lr = True
        # This mode always starts with the full fixed command distribution.
        print("[joint_tracking] constant learning rate; no curriculum/teacher/residual/action guidance", flush=True)
    if args.updates < 1 or args.rollout_steps < 2 or args.num_envs < 1:
        parser.error("updates/num-envs must be positive and rollout-steps >= 2")
    if args.action_std is not None and args.action_std <= 0:
        parser.error("--action-std must be positive")
    if args.checkpoint and STAGE_BY_NAME[args.stage].pitch_control:
        source = Path(args.checkpoint).expanduser().with_suffix("")
        source_contract = Path(f"{source}_contract.json")
        if (source_contract.is_file() and not args.reset_vecnormalize
                and not can_preserve_reward_statistics(json.loads(source_contract.read_text()), args.stage)):
            parser.error(f"转入 {args.stage} 时奖励定义变化，必须加 --reset-vecnormalize；默认建议从头训练。")

    # ---- 资源护栏：subproc 每个环境一个进程（各带一份 MuJoCo 模型），
    # 超过 核数/2 就会把整机拖死（32 环境在 16 核 /14G 机器上实测直接卡死）。
    # 确实想要更多时用 ALLOW_OVERSUBSCRIBE=1 显式放行。
    cores = os.cpu_count() or 4
    if args.vec_env == "subproc" and not os.environ.get("ALLOW_OVERSUBSCRIBE"):
        safe = max(1, min(cores // 2, 8))
        if args.num_envs > safe:
            print(f"[warn] --num-envs {args.num_envs} -> {safe}：{cores} 核机器上 subproc "
                  f"环境建议 ≤ {safe}（想强行超配设 ALLOW_OVERSUBSCRIBE=1）", flush=True)
            args.num_envs = safe
    print(f"[resource] num_envs={args.num_envs} cores={cores} "
          f"torch_threads={torch.get_num_threads()}", flush=True)

    # 默认输出为**绝对路径**，始终落在仓库的 checkpoints/<version>/ 下，
    # 与运行时所在目录无关。每个 version 自动新建一个文件夹。
    from uz05.spec import DEFAULT_MODEL_XML, ORIGINAL_MODEL_XML
    model_suffix = ("_armature001" if MODEL_XML == DEFAULT_MODEL_XML.resolve() else
                    "_model20261104" if MODEL_XML == ORIGINAL_MODEL_XML.resolve() else "")
    version = args.version or f"{args.stage}_v4{model_suffix}"
    checkpoint_root = (REPO_ROOT / "checkpoints").resolve()
    out_path = (Path(args.out).expanduser().resolve() if args.out else
                checkpoint_root / version / "checkpoint")
    if not out_path.is_relative_to(checkpoint_root):
        raise SystemExit(f"checkpoint 输出必须位于 {checkpoint_root} 下: {out_path}")
    out = str(out_path)
    tb = args.tensorboard_log or str(REPO_ROOT / "runs" / version)
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[interface] stage={args.stage}  obs={OBS_DIM}  actor={ACTOR_OBS_DIM}  "
        f"critic_priv={PRIV_OBS_DIM}  action={ACTION_DIM}",
        flush=True,
    )
    print(f"[model] {MODEL_XML} sha256={model_xml_sha256()}", flush=True)

    # 续训时复用上一阶段的奖励归一化统计量（否则 critic 目标尺度会突变）
    stats_path = Path(args.vecnormalize_path).expanduser() if args.vecnormalize_path else None
    if stats_path is None and args.checkpoint and not args.reset_vecnormalize:
        base = Path(args.checkpoint).expanduser()
        candidate = Path(f"{base.with_suffix('') if base.suffix == '.zip' else base}_vecnormalize.pkl")
        if candidate.exists():
            stats_path = candidate

    env_cls = SubprocVecEnv if args.num_envs > 1 and args.vec_env == "subproc" else DummyVecEnv
    from uz05.spec import STAND_LEVELS
    level = next((lv for lv in STAND_LEVELS if lv.level == args.stand_level), STAND_LEVELS[0])
    env = VecMonitor(env_cls([
        make_env(args.stage, r, args.seed, level.level,
                 domain_randomization=not args.disable_domain_randomization,
                 closed_chain_reset=not args.legacy_reset, reset_audit_dir=args.reset_audit_dir,
                 wheel_torque_model=args.wheel_torque_model, command_sampling=args.command_sampling)
        for r in range(args.num_envs)
    ]))
    if args.reset_vecnormalize:
        stats_path = None
    if stats_path is not None:
        if not stats_path.exists():
            raise FileNotFoundError(f"VecNormalize 统计量不存在: {stats_path}")
        env = VecNormalize.load(str(stats_path), env)
        env.training = True
        env.norm_reward = True
        print(f"vecnormalize_loaded: {stats_path}", flush=True)
    else:
        env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0,
                           gamma=training_stage.ppo_gamma)
    env.gamma = training_stage.ppo_gamma

    if args.checkpoint:
        checkpoint = Path(args.checkpoint).expanduser()
        if checkpoint.suffix == ".zip":
            checkpoint = checkpoint.with_suffix("")
        require_compatible_contract(checkpoint, args.stage, args.wheel_torque_model, args.adapt_wheel_torque_model)
        try:
            # Override before _setup_model(): load otherwise restores the source
            # seed, rollout buffer shape and learning-rate schedule as well.
            model = TrainingPPO.load(str(checkpoint), env=env, device=args.device,
                                    seed=args.seed, n_steps=args.rollout_steps,
                                    batch_size=args.batch_size, learning_rate=args.learning_rate,
                                    gamma=training_stage.ppo_gamma,
                                    gae_lambda=training_stage.ppo_gae_lambda,
                                    ent_coef=args.entropy_coef)
        except (ValueError, AssertionError, RuntimeError) as error:
            raise SystemExit(
                f"checkpoint 与当前接口不兼容（obs={OBS_DIM}，actor={ACTOR_OBS_DIM}，"
                f"action={ACTION_DIM}）：{error}\n"
                "请使用同接口版本的 checkpoint，或去掉 --checkpoint 从头训练；"
                "旧版 236/6、226/6 与 73 维 checkpoint 不支持自动迁移。"
            )
        if model.policy.log_std is not None and not args.preserve_action_std:
            import torch as _torch
            if args.action_std is None:
                model.policy.log_std.data.clamp_(-2.3, -0.9)   # std ∈ [0.10, 0.40]
            else:
                model.policy.log_std.data.fill_(float(np.log(max(args.action_std, 1.0e-3))))
        model.ent_coef = args.entropy_coef
        model.gamma = training_stage.ppo_gamma
        model.gae_lambda = training_stage.ppo_gae_lambda
        model.rollout_buffer.gamma = model.gamma
        model.rollout_buffer.gae_lambda = model.gae_lambda
        model.learning_rate = args.learning_rate
        for group in model.policy.optimizer.param_groups:
            group["lr"] = args.learning_rate
        print(f"[inherit] 已从 {checkpoint} 继承策略权重", flush=True)
    else:
        model = TrainingPPO(
            AsymmetricActorCriticPolicy,
            env,
            n_steps=args.rollout_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            n_epochs=5,              # 对齐开源 num_learning_epochs
            gamma=training_stage.ppo_gamma,
            gae_lambda=training_stage.ppo_gae_lambda,
            vf_coef=2.0,             # 对齐开源 value_loss_coef
            max_grad_norm=1.0,       # 对齐开源
            ent_coef=args.entropy_coef,
            policy_kwargs=dict(
                actor_dim=ACTOR_OBS_DIM,
                net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64]),
                activation_fn=nn.ELU,
                # 开源用 init_noise_std=1.0（σ=1.0）。本机实测 σ=1.0 会让策略一上来
                # 就用满幅腿动作把闭链机构改形，故取 σ≈0.37 起步；不使用控制辅助策略。
                log_std_init=(float(np.log(args.action_std)) if args.action_std else -1.0),
            ),
            tensorboard_log=tb,
            verbose=0,
            device=args.device,
            seed=args.seed,
        )

    # Loading also restores the source run's TensorBoard path; keep this run's
    # diagnostics with its own version rather than appending to the old run.
    model.tensorboard_log = tb
    wheel_std = (args.wheel_action_std if args.wheel_action_std is not None
                 else STAGE_BY_NAME[args.stage].wheel_action_std_init)
    if wheel_std is not None and model.policy.log_std is not None and not args.preserve_action_std:
        # Initial channel-specific exploration, learned freely thereafter.
        with torch.no_grad():
            model.policy.log_std[:2].fill_(float(np.log(wheel_std)))
        print(f"train_config_initial_wheel_action_std: {wheel_std}", flush=True)
    if args.preserve_action_std and model.policy.log_std is not None:
        print(f"train_config_preserved_action_std: {model.policy.log_std.detach().exp().cpu().tolist()}", flush=True)

    total_steps = args.updates * args.rollout_steps * args.num_envs
    print(f"train_config_stage: {args.stage}", flush=True)
    print(f"train_config_updates: {args.updates}", flush=True)
    print(f"train_config_total_steps: {total_steps}", flush=True)
    print(f"train_config_num_envs: {args.num_envs}", flush=True)
    print(f"train_config_learning_rate: {args.learning_rate}", flush=True)
    print(f"train_config_gamma: {model.gamma}  gae_lambda: {model.gae_lambda}", flush=True)
    if STAGE_BY_NAME[args.stage].continuous_position:
        print(f"train_config_commands: {asdict(STAGE_BY_NAME[args.stage])}", flush=True)
        print("train_config_actions: [wheel_L_speed, wheel_R_speed, joint_L1_pos, joint_L2_pos, joint_R1_pos, joint_R2_pos]", flush=True)
    else:
        print(f"train_config_stand_level: {level.name}", flush=True)
        print(f"train_config_station_deadband_m: {level.deadband}", flush=True)
        print(f"train_config_station_hard_limit_m: {level.hard_limit}", flush=True)
        print(f"train_config_yaw_hard_limit_deg: {np.degrees(level.yaw_hard_limit):.2f}", flush=True)
    print("train_config_policy_assistance: none (PPO actions only)", flush=True)
    print(f"train_config_domain_randomization: {not args.disable_domain_randomization}", flush=True)
    print("", flush=True)
    (Path(out).parent / "training_config.json").write_text(
        json.dumps({"arguments": vars(args), "stage": asdict(STAGE_BY_NAME[args.stage]),
                    "initialization_provenance": getattr(model,'initialization_provenance',None),
                    "command_sampling": sampling_contract(args.command_sampling),
                    "model_xml_path": str(MODEL_XML), "model_xml_sha256": model_xml_sha256(),
                    "actuators": actuator_contract(args.stage, args.wheel_torque_model),
                    "environment_config": environment_contract(args.stage, not args.disable_domain_randomization, not args.legacy_reset),
                    "precision_objective": asdict(precision_rewards_for_stage(STAGE_BY_NAME[args.stage], EnvParams())) if STAGE_BY_NAME[args.stage].precision_control else None}, indent=2, ensure_ascii=False) + "\n"
    )
    write_contract(out, args.stage, args.updates, version,
                   domain_randomization=not args.disable_domain_randomization,
                   closed_chain_reset=not args.legacy_reset, wheel_torque_model=args.wheel_torque_model,
                   command_sampling=args.command_sampling,
                   initialization_provenance=getattr(model,'initialization_provenance',None))
    model._total_timesteps = total_steps
    if args.initial_state_audit:
        from uz05.experiment_audit import initial_state_audit
        initial_state_audit(model, env, args.initial_state_audit)
    model.learn(
        total_timesteps=total_steps,
        callback=ProgressLogger(args.stage, out, args.save_interval, version,
                                args.log_verbose, args.log_interval,
                                adaptive_lr=not args.disable_adaptive_lr,
                                total_updates=args.updates,
                                tilt_limit_deg=float(np.degrees(level.tilt_limit))),
        log_interval=1,
    )
    model.save(out)
    env.save(f"{out}_vecnormalize.pkl")
    contract = write_contract(out, args.stage, args.updates, version,
                              domain_randomization=not args.disable_domain_randomization,
                              closed_chain_reset=not args.legacy_reset, wheel_torque_model=args.wheel_torque_model,
                              command_sampling=args.command_sampling,
                              initialization_provenance=getattr(model,'initialization_provenance',None))
    print(f"checkpoint: {out}.zip", flush=True)
    print(f"vecnormalize: {out}_vecnormalize.pkl", flush=True)
    print(f"contract: {contract}", flush=True)
    print(f"tensorboard: {tb}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
