"""Train the direct-target PPO policy for UZ-05.

The actor owns all six action channels.  The only transformations after PPO
are the physical leg PD and wheel-speed loops implemented by ``uz05.actuators``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import deque
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure as configure_logger
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from uz05.env import UZ05Env
from uz05.policy_compat import ensure_forward_positive_wheel_actions
from uz05.spec import (
    ACTOR_OBS_DIM,
    ACTION_DIM,
    EnvParams,
    MOTOR_CONTROL_RATE_HZ,
    MOTOR_TICKS_PER_POLICY,
    OBS_DIM,
    POLICY_RATE_HZ,
    REPO_ROOT,
    WHEEL_ACTION_FRAME,
    WHEEL_ANGULAR_TO_BODY_X,
)

torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))

CONTRACT_VERSION = "uz05_direct_ppo_mit_v4_forward_positive_wheels"


class AsymmetricMlpExtractor(nn.Module):
    """Actor uses sensor observations; critic may use the full observation."""

    def __init__(self, actor_dim, observation_dim, net_arch, activation_fn):
        super().__init__()
        if isinstance(net_arch, dict):
            pi_layers = list(net_arch.get("pi", [256, 128, 64]))
            vf_layers = list(net_arch.get("vf", [256, 128, 64]))
        else:
            pi_layers = vf_layers = list(net_arch or [256, 128, 64])
        self.actor_dim = int(actor_dim)
        self.policy_net = self._build(self.actor_dim, pi_layers, activation_fn)
        self.value_net = self._build(int(observation_dim), vf_layers, activation_fn)
        self.latent_dim_pi = pi_layers[-1]
        self.latent_dim_vf = vf_layers[-1]

    @staticmethod
    def _build(in_dim, layers, activation_fn):
        modules: list[nn.Module] = []
        last = int(in_dim)
        for width in layers:
            modules.extend((nn.Linear(last, int(width)), activation_fn()))
            last = int(width)
        return nn.Sequential(*modules)

    def forward(self, obs):
        return self.policy_net(obs[..., :self.actor_dim]), self.value_net(obs)

    def forward_actor(self, obs):
        return self.policy_net(obs[..., :self.actor_dim])

    def forward_critic(self, obs):
        return self.value_net(obs)


class AsymmetricActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, actor_dim=None, **kwargs):
        if actor_dim is None:
            raise ValueError("actor_dim is required")
        self._actor_dim = int(actor_dim)
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = AsymmetricMlpExtractor(
            self._actor_dim, self.features_dim, self.net_arch, self.activation_fn
        )


def make_env(stage: str, rank: int, seed: int, args: argparse.Namespace):
    def _init():
        return UZ05Env(
            stage=stage,
            seed=seed + rank,
            stand_level=args.stand_level,
            init_scale=args.init_scale_start,
            vx_range_override=args.vx_range,
            zero_command_prob_override=args.command_zero_prob,
            reverse_prob_override=args.command_reverse_prob,
            command_accel_limit_override=args.command_accel_limit,
            vx_reward_scale=args.vx_reward_scale_start,
            low_speed_reverse_reward_scale=args.low_speed_reverse_reward_scale,
            low_speed_wrong_direction_weight=args.low_speed_wrong_direction_weight,
            low_speed_wheel_differential_weight=args.low_speed_wheel_differential_weight,
            height_switch_steps=args.height_switch_steps,
            height_switch_prob=args.height_switch_prob,
            height_target_rate_m_s=args.height_target_rate,
            leg_length_reward_weight=args.leg_length_reward_weight,
            leg_length_progress_reward_weight=args.leg_length_progress_reward_weight,
            leg_length_rate_reward_weight=args.leg_length_rate_reward_weight,
            low_speed_stable_motion_weight=args.low_speed_stable_motion_weight,
            low_speed_posture_gate_gain=args.low_speed_posture_gate_gain,
            low_speed_track_sigma_relative=args.low_speed_track_sigma_relative,
            deployment_mode=args.deployment_mode,
            observation_delay_steps=args.observation_delay_steps,
            actuator_delay_steps=args.actuator_delay_steps,
            tilt_hold_steps=args.tilt_hold_steps,
            wheel_speed_kp=args.wheel_speed_kp,
            wheel_speed_ki=args.wheel_speed_ki,
            wheel_integral_limit=args.wheel_integral_limit,
            episode_steps_override=(
                args.horizon_stages[0] if args.horizon_curriculum else args.episode_steps
            ),
        )
    return _init


class TrainingCallback(BaseCallback):
    METRIC_KEYS = (
        "wheel_target_abs", "wheel_current_abs", "wheel_torque_abs_mean",
        "wheel_speed_abs", "wheel_target_skew_rad_s", "wheel_speed_skew_rad_s",
        "leg_target_abs", "leg_torque_abs_mean", "leg_length_error_mm",
        "pitch", "base_height", "body_vx_after", "action_abs",
    )

    def __init__(self, args, total_updates: int, verbose: int = 0):
        super().__init__(verbose)
        self.args = args
        self.total_updates = max(1, int(total_updates))
        self.episode_stats: deque = deque(maxlen=50)
        self.last_info: dict = {}
        self.term_counts: dict[str, int] = {}
        self.rollout_term_counts: dict[str, int] = {}
        self.metric_sums = {key: 0.0 for key in self.METRIC_KEYS}
        self.metric_count = 0
        self.best_score = 0.0
        self.best_update = 0
        self.best_metrics: dict[str, float] = {}
        self.stage_best_scores: dict[int, tuple[float, ...]] = {}
        self.best_eval_score: tuple[float, ...] | None = None
        self.update = 0
        self.start_steps = 0
        self.wall_start = 0.0
        self.last_rollout_time = 0.0
        self.curriculum_stage = (
            max(0, int(args.curriculum_start_stage) - 1)
            if args.velocity_curriculum else 0
        )
        self.curriculum_stage_start_update = 0
        self.curriculum_episodes: deque = deque(maxlen=max(1, int(args.curriculum_window_episodes)))
        self.curriculum_env_stats: list[dict[str, float]] = []
        self.eval_env: UZ05Env | None = None

    def _init_curriculum_env_stats(self) -> None:
        self.curriculum_env_stats = [
            {
                "track_error_sum": 0.0,
                "track_count": 0.0,
                "direction_correct": 0.0,
                "direction_count": 0.0,
                "contact_both_sum": 0.0,
                "step_count": 0.0,
                "motion_peak": 0.0,
                "station_peak": 0.0,
                "pitch_abs": [],
            }
            for _ in range(self.model.n_envs)
        ]

    def _on_training_start(self) -> None:
        self.start_steps = self.num_timesteps
        self.wall_start = time.perf_counter()
        self.last_rollout_time = self.wall_start
        self.eval_env = make_env(
            self.args.stage, self.args.num_envs + 10000, self.args.seed, self.args
        )()
        if self.args.velocity_curriculum:
            self._configure_velocity_curriculum_stage()
        elif self.args.horizon_curriculum:
            horizon = int(self.args.horizon_stages[self.curriculum_stage])
            self.training_env.env_method("set_episode_steps", horizon)
            self.eval_env.set_episode_steps(horizon)
        if self.args.velocity_curriculum or self.args.horizon_curriculum:
            self._init_curriculum_env_stats()

    def _zero_prob_for_stage(self, stage_number: int) -> float:
        if stage_number < self.args.curriculum_zero_start_stage:
            return 0.0
        return float(self.args.curriculum_zero_prob)

    def _reverse_prob_for_stage(self, stage_number: int) -> float:
        if stage_number < self.args.curriculum_reverse_start_stage:
            return 0.0
        return float(self.args.curriculum_reverse_prob)

    def _configure_velocity_curriculum_stage(self) -> None:
        stage_number = self.curriculum_stage + 1
        cap = float(self.args.curriculum_stages[self.curriculum_stage])
        vx_range = (float(self.args.curriculum_min_speed), cap)
        zero_prob = self._zero_prob_for_stage(stage_number)
        reverse_prob = self._reverse_prob_for_stage(stage_number)
        self.training_env.env_method("set_vx_range", vx_range)
        self.training_env.env_method("set_zero_command_prob", zero_prob)
        self.training_env.env_method("set_reverse_prob", reverse_prob)
        if self.eval_env is not None:
            self.eval_env.set_vx_range(vx_range)
            self.eval_env.set_zero_command_prob(zero_prob)
            self.eval_env.set_reverse_prob(reverse_prob)

    def _on_training_end(self) -> None:
        if self.eval_env is not None:
            self.eval_env.close()
            self.eval_env = None

    def _curriculum_step(self, index: int, info: dict) -> None:
        stats = self.curriculum_env_stats[index]
        stats["step_count"] += 1.0
        stats["contact_both_sum"] += float(
            info.get("wheel_contact_left", 0.0) and info.get("wheel_contact_right", 0.0)
        )
        stats["motion_peak"] = max(stats["motion_peak"], float(info.get("motion_max_abs", 0.0)))
        stats["station_peak"] = max(stats["station_peak"], float(info.get("station_max_abs", 0.0)))
        if abs(float(info.get("command_target_vx", 0.0))) > 0.01:
            stats["pitch_abs"].append(abs(float(info.get("pitch", 0.0))))
        if self.args.horizon_curriculum:
            # The first short stage is 32 steps.  Ignore only its command-ramp
            # beginning, not the entire episode as the long-horizon rule does.
            tracking_start = max(8, int(0.40 * self.args.horizon_stages[self.curriculum_stage]))
        else:
            tracking_start = 100
        if int(info.get("episode_steps", 0)) < tracking_start:
            return
        command = float(info.get("command_vx", 0.0))
        if abs(command) <= 0.02:
            return
        actual = float(info.get("body_vx_after", 0.0))
        stats["track_error_sum"] += abs(actual - command)
        stats["track_count"] += 1.0
        if abs(actual) >= 0.03:
            stats["direction_count"] += 1.0
            stats["direction_correct"] += float(actual * command > 0.0)
        else:
            stats["direction_count"] += 1.0

    def _finish_curriculum_episode(self, index: int, info: dict, reason: str) -> None:
        stats = self.curriculum_env_stats[index]
        target = float(info.get("command_target_vx", 0.0))
        sign = 0 if abs(target) <= 0.01 else (1 if target > 0.0 else -1)
        self.curriculum_episodes.append({
            "sign": sign,
            "magnitude": abs(target),
            "survived": float(reason == "time_limit"),
            "vx_mae": (
                stats["track_error_sum"] / stats["track_count"]
                if stats["track_count"] else float("nan")
            ),
            "direction_accuracy": (
                stats["direction_correct"] / stats["direction_count"]
                if stats["direction_count"] else float("nan")
            ),
            "contact_both": stats["contact_both_sum"] / max(stats["step_count"], 1.0),
            "motion_peak": stats["motion_peak"],
            "station_peak": stats["station_peak"],
            "pitch_p95_rad": (
                float(np.quantile(stats["pitch_abs"], 0.95))
                if stats["pitch_abs"] else float("nan")
            ),
        })
        self.curriculum_env_stats[index] = {
            "track_error_sum": 0.0,
            "track_count": 0.0,
            "direction_correct": 0.0,
            "direction_count": 0.0,
            "contact_both_sum": 0.0,
            "step_count": 0.0,
            "motion_peak": 0.0,
            "station_peak": 0.0,
            "pitch_abs": [],
        }

    def _curriculum_summary(self) -> dict[str, float | int | bool]:
        episodes = list(self.curriculum_episodes)
        moving = [item for item in episodes if item["sign"] != 0]
        positive = [item for item in moving if item["sign"] > 0]
        negative = [item for item in moving if item["sign"] < 0]
        zero = [item for item in episodes if item["sign"] == 0]

        def mean(items, key):
            values = [float(item[key]) for item in items if np.isfinite(item[key])]
            return float(np.mean(values)) if values else float("nan")

        def quantile(items, key, q=0.90):
            values = [float(item[key]) for item in items if np.isfinite(item[key])]
            return float(np.quantile(values, q)) if values else float("nan")

        min_count = int(self.args.curriculum_min_per_direction)
        stage_number = self.curriculum_stage + 1
        cap = float(
            self.args.curriculum_stages[self.curriculum_stage]
            if self.args.velocity_curriculum else self.args.vx_range[1]
        )
        near = [item for item in moving if item["magnitude"] >= 0.60 * cap]
        positive_near = [item for item in near if item["sign"] > 0]
        negative_near = [item for item in near if item["sign"] < 0]
        require_reverse = (
            self.args.velocity_curriculum
            and stage_number >= self.args.curriculum_reverse_start_stage
        )
        require_zero = (
            self.args.velocity_curriculum
            and self._zero_prob_for_stage(stage_number) > 0.0
        ) or (
            self.args.horizon_curriculum
            and float(self.args.command_zero_prob or 0.0) > 0.0
        )
        enough_data = (
            len(episodes) >= int(self.args.curriculum_window_episodes)
            and len(positive_near) >= min_count
            and (not require_reverse or len(negative_near) >= min_count)
            and (
                not require_zero
                or len(zero) >= max(
                    3,
                    int(self.args.curriculum_window_episodes * self.args.curriculum_zero_prob * 0.35),
                )
            )
        )
        moving_survival = mean(near, "survived")
        moving_error = mean(near, "vx_mae")
        direction_accuracy = mean(near, "direction_accuracy")
        positive_direction_accuracy = mean(positive_near, "direction_accuracy")
        negative_direction_accuracy = mean(negative_near, "direction_accuracy")
        moving_contact = mean(near, "contact_both")
        drift_p90 = quantile(near, "motion_peak")
        zero_survival = mean(zero, "survived")
        zero_drift_p90 = quantile(zero, "station_peak")
        moving_pitch_p95_deg = float(np.degrees(quantile(near, "pitch_p95_rad", 0.95)))
        qualified = bool(
            enough_data
            and moving_survival >= self.args.curriculum_min_survival
            and moving_error <= self.args.curriculum_max_vx_error
            and direction_accuracy >= self.args.curriculum_min_direction_accuracy
            and positive_direction_accuracy >= self.args.curriculum_min_direction_accuracy
            and (not require_reverse or negative_direction_accuracy >= self.args.curriculum_min_direction_accuracy)
            and moving_contact >= self.args.curriculum_min_contact
            and (not self.args.horizon_curriculum or moving_pitch_p95_deg <= self.args.horizon_max_pitch_deg)
            and (not require_zero or zero_survival >= self.args.curriculum_min_survival)
            and (not require_zero or zero_drift_p90 <= self.args.curriculum_max_zero_drift_m)
        )
        return {
            "episodes": len(episodes), "positive": len(positive), "negative": len(negative),
            "positive_near": len(positive_near), "negative_near": len(negative_near),
            "zero": len(zero), "enough_data": enough_data, "qualified": qualified,
            "stage": self.curriculum_stage + 1,
            "moving_survival": moving_survival, "moving_vx_mae": moving_error,
            "direction_accuracy": direction_accuracy,
            "positive_direction_accuracy": positive_direction_accuracy,
            "negative_direction_accuracy": negative_direction_accuracy,
            "moving_contact_both": moving_contact,
            "moving_drift_p90_m": drift_p90, "zero_survival": zero_survival,
            "zero_drift_p90_m": zero_drift_p90, "speed_cap_m_s": cap,
            "moving_pitch_p95_deg": moving_pitch_p95_deg,
            "episode_horizon_steps": (
                int(self.args.episode_steps)
                if self.args.episode_steps is not None
                else int(
                    self.args.horizon_stages[self.curriculum_stage]
                    if self.args.horizon_curriculum
                    else self.training_env.get_attr("stage")[0].episode_steps
                )
            ),
        }

    def _advance_velocity_curriculum(self, summary: dict[str, float | int | bool]) -> bool:
        if not self.args.velocity_curriculum or not bool(summary["qualified"]):
            return False
        if self.update - self.curriculum_stage_start_update < int(
            self.args.curriculum_min_stage_updates
        ):
            return False
        if self.curriculum_stage + 1 >= len(self.args.curriculum_stages):
            return False
        self.curriculum_stage += 1
        self.curriculum_stage_start_update = self.update
        self._configure_velocity_curriculum_stage()
        self.curriculum_episodes.clear()
        self._init_curriculum_env_stats()
        print(
            f"[curriculum] promoted to speed stage {self.curriculum_stage + 1}/"
            f"{len(self.args.curriculum_stages)}; "
            f"vx=[{self.args.curriculum_min_speed:.2f}, "
            f"{self.args.curriculum_stages[self.curriculum_stage]:.2f}] m/s, "
            f"reverse_p={self._reverse_prob_for_stage(self.curriculum_stage + 1):.2f}, "
            f"zero_p={self._zero_prob_for_stage(self.curriculum_stage + 1):.2f}",
            flush=True,
        )
        return True

    def _advance_horizon_curriculum(self, summary: dict[str, float | int | bool]) -> bool:
        if not self.args.horizon_curriculum or not bool(summary["qualified"]):
            return False
        if self.curriculum_stage + 1 >= len(self.args.horizon_stages):
            return False
        self.curriculum_stage += 1
        horizon = int(self.args.horizon_stages[self.curriculum_stage])
        self.training_env.env_method("set_episode_steps", horizon)
        if self.eval_env is not None:
            self.eval_env.set_episode_steps(horizon)
        self.curriculum_episodes.clear()
        self._init_curriculum_env_stats()
        print(
            f"[curriculum] promoted to horizon stage {self.curriculum_stage + 1}/"
            f"{len(self.args.horizon_stages)}; episode_steps={horizon}",
            flush=True,
        )
        return True

    def _deterministic_eval_summary(self, init_scale: float) -> dict[str, float | int | bool]:
        if self.eval_env is None:
            raise RuntimeError("deterministic evaluation environment is not initialized")
        env = self.eval_env
        env.set_init_scale(init_scale)
        horizon = int(env.episode_steps_override or env.stage.episode_steps)
        tracking_start = max(8, int(0.10 * horizon))
        episodes: list[dict[str, float | int]] = []
        for episode_index in range(int(self.args.deterministic_eval_episodes)):
            obs, _ = env.reset(seed=int(self.args.deterministic_eval_seed) + episode_index)
            errors: list[float] = []
            direction_hits: list[float] = []
            pitch_abs: list[float] = []
            contact_both: list[float] = []
            wheel_targets: list[float] = []
            wheel_currents: list[float] = []
            wheel_speeds: list[float] = []
            wheel_target_skews: list[float] = []
            wheel_speed_skews: list[float] = []
            track_rewards: list[float] = []
            done = False
            info: dict = {}
            while not done:
                action = self.model.predict(obs, deterministic=True)[0]
                obs, _, terminated, truncated, info = env.step(
                    np.asarray(action, dtype=np.float32).reshape(-1)
                )
                done = bool(terminated or truncated)
                command = float(info.get("command_vx", 0.0))
                actual = float(info.get("body_vx_after", 0.0))
                step = int(info.get("episode_steps", 0))
                if step >= tracking_start and abs(command) > 0.02:
                    errors.append(abs(actual - command))
                    direction_hits.append(float(actual * command > 0.0 and abs(actual) >= 0.03))
                pitch_abs.append(abs(float(info.get("pitch", 0.0))))
                contact_both.append(float(
                    info.get("wheel_contact_left", 0.0)
                    and info.get("wheel_contact_right", 0.0)
                ))
                wheel_targets.append(float(info.get("wheel_target_abs", 0.0)))
                wheel_currents.append(float(info.get("wheel_current_abs", 0.0)))
                wheel_speeds.append(float(info.get("wheel_speed_abs", 0.0)))
                wheel_target_skews.append(float(info.get("wheel_target_skew_rad_s", 0.0)))
                wheel_speed_skews.append(float(info.get("wheel_speed_skew_rad_s", 0.0)))
                track_rewards.append(float((info.get("reward_terms") or {}).get("track_vx", 0.0)))
            target = float(info.get("command_target_vx", 0.0))
            episodes.append({
                "sign": 0 if abs(target) <= 0.01 else (1 if target > 0.0 else -1),
                "magnitude": abs(target),
                "survived": float(info.get("termination_reason") == "time_limit"),
                "vx_mae": float(np.mean(errors)) if errors else float("nan"),
                "direction_accuracy": float(np.mean(direction_hits)) if direction_hits else float("nan"),
                "contact_both": float(np.mean(contact_both)) if contact_both else float("nan"),
                "motion_peak": float(info.get("motion_max_abs", 0.0)),
                "station_peak": float(info.get("station_max_abs", 0.0)),
                "pitch_p95_rad": float(np.quantile(pitch_abs, 0.95)) if pitch_abs else float("nan"),
                "wheel_target_abs": float(np.mean(wheel_targets)) if wheel_targets else float("nan"),
                "wheel_current_abs": float(np.mean(wheel_currents)) if wheel_currents else float("nan"),
                "wheel_speed_abs": float(np.mean(wheel_speeds)) if wheel_speeds else float("nan"),
                "wheel_target_skew_rad_s": float(np.mean(wheel_target_skews)) if wheel_target_skews else float("nan"),
                "wheel_speed_skew_rad_s": float(np.mean(wheel_speed_skews)) if wheel_speed_skews else float("nan"),
                "track_reward": float(np.mean(track_rewards)) if track_rewards else float("nan"),
                "steps": int(info.get("episode_steps", 0)),
            })

        moving = [item for item in episodes if item["sign"] != 0]
        cap = float(
            self.args.curriculum_stages[self.curriculum_stage]
            if self.args.velocity_curriculum
            else (env.get_vx_range()[1])
        )
        near = [item for item in moving if item["magnitude"] >= 0.60 * cap]
        positive_near = [item for item in near if item["sign"] > 0]
        negative_near = [item for item in near if item["sign"] < 0]
        zero = [item for item in episodes if item["sign"] == 0]

        def mean(items, key):
            values = [float(item[key]) for item in items if np.isfinite(float(item[key]))]
            return float(np.mean(values)) if values else float("nan")

        def quantile(items, key, q):
            values = [float(item[key]) for item in items if np.isfinite(float(item[key]))]
            return float(np.quantile(values, q)) if values else float("nan")

        stage_number = self.curriculum_stage + 1
        require_reverse = (
            self.args.velocity_curriculum
            and stage_number >= self.args.curriculum_reverse_start_stage
        )
        require_zero = (
            self.args.velocity_curriculum
            and self._zero_prob_for_stage(stage_number) > 0.0
        ) or (
            self.args.horizon_curriculum
            and float(self.args.command_zero_prob or 0.0) > 0.0
        )
        min_count = int(self.args.curriculum_min_per_direction)
        enough_data = (
            len(positive_near) >= min_count
            and (not require_reverse or len(negative_near) >= min_count)
            and (not require_zero or len(zero) >= 3)
        )
        moving_survival = mean(near, "survived")
        moving_error = mean(near, "vx_mae")
        direction_accuracy = mean(near, "direction_accuracy")
        positive_direction = mean(positive_near, "direction_accuracy")
        negative_direction = mean(negative_near, "direction_accuracy")
        moving_contact = mean(near, "contact_both")
        moving_pitch_p95_deg = float(np.degrees(quantile(near, "pitch_p95_rad", 0.95)))
        zero_survival = mean(zero, "survived")
        zero_drift_p90 = quantile(zero, "station_peak", 0.90)
        qualified = bool(
            enough_data
            and moving_survival >= self.args.curriculum_min_survival
            and moving_error <= self.args.curriculum_max_vx_error
            and direction_accuracy >= self.args.curriculum_min_direction_accuracy
            and positive_direction >= self.args.curriculum_min_direction_accuracy
            and (not require_reverse or negative_direction >= self.args.curriculum_min_direction_accuracy)
            and moving_contact >= self.args.curriculum_min_contact
            and (not self.args.horizon_curriculum or moving_pitch_p95_deg <= self.args.horizon_max_pitch_deg)
            and (not require_zero or zero_survival >= self.args.curriculum_min_survival)
            and (not require_zero or zero_drift_p90 <= self.args.curriculum_max_zero_drift_m)
        )
        return {
            "episodes": len(episodes),
            "positive_near": len(positive_near),
            "negative_near": len(negative_near),
            "zero": len(zero),
            "enough_data": enough_data,
            "qualified": qualified,
            "stage": stage_number,
            "moving_survival": moving_survival,
            "moving_vx_mae": moving_error,
            "direction_accuracy": direction_accuracy,
            "positive_direction_accuracy": positive_direction,
            "negative_direction_accuracy": negative_direction,
            "moving_contact_both": moving_contact,
            "moving_wheel_target_rad_s": mean(near, "wheel_target_abs"),
            "moving_wheel_current_a": mean(near, "wheel_current_abs"),
            "moving_wheel_speed_rad_s": mean(near, "wheel_speed_abs"),
            "moving_wheel_target_skew_rad_s": mean(near, "wheel_target_skew_rad_s"),
            "moving_wheel_speed_skew_rad_s": mean(near, "wheel_speed_skew_rad_s"),
            "moving_track_reward": mean(near, "track_reward"),
            "moving_drift_p90_m": quantile(near, "motion_peak", 0.90),
            "moving_pitch_p95_deg": moving_pitch_p95_deg,
            "zero_survival": zero_survival,
            "zero_drift_p90_m": zero_drift_p90,
            "speed_cap_m_s": cap,
            "episode_horizon_steps": horizon,
        }

    @staticmethod
    def _eval_score(summary: dict[str, float | int | bool]) -> tuple[float, ...]:
        def finite(value, missing=-1e6):
            value = float(value)
            return value if np.isfinite(value) else missing

        def negative(value, missing=-1e6):
            value = float(value)
            return -value if np.isfinite(value) else missing

        if np.isfinite(float(summary["moving_vx_mae"])):
            direction_values = [
                float(summary.get(name, float("nan")))
                for name in (
                    "positive_direction_accuracy",
                    "negative_direction_accuracy",
                )
                if np.isfinite(float(summary.get(name, float("nan"))))
            ]
            direction_floor = min(direction_values) if direction_values else -1.0
            return (
                finite(summary["moving_survival"], -1.0),
                direction_floor,
                negative(summary["zero_drift_p90_m"]),
                negative(summary["moving_vx_mae"]),
                finite(summary["moving_contact_both"], -1.0),
                negative(summary["moving_pitch_p95_deg"]),
                finite(summary["zero_survival"], -1.0),
            )
        return (
            finite(summary["zero_survival"], -1.0),
            negative(summary["zero_drift_p90_m"]),
            negative(summary["moving_pitch_p95_deg"]),
        )

    @staticmethod
    def _json_safe(value):
        if isinstance(value, dict):
            return {key: TrainingCallback._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [TrainingCallback._json_safe(item) for item in value]
        if isinstance(value, (np.bool_, bool)):
            return bool(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, (np.floating, float)):
            number = float(value)
            return number if np.isfinite(number) else None
        return value

    def _save_deterministic_candidates(
        self, summary: dict[str, float | int | bool]
    ) -> None:
        prefix = Path(self.args.out)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        stage = int(summary["stage"])
        score = self._eval_score(summary)
        moving_eval = np.isfinite(float(summary["moving_vx_mae"]))
        direction_floor_accuracy = None
        if moving_eval:
            direction_values = [
                float(summary[name])
                for name in (
                    "positive_direction_accuracy",
                    "negative_direction_accuracy",
                )
                if np.isfinite(float(summary[name]))
            ]
            direction_floor_accuracy = (
                min(direction_values) if direction_values else None
            )
        payload = {
            "update": self.update,
            "num_timesteps": int(self.num_timesteps),
            "deterministic": True,
            "direction_floor_accuracy": direction_floor_accuracy,
            "score_order": (
                [
                    "moving_survival", "direction_floor_accuracy",
                    "-zero_drift_p90_m", "-moving_vx_mae",
                    "moving_contact_both", "-moving_pitch_p95_deg",
                    "zero_survival",
                ]
                if moving_eval else [
                    "zero_survival", "-zero_drift_p90_m", "-moving_pitch_p95_deg"
                ]
            ),
            "score": list(score),
            **summary,
        }
        stage_path = prefix.with_name(f"{prefix.name}_stage_{stage}_best")
        stage_priority = (
            stage
            if (self.args.velocity_curriculum or self.args.horizon_curriculum)
            and bool(summary["qualified"])
            else 0
        )
        global_score = (float(stage_priority), *score)
        record = self._json_safe({**payload, "global_score": list(global_score)})
        history_path = prefix.with_name(f"{prefix.name}_deterministic_eval.jsonl")
        with history_path.open("a", encoding="utf-8") as history_file:
            history_file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        if score > self.stage_best_scores.get(stage, (-float("inf"),) * len(score)):
            self.stage_best_scores[stage] = score
            self.model.save(str(stage_path))
            stage_path.with_name(f"{stage_path.name}.eval.json").write_text(
                json.dumps(self._json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8",
            )

        if self.best_eval_score is None or global_score > self.best_eval_score:
            self.best_eval_score = global_score
            self.best_score = float(score[0])
            self.best_update = self.update
            self.best_metrics = dict(summary)
            self.model.save(str(prefix) + "_best")
            prefix.with_name(f"{prefix.name}_best.eval.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8",
            )

    def _run_scheduled_deterministic_eval(self, init_scale: float):
        interval = max(1, int(self.args.deterministic_eval_interval))
        if self.update != 1 and self.update % interval != 0 and self.update != self.total_updates:
            return None
        summary = self._deterministic_eval_summary(init_scale)
        self._save_deterministic_candidates(summary)
        for key, value in summary.items():
            if isinstance(value, (bool, int, float)):
                self.logger.record(f"eval_deterministic/{key}", float(value))
        print(
            "[eval deterministic] update={} stage={}/{} episodes={} cap={:.2f} "
            "survival={:.3f} vx_mae={:.4f} direction={:.3f} contact={:.3f} "
            "wheel_target={:.3f} wheel_speed={:.3f} current={:.3f} "
            "target_skew={:.3f} speed_skew={:.3f} "
            "zero_drift_p90={:.3f} qualified={}".format(
                self.update,
                summary["stage"],
                len(self.args.curriculum_stages if self.args.velocity_curriculum else self.args.horizon_stages)
                if (self.args.velocity_curriculum or self.args.horizon_curriculum) else 1,
                summary["episodes"],
                summary["speed_cap_m_s"],
                summary["moving_survival"],
                summary["moving_vx_mae"],
                summary["direction_accuracy"],
                summary["moving_contact_both"],
                summary["moving_wheel_target_rad_s"],
                summary["moving_wheel_speed_rad_s"],
                summary["moving_wheel_current_a"],
                summary["moving_wheel_target_skew_rad_s"],
                summary["moving_wheel_speed_skew_rad_s"],
                summary["zero_drift_p90_m"],
                summary["qualified"],
            ),
            flush=True,
        )
        return summary

    def _on_step(self) -> bool:
        infos = self.locals.get("infos") or []
        dones = self.locals.get("dones")
        for index, info in enumerate(infos):
            self.last_info = info
            for key in self.METRIC_KEYS:
                value = info.get(key)
                if value is not None and np.isfinite(value):
                    self.metric_sums[key] += float(value)
            self.metric_count += 1
            reason = str(info.get("termination_reason", "running"))
            if reason not in ("running", "time_limit"):
                self.term_counts[reason] = self.term_counts.get(reason, 0) + 1
            if self.args.velocity_curriculum or self.args.horizon_curriculum:
                self._curriculum_step(index, info)
            if dones is not None and index < len(dones) and bool(dones[index]):
                self.rollout_term_counts[reason] = self.rollout_term_counts.get(reason, 0) + 1
                episode = info.get("episode", {})
                self.episode_stats.append({
                    "survived": reason == "time_limit",
                    "return": float(episode.get("r", np.nan)),
                    "length": float(episode.get("l", np.nan)),
                })
                if self.args.velocity_curriculum or self.args.horizon_curriculum:
                    self._finish_curriculum_episode(index, info, reason)
        return True

    def _on_rollout_end(self) -> None:
        self.update += 1
        now = time.perf_counter()
        rollout_steps = int(self.model.n_steps * self.model.n_envs)
        rollout_seconds = max(now - self.last_rollout_time, 1e-9)
        elapsed_seconds = max(now - self.wall_start, 1e-9)
        rollout_fps = rollout_steps / rollout_seconds
        total_fps = max(self.num_timesteps - self.start_steps, 1) / elapsed_seconds
        self.last_rollout_time = now
        progress = min(1.0, self.update / self.total_updates)
        if self.args.velocity_curriculum:
            # Hold reset perturbations fixed while the velocity command itself
            # is being learned, so promotion measures speed skill alone.
            init_scale = self.args.init_scale_start
        elif self.args.horizon_curriculum:
            # Horizon progression may increase reset perturbations only after
            # its deterministic mastery gate promotes the stage.
            stage_count = len(
                self.args.horizon_stages
            )
            stage_progress = (
                self.curriculum_stage / max(stage_count - 1, 1)
                if stage_count > 1 else 0.0
            )
            init_scale = self.args.init_scale_start + (
                self.args.init_scale_end - self.args.init_scale_start
            ) * stage_progress
        else:
            init_scale = self.args.init_scale_start + (
                self.args.init_scale_end - self.args.init_scale_start
            ) * progress
        vx_scale = self.args.vx_reward_scale_start + (
            self.args.vx_reward_scale_end - self.args.vx_reward_scale_start
        ) * progress
        if hasattr(self.training_env, "env_method"):
            self.training_env.env_method("set_init_scale", float(init_scale))
            self.training_env.env_method("set_vx_reward_scale", float(vx_scale))
        self.logger.record("curriculum/init_scale", init_scale)
        self.logger.record("curriculum/vx_reward_scale", vx_scale)
        self.logger.record("policy/action_authority", 1.0)
        rollout_reward = float(np.mean(self.model.rollout_buffer.rewards))
        metric_mean = {
            key: (value / self.metric_count if self.metric_count else float("nan"))
            for key, value in self.metric_sums.items()
        }
        completed = [item for item in self.episode_stats if np.isfinite(item["return"])]
        episode_return = (
            float(np.mean([item["return"] for item in completed]))
            if completed else float("nan")
        )
        survival_steps = (
            float(np.mean([item["length"] for item in completed]))
            if completed else float("nan")
        )
        survival_rate = (
            float(np.mean([item["survived"] for item in completed]))
            if completed else float("nan")
        )
        curriculum_summary = (
            self._curriculum_summary()
            if self.args.velocity_curriculum or self.args.horizon_curriculum else None
        )
        eval_summary = self._run_scheduled_deterministic_eval(init_scale)
        self.logger.record("rollout/reward_mean", rollout_reward)
        self.logger.record("rollout/episode_return_mean", episode_return)
        self.logger.record("rollout/survival_steps_mean", survival_steps)
        self.logger.record("rollout/survival_rate", survival_rate)
        self.logger.record("eval_deterministic/best_survival_rate", self.best_score)
        self.logger.record("eval_deterministic/best_update", self.best_update)
        self.logger.record("time/rollout_fps", rollout_fps)
        self.logger.record("time/total_fps", total_fps)
        for key, value in metric_mean.items():
            self.logger.record(f"rollout/{key}", value)
        if curriculum_summary is not None:
            for key, value in curriculum_summary.items():
                if isinstance(value, (bool, int, float)):
                    self.logger.record(f"curriculum_rollout/{key}", float(value))
        if self.last_info:
            self.logger.record("rollout/pitch_deg", np.degrees(float(self.last_info.get("pitch", 0.0))))
            self.logger.record("rollout/leg_error_mm", float(self.last_info.get("leg_length_error_mm", 0.0)))
            self.logger.record("rollout/wheel_target_rad_s", float(self.last_info.get("wheel_target_abs", 0.0)))
        action_std = torch.exp(self.model.policy.log_std.detach()).cpu().numpy()
        for index, value in enumerate(action_std):
            self.logger.record(f"policy/action_std_{index}", float(value))
        log_interval = max(1, int(self.args.log_interval))
        if self.update % log_interval == 0:
            total_steps = self.total_updates * rollout_steps
            terminations = ", ".join(
                f"{key}={value}" for key, value in sorted(self.rollout_term_counts.items())
            ) or "none"
            print("", flush=True)
            print("=" * 72, flush=True)
            print(f"[PPO] iteration: {self.update}/{self.total_updates}", flush=True)
            print(f"[PPO] total_steps: {self.num_timesteps}/{total_steps}", flush=True)
            print(f"[PPO] init_scale: {init_scale:.3f}", flush=True)
            print(f"[PPO] vx_reward_scale: {vx_scale:.3f}", flush=True)
            print(f"[episode] survival_steps_mean: {survival_steps:.1f}", flush=True)
            print(f"[episode] survival_rate: {survival_rate:.3f}", flush=True)
            print(f"[episode] return_mean: {episode_return:.3f}", flush=True)
            print(f"[rollout] reward_mean: {rollout_reward:.3f}", flush=True)
            print(f"[wheel] target_rad_s: {metric_mean['wheel_target_abs']:.3f}", flush=True)
            print(f"[wheel] current_A: {metric_mean['wheel_current_abs']:.3f}", flush=True)
            print(f"[wheel] torque_Nm: {metric_mean['wheel_torque_abs_mean']:.3f}", flush=True)
            print(f"[wheel] speed_rad_s: {metric_mean['wheel_speed_abs']:.3f}", flush=True)
            print(f"[wheel] target_skew_rad_s: {metric_mean['wheel_target_skew_rad_s']:.3f}", flush=True)
            print(f"[wheel] speed_skew_rad_s: {metric_mean['wheel_speed_skew_rad_s']:.3f}", flush=True)
            print(f"[leg] target_rad: {metric_mean['leg_target_abs']:.3f}", flush=True)
            print(f"[leg] torque_Nm: {metric_mean['leg_torque_abs_mean']:.3f}", flush=True)
            print(f"[leg] length_error_mm: {metric_mean['leg_length_error_mm']:.3f}", flush=True)
            print(f"[state] pitch_deg: {np.degrees(metric_mean['pitch']):.3f}", flush=True)
            print(f"[state] height_m: {metric_mean['base_height']:.4f}", flush=True)
            print(f"[state] body_vx_m_s: {metric_mean['body_vx_after']:.3f}", flush=True)
            print(f"[policy] action_abs: {metric_mean['action_abs']:.3f}", flush=True)
            print(
                "[policy] action_std: "
                + np.array2string(action_std, precision=3, separator=", "),
                flush=True,
            )
            print("[policy] action_authority: 1.000", flush=True)
            print(f"[time] rollout_fps: {rollout_fps:.0f}", flush=True)
            print(f"[time] total_fps: {total_fps:.0f}", flush=True)
            print(f"[best] update: {self.best_update}/{self.total_updates}", flush=True)
            print(f"[episode] terminations: {terminations}", flush=True)
            if curriculum_summary is not None:
                print(
                    "[curriculum rollout stats] stage={}/{} horizon={} cap={:.2f}m/s episodes={}/{} "
                    "(near-cap +{} / -{} / zero{}), surv={:.2f} vx_mae={:.3f} "
                    "dir={:.2f} (+{:.2f}/-{:.2f}) contact={:.3f} pitch_p95={:.1f}deg "
                    "drift_p90={:.3f}m zero_drift_p90={:.3f}m rollout_qualified={}".format(
                        curriculum_summary["stage"],
                        len(self.args.curriculum_stages if self.args.velocity_curriculum else self.args.horizon_stages),
                        curriculum_summary["episode_horizon_steps"],
                        curriculum_summary["speed_cap_m_s"],
                        curriculum_summary["episodes"], self.args.curriculum_window_episodes,
                        curriculum_summary["positive_near"], curriculum_summary["negative_near"],
                        curriculum_summary["zero"], curriculum_summary["moving_survival"],
                        curriculum_summary["moving_vx_mae"], curriculum_summary["direction_accuracy"],
                        curriculum_summary["positive_direction_accuracy"],
                        curriculum_summary["negative_direction_accuracy"],
                        curriculum_summary["moving_contact_both"],
                        curriculum_summary["moving_pitch_p95_deg"],
                        curriculum_summary["moving_drift_p90_m"],
                        curriculum_summary["zero_drift_p90_m"], curriculum_summary["qualified"],
                    ), flush=True
                )
        if eval_summary is not None and eval_summary["qualified"]:
            if self.args.horizon_curriculum:
                self._advance_horizon_curriculum(eval_summary)
            elif self.args.velocity_curriculum:
                self._advance_velocity_curriculum(eval_summary)
        self.metric_sums = {key: 0.0 for key in self.METRIC_KEYS}
        self.metric_count = 0
        self.rollout_term_counts = {}
        if self.args.save_interval > 0 and self.update % self.args.save_interval == 0:
            prefix = Path(self.args.out)
            prefix.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(str(prefix) + f"_iter_{self.update}")


def _parse_pair(values):
    return None if values is None else tuple(float(value) for value in values)


def _write_contract(path: Path, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "contract_version": CONTRACT_VERSION,
        "observation_dim": OBS_DIM,
        "actor_observation_dim": ACTOR_OBS_DIM,
        "action_dim": ACTION_DIM,
        "action_layout": [
            "wheel_velocity_target_left",
            "wheel_velocity_target_right",
            "hip_position_target_0",
            "hip_position_target_1",
            "hip_position_target_2",
            "hip_position_target_3",
        ],
        "wheel_action_frame": WHEEL_ACTION_FRAME,
        "wheel_action_to_joint_velocity_sign": WHEEL_ANGULAR_TO_BODY_X,
        "policy_rate_hz": POLICY_RATE_HZ,
        "motor_control_rate_hz": MOTOR_CONTROL_RATE_HZ,
        "motor_ticks_per_policy": MOTOR_TICKS_PER_POLICY,
        "leg_interface": {
            "mode": "MIT",
            "v_des": 0.0,
            "kp": 100.0,
            "kd": 4.0,
            "t_ff": 0.0,
        },
        "wheel_interface": "velocity_target_to_500hz_PI_to_C620_current",
        "control_architecture": {
            "policy": "single_direct_ppo",
            "policy_outputs": "six_absolute_actuator_targets",
            "teacher_policy": False,
            "residual_policy": False,
            "coordination_controller": False,
        },
        "wheel_speed_loop": {
            "kp_a_per_rad_s": float(args.wheel_speed_kp),
            "ki_a_per_rad_s2": float(args.wheel_speed_ki),
            "error_limit_rad_s": 14.0,
            "integral_limit_a": float(args.wheel_integral_limit),
        },
        "deployment_runtime": "ONNX Runtime CPU",
        "stage": args.stage,
        "deployment_mode": bool(args.deployment_mode),
        "updates": int(args.updates),
        "num_envs": int(args.num_envs),
        "rollout_steps_per_env": int(args.rollout_steps),
        "episode_steps_override": args.episode_steps,
        "total_timesteps": int(args.updates * args.num_envs * args.rollout_steps),
        "learning_rate": float(args.learning_rate),
        "warmstart_log_std": (
            list(args.warmstart_log_std) if args.warmstart_log_std is not None else None
        ),
        "init_scale_schedule": {
            "start": float(args.init_scale_start),
            "end": float(args.init_scale_end),
            "mode": (
                "fixed_during_command_curriculum" if args.velocity_curriculum
                else "mastery_gated_by_horizon_stage" if args.horizon_curriculum
                else "linear_by_update"
            ),
        },
        "target_kl": None if args.target_kl is None else float(args.target_kl),
        "tilt_limit_hold_steps": int(args.tilt_hold_steps),
        "velocity_curriculum": {
            "enabled": bool(args.velocity_curriculum),
            "stages_m_s": list(args.curriculum_stages) if args.velocity_curriculum else [],
            "start_stage": int(args.curriculum_start_stage),
            "minimum_speed_m_s": float(args.curriculum_min_speed),
            "zero_command_probability": float(args.curriculum_zero_prob),
            "zero_start_stage": int(args.curriculum_zero_start_stage),
            "reverse_start_stage": int(args.curriculum_reverse_start_stage),
            "reverse_command_probability": float(args.curriculum_reverse_prob),
            "minimum_updates_per_stage": int(args.curriculum_min_stage_updates),
            "promotion_window_episodes": int(args.curriculum_window_episodes),
            "promotion_min_survival": float(args.curriculum_min_survival),
            "promotion_max_vx_mae_m_s": float(args.curriculum_max_vx_error),
            "promotion_min_direction_accuracy": float(args.curriculum_min_direction_accuracy),
            "promotion_min_contact_rate": float(args.curriculum_min_contact),
            "promotion_max_zero_drift_p90_m": float(args.curriculum_max_zero_drift_m),
        },
        "horizon_curriculum": {
            "enabled": bool(args.horizon_curriculum),
            "stages_policy_steps": list(args.horizon_stages) if args.horizon_curriculum else [],
            "forward_only": bool(args.horizon_curriculum),
            "max_moving_pitch_p95_deg": float(args.horizon_max_pitch_deg),
            "promotion_window_episodes": int(args.curriculum_window_episodes),
            "promotion_min_survival": float(args.curriculum_min_survival),
            "promotion_max_vx_mae_m_s": float(args.curriculum_max_vx_error),
            "promotion_min_direction_accuracy": float(args.curriculum_min_direction_accuracy),
            "promotion_min_contact_rate": float(args.curriculum_min_contact),
        },
        "low_speed_motion_reward": {
            "tracking_mode": "command_scaled_absolute_error_plus_command_signed_alignment",
            "track_sigma_m_s": EnvParams().rewards.low_speed_track_sigma_m_s,
            "track_sigma_relative_fraction": float(args.low_speed_track_sigma_relative),
            "track_sigma_rule": "max(base_sigma, fraction * abs(command_vx))",
            "track_weight": EnvParams().rewards.track_vx,
            "signed_alignment_weight": EnvParams().rewards.track_vx_progress,
            "reverse_reward_scale": float(args.low_speed_reverse_reward_scale),
            "wrong_direction_penalty_weight": float(args.low_speed_wrong_direction_weight),
            "straight_translation_wheel_differential_weight": float(
                args.low_speed_wheel_differential_weight
            ),
            "startup_progress_weight": EnvParams().rewards.low_speed_startup_progress,
            "cruise_progress_weight": EnvParams().rewards.low_speed_cruise_progress,
            "stable_motion_weight": (
                EnvParams().rewards.low_speed_stable_motion
                if args.low_speed_stable_motion_weight is None
                else float(args.low_speed_stable_motion_weight)
            ),
            "stable_pitch_sigma_rad": EnvParams().rewards.low_speed_stable_pitch_sigma_rad,
            "stable_pitch_rate_sigma_rad_s": EnvParams().rewards.low_speed_stable_pitch_rate_sigma_rad_s,
            "stable_penalty_mode": "bounded_excess_quadratic",
            "stable_penalty_cap": EnvParams().rewards.low_speed_stable_penalty_cap,
            "posture_gate": False,
            "pitch_speed_coupling": False,
            "direction_duplicate_penalty": False,
            "zero_hold_position_mode": "planar_norm_smooth_bounded",
            "zero_hold_progress_mode": "planar_potential_difference",
        },
        "deterministic_evaluation": {
            "enabled": True,
            "interval_updates": int(args.deterministic_eval_interval),
            "episodes": int(args.deterministic_eval_episodes),
            "seed": int(args.deterministic_eval_seed),
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="UZ-05 direct PPO training")
    parser.add_argument("--stage", default="stand")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--reset-optimizer", action="store_true")
    parser.add_argument("--updates", type=int, default=800)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--ppo-epochs", type=int, default=5)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--target-kl", type=float, default=0.015)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--initial-log-std", type=float, default=-1.5)
    parser.add_argument(
        "--warmstart-log-std", type=float, nargs=6, default=None,
        metavar=("WHEEL_L", "WHEEL_R", "LEG_1", "LEG_2", "LEG_3", "LEG_4"),
        help="override the six PPO action log standard deviations after loading a checkpoint",
    )
    parser.add_argument("--log-std-min", type=float, default=-4.0)
    parser.add_argument("--log-std-max", type=float, default=0.0)
    parser.add_argument("--stand-level", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--version", default="ppo_direct_v1")
    parser.add_argument("--out", default=None)
    parser.add_argument("--tensorboard-log", default=None)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--init-scale-start", type=float, default=0.0)
    parser.add_argument("--init-scale-end", type=float, default=1.0)
    parser.add_argument("--vx-range", type=float, nargs=2, default=None)
    parser.add_argument("--command-zero-prob", type=float, default=None)
    parser.add_argument("--command-reverse-prob", type=float, default=None)
    parser.add_argument(
        "--velocity-curriculum", action="store_true",
        help="mastery-gated low-speed command progression up to 0.30 m/s",
    )
    parser.add_argument("--curriculum-stages", type=float, nargs="+", default=[0.12, 0.18, 0.24, 0.30])
    parser.add_argument("--curriculum-min-speed", type=float, default=0.08)
    parser.add_argument("--curriculum-start-stage", type=int, default=1)
    parser.add_argument("--curriculum-reverse-start-stage", type=int, default=2)
    parser.add_argument("--curriculum-reverse-prob", type=float, default=0.5)
    parser.add_argument("--curriculum-zero-start-stage", type=int, default=3)
    parser.add_argument("--curriculum-zero-prob", type=float, default=0.10)
    parser.add_argument("--curriculum-window-episodes", type=int, default=80)
    parser.add_argument("--curriculum-min-stage-updates", type=int, default=0)
    parser.add_argument("--curriculum-min-per-direction", type=int, default=8)
    parser.add_argument("--curriculum-min-survival", type=float, default=0.95)
    parser.add_argument("--curriculum-max-vx-error", type=float, default=0.03)
    parser.add_argument("--curriculum-min-direction-accuracy", type=float, default=0.90)
    parser.add_argument("--curriculum-min-contact", type=float, default=0.95)
    parser.add_argument("--curriculum-max-zero-drift-m", type=float, default=0.08)
    parser.add_argument("--deterministic-eval-interval", type=int, default=20)
    parser.add_argument("--deterministic-eval-episodes", type=int, default=60)
    parser.add_argument("--deterministic-eval-seed", type=int, default=10000)
    parser.add_argument(
        "--horizon-curriculum", action="store_true",
        help="forward-only low-speed mastery stages with increasing episode duration",
    )
    parser.add_argument("--horizon-stages", type=int, nargs="+", default=[128, 160, 192, 224, 256, 500, 1000])
    parser.add_argument("--horizon-max-pitch-deg", type=float, default=10.0)
    parser.add_argument("--episode-steps", type=int, default=None)
    parser.add_argument("--command-accel-limit", type=float, default=None)
    parser.add_argument("--vx-reward-scale-start", type=float, default=1.0)
    parser.add_argument("--vx-reward-scale-end", type=float, default=1.0)
    parser.add_argument("--height-switch-steps", type=int, default=0)
    parser.add_argument("--height-switch-prob", type=float, default=0.0)
    parser.add_argument("--height-target-rate", type=float, default=0.0)
    parser.add_argument("--leg-length-reward-weight", type=float, default=None)
    parser.add_argument("--leg-length-progress-reward-weight", type=float, default=None)
    parser.add_argument("--leg-length-rate-reward-weight", type=float, default=None)
    parser.add_argument("--low-speed-stable-motion-weight", type=float, default=None)
    parser.add_argument(
        "--low-speed-reverse-reward-scale", type=float, default=1.0,
        help="multiplier on low-speed reverse-command tracking and signed-alignment rewards",
    )
    parser.add_argument(
        "--low-speed-wrong-direction-weight", type=float, default=0.0,
        help="quadratic penalty on low-speed body velocity opposite to the command",
    )
    parser.add_argument(
        "--low-speed-wheel-differential-weight", type=float, default=0.0,
        help="penalize left-right wheel-speed mismatch only during straight low-speed translation",
    )
    parser.add_argument(
        "--low-speed-track-sigma-relative", type=float,
        default=EnvParams().rewards.low_speed_track_sigma_relative,
        help="minimum tracking sigma as a fraction of |command vx|",
    )
    parser.add_argument(
        "--low-speed-posture-gate-gain", type=float, default=None,
        help="legacy compatibility option; low-speed velocity reward is no longer posture-gated",
    )
    parser.add_argument("--deployment-mode", action="store_true")
    parser.add_argument(
        "--tilt-hold-steps", type=int, default=0,
        help="number of 125 Hz policy steps allowed beyond tilt limit before termination",
    )
    parser.add_argument("--wheel-speed-kp", type=float, default=None)
    parser.add_argument("--wheel-speed-ki", type=float, default=None)
    parser.add_argument("--wheel-integral-limit", type=float, default=None)
    parser.add_argument("--observation-delay-steps", type=int, nargs=2, default=None)
    parser.add_argument(
        "--actuator-delay-steps", type=int, nargs=2, default=None,
        help="actuator delay range in 500 Hz motor ticks",
    )
    args = parser.parse_args()
    if args.low_speed_stable_motion_weight is not None and args.low_speed_stable_motion_weight < 0.0:
        parser.error("--low-speed-stable-motion-weight must be nonnegative")
    if args.low_speed_reverse_reward_scale < 0.0:
        parser.error("--low-speed-reverse-reward-scale must be nonnegative")
    if args.low_speed_wrong_direction_weight < 0.0:
        parser.error("--low-speed-wrong-direction-weight must be nonnegative")
    if args.low_speed_wheel_differential_weight < 0.0:
        parser.error("--low-speed-wheel-differential-weight must be nonnegative")
    if args.low_speed_track_sigma_relative < 0.0:
        parser.error("--low-speed-track-sigma-relative must be nonnegative")
    if args.low_speed_posture_gate_gain is not None and args.low_speed_posture_gate_gain < 0.0:
        parser.error("--low-speed-posture-gate-gain must be nonnegative")
    if args.horizon_curriculum and args.velocity_curriculum:
        parser.error("--horizon-curriculum and --velocity-curriculum are mutually exclusive")
    if args.horizon_curriculum:
        if args.stage != "low_speed":
            parser.error("--horizon-curriculum requires --stage low_speed")
        if args.vx_range is None or args.command_zero_prob is None or args.command_reverse_prob != 0.0:
            parser.error("horizon curriculum requires --vx-range, --command-zero-prob and --command-reverse-prob 0")
        if len(args.horizon_stages) < 2 or any(value <= 0 for value in args.horizon_stages) or any(
            right <= left for left, right in zip(args.horizon_stages, args.horizon_stages[1:])
        ) or args.horizon_stages[-1] != 1000:
            parser.error("horizon stages must increase strictly and end at 1000 policy steps")
        if args.curriculum_window_episodes < 10 or args.curriculum_min_per_direction < 1:
            parser.error("horizon curriculum episode window must be >=10 and minimum forward count >=1")
        if not 0.0 < args.horizon_max_pitch_deg < 17.0:
            parser.error("--horizon-max-pitch-deg must be between 0 and 17")
        if not 0.0 <= args.command_zero_prob < 1.0:
            parser.error("--command-zero-prob must be in [0, 1)")
        args.curriculum_zero_prob = args.command_zero_prob
    if args.warmstart_log_std is not None:
        if args.checkpoint is None:
            parser.error("--warmstart-log-std requires --checkpoint")
        if any(value < args.log_std_min or value > args.log_std_max for value in args.warmstart_log_std):
            parser.error("each --warmstart-log-std value must be within --log-std-min/--log-std-max")
    args.vx_range = _parse_pair(args.vx_range)
    args.observation_delay_steps = None if args.observation_delay_steps is None else tuple(args.observation_delay_steps)
    args.actuator_delay_steps = None if args.actuator_delay_steps is None else tuple(args.actuator_delay_steps)
    if args.velocity_curriculum:
        if args.stage != "low_speed":
            parser.error("--velocity-curriculum requires --stage low_speed")
        if args.vx_range is not None:
            parser.error("do not combine --velocity-curriculum with --vx-range")
        if args.command_zero_prob is not None or args.command_reverse_prob is not None:
            parser.error("use curriculum stage settings for zero and reverse commands")
        if (
            not args.curriculum_stages
            or any(value <= 0.0 for value in args.curriculum_stages)
            or any(b <= a for a, b in zip(args.curriculum_stages, args.curriculum_stages[1:]))
            or args.curriculum_stages[-1] < 0.30
        ):
            parser.error("curriculum stages must increase strictly and end at or above 0.30 m/s")
        if not 1 <= args.curriculum_start_stage <= len(args.curriculum_stages):
            parser.error("--curriculum-start-stage must select an existing speed stage")
        if not 1 <= args.curriculum_reverse_start_stage <= len(args.curriculum_stages) + 1:
            parser.error("--curriculum-reverse-start-stage must be within the curriculum")
        if not 1 <= args.curriculum_zero_start_stage <= len(args.curriculum_stages) + 1:
            parser.error("--curriculum-zero-start-stage must be within the curriculum")
        if (
            args.curriculum_window_episodes < 10
            or args.curriculum_min_per_direction < 1
            or args.curriculum_min_stage_updates < 0
        ):
            parser.error(
                "curriculum window must be >=10, per-direction minimum >=1, "
                "and minimum stage updates >=0"
            )
        if not 0.0 <= args.curriculum_zero_prob < 1.0:
            parser.error("--curriculum-zero-prob must be in [0, 1)")
        if not 0.0 <= args.curriculum_reverse_prob <= 1.0:
            parser.error("--curriculum-reverse-prob must be in [0, 1]")
        if not 0.0 <= args.curriculum_min_survival <= 1.0:
            parser.error("--curriculum-min-survival must be in [0, 1]")
        if not 0.0 <= args.curriculum_min_direction_accuracy <= 1.0:
            parser.error("--curriculum-min-direction-accuracy must be in [0, 1]")
        if not 0.0 <= args.curriculum_min_contact <= 1.0:
            parser.error("--curriculum-min-contact must be in [0, 1]")
        start_stage_number = int(args.curriculum_start_stage)
        args.vx_range = (
            args.curriculum_min_speed,
            float(args.curriculum_stages[start_stage_number - 1]),
        )
        args.command_zero_prob = (
            float(args.curriculum_zero_prob)
            if start_stage_number >= args.curriculum_zero_start_stage else 0.0
        )
        args.command_reverse_prob = (
            float(args.curriculum_reverse_prob)
            if start_stage_number >= args.curriculum_reverse_start_stage else 0.0
        )
    if args.curriculum_min_speed < 0.0 or (
        args.velocity_curriculum and args.curriculum_min_speed >= args.curriculum_stages[0]
    ):
        parser.error("curriculum minimum speed must be non-negative and below the first stage cap")
    if args.tilt_hold_steps < 0:
        raise ValueError("--tilt-hold-steps must be non-negative")
    if args.deterministic_eval_interval <= 0 or args.deterministic_eval_episodes <= 0:
        raise ValueError("deterministic evaluation interval and episodes must be positive")
    if args.wheel_speed_kp is None:
        args.wheel_speed_kp = 0.60
    if args.wheel_speed_ki is None:
        args.wheel_speed_ki = 0.30
    if args.wheel_integral_limit is None:
        args.wheel_integral_limit = 2.0
    if args.wheel_speed_kp < 0.0 or args.wheel_speed_ki < 0.0 or args.wheel_integral_limit <= 0.0:
        raise ValueError("wheel PI gains must be non-negative and integral limit positive")
    if args.out is None:
        args.out = str(REPO_ROOT / "checkpoints" / args.version / "checkpoint")
    if args.tensorboard_log is None:
        args.tensorboard_log = str(REPO_ROOT / "runs" / args.version)

    env_count = max(1, int(args.num_envs))
    if args.updates <= 0:
        raise ValueError("--updates must be a positive number of PPO rollout iterations")
    if args.rollout_steps <= 0:
        raise ValueError("--rollout-steps must be positive")
    if args.episode_steps is not None and args.episode_steps <= 0:
        raise ValueError("--episode-steps must be positive")
    rollout_steps_total = int(args.rollout_steps) * env_count
    total_timesteps = int(args.updates) * rollout_steps_total
    print(
        f"[PPO] training_budget: updates={args.updates}, "
        f"envs={env_count}, rollout_steps_per_env={args.rollout_steps}, "
        f"steps_per_update={rollout_steps_total}, total_timesteps={total_timesteps}",
        flush=True,
    )
    factories = [make_env(args.stage, rank, args.seed, args) for rank in range(env_count)]
    vec_env = DummyVecEnv(factories) if env_count == 1 else SubprocVecEnv(factories)
    vec_env = VecMonitor(vec_env)
    policy_kwargs = dict(
        actor_dim=ACTOR_OBS_DIM,
        net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64]),
        activation_fn=nn.ELU,
    )
    callback = TrainingCallback(args, args.updates, verbose=0)
    if args.checkpoint:
        # n_steps participates in rollout-buffer construction.  Override it
        # during load rather than assigning it afterwards, otherwise resuming
        # with a different --rollout-steps leaves a stale-sized buffer.
        model = PPO.load(
            args.checkpoint,
            env=vec_env,
            device="cpu",
            custom_objects={"n_steps": int(args.rollout_steps)},
        )
        if ensure_forward_positive_wheel_actions(model):
            print("[PPO] migrated legacy joint-positive wheel actions; optimizer state reset", flush=True)
        model.set_env(vec_env)
        model.tensorboard_log = args.tensorboard_log
        model.batch_size = args.batch_size
        model.learning_rate = args.learning_rate
        model.lr_schedule = get_schedule_fn(args.learning_rate)
        model.n_epochs = args.ppo_epochs
        # SB3 evaluates clip_range(progress_remaining) inside PPO.train().
        # Assigning the CLI float directly only breaks the checkpoint-resume
        # path; fresh PPO construction wraps it as a schedule automatically.
        model.clip_range = get_schedule_fn(args.clip_range)
        model.target_kl = args.target_kl
        model.ent_coef = args.ent_coef
        if args.warmstart_log_std is not None:
            new_log_std = torch.as_tensor(
                args.warmstart_log_std,
                dtype=model.policy.log_std.dtype,
                device=model.policy.log_std.device,
            )
            model.policy.log_std.data.copy_(new_log_std)
            print(
                "[PPO] warmstart_action_std: "
                + np.array2string(torch.exp(new_log_std).cpu().numpy(), precision=3, separator=", "),
                flush=True,
            )
        if args.reset_optimizer:
            # Keep the loaded PPO policy/value weights while discarding Adam's
            # accumulated moments.  Do not route this case through fresh PPO
            # construction, which would silently ignore --checkpoint.
            model.policy.optimizer.state.clear()
    else:
        model = PPO(
            AsymmetricActorCriticPolicy,
            vec_env,
            n_steps=args.rollout_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            n_epochs=args.ppo_epochs,
            clip_range=args.clip_range,
            target_kl=args.target_kl,
            ent_coef=args.ent_coef,
            policy_kwargs=policy_kwargs,
            tensorboard_log=args.tensorboard_log,
            device="cpu",
            seed=args.seed,
            verbose=0,
        )
        model.policy.log_std.data.fill_(args.initial_log_std)
        model.wheel_action_frame = WHEEL_ACTION_FRAME
    # Keep TensorBoard records, but remove SB3's large tabular stdout dump.
    model.set_logger(configure_logger(args.tensorboard_log, ["tensorboard"]))
    start_num_timesteps = int(model.num_timesteps)
    model.learn(
        # SB3 counts timesteps across all vectorized environments.  One
        # rollout therefore contributes rollout_steps * env_count steps.
        total_timesteps=total_timesteps,
        callback=callback,
        log_interval=max(1, args.log_interval),
        reset_num_timesteps=False,
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output))
    _write_contract(output.with_suffix(".contract.json"), args)
    vec_env.close()
    actual_timesteps = int(model.num_timesteps) - start_num_timesteps
    print(
        f"[PPO] completed_updates: {callback.update}/{args.updates}",
        flush=True,
    )
    print(
        f"[PPO] completed_timesteps: {actual_timesteps}/{total_timesteps}",
        flush=True,
    )
    print(
        f"[PPO] best_update: {callback.best_update}/{args.updates}",
        flush=True,
    )
    print(
        f"[PPO] best_checkpoint: {Path(args.out).with_name(Path(args.out).name + '_best')}.zip",
        flush=True,
    )
    print(f"saved: {output}.zip", flush=True)


if __name__ == "__main__":
    main()
