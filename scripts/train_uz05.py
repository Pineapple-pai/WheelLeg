"""训练入口：分阶段（capability）训练 UZ-05 轮腿机器人。

**继承机制**：观测维度（actor 38 / critic-full 94）与动作维度（6）在所有阶段完全固定，
所以任何阶段的 checkpoint 都能直接加载到任何其它阶段，**不需要迁移**。
capability 只决定奖励/课程/终止，不改变网络接口。

典型用法::

    # 1) 站立
    python scripts/train_uz05.py --stage stand --updates 800

    # 2) 接着训低速平移（直接继承站立能力）
    python scripts/train_uz05.py --stage low_speed \\
        --checkpoint checkpoints/stand_s2_v1/checkpoint --updates 1500

    # 3) 依次推进：high_speed / steering / rotation / airborne / stairs / jump / recovery
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path

# ---- 线程数必须在 import torch 之前卡死 ----
# subproc 环境的子进程会继承这些变量。不限制的话：32 个环境进程 × 每进程 16 个
# OMP/BLAS 线程 = 500+ 线程抢 16 个核，机器直接卡死（已踩过一次，见 train_uz05_all.sh）。
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize

import torch.nn as nn

from uz05.env import UZ05Env
from uz05.spec import (ACTION_DIM, ACTOR_OBS_DIM, CAPABILITY_ORDER, OBS_DIM,
                       REPO_ROOT, STAGE_BY_NAME, STAGES)

CONTRACT_VERSION = "uz05_iface_v2"


# 常用训练配置集中在这里，避免命令行反复堆叠几十个参数。
# 命令行显式给出的参数优先级更高，例如：
#   --preset leg-length --updates 100 --num-envs 2
TRAINING_PRESETS: dict[str, dict[str, object]] = {
    "leg-length": {
        "stage": "stand",
        "stand_level": 2,
        "checkpoint": str(
            REPO_ROOT / "checkpoints" / "ppo_stand_height_ramped_v4" / "checkpoint"
        ),
        "reset_optimizer": True,
        "updates": 600,
        "rollout_steps": 512,
        "batch_size": 512,
        "num_envs": 4,
        "vec_env": "subproc",
        "version": "ppo_stand_leg_015_027_v2",
        "learning_rate": 3.0e-5,
        "ppo_epochs": 2,
        "clip_range": 0.05,
        "target_kl": 0.005,
        "ent_coef": 0.0,
        "height_switch_steps": 500,
        "height_switch_prob": 1.0,
        "height_rate_limit": 0.90,
        "height_rate_gain": 20.0,
        "leg_feedforward_scale": 1.00,
        "height_target_rate": 0.25,
        "disable_adaptive_lr": True,
        "init_scale_start": 0.50,
        "init_scale_end": 1.0,
        "coord_start": 1.0,
        "coord_end": 1.0,
        # 腿长由闭环独立跟踪；策略只保留很小的轮平衡残差，避免 PPO
        # 在高度切换时把腿关节再次推离当前目标。
        "coord_residual_scale": (0.02, 0.005, 0.0),
        "log_verbose": False,
    },
}

# 版本化 profile：除训练参数外，还固定本版的 checkpoint /
# TensorBoard 目录。每个新版本都在 checkpoints/<version>/ 下单独保存。
TRAINING_PROFILES: dict[str, dict[str, object]] = {
    "v28": {
        **TRAINING_PRESETS["leg-length"],
        "updates": 1000,
        "version": "v28_fixed",
        "out": str(REPO_ROOT / "checkpoints" / "v28_fixed" / "checkpoint"),
        "tensorboard_log": str(REPO_ROOT / "runs" / "v28_fixed"),
        "log_interval": 10,
        "save_interval": 200,
    },
    "v29": {
        **TRAINING_PRESETS["leg-length"],
        "checkpoint": str(
            REPO_ROOT / "checkpoints" / "ppo_stand_height_ramped_v4" / "checkpoint"
        ),
        "updates": 1000,
        "version": "v29_leg_015_027_fast",
        "out": str(REPO_ROOT / "checkpoints" / "v29_leg_015_027_fast" / "checkpoint"),
        "tensorboard_log": str(REPO_ROOT / "runs" / "v29_leg_015_027_fast"),
        "log_interval": 10,
        "save_interval": 200,
    },
    "v30": {
        **TRAINING_PRESETS["leg-length"],
        # v29 的随机探索过大，绝大多数 rollout 在 height_limit 终止。
        # v30 从原始稳定站立策略重新分叉，以低探索只微调轮平衡残差；
        # 腿长本身交给带速度反馈的确定性闭环。
        "checkpoint": str(
            REPO_ROOT / "checkpoints" / "ppo_stand_height_ramped_v4" / "checkpoint"
        ),
        "updates": 1000,
        "version": "v30_leg_tracking_damped",
        "out": str(REPO_ROOT / "checkpoints" / "v30_leg_tracking_damped" / "checkpoint"),
        "tensorboard_log": str(REPO_ROOT / "runs" / "v30_leg_tracking_damped"),
        "learning_rate": 1.0e-5,
        "ppo_epochs": 1,
        "clip_range": 0.03,
        "target_kl": 0.002,
        "initial_log_std": -3.5,
        "log_std_min": -4.0,
        "log_std_max": -3.0,
        "freeze_actor_backbone": True,
        "init_scale_start": 0.20,
        "init_scale_end": 0.60,
        "height_switch_steps": 500,
        "height_target_rate": 0.08,
        "height_rate_limit": 0.35,
        "height_rate_gain": 7.0,
        "height_rate_damping": 1.0,
        "height_rate_brake_threshold": 0.005,
        "height_retract_rate_limit": 0.28,
        "height_retract_slow_rate_limit": 0.14,
        "height_retract_rate_gain": 7.0,
        "height_retract_feedforward_scale": 1.0,
        "leg_feedforward_scale": 1.0,
        "coord_residual_scale": (0.01, 0.002, 0.0),
        "log_interval": 10,
        "save_interval": 200,
    },
    "v31": {
        **TRAINING_PRESETS["leg-length"],
        "checkpoint": str(
            REPO_ROOT / "checkpoints" / "ppo_stand_height_ramped_v4" / "checkpoint"
        ),
        "updates": 1000,
        "version": "v31_leg_tracking_residual",
        "out": str(REPO_ROOT / "checkpoints" / "v31_leg_tracking_residual" / "checkpoint"),
        "tensorboard_log": str(REPO_ROOT / "runs" / "v31_leg_tracking_residual"),
        # 低探索、低学习率，从稳定站立策略学习小幅腿关节残差。
        "learning_rate": 2.0e-5,
        "ppo_epochs": 2,
        "clip_range": 0.04,
        "target_kl": 0.003,
        "initial_log_std": -3.5,
        "log_std_min": -4.0,
        "log_std_max": -3.2,
        "freeze_actor_backbone": False,
        "reset_leg_action_head": True,
        "init_scale_start": 0.10,
        "init_scale_end": 0.60,
        "height_switch_steps": 500,
        "height_target_rate": 0.08,
        "height_rate_limit": 0.35,
        "height_rate_gain": 7.0,
        "height_rate_damping": 1.0,
        "height_rate_brake_threshold": 0.005,
        "height_retract_rate_limit": 0.28,
        "height_retract_slow_rate_limit": 0.14,
        "height_retract_rate_gain": 7.0,
        "height_retract_feedforward_scale": 1.0,
        "leg_feedforward_scale": 1.0,
        # 前 10% 训练仅给极小腿残差，随后平滑开放到 0.12。
        # 0.12 × 0.35 rad = 0.042 rad（约 2.4°），符合实物安全余量。
        "coord_residual_start": (0.005, 0.001, 0.01),
        "coord_residual_scale": (0.015, 0.003, 0.12),
        "coord_residual_warmup": 0.10,
        "leg_residual_slew_rate": 0.30,
        # 给 PPO 明确的腿长收敛与目标附近制动梯度。
        "leg_length_reward_weight": 4.0,
        "leg_length_progress_reward_weight": 120.0,
        "leg_length_rate_reward_weight": 0.25,
        "log_interval": 10,
        "save_interval": 200,
    },
    "v32": {
        **TRAINING_PRESETS["leg-length"],
        # v31 暴露了两个语义错位：动作奖励在惩罚控制器基线，
        # 且 4 个独立腿残差会注入俯仰/左右非对称模式。v32 从原始
        # 稳定 checkpoint 重新分叉，只学实物可执行的对称腿长差模。
        "checkpoint": str(
            REPO_ROOT / "checkpoints" / "ppo_stand_height_ramped_v4" / "checkpoint"
        ),
        "updates": 1000,
        "version": "v32_leg_diff_residual",
        "out": str(REPO_ROOT / "checkpoints" / "v32_leg_diff_residual" / "checkpoint"),
        "tensorboard_log": str(REPO_ROOT / "runs" / "v32_leg_diff_residual"),
        "learning_rate": 5.0e-6,
        "ppo_epochs": 1,
        "clip_range": 0.02,
        "target_kl": 0.001,
        "initial_log_std": -3.7,
        "log_std_min": -4.2,
        "log_std_max": -3.5,
        "freeze_actor_backbone": True,
        "reset_leg_action_head": True,
        "train_leg_diff_only": True,
        "project_leg_length_residual": True,
        "init_scale_start": 0.10,
        "init_scale_end": 0.60,
        "height_switch_steps": 300,
        "height_target_rate": 0.06,
        "height_rate_limit": 0.35,
        "height_rate_gain": 7.0,
        "height_rate_damping": 1.0,
        "height_rate_brake_threshold": 0.005,
        "height_retract_rate_limit": 0.28,
        "height_retract_slow_rate_limit": 0.14,
        "height_retract_rate_gain": 7.0,
        "height_retract_feedforward_scale": 1.0,
        "leg_feedforward_scale": 1.0,
        # 200 轮隔离试验表明：即使 learning_rate=0，动态放权到
        # 0.046 也会在“目标切换 + 域随机化”下引发倾角终止。因此本版
        # 固定在已验证的 0.008，不在同一次训练中自动扩权。
        "coord_residual_start": (0.005, 0.001, 0.008),
        "coord_residual_scale": (0.005, 0.001, 0.008),
        "coord_residual_warmup": 0.0,
        "leg_residual_slew_rate": 0.20,
        "leg_length_reward_weight": 8.0,
        "leg_length_progress_reward_weight": 120.0,
        "leg_length_rate_reward_weight": 0.50,
        "log_interval": 10,
        "save_interval": 200,
    },
}

# v32 的策略约束本身通过了独立回放，但训练环境的执行器域随机化
# 在 reset 间累乘，导致约 60 轮后被控对象参数漂移。v33 在修复为“每局
# 始终相对不变标称值采样”后，从原始稳定 checkpoint 重新训练。
TRAINING_PROFILES["v33"] = {
    **TRAINING_PROFILES["v32"],
    "version": "v33_leg_diff_dr_fixed",
    "out": str(REPO_ROOT / "checkpoints" / "v33_leg_diff_dr_fixed" / "checkpoint"),
    "tensorboard_log": str(REPO_ROOT / "runs" / "v33_leg_diff_dr_fixed"),
}

# v34: v33 已证明环境和基线可稳定长时间运行，但 0.008 残差权限太小，
# PPO 几乎没有足够的跟踪动作空间。在修复后的域随机化环境中已压测
# 0.02 残差的对称维度可存活，本版只提高腿长残差和目标斜率，其余约束不变。
TRAINING_PROFILES["v34"] = {
    **TRAINING_PROFILES["v33"],
    "version": "v34_leg_diff_tracking",
    "out": str(REPO_ROOT / "checkpoints" / "v34_leg_diff_tracking" / "checkpoint"),
    "tensorboard_log": str(REPO_ROOT / "runs" / "v34_leg_diff_tracking"),
    "coord_residual_start": (0.005, 0.001, 0.012),
    "coord_residual_scale": (0.005, 0.001, 0.020),
    "height_target_rate": 0.08,
}

# v35: 以 v34 为起点，统一到网页回放已经验证过的动态腿长过渡配置。
# 关键变化是目标斜坡更快但收腿前馈受限；balance.py 还会在腿长实际运动时
# 临时提高位置/速度保持增益，减少差模伸缩给机体带来的前后漂移。
TRAINING_PROFILES["v35"] = {
    **TRAINING_PROFILES["v34"],
    "version": "v35_height_transition_hold",
    "out": str(REPO_ROOT / "checkpoints" / "v35_height_transition_hold" / "checkpoint"),
    "tensorboard_log": str(REPO_ROOT / "runs" / "v35_height_transition_hold"),
    "height_switch_steps": 500,
    "height_target_rate": 0.25,
    "height_rate_limit": 0.35,
    "height_rate_gain": 20.0,
    "height_retract_rate_limit": 0.35,
    "height_retract_slow_rate_limit": 0.20,
    "height_retract_rate_gain": 30.0,
    "height_retract_feedforward_scale": 0.25,
    "leg_feedforward_scale": 1.0,
}

# v36: 域随机化下的低腿长保护版。
# balance.py 已修复“误差穿过死区时旧收腿积分不清零”的问题；本版再把
# 目标切换间隔拉到 700 步，给 0.15~0.27 m 的反向过渡留出稳定时间，避免
# PPO 在上一段尚未收敛时又学到相反方向的残差。速度阻尼显式固定为 1.0，
# 与实物部署/网页回放保持同一条安全链路。
TRAINING_PROFILES["v36"] = {
    **TRAINING_PROFILES["v35"],
    "version": "v36_height_dr_guard",
    "out": str(REPO_ROOT / "checkpoints" / "v36_height_dr_guard" / "checkpoint"),
    "tensorboard_log": str(REPO_ROOT / "runs" / "v36_height_dr_guard"),
    "height_switch_steps": 700,
    "height_target_rate": 0.25,
    "height_rate_damping": 1.0,
    "height_retract_rate_limit": 0.35,
    "height_retract_slow_rate_limit": 0.20,
    "height_retract_rate_gain": 30.0,
    "height_retract_feedforward_scale": 0.25,
    "leg_feedforward_scale": 1.0,
}

# v37: 高度带宽/快速反向课程。
# v36 已经验证低腿长保护逻辑，但 700 步切换间隔没有覆盖网页端的连续拖动：
# 上一段腿长还没有收敛，目标就可能再次反向。v37 保持 v36 的安全差模速率和
# 小策略残差，只把目标切换压到 250 步（约 2.0 s），并把收腿低速区从默认
# 60 mm 收紧到 30 mm，逐步逼近部署端的快速回放 profile。这里暂不加入前后
# 平移命令，避免把高度过渡漂移和车体位置漂移混成一个训练目标。
TRAINING_PROFILES["v37_height_bandwidth"] = {
    **TRAINING_PROFILES["v36"],
    "checkpoint": str(
        REPO_ROOT / "checkpoints" / "v36_height_dr_guard" / "checkpoint"
    ),
    "updates": 800,
    "version": "v37_height_bandwidth",
    "out": str(REPO_ROOT / "checkpoints" / "v37_height_bandwidth" / "checkpoint"),
    "tensorboard_log": str(REPO_ROOT / "runs" / "v37_height_bandwidth"),
    "learning_rate": 3.0e-6,
    "height_switch_steps": 250,
    "height_switch_prob": 1.0,
    "height_target_rate": 0.25,
    "height_rate_limit": 0.35,
    "height_rate_gain": 20.0,
    "height_rate_damping": 1.0,
    "height_rate_brake_threshold": 0.005,
    "height_retract_rate_limit": 0.35,
    "height_retract_slow_rate_limit": 0.20,
    "height_retract_slow_error_m": 0.030,
    "height_retract_rate_gain": 30.0,
    "height_retract_feedforward_scale": 0.10,
    "leg_feedforward_scale": 1.0,
    "coord_residual_start": (0.005, 0.001, 0.008),
    "coord_residual_scale": (0.005, 0.001, 0.020),
    "coord_residual_warmup": 0.0,
    "leg_residual_slew_rate": 0.20,
    "log_interval": 10,
    "save_interval": 200,
}

# v38: 高度速度专项课程。
# v37 在动态切换下已经稳定，但腿长误差 p95 仍然较大，且策略腿残差几乎为
# 0，说明当前安全控制器承担了全部跟踪任务。v38 只做一次中等幅度增速：把
# 高度差模速率从 0.35 提到 0.50 action/s，把投影后的腿长残差预算从 0.020
# 提到 0.030；仍保持 250 步切换、低腿长保护、速度阻尼和左右对称投影，不把
# 1 秒全幅变化作为硬目标。若 v38 仍在扭矩饱和下无法降低误差，应转为实物
# 扭矩/加速度标定，而不是继续放大 PPO 权限。
TRAINING_PROFILES["v38_height_speed"] = {
    **TRAINING_PROFILES["v37_height_bandwidth"],
    "checkpoint": str(
        REPO_ROOT / "checkpoints" / "v37_height_bandwidth" / "checkpoint"
    ),
    "updates": 1000,
    "version": "v38_height_speed",
    "out": str(REPO_ROOT / "checkpoints" / "v38_height_speed" / "checkpoint"),
    "tensorboard_log": str(REPO_ROOT / "runs" / "v38_height_speed"),
    "learning_rate": 2.0e-6,
    "height_switch_steps": 250,
    "height_target_rate": 0.25,
    "height_rate_limit": 0.50,
    "height_rate_gain": 25.0,
    "height_rate_damping": 1.0,
    "height_rate_brake_threshold": 0.005,
    "height_retract_rate_limit": 0.50,
    "height_retract_slow_rate_limit": 0.30,
    "height_retract_slow_error_m": 0.030,
    "height_retract_rate_gain": 40.0,
    "height_retract_feedforward_scale": 0.10,
    "leg_feedforward_scale": 1.0,
    "coord_residual_start": (0.005, 0.001, 0.008),
    "coord_residual_scale": (0.005, 0.001, 0.030),
    "coord_residual_warmup": 0.0,
    "leg_residual_slew_rate": 0.30,
    "log_interval": 10,
    "save_interval": 200,
}

# v39: 远离目标时允许策略加速、接近目标时保留制动带。
# v38 仍把“腿已朝目标运动”整体视为制动状态，导致策略腿长动作几乎为零，
# 动态腿长误差没有继续下降。v39 只在误差大于 15 mm 时开放符号安全的策略
# 残差；进入 ±15 mm 后恢复原有制动保护。这样不会允许策略把腿推离目标，
# 但给 PPO 一个真实的“加快大误差收敛”梯度。
TRAINING_PROFILES["v39_height_residual_accel"] = {
    **TRAINING_PROFILES["v38_height_speed"],
    "checkpoint": str(
        REPO_ROOT / "checkpoints" / "v38_height_speed" / "checkpoint"
    ),
    "updates": 1000,
    "version": "v39_height_residual_accel",
    "out": str(REPO_ROOT / "checkpoints" / "v39_height_residual_accel" / "checkpoint"),
    "tensorboard_log": str(REPO_ROOT / "runs" / "v39_height_residual_accel"),
    "learning_rate": 2.0e-6,
    "coord_residual_start": (0.005, 0.001, 0.008),
    "coord_residual_scale": (0.005, 0.001, 0.035),
    "coord_residual_warmup": 0.0,
    "leg_residual_slew_rate": 0.30,
    "leg_residual_brake_error_m": 0.015,
    "log_interval": 10,
    "save_interval": 200,
}


def apply_training_preset(args: argparse.Namespace, argv: list[str]) -> None:
    """应用命名训练预设，但不覆盖用户在命令行中显式给出的选项。"""
    config = (TRAINING_PROFILES[args.profile] if args.profile else
              TRAINING_PRESETS[args.preset] if args.preset else None)
    if config is None:
        return
    supplied = {item.split("=", 1)[0] for item in argv if item.startswith("--")}
    for dest, value in config.items():
        option = "--" + dest.replace("_", "-")
        if option not in supplied:
            setattr(args, dest, value)



# ==========================================================================
# 非对称 actor-critic：actor 只吃实机可测量，critic 额外吃仿真真值
# ==========================================================================
from stable_baselines3.common.policies import ActorCriticPolicy


class AsymmetricMlpExtractor(nn.Module):
    """actor 只用观测前 ``actor_dim`` 维；critic 用全部观测。"""

    def __init__(self, actor_dim, observation_dim, net_arch, activation_fn):
        super().__init__()
        self.actor_dim = int(actor_dim)
        pi_layers = list(net_arch.get("pi", [64, 64]))
        vf_layers = list(net_arch.get("vf", [64, 64]))
        self.policy_net = self._build(self.actor_dim, pi_layers, activation_fn)
        self.value_net = self._build(observation_dim, vf_layers, activation_fn)
        self.latent_dim_pi = pi_layers[-1]
        self.latent_dim_vf = vf_layers[-1]

    @staticmethod
    def _build(in_dim, layers, activation_fn):
        modules, last = [], int(in_dim)
        for hidden in layers:
            modules += [nn.Linear(last, hidden), activation_fn()]
            last = hidden
        return nn.Sequential(*modules)

    def forward(self, obs):
        return self.policy_net(obs[..., : self.actor_dim]), self.value_net(obs)

    def forward_actor(self, obs):
        return self.policy_net(obs[..., : self.actor_dim])

    def forward_critic(self, obs):
        return self.value_net(obs)


class AsymmetricActorCriticPolicy(ActorCriticPolicy):
    """SB3 PPO 的非对称策略：critic 看到 critic 专属的特权观测。"""

    def __init__(self, *args, actor_dim=None, **kwargs):
        if actor_dim is None:
            raise ValueError("需要 actor_dim（actor 可见的观测维数）")
        self._actor_dim = int(actor_dim)
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = AsymmetricMlpExtractor(
            self._actor_dim, self.features_dim, self.net_arch, self.activation_fn
        )


# 旧的 73 维契约 → 新的 71 维。（块名从 attitude 拆成 gravity+yaw）
#   旧: attitude[0:3]=gravity [3:6]=roll,pitch,yaw | lin_vel[6:9] | ang_vel[9:12]
#       pos_rel[12:14] | joint_pos[14:18] | joint_vel[18:22] | wheel_vel[22:24]
#       command[24:29] | prev_action[29:39] | leg_length[39:43] | contact[43:49]
#       terrain[49:66] | phase[66:68] | mode[68:73]
OLD_OBS_DIM = 73   # 拆分前的对称观测维数
OLD_V73_TO_V71 = (
    list(range(0, 3)) + [5]                       # gravity + yaw
    + list(range(9, 12))                          # base_ang_vel
    + list(range(14, 18)) + list(range(18, 22))   # joint pos / vel
    + list(range(22, 24))                         # wheel vel
    + list(range(24, 29))                         # command
    + list(range(29, 39))                         # previous_action
    + list(range(66, 68)) + list(range(68, 73))   # phase + mode      -> actor 0:39
    + list(range(6, 9)) + list(range(12, 14))     # lin_vel + pos_rel
    + list(range(39, 43)) + list(range(43, 49))   # leg_length + contact
    + list(range(49, 66))                         # terrain_scan      -> priv 39:71
)


def load_and_migrate_v73(path, env, ppo_kwargs):
    """把旧 73 维对称 checkpoint 读进 39/71 的非对称模型。

    旧 checkpoint 的 policy 是普通 ``ActorCriticPolicy``（actor/critic 共用同一份观测），
    所以不能只换权重 —— 必须**用新的策略类重建模型**，再按列选择拷贝权重。
    整个过程是精确的列选择，**不需要重新训练**。
    """
    old = PPO.load(str(path), device="cpu")
    if int(old.observation_space.shape[0]) != OLD_OBS_DIM:
        return None
    ppo_kwargs = dict(ppo_kwargs)
    ppo_kwargs["policy_kwargs"] = dict(
        ppo_kwargs.get("policy_kwargs", {}), actor_dim=ACTOR_OBS_DIM
    )
    new = PPO(AsymmetricActorCriticPolicy, env, **ppo_kwargs)
    with torch.no_grad():
        for name, mapping in (("policy_net", OLD_V73_TO_V71[:ACTOR_OBS_DIM]),
                              ("value_net", OLD_V73_TO_V71)):
            old_net = getattr(old.policy.mlp_extractor, name)
            new_net = getattr(new.policy.mlp_extractor, name)
            # 第一层：按列选择（输入维度变了）
            first = new_net[0]
            first.weight.copy_(old_net[0].weight.data.index_select(
                1, torch.as_tensor(mapping, dtype=torch.long)))
            if first.bias is not None:
                first.bias.copy_(old_net[0].bias.data)
            # 其余隐藏层维度不变，直接整体拷贝
            for index in range(1, len(old_net)):
                new_net[index].load_state_dict(old_net[index].state_dict())
        new.policy.action_net.load_state_dict(old.policy.action_net.state_dict())
        new.policy.value_net.load_state_dict(old.policy.value_net.state_dict())
        new.policy.log_std.data.copy_(old.policy.log_std.data)
    print(f"[migrate] 73 维对称 → actor {ACTOR_OBS_DIM} / critic {OBS_DIM} 非对称（精确列选择）",
          flush=True)
    return new


def _obs_space(dim):
    import gymnasium.spaces as spaces_
    return spaces_.Box(-np.inf, np.inf, (int(dim),), np.float32)


def make_env(stage: str, rank: int, seed: int, stand_level: int, assist: float,
             init_scale: float = 1.0, lock_stand_leg_actions: bool = False,
             stand_leg_action_limit: float = 1.0, coord_mix: float | None = None,
             coord_residual_scale: tuple[float, float, float] = (0.05, 0.02, 0.05),
             coord_channels: str = "all", height_switch_steps: int = 0,
             height_switch_prob: float = 0.0, height_rate_limit: float | None = None,
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
             vx_range: tuple[float, float] | None = None,
             zero_command_prob: float | None = None,
             reverse_prob: float | None = None):
    def _init():
        env = UZ05Env(stage=stage, seed=seed + rank, stand_level=stand_level,
                      assist_scale=assist, init_scale=init_scale,
                      lock_stand_leg_actions=lock_stand_leg_actions,
                      stand_leg_action_limit=stand_leg_action_limit,
                      coord_mix=coord_mix,
                      coord_residual_scale=coord_residual_scale,
                      coord_channels=coord_channels,
                      height_switch_steps=height_switch_steps,
                      height_switch_prob=height_switch_prob,
                      height_rate_limit=height_rate_limit,
                      height_rate_gain=height_rate_gain,
                      height_rate_damping=height_rate_damping,
                      height_retract_rate_damping=height_retract_rate_damping,
                      height_low_target_brake_damping_scale=(
                          height_low_target_brake_damping_scale),
                      height_rate_brake_threshold=height_rate_brake_threshold,
                      height_reference_jump_reset_m=height_reference_jump_reset_m,
                      height_brake_error_m=height_brake_error_m,
                      height_target_rate_feedforward_scale=(
                          height_target_rate_feedforward_scale),
                      height_hold_error_m=height_hold_error_m,
                      height_hold_rate_limit=height_hold_rate_limit,
                      height_hold_rate_gain=height_hold_rate_gain,
                      height_filter_alpha=height_filter_alpha,
                      height_retract_rate_limit=height_retract_rate_limit,
                      height_retract_slow_rate_limit=height_retract_slow_rate_limit,
                      height_retract_slow_error_m=height_retract_slow_error_m,
                      height_retract_rate_gain=height_retract_rate_gain,
                      height_retract_feedforward_scale=height_retract_feedforward_scale,
                      leg_feedforward_scale=leg_feedforward_scale,
                      height_target_rate_m_s=height_target_rate_m_s,
                      leg_residual_slew_rate=leg_residual_slew_rate,
                      leg_residual_brake_error_m=leg_residual_brake_error_m,
                      project_leg_length_residual=project_leg_length_residual,
                      leg_length_reward_weight=leg_length_reward_weight,
                      leg_length_progress_reward_weight=leg_length_progress_reward_weight,
                      leg_length_rate_reward_weight=leg_length_rate_reward_weight,
                      vx_range_override=vx_range,
                      zero_command_prob_override=zero_command_prob,
                      reverse_prob_override=reverse_prob)
        env.reset(seed=seed + rank)
        return env
    return _init


def write_contract(path: str | Path, stage: str, total_updates: int,
                   version: str = "", deployment: dict | None = None) -> Path:
    """写出接口契约（维度固定，因此所有阶段共享同一份接口）。"""
    path = Path(path)
    if path.suffix == ".zip":
        path = path.with_suffix("")
    contract = {
        "contract_version": CONTRACT_VERSION,
        "stage": stage,
        "version": version,
        "checkpoint_root": "checkpoints",
        "stage_order": list(CAPABILITY_ORDER),
        "observation_shape": OBS_DIM,
        "action_shape": ACTION_DIM,
        "interface_frozen": True,
        "note": (
            "观测与动作维度在所有 capability 阶段固定不变，"
            "checkpoint 可在任意阶段之间直接继承，无需迁移。"
        ),
        "total_updates": int(total_updates),
    }
    if deployment:
        contract["deployment"] = deployment
    target = Path(f"{path}_contract.json")
    target.write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n")
    return target


class ProgressLogger(BaseCallback):
    """默认每轮只打印少量摘要；``--log-verbose`` 打印完整诊断量。"""

    # 核心量：(显示名, info 键, 格式)
    CORE = (
        ("assist_scale",   "assist_scale",         ".3f"),
        ("coord_mix",      "_coord_mix",           ".3f"),
        ("leg_residual_scale", "_leg_residual_scale", ".3f"),
        ("leg_residual_holding", "_leg_residual_holding", ".0f"),
        ("init_scale",     "_init_scale",          ".2f"),
        ("anneal_progress", "_anneal",             ".2f"),
        ("assist_holding", "_holding",             ".0f"),
        ("roll_rad",       "roll",                 "+.4f"),
        ("pitch_rad",      "pitch",                "+.4f"),
        ("height_m",       "base_height",          ".4f"),
        ("vx_body_m_s",    "body_vx",              "+.4f"),
        ("command_vx_m_s", "command_vx",           "+.4f"),
        ("station_err_m",  "station_error",        "+.4f"),
        ("station_tail_m", "station_tail_abs",     ".4f"),
        ("within_5cm",     "station_within_5cm",   ".0f"),
        ("leg_length_L_m", "leg_length_left",      ".4f"),
        ("leg_length_R_m", "leg_length_right",     ".4f"),
        ("leg_target_m",   "leg_length_target",    ".4f"),
        ("leg_error_mm",   "leg_length_error_mm",  "+.1f"),
        ("height_switches", "height_switch_count", ".0f"),
        ("height_settle_steps", "height_settle_steps", ".0f"),
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
        ("balance_torque",   "balance_torque",       "+.4f"),
        ("wheel_force_L_N",  "wheel_force_left",     ".2f"),
        ("wheel_force_R_N",  "wheel_force_right",    ".2f"),
        ("body_contact",     "body_contact",         ".0f"),
        ("mode",             "mode",                 "s"),
    )

    REWARD_TOP = 3        # 只打印贡献最大的几项

    def __init__(self, stage, save_prefix, save_interval, version="",
                 assist_start=None, assist_end=None, verbose=False,
                 log_interval=1, total_updates=0, adaptive=False,
                 hold_below=0.85, resume_above=0.95, adaptive_lr=True,
                 log_std_min=-2.5, log_std_max=-1.5,
                 init_scale_start=1.0, init_scale_end=1.0,
                 coord_start=0.0, coord_end=0.0,
                 coord_residual_start=None, coord_residual_end=None,
                 coord_residual_warmup=0.0):
        super().__init__()
        self.stage = stage
        self.save_prefix = save_prefix
        self.save_interval = max(1, int(save_interval))
        self.version = version
        self.assist_start = assist_start
        self.assist_end = assist_end
        self.assist_now = assist_start
        # 自适应退火：存活率过低时冻结退火进度，恢复后再继续。
        # 这是"辅助太强→策略没压力"与"退火太快→策略崩"之间的唯一平衡点。
        self.adaptive = adaptive
        self.hold_below = hold_below
        self.resume_above = resume_above
        self.anneal_progress = 0.0
        self.holding = False
        self.verbose = verbose
        self.log_interval = max(1, int(log_interval))
        self.total_updates = max(1, int(total_updates or 1))
        self.term_counts: dict[str, int] = {}
        self.last_info: dict = {}
        self._start = 0.0
        # 验收统计：滚动窗口内的 episode 结果
        #   (是否活满整集, 末段稳态漂移, 末端倾角, 末端高度, 全程峰值漂移)
        self.episodes: deque = deque(maxlen=50)
        # RSL-RL 自适应学习率（对齐开源 desired_kl=0.01）
        self.desired_kl = 0.01
        self.lr_min, self.lr_max = 1.0e-5, 2.0e-4
        self.adaptive_lr = bool(adaptive_lr)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.init_scale_start = float(init_scale_start)
        self.init_scale_end = float(init_scale_end)
        self.init_scale_now = self.init_scale_start
        # 腿+轮协同平衡控制器的混合课程（1 = 控制器全权 → 0 = 完全靠策略）
        self.coord_start = None if coord_start is None else float(coord_start)
        self.coord_end = None if coord_end is None else float(coord_end)
        self.coord_now = self.coord_start
        self.coord_residual_start = (None if coord_residual_start is None else
                                     np.asarray(coord_residual_start, dtype=np.float64))
        self.coord_residual_end = (None if coord_residual_end is None else
                                   np.asarray(coord_residual_end, dtype=np.float64))
        self.coord_residual_warmup = float(np.clip(coord_residual_warmup, 0.0, 0.95))
        self.coord_residual_now = (None if self.coord_residual_start is None else
                                   self.coord_residual_start.copy())
        self.coord_residual_progress = 0.0
        self.coord_residual_holding = False
        self.best_drift_cm = float("nan")
        self.best_score = float("inf")
        self.best_paths: tuple[str, str] | None = None
        self._pitch_sq: deque = deque(maxlen=2000)
        self._prate_sq: deque = deque(maxlen=2000)
        self._accel_sq: deque = deque(maxlen=2000)

    def _on_training_start(self) -> None:
        import time
        self._start = time.perf_counter()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos") or []
        dones = self.locals.get("dones")
        for index, info in enumerate(infos):
            reason = info.get("termination_reason", "running")
            if reason not in ("running", "time_limit"):
                self.term_counts[reason] = self.term_counts.get(reason, 0) + 1
            self.last_info = info
            # 只在该 episode 真正结束时记录一次
            if dones is not None and index < len(dones) and bool(dones[index]):
                survived = reason == "time_limit"
                tail = float(info.get("station_tail_mean", float("nan")))
                tilt = max(abs(float(info.get("roll", 99.0))),
                           abs(float(info.get("pitch", 99.0))))
                height = float(info.get("base_height", 0.0))
                peak = float(info.get("station_max_abs", float("nan")))
                self.episodes.append((survived, tail, tilt, height, peak))
            # 站立质量统计（滚动窗口）：pitch 振荡 / pitch 速率 / 前后加速度
            pitch = float(info.get("pitch", 0.0))
            self._pitch_sq.append(pitch * pitch)
            self._prate_sq.append(float(info.get("pitch_rate", 0.0)) ** 2)
            accel = float(info.get("body_accel", 0.0))
            self._accel_sq.append(accel * accel)
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

    def _apply_assist_anneal(self) -> None:
        """每轮推进一次退火（必须在 rollout 边界调用，不能放在 _on_step）。

        PPO-only 训练不使用教师跟踪或辅助退火；保留此回调是为兼容旧命令，
        实际训练时辅助强度必须为 0。
        """
        if self.assist_start is None or self.assist_end is None:
            return
        if self.adaptive and self.episodes:
            survive = float(np.mean([item[0] for item in self.episodes]))
            step = 1.0 / max(1, self.total_updates)
            if survive < self.hold_below:
                # PPO-only 时这个辅助值始终为 0，这个分支只为兼容旧命令。
                self.holding = True
                self.anneal_progress = max(0.0, self.anneal_progress - 2.0 * step)
            elif survive > self.resume_above:
                self.holding = False
        if not self.holding:
            self.anneal_progress = min(
                1.0, self.anneal_progress + 1.0 / max(1, self.total_updates)
            )
        value = self.assist_start + (self.assist_end - self.assist_start) * self.anneal_progress
        envs = self.training_env
        if hasattr(envs, "env_method"):
            envs.env_method("set_assist_scale", float(value))
        self.assist_now = float(value)

    def _apply_init_curriculum(self) -> None:
        progress = min(1.0, self.num_timesteps / max(
            1, self.total_updates * self.training_env.num_envs * self.model.n_steps
        ))
        value = self.init_scale_start + (
            self.init_scale_end - self.init_scale_start
        ) * progress
        if hasattr(self.training_env, "env_method"):
            self.training_env.env_method("set_init_scale", float(value))
        self.init_scale_now = float(value)

    def _apply_coord_curriculum(self) -> None:
        """腿+轮协同控制器的退火：控制器先稳住机体，策略逐步接管。

        与 assist 退火的关键区别：这里是**均匀退火**，不看存活率。因为协同
        控制器本身是稳定的（实测 1000/1000 步站定、pitch RMS 0.1°），退火
        过程中策略看到的是一个始终可控的残差任务，不需要"冻结进度"保护。
        """
        if self.coord_start is None or self.coord_end is None:
            return
        progress = min(1.0, self.num_timesteps / max(
            1, self.total_updates * self.training_env.num_envs * self.model.n_steps
        ))
        value = self.coord_start + (self.coord_end - self.coord_start) * progress
        if hasattr(self.training_env, "env_method"):
            self.training_env.env_method("set_coord_mix", float(value))
        self.coord_now = float(value)

    def _apply_residual_curriculum(self) -> None:
        """逐步开放策略残差，避免未训练腿动作直接冲击实物等效机构。"""
        if self.coord_residual_start is None or self.coord_residual_end is None:
            return
        progress = min(1.0, self.num_timesteps / max(
            1, self.total_updates * self.training_env.num_envs * self.model.n_steps
        ))
        if progress <= self.coord_residual_warmup:
            alpha = 0.0
            self.coord_residual_holding = True
        else:
            # 权限只在最近 episode 足够安全时增加；存活率跌破 85% 时回退。
            # 这与实物分级放权一致：策略没有先证明稳定，就不会得到更大的
            # 关节修正权限。
            survive = (float(np.mean([item[0] for item in self.episodes]))
                       if self.episodes else 0.0)
            step = 1.0 / max(
                1.0, self.total_updates * (1.0 - self.coord_residual_warmup)
            )
            if survive >= 0.95:
                self.coord_residual_progress = min(
                    1.0, self.coord_residual_progress + step
                )
                self.coord_residual_holding = False
            elif survive < 0.90:
                self.coord_residual_progress = max(
                    0.0, self.coord_residual_progress - 2.0 * step
                )
                self.coord_residual_holding = True
            else:
                self.coord_residual_holding = True
            alpha = self.coord_residual_progress
        value = self.coord_residual_start + (
            self.coord_residual_end - self.coord_residual_start
        ) * float(np.clip(alpha, 0.0, 1.0))
        if hasattr(self.training_env, "env_method"):
            self.training_env.env_method(
                "set_coord_residual_scale", tuple(float(v) for v in value)
            )
        self.coord_residual_now = value.copy()

    def _on_rollout_end(self) -> None:
        import time
        # 闭链腿机构的高幅高斯探索会直接破坏本已稳定的辅助平衡解。
        if self.model.policy.log_std is not None:
            self.model.policy.log_std.data.clamp_(self.log_std_min, self.log_std_max)
        self._apply_assist_anneal()
        self._apply_init_curriculum()
        self._apply_coord_curriculum()
        self._apply_residual_curriculum()
        self._adapt_learning_rate()
        update = self.num_timesteps // max(1, self.locals.get("n_steps", 1) * self.training_env.num_envs)
        if update % self.log_interval:
            if self.save_prefix and update and update % self.save_interval == 0:
                self.model.save(f"{self.save_prefix}_iter_{update}")
                if isinstance(self.training_env, VecNormalize):
                    self.training_env.save(f"{self.save_prefix}_iter_{update}_vecnormalize.pkl")
            return
        info = dict(self.last_info or {})
        info["stage"] = self.stage
        if self.assist_now is not None:
            info["assist_scale"] = self.assist_now
        info["_anneal"] = self.anneal_progress
        info["_holding"] = 1.0 if self.holding else 0.0
        info["_init_scale"] = self.init_scale_now
        if self.coord_now is not None:
            info["_coord_mix"] = self.coord_now
        if self.coord_residual_now is not None:
            info["_leg_residual_scale"] = float(self.coord_residual_now[2])
            info["_leg_residual_holding"] = (
                1.0 if self.coord_residual_holding else 0.0
            )
        # 派生量：关节统计合成一行，不再逐关节打印
        try:
            info["_hip_pos_abs"] = float(np.abs([info[f"hip_pos_{i}"] for i in range(4)]).mean())
            info["_hip_vel_abs"] = float(np.abs([info[f"hip_vel_{i}"] for i in range(4)]).mean())
        except KeyError:
            pass

        elapsed = time.perf_counter() - self._start
        episodes = list(self.model.ep_info_buffer)
        if self.verbose:
            print(f"iter: {update}/{self.total_updates}   "
                  f"fps: {self.num_timesteps / elapsed:.0f}   elapsed_s: {elapsed:.0f}",
                  flush=True)
            for label, key, fmt in self.CORE + self.EXTRA:
                if key not in info:
                    continue
                value = info[key]
                try:
                    print(f"{label}: {value if fmt == 's' else format(float(value), fmt)}",
                          flush=True)
                except (TypeError, ValueError):
                    print(f"{label}: {value}", flush=True)
        else:
            leg_mean = 0.5 * (
                float(info.get("leg_length_left", float("nan")))
                + float(info.get("leg_length_right", float("nan")))
            )
            print(f"iter: {update}/{self.total_updates}", flush=True)
            print(f"fps: {self.num_timesteps / elapsed:.0f}", flush=True)
            print(f"elapsed_s: {elapsed:.0f}", flush=True)
            print(f"leg_actual_m: {leg_mean:.4f}", flush=True)
            print(f"leg_target_m: {float(info.get('leg_length_target', float('nan'))):.4f}",
                  flush=True)
            print(f"leg_error_mm: "
                  f"{float(info.get('leg_length_error_mm', float('nan'))):+.1f}", flush=True)
            print(f"pitch_deg: "
                  f"{np.degrees(float(info.get('pitch', float('nan')))):+.2f}", flush=True)
            print(f"drift_cm: "
                  f"{100.0 * float(info.get('station_error', float('nan'))):+.2f}", flush=True)
            print(f"height_m: {float(info.get('base_height', float('nan'))):.3f}", flush=True)
            if self.coord_residual_now is not None:
                print(f"leg_residual_scale: {float(self.coord_residual_now[2]):.3f}",
                      flush=True)
                print(f"leg_residual_holding: "
                      f"{1 if self.coord_residual_holding else 0}", flush=True)
        # ---------------- 验收指标 ----------------
        survive_rate = float("nan")
        drift_cm = float("nan")
        upright_rate = float("nan")
        pitch_rms = float("nan")
        accel_rms = float("nan")
        peak_ok_rate = float("nan")
        ok = False
        if self.episodes:
            survived = [item[0] for item in self.episodes]
            survive_rate = float(np.mean(survived))
            tails = [item[1] for item in self.episodes if item[0] and np.isfinite(item[1])]
            drift_cm = float(np.mean(tails) * 100.0) if tails else float("nan")
            if self.verbose:
                print(f"survive_rate: {survive_rate:.2f}   "
                      f"({int(np.sum(survived))}/{len(survived)} 活满整集)", flush=True)
                if tails:
                    print(f"drift_tail_cm: {drift_cm:.2f}   "
                          f"(仅统计活满整集, n={len(tails)})", flush=True)
                else:
                    print("drift_tail_cm: n/a   (本窗口没有活满整集的 episode)",
                          flush=True)
            upright = [item[2] <= 0.30 for item in self.episodes]
            upright_rate = float(np.mean(upright))
            final_tilt_deg = float(np.degrees(np.mean([item[2] for item in self.episodes])))
            height_ok = [item[3] >= 0.80 * 0.23825 for item in self.episodes]
            peaks = [item[4] for item in self.episodes if item[0] and np.isfinite(item[4])]
            peak_ok_rate = float(np.mean(np.asarray(peaks) < 0.05)) if peaks else 0.0
            if self.verbose:
                print(f"upright_rate: {upright_rate:.2f}   "
                      f"final_tilt_mean_deg: {final_tilt_deg:.2f}", flush=True)
            if self._pitch_sq:
                pitch_rms = float(np.degrees(np.sqrt(np.mean(self._pitch_sq))))
                prate_rms = float(np.sqrt(np.mean(self._prate_sq)))
                accel_rms = float(np.sqrt(np.mean(self._accel_sq)))
                if self.verbose:
                    print(f"pitch_rms_deg: {pitch_rms:.3f}   pitch_rate_rms: {prate_rms:.4f}   "
                          f"body_accel_rms: {accel_rms:.3f}", flush=True)
            assist = info.get("assist_scale")
            if assist is not None:
                ok = (survive_rate >= 0.95 and upright_rate >= 0.95
                      and peak_ok_rate >= 0.95
                      and np.isfinite(drift_cm) and drift_cm < 5.0
                      and float(np.mean(height_ok)) >= 0.95 and assist <= 1e-3)
                if self.verbose:
                    print(f"peak_drift_ok_rate: {peak_ok_rate:.2f}   "
                          f"accept_stand: {1 if ok else 0}   "
                          f"(存活率>=0.95 / 倾角<=0.30rad / "
                          f"全程峰值漂移<5cm / 高度>=80% / assist=0)",
                          flush=True)
            # ---- 检查点选择：按"站得稳 + 不点头 + 不漂"的综合目标 ----
            # 只用 reward 或只用 drift 选点都不对：前者会被 alive/姿态项带偏，
            # 后者会选到"漂移小但一直在抖"的策略。
            if (self.episodes and np.isfinite(drift_cm) and np.isfinite(pitch_rms)
                    and self.coord_now is not None and self.coord_now <= 0.05):
                score = (drift_cm + 2.0 * pitch_rms + 0.5 * accel_rms
                         + (0.0 if survive_rate >= 0.95 else 20.0))
                if score < self.best_score:
                    self.best_score = score
                    self.best_drift_cm = drift_cm
                    if self.save_prefix:
                        self.model.save(f"{self.save_prefix}_best")
                        if isinstance(self.training_env, VecNormalize):
                            self.training_env.save(f"{self.save_prefix}_best_vecnormalize.pkl")
                        self.best_paths = (f"{self.save_prefix}_best.zip",
                                           f"{self.save_prefix}_best_vecnormalize.pkl")
                    print(f"best_checkpoint: score={score:.3f} drift_cm={drift_cm:.2f} "
                          f"pitch_rms_deg={pitch_rms:.3f}", flush=True)
        learning_rate = float(self.model.learning_rate)
        explained_variance = self.logger.name_to_value.get("train/explained_variance")
        approx_kl = self.logger.name_to_value.get("train/approx_kl")
        mean_return = float(np.mean([e["r"] for e in episodes])) if episodes else float("nan")
        mean_ep_len = float(np.mean([e["l"] for e in episodes])) if episodes else float("nan")
        terms = info.get("reward_terms") or {}
        reward_total = float(sum(terms.values())) if terms else float("nan")
        if self.verbose:
            print(f"learning_rate: {learning_rate:.2e}", flush=True)
            for label, value in (("explained_variance", explained_variance),
                                 ("approx_kl", approx_kl)):
                if value is not None and np.isfinite(value):
                    print(f"{label}: {value:+.4f}", flush=True)
            if episodes:
                print(f"return: {mean_return:.1f}   ep_len: {mean_ep_len:.1f}", flush=True)
            if terms:
                top = sorted(terms.items(), key=lambda kv: -abs(kv[1]))[:self.REWARD_TOP]
                print(f"reward_total: {reward_total:+.2f}   "
                      + "   ".join(f"{k}={v:+.2f}" for k, v in top), flush=True)
        else:
            termination = str(info.get("termination_reason", "running"))
            if self.episodes:
                print(f"survive_rate: {survive_rate:.2f}", flush=True)
                print(f"drift_tail_cm: {drift_cm:.2f}", flush=True)
                print(f"pitch_rms_deg: {pitch_rms:.2f}", flush=True)
                print(f"accept_stand: {int(ok)}", flush=True)
            else:
                print("episode_status: collecting", flush=True)
            print(f"termination: {termination}", flush=True)
            kl_text = (f"{float(approx_kl):.4f}" if approx_kl is not None
                       and np.isfinite(approx_kl) else "n/a")
            ev_text = (f"{float(explained_variance):+.3f}" if explained_variance is not None
                       and np.isfinite(explained_variance) else "n/a")
            print(f"return_mean: {mean_return:.1f}", flush=True)
            print(f"reward_total: {reward_total:+.2f}", flush=True)
            print(f"learning_rate: {learning_rate:.1e}", flush=True)
            print(f"approx_kl: {kl_text}", flush=True)
            print(f"explained_variance: {ev_text}", flush=True)
        print("", flush=True)

        if self.save_prefix and update and update % self.save_interval == 0:
            self.model.save(f"{self.save_prefix}_iter_{update}")
            if isinstance(self.training_env, VecNormalize):
                self.training_env.save(f"{self.save_prefix}_iter_{update}_vecnormalize.pkl")

    def _on_training_end(self) -> None:
        if self.term_counts:
            total = sum(self.term_counts.values())
            summary = "   ".join(f"{k}={v}({100.0 * v / total:.0f}%)"
                                 for k, v in sorted(self.term_counts.items(), key=lambda kv: -kv[1]))
            print(f"termination_summary: {summary}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="UZ-05 分阶段强化学习训练")
    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument(
        "--preset", default="", choices=tuple(TRAINING_PRESETS),
        help="通用内置训练配置；leg-length=0.15~0.27 m 动态腿长续训",
    )
    config_group.add_argument(
        "--profile", default="", choices=tuple(TRAINING_PROFILES),
        help="版本化训练配置；自动使用 checkpoints/<version>/ 独立目录",
    )
    parser.add_argument("--stage", default="stand", choices=list(CAPABILITY_ORDER))
    parser.add_argument("--checkpoint", default="", help="从该 checkpoint 继承（任意阶段均可）")
    parser.add_argument("--reset-optimizer", action="store_true",
                        help="续训时清空继承 checkpoint 的 Adam 动量状态，适合奖励/课程微调")
    parser.add_argument("--updates", type=int, default=1500)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--vec-env", choices=("dummy", "subproc"), default="subproc")
    parser.add_argument("--disable-adaptive-lr", action="store_true")
    parser.add_argument("--version", default="",
                        help="本次训练的版本名；checkpoint 存到 checkpoints/<version>/。"
                             "默认 <stage>_v1")
    parser.add_argument("--out", default="", help="显式指定 checkpoint 前缀（覆盖 --version）")
    parser.add_argument("--tensorboard-log", default="")
    parser.add_argument("--save-interval", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)   # 对齐开源
    parser.add_argument("--ppo-epochs", type=int, default=5,
                        help="PPO 每轮优化 epoch；续训微调建议 1")
    parser.add_argument("--clip-range", type=float, default=0.2,
                        help="PPO ratio clip；续训微调建议 0.03~0.05")
    parser.add_argument("--target-kl", type=float, default=None,
                        help="PPO 单轮 KL 上限；续训微调可设 0.001~0.01")
    parser.add_argument("--ent-coef", type=float, default=0.005,
                        help="PPO 熵系数；稳定策略微调建议 0")
    parser.add_argument("--freeze-actor-backbone", action="store_true",
                        help="续训时冻结 actor 特征层，只微调动作头，降低站立策略漂移")
    parser.add_argument("--reset-leg-action-head", action="store_true",
                        help="继承 checkpoint 时将 4 个腿动作输出行清零；用于首次开放腿残差")
    parser.add_argument("--train-leg-diff-only", action="store_true",
                        help="训练时只更新 a3/a5 对称腿长差模动作头")
    parser.add_argument("--initial-log-std", type=float, default=-2.0,
                        help="高斯动作探索初值的 log(std)；站立建议约 -2.0")
    parser.add_argument("--log-std-min", type=float, default=-2.5)
    parser.add_argument("--log-std-max", type=float, default=-1.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vecnormalize-path", default="",
                        help="显式指定奖励归一化统计量；默认自动找 <checkpoint>_vecnormalize.pkl")
    parser.add_argument("--log-verbose", action="store_true",
                        help="打印完整状态量（默认只打印 21 项核心量）")
    parser.add_argument("--log-interval", type=int, default=1,
                        help="每 N 次迭代打印一次")
    parser.add_argument("--stand-level", type=int, default=0,
                        help="站立分级 0=S0_balance / 1=S1_tighten / 2=S2_accept(±5cm 验收)")
    parser.add_argument("--unlock-stand-legs", action="store_true", default=True,
                        help="站立训练时允许腿关节动作参与平衡（★ 现在默认开启："
                             "锁腿会让站立策略永远学不会用腿，只能靠轮子硬撑，"
                             "结果就是 1.6 Hz 的 pitch 点头极限环）")
    parser.add_argument("--lock-stand-legs", dest="unlock_stand_legs",
                        action="store_false",
                        help="回到旧的锁腿行为（仅用于复现历史结果）")
    parser.add_argument("--stand-leg-action-limit", type=float, default=1.0,
                        help="解锁站立腿动作的归一化限幅，默认 1.0（= ±0.35 rad 关节偏置）")
    parser.add_argument("--coord-start", type=float, default=None,
                        help="腿+轮协同平衡控制器混合系数起点（1 = 控制器全权）；"
                             "默认取 stand-level 的建议值")
    parser.add_argument("--coord-end", type=float, default=None,
                        help="协同平衡控制器混合系数终点（0 = 完全靠策略）")
    parser.add_argument("--coord-channels", default="all",
                        choices=("all", "legs", "wheels", "none"),
                        help="★ 通道级分工：all=协同控制器全权；legs=控制器给腿、"
                             "策略学轮（共模+差模）；wheels=控制器给轮、策略学腿；"
                             "none=完全靠策略。让'腿+轮协同'可以分开训练，"
                             "策略能拿到明确的回报梯度。")
    parser.add_argument("--coord-residual-scale", type=float, nargs=3, default=(0.05, 0.02, 0.05),
                        metavar=("WHEEL", "DIFF", "LEG"),
                        help="策略残差在共模轮 / 差模轮 / 腿通道上的缩放")
    parser.add_argument("--coord-residual-start", type=float, nargs=3, default=None,
                        metavar=("WHEEL", "DIFF", "LEG"),
                        help="残差课程起点；省略时与 --coord-residual-scale 相同")
    parser.add_argument("--coord-residual-warmup", type=float, default=0.0,
                        help="训练前段保持起始残差权限的比例（0..0.95）")
    parser.add_argument("--assist-start", type=float, default=None,
                        help="本阶段起始辅助强度（默认取 stand-level 的建议值）")
    parser.add_argument("--assist-adaptive", dest="assist_adaptive",
                        action="store_true", default=True,
                        help="自适应退火（默认开）：存活率或教师跟踪质量低于阈值时冻结/回退退火")
    parser.add_argument("--no-assist-adaptive", dest="assist_adaptive",
                        action="store_false",
                        help="按轮数匀速退火，不看存活率与跟踪质量")
    parser.add_argument("--assist-end", type=float, default=None,
                        help="本阶段结束辅助强度（训练中线性退火）")
    parser.add_argument("--init-scale-start", type=float, default=1.0,
                        help="初始姿态/速度扰动课程起点（0..1）")
    parser.add_argument("--init-scale-end", type=float, default=1.0,
                        help="初始姿态/速度扰动课程终点（0..1）")
    parser.add_argument("--vx-range", type=float, nargs=2, default=None,
                        metavar=("MIN", "MAX"),
                        help="覆盖平移命令的 |vx| 范围，例如 0.05 0.10；单位 m/s")
    parser.add_argument("--command-zero-prob", type=float, default=None,
                        help="零速度 episode 概率；用于平移课程的稳定锚点")
    parser.add_argument("--command-reverse-prob", type=float, default=None,
                        help="非零 vx 取反概率；0.5 表示正反向对称")
    parser.add_argument("--height-switch-steps", type=int, default=0,
                        help="站立 episode 内高度目标切换间隔（控制步）；0=关闭")
    parser.add_argument("--height-switch-prob", type=float, default=0.0,
                        help="到达切换间隔时采样新高度目标的概率（0..1）")
    parser.add_argument("--height-rate-limit", type=float, default=None,
                        help="腿长差模最大变化率（动作单位/s）；默认使用环境标定值")
    parser.add_argument("--height-rate-gain", type=float, default=None,
                        help="腿长 PI 环速率增益；默认使用环境标定值")
    parser.add_argument("--height-rate-damping", type=float, default=None,
                        help="腿长速度反馈阻尼；越大越能抑制目标附近的惯性过冲")
    parser.add_argument("--height-retract-rate-damping", type=float, default=None,
                        help="收腿方向单独的速度阻尼；用于低腿长下冲保护")
    parser.add_argument("--height-low-target-brake-damping-scale", type=float, default=None,
                        help="最低腿长目标附近的额外制动阻尼倍率（>=1）")
    parser.add_argument("--height-rate-brake-threshold", type=float, default=None,
                        help="触发腿长惯性制动的速度阈值（m/s）")
    parser.add_argument("--height-reference-jump-reset-m", type=float, default=None,
                        help="只有超过该目标跳变（m）才清空腿长积分器；斜坡更新不清零")
    parser.add_argument("--height-brake-error-m", type=float, default=None,
                        help="目标附近制动带宽（m）；远离目标时不撤掉同向驱动")
    parser.add_argument("--height-target-rate-feedforward-scale", type=float, default=None,
                        help="目标腿长速度前馈比例（0=关闭）")
    parser.add_argument("--height-hold-error-m", type=float, default=None,
                        help="目标附近切换到低带宽保持控制的误差范围（m）")
    parser.add_argument("--height-hold-rate-limit", type=float, default=None,
                        help="目标附近差模速率上限（动作单位/s）")
    parser.add_argument("--height-hold-rate-gain", type=float, default=None,
                        help="目标附近腿长保持增益")
    parser.add_argument("--height-filter-alpha", type=float, default=None,
                        help="腿差模独立低通系数；越小越平滑")
    parser.add_argument("--height-retract-rate-limit", type=float, default=None,
                        help="收腿方向最大差模变化率（动作单位/s）")
    parser.add_argument("--height-retract-slow-rate-limit", type=float, default=None,
                        help="收腿进入目标附近后的差模变化率")
    parser.add_argument("--height-retract-slow-error-m", type=float, default=None,
                        help="收腿进入低速区的剩余腿长误差阈值（m）")
    parser.add_argument("--height-retract-rate-gain", type=float, default=None,
                        help="收腿方向 PI 环速率增益")
    parser.add_argument("--height-retract-feedforward-scale", type=float, default=None,
                        help="收腿方向保留的目标位形前馈比例（0..1）")
    parser.add_argument("--leg-feedforward-scale", type=float, default=None,
                        help="腿长目标差模前馈比例（0..1）；默认使用保守基线值")
    parser.add_argument("--height-target-rate", type=float, default=None,
                        help="高度目标斜坡速率（m/s）；0/省略=目标立即切换")
    parser.add_argument("--leg-residual-slew-rate", type=float, default=0.30,
                        help="策略腿残差最终动作变化率（归一化动作/s）")
    parser.add_argument("--leg-residual-brake-error-m", type=float, default=None,
                        help="策略腿残差进入制动保护的剩余误差阈值（m）；省略则沿用旧版全程制动")
    parser.add_argument("--project-leg-length-residual", action="store_true",
                        help="将腿策略残差投影为左右对称的单腿长差模")
    parser.add_argument("--leg-length-reward-weight", type=float, default=None,
                        help="腿长位置误差奖励权重")
    parser.add_argument("--leg-length-progress-reward-weight", type=float, default=None,
                        help="腿长误差单步减小的势函数奖励权重")
    parser.add_argument("--leg-length-rate-reward-weight", type=float, default=None,
                        help="目标附近腿长速度惩罚权重")
    parser.add_argument("--norm-reward", dest="norm_reward", action="store_true",
                        default=False,
                        help="VecNormalize 的奖励归一化。★ 站立任务默认关闭：奖励已按"
                             "物理量标定（协同控制器满分为 +7.2/步），而归一化会在"
                             "奖励方差极小时把噪声放大成主梯度，实测会把标定好的"
                             "平衡基线推坏。")
    parser.add_argument("--normalize-reward", dest="norm_reward", action="store_true",
                        help="等价于 --norm-reward（兼容旧命令）")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    apply_training_preset(args, sys.argv[1:])

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
    if args.profile:
        print(f"[profile] {args.profile}", flush=True)
    elif args.preset:
        print(f"[preset] {args.preset}", flush=True)

    # 默认输出为**绝对路径**，始终落在仓库的 checkpoints/<version>/ 下，
    # 与运行时所在目录无关。每个 version 自动新建一个文件夹。
    version = args.version or f"{args.stage}_v1"
    out = args.out or str(REPO_ROOT / "checkpoints" / version / "checkpoint")
    tb = args.tensorboard_log or str(REPO_ROOT / "runs" / version)
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[interface] stage={args.stage}  obs={OBS_DIM}  action={ACTION_DIM}  "
        f"(固定接口，可直接继承任何阶段的 checkpoint)",
        flush=True,
    )

    # 续训时复用上一阶段的奖励归一化统计量（否则 critic 目标尺度会突变）
    stats_path = Path(args.vecnormalize_path).expanduser() if args.vecnormalize_path else None
    if stats_path is None and args.checkpoint:
        base = Path(args.checkpoint).expanduser()
        candidate = Path(f"{base.with_suffix('') if base.suffix == '.zip' else base}_vecnormalize.pkl")
        if candidate.exists():
            stats_path = candidate

    env_cls = SubprocVecEnv if args.num_envs > 1 and args.vec_env == "subproc" else DummyVecEnv
    from uz05.spec import STAND_LEVELS
    level = next((lv for lv in STAND_LEVELS if lv.level == args.stand_level), STAND_LEVELS[0])
    assist_start = level.assist_start if args.assist_start is None else args.assist_start
    assist_end = level.assist_end if args.assist_end is None else args.assist_end
    coord_start = level.coord_start if args.coord_start is None else args.coord_start
    coord_end = level.coord_end if args.coord_end is None else args.coord_end
    residual_start = (tuple(args.coord_residual_scale)
                      if args.coord_residual_start is None
                      else tuple(args.coord_residual_start))
    residual_end = tuple(args.coord_residual_scale)
    env = VecMonitor(env_cls([
        make_env(args.stage, r, args.seed, level.level, assist_start,
                 args.init_scale_start, not args.unlock_stand_legs,
                 args.stand_leg_action_limit, coord_start,
                 residual_start, args.coord_channels,
                 args.height_switch_steps, args.height_switch_prob,
                 args.height_rate_limit, args.height_rate_gain,
                 args.height_rate_damping, args.height_retract_rate_damping,
                 args.height_low_target_brake_damping_scale,
                 args.height_rate_brake_threshold,
                 args.height_reference_jump_reset_m, args.height_brake_error_m,
                 args.height_target_rate_feedforward_scale,
                 args.height_hold_error_m, args.height_hold_rate_limit,
                 args.height_hold_rate_gain, args.height_filter_alpha,
                 args.height_retract_rate_limit, args.height_retract_slow_rate_limit,
                 args.height_retract_slow_error_m,
                 args.height_retract_rate_gain,
                 args.height_retract_feedforward_scale,
                 args.leg_feedforward_scale, args.height_target_rate,
                 args.leg_residual_slew_rate,
                 args.leg_residual_brake_error_m,
                 args.project_leg_length_residual,
                 args.leg_length_reward_weight,
                 args.leg_length_progress_reward_weight,
                 args.leg_length_rate_reward_weight,
                 args.vx_range,
                 args.command_zero_prob,
                 args.command_reverse_prob)
        for r in range(args.num_envs)
    ]))
    if stats_path is not None:
        if not stats_path.exists():
            raise FileNotFoundError(f"VecNormalize 统计量不存在: {stats_path}")
        env = VecNormalize.load(str(stats_path), env)
        env.training = True
        env.norm_reward = bool(args.norm_reward)
        print(f"vecnormalize_loaded: {stats_path}", flush=True)
    else:
        env = VecNormalize(env, norm_obs=False, norm_reward=bool(args.norm_reward),
                           clip_reward=10.0)
    print(f"train_config_norm_reward: {bool(args.norm_reward)}", flush=True)

    if args.checkpoint:
        checkpoint = Path(args.checkpoint).expanduser()
        if checkpoint.suffix == ".zip":
            checkpoint = checkpoint.with_suffix("")
        ppo_kwargs_for_migration = dict(
            n_steps=args.rollout_steps, batch_size=args.batch_size,
            learning_rate=args.learning_rate, n_epochs=args.ppo_epochs,
            clip_range=args.clip_range, gamma=0.99,
            target_kl=args.target_kl,
            gae_lambda=0.95, vf_coef=2.0, max_grad_norm=1.0,
            policy_kwargs=dict(net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64]),
                               activation_fn=nn.ELU, log_std_init=args.initial_log_std),
            tensorboard_log=tb, verbose=0, device=args.device, seed=args.seed,
        )
        model = load_and_migrate_v73(checkpoint, env, ppo_kwargs_for_migration)
        if model is None:
            try:
                model = PPO.load(str(checkpoint), env=env, device=args.device)
            except (ValueError, AssertionError) as error:
                raise SystemExit(
                    f"checkpoint 与新接口不兼容（新接口 obs={OBS_DIM} action={ACTION_DIM}）：{error}\n"
                    "本版已对齐开源接口，旧 checkpoint 请舍弃 —— 去掉 --checkpoint 从零训练。"
                )
        if model.policy.log_std is not None:
            import torch as _torch
            model.policy.log_std.data.clamp_(args.log_std_min, args.log_std_max)
        # PPO.load() carries the parent checkpoint's tensorboard path. Override it
        # explicitly so a resumed experiment cannot append metrics to an older run.
        model.tensorboard_log = tb
        model.n_epochs = args.ppo_epochs
        model.clip_range = lambda _progress: args.clip_range
        model.target_kl = args.target_kl
        model.ent_coef = args.ent_coef
        model.learning_rate = args.learning_rate
        # PPO.train() evaluates lr_schedule on every update. Updating only the
        # optimizer param group leaves the checkpoint's old schedule active,
        # so a requested 1e-6 fine-tune can silently run at the old LR.
        model.lr_schedule = get_schedule_fn(args.learning_rate)
        for group in model.policy.optimizer.param_groups:
            group["lr"] = args.learning_rate
        if args.reset_optimizer:
            # Keep the policy/value weights but discard stale Adam moments from
            # the previous curriculum, which can cause large effective updates
            # even after lowering the nominal learning rate.
            model.policy.optimizer.state.clear()
            print("[inherit] optimizer_state_reset: true", flush=True)
        if args.reset_leg_action_head:
            # 历史站立策略的腿通道长期被 residual_scale=0 屏蔽，输出没有
            # 物理含义。只清零腿动作头，保留姿态/轮平衡特征和轮动作头。
            with torch.no_grad():
                model.policy.action_net.weight[2:6].zero_()
                model.policy.action_net.bias[2:6].zero_()
                if model.policy.log_std is not None:
                    model.policy.log_std.data[2:6].fill_(args.initial_log_std)
            model.policy.optimizer.state.clear()
            print("[inherit] leg_action_head_reset: true", flush=True)
        if args.freeze_actor_backbone:
            for name, parameter in model.policy.named_parameters():
                if name.startswith("mlp_extractor.policy_net."):
                    parameter.requires_grad_(False)
            # Keep the original optimizer parameter groups so the saved PPO
            # checkpoint remains loadable by an ordinary PPO.load(). Frozen
            # tensors receive no gradient; their optimizer slots are harmless.
            model.policy.optimizer.state.clear()
            print("[inherit] actor_backbone_frozen: true", flush=True)
        if args.train_leg_diff_only:
            # Preserve inherited wheel outputs and freeze the two leg common
            # modes. Only a3/a5 may learn; the environment projects them into
            # one symmetric physical leg-length command.
            weight_mask = torch.zeros_like(model.policy.action_net.weight)
            weight_mask[3] = 1.0
            weight_mask[5] = 1.0
            bias_mask = torch.zeros_like(model.policy.action_net.bias)
            bias_mask[3] = 1.0
            bias_mask[5] = 1.0
            model.policy.action_net.weight.register_hook(
                lambda grad, mask=weight_mask: grad * mask
            )
            model.policy.action_net.bias.register_hook(
                lambda grad, mask=bias_mask: grad * mask
            )
            if model.policy.log_std is not None:
                std_mask = torch.zeros_like(model.policy.log_std)
                std_mask[3] = 1.0
                std_mask[5] = 1.0
                model.policy.log_std.register_hook(
                    lambda grad, mask=std_mask: grad * mask
                )
            print("[inherit] trainable_actor_outputs: a3,a5", flush=True)
        print(f"[inherit] 已从 {checkpoint} 继承策略权重（接口固定，无需迁移）", flush=True)
    else:
        model = PPO(
            AsymmetricActorCriticPolicy,
            env,
            n_steps=args.rollout_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            n_epochs=args.ppo_epochs,
            clip_range=args.clip_range,
            target_kl=args.target_kl,
            gamma=0.99,
            gae_lambda=0.95,
            vf_coef=2.0,             # 对齐开源 value_loss_coef
            max_grad_norm=1.0,       # 对齐开源
            ent_coef=args.ent_coef,
            policy_kwargs=dict(
                actor_dim=ACTOR_OBS_DIM,
                net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64]),
                activation_fn=nn.ELU,
                log_std_init=args.initial_log_std,
            ),
            tensorboard_log=tb,
            verbose=0,
            device=args.device,
            seed=args.seed,
        )
        # S0 的零动作在辅助下是已验证的稳定基准；从该均值出发只探索受限残差。
        with torch.no_grad():
            model.policy.action_net.weight.zero_()
            model.policy.action_net.bias.zero_()

    total_steps = args.updates * args.rollout_steps * args.num_envs
    if args.log_verbose:
        print(f"train_config_stage: {args.stage}", flush=True)
        print(f"train_config_updates: {args.updates}", flush=True)
        print(f"train_config_total_steps: {total_steps}", flush=True)
        print(f"train_config_num_envs: {args.num_envs}", flush=True)
        print(f"train_config_learning_rate: {args.learning_rate}", flush=True)
        print(f"train_config_stand_level: {level.name}", flush=True)
        print(f"train_config_station_deadband_m: {level.deadband}", flush=True)
        print(f"train_config_station_hard_limit_m: {level.hard_limit}", flush=True)
        print(f"train_config_assist_anneal: {assist_start} -> {assist_end}", flush=True)
        print(f"train_config_init_scale: {args.init_scale_start} -> {args.init_scale_end}", flush=True)
        print(f"train_config_lock_stand_legs: {not args.unlock_stand_legs}", flush=True)
        print(f"train_config_stand_leg_action_limit: {args.stand_leg_action_limit}", flush=True)
        print(f"train_config_coord_anneal: {coord_start} -> {coord_end}", flush=True)
        print(f"train_config_coord_residual_scale: {tuple(args.coord_residual_scale)}", flush=True)
        print(f"train_config_coord_residual_start: {residual_start}", flush=True)
        print(f"train_config_coord_residual_warmup: {args.coord_residual_warmup}", flush=True)
        print(f"train_config_coord_channels: {args.coord_channels}", flush=True)
        print(f"train_config_height_switch: {args.height_switch_steps} steps @ "
              f"{args.height_switch_prob}", flush=True)
        print(f"train_config_height_rate_limit: {args.height_rate_limit}", flush=True)
        print(f"train_config_height_rate_gain: {args.height_rate_gain}", flush=True)
        print(f"train_config_height_rate_damping: {args.height_rate_damping}", flush=True)
        print(f"train_config_height_retract_rate_damping: "
              f"{args.height_retract_rate_damping}", flush=True)
        print(f"train_config_height_low_target_brake_damping_scale: "
              f"{args.height_low_target_brake_damping_scale}", flush=True)
        print(f"train_config_height_rate_brake_threshold: "
              f"{args.height_rate_brake_threshold}", flush=True)
        print(f"train_config_height_reference_jump_reset_m: "
              f"{args.height_reference_jump_reset_m}", flush=True)
        print(f"train_config_height_brake_error_m: "
              f"{args.height_brake_error_m}", flush=True)
        print(f"train_config_height_target_rate_feedforward_scale: "
              f"{args.height_target_rate_feedforward_scale}", flush=True)
        print(f"train_config_height_hold_error_m: {args.height_hold_error_m}", flush=True)
        print(f"train_config_height_hold_rate_limit: {args.height_hold_rate_limit}", flush=True)
        print(f"train_config_height_hold_rate_gain: {args.height_hold_rate_gain}", flush=True)
        print(f"train_config_height_filter_alpha: {args.height_filter_alpha}", flush=True)
        print(f"train_config_height_retract_rate_limit: "
              f"{args.height_retract_rate_limit}", flush=True)
        print(f"train_config_height_retract_slow_rate_limit: "
              f"{args.height_retract_slow_rate_limit}", flush=True)
        print(f"train_config_height_retract_slow_error_m: "
              f"{args.height_retract_slow_error_m}", flush=True)
        print(f"train_config_height_retract_rate_gain: "
              f"{args.height_retract_rate_gain}", flush=True)
        print(f"train_config_height_retract_feedforward_scale: "
              f"{args.height_retract_feedforward_scale}", flush=True)
        print(f"train_config_leg_feedforward_scale: {args.leg_feedforward_scale}", flush=True)
        print(f"train_config_height_target_rate: {args.height_target_rate}", flush=True)
    else:
        leg_range = STAGE_BY_NAME[args.stage].leg_length_range
        leg_range_text = (f"{leg_range[0]:.3f}~{leg_range[1]:.3f}m"
                          if leg_range is not None else "off")
        print(f"train_stage: {args.stage}", flush=True)
        print(f"stand_level: {level.name}", flush=True)
        print(f"updates: {args.updates}", flush=True)
        print(f"total_steps: {total_steps}", flush=True)
        print(f"num_envs: {args.num_envs}", flush=True)
        print(f"initial_learning_rate: {args.learning_rate:.1e}", flush=True)
        print(f"leg_length_range: {leg_range_text}", flush=True)
        print(f"height_switch_steps: {args.height_switch_steps}", flush=True)
        print(f"height_target_rate_m_s: {args.height_target_rate}", flush=True)
        print(f"height_diff_rate_limit: {args.height_rate_limit}", flush=True)
        print(f"coord_residual_start: {residual_start}", flush=True)
        print(f"coord_residual_end: {residual_end}", flush=True)
        print(f"leg_residual_slew_rate: {args.leg_residual_slew_rate}", flush=True)
        print(f"leg_residual_brake_error_m: {args.leg_residual_brake_error_m}", flush=True)
        print(f"project_leg_length_residual: {args.project_leg_length_residual}", flush=True)
    print("", flush=True)
    model._total_timesteps = total_steps
    model.learn(
        total_timesteps=total_steps,
        callback=ProgressLogger(args.stage, out, args.save_interval, version,
                                assist_start, assist_end, args.log_verbose,
                                args.log_interval, args.updates,
                                args.assist_adaptive,
                                adaptive_lr=not args.disable_adaptive_lr,
                                log_std_min=args.log_std_min,
                                log_std_max=args.log_std_max,
                                init_scale_start=args.init_scale_start,
                                init_scale_end=args.init_scale_end,
                                coord_start=coord_start, coord_end=coord_end,
                                coord_residual_start=residual_start,
                                coord_residual_end=residual_end,
                                coord_residual_warmup=args.coord_residual_warmup),
        log_interval=1,
    )
    model.save(out)
    env.save(f"{out}_vecnormalize.pkl")
    achieved_residual = residual_end
    if hasattr(env, "env_method"):
        try:
            values = env.env_method("get_coord_residual_scale")
            if values:
                achieved_residual = tuple(float(v) for v in values[0])
        except (AttributeError, IndexError, TypeError, ValueError):
            pass
    deployment = {
        "coord_mix": float(coord_end),
        "coord_residual_scale": list(achieved_residual),
        "leg_residual_slew_rate": float(args.leg_residual_slew_rate),
        "leg_residual_brake_error_m": (
            None if args.leg_residual_brake_error_m is None else
            float(args.leg_residual_brake_error_m)
        ),
        "project_leg_length_residual": bool(args.project_leg_length_residual),
        "leg_length_range_m": list(STAGE_BY_NAME[args.stage].leg_length_range or []),
        "height_target_rate_m_s": (
            None if args.height_target_rate is None else float(args.height_target_rate)
        ),
        "height_reference_jump_reset_m": (
            None if args.height_reference_jump_reset_m is None else
            float(args.height_reference_jump_reset_m)
        ),
        "height_brake_error_m": (
            None if args.height_brake_error_m is None else
            float(args.height_brake_error_m)
        ),
        "height_retract_rate_damping": (
            None if args.height_retract_rate_damping is None else
            float(args.height_retract_rate_damping)
        ),
        "height_low_target_brake_damping_scale": (
            None if args.height_low_target_brake_damping_scale is None else
            float(args.height_low_target_brake_damping_scale)
        ),
        "height_target_rate_feedforward_scale": (
            None if args.height_target_rate_feedforward_scale is None else
            float(args.height_target_rate_feedforward_scale)
        ),
        "height_hold_error_m": (
            None if args.height_hold_error_m is None else float(args.height_hold_error_m)
        ),
        "height_hold_rate_limit": (
            None if args.height_hold_rate_limit is None else float(args.height_hold_rate_limit)
        ),
        "height_hold_rate_gain": (
            None if args.height_hold_rate_gain is None else float(args.height_hold_rate_gain)
        ),
        "height_filter_alpha": (
            None if args.height_filter_alpha is None else float(args.height_filter_alpha)
        ),
        "control_dt_s": 0.008,
        "actor_uses_privileged_observations": False,
    }
    contract = write_contract(out, args.stage, args.updates, version, deployment)
    print(f"checkpoint: {out}.zip", flush=True)
    print(f"vecnormalize: {out}_vecnormalize.pkl", flush=True)
    print(f"contract: {contract}", flush=True)
    print(f"tensorboard: {tb}", flush=True)


if __name__ == "__main__":
    main()
