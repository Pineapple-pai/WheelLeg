"""训练入口：分阶段（capability）训练 UZ-05 轮腿机器人。

**继承机制**：观测维度（214）与动作维度（6）在所有阶段完全固定，
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
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize

import torch.nn as nn

from uz05.env import UZ05Env
from uz05.spec import (ACTION_DIM, ACTOR_OBS_DIM, CAPABILITY_ORDER, OBS_DIM,
                       REPO_ROOT, STAGE_BY_NAME, STAGES)

CONTRACT_VERSION = "uz05_iface_v1"



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


def make_env(stage: str, rank: int, seed: int, stand_level: int, assist: float):
    def _init():
        env = UZ05Env(stage=stage, seed=seed + rank, stand_level=stand_level,
                      assist_scale=assist)
        env.reset(seed=seed + rank)
        return env
    return _init


def write_contract(path: str | Path, stage: str, total_updates: int,
                   version: str = "") -> Path:
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
    target = Path(f"{path}_contract.json")
    target.write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n")
    return target


class ProgressLogger(BaseCallback):
    """精简日志：每项数据单独一行，只保留判断训练是否健康所必需的量。

    需要完整 41 项状态时加 ``--log-verbose``。
    """

    # 核心量：(显示名, info 键, 格式)
    CORE = (
        ("assist_scale",   "assist_scale",         ".3f"),
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
                 hold_below=0.50, resume_above=0.90, adaptive_lr=True):
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
        #   (是否活满整集, 末段稳态漂移)
        self.episodes: deque = deque(maxlen=50)
        # RSL-RL 自适应学习率（对齐开源 desired_kl=0.01）
        self.desired_kl = 0.01
        self.lr_min, self.lr_max = 1.0e-5, 2.0e-4
        self.adaptive_lr = bool(adaptive_lr)
        self.best_drift_cm = float("nan")

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
                self.episodes.append((survived, tail))
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
        """每轮推进一次退火（必须在 rollout 边界调用，不能放在 _on_step）。"""
        if self.assist_start is None or self.assist_end is None:
            return
        if self.adaptive and self.episodes:
            survive = float(np.mean([s for s, _ in self.episodes]))
            step = 1.0 / max(1, self.total_updates)
            if survive < self.hold_below:
                # 策略跟不上：不只冻结，还要**回退**辅助，把它拉回能应付的区间。
                # 这样辅助强度会稳定停留在"策略能力边界"附近 —— 那里学习最快。
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

    def _on_rollout_end(self) -> None:
        import time
        self._apply_assist_anneal()
        self._adapt_learning_rate()
        update = self.num_timesteps // max(1, self.locals.get("n_steps", 1) * self.training_env.num_envs)
        if update % self.log_interval:
            if self.save_prefix and update and update % self.save_interval == 0:
                self.model.save(f"{self.save_prefix}_iter_{update}")
            return
        info = dict(self.last_info or {})
        info["stage"] = self.stage
        if self.assist_now is not None:
            info["assist_scale"] = self.assist_now
        info["_anneal"] = self.anneal_progress
        info["_holding"] = 1.0 if self.holding else 0.0
        # 派生量：关节统计合成一行，不再逐关节打印
        try:
            info["_hip_pos_abs"] = float(np.abs([info[f"hip_pos_{i}"] for i in range(4)]).mean())
            info["_hip_vel_abs"] = float(np.abs([info[f"hip_vel_{i}"] for i in range(4)]).mean())
        except KeyError:
            pass

        elapsed = time.perf_counter() - self._start
        episodes = list(self.model.ep_info_buffer)
        print(f"iter: {update}/{self.total_updates}   fps: {self.num_timesteps / elapsed:.0f}   "
              f"elapsed_s: {elapsed:.0f}", flush=True)
        for label, key, fmt in self.CORE + (self.EXTRA if self.verbose else ()):
            if key not in info:
                continue
            value = info[key]
            try:
                print(f"{label}: {value if fmt == 's' else format(float(value), fmt)}", flush=True)
            except (TypeError, ValueError):
                print(f"{label}: {value}", flush=True)
        # ---------------- 验收指标 ----------------
        if self.episodes:
            survived = [s for s, _ in self.episodes]
            survive_rate = float(np.mean(survived))
            tails = [t for s, t in self.episodes if s and np.isfinite(t)]
            drift_cm = float(np.mean(tails) * 100.0) if tails else float("nan")
            print(f"survive_rate: {survive_rate:.2f}   ({int(np.sum(survived))}/{len(survived)} 活满整集)",
                  flush=True)
            if tails:
                print(f"drift_tail_cm: {drift_cm:.2f}   (仅统计活满整集, n={len(tails)})", flush=True)
            else:
                print("drift_tail_cm: n/a   (本窗口没有活满整集的 episode)", flush=True)
            assist = info.get("assist_scale")
            if assist is not None:
                ok = survive_rate >= 0.95 and np.isfinite(drift_cm) and drift_cm < 5.0 and assist <= 1e-3
                print(f"accept_5cm: {1 if ok else 0}   "
                      f"(需同时满足 存活率>=0.95 / 漂移<5cm / assist=0)", flush=True)
        print(f"learning_rate: {float(self.model.learning_rate):.2e}", flush=True)
        for tag in ("train/explained_variance", "train/approx_kl"):
            value = self.logger.name_to_value.get(tag)
            if value is not None and np.isfinite(value):
                print(f"{tag.split('/')[1]}: {value:+.4f}", flush=True)
        if episodes:
            print(f"return: {np.mean([e['r'] for e in episodes]):.1f}   "
                  f"ep_len: {np.mean([e['l'] for e in episodes]):.1f}", flush=True)
        terms = info.get("reward_terms") or {}
        if terms:
            top = sorted(terms.items(), key=lambda kv: -abs(kv[1]))[:self.REWARD_TOP]
            print(f"reward_total: {sum(terms.values()):+.2f}   " +
                  "   ".join(f"{k}={v:+.2f}" for k, v in top), flush=True)
        print("", flush=True)

        if self.save_prefix and update and update % self.save_interval == 0:
            self.model.save(f"{self.save_prefix}_iter_{update}")

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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vecnormalize-path", default="",
                        help="显式指定奖励归一化统计量；默认自动找 <checkpoint>_vecnormalize.pkl")
    parser.add_argument("--log-verbose", action="store_true",
                        help="打印完整状态量（默认只打印 21 项核心量）")
    parser.add_argument("--log-interval", type=int, default=1,
                        help="每 N 次迭代打印一次")
    parser.add_argument("--stand-level", type=int, default=0,
                        help="站立分级 0=S0_balance / 1=S1_tighten / 2=S2_accept(±5cm 验收)")
    parser.add_argument("--assist-start", type=float, default=None,
                        help="本阶段起始辅助强度（默认取 stand-level 的建议值）")
    parser.add_argument("--assist-adaptive", action="store_true",
                        help="自适应退火：存活率低于阈值时冻结退火，避免辅助谷崩溃")
    parser.add_argument("--assist-end", type=float, default=None,
                        help="本阶段结束辅助强度（训练中线性退火）")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

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
    env = VecMonitor(env_cls([
        make_env(args.stage, r, args.seed, level.level, assist_start)
        for r in range(args.num_envs)
    ]))
    if stats_path is not None:
        if not stats_path.exists():
            raise FileNotFoundError(f"VecNormalize 统计量不存在: {stats_path}")
        env = VecNormalize.load(str(stats_path), env)
        env.training = True
        env.norm_reward = True
        print(f"vecnormalize_loaded: {stats_path}", flush=True)
    else:
        env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    if args.checkpoint:
        checkpoint = Path(args.checkpoint).expanduser()
        if checkpoint.suffix == ".zip":
            checkpoint = checkpoint.with_suffix("")
        ppo_kwargs_for_migration = dict(
            n_steps=args.rollout_steps, batch_size=args.batch_size,
            learning_rate=args.learning_rate, n_epochs=5, gamma=0.99,
            gae_lambda=0.95, vf_coef=2.0, max_grad_norm=1.0,
            policy_kwargs=dict(net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64]),
                               activation_fn=nn.ELU, log_std_init=-1.5),
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
            model.policy.log_std.data.clamp_(-2.3, -0.9)   # std ∈ [0.10, 0.40]
        model.learning_rate = args.learning_rate
        for group in model.policy.optimizer.param_groups:
            group["lr"] = args.learning_rate
        print(f"[inherit] 已从 {checkpoint} 继承策略权重（接口固定，无需迁移）", flush=True)
    else:
        model = PPO(
            AsymmetricActorCriticPolicy,
            env,
            n_steps=args.rollout_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            n_epochs=5,              # 对齐开源 num_learning_epochs
            gamma=0.99,
            gae_lambda=0.95,
            vf_coef=2.0,             # 对齐开源 value_loss_coef
            max_grad_norm=1.0,       # 对齐开源
            ent_coef=0.005,          # 对齐开源 entropy_coef
            policy_kwargs=dict(
                actor_dim=ACTOR_OBS_DIM,
                net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64]),
                activation_fn=nn.ELU,
                # 开源用 init_noise_std=1.0（σ=1.0）。本机实测 σ=1.0 会让策略一上来
                # 就用满幅腿动作把闭链机构改形、固定平衡增益失效，故取 σ≈0.37 起步，
                # 由 ent_coef=0.005 自然退火。这是**有意的偏离**，其余超参全部对齐。
                log_std_init=-1.0,
            ),
            tensorboard_log=tb,
            verbose=0,
            device=args.device,
            seed=args.seed,
        )

    total_steps = args.updates * args.rollout_steps * args.num_envs
    print(f"train_config_stage: {args.stage}", flush=True)
    print(f"train_config_updates: {args.updates}", flush=True)
    print(f"train_config_total_steps: {total_steps}", flush=True)
    print(f"train_config_num_envs: {args.num_envs}", flush=True)
    print(f"train_config_learning_rate: {args.learning_rate}", flush=True)
    print(f"train_config_stand_level: {level.name}", flush=True)
    print(f"train_config_station_deadband_m: {level.deadband}", flush=True)
    print(f"train_config_station_hard_limit_m: {level.hard_limit}", flush=True)
    print(f"train_config_assist_anneal: {assist_start} -> {assist_end}", flush=True)
    print("", flush=True)
    model._total_timesteps = total_steps
    model.learn(
        total_timesteps=total_steps,
        callback=ProgressLogger(args.stage, out, args.save_interval, version,
                                assist_start, assist_end, args.log_verbose,
                                args.log_interval, args.updates,
                                args.assist_adaptive,
                                adaptive_lr=not args.disable_adaptive_lr),
        log_interval=1,
    )
    model.save(out)
    env.save(f"{out}_vecnormalize.pkl")
    contract = write_contract(out, args.stage, args.updates, version)
    print(f"checkpoint: {out}.zip", flush=True)
    print(f"vecnormalize: {out}_vecnormalize.pkl", flush=True)
    print(f"contract: {contract}", flush=True)
    print(f"tensorboard: {tb}", flush=True)


if __name__ == "__main__":
    main()
