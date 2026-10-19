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
from uz05.spec import (
    ACTION_DIM, ACTION_SPEC, ACTOR_FRAME_DIM, ACTOR_OBS_BLOCKS, ACTOR_OBS_DIM,
    CAPABILITY_ORDER, OBS_DIM, OBS_SCALE, OBS_SOURCE, PRIV_OBS_BLOCKS, PRIV_OBS_DIM,
    EnvParams, MODEL_XML, REPO_ROOT, STAGE_BY_NAME, STAGES,
)

CONTRACT_VERSION = "uz05_iface_v5"


def model_xml_sha256() -> str:
    """Fingerprint the exact MJCF used to train a checkpoint."""
    return hashlib.sha256(MODEL_XML.read_bytes()).hexdigest()


def actuator_contract() -> dict:
    """Capture actuator and control-period semantics for checkpoint safety."""
    params = EnvParams()
    return {
        "joint": asdict(params.joint),
        "wheel": asdict(params.wheel),
        "control_dt_s": params.control_dt,
    }

def make_env(stage: str, rank: int, seed: int, stand_level: int,
             domain_randomization: bool = True):
    def _init():
        env = UZ05Env(stage=stage, seed=seed + rank, stand_level=stand_level)
        env.params.domain_randomization.enabled = bool(domain_randomization)
        env.reset(seed=seed + rank)
        return env
    return _init


def write_contract(path: str | Path, stage: str, total_updates: int,
                   version: str = "", domain_randomization: bool = True) -> Path:
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
        "actuator_config": actuator_contract(),
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
        "action_blocks": [{"name": name, "width": width, "scale": scale}
                          for name, width, scale in ACTION_SPEC],
        "interface_frozen": True,
        "note": (
            "观测与动作张量形状在所有 capability 阶段固定；接口版本、MJCF 哈希和执行器参数都相同的 "
            "checkpoint 才可直接继承。actor 不接收仿真任务标签；速度、位移、高度和前向地形值仍是带噪"
            "仿真代理，部署前需要用传感器状态估计替换并标定。"
        ),
        "total_updates": int(total_updates),
    }
    target = Path(f"{path}_contract.json")
    target.write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n")
    return target


def require_compatible_contract(checkpoint: Path) -> None:
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
    found = contract.get("contract_version")
    if found != CONTRACT_VERSION:
        raise SystemExit(
            f"checkpoint 接口为 {found!r}，当前代码要求 {CONTRACT_VERSION!r}: {checkpoint}\n"
            "观测接口/执行器模型版本已改变；请使用 v5 checkpoint，或去掉 --checkpoint 从头训练。"
        )
    expected = {
        "model_xml_sha256": model_xml_sha256(),
        "actuator_config": actuator_contract(),
        "observation_shape": OBS_DIM,
        "actor_observation_shape": ACTOR_OBS_DIM,
        "critic_privileged_shape": PRIV_OBS_DIM,
        "action_shape": ACTION_DIM,
    }
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


class ProgressLogger(BaseCallback):
    """精简日志：每项数据单独一行，只保留判断训练是否健康所必需的量。

    需要完整 41 项状态时加 ``--log-verbose``。
    """

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
        import time
        self._adapt_learning_rate()
        update = self.num_timesteps // max(1, self.locals.get("n_steps", 1) * self.training_env.num_envs)
        if update % self.log_interval:
            if self.save_prefix and update and update % self.save_interval == 0:
                self.model.save(f"{self.save_prefix}_iter_{update}")
            return
        info = dict(self.last_info or {})
        info["stage"] = self.stage
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
            survived = [episode[0] for episode in self.episodes]
            survive_rate = float(np.mean(survived))
            tails = [episode[1] for episode in self.episodes
                     if episode[0] and np.isfinite(episode[1])]
            peak_drifts = [episode[2] for episode in self.episodes
                           if episode[0] and np.isfinite(episode[2])]
            peak_yaws = [episode[3] for episode in self.episodes
                         if episode[0] and np.isfinite(episode[3])]
            peak_tilts = [episode[4] for episode in self.episodes
                          if episode[0] and np.isfinite(episode[4])]
            gyro_rms = [episode[5] for episode in self.episodes
                        if episode[0] and np.isfinite(episode[5])]
            drift_cm = float(np.mean(tails) * 100.0) if tails else float("nan")
            max_drift_cm = float(max(peak_drifts) * 100.0) if peak_drifts else float("nan")
            max_yaw_deg = float(np.degrees(max(peak_yaws))) if peak_yaws else float("nan")
            max_tilt_deg = float(np.degrees(max(peak_tilts))) if peak_tilts else float("nan")
            max_gyro_rms = float(max(gyro_rms)) if gyro_rms else float("nan")
            print(f"survive_rate: {survive_rate:.2f}   ({int(np.sum(survived))}/{len(survived)} 活满整集)",
                  flush=True)
            if tails:
                print(f"drift_tail_cm: {drift_cm:.2f}   (仅统计活满整集, n={len(tails)})", flush=True)
            else:
                print("drift_tail_cm: n/a   (本窗口没有活满整集的 episode)", flush=True)
            if peak_drifts and peak_yaws and peak_tilts and gyro_rms:
                print(f"drift_peak_cm: {max_drift_cm:.2f}   yaw_peak_deg: {max_yaw_deg:.2f}   "
                      f"tilt_peak_deg: {max_tilt_deg:.2f}   gyro_rms_rad_s: {max_gyro_rms:.3f}",
                      flush=True)
            ok = (survive_rate >= 0.95 and np.isfinite(max_drift_cm)
                  and max_drift_cm <= 5.0 and np.isfinite(max_yaw_deg)
                  and max_yaw_deg <= 5.0 and np.isfinite(max_tilt_deg)
                  and max_tilt_deg <= self.tilt_limit_deg and np.isfinite(max_gyro_rms)
                  and max_gyro_rms <= 0.20)
            print(f"accept_stand: {1 if ok else 0}   "
                  f"(纯 PPO；存活率>=0.95 / 全程漂移<=5cm / 偏航<=5° / "
                  f"pitch-roll峰值<={self.tilt_limit_deg:.1f}° / 角速度RMS<=0.20rad/s)", flush=True)
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vecnormalize-path", default="",
                        help="显式指定奖励归一化统计量；默认自动找 <checkpoint>_vecnormalize.pkl")
    parser.add_argument("--reset-vecnormalize", action="store_true",
                        help="不继承旧奖励归一化统计；奖励函数变化后续训时使用")
    parser.add_argument("--log-verbose", action="store_true",
                        help="打印完整状态量（默认只打印 21 项核心量）")
    parser.add_argument("--log-interval", type=int, default=1,
                        help="每 N 次迭代打印一次")
    parser.add_argument("--stand-level", type=int, default=0,
                        help="站立分级 0=S0_balance / 1=S1_tighten / 2=S2_accept(±5cm/±5° 验收)")
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
    version = args.version or f"{args.stage}_v4"
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
                 domain_randomization=not args.disable_domain_randomization)
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
        env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    if args.checkpoint:
        checkpoint = Path(args.checkpoint).expanduser()
        if checkpoint.suffix == ".zip":
            checkpoint = checkpoint.with_suffix("")
        require_compatible_contract(checkpoint)
        try:
            model = PPO.load(str(checkpoint), env=env, device=args.device)
        except (ValueError, AssertionError, RuntimeError) as error:
            raise SystemExit(
                f"checkpoint 与当前接口不兼容（obs={OBS_DIM}，actor={ACTOR_OBS_DIM}，"
                f"action={ACTION_DIM}）：{error}\n"
                "请使用同接口版本的 checkpoint，或去掉 --checkpoint 从头训练；"
                "旧版 236/6、226/6 与 73 维 checkpoint 不支持自动迁移。"
            )
        if model.policy.log_std is not None:
            import torch as _torch
            if args.action_std is None:
                model.policy.log_std.data.clamp_(-2.3, -0.9)   # std ∈ [0.10, 0.40]
            else:
                model.policy.log_std.data.fill_(float(np.log(max(args.action_std, 1.0e-3))))
        model.ent_coef = args.entropy_coef
        model.learning_rate = args.learning_rate
        for group in model.policy.optimizer.param_groups:
            group["lr"] = args.learning_rate
        print(f"[inherit] 已从 {checkpoint} 继承策略权重", flush=True)
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

    total_steps = args.updates * args.rollout_steps * args.num_envs
    print(f"train_config_stage: {args.stage}", flush=True)
    print(f"train_config_updates: {args.updates}", flush=True)
    print(f"train_config_total_steps: {total_steps}", flush=True)
    print(f"train_config_num_envs: {args.num_envs}", flush=True)
    print(f"train_config_learning_rate: {args.learning_rate}", flush=True)
    print(f"train_config_stand_level: {level.name}", flush=True)
    print(f"train_config_station_deadband_m: {level.deadband}", flush=True)
    print(f"train_config_station_hard_limit_m: {level.hard_limit}", flush=True)
    print(f"train_config_yaw_hard_limit_deg: {np.degrees(level.yaw_hard_limit):.2f}", flush=True)
    print("train_config_policy_assistance: none (PPO actions only)", flush=True)
    print(f"train_config_domain_randomization: {not args.disable_domain_randomization}", flush=True)
    print("", flush=True)
    model._total_timesteps = total_steps
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
                              domain_randomization=not args.disable_domain_randomization)
    print(f"checkpoint: {out}.zip", flush=True)
    print(f"vecnormalize: {out}_vecnormalize.pkl", flush=True)
    print(f"contract: {contract}", flush=True)
    print(f"tensorboard: {tb}", flush=True)


if __name__ == "__main__":
    main()
