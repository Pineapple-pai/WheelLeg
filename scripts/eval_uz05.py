"""确定性评估 UZ-05 PPO checkpoint 的站立质量。"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from train_uz05 import AsymmetricActorCriticPolicy
from uz05.env import UZ05Env
from uz05.spec import EnvParams


def main() -> None:
    parser = argparse.ArgumentParser(description="评估 UZ-05 PPO checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stage", default="stand")
    parser.add_argument("--stand-level", type=int, default=2)
    parser.add_argument("--assist", type=float, default=0.0)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--init-scale", type=float, default=0.0,
                        help="初始姿态/速度/关节扰动比例；0 表示额定静止站姿")
    parser.add_argument("--policy-current-scale", type=float, default=None,
                        help="仅用于增益扫描，覆盖每单位轮动作对应的电流(A)")
    parser.add_argument("--action-filter-alpha", type=float, default=1.0,
                        help="站立轮/腿动作一阶滤波系数；1 表示不滤波")
    parser.add_argument("--pitch-angle-correction", type=float, default=5.0,
                        help="站立共模轮俯仰角修正 A/rad，正值抵消 pitch")
    parser.add_argument("--pitch-rate-damping", type=float, default=2.0,
                        help="站立共模轮电流速度阻尼 A/(rad/s)，正值抵消 pitch_rate")
    parser.add_argument("--unlock-stand-legs", action="store_true",
                        help="站立时保留策略腿关节动作，用于验证腿部动态支撑")
    parser.add_argument("--stand-leg-action-limit", type=float, default=0.15,
                        help="解锁站立腿动作的归一化限幅")
    parser.add_argument("--zero-action", action="store_true",
                        help="不加载策略，输出全零动作以验证环境/辅助基准")
    parser.add_argument("--accept-tilt-deg", type=float, default=17.0,
                        help="末端直立验收的最大 roll/pitch 绝对值（度）")
    parser.add_argument("--accept-height-ratio", type=float, default=0.80,
                        help="末端高度相对 nominal_stand_height 的最低比例")
    parser.add_argument("--accept-peak-drift-cm", type=float, default=5.0,
                        help="全程前向漂移峰值验收上限（cm）")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    model = None if args.zero_action else PPO.load(str(checkpoint), custom_objects={
        "policy_class": AsymmetricActorCriticPolicy,
    }, device="cpu")
    outcomes: list[dict[str, float | str | bool]] = []
    action_abs: list[float] = []
    wheel_common_abs: list[float] = []
    wheel_differential_abs: list[float] = []
    leg_action_abs: list[float] = []
    policy_current_abs: list[float] = []
    current_abs: list[float] = []
    leg_torque_abs: list[float] = []
    final_tilt: list[float] = []
    final_height: list[float] = []
    pitch_rms: list[float] = []
    pitch_rate_rms: list[float] = []
    common_action_jerk: list[float] = []
    reasons: Counter[str] = Counter()

    for episode in range(args.episodes):
        env = UZ05Env(stage=args.stage, stand_level=args.stand_level,
                      assist_scale=args.assist, seed=args.seed + episode,
                      init_scale=args.init_scale,
                      action_filter_alpha=args.action_filter_alpha,
                      pitch_angle_correction_a_per_rad=args.pitch_angle_correction,
                      pitch_rate_damping_a_per_rad_s=args.pitch_rate_damping,
                      lock_stand_leg_actions=not args.unlock_stand_legs,
                      stand_leg_action_limit=args.stand_leg_action_limit)
        if args.policy_current_scale is not None:
            env.params.wheel.policy_current_scale_a = args.policy_current_scale
        obs, _ = env.reset(seed=args.seed + episode)
        done = False
        info: dict = {}
        episode_pitch: list[float] = []
        episode_pitch_rate: list[float] = []
        episode_common: list[float] = []
        while not done:
            action = np.zeros(env.action_space.shape, dtype=np.float32) if model is None else model.predict(
                obs, deterministic=True,
            )[0]
            action_abs.append(float(np.abs(action).mean()))
            # 策略前两维已经是 [common, differential] 基，不是左右轮。
            wheel_common_abs.append(float(abs(action[0])))
            wheel_differential_abs.append(float(abs(action[1])))
            leg_action_abs.append(float(np.abs(action[2:]).mean()))
            obs, _, terminated, truncated, info = env.step(action)
            episode_pitch.append(float(info.get("pitch", 0.0)))
            episode_pitch_rate.append(float(info.get("pitch_rate", 0.0)))
            episode_common.append(float(np.asarray(action).reshape(-1)[0]))
            current_abs.append(float(np.abs([
                info["wheel_current_left"], info["wheel_current_right"],
            ]).mean()))
            leg_torque_abs.append(float(info.get("leg_torque_abs_mean", 0.0)))
            policy_current_abs.append(float(np.abs([
                info["wheel_target_left"], info["wheel_target_right"],
            ]).mean()))
            done = terminated or truncated
        reason = str(info["termination_reason"])
        reasons[reason] += 1
        outcomes.append({
            "full": bool(truncated and not terminated),
            "tail": float(info["station_tail_mean"]),
            "peak": float(info["station_max_abs"]),
            "tilt": max(abs(float(info.get("roll", 99.0))),
                        abs(float(info.get("pitch", 99.0)))),
            "height": float(info.get("base_height", 0.0)),
            "steps": float(info["episode_steps"]),
            "reason": reason,
        })
        final_tilt.append(outcomes[-1]["tilt"])
        final_height.append(outcomes[-1]["height"])
        pitch_rms.append(float(np.sqrt(np.mean(np.square(episode_pitch)))))
        pitch_rate_rms.append(float(np.sqrt(np.mean(np.square(episode_pitch_rate)))))
        common = np.asarray(episode_common, dtype=np.float64)
        common_action_jerk.append(
            float(np.sqrt(np.mean(np.square(np.diff(common, n=2)))))
            if common.size >= 3 else 0.0
        )
        env.close()

    full = np.asarray([item["full"] for item in outcomes], dtype=np.float64)
    tail = np.asarray([item["tail"] for item in outcomes], dtype=np.float64)
    peak = np.asarray([item["peak"] for item in outcomes], dtype=np.float64)
    tilt = np.asarray(final_tilt, dtype=np.float64)
    height = np.asarray(final_height, dtype=np.float64)
    steps = np.asarray([item["steps"] for item in outcomes], dtype=np.float64)
    tilt_ok = tilt <= np.deg2rad(args.accept_tilt_deg)
    height_ok = height >= args.accept_height_ratio * EnvParams().robot.nominal_stand_height
    peak_ok = peak <= args.accept_peak_drift_cm / 100.0
    accepted = (full * (tail < 0.05) * peak_ok * tilt_ok * height_ok
                * float(args.assist <= 1e-3))
    print(f"checkpoint: {checkpoint}")
    print(f"eval: stand_level={args.stand_level} assist={args.assist:.3f} episodes={args.episodes}")
    print(f"survive_rate: {full.mean():.3f} ({int(full.sum())}/{args.episodes})")
    print(f"tail_drift_cm_mean: {100.0 * tail.mean():.2f}")
    print(f"tail_drift_cm_p95: {100.0 * np.quantile(tail, 0.95):.2f}")
    print(f"peak_drift_cm_p95: {100.0 * np.quantile(peak, 0.95):.2f}")
    print(f"peak_drift_ok_rate: {peak_ok.mean():.3f} ({int(peak_ok.sum())}/{args.episodes})")
    print(f"final_tilt_deg_mean: {np.degrees(tilt).mean():.2f}")
    print(f"final_tilt_deg_p95: {np.degrees(np.quantile(tilt, 0.95)):.2f}")
    print(f"pitch_rms_deg_mean: {np.degrees(np.mean(pitch_rms)):.3f}")
    print(f"pitch_rate_rms_mean: {np.mean(pitch_rate_rms):.5f} rad/s")
    print(f"common_action_jerk_rms_mean: {np.mean(common_action_jerk):.5f}")
    print(f"upright_rate: {tilt_ok.mean():.3f} ({int(tilt_ok.sum())}/{args.episodes})")
    print(f"height_ok_rate: {height_ok.mean():.3f} ({int(height_ok.sum())}/{args.episodes})")
    print(f"accept_5cm_rate: {accepted.mean():.3f} ({int(accepted.sum())}/{args.episodes})")
    print(f"episode_steps_mean: {steps.mean():.1f}")
    print(f"policy_action_abs_mean: {np.mean(action_abs):.4f}")
    print(f"policy_wheel_common_abs_mean: {np.mean(wheel_common_abs):.4f}")
    print(f"policy_wheel_differential_abs_mean: {np.mean(wheel_differential_abs):.4f}")
    print(f"policy_leg_action_abs_mean: {np.mean(leg_action_abs):.4f}")
    print(f"policy_current_abs_mean_A: {np.mean(policy_current_abs):.3f}")
    print(f"wheel_current_abs_mean_A: {np.mean(current_abs):.3f}")
    print(f"leg_torque_abs_mean_Nm: {np.mean(leg_torque_abs):.3f}")
    print("terminations: " + ", ".join(
        f"{reason}={count}" for reason, count in sorted(reasons.items())
    ))


if __name__ == "__main__":
    main()
