"""验证站立时动态腿长目标的跟踪速度和鲁棒性。

示例::

    conda run --no-capture-output -n sim python scripts/validate_height_tracking.py \
        --episodes 5 --switch-steps 250 --rate-limit 0.90

该脚本只验证协同控制器基线，不加载 checkpoint；因此结果表示物理环路
在域随机化下的可行性，不代表 PPO 已经训练完成。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uz05.env import UZ05Env  # noqa: E402


def run_episode(seed: int, steps: int, switch_steps: int, rate_limit: float,
                rate_gain: float | None, feedforward_scale: float | None,
                target_rate: float | None,
                dr_scale: float,
                retract_rate_limit: float,
                retract_slow_rate_limit: float,
                retract_slow_error_m: float,
                retract_rate_gain: float,
                retract_feedforward_scale: float,
                retract_rate_damping: float,
                low_target_brake_damping_scale: float,
                reference_jump_reset_m: float,
                brake_error_m: float,
                target_rate_feedforward_scale: float,
                hold_error_m: float,
                hold_rate_limit: float,
                hold_rate_gain: float,
                height_filter_alpha: float) -> dict[str, float | int | str]:
    env = UZ05Env(
        stage="stand", stand_level=2, assist_scale=0.0, init_scale=1.0,
        coord_mix=1.0, height_switch_steps=switch_steps,
        height_switch_prob=1.0, height_rate_limit=rate_limit,
        height_rate_gain=rate_gain, leg_feedforward_scale=feedforward_scale,
        height_target_rate_m_s=target_rate,
        height_retract_rate_limit=retract_rate_limit,
        height_retract_slow_rate_limit=retract_slow_rate_limit,
        height_retract_slow_error_m=retract_slow_error_m,
        height_retract_rate_gain=retract_rate_gain,
        height_retract_feedforward_scale=retract_feedforward_scale,
        height_retract_rate_damping=retract_rate_damping,
        height_low_target_brake_damping_scale=low_target_brake_damping_scale,
        height_reference_jump_reset_m=reference_jump_reset_m,
        height_brake_error_m=brake_error_m,
        height_target_rate_feedforward_scale=target_rate_feedforward_scale,
        height_hold_error_m=hold_error_m,
        height_hold_rate_limit=hold_rate_limit,
        height_hold_rate_gain=hold_rate_gain,
        height_filter_alpha=height_filter_alpha,
        seed=seed,
    )
    env.stage = replace(env.stage, episode_steps=int(steps))
    env.params.domain_randomization.enabled = dr_scale > 0.0
    # Keep the same S2 randomization shape while allowing a lighter stress probe.
    env.stand_level = replace(env.stand_level, dr_scale=float(np.clip(dr_scale, 0.0, 1.0)))
    env.reset(seed=seed)
    max_error = 0.0
    max_drift = 0.0
    settle_samples: list[int] = []
    previous_switch = 0
    termination = "running"
    last_step = 0
    for step_idx in range(steps):
        _, _, terminated, truncated, info = env.step(np.zeros(6, dtype=np.float32))
        last_step = step_idx + 1
        max_error = max(max_error, abs(float(info["leg_length_error_mm"])))
        max_drift = max(max_drift, abs(float(info["station_error"])))
        switches = int(info["height_switch_count"])
        if switches > previous_switch:
            previous_switch = switches
        settle = int(info["height_settle_steps"])
        if settle >= 0 and len(settle_samples) < switches:
            settle_samples.append(settle)
        termination = str(info["termination_reason"])
        if terminated or truncated:
            break
    env.close()
    return {
        "steps": int(last_step),
        "switches": int(previous_switch),
        "max_error_mm": float(max_error),
        "max_drift_cm": float(max_drift * 100.0),
        "settle_p95_steps": float(np.percentile(settle_samples, 95)) if settle_samples else float("nan"),
        "termination": termination,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--switch-steps", type=int, default=250)
    parser.add_argument("--rate-limit", type=float, default=0.90)
    parser.add_argument("--rate-gain", type=float, default=50.0)
    parser.add_argument("--feedforward-scale", type=float, default=1.0)
    parser.add_argument("--target-rate", type=float, default=0.65,
                        help="高度目标斜坡速率（m/s）；0=阶跃目标")
    parser.add_argument("--retract-rate-limit", type=float, default=1.10)
    parser.add_argument("--retract-slow-rate-limit", type=float, default=0.90)
    parser.add_argument("--retract-slow-error-m", type=float, default=0.020)
    parser.add_argument("--retract-rate-gain", type=float, default=70.0)
    parser.add_argument("--retract-feedforward-scale", type=float, default=0.10)
    parser.add_argument("--retract-rate-damping", type=float, default=0.65)
    parser.add_argument("--low-target-brake-damping-scale", type=float, default=1.0)
    parser.add_argument("--reference-jump-reset-m", type=float, default=0.020)
    parser.add_argument("--brake-error-m", type=float, default=0.015)
    parser.add_argument("--target-rate-feedforward-scale", type=float, default=0.35)
    parser.add_argument("--hold-error-m", type=float, default=0.012)
    parser.add_argument("--hold-rate-limit", type=float, default=0.35)
    parser.add_argument("--hold-rate-gain", type=float, default=15.0)
    parser.add_argument("--height-filter-alpha", type=float, default=0.35)
    parser.add_argument("--dr-scale", type=float, default=1.0,
                        help="域随机化课程比例，0=关闭，1=S2 全量")
    args = parser.parse_args()

    results = [run_episode(seed=i, steps=args.steps,
                           switch_steps=args.switch_steps,
                           rate_limit=args.rate_limit,
                           rate_gain=args.rate_gain,
                           feedforward_scale=args.feedforward_scale,
                           target_rate=args.target_rate,
                           dr_scale=args.dr_scale,
                           retract_rate_limit=args.retract_rate_limit,
                           retract_slow_rate_limit=args.retract_slow_rate_limit,
                           retract_slow_error_m=args.retract_slow_error_m,
                           retract_rate_gain=args.retract_rate_gain,
                           retract_feedforward_scale=args.retract_feedforward_scale,
                           retract_rate_damping=args.retract_rate_damping,
                           low_target_brake_damping_scale=args.low_target_brake_damping_scale,
                           reference_jump_reset_m=args.reference_jump_reset_m,
                           brake_error_m=args.brake_error_m,
                           target_rate_feedforward_scale=args.target_rate_feedforward_scale,
                           hold_error_m=args.hold_error_m,
                           hold_rate_limit=args.hold_rate_limit,
                           hold_rate_gain=args.hold_rate_gain,
                           height_filter_alpha=args.height_filter_alpha)
               for i in range(args.episodes)]
    survived = sum(r["termination"] == "time_limit" for r in results)
    print("episode steps switches max_err_mm drift_cm settle_p95 result")
    for i, r in enumerate(results):
        print(f"{i:7d} {r['steps']:5d} {r['switches']:8d} "
              f"{r['max_error_mm']:11.2f} {r['max_drift_cm']:8.3f} "
              f"{r['settle_p95_steps']:14.1f} {r['termination']}")
    print(f"summary survived={survived}/{len(results)} rate_limit={args.rate_limit:.2f} "
          f"switch_steps={args.switch_steps} dr_scale={args.dr_scale:.2f}")


if __name__ == "__main__":
    main()
