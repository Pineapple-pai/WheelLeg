"""复现网页快速档的腿长往返验收。

该脚本不加载 PPO，直接调用与网页回放相同的腿长目标斜坡和协同控制器，
用于区分“控制器/执行器响应问题”和“策略训练问题”。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uz05.env import UZ05Env  # noqa: E402


PROFILE = dict(
    height_rate_limit=0.90,
    height_rate_gain=50.0,
    height_reference_jump_reset_m=0.020,
    height_brake_error_m=0.015,
    height_target_rate_feedforward_scale=0.35,
    height_hold_error_m=0.006,
    height_hold_rate_limit=0.30,
    height_hold_rate_gain=40.0,
    height_filter_alpha=0.75,
    height_rate_damping=0.50,
    height_retract_rate_damping=0.65,
    height_low_target_brake_damping_scale=1.0,
    height_rate_brake_threshold=0.005,
    height_retract_rate_limit=1.10,
    height_retract_slow_rate_limit=0.90,
    height_retract_slow_error_m=0.020,
    height_retract_rate_gain=70.0,
    height_retract_feedforward_scale=0.10,
    leg_feedforward_scale=1.0,
)


def run_episode(seed: int, init_scale: float, domain_randomization: bool,
                target_rate: float, segment_steps: int) -> dict:
    targets = (0.270, 0.150, 0.270)
    env = UZ05Env(stage="stand", stand_level=2, assist_scale=0.0,
                  seed=seed, init_scale=init_scale, coord_mix=1.0,
                  **PROFILE)
    env.params.domain_randomization.enabled = bool(domain_randomization)
    env.stage = replace(env.stage, episode_steps=3 * segment_steps + 10)
    env.reset(seed=seed)
    internal = float(np.mean(env.sim.leg_lengths()))
    dt = float(env.params.control_dt)
    segment_results = []
    failed = False

    for target in targets:
        leg = []
        drift = []
        torque = []
        error = []
        leg_rate = []
        height_command = []
        for _ in range(segment_steps):
            internal += float(np.clip(
                target - internal,
                -target_rate * dt,
                target_rate * dt,
            ))
            env.set_leg_length_command(internal)
            _, _, terminated, _, info = env.step(
                np.zeros(6, dtype=np.float32)
            )
            leg.append(float(np.mean(env.sim.leg_lengths())))
            drift.append(abs(float(info["station_error"])) * 100.0)
            torque.append(float(info.get("leg_torque_abs_mean", 0.0)))
            error.append(float(info["leg_length_error_mm"]))
            leg_rate.append(float(info.get("leg_length_rate_mean", 0.0)))
            height_command.append(float(env._coord_action[3]))
            if terminated:
                failed = True
                break

        values = np.asarray(error, dtype=np.float64)
        inside = np.flatnonzero(np.abs(values) <= 5.0)
        settle_s = None
        if inside.size:
            for index in inside:
                if np.all(np.abs(values[index:]) <= 5.0):
                    settle_s = float((index + 1) * dt)
                    break
        segment_results.append({
            "target": target,
            "settle_s": settle_s,
            "final_error_mm": float(values[-1]) if values.size else float("nan"),
            "peak_error_mm": float(np.max(np.abs(values))) if values.size else float("nan"),
            "peak_drift_cm": float(max(drift)) if drift else float("nan"),
            "peak_torque_nm": float(max(torque)) if torque else float("nan"),
            "min_leg_m": float(min(leg)) if leg else float("nan"),
            "tail_error_mean_mm": float(np.mean(values[-200:])) if values.size else float("nan"),
            "tail_error_std_mm": float(np.std(values[-200:])) if values.size else float("nan"),
            "tail_leg_rate_rms_m_s": float(
                np.sqrt(np.mean(np.square(np.asarray(leg_rate[-200:], dtype=np.float64))))
            ) if leg_rate else float("nan"),
            "tail_height_action_rate_rms_s": float(
                np.sqrt(np.mean(np.square(np.diff(height_command[-200:]))))
                / max(dt, 1e-9)
            ) if len(height_command) >= 2 else float("nan"),
        })
        if failed:
            break

    env.close()
    return {"failed": failed, "segments": segment_results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--init-scale", type=float, default=0.0)
    parser.add_argument("--target-rate", type=float, default=0.65)
    parser.add_argument("--segment-steps", type=int, default=1500)
    parser.add_argument("--domain-randomization", action="store_true")
    args = parser.parse_args()

    results = [run_episode(
        seed=args.seed + index,
        init_scale=args.init_scale,
        domain_randomization=args.domain_randomization,
        target_rate=args.target_rate,
        segment_steps=args.segment_steps,
    ) for index in range(args.episodes)]
    survived = [item for item in results if not item["failed"]]
    print("episode failed  high1_settle_s  low_settle_s  high2_settle_s "
          "peak_drift_cm  peak_torque_nm  min_leg_m  tail_err_mm  "
          "tail_rate_m_s  cmd_rate_s")
    for index, item in enumerate(results):
        seg = item["segments"] + [{"settle_s": None} for _ in range(
            3 - len(item["segments"])
        )]
        all_segments = item["segments"]
        print(f"{index:7d} {str(item['failed']):>6s} "
              f"{seg[0]['settle_s'] if seg[0]['settle_s'] is not None else float('nan'):14.3f} "
              f"{seg[1]['settle_s'] if seg[1]['settle_s'] is not None else float('nan'):12.3f} "
              f"{seg[2]['settle_s'] if seg[2]['settle_s'] is not None else float('nan'):13.3f} "
              f"{max((x['peak_drift_cm'] for x in all_segments), default=float('nan')):13.2f} "
              f"{max((x['peak_torque_nm'] for x in all_segments), default=float('nan')):15.2f} "
              f"{min((x['min_leg_m'] for x in all_segments), default=float('nan')):10.4f} "
              f"{max((abs(x['tail_error_mean_mm']) for x in all_segments), default=float('nan')):11.2f} "
              f"{max((x['tail_leg_rate_rms_m_s'] for x in all_segments), default=float('nan')):13.3f} "
              f"{max((x['tail_height_action_rate_rms_s'] for x in all_segments), default=float('nan')):10.2f}")
    print(f"summary survived={len(survived)}/{len(results)} "
          f"domain_randomization={args.domain_randomization} "
          f"target_rate={args.target_rate:.2f} m/s")


if __name__ == "__main__":
    main()
