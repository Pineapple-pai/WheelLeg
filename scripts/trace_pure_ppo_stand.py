#!/usr/bin/env python3
"""Record step-level actuator and body data for held-out Pure PPO episodes.

This is a read-only diagnostic: it loads a versioned checkpoint, keeps the
training interface fixed, and writes compressed trace data plus summary stats.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from uz05.pure_ppo_contract import read_and_require_contract
from uz05.pure_ppo_stand import EPISODE_STEPS, PurePPOStandEnv


ROOT = Path(__file__).resolve().parents[1]
TRACE_SEED_BASE = {0: 20_000, 1: 21_000, 2: 22_000}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    contract = read_and_require_contract(
        checkpoint.with_name(f"{checkpoint.stem}_contract.json")
    )
    if int(contract["seed"]) != args.training_seed:
        raise ValueError("checkpoint contract seed does not match --training-seed")
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    model = PPO.load(str(checkpoint), device="cpu")
    env = PurePPOStandEnv()
    headers = [
        "seed", "step", "time_s", "initial_pitch_rad", "initial_pitch_rate_rad_s",
        "pitch_rad", "roll_rad", "pitch_rate_rad_s", "base_vx_m_s", "base_vy_m_s",
        "base_height_m", "planar_drift_m", "action_0", "action_1", "action_2",
        "action_3", "action_4", "action_5", "wheel_target_left_rad_s",
        "wheel_target_right_rad_s", "wheel_speed_before_left_rad_s",
        "wheel_speed_before_right_rad_s", "wheel_speed_after_left_rad_s",
        "wheel_speed_after_right_rad_s", "wheel_current_left_a", "wheel_current_right_a",
        "wheel_torque_left_nm", "wheel_torque_right_nm", "joint_target_0_rad",
        "joint_target_1_rad", "joint_target_2_rad", "joint_target_3_rad",
        "joint_position_0_rad", "joint_position_1_rad", "joint_position_2_rad",
        "joint_position_3_rad", "joint_velocity_0_rad_s", "joint_velocity_1_rad_s",
        "joint_velocity_2_rad_s", "joint_velocity_3_rad_s", "joint_torque_0_nm",
        "joint_torque_1_nm", "joint_torque_2_nm", "joint_torque_3_nm",
        "left_wheel_contact", "right_wheel_contact", "body_contact", "reward",
        "wheel_current_saturated_left", "wheel_current_saturated_right",
        "wheel_torque_saturated_left", "wheel_torque_saturated_right",
        "leg_torque_saturated_0", "leg_torque_saturated_1",
        "leg_torque_saturated_2", "leg_torque_saturated_3", "termination_reason",
    ]
    params = env.params
    wheel_current_limit = float(params.wheel.current_limit_a)
    wheel_torque_limit = float(params.wheel.joint_torque_limit)
    leg_torque_limit = float(params.joint.torque_limit)
    episode_summaries = []
    csv_path = output_dir / "step_trace.csv.gz"
    try:
        with gzip.open(csv_path, "wt", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=headers)
            writer.writeheader()
            for seed in range(
                TRACE_SEED_BASE[args.training_seed],
                TRACE_SEED_BASE[args.training_seed] + args.episodes,
            ):
                observation, reset_info = env.reset(seed=seed)
                local_rows = 0
                wheel_speed_abs_peak = np.zeros(2, dtype=np.float64)
                terminal_reason = "not_finished"
                for step in range(1, EPISODE_STEPS + 1):
                    action, _ = model.predict(observation, deterministic=True)
                    before = env.sim.wheel_velocities().copy()
                    observation, reward, terminated, truncated, info = env.step(action)
                    after = env.sim.wheel_velocities().copy()
                    wheel_speed_abs_peak = np.maximum(
                        wheel_speed_abs_peak, np.maximum(np.abs(before), np.abs(after))
                    )
                    q = env.sim.joint_positions().copy()
                    qd = env.sim.joint_velocities().copy()
                    joint_tau = np.asarray(info["joint_torque_nm"], dtype=np.float64)
                    wheel_tau = np.asarray(info["wheel_torque_nm"], dtype=np.float64)
                    wheel_current = np.asarray(info["wheel_current_a"], dtype=np.float64)
                    contact = np.asarray(info["wheel_contact"], dtype=np.float64)
                    values = {
                        "seed": seed,
                        "step": step,
                        "time_s": step * 0.008,
                        "initial_pitch_rad": reset_info["initial_pitch_rad"],
                        "initial_pitch_rate_rad_s": reset_info["initial_pitch_rate_rad_s"],
                        "pitch_rad": info["pitch_rad"],
                        "roll_rad": info["roll_rad"],
                        "pitch_rate_rad_s": info["pitch_rate_rad_s"],
                        "base_vx_m_s": info["base_vx_m_s"],
                        "base_vy_m_s": info["base_vy_m_s"],
                        "base_height_m": info["base_height_m"],
                        "planar_drift_m": info["planar_drift_m"],
                        **{f"action_{i}": float(action[i]) for i in range(6)},
                        "wheel_target_left_rad_s": info["wheel_target_rad_s"][0],
                        "wheel_target_right_rad_s": info["wheel_target_rad_s"][1],
                        "wheel_speed_before_left_rad_s": before[0],
                        "wheel_speed_before_right_rad_s": before[1],
                        "wheel_speed_after_left_rad_s": after[0],
                        "wheel_speed_after_right_rad_s": after[1],
                        "wheel_current_left_a": wheel_current[0],
                        "wheel_current_right_a": wheel_current[1],
                        "wheel_torque_left_nm": wheel_tau[0],
                        "wheel_torque_right_nm": wheel_tau[1],
                        **{f"joint_target_{i}_rad": info["joint_target_rad"][i] for i in range(4)},
                        **{f"joint_position_{i}_rad": q[i] for i in range(4)},
                        **{f"joint_velocity_{i}_rad_s": qd[i] for i in range(4)},
                        **{f"joint_torque_{i}_nm": joint_tau[i] for i in range(4)},
                        "left_wheel_contact": contact[0],
                        "right_wheel_contact": contact[1],
                        "body_contact": info["body_contact"],
                        "reward": reward,
                        "wheel_current_saturated_left": abs(wheel_current[0]) >= 0.99 * wheel_current_limit,
                        "wheel_current_saturated_right": abs(wheel_current[1]) >= 0.99 * wheel_current_limit,
                        "wheel_torque_saturated_left": abs(wheel_tau[0]) >= 0.99 * wheel_torque_limit,
                        "wheel_torque_saturated_right": abs(wheel_tau[1]) >= 0.99 * wheel_torque_limit,
                        **{f"leg_torque_saturated_{i}": abs(joint_tau[i]) >= 0.99 * leg_torque_limit for i in range(4)},
                        "termination_reason": info["termination_reason"],
                    }
                    writer.writerow(values)
                    local_rows += 1
                    terminal_reason = info["termination_reason"]
                    if terminated or truncated:
                        break
                episode_summaries.append({
                    "seed": seed,
                    "steps": local_rows,
                    "termination_reason": terminal_reason,
                    "wheel_speed_abs_peak_rad_s": float(np.max(wheel_speed_abs_peak)),
                    "wheel_speed_abs_peak_each_rad_s": wheel_speed_abs_peak.tolist(),
                    **(info.get("episode_metrics") or {}),
                })
                stream.flush()
                if len(episode_summaries) % 10 == 0:
                    print(f"traced {len(episode_summaries)}/{args.episodes} episodes", flush=True)
    finally:
        env.close()

    (output_dir / "episode_metrics.json").write_text(
        json.dumps(episode_summaries, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    scalar_stats = {}
    keys = (
        "pitch_rms_rad", "pitch_peak_rad", "drift_max_m", "wheel_speed_abs_peak_rad_s",
        "wheel_current_saturation_fraction", "wheel_torque_saturation_fraction",
        "wheel_saturation_fraction", "leg_torque_saturation_fraction",
    )
    for key in keys:
        values = [row[key] for row in episode_summaries if key in row]
        if values:
            samples = np.asarray(values, dtype=np.float64)
            if samples.ndim == 1:
                scalar_stats[key] = {
                    "mean": float(np.mean(samples)),
                    "p95": float(np.percentile(samples, 95)),
                    "max": float(np.max(samples)),
                }
            else:
                scalar_stats[key] = {
                    "mean_by_channel": np.mean(samples, axis=0).tolist(),
                    "p95_by_channel": np.percentile(samples, 95, axis=0).tolist(),
                    "max_by_channel": np.max(samples, axis=0).tolist(),
                }
    report = {
        "checkpoint": str(checkpoint),
        "training_seed": args.training_seed,
        "held_out_seed_start": TRACE_SEED_BASE[args.training_seed],
        "episodes": args.episodes,
        "trace_csv_gzip": str(csv_path),
        "trace_rows": int(sum(row["steps"] for row in episode_summaries)),
        "episode_metric_stats": scalar_stats,
        "note": "Actual wheel speeds are logged before and after each 8 ms policy step; controller targets and actuator outputs are logged for that same step.",
    }
    (output_dir / "trace_summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
