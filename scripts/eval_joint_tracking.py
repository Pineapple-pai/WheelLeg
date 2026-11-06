"""Evaluate independent held commands, including zero speed, at several heights."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from train_uz05 import require_compatible_contract
from uz05.env import UZ05Env
from uz05.spec import EnvParams, env_params_for_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--episodes-per-case", type=int, default=10)
    parser.add_argument("--duration", type=float, default=None, help="episode horizon in seconds")
    parser.add_argument("--perturbations", action="store_true", help="scheduled external pushes")
    parser.add_argument("--domain-randomization", action="store_true")
    parser.add_argument("--moving-starts", action="store_true", help="include the training mixture of moving initial states")
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--output", default="")
    parser.add_argument("--ablation", choices=("none", "wheel_zero", "joint_hold"), default="none")
    parser.add_argument("--ablation-start-seconds", type=float, default=2.0)
    parser.add_argument("--heights", type=float, nargs="+", default=None)
    parser.add_argument("--speed", type=float, default=None, help="translation speed magnitude in m/s")
    parser.add_argument("--cases", nargs="+", choices=("stand", "forward", "reverse", "yaw_left", "yaw_right", "mixed"))
    args = parser.parse_args()
    if args.episodes_per_case < 1:
        parser.error("--episodes-per-case must be positive")
    checkpoint = Path(args.checkpoint).expanduser().with_suffix("")
    contract = json.loads(Path(f"{checkpoint}_contract.json").read_text())
    stage = contract["stage"]
    require_compatible_contract(checkpoint, stage)
    if args.duration is not None and args.duration < .008:
        parser.error("--duration must be >= 0.008")
    params = env_params_for_stage(stage)
    params.domain_randomization.enabled = args.domain_randomization
    params.robust.pushes_enabled = args.perturbations
    raw = UZ05Env(stage, params=params)
    if not args.moving_starts:
        raw.stage = replace(raw.stage, moving_start_probability=0.0)
    if args.heights is None:
        args.heights = [0.22, 0.27, 0.32] if raw.stage.transition_tracking else [0.245, 0.27, 0.295]
    speed = args.speed if args.speed is not None else (2.0 if raw.stage.transition_tracking else .3)
    if not np.isfinite(speed) or speed <= 0:
        parser.error("--speed must be finite and positive")
    if args.duration is not None:
        raw.stage = replace(raw.stage, episode_steps=int(args.duration / raw.params.control_dt))
    env = VecNormalize.load(f"{checkpoint}_vecnormalize.pkl", DummyVecEnv([lambda: raw]))
    env.training = False
    env.norm_reward = False
    model = PPO.load(checkpoint, env=env, device="cpu")
    results = []
    # Include held-out intermediate heights and both velocity signs.
    cases = {"stand": (0, 0), "forward": (speed, 0), "reverse": (-speed, 0),
             "yaw_left": (0, 1), "yaw_right": (0, -1), "mixed": (speed, -1)}
    for height in args.heights:
        for case in args.cases or cases:
            vx, yaw = cases[case]
            errors, lengths, survivors, leg_vel, wheel_vel, stop_drift = [], [], [], [], [], []
            settled_errors, angles, saturations, failures, episode_details = [], [], [], Counter(), []
            stance_samples, output_samples, hold_samples, support_samples = [], [], [], []
            settled_support, settled_leg_spans = [], []
            settled_posture = []
            settled_legs, settled_fore_aft_difference = [], []
            settled_velocities, settled_heights, yaw_peaks = [], [], []
            envelope_samples = []
            axle_samples, post_first_second_axles = [], []
            for episode in range(args.episodes_per_case):
                env.seed(args.seed + episode)
                env.reset()
                raw.command[:] = [vx, 0, yaw, height, 0]
                raw.command_target[:] = raw.command
                raw.command_steps_left = raw.stage.episode_steps + 1
                obs = raw._obs()[None, :]
                held_position_action = np.arctan2(np.sin(raw.sim.joint_positions()),
                                                  np.cos(raw.sim.joint_positions())) / np.pi
                episode_errors = []
                episode_axles = []
                episode_legs = []
                peak_tilt = 0.0
                previous_yaw = raw.nominal_yaw
                yaw_change = yaw_peak = 0.0
                for step in range(raw.stage.episode_steps):
                    action, _ = model.predict(obs, deterministic=True)
                    intervention_step = int(args.ablation_start_seconds / raw.params.control_dt)
                    if step == intervention_step:
                        held_position_action = np.arctan2(np.sin(raw.sim.joint_positions()),
                                                          np.cos(raw.sim.joint_positions())) / np.pi
                    if step >= intervention_step:
                        if args.ablation == "wheel_zero":
                            action[:, :2] = 0.0
                        elif args.ablation == "joint_hold":
                            action[:, 2:] = held_position_action
                    obs, _, done, infos = env.step(action)
                    info = infos[0]
                    # VecEnv may auto-reset on done; use post-step geometry
                    # supplied in info when available, otherwise the live state.
                    if 'leg_forward_reach_left_m' in info:
                        envelope = info
                    elif not done[0]:
                        envelope = raw.sim.leg_envelope()
                    else:
                        envelope = None
                    if envelope is not None:
                        envelope_samples.append([envelope[f'leg_{direction}_reach_{side}_m']
                                                 for direction in ('forward','backward') for side in ('left','right')])
                    delta_yaw = info["yaw"] - previous_yaw
                    yaw_change += float(np.arctan2(np.sin(delta_yaw), np.cos(delta_yaw)))
                    previous_yaw = info["yaw"]
                    yaw_peak = max(yaw_peak, abs(yaw_change))
                    errors.append([info["tracking_vx_error"], info["tracking_yaw_error"],
                                   info["tracking_height_error"]])
                    support_samples.append([info["leg_drop_difference_m"], info["leg_differential_velocity_m_s"], info["wheel_dual_support"], info["push_force_n"]])
                    output_samples.append([info["wheel_target_change_sq"], info["joint_target_change_sq"],
                                           info["wheel_saturation_fraction"], info["effort_change_sq"]])
                    if info["zero_command"]:
                        hold_samples.append([info["hold_planar_error_m"], info["hold_heading_error_rad"]])
                    stance_samples.append([info["wheel_stagger_m"], info["wheel_offset_max_m"]])
                    axle_samples.append(info['axle_line_offset_m'])
                    episode_axles.append(info['axle_line_offset_m'])
                    episode_errors.append(errors[-1])
                    if step >= 125:
                        post_first_second_axles.append(info['axle_line_offset_m'])
                        settled_errors.append(errors[-1])
                        # Tracking errors refer to the post-step state; body_vx
                        # in historical info dictionaries refers to pre-step.
                        settled_velocities.append([vx + info["tracking_vx_error"], yaw + info["tracking_yaw_error"]])
                        settled_heights.append(height + info["tracking_height_error"])
                        settled_support.append(support_samples[-1])
                        episode_legs.append([info["leg_length_left"], info["leg_length_right"]])
                        settled_posture.append([info["pitch"], info.get("wheel_acceleration_sq", 0.0), info["roll"]])
                        settled_legs.append(episode_legs[-1])
                        settled_fore_aft_difference.append(info["wheel_fore_aft_left_m"] - info["wheel_fore_aft_right_m"])
                    peak_tilt = max(peak_tilt, abs(info["roll"]), abs(info["pitch"]))
                    angles.append([info[f"hip_pos_{j}"] for j in range(4)])
                    saturations.append(info["joint_saturation_per_joint"])
                    leg_vel.append([info[f"hip_vel_{j}"] for j in range(4)])
                    wheel_vel.append([info["wheel_vel_left"], info["wheel_vel_right"]])
                    if done[0]:
                        lengths.append(step + 1)
                        survivors.append(info["termination_reason"] == "time_limit")
                        failures[info["termination_reason"]] += 1
                        episode_details.append({
                            "seed": args.seed + episode, "steps": step + 1,
                            "reason": info["termination_reason"], "peak_tilt_deg": float(np.degrees(peak_tilt)),
                            "rmse_vx_yaw_height": np.sqrt(np.square(episode_errors).mean(axis=0)).tolist(),
                            "yaw_change_peak_deg": float(np.degrees(yaw_peak)),
                            "axle_line_offset_peak_m": float(np.max(episode_axles)),
                            "post_first_second_axle_mean_m": (float(np.mean(episode_axles[125:]))
                                                                 if len(episode_axles)>125 else None),
                        })
                        if vx == 0 and yaw == 0:
                            stop_drift.append(info["station_max_abs"])
                        break
                yaw_peaks.append(yaw_peak)
                if episode_legs:
                    settled_leg_spans.append(np.ptp(episode_legs, axis=0))
            results.append({
                "axle_line_offset_peak_m": float(np.max(axle_samples)),
                "post_first_second_axle_mean_m": (float(np.mean(post_first_second_axles)) if post_first_second_axles else None),
                "post_first_second_axle_p95_m": (float(np.percentile(post_first_second_axles,95)) if post_first_second_axles else None),
                "leg_reach_peak_forwardL_forwardR_backwardL_backwardR_m": np.max(envelope_samples,axis=0).tolist() if envelope_samples else None,
                "settled_vx_mean_m_s": float(np.mean(np.asarray(settled_velocities)[:, 0])) if settled_velocities else None,
                "settled_yaw_rate_mean_rad_s": float(np.mean(np.asarray(settled_velocities)[:, 1])) if settled_velocities else None,
                "yaw_change_peak_deg": float(np.degrees(max(yaw_peaks))),
                "settled_height_std_m": float(np.std(settled_heights)) if settled_heights else None,
                "settled_height_mean_m": float(np.mean(settled_heights)) if settled_heights else None,
                "settled_pitch_peak_deg": float(np.degrees(np.max(np.abs(np.asarray(settled_posture)[:, 0])))) if settled_posture else None,
                "settled_leg_mean_left_right_m": np.mean(settled_legs, axis=0).tolist() if settled_legs else None,
                "settled_leg_difference_rms_m": float(np.sqrt(np.mean(np.diff(settled_legs, axis=1)**2))) if settled_legs else None,
                "settled_leg_difference_peak_m": float(np.max(np.abs(np.diff(settled_legs, axis=1)))) if settled_legs else None,
                "settled_leg_drop_difference_rms_m": float(np.sqrt(np.mean(np.asarray(settled_support)[:, 0]**2))) if settled_support else None,
                "settled_wheel_fore_aft_difference_rms_m": float(np.sqrt(np.mean(np.square(settled_fore_aft_difference)))) if settled_fore_aft_difference else None,
                "settled_roll_mean_deg": float(np.degrees(np.mean(np.asarray(settled_posture)[:, 2]))) if settled_posture else None,
                "settled_roll_rms_deg": float(np.degrees(np.sqrt(np.mean(np.asarray(settled_posture)[:, 2]**2)))) if settled_posture else None,
                "settled_pitch_mean_deg": float(np.degrees(np.mean(np.asarray(settled_posture)[:, 0]))) if settled_posture else None,
                "settled_pitch_rms_deg": float(np.degrees(np.sqrt(np.mean(np.asarray(settled_posture)[:, 0]**2)))) if settled_posture else None,
                "settled_pitch_within_5deg_fraction": float(np.mean(np.abs(np.asarray(settled_posture)[:, 0]) <= np.radians(5))) if settled_posture else None,
                "settled_wheel_acceleration_rms_rad_s2": float(np.sqrt(np.mean(np.asarray(settled_posture)[:, 1]))) if settled_posture and raw.stage.pitch_control else None,
                "settled_leg_length_span_mean_m": np.mean(settled_leg_spans, axis=0).tolist() if settled_leg_spans else None,
                "settled_leg_differential_velocity_rms_m_s": float(np.sqrt(np.mean(np.asarray(settled_support)[:, 1]**2))) if settled_support else None,
                "settled_dual_wheel_support_fraction": float(np.mean(np.asarray(settled_support)[:, 2])) if settled_support else None,
                "leg_drop_difference_rms_m": float(np.sqrt(np.mean(np.asarray(support_samples)[:, 0]**2))),
                "leg_differential_velocity_rms_m_s": float(np.sqrt(np.mean(np.asarray(support_samples)[:, 1]**2))),
                "dual_wheel_support_fraction": float(np.mean(np.asarray(support_samples)[:, 2])),
                "push_active_fraction": float(np.mean(np.asarray(support_samples)[:, 3] > 0)),
                "wheel_target_change_rms_rad_s": float(np.sqrt(np.mean(np.asarray(output_samples)[:, 0]))),
                "joint_target_change_rms_rad": float(np.sqrt(np.mean(np.asarray(output_samples)[:, 1]))),
                "wheel_saturation_fraction": float(np.mean(np.asarray(output_samples)[:, 2])),
                "effort_change_rms": float(np.sqrt(np.mean(np.asarray(output_samples)[:, 3]))),
                "zero_command_peak_planar_drift_m": float(np.max(np.asarray(hold_samples)[:, 0])) if hold_samples else None,
                "zero_command_peak_heading_error_rad": float(np.max(np.asarray(hold_samples)[:, 1])) if hold_samples else None,
                "zero_command_within_5cm_fraction": float(np.mean(np.asarray(hold_samples)[:, 0] <= .05)) if hold_samples else None,
                "wheel_stagger_max_m": float(np.max(np.asarray(stance_samples)[:, 0])),
                "stance_valid_fraction": float(np.mean(
                    (np.asarray(stance_samples)[:, 0] <= raw.stage.wheel_stagger_limit_m)
                    & (np.asarray(stance_samples)[:, 1] <= raw.stage.wheel_offset_limit_m))),
                "samples": len(errors),
                "height_m": height, "vx_m_s": vx, "yaw_rad_s": yaw,
                "survival_rate": float(np.mean(survivors)),
                "episode_steps_mean": float(np.mean(lengths)),
                "rmse_vx_yaw_height": np.sqrt(np.square(errors).mean(axis=0)).tolist(),
                "joint_velocity_rms_rad_s": np.sqrt(np.square(leg_vel).mean(axis=0)).tolist(),
                "wheel_velocity_rms_rad_s": np.sqrt(np.square(wheel_vel).mean(axis=0)).tolist(),
                "settled_rmse_vx_yaw_height": (np.sqrt(np.square(settled_errors).mean(axis=0)).tolist()
                                                     if settled_errors else None),
                "joint_angle_span_rad": np.ptp(angles, axis=0).tolist(),
                "joint_torque_saturation_fraction": np.mean(saturations, axis=0).tolist(),
                "termination_counts": dict(failures),
                "episodes": episode_details,
                "zero_command_peak_x_drift_m": max(stop_drift) if stop_drift else None,
            })
            print(f"h={height:.3f} vx={vx:+.1f} yaw={yaw:+.1f}: "
                  f"survival={np.mean(survivors):.0%} length={np.mean(lengths):.0f}", flush=True)
    env.close()
    report = json.dumps({"checkpoint": str(checkpoint), "ablation": args.ablation,
                         "episodes_per_case": args.episodes_per_case, "seed_base": args.seed,
                         "ablation_start_seconds": args.ablation_start_seconds,
                         "domain_randomization": args.domain_randomization, "perturbations": args.perturbations,
                         "moving_start_probability": raw.stage.moving_start_probability,
                         "translation_speed_m_s": speed,
                         "episode_seconds": raw.stage.episode_steps * raw.params.control_dt,
                         "results": results}, indent=2)
    print(report)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report + "\n")


if __name__ == "__main__":
    main()
