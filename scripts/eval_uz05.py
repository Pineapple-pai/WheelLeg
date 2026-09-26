"""Evaluate a direct-target PPO checkpoint on UZ-05."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from train_uz05 import AsymmetricActorCriticPolicy
from uz05.env import UZ05Env
from uz05.onnx_policy import OnnxPolicy
from uz05.policy_compat import ensure_forward_positive_wheel_actions
from uz05.spec import ACTOR_OBS_DIM, EnvParams


def _mean(values):
    return float(np.mean(values)) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="evaluate UZ-05 PPO")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--checkpoint")
    source.add_argument("--onnx")
    parser.add_argument("--stage", default="stand")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--stand-level", type=int, default=2)
    parser.add_argument("--init-scale", type=float, default=0.0)
    parser.add_argument("--vx-range", type=float, nargs=2, default=None)
    parser.add_argument("--command-zero-prob", type=float, default=None)
    parser.add_argument("--command-reverse-prob", type=float, default=None)
    parser.add_argument("--command-accel-limit", type=float, default=None)
    parser.add_argument("--episode-steps", type=int, default=None)
    parser.add_argument("--deployment-mode", action="store_true")
    parser.add_argument("--observation-delay-steps", type=int, nargs=2, default=None)
    parser.add_argument(
        "--actuator-delay-steps", type=int, nargs=2, default=None,
        help="actuator delay range in 500 Hz motor ticks",
    )
    parser.add_argument("--zero-action", action="store_true")
    parser.add_argument("--accept-tilt-deg", type=float, default=17.0)
    parser.add_argument("--accept-height-ratio", type=float, default=0.80)
    parser.add_argument("--accept-peak-drift-cm", type=float, default=5.0)
    parser.add_argument("--accept-vx-error", type=float, default=0.07)
    parser.add_argument("--accept-direction-accuracy", type=float, default=0.80)
    args = parser.parse_args()

    if not args.zero_action and not (args.checkpoint or args.onnx):
        parser.error("one of --checkpoint or --onnx is required unless --zero-action is used")

    model = None
    onnx_policy = None
    if not args.zero_action and args.checkpoint:
        model = PPO.load(
            str(Path(args.checkpoint)),
            custom_objects={"policy_class": AsymmetricActorCriticPolicy},
            device="cpu",
        )
        if ensure_forward_positive_wheel_actions(model):
            print("checkpoint_wheel_action_migrated: legacy joint-positive -> body-forward-positive")
    elif not args.zero_action:
        onnx_policy = OnnxPolicy(args.onnx)
    outcomes = []
    pitch_rms, pitch_rate_rms, action_abs = [], [], []
    current_abs, wheel_target_abs, leg_target_abs = [], [], []
    vx_errors, vx_rel_errors, zero_vx = [], [], []
    vx_commands, vx_actuals, wheel_contact_left, wheel_contact_right = [], [], [], []
    airborne_steps, body_contact_steps = [], []
    reasons: Counter[str] = Counter()
    class_reasons = defaultdict(Counter)
    class_episode_stats = defaultdict(list)
    class_vx_error = defaultdict(list)
    class_vx_actual = defaultdict(list)
    class_vx_target = defaultdict(list)
    class_action_rows = defaultdict(list)
    class_reward_sums = defaultdict(lambda: defaultdict(float))
    class_reward_steps = Counter()

    for episode in range(args.episodes):
        env = UZ05Env(
            stage=args.stage,
            stand_level=args.stand_level,
            seed=args.seed + episode,
            init_scale=args.init_scale,
            vx_range_override=None if args.vx_range is None else tuple(args.vx_range),
            zero_command_prob_override=args.command_zero_prob,
            reverse_prob_override=args.command_reverse_prob,
            command_accel_limit_override=args.command_accel_limit,
            deployment_mode=args.deployment_mode,
            observation_delay_steps=None if args.observation_delay_steps is None else tuple(args.observation_delay_steps),
            actuator_delay_steps=None if args.actuator_delay_steps is None else tuple(args.actuator_delay_steps),
            episode_steps_override=args.episode_steps,
        )
        obs, _ = env.reset(seed=args.seed + episode)
        pitches, pitch_rates, common = [], [], []
        episode_actual, episode_vx_abs_error, episode_direction = [], [], []
        done = False
        info = {}
        while not done:
            if args.zero_action:
                action = np.zeros(6, dtype=np.float32)
            elif onnx_policy is not None:
                action = onnx_policy.predict(obs[:ACTOR_OBS_DIM])
            else:
                action = model.predict(obs, deterministic=True)[0]
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            action_abs.append(float(np.abs(action).mean()))
            obs, _, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            pitches.append(float(info.get("pitch", 0.0)))
            pitch_rates.append(float(info.get("pitch_rate", 0.0)))
            common.append(float(action[0]))
            current_abs.append(float(info.get("wheel_current_abs", 0.0)))
            wheel_target_abs.append(float(info.get("wheel_target_abs", 0.0)))
            leg_target_abs.append(float(info.get("leg_target_abs", 0.0)))
            command = float(info.get("command_vx", 0.0))
            target_command = float(info.get("command_target_vx", command))
            body_vx = float(info.get("body_vx_after", 0.0))
            vx_commands.append(command)
            vx_actuals.append(body_vx)
            wheel_contact_left.append(float(info.get("wheel_contact_left", 0.0)))
            wheel_contact_right.append(float(info.get("wheel_contact_right", 0.0)))
            airborne_steps.append(float(info.get("airborne", 0.0)))
            body_contact_steps.append(float(info.get("body_contact", 0.0)))
            group = (
                "zero" if abs(target_command) <= 0.01
                else ("forward" if target_command > 0.0 else "reverse")
            )
            class_action_rows[group].append(action.copy())
            class_reward_steps[group] += 1
            for name, value in (info.get("reward_terms") or {}).items():
                class_reward_sums[group][name] += float(value)
            episode_actual.append(body_vx)
            if abs(command) > 0.01:
                error = body_vx - command
                vx_errors.append(error)
                vx_rel_errors.append(error / max(abs(command), 0.06))
                class_vx_error[group].append(abs(error))
                class_vx_actual[group].append(body_vx)
                class_vx_target[group].append(command)
                episode_vx_abs_error.append(abs(error))
                episode_direction.append(float(body_vx * command > 0.0 and abs(body_vx) >= 0.03))
            else:
                zero_vx.append(abs(body_vx))
        reason = str(info.get("termination_reason", "unknown"))
        reasons[reason] += 1
        final_target = float(info.get("command_target_vx", 0.0))
        final_group = (
            "zero" if abs(final_target) <= 0.01
            else ("forward" if final_target > 0.0 else "reverse")
        )
        class_reasons[final_group][reason] += 1
        use_motion = args.stage == "low_speed"
        zero_command = abs(final_target) <= 0.01
        if use_motion and not zero_command:
            drift_tail = float(info.get("motion_tail_mean", 0.0))
            drift_peak = float(info.get("motion_max_abs", 0.0))
        elif use_motion:
            drift_tail = float(info.get("station_tail_mean", 0.0))
            drift_peak = float(info.get("station_max_abs", 0.0))
        else:
            drift_tail = float(info.get("station_tail_mean", 0.0))
            drift_peak = float(info.get("station_max_abs", 0.0))
        outcome = {
            "full": bool(truncated and not terminated),
            "tail": drift_tail,
            "peak": drift_peak,
            "tilt": max(abs(float(info.get("roll", 99.0))), abs(float(info.get("pitch", 99.0)))),
            "height": float(info.get("base_height", 0.0)),
            "steps": int(info.get("episode_steps", 0)),
            "reason": reason,
            "target_vx": final_target,
            "actual_vx_mean": _mean(episode_actual),
            "vx_mae": _mean(episode_vx_abs_error),
            "direction_accuracy": _mean(episode_direction),
        }
        outcomes.append(outcome)
        class_episode_stats[final_group].append(outcome)
        pitch_rms.append(float(np.sqrt(np.mean(np.square(pitches)))))
        pitch_rate_rms.append(float(np.sqrt(np.mean(np.square(pitch_rates)))))
        env.close()

    full = np.asarray([item["full"] for item in outcomes], dtype=np.float64)
    tail = np.asarray([item["tail"] for item in outcomes], dtype=np.float64)
    peak = np.asarray([item["peak"] for item in outcomes], dtype=np.float64)
    tilt = np.asarray([item["tilt"] for item in outcomes], dtype=np.float64)
    height = np.asarray([item["height"] for item in outcomes], dtype=np.float64)
    zero_mask = np.asarray([abs(float(item["target_vx"])) <= 0.01 for item in outcomes])
    peak_ok = np.asarray([
        item["peak"] <= args.accept_peak_drift_cm / 100.0
        if abs(float(item["target_vx"])) <= 0.01 else True
        for item in outcomes
    ], dtype=np.float64)
    zero_peak = peak[zero_mask]
    zero_peak_ok_rate = (
        float(np.mean(zero_peak <= args.accept_peak_drift_cm / 100.0))
        if zero_peak.size else float("nan")
    )
    moving_peak = peak[~zero_mask]
    tracking_ok = np.asarray([
        True if abs(float(item["target_vx"])) <= 0.01 else (
            np.isfinite(item["vx_mae"])
            and item["vx_mae"] <= args.accept_vx_error
            and np.isfinite(item["direction_accuracy"])
            and item["direction_accuracy"] >= args.accept_direction_accuracy
        )
        for item in outcomes
    ], dtype=np.float64)
    tilt_ok = tilt <= np.deg2rad(args.accept_tilt_deg)
    height_ok = height >= args.accept_height_ratio * EnvParams().robot.nominal_stand_height
    accepted = full * peak_ok * tilt_ok * height_ok * tracking_ok
    print(f"policy_source: {args.onnx or args.checkpoint or 'zero_action'}")
    print(f"contract: direct PPO -> wheel speed target + leg position target")
    print(f"survive_rate: {full.mean():.3f} ({int(full.sum())}/{args.episodes})")
    print(f"tail_drift_cm_mean: {100 * tail.mean():.2f}")
    print(f"moving_path_error_cm_p95: {100 * np.quantile(moving_peak, 0.95):.2f}" if moving_peak.size else "moving_path_error_cm_p95: nan")
    print(f"zero_command_drift_cm_p95: {100 * np.quantile(zero_peak, 0.95):.2f}" if zero_peak.size else "zero_command_drift_cm_p95: nan")
    print(
        "zero_command_drift_ok_rate: "
        + (f"{zero_peak_ok_rate:.3f} ({int(np.sum(zero_peak <= args.accept_peak_drift_cm / 100.0))}/{zero_peak.size})"
           if zero_peak.size else "nan (0/0)")
    )
    print(f"command_tracking_ok_rate: {tracking_ok.mean():.3f} ({int(tracking_ok.sum())}/{args.episodes})")
    print(f"upright_rate: {tilt_ok.mean():.3f} ({int(tilt_ok.sum())}/{args.episodes})")
    print(f"height_ok_rate: {height_ok.mean():.3f} ({int(height_ok.sum())}/{args.episodes})")
    print(f"accept_rate: {accepted.mean():.3f} ({int(accepted.sum())}/{args.episodes})")
    print(f"pitch_rms_deg_mean: {np.degrees(np.mean(pitch_rms)):.3f}")
    print(f"pitch_rate_rms_mean: {_mean(pitch_rate_rms):.5f} rad/s")
    print(f"policy_action_abs_mean: {_mean(action_abs):.4f}")
    print(f"wheel_target_abs_mean_rad_s: {_mean(wheel_target_abs):.3f}")
    print(f"wheel_current_abs_mean_A: {_mean(current_abs):.3f}")
    print(f"leg_target_abs_mean_rad: {_mean(leg_target_abs):.3f}")
    print(f"vx_error_mae_m_s: {_mean([abs(value) for value in vx_errors]):.4f}")
    print(f"vx_error_rel_mae: {_mean([abs(value) for value in vx_rel_errors]):.3f}")
    print(f"vx_command_mean_m_s: {_mean(vx_commands):.4f}")
    print(f"vx_actual_mean_m_s: {_mean(vx_actuals):.4f}")
    print(f"vx_signed_error_mean_m_s: {_mean([a - c for a, c in zip(vx_actuals, vx_commands)]):.4f}")
    print(f"vx_zero_actual_abs_mean_m_s: {_mean(zero_vx):.4f}")
    print(f"wheel_contact_left_rate: {_mean(wheel_contact_left):.4f}")
    print(f"wheel_contact_right_rate: {_mean(wheel_contact_right):.4f}")
    print(f"wheel_contact_both_rate: {_mean([l * r for l, r in zip(wheel_contact_left, wheel_contact_right)]):.4f}")
    print(f"airborne_step_rate: {_mean(airborne_steps):.4f}")
    print(f"body_contact_step_rate: {_mean(body_contact_steps):.4f}")
    print("terminations: " + ", ".join(f"{key}={value}" for key, value in sorted(reasons.items())))
    action_names = ("wheel_left", "wheel_right", "leg_1", "leg_2", "leg_3", "leg_4")
    for group in ("forward", "reverse", "zero"):
        episodes = class_episode_stats[group]
        if not episodes:
            continue
        group_full = [float(item["full"]) for item in episodes]
        group_peak = [float(item["peak"]) for item in episodes]
        print(
            f"[{group}] episodes={len(episodes)} survive={np.mean(group_full):.3f} "
            f"steps_mean={_mean([item['steps'] for item in episodes]):.1f} "
            f"target_vx_mean={_mean([item['target_vx'] for item in episodes]):.4f} "
            f"actual_vx_mean={_mean([item['actual_vx_mean'] for item in episodes]):.4f} "
            f"peak_position_error_p90_cm={100*np.quantile(group_peak, 0.90):.2f}"
        )
        print(
            f"[{group}] terminations: "
            + ", ".join(
                f"{key}={value}" for key, value in sorted(class_reasons[group].items())
            )
        )
        if group in class_vx_error:
            direction_accuracy = _mean([
                float(actual * target > 0.0 and abs(actual) >= 0.03)
                for actual, target in zip(class_vx_actual[group], class_vx_target[group])
            ])
            print(
                f"[{group}] vx_mae_m_s={_mean(class_vx_error[group]):.4f} "
                f"direction_accuracy={direction_accuracy:.3f}"
            )
        actions = np.asarray(class_action_rows[group], dtype=np.float64)
        if actions.size:
            print(
                f"[{group}] action_mean: "
                + ", ".join(f"{name}={value:.3f}" for name, value in zip(action_names, actions.mean(axis=0)))
            )
            print(
                f"[{group}] action_abs_mean: "
                + ", ".join(f"{name}={value:.3f}" for name, value in zip(action_names, np.abs(actions).mean(axis=0)))
            )
        steps = class_reward_steps[group]
        if steps:
            means = {
                name: value / steps for name, value in class_reward_sums[group].items()
            }
            ranked = sorted(means.items(), key=lambda item: abs(item[1]), reverse=True)[:16]
            focused = (
                "station", "station_vel", "station_progress",
                "motion_position_error", "motion_position_progress",
                "track_vx", "track_vx_error", "track_vx_forward_error",
                "track_vx_reverse_error", "alive", "termination",
            )
            ranked_names = {name for name, _ in ranked}
            ranked.extend((name, means.get(name, 0.0)) for name in focused if name not in ranked_names)
            print(
                f"[{group}] reward_terms_mean_per_step: "
                + ", ".join(f"{name}={value:.3f}" for name, value in ranked)
            )


if __name__ == "__main__":
    main()
