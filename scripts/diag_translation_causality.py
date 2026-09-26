"""Paired, read-only probes for wheel response and translation reward."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from train_uz05 import AsymmetricActorCriticPolicy
from uz05.actuators import WHEEL_SPEED_SCALE
from uz05.env import UZ05Env
from uz05.policy_compat import ensure_forward_positive_wheel_actions
from uz05.spec import EnvParams, WHEEL_ANGULAR_TO_BODY_X


def load_policy(path: str):
    model = PPO.load(
        str(Path(path)),
        custom_objects={"policy_class": AsymmetricActorCriticPolicy},
        device="cpu",
    )
    ensure_forward_positive_wheel_actions(model)
    return model


def run_trial(model, *, seed: int, command: float, wheel_action: float | None,
              steps: int, fixed_legs: bool = False, warmup_steps: int = 0,
              wheel_delta: float = 0.0, ramp_steps: int = 0,
              command_ramp: bool = False, wheel_delta_ramp_steps: int = 0,
              symmetrize_wheels: bool = False):
    env = UZ05Env(
        stage="low_speed", seed=seed, stand_level=2, init_scale=0.2,
        deployment_mode=True, vx_range_override=(abs(command), abs(command)),
        zero_command_prob_override=1.0,
    )
    try:
        obs, _ = env.reset(seed=seed)
        # Paired branches share an identical reset.  Holding the command at
        # its target avoids interpreting the 0.75 m/s^2 ramp as actuator lag.
        env.command_target[0] = command
        if not command_ramp:
            env.command[0] = command
        env._translation_episode_active = abs(command) > 0.01
        obs = env._obs()
        fixed_leg_action = model.predict(obs, deterministic=True)[0][2:].copy()
        for _ in range(warmup_steps):
            action = np.asarray(model.predict(obs, deterministic=True)[0], dtype=np.float32).copy()
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                return []
        rows = []
        for step_index in range(steps):
            action = np.asarray(model.predict(obs, deterministic=True)[0], dtype=np.float32).copy()
            if symmetrize_wheels:
                action[:2] = float(np.mean(action[:2]))
            if fixed_legs:
                action[2:] = fixed_leg_action
            if wheel_action is not None:
                fraction = min(1.0, (step_index + 1) / ramp_steps) if ramp_steps else 1.0
                action[:2] = wheel_action * fraction
            delta_fraction = (
                min(1.0, (step_index + 1) / wheel_delta_ramp_steps)
                if wheel_delta_ramp_steps else 1.0
            )
            action[:2] += wheel_delta * delta_fraction
            np.clip(action, -1.0, 1.0, out=action)
            obs, reward, terminated, truncated, info = env.step(action)
            rows.append((reward, info, action.copy()))
            if terminated or truncated:
                break
        return rows
    finally:
        env.close()


def report(label: str, results: list[list[tuple]]):
    keys = (
        "body_vx_after", "wheel_body_vx", "wheel_target_abs",
        "wheel_current_abs", "wheel_slip_m_s", "pitch",
        "yaw_rate",
        "wheel_contact_left", "wheel_contact_right",
    )
    print(f"{label}: trials={len(results)}", flush=True)
    for key in keys:
        values = [float(info[key]) for rows in results for _, info, _ in rows]
        print(f"  {key}: {np.mean(values):+.5f}", flush=True)
    for key in ("track_vx", "track_vx_progress", "pitch_speed_coupling",
                "upright", "wrong_direction", "wheel_slip"):
        values = [float(info["reward_terms"].get(key, 0.0))
                  for rows in results for _, info, _ in rows]
        print(f"  reward/{key}: {np.mean(values):+.5f}", flush=True)
    action_delta = [float(action[0] - action[1]) for rows in results for _, _, action in rows]
    action_common = [float((action[0] + action[1]) * 0.5) for rows in results for _, _, action in rows]
    print(f"  policy_wheel_common_action: {np.mean(action_common):+.5f}", flush=True)
    print(f"  policy_wheel_differential_action: {np.mean(action_delta):+.5f}", flush=True)
    rewards = [float(reward) for rows in results for reward, _, _ in rows]
    print(f"  reward/total: {np.mean(rewards):+.5f}", flush=True)
    print(f"  mean_steps: {np.mean([len(rows) for rows in results]):.1f}", flush=True)
    reasons = defaultdict(int)
    for rows in results:
        reasons[str(rows[-1][1]["termination_reason"])] += 1
    print(f"  final_reasons: {dict(reasons)}", flush=True)
    early = [row for rows in results for row in rows[:20]]
    if len(early) == 20 * len(results):
        print(f"  first20_reward: {np.mean([r for r, _, _ in early]):+.5f}", flush=True)
        print(f"  first20_body_vx: {np.mean([i['body_vx_after'] for _, i, _ in early]):+.5f}", flush=True)
        print(f"  first20_pitch_deg: {np.degrees(np.mean([i['pitch'] for _, i, _ in early])):+.3f}", flush=True)
    first = [rows[0] for rows in results]
    print(f"  first1_reward: {np.mean([r for r, _, _ in first]):+.5f}", flush=True)
    print(f"  first1_track_vx: {np.mean([i['reward_terms']['track_vx'] for _, i, _ in first]):+.5f}", flush=True)
    print(f"  first1_upright: {np.mean([i['reward_terms']['upright'] for _, i, _ in first]):+.5f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stand-checkpoint", required=True)
    parser.add_argument("--moving-checkpoint", required=True)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--ramp-only", action="store_true")
    parser.add_argument("--symmetry-only", action="store_true")
    args = parser.parse_args()

    stand = load_policy(args.stand_checkpoint)
    moving = load_policy(args.moving_checkpoint)
    seeds = range(101, 101 + args.seeds)
    if args.symmetry_only:
        print("TEST 4: paired policy rollouts with/without equal wheel targets", flush=True)
        for symmetrize in (False, True):
            results = [run_trial(
                moving, seed=seed, command=0.10, wheel_action=None,
                steps=args.steps, command_ramp=True,
                symmetrize_wheels=symmetrize,
            ) for seed in seeds]
            report("equal_wheel_targets" if symmetrize else "policy_wheel_targets", results)
        return
    if args.ramp_only:
        print("TEST 3: command ramp plus wheel bias atop the active policy", flush=True)
        for command in (0.10, -0.10):
            feedforward = command / (
                WHEEL_SPEED_SCALE * abs(WHEEL_ANGULAR_TO_BODY_X)
                * EnvParams().robot.wheel_radius
            )
            for label, wheel_delta, ramp in (("v8_policy", 0.0, 0),
                                             ("plus_half_speed_0.25s", feedforward * 0.5, 31),
                                             ("plus_half_speed_0.50s", feedforward * 0.5, 63),
                                             ("plus_full_speed_0.50s", feedforward, 63),
                                             ("plus_full_speed_1.00s", feedforward, 125)):
                results = [run_trial(moving, seed=seed, command=command,
                                     wheel_action=None, steps=args.steps,
                                     command_ramp=True, wheel_delta=wheel_delta,
                                     wheel_delta_ramp_steps=ramp)
                           for seed in seeds]
                report(f"command={command:+.2f} {label}", results)
        return
    print("TEST 1A: standing leg policy remains active, zero velocity command", flush=True)
    for wheel in (0.0, 0.12, -0.12, 0.24, -0.24):
        results = [run_trial(moving, seed=seed, command=0.0,
                             wheel_action=wheel, steps=args.steps)
                   for seed in seeds]
        report(f"wheel_action={wheel:+.2f}", results)

    print("TEST 1B: paired wheel perturbation after stable policy warmup", flush=True)
    for delta in (0.0, 0.08, -0.08, 0.16, -0.16):
        results = [run_trial(moving, seed=seed, command=0.0,
                             wheel_action=None, steps=20, warmup_steps=100,
                             wheel_delta=delta)
                   for seed in seeds]
        if all(results):
            report(f"wheel_delta={delta:+.2f}", results)
        else:
            print(f"wheel_delta={delta:+.2f}: warmup failed", flush=True)

    print("TEST 1C: fixed standing-leg target (short open-loop control)", flush=True)
    for wheel in (0.0, 0.12, -0.12, 0.24, -0.24):
        results = [run_trial(stand, seed=seed, command=0.0,
                             wheel_action=wheel, steps=args.steps, fixed_legs=True)
                   for seed in seeds]
        report(f"wheel_action={wheel:+.2f}", results)

    print("TEST 2: paired moving-command rollouts, identical reset per branch", flush=True)
    for command in (0.10, -0.10):
        feedforward = command / (
            WHEEL_SPEED_SCALE * abs(WHEEL_ANGULAR_TO_BODY_X)
            * EnvParams().robot.wheel_radius
        )
        for label, wheel in (("zero_wheel", 0.0), ("v8_policy", None),
                             ("correct_wheel", feedforward), ("opposite_wheel", -feedforward)):
            results = [run_trial(moving, seed=seed, command=command,
                                 wheel_action=wheel, steps=args.steps)
                       for seed in seeds]
            report(f"command={command:+.2f} {label} wheel={wheel}", results)


if __name__ == "__main__":
    main()
