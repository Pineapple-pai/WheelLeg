"""Scan the loaded leg-extension envelope.

This separates three limits that are easy to conflate:

* requested leg length (the command sent to the height loop),
* achieved leg length (including controller differential saturation), and
* standing stability during the extension transient.

Examples::

    conda run --no-capture-output -n sim python scripts/scan_leg_extension.py
    conda run --no-capture-output -n sim python scripts/scan_leg_extension.py \
        --rates 0.40 0.20 --targets 0.210 0.240 0.270 0.300 0.320 0.325
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uz05.env import UZ05Env  # noqa: E402


def run(target: float, rate: float, steps: int, settle: int) -> dict[str, float | int | str | bool]:
    env = UZ05Env(
        stage="stand", stand_level=2, assist_scale=0.0, seed=0,
        init_scale=0.0, lock_stand_leg_actions=False,
        stand_leg_action_limit=1.0, coord_mix=1.0,
    )
    env.params.domain_randomization.enabled = False
    env.stage = replace(env.stage, episode_steps=steps)
    env.reset(seed=0)
    env.coordinated.p.height_rate_limit = float(rate)
    env.coordinated.enable_leg_length(0.15, 0.35)
    # Use the same public command path as deployment.  Setting the controller
    # reference directly is incorrect: env.step() synchronizes it from
    # command[3] again and silently restores the episode's sampled target.
    env.set_leg_length_command(float(target))

    samples: list[tuple[float, float, float, float, float, float]] = []
    info: dict = {}
    n = 0
    for k in range(steps):
        _, _, terminated, truncated, info = env.step(np.zeros(6, dtype=np.float32))
        n = k + 1
        if k >= settle:
            leg = float(np.mean(env.sim.leg_lengths()))
            q = env.sim.joint_positions()
            samples.append((
                leg,
                float(env.sim.data.qpos[2]),
                float(info.get("station_error", 0.0)),
                float(info.get("pitch", 0.0)),
                float(q[0]),
                float(q[1]),
            ))
        if terminated or truncated:
            break

    arr = np.asarray(samples, dtype=np.float64)
    p = env.coordinated.p
    result: dict[str, float | int | str | bool] = {
        "target_m": float(target),
        "rate_action_s": float(rate),
        "steps": int(n),
        "survived": bool(n >= steps and info.get("termination_reason") == "time_limit"),
        "termination": str(info.get("termination_reason", "")),
        "height_diff_limit": float(p.height_diff_limit),
        "height_diff_final": float(env.coordinated.height_diff),
    }
    if arr.size:
        leg_mean = float(np.mean(arr[:, 0]))
        result.update({
            "leg_mean_m": leg_mean,
            "base_z_mean_m": float(np.mean(arr[:, 1])),
            "drift_peak_cm": float(np.max(np.abs(arr[:, 2])) * 100.0),
            "pitch_peak_deg": float(np.max(np.abs(arr[:, 3])) * 180.0 / np.pi),
            "q2_mean_rad": float(np.mean(arr[:, 4])),
            "q4_mean_rad": float(np.mean(arr[:, 5])),
            "leg_error_mm": float((leg_mean - target) * 1000.0),
        })
        # A run that merely stays upright while ignoring the requested length
        # is not a successful extension scan.
        result["survived"] = bool(result["survived"] and abs(leg_mean - target) <= 0.015)
        if n >= steps and info.get("termination_reason") == "time_limit" and not result["survived"]:
            result["termination"] = "tracking_error"
    else:
        result.update({
            "leg_mean_m": float("nan"), "base_z_mean_m": float("nan"),
            "drift_peak_cm": float("nan"), "pitch_peak_deg": float("nan"),
            "q2_mean_rad": float("nan"), "q4_mean_rad": float("nan"),
            "leg_error_mm": float("nan"),
        })
    env.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=float, nargs="*",
                        default=[0.210, 0.240, 0.270, 0.300, 0.320, 0.325, 0.330])
    parser.add_argument("--rates", type=float, nargs="*", default=[0.40, 0.20])
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--settle", type=int, default=2000)
    args = parser.parse_args()

    print("target rate   leg_m  err_mm  base_z  drift_cm pitch_deg  diff/lim   q2       q4   steps result")
    for rate in args.rates:
        for target in args.targets:
            r = run(target, rate, args.steps, min(args.settle, args.steps - 1))
            print(
                f"{r['target_m']:6.3f} {r['rate_action_s']:4.2f} "
                f"{r['leg_mean_m']:7.4f} {r['leg_error_mm']:7.1f} "
                f"{r['base_z_mean_m']:7.4f} {r['drift_peak_cm']:8.3f} "
                f"{r['pitch_peak_deg']:8.3f} "
                f"{r['height_diff_final']:5.3f}/{r['height_diff_limit']:5.3f} "
                f"{r['q2_mean_rad']:+7.3f} {r['q4_mean_rad']:+7.3f} "
                f"{r['steps']:5d} {'PASS' if r['survived'] else r['termination']}"
            )


if __name__ == "__main__":
    main()
