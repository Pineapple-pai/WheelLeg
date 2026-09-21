"""在"平衡位形重置"下扫描加性线性平衡律，找 survive=1 且漂移最小的解。

tau = -Kp*(pitch + Kd*pitch_rate) - Kv*(vx - vx_cmd) - Kx*sat(x, deadband)
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uz05.env import UZ05Env  # noqa: E402


def rollout(env, g, episodes, seed0, steps, reset_hip, reset_h):
    kp, kd, kv, kx, db = g
    scale = env.params.wheel.policy_current_scale_a
    surv, tails, peaks, pstd = [], [], [], []
    for ep in range(episodes):
        env.reset(seed=seed0 + ep)
        if reset_hip is not None:
            env.sim.data.qpos[env.sim.hip_qpos_adr] = np.array(
                [-reset_hip, reset_hip, -reset_hip, reset_hip])
        if reset_h is not None:
            env.sim.data.qpos[2] = reset_h
        env.sim.data.qvel[:] = 0.0
        env.sim.forward()
        env.nominal_xy = env.sim.data.qpos[:2].copy()
        pitchs = []
        info: dict = {}
        done = False
        n = 0
        while not done and n < steps:
            p = info.get("pitch", 0.0)
            pr = info.get("pitch_rate", 0.0)
            vx = info.get("body_vx", 0.0)
            x = info.get("station_error", 0.0)
            tau = -kp * (p + kd * pr) - kv * (vx - env.command[0])
            if abs(x) > db:
                tau -= kx * np.copysign(abs(x) - db, x)
            cur = float(np.clip(tau / 0.246, -8.0, 8.0))
            a = np.zeros(6)
            a[0] = np.clip(cur / scale, -1.0, 1.0)
            _, _, term, trunc, info = env.step(a)
            pitchs.append(info["pitch"])
            n += 1
            done = term or trunc
        surv.append(bool(trunc and not term))
        tails.append(info["station_tail_mean"])
        peaks.append(info["station_max_abs"])
        pstd.append(np.std(pitchs))
    return (float(np.mean(surv)), float(np.mean(tails)), float(np.mean(peaks)),
            float(np.mean(pstd)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--reset-hip", type=float, default=0.115)
    ap.add_argument("--reset-height", type=float, default=0.260)
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    env = UZ05Env(stage="stand", stand_level=2, assist_scale=0.0)
    env.params.noise.enabled = False
    env.params.wheel.policy_current_scale_a = 8.0
    env.params.station_hard_limit = 10.0
    env.tilt_limit = 1.0
    env.params.terminate_on_drift = 10.0
    env.params.terminate_lateral_vel = 10.0
    env.params.robot.leg_length_min, env.params.robot.leg_length_max = 0.10, 0.40

    grid = list(itertools.product(
        [10.0, 14.0, 20.0, 28.7, 40.0],      # kp
        [0.15, 0.25, 0.36, 0.55],            # kd (s)
        [8.0, 14.0, 20.0, 26.8],             # kv
        [10.0, 30.0, 60.0, 100.0],           # kx
        [0.0, 0.02, 0.05],                   # deadband
    ))
    print(f"grid {len(grid)}")
    rows = []
    for g in grid:
        s, t, p, ps = rollout(env, g, args.episodes, 4000, args.steps,
                              args.reset_hip, args.reset_height)
        rows.append((s, t, p, ps, g))
    rows.sort(key=lambda r: (-r[0], r[2], r[1]))
    print(f"{'surv':>5} {'tail_cm':>8} {'peak_cm':>8} {'p_std':>7}  "
          f"{'kp':>6} {'kd':>5} {'kv':>6} {'kx':>6} {'db':>5}")
    for s, t, p, ps, g in rows[: args.top]:
        print(f"{s:5.2f} {100*t:8.2f} {100*p:8.2f} {ps:7.4f}  "
              f"{g[0]:6.1f} {g[1]:5.2f} {g[2]:6.1f} {g[3]:6.1f} {g[4]:5.2f}")
    env.close()


if __name__ == "__main__":
    main()
