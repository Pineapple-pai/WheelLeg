"""教师增益扫描（走真实 ActuatorBank/BalanceController 代码路径）。

assist=1.0、零动作、放宽终止阈值，直接测"教师自己能站多好"：
  survive（是否活满整集）/ tail_cm（末段均值漂移）/ peak_cm / pitch_std
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uz05.env import UZ05Env  # noqa: E402


def rollout(env, g, episodes, seed0, steps):
    kp, kd, kv, ki, kx, th = g
    b = env.params.balance
    b.speed_sign = 1.0
    b.station_sign = 1.0
    b.pitch_kp, b.pitch_kd = kp, kd
    b.body_speed_kp, b.body_speed_ki = kv, ki
    b.station_kp = kx
    b.theta_max = th
    surv, tails, peaks, pstd = [], [], [], []
    for ep in range(episodes):
        env.reset(seed=seed0 + ep)
        a = np.zeros(6)
        info: dict = {}
        pitch = []
        n = 0
        done = False
        while not done and n < steps:
            _, _, term, trunc, info = env.step(a)
            pitch.append(info["pitch"])
            n += 1
            done = term or trunc
        surv.append(bool(trunc and not term))
        tails.append(info["station_tail_mean"])
        peaks.append(info["station_max_abs"])
        pstd.append(np.std(pitch))
    return float(np.mean(surv)), float(np.mean(tails)), float(np.mean(peaks)), float(np.mean(pstd))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    env = UZ05Env(stage="stand", stand_level=2, assist_scale=1.0)
    env.params.noise.enabled = False
    # 放宽终止：先看本体能站多好；阈值另算
    env.params.station_hard_limit = 10.0
    env.params.station_hold_steps = 10**9
    env.tilt_limit = 1.0
    env.params.terminate_on_drift = 10.0
    env.params.terminate_lateral_vel = 10.0

    grid = list(itertools.product(
        [20.0, 30.0, 45.0],              # pitch_kp
        [0.03, 0.06, 0.12],              # pitch_kd (kd_eff = kp*kd)
        [4.0, 10.0, 20.0, 36.0],         # body_speed_kp (Nm/(m/s)) -> kv = /kp
        [0.0],                           # body_speed_ki
        [0.5, 1.5, 3.0],                 # station_kp (rad/m)
        [0.05, 0.10],                    # theta_max
    ))
    print(f"grid {len(grid)}")
    rows = []
    for g in grid:
        s, t, p, ps = rollout(env, g, args.episodes, 4000, args.steps)
        rows.append((s, t, p, ps, g))
    rows.sort(key=lambda r: (-r[0], r[2], r[1]))
    print(f"{'surv':>5} {'tail_cm':>8} {'peak_cm':>8} {'p_std':>7}  "
          f"{'kp':>5} {'kd':>5} {'kv':>5} {'ki':>4} {'kx':>5} {'thmax':>6}")
    for s, t, p, ps, g in rows[: args.top]:
        print(f"{s:5.2f} {100*t:8.2f} {100*p:8.2f} {ps:7.4f}  "
              f"{g[0]:5.1f} {g[1]:5.2f} {g[2]:5.1f} {g[3]:4.1f} {g[4]:5.2f} {g[5]:6.3f}")
    env.close()


if __name__ == "__main__":
    main()
