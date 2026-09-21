"""可行性扫描：assist=0 下，纯线性轮电流反馈能达到多好的站立质量？

结论决定"±5cm 稳定站立"是否物理可达、以及课程/终止阈值应该定在哪。
扫描时把终止阈值放宽（只保留真正摔倒的判据），先测本体稳定性；
再单独报告达到该稳定性所需的终止阈值。
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uz05.env import UZ05Env  # noqa: E402


class Law:
    """tau = -Kp*(pitch + Kd*pitch_rate) - Kv*(vx - vx_cmd) - Kx*(x-dead) - Kd_x*vx"""

    def __init__(self, kp, kd, kv, kx, deadband, kdx=0.0):
        self.kp, self.kd, self.kv, self.kx, self.deadband, self.kdx = (
            kp, kd, kv, kx, deadband, kdx)

    def current(self, pitch, pitch_rate, vx, x, cmd_vx):
        torque = -self.kp * (pitch + self.kd * pitch_rate)
        torque -= self.kv * (vx - cmd_vx)
        if abs(x) > self.deadband:
            torque -= self.kx * np.copysign(abs(x) - self.deadband, x)
        torque -= self.kdx * vx
        return torque / 0.246


def rollout(env, law, episodes, seed0, max_steps=1000):
    tails, peaks, surv, steps = [], [], [], []
    drift_peak = []
    for ep in range(episodes):
        env.reset(seed=seed0 + ep)
        done = False
        info = {}
        n = 0
        while not done and n < max_steps:
            pitch = info.get("pitch", 0.0)
            pitch_rate = info.get("pitch_rate", 0.0)
            vx = info.get("body_vx", 0.0)
            x = info.get("station_error", 0.0)
            cur = law.current(pitch, pitch_rate, vx, x, env.command[0])
            scale = env.params.wheel.policy_current_scale_a
            a = np.zeros(6)
            a[0] = np.clip(cur / scale, -1.0, 1.0)
            _, _, term, trunc, info = env.step(a)
            n += 1
            done = term or trunc
        tails.append(info["station_tail_mean"] if "station_tail_mean" in info else 0.0)
        peaks.append(info["station_max_abs"])
        drift_peak.append(abs(info["station_error"]))
        surv.append(bool(trunc and not term))
        steps.append(info["episode_steps"])
    return (float(np.mean(surv)), float(np.mean(tails)), float(np.mean(peaks)),
            float(np.mean(steps)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--loose", action="store_true", default=True,
                    help="放宽终止阈值，只保留真摔倒判据")
    args = ap.parse_args()

    env = UZ05Env(stage="stand", stand_level=2, assist_scale=0.0)
    env.params.noise.enabled = False
    env.params.wheel.policy_current_scale_a = 8.0
    # 放宽终止阈值：测"本体能站多好"
    env.params.station_hard_limit = 10.0
    env.target_lateral = 10.0
    env.params.terminate_on_drift = 10.0
    env.params.terminate_lateral_vel = 10.0
    env.tilt_limit = 1.0
    env.params.robot.leg_length_min, env.params.robot.leg_length_max = 0.10, 0.40

    grid = list(itertools.product(
        [14.0, 20.0, 28.7, 40.0],           # kp
        [0.20, 0.36, 0.60],                 # kd
        [8.0, 16.0, 26.8, 40.0],            # kv
        [0.0, 40.0, 80.0, 120.0, 200.0],    # kx
        [0.02, 0.05],                       # deadband
    ))
    print(f"grid size: {len(grid)}  episodes/each: {args.episodes}")
    best = []
    for kp, kd, kv, kx, db in grid:
        law = Law(kp, kd, kv, kx, db)
        surv, tail, peak, steps = rollout(env, law, args.episodes, 5000, args.steps)
        best.append((surv, tail, peak, steps, kp, kd, kv, kx, db))
    best.sort(key=lambda t: (-t[0], t[1]))
    print(f"{'surv':>5} {'tail_cm':>8} {'peak_cm':>8} {'steps':>7}  "
          f"{'kp':>6} {'kd':>5} {'kv':>6} {'kx':>6} {'db':>5}")
    for surv, tail, peak, steps, kp, kd, kv, kx, db in best[:25]:
        print(f"{surv:5.2f} {100*tail:8.2f} {100*peak:8.2f} {steps:7.0f}  "
              f"{kp:6.1f} {kd:5.2f} {kv:6.1f} {kx:6.1f} {db:5.2f}")
    env.close()


if __name__ == "__main__":
    main()
