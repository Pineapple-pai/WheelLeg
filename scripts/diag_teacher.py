"""教师控制器扫描：级联式（外环俯仰指令 + 内环俯仰 PD + 力矩预算），
在 assist=0 下寻找能稳定定点站立的线性反馈解。

theta_ref = clip(Kv*vx + Kx*x_err, ±th_max)      # vx/x_err 为正表示需要向 +X 修正
tau       = Kp*(theta_ref - pitch) - Kd*pitch_rate
current   = clip(tau / torque_per_amp, ±I_MAX)
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uz05.env import UZ05Env  # noqa: E402

TAU_PER_AMP = 0.246


def rollout(env, gains, episodes, seed0, steps, reset_hip, reset_h, loose):
    kp, kd, kv, kx, thmax, imax = gains
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
        env.params.station_hard_limit = 10.0 if loose else 0.05
        env.tilt_limit = 0.60 if loose else 0.30
        pitch = []
        info: dict = {}
        done = False
        n = 0
        while not done and n < steps:
            p = info.get("pitch", 0.0)
            pr = info.get("pitch_rate", 0.0)
            vx = info.get("body_vx", 0.0)
            x = info.get("station_error", 0.0)
            th_ref = float(np.clip(kv * (vx - env.command[0]) + kx * x, -thmax, thmax))
            tau = kp * (th_ref - p) - kd * pr
            cur = float(np.clip(tau / TAU_PER_AMP, -imax, imax))
            a = np.zeros(6)
            a[0] = np.clip(cur / scale, -1.0, 1.0)
            _, _, term, trunc, info = env.step(a)
            pitch.append(info["pitch"])
            n += 1
            done = term or trunc
        surv.append(bool(trunc and not term))
        tails.append(info["station_tail_mean"])
        peaks.append(info["station_max_abs"])
        pstd.append(np.std(pitch))
    return (float(np.mean(surv)), float(np.mean(tails)), float(np.mean(peaks)),
            float(np.mean(pstd)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--reset-hip", type=float, default=0.115)
    ap.add_argument("--reset-height", type=float, default=0.260)
    ap.add_argument("--loose", action="store_true")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--stage", default="cascade")
    args = ap.parse_args()

    env = UZ05Env(stage="stand", stand_level=2, assist_scale=0.0)
    env.params.noise.enabled = False
    env.params.wheel.policy_current_scale_a = 8.0
    env.params.terminate_on_drift = 10.0
    env.params.terminate_lateral_vel = 10.0

    grid = list(itertools.product(
        [20.0, 30.0, 45.0],              # kp
        [0.05, 0.10, 0.18],              # kd
        [0.3, 0.8],                      # kv
        [0.5, 1.5, 3.0],                 # kx
        [0.03, 0.06, 0.10],              # thmax
        [8.0],                           # imax (A)
    ))
    print(f"grid {len(grid)}  loose={args.loose}")
    rows = []
    for gains in grid:
        surv, tail, peak, pstd = rollout(env, gains, args.episodes, 3000, args.steps,
                                         args.reset_hip, args.reset_height, args.loose)
        rows.append((surv, tail, peak, pstd, gains))
    rows.sort(key=lambda t: (-t[0], t[2], t[1]))
    print(f"{'surv':>5} {'tail_cm':>8} {'peak_cm':>8} {'p_std':>7}  "
          f"{'kp':>5} {'kd':>5} {'kv':>5} {'kx':>5} {'thmax':>6} {'imax':>5}")
    for surv, tail, peak, pstd, g in rows[: args.top]:
        print(f"{surv:5.2f} {100*tail:8.2f} {100*peak:8.2f} {pstd:7.4f}  "
              f"{g[0]:5.1f} {g[1]:5.2f} {g[2]:5.2f} {g[3]:5.2f} {g[4]:6.3f} {g[5]:5.1f}")
    env.close()


if __name__ == "__main__":
    main()
