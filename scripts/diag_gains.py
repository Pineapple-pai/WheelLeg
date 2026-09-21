"""聚焦调参：在新的候选平衡增益下，测真实终止配置下的站立质量。

用法:
  conda run -n sim python -u scripts/diag_gains.py --kp 14 --kd 0.2 --kv 16 --kx 80 --db 0.02
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uz05.env import UZ05Env  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kp", type=float, default=14.0)
    ap.add_argument("--kd", type=float, default=0.20)
    ap.add_argument("--kv", type=float, default=16.0)
    ap.add_argument("--kx", type=float, default=80.0)
    ap.add_argument("--db", type=float, default=0.02)
    ap.add_argument("--kdx", type=float, default=0.0)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--station-limit", type=float, default=0.10)
    ap.add_argument("--tilt-limit", type=float, default=0.30)
    ap.add_argument("--noise", action="store_true", default=False)
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--reset-hip", type=float, default=None,
                    help="重置髋关节角（正负号按 [-,+,-,+] 交替），None=保持原逻辑(±0.05)")
    ap.add_argument("--reset-height", type=float, default=None)
    ap.add_argument("--theta-max", type=float, default=None, help="级联外环俯仰指令限幅")
    args = ap.parse_args()

    env = UZ05Env(stage="stand", stand_level=2, assist_scale=0.0)
    env.params.noise.enabled = bool(args.noise)
    env.params.wheel.policy_current_scale_a = 8.0
    env.params.station_hard_limit = args.station_limit
    env.params.balance.station_deadband = args.db
    env.tilt_limit = args.tilt_limit
    scale = env.params.wheel.policy_current_scale_a

    def current(info):
        pitch = info.get("pitch", 0.0)
        pr = info.get("pitch_rate", 0.0)
        vx = info.get("body_vx", 0.0)
        x = info.get("station_error", 0.0)
        if args.theta_max is not None:
            # 级联形式：外环给俯仰指令（限幅），内环做俯仰 PD
            theta_ref = args.kv * (env.command[0] - vx) - args.kx * x
            theta_ref = float(np.clip(theta_ref, -args.theta_max, args.theta_max))
            t = args.kp * (theta_ref - pitch) - args.kp * args.kd * pr
        else:
            t = -args.kp * (pitch + args.kd * pr)
            t -= args.kv * (vx - env.command[0])
            if abs(x) > args.db:
                t -= args.kx * np.copysign(abs(x) - args.db, x)
            t -= args.kdx * vx
        return t / 0.246

    surv, tails, peaks, reasons = [], [], [], []
    pitch_hist = []
    for ep in range(args.episodes):
        env.reset(seed=args.seed0 + ep)
        if args.reset_hip is not None:
            env.sim.data.qpos[env.sim.hip_qpos_adr] = np.array(
                [-args.reset_hip, args.reset_hip, -args.reset_hip, args.reset_hip])
        if args.reset_height is not None:
            env.sim.data.qpos[2] = args.reset_height
        env.sim.forward()
        env.nominal_xy = env.sim.data.qpos[:2].copy()
        info: dict = {}
        done = False
        n = 0
        trace = []
        while not done and n < args.steps:
            a = np.zeros(6)
            a[0] = np.clip(current(info) / scale, -1.0, 1.0)
            _, _, term, trunc, info = env.step(a)
            n += 1
            done = term or trunc
            pitch_hist.append(info["pitch"])
            if args.trace and (n % 20 == 0 or n < 5):
                trace.append((n, info["pitch"], info["station_error"], info["body_vx"],
                              info["wheel_current_left"], info["base_height"]))
        surv.append(bool(trunc and not term))
        tails.append(info["station_tail_mean"])
        peaks.append(info["station_max_abs"])
        reasons.append(info["termination_reason"])
    print(f"gains kp={args.kp} kd={args.kd} kv={args.kv} kx={args.kx} db={args.db} kdx={args.kdx} "
          f"station_lim={args.station_limit} tilt={args.tilt_limit} noise={args.noise}")
    print(f"survive={np.mean(surv):.2f} tail_cm={100*np.mean(tails):.2f} peak_cm={100*np.mean(peaks):.2f} "
          f"pitch_std={np.std(pitch_hist):.4f}")
    for s, t, p, r in zip(surv, tails, peaks, reasons):
        print(f"  surv={int(s)} tail={100*t:6.2f}cm peak={100*p:6.2f}cm  {r}")
    if args.trace:
        print("  step  pitch    x_err   vx      curL    height")
        for row in trace:
            print(f"  {row[0]:5d} {row[1]:+.4f} {row[2]:+.4f} {row[3]:+.3f} {row[4]:+.2f} {row[5]:.4f}")
    env.close()


if __name__ == "__main__":
    main()
