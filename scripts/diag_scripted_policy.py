"""可行性检验：把标定好的手写平衡律从"辅助"搬到"策略动作"通道，
在 assist=0 下跑，验证物理上确实能靠轮电流反馈站住。

如果这个脚本能站住而 PPO 不能，问题在奖励/课程/优化，不在物理。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uz05.env import UZ05Env  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stand-level", type=int, default=2)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--gain", type=float, default=1.0, help="平衡律增益缩放")
    ap.add_argument("--no-noise", action="store_true")
    args = ap.parse_args()

    for level in (0, 1, 2):
        if args.stand_level >= 0 and level != args.stand_level:
            continue
        env = UZ05Env(stage="stand", stand_level=level, assist_scale=0.0)
        if args.no_noise:
            env.params.noise.enabled = False
        b = env.params.balance
        scale = env.params.wheel.policy_current_scale_a
        surv, tails, peaks = [], [], []
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=1000 + ep)
            done = False
            info: dict = {}
            while not done:
                pitch = info.get("pitch", 0.0)
                pitch_rate = info.get("pitch_rate", 0.0)
                vx = info.get("body_vx", 0.0)
                x_err = info.get("station_error", 0.0)
                torque = -b.pitch_kp * (pitch + b.pitch_kd * pitch_rate)
                torque -= b.body_speed_kp * (vx - env.command[0])
                if abs(x_err) > b.station_deadband:
                    boundary = np.copysign(abs(x_err) - b.station_deadband, x_err)
                    torque -= b.station_kp * boundary
                torque *= args.gain
                current = torque / env.params.wheel.torque_per_amp_joint
                action = np.zeros(6, dtype=np.float32)
                action[0] = np.clip(current / scale, -1.0, 1.0)
                obs, rew, term, trunc, info = env.step(action)
                done = term or trunc
            surv.append(bool(trunc and not term))
            tails.append(info["station_tail_mean"])
            peaks.append(info["station_max_abs"])
        print(f"[scripted-balance assist=0] {env.stand_level.name} "
              f"action_limit={env.action_limit} cur_scale={scale:.1f}A "
              f"hard={env.params.station_hard_limit} tilt={env.tilt_limit} "
              f"survive={np.mean(surv):.2f} tail_cm={100*np.mean(tails):.2f} "
              f"peak_cm={100*np.mean(peaks):.2f}")
        env.close()


if __name__ == "__main__":
    main()
