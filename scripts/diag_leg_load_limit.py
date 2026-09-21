"""腿长上限的**负载**归因：关节 PD 在负载下的柔量吃掉了多少行程？

无重力实测机构能到 0.3765 m（`diag_leg_diff_sign.py`），但带载只能到 0.2819 m。
差值来自关节有限刚度：支撑力在髋关节上产生力矩，PD 必须偏出 δ 才能顶住。

本脚本：无重力 vs 有重力，同一差模指令下的腿长差 + 关节目标跟踪误差。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uz05.env import UZ05Env  # noqa: E402

BASE = (-0.10055, 0.09896)   # pd_neutral 的 L2 / L4


def measure(env: UZ05Env, diff: float, gravity: float, kp: float,
            n: int = 4000) -> dict:
    sim = env.sim
    sim.model.opt.gravity[:] = [0.0, 0.0, -gravity]
    q = np.array([BASE[0] + diff, BASE[1] - diff, BASE[0] + diff, BASE[1] - diff])
    sim.data.qpos[:3] = [0.0, 0.0, 0.30]
    sim.data.qpos[3:7] = [1, 0, 0, 0]
    sim.data.qvel[:] = 0.0
    sim.data.qpos[sim.hip_qpos_adr] = q
    for _ in range(n):
        err = q - sim.joint_positions()
        sim.data.ctrl[:4] = np.clip(kp * err - 4.0 * sim.joint_velocities(), -20, 20)
        sim.data.ctrl[4:] = 0.0
        for _ in range(4):
            sim.step()
    L = sim.leg_lengths()
    actual = sim.joint_positions()
    tau = np.clip(kp * (q - actual), -20, 20)
    return {"leg": float(L.mean()), "lr_mm": float((L[0] - L[1]) * 1000),
            "q_err": float(np.abs(q - actual).mean()),
            "tau": float(np.abs(tau).mean()),
            "z": float(sim.data.qpos[2])}


def main() -> None:
    env = UZ05Env(stage="stand", stand_level=2, assist_scale=0.0, seed=0,
                  init_scale=0.0)
    env.params.domain_randomization.enabled = False
    env.reset(seed=0)
    print(f"{'diff':>6} {'grav':>6} {'kp':>5} {'leg':>9} {'关节误差':>9} {'|tau|':>7} {'z':>8}")
    for kp in (100.0, 300.0):
        for grav in (0.0, 9.81):
            for diff in (0.0, 0.3, 0.6, 0.9, 1.2):
                r = measure(env, diff, grav, kp)
                print(f"{diff:>6.2f} {grav:>6.2f} {kp:>5.0f} {r['leg']:>9.4f} "
                      f"{r['q_err']:>9.4f} {r['tau']:>7.2f} {r['z']:>8.4f}")
        print()
    env.close()


if __name__ == "__main__":
    main()
