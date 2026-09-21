"""左右腿差模的**符号约定**：差模应该让两腿同向伸缩，不能把腿差开。

`ActuatorBank` 目前把差模按「左右同号」下发::

    q2_left  = neutral2 + (c + d)·s      q2_right = neutral2 + (c + d)·s
    q4_left  = neutral4 + (c − d)·s      q4_right = neutral4 + (c − d)·s

但左右腿的 hinge 轴都指向 +x，关节角符号相对机构是**反对称**的
（实测额定位形 q2≈−0.194、q4≈+0.194，两腿数值相同、方向相反）。
本脚本在无重力下扫差模，看哪种约定能保持左右腿长一致。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uz05.env import UZ05Env  # noqa: E402

CONVENTIONS = {
    "同号 (c+d, c-d) 两腿相同": lambda c, d: (c + d, c - d, c + d, c - d),
    "反号 (c+d, c-d) L / (c-d, c+d) R": lambda c, d: (c + d, c - d, c - d, c + d),
    "反号 (c+d) L / (c-d) R": lambda c, d: (c + d, c + d, c - d, c - d),
}
BASE = (-0.10055, 0.09896)   # pd_neutral 的 L2 / L4


def settle(env: UZ05Env, q: np.ndarray, n: int = 3000) -> tuple[float, float]:
    sim = env.sim
    for _ in range(n):
        err = q - sim.joint_positions()
        sim.data.ctrl[:4] = np.clip(200.0 * err - 6.0 * sim.joint_velocities(), -20, 20)
        sim.data.ctrl[4:] = 0.0
        for _ in range(4):
            sim.step()
    L = sim.leg_lengths()
    return float(L.mean()), float(L[0] - L[1])


def main() -> None:
    env = UZ05Env(stage="stand", stand_level=2, assist_scale=0.0, seed=0,
                  init_scale=0.0)
    env.params.domain_randomization.enabled = False
    env.reset(seed=0)
    env.sim.model.opt.gravity[:] = 0.0     # 无载，只看机构
    print(f"{'convention':36s} {'d':>6} {'leg':>9} {'L/R(mm)':>9}")
    for name, fn in CONVENTIONS.items():
        for d in (0.0, 0.15, 0.3, 0.5):
            env.sim.data.qpos[:3] = [0.0, 0.0, 0.30]
            env.sim.data.qpos[3:7] = [1, 0, 0, 0]
            env.sim.data.qvel[:] = 0.0
            a, b, cc, dd = fn(BASE[0], d)
            # q2 通道取 a/cc（L2/R2），q4 通道取 b/dd（L4/R4）
            env.sim.data.qpos[env.sim.hip_qpos_adr] = np.array([a, b, cc, dd])
            L, lr = settle(env, np.array([a, b, cc, dd]))
            flag = "  <-- 对称" if abs(lr) < 0.01 else ""
            print(f"{name:36s} {d:>6.2f} {L:>9.4f} {lr * 1000:>+9.2f}{flag}")
        print()
    env.close()


if __name__ == "__main__":
    main()
