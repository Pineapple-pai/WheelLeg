"""确定左右腿的驱动符号约定（镜像方式）。

腿是绕 +x 的 hinge，左右腿在 y 上反对称。哪种符号组合能让两条腿保持
**相同腿长**必须实测。这里对几种符号组合各扫一遍，报告左右腿长差。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uz05.env import UZ05Env  # noqa: E402

SETTLE, SUBSTEPS = 80, 4


def hold(env: UZ05Env, q: np.ndarray, settle: int = SETTLE) -> None:
    sim = env.sim
    for _ in range(settle):
        err = q - sim.joint_positions()
        torque = np.clip(200.0 * err - 6.0 * sim.joint_velocities(), -20.0, 20.0)
        sim.data.ctrl[:4] = torque
        sim.data.ctrl[4:] = 0.0
        for _ in range(SUBSTEPS):
            sim.step()


MIRRORS = {
    "(q2,q4 | -q2,-q4)": lambda a, b: (a, b, -a, -b),
    "(q2,q4 | q2,q4)": lambda a, b: (a, b, a, b),
    "(q2,q4 | -q2,q4)": lambda a, b: (a, b, -a, b),
    "(q2,q4 | q2,-q4)": lambda a, b: (a, b, a, -b),
    "(q2,q4 | -q4,-q2)": lambda a, b: (a, b, -b, -a),
    "(q2,q4 | q4,q2)": lambda a, b: (a, b, b, a),
}


def main() -> None:
    env = UZ05Env(stage="stand", stand_level=2, assist_scale=0.0, seed=0,
                  init_scale=0.0)
    env.params.domain_randomization.enabled = False
    env.reset(seed=0)
    base = np.asarray(env.params.robot.stand_joint_pos, dtype=np.float64)
    print(f"nominal q2={base[0]:+.4f} q4={base[1]:+.4f} "
          f"leg={env.sim.leg_lengths().round(4)}")
    for name, fn in MIRRORS.items():
        env.reset(seed=0)
        hold(env, base, settle=150)
        l0 = env.sim.leg_lengths().copy()
        print(f"\n-- {name} --  nominal leg={l0.round(4)} diff={l0[0] - l0[1]:+.4f}")
        for d in (-0.3, -0.15, 0.15, 0.3):
            q = np.asarray(fn(base[0] + d, base[1] + d), dtype=np.float64)
            hold(env, q)
            L = env.sim.leg_lengths()
            rel = env.sim.body_frame(env.sim.data.site_xpos[env.sim.wheel_sites[0]]
                                     - env.sim.data.site_xpos[env.sim.hip_sites[0]])
            print(f"   d={d:+.2f} q={np.round(q, 3)} legL={L[0]:.4f} legR={L[1]:.4f} "
                  f"diff={L[0] - L[1]:+.4f} rel_x={rel[0]:+.4f}")
    env.close()


if __name__ == "__main__":
    main()
